"""Per-frame single-channel rule for replaced_saturated rows in the merged-
catalog residual render (build_mergedcat_residuals -> _satstar_render_covered).

Each saturated star must be subtracted by exactly one channel in each frame:
the frame's satstar model where that frame's satstar pass accepted a fit of
the star, the merged-catalog render everywhere else.  The model-threshold
test alone reads 'covered' from the wing of a bright accepted neighbour, so a
star that the frame's satstar pass rejected (and the daophot hand-off fitted)
was subtracted by neither channel.

The default (MERGEDCAT_SATSTAR_COVER unset) keeps the model-threshold rule;
``accepted`` is opt-in and never applies to MIRI.
"""
import os
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from astropy.nddata import NDData
from astropy.table import Table
from astropy.wcs import WCS
from photutils.psf import GriddedPSFModel

import jwst_gc_pipeline.photometry.crowdsource_catalogs_long as ccl
import jwst_gc_pipeline.reduction.filtering as filtering
from jwst_gc_pipeline.mast_names import jw_prefix
from jwst_gc_pipeline.photometry.crowdsource_catalogs_long import (
    _load_frame_satstar_cover,
    _mergedcat_satstar_cover_mode,
    _mergedcat_satstar_cover_radius_fwhm,
    _satstar_render_covered,
    _uncovered_satstar_rows,
)
from jwst_gc_pipeline.photometry.naming import _inst_token, frame_identity


def _model_with_star(shape=(80, 80), x0=20.0, y0=20.0, amp=5.0e4, sigma=1.1):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    r2 = (xx - x0) ** 2 + (yy - y0) ** 2
    # Gaussian core + r^-3 wing, so the wing stays above the cover threshold
    # several pixels out, as a bright F480M satstar model does
    return amp * np.exp(-0.5 * r2 / sigma ** 2) + 2.0e4 / np.maximum(r2, 1.0) ** 1.5


def test_neighbour_wing_is_not_coverage_when_the_frame_did_not_fit_the_star():
    sm = _model_with_star()
    # star B sits 6 px from the accepted satstar A, on A's wing (model > 10)
    bx, by = 26.0, 20.0
    assert sm[int(by) - 3:int(by) + 4, int(bx) - 3:int(bx) + 4].max() > 10.0
    legacy = _satstar_render_covered([bx], [by], sm, 10.0)
    assert legacy.tolist() == [True]          # the old test hands B to nobody
    acc = np.array([[20.0, 20.0]])            # only A was accepted in this frame
    fixed = _satstar_render_covered([bx], [by], sm, 10.0, acc_xy=acc,
                                    cover_radius_px=1.5 * 2.574)
    assert fixed.tolist() == [False]          # B is rendered from the catalog


def test_star_fitted_by_the_frame_stays_covered():
    sm = _model_with_star()
    acc = np.array([[20.3, 19.8]])
    cov = _satstar_render_covered([20.0], [20.0], sm, 10.0, acc_xy=acc,
                                  cover_radius_px=1.5 * 2.574)
    assert cov.tolist() == [True]             # left to the satstar model


def test_no_model_covers_nothing_and_empty_accepted_list_covers_nothing():
    assert _satstar_render_covered([20.0], [20.0], None, 10.0).tolist() == [False]
    sm = _model_with_star()
    cov = _satstar_render_covered([20.0, 50.0], [20.0, 50.0], sm, 10.0,
                                  acc_xy=np.zeros((0, 2)), cover_radius_px=3.9)
    assert cov.tolist() == [False, False]


def test_nonfinite_and_off_grid_rows_are_not_covered():
    sm = _model_with_star()
    cov = _satstar_render_covered([np.nan, -5.0, 20.0], [20.0, 20.0, np.inf], sm, 10.0,
                                  acc_xy=np.array([[20.0, 20.0]]), cover_radius_px=3.9)
    assert cov.tolist() == [False, False, False]


def test_legacy_mode_matches_model_threshold_only():
    sm = _model_with_star()
    xs, ys = [20.0, 26.0, 60.0], [20.0, 20.0, 60.0]
    legacy = _satstar_render_covered(xs, ys, sm, 10.0)
    assert legacy.tolist() == [True, True, False]


# --- MERGEDCAT_SATSTAR_COVER parsing, default, MIRI gate ---------------------

def test_cover_mode_env(monkeypatch):
    monkeypatch.delenv('MERGEDCAT_SATSTAR_COVER', raising=False)
    assert _mergedcat_satstar_cover_mode() == 'model'      # default: old rule
    monkeypatch.setenv('MERGEDCAT_SATSTAR_COVER', '')
    assert _mergedcat_satstar_cover_mode() == 'model'
    monkeypatch.setenv('MERGEDCAT_SATSTAR_COVER', ' Accepted ')
    assert _mergedcat_satstar_cover_mode() == 'accepted'
    monkeypatch.setenv('MERGEDCAT_SATSTAR_COVER', 'model')
    assert _mergedcat_satstar_cover_mode() == 'model'
    monkeypatch.setenv('MERGEDCAT_SATSTAR_COVER', 'bogus')
    with pytest.raises(ValueError):
        _mergedcat_satstar_cover_mode()


@pytest.mark.parametrize('filtername, expected', [
    ('F480M', 'accepted'), ('F212N', 'accepted'), ('f200w', 'accepted'),
    ('F770W', 'model'), ('F2550W', 'model'), ('f1130w', 'model'),
])
def test_accepted_cover_is_gated_off_for_miri(monkeypatch, filtername, expected):
    monkeypatch.delenv('GC_INSTRUMENT_OVERRIDE', raising=False)
    monkeypatch.setenv('MERGEDCAT_SATSTAR_COVER', 'accepted')
    assert _mergedcat_satstar_cover_mode(filtername) == expected


def test_miri_gate_still_rejects_a_malformed_value(monkeypatch):
    monkeypatch.delenv('GC_INSTRUMENT_OVERRIDE', raising=False)
    monkeypatch.setenv('MERGEDCAT_SATSTAR_COVER', 'acepted')
    with pytest.raises(ValueError):
        _mergedcat_satstar_cover_mode('F770W')


def test_cover_radius_env(monkeypatch):
    monkeypatch.delenv('MERGEDCAT_SATSTAR_COVER_RADIUS_FWHM', raising=False)
    assert _mergedcat_satstar_cover_radius_fwhm() == 1.5
    monkeypatch.setenv('MERGEDCAT_SATSTAR_COVER_RADIUS_FWHM', ' ')
    assert _mergedcat_satstar_cover_radius_fwhm() == 1.5   # blank = default
    monkeypatch.setenv('MERGEDCAT_SATSTAR_COVER_RADIUS_FWHM', '2.25')
    assert _mergedcat_satstar_cover_radius_fwhm() == 2.25


@pytest.mark.parametrize('value', ['wide', '0', '-1', 'nan', 'inf'])
def test_cover_radius_rejects_a_value_that_covers_nothing(monkeypatch, value):
    monkeypatch.setenv('MERGEDCAT_SATSTAR_COVER_RADIUS_FWHM', value)
    with pytest.raises(ValueError, match='MERGEDCAT_SATSTAR_COVER_RADIUS_FWHM'):
        _mergedcat_satstar_cover_radius_fwhm()


# --- the render's row selection ---------------------------------------------

def _replaced_loop(sxx, syy, sat_flux, satstar_sm, thresh, nx, ny, half_w, half_h):
    """The per-row loop build_mergedcat_residuals used before the cover switch
    (copied verbatim, returns the rendered row indices)."""
    out = []
    for k in range(len(sxx)):
        sx, sy, sf = float(sxx[k]), float(syy[k]), float(sat_flux[k])
        if not (np.isfinite(sx) and np.isfinite(sy) and np.isfinite(sf)):
            continue
        if not (-half_w < sx < nx + half_w and -half_h < sy < ny + half_h):
            continue
        covered = False
        if satstar_sm is not None:
            xi, yi = int(round(sx)), int(round(sy))
            if 0 <= xi < nx and 0 <= yi < ny:
                sub = satstar_sm[max(0, yi - 3):yi + 4,
                                 max(0, xi - 3):xi + 4]
                covered = (np.isfinite(sub).any()
                           and np.nanmax(sub) > thresh)
        if not covered:
            out.append(k)
    return out


@pytest.mark.parametrize('model_shape', [(80, 80), (70, 90), (90, 70), None])
def test_default_selection_reproduces_the_replaced_loop(model_shape):
    rng = np.random.default_rng(972)
    ny, nx = 80, 80
    half_w = half_h = 10
    n = 400
    sxx = rng.uniform(-15, nx + 15, n)
    syy = rng.uniform(-15, ny + 15, n)
    flux = rng.uniform(1e3, 1e6, n)
    sxx[::37] = np.nan
    syy[::41] = np.inf
    flux[::43] = np.nan
    # rows exactly on the frame / stamp edges
    sxx[:6] = [-half_w, nx + half_w, -0.5, nx - 0.5, 0.0, nx - 1.0]
    sm = None
    if model_shape is not None:
        sm = np.zeros(model_shape)
        for x0, y0 in rng.uniform(0, 80, (12, 2)):
            sm = sm + _model_with_star(model_shape, x0, y0, amp=rng.uniform(1e2, 1e5))
    want = _replaced_loop(sxx, syy, flux, sm, 10.0, nx, ny, half_w, half_h)
    got, n_wing = _uncovered_satstar_rows(sxx, syy, flux, sm, 10.0, (ny, nx),
                                          half_w, half_h)
    assert got.tolist() == want
    assert n_wing == 0


def test_accepted_selection_renders_the_neighbour_wing_star():
    sm = _model_with_star()
    sxx, syy, flux = [20.0, 26.0, 60.0], [20.0, 20.0, 60.0], [1e6, 3e4, 2e4]
    got, n_wing = _uncovered_satstar_rows(sxx, syy, flux, sm, 10.0, sm.shape,
                                          10, 10)
    assert got.tolist() == [2]                # B (index 1) left to A's wing
    got, n_wing = _uncovered_satstar_rows(sxx, syy, flux, sm, 10.0, sm.shape,
                                          10, 10, acc_xy=[[20.1, 20.0]],
                                          cover_radius_px=3.86)
    assert got.tolist() == [1, 2]             # B rendered from the catalog
    assert n_wing == 1


# --- which satstar catalog goes with which model (_load_frame_satstar_cover) --

SFX = '_m6'


def _frame(tmp_path):
    return str(tmp_path / 'jw01234001001_02101_00001_nrcblong_cal.fits')


def _write_model(path, x0=20.0, y0=20.0):
    fits.PrimaryHDU(_model_with_star(x0=x0, y0=y0).astype('float32')).writeto(path)


def _write_catalog(path, xy):
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    Table({'xcentroid': xy[:, 0], 'ycentroid': xy[:, 1],
           'flux_fit': np.full(len(xy), 1e5)}).write(path, format='fits')


def _paths(frame):
    stem = frame[:-len('.fits')]
    return {k: f'{stem}{SFX}_{k}.fits' for k in (
        'satstar_model', 'satstar_catalog',
        'extended_satstar_model', 'extended_satstar_catalog')}


def test_plain_model_reads_the_plain_catalog(tmp_path):
    frame = _frame(tmp_path)
    p = _paths(frame)
    _write_model(p['satstar_model'])
    _write_catalog(p['satstar_catalog'], [[20.0, 20.0], [50.0, 40.0]])
    sm, acc = _load_frame_satstar_cover(frame, SFX, 'accepted')
    assert sm is not None and sm.shape == (80, 80)
    assert acc.tolist() == [[20.0, 20.0], [50.0, 40.0]]


def test_extended_model_reads_the_extended_catalog(tmp_path):
    frame = _frame(tmp_path)
    p = _paths(frame)
    _write_model(p['satstar_model'])
    _write_model(p['extended_satstar_model'])
    _write_catalog(p['satstar_catalog'], [[20.0, 20.0]])
    # the extended catalog adds a forced fit at (60, 60)
    _write_catalog(p['extended_satstar_catalog'], [[20.0, 20.0], [60.0, 60.0]])
    sm, acc = _load_frame_satstar_cover(frame, SFX, 'accepted')
    assert acc.tolist() == [[20.0, 20.0], [60.0, 60.0]]


def test_forced_fit_in_the_extended_catalog_stays_covered(tmp_path):
    """A star fitted only by the forced (extended) pass is in the extended
    model; reading the plain catalog would render it a second time."""
    frame = _frame(tmp_path)
    p = _paths(frame)
    ext = _model_with_star(x0=20.0, y0=20.0) + _model_with_star(x0=60.0, y0=60.0)
    fits.PrimaryHDU(ext.astype('float32')).writeto(p['extended_satstar_model'])
    _write_catalog(p['extended_satstar_catalog'], [[20.0, 20.0], [60.2, 59.9]])
    _write_catalog(p['satstar_catalog'], [[20.0, 20.0]])
    sm, acc = _load_frame_satstar_cover(frame, SFX, 'accepted')
    got, _ = _uncovered_satstar_rows([20.0, 60.0], [20.0, 60.0], [1e6, 1e6], sm,
                                     10.0, sm.shape, 10, 10, acc_xy=acc,
                                     cover_radius_px=3.86)
    assert got.tolist() == []                 # both left to the model


def test_extended_model_without_its_catalog_falls_back_to_model_test(tmp_path, capsys):
    frame = _frame(tmp_path)
    p = _paths(frame)
    _write_model(p['extended_satstar_model'])
    _write_catalog(p['satstar_catalog'], [[20.0, 20.0]])   # not the model's catalog
    sm, acc = _load_frame_satstar_cover(frame, SFX, 'accepted')
    assert sm is not None
    assert acc is None
    assert 'model-threshold cover test' in capsys.readouterr().out


def test_missing_catalog_falls_back_to_model_test(tmp_path, capsys):
    frame = _frame(tmp_path)
    p = _paths(frame)
    _write_model(p['satstar_model'])
    sm, acc = _load_frame_satstar_cover(frame, SFX, 'accepted')
    assert sm is not None and acc is None
    assert 'model-threshold cover test' in capsys.readouterr().out


def test_catalog_without_centroid_columns_falls_back_to_model_test(tmp_path, capsys):
    frame = _frame(tmp_path)
    p = _paths(frame)
    _write_model(p['satstar_model'])
    Table({'x_fit': [20.0], 'y_fit': [20.0]}).write(p['satstar_catalog'], format='fits')
    sm, acc = _load_frame_satstar_cover(frame, SFX, 'accepted')
    assert sm is not None and acc is None
    assert 'has no xcentroid/ycentroid' in capsys.readouterr().out


@pytest.mark.parametrize('payload', [b'', b'not a FITS file' * 20])
def test_unreadable_catalog_falls_back_to_model_test(tmp_path, capsys, payload):
    frame = _frame(tmp_path)
    p = _paths(frame)
    _write_model(p['satstar_model'])
    with open(p['satstar_catalog'], 'wb') as fh:
        fh.write(payload)
    sm, acc = _load_frame_satstar_cover(frame, SFX, 'accepted')
    assert sm is not None and acc is None
    assert 'could not read satstar catalog' in capsys.readouterr().out


def test_model_mode_reads_no_catalog(tmp_path):
    frame = _frame(tmp_path)
    p = _paths(frame)
    _write_model(p['satstar_model'])
    with open(p['satstar_catalog'], 'wb') as fh:   # would be unreadable
        fh.write(b'')
    sm, acc = _load_frame_satstar_cover(frame, SFX, 'model')
    assert sm is not None and acc is None


def test_no_model_means_no_cover_and_no_catalog_read(tmp_path):
    frame = _frame(tmp_path)
    p = _paths(frame)
    _write_catalog(p['satstar_catalog'], [[20.0, 20.0]])
    assert _load_frame_satstar_cover(frame, SFX, 'accepted') == (None, None)


def test_unreadable_extended_model_is_not_replaced_by_the_plain_one(tmp_path, capsys):
    frame = _frame(tmp_path)
    p = _paths(frame)
    with open(p['extended_satstar_model'], 'wb') as fh:
        fh.write(b'')
    _write_model(p['satstar_model'])
    _write_catalog(p['satstar_catalog'], [[20.0, 20.0]])
    assert _load_frame_satstar_cover(frame, SFX, 'accepted') == (None, None)
    assert 'could not read satstar model' in capsys.readouterr().out


def test_model_nans_become_zero(tmp_path):
    frame = _frame(tmp_path)
    p = _paths(frame)
    m = _model_with_star().astype('float32')
    m[0, 0] = np.nan
    fits.PrimaryHDU(m).writeto(p['satstar_model'])
    sm, _ = _load_frame_satstar_cover(frame, SFX, 'model')
    assert sm[0, 0] == 0.0 and np.isfinite(sm).all()


# --- end to end through build_mergedcat_residuals ---------------------------
# One synthetic NIRCam frame, rendered by the real build_mergedcat_residuals.
# Only the PSF-grid builder, the OPD preflight, the FWHM table lookup, the
# JWST-datamodel writer and the resample are replaced; the satstar model and
# catalog, the raw per-frame products and the merged catalog are real files.
#
#   A  saturated; this frame's satstar pass ACCEPTED it: it is in the satstar
#      model (subtracted from the render base) and in the satstar catalog.
#   B  replaced_saturated merged row (another frame accepted it); this frame's
#      satstar pass rejected it, so it is in the base, and it sits on A's
#      satstar-model wing (7x7 max > 10), 6 px from A.
#   C  an ordinary merged row.
#
# Under ``accepted`` the render must subtract B with the catalog PSF and leave
# A to the satstar model; under ``model`` A's wing marks B covered and B stays
# in the residual.  A render whose call site dropped ``acc_xy`` (or the cover
# radius) behaves like ``model`` and fails the ``accepted`` case.

E2E_SHAPE = (64, 64)
E2E_FWHM_PIX = 2.574                      # F480M; 1.5 FWHM = 3.86 px
E2E_A, E2E_B, E2E_C = (24.0, 30.0), (30.0, 30.0), (48.0, 12.0)
E2E_FLUX = {'A': 1.0e6, 'B': 3.0e4, 'C': 2.0e4}
E2E_PID, E2E_FIELD, E2E_ITER, E2E_SATLABEL = '1234', '001', 'm7', 'm6'


def _e2e_grid():
    sigma = E2E_FWHM_PIX / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    yy, xx = np.mgrid[-12:13, -12:13]
    psf = np.exp(-0.5 * (xx ** 2 + yy ** 2) / sigma ** 2)
    psf /= psf.sum()
    ny, nx = E2E_SHAPE
    xy = [(0, 0), (nx - 1, 0), (0, ny - 1), (nx - 1, ny - 1)]
    return GriddedPSFModel(NDData(np.stack([psf] * len(xy)),
                                  meta={'grid_xypos': xy, 'oversampling': 1}))


def _e2e_wcs():
    ww = WCS(naxis=2)
    ww.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    ww.wcs.crval = [266.5, -28.9]
    ww.wcs.crpix = [32.5, 32.5]
    ww.wcs.cdelt = [-0.063 / 3600, 0.063 / 3600]
    return ww


def _e2e_options():
    return SimpleNamespace(cutout_region='', each_exposure=True, blur=False,
                           target='e2e', desaturated=False, epsf=False,
                           group=False, iteration_label=E2E_ITER, bgsub=False,
                           use_iter3_residual_bg=False,
                           mergedcat_render_threads=1)


def _e2e_setup(tmp_path, grid):
    """Write the frame, its raw basic residual/model, its m6 satstar model and
    catalog, and the merged catalog.  Returns (frame, merged_cat_path)."""
    pipeline_dir = tmp_path / 'F480M' / 'pipeline'
    pipeline_dir.mkdir(parents=True)
    ww = _e2e_wcs()
    frame = str(pipeline_dir / 'jw01234001001_02101_00001_nrcblong_crf.fits')
    phdr = fits.Header({'INSTRUME': 'NIRCAM', 'TELESCOP': 'JWST',
                        'DATE-OBS': '2023-04-01', 'FILTER': 'F480M'})
    fits.HDUList([fits.PrimaryHDU(header=phdr),
                  fits.ImageHDU(np.zeros(E2E_SHAPE, 'float32'),
                                header=ww.to_header(), name='SCI'),
                  fits.ImageHDU(np.ones(E2E_SHAPE, 'float32'), name='ERR'),
                  ]).writeto(frame)

    # render base = data - satstar model: A is gone, B and C are in it
    base = ccl._render_model_from_table(
        Table({'x_fit': [E2E_B[0], E2E_C[0]], 'y_fit': [E2E_B[1], E2E_C[1]],
               'flux_fit': [E2E_FLUX['B'], E2E_FLUX['C']]}),
        grid, E2E_SHAPE, (21, 21))
    options = _e2e_options()
    visit, vgroup, exposure, det = frame_identity(frame, field=E2E_FIELD)
    tokens = ccl._predict_output_tokens(options, visit, vgroup, exposure, E2E_ITER)
    stem = (f'{pipeline_dir}/{jw_prefix(E2E_PID)}-o{E2E_FIELD}_t001_'
            f'{_inst_token("F480M")}_clear-f480m-{det}{"".join(tokens)}'
            f'_daophot_basic')
    for suffix, data in (('residual', base), ('model', np.zeros(E2E_SHAPE))):
        fits.HDUList([fits.PrimaryHDU(),
                      fits.ImageHDU(np.asarray(data, 'float32'),
                                    header=ww.to_header(), name='SCI'),
                      ]).writeto(f'{stem}_{suffix}.fits')

    sfx = f'_{E2E_SATLABEL}'
    fits.PrimaryHDU(_model_with_star(E2E_SHAPE, *E2E_A).astype('float32')).writeto(
        frame.replace('.fits', f'{sfx}_satstar_model.fits'))
    _write_catalog(frame.replace('.fits', f'{sfx}_satstar_catalog.fits'), [E2E_A])

    sky = ww.pixel_to_world([E2E_A[0], E2E_B[0], E2E_C[0]],
                            [E2E_A[1], E2E_B[1], E2E_C[1]])
    merged = str(tmp_path / 'merged_f480m_m7.fits')
    Table({'ra': sky.ra.deg, 'dec': sky.dec.deg,
           'flux_fit': [E2E_FLUX['A'], E2E_FLUX['B'], E2E_FLUX['C']],
           'replaced_saturated': [True, True, False]}).write(merged)
    return frame, merged


def _run_e2e(tmp_path, monkeypatch, cover_mode):
    """Run build_mergedcat_residuals on the synthetic frame.  Returns
    (rendered positions per _render_model_from_table call, residual, model)."""
    for var in ('MERGEDCAT_SATSTAR_COVER_RADIUS_FWHM', 'MERGE_RENDER_PSF_SHAPE',
                'MERGE_RENDER_FWHM_MULT', 'MERGE_SATSTAR_RENDER_CAP',
                'GC_INSTRUMENT_OVERRIDE'):
        monkeypatch.delenv(var, raising=False)
    if cover_mode is None:
        monkeypatch.delenv('MERGEDCAT_SATSTAR_COVER', raising=False)
    else:
        monkeypatch.setenv('MERGEDCAT_SATSTAR_COVER', cover_mode)
    grid = _e2e_grid()
    frame, merged = _e2e_setup(tmp_path, grid)

    monkeypatch.setattr(ccl.psf_preflight, 'preflight_psf_data',
                        lambda *a, **k: None)
    monkeypatch.setattr(ccl, 'get_psf_model', lambda *a, **k: (grid, None))
    monkeypatch.setattr(filtering, 'get_fwhm',
                        lambda hdr, *a, **k: (0.164, E2E_FWHM_PIX))
    saved = {}

    def _save(input_filename, output_filename, data, clear_dq=False):
        saved[output_filename] = np.array(data, dtype=float)
    monkeypatch.setattr(ccl, 'save_residual_datamodel', _save)
    monkeypatch.setattr(ccl, '_resample_to_i2d',
                        lambda files, pdir, name, **k: os.path.join(pdir, f'{name}_i2d.fits'))
    calls = []
    _real_render = ccl._render_model_from_table

    def _spy(table, psf_model, shape, psf_shape):
        calls.append(sorted((round(float(x), 3), round(float(y), 3))
                            for x, y in zip(table['x_fit'], table['y_fit'])))
        return _real_render(table, psf_model, shape, psf_shape)
    monkeypatch.setattr(ccl, '_render_model_from_table', _spy)

    out = ccl.build_mergedcat_residuals(
        str(tmp_path), str(tmp_path), merged, 'F480M', E2E_PID, E2E_FIELD,
        'nrcb', _e2e_options(), [frame], E2E_ITER, ['basic'],
        satstar_label=E2E_SATLABEL, write_model_i2d=False)
    assert list(out) == ['basic']
    resid = [v for k, v in saved.items() if k.endswith('_mergedcat_residual.fits')]
    model = [v for k, v in saved.items() if k.endswith('_mergedcat_model.fits')]
    assert len(resid) == 1 and len(model) == 1
    return calls, resid[0], model[0]


def _peak_px(xy):
    return int(round(xy[1])), int(round(xy[0]))


def _b_peak():
    """B's rendered peak pixel (the catalog PSF at B's integer position)."""
    zero = np.zeros((1, 1))
    return E2E_FLUX['B'] * float(_e2e_grid().evaluate(zero, zero, 1.0, 0.0, 0.0)[0, 0])


@pytest.mark.filterwarnings(
    'ignore::jwst_gc_pipeline.frame_wcs.MissingGwcsWarning')
def test_e2e_accepted_renders_the_rejected_star_and_leaves_the_accepted_one_to_the_model(
        tmp_path, monkeypatch, capsys):
    calls, resid, model = _run_e2e(tmp_path, monkeypatch, 'accepted')
    b_peak = _b_peak()
    # catalog-PSF renders: C (ordinary row), then B (the uncovered satstar
    # group); A never, since this frame's satstar model subtracts it
    assert calls == [[E2E_C], [E2E_B]]
    # B and C are subtracted once, A is not subtracted a second time
    assert np.abs(resid).max() < 1e-4 * b_peak
    # model i2d: A comes from the satstar model, B from the catalog PSF
    sm = _model_with_star(E2E_SHAPE, *E2E_A)
    assert model[_peak_px(E2E_A)] == pytest.approx(sm[_peak_px(E2E_A)], rel=1e-5)
    assert (model[_peak_px(E2E_B)] - sm[_peak_px(E2E_B)]
            == pytest.approx(b_peak, rel=1e-4))
    out = capsys.readouterr().out
    assert 'satstar cover test = accepted (accepted fit of the frame within 3.86 px)' in out
    assert ('rendering 1 saturated stars NOT covered by this frame\'s satstar '
            'model' in out)
    assert '1 of them sit on satstar-model flux > 10' in out


@pytest.mark.filterwarnings(
    'ignore::jwst_gc_pipeline.frame_wcs.MissingGwcsWarning')
@pytest.mark.parametrize('cover_mode', [None, 'model'])
def test_e2e_model_rule_leaves_the_neighbour_wing_star_in_the_residual(
        tmp_path, monkeypatch, capsys, cover_mode):
    """The default (and explicit ``model``) render keeps the old rule: A's
    wing covers B, so neither channel subtracts B in this frame."""
    calls, resid, model = _run_e2e(tmp_path, monkeypatch, cover_mode)
    b_peak = _b_peak()
    assert calls == [[E2E_C]]
    assert resid[_peak_px(E2E_B)] == pytest.approx(b_peak, rel=1e-4)
    assert np.abs(resid[_peak_px(E2E_C)]) < 1e-4 * b_peak
    assert np.abs(resid[_peak_px(E2E_A)]) < 1e-2 * b_peak
    out = capsys.readouterr().out
    assert 'satstar cover test = model' in out
    assert 'NOT covered by this frame' not in out
