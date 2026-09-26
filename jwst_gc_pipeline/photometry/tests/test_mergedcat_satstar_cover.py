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
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from jwst_gc_pipeline.photometry.crowdsource_catalogs_long import (
    _load_frame_satstar_cover,
    _mergedcat_satstar_cover_mode,
    _mergedcat_satstar_cover_radius_fwhm,
    _satstar_render_covered,
    _uncovered_satstar_rows,
)


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
