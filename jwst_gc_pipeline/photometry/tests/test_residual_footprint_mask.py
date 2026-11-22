"""The opt-in satstar footprint mask of the merged-catalog residual i2d (env
``MERGEDCAT_FOOTPRINT_MASK_KFAC``, off by default): the switch, the disk
geometry, the catalog reader and the hand-over from ``build_mergedcat_residuals``."""
import math
import os

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

import jwst_gc_pipeline.photometry.crowdsource_catalogs_long as L
from jwst_gc_pipeline.photometry import merge_catalogs
from jwst_gc_pipeline.photometry import residual_footprint_mask as fm
from jwst_gc_pipeline.photometry.tests.test_mergedcat_render_procs import _setup
from jwst_gc_pipeline.photometry.tests.test_mergedcat_satstar_cover import (
    E2E_FIELD, E2E_FWHM_PIX, E2E_ITER, E2E_PID, _e2e_grid, _e2e_options,
)
import jwst_gc_pipeline.reduction.filtering as filtering

pytestmark = [
    pytest.mark.filterwarnings(
        'ignore::jwst_gc_pipeline.frame_wcs.MissingGwcsWarning'),
    pytest.mark.filterwarnings(
        'ignore:This process .* is multi-threaded:DeprecationWarning'),
]

SCALE = 0.063                       # arcsec per pixel of the test i2d: the F480M native scale
SHAPE = (60, 80)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(fm.ENV_KFAC, raising=False)
    monkeypatch.delenv(fm.ENV_MARGIN, raising=False)


def _i2d_wcs(scale=SCALE):
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = [266.5, -28.9]
    w.wcs.crpix = [40.5, 30.5]
    w.wcs.cdelt = [-scale / 3600.0, scale / 3600.0]
    return w


def _write_i2d(path, data=None, wcs=None):
    wcs = wcs or _i2d_wcs()
    data = np.ones(SHAPE, 'float32') if data is None else data
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(np.asarray(data, 'float32'), header=wcs.to_header(),
                                name='SCI')]).writeto(path)


def _satstars(wcs, xy, area):
    sky = wcs.pixel_to_world(np.array([p[0] for p in xy], float),
                             np.array([p[1] for p in xy], float))
    return Table({'skycoord_fit': sky, 'sat_area': np.asarray(area, float)})


def _disk(x, y, radius):
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    return np.hypot(xx - x, yy - y) <= radius


# --- the switch ---------------------------------------------------------------------

def test_off_by_default_and_the_margin_is_not_read(monkeypatch):
    assert fm.mask_params() == (0.0, fm.DEFAULT_MARGIN_PX) == (0.0, 2.0)
    monkeypatch.setenv(fm.ENV_KFAC, '  ')
    assert fm.mask_params()[0] == 0.0
    monkeypatch.setenv(fm.ENV_KFAC, '0')
    monkeypatch.setenv(fm.ENV_MARGIN, 'not a number')
    assert fm.mask_params() == (0.0, 2.0)


@pytest.mark.parametrize('kfac, margin, expected', [
    ('1', None, (1.0, 2.0)),
    ('1.5', '3', (1.5, 3.0)),
    ('0.5', '0', (0.5, 0.0)),
])
def test_switch_values(monkeypatch, kfac, margin, expected):
    monkeypatch.setenv(fm.ENV_KFAC, kfac)
    if margin is not None:
        monkeypatch.setenv(fm.ENV_MARGIN, margin)
    assert fm.mask_params() == expected


@pytest.mark.parametrize('name, value', [
    (fm.ENV_KFAC, 'abc'), (fm.ENV_KFAC, '-1'), (fm.ENV_KFAC, 'nan'),
    (fm.ENV_KFAC, 'inf'), (fm.ENV_MARGIN, 'abc'), (fm.ENV_MARGIN, '-0.5'),
])
def test_switch_typo_raises_naming_the_variable(monkeypatch, name, value):
    monkeypatch.setenv(fm.ENV_KFAC, '1')
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        fm.mask_params()


def test_native_pixel_scale_follows_the_channel():
    assert fm.native_pixel_scale_arcsec('F212N') == pytest.approx(0.031)
    assert fm.native_pixel_scale_arcsec('F480M', 'NIRCAM') == pytest.approx(0.063)
    assert fm.native_pixel_scale_arcsec('F770W', 'MIRI') == pytest.approx(0.11)


# --- the disks ----------------------------------------------------------------------

def test_mask_disks_blank_exactly_the_pixels_within_the_radius():
    sci = np.ones(SHAPE, 'float32')
    n = fm.mask_disks(sci, [30.0, 55.3], [25.0, 40.6], [4.78, 3.2])
    expected = _disk(30.0, 25.0, 4.78) | _disk(55.3, 40.6, 3.2)
    assert np.array_equal(np.isnan(sci), expected)
    assert n == int(expected.sum())
    assert np.all(sci[~expected] == 1.0)


def test_mask_disks_clip_at_the_edge_and_count_only_finite_pixels():
    sci = np.ones(SHAPE, 'float32')
    sci[2, 3] = np.nan                                  # already blank: not counted again
    n = fm.mask_disks(sci, [0.0], [0.0], [5.0])
    expected = _disk(0.0, 0.0, 5.0)
    assert expected[2, 3]                               # inside the disk, so it was blank twice
    assert np.array_equal(np.isnan(sci), expected)
    assert n == int(expected.sum()) - 1


def test_mask_disks_skip_rows_without_a_finite_position_or_radius_or_on_the_image():
    sci = np.ones(SHAPE, 'float32')
    n = fm.mask_disks(sci, [np.nan, 10.0, 10.0, 500.0], [10.0, np.inf, 10.0, 10.0],
                      [3.0, 3.0, np.nan, 3.0])
    assert n == 0
    assert np.all(sci == 1.0)


# --- the i2d ------------------------------------------------------------------------

def test_residual_i2d_radius_is_the_saturated_area_radius_times_kfac_plus_margin(tmp_path):
    path = str(tmp_path / 'resid_i2d.fits')
    data = np.arange(np.prod(SHAPE), dtype='float32').reshape(SHAPE)
    _write_i2d(path, data)
    stars = _satstars(_i2d_wcs(), [(30.0, 25.0), (55.3, 40.6)], [20.0, 45.0])
    out = fm.mask_residual_i2d(path, stars, 'F480M', 'NIRCAM', kfac=1.5, margin_px=1.0)
    r = [1.5 * math.sqrt(a / math.pi) + 1.0 for a in (20.0, 45.0)]
    expected = _disk(30.0, 25.0, r[0]) | _disk(55.3, 40.6, r[1])
    with fits.open(path) as h:
        sci = h['SCI'].data
        assert np.array_equal(np.isnan(sci), expected)
        assert np.array_equal(sci[~expected], data[~expected])
        assert [h['SCI'].header[k] for k in ('FPMASKK', 'FPMASKM', 'FPMASKN')] == [1.5, 1.0, 2]
    assert out['rows'] == 2
    assert out['pixels'] == int(expected.sum())
    assert out['finite'] == data.size
    assert out['scale'] == pytest.approx(SCALE)
    assert out['radius_max'] == pytest.approx(max(r))
    assert out['radius_p50'] == pytest.approx(np.median(r))


def test_residual_i2d_radius_is_in_mosaic_pixels_not_native_ones(tmp_path):
    # the mosaic pixel is half the F480M native pixel, so every native pixel of the
    # footprint radius is two mosaic pixels
    path = str(tmp_path / 'resid_i2d.fits')
    wcs = _i2d_wcs(SCALE / 2.0)
    _write_i2d(path, wcs=wcs)
    stars = _satstars(wcs, [(40.0, 30.0)], [20.0])
    fm.mask_residual_i2d(path, stars, 'F480M', 'NIRCAM', kfac=1.0, margin_px=2.0)
    expected = _disk(40.0, 30.0, 2.0 * math.sqrt(20.0 / math.pi) + 2.0)
    with fits.open(path) as h:
        assert np.array_equal(np.isnan(h['SCI'].data), expected)


def test_residual_i2d_rows_without_a_positive_saturated_area_have_no_footprint(tmp_path):
    path = str(tmp_path / 'resid_i2d.fits')
    _write_i2d(path)
    stars = _satstars(_i2d_wcs(), [(10.0, 10.0), (20.0, 20.0), (30.0, 30.0), (40.0, 40.0)],
                      [0.0, np.nan, -3.0, 12.0])
    out = fm.mask_residual_i2d(path, stars, 'F480M', 'NIRCAM', kfac=1.0, margin_px=2.0)
    assert out['rows'] == 1
    with fits.open(path) as h:
        assert np.array_equal(np.isnan(h['SCI'].data),
                              _disk(40.0, 40.0, math.sqrt(12.0 / math.pi) + 2.0))


def test_residual_i2d_with_no_footprint_row_is_unchanged_but_records_the_setting(tmp_path):
    path = str(tmp_path / 'resid_i2d.fits')
    _write_i2d(path)
    stars = _satstars(_i2d_wcs(), [(10.0, 10.0)], [0.0])
    out = fm.mask_residual_i2d(path, stars, 'F480M', 'NIRCAM', kfac=1.0, margin_px=2.0)
    assert out['rows'] == 0 and out['pixels'] == 0
    with fits.open(path) as h:
        assert np.all(h['SCI'].data == 1.0)
        assert h['SCI'].header['FPMASKN'] == 0


# --- the catalog ---------------------------------------------------------------------

PID, FIELD = '10678', '105'


def _cache_path(basepath, filtername='F212N'):
    return merge_catalogs.consolidated_satstar_cache_path(
        basepath, filtername, merge_catalogs.satstar_obs_scope(PID, FIELD))


def _write_cache(basepath, tbl, filtername='F212N'):
    path = _cache_path(basepath, filtername)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tbl.write(path)
    return path


def test_read_satstars_returns_the_consolidated_cache_of_the_observation(tmp_path):
    stars = _satstars(_i2d_wcs(), [(30.0, 25.0), (50.0, 40.0)], [20.0, 30.0])
    stars['flux_fit'] = [1.0e6, 2.0e6]
    _write_cache(str(tmp_path), stars)
    got = fm.read_satstars(str(tmp_path), 'F212N', PID, FIELD)
    assert len(got) == 2
    assert np.allclose(got['sat_area'], [20.0, 30.0])
    assert got['skycoord_fit'].separation(stars['skycoord_fit']).arcsec.max() < 1e-6


def test_read_satstars_without_a_cache_names_the_path(tmp_path):
    with pytest.raises(fm.FootprintMaskUnavailable, match='consolidated_satstar_catalog'):
        fm.read_satstars(str(tmp_path), 'F212N', PID, FIELD)


@pytest.mark.parametrize('drop', ['sat_area', 'skycoord_fit'])
def test_read_satstars_without_the_footprint_columns_names_the_column(tmp_path, drop):
    stars = _satstars(_i2d_wcs(), [(30.0, 25.0)], [20.0])
    stars.remove_column(drop)
    stars['flux_fit'] = [1.0e6]
    _write_cache(str(tmp_path), stars)
    with pytest.raises(fm.FootprintMaskUnavailable, match=drop):
        fm.read_satstars(str(tmp_path), 'F212N', PID, FIELD)


def test_read_satstars_leaves_the_primary_catalog_to_the_merge(tmp_path, monkeypatch):
    # a stale consolidated cache beside a primary catalog is not what the merge read
    _write_cache(str(tmp_path), _satstars(_i2d_wcs(), [(30.0, 25.0)], [20.0]))
    monkeypatch.setattr(merge_catalogs, '_obs_filters_for', lambda t: {'10678': ['f212n']})
    monkeypatch.setattr(merge_catalogs, '_primary_satstar_catalog_path',
                        lambda *a, **k: '/x/primary_satstar_catalog.fits')
    with pytest.raises(fm.FootprintMaskUnavailable, match='primary_satstar_catalog'):
        fm.read_satstars(str(tmp_path), 'F212N', PID, FIELD, target='gc-treasury')


def test_read_satstars_for_a_target_with_no_primary_reads_the_cache(tmp_path, monkeypatch):
    _write_cache(str(tmp_path), _satstars(_i2d_wcs(), [(30.0, 25.0)], [20.0]))
    monkeypatch.setattr(merge_catalogs, '_obs_filters_for', lambda t: {'10678': ['f212n']})
    monkeypatch.setattr(merge_catalogs, '_primary_satstar_catalog_path',
                        lambda *a, **k: None)
    assert len(fm.read_satstars(str(tmp_path), 'F212N', PID, FIELD, target='gc-treasury')) == 1


def test_read_satstars_for_an_unregistered_target_does_not_look_for_a_primary(tmp_path,
                                                                             monkeypatch):
    _write_cache(str(tmp_path), _satstars(_i2d_wcs(), [(30.0, 25.0)], [20.0]))

    def _never(*a, **k):
        raise AssertionError('the primary-catalog lookup is for registered targets only')

    monkeypatch.setattr(merge_catalogs, '_primary_satstar_catalog_path', _never)
    assert len(fm.read_satstars(str(tmp_path), 'F212N', PID, FIELD, target='e2e')) == 1


# --- build_mergedcat_residuals -> the mask -----------------------------------------------

def _build(root, monkeypatch, capsys, env=None, cache=True, written=None):
    """Run build_mergedcat_residuals on the render-procs fixture with ``_resample_to_i2d``
    writing a real i2d, optionally with the footprint mask on and a consolidated
    satstar cache beside it.  ``written`` collects the per-frame files the residual
    writer was given.  Returns (residual i2d path, captured stdout)."""
    for var in ('MERGEDCAT_SATSTAR_COVER', 'MERGEDCAT_SATSTAR_COVER_RADIUS_FWHM',
                'MERGE_RENDER_PSF_SHAPE', 'MERGE_RENDER_FWHM_MULT',
                'MERGE_SATSTAR_RENDER_CAP', 'GC_INSTRUMENT_OVERRIDE'):
        monkeypatch.delenv(var, raising=False)
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    root.mkdir()
    frames, merged = _setup(root)
    if cache:
        path = merge_catalogs.consolidated_satstar_cache_path(
            str(root), 'F480M', merge_catalogs.satstar_obs_scope(E2E_PID, E2E_FIELD))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _satstars(_i2d_wcs(), [(30.0, 25.0), (55.3, 40.6)], [20.0, 45.0]).write(path)
    written = [] if written is None else written

    def _resample(files, pdir, name, **kw):
        out = os.path.join(pdir, f'{name}_i2d.fits')
        _write_i2d(out)
        return out

    monkeypatch.setattr(L.psf_preflight, 'preflight_psf_data', lambda *a, **k: None)
    monkeypatch.setattr(L, 'get_psf_model', lambda *a, module=None, **k: (_e2e_grid(), None))
    monkeypatch.setattr(filtering, 'get_fwhm', lambda hdr, *a, **k: (0.164, E2E_FWHM_PIX))
    monkeypatch.setattr(L, 'save_residual_datamodel',
                        lambda input_filename, output_filename, data, **kw:
                        written.append(output_filename))
    monkeypatch.setattr(L, '_resample_to_i2d', _resample)
    outpaths = L.build_mergedcat_residuals(
        str(root), str(root), merged, 'F480M', E2E_PID, E2E_FIELD, 'merged',
        _e2e_options(), frames, E2E_ITER, ['basic'], satstar_label=None,
        write_model_i2d=False)
    return outpaths['basic'], capsys.readouterr().out


def test_switch_off_leaves_the_residual_i2d_alone(tmp_path, monkeypatch, capsys):
    path, out = _build(tmp_path / 'a', monkeypatch, capsys)
    with fits.open(path) as h:
        assert np.all(h['SCI'].data == 1.0)
        assert 'FPMASKK' not in h['SCI'].header
    assert 'footprint mask' not in out and 'resid i2d footprint' not in out


def test_switch_on_blanks_the_footprints_of_the_consolidated_satstars(tmp_path, monkeypatch,
                                                                       capsys):
    path, out = _build(tmp_path / 'b', monkeypatch, capsys,
                       env={fm.ENV_KFAC: '1.5', fm.ENV_MARGIN: '1'})
    r = [1.5 * math.sqrt(a / math.pi) + 1.0 for a in (20.0, 45.0)]
    expected = _disk(30.0, 25.0, r[0]) | _disk(55.3, 40.6, r[1])
    with fits.open(path) as h:
        assert np.array_equal(np.isnan(h['SCI'].data), expected)
        assert h['SCI'].header['FPMASKN'] == 2
    assert f"NaN'd {int(expected.sum())} px" in out
    assert 'around 2 satstar rows' in out


def test_switch_on_without_a_cache_warns_and_keeps_the_image(tmp_path, monkeypatch, capsys):
    path, out = _build(tmp_path / 'c', monkeypatch, capsys,
                       env={fm.ENV_KFAC: '1'}, cache=False)
    with fits.open(path) as h:
        assert np.all(h['SCI'].data == 1.0)
    assert 'WARNING mergedcat: resid i2d footprint mask skipped' in out


def test_a_typo_in_the_switch_fails_before_any_residual_is_rendered(tmp_path, monkeypatch,
                                                                    capsys):
    written = []
    with pytest.raises(ValueError, match=fm.ENV_KFAC):
        _build(tmp_path / 'd', monkeypatch, capsys, env={fm.ENV_KFAC: 'oops'},
               written=written)
    assert written == []
