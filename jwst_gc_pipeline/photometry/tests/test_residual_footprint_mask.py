"""The opt-in satstar footprint mask of the merged-catalog residual i2d (env
``MERGEDCAT_FOOTPRINT_MASK_KFAC``, off by default, applied to the m7 pass unless
``MERGEDCAT_FOOTPRINT_MASK_PASSES`` says otherwise): the switch, the pass scope,
the disk geometry, the ring-zone extension of the disks of the brightest rows
(``MERGEDCAT_FOOTPRINT_MASK_RING_FLUX_MIN`` and ``..._RING_RADIUS_PX``), the catalog
reader, the gate-rejected fits that ``MERGEDCAT_FOOTPRINT_MASK_REJECTED`` adds to the
disks, and the hand-over from ``build_mergedcat_residuals``."""
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
    monkeypatch.delenv(fm.ENV_PASSES, raising=False)
    monkeypatch.delenv(fm.ENV_RING_FLUX_MIN, raising=False)
    monkeypatch.delenv(fm.ENV_RING_RADIUS_PX, raising=False)
    monkeypatch.delenv(fm.ENV_REJECTED, raising=False)


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


def _satstars(wcs, xy, area, flux=None):
    sky = wcs.pixel_to_world(np.array([p[0] for p in xy], float),
                             np.array([p[1] for p in xy], float))
    tbl = Table({'skycoord_fit': sky, 'sat_area': np.asarray(area, float)})
    if flux is not None:
        tbl['flux_fit'] = np.asarray(flux, float)
    return tbl


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


def test_the_ring_extension_is_off_by_default_and_unread_while_it_is_off(monkeypatch):
    assert fm.ring_params() == (None, 0.0)
    # the mask itself is off: neither ring variable is read
    monkeypatch.setenv(fm.ENV_RING_FLUX_MIN, 'abc')
    monkeypatch.setenv(fm.ENV_RING_RADIUS_PX, '-3')
    assert fm.ring_params() == (None, 0.0)
    monkeypatch.setenv(fm.ENV_KFAC, '1')
    with pytest.raises(ValueError, match=fm.ENV_RING_FLUX_MIN):
        fm.ring_params()
    # a flux of 0 or a blank one is off, and the radius is then not read
    monkeypatch.setenv(fm.ENV_RING_FLUX_MIN, '0')
    assert fm.ring_params() == (None, 0.0)
    monkeypatch.setenv(fm.ENV_RING_FLUX_MIN, '  ')
    assert fm.ring_params() == (None, 0.0)


@pytest.mark.parametrize('flux, radius, expected', [
    ('1e6', '25', (1e6, 25.0)), ('500000', '12.5', (5e5, 12.5)),
])
def test_ring_values(monkeypatch, flux, radius, expected):
    monkeypatch.setenv(fm.ENV_KFAC, '1')
    monkeypatch.setenv(fm.ENV_RING_FLUX_MIN, flux)
    monkeypatch.setenv(fm.ENV_RING_RADIUS_PX, radius)
    assert fm.ring_params() == expected


@pytest.mark.parametrize('name, value', [
    (fm.ENV_RING_FLUX_MIN, 'abc'), (fm.ENV_RING_FLUX_MIN, '-1'),
    (fm.ENV_RING_FLUX_MIN, 'nan'), (fm.ENV_RING_FLUX_MIN, 'inf'),
    (fm.ENV_RING_RADIUS_PX, 'abc'), (fm.ENV_RING_RADIUS_PX, '-3'),
    (fm.ENV_RING_RADIUS_PX, 'inf'),
])
def test_ring_typo_raises_naming_the_variable(monkeypatch, name, value):
    monkeypatch.setenv(fm.ENV_KFAC, '1')
    monkeypatch.setenv(fm.ENV_RING_FLUX_MIN, '1e6')
    monkeypatch.setenv(fm.ENV_RING_RADIUS_PX, '25')
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        fm.ring_params()


@pytest.mark.parametrize('radius', [None, '', '0'])
def test_a_ring_flux_without_a_radius_raises_naming_both_variables(monkeypatch, radius):
    monkeypatch.setenv(fm.ENV_KFAC, '1')
    monkeypatch.setenv(fm.ENV_RING_FLUX_MIN, '1e6')
    if radius is not None:
        monkeypatch.setenv(fm.ENV_RING_RADIUS_PX, radius)
    with pytest.raises(ValueError, match=fm.ENV_RING_RADIUS_PX) as err:
        fm.ring_params()
    assert fm.ENV_RING_FLUX_MIN in str(err.value)


def test_the_mask_is_scoped_to_m7_by_default():
    assert fm.mask_passes() == fm.DEFAULT_PASSES == ('m7',)


@pytest.mark.parametrize('raw, expected', [
    ('m7', ('m7',)), ('m6,m7', ('m6', 'm7')), ('m3 m4  m7', ('m3', 'm4', 'm7')),
    ('M7, M2', ('m7', 'm2')), ('  ', ('m7',)),
])
def test_pass_list_values(monkeypatch, raw, expected):
    monkeypatch.setenv(fm.ENV_PASSES, raw)
    assert fm.mask_passes() == expected


@pytest.mark.parametrize('raw', ['m1', 'm12', 'm8', 'm7,m9', 'final', 'iter7'])
def test_pass_list_typo_raises_naming_the_variable(monkeypatch, raw):
    monkeypatch.setenv(fm.ENV_PASSES, raw)
    with pytest.raises(ValueError, match=fm.ENV_PASSES):
        fm.mask_passes()


def test_applies_to_needs_the_switch_on_and_the_pass_selected(monkeypatch):
    assert not any(fm.applies_to(label) for label in fm.PASS_LABELS)
    monkeypatch.setenv(fm.ENV_KFAC, '1')
    assert [label for label in fm.PASS_LABELS if fm.applies_to(label)] == ['m7']
    assert not fm.applies_to(None) and not fm.applies_to('')
    monkeypatch.setenv(fm.ENV_PASSES, 'm3, m7')
    assert [label for label in fm.PASS_LABELS if fm.applies_to(label)] == ['m3', 'm7']


def test_a_stray_pass_list_cannot_break_a_run_with_the_switch_off(monkeypatch):
    monkeypatch.setenv(fm.ENV_PASSES, 'not a pass')
    assert fm.applies_to('m7') is False
    monkeypatch.setenv(fm.ENV_KFAC, '0')
    assert fm.applies_to('m7') is False


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


# --- the ring-zone extension of the i2d mask ---------------------------------------------

def test_residual_i2d_ring_extension_widens_the_disks_of_the_bright_rows(tmp_path):
    path = str(tmp_path / 'resid_i2d.fits')
    _write_i2d(path)
    stars = _satstars(_i2d_wcs(), [(20.0, 20.0), (55.0, 40.0), (40.0, 10.0)],
                      [20.0, 20.0, 20.0], flux=[2e6, 1e6, 9.9e5])
    out = fm.mask_residual_i2d(path, stars, 'F480M', 'NIRCAM', kfac=1.0, margin_px=2.0,
                               ring_flux_min=1e6, ring_radius_px=9.3)
    small = math.sqrt(20.0 / math.pi) + 2.0
    # the flux of the second row is exactly the minimum: included; the third is below it
    expected = _disk(20.0, 20.0, 9.3) | _disk(55.0, 40.0, 9.3) | _disk(40.0, 10.0, small)
    with fits.open(path) as h:
        assert np.array_equal(np.isnan(h['SCI'].data), expected)
        hdr = h['SCI'].header
        assert [hdr[k] for k in ('FPMASKN', 'FPRINGF', 'FPRINGR', 'FPRINGN')] == [3, 1e6, 9.3, 2]
    assert out['rows'] == 3 and out['ring_rows'] == 2
    assert out['pixels'] == int(expected.sum())
    assert out['radius_max'] == pytest.approx(9.3)


def test_residual_i2d_ring_extension_never_shrinks_a_larger_footprint_disk(tmp_path):
    path = str(tmp_path / 'resid_i2d.fits')
    _write_i2d(path)
    stars = _satstars(_i2d_wcs(), [(40.0, 30.0)], [400.0], flux=[3e6])
    out = fm.mask_residual_i2d(path, stars, 'F480M', 'NIRCAM', kfac=1.0, margin_px=2.0,
                               ring_flux_min=1e6, ring_radius_px=6.0)
    with fits.open(path) as h:
        assert np.array_equal(np.isnan(h['SCI'].data),
                              _disk(40.0, 30.0, math.sqrt(400.0 / math.pi) + 2.0))
    assert out['ring_rows'] == 1       # counted: at or above the flux, widened or not


def test_residual_i2d_ring_extension_skips_rows_without_a_footprint_or_a_finite_flux(tmp_path):
    path = str(tmp_path / 'resid_i2d.fits')
    _write_i2d(path)
    stars = _satstars(_i2d_wcs(), [(15.0, 15.0), (40.0, 30.0), (65.0, 45.0)],
                      [0.0, 12.0, 12.0], flux=[5e6, np.nan, 5e6])
    out = fm.mask_residual_i2d(path, stars, 'F480M', 'NIRCAM', kfac=1.0, margin_px=2.0,
                               ring_flux_min=1e6, ring_radius_px=8.3)
    expected = (_disk(40.0, 30.0, math.sqrt(12.0 / math.pi) + 2.0)
                | _disk(65.0, 45.0, 8.3))
    with fits.open(path) as h:
        assert np.array_equal(np.isnan(h['SCI'].data), expected)
    assert out['rows'] == 2 and out['ring_rows'] == 1


def test_residual_i2d_ring_radius_is_in_mosaic_pixels_not_native_ones(tmp_path):
    path = str(tmp_path / 'resid_i2d.fits')
    wcs = _i2d_wcs(SCALE / 2.0)
    _write_i2d(path, wcs=wcs)
    stars = _satstars(wcs, [(40.0, 30.0)], [20.0], flux=[2e6])
    fm.mask_residual_i2d(path, stars, 'F480M', 'NIRCAM', kfac=1.0, margin_px=2.0,
                         ring_flux_min=1e6, ring_radius_px=10.3)
    # the footprint radius there is 2 * 2.52 + 2 = 7.05 mosaic pixels
    with fits.open(path) as h:
        assert np.array_equal(np.isnan(h['SCI'].data), _disk(40.0, 30.0, 10.3))


def test_residual_i2d_without_the_ring_extension_records_no_ring_setting(tmp_path):
    path = str(tmp_path / 'resid_i2d.fits')
    _write_i2d(path)
    stars = _satstars(_i2d_wcs(), [(40.0, 30.0)], [20.0], flux=[5e7])
    out = fm.mask_residual_i2d(path, stars, 'F480M', 'NIRCAM', kfac=1.0, margin_px=2.0)
    assert out['ring_rows'] == 0
    with fits.open(path) as h:
        assert np.array_equal(np.isnan(h['SCI'].data),
                              _disk(40.0, 30.0, math.sqrt(20.0 / math.pi) + 2.0))
        assert not [k for k in h['SCI'].header if k.startswith('FPRING')]


def test_residual_i2d_ring_extension_needs_the_flux_column_and_leaves_the_file_alone(tmp_path):
    path = str(tmp_path / 'resid_i2d.fits')
    _write_i2d(path)
    stars = _satstars(_i2d_wcs(), [(40.0, 30.0)], [20.0])
    with pytest.raises(fm.FootprintMaskUnavailable, match='flux_fit'):
        fm.mask_residual_i2d(path, stars, 'F480M', 'NIRCAM', kfac=1.0, margin_px=2.0,
                             ring_flux_min=1e6, ring_radius_px=9.0)
    with fits.open(path) as h:
        assert np.all(h['SCI'].data == 1.0)
        assert 'FPMASKK' not in h['SCI'].header


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


def test_read_satstars_reads_the_flux_column_only_for_the_ring_extension(tmp_path):
    stars = _satstars(_i2d_wcs(), [(30.0, 25.0)], [20.0])
    _write_cache(str(tmp_path), stars)
    assert len(fm.read_satstars(str(tmp_path), 'F212N', PID, FIELD)) == 1
    with pytest.raises(fm.FootprintMaskUnavailable, match='flux_fit'):
        fm.read_satstars(str(tmp_path), 'F212N', PID, FIELD, need_flux=True)


# --- build_mergedcat_residuals -> the mask -----------------------------------------------

def _build(root, monkeypatch, capsys, env=None, cache=True, written=None,
           iteration_label=E2E_ITER, flux=None, rejected=None):
    """Run build_mergedcat_residuals on the render-procs fixture with ``_resample_to_i2d``
    writing a real i2d, optionally with the footprint mask on and a consolidated
    satstar cache beside it (with a ``flux_fit`` column when ``flux`` is given).
    ``written`` collects the per-frame files the residual writer was given.
    ``iteration_label`` is the pass the residual is built for.  ``rejected`` is a
    table written as the ``iteration_label`` rejected-satstar file of the first
    frame.
    Returns (residual i2d path, captured stdout)."""
    for var in ('MERGEDCAT_SATSTAR_COVER', 'MERGEDCAT_SATSTAR_COVER_RADIUS_FWHM',
                'MERGE_RENDER_PSF_SHAPE', 'MERGE_RENDER_FWHM_MULT',
                'MERGE_SATSTAR_RENDER_CAP', 'GC_INSTRUMENT_OVERRIDE'):
        monkeypatch.delenv(var, raising=False)
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    root.mkdir()
    frames, merged = _setup(root, iteration_label)
    if cache:
        path = merge_catalogs.consolidated_satstar_cache_path(
            str(root), 'F480M', merge_catalogs.satstar_obs_scope(E2E_PID, E2E_FIELD))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _satstars(_i2d_wcs(), [(30.0, 25.0), (55.3, 40.6)], [20.0, 45.0],
                  flux=flux).write(path)
    if rejected is not None:
        rejected.write(frames[0].replace('.fits', f'_{iteration_label}_satstar_rejected.fits'))
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
        _e2e_options(), frames, iteration_label, ['basic'], satstar_label=None,
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


def test_the_switch_leaves_the_residual_i2d_of_an_earlier_pass_alone(tmp_path, monkeypatch,
                                                                     capsys):
    path, out = _build(tmp_path / 'e', monkeypatch, capsys,
                       env={fm.ENV_KFAC: '1'}, iteration_label='m3')
    with fits.open(path) as h:
        assert np.all(h['SCI'].data == 1.0)
        assert 'FPMASKK' not in h['SCI'].header
    assert f'pass m3 is not in {fm.ENV_PASSES} (m7)' in out
    assert 'resid i2d footprint mask' not in out


def test_the_pass_list_selects_the_pass_to_mask(tmp_path, monkeypatch, capsys):
    path, out = _build(tmp_path / 'f', monkeypatch, capsys,
                       env={fm.ENV_KFAC: '1', fm.ENV_PASSES: 'm3,m7'},
                       iteration_label='m3')
    with fits.open(path) as h:
        assert np.isnan(h['SCI'].data).any()
        assert h['SCI'].header['FPMASKN'] == 2
    assert 'the m3 residual i2d is NaN inside' in out


def test_a_typo_in_the_pass_list_fails_before_any_residual_is_rendered(tmp_path, monkeypatch,
                                                                       capsys):
    written = []
    with pytest.raises(ValueError, match=fm.ENV_PASSES):
        _build(tmp_path / 'g', monkeypatch, capsys,
               env={fm.ENV_KFAC: '1', fm.ENV_PASSES: 'm77'}, written=written)
    assert written == []


def test_the_ring_extension_widens_the_bright_row_in_the_render(tmp_path, monkeypatch, capsys):
    path, out = _build(tmp_path / 'h', monkeypatch, capsys,
                       env={fm.ENV_KFAC: '1.5', fm.ENV_MARGIN: '1',
                            fm.ENV_RING_FLUX_MIN: '1e6', fm.ENV_RING_RADIUS_PX: '12.3'},
                       flux=[2e6, 1e5])
    r = [1.5 * math.sqrt(a / math.pi) + 1.0 for a in (20.0, 45.0)]
    expected = _disk(30.0, 25.0, 12.3) | _disk(55.3, 40.6, r[1])
    with fits.open(path) as h:
        assert np.array_equal(np.isnan(h['SCI'].data), expected)
        assert h['SCI'].header['FPRINGN'] == 1
    assert 'and inside at least 12.3 px of every row with flux_fit >= 1e+06' in out
    assert '(1 of them out to at least 12.3 px)' in out


def test_the_ring_extension_without_a_flux_column_warns_and_keeps_the_image(tmp_path, monkeypatch,
                                                                           capsys):
    path, out = _build(tmp_path / 'i', monkeypatch, capsys,
                       env={fm.ENV_KFAC: '1', fm.ENV_RING_FLUX_MIN: '1e6',
                            fm.ENV_RING_RADIUS_PX: '12'})
    with fits.open(path) as h:
        assert np.all(h['SCI'].data == 1.0)
    assert 'WARNING mergedcat: resid i2d footprint mask skipped' in out
    assert 'flux_fit' in out


def test_a_ring_typo_fails_before_any_residual_is_rendered(tmp_path, monkeypatch, capsys):
    written = []
    with pytest.raises(ValueError, match=fm.ENV_RING_RADIUS_PX):
        _build(tmp_path / 'j', monkeypatch, capsys, written=written,
               env={fm.ENV_KFAC: '1', fm.ENV_RING_FLUX_MIN: '1e6'})
    assert written == []


def test_ring_variables_do_nothing_while_the_mask_is_off(tmp_path, monkeypatch, capsys):
    path, out = _build(tmp_path / 'k', monkeypatch, capsys,
                       env={fm.ENV_RING_FLUX_MIN: 'oops', fm.ENV_RING_RADIUS_PX: '-1'})
    with fits.open(path) as h:
        assert np.all(h['SCI'].data == 1.0)
    assert 'footprint mask' not in out


# --- the gate-rejected fits (MERGEDCAT_FOOTPRINT_MASK_REJECTED) ----------------------------

def test_the_rejected_extension_is_off_by_default_and_unread_while_the_mask_is_off(monkeypatch):
    assert fm.rejected_enabled() is False
    monkeypatch.setenv(fm.ENV_REJECTED, 'oops')
    assert fm.rejected_enabled() is False


@pytest.mark.parametrize('raw, expected', [
    ('1', True), ('true', True), (' YES ', True), ('On', True),
    ('0', False), ('false', False), ('no', False), ('OFF', False), ('', False), ('  ', False)])
def test_rejected_switch_values(monkeypatch, raw, expected):
    monkeypatch.setenv(fm.ENV_KFAC, '1')
    monkeypatch.setenv(fm.ENV_REJECTED, raw)
    assert fm.rejected_enabled() is expected


@pytest.mark.parametrize('raw', ['maybe', '2', 'tru'])
def test_rejected_switch_typo_raises_naming_the_variable(monkeypatch, raw):
    monkeypatch.setenv(fm.ENV_KFAC, '1')
    monkeypatch.setenv(fm.ENV_REJECTED, raw)
    with pytest.raises(ValueError, match=fm.ENV_REJECTED):
        fm.rejected_enabled()


def _flagged(wcs, xy, area, flag, flux=None):
    tbl = _satstars(wcs, xy, area, flux=flux)
    tbl[fm.MASKABLE_COLUMN] = np.asarray(flag, bool)
    return tbl


def test_residual_i2d_masks_the_rejected_rows_with_the_same_radius(tmp_path):
    path = str(tmp_path / 'resid_i2d.fits')
    _write_i2d(path)
    wcs = _i2d_wcs()
    stars = _satstars(wcs, [(20.0, 20.0)], [20.0])
    rejected = _satstars(wcs, [(55.0, 40.0), (40.0, 10.0)], [45.0, 0.0])
    out = fm.mask_residual_i2d(path, stars, 'F480M', 'NIRCAM', kfac=1.0, margin_px=2.0,
                               rejected=rejected)
    r = [math.sqrt(a / math.pi) + 2.0 for a in (20.0, 45.0)]
    # the second rejected row has no saturated area: no footprint
    expected = _disk(20.0, 20.0, r[0]) | _disk(55.0, 40.0, r[1])
    with fits.open(path) as h:
        assert np.array_equal(np.isnan(h['SCI'].data), expected)
        hdr = h['SCI'].header
        assert hdr['FPMASKN'] == 1 and hdr['FPREJN'] == 1
    assert out['rows'] == 1 and out['rejected_rows'] == 1
    assert out['pixels'] == int(expected.sum())
    assert out['radius_max'] == pytest.approx(r[1])


def test_residual_i2d_without_rejected_rows_records_no_rejected_count(tmp_path):
    path = str(tmp_path / 'resid_i2d.fits')
    _write_i2d(path)
    out = fm.mask_residual_i2d(path, _satstars(_i2d_wcs(), [(20.0, 20.0)], [20.0]),
                               'F480M', 'NIRCAM', kfac=1.0, margin_px=2.0)
    assert out['rejected_rows'] == 0
    with fits.open(path) as h:
        assert 'FPREJN' not in h['SCI'].header


def test_residual_i2d_with_an_empty_rejected_table_records_a_zero_count(tmp_path):
    path = str(tmp_path / 'resid_i2d.fits')
    _write_i2d(path)
    wcs = _i2d_wcs()
    rejected = _satstars(wcs, [(55.0, 40.0)], [45.0])[:0]
    out = fm.mask_residual_i2d(path, _satstars(wcs, [(20.0, 20.0)], [20.0]),
                               'F480M', 'NIRCAM', kfac=1.0, margin_px=2.0, rejected=rejected)
    expected = _disk(20.0, 20.0, math.sqrt(20.0 / math.pi) + 2.0)
    assert out['rows'] == 1 and out['rejected_rows'] == 0
    with fits.open(path) as h:
        assert np.array_equal(np.isnan(h['SCI'].data), expected)
        assert h['SCI'].header['FPREJN'] == 0


def test_residual_i2d_masks_the_rejected_rows_alone_when_there_is_no_consolidated_row(tmp_path):
    path = str(tmp_path / 'resid_i2d.fits')
    _write_i2d(path)
    wcs = _i2d_wcs()
    out = fm.mask_residual_i2d(path, _satstars(wcs, [(20.0, 20.0)], [0.0]), 'F480M',
                               'NIRCAM', kfac=1.0, margin_px=2.0,
                               rejected=_satstars(wcs, [(55.0, 40.0)], [45.0]))
    assert out['rows'] == 0 and out['rejected_rows'] == 1
    assert out['pixels'] == int(_disk(55.0, 40.0, math.sqrt(45.0 / math.pi) + 2.0).sum())


def test_residual_i2d_ring_extension_widens_the_rejected_rows_too(tmp_path):
    path = str(tmp_path / 'resid_i2d.fits')
    _write_i2d(path)
    wcs = _i2d_wcs()
    stars = _satstars(wcs, [(20.0, 20.0)], [20.0], flux=[2e6])
    rejected = _satstars(wcs, [(55.0, 40.0), (40.0, 10.0)], [20.0, 20.0], flux=[1e6, 5e5])
    out = fm.mask_residual_i2d(path, stars, 'F480M', 'NIRCAM', kfac=1.0, margin_px=2.0,
                               ring_flux_min=1e6, ring_radius_px=9.3, rejected=rejected)
    small = math.sqrt(20.0 / math.pi) + 2.0
    expected = _disk(20.0, 20.0, 9.3) | _disk(55.0, 40.0, 9.3) | _disk(40.0, 10.0, small)
    with fits.open(path) as h:
        assert np.array_equal(np.isnan(h['SCI'].data), expected)
        hdr = h['SCI'].header
        assert [hdr[k] for k in ('FPMASKN', 'FPREJN', 'FPRINGN')] == [1, 2, 2]
    assert out['rows'] == 1 and out['rejected_rows'] == 2 and out['ring_rows'] == 2


def test_residual_i2d_ring_extension_needs_the_flux_column_of_the_rejected_rows_too(tmp_path):
    path = str(tmp_path / 'resid_i2d.fits')
    _write_i2d(path)
    wcs = _i2d_wcs()
    stars = _satstars(wcs, [(40.0, 30.0)], [20.0], flux=[1e6])
    rejected = _satstars(wcs, [(20.0, 20.0)], [20.0])
    with pytest.raises(fm.FootprintMaskUnavailable,
                       match='rejected satstar catalog has no flux_fit'):
        fm.mask_residual_i2d(path, stars, 'F480M', 'NIRCAM', kfac=1.0, margin_px=2.0,
                             ring_flux_min=1e6, ring_radius_px=9.0, rejected=rejected)
    with fits.open(path) as h:
        assert np.all(h['SCI'].data == 1.0)
        assert 'FPMASKK' not in h['SCI'].header


# --- the rejected-fit reader ---------------------------------------------------------------

#: the observation the readers are scoped to (PID/FIELD above) and one exposure of it in
#: the two frame variants that coexist in a tree
FRAME_A = 'jw10678105001_02101_00001_nrca1_destreak_o105_crf'
FRAME_A_ALIGN = 'jw10678105001_02101_00001_nrca1_align_o105_crf'
FRAME_B = 'jw10678105001_02101_00002_nrca1_destreak_o105_crf'
FRAME_OTHER_OBS = 'jw10678130001_02101_00001_nrca1_destreak_o130_crf'


def _moved_wcs():
    """The WCS the rejected files were written under: 90 mas from the frames'."""
    w = _i2d_wcs()
    w.wcs.crval = [w.wcs.crval[0] + 0.09 / 3600.0 / math.cos(math.radians(-28.9)),
                   w.wcs.crval[1]]
    return w


def _rejected_file(root, stem, phase, xy, area, flag=True, filtername='F212N', flux=None):
    """Write ``stem``'s frame (if it is not there) and its ``phase`` rejected file with the
    stored sky under the moved WCS.  ``flag`` is one flag per row (a bool for every row), or
    None to leave the column out.  Returns the frame path."""
    pdir = root / filtername / 'pipeline'
    pdir.mkdir(parents=True, exist_ok=True)
    frame = pdir / f'{stem}.fits'
    if not frame.exists():
        _write_i2d(str(frame))
    x = np.array([p[0] for p in xy], float)
    y = np.array([p[1] for p in xy], float)
    tbl = Table({'xcentroid': x, 'ycentroid': y,
                 'skycoord_fit': _moved_wcs().pixel_to_world(x, y),
                 'sat_area': np.asarray(area, float)})
    if flux is not None:
        tbl['flux_fit'] = np.asarray(flux, float)
    if flag is not None:
        tbl[fm.MASKABLE_COLUMN] = (np.full(len(xy), flag) if isinstance(flag, bool)
                                   else np.asarray(flag, bool))
    tbl.write(str(pdir / f'{stem}_{phase}_satstar_rejected.fits'))
    return str(frame)


def _sky_sep_mas(tbl, x, y):
    now = _i2d_wcs().pixel_to_world(np.asarray(x, float), np.asarray(y, float))
    return tbl['skycoord_fit'].separation(now).to('mas').value


def test_read_maskable_rejected_returns_the_flagged_rows_on_the_current_frame(tmp_path):
    frame = _rejected_file(tmp_path, FRAME_A, 'm7', [(10.0, 10.0), (30.0, 20.0), (50.0, 40.0)],
                           [20.0, 20.0, 45.0], flag=[True, False, True])
    got = fm.read_maskable_rejected(str(tmp_path), 'F212N', PID, FIELD, phase='m7',
                                    frames=[frame])
    assert len(got) == 2
    assert list(got['sat_area']) == [20.0, 45.0]
    # stored under the moved WCS, served on the frame's current one
    assert np.all(_sky_sep_mas(got, [10.0, 50.0], [10.0, 40.0]) < 0.01)
    assert np.all(_moved_wcs().pixel_to_world([10.0, 50.0], [10.0, 40.0])
                  .separation(got['skycoord_fit']).to('mas').value > 50)


def test_read_maskable_rejected_follows_the_phase_of_the_merge(tmp_path):
    frame = _rejected_file(tmp_path, FRAME_A, 'm6', [(5.0, 5.0)], [20.0])
    _rejected_file(tmp_path, FRAME_A, 'm7', [(10.0, 10.0)], [20.0])
    got = fm.read_maskable_rejected(str(tmp_path), 'F212N', PID, FIELD, phase='m7',
                                    frames=[frame])
    assert np.all(_sky_sep_mas(got, [10.0], [10.0]) < 0.01) and len(got) == 1
    got = fm.read_maskable_rejected(str(tmp_path), 'F212N', PID, FIELD, phase='m6',
                                    frames=[frame])
    assert np.all(_sky_sep_mas(got, [5.0], [5.0]) < 0.01) and len(got) == 1


def test_read_maskable_rejected_follows_the_frame_variant_of_the_merge(tmp_path):
    frame_a = _rejected_file(tmp_path, FRAME_A, 'm7', [(10.0, 10.0)], [20.0])
    frame_b = _rejected_file(tmp_path, FRAME_A_ALIGN, 'm7', [(20.0, 20.0)], [20.0])
    got = fm.read_maskable_rejected(str(tmp_path), 'F212N', PID, FIELD, phase='m7',
                                    frames=[frame_a])
    assert len(got) == 1 and np.all(_sky_sep_mas(got, [10.0], [10.0]) < 0.01)
    got = fm.read_maskable_rejected(str(tmp_path), 'F212N', PID, FIELD, phase='m7',
                                    frames=[frame_b])
    assert len(got) == 1 and np.all(_sky_sep_mas(got, [20.0], [20.0]) < 0.01)


def test_read_maskable_rejected_is_scoped_to_the_observation(tmp_path):
    frame = _rejected_file(tmp_path, FRAME_A, 'm7', [(10.0, 10.0)], [20.0])
    _rejected_file(tmp_path, FRAME_OTHER_OBS, 'm7', [(40.0, 40.0)], [20.0])
    got = fm.read_maskable_rejected(str(tmp_path), 'F212N', PID, FIELD, phase='m7',
                                    frames=[frame])
    assert len(got) == 1 and np.all(_sky_sep_mas(got, [10.0], [10.0]) < 0.01)


def test_read_maskable_rejected_masks_nothing_of_a_file_written_without_the_flag(tmp_path):
    frame = _rejected_file(tmp_path, FRAME_A, 'm7', [(10.0, 10.0)], [20.0])
    # an exposure an older satstar run wrote: no flag column
    _rejected_file(tmp_path, FRAME_B, 'm7', [(40.0, 40.0)], [20.0], flag=None)
    got = fm.read_maskable_rejected(str(tmp_path), 'F212N', PID, FIELD, phase='m7',
                                    frames=[frame])
    assert len(got) == 1 and np.all(_sky_sep_mas(got, [10.0], [10.0]) < 0.01)


def test_read_maskable_rejected_with_every_row_unflagged_returns_an_empty_table(tmp_path):
    frame = _rejected_file(tmp_path, FRAME_A, 'm7', [(10.0, 10.0)], [20.0], flag=False)
    got = fm.read_maskable_rejected(str(tmp_path), 'F212N', PID, FIELD, phase='m7',
                                    frames=[frame])
    assert len(got) == 0


def test_read_maskable_rejected_without_the_flag_column_names_it(tmp_path):
    frame = _rejected_file(tmp_path, FRAME_A, 'm7', [(10.0, 10.0)], [20.0], flag=None)
    with pytest.raises(fm.FootprintMaskUnavailable, match=fm.MASKABLE_COLUMN):
        fm.read_maskable_rejected(str(tmp_path), 'F212N', PID, FIELD, phase='m7',
                                  frames=[frame])


def test_read_maskable_rejected_without_rejected_files_says_so(tmp_path):
    with pytest.raises(fm.FootprintMaskUnavailable, match='no gate-rejected satstar files'):
        fm.read_maskable_rejected(str(tmp_path), 'F212N', PID, FIELD, phase='m7')


def test_read_maskable_rejected_reads_the_flux_column_only_for_the_ring_extension(tmp_path):
    frame = _rejected_file(tmp_path, FRAME_A, 'm7', [(10.0, 10.0)], [20.0])
    assert len(fm.read_maskable_rejected(str(tmp_path), 'F212N', PID, FIELD, phase='m7',
                                         frames=[frame])) == 1
    with pytest.raises(fm.FootprintMaskUnavailable, match='flux_fit'):
        fm.read_maskable_rejected(str(tmp_path), 'F212N', PID, FIELD, phase='m7',
                                  frames=[frame], need_flux=True)


def test_read_maskable_rejected_leaves_the_primary_catalog_to_the_merge(tmp_path, monkeypatch):
    frame = _rejected_file(tmp_path, FRAME_A, 'm7', [(10.0, 10.0)], [20.0])
    monkeypatch.setattr(merge_catalogs, '_obs_filters_for', lambda t: {'10678': ['f212n']})
    monkeypatch.setattr(merge_catalogs, '_primary_satstar_catalog_path',
                        lambda *a, **k: '/x/primary_satstar_catalog.fits')
    with pytest.raises(fm.FootprintMaskUnavailable, match='primary_satstar_catalog'):
        fm.read_maskable_rejected(str(tmp_path), 'F212N', PID, FIELD, target='gc-treasury',
                                  phase='m7', frames=[frame])


@pytest.mark.parametrize('target, passed', [('gc-treasury', {'target': 'gc-treasury'}),
                                            (None, {}), ('', {})])
def test_read_maskable_rejected_hands_the_merges_scope_to_the_loader(monkeypatch, target, passed):
    seen = {}

    def _loader(filtername, **kwargs):
        seen.update(kwargs, filtername=filtername)
        return _flagged(_i2d_wcs(), [(10.0, 10.0)], [20.0], [True])

    monkeypatch.setattr(merge_catalogs, '_obs_filters_for', lambda t: None)
    monkeypatch.setattr(merge_catalogs, 'load_rejected_satstar_catalog', _loader)
    got = fm.read_maskable_rejected(
        '/base', 'F212N', PID, FIELD, target=target, phase='m7',
        frames=[f'/p/{FRAME_A}.fits', f'/p/{FRAME_A_ALIGN}.fits', '/p/not_an_exposure.fits'])
    assert len(got) == 1
    assert seen == dict(passed, filtername='F212N', basepath='/base', proposal_id=PID,
                        field=FIELD, phase='m7', refresh_sky=True,
                        frame_variants=frozenset({'destreak_o105_crf', 'align_o105_crf'}))


# --- build_mergedcat_residuals -> the rejected fits --------------------------------------

@pytest.fixture
def _registered_target(monkeypatch):
    """The render fixture's target ('e2e') is not in the target registry, and the loader of the
    rejected files looks its primary catalog up there; a real run's target is always registered.
    Register it with no filters: it then has no primary catalog."""
    monkeypatch.setattr(merge_catalogs, '_obs_filters_for', lambda target: {})


#: a flagged and an unflagged rejected fit of the first frame, beside the two consolidated rows
#: at (30, 25) and (55.3, 40.6) of ``_build``
def _rejected_of_build(flux=None):
    return _flagged(_i2d_wcs(), [(12.0, 45.0), (70.0, 10.0)], [30.0, 30.0], [True, False],
                    flux=flux)


@pytest.mark.usefixtures('_registered_target')
def test_the_rejected_extension_blanks_the_flagged_rejected_fits_in_the_render(
        tmp_path, monkeypatch, capsys):
    path, out = _build(tmp_path / 'r1', monkeypatch, capsys, rejected=_rejected_of_build(),
                       env={fm.ENV_KFAC: '1.5', fm.ENV_MARGIN: '1', fm.ENV_REJECTED: '1'})
    r = [1.5 * math.sqrt(a / math.pi) + 1.0 for a in (20.0, 45.0, 30.0)]
    expected = _disk(30.0, 25.0, r[0]) | _disk(55.3, 40.6, r[1]) | _disk(12.0, 45.0, r[2])
    with fits.open(path) as h:
        assert np.array_equal(np.isnan(h['SCI'].data), expected)
        assert h['SCI'].header['FPMASKN'] == 2 and h['SCI'].header['FPREJN'] == 1
    assert f"and of every gate-rejected satstar fit flagged {fm.MASKABLE_COLUMN}" in out
    assert f"1 gate-rejected satstar fit(s) flagged {fm.MASKABLE_COLUMN}" in out
    assert 'around 2 satstar rows and 1 gate-rejected fit(s)' in out


@pytest.mark.usefixtures('_registered_target')
def test_the_rejected_fits_stay_in_the_image_while_the_extension_is_off(tmp_path, monkeypatch,
                                                                       capsys):
    path, out = _build(tmp_path / 'r2', monkeypatch, capsys, rejected=_rejected_of_build(),
                       env={fm.ENV_KFAC: '1.5', fm.ENV_MARGIN: '1'})
    r = [1.5 * math.sqrt(a / math.pi) + 1.0 for a in (20.0, 45.0)]
    with fits.open(path) as h:
        assert np.array_equal(np.isnan(h['SCI'].data),
                              _disk(30.0, 25.0, r[0]) | _disk(55.3, 40.6, r[1]))
        assert 'FPREJN' not in h['SCI'].header
    assert 'gate-rejected' not in out


@pytest.mark.usefixtures('_registered_target')
def test_the_rejected_extension_without_the_flag_column_warns_and_masks_the_consolidated_rows(
        tmp_path, monkeypatch, capsys):
    rejected = _rejected_of_build()
    rejected.remove_column(fm.MASKABLE_COLUMN)
    path, out = _build(tmp_path / 'r3', monkeypatch, capsys, rejected=rejected,
                       env={fm.ENV_KFAC: '1.5', fm.ENV_MARGIN: '1', fm.ENV_REJECTED: '1'})
    r = [1.5 * math.sqrt(a / math.pi) + 1.0 for a in (20.0, 45.0)]
    with fits.open(path) as h:
        assert np.array_equal(np.isnan(h['SCI'].data),
                              _disk(30.0, 25.0, r[0]) | _disk(55.3, 40.6, r[1]))
        assert 'FPREJN' not in h['SCI'].header
    assert 'WARNING mergedcat: resid i2d footprint mask: the gate-rejected satstar fits are ' \
           f'skipped' in out
    assert fm.MASKABLE_COLUMN in out


@pytest.mark.usefixtures('_registered_target')
def test_the_rejected_extension_without_rejected_files_warns_and_masks_the_consolidated_rows(
        tmp_path, monkeypatch, capsys):
    path, out = _build(tmp_path / 'r4', monkeypatch, capsys,
                       env={fm.ENV_KFAC: '1.5', fm.ENV_MARGIN: '1', fm.ENV_REJECTED: '1'})
    with fits.open(path) as h:
        assert h['SCI'].header['FPMASKN'] == 2 and 'FPREJN' not in h['SCI'].header
        assert np.isnan(h['SCI'].data).any()
    assert 'the gate-rejected satstar fits are skipped: no gate-rejected satstar files' in out


@pytest.mark.usefixtures('_registered_target')
def test_the_rejected_ring_extension_needs_the_flux_of_the_rejected_fits(tmp_path, monkeypatch,
                                                                         capsys):
    path, out = _build(tmp_path / 'r5', monkeypatch, capsys, rejected=_rejected_of_build(),
                       env={fm.ENV_KFAC: '1', fm.ENV_REJECTED: '1',
                            fm.ENV_RING_FLUX_MIN: '1e6', fm.ENV_RING_RADIUS_PX: '12'},
                       flux=[2e6, 1e5])
    with fits.open(path) as h:
        assert 'FPREJN' not in h['SCI'].header
        assert h['SCI'].header['FPRINGN'] == 1
    assert 'the gate-rejected satstar fits are skipped' in out and 'flux_fit' in out


@pytest.mark.usefixtures('_registered_target')
def test_the_rejected_ring_extension_widens_the_bright_rejected_fit(tmp_path, monkeypatch,
                                                                     capsys):
    path, out = _build(tmp_path / 'r6', monkeypatch, capsys,
                       rejected=_rejected_of_build(flux=[3e6, 3e6]),
                       env={fm.ENV_KFAC: '1', fm.ENV_MARGIN: '1', fm.ENV_REJECTED: '1',
                            fm.ENV_RING_FLUX_MIN: '1e6', fm.ENV_RING_RADIUS_PX: '9.3'},
                       flux=[2e6, 1e5])
    small = math.sqrt(45.0 / math.pi) + 1.0
    # the unflagged rejected fit at (70, 10) is not masked however bright it is
    expected = _disk(30.0, 25.0, 9.3) | _disk(55.3, 40.6, small) | _disk(12.0, 45.0, 9.3)
    with fits.open(path) as h:
        assert np.array_equal(np.isnan(h['SCI'].data), expected)
        assert h['SCI'].header['FPRINGN'] == 2


@pytest.mark.usefixtures('_registered_target')
def test_a_typo_in_the_rejected_switch_fails_before_any_residual_is_rendered(
        tmp_path, monkeypatch, capsys):
    written = []
    with pytest.raises(ValueError, match=fm.ENV_REJECTED):
        _build(tmp_path / 'r7', monkeypatch, capsys, written=written,
               env={fm.ENV_KFAC: '1', fm.ENV_REJECTED: 'maybe'})
    assert written == []


@pytest.mark.usefixtures('_registered_target')
def test_the_rejected_switch_does_nothing_while_the_mask_is_off(tmp_path, monkeypatch, capsys):
    path, out = _build(tmp_path / 'r8', monkeypatch, capsys, rejected=_rejected_of_build(),
                       env={fm.ENV_REJECTED: 'maybe'})
    with fits.open(path) as h:
        assert np.all(h['SCI'].data == 1.0)
    assert 'footprint mask' not in out


@pytest.mark.usefixtures('_registered_target')
def test_the_rejected_files_are_not_read_for_a_pass_the_mask_skips(tmp_path, monkeypatch,
                                                                  capsys):
    def _never(*a, **k):
        raise AssertionError('the rejected files are read for the masked pass only')

    monkeypatch.setattr(merge_catalogs, 'load_rejected_satstar_catalog', _never)
    path, out = _build(tmp_path / 'r9', monkeypatch, capsys, iteration_label='m3',
                       rejected=_rejected_of_build(),
                       env={fm.ENV_KFAC: '1', fm.ENV_REJECTED: '1'})
    with fits.open(path) as h:
        assert np.all(h['SCI'].data == 1.0)
