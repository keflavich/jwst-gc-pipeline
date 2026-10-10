"""Pixel-area correction of per-frame and satstar fluxes (``PHOT_PIXEL_AREA``,
issue #1151).

The PSF fit runs on MJy/sr with a unit-sum PSF and the merge calibrates with a
constant pixel area, so a star on a larger-than-nominal pixel reads faint by
``2.5 log10(AREA)``.  With the switch on, each per-frame catalogue and each
per-exposure satstar catalogue is scaled by ``AREA`` at the star's frame pixel
before the exposures are combined.
"""
import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from jwst_gc_pipeline.photometry import merge_catalogs as MC
from jwst_gc_pipeline.photometry import pixel_area as PA

SHAPE = (128, 128)
PIXSCALE = 31.2 / 3600.0 / 1000.0  # NIRCam SW, deg/px


def _area():
    yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1]]
    return (1.0 + 1e-3 * xx + 1e-4 * yy).astype('float32')


def _wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crpix = [64.5, 64.5]
    w.wcs.crval = [266.5, -28.7]
    w.wcs.cdelt = [-PIXSCALE, PIXSCALE]
    return w


def _write_frame(path, area=True, area_shape=SHAPE):
    hdus = [fits.PrimaryHDU(),
            fits.ImageHDU(np.zeros(SHAPE, dtype='float32'), name='SCI')]
    hdus[1].header.update(_wcs().to_header(relax=True))
    if area:
        data = _area() if area_shape == SHAPE else np.ones(area_shape, 'float32')
        hdus.append(fits.ImageHDU(data, name='AREA'))
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return str(path)


def _frame_table(x, y, frame):
    tbl = Table({'x_fit': np.asarray(x, float), 'y_fit': np.asarray(y, float),
                 'flux_fit': np.full(len(x), 100.0) * u.MJy / u.sr,
                 'flux_err': np.full(len(x), 2.0),
                 'flux_init': np.full(len(x), 90.0),
                 'local_bkg': np.full(len(x), 5.0)})
    tbl.meta['FILENAME'] = frame
    return tbl


@pytest.mark.parametrize('value, expected', [
    (None, False), ('0', False), ('', False), ('no', False),
    ('1', True), ('true', True), (' On ', True), ('YES', True)])
def test_switch_parsing(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(PA.PIXEL_AREA_ENV, raising=False)
    else:
        monkeypatch.setenv(PA.PIXEL_AREA_ENV, value)
    assert PA.pixel_area_enabled() is expected


def test_area_at_nearest_pixel_edge_clip_and_fallbacks():
    area = _area()
    area[10, 20] = np.nan
    area[11, 20] = 0.0
    x = np.array([5.4, 5.6, -3.0, 500.0, np.nan, 20.0, 20.0])
    y = np.array([7.0, 7.0, 2.0, 140.0, 3.0, 10.0, 11.0])
    got = PA.area_at(area, x, y)
    assert got[0] == pytest.approx(area[7, 5])
    assert got[1] == pytest.approx(area[7, 6])
    assert got[2] == pytest.approx(area[2, 0])
    assert got[3] == pytest.approx(area[127, 127])
    np.testing.assert_array_equal(got[4:], 1.0)


def test_flux_columns_scale_and_surface_brightness_does_not(tmp_path):
    frame = _write_frame(tmp_path / 'exp_crf.fits')
    x, y = np.array([10.0, 100.0]), np.array([20.0, 60.0])
    tbl = _frame_table(x, y, frame)
    factor = PA.apply_pixel_area(tbl, frame)
    expect = _area()[y.astype(int), x.astype(int)]
    np.testing.assert_allclose(factor, expect)
    np.testing.assert_allclose(tbl['flux_fit'].value, 100.0 * expect)
    assert tbl['flux_fit'].unit == u.MJy / u.sr
    np.testing.assert_allclose(tbl['flux_err'], 2.0 * expect)
    np.testing.assert_allclose(tbl['flux_init'], 90.0 * expect)
    np.testing.assert_array_equal(tbl['local_bkg'], 5.0)
    assert tbl.meta[PA.META_KEY] == 1


def test_correction_is_never_applied_twice(tmp_path):
    frame = _write_frame(tmp_path / 'exp_crf.fits')
    tbl = _frame_table([100.0], [100.0], frame)
    PA.apply_pixel_area(tbl, frame)
    once = np.array(tbl['flux_fit'])
    assert PA.apply_pixel_area(tbl, frame) is None
    np.testing.assert_array_equal(tbl['flux_fit'], once)


def test_masked_flux_keeps_its_mask(tmp_path):
    frame = _write_frame(tmp_path / 'exp_crf.fits')
    tbl = Table({'x_fit': [10.0, 20.0], 'y_fit': [10.0, 20.0]}, masked=True)
    tbl['flux_fit'] = np.ma.MaskedArray([1.0, 2.0], mask=[False, True])
    PA.apply_pixel_area(tbl, frame)
    assert list(tbl['flux_fit'].mask) == [False, True]


def test_missing_area_extension_is_an_error(tmp_path):
    frame = _write_frame(tmp_path / 'exp_crf.fits', area=False)
    with pytest.raises(PA.PixelAreaError, match='no AREA extension'):
        PA.apply_pixel_area(_frame_table([1.0], [1.0], frame), frame)


def test_area_shape_must_match_sci(tmp_path):
    frame = _write_frame(tmp_path / 'exp_crf.fits', area_shape=(64, 64))
    with pytest.raises(PA.PixelAreaError, match='differs from SCI'):
        PA.apply_pixel_area(_frame_table([1.0], [1.0], frame), frame)


def test_missing_frame_is_an_error(tmp_path):
    frame = str(tmp_path / 'gone_crf.fits')
    with pytest.raises(FileNotFoundError):
        PA.apply_pixel_area(_frame_table([1.0], [1.0], frame), frame)


def test_frames_need_their_filename(tmp_path):
    frame = _write_frame(tmp_path / 'exp_crf.fits')
    tbl = _frame_table([1.0], [1.0], frame)
    del tbl.meta['FILENAME']
    with pytest.raises(PA.PixelAreaError, match='FILENAME'):
        PA.apply_pixel_area_to_frames([tbl])


def test_each_frame_reads_its_own_area(tmp_path):
    f1 = _write_frame(tmp_path / 'a_crf.fits')
    f2 = str(tmp_path / 'b_crf.fits')
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(np.zeros(SHAPE, 'float32'), name='SCI'),
                  fits.ImageHDU(np.full(SHAPE, 1.02, 'float32'), name='AREA')]
                 ).writeto(f2)
    t1, t2 = _frame_table([50.0], [50.0], f1), _frame_table([50.0], [50.0], f2)
    PA.apply_pixel_area_to_frames([t1, t2])
    assert t1['flux_fit'][0] == pytest.approx(100.0 * _area()[50, 50])
    assert t2['flux_fit'][0] == pytest.approx(102.0)


def _satstar_catalog(tmp_path, x, y):
    """A per-exposure satstar catalogue as the fitter writes it: cutout-local
    x_fit/y_fit (about 81 px) and the frame position in xcentroid/ycentroid."""
    frame = _write_frame(tmp_path / 'exp_crf.fits')
    sky = _wcs().pixel_to_world(x, y)
    tbl = Table({'x_fit': np.full(len(x), 81.0), 'y_fit': np.full(len(x), 81.0),
                 'xcentroid': np.asarray(x, float),
                 'ycentroid': np.asarray(y, float),
                 'flux_fit': np.full(len(x), 1.0e6)})
    tbl['skycoord_fit'] = sky
    cat = str(tmp_path / 'exp_crf_m3_satstar_catalog.fits')
    tbl.write(cat, overwrite=True)
    return frame, cat


def test_satstar_reader_uses_the_frame_pixel(tmp_path, monkeypatch):
    monkeypatch.setenv(PA.PIXEL_AREA_ENV, '1')
    x, y = np.array([5.0, 120.0]), np.array([110.0, 15.0])
    _, cat = _satstar_catalog(tmp_path, x, y)
    got = MC._read_satstar_catalog_on_current_frame(cat, {})
    expect = _area()[y.astype(int), x.astype(int)]
    np.testing.assert_allclose(got['flux_fit'], 1.0e6 * expect, rtol=1e-6)


def test_satstar_reader_is_unchanged_when_off(tmp_path, monkeypatch):
    monkeypatch.delenv(PA.PIXEL_AREA_ENV, raising=False)
    _, cat = _satstar_catalog(tmp_path, np.array([120.0]), np.array([15.0]))
    got = MC._read_satstar_catalog_on_current_frame(cat, {})
    np.testing.assert_array_equal(got['flux_fit'], 1.0e6)


def test_switch_is_part_of_the_satstar_cache_key(monkeypatch):
    monkeypatch.delenv('SATSTAR_FLUX_STAT', raising=False)
    monkeypatch.delenv(PA.PIXEL_AREA_ENV, raising=False)
    off = MC._satstar_dedup_alg_tag()
    assert off == MC._SATSTAR_DEDUP_ALG
    monkeypatch.setenv(PA.PIXEL_AREA_ENV, '1')
    assert MC._satstar_dedup_alg_tag() == f'{off}-pam'
