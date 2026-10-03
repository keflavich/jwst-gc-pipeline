"""``--inject-stars`` (photometry/injection.py): the reference-field truth.

The injected stars are the truth list every reference-field completeness
number is measured against, so the injection itself has to be exact: the
right sky position, the right flux in image units, the same pixels in every
phase, and no injection outside a cutout run.
"""
import warnings

import numpy as np
import pytest
from astropy.io import fits
from astropy.nddata import NDData
from astropy.table import Table
from astropy.wcs import WCS

from jwst_gc_pipeline.photometry import injection

SHAPE = (120, 140)
PIXAR_SR = 2.29e-14
FWHM = 2.0


def _wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crpix = [70.5, 60.5]
    w.wcs.crval = [266.55, -28.80]
    w.wcs.cdelt = [-0.031 / 3600, 0.031 / 3600]
    return w


def _grid():
    from photutils.psf import GriddedPSFModel
    stamp = 51
    yy, xx = np.mgrid[0:stamp, 0:stamp]
    c = (stamp - 1) / 2
    s = FWHM / 2.3548
    g = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * s ** 2))
    g /= g.sum()
    pos = [(0, 0), (0, SHAPE[0]), (SHAPE[1], 0), (SHAPE[1], SHAPE[0])]
    return GriddedPSFModel(NDData(np.array([g] * 4),
                                  meta={'grid_xypos': pos, 'oversampling': 1}))


def _frame(path, sky=5.0):
    hdr = _wcs().to_header()
    hdr['PIXAR_SR'] = PIXAR_SR
    sci = np.full(SHAPE, sky, dtype=np.float32)
    err = np.full(SHAPE, 0.5, dtype=np.float32)
    vp = np.full(SHAPE, 0.2, dtype=np.float32)
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(sci, header=hdr, name='SCI'),
                  fits.ImageHDU(err, name='ERR'),
                  fits.ImageHDU(vp, name='VAR_POISSON')]).writeto(path)
    return path


def _table(xy, flux_img, filt='F182M'):
    sky = _wcs().pixel_to_world(*np.array(xy).T)
    return Table({'ra': sky.ra.deg, 'dec': sky.dec.deg,
                  injection.flux_column(filt): np.asarray(flux_img) * 1e6 * PIXAR_SR})


def _inject(path, tbl, seed=0):
    rng = np.random.default_rng(injection.frame_seed(seed, 'jw_orig_nrca1_crf.fits'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return injection.inject_table_into_frame(path, tbl, _grid(), 'F182M', rng)


def test_injects_at_sky_position_with_image_unit_flux(tmp_path):
    path = _frame(tmp_path / 'f.fits')
    xy = [(40.3, 50.7), (95.6, 30.2)]
    flux = [400.0, 900.0]
    n = _inject(path, _table(xy, flux))
    assert n == 2
    with fits.open(path) as h:
        added = h['SCI'].data.astype(float) - 5.0
        assert h['SCI'].header['INJNSTAR'] == 2
        assert (h['ERR'].data >= 0.5).all()
    for (x, y), f in zip(xy, flux):
        box = added[int(y) - 10:int(y) + 11, int(x) - 10:int(x) + 11]
        assert box.sum() == pytest.approx(f, rel=0.05)
        cy, cx = np.unravel_index(np.argmax(box), box.shape)
        assert abs(cx + int(x) - 10 - x) <= 1 and abs(cy + int(y) - 10 - y) <= 1


def test_same_seed_same_pixels_every_phase(tmp_path):
    tbl = _table([(60.0, 60.0)], [500.0])
    a = _frame(tmp_path / 'a.fits')
    b = _frame(tmp_path / 'b.fits')
    _inject(a, tbl, seed=3)
    _inject(b, tbl, seed=3)
    assert np.array_equal(fits.getdata(a, 'SCI'), fits.getdata(b, 'SCI'))


def test_off_frame_and_missing_band(tmp_path):
    path = _frame(tmp_path / 'f.fits')
    assert _inject(path, _table([(-50.0, 10.0)], [500.0])) == 0
    with pytest.raises(ValueError, match='flux_jy_F182M'):
        _inject(path, _table([(60.0, 60.0)], [500.0], filt='F212N'))


def test_inject_requires_cutout(monkeypatch):
    """A full-frame run would rewrite production frame pixels."""
    from jwst_gc_pipeline.photometry import cataloging as C

    class Opt:
        inject_stars = 'x.ecsv'
        cutout_region = ''
        desaturated = False
        epsf = False
        blur = False
        group = False
        bgsub = False

    monkeypatch.setattr(C, 'Table', Table)
    with pytest.raises(ValueError, match='only allowed together with --cutout-region'):
        C._prepare_frame_for_photometry(
            Opt(), 'F182M', 'nrca1', '001', '/nonexistent', 'f.fits', '2221',
            exposurenumber=1, visit_id=1, vgroup_id='0310e', bg_boxsizes={},
            use_webbpsf=True, pupil='clear', resbg_path=None, satstar_label='')
