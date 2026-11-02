"""The smoothed-residual background box must be wide enough to reject a faint
star that is missing from the source mask (#1039).

``_build_source_masked_bg`` masks the catalogued sources and median-filters
the residual mosaic.  With the historical 3 px box, a 2 px FWHM star that is in
neither the vetted catalog nor the seed keeps about half its peak in the
"background"; the next phase subtracts it and the star's fit comes out ~1 mag
faint (wd2: 6.8% of dolphot-matched band magnitudes).  A box of ~7 FWHM
(``median_size <= 0``) rejects the star and still follows diffuse emission on
scales larger than the box.
"""
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from jwst_gc_pipeline.photometry.cataloging import (
    _build_source_masked_bg, _resolve_residual_bg_median_size)
from jwst_gc_pipeline.photometry.manual_defaults import MANUAL_DEFAULTS

FILT = 'F200W'          # present in reduction/fwhm_table.ecsv
PIX_AS = 0.031          # SW i2d pixel scale
DIFFUSE = 5.0
PEAK = 100.0


def _wcs():
    w = WCS(naxis=2)
    w.wcs.crpix = [100, 100]
    w.wcs.crval = [161.03, -59.75]
    w.wcs.cdelt = [-PIX_AS / 3600, PIX_AS / 3600]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    return w


def _gauss(shape, x0, y0, amp, sig):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return amp * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sig ** 2))


def _write(tmp_path, nebula_amp=0.0):
    w = _wcs()
    shape = (200, 200)
    fwhm_px = 0.066 / PIX_AS
    star_xy = (120.0, 80.0)
    data = (np.full(shape, DIFFUSE, dtype=float)
            + _gauss(shape, *star_xy, PEAK, fwhm_px / 2.3548)
            + _gauss(shape, 100.0, 100.0, nebula_amp, 25.0))
    mc = str(tmp_path / 'sim_clear-f200w-nrca_resbgsub_m6_daophot_basic_mergedcat_residual_i2d.fits')
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(data=data.astype('float32'), header=w.to_header(),
                                name='SCI')]).writeto(mc)
    vet = str(tmp_path / 'vetted.fits')
    # an empty vetted catalog: the star is uncatalogued
    Table({'skycoord': w.pixel_to_world(np.array([10.0]), np.array([10.0]))}
          ).write(vet, overwrite=True)
    return mc, vet, star_xy


def _at(path, xy):
    return float(fits.getdata(path)[int(round(xy[1])), int(round(xy[0]))])


def test_resolve_median_size():
    assert _resolve_residual_bg_median_size(3, 2.1) == 3
    assert _resolve_residual_bg_median_size(25, 2.1) == 25
    # PSF-scaled: 7 x FWHM, rounded up to odd
    assert _resolve_residual_bg_median_size(0, 2.13) == 15
    assert _resolve_residual_bg_median_size(0, 2.4) == 17
    assert _resolve_residual_bg_median_size(-1, 2.4) == 17
    assert _resolve_residual_bg_median_size(None, 0.1) == 3
    for f in np.linspace(0.5, 6, 23):
        assert _resolve_residual_bg_median_size(0, f) % 2 == 1


def test_default_is_unchanged():
    assert MANUAL_DEFAULTS['manual_residual_bg_median_size'] == 3


def test_box3_absorbs_uncatalogued_star(tmp_path):
    """The bug: a 3 px median keeps about half of an uncatalogued star's peak."""
    mc, vet, star_xy = _write(tmp_path)
    out = _build_source_masked_bg(mc, vet, FILT, median_size=3)
    assert _at(out, star_xy) - DIFFUSE > 0.3 * PEAK


def test_psf_scaled_box_rejects_uncatalogued_star(tmp_path):
    mc, vet, star_xy = _write(tmp_path)
    out = _build_source_masked_bg(mc, vet, FILT, median_size=0)
    assert abs(_at(out, star_xy) - DIFFUSE) < 0.02 * PEAK


def test_psf_scaled_box_follows_diffuse_emission(tmp_path):
    """Emission on scales well above the box stays in the background."""
    mc, vet, star_xy = _write(tmp_path, nebula_amp=40.0)
    out = _build_source_masked_bg(mc, vet, FILT, median_size=0)
    bg = fits.getdata(out)
    assert abs(bg[100, 100] - (DIFFUSE + 40.0)) < 2.0
    assert abs(bg[100, 150] - (DIFFUSE + 40.0 * np.exp(-50 ** 2 / (2 * 25.0 ** 2)))) < 2.0
