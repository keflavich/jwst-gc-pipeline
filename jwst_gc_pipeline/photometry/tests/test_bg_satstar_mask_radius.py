"""The residual bg must be able to mask unmodeled saturated stars at a wider
radius than the 2 FWHM source disks.

Saturated stars the phase did not model keep PSF wings in the residual mosaic
out to ~8 px (F150W, FWHM 1.6 px).  The 2 FWHM disk leaves the wings in, the
interpolation fills the disk from them, and the bg bumps by 3-5% of the star's
flux; the next phase's hand-off fit then reads +0.05..+0.13 mag too faint.
``satstar_mask_radius_fwhm`` masks the per-frame satstar positions wider
(wd2 F150W: 3.75 FWHM puts those stars on the control scale).
"""
import os
import re

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from jwst_gc_pipeline.photometry.cataloging import (
    _build_source_masked_bg, _phase_satstar_product_paths)
from jwst_gc_pipeline.photometry.manual_defaults import MANUAL_DEFAULTS

FILT = 'F150W'          # present in reduction/fwhm_table.ecsv
PIX_AS = 0.031
FWHM_PX = 1.6           # approximate; the test only needs wings >> 2 FWHM
DIFFUSE = 5.0
PEAK = 2000.0


def _wcs():
    w = WCS(naxis=2)
    w.wcs.crpix = [100, 100]
    w.wcs.crval = [161.03, -59.75]
    w.wcs.cdelt = [-PIX_AS / 3600, PIX_AS / 3600]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    return w


def _write(tmp_path):
    """Flat bg plus a bright star whose wings (Gaussian core + broad halo,
    sigma 3.2 px ~ 2 FWHM) extend to ~4 FWHM."""
    w = _wcs()
    shape = (200, 200)
    star_xy = (120.0, 80.0)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    r2 = (xx - star_xy[0]) ** 2 + (yy - star_xy[1]) ** 2
    data = (np.full(shape, DIFFUSE)
            + PEAK * np.exp(-r2 / (2 * (FWHM_PX / 2.3548) ** 2))
            + 0.05 * PEAK * np.exp(-r2 / (2 * 3.2 ** 2)))
    mc = str(tmp_path / 'sim_clear-f150w-nrca_resbgsub_m6_daophot_basic_mergedcat_residual_i2d.fits')
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(data=data.astype('float32'), header=w.to_header(),
                                name='SCI')]).writeto(mc)
    vet = str(tmp_path / 'vetted.fits')
    # the star is in neither the vetted catalog nor the seed
    Table({'skycoord': w.pixel_to_world(np.array([10.0]), np.array([10.0]))}
          ).write(vet, overwrite=True)
    sat = str(tmp_path / 'frame_resbgsub_m6_satstar_catalog.fits')
    Table({'skycoord_fit': w.pixel_to_world(np.array([star_xy[0]]),
                                            np.array([star_xy[1]]))}
          ).write(sat, overwrite=True)
    return mc, vet, sat, star_xy


def _at(path, xy):
    return float(fits.getdata(path)[int(round(xy[1])), int(round(xy[0]))])


def test_default_is_bit_identical(tmp_path):
    mc, vet, sat, _ = _write(tmp_path)
    base = fits.getdata(_build_source_masked_bg(mc, vet, FILT)).copy()
    for kw in (dict(satstar_catalogs=[sat]),
               dict(satstar_mask_radius_fwhm=3.75),
               dict(satstar_catalogs=[sat], satstar_mask_radius_fwhm=0.0)):
        out = fits.getdata(_build_source_masked_bg(mc, vet, FILT, **kw))
        assert np.array_equal(out, base, equal_nan=True)


def test_satstar_mask_removes_wing_bump(tmp_path):
    mc, vet, sat, xy = _write(tmp_path)
    off = _build_source_masked_bg(mc, vet, FILT, satstar_catalogs=[sat],
                                  satstar_mask_radius_fwhm=0.0)
    bump_off = _at(off, xy) - DIFFUSE
    assert bump_off > 0.02 * PEAK
    on = _build_source_masked_bg(mc, vet, FILT, satstar_catalogs=[sat],
                                 satstar_mask_radius_fwhm=3.75)
    # near the surrounding level: the wing bump drops by >5x
    assert abs(_at(on, xy) - DIFFUSE) < 0.2 * bump_off


def test_missing_satstar_file_is_skipped(tmp_path):
    mc, vet, sat, _ = _write(tmp_path)
    base = fits.getdata(_build_source_masked_bg(mc, vet, FILT)).copy()
    out = _build_source_masked_bg(
        mc, vet, FILT, satstar_catalogs=[str(tmp_path / 'nope_satstar_catalog.fits')],
        satstar_mask_radius_fwhm=3.75)
    assert np.array_equal(fits.getdata(out), base, equal_nan=True)


def test_skycoord_fallback_and_nonfinite(tmp_path):
    """A catalog with only ``skycoord`` is used; NaN positions are skipped."""
    mc, vet, sat, xy = _write(tmp_path)
    w = _wcs()
    p = str(tmp_path / 'f2_resbgsub_m6_satstar_catalog.fits')
    Table({'skycoord': w.pixel_to_world(np.array([xy[0], np.nan]),
                                        np.array([xy[1], np.nan]))}).write(p)
    # both calls write the same output name, so read each value in turn
    bump_off = _at(_build_source_masked_bg(mc, vet, FILT), xy) - DIFFUSE
    out = _build_source_masked_bg(mc, vet, FILT, satstar_catalogs=[p],
                                  satstar_mask_radius_fwhm=3.75)
    assert abs(_at(out, xy) - DIFFUSE) < 0.2 * bump_off


def test_product_paths():
    paths = _phase_satstar_product_paths(
        ['/d/jw_nrca1_crf.fits'], '_resbgsub_m6')
    assert paths == ['/d/jw_nrca1_crf_resbgsub_m6_satstar_catalog.fits',
                     '/d/jw_nrca1_crf_resbgsub_m6_extended_satstar_catalog.fits',
                     '/d/jw_nrca1_crf_resbgsub_m6_satstar_rejected.fits']
    cut = _phase_satstar_product_paths(
        ['/d/jw_nrca1_crf.fits'], '_resbgsub_m6', cutout_label='c1',
        pipeline_dir='/p')
    assert cut[0] == '/p/jw_nrca1_crf_cutout_c1_resbgsub_m6_satstar_catalog.fits'
    assert _phase_satstar_product_paths([], '_m6') == []


def test_option_default_and_parser_flag():
    assert MANUAL_DEFAULTS['manual_residual_bg_satstar_mask_fwhm'] == 0.0
    # the parser is built inside main(); check the registration in source
    src = open(os.path.join(os.path.dirname(__file__), '..',
                            'crowdsource_catalogs_long.py')).read()
    m = re.search(r'add_option\("--residual-bg-satstar-mask-fwhm",(.*?)help=', src, re.S)
    assert m is not None
    assert 'manual_residual_bg_satstar_mask_fwhm' in m.group(1)
    assert 'type=float' in m.group(1)
