"""Regression tests for cataloging.py (manual-iteration PSF photometry path).

Pins the MIRI-vs-NIRCam divergence in ``_filter_extended_emission`` so the
Phase-4 untangle (splitting the two keep-logics) can't change behavior:

* NIRCam path (``min_prominence == 0``): keep = star_like (qfit/flags/peakSB)
  AND local-SNR cut.  A high-qfit low-SNR source is dropped.
* MIRI path (``min_prominence > 0``): the deep-i2d prominence is the SOLE
  discriminator; star_like/SNR are BYPASSED (a bright real MIRI star on
  extended emission has bad qfit but high prominence -> kept; a flat-emission
  bump has low/NaN prominence -> dropped, including off-i2d sources).

Importing cataloging pulls crowdsource_catalogs_long (webbpsf) -> slow cold.
"""
import numpy as np
import pytest

from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

from jwst_gc_pipeline.photometry import cataloging as C


def _wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crpix = [30.5, 30.5]
    w.wcs.crval = [266.55, -28.80]
    w.wcs.cdelt = [-1.0 / 3600, 1.0 / 3600]
    w.wcs.cunit = ['deg', 'deg']
    return w


class TestFilterExtendedEmissionNircam:
    def test_keeps_starlike_drops_high_qfit_low_snr(self):
        # No i2d image -> peakSB/prominence skipped; NIRCam keep-logic only.
        t = Table({
            'qfit': [0.1, 0.5],            # row0 star-like, row1 not
            'flags': [0, 0],
            'flux': [100.0, 100.0],
            'flux_err': [10.0, 10.0],      # snr = 10 for both (>= local_snr_min)
        })
        out = C._filter_extended_emission(t, min_prominence=0.0,
                                          qfit_max=0.2, local_snr_min=5.0,
                                          label='nircam-test')
        assert len(out) == 1
        np.testing.assert_allclose(out['qfit'][0], 0.1)

    def test_low_snr_dropped_even_if_starlike(self):
        t = Table({
            'qfit': [0.1],
            'flags': [0],
            'flux': [10.0],
            'flux_err': [10.0],            # snr = 1 < local_snr_min
        })
        out = C._filter_extended_emission(t, min_prominence=0.0,
                                          qfit_max=0.2, local_snr_min=5.0,
                                          label='nircam-test')
        assert len(out) == 0


class TestFilterExtendedEmissionMiri:
    def _image_and_catalog(self):
        ww = _wcs()
        ny = nx = 60
        yy, xx = np.mgrid[0:ny, 0:nx]
        img = np.ones((ny, nx)) + 0.01 * np.sin(xx) * np.cos(yy)  # texture -> MAD>0
        # bright narrow peak at (30, 30) -> high prominence
        img += 200.0 * np.exp(-(((xx - 30) ** 2 + (yy - 30) ** 2) / 2.0))
        # source A on the peak, source B on flat texture far from any peak
        sc = SkyCoord([ww.pixel_to_world(30, 30), ww.pixel_to_world(20, 20)])
        t = Table({
            'qfit': [0.9, 0.9],            # both bad qfit: NIRCam path would drop both
            'flags': [0, 0],
            'flux': [1.0, 1.0],
            'flux_err': [1.0, 1.0],        # snr 1: NIRCam path would drop both
            'skycoord': sc,
        })
        return img, ww, t

    def test_prominence_alone_keeps_real_star_bypasses_starlike(self):
        img, ww, t = self._image_and_catalog()
        out = C._filter_extended_emission(t, data_i2d_image=img, ww_i2d=ww,
                                          min_prominence=10.0, label='miri-test')
        # Only the prominent source A survives; star_like/snr are bypassed.
        assert len(out) == 1
        # surviving source is the one at the peak (RA/Dec of pixel 30,30)
        kept = SkyCoord(out['skycoord'])
        peak = ww.pixel_to_world(30, 30)
        assert kept[0].separation(peak).arcsec < 0.5

    def test_off_i2d_nan_prominence_dropped(self):
        # A source whose prominence cannot be measured (NaN) must be dropped
        # by the MIRI gate (off-i2d / other-obs footprint).
        img, ww, t = self._image_and_catalog()
        out = C._filter_extended_emission(t, data_i2d_image=img, ww_i2d=ww,
                                          min_prominence=10.0, label='miri-test')
        kept = SkyCoord(out['skycoord'])
        flat = ww.pixel_to_world(20, 20)
        assert all(k.separation(flat).arcsec > 0.5 for k in kept)
