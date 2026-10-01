"""Vetting bright-star branch: data-i2d prominence in place of peak_SB.

The old branch keeps a source when peak_SB > peak_over_bkg * local_bkg AND
local_bkg > 0.  local_bkg is fit on background-subtracted frames and scatters
about zero, so the sign of local_bkg decides it: a clean star with a slightly
negative local_bkg is dropped and an emission knot with a slightly positive
local_bkg is kept.  ``star_prom_min > 0`` tests the rise above the local
annulus instead; sources without a measured prominence keep the old test.
"""
import numpy as np
from astropy.table import Table
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter

from jwst_gc_pipeline.photometry.cataloging import _filter_extended_emission
from jwst_gc_pipeline.photometry.manual_defaults import MANUAL_DEFAULTS

NY = NX = 200
STAR, KNOT, EDGE = (50, 50), (150, 150), (4, 100)


def _wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crpix = [NX / 2, NY / 2]
    w.wcs.crval = [266.5, -28.7]
    w.wcs.cdelt = [-0.063 / 3600, 0.063 / 3600]
    return w


def _gauss(data, x, y, amp, sig=1.0):
    yy, xx = np.mgrid[0:NY, 0:NX]
    data += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sig ** 2))


def _image(seed=3):
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 1.0, (NY, NX))
    _gauss(data, *STAR, amp=30.0)
    # the knot: one peak of PSF-scale structure inside a bright patch, no
    # higher than its neighbours
    struct = gaussian_filter(rng.normal(0, 1, (NY, NX)), 1.2)
    struct *= 8.0 / np.std(struct)
    patch = (slice(KNOT[1] - 18, KNOT[1] + 18), slice(KNOT[0] - 18, KNOT[0] + 18))
    data[patch] += 40.0 + struct[patch]
    _gauss(data, *KNOT, amp=10.0)
    _gauss(data, *EDGE, amp=30.0)
    return data


def _run(star_prom_min):
    w = _wcs()
    xy = np.array([STAR, KNOT, EDGE], float)
    cat = Table({'skycoord': w.pixel_to_world(xy[:, 0], xy[:, 1]),
                 # all fail qfit <= 0.2; S/N 8 clears local_snr_min 5 but not
                 # the bright-isolated keep
                 'qfit': np.full(3, 0.5), 'flags': np.zeros(3),
                 # star: local_bkg slightly negative; knot + edge: slightly positive
                 'local_bkg': np.array([-0.05, 0.05, 0.05]),
                 'flux': np.full(3, 80.0), 'flux_err': np.full(3, 10.0),
                 'group_size': np.ones(3), 'id': np.arange(3)})
    out = _filter_extended_emission(cat, data_i2d_image=_image(), ww_i2d=w,
                                    star_prom_min=star_prom_min,
                                    sky_clean_keep=False, label='test')
    return set(np.asarray(out['id']).tolist()), cat


def test_peak_sb_branch_follows_local_bkg_sign():
    kept, cat = _run(0.0)
    assert kept == {1, 2}            # star dropped, knot kept
    prom = np.asarray(cat['prominence'])
    assert prom[0] > 10 and prom[1] < 5 and not np.isfinite(prom[2])


def test_prominence_branch_keeps_star_drops_knot():
    kept, _ = _run(5.0)
    assert 0 in kept and 1 not in kept


def test_unmeasured_prominence_keeps_peak_sb_test():
    # the edge source (within 10 px of the border) has no prominence and
    # passes peak_SB; the prominence branch leaves it to that test
    kept, _ = _run(5.0)
    assert 2 in kept


def test_pipeline_default_on():
    assert MANUAL_DEFAULTS['manual_ext_star_prom_min'] == 5.0
