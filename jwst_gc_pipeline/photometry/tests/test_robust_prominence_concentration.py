"""The neighbour-robust prominence branch refuses fits whose flux is not
concentrated in a PSF core.

Next to a bright star the 25th-percentile annulus floor reads the dark side of
the star's wing, so a fit to a bump in that wing reads a high robust
prominence.  The fit's flux is spread over the bump: its data-i2d core flux per
unit fitted flux is well below that of the field's prominent stars.
"""
import numpy as np
from astropy.table import Table
from astropy.wcs import WCS

from jwst_gc_pipeline.photometry.cataloging import (_core_concentration,
                                                    _filter_extended_emission)
from jwst_gc_pipeline.photometry.manual_defaults import MANUAL_DEFAULTS

NY = NX = 160
CAL = [(25, 25), (60, 25), (95, 25), (130, 25), (25, 60), (130, 60)]
STAR, BUMP = (50, 110), (110, 110)


def _wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crpix = [NX / 2, NY / 2]
    w.wcs.crval = [266.5, -28.7]
    w.wcs.cdelt = [-0.031 / 3600, 0.031 / 3600]
    return w


def _gauss(data, x, y, amp, sig):
    yy, xx = np.mgrid[0:NY, 0:NX]
    data += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sig ** 2))


def _run(conc, conc_ref_min_n=5):
    """Six prominent PSF stars (sigma 1 px), a faint PSF star and a broad bump
    (sigma 2.5 px) fit with a PSF-sized flux.  Faint star and bump fail qfit."""
    rng = np.random.default_rng(4)
    data = rng.normal(0.0, 1.0, (NY, NX))
    for x, y in CAL:
        _gauss(data, x, y, 60.0, 1.0)
    _gauss(data, *STAR, 4.5, 1.0)
    _gauss(data, *BUMP, 4.0, 2.5)
    xy = np.array(CAL + [STAR, BUMP], float)
    n = len(xy)
    flux = np.r_[np.full(len(CAL), 2 * np.pi * 60.0), 2 * np.pi * 4.5, 60.0]
    w = _wcs()
    cat = Table({'id': np.arange(n), 'skycoord': w.pixel_to_world(xy[:, 0], xy[:, 1]),
                 'qfit': np.r_[np.full(len(CAL), 0.05), 0.5, 0.5],
                 'flags': np.zeros(n), 'local_bkg': np.zeros(n),
                 'flux': flux, 'flux_err': flux / 10.0, 'group_size': np.ones(n)})
    out = _filter_extended_emission(cat, data_i2d_image=data, ww_i2d=w,
                                    star_prom_min=5.0, star_prom_robust_min=4.0,
                                    star_prom_robust_conc=conc,
                                    conc_ref_min_n=conc_ref_min_n,
                                    sky_clean_keep=False, label='test')
    return set(np.asarray(out['id']).tolist()), cat


def test_core_concentration_of_a_psf():
    rng = np.random.default_rng(0)
    data = rng.normal(0.0, 0.01, (NY, NX))
    _gauss(data, 80.3, 79.6, 100.0, 1.0)
    core, sig = _core_concentration(data, np.array([80.3, 5.0]), np.array([79.6, 80.0]))
    # the r <= 1.5 px core of a sigma = 1 px Gaussian centred off-pixel holds
    # ~2/3 of the flux
    assert 0.55 < core[0] / (2 * np.pi * 100.0) < 0.8
    assert 0 < sig[0] < 0.1
    assert np.isnan(core[1]) and np.isnan(sig[1])     # within 10 px of the edge


def test_robust_branch_refuses_unconcentrated_fit():
    star, bump = len(CAL), len(CAL) + 1
    kept, cat = _run(0.0)
    prom = np.asarray(cat['prominence'])
    rob = np.asarray(cat['prominence_robust'])
    # both pass only through the robust branch
    assert np.all(prom[[star, bump]] < 5) and np.all(rob[[star, bump]] >= 4)
    assert {star, bump} <= kept
    assert 'core_concentration' not in cat.colnames
    kept, cat = _run(0.6)
    conc = np.asarray(cat['core_concentration'])
    assert np.all(np.abs(conc[:len(CAL)] - 1) < 0.1)
    assert conc[star] > 0.6 and conc[bump] < 0.6
    assert star in kept and bump not in kept
    assert set(range(len(CAL))) <= kept


def test_guard_off_without_calibration_stars():
    kept, cat = _run(0.6, conc_ref_min_n=len(CAL) + 1)
    assert len(CAL) + 1 in kept
    assert 'core_concentration' not in cat.colnames


def test_pipeline_default():
    assert MANUAL_DEFAULTS['manual_ext_star_prom_robust_conc'] == 0.6


def test_pipeline_defaults_without_data_i2d():
    """No data i2d (or no skycoord): prominence is NaN everywhere, every
    source keeps the peak_SB test, and the concentration guard is skipped."""
    n = 4
    cat = Table({'id': np.arange(n), 'qfit': np.array([0.05, 0.5, 0.5, 0.5]),
                 'flags': np.zeros(n), 'local_bkg': np.array([1.0, 1.0, 1.0, -1.0]),
                 'peak_sb': np.array([5.0, 50.0, 5.0, 50.0]),
                 'flux': np.full(n, 100.0), 'flux_err': np.full(n, 10.0),
                 'group_size': np.ones(n)})
    md = MANUAL_DEFAULTS
    out = _filter_extended_emission(
        cat, data_i2d_image=None, ww_i2d=None,
        star_prom_min=md['manual_ext_star_prom_min'],
        star_prom_robust_min=8.0,
        star_prom_robust_conc=md['manual_ext_star_prom_robust_conc'],
        label='test')
    assert np.all(np.isnan(np.asarray(cat['prominence'])))
    assert 'core_concentration' not in cat.colnames
    assert 0 in set(np.asarray(out['id']).tolist())
