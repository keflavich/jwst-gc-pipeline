"""The vetting qfit gate carries a pixel-noise term.

qfit = sum|resid| / flux.  For a PERFECT PSF fit the residual is pixel noise,
so qfit ~ c / (S/N) with c ~ 3.4 (Brick F182M dark-sky stars, which the
sky-clean tier keeps without looking at qfit: median qfit x S/N 3.4, 90th
percentile 4.3 at S/N 5-10).  A flat qfit <= 0.2 is then reachable only at
S/N >~ 17, and every fainter real star is vetted out and left unsubtracted in
the residual.  The gate is now qfit <= sqrt(qfit_max^2 + (k/S/N)^2).
"""
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

from jwst_gc_pipeline.photometry.cataloging import _filter_extended_emission
from jwst_gc_pipeline.photometry.manual_defaults import MANUAL_DEFAULTS


def _mk(qfit, snr):
    n = len(qfit)
    return Table({
        'id': np.arange(n),
        'skycoord': SkyCoord((266.5 + np.arange(n) * 0.001) * u.deg,
                             np.full(n, -28.8) * u.deg),
        'qfit': np.asarray(qfit, float),
        'flux': np.asarray(snr, float) * 10.0,
        'flux_err': np.full(n, 10.0),
        'flags': np.zeros(n, int),
        'local_bkg': np.zeros(n),
    })


def _kept(t, k):
    out = _filter_extended_emission(t, qfit_max=0.2, local_snr_min=5.0,
                                    qfit_snr_k=k, sky_clean_keep=False)
    return np.isin(t['id'], out['id'])


def test_noise_limited_faint_star_kept():
    # S/N 8 star at qfit 3.6/8 = 0.45: the noise floor of a perfect fit
    t = _mk(qfit=[0.45], snr=[8.0])
    assert not _kept(t, 0.0)[0]          # flat cut deletes it
    assert _kept(t, 5.0)[0]              # ceiling sqrt(0.04 + 0.39) = 0.66


def test_bright_poor_fit_still_rejected():
    # at S/N 60 the noise term is 5/60 = 0.08: qfit 0.45 is PSF mismatch
    # (and above the bright-isolated keep's 0.4)
    t = _mk(qfit=[0.45], snr=[60.0])
    assert not _kept(t, 5.0)[0]


def test_noise_term_does_not_bypass_snr_floor():
    # qfit 1.0 at S/N 4 is inside the ceiling (sqrt(0.04 + 1.56) = 1.26) but
    # below local_snr_min = 5: only qfit <= qfit_max is S/N-exempt
    t = _mk(qfit=[1.0], snr=[4.0])
    assert not _kept(t, 5.0)[0]


def test_nonfinite_snr_gets_flat_cut():
    t = _mk(qfit=[0.5, 0.15], snr=[1.0, 1.0])
    t['flux_err'] = [np.nan, np.nan]
    kept = _kept(t, 5.0)
    assert not kept[0] and kept[1]


def test_k_zero_is_flat_cut():
    rng = np.random.default_rng(0)
    q = rng.uniform(0, 1, 200)
    s = rng.uniform(5, 19, 200)       # below the bright-isolated keep
    t = _mk(q, s)
    assert np.array_equal(_kept(t, 0.0), q <= 0.2)


def test_pipeline_default_on():
    assert MANUAL_DEFAULTS['manual_ext_qfit_snr_k'] == 5.0
