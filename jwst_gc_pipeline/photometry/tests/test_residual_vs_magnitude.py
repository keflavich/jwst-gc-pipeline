"""Position-vs-brightness systematics check (residual_vs_magnitude).

A NIRSpec pointing reference catalog must show NO positional trend with source
brightness (saturation-core bias, wing-fit substitution, nonlinearity all
imprint one).  Synthetic tests: clean catalogs pass; a bright-end step and a
smooth mas/mag drift are both flagged; the verified-global-tie precondition is
enforced.
"""
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from jwst_gc_pipeline.photometry.astrometry_offsets import (
    GlobalTieNotVerifiedError, measure_offset, residual_vs_magnitude,
)

RA0, DEC0 = 266.54, -28.70
COSD = np.cos(np.radians(DEC0))


def _cat(n=6000, seed=0, bright_step_mas=0.0, step_below_mag=12.0,
         slope_mas_per_mag=0.0, noise_mas=1.0):
    rng = np.random.default_rng(seed)
    ra = RA0 + (rng.random(n) - 0.5) * 0.02
    dec = DEC0 + (rng.random(n) - 0.5) * 0.02
    mag = rng.uniform(10.0, 20.0, n)
    dra = np.where(mag < step_below_mag, bright_step_mas, 0.0)
    dra = dra + slope_mas_per_mag * (mag - 15.0)
    ra_obs = ra + (dra + rng.normal(0, noise_mas, n)) / 3.6e6 / COSD
    dec_obs = dec + rng.normal(0, noise_mas, n) / 3.6e6
    cat = SkyCoord(ra_obs * u.deg, dec_obs * u.deg)
    ref = SkyCoord(ra * u.deg, dec * u.deg)
    return cat, ref, mag


def test_clean_catalog_passes():
    cat, ref, mag = _cat()
    g = measure_offset(cat, ref)
    r = residual_vs_magnitude(cat, ref, mag, g, context="clean")
    assert r["n_bins"] >= 5
    assert r["n_flagged"] == 0
    assert not r["slope_significant"]
    assert r["clean"]


def test_bright_end_step_flagged():
    # bright stars (mag < 12) sit 10 mas off: the satstar/wing-fit class
    cat, ref, mag = _cat(bright_step_mas=10.0)
    g = measure_offset(cat, ref)
    r = residual_vs_magnitude(cat, ref, mag, g, context="step")
    flagged = [b for b in r["bins"] if b["flagged"]]
    assert flagged, r
    assert all(b["mag_hi"] <= 12.5 for b in flagged)
    assert not r["clean"]


def test_smooth_drift_flagged_by_slope():
    # 1.5 mas/mag drift: each bin can hide under a per-bin tolerance while the
    # range accumulates 15 mas end to end -- the slope term must catch it
    cat, ref, mag = _cat(slope_mas_per_mag=1.5)
    g = measure_offset(cat, ref)
    r = residual_vs_magnitude(cat, ref, mag, g, tol_mas=8.0, context="drift")
    # residuals are (ref - cat) = the CORRECTION: a catalog drifting +1.5
    # mas/mag needs a -1.5 mas/mag correction
    assert r["slope_dra_mas_per_mag"] == pytest.approx(-1.5, abs=0.3)
    assert r["slope_significant"]
    assert not r["clean"]


def test_requires_verified_tie():
    cat, ref, mag = _cat()
    with pytest.raises(GlobalTieNotVerifiedError):
        residual_vs_magnitude(cat, ref, mag, None, context="no-tie")
    with pytest.raises(GlobalTieNotVerifiedError):
        residual_vs_magnitude(cat, ref, mag,
                              dict(ok=True, off=20000.0, dra=0, ddec=0,
                                   swept=True), context="swept")


def test_nan_magnitudes_excluded():
    cat, ref, mag = _cat()
    mag[::3] = np.nan
    g = measure_offset(cat, ref)
    r = residual_vs_magnitude(cat, ref, mag, g, context="nans")
    assert r["n_bins"] >= 3
    assert r["clean"]
