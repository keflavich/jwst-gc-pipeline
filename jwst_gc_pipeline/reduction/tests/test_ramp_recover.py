"""Unit tests for the ramp-SLOPE saturated-star recovery
(saturated_star_finding.ramp_slope_map / ramp_recover_saturated).

The design premise (validated on real frames 2026-07-18): cal(MJy/sr) is
proportional to the ramp-fit SLOPE, not group-0; saturation must be read from
the ramp DATA (the _ramp.fits GROUPDQ broadcasts SATURATED to every group), and
a per-pixel slope + LOCAL cal/slope calibration recovers the rim crowding-immune.
"""
import numpy as np
import pytest

from jwst_gc_pipeline.reduction.saturated_star_finding import (
    ramp_slope_map, ramp_recover_saturated)
try:
    from jwst.datamodels import dqflags
    PSAT = dqflags.pixel['SATURATED']
except Exception:  # pragma: no cover
    PSAT = 2

CEIL = 50000.0


def _linear_ramp(rate, ng, ceiling=CEIL, pedestal=0.0):
    """A pixel's ramp DN per group: pedestal + rate*t, clipped at the well."""
    t = np.arange(ng, dtype=float)
    return np.minimum(pedestal + rate * t, ceiling)


def test_slope_recovers_true_rate_over_presat_groups():
    """A pixel that saturates partway up the ramp: the slope must equal the
    true pre-saturation rate (not be dragged down by the railed groups)."""
    ng = 7
    rate = 12000.0                         # rails at group ~4
    ramp = np.zeros((ng, 3, 3))
    ramp[:, 1, 1] = _linear_ramp(rate, ng)
    slope, n_good = ramp_slope_map(ramp, ceiling=CEIL)
    # only leading below-ceiling groups counted; slope ~ true rate
    assert n_good[1, 1] >= 2
    assert np.isfinite(slope[1, 1])
    assert abs(slope[1, 1] - rate) / rate < 0.05


def test_group0_railed_is_deep_core():
    """A pixel already at the well in group-0 has no usable groups -> NaN slope
    (deep core, unrecoverable from data)."""
    ng = 7
    ramp = np.full((ng, 3, 3), CEIL)       # railed from group 0
    slope, n_good = ramp_slope_map(ramp, ceiling=CEIL)
    assert n_good[1, 1] == 0
    assert not np.isfinite(slope[1, 1])


def test_saturation_read_from_data_not_broadcast_groupdq():
    """GROUPDQ that flags SATURATED on EVERY group (the real _ramp.fits pattern)
    must NOT zero out a recoverable pixel -- saturation comes from the data."""
    ng = 7
    ramp = np.zeros((ng, 3, 3))
    ramp[:, 1, 1] = _linear_ramp(9000.0, ng)
    gdq = np.full((ng, 3, 3), dqflags.group['SATURATED'], dtype=int)  # broadcast
    slope, n_good = ramp_slope_map(ramp, gdq, ceiling=CEIL)
    assert n_good[1, 1] >= 2 and np.isfinite(slope[1, 1])


def test_recover_fills_saturated_rim_with_slope_times_local_K():
    """End-to-end: a bright saturated star on a field of unsaturated calibrators
    with a known cal = K*slope. The recovered rim (NaN-blanked cal) must come
    back at ~K*slope, and the deep core (railed at g0) stays deep."""
    rng = np.random.default_rng(0)
    ny = nx = 96
    ng = 7
    K = 0.17
    # unsaturated calibrator field: random rates -> cal = K*slope (+small noise)
    rates = rng.uniform(200, 4000, size=(ny, nx))
    ramp = rates[None] * np.arange(ng)[:, None, None]
    ramp = np.minimum(ramp, CEIL)
    cal = K * rates * (1 + 0.02 * rng.standard_normal((ny, nx)))
    dq = np.zeros((ny, nx), int)
    # inject a saturated star at center: high rate -> rails partway
    cy = cx = 48
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(yy - cy, xx - cx)
    star_rate = 60000.0 * np.exp(-(r ** 2) / (2 * 2.5 ** 2))
    ramp[:, r < 12] = np.minimum((star_rate[r < 12][None] *
                                  np.arange(ng)[:, None]), CEIL)
    # cal + DQ: saturated core NaN-blanked & flagged where group0 rails
    railed_g0 = ramp[0] >= CEIL
    sat_region = (r < 6)
    dq[sat_region] |= PSAT
    cal[sat_region] = np.nan
    recovered, rim, deep, Kg = ramp_recover_saturated(cal, dq, ramp,
                                                      slope_min=20.0,
                                                      cal_block=32)
    assert Kg == pytest.approx(K, rel=0.1)
    # rim pixels that had a measurable slope are filled (finite) and positive
    assert rim.sum() > 0
    assert np.isfinite(recovered[rim]).all()
    # a mid-rim saturated pixel with pre-sat groups is recovered near K*slope
    ring = sat_region & ~railed_g0
    if ring.any():
        slope, ngood = ramp_slope_map(ramp, ceiling=CEIL)
        rec_ring = recovered[ring & rim]
        exp_ring = (K * slope)[ring & rim]
        good = np.isfinite(rec_ring) & np.isfinite(exp_ring) & (exp_ring > 0)
        assert good.sum() > 0
        assert np.nanmedian(rec_ring[good] / exp_ring[good]) == pytest.approx(1.0, rel=0.15)


def test_no_ramp_is_noop():
    cal = np.ones((10, 10))
    dq = np.zeros((10, 10), int); dq[5, 5] |= PSAT
    rec, rim, deep, K = ramp_recover_saturated(cal, dq, None)
    assert np.array_equal(rec, cal) and rim.sum() == 0
