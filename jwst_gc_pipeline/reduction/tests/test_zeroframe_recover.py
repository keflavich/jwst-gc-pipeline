"""Regressions for zeroframe_recover_saturated (the satstar fit anchor).

Brick F182M findings (2026-07-10) pinned here:
1. group-0 <= 0 is INVALID, never 'clean' (the ramp ZEROFRAME extension is
   ZEROED at flagged pixels; treating 0 as clean produced R*0 = 0
   "recoveries" that the fit then discarded as cutout==0).
2. The group-0 saturation ceiling must come from the saturated-pixel pile-up
   plateau, not a whole-frame percentile (on a mostly-dark frame the global
   p99.9 sat 20x below the true rail, mislabelling every bright rim pixel as
   deep core).
3. R is calibrated in the BRIGHT regime (it drifts ~25% from faint to bright
   pixels; the rim pixels being recovered are bright).
"""
import numpy as np

from jwst_gc_pipeline.reduction.saturated_star_finding import (
    zeroframe_recover_saturated)

SATBIT = 2


def _scene(rail=48000.0, r_true=0.17):
    """Dark frame + one saturated star: deep core at the group-0 rail,
    recoverable rim, bright unsaturated calibration pixels."""
    ny = nx = 120
    g0 = np.random.default_rng(7).normal(10, 3, (ny, nx))
    data = g0 * r_true + np.random.default_rng(8).normal(0, 0.05, (ny, nx))
    dq = np.zeros((ny, nx), dtype=np.uint32)
    # bright unsaturated calibration pixels: an 8x8 block spread over the
    # bright regime (the R estimator needs >=50 such pixels)
    rng = np.random.default_rng(9)
    vals = rng.uniform(21000, 40000, (8, 8))
    g0[100:108, 8:16] = vals
    data[100:108, 8:16] = vals * r_true
    # the saturated star at (60, 60): 3x3 deep core at the rail, ring rim
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.hypot(xx - 60, yy - 60)
    core = rr < 1.5
    rim = (rr >= 1.5) & (rr < 4)
    g0[core] = rail
    g0[rim] = 20000.0
    dq[core | rim] = SATBIT
    data[core] = np.nan                       # crf blanks the core
    data[rim] = 20000.0 * r_true * 1.15       # charge-migration inflated rim
    return data, dq, g0, core, rim, r_true


def test_rim_recovered_core_masked():
    data, dq, g0, core, rim, r_true = _scene()
    rec, rim_mask, deep, R = zeroframe_recover_saturated(data, dq, g0)
    assert np.isfinite(R) and abs(R / r_true - 1) < 0.05
    # rim pixels rewritten to ~R*g0 (de-inflated), core flagged deep
    assert rim_mask[rim].all()
    assert np.allclose(rec[rim], R * 20000.0, rtol=0.05)
    assert deep[core].all()


def test_zeroed_group0_is_invalid_not_clean():
    """Pixels with group-0 == 0 (zeroed ZEROFRAME ext) must be deep-core, not
    'recovered' to R*0 = 0."""
    data, dq, g0, core, rim, r_true = _scene()
    g0[rim] = 0.0
    rec, rim_mask, deep, R = zeroframe_recover_saturated(data, dq, g0)
    assert not rim_mask[rim].any(), "g0==0 must never count as recoverable"
    assert deep[rim].all()
    # data untouched there
    assert np.allclose(rec[rim], data[rim], equal_nan=True)


def test_ceiling_from_pileup_plateau():
    """A mostly-dark frame: the whole-frame percentile would put the ceiling
    ~30 DN and consign the 20k rim to 'deep core'; the plateau-based ceiling
    must keep it recoverable."""
    data, dq, g0, core, rim, r_true = _scene()
    rec, rim_mask, deep, R = zeroframe_recover_saturated(data, dq, g0)
    assert rim_mask[rim].all(), "bright rim below the rail must be recoverable"
