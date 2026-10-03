"""NIRCam recovered-core cap (NIRCAM_SATSTAR_RECOVERED_CAP) reads only the
fitted star's own pixels.

Before the fix the cap read ``satmask_combined``: this star's dilated deep
core plus every OTHER saturated source's deep core in the cutout.  A weakly
saturated star whose own deep core was empty (ZEROFRAME anchor recovered all
its SATURATED pixels, or SATSTAR_ZF_KEEP_FINITE kept them) was then capped by
a neighbour's model-subtracted core, ~1% of its flux, and rejected by the
fit-quality gate (wd2 F150W nrcb1: 245 cap firings, median capped/fitted flux
0.011).
"""
import numpy as np
from scipy.ndimage import binary_dilation

from jwst_gc_pipeline.reduction.saturated_star_finding import (
    recovered_cap_region, recovered_core_peak)


def _disk(shape, x, y, r):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2


def _scene():
    """Cutout with the fitted star A (weakly saturated: SAT component with
    finite rates, empty deep core) and a neighbour B whose deep core holds a
    small model-subtracted residual."""
    shape = (41, 41)
    cutout = np.full(shape, 2.0)
    comp_a = _disk(shape, 20, 20, 2)
    cutout[comp_a] = 1200.0
    cutout[20, 20] = 1800.0                     # A's measured peak
    deep_b = _disk(shape, 33, 8, 1)
    cutout[deep_b] = 5.0                        # B's core after subtraction
    unrec = np.zeros(shape, bool)
    return cutout, comp_a, deep_b, unrec


def test_region_is_own_component_when_own_deep_core_empty():
    cutout, comp_a, deep_b, unrec = _scene()
    own_deep = np.zeros_like(comp_a)
    region = recovered_cap_region(comp_a, own_deep,
                                  binary_dilation(own_deep, iterations=2))
    assert np.array_equal(region, comp_a)
    assert not (region & deep_b).any()


def test_region_is_dilated_own_deep_core_when_present():
    _, comp_a, deep_b, _ = _scene()
    own_deep = _disk(comp_a.shape, 20, 20, 1)
    expanded = binary_dilation(own_deep, iterations=2)
    region = recovered_cap_region(comp_a, own_deep, expanded)
    assert np.array_equal(region, expanded)
    assert not (region & deep_b).any()


def test_cap_reads_own_peak_not_neighbour_core():
    """The new region gives A's peak; the old satmask_combined region
    (empty own deep core | B's deep core) gave B's residual."""
    cutout, comp_a, deep_b, unrec = _scene()
    own_deep = np.zeros_like(comp_a)
    expanded = binary_dilation(own_deep, iterations=2)
    region = recovered_cap_region(comp_a, own_deep, expanded)
    peak, lost, nrec = recovered_core_peak(cutout, region, unrec)
    assert peak == 1800.0
    assert lost == 0.0 and nrec == int(comp_a.sum())
    old_peak, _, _ = recovered_core_peak(cutout, expanded | deep_b, unrec)
    assert old_peak == 5.0


def test_cap_skipped_when_core_mostly_lost():
    cutout, comp_a, _, _ = _scene()
    unrec = comp_a.copy()
    unrec[20, 18] = False                       # 1 of 13 pixels recoverable
    peak, lost, nrec = recovered_core_peak(cutout, comp_a, unrec, max_lost=0.2)
    assert np.isnan(peak)
    assert lost > 0.9 and nrec == 1


def test_cap_skipped_with_too_few_measured_pixels():
    cutout, comp_a, _, unrec = _scene()
    region = np.zeros_like(comp_a)
    region[20, 20] = region[20, 21] = True
    peak, lost, nrec = recovered_core_peak(cutout, region, unrec)
    assert np.isnan(peak) and nrec == 2


def test_empty_region_skips_cap():
    cutout, comp_a, _, unrec = _scene()
    peak, lost, nrec = recovered_core_peak(cutout, np.zeros_like(comp_a), unrec)
    assert np.isnan(peak) and lost == 1.0 and nrec == 0
