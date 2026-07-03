"""Regression test for the neighbour-robust satstar seed prominence
(cloudc 2526 F770W: 28 by-eye-real faint saturated stars dropped by the gate
because a bright NEIGHBOUR in the annulus inflated the MAD denominator and
crushed the prominence below seed_prominence_min).  ``robust=True`` measures the
background as the annulus emission FLOOR (25th pct) and the spread from the lower
half, so a single bright neighbour cannot inflate the denominator.

See project_cloudc_f770w_satstar_gate_miscalib.
"""
import numpy as np

from jwst_gc_pipeline.reduction.saturated_star_finding import _seed_prominence


def _scene_faint_star_on_structured_emission():
    """A genuine faint saturated star at centre (compact bright wing ring) whose
    prominence ANNULUS is filled with structured bright emission (a filament
    covering ~half the annulus).  The bright half lifts the annulus median AND
    MAD, so the non-robust (core-median)/MAD prominence collapses, while the
    robust (25th-pct floor + lower-half spread) estimate -- which reads the faint
    inter-filament floor -- keeps the real star's prominence high.  This is the
    cloudc 2526 F770W failure mode (28 by-eye stars dropped)."""
    ny = nx = 81
    yy, xx = np.mgrid[0:ny, 0:nx]
    cy = cx = 40
    r = np.hypot(xx - cx, yy - cy)
    data = np.full((ny, nx), 300.0, dtype=float)          # emission floor
    # bright filament covering ~2/3 of the prominence annulus -> the annulus
    # MEDIAN jumps to the bright level, so the non-robust (core-median)/MAD goes
    # negative for the (fainter) real star and it is dropped.
    data[yy < cy + 6] = 1000.0
    # compact target: bright PSF wing ring (~620) just outside a zeroed
    # saturated core (mimics the DQ-SATURATED core set to 0 upstream).
    data += 1700.0 * np.exp(-(r**2 / (2 * 2.2**2)))
    data[r < 3] = 0.0                                     # zeroed saturated core
    return data, (float(cy), float(cx))


def test_robust_prominence_lifts_star_on_structured_emission():
    data, com = _scene_faint_star_on_structured_emission()
    prom_orig, core_orig = _seed_prominence(data, com, 25, robust=False)
    prom_rob, core_rob = _seed_prominence(data, com, 25, robust=True)
    # same absolute wing-ring core either way (robust changes only the
    # background/denominator, never the measured core)
    assert np.isfinite(core_orig) and core_orig == core_rob
    # emission filling the annulus raises the median + MAD; the robust estimate
    # reads the lower emission FLOOR and a smaller lower-half spread, so it lifts
    # the prominence of a real star sitting on that emission strictly above the
    # non-robust value -- the cloudc 2526 recovery mechanism.
    assert np.isfinite(prom_rob) and prom_rob > prom_orig


def test_robust_keeps_diffuse_emission_low():
    """A flat DQ-saturated emission patch (no compact core) must still score a
    LOW robust prominence -- the fix must not turn emission into a star."""
    rng = np.random.RandomState(0)
    data = 300.0 + rng.normal(0.0, 3.0, (81, 81))
    prom_rob, _ = _seed_prominence(data, (40.0, 40.0), 25, robust=True)
    assert prom_rob < 8.0
