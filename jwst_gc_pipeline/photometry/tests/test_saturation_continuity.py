"""Unit tests for the saturation-continuity certification metric.

The metric is the pass/fail gate for satstar photometry: across the
saturation boundary the satstar-fit and normal-photometry color medians must
agree (<0.05 mag goal / <0.10 certification floor).  Synthetic catalogs pin
the C1 (transition-bin jump) behaviour.
"""
import numpy as np
import pytest
from astropy.table import Table

from jwst_gc_pipeline.photometry.saturation_continuity import (
    saturation_continuity, assert_saturation_continuity,
    degenerate_pair_flatness)


def _cat(jump=0.0, n=4000, seed=1):
    """Two-band catalog: color locus flat at 0.5; stars brighter than
    mag_B=13 are replaced_saturated in A with an optional color JUMP."""
    rng = np.random.default_rng(seed)
    magB = rng.uniform(10.5, 17, n)
    sat = magB < 13 + rng.normal(0, 0.3, n)   # fuzzy boundary -> mixed bins
    color = 0.5 + rng.normal(0, 0.05, n) + np.where(sat, jump, 0.0)
    return Table({
        'mag_vega_a': magB + color,
        'mag_vega_b': magB,
        'replaced_saturated_a': sat,
        'replaced_saturated_b': np.zeros(n, bool),
        'forced_filled_a': np.zeros(n, bool),
        'forced_filled_b': np.zeros(n, bool),
        'independently_detected_b': np.ones(n, bool),
    })


def test_continuous_catalog_passes():
    r = saturation_continuity(_cat(jump=0.0), 'a', 'b')
    assert np.isfinite(r['metric']) and r['metric'] < 0.05
    assert_saturation_continuity(_cat(jump=0.0), [('a', 'b')], threshold=0.10)


def test_jump_detected_and_fails():
    r = saturation_continuity(_cat(jump=0.4), 'a', 'b')
    assert np.isfinite(r['metric']) and abs(r['metric'] - 0.4) < 0.1
    with pytest.raises(AssertionError):
        assert_saturation_continuity(_cat(jump=0.4), [('a', 'b')], threshold=0.10)


def test_saturation_continuity_is_directional():
    """Interface + directionality regression, not a guard against human misuse
    (that isn't unit-testable). Pins that (a) the ``band_sat=``/``band_ref=``
    keywords exist and the positional order is (band_sat, band_ref), and (b) the
    metric is genuinely order-dependent: 'a' carries the replaced_saturated
    population and 'b' does not, so the reversed order has no SAT population --
    'no-sat-population', not a mirror-image metric. The names make that
    order-dependence readable at the call site; the metric itself was already
    directional before the rename."""
    cat = _cat(jump=0.4)
    fwd = saturation_continuity(cat, band_sat='a', band_ref='b')
    rev = saturation_continuity(cat, band_sat='b', band_ref='a')
    assert np.isfinite(fwd['metric']) and abs(fwd['metric'] - 0.4) < 0.1
    assert rev['kind'] == 'no-sat-population'
    # positional and keyword forms agree (positional order is band_sat, band_ref)
    assert saturation_continuity(cat, 'a', 'b')['metric'] == fwd['metric']


def _flatcat(bright_dev=0.0, bright_flag='is_saturated', n=8000, seed=3):
    """Color-flat locus at 0.5 for mag_B >= 13; the bright end (mag_B < 13)
    carries ``bright_dev`` and is tagged with ``bright_flag`` in band A only.
    This is the release picture: a bright regime that is off-locus but flagged,
    so the SCIENCE subset (flags cut) must read flat while the flag-inclusive
    metric sees the offset."""
    rng = np.random.default_rng(seed)
    magB = rng.uniform(9.5, 19, n)
    bright = magB < 13
    color = 0.5 + rng.normal(0, 0.03, n) + np.where(bright, bright_dev, 0.0)
    t = Table({
        'mag_vega_a': magB + color,
        'mag_vega_b': magB,
        'is_saturated_a': np.zeros(n, bool),
        'is_saturated_b': np.zeros(n, bool),
        'replaced_saturated_a': np.zeros(n, bool),
        'replaced_saturated_b': np.zeros(n, bool),
        'forced_filled_a': np.zeros(n, bool),
        'forced_filled_b': np.zeros(n, bool),
    })
    t[f'{bright_flag}_a'] = bright
    return t


def test_science_subset_ignores_flagged_bright_offset():
    # deep-core / recovered rows are flagged and off-locus; science subset is flat
    cat = _flatcat(bright_dev=0.4, bright_flag='is_saturated')
    r_full = degenerate_pair_flatness(cat, 'a', 'b', include_flags=True)
    r_sci = degenerate_pair_flatness(cat, 'a', 'b', science_only=True)
    assert r_full['metric'] >= 0.10          # flag-inclusive sees the offset
    assert r_sci['metric'] < 0.05            # science subset is flat


def test_science_subset_also_drops_replaced_saturated():
    cat = _flatcat(bright_dev=0.3, bright_flag='replaced_saturated')
    r_full = degenerate_pair_flatness(cat, 'a', 'b', include_flags=True)
    r_sci = degenerate_pair_flatness(cat, 'a', 'b', science_only=True)
    assert r_full['metric'] >= 0.10
    assert r_sci['metric'] < 0.05


def test_science_subset_catches_unflagged_drift():
    # a drift in UNFLAGGED stars must still fail the science gate
    cat = _flatcat(bright_dev=0.0)
    # push a mid-mag unflagged bump into band A
    magB = np.asarray(cat['mag_vega_b'])
    bump = (magB >= 14) & (magB < 15)
    cat['mag_vega_a'][bump] += 0.3
    r_sci = degenerate_pair_flatness(cat, 'a', 'b', science_only=True)
    assert r_sci['metric'] >= 0.10
    # The metric only scans bins BRIGHTER than the 40th percentile of mag_B, so
    # the bump must fall inside that range to be seen. _flatcat flags mag_B<13
    # is_saturated, which cuts a third of the rows and lifts p_lo above 15 -- pin
    # that the failing bin IS the 14-15 bump so the test cannot silently stop
    # exercising it if the plateau window moves.
    assert 14.0 <= r_sci['worst_bin']['magB_lo'] < 15.0
