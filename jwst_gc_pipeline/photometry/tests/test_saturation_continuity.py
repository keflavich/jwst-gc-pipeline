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
    saturation_continuity, assert_saturation_continuity)


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
