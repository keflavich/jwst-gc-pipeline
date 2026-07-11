"""Regression tests for validate_offsets_table -- the offsets-table sanity
validator that catches the brick-1182 VISIT-COLLAPSE bug (a builder writing one
visit's offset for every visit).
"""
import numpy as np
import pytest
from astropy.table import Table

from jwst_gc_pipeline.reduction.validate_offsets_table import (
    flag_collapsed_visits, flag_insane_magnitudes, validate_offsets_table)


def _table(rows):
    """rows: list of (visit, exposure, filt, dra, ddec)."""
    t = Table()
    t['Visit'] = [r[0] for r in rows]
    t['Exposure'] = [r[1] for r in rows]
    t['Filter'] = [r[2] for r in rows]
    t['dra'] = [r[3] for r in rows]
    t['ddec'] = [r[4] for r in rows]
    return t


def test_collapsed_visits_flagged():
    # The actual brick-1182 bug: v001 got v002's +1.9" for every filter.
    t = _table([
        ('jw01182004001', 1, 'F200W', 1.9, 0.98),
        ('jw01182004002', 1, 'F200W', 1.9, 0.98),   # identical -> collapse
    ])
    issues = flag_collapsed_visits(t)
    assert len(issues) == 1
    assert 'COLLAPSE' in issues[0] and 'F200W' in issues[0]


def test_distinct_visits_pass():
    # The CORRECT table: v001 = -17.5", v002 = +1.9" (differ by ~20").
    t = _table([
        ('jw01182004001', 1, 'F200W', -17.48, 13.55),
        ('jw01182004002', 1, 'F200W', 1.90, 0.98),
    ])
    assert flag_collapsed_visits(t) == []


def test_single_visit_not_flagged():
    # One visit -> nothing to compare, must not false-positive.
    t = _table([('jw02221001001', 1, 'F182M', -0.17, 0.05)])
    assert flag_collapsed_visits(t) == []


def test_collapse_flagged_per_filter_independently():
    # F200W collapsed, F115W fine -> exactly one flag (F200W).
    t = _table([
        ('jw01182004001', 1, 'F200W', 1.9, 0.98),
        ('jw01182004002', 1, 'F200W', 1.9, 0.98),
        ('jw01182004001', 1, 'F115W', -17.5, 13.5),
        ('jw01182004002', 1, 'F115W', 1.9, 0.98),
    ])
    issues = flag_collapsed_visits(t)
    assert len(issues) == 1 and 'F200W' in issues[0]


def test_near_but_not_identical_still_flags_below_tol():
    # 3 mas apart (< 5 mas tol) -> still collapse (independent visits never agree this well).
    t = _table([
        ('jw01182004001', 1, 'F200W', 1.900, 0.980),
        ('jw01182004002', 1, 'F200W', 1.903, 0.981),
    ])
    assert len(flag_collapsed_visits(t)) == 1
    # ... but a genuine >tol difference passes
    t2 = _table([
        ('jw01182004001', 1, 'F200W', 1.900, 0.980),
        ('jw01182004002', 1, 'F200W', 1.950, 1.030),  # ~65 mas apart
    ])
    assert flag_collapsed_visits(t2) == []


def test_insane_magnitude_flagged():
    # A unit error (mas written as arcsec) -> huge offset.
    t = _table([('jw01182004001', 1, 'F200W', 1911.0, 977.0)])
    assert len(flag_insane_magnitudes(t)) == 1
    # sane magnitudes pass
    assert flag_insane_magnitudes(_table([('jw01182004001', 1, 'F200W', -17.5, 13.5)])) == []


def test_validate_combines_checks():
    t = _table([
        ('jw01182004001', 1, 'F200W', 1.9, 0.98),
        ('jw01182004002', 1, 'F200W', 1.9, 0.98),
    ])
    assert validate_offsets_table(t)  # non-empty -> fails


def test_missing_columns_raises():
    t = Table(); t['foo'] = [1, 2]
    with pytest.raises(KeyError):
        flag_collapsed_visits(t)
