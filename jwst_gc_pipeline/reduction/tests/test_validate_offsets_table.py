"""Collapse detection for the per-visit offsets table (brick-1182 v001 signature)."""
import pytest
from astropy.table import Table

from jwst_gc_pipeline.reduction.validate_offsets_table import (
    flag_collapsed_visits, assert_offsets_table_sane, CollapsedOffsetsTableError)


def _tbl(rows):
    return Table(rows=rows, names=('Visit', 'Filter', 'dra (arcsec)', 'ddec (arcsec)'))


# Real pre-fix VIRAC2locked values: v001 collapsed onto v002 (both ~+1.9") for all bands.
COLLAPSED = _tbl([
    ('jw01182004001', 'F200W', 1.908, 0.979),
    ('jw01182004002', 'F200W', 1.911, 0.978),
    ('jw01182004001', 'F115W', 1.908, 0.979),
    ('jw01182004002', 'F115W', 1.919, 0.978),
])

# Real post-fix values: v001 = -17.5, v002 = +1.9 -> ~20" apart, clean.
CLEAN = _tbl([
    ('jw01182004001', 'F200W', -17.484, 13.546),
    ('jw01182004002', 'F200W', 1.911, 0.978),
    ('jw01182004001', 'F115W', -17.538, 13.466),
    ('jw01182004002', 'F115W', 1.919, 0.978),
])


def test_flags_collapsed_visits():
    issues = flag_collapsed_visits(COLLAPSED)
    filters = {i['filter'] for i in issues}
    assert filters == {'F200W', 'F115W'}      # both bands flagged
    assert all(i['sep_arcsec'] < 0.02 for i in issues)


def test_clean_table_passes():
    assert flag_collapsed_visits(CLEAN) == []


def test_assert_warns_not_raises_by_default(recwarn):
    issues = assert_offsets_table_sane(COLLAPSED, context="test")
    assert issues
    assert any("COLLAPSED OFFSETS TABLE" in str(w.message) for w in recwarn.list)


def test_assert_can_raise():
    with pytest.raises(CollapsedOffsetsTableError):
        assert_offsets_table_sane(COLLAPSED, raise_on_issue=True)


def test_clean_no_warn(recwarn):
    assert assert_offsets_table_sane(CLEAN) == []
    assert not any("COLLAPSED" in str(w.message) for w in recwarn.list)
