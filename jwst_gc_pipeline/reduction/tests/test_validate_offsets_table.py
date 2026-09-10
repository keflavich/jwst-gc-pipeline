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


# ---------------------------------------------------------------------------
# The amplitude floor (review of PR #770)
# ---------------------------------------------------------------------------
#
# `flag_collapsed_visits` tested only whether two visits AGREE.  Two visits
# that each legitimately needed no correction agree too, and that is the
# common case on a table the m2 re-tie writes -- `astrometry_checkpoint` had
# already had to write the exemption at one of its own call sites ("any two
# visits agree within 20 mas by construction -- flagging that would be a
# category error").  Once the apply path RAISES on the finding, that class
# stops a reduction.

#: Two visits that each measured ~no correction.  Per-visit medians read
#: 2026-09-09 from the live tables the review named -- m4
#: ``Offsets_JWST_Brick1979_consensus.csv`` F150W2 and ngc6334
#: ``Offsets_JWST_Brick7213_consensus.csv`` F162M.
WELL_ALIGNED = _tbl([
    ('jw01979002001', 'F150W2', +0.0010, -0.0005),   # |1.1| mas
    ('jw01979003001', 'F150W2', +0.0011, +0.0020),   # |2.3| mas, 2.5 mas apart
    ('jw07213001001', 'F162M', -0.0015, +0.0016),    # |2.2| mas
    ('jw07213001002', 'F162M', -0.0015, -0.0084),    # |8.5| mas, 10.0 mas apart
])


def test_two_visits_that_both_needed_no_correction_are_not_a_collapse():
    """The false-positive class the floor exists for.

    Nothing was overwritten here, because nothing was applied: both visits
    carry ~1-10 mas.  At that scale the tolerance used to call the two equal
    is as big as the value being compared, so "they agree" says nothing about
    whether one was copied from the other.
    """
    assert flag_collapsed_visits(WELL_ALIGNED) == []


def test_the_floor_still_catches_the_brick_1182_signature():
    """+1.9" shared while one visit truly needed -17.5" -- the real failure."""
    issues = flag_collapsed_visits(COLLAPSED)
    assert {i['filter'] for i in issues} == {'F200W', 'F115W'}
    assert all(i['amp_arcsec'] > 0.02 for i in issues)


def test_the_floor_is_on_the_shared_VALUE_not_on_the_agreement():
    """A pair agreeing to 1 mas is flagged or not by its amplitude alone."""
    below = _tbl([('v1', 'F200W', 0.010, 0.0), ('v2', 'F200W', 0.011, 0.0)])
    above = _tbl([('v1', 'F200W', 0.030, 0.0), ('v2', 'F200W', 0.031, 0.0)])
    assert flag_collapsed_visits(below) == []
    assert len(flag_collapsed_visits(above)) == 1


def test_the_message_reports_the_shared_value(recwarn):
    """An operator has to be able to see WHY the pair was called a collapse."""
    assert_offsets_table_sane(COLLAPSED, context="test")
    msgs = [str(w.message) for w in recwarn.list
            if 'COLLAPSED OFFSETS TABLE' in str(w.message)]
    assert msgs
    assert 'shared value' in msgs[0]
