"""Collapse detection for the per-visit offsets table (brick-1182 v001 signature)."""
import pytest
from astropy.table import Table

from jwst_gc_pipeline.reduction.validate_offsets_table import (
    flag_collapsed_visits, assert_offsets_table_sane, CollapsedOffsetsTableError,
    unmeasurable_visit_cells)


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


# ---------------------------------------------------------------------------
# Whole-consensus tie rows (issue #831)
# ---------------------------------------------------------------------------
#
# A consensus table holds per-exposure rows plus one bulk consensus->reference
# row per (visit, filter): Module == "all", Exposure == -1.  The per-visit
# median over both kinds follows the row count, not the astrometry.  Values
# below are read from ngc6334 ``Offsets_JWST_Brick6778_consensus.csv``
# (2026-11-13): per-exposure ~10 mas, tie ~89 mas.

def _ctbl(rows):
    """Rows of (visit, filter, module, exposure, dra, ddec) in arcsec."""
    return Table(rows=rows, names=('Visit', 'Filter', 'Module', 'Exposure',
                                   'dra (arcsec)', 'ddec (arcsec)'))


def _tie(visit, filt, dra, ddec):
    return (visit, filt, 'all', -1, dra, ddec)


F187N_TWO_ROW_VISITS = _ctbl(
    [('jw06778001001', 'F187N', 'nrca3', 1, -0.0102, 0.0010),
     _tie('jw06778001001', 'F187N', -0.0871, 0.0158),
     ('jw06778001002', 'F187N', 'nrca2', 2, -0.0094, 0.0012),
     _tie('jw06778001002', 'F187N', -0.0859, 0.0152)]
    + [('jw06778001003', 'F187N', f'nrca{k % 4 + 1}', k, -0.0114 + 0.0004 * k,
        0.0009 * (k % 3)) for k in range(1, 10)])

F470N_TIE_ONLY = _ctbl([
    _tie('jw06778001001', 'F470N', -0.0901, 0.0120),
    _tie('jw06778001002', 'F470N', -0.0915, 0.0118),
])


def test_tie_rows_do_not_enter_the_visit_median():
    """Two visits of two rows each (one exposure, one tie) are not a collapse."""
    assert flag_collapsed_visits(F187N_TWO_ROW_VISITS) == []


def test_the_fixture_flags_without_the_exclusion():
    """The same rows with the Module/Exposure columns removed reproduce #831."""
    stripped = F187N_TWO_ROW_VISITS['Visit', 'Filter', 'dra (arcsec)',
                                    'ddec (arcsec)']
    assert flag_collapsed_visits(stripped)


def test_tie_only_visits_are_unmeasurable_not_collapsed(recwarn):
    assert flag_collapsed_visits(F470N_TIE_ONLY) == []
    assert unmeasurable_visit_cells(F470N_TIE_ONLY) == [
        ('F470N', 'jw06778001001'), ('F470N', 'jw06778001002')]
    assert assert_offsets_table_sane(F470N_TIE_ONLY, raise_on_issue=True) == []
    msgs = [str(w.message) for w in recwarn.list if 'unmeasurable' in str(w.message)]
    assert len(msgs) == 1
    assert 'F470N jw06778001001' in msgs[0]


def test_unmeasurable_warning_is_one_line_with_a_count(recwarn):
    rows = [_tie(f'jw06778001{k:03d}', 'F470N', -0.09, 0.012 + 0.001 * k)
            for k in range(1, 15)]
    assert len(unmeasurable_visit_cells(_ctbl(rows))) == 14
    assert_offsets_table_sane(_ctbl(rows))
    msgs = [str(w.message) for w in recwarn.list if 'unmeasurable' in str(w.message)]
    assert len(msgs) == 1
    assert '14 (filter, visit) cell(s)' in msgs[0]
    assert 'and 4 more' in msgs[0]


def test_cells_with_an_exposure_row_are_not_listed_unmeasurable():
    assert unmeasurable_visit_cells(F187N_TWO_ROW_VISITS) == []


def test_miri_tie_only_visits_do_not_flag():
    """cloudef 2092 F770W: four visits, one tie row each, ~-0.07/-0.13 arcsec."""
    rows = [_tie(f'jw02092004{k:03d}', 'F770W', -0.070 - 0.002 * (k % 2), -0.130)
            for k in (2, 3, 6, 15)]
    t = _ctbl(rows)
    assert flag_collapsed_visits(t) == []
    assert len(unmeasurable_visit_cells(t)) == 4


def test_collapse_in_exposure_rows_still_flagged_with_ties_present():
    """Identical large per-exposure rows in two visits remain the signature."""
    t = _ctbl([
        ('v1', 'F200W', 'nrca1', 1, 1.908, 0.979),
        ('v1', 'F200W', 'nrca2', 2, 1.909, 0.979),
        _tie('v1', 'F200W', 0.010, 0.002),
        ('v2', 'F200W', 'nrca1', 1, 1.911, 0.978),
        ('v2', 'F200W', 'nrca2', 2, 1.910, 0.978),
        _tie('v2', 'F200W', 0.030, -0.004),
    ])
    issues = flag_collapsed_visits(t)
    assert len(issues) == 1
    assert issues[0]['amp_arcsec'] > 1.0


def test_a_tie_needs_both_sentinels():
    """Module == "all" with a real Exposure number is a per-exposure row."""
    t = _ctbl([('v1', 'F200W', 'all', 1, 0.050, 0.0),
               ('v2', 'F200W', 'all', 1, 0.051, 0.0)])
    assert len(flag_collapsed_visits(t)) == 1
    assert unmeasurable_visit_cells(t) == []


def test_nan_rows_are_skipped_not_compared():
    t = _ctbl([('v1', 'F200W', 'nrca1', 1, float('nan'), float('nan')),
               ('v2', 'F200W', 'nrca1', 1, 0.050, 0.0)])
    assert flag_collapsed_visits(t) == []


def test_tables_without_module_columns_unchanged():
    assert len(flag_collapsed_visits(COLLAPSED)) == 2
    assert flag_collapsed_visits(CLEAN) == []
    assert unmeasurable_visit_cells(COLLAPSED) == []


def test_bulk_sentinels_match_checkpoint():
    from jwst_gc_pipeline.photometry import astrometry_checkpoint as ac
    from jwst_gc_pipeline.reduction import validate_offsets_table as vot
    assert vot._BULK_MODULE == ac.BULK_MODULE
    assert vot._BULK_EXPOSURE == ac.BULK_EXPOSURE
