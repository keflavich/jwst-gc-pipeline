"""The two column pairs are AS-BUILT and AS-CORRECTED, not two copies (#319).

#319 read them as duplicates of one quantity, concluded "39 of 40 rows
disagree", and stopped a gc2211 re-reduction over it.  Measured across all ten
live locked tables (1164 rows), 678 differ and **every one** is reconstructed
from
`prov_dra_added_mas` / `prov_ddec_added_mas` to <0.1 mas:

    gc2211 o023 F277W exp1   ddec gap 14986.2 mas   prov_ddec 14986.2 mas
                             dra  gap -7163.5 mas   prov_dra  -6269.7 mas
                                                    / cos(28.9 deg) = -7164.1

So a gap is normal and its SIZE says nothing.  A row sits in one of two
legitimate states -- pairs in sync (`gap == 0`, where the builder starts it and
where `update_offsets_table` returns it by HEALING an explained gap), or never
healed (`gap == prov`).  What is worth catching is a gap that is NEITHER: one
pair updated without the other.
"""
import warnings

import numpy as np
import pytest
from astropy.table import Table

from jwst_gc_pipeline.reduction.validate_offsets_table import (
    DivergedColumnPairError, assert_offsets_table_sane,
    flag_diverged_column_pairs)

COSD = np.cos(np.radians(28.9))


def _tbl(prov_dra=0.0, prov_ddec=0.0, dra=1.0, ddec=-2.0,
         dra_as=None, ddec_as=None):
    return Table({
        "Visit": ["jw02211023001"], "Filter": ["F277W"], "Exposure": [1],
        "Module": ["nrcalong"],
        "dra": [dra], "ddec": [ddec],
        "dra (arcsec)": [dra if dra_as is None else dra_as],
        "ddec (arcsec)": [ddec if ddec_as is None else ddec_as],
        "prov_dra_added_mas": [prov_dra], "prov_ddec_added_mas": [prov_ddec]})


def test_a_gap_that_equals_the_recorded_corrections_is_NOT_flagged():
    """The gc2211 numbers, which #319 reported as 39-of-40 divergent."""
    t = _tbl(prov_dra=-6269.7, prov_ddec=14986.2,
             dra=3.1262, ddec=-1.8226,
             dra_as=3.1262 - 7.1635, ddec_as=-1.8226 + 14.9862)
    assert flag_diverged_column_pairs(t) == []


def test_a_pair_updated_without_the_other_IS_flagged():
    """The mechanism #319 suspected: a writer touches `(arcsec)` only."""
    t = _tbl(prov_dra=0.0, prov_ddec=0.0, ddec_as=-2.0 + 0.05)
    bad = flag_diverged_column_pairs(t)
    assert len(bad) == 1, bad
    assert abs(bad[0]["ddec_gap_mas"] - 50.0) < 0.1


def test_a_recorded_correction_with_NO_gap_is_the_HEALED_state():
    """27 cloudc rows and 4 sgrc rows record a 1-7 mas m2 correction while both
    pairs are identical.  That is not a broken table: `update_offsets_table`
    HEALS an explained gap into the plain pair before applying a correction, and
    `prov_*` keeps accumulating past the heal, so a corrected row ends at gap 0
    with nonzero provenance BY DESIGN.

    Requiring `gap == prov` unconditionally therefore flagged every corrected
    row on every table -- it took the branch's CI red across 8 tests, including
    `test_offsets_column_pair_sync`, which asserts exactly this heal."""
    t = _tbl(prov_dra=4.69, prov_ddec=1.31)      # gap 0, corrections recorded
    assert flag_diverged_column_pairs(t) == []


def _tbl_unexplained():
    """A gap that is neither 0 (healed/in sync) nor the recorded provenance."""
    return _tbl(prov_dra=4.69, prov_ddec=1.31, ddec_as=-2.0 + 0.050)


def test_a_gap_that_is_neither_zero_nor_the_provenance_IS_flagged():
    """The only remaining state, and the one no writer produces: 50 mas of Dec
    gap against 1.31 mas of recorded correction.  One pair moved by an amount
    the other never got."""
    bad = flag_diverged_column_pairs(_tbl_unexplained())
    assert len(bad) == 1, bad
    assert abs(bad[0]["ddec_gap_mas"] - 50.0) < 0.1
    assert abs(bad[0]["prov_ddec_mas"] - 1.31) < 0.1


def test_the_RA_axis_is_bounded_not_exact():
    """`dec_deg` is not stored per row and the apply loop divides on-sky mas by
    cos(dec), so the RA gap is only known to lie between `prov` and
    `prov/cos(30 deg)`.  Both ends must pass."""
    for factor in (1.0, 1.0 / np.cos(np.radians(30.0)), 1.0 / COSD):
        t = _tbl(prov_dra=-6269.7, dra_as=1.0 + (-6269.7 * factor) / 1000.0)
        assert flag_diverged_column_pairs(t) == [], factor
    # ... and beyond that bound it is flagged
    t = _tbl(prov_dra=-6269.7, dra_as=1.0 + (-6269.7 * 1.3) / 1000.0)
    assert flag_diverged_column_pairs(t)


def test_a_single_pair_table_is_not_checked():
    """Builder output and the GNS/consensus tables carry one pair; there is no
    invariant to hold."""
    t = Table({"Visit": ["v"], "Filter": ["F212N"], "Exposure": [1],
               "dra": [1.0], "ddec": [2.0]})
    assert flag_diverged_column_pairs(t) == []


def test_the_gate_WARNS_rather_than_stopping():
    """The reducer reads the `(arcsec)` pair and that pair is still
    self-consistent -- what is lost is the ability to say how it got there.
    Stopping 31 real rows over a broken audit trail would be the wrong trade,
    and it is escalatable for anyone who wants it to be."""
    t = _tbl_unexplained()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        issues = assert_offsets_table_sane(t, context="test")
    assert len(caught) == 1
    assert "PROVENANCE BROKEN" in str(caught[0].message)
    assert any(i.get("kind") == "diverged_column_pair" for i in issues)


def test_it_can_be_escalated_to_a_stop():
    with pytest.raises(DivergedColumnPairError, match="PROVENANCE BROKEN"):
        assert_offsets_table_sane(_tbl_unexplained(), context="test",
                                  raise_on_diverged=True)


def test_the_COLLAPSE_switch_does_not_escalate_a_divergence():
    """They are separate findings and need separate switches.  Sharing one made
    every existing `raise_on_issue=True` caller -- the m2 checkpoint and the
    release gate among them -- stop on the weaker one."""
    t = _tbl_unexplained()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        issues = assert_offsets_table_sane(t, context="test",
                                           raise_on_issue=True)
    assert any(i.get("kind") == "diverged_column_pair" for i in issues)
    assert any("PROVENANCE BROKEN" in str(c.message) for c in caught)


def test_a_clean_table_neither_warns_nor_reports():
    t = _tbl()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert assert_offsets_table_sane(t) == []
    assert not caught
