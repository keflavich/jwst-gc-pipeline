"""``prov_source`` must record the whole source string, not its first N chars.

An offsets table is stored as CSV and read back with ``Table.read``, so each of
its text columns is typed to the longest string that file happened to contain --
``<U23`` for a table whose provenance so far has only ever said ``'m2
visit-consensus'``.  Assigning a longer string into a numpy string column
truncates it silently: no error, no warning, the leading characters kept and the
rest dropped.

The string the m2 checkpoint writes when it pools several detectors' corrections
into one is 58 characters:

    'm2 visit-consensus [median of 2, ptp 1.51mas: nrcb3,nrcb4]'

Stored in a ``<U23`` column that becomes ``'m2 visit-consensus [med'`` -- the
detector list, the part that says which measurements the median came from, is
exactly what is cut.  Six of the thirteen live offsets tables (arches, both
cloudef tables, quintuplet, sgra, sgrb2) are ``<U23`` today; the three that
already carry the long form (cloudc, sgrc, sickle) are wide enough only because
a long string reached them first.  So which tables lose their provenance depends
on the order their rows were written.

Issue #348.
"""
import numpy as np
import pytest
from astropy.table import Table

from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    PROV_TEXT_COLUMNS, PROV_TEXT_MAX_CHARS, PROV_TEXT_MIN_CHARS,
    _widen_prov_text_columns, update_offsets_table)

#: What m2 writes for a median pooled over two detectors -- 58 characters, and
#: the form six live tables cannot hold.
POOLED_SOURCE = "m2 visit-consensus [median of 2, ptp 1.51mas: nrcb3,nrcb4]"

#: The same for FOUR detectors -- 70 characters.  Four is the number the pooler
#: is built for (one module's detectors), so this is the ordinary case, not the
#: extreme one, and a live example already sits truncated in cloudc's table.
POOLED_SOURCE_4 = ("m2 visit-consensus [median of 4, ptp 3.42mas: "
                   "nrcb1,nrcb2,nrcb3,nrcb4]")

#: A pooled median on top of a cross-band tie -- the longest form the pipeline
#: produces, 102 characters.  w51's table already carries the 62-character tie
#: this is built from.
POOLED_ON_CROSSBAND = ("m2 consensus->reference (cross-band tied-F210M, "
                       "contrast>2900) [median of 4, ptp 3.42mas: "
                       "nrcb1,nrcb2]")


def _narrow_table(tmp_path, prov_source="m2 visit-consensus"):
    """A table whose ``prov_source`` column is as wide as its content and no more.

    This is not a contrived fixture: it is what ``Table.read`` produces from a
    CSV every one of whose rows says ``'m2 visit-consensus'``.
    """
    t = Table()
    t["Visit"] = ["jw01939001001"] * 4
    t["Exposure"] = [1, 2, 3, 4]
    t["Filter"] = ["F212N"] * 4
    t["Module"] = ["nrcb"] * 4
    t["dra (arcsec)"] = np.array([0.10, 0.11, 0.12, 0.13])
    t["ddec (arcsec)"] = np.array([-0.20, -0.21, -0.22, -0.23])
    t["prov_stage"] = ["m2"] * 4
    t["prov_date"] = ["2026-08-08T00:00:00Z"] * 4
    t["prov_source"] = [prov_source] * 4
    t["prov_dra_added_mas"] = np.zeros(4)
    t["prov_ddec_added_mas"] = np.zeros(4)
    p = tmp_path / "Offsets_JWST_Brick1939_VIRAC2locked.csv"
    t.write(p, format="ascii.csv", overwrite=True)
    return str(p)


def _corr(exposure, source, dra=1.51, ddec=-0.80):
    return dict(visit="jw01939001001", exposure=exposure, module="nrcb",
                filtername="F212N", dra_onsky_mas=dra, ddec_onsky_mas=ddec,
                dec_deg=-29.0, source=source)


def test_the_fixture_really_is_too_narrow(tmp_path):
    """Guard the premise: without the widening this table WOULD truncate.

    If astropy ever stops typing a CSV text column to its longest value, this
    test file stops testing anything, and this is the assertion that says so.
    """
    t = Table.read(_narrow_table(tmp_path), format="ascii.csv")
    assert t["prov_source"].dtype.kind == "U"
    assert t["prov_source"].dtype.itemsize // 4 < len(POOLED_SOURCE)


def test_a_pooled_source_string_survives_a_narrow_column(tmp_path):
    """The defect: 58 characters in, 58 characters out."""
    p = _narrow_table(tmp_path)
    update_offsets_table(p, [_corr(2, POOLED_SOURCE)], stage="m2")
    t = Table.read(p, format="ascii.csv")
    assert t["prov_source"][1] == POOLED_SOURCE


def test_the_detector_list_is_what_truncation_removes(tmp_path):
    """Stated as the consequence, so a partial fix cannot pass.

    A regression that widened to, say, 32 characters would keep
    ``'m2 visit-consensus [median of 2,'`` -- still a plausible-looking
    provenance string, and still missing the detectors.
    """
    p = _narrow_table(tmp_path)
    update_offsets_table(p, [_corr(2, POOLED_SOURCE)], stage="m2")
    t = Table.read(p, format="ascii.csv")
    assert "nrcb3,nrcb4" in t["prov_source"][1]


def test_the_other_rows_keep_their_own_provenance(tmp_path):
    """Widening a column must not disturb the values already in it."""
    p = _narrow_table(tmp_path)
    update_offsets_table(p, [_corr(2, POOLED_SOURCE)], stage="m2")
    t = Table.read(p, format="ascii.csv")
    for row in (0, 2, 3):
        assert t["prov_source"][row] == "m2 visit-consensus"


def test_the_four_detector_pooling_survives(tmp_path):
    """The case a fixed 64-character cap still cut, and it is the ORDINARY one.

    A module has four detectors, so a median pooled over four is what the pooler
    is built for; its source string is 70 characters.  The first version of this
    fix capped at 64 and would have gone on dropping the last detector and the
    closing bracket -- and cloudc's live table already carries exactly that,
    stored as `'...nrcb1,nrcb2,'`.
    """
    assert len(POOLED_SOURCE_4) > PROV_TEXT_MIN_CHARS
    p = _narrow_table(tmp_path)
    update_offsets_table(p, [_corr(2, POOLED_SOURCE_4)], stage="m2")
    t = Table.read(p, format="ascii.csv")
    assert t["prov_source"][1] == POOLED_SOURCE_4
    assert t["prov_source"][1].endswith("nrcb4]")


def test_the_longest_form_the_pipeline_produces_survives(tmp_path):
    """A pooled median on top of a cross-band tie: 102 characters."""
    p = _narrow_table(tmp_path)
    update_offsets_table(p, [_corr(2, POOLED_ON_CROSSBAND)], stage="m2")
    t = Table.read(p, format="ascii.csv")
    assert t["prov_source"][1] == POOLED_ON_CROSSBAND


def test_a_source_over_the_hard_bound_is_cut_and_announced(tmp_path, capsys):
    """The bound exists only against absurd input, and it says so when it bites.

    Everything the pipeline produces fits; this is the backstop that stops one
    caller widening every row of a shared table.  A cut here is ANNOUNCED,
    because a provenance string cut without a word is the whole of #348.
    """
    absurd = "x" * (PROV_TEXT_MAX_CHARS + 50)
    p = _narrow_table(tmp_path)
    update_offsets_table(p, [_corr(2, absurd)], stage="m2")
    assert "provenance value truncated" in capsys.readouterr().out
    t = Table.read(p, format="ascii.csv")
    assert t["prov_source"][1] == absurd[:PROV_TEXT_MAX_CHARS]


def test_an_already_wide_table_is_left_alone(tmp_path):
    """Idempotent: the widening only ever grows a column."""
    p = _narrow_table(tmp_path, prov_source="x" * (PROV_TEXT_MIN_CHARS + 40))
    before = Table.read(p, format="ascii.csv")["prov_source"].dtype
    update_offsets_table(p, [_corr(2, "m2 visit-consensus")], stage="m2")
    after = Table.read(p, format="ascii.csv")
    assert after["prov_source"][0] == "x" * (PROV_TEXT_MIN_CHARS + 40)
    assert after["prov_source"].dtype.itemsize >= before.itemsize


def test_an_empty_provenance_cell_is_not_rewritten(tmp_path):
    """The blocker: widening must not resurrect what a mask is hiding.

    Six of the thirteen live tables carry these columns MASKED -- 754 rows
    between them, on rows no correction is touching.  ``np.asarray`` on a
    ``MaskedColumn`` returns the underlying data and discards the mask, and what
    astropy's CSV reader leaves under an empty text cell is the literal string
    ``'0'``.  Widening that way rewrites every one of those cells to ``0``,
    which is a false provenance record written by the fix for missing
    provenance.  ``prov_stage`` is read back into the alignment header, so a
    ``stage='0'`` propagates out of the table.
    """
    t = Table.read(_narrow_table(tmp_path), format="ascii.csv")
    for col in PROV_TEXT_COLUMNS:                      # empty on rows 0, 2, 3
        t[col] = np.ma.array(list(t[col]), mask=[True, False, True, True])
    p = str(tmp_path / "masked.csv")
    t.write(p, format="ascii.csv", overwrite=True)
    raw_before = open(p).read().splitlines()

    update_offsets_table(p, [_corr(2, POOLED_SOURCE_4)], stage="m2")

    raw_after = open(p).read().splitlines()
    changed = [i for i, (b, a) in enumerate(zip(raw_before, raw_after)) if b != a]
    assert changed == [2], (
        f"only the corrected row may change; changed rows {changed}")
    after = Table.read(p, format="ascii.csv")
    assert not any(str(v) == "0" for v in after["prov_source"]), \
        "an empty provenance cell was rewritten to the literal '0'"
    assert not any(str(v) == "0" for v in after["prov_stage"])


def test_the_column_description_survives_widening():
    """``astype`` keeps a column's description and metadata; ``asarray`` does not."""
    t = Table()
    t["prov_source"] = np.array(["m2"] * 3, dtype="U2")
    t["prov_source"].description = "where this correction came from"
    _widen_prov_text_columns(t)
    assert t["prov_source"].description == "where this correction came from"


def test_a_column_that_is_empty_in_every_row_is_still_widened():
    """A text column that is blank throughout a CSV reads back as int64.

    Assigning text into it raises rather than truncating, which is the same
    "column typed by its data" failure from the other end.
    """
    t = Table()
    t["prov_source"] = np.zeros(3, dtype=int)          # what Table.read gives
    _widen_prov_text_columns(t, len(POOLED_SOURCE_4))
    t["prov_source"][0] = POOLED_SOURCE_4
    assert t["prov_source"][0] == POOLED_SOURCE_4


@pytest.mark.parametrize("column", PROV_TEXT_COLUMNS)
def test_every_free_text_provenance_column_is_widened(column):
    """``prov_stage`` and ``prov_date`` are short today and narrow by the same rule.

    ``prov_date`` is fixed-width ISO and ``prov_stage`` is ``m2``..``m12``, so
    neither truncates in practice -- but both are numpy string columns sized by
    whatever CSV they were read from, so both are one longer value away from the
    same failure.  Asserted on the in-memory table, because a CSV round-trip
    re-narrows every column to its own content and would hide the result.
    """
    t = Table()
    t[column] = np.array(["short"] * 3, dtype="U5")
    _widen_prov_text_columns(t)
    assert t[column].dtype.itemsize // 4 == PROV_TEXT_MIN_CHARS
    assert list(t[column]) == ["short"] * 3


def test_widening_does_not_touch_an_unrelated_column():
    """It is keyed on a fixed column list, not on dtype: nothing else moves."""
    t = Table()
    t["Module"] = np.array(["nrcb"] * 3, dtype="U4")
    t["prov_source"] = np.array(["m2"] * 3, dtype="U2")
    _widen_prov_text_columns(t)
    assert t["Module"].dtype.itemsize // 4 == 4
    assert t["prov_source"].dtype.itemsize // 4 == PROV_TEXT_MIN_CHARS
