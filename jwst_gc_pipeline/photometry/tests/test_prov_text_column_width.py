"""``prov_source`` must record the whole source string, not its first N chars.

An offsets table is stored as CSV and read back with ``Table.read``, so each of
its text columns is typed to the longest string that file happened to contain --
``<U23`` for a table whose provenance so far has only ever said ``'m2
visit-consensus'``.  Assigning a longer string into a numpy string column
truncates it: the leading characters are kept and the rest dropped.  astropy
does emit a ``StringTruncateWarning``, but it is one line in a log carrying
thousands, and nothing downstream can tell a truncated value from a short one --
which is the part that matters.

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
#: produces.  w51's live table already carries the 62-character base.  114
#: characters over four detectors; the eight-detector form below is 138, which
#: is the true maximum and still well under PROV_TEXT_MAX_CHARS.
#:
#: (An earlier version of this constant said "median of 4" while listing two
#: detectors -- a string the pooler cannot emit -- and was quoted as the
#: pipeline's longest form at 102 characters.  Both wrong.)
POOLED_ON_CROSSBAND = ("m2 consensus->reference (cross-band tied-F210M, "
                       "contrast>2900) [median of 4, ptp 3.42mas: "
                       "nrcb1,nrcb2,nrcb3,nrcb4]")

#: The genuine maximum: eight detectors on the longest base.
POOLED_ON_CROSSBAND_8 = ("m2 consensus->reference (cross-band tied-F210M, "
                         "contrast>2900) [median of 8, ptp 3.42mas: "
                         "nrca1,nrca2,nrca3,nrca4,nrcb1,nrcb2,nrcb3,nrcb4]")


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


@pytest.mark.parametrize("source", [POOLED_ON_CROSSBAND, POOLED_ON_CROSSBAND_8])
def test_the_longest_form_the_pipeline_produces_survives(tmp_path, source):
    """A pooled median on the longest base: 114 chars over 4 detectors, 138 over 8."""
    assert len(source) > PROV_TEXT_MIN_CHARS
    assert len(source) < PROV_TEXT_MAX_CHARS
    p = _narrow_table(tmp_path)
    update_offsets_table(p, [_corr(2, source)], stage="m2")
    t = Table.read(p, format="ascii.csv")
    assert t["prov_source"][1] == source


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


def test_the_revert_tool_does_not_narrow_a_wide_column(tmp_path):
    """`revert_broadcast_provenance` writes into live tables and must not cut them.

    It previously hardcoded ``astype("U64")``, which NARROWS any column already
    wider than 64 -- and the longest string this pipeline writes is 102
    characters.  Not hypothetical: gc2211's live table carries 240 rows written
    by that script, and its ``prov_source`` column is ``<U27``.
    """
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location(
        "revert_broadcast_provenance",
        os.path.join(os.path.dirname(__file__), "..", "..", "..",
                     "scripts", "reduction", "revert_broadcast_provenance.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    t = Table()
    t["prov_stage"] = np.array(["m2"] * 3, dtype="U2")
    t["prov_date"] = np.array(["2026-08-10T00:00:00Z"] * 3)
    t["prov_source"] = np.array([POOLED_ON_CROSSBAND] * 3)
    assert t["prov_source"].dtype.itemsize // 4 == len(POOLED_ON_CROSSBAND)

    _widen_prov_text_columns(t, PROV_TEXT_MIN_CHARS)
    assert t["prov_source"][0] == POOLED_ON_CROSSBAND, (
        "widening to the baseline must not shorten a column already wider")
    # and the tool reaches for the same helper rather than a literal width.
    # Comments are stripped first, so the explanation of WHY the literal was
    # wrong does not itself trip the check.
    code = "\n".join(l.split("#", 1)[0] for l in open(mod.__file__))
    assert "_widen_prov_text_columns" in code
    assert 'astype("U' not in code, "a hardcoded column width is back"
