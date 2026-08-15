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
into one is 59 characters:

    'm2 visit-consensus [mean of 2, maxsep 1.51mas: nrcb3,nrcb4]'

Stored in a ``<U23`` column that becomes ``'m2 visit-consensus [mea'`` -- the
detector list, the part that says which measurements the pooled value came from, is
exactly what is cut.  Six of the thirteen live offsets tables (arches, both
cloudef tables, quintuplet, sgra, sgrb2) are ``<U23`` today; the three that
already carry the long form (cloudc, sgrc, sickle) are wide enough only because
a long string reached them first.  So which tables lose their provenance depends
on the order their rows were written.

Issue #348.
"""
import inspect

import numpy as np
import pytest
from astropy.table import Table

from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    PROV_TEXT_COLUMNS, PROV_TEXT_MAX_CHARS, PROV_TEXT_MIN_CHARS,
    _widen_prov_text_columns, update_offsets_table)

#: What m2 writes for a value pooled over two detectors: 59 characters, and a
#: form six live tables cannot hold.  (It was 58 when the statistic was the
#: median -- "mean" is two characters shorter, and "maxsep" three longer than
#: the "ptp" it replaced, so the string grew by one.)
POOLED_SOURCE = "m2 visit-consensus [mean of 2, maxsep 1.51mas: nrcb3,nrcb4]"

#: The same for FOUR detectors -- 71 characters.  Four is the number the pooler
#: is built for (one module's detectors), so this is the ordinary case, not the
#: extreme one, and a live example already sits truncated in cloudc's table.
POOLED_SOURCE_4 = ("m2 visit-consensus [mean of 4, maxsep 3.42mas: "
                   "nrcb1,nrcb2,nrcb3,nrcb4]")

#: The longest source string the pipeline can emit, BUILT rather than quoted.
#:
#: FOUR literal maxima have been wrong in this file's history, each for its own
#: reason, so this one is assembled from the code that decides its parts:
#:
#:   102  said "mean of 4" while listing only two detectors.
#:   138  listed eight detectors across both modules.  `_assert_poolable`
#:        refuses any group spanning module families --
#:            _assert_poolable(8 corrections, ['nrca1'..'nrcb4'], ...)
#:              -> OffsetsTableUpdateError: corrections spanning module families
#:   114  put a four-detector pooled value on w51's 62-character base,
#:        'm2 consensus->reference (cross-band tied-F210M, contrast>2900)'.
#:        That base is REAL -- three rows of w51's live table carry it -- but no
#:        code at this head writes it (the only sources emitted are
#:        f"{stage} visit-consensus" and f"{stage} consensus->reference"), and it
#:        is a per-visit BULK correction, which the pooler passes through without
#:        pooling.  So it can never take a "[mean of N ...]" suffix.
#:   70/71 fixed the stage token at "m2" and missed "m12", which is in
#:        `CORRECTION_STAGES` and fully wired (cataloging.py passes merge_label
#:        as stage; versioning/rerun.py seeds it).
#:
#: It is not FORMATTED here either.  A previous version of this helper built the
#: string from a hand-copied template, and a reviewer defeated it three ways
#: without any test noticing: rebuilding it on the `consensus->reference` base
#: (the bulk-only one, which is exactly how 114 was arrived at), renaming the
#: producer in the module, and changing the suffix wording.  All three left a
#: stale fixture and a green suite.
#:
#: So the fixture is EMITTED: four synthetic detector corrections go through
#: `pool_corrections_to_table_granularity`, and whatever it returns is what gets
#: tested, so a change to the suffix wording follows automatically.
#:
#: Note what that does NOT settle.  The pooler decides what to pool from the
#: correction's SHAPE -- a bulk correction is one with no exposure and no module
#: -- not from its source text, so handing it the `consensus->reference` string
#: on per-detector corrections pools it happily.  That is why 114 looked
#: plausible.  What makes the longer base un-poolable is that it is only ever
#: written on a BULK correction, and bulk corrections pass through untouched;
#: `test_a_bulk_correction_is_never_given_a_pooled_suffix` pins that directly.
#:
#: Its LENGTH is deliberately not asserted: `ASTROM_MAX_POOL_SPREAD_MAS` is an
#: operator setting, so the spread field's width is not fixed by the source at
#: all.
POOLED_DETECTORS = ("nrcb1", "nrcb2", "nrcb3", "nrcb4")   # one module's four


def _pool_synthetic(base, detectors=POOLED_DETECTORS, spread=None):
    """Run the real pooler over one module's detectors and return its ``source``.

    ``spread`` is the peak-to-peak of the corrections handed in.  The value the
    pooler REPORTS is its own statistic over them, which is not the same number,
    so this does not try to dictate it -- the caller asks only that the reported
    field be two digits wide, which is what makes the string its widest.
    """
    from astropy.table import Table as _T
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        MAX_POOL_SPREAD_MAS, pool_corrections_to_table_granularity)
    if spread is None:
        spread = MAX_POOL_SPREAD_MAS - 0.01
    n = len(detectors)
    offsets = np.linspace(-spread / 2, spread / 2, n)
    tbl = _T({"Visit": ["jw01939001001"] * n, "Exposure": [1] * n,
              "Filter": ["F212N"] * n, "Module": ["nrcb"] * n})
    corrections = [
        dict(visit="jw01939001001", exposure=1, module=det, filtername="F212N",
             dra_onsky_mas=float(o), ddec_onsky_mas=0.0, dec_deg=-29.0,
             source=base)
        for det, o in zip(detectors, offsets)]
    pooled = pool_corrections_to_table_granularity(
        corrections, "Offsets_JWST_Brick1939_VIRAC2locked.csv", tbl=tbl)
    assert len(pooled) == 1, (
        f"{base!r} did not pool into one row; got {len(pooled)}")
    return pooled[0]["source"]


def _longest_emittable_source():
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        CORRECTION_STAGES)
    stage = max(CORRECTION_STAGES, key=len)
    return _pool_synthetic(f"{stage} visit-consensus")


POOLED_SOURCE_MAX = _longest_emittable_source()

#: A stage label longer than the "m2"/"m12" the source string uses.  The `stage`
#: ARGUMENT is not restricted to `CORRECTION_STAGES`:
#: `apply_m2_checkpoint_corrections.py` passes "m2cycle2", and brick's two
#: VIRAC2locked tables carry it on 240 rows between them.  So `prov_stage` is a
#: column a narrowing can truncate on live data, which is why the revert test
#: asserts all three provenance columns rather than `prov_source` alone.
LONG_STAGE = "m2cycle2"


def _emitted_sources():
    """Every ``source=f"{stage} ..."`` the checkpoint writes, with its SHAPE.

    Returns ``{(source_text_after_the_stage, is_bulk), ...}`` by walking the
    module's AST, not its text.  Two reasons it is not a regex:

    * a text match sees only the quote style it was written for, and
      `source=f'{stage} ...'` in single quotes slips straight past it.  That is
      the defect the previous guard in this file was replaced for; re-creating
      it in the replacement is how it came back.
    * the SHAPE matters as much as the string.  ``_is_bulk_correction`` is "no
      exposure AND no module", and the longer base can only stay un-poolable
      while it is written on a bulk correction.  Reading the same keyword
      arguments the literal is written beside is what pins that at the call
      site, rather than asserting it about a hand-built dict elsewhere.
    """
    import ast
    from jwst_gc_pipeline.photometry import astrometry_checkpoint as _ac
    tree = ast.parse(inspect.getsource(_ac))
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        src = kw.get("source")
        if not isinstance(src, ast.JoinedStr):
            continue
        # f"{stage} <literal>" -> take the literal tail
        tail = "".join(v.value for v in src.values
                       if isinstance(v, ast.Constant) and isinstance(v.value, str))
        tail = tail.strip()
        if not tail:
            continue
        is_bulk = all(
            isinstance(kw.get(a), ast.Constant) and kw[a].value is None
            for a in ("exposure", "module"))
        found.add((tail, is_bulk))
    return found

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
    """The defect: 59 characters in, 59 characters out."""
    p = _narrow_table(tmp_path)
    update_offsets_table(p, [_corr(2, POOLED_SOURCE)], stage="m2")
    t = Table.read(p, format="ascii.csv")
    assert t["prov_source"][1] == POOLED_SOURCE


def test_the_detector_list_is_what_truncation_removes(tmp_path):
    """Stated as the consequence, so a partial fix cannot pass.

    A regression that widened to, say, 32 characters would keep
    ``'m2 visit-consensus [mean of 2,'`` -- still a plausible-looking
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

    A module has four detectors, so a value pooled over four is what the pooler
    is built for; its source string is 71 characters.  The first version of this
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
    """The widest emittable source: longest stage, four detectors, widest spread.

    No literal length is asserted -- four of those have been wrong.  What is
    asserted is that the built string is longer than the floor (so it exercises
    the widening at all) and inside the outer bound.
    """
    assert len(POOLED_SOURCE_MAX) > PROV_TEXT_MIN_CHARS
    assert len(POOLED_SOURCE_MAX) < PROV_TEXT_MAX_CHARS
    p = _narrow_table(tmp_path)
    update_offsets_table(p, [_corr(2, POOLED_SOURCE_MAX)], stage="m2")
    t = Table.read(p, format="ascii.csv")
    assert t["prov_source"][1] == POOLED_SOURCE_MAX


def test_the_fixture_is_a_string_the_pooler_can_actually_emit(tmp_path):
    """Guards the fixture itself -- THREE times now it has not been.

    Two things have to hold, and only the first was checked before:

      1. the detector list is one module's four, because `_assert_poolable`
         refuses a group spanning module families; and
      2. the BASE it is appended to is a string the checkpoint actually writes.

    (2) is what the 114-character fixture failed: its base was real -- it is on
    disk in w51's table -- but nothing at this head emits it, and it belongs to a
    bulk correction, which is passed through unpooled and so never takes a
    "[mean of N ...]" suffix at all.
    """
    import inspect
    import re
    from jwst_gc_pipeline.photometry import astrometry_checkpoint as ac
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        _assert_poolable, OffsetsTableUpdateError)
    from astropy.table import Table as _T

    mods = ["nrcb1", "nrcb2", "nrcb3", "nrcb4"]
    for m in mods:                      # every detector named must appear
        assert m in POOLED_SOURCE_MAX
    assert f"mean of {len(mods)}" in POOLED_SOURCE_MAX
    _assert_poolable([{}] * len(mods), mods, "row", _T(), "t.csv")   # allowed
    with pytest.raises(OffsetsTableUpdateError, match="module families"):
        _assert_poolable([{}] * 8, ["nrca1", "nrca2", "nrca3", "nrca4"] + mods,
                         "row", _T(), "t.csv")

    # (2): the part before " [mean of" must be a source the module emits.
    #
    # An `any()` over hardcoded literals is not enough: renaming one producer
    # leaves the other satisfying it while the fixture goes stale.  So compare
    # the SET -- and find it by walking the AST rather than by matching source
    # TEXT, because a text match is defeated by a quote character, which is the
    # exact defect this guard's predecessor was replaced for.
    base = POOLED_SOURCE_MAX.split(" [mean of")[0]
    emitted = {src for src, _bulk in _emitted_sources()}
    assert emitted == {"visit-consensus", "consensus->reference"}, (
        f"the module's source-string producers are now {sorted(emitted)}; "
        f"rebuild the fixture from them rather than assuming the old pair")
    stage, _, rest = base.partition(" ")
    assert rest == "visit-consensus", (
        f"{rest!r} is not the base a POOLED correction carries -- see "
        f"test_a_bulk_correction_is_never_given_a_pooled_suffix")
    assert stage in ac.CORRECTION_STAGES, (
        f"{stage!r} is not a stage that writes corrections; the maximum has to "
        f"use the LONGEST of CORRECTION_STAGES, which is where the 70/71 "
        f"measurement went wrong by fixing it at 'm2'")
    assert stage == max(ac.CORRECTION_STAGES, key=len)
    # ...and the spread must be one the pooler would admit.
    # `maxsep`, not `ptp`: the dispersion is the largest separation between
    # any two members as VECTORS, not a peak-to-peak of their magnitudes,
    # and the emitted token was renamed with it.
    spread = float(POOLED_SOURCE_MAX.split("maxsep ")[1].split("mas")[0])
    assert spread <= ac.MAX_POOL_SPREAD_MAS


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


def _load_revert_tool():
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location(
        "revert_broadcast_provenance",
        os.path.join(os.path.dirname(__file__), "..", "..", "..",
                     "scripts", "reduction", "revert_broadcast_provenance.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _broadcast_table(path, prov_source):
    """A table `revert_broadcast_provenance` will act on.

    Two visits of one filter carrying the IDENTICAL prov_* pair is what
    ``flag_broadcast_provenance`` fires on -- distinct pointings cannot need the
    same correction to a fraction of a mas -- so this is the smallest input that
    reaches the tool's write path.  ``prov_source`` starts as wide as its own
    content, which is what a CSV round-trip produces.
    """
    t = Table()
    t["Visit"] = np.array(["001", "002"], dtype="U3")
    t["Filter"] = np.array(["F405N", "F405N"], dtype="U5")
    t["dra"] = np.array([0.10, 0.20])
    t["ddec"] = np.array([0.30, 0.40])
    t["dra (arcsec)"] = np.array([1.10, 1.20])
    t["ddec (arcsec)"] = np.array([1.30, 1.40])
    t["prov_dra_added_mas"] = np.array([1000.0, 1000.0])
    t["prov_ddec_added_mas"] = np.array([2000.0, 2000.0])
    t["prov_stage"] = np.array([LONG_STAGE] * 2, dtype=f"U{len(LONG_STAGE)}")
    t["prov_date"] = np.array(["2026-08-10T00:00:00Z"] * 2)
    t["prov_source"] = np.array([prov_source] * 2)
    t.write(path, format="ascii.csv", overwrite=True)
    return t


def test_the_revert_tool_does_not_narrow_a_wide_column(tmp_path):
    """`revert_broadcast_provenance` writes into live tables and must not cut them.

    It previously hardcoded ``astype("U64")``, which NARROWS any column already
    wider than 64.  Not hypothetical: gc2211's live table carries 240 rows
    written by that script, and its ``prov_source`` column is ``<U27``.

    This exercises the tool's ACTUAL WRITE PATH (``revert(..., apply=True)``)
    rather than asserting on its source text.  A source-text check is defeated
    by a quote character -- ``astype('U64')`` with single quotes passes it -- and
    it cannot see whether the write itself truncates.
    """
    mod = _load_revert_tool()
    p = str(tmp_path / "Offsets_JWST_Brick9999_VIRAC2locked.csv")
    _broadcast_table(p, POOLED_SOURCE_MAX)

    before = Table.read(p, format="ascii.csv")
    assert before["prov_source"].dtype.itemsize // 4 == len(POOLED_SOURCE_MAX)
    assert mod.revert(p, apply=False) == 2, "fixture must reach the write path"

    assert mod.revert(p, apply=True) == 2
    after = Table.read(p, format="ascii.csv")

    # the reverted rows say a revert happened, in full
    assert list(after["prov_source"]) == ["revert_broadcast_provenance"] * 2
    assert list(after["prov_stage"]) == ["revert"] * 2
    # and the applied pair was restored from the as-built one
    assert list(after["dra (arcsec)"]) == [0.10, 0.20]

    # A row the revert does not touch keeps its full-length source.  Same table,
    # one filter reverted and one left alone.
    p2 = str(tmp_path / "Offsets_JWST_Brick9998_VIRAC2locked.csv")
    t = _broadcast_table(p2, POOLED_SOURCE_MAX)
    t.add_row(["003", "F212N", 0.5, 0.6, 1.5, 1.6, 5.0, 6.0,
               LONG_STAGE, "2026-08-10T00:00:00Z", POOLED_SOURCE_MAX])
    t.write(p2, format="ascii.csv", overwrite=True)
    mod.revert(p2, apply=True)
    kept = Table.read(p2, format="ascii.csv")
    untouched = kept[np.asarray([str(f) for f in kept["Filter"]]) == "F212N"]
    # ALL THREE columns, not just prov_source.  A narrowing of prov_stage alone
    # passed this test while truncating live data: brick's two VIRAC2locked
    # tables carry prov_stage='m2cycle2' on 240 rows between them, and that is
    # the column read back into the frame's alignment header.
    for col, want in (("prov_source", POOLED_SOURCE_MAX),
                      ("prov_stage", LONG_STAGE),
                      ("prov_date", "2026-08-10T00:00:00Z")):
        assert untouched[col][0] == want, (
            f"the write narrowed {col} and cut an untouched row's provenance")


def test_widening_to_the_baseline_never_shortens_a_wider_column(tmp_path):
    """The helper's floor is a floor: a column already wider is left alone."""
    t = Table()
    t["prov_stage"] = np.array(["m2"] * 3, dtype="U2")
    t["prov_date"] = np.array(["2026-08-10T00:00:00Z"] * 3)
    t["prov_source"] = np.array([POOLED_SOURCE_MAX] * 3)
    assert t["prov_source"].dtype.itemsize // 4 == len(POOLED_SOURCE_MAX)
    _widen_prov_text_columns(t, PROV_TEXT_MIN_CHARS)
    assert t["prov_source"][0] == POOLED_SOURCE_MAX


def test_an_object_dtype_column_is_not_narrowed_into_shape(tmp_path):
    """Object columns hold strings of any length; widening must not cut them.

    ``_string_column_chars`` returns None for object dtype, which used to mean
    "widen to the floor unconditionally" -- so a 103-character value in an
    object column came back at 64, silently, from the function whose whole job
    is to stop silent cuts.
    """
    long_value = "m2 visit-consensus " + "x" * 84          # 103 characters
    t = Table()
    t["prov_source"] = np.array([long_value, "m2"], dtype=object)
    assert t["prov_source"].dtype.kind == "O"
    _widen_prov_text_columns(t, PROV_TEXT_MIN_CHARS)
    assert t["prov_source"][0] == long_value
    assert t["prov_source"].dtype.itemsize // 4 >= len(long_value)


def test_a_bulk_correction_is_never_given_a_pooled_suffix():
    """Why the longer of the two bases can never reach the pooled form.

    `consensus->reference` is five characters longer than `visit-consensus`, so
    a pooled value written on top of it would be the longest string in the
    file -- which is exactly the reasoning that produced the retracted 114.

    It cannot happen, but NOT because the pooler inspects the text.  It pools on
    the correction's SHAPE: `_is_bulk_correction` is "no exposure AND no
    module", and the `consensus->reference` string is only ever written on such
    a correction (`astrometry_checkpoint.py`, the per-visit reference tie).
    Bulk corrections are passed through untouched, so they never acquire a
    `[mean of N ...]` suffix.

    Pinned here directly, because the fixture guard cannot see it: hand the
    pooler that same string on PER-DETECTOR corrections and it pools it happily.
    """
    from astropy.table import Table as _T
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        _is_bulk_correction, pool_corrections_to_table_granularity)

    base = "m12 consensus->reference (cross-band tied-F210M, contrast>2900)"
    bulk = dict(visit="jw01939001001", exposure=None, module=None,
                filtername="F212N", dra_onsky_mas=1.5, ddec_onsky_mas=-0.8,
                dec_deg=-29.0, source=base)
    assert _is_bulk_correction(bulk)

    tbl = _T({"Visit": ["jw01939001001"], "Exposure": [1],
              "Filter": ["F212N"], "Module": ["nrcb"]})
    out = pool_corrections_to_table_granularity(
        [bulk], "Offsets_JWST_Brick1939_VIRAC2locked.csv", tbl=tbl)
    assert len(out) == 1
    assert out[0]["source"] == base, "a bulk correction was pooled"
    assert "[mean of" not in out[0]["source"]

    # and the shape is what decides it, not the text: the same string on
    # per-detector corrections DOES pool.  This is the trap 114 fell into.
    assert "[mean of" in _pool_synthetic(base)

    # THE INVARIANT, PINNED WHERE IT LIVES.  The two assertions above are about
    # a dict built here; neither says anything about how the module writes that
    # string.  Add one per-exposure producer carrying the longer base and both
    # still pass, while the longest emittable source jumps from 72 to 77 -- the
    # retracted 114 made real by the guard built to catch it.  So read the SHAPE
    # off the call sites: every source longer than the one the fixture uses must
    # be written on a bulk correction, and the fixture's own base must not be.
    by_src = dict(_emitted_sources())
    fixture_base = POOLED_SOURCE_MAX.split(" [mean of")[0].partition(" ")[2]
    assert by_src[fixture_base] is False, (
        f"{fixture_base!r} is now written on a bulk correction, which the "
        f"pooler passes through -- the fixture can no longer be pooled")
    for src, is_bulk in by_src.items():
        if len(src) > len(fixture_base):
            assert is_bulk, (
                f"{src!r} is longer than the fixture's base and is written on a "
                f"NON-bulk correction, so it can take a pooled suffix -- the "
                f"longest emittable source is no longer POOLED_SOURCE_MAX")
