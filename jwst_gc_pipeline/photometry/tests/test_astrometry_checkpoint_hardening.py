"""Astrometry-checkpoint hardening (issue #111): three ways the gate could be
silently neutered rather than loudly fail.

1. The m2 correction FLOOR read the correction magnitude with
   ``.get(..., 0.0)``.  A correction source that omitted the keys would read
   magnitude 0 for every correction, so every correction would be "sub-floor"
   and the checkpoint would PASS without ever applying anything.
2. The frozen-stage baseline READER interpolated the filter name bare while the
   WRITER used ``filtername or 'all'``.  A ``filtername=None`` caller wrote
   ``checkpoint_m2_all`` and the reader asked for a name that does not exist ->
   "no m2 baseline record found" -> a FALSE regression at every frozen stage.
3. A frozen stage whose reference tie both MOVED since the m2 freeze and went
   INCOHERENT (``apply_ok`` False) fell into the could-not-verify branch and the
   run continued -- a silent exit from the gate in exactly the case the gate
   exists for.  The blocking condition is MOVEMENT (the frozen-stage contract
   everywhere else in the module), not the absolute magnitude of the tie: an
   incoherent tie is never applied, so the residual m2 passed with survives into
   every later stage, and testing it absolutely re-trips on scatter m2 already
   tolerated.

All of these are hermetic: the heavy consensus/reference numerics are
monkeypatched (they are covered by test_visit_consensus).
"""
import json
import os
import types

import pytest
from astropy.table import Table

import jwst_gc_pipeline.photometry.astrometry_checkpoint as _ac
from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    REFERENCE_APPLY_MIN_MAS, AstrometryRegressionError, run_visit_checkpoint,
)
from jwst_gc_pipeline.photometry.cataloging import (
    _run_astrometry_stage_checkpoint,
)
from .test_astrometry_checkpoint import (
    _DUMMY_REFCAT, _patch_consensus_and_tie, _tiny_visit_table,
    _write_m2_baseline,
)


# ---------------------------------------------------------------------------
# 1. the m2 correction floor must read the magnitude LOUDLY
# ---------------------------------------------------------------------------

def _perframe_layout(tmp_path, filt="f212n", label="m2"):
    """One readable per-frame catalog where the stage checkpoint globs for it."""
    d = tmp_path / filt.upper()
    d.mkdir(parents=True, exist_ok=True)
    name = (f"{filt}_nrcb1_visit001_vgroup02101_exp00001_{label}"
            f"_daophot_basic.fits")
    Table({"skycoord_ra": [266.5], "skycoord_dec": [-28.7]}).write(
        d / name, overwrite=True)
    return str(d / name)


def _stub_checkpoint(tmp_path, monkeypatch, corrections, filt="f212n"):
    """Drive _run_astrometry_stage_checkpoint's floor filter over ``corrections``
    without running any real photometry: run_visit_checkpoint is replaced by a
    stub that returns exactly these corrections."""
    _perframe_layout(tmp_path, filt=filt)

    def _fake_checkpoint(tables, stage, **kw):
        return dict(stage=stage, corrections=list(corrections), failures=[],
                    unverified=[], passed=True,
                    record_path=str(tmp_path / "rec.json"))

    monkeypatch.setattr(_ac, "run_visit_checkpoint", _fake_checkpoint)


def _invoke(tmp_path, monkeypatch, corrections, floor="4.0", filt="f212n"):
    monkeypatch.setenv("ASTROM_M2_CORRECTION_FLOOR_MAS", floor)
    monkeypatch.delenv("ASTROM_CHECKPOINT_APPLY", raising=False)
    monkeypatch.delenv("ASTROM_CHECKPOINT_WARN_ONLY", raising=False)
    _stub_checkpoint(tmp_path, monkeypatch, corrections, filt=filt)
    return _run_astrometry_stage_checkpoint(
        "m2", "nrcb", filt, str(tmp_path), str(tmp_path), "2221",
        types.SimpleNamespace(cutout_region=""), {"refcat": None},
        context="test")


def _corr(dra=None, ddec=None, **extra):
    c = dict(visit="001", exposure=1, module="nrcb", filtername="F212N",
             dec_deg=-28.7, source="test")
    if dra is not None:
        c["dra_onsky_mas"] = dra
    if ddec is not None:
        c["ddec_onsky_mas"] = ddec
    c.update(extra)
    return c


def test_floor_raises_when_a_correction_lacks_the_magnitude_keys(
        tmp_path, monkeypatch):
    """The defect this closes: a correction source that omits the magnitude keys
    used to read magnitude 0 -> "sub-floor" -> PASS.  A 30 mas misalignment
    would have been silently discarded."""
    with pytest.raises(KeyError, match="dra_onsky_mas"):
        _invoke(tmp_path, monkeypatch, [_corr(dra=30.0)])       # ddec missing


def test_floor_raises_when_both_magnitude_keys_are_absent(tmp_path, monkeypatch):
    with pytest.raises(KeyError, match="silently"):
        _invoke(tmp_path, monkeypatch, [_corr()])


def test_floor_still_passes_genuinely_subfloor_corrections(tmp_path, monkeypatch):
    """Positive control: WITH the keys, a sub-floor correction is still a PASS
    (the floor's intended behaviour is unchanged)."""
    _invoke(tmp_path, monkeypatch, [_corr(dra=1.0, ddec=0.5)])   # must not raise


def test_floor_off_does_not_inspect_the_keys(tmp_path, monkeypatch):
    """With no floor configured (the default) the filter never runs, so a
    key-less correction is not this function's problem to diagnose -- it flows
    on to the apply path, which is already key-loud."""
    # ASTROM_CHECKPOINT_APPLY unset -> the apply path is a no-op, so this simply
    # must not raise KeyError from the floor.
    monkeypatch.setenv("ASTROM_M2_CORRECTION_FLOOR_MAS", "0")
    monkeypatch.delenv("ASTROM_CHECKPOINT_APPLY", raising=False)
    monkeypatch.delenv("ASTROM_CHECKPOINT_WARN_ONLY", raising=False)
    _stub_checkpoint(tmp_path, monkeypatch, [_corr()])
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        AstrometryCorrectionRequiredError)
    with pytest.raises(AstrometryCorrectionRequiredError):
        _run_astrometry_stage_checkpoint(
            "m2", "nrcb", "f212n", str(tmp_path), str(tmp_path), "2221",
            types.SimpleNamespace(cutout_region=""), {"refcat": None},
            context="test")


# ---------------------------------------------------------------------------
# 2. baseline reader/writer record-name symmetry
# ---------------------------------------------------------------------------

def test_record_name_is_the_single_source_of_truth():
    assert _ac._record_name("m2", "F212N") == "checkpoint_m2_F212N"
    assert _ac._record_name("m2", None) == "checkpoint_m2_all"
    assert _ac._record_name("m2", "") == "checkpoint_m2_all"


def test_baseline_reader_finds_the_filterless_record(tmp_path, monkeypatch):
    """The writer's ``or 'all'`` fallback, mirrored at the reader.  m2 was
    written by a ``filtername=None`` caller (-> checkpoint_m2_all) while the
    frozen-stage reader keys on the per-group filter (F212N).  Before the fix the
    reader missed the baseline entirely and the STABLE tie raised a FALSE
    regression."""
    _write_m2_baseline(str(tmp_path), 10.0, 0.0, filt="all")
    _patch_consensus_and_tie(monkeypatch, dra_now=10.0, ddec_now=0.0)
    rec = run_visit_checkpoint([_tiny_visit_table()], "m3", refcat=_DUMMY_REFCAT,
                               filtername="F212N", record_dir=str(tmp_path),
                               context="test")
    assert rec["passed"], rec["failures"]
    assert rec["failures"] == []


def test_filterless_writer_then_reader_round_trip(tmp_path, monkeypatch):
    """End-to-end: an m2 run with filtername=None writes the record, and the
    frozen m3 run reads that same record back and finds the tie STABLE."""
    _patch_consensus_and_tie(monkeypatch, dra_now=10.0, ddec_now=0.0)
    m2 = run_visit_checkpoint([_tiny_visit_table()], "m2", refcat=_DUMMY_REFCAT,
                              filtername=None, record_dir=str(tmp_path),
                              context="test")
    assert m2["correcting"] and len(m2["corrections"]) == 1
    assert os.path.exists(tmp_path / "checkpoint_m2_all_latest.json")
    m3 = run_visit_checkpoint([_tiny_visit_table()], "m3", refcat=_DUMMY_REFCAT,
                              filtername=None, record_dir=str(tmp_path),
                              context="test")
    assert m3["passed"], m3["failures"]


def test_exact_filter_record_wins_over_the_filterless_one(tmp_path, monkeypatch):
    """When both spellings exist the exact-filter record is authoritative: the
    F212N baseline (10,0) matches the current tie -> STABLE, even though the
    ``_all`` record says (99,0) and would look like a huge movement."""
    _write_m2_baseline(str(tmp_path), 10.0, 0.0, filt="F212N")
    _write_m2_baseline(str(tmp_path), 99.0, 0.0, filt="all")
    _patch_consensus_and_tie(monkeypatch, dra_now=10.0, ddec_now=0.0)
    rec = run_visit_checkpoint([_tiny_visit_table()], "m3", refcat=_DUMMY_REFCAT,
                               filtername="F212N", record_dir=str(tmp_path),
                               context="test")
    assert rec["passed"], rec["failures"]


# ---------------------------------------------------------------------------
# 3. frozen-stage incoherence is a HARD stop, not a soft "unverified"
# ---------------------------------------------------------------------------

_MOVED = REFERENCE_APPLY_MIN_MAS + 8.0     # comfortably above the apply floor


def test_frozen_stage_incoherent_moved_tie_raises(tmp_path, monkeypatch):
    """The soft exit this closes: m2 froze the tie at (0.5, 0); it has since
    moved to (10, 0) AND apply_ok is False, so it can neither be verified nor
    corrected over.  It used to be recorded as `unverified` and the run
    continued."""
    monkeypatch.delenv("ALLOW_LATE_STAGE_ASTROM_SHIFT", raising=False)
    monkeypatch.delenv("ASTROM_ALLOW_FROZEN_INCOHERENT_TIE", raising=False)
    _write_m2_baseline(str(tmp_path), 0.5, 0.0)
    _patch_consensus_and_tie(monkeypatch, dra_now=_MOVED, ddec_now=0.0,
                             apply_ok=False)
    with pytest.raises(AstrometryRegressionError, match="incoherent"):
        run_visit_checkpoint([_tiny_visit_table()], "m3", refcat=_DUMMY_REFCAT,
                             filtername="F212N", record_dir=str(tmp_path),
                             context="test")


def test_frozen_stage_incoherent_tie_that_did_not_move_does_not_raise(
        tmp_path, monkeypatch):
    """The blocking condition is MOVEMENT, not absolute magnitude.

    The sgra shape, straight off disk: a ~49 mas tie that reads the SAME ~49 mas
    at m2 (where it PASSed, because an incoherent tie is never applied) and at
    m3-m6, having moved 0.1-0.5 mas.  Nothing regressed between the freeze and
    now, so the frozen-stage gate has nothing to say; the residual stays in
    `unverified` where m2 left it.  Testing the absolute magnitude here would
    re-trip on scatter m2 already tolerated -- the same failure mode as the
    brick F182M m3 "MOVED 5.86 mas" false regression.
    """
    monkeypatch.delenv("ALLOW_LATE_STAGE_ASTROM_SHIFT", raising=False)
    monkeypatch.delenv("ASTROM_ALLOW_FROZEN_INCOHERENT_TIE", raising=False)
    _write_m2_baseline(str(tmp_path), 49.0, 0.0)
    _patch_consensus_and_tie(monkeypatch, dra_now=49.2, ddec_now=0.0,
                             apply_ok=False)
    rec = run_visit_checkpoint([_tiny_visit_table()], "m5", refcat=_DUMMY_REFCAT,
                               filtername="F212N", record_dir=str(tmp_path),
                               context="test")
    assert rec["failures"] == []
    assert rec["passed"]
    assert rec["unverified"]              # still audited, just not blocking
    assert not rec["all_verified"]


def test_frozen_stage_incoherent_tie_without_a_baseline_fails_closed(
        tmp_path, monkeypatch):
    """No m2 record => the movement question cannot be answered at all, so the
    gate blocks (mirrors the coherent branch's no-baseline behaviour)."""
    monkeypatch.delenv("ALLOW_LATE_STAGE_ASTROM_SHIFT", raising=False)
    monkeypatch.delenv("ASTROM_ALLOW_FROZEN_INCOHERENT_TIE", raising=False)
    _patch_consensus_and_tie(monkeypatch, dra_now=_MOVED, ddec_now=0.0,
                             apply_ok=False)
    with pytest.raises(AstrometryRegressionError, match="NO m2 baseline record"):
        run_visit_checkpoint([_tiny_visit_table()], "m3", refcat=_DUMMY_REFCAT,
                             filtername="F212N", record_dir=str(tmp_path),
                             context="test")


def test_frozen_stage_incoherent_tie_movement_reads_the_samestar_baseline(
        tmp_path, monkeypatch):
    """The movement test must read the SAME estimator the current tie reports
    (the same-star reported bulk), not the histogram ``vs_full`` -- otherwise it
    reproduces the brick F182M m3 false regression on the incoherence path
    instead of the coherent one.  m2 recorded bulk (+1,-6) with vs_full
    (+6.7,-7.5); the tie still reads (+1,-6) and is incoherent -> no movement,
    no stop."""
    monkeypatch.delenv("ALLOW_LATE_STAGE_ASTROM_SHIFT", raising=False)
    monkeypatch.delenv("ASTROM_ALLOW_FROZEN_INCOHERENT_TIE", raising=False)
    _write_m2_baseline(str(tmp_path), 1.0, -6.0, vs_full=(6.7, -7.5))
    _patch_consensus_and_tie(monkeypatch, dra_now=1.0, ddec_now=-6.0,
                             apply_ok=False)
    rec = run_visit_checkpoint([_tiny_visit_table()], "m3", refcat=_DUMMY_REFCAT,
                               filtername="F212N", record_dir=str(tmp_path),
                               context="test")
    assert rec["failures"] == []
    assert rec["passed"]


def test_frozen_stage_incoherent_tie_without_a_finite_bulk_fails_closed(
        tmp_path, monkeypatch):
    """A tie that reports an offset but no finite bulk cannot be differenced
    against the baseline; the movement question is unanswerable, so block."""
    monkeypatch.delenv("ALLOW_LATE_STAGE_ASTROM_SHIFT", raising=False)
    monkeypatch.delenv("ASTROM_ALLOW_FROZEN_INCOHERENT_TIE", raising=False)
    _write_m2_baseline(str(tmp_path), 1.0, -6.0)
    _patch_consensus_and_tie(monkeypatch, dra_now=_MOVED, ddec_now=0.0,
                             apply_ok=False)
    real_tie = _ac.measure_reference_tie

    def _no_bulk(*args, **kwargs):
        tie = real_tie(*args, **kwargs)
        tie["dra_mas"] = None
        tie["ddec_mas"] = None
        return tie

    monkeypatch.setattr(_ac, "measure_reference_tie", _no_bulk)
    with pytest.raises(AstrometryRegressionError, match="no finite bulk"):
        run_visit_checkpoint([_tiny_visit_table()], "m3", refcat=_DUMMY_REFCAT,
                             filtername="F212N", record_dir=str(tmp_path),
                             context="test")


def test_frozen_stage_incoherent_tie_is_also_recorded_unverified(
        tmp_path, monkeypatch):
    """The audit trail is preserved: the case still appears in `unverified` (so
    the record's all_verified is False), it is merely ALSO blocking."""
    monkeypatch.setenv("ALLOW_LATE_STAGE_ASTROM_SHIFT", "1")
    _write_m2_baseline(str(tmp_path), 0.5, 0.0)
    _patch_consensus_and_tie(monkeypatch, dra_now=_MOVED, ddec_now=0.0,
                             apply_ok=False)
    rec = run_visit_checkpoint([_tiny_visit_table()], "m4", refcat=_DUMMY_REFCAT,
                               filtername="F212N", record_dir=str(tmp_path),
                               context="test")
    assert not rec["passed"]
    assert not rec["all_verified"]
    assert rec["unverified"] and rec["failures"]
    assert "not trustworthy" in rec["unverified"][0]


def test_frozen_stage_incoherent_tie_narrow_escape(tmp_path, monkeypatch):
    """A narrow escape exists so the global warn-only switch is not the only way
    out (same shape as ASTROM_ALLOW_MISSING_PERFRAME)."""
    monkeypatch.delenv("ALLOW_LATE_STAGE_ASTROM_SHIFT", raising=False)
    monkeypatch.setenv("ASTROM_ALLOW_FROZEN_INCOHERENT_TIE", "1")
    _write_m2_baseline(str(tmp_path), 0.5, 0.0)
    _patch_consensus_and_tie(monkeypatch, dra_now=_MOVED, ddec_now=0.0,
                             apply_ok=False)
    rec = run_visit_checkpoint([_tiny_visit_table()], "m4", refcat=_DUMMY_REFCAT,
                               filtername="F212N", record_dir=str(tmp_path),
                               context="test")
    assert rec["passed"]              # demoted back to the old soft behaviour
    assert rec["failures"] == []
    assert rec["unverified"]


def test_correcting_stage_incoherent_tie_is_unchanged(tmp_path, monkeypatch):
    """m2 is the CORRECTING stage: an untrustworthy tie there is still merely
    unverified (there is nothing frozen yet to regress from)."""
    monkeypatch.delenv("ALLOW_LATE_STAGE_ASTROM_SHIFT", raising=False)
    monkeypatch.delenv("ASTROM_ALLOW_FROZEN_INCOHERENT_TIE", raising=False)
    _patch_consensus_and_tie(monkeypatch, dra_now=_MOVED, ddec_now=0.0,
                             apply_ok=False)
    rec = run_visit_checkpoint([_tiny_visit_table()], "m2", refcat=_DUMMY_REFCAT,
                               filtername="F212N", record_dir=str(tmp_path),
                               context="test")
    assert rec["passed"]
    assert rec["corrections"] == []   # not trustworthy -> not applied
    assert rec["unverified"]


def test_frozen_stage_subfloor_incoherent_tie_does_not_raise(tmp_path, monkeypatch):
    """Scope check: the new raise is conditioned on the tie having MOVED past
    REFERENCE_APPLY_MIN_MAS.  An incoherent tie whose offset is below the apply
    floor is the measurement floor, not a regression -- and a sparse field that
    can never measure a tie must not hard-stop every frozen stage."""
    monkeypatch.delenv("ALLOW_LATE_STAGE_ASTROM_SHIFT", raising=False)
    monkeypatch.delenv("ASTROM_ALLOW_FROZEN_INCOHERENT_TIE", raising=False)
    _patch_consensus_and_tie(monkeypatch,
                             dra_now=REFERENCE_APPLY_MIN_MAS / 2.0, ddec_now=0.0,
                             apply_ok=False)
    rec = run_visit_checkpoint([_tiny_visit_table()], "m5", refcat=_DUMMY_REFCAT,
                               filtername="F212N", record_dir=str(tmp_path),
                               context="test")
    assert rec["passed"]
    assert rec["failures"] == []
    assert rec["unverified"] == []


def test_frozen_stage_incoherent_tie_failure_is_in_the_record(tmp_path, monkeypatch):
    """The written record carries the blocking reason, so the ladder audit can
    see it without re-running anything."""
    monkeypatch.setenv("ALLOW_LATE_STAGE_ASTROM_SHIFT", "1")
    _write_m2_baseline(str(tmp_path), 0.5, 0.0)
    _patch_consensus_and_tie(monkeypatch, dra_now=_MOVED, ddec_now=0.0,
                             apply_ok=False)
    run_visit_checkpoint([_tiny_visit_table()], "m6", refcat=_DUMMY_REFCAT,
                         filtername="F212N", record_dir=str(tmp_path),
                         context="test")
    with open(tmp_path / "checkpoint_m6_F212N_latest.json") as fh:
        rec = json.load(fh)
    assert rec["failures"]
    assert "FROZEN stage AND the reference tie is incoherent" in rec["failures"][0]
    assert "MOVED" in rec["failures"][0]        # states how far, from what
    assert not rec["passed"]
