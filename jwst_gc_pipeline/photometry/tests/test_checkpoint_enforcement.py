"""Where a FROZEN-stage astrometry failure stops the pipeline.

m3 and later cannot change the astrometry -- the solution is frozen, which is
what makes a shift there a defect -- so the check is a MEASUREMENT wired up as
a CONTROL.  Raising inside the stage bought one thing, not spending compute on
a run that would be refused, and cost three:

* the chain is ``afterok``, so one filter's raise discarded every other
  filter's finished stages (cloudef 002 spent ten re-tie iterations reaching m2
  and lost all of it at m3, 2026-08-18);
* the products that would let someone diagnose the shift were never made;
* every frozen-stage failure diagnosed so far turned out to be a comparison
  artefact rather than movement.

So the default moves the stop to the release gate.  These tests pin that the
move is a MOVE and not a removal: the failure is still measured, still recorded
with ``passed: false``, still printed, and ``check_astrometry_checkpoints.py``
refuses the field.  m2 -- the one stage that can still correct -- still raises
in place.
"""
import json
import os

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from jwst_gc_pipeline.photometry import astrometry_checkpoint as _ac
from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    AstrometryCorrectionRequiredError, AstrometryRegressionError,
    CHECKPOINT_ENFORCE_ENV, ENFORCE_AT_RELEASE, ENFORCE_AT_STAGE,
    checkpoint_enforcement, run_visit_checkpoint)
from .test_visit_consensus import RA0, DEC0, _exposure_table, _field

_DUMMY_REFCAT = dict(all=None, sparse=None, mag=None, dense=True)


def _tiny_visit_table():
    ra, dec = _field(n=5)
    return _exposure_table(ra, dec, exposure=1)


def _patch(monkeypatch, dra_now, ddec_now):
    coords = SkyCoord(ra=[RA0, RA0] * u.deg, dec=[DEC0, DEC0] * u.deg,
                      frame="icrs")

    def _fake_consensus(tables, context="", **kw):
        return dict(coords=coords, mag=None, exposures=[],
                    anchor_key=("001", 1, "nrcb1", "F212N"),
                    scatter_mas=np.array([1.0]), consensus_ok=True, skipped=[])

    def _fake_tie(cons_coords, ref_all, ref_sparse, **kw):
        return dict(off_mas=float(np.hypot(dra_now, ddec_now)), apply_ok=True,
                    dra_mas=float(dra_now), ddec_mas=float(ddec_now),
                    cross_reference={"agree": True, "sep_mas": 0.0},
                    cross_reference_gross_ok=True, per_tile={"clean": True},
                    swept=False,
                    vs_full={"dra": float(dra_now), "ddec": float(ddec_now)})

    monkeypatch.setattr(_ac, "build_visit_consensus", _fake_consensus)
    monkeypatch.setattr(_ac, "measure_reference_tie", _fake_tie)


def _m2_baseline(record_dir, dra, ddec, filt="F212N"):
    rec = dict(visits=[dict(visit="001", reference_tie=dict(
        apply_ok=True, dra_mas=dra, ddec_mas=ddec,
        vs_full=dict(dra=dra, ddec=ddec)))])
    with open(os.path.join(record_dir, f"checkpoint_m2_{filt}_latest.json"),
              "w") as fh:
        json.dump(rec, fh)


def _moved_run(tmp_path, monkeypatch, stage="m3"):
    """m2 froze the tie at (10, 0); this stage reads (40, 0) -- a real 30 mas
    shift, not an artefact, so it fails under either policy."""
    _m2_baseline(str(tmp_path), 10.0, 0.0)
    _patch(monkeypatch, dra_now=40.0, ddec_now=0.0)
    return lambda: run_visit_checkpoint(
        [_tiny_visit_table()], stage, refcat=_DUMMY_REFCAT,
        filtername="F212N", record_dir=str(tmp_path), context="test")


# ---------------------------------------------------------------------------
# the switch itself
# ---------------------------------------------------------------------------

def test_the_default_is_to_defer_to_the_release_gate(monkeypatch):
    monkeypatch.delenv(CHECKPOINT_ENFORCE_ENV, raising=False)
    assert checkpoint_enforcement() == ENFORCE_AT_RELEASE


@pytest.mark.parametrize("value", ["stage", "STAGE", " stage "])
def test_stage_enforcement_is_selectable(monkeypatch, value):
    monkeypatch.setenv(CHECKPOINT_ENFORCE_ENV, value)
    assert checkpoint_enforcement() == ENFORCE_AT_STAGE


@pytest.mark.parametrize("typo", ["relase", "1", "yes", "release-gate", "off"])
def test_a_TYPO_enforces_at_the_stage_rather_than_choosing_the_lenient_branch(
        monkeypatch, typo):
    """A misspelled value must not silently pick the permissive policy.  Anything
    that is not exactly "release" is read as "stage", which is the stricter of
    the two -- a typo then costs a stopped chain, not a shipped misalignment."""
    monkeypatch.setenv(CHECKPOINT_ENFORCE_ENV, typo)
    assert checkpoint_enforcement() == ENFORCE_AT_STAGE


# ---------------------------------------------------------------------------
# what each policy does with a real frozen-stage failure
# ---------------------------------------------------------------------------

def test_at_the_default_a_frozen_failure_does_not_raise(tmp_path, monkeypatch):
    monkeypatch.delenv(CHECKPOINT_ENFORCE_ENV, raising=False)
    rec = _moved_run(tmp_path, monkeypatch)()
    assert rec["passed"] is False
    assert rec["failures"], 'the failure must still be MEASURED and recorded'


def test_at_the_default_the_failure_is_still_written_to_the_record(
        tmp_path, monkeypatch):
    """The release gate reads the record, so a deferral that did not persist
    would be a deletion."""
    monkeypatch.delenv(CHECKPOINT_ENFORCE_ENV, raising=False)
    _moved_run(tmp_path, monkeypatch)()
    with open(os.path.join(str(tmp_path), 'checkpoint_m3_F212N_latest.json')) as fh:
        on_disk = json.load(fh)
    assert on_disk["passed"] is False
    assert any("MOVED" in f for f in on_disk["failures"]), on_disk["failures"]


def test_at_the_default_it_says_the_release_gate_will_refuse(tmp_path,
                                                             monkeypatch,
                                                             capsys):
    """A silent continue is the failure mode to avoid: the operator has to be
    able to tell a deferred failure from a pass, in the log, at the time."""
    monkeypatch.delenv(CHECKPOINT_ENFORCE_ENV, raising=False)
    _moved_run(tmp_path, monkeypatch)()
    out = capsys.readouterr().out
    assert "MOVED" in out
    assert "check_astrometry_checkpoints.py" in out, out
    assert "release gate refuses" in out, out
    assert f"{CHECKPOINT_ENFORCE_ENV}={ENFORCE_AT_STAGE}" in out, (
        'the log must say how to get the old behaviour back')


def test_stage_enforcement_still_raises_where_it_measured(tmp_path, monkeypatch):
    monkeypatch.setenv(CHECKPOINT_ENFORCE_ENV, ENFORCE_AT_STAGE)
    with pytest.raises(AstrometryRegressionError):
        _moved_run(tmp_path, monkeypatch)()


def test_deferral_applies_at_every_frozen_stage(tmp_path, monkeypatch):
    monkeypatch.delenv(CHECKPOINT_ENFORCE_ENV, raising=False)
    for stage in ("m3", "m4", "m5", "m6"):
        d = tmp_path / stage
        d.mkdir()
        rec = _moved_run(d, monkeypatch, stage=stage)()
        assert rec["passed"] is False, stage


# ---------------------------------------------------------------------------
# m2 is NOT deferred
# ---------------------------------------------------------------------------

def test_the_CORRECTING_checkpoint_emits_a_CORRECTION_not_a_deferral(
        tmp_path, monkeypatch, capsys):
    """m2 is the one stage where the astrometry can still change, and its
    response to a measured offset is to CORRECT it -- write the correction to
    the offsets table, stale-tag the mosaics, and have its caller stop the run
    for regeneration.  That is a control action, and no later gate can perform
    it, so the deferral must not touch this path: an m2 that merely recorded a
    4" offset and continued would run the whole chain on frames it had already
    declared wrong."""
    monkeypatch.delenv(CHECKPOINT_ENFORCE_ENV, raising=False)
    _patch(monkeypatch, dra_now=4000.0, ddec_now=0.0)
    rec = run_visit_checkpoint([_tiny_visit_table()], "m2", refcat=_DUMMY_REFCAT,
                               filtername="F212N", basepath=str(tmp_path),
                               record_dir=str(tmp_path), context="test")
    assert rec["correcting"] is True
    assert rec["corrections"], 'm2 must emit the correction, not record a note'
    assert rec["failures"] == [], 'a correctable offset is not a frozen failure'
    out = capsys.readouterr().out
    assert "release gate refuses" not in out, (
        'the deferral message must never appear at a correcting stage')


def test_the_old_per_stage_override_still_works(tmp_path, monkeypatch):
    """`ALLOW_LATE_STAGE_ASTROM_SHIFT=1` predates this and is still the way to
    proceed past a frozen-stage failure without recording it as blocking."""
    monkeypatch.setenv(CHECKPOINT_ENFORCE_ENV, ENFORCE_AT_STAGE)
    monkeypatch.setenv("ALLOW_LATE_STAGE_ASTROM_SHIFT", "1")
    rec = _moved_run(tmp_path, monkeypatch)()
    assert rec["passed"] is False


def test_a_CORRECTING_stage_with_real_failures_neither_raises_nor_defers(
        tmp_path, monkeypatch, capsys):
    """m2 can record `failures` -- a duplicated exposure identity is one -- and
    the frozen-stage dispatch must not touch them.

    Without the `not correcting` guard, m2 would either raise here (stopping a
    run whose caller already handles this) or, worse, print the deferral and
    hand the release gate a correcting-stage record to refuse on.  Both are
    wrong for a stage whose job is to correct and re-run."""
    from jwst_gc_pipeline.photometry.visit_consensus import DuplicateExposureError
    monkeypatch.delenv(CHECKPOINT_ENFORCE_ENV, raising=False)

    def _dupe(tables, context="", **kw):
        raise DuplicateExposureError("('1', 1, 'nrca1') appears twice")

    monkeypatch.setattr(_ac, "build_visit_consensus", _dupe)
    rec = run_visit_checkpoint([_tiny_visit_table()], "m2", refcat=None,
                               filtername="F212N", record_dir=str(tmp_path),
                               context="test")
    assert rec["correcting"] is True
    assert rec["failures"], 'the duplicate must still be recorded'
    out = capsys.readouterr().out
    assert "release gate refuses" not in out, out
    assert "ASTROMETRY REGRESSION" not in out, out
