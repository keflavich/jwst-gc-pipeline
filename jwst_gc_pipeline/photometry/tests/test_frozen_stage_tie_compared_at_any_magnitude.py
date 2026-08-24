"""A frozen stage must compare its reference tie against the m2 baseline
whatever the CURRENT value reads.

Issue #398.  The whole frozen-stage block sat inside

    if np.isfinite(off) and off > REFERENCE_APPLY_MIN_MAS:

whose constant is documented as an APPLY threshold -- "a reference correction
is only APPLIED when it exceeds this".  Nesting the STABILITY comparison inside
it conflates two questions.  A tie frozen at 50 mas that a later stage
re-measures at 1 mas has moved 49 mas, and 1 is below 2, so nothing was
compared and the stage passed.  `ASTROMETRY_CHECKPOINTS.md` promises "any
measured shift raises" at m3-m6.

Measured on the checkpoint records under
`/orange/adamginsburg/jwst/*/astrometry_checkpoints/` (2026-08-23): of 204
m3-m6 visit entries carrying a `reference_tie`, 121 read <= 2 mas and were
therefore never compared; three of those sit on an m2 baseline above 2 mas, and
one -- sickle F335M m5, m2 3.08 mas, m5 1.75 mas -- would read a 2.17 mas
delta.

The fix splits the two questions rather than deleting the threshold, because
the same `if` also guards the `apply_ok is False` arm, which appends to
`unverified_blocking`.  That arm is about MAGNITUDE: a 0.22 mas tie that could
not be certified is a sub-floor number, not a defect to stop a run on.  One
such entry exists on disk today (sickle F480M o007 m4), and opening that arm
along with the comparison would have turned it into a blocking item.

The fakes come from `test_frozen_stage_symmetric_baseline`, whose tie depends
both on WHICH stars it is handed and on WHERE they are -- a constant-returning
fake cannot express "the stage moved onto the reference".
"""
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from jwst_gc_pipeline.photometry import astrometry_checkpoint as _ac
from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    AstrometryRegressionError)

from .test_frozen_stage_symmetric_baseline import (
    _install_fakes, _m2_star_grid, _run, _write_m2_consensus_catalog,
    _write_m2_record, _zero_mean_jitter)

#: Where m2 froze the tie, in mas of Dec.  Well above REFERENCE_APPLY_MIN_MAS,
#: so m2 itself measured and applied it.
M2_FROZEN_DDEC = 50.0

#: What the later stage reads.  BELOW REFERENCE_APPLY_MIN_MAS -- which is the
#: whole point: the stage has moved 49 mas and its current value is small.
STAGE_DDEC = 1.0


@pytest.fixture(autouse=True)
def _enforce_at_the_stage(monkeypatch):
    """#442 defers the stop to the release gate by default; these tests assert
    on the exception, so they run under `stage` enforcement, as
    `test_frozen_stage_symmetric_baseline` does for the same reason."""
    monkeypatch.setenv('ASTROM_CHECKPOINT_ENFORCE', 'stage')


def _moved_onto_the_reference(tmp_path, monkeypatch, n_stars=400,
                              stage_ddec=STAGE_DDEC, write_m2_record=True,
                              **fake_kw):
    """m2 froze the tie at +50 mas; the stage's stars have moved 49 mas so its
    own tie reads +1 mas.  No population change at all -- every m2 star is
    re-detected -- so the survivor re-measure sees the same 49 mas and cannot
    explain it away as a baseline artefact.
    """
    coords = _m2_star_grid(n_stars)
    per_star = np.full(n_stars, M2_FROZEN_DDEC)
    moved_deg = (stage_ddec - M2_FROZEN_DDEC) / 3.6e6
    stage = SkyCoord(
        ra=coords.ra.deg * u.deg,
        dec=(coords.dec.deg + _zero_mean_jitter(n_stars) + moved_deg) * u.deg,
        frame="icrs")
    basepath = str(tmp_path)
    cat = _write_m2_consensus_catalog(basepath, coords)
    if write_m2_record:
        _write_m2_record(basepath, 0.0, M2_FROZEN_DDEC, cat)
    _install_fakes(monkeypatch, coords, per_star, stage, **fake_kw)
    return basepath


# ---------------------------------------------------------------------------
# the comparison itself
# ---------------------------------------------------------------------------

def test_a_tie_that_moved_under_the_apply_floor_is_still_compared(
        tmp_path, monkeypatch):
    """The case in the issue: frozen at 50 mas, now reads 1 mas, moved 49.

    Without the fix this passes silently, because 1 < REFERENCE_APPLY_MIN_MAS.
    """
    assert STAGE_DDEC < _ac.REFERENCE_APPLY_MIN_MAS, (
        'the premise: the stage tie must be BELOW the apply threshold, or this '
        'test does not exercise the gate at all')
    assert (M2_FROZEN_DDEC - STAGE_DDEC) > _ac.STAGE_STABILITY_TOL_MAS

    basepath = _moved_onto_the_reference(tmp_path, monkeypatch)
    with pytest.raises(AstrometryRegressionError) as ex:
        _run(basepath, stage="m3")
    msg = str(ex.value)
    assert "consensus->reference MOVED" in msg, msg
    assert f"{M2_FROZEN_DDEC - STAGE_DDEC:.2f} mas" in msg, msg


def test_the_recorded_verdict_names_both_ties(tmp_path, monkeypatch):
    """A reader of the record must be able to see the two numbers that were
    differenced, not only that something failed."""
    basepath = _moved_onto_the_reference(tmp_path, monkeypatch)
    with pytest.raises(AstrometryRegressionError):
        _run(basepath, stage="m4")
    import json
    import os
    with open(os.path.join(basepath,
                           "checkpoint_m4_F212N_latest.json")) as fh:
        rec = json.load(fh)
    assert rec["passed"] is False
    assert rec["failures"], rec
    sym = rec["visits"][0]["symmetric_baseline"]
    assert sym["m2_ddec_mas"] == pytest.approx(M2_FROZEN_DDEC, abs=1e-3)
    assert sym["stage_ddec_mas"] == pytest.approx(STAGE_DDEC, abs=1e-3)


def test_a_frozen_tie_that_did_not_move_still_passes(tmp_path, monkeypatch):
    """The other side of the same change: the comparison now runs on every
    finite tie, so it must not start failing the stages that are fine.  Frozen
    at 50 mas, still reads 50 mas."""
    basepath = _moved_onto_the_reference(tmp_path, monkeypatch,
                                         stage_ddec=M2_FROZEN_DDEC)
    rec = _run(basepath, stage="m3")
    assert rec["passed"], rec["failures"]
    assert rec["failures"] == []


def test_a_correcting_stage_is_unaffected(tmp_path, monkeypatch):
    """m2 has nothing frozen to compare against, and a sub-floor tie is not
    worth applying, so the correcting side keeps the magnitude gate: no
    correction is emitted for a 1 mas tie."""
    basepath = _moved_onto_the_reference(tmp_path, monkeypatch)
    rec = _run(basepath, stage="m2")
    assert rec["passed"], rec["failures"]
    assert rec["visits"][0].get("symmetric_baseline") is None
    tie = rec["visits"][0]["reference_tie"]
    assert tie["off_mas"] < _ac.REFERENCE_APPLY_MIN_MAS


# ---------------------------------------------------------------------------
# the arms that STAY gated on magnitude
# ---------------------------------------------------------------------------

def test_a_sub_floor_uncertified_tie_does_not_block(tmp_path, monkeypatch):
    """`apply_ok: false` under the apply floor stays silent.

    This arm appends to `unverified_blocking`, which stops a release.  sickle
    F480M o007 m4 reads 0.22 mas with `apply_ok: false` on disk today; opening
    this arm along with the stability comparison would make that a blocking
    item, which is a fabricated failure -- the tie is smaller than the
    threshold at which anyone would have acted on it.
    """
    basepath = _moved_onto_the_reference(tmp_path, monkeypatch,
                                         stage_ddec=0.22, apply_ok=False)
    rec = _run(basepath, stage="m3")
    tie = rec["visits"][0]["reference_tie"]
    assert tie["off_mas"] < _ac.REFERENCE_APPLY_MIN_MAS, tie["off_mas"]
    assert tie["apply_ok"] is False
    blocking = [u for u in (rec.get("unverified") or [])
                if "not trustworthy" in u]
    assert blocking == [], blocking
    assert rec["passed"], rec["failures"]


def test_an_uncertified_tie_OVER_the_floor_still_blocks(tmp_path, monkeypatch):
    """The premise of the test above: the same arm with a large value is the
    cloudc F410M case (#312) and must keep blocking."""
    basepath = _moved_onto_the_reference(tmp_path, monkeypatch,
                                         stage_ddec=M2_FROZEN_DDEC,
                                         apply_ok=False)
    rec = _run(basepath, stage="m3")
    assert rec["visits"][0]["reference_tie"]["off_mas"] > \
        _ac.REFERENCE_APPLY_MIN_MAS
    assert any("not trustworthy" in u for u in (rec.get("unverified") or [])), \
        rec.get("unverified")
    assert rec["passed"] is False


def test_a_sub_floor_tie_with_no_m2_record_says_nothing(tmp_path, monkeypatch):
    """No baseline and no magnitude: there is nothing to report.

    With no m2 record the frozen branch's last arm calls the tie a late-stage
    offset and fails closed, which is right for a tie worth acting on and
    fabricated for a 1 mas one.
    """
    basepath = _moved_onto_the_reference(tmp_path, monkeypatch,
                                         write_m2_record=False)
    rec = _run(basepath, stage="m3")
    assert rec["passed"], rec["failures"]
    assert rec["failures"] == []


def test_no_m2_record_and_a_tie_OVER_the_floor_still_fails_closed(
        tmp_path, monkeypatch):
    """The premise of the test above."""
    basepath = _moved_onto_the_reference(tmp_path, monkeypatch,
                                         stage_ddec=M2_FROZEN_DDEC,
                                         write_m2_record=False)
    with pytest.raises(AstrometryRegressionError) as ex:
        _run(basepath, stage="m3")
    assert "no m2 baseline record found" in str(ex.value), str(ex.value)


# ---------------------------------------------------------------------------
# the constant says what it gates
# ---------------------------------------------------------------------------

def test_the_apply_threshold_documents_that_it_is_not_the_stability_gate():
    import inspect
    src = inspect.getsource(_ac)
    idx = src.index("REFERENCE_APPLY_MIN_MAS = 2.0")
    preamble = src[max(0, idx - 900):idx]
    assert "stability" in preamble.lower(), (
        "REFERENCE_APPLY_MIN_MAS is an APPLY threshold; its comment must say "
        "it does not gate the frozen-stage stability comparison (issue #398), "
        "or the next reader re-nests them")
