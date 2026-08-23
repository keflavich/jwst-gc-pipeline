"""A frozen-stage failure at a stage the release does not ship, which a LATER
stage of the same chain measured as passing, is answered rather than live.

The frozen ladder asks ONE question at every stage: is the solution still the
one m2 froze?  A later stage answering "no movement" measures that same
property, on the same exposures, against the same m2 baseline, after the
failing stage.  An excursion a later stage does not see did not reach the
products that later stage describes.

brick F200W is the case (issue #258).  One chain, 22-24 July:

    m2  2/7 = (+1.90, -0.31)   the freeze
    m5  2/7 = (+4.06, -1.10)   MOVED 2.30 mas   <- FAILED, tol 2.0
    m6  2/7 = (+2.56, -0.42)   moved 0.68 mas   -- passed
    m7  2/7 = (+2.41, -0.34)   moved 0.51 mas   -- passed

and brick ships the m7 products.  The release was refused over a transient in
an intermediate nobody downloads, while two later measurements of the same
property on the shipped stage both said the solution held.

`--shipped-stages` is what keeps this from becoming "a later pass forgives an
earlier failure": a failure at a stage the release SHIPS is never answered, and
declaring nothing supersedes nothing.
"""
import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "check_astrometry_checkpoints",
    Path(__file__).resolve().parents[3] / "scripts" / "release"
    / "check_astrometry_checkpoints.py")
cac = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cac)


def _rec(tmp_path, field, stage, filt, date, passed, obs=None, failures=()):
    d = tmp_path / field / "astrometry_checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    name = f"checkpoint_{stage}_{filt}" + (f"_o{obs}" if obs else "") + "_latest.json"
    (d / name).write_text(json.dumps({
        "stage": stage, "filtername": filt, "date": date, "passed": passed,
        "failures": list(failures), "visits": []}))


@pytest.fixture
def brick(tmp_path, monkeypatch):
    """brick's real shape: m2 freeze, m5 fails, m6 and m7 pass afterwards."""
    monkeypatch.setattr(cac, "BASE", str(tmp_path))
    _rec(tmp_path, "brick", "m2", "F200W", "2026-07-22T13:16:17Z", True)
    _rec(tmp_path, "brick", "m5", "F200W", "2026-07-23T20:14:42Z", False,
         failures=["exposure ('2', 7, 'nrca2', 'F200W') MOVED 2.30 mas"])
    _rec(tmp_path, "brick", "m6", "F200W", "2026-07-24T06:17:02Z", True)
    _rec(tmp_path, "brick", "m7", "F200W", "2026-07-24T15:46:44Z", True)
    return tmp_path


def test_a_later_passing_stage_answers_an_unshipped_failure(brick, capsys):
    rc = cac.main(["--field", "brick", "--shipped-stages", "m7,m8"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "0 FAILED" in out and "1 answered by a later stage" in out
    assert "ANSWERED BY A LATER STAGE m5/F200W" in out
    # the original failure text is still shown -- answered is not hidden
    assert "MOVED 2.30 mas" in out


def test_a_failure_at_a_SHIPPED_stage_is_never_answered(brick, capsys):
    """If the release ships the failing stage's products, the failure is about
    something a user downloads."""
    rc = cac.main(["--field", "brick", "--shipped-stages", "m5,m7"])
    out = capsys.readouterr().out
    assert rc == 1, out
    assert "1 FAILED" in out and "answered by a later stage" not in out
    assert "FAILED m5/F200W" in out


def test_declaring_no_shipped_stages_supersedes_nothing(brick, capsys):
    """Fail closed: a caller that does not say what it ships gets the old
    behaviour, not a free pass."""
    assert cac.main(["--field", "brick"]) == 1
    assert "FAILED m5/F200W" in capsys.readouterr().out


def test_an_EARLIER_passing_stage_does_not_answer_a_later_failure(tmp_path,
                                                                  monkeypatch,
                                                                  capsys):
    """Direction matters. m3 passing before m5 failed says nothing about m5 --
    the excursion happened after it."""
    monkeypatch.setattr(cac, "BASE", str(tmp_path))
    _rec(tmp_path, "f", "m2", "F200W", "2026-07-22T00:00:00Z", True)
    # m3 RE-RUN after m5, so its date is later while its stage is earlier: the
    # stage index is what decides, not which record was written last.  A stage
    # BEFORE the excursion cannot have measured whether the excursion persisted.
    _rec(tmp_path, "f", "m3", "F200W", "2026-07-25T00:00:00Z", True)
    _rec(tmp_path, "f", "m5", "F200W", "2026-07-24T00:00:00Z", False,
         failures=["MOVED 2.30 mas"])
    assert cac.main(["--field", "f", "--shipped-stages", "m7"]) == 1
    assert "answered by a later stage" not in capsys.readouterr().out


def test_a_later_stage_that_also_failed_answers_nothing(tmp_path, monkeypatch,
                                                        capsys):
    """Only a PASS answers. A later stage that also failed measured the same
    property and agrees something is wrong -- both failures stay live, which is
    what BOTH counts have to show."""
    monkeypatch.setattr(cac, "BASE", str(tmp_path))
    _rec(tmp_path, "f", "m2", "F200W", "2026-07-22T00:00:00Z", True)
    _rec(tmp_path, "f", "m5", "F200W", "2026-07-23T00:00:00Z", False,
         failures=["MOVED 2.30 mas"])
    _rec(tmp_path, "f", "m7", "F200W", "2026-07-24T00:00:00Z", False,
         failures=["MOVED 3.00 mas"])
    assert cac.main(["--field", "f", "--shipped-stages", "m8"]) == 1
    out = capsys.readouterr().out
    assert "2 FAILED" in out, out
    assert "answered by a later stage" not in out


def test_a_later_stage_of_a_DIFFERENT_filter_answers_nothing(tmp_path,
                                                             monkeypatch):
    """Same chain means same (filter, observation): F187N passing at m7 says
    nothing about F200W at m5."""
    monkeypatch.setattr(cac, "BASE", str(tmp_path))
    _rec(tmp_path, "f", "m2", "F200W", "2026-07-22T00:00:00Z", True)
    _rec(tmp_path, "f", "m5", "F200W", "2026-07-23T00:00:00Z", False,
         failures=["MOVED 2.30 mas"])
    _rec(tmp_path, "f", "m7", "F187N", "2026-07-24T00:00:00Z", True)
    assert cac.main(["--field", "f", "--shipped-stages", "m7"]) == 1


# ---- observation scoping: both spellings of an observation key ----

def test_scoping_accepts_proposal_obs_keys_as_well_as_bare_obsids(tmp_path,
                                                                  monkeypatch,
                                                                  capsys):
    """`stage_release` passes `_release_observations` -- PROPOSAL-obs keys like
    `02221-002` -- while a record name carries the bare obsid `002`.  Parsing
    the former as the latter matched nothing, so every TOKENISED record was
    dropped; with the newest m2 among them, a SUPERSEDED failure came back as
    live and refused the field.

    Measured on the archive 2026-08-20: cloudc read "1 FAILED" (m3/F182M,
    2026-08-06) under the proposal-obs spelling and "0 FAILED, superseded by the
    2026-08-12 m2" under the bare one.  arches the same, with m3/F212N.
    """
    monkeypatch.setattr(cac, "BASE", str(tmp_path))
    # the newest m2 is the TOKENISED one; the frozen failure predates it
    _rec(tmp_path, "cloudc", "m2", "F182M", "2026-08-06T04:02:54Z", True)
    _rec(tmp_path, "cloudc", "m2", "F182M", "2026-08-12T11:51:09Z", True, obs="002")
    _rec(tmp_path, "cloudc", "m3", "F182M", "2026-08-06T10:31:35Z", False,
         failures=["MOVED 3.07 mas"])

    assert cac.main(["--field", "cloudc", "--scan",
                     "--observations", "02221-002"]) != 1
    out = capsys.readouterr().out
    assert "0 FAILED" in out, out
    assert "SUPERSEDED m3/F182M" in out

    # ...and the bare spelling keeps working
    assert cac.main(["--field", "cloudc", "--scan",
                     "--observations", "002"]) != 1
    assert "0 FAILED" in capsys.readouterr().out
