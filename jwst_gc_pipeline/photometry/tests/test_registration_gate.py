"""Tests for the tailing registration failsafe wrapper."""
from pathlib import Path

from jwst_gc_pipeline.photometry import registration_gate as rg


def test_failsafe_script_resolves():
    # The wrapper must point at the real per-tile failsafe it shells out to.
    assert rg._FAILSAFE.name == "registration_failsafes.py"
    assert rg._FAILSAFE.is_file(), f"missing {rg._FAILSAFE}"


def test_missing_field_does_not_crash_non_strict(monkeypatch):
    # A bogus field: the failsafe reports could-not-verify / no products. Non-strict
    # must NOT raise (never wedge a run); strict may raise but must not hang.
    res = rg.run_registration_gate("definitely-not-a-real-field", strict=False, verbose=False)
    assert res["field"] == "definitely-not-a-real-field"
    assert "returncode" in res
    # rc 0 or 1 both allowed here; the contract is only "no exception when non-strict".


def test_strict_raises_on_fail(monkeypatch):
    # Force the underlying scan to look like a FAIL (rc=1) and confirm strict raises.
    class _R:
        returncode = 1

    monkeypatch.setattr(rg.subprocess, "run", lambda *a, **k: _R())
    monkeypatch.setattr(rg._FAILSAFE.__class__, "is_file", lambda self: True)
    import pytest
    with pytest.raises(rg.RegistrationGateError):
        rg.assert_field_registered("brick", verbose=False)


def _fake_run_writing_json(payload):
    """Return a subprocess.run stand-in that writes ``payload`` to the --json
    path in the command (mirroring registration_failsafes --scan --json)."""
    import json as _json

    class _R:
        returncode = 0

    def _run(cmd, *a, **k):
        json_path = cmd[cmd.index("--json") + 1]
        with open(json_path, "w") as fh:
            _json.dump(payload, fh)
        return _R()

    return _run


def test_rc0_pass_reports_passed_true(monkeypatch):
    monkeypatch.setattr(rg.subprocess, "run",
                        _fake_run_writing_json({"field": "brick", "PASS": True}))
    monkeypatch.setattr(rg._FAILSAFE.__class__, "is_file", lambda self: True)
    res = rg.run_registration_gate("brick", verbose=False)
    assert res["passed"] is True
    assert res["could_not_verify"] is False
    assert res["returncode"] == 0


def test_rc0_could_not_verify_is_truthful(monkeypatch):
    # The failsafe exits 0 for could-not-verify (e.g. <2 bands); the gate used
    # to report passed=True/could_not_verify=False for that case.
    monkeypatch.setattr(rg.subprocess, "run", _fake_run_writing_json(
        {"field": "brick", "bands": ["F200W"],
         "error": "need >=2 bands for cross-band"}))
    monkeypatch.setattr(rg._FAILSAFE.__class__, "is_file", lambda self: True)
    res = rg.run_registration_gate("brick", verbose=False)
    assert res["passed"] is False
    assert res["could_not_verify"] is True
    assert res["returncode"] == 0


def test_rc0_without_readable_json_is_could_not_verify(monkeypatch):
    # A failsafe that produced no readable --json result cannot be called
    # verified (passed must NOT default to True on rc==0).
    class _R:
        returncode = 0

    monkeypatch.setattr(rg.subprocess, "run", lambda *a, **k: _R())
    monkeypatch.setattr(rg._FAILSAFE.__class__, "is_file", lambda self: True)
    res = rg.run_registration_gate("brick", verbose=False)
    assert res["passed"] is False
    assert res["could_not_verify"] is True
