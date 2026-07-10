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
