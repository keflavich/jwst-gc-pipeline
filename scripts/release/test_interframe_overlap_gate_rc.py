"""Issue #393: the scan's two refusal arms are independent.

``check_filter`` sets ``could_not_verify`` and ``PASS`` separately, so one
filter can carry BOTH a measured misregistration and a separate pair nothing
could arbitrate.  ``main`` used ``elif``, so such a filter set only
``any_noverify`` and the run printed the rc=2 message -- which names "no
matchable crf frames / no detections", a cause that did not occur -- for a field
whose actual finding was exposures more than ``TOL_MAS`` apart.  Both are
refusals (``stage_release`` rejects any rc != 0), so what was lost is the
operator being told which one it was.

The mirror-image mistake is equally easy: ``PASS`` is False for EVERY
could-not-verify filter, so deriving ``any_fail`` from ``not PASS`` would report
a field that measured nothing as misregistered.  These tests pin both
directions, plus the cause-neutral rc=2 wording.

Hermetic: ``check_filter`` and ``field_filters`` are replaced with canned
verdict dicts, so nothing touches disk.
"""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "check_interframe_overlap",
    Path(__file__).with_name("check_interframe_overlap.py"))
ck = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ck)


def _run(monkeypatch, capsys, verdicts):
    """Run ``main --field f --scan`` over canned per-filter verdicts."""
    filts = [v["filt"] for v in verdicts]
    by_filt = {v["filt"]: v for v in verdicts}
    monkeypatch.setattr(ck, "field_filters", lambda field: filts)
    monkeypatch.setattr(
        ck, "check_filter",
        lambda field, filt, **kw: dict(by_filt[filt], field=field))
    rc = ck.main(["--field", "testfield", "--scan"])
    return rc, capsys.readouterr().out


def _both():
    """One filter with a real FAIL pair AND an unarbitrated deferred pair."""
    return dict(filt="F200W", PASS=False, could_not_verify=True,
                ext_fail=False, n_fail=1)


def _noverify_only():
    """The no-crf-frames early return: PASS=False, nothing measured."""
    return dict(filt="F410M", PASS=False, could_not_verify=True,
                note="no crf frames matched")


def _fail_only():
    return dict(filt="F187N", PASS=False, could_not_verify=False,
                ext_fail=False, n_fail=2)


def _pass():
    return dict(filt="F444W", PASS=True, could_not_verify=False,
                ext_fail=False, n_fail=0)


def test_filter_that_is_both_fail_and_unverified_exits_1_and_says_fail(
        monkeypatch, capsys):
    """The #393 case: both flags on ONE filter -> rc 1 and the FAIL line."""
    rc, out = _run(monkeypatch, capsys, [_both()])
    assert rc == 1, out
    assert "OVERLAP GATE: FAIL for testfield" in out
    assert "inter-frame misregistration" in out


def test_filter_that_is_both_still_reports_the_could_not_verify_arm(
        monkeypatch, capsys):
    """A field carrying both refusals is described by BOTH lines, so the
    operator learns that a pair was also left unarbitrated."""
    _, out = _run(monkeypatch, capsys, [_both()])
    assert "OVERLAP GATE: COULD NOT VERIFY testfield" in out


def test_fail_in_one_filter_and_noverify_in_another_exits_1(monkeypatch, capsys):
    """Across filters, a measured FAIL owns the exit code."""
    rc, out = _run(monkeypatch, capsys, [_noverify_only(), _fail_only()])
    assert rc == 1, out
    assert "OVERLAP GATE: FAIL for testfield" in out
    assert "OVERLAP GATE: COULD NOT VERIFY testfield" in out


def test_pure_could_not_verify_stays_rc_2_and_prints_no_fail_line(
        monkeypatch, capsys):
    """The mirror-image regression: `PASS` is False here too, so an `any_fail`
    derived from `not PASS` would call an unmeasured filter misregistered."""
    rc, out = _run(monkeypatch, capsys, [_noverify_only()])
    assert rc == 2, out
    assert "OVERLAP GATE: FAIL for testfield" not in out
    assert "OVERLAP GATE: COULD NOT VERIFY testfield" in out


def test_could_not_verify_message_does_not_name_one_cause(monkeypatch, capsys):
    """Several arms reach rc=2; the message points at the per-filter lines."""
    _, out = _run(monkeypatch, capsys, [_noverify_only()])
    assert "no matchable crf frames / no detections" not in out


def test_all_clean_exits_0(monkeypatch, capsys):
    rc, out = _run(monkeypatch, capsys, [_pass(), _pass() | {"filt": "F212N"}])
    assert rc == 0, out
    assert "OVERLAP GATE" not in out


def test_ext_fail_alone_is_a_measured_failure(monkeypatch, capsys):
    """`ext_fail` (the whole filter tied wrong against the reference) is the
    other measured-failure arm and must reach rc 1."""
    rc, out = _run(monkeypatch, capsys, [
        dict(filt="F335M", PASS=False, could_not_verify=False,
             ext_fail=True, n_fail=0)])
    assert rc == 1, out
    assert "OVERLAP GATE: FAIL for testfield" in out


def test_not_in_release_band_is_neither(monkeypatch, capsys):
    """A skipped band still counts as neither passed nor failed, and a scan
    whose every band is skipped is rc 2 (#452), unchanged by this fix."""
    rc, out = ck, None
    monkeypatch.setattr(ck, "field_filters", lambda field: ["F090W"])
    monkeypatch.setattr(
        ck, "check_filter",
        lambda field, filt, **kw: dict(field=field, filt=filt, PASS=None,
                                       not_in_release=True))
    rc = ck.main(["--field", "testfield", "--scan"])
    out = capsys.readouterr().out
    assert rc == 2, out
    assert "every band was skipped" in out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
