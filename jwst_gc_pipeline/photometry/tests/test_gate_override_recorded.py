"""An override that let a run past a red gate must be visible on disk.

`_defer_to_release` printed one WARNING line to stdout and returned True.  The
checkpoint record it walked past was byte-identical whether the run had stopped
at the gate or continued through it, so `passed: false` alone cannot say which
happened -- and the SLURM log is the only other copy.

Issue #258 is what that costs.  `brick/checkpoint_m5_F200W_latest.json` has been
red since 2026-07-23 with four nrca2 exposures 2.3 mas past
`STAGE_STABILITY_TOL_MAS`, and the run continued, so
`ALLOW_LATE_STAGE_ASTROM_SHIFT=1` was set.  Two weeks later nothing on disk says
by whom, why, or against which of the four failures -- and CLAUDE.md requires
written justification for exactly this override.

The record now carries `gate_override`, on the same reasoning that already puts
`correction_floor_mas` in `tolerances`: whatever changed the outcome belongs
beside the outcome.
"""
import json
import os

import pytest

from jwst_gc_pipeline.photometry import astrometry_checkpoint as AC


ENV = "ALLOW_LATE_STAGE_ASTROM_SHIFT"
REASON_ENV = "ALLOW_LATE_STAGE_ASTROM_SHIFT_REASON"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (ENV, REASON_ENV, "ASTROM_CHECKPOINT_ENFORCE"):
        monkeypatch.delenv(name, raising=False)


def test_the_reason_variable_is_named_after_the_override():
    """One rule, so an operator who knows the override knows where the
    justification goes without consulting a table."""
    assert AC.override_reason_env(ENV) == REASON_ENV
    assert (AC.override_reason_env("ALLOW_CROSSFILTER_ASTROM_FAIL")
            == "ALLOW_CROSSFILTER_ASTROM_FAIL_REASON")


def test_an_untouched_gate_records_that_it_was_not_overridden():
    state = AC.gate_override_state(ENV)
    assert state["used"] is False
    assert state["reason"] == ""
    assert state["env"] == ENV


def test_an_override_with_a_reason_records_both(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setenv(REASON_ENV,
                       "brick m5 F200W nrca2: transient, m6/m7 green, #258")
    state = AC.gate_override_state(ENV)
    assert state["used"] is True
    assert "#258" in state["reason"]


def test_an_override_with_no_reason_is_recorded_as_having_none(monkeypatch):
    """The state CLAUDE.md forbids has to be distinguishable from a clean run
    AND from a justified override -- an empty reason on an unused override
    would collapse the first two together."""
    monkeypatch.setenv(ENV, "1")
    state = AC.gate_override_state(ENV)
    assert state["used"] is True
    assert state["reason"] == ""
    assert state["reason_env"] == REASON_ENV


def test_a_reason_set_without_the_override_is_not_recorded_as_a_waiver(
        monkeypatch):
    """A leftover reason in the environment must not make an un-overridden run
    look like a deliberate waiver."""
    monkeypatch.setenv(REASON_ENV, "left over from last week")
    state = AC.gate_override_state(ENV)
    assert state["used"] is False
    assert state["reason"] == ""


def test_the_override_justification_is_printed_at_the_gate(monkeypatch, capsys):
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setenv(REASON_ENV, "because the m6 record is green")
    assert AC._defer_to_release("MOVED 2.30 mas", "the frozen-stage check", ENV)
    out = capsys.readouterr().out
    assert "because the m6 record is green" in out


def test_an_unjustified_override_says_so_at_the_gate(monkeypatch, capsys):
    monkeypatch.setenv(ENV, "1")
    assert AC._defer_to_release("MOVED 2.30 mas", "the frozen-stage check", ENV)
    out = capsys.readouterr().out
    assert "NO JUSTIFICATION RECORDED" in out
    assert REASON_ENV in out, 'the message must name the variable to set'


# ---------------------------------------------------------------------------
# the record itself -- written BEFORE the override was consulted, which is why
# the state has to be resolved up front rather than stamped on afterwards
# ---------------------------------------------------------------------------

def _on_disk(tmp_path):
    """The record `run_visit_checkpoint` actually wrote.

    Read from the FILE, not from the returned dict.  The record is serialised
    several statements before `_defer_to_release` is reached, so a field
    stamped on the returned object after the override was consulted would
    satisfy a test that reads the return value and still be missing from every
    record on disk -- which is the defect.
    """
    names = [f for f in os.listdir(tmp_path) if f.startswith('checkpoint_')
             and f.endswith('.json') and 'latest' not in f]
    assert names, f'no record written into {tmp_path}'
    with open(os.path.join(tmp_path, sorted(names)[-1])) as fh:
        return json.load(fh)


def test_a_failing_frozen_stage_record_carries_the_override_state(
        tmp_path, monkeypatch):
    """The #258 shape end to end: a frozen stage moves, the override lets the
    run continue, and the file on disk says so."""
    from .test_astrometry_checkpoint import _visit_tables

    monkeypatch.setenv(ENV, "1")
    monkeypatch.setenv(REASON_ENV, "#258, transient at m5 only")
    rec = AC.run_visit_checkpoint(_visit_tables(misaligned={2: (8.0, 0.0)}),
                                  "m5", filtername="F212N",
                                  record_dir=str(tmp_path), context="test")
    assert not rec["passed"] and rec["failures"]

    disk = _on_disk(tmp_path)
    assert disk["gate_override"]["used"] is True
    assert disk["gate_override"]["reason"] == "#258, transient at m5 only"
    assert disk["gate_override"]["env"] == ENV


def test_an_unjustified_override_is_distinguishable_on_disk(tmp_path,
                                                            monkeypatch):
    """The state CLAUDE.md forbids, and the one #258 is in: overridden, with
    nothing said about why."""
    from .test_astrometry_checkpoint import _visit_tables

    monkeypatch.setenv(ENV, "1")
    rec = AC.run_visit_checkpoint(_visit_tables(misaligned={2: (8.0, 0.0)}),
                                  "m5", filtername="F212N",
                                  record_dir=str(tmp_path), context="test")
    assert not rec["passed"]
    disk = _on_disk(tmp_path)
    assert disk["gate_override"]["used"] is True
    assert disk["gate_override"]["reason"] == ""


def test_a_failure_that_STOPPED_the_run_is_distinguishable_from_one_that_did_not(
        tmp_path, monkeypatch):
    """Both leave `passed: false`.  Without this field they are the same
    record, which is why #258 cannot be answered from disk."""
    from .test_astrometry_checkpoint import _visit_tables

    monkeypatch.setenv("ASTROM_CHECKPOINT_ENFORCE", "stage")
    with pytest.raises(AC.AstrometryRegressionError):
        AC.run_visit_checkpoint(_visit_tables(misaligned={2: (8.0, 0.0)}),
                                "m5", filtername="F212N",
                                record_dir=str(tmp_path), context="test")
    disk = _on_disk(tmp_path)
    assert disk["passed"] is False
    assert disk["gate_override"]["used"] is False, (
        'the run stopped at the gate; nothing was waived')


def test_a_correcting_stage_records_no_override_state(tmp_path, monkeypatch):
    """m2 never defers -- its stop is what sends the field back for
    regeneration -- so `used: false` there would describe a gate that was never
    consulted."""
    from .test_astrometry_checkpoint import _visit_tables

    monkeypatch.setenv(ENV, "1")
    AC.run_visit_checkpoint(_visit_tables(misaligned={2: (8.0, 0.0)}),
                            "m2", filtername="F212N",
                            record_dir=str(tmp_path), context="test")
    assert _on_disk(tmp_path)["gate_override"] is None


def test_a_clean_frozen_stage_records_no_override_state(tmp_path, monkeypatch):
    from .test_astrometry_checkpoint import _visit_tables

    monkeypatch.setenv(ENV, "1")
    rec = AC.run_visit_checkpoint(_visit_tables(), "m5", filtername="F212N",
                                  record_dir=str(tmp_path), context="test")
    assert rec["passed"]
    assert _on_disk(tmp_path)["gate_override"] is None


def test_the_record_builder_resolves_the_override_before_writing():
    """`_write_record` runs several statements before `_defer_to_release`, so a
    later stamp would never reach the file on disk.  Pinned on the source order
    because the failure mode is an ordering one: a test that reads the returned
    dict passes while the file is missing the field."""
    import inspect
    src = inspect.getsource(AC.run_visit_checkpoint)
    assert 'gate_override' in src, 'the record no longer carries the override'
    build = src.index('gate_override = ')
    write = src.index('_write_record(record_dir')
    defer = src.index('_defer_to_release(')
    assert build < write < defer, (
        'the override state must be resolved before the record is written; '
        'stamping it after `_defer_to_release` leaves the file on disk '
        'without it, which is the whole defect')


# ---------------------------------------------------------------------------
# the BROAD override (issue #581): ASTROM_CHECKPOINT_WARN_ONLY
#
# It demotes every blocking check at every stage, a CORRECTING stage's stop
# included, and it was the one override the record never mentioned.  arches ran
# its m2 repair pass under it on 2026-08-28 with a written justification: the
# record reads `passed=True correcting=True gate_override=None`, the release
# gate reports `0 FAILED` and exit 0, and the reason lives only in a SLURM log.
# ---------------------------------------------------------------------------

WARN_ONLY = "ASTROM_CHECKPOINT_WARN_ONLY"
WARN_ONLY_REASON = "ASTROM_CHECKPOINT_WARN_ONLY_REASON"


@pytest.fixture(autouse=True)
def _clean_warn_only(monkeypatch):
    for name in (WARN_ONLY, WARN_ONLY_REASON):
        monkeypatch.delenv(name, raising=False)


def test_the_broad_override_is_the_one_named_in_the_module():
    """Spelled once, and identically to the demotion sites in cataloging.py --
    a second spelling records an override that was never consulted, or misses
    one that was."""
    assert AC.WARN_ONLY_ENV == WARN_ONLY
    assert AC.override_reason_env(AC.WARN_ONLY_ENV) == WARN_ONLY_REASON


def test_a_PASSING_correcting_stage_record_carries_the_broad_override(
        tmp_path, monkeypatch):
    """The arches shape end to end.  m2 measures a correction, WARN_ONLY
    demotes the stop, and the record on disk must say a gate was demoted --
    `passed` is true and the stage is correcting, so both conditions that guard
    the narrow overrides exclude exactly this record."""
    from .test_astrometry_checkpoint import _visit_tables

    monkeypatch.setenv(WARN_ONLY, "1")
    monkeypatch.setenv(WARN_ONLY_REASON,
                       "arches F323N: 4.02/4.00 mas against a 4.0 floor")
    rec = AC.run_visit_checkpoint(_visit_tables(misaligned={2: (8.0, 0.0)}),
                                  "m2", filtername="F323N",
                                  record_dir=str(tmp_path), context="test")
    assert rec["correcting"] and rec["passed"], (
        'the arches record is a PASSING correcting-stage one')

    disk = _on_disk(tmp_path)
    assert disk["gate_override"] is not None, (
        'a run that demoted every blocking check must not be indistinguishable '
        'from one that met them')
    assert disk["gate_override"]["env"] == WARN_ONLY
    assert disk["gate_override"]["used"] is True
    assert "4.02/4.00 mas" in disk["gate_override"]["reason"]
    assert disk["gate_override"]["reason_env"] == WARN_ONLY_REASON


def test_a_CLEAN_frozen_stage_run_under_the_broad_override_records_it(
        tmp_path, monkeypatch):
    """No failures anywhere, so the narrow override is never reached -- but the
    run still had every blocking check demoted, and the pass is not an
    unassisted one."""
    from .test_astrometry_checkpoint import _visit_tables

    monkeypatch.setenv(WARN_ONLY, "1")
    rec = AC.run_visit_checkpoint(_visit_tables(), "m5", filtername="F212N",
                                  record_dir=str(tmp_path), context="test")
    assert rec["passed"] and not rec["failures"]
    disk = _on_disk(tmp_path)
    assert disk["gate_override"]["env"] == WARN_ONLY
    assert disk["gate_override"]["used"] is True


def test_an_unjustified_broad_override_is_recorded_as_having_no_reason(
        tmp_path, monkeypatch):
    """The state CLAUDE.md forbids, for the broadest of the three."""
    from .test_astrometry_checkpoint import _visit_tables

    monkeypatch.setenv(WARN_ONLY, "1")
    AC.run_visit_checkpoint(_visit_tables(), "m5", filtername="F212N",
                            record_dir=str(tmp_path), context="test")
    disk = _on_disk(tmp_path)
    assert disk["gate_override"]["used"] is True
    assert disk["gate_override"]["reason"] == ""


def test_the_broad_override_is_what_a_both_set_run_records(tmp_path,
                                                           monkeypatch):
    """WARN_ONLY demotes this stage's raise whether or not the narrow one is
    set, so it is the one that describes what let the run past."""
    from .test_astrometry_checkpoint import _visit_tables

    monkeypatch.setenv(ENV, "1")
    monkeypatch.setenv(WARN_ONLY, "1")
    AC.run_visit_checkpoint(_visit_tables(misaligned={2: (8.0, 0.0)}),
                            "m5", filtername="F212N",
                            record_dir=str(tmp_path), context="test")
    assert _on_disk(tmp_path)["gate_override"]["env"] == WARN_ONLY


def test_a_run_without_the_broad_override_is_unchanged(tmp_path):
    """The variable unset records nothing new: a clean frozen pass still
    carries `None`, so `gate_override` keeps meaning "a gate was touched"."""
    from .test_astrometry_checkpoint import _visit_tables

    AC.run_visit_checkpoint(_visit_tables(), "m5", filtername="F212N",
                            record_dir=str(tmp_path), context="test")
    assert _on_disk(tmp_path)["gate_override"] is None


def test_the_crossfilter_record_carries_it_too(monkeypatch):
    """m7 is demoted by the same variable (`cataloging.
    _run_crossfilter_astrom_checkpoint`), so its record answers the same
    question."""
    monkeypatch.setenv(WARN_ONLY, "1")
    assert AC.record_gate_override("ALLOW_CROSSFILTER_ASTROM_FAIL",
                                   False)["env"] == WARN_ONLY
    import inspect
    src = inspect.getsource(AC.run_crossfilter_checkpoint)
    assert "record_gate_override(" in src, (
        'the m7 record must resolve its override through the same helper, or '
        'the broad one is recorded at m2..m6 and silently dropped at m7')
