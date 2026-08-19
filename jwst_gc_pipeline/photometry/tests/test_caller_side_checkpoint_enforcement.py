"""The frozen-stage deferral has to hold at BOTH enforcement points.

`run_visit_checkpoint` decides whether a frozen-stage failure raises where it is
measured or is recorded for the release gate (`ASTROM_CHECKPOINT_ENFORCE`,
#442).  `cataloging._run_astrometry_stage_checkpoint` then re-read the record's
``failures`` and raised `AstrometryCheckpointFailedError` regardless, so the
deferral changed nothing about whether the chain survived.

Measured on 2026-08-19, on chains resubmitted from `main` specifically to use
the deferral:

    sickle  m3   "...the release gate refuses this field"  then rc=1, chain dead
    cloudef m3   same, 6985 s in

Both logs contain the deferral message AND the death, three lines apart.  So the
policy now lives in one function, `frozen_failure_is_deferred`, and both points
ask it.
"""
import pytest

from jwst_gc_pipeline.photometry import cataloging as _cat
from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    CHECKPOINT_ENFORCE_ENV, CORRECTION_STAGES, ENFORCE_AT_RELEASE,
    ENFORCE_AT_STAGE, frozen_failure_is_deferred)


# ---------------------------------------------------------------------------
# the policy itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('stage', ['m3', 'm4', 'm5', 'm6', 'm7'])
def test_a_frozen_stage_is_deferred_at_the_default(monkeypatch, stage):
    monkeypatch.delenv(CHECKPOINT_ENFORCE_ENV, raising=False)
    assert frozen_failure_is_deferred(stage) is True


@pytest.mark.parametrize('stage', ['m1', 'm2', 'm12'])
def test_a_CORRECTING_stage_is_never_deferred(monkeypatch, stage):
    """m2's stop is what sends the field back for regeneration, and nothing
    downstream can do that for it.  Deferring it would run the whole chain on
    frames m2 has already declared wrong."""
    monkeypatch.delenv(CHECKPOINT_ENFORCE_ENV, raising=False)
    assert stage in CORRECTION_STAGES
    assert frozen_failure_is_deferred(stage) is False


def test_stage_enforcement_defers_nothing(monkeypatch):
    monkeypatch.setenv(CHECKPOINT_ENFORCE_ENV, ENFORCE_AT_STAGE)
    assert frozen_failure_is_deferred('m3') is False


@pytest.mark.parametrize('typo', ['relase', '1', 'yes', 'off', ''])
def test_a_TYPO_defers_nothing(monkeypatch, typo):
    """Anything that is not exactly "release" is the strict branch, so a
    misspelling costs a stopped chain rather than a shipped misalignment.  The
    empty string is the unset case and takes the default."""
    monkeypatch.setenv(CHECKPOINT_ENFORCE_ENV, typo)
    expected = (typo == '')      # unset -> default -> deferred
    assert frozen_failure_is_deferred('m3') is expected


# ---------------------------------------------------------------------------
# both enforcement points ask it
# ---------------------------------------------------------------------------

def _hook_source():
    import inspect
    return inspect.getsource(_cat._run_astrometry_stage_checkpoint)


def test_the_caller_side_raise_asks_the_policy():
    """The regression this file exists for: `cataloging` raising
    `AstrometryCheckpointFailedError` without consulting the policy is what made
    #442 a no-op for the chain."""
    src = _hook_source()
    n_raise = src.count('raise AstrometryCheckpointFailedError')
    n_ask = src.count('frozen_failure_is_deferred(merge_label)')
    assert n_raise == 2, f'expected two raise sites, found {n_raise}'
    assert n_ask == n_raise, (
        f'{n_raise} raise site(s) but {n_ask} consult the deferral policy; '
        f'a raise that does not ask is a second enforcement point that can '
        f'disagree with the first')


def test_the_measured_failures_are_still_printed():
    """Deferring is not going quiet.  Both branches print the failure list
    before deciding, so an operator sees it where it was measured rather than
    only at the release."""
    src = _hook_source()
    head, _, tail = src.partition('elif frozen_failure_is_deferred(merge_label)')
    assert 'failure(s)' in head, 'the failures must be formatted before the branch'
    assert tail, 'the deferral branch must exist'


def test_the_deferral_says_where_the_stop_moved_to():
    src = _hook_source()
    assert 'the chain CONTINUES' in src
    assert 'release gate refuses this field' in src
    assert 'check_astrometry_checkpoints.py' in src
    assert f'{{CHECKPOINT_ENFORCE_ENV}}={{ENFORCE_AT_STAGE}}' in src, (
        'the message must say how to restore the old behaviour')


def test_warn_only_is_checked_before_the_policy():
    """`ASTROM_CHECKPOINT_WARN_ONLY=1` predates this and still wins, at every
    stage including a correcting one."""
    src = _hook_source()
    for block in src.split('raise AstrometryCheckpointFailedError')[:-1]:
        assert block.rindex('if warn_only:') < block.rindex(
            'frozen_failure_is_deferred(merge_label)'), (
            'warn_only must be tested before the deferral policy')
