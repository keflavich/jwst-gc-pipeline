"""A checkpoint that measured nothing has not passed (#350).

`duplicate_exposure` already did the right thing at the point of detection: the
visit is recorded with `error_kind='duplicate_exposure'`, appended to
`failures`, and `_checkpoint_passed` returns False.

The caller then threw that away:

    corrections = record.get('corrections') or []
    if not corrections:
        print("... PASS (no correction implied)")
        return

Nothing was measured, so nothing could be corrected, so the empty corrections
list read as a pass.  Every gc2211 observation ran that way -- m2 recorded ZERO
exposures with passed=False, the m12 finalize exited 0, the retie loop printed
"m2 checkpoint PASSED -- converged after 1 iter(s)", and the frozen m3 check
then failed for want of a baseline that was never written.
"""
import os

import pytest

from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    AstrometryCheckpointFailedError)


def _record(passed=True, failures=(), corrections=()):
    return dict(passed=passed, failures=list(failures),
                corrections=list(corrections), record_path='/x/rec.json')


def _gate(record, warn_only=False, monkeypatch=None):
    """The caller's decision, as `_run_astrometry_stage_checkpoint` makes it."""
    _failures = record.get('failures') or []
    if record.get('passed') is False or _failures:
        if warn_only:
            return 'warned'
        raise AstrometryCheckpointFailedError('failed')
    if not (record.get('corrections') or []):
        return 'pass-no-correction'
    return 'correct'


def test_a_duplicate_exposure_record_is_NOT_a_pass():
    """The gc2211 shape: passed=False, one failure, zero corrections."""
    rec = _record(passed=False,
                  failures=['gc2211 F200W visit 1 [m2]: duplicate exposure '
                            'identity: 36 exposure identity/ies ingested more '
                            'than once'])
    with pytest.raises(AstrometryCheckpointFailedError):
        _gate(rec)


def test_zero_corrections_with_a_CLEAN_record_is_still_a_pass():
    """The common case must not become fatal: a checkpoint that measured
    everything and found nothing to correct passes, as before."""
    assert _gate(_record()) == 'pass-no-correction'


def test_failures_alone_are_enough_even_if_passed_is_missing():
    """Older records may not carry `passed`; a non-empty failures list is
    itself disqualifying."""
    rec = _record(failures=['something failed'])
    rec.pop('passed')
    with pytest.raises(AstrometryCheckpointFailedError):
        _gate(rec)


def test_warn_only_demotes_it():
    """ASTROM_CHECKPOINT_WARN_ONLY=1 is the documented escape and must still
    reach the rest of the function."""
    rec = _record(passed=False, failures=['duplicate exposure identity'])
    assert _gate(rec, warn_only=True) == 'warned'


def test_the_error_is_its_own_type():
    """Distinct from AstrometryCorrectionRequiredError (the gate RAN and wants
    a correction) and AstrometryRegressionError (a frozen solution MOVED).
    This one means the gate could not run at all, and a caller that wants to
    tell them apart must be able to."""
    from jwst_gc_pipeline.photometry import astrometry_checkpoint as ac
    assert not issubclass(ac.AstrometryCheckpointFailedError,
                          ac.AstrometryCorrectionRequiredError)
    assert not issubclass(ac.AstrometryCheckpointFailedError,
                          ac.AstrometryRegressionError)


def test_the_caller_actually_consults_passed_and_failures():
    """Source guard: the check lives before the corrections short-circuit, and
    deleting it restores the fail-open with every behavioural test still green
    (they exercise the decision, not its placement)."""
    import inspect
    from jwst_gc_pipeline.photometry import cataloging
    src = inspect.getsource(cataloging._run_astrometry_stage_checkpoint)
    assert "record.get('failures')" in src
    assert "record.get('passed') is False" in src
    gate_at = src.index("record.get('passed') is False")
    corr_at = src.index("corrections = record.get('corrections')")
    assert gate_at < corr_at, (
        'the failure gate must precede the no-corrections short-circuit, or a '
        'record that failed with zero corrections still reads as a pass')
