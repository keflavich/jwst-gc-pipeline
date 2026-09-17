"""A checkpoint that measured nothing has not passed (#350) -- and one that
measured and refused is a different thing, which must not be reported as the
first.

`duplicate_exposure` already did the right thing at the point of detection: the
visit is recorded with `error_kind='duplicate_exposure'`, appended to
`failures`, and `_checkpoint_passed` returns False.

The caller then threw that away::

    corrections = record.get('corrections') or []
    if not corrections:
        print("... PASS (no correction implied)")
        return

Nothing was measured, so nothing could be corrected, so the empty corrections
list read as a pass.  Every gc2211 observation ran that way -- m2 recorded ZERO
exposures with passed=False, the m12 finalize exited 0, the retie loop printed
"m2 checkpoint PASSED -- converged after 1 iter(s)", and the frozen m3 check
then failed for want of a baseline that was never written.

## The three shapes, which are not the same thing

`_checkpoint_passed` returns False for two unrelated reasons, and #341 made the
second reachable with an EMPTY failures list:

  * ``failures`` non-empty -- the gate could NOT RUN.  Malformed inputs; nothing
    was measured, so nothing was checked.  Fatal.
  * ``failures`` empty, ``unverified_blocking`` non-empty -- the gate RAN and
    REFUSED a measured item (#312/#341): a gross consensus->reference tie, a
    module-antisymmetric alias.  Reporting that as "0 failure(s); the gate did
    NOT run" is false on both counts, and 14 live m2 records already have
    exactly that shape (cloudc F410M, cloudef F480M, gc2211 F200W_o023...).
  * neither list populated -- the record cannot say what happened, which is the
    fail-open this exists to close.

These drive the REAL `_run_astrometry_stage_checkpoint`: an earlier version
tested a local copy of the decision, which can drift from the original while
staying green.
"""
import os
import types

import pytest
from astropy.table import Table

from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    ALLOW_UNVERIFIED_ENV, AstrometryCheckpointFailedError, _checkpoint_passed)


def _record(passed=True, failures=(), corrections=(), blocking=()):
    return dict(passed=passed, failures=list(failures),
                corrections=list(corrections),
                unverified_blocking=list(blocking),
                record_path='/x/rec.json')


def _run(tmp_path, monkeypatch, record, warn_only=False, filt='F200W',
         stage='m2'):
    """Call the real caller with a canned checkpoint record.

    Only `run_visit_checkpoint` and the reference-catalog load are stubbed --
    everything from the per-frame glob through the decision is the shipping
    code path.

    `stage` selects which enforcement policy the decision runs under:
    `frozen_failure_is_deferred` defers at a FROZEN stage ('m3'...) and never
    at a CORRECTION_STAGES one ('m2'), so a test about the deferred branch has
    to be able to ask for the other half.
    """
    from jwst_gc_pipeline.photometry import cataloging

    cut_bp = str(tmp_path)
    d = tmp_path / filt.upper()
    d.mkdir(parents=True)
    (d / f'{filt.lower()}_nrca1_visit001_vgroup02201_exp00001_{stage}_'
         f'daophot_basic.fits').write_bytes(b'')
    tbl = Table({'x': [1.0]})
    tbl.write(str(d / f'{filt.lower()}_nrca1_visit001_vgroup02201_exp00001_'
                      f'{stage}_daophot_basic.fits'), overwrite=True)

    # `run_visit_checkpoint` is imported inside the function, so patch it at
    # the source module.  A pre-populated refcat cache skips the catalog load.
    from jwst_gc_pipeline.photometry import astrometry_checkpoint
    monkeypatch.setattr(astrometry_checkpoint, 'run_visit_checkpoint',
                        lambda *a, **kw: record)
    monkeypatch.setenv('ASTROM_CHECKPOINT_WARN_ONLY', '1' if warn_only else '')

    options = types.SimpleNamespace(field='004', proposal_id='1182',
                                    target='brick')
    return cataloging._run_astrometry_stage_checkpoint(
        stage, 'nrca', filt, cut_bp, cut_bp, '1182', options,
        {'refcat': {'all': None, 'sparse': None}}, context='test')


# ---------------------------------------------------------------------------
# 1. the gate could not run
# ---------------------------------------------------------------------------

def test_a_duplicate_exposure_record_is_NOT_a_pass(tmp_path, monkeypatch):
    """The gc2211 shape: passed=False, one failure, zero corrections."""
    rec = _record(passed=False,
                  failures=['gc2211 F200W visit 1 [m2]: duplicate exposure '
                            'identity: 36 exposure identity/ies ingested more '
                            'than once'])
    with pytest.raises(AstrometryCheckpointFailedError) as ex:
        _run(tmp_path, monkeypatch, rec)
    assert 'the gate did NOT run' in str(ex.value)


def test_failures_alone_are_enough_even_if_passed_is_missing(tmp_path,
                                                              monkeypatch):
    """Older records may not carry `passed`; a non-empty failures list is
    itself disqualifying."""
    rec = _record(failures=['something failed'])
    rec.pop('passed')
    with pytest.raises(AstrometryCheckpointFailedError):
        _run(tmp_path, monkeypatch, rec)


def test_warn_only_demotes_it(tmp_path, monkeypatch, capsys):
    """ASTROM_CHECKPOINT_WARN_ONLY=1 is the documented escape and must still
    reach the rest of the function."""
    rec = _record(passed=False, failures=['duplicate exposure identity'])
    _run(tmp_path, monkeypatch, rec, warn_only=True)
    assert 'WARN_ONLY=1 -- continuing' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 2. the gate RAN and refused -- #312/#341, not this PR's failure mode
# ---------------------------------------------------------------------------

def test_a_MEASURED_AND_REFUSED_record_is_not_called_a_gate_that_did_not_run(
        tmp_path, monkeypatch, capsys):
    """#341 made `passed=False` with an EMPTY failures list reachable: a gross
    consensus->reference tie is measured, refused, and recorded as blocking.
    Calling that "0 failure(s); the gate did NOT run" is wrong on both counts
    and would convert a documented advisory into a fatal blaming malformed
    inputs."""
    rec = _record(passed=False, blocking=[
        'cloudc F410M/nrcblong visit 2 [m2]: consensus is 731 mas off VIRAC2 '
        '-- MEASURED and refused'])
    _run(tmp_path, monkeypatch, rec)          # must not raise
    out = capsys.readouterr().out
    assert 'MEASURED and REFUSED' in out
    assert 'the gate ran' in out
    assert 'did NOT run' not in out
    assert '0 failure(s)' not in out


def test_a_refused_record_still_reports_the_item_and_its_hatch(tmp_path,
                                                               monkeypatch,
                                                               capsys):
    rec = _record(passed=False, blocking=['the offending item'])
    _run(tmp_path, monkeypatch, rec)
    out = capsys.readouterr().out
    assert 'the offending item' in out
    assert f'{ALLOW_UNVERIFIED_ENV}=1' in out


def test_the_hatch_it_names_is_one_the_code_reads(tmp_path, monkeypatch,
                                                  capsys):
    """The env var this message tells the operator to set must be the one
    `_checkpoint_passed` consults.

    It was not: the line read `ALLOW_UNVERIFIED_ASTROM=1` while the reader is
    `ALLOW_UNVERIFIED_ASTROM_CHECKPOINT` (issue #400 item 1).  Setting the name
    in the message changed nothing, and the next lever to hand
    (`ASTROM_CHECKPOINT_WARN_ONLY=1`) demotes every blocking failure rather
    than the one refused item.  The old assertion pinned the typo, so widening
    the message alone would not have caught it.

    Asserted behaviourally: setting exactly the name the message prints must
    turn this record into a pass.
    """
    rec = _record(passed=False, blocking=['the offending item'])
    _run(tmp_path, monkeypatch, rec)
    printed = [tok.rstrip('=1') for tok in capsys.readouterr().out.split()
               if tok.endswith('=1') and tok[0].isupper()]
    hatches = [name for name in printed if name.startswith('ALLOW_UNVERIFIED')]
    assert hatches, 'the refused-record message names no ALLOW_UNVERIFIED hatch'
    for name in hatches:
        monkeypatch.setenv(name, '1')
        assert _checkpoint_passed([], ['the offending item']) is True, (
            f'{name} is printed as the escape hatch but does not make a '
            f'measured-and-refused record pass')
        monkeypatch.delenv(name)


def test_a_refused_record_does_NOT_then_announce_a_PASS(tmp_path, monkeypatch,
                                                        capsys):
    """The o135 shape (#871): refused, zero corrections, and the very next line
    called it a pass.

    `corrections` is empty precisely BECAUSE the tie was refused -- a tie m2
    declines to apply is a tie m2 does not write -- so the branch that
    announces a clean checkpoint fired on the refused one too, and its PASS was
    the last word in the log.  gc-treasury o135 m12 finalize 41922332
    (2026-09-13) printed, for F212N/merged:

        ASTROM CHECKPOINT [m2]: NOT A PASS -- 1 item(s) were MEASURED and
          refused ... consensus->reference offset 62.45 mas
        astrom checkpoint [m2] F212N/merged: NOT A PASS -- 1 item(s) ...
        astrom checkpoint [m2] F212N/merged: PASS (no correction implied)

    The finalize exited 0 and m3 froze the 62 mas in.
    """
    rec = _record(passed=False, corrections=[], blocking=[
        'gc-treasury F212N/merged F212N visit 1 [m2]: consensus->reference '
        'offset 62.45 mas but the tie is not trustworthy -- NOT applying'])
    _run(tmp_path, monkeypatch, rec)          # still not fatal (#312/#341)
    out = capsys.readouterr().out
    assert 'MEASURED and REFUSED' in out
    assert 'PASS (no correction implied)' not in out, (
        'a refused record announced itself as a pass three lines after being '
        'refused, and the PASS was what the operator and the retie loop read')
    assert 'NOT a pass' in out


def test_a_warn_only_demotion_does_NOT_then_announce_a_PASS(tmp_path,
                                                            monkeypatch,
                                                            capsys):
    """ASTROM_CHECKPOINT_WARN_ONLY=1 asks the chain to CONTINUE past a
    failure.  It does not turn the failure into a pass, and the demoted record
    reaches the corrections short-circuit with an empty list like any other."""
    rec = _record(passed=False, failures=['duplicate exposure identity'])
    _run(tmp_path, monkeypatch, rec, warn_only=True)
    out = capsys.readouterr().out
    assert 'WARN_ONLY=1 -- continuing' in out
    assert 'PASS (no correction implied)' not in out


def test_a_DEFERRED_frozen_failure_does_NOT_then_announce_a_PASS(
        tmp_path, monkeypatch, capsys):
    """The deferral says "the release gate refuses this field" and continues.
    Printing PASS after that sentence contradicts it in the same log."""
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        CHECKPOINT_ENFORCE_ENV, ENFORCE_AT_RELEASE)
    monkeypatch.setenv(CHECKPOINT_ENFORCE_ENV, ENFORCE_AT_RELEASE)
    rec = _record(passed=False, failures=['consensus->reference MOVED 62 mas'])
    _run(tmp_path, monkeypatch, rec, stage='m3')      # frozen -> deferred
    out = capsys.readouterr().out
    assert 'the release gate refuses this field' in out
    assert 'PASS (no correction implied)' not in out


def test_the_opt_in_hatch_still_produces_a_real_PASS(tmp_path, monkeypatch,
                                                     capsys):
    """ALLOW_UNVERIFIED_ASTROM_CHECKPOINT=1 is the documented way to proceed
    past a measured-and-refused item.  It works upstream -- `_checkpoint_passed`
    returns True, so the record arrives here with passed=True -- and such a
    record must still read as a pass.  Narrowing the PASS line must not take
    the hatch away."""
    monkeypatch.setenv(ALLOW_UNVERIFIED_ENV, '1')
    rec = _record(passed=True, blocking=['the operator accepted this one'])
    _run(tmp_path, monkeypatch, rec)
    assert 'PASS (no correction implied)' in capsys.readouterr().out


def test_the_PASS_line_is_gated_on_the_record_not_on_list_length():
    """Source guard: the corrections short-circuit must consult the verdict.

    Reverting to a bare `if not corrections: print(PASS)` restores #871, and
    every behavioural test above would still pass if the gate were moved to
    somewhere that does not run for the deferred branch.

    Anchored on the f-string that BUILDS the line (`{module}: PASS `), not on
    the rendered text: the comment above it quotes the o135 log verbatim, so
    the rendered form appears earlier in the source and matching that would
    measure the comment instead of the code.
    """
    import inspect

    from jwst_gc_pipeline.photometry import cataloging
    src = inspect.getsource(cataloging._run_astrometry_stage_checkpoint)
    corr_at = src.index("corrections = record.get('corrections')")
    tail = src[corr_at:]
    pass_at = tail.index('{module}: PASS ')
    assert "record.get('passed') is False" in tail[:pass_at], (
        'the PASS line is reached without re-reading the verdict, so a '
        'passed=false record with zero corrections announces a pass again')


def test_failures_WIN_when_both_lists_are_populated(tmp_path, monkeypatch):
    """A record with both is a gate that could not run; the harsher, accurate
    reading takes precedence."""
    rec = _record(passed=False, failures=['duplicate exposure identity'],
                  blocking=['also refused something'])
    with pytest.raises(AstrometryCheckpointFailedError) as ex:
        _run(tmp_path, monkeypatch, rec)
    assert 'the gate did NOT run' in str(ex.value)


# ---------------------------------------------------------------------------
# 3. the record cannot say what happened
# ---------------------------------------------------------------------------

def test_passed_False_with_NEITHER_list_is_still_fatal(tmp_path, monkeypatch):
    """The fail-open shape: nothing in the record explains the verdict, so it
    cannot be treated as a pass."""
    rec = _record(passed=False)
    with pytest.raises(AstrometryCheckpointFailedError) as ex:
        _run(tmp_path, monkeypatch, rec)
    assert 'cannot say what was checked' in str(ex.value)


def test_zero_corrections_with_a_CLEAN_record_is_still_a_pass(tmp_path,
                                                              monkeypatch,
                                                              capsys):
    """The common case must not become fatal: a checkpoint that measured
    everything and found nothing to correct passes, as before."""
    _run(tmp_path, monkeypatch, _record())
    assert 'PASS (no correction implied)' in capsys.readouterr().out


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
    deleting it restores the fail-open."""
    import inspect

    from jwst_gc_pipeline.photometry import cataloging
    src = inspect.getsource(cataloging._run_astrometry_stage_checkpoint)
    assert "record.get('failures')" in src
    assert "record.get('passed') is False" in src
    assert "record.get('unverified_blocking')" in src, (
        'without this the two failure modes cannot be told apart')
    gate_at = src.index("record.get('failures')")
    corr_at = src.index("corrections = record.get('corrections')")
    assert gate_at < corr_at, (
        'the failure gate must precede the no-corrections short-circuit, or a '
        'record that failed with zero corrections still reads as a pass')


def test_the_gate_uses_the_functions_own_warn_only_local():
    """The new branches must reuse `warn_only`, not re-read the environment.

    The function reads the env once BEFORE `warn_only` exists (the
    no-per-frame-catalogs branch runs earlier and returns), and once to define
    it.  Anything after that assignment is a third copy that can drift from the
    demotion path used twenty lines above -- which is what the review caught.
    """
    import inspect

    from jwst_gc_pipeline.photometry import cataloging
    src = inspect.getsource(cataloging._run_astrometry_stage_checkpoint)
    assign = src.index("warn_only = os.environ.get(")
    after = src[assign + len("warn_only = os.environ.get("):]
    assert "os.environ.get('ASTROM_CHECKPOINT_WARN_ONLY'" not in after, (
        're-reading the env after `warn_only` is defined lets the demotion '
        'switch drift between branches')
    assert after.count('if warn_only:') >= 2, (
        'the new gate branches should demote through the same local')
