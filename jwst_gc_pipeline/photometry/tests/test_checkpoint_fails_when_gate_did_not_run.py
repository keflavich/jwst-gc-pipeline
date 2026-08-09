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
    AstrometryCheckpointFailedError)


def _record(passed=True, failures=(), corrections=(), blocking=()):
    return dict(passed=passed, failures=list(failures),
                corrections=list(corrections),
                unverified_blocking=list(blocking),
                record_path='/x/rec.json')


def _run(tmp_path, monkeypatch, record, warn_only=False, filt='F200W'):
    """Call the real caller with a canned checkpoint record.

    Only `run_visit_checkpoint` and the reference-catalog load are stubbed --
    everything from the per-frame glob through the decision is the shipping
    code path.
    """
    from jwst_gc_pipeline.photometry import cataloging

    cut_bp = str(tmp_path)
    d = tmp_path / filt.upper()
    d.mkdir(parents=True)
    (d / f'{filt.lower()}_nrca1_visit001_vgroup02201_exp00001_m2_'
         f'daophot_basic.fits').write_bytes(b'')
    tbl = Table({'x': [1.0]})
    tbl.write(str(d / f'{filt.lower()}_nrca1_visit001_vgroup02201_exp00001_m2_'
                      f'daophot_basic.fits'), overwrite=True)

    # `run_visit_checkpoint` is imported inside the function, so patch it at
    # the source module.  A pre-populated refcat cache skips the catalog load.
    from jwst_gc_pipeline.photometry import astrometry_checkpoint
    monkeypatch.setattr(astrometry_checkpoint, 'run_visit_checkpoint',
                        lambda *a, **kw: record)
    monkeypatch.setenv('ASTROM_CHECKPOINT_WARN_ONLY', '1' if warn_only else '')

    options = types.SimpleNamespace(field='004', proposal_id='1182',
                                    target='brick')
    return cataloging._run_astrometry_stage_checkpoint(
        'm2', 'nrca', filt, cut_bp, cut_bp, '1182', options,
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
    assert 'ALLOW_UNVERIFIED_ASTROM=1' in out


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
