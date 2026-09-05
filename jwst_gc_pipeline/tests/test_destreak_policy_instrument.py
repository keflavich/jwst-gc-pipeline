"""The lineage token is decided from a RESOLVED instrument, or not at all.

``destreak_policy`` answers two questions off one fact: whether stage 1
destreaks (``destreaks``) and which per-exposure frames stage 2 reads
(``crf_suffix``).  Both hinge on "is this NIRCam", and both answers are
damaging when that is decided wrongly:

* guessing NIRCam for a MIRI observation asks for a ``destreak_``/``align_``
  token no MIRI frame carries, and cataloging finds zero frames (#647);
* guessing not-NIRCam drops the token from a NIRCam run, and brick, cloudc,
  cloudef and sickle each carry BOTH ``o<obs>_crf`` and
  ``destreak_o<obs>_crf`` in one pipeline directory -- an untokened glob reads
  both at once, which is the ~106 mas two-lineage catalog.

So the resolution has to fail closed on both sides, and this file pins the two
ways it did not:

1. ``_instrument_from_filter`` returns its ``instrument`` argument UNCHANGED
   when it does not recognise it, so ``resolved == 'NIRCam'`` was a whitelist
   whose default was "drop the token": ``instrument=''`` and
   ``instrument='nrcalong'`` both produced the untokened NIRCam suffix.
2. The resolution falls back to ``GC_INSTRUMENT_OVERRIDE``, which a shell that
   last ran a MIRI or NIRISS job still has exported and ``sbatch --export=ALL``
   carries into the next job.  Every call site therefore names its instrument
   rather than leaving it to the process environment.
"""
import ast
import fnmatch
import subprocess
from pathlib import Path

import pytest

from jwst_gc_pipeline.reduction import destreak_policy
from jwst_gc_pipeline.reduction.destreak_policy import (
    UnknownInstrumentError, crf_suffix, destreaks)

REPO = Path(__file__).resolve().parents[2]

#: The policy functions whose answer is a lineage decision.
POLICY_CALLS = ('destreaks', 'crf_suffix', 'suffixes_by_filter')


@pytest.mark.parametrize('instrument', ['', 'nrcalong', 'nrca1', 'nircam ',
                                        'NIRCAMLONG', 'unknown'])
def test_an_unresolvable_instrument_raises_rather_than_dropping_the_token(
        instrument):
    """The fail-closed polarity.

    Each of these used to return the bare ``o001_crf`` -- not because the
    caller said MIRI, but because it said something the resolver did not know
    and ``== 'NIRCam'`` was False.
    """
    with pytest.raises(UnknownInstrumentError):
        crf_suffix('brick', 'F410M', '001', instrument=instrument)
    with pytest.raises(UnknownInstrumentError):
        destreaks('brick', 'F410M', True, instrument=instrument)


def test_what_the_dropped_token_would_have_globbed():
    """Why that raise is worth having, stated as the glob it prevents.

    brick/F410M holds both lineages.  ``get_filenames`` builds
    ``*{detector}*{suffix}.fits``, so the untokened suffix the unresolved
    instrument used to return matches BOTH files for the same exposure, while
    the tokened one matches one.
    """
    both_lineages = ['jw02221001001_02101_00001_nrcalong_o001_crf.fits',
                     'jw02221001001_02101_00001_nrcalong_destreak_o001_crf.fits']
    untokened = [f for f in both_lineages
                 if fnmatch.fnmatch(f, '*nrcalong*o001_crf.fits')]
    tokened = [f for f in both_lineages
               if fnmatch.fnmatch(f, '*nrcalong*destreak_o001_crf.fits')]
    assert len(untokened) == 2, 'one exposure, two lineages, one glob'
    assert len(tokened) == 1


@pytest.mark.parametrize('instrument,expected', [
    ('nircam', 'destreak_o001_crf'),
    ('NIRCam', 'destreak_o001_crf'),
    ('NIRCAM', 'destreak_o001_crf'),
    (None, 'destreak_o001_crf'),
    ('niriss', 'o001_crf'),
    ('miri', 'o001_crf'),
])
def test_a_resolved_instrument_still_answers(instrument, expected):
    """The raise is for unresolvable spellings only; the three instruments the
    project reduces, in any case, keep answering -- including ``None``, which
    means "use the filter-name heuristic"."""
    assert crf_suffix('brick', 'F410M', '001', instrument=instrument) == expected


def test_a_stray_instrument_override_cannot_answer_for_a_named_caller(
        monkeypatch):
    """``GC_INSTRUMENT_OVERRIDE=miri`` left over in the submitting shell made
    ``destreaks('brick', 'F410M', True)`` return False, so a NIRCam reduction
    skipped destreaking and printed the field-policy message while doing it.
    A caller that names its instrument is immune."""
    monkeypatch.setenv('GC_INSTRUMENT_OVERRIDE', 'miri')
    assert destreaks('brick', 'F410M', True) is False, (
        'the env fallback itself is unchanged: this is the trap')
    assert destreaks('brick', 'F410M', True, instrument='nircam') is True
    assert crf_suffix('brick', 'F410M', '001',
                      instrument='nircam') == 'destreak_o001_crf'


def _tracked_python_files():
    out = subprocess.run(['git', '-C', str(REPO), 'ls-files', '*.py'],
                         capture_output=True, text=True, check=True)
    return [REPO / line for line in out.stdout.split('\n') if line]


def test_every_policy_call_site_names_its_instrument():
    """No production caller may leave the instrument to the environment.

    The four sites: ``run_pipeline._plan`` (which knows it from the command
    line), ``PipelineRerunNIRCAM-LONG`` (which IS the NIRCam driver),
    ``check_interframe_overlap._reduction_lineage`` (MIRI already returned
    above it) and ``exposure_bundle.enumerate_field_exposures`` (which walks a
    field's NIRCam and MIRI filter directories in one loop, so no single
    process-wide instrument answers for it).
    """
    unnamed = []
    for path in _tracked_python_files():
        if path.name.startswith('test_') or path.name == 'destreak_policy.py':
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (func.attr if isinstance(func, ast.Attribute)
                    else getattr(func, 'id', None))
            if name not in POLICY_CALLS:
                continue
            if not any(kw.arg == 'instrument' for kw in node.keywords):
                unnamed.append(f'{path.relative_to(REPO)}:{node.lineno}')
    assert not unnamed, (
        'these destreak_policy call sites leave the instrument to '
        f'GC_INSTRUMENT_OVERRIDE: {unnamed}')


def test_the_nircam_driver_names_nircam():
    """The site the env leak was demonstrated on, pinned by the value it
    passes rather than only by the presence of the keyword."""
    driver = REPO / 'jwst_gc_pipeline/reduction/PipelineRerunNIRCAM-LONG.py'
    tree = ast.parse(driver.read_text(), filename=str(driver))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, 'id', None) == 'destreaks']
    assert len(calls) == 1, 'one policy call in the NIRCam driver'
    named = {kw.arg: kw.value for kw in calls[0].keywords}
    assert 'instrument' in named
    assert isinstance(named['instrument'], ast.Constant)
    assert named['instrument'].value == 'nircam'


def test_the_policy_module_lists_the_instruments_it_resolves():
    """The whitelist is a named constant so the raise can quote it."""
    assert destreak_policy.KNOWN_INSTRUMENTS == ('NIRCam', 'MIRI', 'NIRISS')
