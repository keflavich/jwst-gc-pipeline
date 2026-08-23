"""``fix_alignment`` resolves its proposal from the frame's own PROGRAM header.

Issue #440.  All three ``fix_alignment`` implementations used to recover the
proposal by parsing the input basename, so a frame renamed or copied into
another field's tree was attributed to whatever the new name said, silently:
the proposal chooses the offsets table, the reference catalog and the module
policy, and every one of those lookups succeeds for the wrong proposal instead
of failing.  ``mod = ImageModel(fn)`` is open on the line above the
assignment, and ``mod.meta.observation.program_number`` is the ``PROGRAM``
keyword -- it travels inside the file and survives the rename.

Two halves are pinned here.  ``proposal_id_from_datamodel`` is pinned by
VALUE, including the de-pad (``'02221'`` -> ``'2221'``, so a caller comparing
against the literal ``'2221'`` keeps working) and the filename fallback for a
model with no ``PROGRAM``.  The three call sites are pinned by SOURCE, in the
style of ``test_five_digit_proposal_call_sites.py``: importing the reduce
drivers pulls in the JWST calibration stack, and ``fix_alignment`` opens a
real frame, applies the DVA correction and reads an offsets table before it
reaches the line under test, so there is nothing callable to assert against
here.  The source assertions fail if a driver goes back to
``proposal_id_from_filename(fn)``.

The third assertion is that the NIRCam-LONG driver holds ONE
``if proposal_id is None:`` in ``fix_alignment``.  It used to hold two -- the
filename parse, then a late fallback to ``program_number`` that could not run,
because the parse either returns a value or raises.  With the header read
first, that second branch is the same read a second time.

Vocabulary.  ``fix_alignment`` is the per-frame astrometric correction the
reduce applies before resampling; a *datamodel* is the ``jwst`` package's
in-memory representation of a frame, whose ``meta`` tree mirrors the FITS
header.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from jwst_gc_pipeline.mast_names import proposal_id_from_datamodel

REDUCTION = Path(__file__).resolve().parents[1] / 'reduction'

DRIVERS = {
    'nircam': REDUCTION / 'PipelineRerunNIRCAM-LONG.py',
    'miri': REDUCTION / 'PipelineMIRI.py',
    'niriss': REDUCTION / 'PipelineRerunNIRISS.py',
}


def _model(program_number):
    """A stand-in for an open ``ImageModel``: only ``meta.observation`` is read."""
    return SimpleNamespace(
        meta=SimpleNamespace(observation=SimpleNamespace(
            program_number=program_number)))


def _fix_alignment_node(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'fix_alignment':
            return node
    raise AssertionError(f'{path.name} defines no fix_alignment')


# ---------------------------------------------------------------------------
# the helper, by value
# ---------------------------------------------------------------------------

def test_header_program_is_depadded_to_the_pipeline_key():
    """``PROGRAM='02221'`` is the padded MAST form; the pipeline key is '2221'."""
    assert proposal_id_from_datamodel(_model('02221')) == '2221'


def test_five_digit_program_keeps_all_five_digits():
    assert proposal_id_from_datamodel(_model('10678')) == '10678'


def test_header_wins_over_a_disagreeing_filename():
    """The point of the change: a frame copied under another proposal's name
    still reports the proposal it was observed under."""
    renamed = 'jw01182004001_02101_00001_nrca1_cal.fits'
    assert proposal_id_from_datamodel(_model('02221'), renamed) == '2221'


def test_filename_is_the_fallback_when_the_header_is_empty():
    frame = 'jw10678001001_02101_00001_nrca1_cal.fits'
    assert proposal_id_from_datamodel(_model(None), frame) == '10678'
    assert proposal_id_from_datamodel(_model('  '), frame) == '10678'


def test_no_program_and_no_filename_raises():
    with pytest.raises(ValueError, match='program_number'):
        proposal_id_from_datamodel(_model(None))


def test_a_program_that_is_not_a_proposal_raises():
    with pytest.raises(ValueError):
        proposal_id_from_datamodel(_model('not-a-program'))


# ---------------------------------------------------------------------------
# the three call sites, by source
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('instrument', sorted(DRIVERS))
def test_fix_alignment_reads_the_datamodel_not_the_filename(instrument):
    source = ast.unparse(_fix_alignment_node(DRIVERS[instrument]))
    assert 'proposal_id_from_datamodel(mod, fn)' in source, (
        f'{DRIVERS[instrument].name}: fix_alignment must resolve proposal_id '
        f'from the open datamodel (issue #440)')
    assert 'proposal_id_from_filename' not in source, (
        f'{DRIVERS[instrument].name}: fix_alignment still parses the basename')


def test_nircam_fix_alignment_has_one_proposal_id_fallback():
    """The second ``if proposal_id is None:`` was unreachable; it is gone."""
    node = _fix_alignment_node(DRIVERS['nircam'])
    guards = [n for n in ast.walk(node)
              if isinstance(n, ast.If)
              and 'proposal_id is None' in ast.unparse(n.test)]
    assert len(guards) == 1, (
        f'expected one proposal_id fallback in NIRCam fix_alignment, '
        f'found {len(guards)}')
