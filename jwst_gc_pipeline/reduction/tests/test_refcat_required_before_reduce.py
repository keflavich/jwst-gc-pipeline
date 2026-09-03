"""The reduce driver refuses a VIRAC2-framed field with no reference catalog.

``get_existing_reference_astrometric_catalog_path`` used to return ``None`` for
every field the registry had no entry for, on the reading that such a field
"sets its absolute zero point some other way".  That is true of m4, m92,
ngc6397, w51, wd1 obs003 and wd2 obs003 -- all Gaia-framed or unconfigured --
and false of a field whose ``alignment_config`` frame is VIRAC2, whose tie is
MADE against that catalog.  For those the reduce ran to completion and the
first refusal came at the m2 checkpoint, hours later.

That matters for program 10678 specifically, whose 139 tiles are reduced by an
automated trigger: ``data_qa.pipeline_trigger`` submits
``scripts/reduction/submit_reduction.sbatch``, which runs this driver directly
and never passes through ``run_pipeline.build_plan``, where the plan-time
refusal lives.

``PipelineRerunNIRCAM-LONG.py`` has a hyphen and imports the whole JWST stack,
so the one function is lifted out by parsing, the same approach as
``test_field_normalised_at_entry.py``.
"""
import ast
import os
import pathlib

import pytest

from jwst_gc_pipeline import fields as F
from jwst_gc_pipeline.reduction import alignment_config as ac

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DRIVER = REPO_ROOT / 'jwst_gc_pipeline' / 'reduction' / 'PipelineRerunNIRCAM-LONG.py'
_TREE = ast.parse(DRIVER.read_text(), filename=str(DRIVER))


def _getter():
    """The driver's existence-checked getter, compiled on its own."""
    for node in ast.walk(_TREE):
        if (isinstance(node, ast.FunctionDef)
                and node.name == 'get_existing_reference_astrometric_catalog_path'):
            module = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(module)
            namespace = {'os': os, 'field_registry': F}
            exec(compile(module, str(DRIVER), 'exec'), namespace)
            return namespace['get_existing_reference_astrometric_catalog_path']
    raise AssertionError(f'{DRIVER} no longer defines the getter')


GETTER = _getter()


def test_an_unregistered_virac2_framed_tile_stops_the_reduce(tmp_path):
    """10678 registers no reference catalog for any of its 139 tiles (one
    CMZ-wide file would be the wrong sky for nearly all of them), and its
    alignment config declares the VIRAC2 frame.  The reduce must stop here,
    not at m2 after a full reduction."""
    with pytest.raises(F.FieldRegistryError) as excinfo:
        GETTER(str(tmp_path), '10678', '088')
    message = str(excinfo.value)
    assert 'VIRAC2' in message
    assert '088' in message
    # it names what the operator has to do, not just what is missing
    assert 'reference_catalog:' in message
    assert 'build_treasury_refcats' in message


def test_a_gaia_framed_field_with_no_entry_still_reduces(tmp_path):
    """The behaviour the ``None`` return exists for.  wd1 obs003 and wd2 obs003
    have no ``reference_catalog`` key and a Gaia frame; they reduce today and
    must keep doing so."""
    assert GETTER(str(tmp_path), '1905', '003') is None
    assert GETTER(str(tmp_path), '3523', '003') is None


def test_an_unknown_proposal_still_reduces(tmp_path):
    """Nothing in fields.yaml and nothing in alignment_config: no reference is
    declared, so none is required."""
    assert GETTER(str(tmp_path), '99999', '001') is None


def test_a_registered_catalog_whose_file_is_absent_still_raises(tmp_path):
    """The pre-existing refusal, unchanged: a field wired to a catalog that
    was never built stops before producing products naming a missing file."""
    with pytest.raises(FileNotFoundError, match='MISSING'):
        GETTER(str(tmp_path), '2221', '001')


def test_the_reduce_and_the_m2_checkpoint_ask_the_same_question():
    """Two halves of one field's tie must not disagree about whether the
    catalog is optional.  ``cataloging._refcat_is_required`` is the m2 side."""
    from jwst_gc_pipeline.photometry import cataloging
    for proposal, field in (('10678', '088'), ('1905', '003'),
                            ('3523', '003'), ('2221', '001'),
                            ('99999', '001'), (None, None)):
        assert (cataloging._refcat_is_required(proposal, field)
                is ac.reference_catalog_required(proposal, field)), (
                    proposal, field)


@pytest.mark.parametrize('proposal,field,expected', [
    ('10678', '088', True),     # gc-treasury, VIRAC2, proposal-wide entry
    ('10678', '139', True),
    ('2221', '001', True),      # brick, VIRAC2
    ('1905', '003', False),     # wd1, Gaia
    ('3523', '003', False),     # wd2, no alignment entry
    ('1979', '002', False),     # m4, Gaia
    (None, None, False),
])
def test_the_requirement_follows_the_declared_frame(proposal, field, expected):
    assert ac.reference_catalog_required(proposal, field) is expected
