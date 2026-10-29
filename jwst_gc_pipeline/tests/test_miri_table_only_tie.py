"""gc3571 MIRI: the consensus offsets table is the only per-frame correction
(``FieldAlignment.table_only_tie``).

Image3's absolute tweakreg tied every frame back to the VIRAC2 catalog the m2
checkpoint checks, starting from the fix_alignment-shifted WCS, so it cancelled
the table's bulk row: o001 F560W read ~47 mas off VIRAC2 at m2 both before and
after the regeneration that applied it.
"""
from dataclasses import replace
from pathlib import Path

import pytest

from jwst_gc_pipeline.reduction import alignment_config as AC


@pytest.mark.parametrize('field', ['001', '003', '004', '005', '016'])
def test_gc3571_is_table_only(field):
    assert AC.table_only_tie('3571', field)
    assert AC.table_only_tie(3571, field)


@pytest.mark.parametrize('proposal, field', [
    ('1266', '004'),     # same shape of field, nothing measured yet
    ('10678', '040'),    # skips tweakreg through InheritedBulk instead
    ('2221', '001'),
    ('9999', '001'),     # unregistered
])
def test_other_fields_keep_tweakreg(proposal, field):
    assert not AC.table_only_tie(proposal, field)


def test_flag_needs_a_consensus_table(monkeypatch):
    cfg = replace(AC.resolve('3571', '001'), source=AC.TABLE_LOCKED)
    monkeypatch.setattr(AC, 'resolve', lambda proposal_id, field: cfg)
    assert not AC.table_only_tie('3571', '001')


def test_the_miri_reducer_skips_tweakreg_on_a_table_only_field():
    src = (Path(AC.__file__).resolve().parent / 'PipelineMIRI.py').read_text()
    branch = src.split('elif table_only_tie(proposal_id, field):')[1]
    branch = branch.split('if abs_refcat is not None:')[0]
    # the whole TweakRegStep: the relative fit runs after the table shift too
    assert "tweakreg_parameters['skip'] = True" in branch
    assert 'abs_refcat = None' in branch
