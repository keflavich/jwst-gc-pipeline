"""Which observations the reduce knows are single-module.

`MODULES_BY_PROPOSAL_FIELD_FILTER` is the one place the pipeline records that an
observation was taken with only part of NIRCam.  `get_allowed_modules` narrows a
module request through it and `scripts/reduction/preflight_reduce_inputs.py`
parses it, so an observation IN the registry is protected for every caller and
one that is missing reproduces the same abort for each of them:

    ValueError: No nrca members found in ... jw02211-o050_..._asn.json for
    filter F200W field 050 proposal 2211.

The registry is read here by `ast.parse` rather than by importing the driver,
which pulls in the whole JWST stack; that is the same reading the preflight
does (`reduce_module_policy`), so a spelling the preflight cannot parse fails
here too.

Facts pinned (issue #436), counted from the `_cal` frames on disk 2026-08-22:

  * sickle 3958/007 -- nrcb only, SW as four detectors, LW as the family;
  * gc2211 2211/050 -- nrcb only, F200W (48 frames, nrcb1-4) and F277W
    (12 frames, nrcblong), while 023/028/046/049 carry both modules.
"""
import ast
from pathlib import Path

import pytest

_DRIVER = (Path(__file__).resolve().parents[1] / 'PipelineRerunNIRCAM-LONG.py')


def _module_family(module):
    """`_module_group` from the driver, without importing it."""
    if module == 'merged':
        return 'merged'
    if module.startswith('nrca'):
        return 'nrca'
    if module.startswith('nrcb'):
        return 'nrcb'
    return module


def _policy():
    tree = ast.parse(_DRIVER.read_text(), filename=str(_DRIVER))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (isinstance(target, ast.Name)
                    and target.id == 'MODULES_BY_PROPOSAL_FIELD_FILTER'):
                return ast.literal_eval(node.value)
    raise AssertionError(
        f'MODULES_BY_PROPOSAL_FIELD_FILTER not found in {_DRIVER}')


def test_the_registry_is_literal_enough_for_the_preflight_to_parse():
    policy = _policy()
    assert isinstance(policy, dict) and policy


@pytest.mark.parametrize('proposal,obsid,filtername', [
    ('3958', '007', 'F187N'),
    ('3958', '007', 'F480M'),
    ('2211', '050', 'F200W'),
    ('2211', '050', 'F277W'),
])
def test_single_module_observations_are_registered_as_module_b_only(
        proposal, obsid, filtername):
    entry = _policy()[proposal][obsid][filtername]
    assert entry, f'{proposal}/{obsid} {filtername} has an empty module list'
    assert {_module_family(m) for m in entry} == {'nrcb'}


def test_gc2211_050_is_the_only_restricted_gc2211_observation():
    """Its four siblings carry both modules; restricting them would drop half
    of each mosaic."""
    gc2211 = _policy()['2211']
    assert set(gc2211) == {'050'}


def test_a_module_request_for_gc2211_050_narrows_to_b_and_cannot_ask_for_merged():
    """The two verdicts `get_allowed_modules` reaches on this entry: narrow a
    request that contains nrcb, and raise on one that does not (a `merged`
    mosaic needs an nrca half that this observation does not have)."""
    entry = _policy()['2211']['050']['F200W']
    allowed = {_module_family(m) for m in entry}

    requested = ('nrca', 'nrcb', 'merged')
    assert {m for m in requested if m in allowed} == {'nrcb'}

    for impossible in (('nrca',), ('merged',), ('nrca', 'merged')):
        assert not {m for m in impossible if m in allowed}, impossible
