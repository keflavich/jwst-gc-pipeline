"""``--field`` is padded at the MIRI and NIRISS drivers' entry points too.

#528 normalised ``--field`` in ``PipelineRerunNIRCAM-LONG.py``; the other two
reduce drivers interpolate the raw value into exactly the same shapes and were
left out of it (issue #438).  For MIRI those are the uncal download filter
(``jw{PPPPP}{field}*_uncal.fits``), the association search
(``jw{PPPPP}-o{field}*_image3_*asn.json``), the drizzle product name
(``...-o{field}_t001_miri_{filt}``), the ``_o{field}_crf`` frames and the
per-exposure i2d regeneration glob; for NIRISS the same set with
``_t001_niriss_``.  Both drivers also compare ``field`` against a three-digit
literal in their region sanity checks (``assert field == '002'`` for brick's
MIRI, ``assert field == '012'`` for the one registered NIRISS observation), so
an unpadded value stops the run against a registry that holds the observation
it names.

The pin is PLACEMENT, not existence: a padding that lands after the first read
of ``field`` leaves precisely the names it was meant to fix.  Both drivers pull
in the JWST stack at import and ``main`` cannot be run in a test, so they are
read by parsing -- the approach ``test_field_normalised_at_entry.py`` uses for
the NIRCam driver.
"""
import ast
import fnmatch
import pathlib

import pytest

from jwst_gc_pipeline.mast_names import jw_prefix
from jwst_gc_pipeline.reduction.mast_obs_scope import observation_number

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_REDUCTION = REPO_ROOT / 'jwst_gc_pipeline' / 'reduction'

#: driver -> the product token its level-3 names carry, for the glob check.
DRIVERS = {
    'PipelineMIRI.py': 'miri',
    'PipelineRerunNIRISS.py': 'niriss',
}

#: The shared helper both drivers must route the value through.  Named rather
#: than "any call", so a driver that re-implements padding locally -- the state
#: this file exists to prevent a third copy of -- does not pass.
_PAD = 'observation_number'


def _tree(driver):
    path = _REDUCTION / driver
    return path, ast.parse(path.read_text(), filename=str(path))


def _function(tree, path, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f'{path} no longer defines {name}()')


def _padding_assignments(body, target):
    """``<target> = ...observation_number(...)...`` statements in ``body``."""
    found = []
    for node in ast.walk(ast.Module(body=list(body), type_ignores=[])):
        if not isinstance(node, ast.Assign):
            continue
        if target not in [t.id for t in node.targets if isinstance(t, ast.Name)]:
            continue
        if any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
               and c.func.id == _PAD for c in ast.walk(node.value)):
            found.append(node)
    return found


@pytest.mark.parametrize('driver', sorted(DRIVERS))
def test_the_driver_imports_the_shared_padding_helper(driver):
    """One rule, one home.  A local re-implementation is how the NIRCam driver
    and ``mast_obs_scope`` came to hold the same three lines twice."""
    path, tree = _tree(driver)
    imported = any(
        isinstance(node, ast.ImportFrom)
        and node.module == 'jwst_gc_pipeline.reduction.mast_obs_scope'
        and any(alias.name == _PAD for alias in node.names)
        for node in ast.walk(tree))
    assert imported, (
        f'{path.name} does not import {_PAD} from '
        f'jwst_gc_pipeline.reduction.mast_obs_scope')


@pytest.mark.parametrize('driver', sorted(DRIVERS))
def test_main_pads_field_before_it_reads_it(driver):
    """The pin.  Every glob, product name and region assert in ``main`` is
    built from ``field``, so the padding has to precede the first of them."""
    path, tree = _tree(driver)
    main = _function(tree, path, 'main')
    assignments = _padding_assignments(main.body, 'field')
    assert assignments, (
        f'{path.name}: main() does not pad `field`, so every glob and product '
        f'name below it keeps the raw --field spelling')
    pad = min(assignments, key=lambda n: n.lineno)

    reads = [n.lineno for n in ast.walk(main)
             if isinstance(n, ast.Name) and n.id == 'field'
             and not (pad.lineno <= n.lineno <= (pad.end_lineno or pad.lineno))]
    assert reads, f'{path.name}: main() no longer uses `field`; this test is stale'
    assert min(reads) > pad.lineno, (
        f'{path.name}: main() reads `field` at line {min(reads)}, before the '
        f'padding at line {pad.lineno}; the names built above keep the raw '
        f'spelling')


@pytest.mark.parametrize('driver', sorted(DRIVERS))
def test_the_cli_pads_every_comma_separated_field(driver):
    """``field_to_reg_mapping[field]`` keys on the three-digit spelling and the
    CLI reaches it before ``main``, so an unpadded part raises there first."""
    path, tree = _tree(driver)
    assigns = [n for n in ast.walk(tree)
               if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'fields'
                       for t in n.targets)]
    assert assigns, f'{path.name}: the CLI no longer builds `fields`; test is stale'
    assert any(_padding_assignments([n], 'fields') for n in assigns), (
        f'{path.name}: the CLI splits --field without padding the parts, so '
        f"`--field 2` raises KeyError: '2' against a registry holding that "
        f'observation')


@pytest.mark.parametrize('spelling,expected', [('2', '002'), ('02', '002'),
                                               ('002', '002'), ('12', '012')])
def test_padding_is_what_makes_the_names_match(spelling, expected):
    """The consequence, on real MIRI and NIRISS names.  The padded spelling
    matches; the raw one does not.

    The prefix comes from ``jw_prefix``, the helper the two drivers themselves
    build these globs with, so the patterns tested here are the ones the
    drivers form and this file carries no ``jw0``-plus-proposal spelling of
    its own -- the 4-digit assumption issue #414's grep guard bans.
    """
    field = observation_number(spelling)
    assert field == expected
    for proposal_id, token, filtername, detector in (
            ('2221', 'miri', 'f2550w', 'mirimage'),
            ('4147', 'niriss', 'f150w', 'nis')):
        prefix = jw_prefix(proposal_id)
        asn = (f'{prefix}-o{expected}_t001_{token}_{filtername}'
               f'_image3_00001_asn.json')
        uncal = f'{prefix}{expected}001_02101_00001_{detector}_uncal.fits'
        assert fnmatch.fnmatch(asn, f'{prefix}-o{field}*_image3_*asn.json')
        assert fnmatch.fnmatch(uncal, f'{prefix}{field}*_uncal.fits')
        if spelling != expected:
            assert not fnmatch.fnmatch(
                asn, f'{prefix}-o{spelling}*_image3_*asn.json')
            assert not fnmatch.fnmatch(
                uncal, f'{prefix}{spelling}*_uncal.fits')


@pytest.mark.parametrize('passthrough', ['*', '002-998', '001-002'])
def test_a_joint_registration_reaches_the_names_unchanged(passthrough):
    """MIRI is where the joint spellings live -- sgrb2's '002-998' and
    sickle's '001-002' -- so the padding must hand a non-number back."""
    assert observation_number(passthrough) == passthrough
