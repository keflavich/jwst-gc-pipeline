"""``--field`` is padded at the reduce driver's entry points, once.

#514 padded the value where the single-module registry is read, which closed
the silent half of issue #438.  Everything else in the driver still built names
from the raw string: the association search
(``jw02221-o{field}*_image3_*asn.json``), the uncal download filter
(``jw02221{field}*_uncal.fits``), the drizzle product names
(``...-o{field}_t001_nircam_clear-...``) and the ``_o{field}_crf`` frames.  MAST
and the products spell an observation with three digits, so ``--field 1`` built
``jw02221-o1*``, matched nothing, and the run stopped at "Did not find any
NIRCam asn files" -- a loud stop for a reason that is not the real one -- while
the CLI raised ``KeyError: '1'`` from ``field_to_reg_mapping`` before ``main``
was ever entered.

The fix normalises at the two entry points (``main``'s first statement, and the
CLI's ``fields`` split), so every name below is built from the canonical
spelling.  This pins the placement: a normalisation that happens after the first
use of ``field`` leaves the names it was supposed to fix, which is the state
this file exists to prevent.

``PipelineRerunNIRCAM-LONG.py`` has a hyphen and imports the whole JWST stack,
and ``main`` cannot be run in a test, so the driver is read by parsing --
the same approach as ``test_registry_obs_key_padding.py``.
"""
import ast
import fnmatch
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DRIVER = REPO_ROOT / 'jwst_gc_pipeline' / 'reduction' / 'PipelineRerunNIRCAM-LONG.py'

_SOURCE = DRIVER.read_text()
_TREE = ast.parse(_SOURCE, filename=str(DRIVER))


def _function(name):
    for node in ast.walk(_TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f'{DRIVER} no longer defines {name}()')


def _registry_obs_key():
    """The driver's own padding helper, compiled on its own."""
    node = _function('registry_obs_key')
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(DRIVER), 'exec'), namespace)
    return namespace['registry_obs_key']


def _normalising_assignments(body, target='field'):
    """``<target> = ...registry_obs_key(...)...`` statements in ``body``."""
    found = []
    for node in ast.walk(ast.Module(body=list(body), type_ignores=[])):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if target not in targets:
            continue
        calls = [c for c in ast.walk(node.value)
                 if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                 and c.func.id == 'registry_obs_key']
        if calls:
            found.append(node)
    return found


def test_main_normalises_field_before_it_uses_it():
    """The pin.  Every product name and glob in ``main`` is interpolated from
    ``field``, so the padding has to precede the first of them."""
    main = _function('main')
    assignments = _normalising_assignments(main.body)
    assert assignments, (
        "main() does not pad `field`: every glob and product name below is "
        "built from the raw --field, so `--field 1` writes jw*-o1* names")
    normalise = min(assignments, key=lambda n: n.lineno)

    uses = [n.lineno for n in ast.walk(main)
            if isinstance(n, ast.Name) and n.id == 'field'
            and not (normalise.lineno <= n.lineno <= (normalise.end_lineno
                                                      or normalise.lineno))]
    assert uses, 'main() no longer uses `field`; this test is stale'
    assert min(uses) > normalise.lineno, (
        f'main() reads `field` at line {min(uses)}, before the padding at line '
        f'{normalise.lineno}: the names built above it keep the raw spelling')


def test_the_cli_pads_every_comma_separated_field():
    """``field_to_reg_mapping`` and ``get_allowed_modules`` both key on the
    padded spelling, and the CLI reaches them before ``main``."""
    assigns = [n for n in ast.walk(_TREE)
               if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'fields'
                       for t in n.targets)]
    assert assigns, "the CLI no longer builds `fields`; this test is stale"
    assert any(_normalising_assignments([n], target='fields') for n in assigns), (
        "the CLI splits --field without padding the parts, so `--field 1` "
        "raises KeyError: '1' against a registry holding that observation")


@pytest.mark.parametrize('spelling', ['1', '01', '001'])
def test_padding_is_what_makes_the_association_glob_match(spelling):
    """The consequence, on the two patterns the issue quotes.  The padded
    spelling matches a real association / uncal name; the raw one does not."""
    field = _registry_obs_key()(spelling)
    asn = 'jw02221-o001_t001_nircam_clear-f212n-nrca_image3_00001_asn.json'
    uncal = 'jw02221001001_02101_00001_nrca1_uncal.fits'
    assert fnmatch.fnmatch(asn, f'jw02221-o{field}*_image3_*_asn.json')
    assert fnmatch.fnmatch(uncal, f'jw02221{field}*_uncal.fits')
    if spelling != '001':
        assert not fnmatch.fnmatch(asn, f'jw02221-o{spelling}*_image3_*_asn.json')
        assert not fnmatch.fnmatch(uncal, f'jw02221{spelling}*_uncal.fits')


@pytest.mark.parametrize('passthrough', ['*', '002-998'])
def test_a_non_number_field_reaches_the_names_unchanged(passthrough):
    """Padding must not rewrite a joint registration ('002-998', sgrb2's MIRI
    pair) or the wildcard obsid a not-yet-executed program registers."""
    assert _registry_obs_key()(passthrough) == passthrough
