"""An unpadded ``--field`` must still find the single-module registry.

``MODULES_BY_PROPOSAL_FIELD_FILTER`` keys an observation with three digits
(``'007'``, ``'050'``), the way MAST and the association names spell it.
``--field`` is taken as typed, so ``7`` and ``007`` name the same observation
and only one of them matched the registry.

The miss is SILENT, which is what makes it worth pinning: ``get_allowed_modules``
leaves ``allowed_modules`` None on a miss and returns the request UNRESTRICTED
with nothing printed, so ``--field 7`` on sickle 3958/007 (module B only) asks
for nrca as well and the reduce goes looking for members that were never taken.
``preflight_reduce_inputs.allowed_modules`` reads the same registry by the same
key and answered the same way, so the ten-second preflight agreed with the
wrong answer.

Both readers are exercised here.  The driver's copy is read by parsing rather
than importing (``PipelineRerunNIRCAM-LONG.py`` has a hyphen and pulls in the
whole JWST stack): the registry literal and the three functions that consume it
are compiled into a private namespace.  Issue #438.
"""
import ast
import importlib.util
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_DRIVER = REPO_ROOT / 'jwst_gc_pipeline' / 'reduction' / 'PipelineRerunNIRCAM-LONG.py'

_WANTED_FUNCS = ('registry_obs_key', '_module_group', 'get_allowed_modules')


def _driver_namespace():
    """The registry + the functions that read it, executed on their own."""
    tree = ast.parse(_DRIVER.read_text(), filename=str(_DRIVER))
    kept = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _WANTED_FUNCS:
            kept.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name)
                and t.id == 'MODULES_BY_PROPOSAL_FIELD_FILTER'
                for t in node.targets):
            kept.append(node)
    names = {n.name for n in kept if isinstance(n, ast.FunctionDef)}
    missing = set(_WANTED_FUNCS) - names
    assert not missing, f'{_DRIVER} no longer defines {sorted(missing)}'
    module = ast.Module(body=kept, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(_DRIVER), 'exec'), namespace)
    assert namespace['MODULES_BY_PROPOSAL_FIELD_FILTER'], 'registry is empty'
    return namespace


_PREFLIGHT_SPEC = importlib.util.spec_from_file_location(
    'preflight_reduce_inputs',
    REPO_ROOT / 'scripts' / 'reduction' / 'preflight_reduce_inputs.py')
preflight = importlib.util.module_from_spec(_PREFLIGHT_SPEC)
_PREFLIGHT_SPEC.loader.exec_module(preflight)


#: (proposal, padded obsid, unpadded spelling, filter) for every registered
#: single-module observation, so a new entry is covered without editing this.
def _registered_observations():
    registry = _driver_namespace()['MODULES_BY_PROPOSAL_FIELD_FILTER']
    for proposal, by_obs in registry.items():
        for obsid, by_filter in by_obs.items():
            for filtername in by_filter:
                yield proposal, obsid, str(int(obsid)), filtername


_CASES = sorted(_registered_observations())


def test_the_registry_keys_observations_with_three_digits():
    """The premise: every key is padded, so an unpadded field cannot match by
    string equality."""
    registry = _driver_namespace()['MODULES_BY_PROPOSAL_FIELD_FILTER']
    for by_obs in registry.values():
        for obsid in by_obs:
            assert obsid == f'{int(obsid):03d}', obsid


@pytest.mark.parametrize('spelling,expected',
                         [('7', '007'), ('07', '007'), ('007', '007'),
                          ('050', '050'), ('50', '050'), (7, '007'),
                          ('  7 ', '007')])
def test_registry_obs_key_pads_a_number(spelling, expected):
    assert _driver_namespace()['registry_obs_key'](spelling) == expected
    assert preflight.registry_obs_key(spelling) == expected


@pytest.mark.parametrize('passthrough', ['*', '001-002', 'brick', ''])
def test_registry_obs_key_leaves_a_non_number_alone(passthrough):
    """The wildcard obsid a not-yet-executed program registers must survive:
    padding it would invent an observation."""
    assert _driver_namespace()['registry_obs_key'](passthrough) == passthrough
    assert preflight.registry_obs_key(passthrough) == passthrough


@pytest.mark.parametrize('proposal,obsid,unpadded,filtername', _CASES)
def test_unpadded_field_narrows_the_modules_the_padded_one_does(
        proposal, obsid, unpadded, filtername):
    """The driver reader.  Both spellings must reach the same entry."""
    namespace = _driver_namespace()
    get_allowed_modules = namespace['get_allowed_modules']
    requested = ['nrca', 'nrcb']
    padded_answer = get_allowed_modules(proposal, obsid, requested,
                                        filtername=filtername)
    assert padded_answer != requested, (
        f'{proposal}/{obsid} {filtername} is registered single-module, so the '
        f'padded spelling must narrow the request')
    assert get_allowed_modules(proposal, unpadded, requested,
                               filtername=filtername) == padded_answer


@pytest.mark.parametrize('proposal,obsid,unpadded,filtername', _CASES)
def test_unpadded_field_refuses_the_module_the_padded_one_refuses(
        proposal, obsid, unpadded, filtername):
    """The loud half: a request naming only the module this observation does
    NOT have raises for both spellings, in both readers.  Under the unpadded
    spelling this used to return the impossible module and reduce toward
    'No nrca members found'."""
    namespace = _driver_namespace()
    get_allowed_modules = namespace['get_allowed_modules']
    module_group = namespace['_module_group']
    entry = namespace['MODULES_BY_PROPOSAL_FIELD_FILTER'][proposal][obsid][filtername]
    families = {module_group(m) for m in entry}
    impossible = sorted({'nrca', 'nrcb'} - families)
    assert impossible, f'{proposal}/{obsid} {filtername} restricts nothing'

    for spelling in (obsid, unpadded):
        with pytest.raises(ValueError):
            get_allowed_modules(proposal, spelling, impossible,
                                filtername=filtername)
        with pytest.raises(preflight.NoAllowedModules):
            preflight.allowed_modules(proposal, spelling, filtername,
                                      set(impossible))


@pytest.mark.parametrize('proposal,obsid,unpadded,filtername', _CASES)
def test_preflight_narrows_on_both_spellings(
        proposal, obsid, unpadded, filtername):
    """The preflight reader, on a request that survives the narrowing."""
    requested = {'nrca', 'nrcb'}
    padded = preflight.allowed_modules(proposal, obsid, filtername, requested)
    assert padded != requested, (
        f'{proposal}/{obsid} {filtername} must narrow in the preflight too')
    assert preflight.allowed_modules(proposal, unpadded, filtername,
                                     requested) == padded


def test_an_unregistered_observation_is_still_unrestricted():
    """Padding must not invent a restriction: a field with no entry keeps the
    full request, which is every field but the registered few."""
    namespace = _driver_namespace()
    requested = ['nrca', 'nrcb']
    assert namespace['get_allowed_modules'](
        '2221', '1', requested, filtername='F212N') == requested
    assert preflight.allowed_modules(
        '2221', '1', 'F212N', set(requested)) == set(requested)
