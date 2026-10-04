"""The AUTO seed switches reach every call site.

``_phase_seed_switches`` resolves the own-band m7 seed and the loose seed
roundness window for one band.  ``run_manual_pipeline`` and
``annotate_independent_detection`` read both switches from it.
``run_manual_pipeline`` is not driven by any test, so the wiring tests read
its source: a call site that ignores the switch, forces it, or resolves it
without the band's instrument would otherwise pass every resolver test.
"""
import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest

from jwst_gc_pipeline.photometry import cataloging
from jwst_gc_pipeline.photometry.cataloging import _phase_seed_switches


@pytest.fixture(autouse=True)
def _no_instrument_override(monkeypatch):
    # the filter-name heuristic decides MIRI in these tests
    monkeypatch.delenv('GC_INSTRUMENT_OVERRIDE', raising=False)


@pytest.mark.parametrize('target,filt,expected', [
    # star-dominated NIRCam fields: both on
    ('brick', 'F212N', (True, 0.8)),
    ('sgrb2', 'F182M', (True, 0.8)),
    ('sgra', 'F405N', (True, 0.8)),
    # extended-emission targets: both off
    ('w51', 'F187N', (False, 0.0)),
    ('sickle', 'F212N', (False, 0.0)),
    ('wd2', 'F444W', (False, 0.0)),
    ('ngc6334', 'F200W', (False, 0.0)),
    # a MIRI band on a star field: both off
    ('brick', 'F770W', (False, 0.0)),
    ('cloudc', 'F1130W', (False, 0.0)),
])
def test_defaults_per_target_and_band(target, filt, expected):
    assert _phase_seed_switches(SimpleNamespace(target=target), filt) == expected


def test_extended_emission_flag_overrides_target():
    on = SimpleNamespace(target='brick', extended_emission=True)
    off = SimpleNamespace(target='w51', extended_emission=False)
    assert _phase_seed_switches(on, 'F212N') == (False, 0.0)
    assert _phase_seed_switches(off, 'F187N') == (True, 0.8)
    # MIRI stays off with --no-extended-emission
    assert _phase_seed_switches(
        SimpleNamespace(target='brick', extended_emission=False), 'F770W') == (False, 0.0)


def test_explicit_values_used_verbatim():
    opts = dict(manual_m7_seed_own_band=True, manual_seed_round_loose_max=0.6)
    for target, filt in (('w51', 'F187N'), ('brick', 'F770W'), ('brick', 'F212N')):
        assert _phase_seed_switches(SimpleNamespace(target=target, **opts), filt) == (True, 0.6)
    opts = dict(manual_m7_seed_own_band=False, manual_seed_round_loose_max=0.0)
    assert _phase_seed_switches(SimpleNamespace(target='brick', **opts), 'F212N') == (False, 0.0)


def _tree(func):
    return ast.parse(textwrap.dedent(inspect.getsource(func)))


def _calls(tree, name):
    return [c for c in ast.walk(tree) if isinstance(c, ast.Call)
            and getattr(c.func, 'id', None) == name]


def _switch_targets(tree, options_name):
    """The names the one ``_phase_seed_switches(<options>, filt)`` call binds."""
    assigns = [a for a in ast.walk(tree) if isinstance(a, ast.Assign)
               and isinstance(a.value, ast.Call)
               and getattr(a.value.func, 'id', None) == '_phase_seed_switches']
    assert len(assigns) == 1
    assert len(_calls(tree, '_phase_seed_switches')) == 1
    call = assigns[0].value
    assert [getattr(a, 'id', None) for a in call.args] == [options_name, 'filt']
    assert not call.keywords
    (target,) = assigns[0].targets
    assert isinstance(target, ast.Tuple) and len(target.elts) == 2
    return [getattr(e, 'id', None) for e in target.elts]


def _assigned_names(tree):
    return [t.id for a in ast.walk(tree) if isinstance(a, (ast.Assign, ast.AugAssign))
            for t in ast.walk(a.targets[0] if isinstance(a, ast.Assign) else a.target)
            if isinstance(t, ast.Name)]


@pytest.mark.parametrize('func', [cataloging.run_manual_pipeline,
                                  cataloging.annotate_independent_detection])
def test_no_call_site_resolves_a_switch_on_its_own(func):
    tree = _tree(func)
    assert not _calls(tree, '_auto_m7_seed_own_band')
    assert not _calls(tree, '_auto_seed_round_loose_max')


def test_run_manual_pipeline_gates_m7_own_band_on_the_switch():
    tree = _tree(cataloging.run_manual_pipeline)
    own, loose = _switch_targets(tree, 'opts_phase')
    assert _assigned_names(tree).count(own) == 1
    # the switch alone decides whether the m7 band seed is built
    gates = [n for n in ast.walk(tree) if isinstance(n, ast.If)
             and _calls(ast.Module(body=n.body, type_ignores=[]), '_build_m7_band_seed')
             and not _calls(ast.Module(body=n.orelse, type_ignores=[]), '_build_m7_band_seed')]
    switch_gates = [g for g in gates
                    if isinstance(g.test, ast.Name) and g.test.id == own]
    assert len(switch_gates) == 1
    # and no other test on the way to the build mentions a constant or the switch
    outer = [g for g in gates if g is not switch_gates[0]]
    for g in outer:
        assert not any(isinstance(n, ast.Constant) and isinstance(n.value, bool)
                       for n in ast.walk(g.test)), ast.unparse(g.test)


def test_run_manual_pipeline_passes_the_loose_switch_unchanged():
    tree = _tree(cataloging.run_manual_pipeline)
    own, loose = _switch_targets(tree, 'opts_phase')
    assert _assigned_names(tree).count(loose) == 1
    (call,) = _calls(tree, '_build_i2d_augmented_seed')
    kw = {k.arg: k.value for k in call.keywords}
    assert isinstance(kw['round_loose_max'], ast.Name)
    assert kw['round_loose_max'].id == loose


def test_annotate_counts_m7_detections_only_when_the_switch_is_on():
    tree = _tree(cataloging.annotate_independent_detection)
    own, _ = _switch_targets(tree, 'options')
    assert _assigned_names(tree).count(own) == 1
    loops = [f for f in ast.walk(tree) if isinstance(f, ast.For)
             and _calls(ast.Module(body=f.body, type_ignores=[]), 'm7_band_seed_path')]
    # the innermost one: the per-module loop over the m7 band seeds
    loops = [f for f in loops
             if not any(g is not f and isinstance(g, ast.For) and g in loops
                        for g in ast.walk(f))]
    assert len(loops) == 1
    it = loops[0].iter
    assert isinstance(it, ast.IfExp)
    assert isinstance(it.test, ast.Name) and it.test.id == own
    assert isinstance(it.orelse, ast.List) and not it.orelse.elts
