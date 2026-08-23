"""A name used but never imported is a NameError waiting for the right input.

`psf_preflight` was called at four sites in `crowdsource_catalogs_long.py` and
imported at none.  Nothing caught it:

* the module imports fine -- the failure is at CALL time, not import time;
* #358's own test asserted the call sites exist by reading the SOURCE TEXT
  (`assert 'missing_local_data_message' in src`), which a missing import
  satisfies perfectly;
* the paths that reach those lines need real survey data, so CI never ran them.

It surfaced two days later on the cluster, after the astrometry had already
passed, killing the m12 finalize of two fields::

    NameError: name 'psf_preflight' is not defined
    MergedcatMosaicError: [m12] nrca/F182M: mergedcat residual / model i2d
    mosaic build failed: name 'psf_preflight' is not defined

This checks the whole file rather than that one name: every name LOADED must be
bound somewhere the interpreter can see it.
"""
import ast
import builtins
import os

import pytest

PHOTOMETRY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE = os.path.dirname(PHOTOMETRY)

#: The one file where "bound nowhere in this module" is the intended design
#: rather than a defect: it rebuilds its predecessor's namespace at import time
#: with ``globals().update(vars(crowdsource_catalogs_long))``, so several
#: hundred of its names are bound by that call and by nothing a parser can see.
#: It is frozen legacy code reached only via ``--legacy-iterations``.
DENY = {
    os.path.join('photometry', 'legacy', 'crowdsource_step.py'),
}

#: PEP 695 type-parameter nodes (Python 3.12+).  Looked up by name so this file
#: still imports on an older interpreter, where the names are absent and the
#: syntax that produces them is a SyntaxError anyway.
_TYPE_PARAM_NODES = tuple(
    t for t in (getattr(ast, n, None)
                for n in ('TypeVar', 'ParamSpec', 'TypeVarTuple'))
    if t is not None) or (type(None),)


def _package_modules():
    """Every .py file in the package except the namespace-copying legacy one.

    Started as a six-name allowlist of files whose call sites need survey data.
    That was the wrong axis: needing survey data is what makes a defect
    *survive* CI, and it is not a property of a file that anyone maintains as
    the tree grows.  Two of the three unbound names in #379 were in files
    nobody had thought to list.  Sweeping everything costs ~2 s.
    """
    out = []
    for root, dirs, files in os.walk(PACKAGE):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for name in sorted(files):
            if not name.endswith('.py'):
                continue
            rel = os.path.relpath(os.path.join(root, name), PACKAGE)
            if rel not in DENY:
                out.append(rel)
    return out


MODULES = _package_modules()


def _bound_names(tree):
    """Every name the module binds: imports, assignments, defs, args, globals.

    Deliberately generous -- the point is to find names bound NOWHERE, which is
    an unambiguous defect, not to do real scope analysis.
    """
    # Module-level dunders the interpreter provides; not imports.
    bound = set(dir(builtins)) | {'__file__', '__name__', '__doc__',
                                  '__package__', '__spec__', '__loader__'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                bound.add((a.asname or a.name).split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                bound.add(a.asname or a.name)
        elif isinstance(node, ast.Lambda):
            # lambda parameters are bound too -- `key=lambda f: ...` is the
            # commonest form in this tree and is not a missing import.
            for a in (list(node.args.args) + list(node.args.posonlyargs)
                      + list(node.args.kwonlyargs)):
                bound.add(a.arg)
            for a in (node.args.vararg, node.args.kwarg):
                if a is not None:
                    bound.add(a.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            bound.add(node.name)
            args = getattr(node, 'args', None)
            if args is not None:
                for a in (list(args.args) + list(args.posonlyargs)
                          + list(args.kwonlyargs)):
                    bound.add(a.arg)
                for a in (args.vararg, args.kwarg):
                    if a is not None:
                        bound.add(a.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            # `case [a, b]:`, `case {'k': v}:`, `case str() as s:` and
            # `case [*rest]:` bind through a plain string ATTRIBUTE rather than
            # an `ast.Name(Store)`, so the branch above cannot see them.  The
            # package has no `match` statement today; under the old six-file
            # allowlist that was negligible, and sweeping all 278 files makes
            # the first one anyone writes a spurious CI failure (#405 item 4).
            bound.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            # `case {**rest}:` -- the same binding, under another attribute.
            bound.add(node.rest)
        elif isinstance(node, _TYPE_PARAM_NODES) and node.name:
            # PEP 695: `def f[T](x: T) -> T:` and `class C[T]:` bind `T` on an
            # `ast.TypeVar`/`ParamSpec`/`TypeVarTuple`, again by string
            # attribute.  Same exposure as `match`.
            bound.add(node.name)
        elif isinstance(node, ast.Global):
            bound.update(node.names)
        elif isinstance(node, ast.alias):
            bound.add(node.asname or node.name.split('.')[0])
    return bound


def _loaded_names(tree):
    return {n.id: n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


@pytest.mark.parametrize('filename', MODULES)
def test_every_name_used_is_bound_somewhere(filename):
    path = os.path.join(PACKAGE, filename)
    if not os.path.exists(path):
        pytest.skip(f'{filename} not present')
    with open(path) as fh:
        tree = ast.parse(fh.read(), filename=path)
    bound = _bound_names(tree)
    unbound = {name: line for name, line in _loaded_names(tree).items()
               if name not in bound}
    assert not unbound, (
        f'{filename}: name(s) used but bound nowhere in the module -- a '
        f'NameError on whichever input reaches them: '
        + ', '.join(f'{n} (line {ln})' for n, ln in sorted(unbound.items())))


def test_the_regression_this_exists_for():
    """`psf_preflight` is used in crowdsource_catalogs_long and must be imported.

    Pinned by name because a generic check is easy to weaken later, and this
    particular one cost two fields their m12 finalize after their astrometry
    had already passed.
    """
    path = os.path.join(PHOTOMETRY, 'crowdsource_catalogs_long.py')
    with open(path) as fh:
        tree = ast.parse(fh.read(), filename=path)
    used = 'psf_preflight' in _loaded_names(tree)
    assert used, 'call sites gone -- delete this test with them'
    assert 'psf_preflight' in _bound_names(tree), (
        'psf_preflight is called but never imported')


def _undefined_message_types(pyflakes_messages):
    """Every pyflakes message class that means "this line raises NameError".

    Filtering on ``UndefinedName`` alone drops two of the three:

    ``UndefinedLocal``
        "local variable 'x' defined in enclosing scope ... referenced before
        assignment" -- the closure form of the same defect.
    ``UndefinedExport``
        ``__all__`` naming a symbol the module does not define, which raises
        ``AttributeError`` on ``from module import *``.

    The package has zero of either today, so widening changes no verdict now.
    It changes what the sweep can claim: with one class the assertion text
    ("used where the interpreter cannot resolve them") described coverage the
    filter did not have (#405 item 3).
    """
    out = [pyflakes_messages.UndefinedName]
    for name in ('UndefinedLocal', 'UndefinedExport'):
        cls = getattr(pyflakes_messages, name, None)
        if cls is not None:
            out.append(cls)
    return tuple(out)


def _scope_undefined(source, label):
    """The NameError-class pyflakes findings in one source string.

    Factored out of the sweep so the class filter can be pinned against
    synthetic sources: the tree has no ``UndefinedLocal`` and no
    ``UndefinedExport``, so a sweep over the real files cannot tell a widened
    filter from a narrow one.
    """
    from pyflakes import checker as pyflakes_checker
    from pyflakes import messages as pyflakes_messages
    tree = ast.parse(source, filename=label)
    types = _undefined_message_types(pyflakes_messages)
    return [f'{label}:{msg.lineno} {msg.message % msg.message_args}'
            for msg in pyflakes_checker.Checker(tree, filename=label).messages
            if isinstance(msg, types)]


def test_no_name_is_used_outside_the_scope_that_binds_it():
    """The check above pools every binding in a file, so it cannot see scope.

    That blind spot cost a real run.  ``make_reference_from_pipeline_catalogs``
    ended ``main()`` with ``print(f"... {len(vvv)} rows")``.  ``vvv`` is bound
    inside ``fetch_vvv_catalog()`` and nowhere else, so the pooled-binding
    check sees it as bound and the interpreter does not: every complete run of
    that script raised ``NameError`` on that line, after all of its outputs had
    already been written, and exited non-zero on a run that had succeeded.

    ``pyflakes`` does per-scope analysis, so it catches that case as well as
    the bound-nowhere ones.  It is in the ``test`` extra; skipped rather than
    failed when absent, so the pooled check above still runs without it.
    """
    pytest.importorskip('pyflakes.checker')
    pytest.importorskip('pyflakes.messages')

    undefined = []
    for rel in MODULES:
        path = os.path.join(PACKAGE, rel)
        with open(path) as fh:
            undefined.extend(_scope_undefined(fh.read(), rel))

    assert not undefined, (
        'name(s) used where the interpreter cannot resolve them -- a NameError '
        'on whichever input reaches the line:\n  ' + '\n  '.join(undefined))


# ---------------------------------------------------------------------------
# what the two checks above CAN see -- pinned against synthetic sources,
# because the package contains no instance of any of these constructs, so the
# sweeps themselves pass either way (issue #405 items 3 and 4)
# ---------------------------------------------------------------------------

MATCH_AND_TYPE_PARAM_SOURCE = '''
def dispatch(x):
    match x:
        case [a, b]:
            return a + b
        case {"k": v}:
            return v
        case {"j": 1, **leftover}:
            return leftover
        case [1, *rest]:
            return rest
        case str() as s:
            return s
    return None


def identity[T](x: T) -> T:
    return x


class Box[U]:
    def get(self) -> U:
        raise NotImplementedError
'''


def test_match_captures_and_type_parameters_count_as_BOUND():
    """`case [a, b]:` and `def f[T](...)` bind by string attribute.

    Neither goes through `ast.Name(Store)`, so the pooled binding check used to
    report every one of them as "used but bound nowhere" -- a spurious CI
    failure on the first `match` statement or PEP 695 signature anyone writes.
    The package has none today, so nothing but this test can tell.
    """
    tree = ast.parse(MATCH_AND_TYPE_PARAM_SOURCE)
    bound = _bound_names(tree)
    unbound = {n: ln for n, ln in _loaded_names(tree).items() if n not in bound}
    assert not unbound, (
        'match captures / PEP 695 type parameters reported as unbound: '
        + ', '.join(f'{n} (line {ln})' for n, ln in sorted(unbound.items())))
    for name in ('a', 'b', 'v', 'leftover', 'rest', 's', 'T', 'U'):
        assert name in bound, f'{name} is bound by the source but not collected'


def test_the_pooled_sweep_passes_on_the_match_and_type_parameter_source(tmp_path):
    """End to end through the sweep's own assertion, not just the helper."""
    path = tmp_path / 'usesmatch.py'
    path.write_text(MATCH_AND_TYPE_PARAM_SOURCE)
    tree = ast.parse(path.read_text(), filename=str(path))
    unbound = {n: ln for n, ln in _loaded_names(tree).items()
               if n not in _bound_names(tree)}
    assert not unbound


def test_the_scope_check_sees_referenced_before_assignment():
    """pyflakes calls the closure form `UndefinedLocal`, a different class.

    Filtering on `UndefinedName` alone let it through, while the assertion text
    claimed every name "used where the interpreter cannot resolve them".
    """
    pytest.importorskip('pyflakes.checker')
    src = (
        'def outer():\n'
        '    x = 1\n'
        '    def inner():\n'
        '        print(x)\n'
        '        x = 2\n'
        '    return inner\n')
    found = _scope_undefined(src, 'synthetic.py')
    assert found, 'referenced-before-assignment was filtered out'
    assert any('referenced before assignment' in f for f in found), found


def test_the_scope_check_sees_an___all___naming_a_missing_symbol():
    """`__all__ = ['nope']` is `UndefinedExport`, also a different class.

    It raises `AttributeError` on `from module import *` rather than at import,
    so nothing else in the suite would catch it.
    """
    pytest.importorskip('pyflakes.checker')
    found = _scope_undefined("__all__ = ['nope']\n", 'synthetic.py')
    assert found, "an __all__ entry bound nowhere was filtered out"
    assert any('nope' in f for f in found), found


def test_a_clean_source_produces_no_scope_finding():
    """The widened filter must not start reporting healthy code.

    `UnusedVariable`, `UnusedImport` and friends are pyflakes messages too, and
    the tree has plenty of them; only the NameError classes belong here.
    """
    pytest.importorskip('pyflakes.checker')
    src = (
        'import os\n'
        '\n'
        '__all__ = ["where"]\n'
        '\n'
        '\n'
        'def where():\n'
        '    unused = 1\n'
        '    return os.getcwd()\n')
    assert _scope_undefined(src, 'synthetic.py') == []


def test_pyflakes_is_pinned_to_the_major_version_whose_api_this_uses():
    """This file uses `pyflakes.checker.Checker` and `msg.message_args`.

    Neither is a documented API, so an unbounded `pyflakes` in the test extra
    can turn a green gate into an ImportError or an AttributeError on an
    unrelated upgrade.  The bound belongs in `pyproject.toml` (#405 item 5).
    """
    pyproject = os.path.join(os.path.dirname(PACKAGE), 'pyproject.toml')
    assert os.path.exists(pyproject), f'no pyproject.toml at {pyproject}'
    with open(pyproject) as fh:
        text = fh.read()
    assert '"pyflakes"' not in text, (
        'pyflakes is unpinned in pyproject.toml; this file depends on its '
        'undocumented Checker API')
    assert 'pyflakes>=3,<4' in text, (
        'pyflakes must carry a major-version bound in pyproject.toml')
