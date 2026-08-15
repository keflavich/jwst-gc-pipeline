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
    pyflakes_checker = pytest.importorskip('pyflakes.checker')
    pyflakes_messages = pytest.importorskip('pyflakes.messages')

    undefined = []
    for rel in MODULES:
        path = os.path.join(PACKAGE, rel)
        with open(path) as fh:
            tree = ast.parse(fh.read(), filename=path)
        for msg in pyflakes_checker.Checker(tree, filename=path).messages:
            if isinstance(msg, pyflakes_messages.UndefinedName):
                undefined.append(f'{rel}:{msg.lineno} {msg.message % msg.message_args}')

    assert not undefined, (
        'name(s) used where the interpreter cannot resolve them -- a NameError '
        'on whichever input reaches the line:\n  ' + '\n  '.join(undefined))
