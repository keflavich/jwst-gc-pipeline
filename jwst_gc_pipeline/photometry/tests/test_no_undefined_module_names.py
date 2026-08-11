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

#: Files whose call sites need survey data, so an unbound name in them cannot
#: be caught by running the code in CI.  These are exactly the ones that need a
#: static check.
MODULES = [
    'crowdsource_catalogs_long.py',
    'cataloging.py',
    'merge_catalogs.py',
    'astrometry_checkpoint.py',
    'visit_consensus.py',
    'psf_preflight.py',
]


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
    path = os.path.join(PHOTOMETRY, filename)
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
