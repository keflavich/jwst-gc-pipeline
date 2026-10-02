"""Satstar PSF seed uses the float centre of mass, not the rounded pixel (#1042).

``get_saturated_stars`` rounds the saturated-mask centre of mass to
``xcen``/``ycen`` to place the cutout window.  It also seeded the PSF model and
centred the position bounds on those integers, so with
``NIRCAM_SATSTAR_LOCK_POS=1`` (flux-only fit) the model sat up to 0.7 px off
the star.  On wd2 the locked fits lay a median 0.4-0.45 px from the dolphot
positions and the flux bias grew with that offset; seeding at the float
``xf``/``yf`` halved the offset and the cross-frame position scatter.

The PSF fitting in ``get_saturated_stars`` needs PSF grids and a full frame,
so these are call-site guards on the source.
"""
import ast
import inspect
import textwrap

import pytest

from jwst_gc_pipeline.reduction import saturated_star_finding as ssf


def _assigned(tree, name):
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
            and any(getattr(t, 'id', None) == name for t in n.targets)]


def _names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


@pytest.fixture(scope='module')
def tree():
    return ast.parse(textwrap.dedent(inspect.getsource(ssf.get_saturated_stars)))


@pytest.mark.parametrize('name', ['x_init', 'y_init',
                                  'low_x', 'high_x', 'low_y', 'high_y'])
def test_seed_and_bounds_do_not_use_the_rounded_centre(tree, name):
    values = _assigned(tree, name)
    assert values, name
    for v in values:
        assert not _names(v) & {'xcen', 'ycen'}, ast.unparse(v)


def test_seed_uses_the_float_centre_of_mass(tree):
    for name, var in (('x_init', 'xf'), ('y_init', 'yf')):
        assert all(var in _names(v) for v in _assigned(tree, name)), name


def test_bounds_are_centred_on_the_seed(tree):
    for name, var in (('low_x', 'x_init'), ('high_x', 'x_init'),
                      ('low_y', 'y_init'), ('high_y', 'y_init')):
        assert all(var in _names(v) for v in _assigned(tree, name)), name


# Sky lookups of the star position: flux-drop and flux-override matching
# (``_wpos_drop``, ``_wpos``), the MIRI coadd-coverage check (``_scov``) and
# the coadd seed gate (``_skyc``).  These must also use the float centre.
@pytest.mark.parametrize('name', ['_wpos_drop', '_wpos', '_scov', '_skyc'])
def test_sky_lookups_do_not_use_the_rounded_centre(tree, name):
    values = _assigned(tree, name)
    assert values, name
    for v in values:
        assert not _names(v) & {'xcen', 'ycen'}, ast.unparse(v)


@pytest.mark.parametrize('name', ['_wpos_drop', '_wpos', '_scov'])
def test_sky_lookups_use_the_float_centre(tree, name):
    for v in _assigned(tree, name):
        assert {'xf', 'yf'} <= _names(v), ast.unparse(v)


if __name__ == '__main__':
    pytest.main([__file__])
