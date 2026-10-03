"""Satstar fits evaluate the PSF grid at the star's DETECTOR position (#1055).

``get_saturated_stars`` fits each saturated star on ``data[y0:y1, x0:x1]``
with ``x_0``/``y_0`` in cutout pixels.  ``GriddedPSFModel`` picks and
interpolates its nodes from ``x_0``/``y_0``, and the grids from ``get_psf``
have nodes at detector positions (0.5 ... 2047.5).  Without the cutout
origin, every star got the PSF of detector pixel ~(81, 81), the lower-left
node, wherever it really was (a 2-3% flux bias on a real F410M grid; 3-6% on
F480M per the issue).

A one-node grid cannot see this bug, so the grid here has four nodes whose
PSFs have very different widths.  The full ``get_saturated_stars`` needs real
PSF grids and a full frame, so its call sites are checked on the source, as in
``test_satstar_float_seed.py``.
"""
import ast
import inspect
import textwrap

import numpy as np
import pytest
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.nddata import NDData
from astropy.table import QTable
from photutils.psf import GriddedPSFModel, PSFPhotometry

from jwst_gc_pipeline.reduction import saturated_star_finding as ssf

NODES = (0.5, 2047.5)
# node sigma (px): narrow at the lower-left node, wide everywhere else
SIGMAS = {(0.5, 0.5): 1.2, (2047.5, 0.5): 3.0,
          (0.5, 2047.5): 3.0, (2047.5, 2047.5): 3.0}


def _grid(oversampling=2, half=40):
    n = 2 * half * oversampling + 1
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    data, xypos = [], []
    for y in NODES:
        for x in NODES:
            s = SIGMAS[(x, y)] * oversampling
            g = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * s ** 2))
            data.append(g / g.sum() * oversampling ** 2)
            xypos.append((x, y))
    return GriddedPSFModel(NDData(np.array(data),
                                  meta={'grid_xypos': xypos,
                                        'oversampling': oversampling}))


@pytest.fixture(scope='module')
def grid():
    return _grid()


# a star far from the lower-left node, and its production cutout (pad = 81)
X, Y, PAD = 1700.3, 1500.6, 81
X0, Y0 = int(round(X)) - PAD, int(round(Y)) - PAD


def _cutout_pixels():
    return np.mgrid[0:2 * PAD, 0:2 * PAD].astype(float)


def test_node_psfs_differ(grid):
    """Guard on the test itself: cutout-coordinate evaluation must be wrong."""
    yy, xx = _cutout_pixels()
    det = grid.evaluate(xx + X0, yy + Y0, 1.0, X, Y)
    cut = grid.evaluate(xx, yy, 1.0, X - X0, Y - Y0)
    assert np.max(np.abs(det - cut)) > 0.1 * det.max()


def test_cutout_view_equals_detector_evaluation(grid):
    yy, xx = _cutout_pixels()
    det = grid.evaluate(xx + X0, yy + Y0, 1.0, X, Y)
    view = ssf.psf_in_cutout_coords(grid, X0, Y0)
    np.testing.assert_allclose(view.evaluate(xx, yy, 1.0, X - X0, Y - Y0),
                               det, rtol=1e-12, atol=1e-15)
    # the same through Model.__call__ with parameters set (satstar_implied_peak)
    m = view.copy()
    m.flux, m.x_0, m.y_0 = 1.0, X - X0, Y - Y0
    np.testing.assert_allclose(m(xx, yy), det, rtol=1e-12, atol=1e-15)


def test_cutout_view_shares_the_grid(grid):
    view = ssf.psf_in_cutout_coords(grid, X0, Y0)
    assert view.data is grid.data
    assert isinstance(view, GriddedPSFModel)
    # copies keep the origin (PSFPhotometry and satstar_implied_peak copy)
    assert (view.copy()._cutout_xoff, view.copy()._cutout_yoff) == (X0, Y0)
    # the original grid is untouched
    yy, xx = _cutout_pixels()
    np.testing.assert_array_equal(grid.evaluate(xx, yy, 1.0, 5.0, 5.0),
                                  _grid().evaluate(xx, yy, 1.0, 5.0, 5.0))


def test_nested_views_add_their_origins(grid):
    yy, xx = _cutout_pixels()
    nested = ssf.psf_in_cutout_coords(ssf.psf_in_cutout_coords(grid, 1000, 900),
                                      X0 - 1000, Y0 - 900)
    np.testing.assert_allclose(nested.evaluate(xx, yy, 1.0, X - X0, Y - Y0),
                               grid.evaluate(xx + X0, yy + Y0, 1.0, X, Y),
                               rtol=1e-12, atol=1e-15)


def test_non_gridded_models_pass_through():
    from photutils.psf import CircularGaussianPRF
    m = CircularGaussianPRF(fwhm=2.0)
    assert ssf.psf_in_cutout_coords(m, 100, 200) is m


def _masked_core_fit(model, data, mask):
    init = QTable()
    init['x'] = [X - X0]
    init['y'] = [Y - Y0]
    phot = PSFPhotometry(psf_model=model, fit_shape=41, aperture_radius=10,
                         fitter=LevMarLSQFitter())
    return phot(data, init_params=init, mask=mask)


def test_masked_core_fit_recovers_flux_at_detector_position(grid):
    """The production fit: a saturated (masked) core in a cutout far from the
    lower-left node.  The cutout view recovers the flux; the bare grid does
    not."""
    flux = 1.0e6
    yy, xx = _cutout_pixels()
    data = grid.evaluate(xx + X0, yy + Y0, flux, X, Y)
    mask = (xx - (X - X0)) ** 2 + (yy - (Y - Y0)) ** 2 < 3.0 ** 2

    res = _masked_core_fit(ssf.psf_in_cutout_coords(grid, X0, Y0), data, mask)
    assert float(res['flux_fit'][0]) == pytest.approx(flux, rel=1e-4)
    assert float(res['x_fit'][0]) + X0 == pytest.approx(X, abs=1e-3)
    assert float(res['y_fit'][0]) + Y0 == pytest.approx(Y, abs=1e-3)

    bare = _masked_core_fit(grid, data, mask)
    assert abs(float(bare['flux_fit'][0]) / flux - 1) > 0.05


# ---- call sites in get_saturated_stars -------------------------------------

@pytest.fixture(scope='module')
def tree():
    return ast.parse(textwrap.dedent(inspect.getsource(ssf.get_saturated_stars)))


def _assigned(tree, name):
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
            and any(getattr(t, 'id', None) == name for t in n.targets)]


def _calls_view(node):
    return any(isinstance(n, ast.Call)
               and getattr(n.func, 'id', None) == 'psf_in_cutout_coords'
               and len(n.args) == 3
               and [getattr(a, 'id', None) for a in n.args[1:]] == ['x0', 'y0']
               for n in ast.walk(node))


@pytest.mark.parametrize('name', ['_forced_psf'])
def test_forced_psf_is_a_cutout_view(tree, name):
    values = _assigned(tree, name)
    assert values, name
    assert all(_calls_view(v) for v in values), name


def test_infov_fit_psf_is_a_cutout_view(tree):
    """``_infov_psf`` is the fit, model-image, envelope and halo-hook PSF.  The
    only non-view assignment allowed is the per-source default before the
    cutout exists."""
    values = _assigned(tree, '_infov_psf')
    views = [v for v in values if _calls_view(v)]
    assert views, 'in-FOV PSF is never the cutout view'
    for v in values:
        if not _calls_view(v):
            assert isinstance(v, ast.Name) and v.id == 'big_grid', ast.unparse(v)
    # the PSFPhotometry model is the view
    fits = [kw.value for n in ast.walk(tree) if isinstance(n, ast.Call)
            and getattr(n.func, 'id', None) == 'PSFPhotometry'
            for kw in n.keywords if kw.arg == 'psf_model']
    assert fits and all(getattr(v, 'id', None) == '_psf_for_fit' for v in fits)
    psf_for_fit = [v for v in _assigned(tree, '_psf_for_fit')
                   if not (isinstance(v, ast.Name) and v.id == 'big_grid')]
    assert psf_for_fit and all(getattr(v, 'id', None) == '_infov_psf'
                               for v in psf_for_fit)


PSF_NAMES = {'big_grid', 'big_grid_large', '_infov_psf', '_psf_for_fit',
             '_psf_for_model', '_forced_psf'}


def test_no_psf_is_called_with_shifted_coordinates(tree):
    """``psf(xx - x_fit, yy - y_fit)`` leaves ``x_0`` at its default, so the
    grid picks the node at detector (0, 0).  Every evaluation goes through
    ``.evaluate(x, y, flux, x_0, y_0)`` on a cutout view instead."""
    bad = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Call)
           and isinstance(n.func, ast.Name) and n.func.id in PSF_NAMES]
    assert not bad, bad


def test_evaluations_use_a_cutout_view(tree):
    """``.evaluate`` is only called on the cutout views, never on the bare
    detector grids."""
    bad = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Call)
           and isinstance(n.func, ast.Attribute) and n.func.attr == 'evaluate'
           and getattr(n.func.value, 'id', None) in {'big_grid', 'big_grid_large'}]
    assert not bad, bad


# ---- no double correction: every PSF is read ONCE at the detector position --
# A cutout view given DETECTOR coordinates adds the origin a second time; the
# bare grid given CUTOUT coordinates adds it zero times (#1055).  Both read
# the wrong node.  The halo-mode hooks are the two sites that take different
# coordinate systems, so each is pinned to its pairing.

def _calls(tree, name):
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and getattr(n.func, 'id', None) == name]


def _names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def test_halo_mode_hook_reads_the_cutout_view_in_cutout_coordinates(tree):
    calls = _calls(tree, 'satstar_halo_mode_ratios')
    assert calls
    for c in calls:
        data, _, _, psf, pos = c.args[:5]
        assert getattr(data, 'id', None) == 'cutout_fit', ast.unparse(c)
        assert getattr(psf, 'id', None) == '_psf_for_fit', ast.unparse(c)
        assert "result['x_fit']" in ast.unparse(pos) and "result['y_fit']" in ast.unparse(pos)


def test_wide_halo_mode_hook_reads_the_bare_grid_in_detector_coordinates(tree):
    calls = _calls(tree, 'satstar_halo_mode_ratios_wide')
    assert calls
    for c in calls:
        data, _, _, psf, pos = c.args[:5]
        assert getattr(data, 'id', None) == 'data', ast.unparse(c)
        # only the bare grids: no view, no call that could build one
        assert not any(isinstance(n, ast.Call) for n in ast.walk(psf)), ast.unparse(psf)
        assert 'big_grid' in _names(psf), ast.unparse(psf)
        assert _names(psf) <= {'big_grid', 'big_grid_large', '_use_large_infov'}, ast.unparse(psf)
        assert {'x_centroid', 'y_centroid'} <= _names(pos), ast.unparse(pos)
    # ... and x_centroid / y_centroid are the fitted cutout position plus the origin
    src = {ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Assign)}
    assert "result['xcentroid'] = result['x_fit'] + x0" in src
    assert "result['ycentroid'] = result['y_fit'] + y0" in src
    assert "x_centroid = np.asarray(result['xcentroid'], dtype=float)" in src
    assert "y_centroid = np.asarray(result['ycentroid'], dtype=float)" in src


def test_wide_halo_ratios_refuse_a_cutout_view(grid):
    from jwst_gc_pipeline.photometry.satstar_halo_modes import satstar_halo_mode_ratios_wide
    frame = np.zeros((2048, 2048))
    err = np.ones_like(frame)
    dq = np.zeros(frame.shape, np.uint32)
    for view in (ssf.psf_in_cutout_coords(grid, X0, Y0),
                 ssf.psf_in_cutout_coords(grid, 0, 0),
                 ssf.psf_in_cutout_coords(ssf.psf_in_cutout_coords(grid, X0, Y0), 5, 5)):
        with pytest.raises(TypeError, match='cutout origin twice'):
            satstar_halo_mode_ratios_wide(frame, err, dq, view, [(X, Y)], 400.0)
    # the bare grid is accepted
    satstar_halo_mode_ratios_wide(frame, err, dq, grid, [(X, Y)], 400.0)


def test_implied_peak_reads_the_grid_once_at_the_detector_position(grid):
    """The implied-peak gate copies ``_psf_for_fit``: the copy keeps the
    origin, so the view at cutout coordinates is the grid at the detector
    position, and either wrong pairing is not."""
    view = ssf.psf_in_cutout_coords(grid, X0, Y0)
    assert view.copy()._cutout_xoff == X0 and view.copy()._cutout_yoff == Y0
    truth = float(grid.evaluate(np.array([X]), np.array([Y]), 1.0, X, Y)[0])
    once = ssf.satstar_implied_peak(1.0, view, X - X0, Y - Y0)
    twice = ssf.satstar_implied_peak(1.0, view, X, Y)
    never = ssf.satstar_implied_peak(1.0, grid, X - X0, Y - Y0)
    assert once == pytest.approx(truth, rel=1e-9)
    assert abs(twice / truth - 1) > 0.1 and abs(never / truth - 1) > 0.1


if __name__ == '__main__':
    pytest.main([__file__])
