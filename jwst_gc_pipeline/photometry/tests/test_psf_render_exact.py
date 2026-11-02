"""The grid=True stamp path renders the same bits as GriddedPSFModel.evaluate.

`_render_model_from_table` evaluates each stamp through
`psf_render.stamp_function`.  For a gridded PSF that is a tensor-product
(``grid=True``) spline evaluation of the stamp's row and column vectors, which
must reproduce ``evaluate`` on the stamp's meshgrid exactly -- the rendered
model is subtracted to make the released residual mosaic.
"""
import numpy as np
import pytest
from astropy.nddata import NDData
from astropy.table import Table
from photutils.psf import GriddedPSFModel

from jwst_gc_pipeline.photometry import crowdsource_catalogs_long as ccl
from jwst_gc_pipeline.photometry import psf_render
from jwst_gc_pipeline.photometry.psf_fitting import CachingGriddedPSFModel

SHAPE = (160, 180)


def _nddata(n=3, oversampling=2):
    """An n x n grid of elliptical Gaussians whose widths change across the
    frame, so the bilinear interpolation between ePSFs matters.  The ePSF
    footprint is +-12.5 px, inside a 31 x 31 stamp, so the fill_value branch
    is exercised too."""
    half = 25
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1] / oversampling
    data, xy = [], []
    for j, gy in enumerate(np.linspace(0, SHAPE[0] - 1, n)):
        for i, gx in enumerate(np.linspace(0, SHAPE[1] - 1, n)):
            sx, sy = 1.0 + 0.35 * i, 1.2 + 0.25 * j
            psf = np.exp(-0.5 * ((xx / sx) ** 2 + (yy / sy) ** 2))
            data.append(psf / psf.sum())
            xy.append((float(gx), float(gy)))
    return NDData(np.array(data),
                  meta={'grid_xypos': xy, 'oversampling': oversampling})


def _grid(cls=GriddedPSFModel, n=3, fill_value=0.0):
    return cls(_nddata(n), fill_value=fill_value)


def _stars(n=300, seed=7):
    """Positions inside, at and beyond the frame edges (beyond the ePSF grid
    too, so the bounding-point clamp is used), mixed-sign fluxes, and rows the
    renderer must skip."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-12, SHAPE[1] + 12, n)
    y = rng.uniform(-12, SHAPE[0] + 12, n)
    f = rng.lognormal(5, 2, n) * np.where(rng.uniform(size=n) < 0.1, -1, 1)
    x[:3] = [np.nan, 20.0, 30.0]
    f[1] = np.inf
    y[2] = -np.inf
    return Table({'x_fit': x, 'y_fit': y, 'flux_fit': f})


def _render(grid, psf_shape, monkeypatch, grid_path):
    monkeypatch.setenv('PSF_RENDER_GRID', '1' if grid_path else '0')
    return ccl._render_model_from_table(_stars(), grid, SHAPE, psf_shape)


@pytest.mark.parametrize('cls', [GriddedPSFModel, CachingGriddedPSFModel])
@pytest.mark.parametrize('n', [1, 2, 3])
@pytest.mark.parametrize('psf_shape', [(21, 21), (31, 31), (20, 25)])
@pytest.mark.parametrize('fill_value', [0.0, 1e-3])
def test_the_grid_path_renders_the_same_bits(monkeypatch, cls, n, psf_shape,
                                             fill_value):
    grid = _grid(cls, n=n, fill_value=fill_value)
    fast = _render(grid, psf_shape, monkeypatch, True)
    ref = _render(grid, psf_shape, monkeypatch, False)
    assert fast.dtype == ref.dtype == np.float32
    assert np.count_nonzero(ref) > 0
    assert np.array_equal(fast.view(np.uint32), ref.view(np.uint32))


def test_the_grid_path_is_taken(monkeypatch):
    """evaluate() runs once per render (the first-stamp check), not per star."""
    grid = _grid()
    calls = []
    real = GriddedPSFModel.evaluate

    def _count(self, *a, **k):
        calls.append(1)
        return real(self, *a, **k)

    monkeypatch.setattr(GriddedPSFModel, 'evaluate', _count)
    monkeypatch.setattr(psf_render.GriddedPSFModel, 'evaluate', _count)
    monkeypatch.setenv('PSF_RENDER_GRID', '1')
    ccl._render_model_from_table(_stars(), grid, SHAPE, (21, 21))
    assert len(calls) == 1
    calls.clear()
    monkeypatch.setenv('PSF_RENDER_GRID', '0')
    ccl._render_model_from_table(_stars(), grid, SHAPE, (21, 21))
    assert len(calls) > 250


def test_a_model_with_its_own_evaluate_is_rendered_by_it(monkeypatch):
    class Doubled(GriddedPSFModel):
        def evaluate(self, x, y, flux, x_0, y_0):
            return 2.0 * super().evaluate(x, y, flux, x_0, y_0)

    monkeypatch.setenv('PSF_RENDER_GRID', '1')
    doubled = Doubled(_nddata())
    assert not isinstance(psf_render.stamp_function(doubled),
                          psf_render._GridStamp)
    plain = ccl._render_model_from_table(_stars(), _grid(), SHAPE, (21, 21))
    twice = ccl._render_model_from_table(_stars(), doubled, SHAPE, (21, 21))
    np.testing.assert_allclose(twice, 2.0 * plain, rtol=1e-6)


def test_a_first_stamp_that_differs_falls_back_to_evaluate(monkeypatch, capsys):
    grid = _grid()
    real_fast = psf_render._GridStamp._fast

    def _off(self, *a, **k):
        return real_fast(self, *a, **k) * (1 + 1e-12)

    monkeypatch.setattr(psf_render._GridStamp, '_fast', _off)
    fast = _render(grid, (21, 21), monkeypatch, True)
    assert "differs from evaluate() on the first stamp" in capsys.readouterr().out
    ref = _render(grid, (21, 21), monkeypatch, False)
    assert np.array_equal(fast.view(np.uint32), ref.view(np.uint32))


def test_a_grid_path_that_raises_falls_back_to_evaluate(monkeypatch, capsys):
    grid = _grid()

    def _gone(self, *a, **k):
        raise AttributeError("'GriddedPSFModel' object has no attribute "
                             "'_find_bounding_points'")

    monkeypatch.setattr(psf_render._GridStamp, '_grid', _gone)
    fast = _render(grid, (21, 21), monkeypatch, True)
    assert "raised AttributeError" in capsys.readouterr().out
    ref = _render(grid, (21, 21), monkeypatch, False)
    assert np.array_equal(fast.view(np.uint32), ref.view(np.uint32))


def test_missing_photutils_internals_use_evaluate(monkeypatch):
    monkeypatch.setenv('PSF_RENDER_GRID', '1')
    monkeypatch.setattr(psf_render, '_INTERNALS',
                        psf_render._INTERNALS + ('_not_in_this_photutils',))
    assert not isinstance(psf_render.stamp_function(_grid()),
                          psf_render._GridStamp)


@pytest.mark.parametrize('cls', [GriddedPSFModel, CachingGriddedPSFModel])
@pytest.mark.parametrize('n', [1, 2, 3])
@pytest.mark.parametrize('fill_value', [0.0, 1e-3, None])
def test_every_grid_stamp_is_the_evaluate_stamp(cls, n, fill_value):
    """`_fast` against `evaluate`, star by star.  A render checks only its
    first stamp and falls back to `evaluate` on a difference, which hides a
    regression in the grid math from the render-level tests above.  Stamps
    inside the ePSF footprint (21 x 21), past it (31 x 31, 20 x 25 against its
    +-12.5 px), and off the frame."""
    grid = _grid(cls, n=n, fill_value=fill_value)
    fast = psf_render._GridStamp(grid, cls is CachingGriddedPSFModel)
    ref = psf_render._meshgrid_stamp(grid)
    rng = np.random.default_rng(11)
    for sy, sx in [(21, 21), (31, 31), (20, 25)]:
        for _ in range(40):
            x0 = rng.uniform(-12, SHAPE[1] + 12)
            y0 = rng.uniform(-12, SHAPE[0] + 12)
            flux = rng.lognormal(5, 2) * rng.choice([-1, 1])
            x_lo = int(np.floor(x0)) - sx // 2
            y_lo = int(np.floor(y0)) - sy // 2
            args = (x_lo, x_lo + sx, y_lo, y_lo + sy, flux, x0, y0)
            got, want = fast._fast(*args), ref(*args)
            assert got.shape == want.shape == (sy, sx)
            assert np.array_equal(got, want, equal_nan=True), args
