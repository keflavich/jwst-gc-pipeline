"""The epsf_map full-field residual script renders with the pipeline renderer.

``scripts/analysis/epsf_map/fullfield_residual.py`` takes
``_render_model_from_table`` out of ``crowdsource_catalogs_long`` and execs it
on its own, because the host module needs the jwst package at import time.
The exec namespace has to hold every global the function uses.  Since #1220
the function calls ``psf_render.stamp_function``; with only ``np`` in the
namespace the script's first render raised ``NameError``.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.nddata import NDData
from astropy.table import Table
from photutils.psf import GriddedPSFModel

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "analysis" / "epsf_map" / "fullfield_residual.py")

ONE_STAR = Table({"x_fit": [20.3], "y_fit": [15.6], "flux_fit": [1000.0]})
SHAPE = (40, 50)
PSF_SHAPE = (21, 21)


@pytest.fixture
def script_renderer(monkeypatch):
    """The renderer exactly as the script builds it.  The script puts its
    own directory on sys.path and imports its sibling ``epsf``; both are
    undone afterwards."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    had_epsf = "epsf" in sys.modules
    spec = importlib.util.spec_from_file_location("_fullfield_residual", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    yield mod.load_pipeline_renderer()
    if not had_epsf:
        sys.modules.pop("epsf", None)


class _Gaussian:
    """A model with only ``evaluate``, like the script's SpatialPSF:
    stamp_function evaluates it on the stamp's meshgrid."""

    sigma = 1.5

    def evaluate(self, x, y, flux, x_0, y_0):
        r2 = (x - x_0) ** 2 + (y - y_0) ** 2
        return flux * np.exp(-0.5 * r2 / self.sigma ** 2) / (2 * np.pi * self.sigma ** 2)


def _gridded():
    """A one-ePSF photutils grid: stamp_function takes the grid=True path."""
    yy, xx = np.mgrid[-25:26, -25:26] / 2
    psf = np.exp(-0.5 * (xx ** 2 + yy ** 2) / 1.5 ** 2)
    return GriddedPSFModel(NDData(psf[None] / psf.sum(),
                                  meta={"grid_xypos": [(20.0, 20.0)],
                                        "oversampling": 2}))


@pytest.mark.parametrize("model", [_Gaussian(), _gridded()],
                         ids=["meshgrid", "gridded"])
def test_the_script_renders_one_star(script_renderer, model):
    img = script_renderer(ONE_STAR, model, SHAPE, PSF_SHAPE)
    assert img.dtype == np.float32 and img.shape == SHAPE
    assert np.unravel_index(np.argmax(img), SHAPE) == (16, 20)
    assert np.count_nonzero(img) == PSF_SHAPE[0] * PSF_SHAPE[1]


def test_the_meshgrid_star_carries_its_flux(script_renderer):
    img = script_renderer(ONE_STAR, _Gaussian(), SHAPE, PSF_SHAPE)
    assert img.sum() == pytest.approx(1000.0, rel=1e-4)


@pytest.mark.parametrize("model", [_Gaussian(), _gridded()],
                         ids=["meshgrid", "gridded"])
def test_the_script_copy_matches_the_pipeline_function(script_renderer, model):
    from jwst_gc_pipeline.photometry import crowdsource_catalogs_long as ccl
    ref = ccl._render_model_from_table(ONE_STAR, model, SHAPE, PSF_SHAPE)
    img = script_renderer(ONE_STAR, model, SHAPE, PSF_SHAPE)
    assert np.array_equal(img.view(np.uint32), ref.view(np.uint32))
