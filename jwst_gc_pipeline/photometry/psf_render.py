"""Exact, fast evaluation of a gridded PSF on a rectangular render stamp.

``_render_model_from_table`` (the mergedcat residual/model render of the m7
finalize) places one stamp per catalog row and evaluates it with
``GriddedPSFModel.evaluate`` on the stamp's meshgrid.  ``evaluate`` hands
every pixel of that meshgrid to the bounding ePSF splines POINTWISE
(``RectBivariateSpline(..., grid=False)``, FITPACK ``bispeu``).  A stamp is a
rectangle, so the same splines can be evaluated on its column and row
VECTORS instead (``grid=True``, FITPACK ``bispev``).  Both routines run the
same per-point kernel (``fpbisp``); the grid routine computes the B-spline
basis once per column and once per row instead of once per pixel.  The
values are the same bits.

The sgrb2 m7 finalize spent 22.5 h of its 72 h wall in this loop
(0.83 ms per star).  On sgrb2's F182M and F405N grids the grid path renders
the same stars 4.4x and 2.8x faster, into a bitwise-identical image (#1220).

`stamp_function` returns the grid path only for the two ``evaluate``
implementations it reproduces (`GriddedPSFModel` and
`CachingGriddedPSFModel`), and only when the photutils internals it calls
exist.  Every grid-path evaluator compares its first stamp with
``evaluate`` and, on any difference, uses ``evaluate`` for the rest of the
render.  ``PSF_RENDER_GRID=0`` turns the grid path off.
"""
import os

import numpy as np
from photutils.psf import GriddedPSFModel

#: photutils ``GriddedPSFModel`` internals the grid path calls, so that the
#: bounding-point choice, bilinear weights and spline construction are
#: photutils' own.
_INTERNALS = ('_calc_interpolator', '_find_bounding_points',
              '_calc_bilinear_weights')


def _meshgrid_stamp(psf_model):
    """The reference: ``psf_model.evaluate`` on the stamp's meshgrid."""
    def stamp(x_lo, x_hi, y_lo, y_hi, flux, x_0, y_0):
        yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
        return psf_model.evaluate(xx.astype(float), yy.astype(float),
                                  flux, x_0, y_0)
    return stamp


class _GridStamp:
    """``GriddedPSFModel.evaluate`` on a rectangle, through ``grid=True``.

    Mirrors ``evaluate`` operation for operation: the same sample
    coordinates, the same bounding ePSFs and weights (photutils' own
    helpers), the same summation order over the nonzero weights, flux applied
    to the sum, then ``fill_value`` outside the ePSF footprint.  For a
    `CachingGriddedPSFModel` it renders the unit stamp and scales it by
    ``float(flux)`` afterwards, as that class's ``evaluate`` does.
    """

    def __init__(self, psf_model, caching):
        self._model = psf_model
        self._caching = caching
        self._reference = _meshgrid_stamp(psf_model)
        self._checked = False
        self._fallback = False

    def _grid(self, x_lo, x_hi, y_lo, y_hi, flux, x_0, y_0):
        m = self._model
        xi = m.oversampling[1] * (np.arange(x_lo, x_hi, dtype=float) - x_0)
        yi = m.oversampling[0] * (np.arange(y_lo, y_hi, dtype=float) - y_0)
        xi += m.origin[0]
        yi += m.origin[1]
        # RectBivariateSpline holds the ePSF in (x, y) order, so grid=True
        # returns [x, y]; the stamp is [y, x].
        if m.data.shape[0] == 1:
            values = flux * m._calc_interpolator(0)(xi, yi, grid=True).T
        else:
            grid_idx, grid_xy = m._find_bounding_points(x_0, y_0)
            weights = m._calc_bilinear_weights(x_0, y_0, grid_xy)
            result = 0
            for gidx, weight in zip(grid_idx, weights):
                if weight != 0:
                    result += m._calc_interpolator(gidx)(xi, yi,
                                                         grid=True) * weight
            values = flux * result.T
        if m.fill_value is not None:
            ny, nx = m.data.shape[1:]
            bad_x = (xi < 0) | (xi > nx - 1)
            bad_y = (yi < 0) | (yi > ny - 1)
            if bad_x.any() or bad_y.any():
                values[bad_y[:, None] | bad_x[None, :]] = m.fill_value
        return values

    def _fast(self, x_lo, x_hi, y_lo, y_hi, flux, x_0, y_0):
        if self._caching:
            return float(flux) * self._grid(x_lo, x_hi, y_lo, y_hi, 1.0,
                                            x_0, y_0)
        return self._grid(x_lo, x_hi, y_lo, y_hi, flux, x_0, y_0)

    def __call__(self, x_lo, x_hi, y_lo, y_hi, flux, x_0, y_0):
        if self._fallback:
            return self._reference(x_lo, x_hi, y_lo, y_hi, flux, x_0, y_0)
        if self._checked:
            return self._fast(x_lo, x_hi, y_lo, y_hi, flux, x_0, y_0)
        self._checked = True
        reference = self._reference(x_lo, x_hi, y_lo, y_hi, flux, x_0, y_0)
        why = None
        try:
            stamp = self._fast(x_lo, x_hi, y_lo, y_hi, flux, x_0, y_0)
        except (AttributeError, TypeError, ValueError, IndexError) as ex:
            why = f"raised {type(ex).__name__}: {ex}"
        else:
            if not (np.shape(stamp) == np.shape(reference)
                    and np.array_equal(stamp, reference, equal_nan=True)):
                why = "differs from evaluate() on the first stamp"
        if why is not None:
            self._fallback = True
            print(f"psf_render: the grid=True stamp path for "
                  f"{type(self._model).__name__} {why}; rendering with "
                  f"evaluate() instead", flush=True)
            return reference
        return stamp


def stamp_function(psf_model):
    """``f(x_lo, x_hi, y_lo, y_hi, flux, x_0, y_0)`` -> the model on rows
    ``y_lo:y_hi`` and columns ``x_lo:x_hi`` ([y, x]), exactly as
    ``psf_model.evaluate`` gives it on their meshgrid.

    The grid path for a `GriddedPSFModel` or `CachingGriddedPSFModel` (and
    subclasses that keep their ``evaluate``); ``evaluate`` on the meshgrid for
    any other model, when photutils lacks an internal the grid path calls, or
    with ``PSF_RENDER_GRID=0``.
    """
    reference = _meshgrid_stamp(psf_model)
    if os.environ.get('PSF_RENDER_GRID', '1') == '0':
        return reference
    from jwst_gc_pipeline.photometry.psf_fitting import CachingGriddedPSFModel
    evaluate = type(psf_model).evaluate
    if evaluate is GriddedPSFModel.evaluate:
        caching = False
    elif evaluate is CachingGriddedPSFModel.evaluate:
        caching = True
    else:
        return reference
    if not all(hasattr(psf_model, name) for name in _INTERNALS):
        return reference
    return _GridStamp(psf_model, caching)
