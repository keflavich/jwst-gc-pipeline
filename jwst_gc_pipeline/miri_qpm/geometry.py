"""Static pane geometry: a pane pixel (x, y) sees the sky the cal WCS assigns to (x + dx, y + dy).

dx, dy are a per-pane affine function of the detector position, fitted on pane stars of every GC F770W frame
(data/F770W/geometry_F770W.json; evaluated exactly as fullframe_preview.region_shift).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def region_shift(models, r, x, y):
    cx, cy = models[r]
    A = np.c_[np.ones(len(x)), (x - 500) / 500, (y - 500) / 500]
    return A @ cx, A @ cy


def pane_positions(ref):
    """Detector coordinates (float, full-frame 0-based) of every crop pixel, with the static pane shift applied:
    arrays X, Y of the crop shape.  Pane pixels take their pane's shift; every other pixel takes the shift of the
    nearest pane (the affine model extrapolated), so the drizzle pixmap is continuous across pane edges.  Drizzle's
    square kernel builds each pixel's corners from its neighbours' positions: an unshifted neighbour 3-4 px away
    squeezes or folds the footprint of the outermost pane pixel (negative weights, lost output pixels)."""
    ny, nx = ref.region.shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    xx += ref.x0
    yy += ref.y0
    X, Y = xx.copy(), yy.copy()
    iy, ix = ndimage.distance_transform_edt(ref.region == 0, return_distances=False, return_indices=True)
    nearest = ref.region[iy, ix]
    for g in (1, 2, 3):
        m = nearest == g
        dx, dy = region_shift(ref.models, g, xx[m], yy[m])
        X[m] += dx
        Y[m] += dy
    return X, Y


def apply_sim(p, c, X, Y):
    """Similarity transform in reference-mosaic pixels (registration); identity for p = (1, 0, 0, 0)."""
    u, v = X - c[0], Y - c[1]
    return p[0] * u - p[1] * v + c[0] + p[2], p[1] * u + p[0] * v + c[1] + p[3]


def det_to_ref(wcs, rw, X, Y, p=(1.0, 0.0, 0.0, 0.0), c=(0.0, 0.0)):
    """Detector (x, y) -> reference-mosaic pixel (x, y), through the cal WCS and the registration."""
    px, py = rw.world_to_pixel_values(*wcs(X, Y, with_bounding_box=False))
    return apply_sim(p, c, np.asarray(px), np.asarray(py))
