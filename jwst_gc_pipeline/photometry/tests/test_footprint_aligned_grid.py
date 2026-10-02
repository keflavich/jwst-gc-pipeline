"""Footprint-aligned region-map grid (issue #989).

gc-treasury o084 F212N's m2 checkpoint blocked on one region-map cell (ix=3,
iy=1): 46 matched pairs against a tile median of 236, residual 57.3 mas.  That
cell is the RIGHT-EDGE COLUMN of ``local_residual_map``'s plain RA/Dec-axis
grid, which anchors at min(ra)/min(dec) and steps by a FIXED 45" -- so unless
a footprint's extent happens to be an exact multiple of 45", the last
row/column is whatever remainder is left over, not a full cell.  Every sparse
flagged cell across the fleet fell in that same edge column (ix at its max),
which is the tell that this is a BINNING artifact, not a spatial one: o084's
own consensus catalog measures 145.3" x 356.1" against 45" cells (3.23 x 7.91
cells), so the last column is a ~10" sliver that still collects whatever stars
the rotated mosaic corner puts inside it.

``footprint_aligned_grid`` fixes this by fitting the minimum-bounding-
rectangle angle of the footprint itself (rotating calipers over its convex
hull) and sizing each axis so ``round(extent / cell_arcsec)`` cells divide the
extent EVENLY -- no remainder, at any rotation.  These tests hold the grid's
own geometry (rotation recovery, even division, graceful fallback) separately
from ``test_same_star_region_map.py``'s behavioural tests (seam still caught,
low-coverage never relaxes a gate).
"""
import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord

from jwst_gc_pipeline.photometry.astrometry_offsets import (
    footprint_aligned_grid, assign_footprint_cells, _footprint_cell_center_radec,
    _min_area_rotation_deg, _rotate_xy, _inverse_rotate_uv, _tangent_plane_xy,
    REGION_FOOTPRINT_ALIGN_MIN_POINTS)

RA0, DEC0 = 266.60, -28.50
COSD = float(np.cos(np.radians(DEC0)))


def _sky(x_arcsec, y_arcsec):
    return SkyCoord((RA0 + np.asarray(x_arcsec) / 3600.0 / COSD) * u.deg,
                    (DEC0 + np.asarray(y_arcsec) / 3600.0) * u.deg, frame="icrs")


def _rotated_rectangle(width, height, pa_deg, n=2000, seed=0):
    """A uniformly-filled rectangle (arcsec, in a local tangent frame) of the
    given size, rotated by ``pa_deg`` from the RA/Dec axes."""
    rng = np.random.RandomState(seed)
    u_ = (rng.rand(n) - 0.5) * width
    v_ = (rng.rand(n) - 0.5) * height
    # _rotate_xy rotates (x, y) BY -pa into (u, v); to place a rectangle at a
    # KNOWN pa in (x, y), apply the inverse.
    x, y = _inverse_rotate_uv(u_, v_, pa_deg)
    return x, y


def test_rotation_recovers_a_known_position_angle():
    """A rectangle deliberately tilted 27 deg from the RA/Dec axes: the fitted
    grid angle must land at 27 (mod 90, since a rectangle's bounding box is
    ambiguous by a quarter turn)."""
    x, y = _rotated_rectangle(200.0, 80.0, pa_deg=27.0, n=3000, seed=1)
    coords = _sky(x, y)
    grid = footprint_aligned_grid(coords, cell_arcsec=45.0)
    assert grid["ok"] is True, grid["reason"]
    resid = min(abs(grid["pa_deg"] - 27.0) % 90.0,
               90.0 - abs(grid["pa_deg"] - 27.0) % 90.0)
    assert resid < 2.0, grid


def test_axis_aligned_rectangle_fits_near_zero_or_ninety():
    x, y = _rotated_rectangle(200.0, 80.0, pa_deg=0.0, n=3000, seed=2)
    coords = _sky(x, y)
    grid = footprint_aligned_grid(coords, cell_arcsec=45.0)
    assert grid["ok"] is True
    resid = min(grid["pa_deg"] % 90.0, 90.0 - grid["pa_deg"] % 90.0)
    assert resid < 2.0, grid


def test_cell_size_divides_the_footprint_evenly_no_remainder():
    """The whole point of the fix: whatever the fitted extent, the returned
    per-axis cell size times its cell count reproduces that extent exactly --
    there is no leftover sliver, unlike the plain
    ``floor((pos - min) / fixed_cell)`` grid it replaces."""
    x, y = _rotated_rectangle(151.0, 362.0, pa_deg=41.0, n=4000, seed=3)
    coords = _sky(x, y)
    grid = footprint_aligned_grid(coords, cell_arcsec=45.0)
    assert grid["ok"] is True
    ra = np.asarray(coords.ra.deg, dtype=float)
    dec = np.asarray(coords.dec.deg, dtype=float)
    xt, yt = _tangent_plane_xy(ra, dec, grid["ra0"], grid["dec0"])
    ut, vt = _rotate_xy(xt, yt, grid["pa_deg"])
    extent_u = ut.max() - ut.min()
    extent_v = vt.max() - vt.min()
    assert grid["cell_u_arcsec"] * grid["n_u"] == pytest.approx(extent_u, abs=1e-6)
    assert grid["cell_v_arcsec"] * grid["n_v"] == pytest.approx(extent_v, abs=1e-6)
    # and every point's assigned cell is in-bounds -- no index falls in a
    # remainder column that does not exist any more
    ix, iy = assign_footprint_cells(ra, dec, grid)
    assert ix.min() >= 0 and ix.max() < grid["n_u"]
    assert iy.min() >= 0 and iy.max() < grid["n_v"]


def test_o084_like_extent_has_no_sliver_edge_column():
    """Reproduces the o084 F212N defect directly: a footprint whose extent
    (145" x 356") is nowhere near a multiple of 45" is exactly the case the
    plain grid slivers and the aligned grid does not.  Assert the aligned
    grid's per-cell population has no outlier-thin column, unlike the plain
    ``floor`` grid built the old way alongside it for comparison."""
    x, y = _rotated_rectangle(145.3, 356.1, pa_deg=1.0, n=6000, seed=4)
    coords = _sky(x, y)
    ra = np.asarray(coords.ra.deg, dtype=float)
    dec = np.asarray(coords.dec.deg, dtype=float)

    # the OLD plain grid this replaces: fixed 45" steps from the min corner
    dec_mid = float(np.median(dec))
    cell_deg_dec = 45.0 / 3600.0
    cell_deg_ra = 45.0 / 3600.0 / max(np.cos(np.radians(dec_mid)), 1e-6)
    r0, d0 = float(ra.min()), float(dec.min())
    old_ix = np.floor((ra - r0) / cell_deg_ra).astype(int)
    old_iy = np.floor((dec - d0) / cell_deg_dec).astype(int)
    old_counts = {}
    for a_, b_ in zip(old_ix, old_iy):
        old_counts[(a_, b_)] = old_counts.get((a_, b_), 0) + 1
    old_edge_col = max(old_ix)
    old_edge_counts = [n for (cx, cy), n in old_counts.items() if cx == old_edge_col]
    old_full_median = float(np.median([n for (cx, cy), n in old_counts.items()
                                       if cx != old_edge_col]))
    # the plain grid's edge column is thin relative to the field median --
    # reproducing the defect this fix targets.
    assert min(old_edge_counts) < 0.3 * old_full_median, (old_edge_counts, old_full_median)

    # the ALIGNED grid: every cell (edge or not) is comparable, because there
    # is no remainder column left to be thin.
    grid = footprint_aligned_grid(coords, cell_arcsec=45.0)
    assert grid["ok"] is True, grid["reason"]
    new_ix, new_iy = assign_footprint_cells(ra, dec, grid)
    new_counts = {}
    for a_, b_ in zip(new_ix, new_iy):
        new_counts[(a_, b_)] = new_counts.get((a_, b_), 0) + 1
    counts = np.array(list(new_counts.values()), dtype=float)
    med = float(np.median(counts))
    assert counts.min() >= 0.5 * med, (sorted(new_counts.items()), med)


def test_too_few_points_falls_back_with_a_reason():
    n = REGION_FOOTPRINT_ALIGN_MIN_POINTS - 1
    x, y = _rotated_rectangle(100.0, 100.0, pa_deg=10.0, n=n, seed=5)
    coords = _sky(x, y)
    grid = footprint_aligned_grid(coords, cell_arcsec=45.0)
    assert grid["ok"] is False
    assert grid["reason"]
    assert grid["n_points"] == n


def test_cell_center_roundtrips_through_assign_and_back():
    x, y = _rotated_rectangle(200.0, 200.0, pa_deg=33.0, n=3000, seed=6)
    coords = _sky(x, y)
    grid = footprint_aligned_grid(coords, cell_arcsec=45.0)
    assert grid["ok"] is True
    ra = np.asarray(coords.ra.deg, dtype=float)
    dec = np.asarray(coords.dec.deg, dtype=float)
    ix, iy = assign_footprint_cells(ra, dec, grid)
    cra, cdec = _footprint_cell_center_radec(ix, iy, grid)
    # a star's cell center sits within one cell diagonal of the star itself
    dra = (ra - cra) * COSD * 3600.0
    ddec = (dec - cdec) * 3600.0
    sep = np.hypot(dra, ddec)
    max_diag = np.hypot(grid["cell_u_arcsec"], grid["cell_v_arcsec"])
    assert sep.max() <= max_diag, sep.max()


def test_degenerate_collinear_input_does_not_raise():
    """A convex hull needs 3 non-collinear points; a perfectly straight line of
    stars is a real (if extreme) degenerate case a real catalog will not hand
    it, but the QhullError path must not raise -- it must fall back to angle
    0 rather than crash a checkpoint run."""
    x = np.linspace(-100.0, 100.0, 50)
    y = np.zeros_like(x)
    coords = _sky(x, y)
    grid = footprint_aligned_grid(coords, cell_arcsec=45.0)
    # Either a graceful ok=True (angle 0 fallback) or ok=False with a reason --
    # never an exception, and never a NaN cell geometry silently used.
    if grid["ok"]:
        assert np.isfinite(grid["pa_deg"])
        assert grid["n_u"] >= 1 and grid["n_v"] >= 1
    else:
        assert grid["reason"]
