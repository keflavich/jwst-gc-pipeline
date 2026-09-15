"""The binning has to tile the plane and has to be shared between maps."""
import math

import numpy as np
import pytest

from jwst_gc_pipeline.quicklook import hexbin


def test_every_point_lands_in_exactly_one_cell():
    rng = np.random.default_rng(0)
    u, v = rng.random(5000), rng.random(5000)
    q, r = hexbin.assign(u, v, hexbin.hex_radius(20))
    assert len(q) == len(u) and len(r) == len(u)
    assert np.isfinite(q).all() and np.isfinite(r).all()


def test_a_point_is_assigned_to_its_nearest_centre():
    """Cube rounding has three branches and only two of them touch (q, r).

    Getting the branch wrong still returns a plausible-looking integer pair, so
    the test that catches it is the geometric one: the cell a point lands in
    must be the cell whose CENTRE is nearest, checked against a brute-force
    search over the neighbourhood.
    """
    radius = hexbin.hex_radius(12)
    rng = np.random.default_rng(1)
    u, v = rng.random(400), rng.random(400)
    q, r = hexbin.assign(u, v, radius)
    for i in range(len(u)):
        cx, cy = hexbin.centre(q[i], r[i], radius)
        best = (cx - u[i]) ** 2 + (cy - v[i]) ** 2
        for dq in (-1, 0, 1):
            for dr in (-1, 0, 1):
                ox, oy = hexbin.centre(q[i] + dq, r[i] + dr, radius)
                d = (ox - u[i]) ** 2 + (oy - v[i]) ** 2
                assert d >= best - 1e-12, (
                    f'point {i} was assigned to a cell that is not the nearest')


def test_counts_are_conserved():
    grid = hexbin.Grid(0.0, 1.0, 10.0, 20.0, nx=15)
    rng = np.random.default_rng(2)
    x = rng.uniform(0.05, 0.95, 3000)
    y = rng.uniform(10.5, 19.5, 3000)
    cells, peak, n = grid.bin(x, y)
    assert n == 3000
    assert sum(c[2] for c in cells) == 3000
    assert peak == max(c[2] for c in cells)


def test_points_outside_the_extent_are_dropped_not_piled_on_the_edge():
    grid = hexbin.Grid(0.0, 1.0, 0.0, 1.0, nx=10)
    x = np.array([0.5, 0.5, 5.0, -3.0, np.nan])
    y = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    cells, peak, n = grid.bin(x, y)
    assert n == 2 and peak == 2 and len(cells) == 1


def test_two_datasets_on_one_grid_share_cell_indices():
    """The point of the page: the per-field overlay must land on the cells of
    the background it is drawn over."""
    grid = hexbin.Grid(0.0, 1.0, 0.0, 1.0, nx=20)
    rng = np.random.default_rng(3)
    x, y = rng.random(2000), rng.random(2000)
    all_cells, _, _ = grid.bin(x, y)
    half_cells, _, _ = grid.bin(x[:600], y[:600])
    keys = {(c[0], c[1]) for c in all_cells}
    assert {(c[0], c[1]) for c in half_cells} <= keys


def test_hexagons_tile_without_gaps():
    """Row spacing is 1.5R against a hexagon height of 2R, and column spacing
    sqrt(3)R against a width of sqrt(3)R -- so neighbours touch or overlap.
    A grid that leaves gaps draws a moire pattern instead of a density map."""
    radius = hexbin.hex_radius(30)
    c00 = hexbin.centre(0, 0, radius)
    right = hexbin.centre(1, 0, radius)
    below = hexbin.centre(0, 1, radius)
    assert right[0] - c00[0] == pytest.approx(math.sqrt(3) * radius)
    assert below[1] - c00[1] == pytest.approx(1.5 * radius)
    assert below[1] - c00[1] < 2 * radius


def test_extent_ignores_the_outlier_tail():
    x = np.concatenate([np.linspace(0, 1, 1000), [500.0]])
    y = np.concatenate([np.linspace(10, 20, 1000), [-900.0]])
    xlo, xhi, ylo, yhi = hexbin.percentile_extent(x, y)
    assert xhi < 2 and ylo > 0


def test_a_degenerate_extent_is_refused():
    with pytest.raises(ValueError):
        hexbin.Grid(1.0, 1.0, 0.0, 1.0)
