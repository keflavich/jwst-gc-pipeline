"""The dense-reference spatial gate must be measured same-star, not by a
per-tile histogram whose peak is a handful of counts (issue #610).

cloudef's m7 cross-filter checkpoint blocked on ONE per-tile cell that reported
5953.8 mas at contrast 4.5, on a field whose anchor ties to VIRAC2 at 0.334 mas
same-star over 3682 pairs and whose 1202-cell same-star local map is clean.
Reproducing the grid on the real catalogs shows why: in each cell the true peak
bin at zero clears the tallest noise bin by a margin of -4 to +17 COUNTS out of
20 000-380 000 pairs in the window, and in four cells the margin went negative,
so those cells reported the densest noise bin (0.93"-5.95") as a measured
offset.  One of the four fell under the contrast floor and blocked the field;
the other three passed.

So on a small, verified tie the spatial check is now the same-star region map,
which on that same field reads 26 cells of 40-192 matched pairs, worst 21 mas,
median 3-sigma 23 mas.  These tests hold the two properties that make the swap
safe: a real seam still fails, in BOTH of the two ways a seam can present.
"""
import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord

from jwst_gc_pipeline.photometry.astrometry_offsets import (
    measure_offset, same_star_region_map)
from jwst_gc_pipeline.photometry import visit_consensus as _vc
from jwst_gc_pipeline.photometry.visit_consensus import measure_reference_tie

RA0, DEC0 = 266.60, -28.50
COSD = float(np.cos(np.radians(DEC0)))


def _sky(x_arcsec, y_arcsec):
    """Tangent-plane arcsec offsets from (RA0, DEC0) -> SkyCoord."""
    return SkyCoord((RA0 + np.asarray(x_arcsec) / 3600.0 / COSD) * u.deg,
                    (DEC0 + np.asarray(y_arcsec) / 3600.0) * u.deg, frame="icrs")


def _field(n_common=1400, n_ref_only=2600, width=120.0, ref_width=150.0,
           scatter_mas=40.0, tie_mas=(15.0, -8.0), seed=11):
    """A GC-like pair: a JWST footprint, and a DENSE reference that covers more
    sky than it, sharing ``n_common`` stars measured to ``scatter_mas``.

    Returns ``(x, y, ref)`` with the COMMON stars first in ``x``/``y`` so a test
    can displace a chosen subset of them.
    """
    rng = np.random.RandomState(seed)
    x = (rng.rand(n_common) - 0.5) * width
    y = (rng.rand(n_common) - 0.5) * width
    # the reference sees those same stars, with its own positional scatter...
    rx = x + rng.randn(n_common) * scatter_mas / 1000.0
    ry = y + rng.randn(n_common) * scatter_mas / 1000.0
    # ...plus a population the JWST catalog does not share (the wrong-pair
    # background that makes the per-tile histogram a low-count statistic)
    ox = (rng.rand(n_ref_only) - 0.5) * ref_width
    oy = (rng.rand(n_ref_only) - 0.5) * ref_width
    ref = _sky(np.concatenate([rx, ox]), np.concatenate([ry, oy]))
    # the JWST frame sits `tie_mas` off the reference: a small, real bulk tie
    x = x - tie_mas[0] / 1000.0
    y = y - tie_mas[1] / 1000.0
    return x, y, ref


def _tie(a, ref):
    res = measure_offset(a, ref, sweep=True, context="region-map-test")
    assert res is not None and res["ok"] and not res["swept"], res
    return res


def test_clean_field_gives_a_measurable_clean_region_map():
    x, y, ref = _field()
    a = _sky(x, y)
    m = same_star_region_map(a, ref, _tie(a, ref), context="clean")
    assert m["measurable"] is True, m["reason"]
    assert m["n_cells"] >= 4, m
    assert m["n_flagged"] == 0 and m["n_uncovered"] == 0, m
    assert m["clean"] is True, m["reason"]


def test_seam_inside_the_match_radius_is_flagged():
    """brick-1182 F200W class: a ~90 mas residual confined to one strip, which
    a rigid tie cannot remove and a field-pooled number averages away."""
    x, y, ref = _field()
    strip = y > 40.0
    y = y.copy()
    y[strip] += 90.0 / 1000.0          # 90 mas, well inside the 0.3" match radius
    a = _sky(x, y)
    m = same_star_region_map(a, ref, _tie(a, ref), context="seam")
    assert m["measurable"] is True, m["reason"]
    assert m["n_flagged"] >= 1, m
    assert m["clean"] is False
    assert m["worst_off_mas"] > 60.0, m
    assert "above 15.0 mas" in m["reason"], m["reason"]


def test_region_displaced_beyond_the_match_radius_is_reported_uncovered():
    """brick-1182 v001 class: half a mosaic ~20" out of place.  Its sources are
    all still there and every one of them loses its partner, so the residual
    statistic is blind to it and the COVERAGE test is what sees it."""
    x, y, ref = _field()
    y = y.copy()
    y[y > 0.0] += 20.0                  # 20 arcsec: half the mosaic out of place
    a = _sky(x, y)
    m = same_star_region_map(a, ref, _tie(a, ref), context="displaced")
    assert m["measurable"] is True, m["reason"]
    assert m["n_uncovered"] >= 1, m
    assert m["clean"] is False
    assert "displaced beyond" in m["reason"], m["reason"]


def test_a_sparse_footprint_edge_is_not_called_a_seam():
    """The cloudef corner cell (5,0) holds 38% of its box and 49 matched stars
    against a typical 130.  Thin coverage must read as unmeasured, never as a
    displaced region -- that conflation is the defect this replaces."""
    x, y, ref = _field()
    keep = (y < 40.0) | (np.random.RandomState(5).rand(len(y)) < 0.08)
    a = _sky(x[keep], y[keep])
    m = same_star_region_map(a, ref, _tie(a, ref), context="thin edge")
    assert m["measurable"] is True, m["reason"]
    assert m["n_uncovered"] == 0, m["uncovered_cells"]
    assert m["clean"] is True, m["reason"]


def test_reference_tie_gates_on_the_region_map_when_the_tie_is_small(monkeypatch):
    """The cloudef regression: a per-tile histogram cell that could not measure
    must no longer decide the field when same-star pairing is unambiguous."""
    x, y, ref = _field()
    a = _sky(x, y)
    real_grid = _vc.measure_offset_grid

    def _unclean_grid(*args, **kwargs):
        grid = real_grid(*args, **kwargs)
        grid["clean"] = False           # as cloudef's 36th cell made it
        grid["n_ok"] = max(grid["n_total"] - 1, 0)
        return grid

    monkeypatch.setattr(_vc, "measure_offset_grid", _unclean_grid)
    tie = measure_reference_tie(a, ref, ref[::40], dense=True,
                                grid_nx=3, grid_ny=3, context="cloudef-like")
    assert tie["bulk_source"] == "same-star", tie["bulk_source"]
    assert tie["per_tile"]["clean"] is False
    assert tie["per_tile_source"] == "same-star-region", tie["per_tile_source"]
    assert tie["per_tile_same_star"]["clean"] is True, tie["per_tile_same_star"]["reason"]
    assert tie["per_tile_ok"] is True
    assert tie["apply_ok"] is True


def test_reference_tie_still_fails_a_real_seam():
    """Same path, real 90 mas strip: the gate must stay red, and say so through
    the region map rather than through the histogram grid."""
    x, y, ref = _field()
    y = y.copy()
    y[y > 40.0] += 90.0 / 1000.0
    a = _sky(x, y)
    tie = measure_reference_tie(a, ref, ref[::40], dense=True,
                                grid_nx=3, grid_ny=3, context="seam")
    assert tie["per_tile_source"] == "same-star-region", tie["per_tile_source"]
    assert tie["per_tile_ok"] is False
    assert tie["apply_ok"] is False


def test_reference_tie_keeps_the_histogram_grid_on_an_unverified_tie(monkeypatch):
    """A large/swept tie cannot be paired same-star, so the histogram grid --
    the estimator that DOES work on a grossly shifted frame -- keeps the gate.
    Nothing about that regime changes."""
    x, y, ref = _field(tie_mas=(0.0, 0.0))
    a = _sky(x + 25.0, y)               # 25" out: the sweep finds it, pairing cannot
    tie = measure_reference_tie(a, ref, ref[::40], dense=True,
                                grid_nx=3, grid_ny=3, context="gross")
    assert tie["same_star"] is None, tie["same_star"]
    assert tie["per_tile_source"] == "histogram-grid", tie["per_tile_source"]
    assert tie["per_tile_same_star"] is None


def test_reference_tie_falls_back_to_the_histogram_grid_when_regions_are_starved():
    """A reference too SPARSE for the region map keeps the histogram grid as the
    gate, so a dense-flagged Gaia-only field still blocks.

    This is the guard that ``test_gaia_only_reference_per_tile_does_not_gate``
    and ``test_measure_bulk_offset_signs_off_on_a_gaia_only_reference`` used to
    carry through ``assert not tie_dense["apply_ok"]``.  Those two run a 90"
    field of 400 perfectly-paired stars -- ~180 stars/arcmin^2, denser than
    VIRAC2 over the Brick -- so since #610 the region map measures it, finds no
    spatial structure (there is none) and passes.  A REAL Gaia-only reference is
    nowhere near that: at 150 stars over a 120" box no 45" cell reaches the 40
    matched pairs a region needs, ``measurable`` is False, and the fallback
    keeps the old verdict.  ``measurable=False`` must never read as a pass.
    """
    rng = np.random.RandomState(7)
    n, width = 150, 120.0
    x = (rng.rand(n) - 0.5) * width
    y = (rng.rand(n) - 0.5) * width
    ref = _sky(x + rng.randn(n) * 0.040, y + rng.randn(n) * 0.040)
    a = _sky(x - 0.015, y + 0.008)      # small, pairable 15/8 mas tie

    tie = measure_reference_tie(a, ref, ref, dense=True,
                                grid_nx=6, grid_ny=6, context="starved")
    assert tie["same_star"] is not None          # the tie IS pairable...
    assert tie["per_tile_same_star"]["measurable"] is False   # ...the map is not
    assert tie["per_tile_same_star"]["clean"] is False
    assert tie["per_tile_source"] == "histogram-grid"
    assert tie["per_tile_ok"] is False
    assert tie["apply_ok"] is False
