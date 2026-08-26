"""Hermetic tests for the own_catalog FAIL contrast-margin (FAIL_MIN_RATIO).

A cell FAILs only on a large offset AND confident contrast.  A real localized seam
doubles stars into a SHARP high-contrast peak; a bright-star-crowded, sparse cell
throws a floor-level peak (ratio ~ MIN_PEAK_RATIO) at a spurious large offset -- not
a seam, so it must NOT FAIL (brick F405N: 7 cells at 80 mas / peak_bg 5-8; the
same-star m7 check of those regions read <=22 mas, 2026-07).  These build synthetic
det/truth sets so ``per_cell`` runs on controlled geometry -- no data files.
"""
import importlib.util
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

_spec = importlib.util.spec_from_file_location(
    "registration_failsafes",
    Path(__file__).with_name("registration_failsafes.py"))
rf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rf)

RA0, DEC0 = 266.5, -28.7
COSD = np.cos(np.deg2rad(DEC0))


def _grid(n_side=140, extent_arcsec=40.0, seed=0):
    rng = np.random.default_rng(seed)
    g = np.linspace(0, extent_arcsec, n_side)
    xx, yy = np.meshgrid(g, g)
    x = xx.ravel() + rng.normal(0, 0.02, xx.size)
    y = yy.ravel() + rng.normal(0, 0.02, yy.size)
    return x, y


def _sc_xy(x, y):
    return SkyCoord((RA0 + (x / 3600.0) / COSD) * u.deg, (DEC0 + y / 3600.0) * u.deg)


def _weak_spurious_field(extent=40.0, shift_frac=0.15, seed=1):
    """Reviewer's construction: truth = a regular star grid; detections = a minority
    (``shift_frac``) of the stars shifted +80 mas PLUS an EQUAL number of uniform-random
    detections.  The random detections make the wrong-pair background comparable to the
    coherent 80 mas peak, so per-cell peaks land at ~80 mas with FLOOR-LEVEL contrast
    (ratio ~ 5-10) -- the exact regime the margin governs (and the brick F405N artifact)."""
    x, y = _grid(extent_arcsec=extent, seed=seed)
    truth = _sc_xy(x, y)
    rng = np.random.default_rng(seed + 100)
    n = x.size
    shifted = rng.random(n) < shift_frac
    dx, dy = x[shifted], y[shifted] + 80e-3 / 3600.0 * 3600.0   # +80 mas in dec (arcsec units of y)
    m = shifted.sum()
    rx = rng.uniform(0, extent, m)
    ry = rng.uniform(0, extent, m)
    det = _sc_xy(np.concatenate([dx, rx]), np.concatenate([dy, ry]))
    return det, truth


def test_margin_demotes_lowcontrast_highoff_and_would_fail_at_floor():
    """THE mutation test. On the weak-spurious field:
      - at the strict floor (fail_min_ratio=MIN_PEAK_RATIO) the low-contrast high-offset
        cells are FAILs;
      - at FAIL_MIN_RATIO those same cells become verified-but-not-failed and are
        surfaced in n_unconfident_highoff.
    Conservation (strict n_fail == relaxed n_fail + relaxed n_unconfident) pins the
    behaviour: reverting FAIL_MIN_RATIO to the floor makes n_unconfident_highoff 0 and
    breaks the first assert.

    ``min_seam_cells`` is set past the size of the whole grid so the seam axis (#170)
    cannot fire, isolating the CONTRAST axis this test was written to pin.  The seam
    axis DOES fail this field -- the construction shifts 15 % of the stars everywhere,
    so its high-offset cells form one large connected patch, which is the behaviour
    change that axis exists for.  That is asserted in
    ``test_seam_axis_fails_the_fieldwide_weak_construction`` below, and the scattered
    case that the brick F405N false positives actually were is asserted in
    ``test_seam_mask_ignores_scattered_singletons``."""
    det, truth = _weak_spurious_field()
    off_axis = rf.GRID * rf.GRID + 1        # larger than the grid -> seam axis disabled
    strict = rf.per_cell(det, None, truth, "strict", fail_min_ratio=rf.MIN_PEAK_RATIO,
                         min_seam_cells=off_axis)
    relaxed = rf.per_cell(det, None, truth, "relaxed", fail_min_ratio=rf.FAIL_MIN_RATIO,
                          min_seam_cells=off_axis)
    assert relaxed["n_unconfident_highoff"] > 0            # fails if margin == floor
    assert relaxed["n_fail"] < strict["n_fail"]           # margin actually demoted some
    # every demoted cell was a strict FAIL (nothing invented, nothing lost)
    assert strict["n_fail"] == relaxed["n_fail"] + relaxed["n_unconfident_highoff"]
    assert strict.get("n_unconfident_highoff", 0) == 0    # floor has no unconfident band
    # demoted cells carry npairs for triage (nit)
    assert all("npairs" in c for c in relaxed["unconfident_highoff_cells"])


def test_coherent_high_contrast_seam_FAILS_even_at_margin():
    """A whole-field 90 mas rigid offset -> every star shifted -> sharp, high-contrast
    peak -> MUST fail even at the raised bar (seam sensitivity retained)."""
    x, y = _grid(seed=7)
    truth = _sc_xy(x, y)
    det = _sc_xy(x, y + 90e-3 / 3600.0 * 3600.0)          # +90 mas dec, all stars
    r = rf.per_cell(det, None, truth, "seam", fail_min_ratio=rf.FAIL_MIN_RATIO)
    assert not r["PASS"] and r["n_fail"] > 0
    assert r["worst"][0]["peak_bg"] >= rf.FAIL_MIN_RATIO


def test_companion_checks_keep_the_strict_floor():
    """The margin is own-catalog-only: the default fail_min_ratio is the strict floor,
    so cross_band / per_module are unchanged. On the weak-spurious field the default
    (strict) call still FAILs where the relaxed own-catalog call would not."""
    det, truth = _weak_spurious_field(seed=3)
    default = rf.per_cell(det, None, truth, "default")          # no fail_min_ratio arg
    own = rf.per_cell(det, None, truth, "own", fail_min_ratio=rf.FAIL_MIN_RATIO)
    assert default["n_fail"] >= own["n_fail"]
    assert default["n_fail"] > 0                                # strict still fails these


def test_fail_min_ratio_default_is_the_floor():
    """Default behaviour is the historic strict gate (no silent relaxation)."""
    import inspect
    assert inspect.signature(rf.per_cell).parameters["fail_min_ratio"].default == rf.MIN_PEAK_RATIO
    assert rf.FAIL_MIN_RATIO > rf.MIN_PEAK_RATIO


def test_clean_field_passes_full_coverage():
    """Perfectly-registered field: 0 fail, cells verify (coverage intact -- the margin
    removes no detections)."""
    x, y = _grid(seed=9)
    truth = _sc_xy(x, y); det = _sc_xy(x, y)
    r = rf.per_cell(det, None, truth, "clean", fail_min_ratio=rf.FAIL_MIN_RATIO)
    assert r["PASS"] and r["n_fail"] == 0 and r["verified_cells"] > 0


# ---- seam contiguity: the second, amplitude-free axis (issue #170) -----------------


def test_seam_mask_requires_three_touching_cells():
    """The measured lower bound. Two of the seven brick F405N false positives are
    4-adjacent -- component sizes were [2, 1, 1, 1, 1, 1] -- so a 2-cell bar re-creates
    the exact false alarm #166/#172 removed."""
    g = np.zeros((20, 20), bool)
    g[5, 5] = g[5, 6] = True                 # an adjacent PAIR, as on brick F405N
    keep, sizes = rf.seam_mask(g)
    assert not keep.any() and sizes == []
    g[5, 7] = True                           # now three in a row
    keep, sizes = rf.seam_mask(g)
    assert keep.sum() == 3 and sizes == [3]


def test_seam_mask_ignores_scattered_singletons():
    """Wrong-pair noise is scattered; five isolated cells are not a seam at any count."""
    g = np.zeros((20, 20), bool)
    for i, j in ((1, 1), (4, 9), (9, 4), (14, 15), (18, 2)):
        g[i, j] = True
    keep, sizes = rf.seam_mask(g)
    assert not keep.any() and sizes == []


def test_seam_mask_uses_edge_connectivity_not_corners():
    """Diagonal-only contact is one corner's worth and must not build a component."""
    g = np.zeros((20, 20), bool)
    g[3, 3] = g[4, 4] = g[5, 5] = True       # a diagonal chain of three
    keep, sizes = rf.seam_mask(g)
    assert not keep.any() and sizes == []


def test_seam_mask_counts_a_real_seam_strip():
    """A one-cell-wide strip across the grid is the shape this axis exists for."""
    g = np.zeros((20, 20), bool)
    g[:, 7] = True
    keep, sizes = rf.seam_mask(g)
    assert keep.sum() == 20 and sizes == [20]


def test_seam_mask_is_empty_on_an_empty_grid():
    keep, sizes = rf.seam_mask(np.zeros((20, 20), bool))
    assert not keep.any() and sizes == []


def test_seam_axis_fails_the_fieldwide_weak_construction():
    """The behaviour change, on #172's own synthetic field.

    15 % of the stars are displaced +80 mas EVERYWHERE, so the high-offset cells form
    one large connected patch.  The contrast bar alone passes many of them (that is what
    ``test_margin_demotes_lowcontrast_highoff_and_would_fail_at_floor`` records); the
    seam axis fails them on shape.  A misregistration covering the field is exactly what
    a release gate must not let through on a density technicality.
    """
    det, truth = _weak_spurious_field()
    off_axis = rf.GRID * rf.GRID + 1
    without = rf.per_cell(det, None, truth, "contrast only",
                          fail_min_ratio=rf.FAIL_MIN_RATIO, min_seam_cells=off_axis)
    with_seam = rf.per_cell(det, None, truth, "both axes",
                            fail_min_ratio=rf.FAIL_MIN_RATIO)
    assert without["n_unconfident_highoff"] > 0        # the contrast bar tolerated some
    assert with_seam["n_fail"] > without["n_fail"]     # the seam axis fails them
    assert with_seam["n_fail_seam_only"] > 0           # ... on shape alone
    assert max(with_seam["seam_component_sizes"]) >= rf.MIN_SEAM_CELLS
    # the two reports stay disjoint: a cell that now fails is not also "tolerated"
    assert (with_seam["n_unconfident_highoff"]
            + with_seam["n_fail_seam_only"] == without["n_unconfident_highoff"])


def test_seam_axis_is_strictly_additive():
    """It can only ADD failures.  A clean field stays a PASS with the axis on."""
    x, y = _grid(seed=9)
    truth = _sc_xy(x, y)
    det = _sc_xy(x, y)
    r = rf.per_cell(det, None, truth, "clean", fail_min_ratio=rf.FAIL_MIN_RATIO)
    assert r["PASS"] and r["n_fail"] == 0 and r["n_fail_seam"] == 0
    assert r["seam_component_sizes"] == []
    # and a high-contrast whole-field seam still fails, with both axes agreeing
    det2 = _sc_xy(x, y + 90e-3)
    r2 = rf.per_cell(det2, None, truth, "seam", fail_min_ratio=rf.FAIL_MIN_RATIO)
    assert not r2["PASS"] and r2["n_fail"] > 0 and r2["n_fail_seam"] > 0


def test_min_seam_cells_default_and_constant():
    """No silent relaxation: the default is the module constant, and it is 3."""
    import inspect
    assert (inspect.signature(rf.per_cell).parameters["min_seam_cells"].default
            == rf.MIN_SEAM_CELLS)
    assert rf.MIN_SEAM_CELLS == 3
