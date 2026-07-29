"""Hermetic tests for the two NEW own_catalog fail axes (issue #170).

The historic discriminant is ``ratio = H.max()/median(H[H>0])``.  Two additions:

* ``sig``    -- a density-FLAT Poisson significance of the peak over the EXPECTED
                wrong-pair background.  The expectation must be the mean over the search
                disk, not ``median(H[H>0])``: the latter is exactly 1 in every verified
                cell of every real field measured, which makes the naive
                ``(H.max()-bg)/sqrt(bg)`` identically ``ratio-1`` -- a relabelling.
* ``contig`` -- an amplitude-FREE axis: a vector-coherent 4-connected patch of
                >=MIN_SEAM_CELLS high-offset cells is seam-shaped no matter how weak the
                contrast, while a scattered singleton is not.

Synthetic star fields only -- no data files.  The numeric anchors below are MEASURED on
brick F405N (2026-07); see the calibration table in the PR for #170.
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

# ---- MEASURED calibration anchors (brick F405N own_catalog, 2026-07) ----------------
# (a) the 7 artifact cells: 80 mas peak offset, same-star m7 truth <=22 mas -> must PASS
ARTIFACT_SIG_MAX = 49.32         # measured max over the 7 cells (range 32.55-49.32)
ARTIFACT_RATIO_MAX = 8.0         # measured max (range 5-8)
# (b) injected +90 mas seams (full field / half field / 10%-wide band), excluding those
#     same 7 cells -> must FAIL.  The MEDIANs are the anchor, not the minima: the two
#     populations overlap at the low end (both start at sig ~32.6), so a usable bar must
#     sit below the seam's typical cell, not below its weakest one.
SEAM_SIG_MEDIAN_MIN = 60.7       # smallest per-cell median over the three injected seams
SEAM_RATIO_MEDIAN_MIN = 12.0     # ditto for the historic ratio


def _sc_xy(x, y):
    """arcsec offsets from (RA0, DEC0) -> SkyCoord."""
    return SkyCoord((RA0 + (x / 3600.0) / COSD) * u.deg, (DEC0 + y / 3600.0) * u.deg)


def _cellular_field(shift_cells, extent=40.0, det_per_cell=7, truth_per_arcsec2=25.0,
                    shift_mas=80.0, seed=5):
    """Truth = a deep uniform catalogue; detections = a SPARSE bright subset, laid down
    ``det_per_cell`` per ``per_cell`` grid cell so every cell's peak height is known and
    identical.  Detections in ``shift_cells`` (a list of (i, j) grid indices) are
    displaced by a COMMON ``shift_mas`` in Dec.

    With det_per_cell in 5..9 the peak is above MIN_PEAK_RATIO (cell verifies) but below
    FAIL_MIN_RATIO, and the deep truth makes the pair count -- hence the Poisson
    background -- large enough that ``sig`` is also well below FAIL_MIN_SIG.  So NEITHER
    amplitude axis can fire and only contiguity decides: exactly the regime #170 is about.
    """
    rng = np.random.default_rng(seed)
    n_truth = int(truth_per_arcsec2 * extent ** 2)
    tx = rng.uniform(0, extent, n_truth)
    ty = rng.uniform(0, extent, n_truth)
    cw = extent / rf.GRID
    shift = set(map(tuple, shift_cells))
    dx, dy = [], []
    for i in range(rf.GRID):
        for j in range(rf.GRID):
            # place detections ON truth positions inside this cell (so they have matches)
            m = ((tx >= i * cw) & (tx < (i + 1) * cw) & (ty >= j * cw) & (ty < (j + 1) * cw))
            idx = np.flatnonzero(m)
            pick = rng.choice(idx, size=det_per_cell, replace=False)
            sx, sy = tx[pick], ty[pick]
            if (i, j) in shift:
                sy = sy + shift_mas / 1000.0
            dx.append(sx); dy.append(sy)
    det = _sc_xy(np.concatenate(dx), np.concatenate(dy))
    return det, _sc_xy(tx, ty)


def _sparse_coherent_field(shift_cells, extent=80.0, det_per_cell=8,
                           truth_per_arcsec2=1.3, shift_mas=80.0, seed=11):
    """Same construction with a SHALLOW truth catalogue.  Fewer chance pairs -> the same
    peak height is far more significant, so ``sig`` fires where the bare peak-count
    ``ratio`` (which cannot see the pair count at all) does not."""
    return _cellular_field(shift_cells, extent=extent, det_per_cell=det_per_cell,
                           truth_per_arcsec2=truth_per_arcsec2, shift_mas=shift_mas,
                           seed=seed)


def _own(det, truth, label="own"):
    return rf.per_cell(det, None, truth, label, fail_min_ratio=rf.FAIL_MIN_RATIO)


# ---------------------------------------------------------------- significance ------
def test_significance_is_not_the_naive_ratio_minus_one():
    """MUTATION LOCK on the background estimator.  ``median(H[H>0])`` is 1 in this
    regime, so the literal formula from #170 would give sig == ratio-1 and change
    nothing.  Using the expected (mean-over-disk) background makes them differ by a
    large factor."""
    H = np.zeros((125, 125))
    H[10, 10] = 8.0
    rng = np.random.default_rng(0)
    for a, b in zip(rng.integers(0, 125, 200), rng.integers(0, 125, 200)):
        if (a, b) != (10, 10):
            H[a, b] = 1.0
    peak, ratio, sig = rf._peak_stats(H, npairs=int(H.sum()), n_bg_bins=12270)
    assert peak == 8.0
    assert ratio == 8.0                       # median of occupied bins is 1 -> bare count
    assert sig > 5 * (ratio - 1.0)            # fails if the naive background is restored
    assert 55.0 < sig < 75.0                  # (8 - 208/12270)/sqrt(208/12270)


def test_significance_is_density_flat_where_the_ratio_is_not():
    """Double the source density at a FIXED match fraction: the bare ratio scales with
    density, the significance does not.  This is the whole point of #170."""
    def stats(scale):
        peak = 8.0 * scale
        H = np.zeros((125, 125)); H[10, 10] = peak
        n = int(300 * scale ** 2)             # chance pairs ~ n_det * n_truth ~ density^2
        rng = np.random.default_rng(1)
        for a, b in zip(rng.integers(0, 125, 400), rng.integers(0, 125, 400)):
            if (a, b) != (10, 10):
                H[a, b] = 1.0
        return rf._peak_stats(H, npairs=n, n_bg_bins=12270)[1:]

    r1, s1 = stats(1.0)
    r4, s4 = stats(4.0)
    assert r4 == 4 * r1                        # ratio is density-COUPLED (here: linear)
    assert abs(s4 - s1) / s1 < 0.05            # significance is density-FLAT


def test_threshold_separates_the_two_measured_brick_populations():
    """CALIBRATION LOCK.  FAIL_MIN_SIG must sit strictly between the measured artifact
    ceiling (must not fail) and the measured seam floor (must fail).  Reverting the
    threshold to a no-op in either direction (0 -> everything fails, inf -> nothing
    fails) breaks this."""
    assert ARTIFACT_SIG_MAX < rf.FAIL_MIN_SIG <= SEAM_SIG_MEDIAN_MIN
    # the same statement for the transitional ratio bar, which #170 keeps
    assert ARTIFACT_RATIO_MAX < rf.FAIL_MIN_RATIO <= SEAM_RATIO_MEDIAN_MIN


def _cell_hist(peak, npairs):
    """An offset histogram with the measured shape of a real cell: one bin holding
    ``peak`` counts and the remaining pairs spread one-per-bin (which is what makes
    median(H[H>0]) == 1 on real data)."""
    H = np.zeros((len(rf.HB) - 1, len(rf.HB) - 1))
    H[40, 40] = peak
    rng = np.random.default_rng(3)
    n = max(int(npairs - peak), 0)
    a = rng.integers(0, H.shape[0], n); b = rng.integers(0, H.shape[1], n)
    for p, q in zip(a, b):
        if (p, q) != (40, 40):
            H[p, q] = 1.0
    return H


def test_measured_brick_cells_land_on_the_correct_side_of_the_bar():
    """Straight from the 2026-07 brick F405N measurement, with the module's own bin
    count -- the strongest artifact cell must stay under the bar and a typical injected
    seam cell must clear it."""
    # artifact cell (6,16): peak 8 counts over 321 pairs  -> sig 49.3, must NOT fail
    art = rf._peak_stats(_cell_hist(8, 321), npairs=321)
    assert art[1] == 8.0 and 45.0 < art[2] < 53.0
    assert art[2] < rf.FAIL_MIN_SIG
    # injected-seam cell (0,1): peak 18 counts over 674 pairs -> sig 76.6, must fail
    seam = rf._peak_stats(_cell_hist(18, 674), npairs=674)
    assert 72.0 < seam[2] < 82.0
    assert seam[2] >= rf.FAIL_MIN_SIG
    # the ORDERING is what the naive background destroys: there, sig == ratio - 1, so the
    # separation would be 7 vs 17 -- the same 2.4x the ratio already gives, no gain.
    assert seam[2] / art[2] > 1.5


def test_sig_path_fails_a_sparse_coherent_cell_the_ratio_bar_misses():
    """A cell with a low peak COUNT but few chance pairs is significant even though the
    bare ratio is under FAIL_MIN_RATIO.  It is a lone cell, so contiguity cannot fire:
    the fail must come from ``sig`` alone.  Reverting the sig path makes this PASS."""
    r = _own(*_sparse_coherent_field([(9, 9)]))
    assert not r["PASS"]
    assert r["n_fail_by_path"]["sig"] > 0
    assert r["n_fail_by_path"]["ratio"] == 0      # bare peak count is under the bar
    assert r["n_fail_by_path"]["contig"] == 0     # single cell, no patch
    assert r["worst"][0]["paths"] == ["sig"]
    assert r["worst"][0]["peak_bg"] < rf.FAIL_MIN_RATIO
    assert r["worst"][0]["peak_sig"] >= rf.FAIL_MIN_SIG


# ---------------------------------------------------------------- contiguity --------
def test_contiguous_patch_fails_but_the_same_count_scattered_does_not():
    """#170's acceptance test.  Identical cell count, identical (low) contrast, identical
    shift: only the SHAPE differs.  Contiguous -> FAIL; scattered -> no fail."""
    contiguous = [(8, 8), (8, 9), (8, 10), (9, 9)]
    scattered = [(2, 3), (7, 14), (13, 5), (17, 11)]
    rc = _own(*_cellular_field(contiguous, seed=5), label="contiguous")
    rs = _own(*_cellular_field(scattered, seed=5), label="scattered")

    # neither population can fire an amplitude axis -- that is what makes this a test of
    # contiguity and not a re-test of the contrast margin
    assert rc["n_fail_by_path"]["ratio"] == 0 and rc["n_fail_by_path"]["sig"] == 0
    assert rs["n_fail_by_path"]["ratio"] == 0 and rs["n_fail_by_path"]["sig"] == 0

    assert not rc["PASS"] and rc["n_fail_by_path"]["contig"] >= rf.MIN_SEAM_CELLS
    assert rs["PASS"] and rs["n_fail"] == 0
    assert rs["n_unconfident_highoff"] > 0        # still surfaced, never silently hidden


def test_contiguity_needs_more_than_two_cells():
    """MIN_SEAM_CELLS is 3, not 2, BECAUSE two of the seven brick F405N artifact cells
    are 4-adjacent and share a common peak vector.  A 2-cell bar re-creates the false
    FAIL this thread is about, so the 2-cell case must not fail."""
    assert rf.MIN_SEAM_CELLS >= 3
    r = _own(*_cellular_field([(8, 8), (8, 9)]), label="pair")
    assert r["PASS"] and r["n_fail"] == 0
    assert r["n_unconfident_highoff"] >= 2


def test_contiguity_requires_a_common_offset_vector():
    """Touching cells that disagree in offset DIRECTION are not one displacement."""
    highoff = np.zeros((rf.GRID, rf.GRID), bool)
    offx = np.full((rf.GRID, rf.GRID), np.nan)
    offy = np.full((rf.GRID, rf.GRID), np.nan)
    for n, (i, j) in enumerate([(4, 4), (4, 5), (5, 4)]):
        highoff[i, j] = True
        ang = 2 * np.pi * n / 3.0
        offx[i, j], offy[i, j] = 90 * np.cos(ang), 90 * np.sin(ang)
    assert not rf._seam_components(highoff, offx, offy).any()
    # ... and the same three cells DO fire once they share a vector
    offx[highoff], offy[highoff] = 0.0, 90.0
    assert rf._seam_components(highoff, offx, offy).sum() == 3


def test_a_minority_of_incoherent_cells_does_not_veto_a_real_seam():
    """A field-wide seam labels as ONE big component containing a few cells that peak
    elsewhere; an all-members coherence rule would discard the entire seam."""
    highoff = np.zeros((rf.GRID, rf.GRID), bool)
    highoff[4:12, 4:12] = True
    offx = np.where(highoff, 0.0, np.nan)
    offy = np.where(highoff, 90.0, np.nan)
    offx[5, 5], offy[5, 5] = 400.0, -600.0        # one rogue cell inside the patch
    seam = rf._seam_components(highoff, offx, offy)
    assert seam.sum() == highoff.sum() - 1
    assert not seam[5, 5]


# ---------------------------------------------------------------- plumbing ----------
def test_both_statistics_are_reported_side_by_side():
    """#170 step 1: both discriminants travel in the result dict so they can be compared
    on real data."""
    r = _own(*_cellular_field([(8, 8), (8, 9), (8, 10)]))
    cells = r["worst"] + r["unconfident_highoff_cells"]
    assert cells
    for c in cells:
        assert set(("peak_bg", "peak_sig", "npairs", "offset_mas", "paths")) <= set(c)
    assert set(r["n_fail_by_path"]) == {"ratio", "sig", "contig"}


def test_clean_field_still_passes_with_all_three_axes():
    """No new axis may fire on a perfectly-registered field."""
    r = _own(*_cellular_field([]))
    assert r["PASS"] and r["n_fail"] == 0 and r["verified_cells"] > 0
    assert r["n_unconfident_highoff"] == 0
