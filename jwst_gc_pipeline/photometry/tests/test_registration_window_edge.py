"""The own-catalog/cross-band per-cell check must not grade a window-edge peak.

``registration_failsafes.per_cell`` histograms every det-truth pair inside a FIXED
2.5" disk (``MX``) and takes the arg-max.  It cannot sweep that window, so a cell
holding no true counterpart pair still returns a peak: the largest bin of the
wrong-pair background, whose radius is distributed as ``r dr`` over the disk and
therefore piles up near the rim.  ``gc2211_o028`` F150W merged is that population --
138 of 139 verified cells high-offset, median 1826 mas (0.73*MX), worst 2487 mas
(0.995*MX), all at peak_bg 5.0-7.0, i.e. the verify floor (issue #588).  A real 1.8"
misregistration produces the same arg-max, and this estimator cannot separate them,
so grading such a cell asserts a measurement that was not made.

``WINDOW_EDGE_FRAC`` withdraws the claim above half the window and reports the cells
instead.  These tests pin BOTH directions: the edge population stops failing, and the
60-1250 mas regime the two fail axes were calibrated on (#170/#179 injected +90 mas
seams; ``OFF_MAX`` = 60) fails exactly as before.
"""
import importlib.util
import inspect
import pathlib

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "registration_failsafes",
    REPO_ROOT / "scripts" / "release" / "registration_failsafes.py")
rf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rf)

RA0, DEC0 = 266.5, -28.7
COSD = np.cos(np.deg2rad(DEC0))
NO_EDGE = 10.0          # window_edge_frac past 1 -> the arm cannot fire (pre-#588 code)


def _grid(n_side=140, extent_arcsec=40.0, seed=0):
    rng = np.random.default_rng(seed)
    g = np.linspace(0, extent_arcsec, n_side)
    xx, yy = np.meshgrid(g, g)
    return (xx.ravel() + rng.normal(0, 0.02, xx.size),
            yy.ravel() + rng.normal(0, 0.02, yy.size))


def _sc_xy(x, y):
    """x, y in arcsec offsets from (RA0, DEC0)."""
    return SkyCoord((RA0 + (x / 3600.0) / COSD) * u.deg, (DEC0 + y / 3600.0) * u.deg)


def _shifted_field(shift_mas, seed=5):
    """Truth = a star grid; detections = the same stars displaced by ``shift_mas``
    in Dec.  Every pair is a true counterpart, so the peak is sharp and lands at
    exactly ``shift_mas`` -- the cleanest possible reading of a rigid offset."""
    x, y = _grid(seed=seed)
    return _sc_xy(x, y + shift_mas / 1000.0), _sc_xy(x, y)


def test_a_two_arcsec_peak_is_not_graded_but_is_reported():
    """THE mutation test.  A 2000 mas (0.8*MX) displacement is inside the search disk,
    so pre-#588 it read as a confident 2 arcsec offset and failed on both axes.  It is
    also indistinguishable from the wrong-pair arg-max of an unswept 2.5" window, so
    the check now declines it and says so.  Reverting the arm (or raising
    WINDOW_EDGE_FRAC past 1, as ``before`` does here) restores the failure."""
    det, truth = _shifted_field(2000.0)
    before = rf.per_cell(det, None, truth, "unguarded",
                         fail_min_ratio=rf.FAIL_MIN_RATIO, window_edge_frac=NO_EDGE)
    after = rf.per_cell(det, None, truth, "guarded", fail_min_ratio=rf.FAIL_MIN_RATIO)
    assert before["n_fail"] > 0 and not before["PASS"]      # the pre-#588 verdict
    assert before["n_window_edge"] == 0                     # ... with nothing reported
    assert after["n_fail"] == 0 and after["PASS"]           # no longer graded
    assert after["n_window_edge"] == before["n_fail"] + before["n_unconfident_highoff"]
    # the state is visible, with the tell that names it
    assert after["window_edge_cells"], after
    worst = after["window_edge_cells"][0]
    assert worst["window_edge_fraction"] > rf.WINDOW_EDGE_FRAC
    assert worst["offset_mas"] > 1900


def test_the_calibrated_seam_regime_still_fails():
    """+90 mas is the amplitude #179 injected and #170 calibrated the seam axis on.
    It sits far below 0.5*MX, so the arm must not touch it."""
    det, truth = _shifted_field(90.0, seed=7)
    r = rf.per_cell(det, None, truth, "90 mas", fail_min_ratio=rf.FAIL_MIN_RATIO)
    assert not r["PASS"] and r["n_fail"] > 0 and r["n_fail_seam"] > 0
    assert r["n_window_edge"] == 0


def test_the_top_of_the_graded_band_still_fails():
    """Just under the threshold (1200 mas < 0.5*MX = 1250) is still graded, so the arm
    is a cut at WINDOW_EDGE_FRAC and not a blanket relaxation of large offsets."""
    det, truth = _shifted_field(1200.0, seed=11)
    r = rf.per_cell(det, None, truth, "1200 mas", fail_min_ratio=rf.FAIL_MIN_RATIO)
    assert not r["PASS"] and r["n_fail"] > 0
    assert r["n_window_edge"] == 0


def test_a_clean_field_reports_no_edge_cells():
    """No false tagging: a registered field grades as before, with an empty report."""
    x, y = _grid(seed=9)
    r = rf.per_cell(_sc_xy(x, y), None, _sc_xy(x, y), "clean",
                    fail_min_ratio=rf.FAIL_MIN_RATIO)
    assert r["PASS"] and r["n_fail"] == 0 and r["verified_cells"] > 0
    assert r["n_window_edge"] == 0 and r["window_edge_cells"] == []


def test_edge_cells_are_disjoint_from_the_other_two_reports():
    """Three states partition the verified high-offset cells: failed, tolerated
    sub-margin, and could-not-measure.  A cell must appear in exactly one, or a
    reader cannot tell what the gate concluded about it."""
    det, truth = _shifted_field(2000.0, seed=13)
    before = rf.per_cell(det, None, truth, "unguarded",
                         fail_min_ratio=rf.FAIL_MIN_RATIO, window_edge_frac=NO_EDGE)
    after = rf.per_cell(det, None, truth, "guarded", fail_min_ratio=rf.FAIL_MIN_RATIO)
    high_off_total = before["n_fail"] + before["n_unconfident_highoff"]
    assert (after["n_fail"] + after["n_unconfident_highoff"] + after["n_window_edge"]
            == high_off_total)
    assert after["n_window_edge"] > 0


def test_a_field_with_no_tie_at_all_is_reported_not_failed():
    """The gc2211_o028 shape, synthesised: detections that have NO counterpart in the
    truth set, matched against a truth set dense enough to fill the histogram.  Every
    pair is a chance coincidence, so the arg-max wanders the disk and lands mostly
    beyond half of it -- 'no tie exists inside 2.5 arcsec', which is what o028's
    per-exposure catalog (294704 rows against 23296 detections) produces."""
    rng = np.random.default_rng(3)
    extent = 40.0
    det = _sc_xy(rng.uniform(0, extent, 6000), rng.uniform(0, extent, 6000))
    truth = _sc_xy(rng.uniform(0, extent, 60000), rng.uniform(0, extent, 60000))
    r = rf.per_cell(det, None, truth, "no tie", fail_min_ratio=rf.FAIL_MIN_RATIO)
    assert r["verified_cells"] > 0, r          # the cells do verify, as o028's did
    assert r["n_window_edge"] > 0, r
    # and the population sits where a uniform disk background puts it: median radius
    # 0.71*MX for p(r) ~ r dr.  o028 read 1826 mas against that prediction's 1768.
    edge_offsets = [c["offset_mas"] for c in r["window_edge_cells"]]
    assert min(edge_offsets) > rf.WINDOW_EDGE_FRAC * rf.MX.to(u.mas).value


def test_window_edge_frac_default_and_bounds():
    """No silent relaxation, and the graded band keeps its calibrated range."""
    assert (inspect.signature(rf.per_cell).parameters["window_edge_frac"].default
            == rf.WINDOW_EDGE_FRAC)
    assert 0.0 < rf.WINDOW_EDGE_FRAC < 1.0
    # the band that stays graded spans 60 mas -> 1250 mas: the whole regime the
    # contrast and seam axes were calibrated on, with 20x headroom over OFF_MAX.
    assert rf.WINDOW_EDGE_FRAC * rf.MX.to(u.mas).value > 20 * rf.OFF_MAX
