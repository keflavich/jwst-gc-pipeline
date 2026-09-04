"""A window-edge peak must be RESOLVED before it is graded -- never waved through.

``registration_failsafes.per_cell`` histograms every det-truth pair inside a fixed 2.5"
disk (``MX``) and takes the arg-max.  Two very different things put that arg-max near
the rim:

* a REAL rigid misregistration of 1.3-2.5", the loudest failure this gate can see;
* the wrong-pair background of a window too narrow to hold any true pair, whose arg-max
  radius is distributed as ``r dr`` and therefore piles up near the rim (median
  0.71*MX = 1768 mas).  ``gc2211_o028`` F150W merged is that population: 138 of 139
  verified cells high-offset, median 1826 mas, worst 2487 mas (0.995*MX), all at
  peak_bg 5.0-7.0 -- the verify floor (issue #588).

The radius alone does not separate them, so ungrading on radius alone would turn a clean
2" misregistration into a PASS.  What separates them is measured: CONTRAST at the base
window (a rigid shift piles true counterparts into one bin: 16-24 against the
background's 5-7) and the SWEEP (CLAUDE.md: a real tie reads the SAME offset at every
window that can contain it, while the background's arg-max moves with the window).

These tests pin all three outcomes: a real offset FAILS -- including one beyond the base
window, which the sweep now measures at its true value; a field with no tie is
COULD-NOT-VERIFY (``PASS`` None -> exit 2), never a pass; and the 60-1250 mas band both
fail axes were calibrated on (#170/#179 injected +90 mas seams; ``OFF_MAX`` = 60) is
untouched.
"""
import importlib.util
import inspect
import pathlib

import numpy as np
import pytest
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
EXTENT = 40.0           # arcsec on a side of the synthetic field
OWN = dict(fail_min_ratio=rf.FAIL_MIN_RATIO)   # the own-catalog bar, where the arm lives


def _sc(x, y):
    """x, y in arcsec offsets from (RA0, DEC0)."""
    return SkyCoord((RA0 + (x / 3600.0) / COSD) * u.deg, (DEC0 + y / 3600.0) * u.deg)


def _grid(n_side=140, extent=EXTENT, seed=0):
    """A star grid dense enough that a cell's wrong-pair background reaches the verify
    floor (~1 pair per 40x40 mas bin), which is the regime o028 sits in."""
    rng = np.random.default_rng(seed)
    g = np.linspace(0, extent, n_side)
    xx, yy = np.meshgrid(g, g)
    return (xx.ravel() + rng.normal(0, 0.02, xx.size),
            yy.ravel() + rng.normal(0, 0.02, yy.size))


def _shifted_field(shift_mas, seed=5, above=None):
    """Truth = a star grid; detections = the same stars displaced by ``shift_mas`` in
    Dec.  Every displaced pair is a true counterpart, so the peak is sharp and lands at
    exactly ``shift_mas`` -- the cleanest possible reading of a rigid offset.

    ``above``: displace only the stars with ``y > above`` (arcsec), i.e. a coherent
    PATCH of the field, which is the shape a real seam / misplaced visit makes.
    """
    x, y = _grid(seed=seed)
    shift = np.where(y > above, shift_mas, 0.0) if above is not None else shift_mas
    return _sc(x, y + shift / 1000.0), _sc(x, y)


# ---------------------------------------------------------------------------
# a REAL rigid offset still fails -- at every amplitude, including past the window
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shift", [1300.0, 2000.0, 2400.0])
def test_a_rigid_offset_past_half_the_window_still_fails(shift):
    """THE regression this file exists for.  1.3-2.4" is beyond WINDOW_EDGE_FRAC*MX, so
    it is exactly the population the window-edge arm looks at -- and it is a clean,
    sharp, rigid misregistration.  It must FAIL, and be reported at its true offset.

    Ungrading it on radius alone (the shape of the first attempt at #588) makes all
    three of these PASS.
    """
    det, truth = _shifted_field(shift)
    r = rf.per_cell(det, None, truth, f"rigid {shift:.0f}", **OWN)
    assert r["PASS"] is False, r
    assert r["n_fail"] > 0
    # graded, not withdrawn: the contrast layer resolves it at the base window
    assert r["n_window_edge"] == 0, r["window_edge_cells"]
    assert abs(r["worst"][0]["offset_mas"] - shift) < 100, r["worst"][:2]
    assert r["worst"][0]["peak_bg"] >= rf.FAIL_MIN_RATIO, r["worst"][:2]


def test_a_rigid_offset_beyond_the_search_window_is_measured_by_the_sweep():
    """cloudc F410M's shape: 4.06", larger than the 2.5" the base window can hold, in a
    coherent quarter of the field.  The base window has no true pair to find there, so
    its arg-max is the background and the largest offset the check can even NAME is
    2461 mas -- the alias.  The sweep re-measures at 5" and 10", finds 4060 mas, and the
    field fails ON THE TRUE NUMBER.  This is capability the check did not have before.
    """
    det, truth = _shifted_field(4060.0, above=30.0)
    r = rf.per_cell(det, None, truth, "gross 4060", **OWN)
    assert r["PASS"] is False, r
    assert r["n_edge_swept_confirmed"] > 0, r
    assert abs(r["worst"][0]["offset_mas"] - 4060.0) < 200, r["worst"][:2]
    # ... and without the sweep the true offset is not merely un-failed: it is
    # unnameable, because every number the base window can return is inside its own disk
    nos = rf.per_cell(det, None, truth, "gross 4060 unswept", sweep=False, **OWN)
    assert nos["n_window_edge"] > 0
    assert nos["PASS"] is not True                       # never a pass, either way
    assert max(c["offset_mas"] for c in nos["window_edge_cells"]) < rf.MX.to(u.mas).value
    assert max(c["offset_mas"] for c in r["worst"]) > rf.MX.to(u.mas).value


def test_an_offset_beyond_even_the_swept_window_is_never_a_pass():
    """brick-1182's shape: the whole field displaced 20.4", past every window this check
    can search.  Nothing resolves anywhere the two footprints overlap, so those cells are
    withdrawn as a 132-cell patch -- which is could-not-verify, and blocks.  A gross
    shift must never come out green, whether or not the estimator can put a number on it.

    (Measured at ``sweep_factors=(2.0,)`` to keep the test cheap; the widest window
    changes which cells resolve, not whether a 20.4" offset can be resolved at all.)
    """
    det, truth = _shifted_field(20400.0)
    r = rf.per_cell(det, None, truth, "gross 20400", sweep_factors=(2.0,), **OWN)
    assert r["PASS"] is not True, r
    assert r["n_window_edge"] > 0 and r["n_window_edge_patch"] >= rf.MIN_SEAM_CELLS, r
    assert max(r["window_edge_component_sizes"]) >= rf.MIN_SEAM_CELLS


def test_a_low_contrast_real_offset_is_caught_by_the_sweep():
    """The case the contrast layer alone would miss, and the one that makes the sweep
    load-bearing rather than decorative: a real 2" shift carried by only ~6 counterparts
    per cell inside a truth set 15x denser (the own-catalog regime -- one row per
    exposure).  Its peak clears the verify floor but not FAIL_MIN_RATIO, so the contrast
    layer cannot confirm it; the sweep does, because 2000 mas is where the peak sits at
    5" and at 10" alike.  84 cells are confirmed that way, and the field FAILs at 2000
    mas with every failing cell at peak_bg 5-6, i.e. below the confident bar.
    """
    rng = np.random.default_rng(17)
    tx, ty = rng.uniform(0, EXTENT, 2400), rng.uniform(0, EXTENT, 2400)
    nx, ny = rng.uniform(0, EXTENT, 36000), rng.uniform(0, EXTENT, 36000)
    det = _sc(tx, ty + 2.0)
    truth = _sc(np.concatenate([tx, nx]), np.concatenate([ty, ny]))
    r = rf.per_cell(det, None, truth, "low-contrast 2000", **OWN)
    assert r["PASS"] is False, r
    assert r["n_edge_swept_confirmed"] > 0, r
    assert r["worst"][0]["peak_bg"] < rf.FAIL_MIN_RATIO, r["worst"][:2]   # sub-margin
    assert abs(r["median_graded_offset_mas"] - 2000.0) < 150, r
    assert any(abs(c["offset_mas"] - 2000.0) < 150 for c in r["worst"]), r["worst"][:3]


# ---------------------------------------------------------------------------
# no tie at all: could-not-verify, never a pass
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def no_tie():
    """The gc2211_o028 shape, synthesised: detections with NO counterpart in a truth set
    10x denser.  Every pair is a chance coincidence, so the arg-max wanders the disk and
    lands mostly beyond half of it -- 'no tie exists inside 2.5 arcsec', which is what
    o028's per-exposure catalog (294704 rows against 23296 detections) produces.

    ``sweep_factors=(2.0,)`` keeps one shared fixture affordable; a background arg-max
    is refuted by the first widening (it moves from 0.71*2.5" to 0.71*5"), so the second
    window changes the cost, not the verdict.
    """
    rng = np.random.default_rng(3)
    det = _sc(rng.uniform(0, EXTENT, 6000), rng.uniform(0, EXTENT, 6000))
    truth = _sc(rng.uniform(0, EXTENT, 60000), rng.uniform(0, EXTENT, 60000))
    return rf.per_cell(det, None, truth, "no tie", sweep_factors=(2.0,), **OWN)


def test_a_field_with_no_tie_is_withdrawn_wholesale_and_never_passes(no_tie):
    """302 of 368 verified cells are withdrawn as one 298-cell patch: the check says it
    could not measure this mosaic, instead of reporting a 1.9" misregistration it never
    saw.  It is not a pass in any of its parts."""
    r = no_tie
    assert r["verified_cells"] > 0, r           # the cells do verify, as o028's did
    assert r["n_window_edge"] > 0, r
    assert r["PASS"] is not True, r
    assert r["n_window_edge_patch"] >= rf.MIN_SEAM_CELLS, r
    assert r["n_window_edge"] > 5 * r["n_fail"], r   # the withdrawn population dominates
    # the population sits where a uniform disk background puts it: p(r) ~ r dr, median
    # 0.71*MX = 1768 mas.  o028 read 1826 against that prediction.
    assert min(c["offset_mas"] for c in r["window_edge_cells"]) > (
        rf.WINDOW_EDGE_FRAC * rf.MX.to(u.mas).value)
    # every withdrawn cell carries the windows that failed to resolve it, so the record
    # says what was TRIED, not only what was concluded
    assert all(len(c["swept_windows"]) >= 2 for c in r["window_edge_cells"]), r
    # ... and the sweep moved the peak rather than reproducing it -- the whole test
    assert any(abs(c["swept_windows"][1][1] - c["swept_windows"][0][1]) > rf.SWEEP_AGREE_MAS
               for c in r["window_edge_cells"]), r["window_edge_cells"][:3]


def test_the_withdrawn_cells_are_disjoint_from_the_two_graded_reports(no_tie):
    """Four states partition the verified cells: failed, tolerated sub-margin,
    could-not-measure, and in-tolerance.  A cell must appear in exactly one, or a reader
    cannot tell what the gate concluded about it."""
    r = no_tie
    assert r["n_unconfident_highoff"] + r["n_window_edge"] <= r["verified_cells"]
    edge_pos = {(c["ra"], c["dec"]) for c in r["window_edge_cells"]}
    other = {(c["ra"], c["dec"]) for c in r["unconfident_highoff_cells"]} | {
        (c["ra"], c["dec"]) for c in r["worst"]}
    assert not (edge_pos & other), (edge_pos, other)


# ---------------------------------------------------------------------------
# the calibrated band and the strict checks are untouched
# ---------------------------------------------------------------------------

def test_the_calibrated_seam_regime_still_fails():
    """+90 mas is the amplitude #179 injected and #170 calibrated the seam axis on."""
    det, truth = _shifted_field(90.0, seed=7)
    r = rf.per_cell(det, None, truth, "90 mas", **OWN)
    assert r["PASS"] is False and r["n_fail"] > 0 and r["n_fail_seam"] > 0
    assert r["n_window_edge"] == 0


def test_the_top_of_the_graded_band_still_fails():
    """Just under the threshold (1200 mas < 0.5*MX = 1250): graded by the base window
    with no sweep needed."""
    det, truth = _shifted_field(1200.0, seed=11)
    r = rf.per_cell(det, None, truth, "1200 mas", **OWN)
    assert r["PASS"] is False and r["n_fail"] > 0
    assert r["n_window_edge"] == 0 and r["n_edge_confident"] == 0


def test_a_clean_field_reports_no_edge_cells():
    """No false tagging: a registered field grades as before, with an empty report."""
    x, y = _grid(seed=9)
    r = rf.per_cell(_sc(x, y), None, _sc(x, y), "clean", **OWN)
    assert r["PASS"] is True and r["n_fail"] == 0 and r["verified_cells"] > 0
    assert r["n_window_edge"] == 0 and r["window_edge_cells"] == []


def test_the_strict_checks_are_not_relaxed_at_all():
    """The cross-band and per-module legs run at ``fail_min_ratio = MIN_PEAK_RATIO``,
    which IS the verify floor, so every verified cell clears the contrast layer and the
    arm never fires for them.  The diagnosis behind it (an own-catalog truth 12.6x
    denser than the detection list) is specific to the own-catalog leg."""
    det, truth = _shifted_field(2000.0, seed=5)
    r = rf.per_cell(det, None, truth, "strict 2000")        # default = strict bar
    assert r["PASS"] is False and r["n_fail"] > 0
    assert r["n_window_edge"] == 0                          # nothing withdrawn
    assert r["n_edge_swept_confirmed"] == 0                 # nothing needed sweeping


# ---------------------------------------------------------------------------
# the verdict has to survive the callers: None blocks, and says why
# ---------------------------------------------------------------------------

def _stub_field(tmp_path, monkeypatch, verdict):
    """An arches-shaped two-band field whose per-cell verdict is ``verdict``."""
    from astropy.io import fits
    from astropy.wcs import WCS
    monkeypatch.setattr(rf, "BASE", str(tmp_path))
    for filt in ("F212N", "F187N"):
        p = tmp_path / "arches" / filt / "pipeline"
        p.mkdir(parents=True, exist_ok=True)
        for mod, ra in (("nrca", 266.0), ("nrcb", 266.1)):
            w = WCS(naxis=2)
            w.wcs.crpix = [128, 128]
            w.wcs.crval = [ra, -28.0]
            w.wcs.cdelt = [-1 / 3600.0, 1 / 3600.0]
            w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            fits.HDUList([fits.PrimaryHDU(),
                          fits.ImageHDU(np.ones((256, 256), "f4"), w.to_header())]
                         ).writeto(p / f"jw02045-o001_t001_nircam_clear-"
                                     f"{filt.lower()}-{mod}_i2d.fits", overwrite=True)

    def _detect(path, thr=30.0):
        n = 20
        return (SkyCoord(np.linspace(266.0, 266.01, n) * u.deg,
                         np.linspace(-28.0, -27.99, n) * u.deg), np.ones(n))

    def _per_cell(det, flux, truth, label, bright_pct=None, fail_min_ratio=None):
        return dict(label=label, PASS=verdict, n_fail=0 if verdict is not False else 3,
                    n_window_edge=17, window_edge_frac=rf.WINDOW_EDGE_FRAC,
                    window_edge_component_sizes=[9, 5, 3])

    monkeypatch.setattr(rf, "detect", _detect)
    monkeypatch.setattr(rf, "per_cell", _per_cell)
    monkeypatch.setattr(rf, "catalog_sc", lambda field, filt, view="merged": None)


def test_a_could_not_verify_check_blocks_the_field_and_names_itself(tmp_path, monkeypatch, capsys):
    """``PASS`` None must not be read as a pass by the caller chain.  It becomes an
    UNRESOLVED entry, the field verdict is None, and ``main`` returns 2 -- the exit code
    ``stage_release.py`` refuses on with "could NOT VERIFY"."""
    _stub_field(tmp_path, monkeypatch, None)
    res = rf.scan_field("arches", verbose=False, images_only=True)
    assert res["PASS"] is None, res
    assert any("could not resolve a tie" in u for u in res["unresolved"]), res["unresolved"]
    assert rf.main(["--field", "arches", "--scan", "--images-only"]) == 2


def test_a_failing_check_still_exits_one(tmp_path, monkeypatch):
    """And a measured failure is still a FAIL (exit 1), not downgraded to unverified."""
    _stub_field(tmp_path, monkeypatch, False)
    assert rf.scan_field("arches", verbose=False, images_only=True)["PASS"] is False
    assert rf.main(["--field", "arches", "--scan", "--images-only"]) == 1


def test_a_passing_field_still_exits_zero(tmp_path, monkeypatch):
    _stub_field(tmp_path, monkeypatch, True)
    assert rf.scan_field("arches", verbose=False, images_only=True)["PASS"] is True
    assert rf.main(["--field", "arches", "--scan", "--images-only"]) == 0


# ---------------------------------------------------------------------------
# the knobs
# ---------------------------------------------------------------------------

def test_window_edge_and_sweep_defaults():
    """No silent relaxation: the arm is on, it sweeps by default, and the sweep reaches
    past the base window (otherwise 'confirm by widening' cannot mean anything)."""
    sig = inspect.signature(rf.per_cell).parameters
    assert sig["window_edge_frac"].default == rf.WINDOW_EDGE_FRAC
    assert sig["sweep"].default is True
    assert sig["sweep_factors"].default == rf.SWEEP_FACTORS
    assert 0.0 < rf.WINDOW_EDGE_FRAC < 1.0
    assert min(rf.SWEEP_FACTORS) > 1.0 and max(rf.SWEEP_FACTORS) >= 4.0
    # the band graded by the base window alone spans 60 mas -> 1250 mas: the whole
    # regime the contrast and seam axes were calibrated on, 20x over OFF_MAX
    assert rf.WINDOW_EDGE_FRAC * rf.MX.to(u.mas).value > 20 * rf.OFF_MAX
    # a real tie has to reproduce to within a few histogram bins, not "anywhere"
    assert rf.SWEEP_AGREE_MAS <= 5 * rf.XBIN * 1000
