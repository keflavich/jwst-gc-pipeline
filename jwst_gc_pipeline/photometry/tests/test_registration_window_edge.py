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

    ``sweep_factors=(2.0,)`` keeps one shared fixture affordable: the full-size field
    at the shipped ``SWEEP_FACTORS`` searches 10" around 6000 detections in a truth set
    of 60000 and takes tens of minutes, which is not a CI cost.

    It is a COST reduction and nothing more.  This fixture's own numbers are therefore
    the numbers at ``(2.0,)``, and the review of #758 was right that the earlier claim
    here -- "the second window changes the cost, not the verdict" -- was an assumption:
    at the shipped default the second window confirms a fraction of these
    pure-background cells and grades them at multi-arcsecond offsets.  What the fixture
    can pin cheaply is the field-level SHAPE (withdrawn wholesale, never a pass); the
    confirmation rule at the shipped ``SWEEP_FACTORS`` is pinned directly, one cell at a
    time, by ``test_the_sweep_does_not_confirm_pure_wrong_pair_background`` and its
    real-tie counterpart below.
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



# ---------------------------------------------------------------------------
# what the sweep's confirmation actually rests on (review of #758)
# ---------------------------------------------------------------------------
#
# The rule this arm shipped with was "the smallest resolving window's offset is
# reproduced by a second resolving window".  The review of #758 showed that agreement
# is arithmetic rather than evidence, and these tests pin the statistic that replaced
# it -- on both sides, so the bar cannot be moved in either direction unnoticed.

def _one_cell_field(n_det=15, truth_per_sq_arcsec=37.5, seed=21, shift_mas=None):
    """One cell's detections plus a uniform truth set at the gc2211_o028 pair density.

    ``n_det`` is set so the cell's wrong-pair background is ~0.9 pairs per 40 mas bin --
    the regime o028 sits in (peak/background 5-7, the verify floor) and the regime the
    whole #588 argument is about.  ``shift_mas`` adds one true counterpart per detection
    at that Dec offset.
    """
    rng = np.random.default_rng(seed)
    dx = rng.uniform(-0.5, 0.5, n_det)
    dy = rng.uniform(-0.5, 0.5, n_det)
    pad = 12.0
    n_truth = int(truth_per_sq_arcsec * (2 * pad) ** 2)
    tx = rng.uniform(-pad, pad, n_truth)
    ty = rng.uniform(-pad, pad, n_truth)
    if shift_mas is not None:
        tx = np.concatenate([tx, dx])
        ty = np.concatenate([ty, dy - shift_mas / 1000.0])
    return _sc(dx, dy), _sc(tx, ty)


def _sweep_one_cell(det, truth, base_off=2400.0, base_ratio=6.0, **kw):
    """Drive ``sweep_cell_windows`` on a single cell at the SHIPPED sweep factors.

    A near-edge cell by definition has ``off > WINDOW_EDGE_FRAC * MX``, so its base
    measurement can never resolve and only the swept windows can confirm anything;
    ``base`` is that arriving reading.
    """
    base = {0: rf._PeakStats(off=base_off, ratio=base_ratio, dra_mas=0.0,
                             dde_mas=base_off, peak=base_ratio, lam=0.9,
                             n_bins=12272.0, expected=1.0)}
    return rf.sweep_cell_windows(det, truth, np.zeros(len(det), int), [0], base,
                                 pairs_per_det=736.0, **kw)


def test_two_nested_windows_agree_arithmetically_so_agreement_is_not_evidence():
    """The review's B1, as a property of the estimator rather than a sample of one.

    ``SWEEP_FACTORS`` are nested radius searches on a SHARED 40 mas bin grid, so the
    wider window's histogram equals the narrower one's bin for bin inside the narrower
    disk.  Whenever the wider window RESOLVES -- arg-max within ``WINDOW_EDGE_FRAC`` of
    its own window, i.e. inside the narrower window's disk -- it therefore reports the
    narrower window's arg-max exactly.  "The two windows agree" is then arithmetic and
    carries no information, which is why the confirmation rule no longer rests on it.
    """
    bin_mas = rf.XBIN * 1000
    # the mechanism: every window is a whole number of bins from the origin, so the
    # narrower grid's edges are a subset of the wider one's
    mx_mas = rf.MX.to(u.mas).value
    for fac in rf.SWEEP_FACTORS:
        n = (fac * mx_mas) / bin_mas
        assert abs(n - round(n)) < 1e-6, (fac, n)
    rng = np.random.default_rng(4)
    agreed = resolved = 0
    for _ in range(150):
        n = 30000                                     # a uniform wrong-pair background
        r = 10000 * np.sqrt(rng.random(n))
        th = rng.uniform(0, 2 * np.pi, n)
        dra, dde = r * np.cos(th), r * np.sin(th)
        narrow = rf._hist_peak_stats(dra[r <= 5000], dde[r <= 5000], 5000.0)
        wide = rf._hist_peak_stats(dra, dde, 10000.0)
        if wide.off > rf.WINDOW_EDGE_FRAC * 10000.0:  # the wider window did not resolve
            continue
        resolved += 1
        agreed += int(abs(wide.dra_mas - narrow.dra_mas) <= rf.SWEEP_AGREE_MAS
                      and abs(wide.dde_mas - narrow.dde_mas) <= rf.SWEEP_AGREE_MAS)
    assert resolved >= 5, resolved
    assert agreed == resolved, (agreed, resolved)     # it never once disagreed


def test_the_sweep_does_not_confirm_pure_wrong_pair_background():
    """The consequence B1 predicted, and the failure this fix exists for.

    These cells have NO counterpart at any offset -- every pair is a chance coincidence
    at o028's density -- so nothing in them may be confirmed and graded as a
    multi-arcsecond misregistration.  What rejects them is the look-elsewhere statistic:
    a background arg-max is the extreme value of that cell's own background over the
    bins the window searched, so ``expected`` is order 1 at EVERY window, while
    ``SWEEP_MAX_EXPECTED_BINS`` is 0.01.

    The second count is what makes the first mean something: these cells DO clear the
    shipped rule's conditions (peak clear of its own rim, at or above the verify floor),
    repeatedly -- so a rule that then confirms on cross-window agreement, which is
    arithmetic, confirms wrong-pair background.  Both numbers are asserted, so removing
    the look-elsewhere bar turns this red rather than merely trivially green.
    """
    confirmations, merged_rule_resolutions = [], 0
    for seed in range(12):
        det, truth = _one_cell_field(seed=100 + seed)
        confirmed, meas = _sweep_one_cell(det, truth)
        merged_rule_resolutions += sum(
            1 for (w, st) in meas[0]
            if w > rf.MX.to(u.mas).value and np.isfinite(st.off)
            and st.ratio >= rf.MIN_PEAK_RATIO and st.off <= rf.WINDOW_EDGE_FRAC * w)
        if confirmed:
            confirmations.append((seed, confirmed[0],
                                  [(w, st.off, st.ratio, st.expected) for w, st in meas[0]]))
    assert merged_rule_resolutions >= 4, merged_rule_resolutions
    assert confirmations == [], confirmations


def test_the_sweep_still_confirms_a_real_tie_beyond_the_base_window():
    """The other side of the same bar, at cloudc F410M's amplitude: a real 4.06"
    displacement, invisible to the 2.5" base window, in a truth set at the same o028
    density.  It is confirmed and graded at its true value -- so the fix is not "confirm
    less"."""
    det, truth = _one_cell_field(seed=21, shift_mas=4060.0)
    confirmed, meas = _sweep_one_cell(det, truth)
    assert 0 in confirmed, [(w, st.off, st.ratio, st.expected) for (w, st) in meas[0]]
    assert abs(confirmed[0][0] - 4060.0) < rf.SWEEP_AGREE_MAS, confirmed[0]


def test_the_swept_value_is_taken_at_the_widest_resolving_window():
    """B4's second half.  The NARROWEST resolving window is the one whose true
    counterpart is most likely to sit outside it, so it is the one most likely to be
    reading an alias.  A 4.06" tie cannot resolve at 5" (4060 > 0.5 * 5000); the graded
    value is the 10" reading."""
    det, truth = _one_cell_field(seed=21, shift_mas=4060.0)
    confirmed, meas = _sweep_one_cell(det, truth)
    at5 = [st for (w, st) in meas[0] if abs(w - 5000.0) < 1]
    assert at5 and at5[0].off > rf.WINDOW_EDGE_FRAC * 5000.0, at5
    assert abs(confirmed[0][0] - 4060.0) < rf.SWEEP_AGREE_MAS


@pytest.mark.parametrize("base_dde, confirms", [(-3000.0, True), (+3000.0, False)])
def test_the_sweep_compares_offset_vectors_not_magnitudes(base_dde, confirms):
    """The review's B4.  ``_hist_peak`` returns a hypot, so two peaks at the same RADIUS
    in OPPOSITE directions used to count as the same reading.

    The cell carries a real tie at (0, -3000) mas.  The arriving base reading is given
    the same magnitude, once with the same sign and once with the opposite one; only the
    first is the same measurement, and only the first may confirm.  ``window_edge_frac``
    is raised for this test so that the base window resolves too -- with two resolving
    windows in play there is a comparison to make, which the nested sweep factors alone
    cannot arrange (see the test above).
    """
    det, truth = _one_cell_field(seed=21, shift_mas=3000.0)
    base = {0: rf._PeakStats(off=3000.0, ratio=20.0, dra_mas=0.0, dde_mas=base_dde,
                             peak=20.0, lam=0.9, n_bins=12272.0, expected=1e-12)}
    confirmed, meas = rf.sweep_cell_windows(
        det, truth, np.zeros(len(det), int), [0], base,
        window_edge_frac=1.5, pairs_per_det=736.0)
    swept = [(w, st.dra_mas, st.dde_mas, st.expected) for (w, st) in meas[0]]
    assert (0 in confirmed) is confirms, (base_dde, confirmed, swept)


def test_the_look_elsewhere_statistic_carries_the_window_and_a_ratio_does_not():
    """Why a contrast floor calibrated at the 2.5" base window (#179) cannot be reused
    at 5" and 10": the same background peak COUNT is less surprising at a wider window,
    because more bins were searched.  ``ratio`` does not know that; ``expected`` does."""
    rng = np.random.default_rng(8)
    n = 20000
    r = 10000 * np.sqrt(rng.random(n))
    th = rng.uniform(0, 2 * np.pi, n)
    dra, dde = r * np.cos(th), r * np.sin(th)
    inner = r <= rf.MX.to(u.mas).value
    base = rf._hist_peak_stats(dra[inner], dde[inner], rf.MX.to(u.mas).value)
    wide = rf._hist_peak_stats(dra, dde, 10000.0)
    assert wide.n_bins > 10 * base.n_bins
    # both are the extreme value of a background: order 1 expected bins, at both windows
    assert 0.01 < base.expected < 100 and 0.01 < wide.expected < 100, (base, wide)
    # a real tie is nowhere near it
    st = rf._hist_peak_stats(np.concatenate([dra, np.zeros(60)]),
                             np.concatenate([dde, np.full(60, 4060.0)]), 10000.0)
    assert st.expected < rf.SWEEP_MAX_EXPECTED_BINS, st


# ---------------------------------------------------------------------------
# one region, one quorum (review B5)
# ---------------------------------------------------------------------------

def _quorate(mask):
    keep, _ = rf.seam_mask(mask, min_cells=rf.MIN_SEAM_CELLS)
    return bool(keep.any())


def test_a_region_split_across_the_two_axes_is_counted_once():
    """The review's B5.  One physical misregistration lands partly in cells the estimator
    MEASURED (``highoff``) and partly in cells it COULD NOT (``edge``).  Testing each
    half against ``MIN_SEAM_CELLS`` separately leaves a 2 + 2 region under the quorum on
    both axes at once, so it blocks on neither.  Counted on the union it is one
    component of 4, and it blocks.

    Driven through ``could_not_verify_patch`` -- the function ``per_cell`` calls, not a
    copy of it.  A synthetic field whose cells happened to split exactly 2/2 would pin
    the field generator instead of the rule.
    """
    highoff = np.zeros((6, 6), bool)
    edge = np.zeros((6, 6), bool)
    highoff[2, 2] = highoff[2, 3] = True
    edge[3, 2] = edge[3, 3] = True                     # 4-connected to the pair above
    assert not _quorate(highoff), "2 measured cells alone are under the quorum"
    assert not _quorate(edge), "2 withdrawn cells alone are under the quorum"
    edge_patch, edge_sizes, region, region_sizes = rf.could_not_verify_patch(
        highoff, edge, min_cells=rf.MIN_SEAM_CELLS)
    assert edge_patch.any(), "a 2+2 region blocks on neither axis if each is counted alone"
    assert region_sizes == [4] and edge_sizes == [4]
    assert int(region.sum()) == 4


def test_the_union_quorum_does_not_invent_measured_failures():
    """The union arms the quorum; it does not move cells between axes.  A withdrawn cell
    never becomes a measured failure because it touches one -- the gate would then be
    reporting a misregistration it did not measure, which is the defect #588 was opened
    for.  So the SEAM (fail) axis is still counted on ``highoff`` alone."""
    highoff = np.zeros((6, 6), bool)
    edge = np.zeros((6, 6), bool)
    highoff[2, 2] = True                               # one measured cell ...
    edge[2, 3] = edge[2, 4] = edge[3, 4] = True        # ... beside a quorate withdrawn patch
    edge_patch, _sizes, region, _rsizes = rf.could_not_verify_patch(
        highoff, edge, min_cells=rf.MIN_SEAM_CELLS)
    assert edge_patch.sum() == 3 and not edge_patch[2, 2]
    seam, seam_sizes = rf.seam_mask(highoff, min_cells=rf.MIN_SEAM_CELLS)
    assert not seam.any() and seam_sizes == []         # the lone measured cell is not failed
    assert region[2, 2]                                # though it is inside the region


def test_the_sweep_bar_defaults_are_not_silently_relaxed():
    sig = inspect.signature(rf.sweep_cell_windows).parameters
    assert sig["max_expected_bins"].default == rf.SWEEP_MAX_EXPECTED_BINS
    assert 0 < rf.SWEEP_MAX_EXPECTED_BINS <= 0.05     # two orders under a background max
