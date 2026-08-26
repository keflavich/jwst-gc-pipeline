"""Hermetic tests for the same-star external-reference arbiter added to
check_interframe_overlap (the 2026-07 fix for the sparse 2221 nrca-long|nrcb-long
false FAIL).

Context: the inter-frame overlap gate could not measure the thin 2221 LW
inter-module overlap reference-free (0 mutual-coverage tiles), and its old
per-cell offset-histogram vs VIRAC (`measure_offset_grid`) was fooled by the
dense-field wrong-pair bias -- it read a 58" worst cell where the SAME-STAR tie
of the identical data is 3 mas.  `_samestar_ref_grid` uses `local_residual_map`
(real matched pairs, per-cell significance), which is density-immune, and gates a
gross / window-swept global tie directly.  These tests pin its behaviour on
controlled synthetic star fields -- no data files.

The second half of the file is the issue #174 hardening: the arbiter's per-cell
tolerance (it was being handed the GROSS global ceiling), its cell size (one 30"
cell averages away any sliver narrower than itself), scoping to the deferred
PAIR's own overlap footprint instead of one field-wide boolean, the multi-radius
NN-collapse check, and the ``ok=False`` global tie that used to escape a blocking
gate as an uncaught ``GlobalTieNotVerifiedError``.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord

from jwst_gc_pipeline.photometry.astrometry_offsets import GlobalTieNotVerifiedError

_spec = importlib.util.spec_from_file_location(
    "check_interframe_overlap",
    Path(__file__).with_name("check_interframe_overlap.py"))
ck = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ck)

RA0, DEC0 = 266.5, -28.7
COSD = float(np.cos(np.deg2rad(DEC0)))


def _field(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    ra = RA0 + rng.uniform(-0.02, 0.02, n)
    dec = DEC0 + rng.uniform(-0.02, 0.02, n)
    return ra, dec


def _sc(ra, dec):
    return SkyCoord(ra * u.deg, dec * u.deg)


def _shift(ra, dra_mas, mask=None):
    r = ra.copy()
    m = np.ones(len(ra), bool) if mask is None else mask
    r[m] = ra[m] + dra_mas / 3.6e6 / COSD
    return r


def test_clean_field_is_clean():
    ra, dec = _field()
    ref = _sc(ra, dec)
    g = ck._samestar_ref_grid(ref, ref, max_off_mas=80.0)
    assert g["clean"] and g["worst_off_mas"] == 0


def test_gross_coherent_offset_is_dirty():
    """A gross rigid offset (brick-1182 v001 ~20" class) -- the global tie gates it."""
    ra, dec = _field(seed=1)
    ref = _sc(ra, dec)
    src = _sc(_shift(ra, 20000.0), dec)     # 20 arcsec
    g = ck._samestar_ref_grid(src, ref, max_off_mas=80.0)
    assert not g["clean"] and g["worst_off_mas"] > 80.0


def test_moderate_coherent_offset_is_dirty():
    """A coherent 150 mas offset > max_off_mas is caught by the global tie."""
    ra, dec = _field(seed=2)
    ref = _sc(ra, dec)
    src = _sc(_shift(ra, 150.0), dec)
    g = ck._samestar_ref_grid(src, ref, max_off_mas=80.0)
    assert not g["clean"]


def test_localized_full_population_seam_is_dirty():
    """A localized region shifted 150 mas (whole cell population) is flagged by the
    per-cell same-star residual map -- the sensitivity the arbiter must keep."""
    ra, dec = _field(seed=3)
    ref = _sc(ra, dec)
    corner = (ra > RA0) & (dec > DEC0)
    src = _sc(_shift(ra, 150.0, corner), dec)
    g = ck._samestar_ref_grid(src, ref, max_off_mas=80.0)
    assert not g["clean"]


def test_within_tolerance_offset_is_clean():
    """A coherent offset below max_off_mas is within tolerance -> clean."""
    ra, dec = _field(seed=4)
    ref = _sc(ra, dec)
    src = _sc(_shift(ra, 40.0), dec)         # 40 mas < 80
    g = ck._samestar_ref_grid(src, ref, max_off_mas=80.0)
    assert g["clean"]


# ---------------------------------------------------------------------------------
# Issue #174 hardening.  Every test below FAILED against the pre-#174 arbiter.
#
# All of these carry per-star JITTER, unlike the tests above.  With exact synthetic
# positions every cell's MAD is zero, so its standard error is zero and
# ``local_residual_map``'s significance requirement (|mean| > nsigma*sem) can never
# be met -- a noiseless field cannot exercise a significance-gated flag at all.
# ---------------------------------------------------------------------------------

JITTER_MAS = 8.0


def _noisy(ra, dec, seed, jitter_mas=JITTER_MAS):
    """Detections of the same stars, with a realistic per-star measurement jitter."""
    rng = np.random.default_rng(seed + 9973)
    return (ra + rng.normal(0, jitter_mas, len(ra)) / 3.6e6 / COSD,
            dec + rng.normal(0, jitter_mas, len(dec)) / 3.6e6)


def _quadrant_case(mas, seed=21, n=4000):
    ra, dec = _field(n=n, seed=seed)
    ref = _sc(ra, dec)
    sra, sdec = _noisy(ra, dec, seed)
    q = (ra > RA0) & (dec > DEC0)
    sra[q] += mas / 3.6e6 / COSD
    return _sc(sra, sdec), ref


@pytest.mark.parametrize("mas", [55.0, 79.0])
def test_quadrant_seam_between_tol_and_the_gross_ceiling_is_dirty(mas):
    """#174 (1): the arbiter was called with ``max_off_mas=80`` and passed it
    straight through as the PER-CELL tolerance, so nothing between the gate's own
    30 mas and 80 mas was resolvable -- 55 and 79 mas quadrant seams both read
    clean.  The two thresholds do different jobs."""
    src, ref = _quadrant_case(mas)
    g = ck._samestar_ref_grid(src, ref, max_off_mas=80.0)
    assert g["measurable"], g["reason"]
    assert not g["clean"], f"{mas} mas seam read clean: {g}"


def test_percell_tolerance_is_the_gate_tolerance_not_the_gross_ceiling():
    """The same data, judged at the two thresholds: the gross global ceiling must
    not be reusable as the per-cell tolerance."""
    src, ref = _quadrant_case(55.0)
    tight = ck._samestar_ref_grid(src, ref, max_off_mas=80.0)          # tol = TOL_MAS
    loose = ck._samestar_ref_grid(src, ref, max_off_mas=80.0, tol_mas=80.0)
    assert not tight["clean"] and loose["clean"]
    assert ck.TOL_MAS < ck.GRID_MAX_OFF_MAS


def test_strip_narrower_than_the_old_single_cell_is_dirty():
    """#174 (2): with one 30" cell, a strip narrower than the cell is a minority of
    every cell it touches and the per-cell MEDIAN averages it away -- a 14" strip
    shifted 150 mas read clean with worst=0.0.  The ladder resolves it at 16"."""
    seed = 22
    ra, dec = _field(seed=seed)
    ref = _sc(ra, dec)
    sra, sdec = _noisy(ra, dec, seed)
    strip = (dec > DEC0) & (dec < DEC0 + 14.0 / 3600.0)
    sra[strip] += 150.0 / 3.6e6 / COSD
    g = ck._samestar_ref_grid(_sc(sra, sdec), ref, max_off_mas=80.0)
    assert g["measurable"], g["reason"]
    assert not g["clean"], g
    assert g["worst_off_mas"] > ck.TOL_MAS


def test_a_scale_without_enough_stars_is_unmeasurable_not_clean():
    """#174 (2), the other half: a cell too small to hold ``min_stars`` must report
    NOTHING, not "clean".  Finer scales contribute no cells at this density, and a
    field too sparse for ANY scale is could-not-verify."""
    ra, dec = _field(seed=23)
    ref = _sc(ra, dec)
    g = ck._samestar_ref_grid(_sc(*_noisy(ra, dec, 23)), ref, max_off_mas=80.0)
    # 0.22 stars/arcsec^2: a 4" cell holds ~3.5 -- those scales must stay silent
    fine = [s for s in g["scales"] if s["cell_arcsec"] <= 4.0]
    assert fine and all(s["n_measured"] == 0 for s in fine), g["scales"]
    assert g["clean"] and g["measurable"]        # the coarse scales still measure

    sparse_ra, sparse_dec = _field(n=150, seed=24)
    sref = _sc(sparse_ra, sparse_dec)
    gs = ck._samestar_ref_grid(_sc(*_noisy(sparse_ra, sparse_dec, 24)), sref,
                               max_off_mas=80.0)
    assert not gs["measurable"] and not gs["clean"]
    assert "no residual-map cell" in gs["reason"]


def _sliver_pair(seed=31, n=4000, sliver_arcsec=4.0, seam_mas=0.0, seam_frac=1.0):
    """Two exposure groups that overlap only in a thin Dec sliver -- the geometry of
    every pair this arbiter is authoritative for (0 mutual-coverage tiles).  The
    seam is applied to group B INSIDE the overlap only."""
    ra, dec = _field(n=n, seed=seed)
    ref = _sc(ra, dec)
    half = sliver_arcsec / 2.0 / 3600.0
    in_a, in_b = dec <= DEC0 + half, dec >= DEC0 - half
    a_ra, a_dec = _noisy(ra[in_a], dec[in_a], seed)
    b_ra, b_dec = _noisy(ra[in_b], dec[in_b], seed + 1)
    if seam_mas:
        band = b_dec <= DEC0 + half
        if seam_frac < 1.0:
            band &= b_ra > np.quantile(b_ra, 1.0 - seam_frac)
        b_ra[band] += seam_mas / 3.6e6 / COSD
    return ref, _sc(a_ra, a_dec), _sc(b_ra, b_dec)


def test_field_wide_map_is_blind_to_the_sliver_the_pair_arbiter_resolves():
    """#174 (3): ``ext_fail`` is one field-wide boolean, so one clean map cleared
    EVERY deferred pair -- and a 4" sliver seam is invisible field-wide at any cell
    size the star density supports.  Scoped to the pair's own overlap footprint the
    same seam is the whole population, and is measured."""
    ref, a, b = _sliver_pair(sliver_arcsec=4.0, seam_mas=150.0)
    allsrc = SkyCoord(np.concatenate([a.ra.deg, b.ra.deg]) * u.deg,
                      np.concatenate([a.dec.deg, b.dec.deg]) * u.deg)
    field = ck._samestar_ref_grid(allsrc, ref, max_off_mas=80.0)
    assert field["clean"], "field-wide map unexpectedly resolved the sliver"

    pair = ck._samestar_pair_footprint(a, b, ref, field["global_tie"])
    assert pair["measurable"], pair["reason"]
    assert not pair["clean"], pair
    assert pair["worst_off_mas"] > ck.TOL_MAS


def test_pair_footprint_arbiter_clears_a_registered_sliver():
    """The regression direction: a deferred pair that IS registered must come back
    clean on its own footprint (the brick F405N case, ~10 mas)."""
    ref, a, b = _sliver_pair(sliver_arcsec=4.0, seam_mas=0.0)
    field = ck._samestar_ref_grid(
        SkyCoord(np.concatenate([a.ra.deg, b.ra.deg]) * u.deg,
                 np.concatenate([a.dec.deg, b.dec.deg]) * u.deg),
        ref, max_off_mas=80.0)
    pair = ck._samestar_pair_footprint(a, b, ref, field["global_tie"])
    assert pair["measurable"] and pair["clean"], pair
    assert pair["n_common"] >= ck.SAMESTAR_MIN_STARS


def test_nn_collapsed_minority_in_the_footprint_is_not_reported_clean():
    """#174 (4) / the issue's own case: a MINORITY of one frame shifted by ~the
    match radius pairs with the wrong neighbour at 0.3" -- its residuals scatter,
    the median stays at zero and the pair reads clean.  Shrink the radius and those
    matches simply disappear: the two frames then keep very different fractions of
    their same-star matches, which no offset histogram is needed to see."""
    ref, a, b = _sliver_pair(seed=33, sliver_arcsec=6.0, seam_mas=300.0,
                             seam_frac=0.4)
    field = ck._samestar_ref_grid(
        SkyCoord(np.concatenate([a.ra.deg, b.ra.deg]) * u.deg,
                 np.concatenate([a.dec.deg, b.dec.deg]) * u.deg),
        ref, max_off_mas=80.0)
    pair = ck._samestar_pair_footprint(a, b, ref, field["global_tie"])
    assert not pair["clean"], pair
    # the magnitude test alone does NOT see it -- the median is carried by the
    # unshifted majority; it is the radius dependence that does
    widest = pair["per_radius"][-1]
    assert widest["off_mas"] < ck.TOL_MAS, widest


def test_low_contrast_global_tie_is_a_verdict_not_a_traceback(monkeypatch):
    """``local_residual_map`` raises unless the global tie has ok=True, swept=False
    AND off << match radius.  The arbiter checked the last two only, so a
    low-contrast tie (``ok=False``, not swept, small offset -- the combination a
    star field with no real tie produces) escaped a BLOCKING gate as an uncaught
    ``GlobalTieNotVerifiedError``.  It must be a could-not-verify verdict."""
    ra, dec = _field(seed=41)
    a = _sc(*_noisy(ra, dec, 41))
    ref = _sc(ra, dec)
    weak = dict(dra=5.0, ddec=5.0, off=7.07, ok=False, swept=False, contrast=1.4,
                npairs=500, n_peak=12, window_arcsec=3.0)
    monkeypatch.setattr(ck, "measure_offset", lambda *args, **kw: weak)

    g = ck._samestar_ref_grid(a, ref, max_off_mas=80.0)     # must not raise
    assert not g["measurable"] and not g["clean"]
    assert "NOT verified" in g["reason"], g["reason"]
    # ... and the raise it is standing in front of is real
    with pytest.raises(GlobalTieNotVerifiedError):
        ck.local_residual_map(a, ref, weak, cell_arcsec=30.0, min_stars=20)


def test_global_tie_too_close_to_the_match_radius_is_could_not_verify(monkeypatch):
    """Third precondition, same class: a tie that is not << the match radius makes
    matched pairs ambiguous, and the library raises.  Verdict, not traceback."""
    ra, dec = _field(seed=43)
    a = _sc(*_noisy(ra, dec, 43))
    ref = _sc(ra, dec)
    near = dict(dra=140.0, ddec=0.0, off=140.0, ok=True, swept=False, contrast=40.0,
                npairs=500, n_peak=120, window_arcsec=3.0)
    monkeypatch.setattr(ck, "measure_offset", lambda *args, **kw: near)
    g = ck._samestar_ref_grid(a, ref, max_off_mas=300.0)
    assert not g["measurable"] and not g["clean"]
    assert "match radius" in g["reason"], g["reason"]


# --- issue #403: the 20-star floor is a COUNT, not a PRECISION ----------------
# The arbiter demands significance to FAIL (``off_mas > nsigma * sem_mas``) and
# demanded nothing to PASS, so a pair measured too coarsely to resolve a
# tolerance-sized seam was protected from flagging twice over -- once by the noise
# and once by that clause -- and fell straight through to "clean".  These pin the
# missing half: a clean verdict requires that a seam AT the tolerance could have
# been flagged.
_ZERO_TIE = dict(dra=0.0, ddec=0.0, off=0.0, ok=True, swept=False, contrast=50.0,
                 npairs=500, n_peak=100, window_arcsec=3.0)


def _footprint_pair(n_stars, jitter_mas, seam_mas=0.0, seed=101, box_arcsec=20.0):
    """A mutual-coverage footprint holding ``n_stars`` reference stars, each
    detected once by group A and once by group B with ``jitter_mas`` per-axis
    measurement noise.  ``seam_mas`` is a rigid RA shift applied to B only, i.e.
    the inter-frame misregistration this arbiter exists to catch.  ``n_stars`` is
    well above ``SAMESTAR_MIN_STARS`` and the jitter is small against the smallest
    match radius, so every radius keeps its matches and the existing
    spread-across-radii and keep-ratio mitigations do NOT fire: what is left is
    precision alone."""
    rng = np.random.default_rng(seed)
    half = box_arcsec / 2.0 / 3600.0
    ra = RA0 + rng.uniform(-half, half, n_stars) / COSD
    dec = DEC0 + rng.uniform(-half, half, n_stars)
    dra_a = rng.normal(0, jitter_mas, n_stars) / 3.6e6 / COSD
    ddec_a = rng.normal(0, jitter_mas, n_stars) / 3.6e6
    dra_b = rng.normal(0, jitter_mas, n_stars) / 3.6e6 / COSD
    ddec_b = rng.normal(0, jitter_mas, n_stars) / 3.6e6
    a = _sc(ra + dra_a, dec + ddec_a)
    b = _sc(ra + dra_b + seam_mas / 3.6e6 / COSD, dec + ddec_b)
    return _sc(ra, dec), a, b


def test_a_real_seam_lost_in_the_per_star_scatter_is_not_reported_clean():
    """The case in the issue, with the seam actually present: 60 stars at ~60 mas
    per-star scatter measure a REAL 60 mas seam as 55 mas +- 18, which is below
    the 3-sigma the flag rule requires, so the pair used to come back CLEAN with a
    misregistration in it.  It must be could-not-verify."""
    ref, a, b = _footprint_pair(n_stars=60, jitter_mas=60.0, seam_mas=60.0)
    v = ck._samestar_pair_footprint(a, b, ref, _ZERO_TIE)
    assert v["n_common"] >= ck.SAMESTAR_MIN_STARS, v        # the count floor is met
    assert v["worst_off_mas"] > ck.TOL_MAS, v               # the seam IS in the data
    assert 3.0 * v["sem_mas"] > ck.TOL_MAS, v               # and cannot be resolved
    assert not v["clean"] and not v["measurable"], v
    assert "could not have been flagged" in v["reason"], v["reason"]


def test_coarse_astrometry_with_no_seam_is_could_not_verify_not_clean():
    """Same geometry, no seam: still could-not-verify, because the measurement
    could not have SEEN one.  A pass has to mean something."""
    ref, a, b = _footprint_pair(n_stars=60, jitter_mas=60.0, seam_mas=0.0)
    v = ck._samestar_pair_footprint(a, b, ref, _ZERO_TIE)
    assert v["n_common"] >= ck.SAMESTAR_MIN_STARS, v
    assert not v["clean"] and not v["measurable"], v
    assert "could not have been flagged" in v["reason"], v["reason"]


def test_precise_stars_still_clear_the_pair():
    """The regression direction: the same geometry measured well enough to resolve
    a tolerance-sized seam is still CLEAN, and now records its precision."""
    ref, a, b = _footprint_pair(n_stars=60, jitter_mas=8.0)
    v = ck._samestar_pair_footprint(a, b, ref, _ZERO_TIE)
    assert v["clean"] and v["measurable"], v
    assert 3.0 * v["sem_mas"] < ck.TOL_MAS, v
    assert "resolvable" in v["reason"], v["reason"]


def test_a_resolvable_seam_still_fails():
    """The measurability condition guards the PASS only: a seam the measurement CAN
    resolve is still a FAIL with measurable=True, so the pair still blocks."""
    ref, a, b = _footprint_pair(n_stars=60, jitter_mas=8.0, seam_mas=150.0)
    v = ck._samestar_pair_footprint(a, b, ref, _ZERO_TIE)
    assert v["measurable"] and not v["clean"], v
    assert v["worst_off_mas"] > ck.TOL_MAS, v


def test_the_precision_requirement_is_the_only_thing_holding_the_noisy_pair(monkeypatch):
    """With the condition switched off the noisy pair passes every OTHER check --
    which is what makes it the missing half rather than a duplicate of the
    spread-across-radii or keep-ratio mitigations."""
    ref, a, b = _footprint_pair(n_stars=60, jitter_mas=60.0, seam_mas=60.0)
    monkeypatch.setattr(ck, "SAMESTAR_REQUIRE_RESOLVABLE", False)
    v = ck._samestar_pair_footprint(a, b, ref, _ZERO_TIE)
    assert v["clean"] and v["measurable"], v
    assert 3.0 * v["sem_mas"] > ck.TOL_MAS, v


# --- issue #411: a gross global tie must beat its RUNNER-UP, not just a typical bin
def _gross(margin, off_mas=378.0):
    return dict(dra=off_mas, ddec=0.0, off=off_mas, ok=True, swept=False,
                contrast=546.0, peak_margin=margin, npairs=50000, n_peak=546,
                window_arcsec=3.0)


def test_a_gross_tie_with_no_margin_is_could_not_verify(monkeypatch):
    """w51 F140M: the pooled tie read 378 mas at contrast 546 with ``ok=True``
    while all 64 of that filter's frames sit within 24 mas of the same catalogue.
    The winning bin beat the true zero bin by 1.7%, so the arg-max was a coin
    flip.  ``contrast`` measures the peak against a TYPICAL bin (median non-empty
    bin = 1 pair over a 3" window) and cannot see that; the verdict must be
    could-not-verify rather than a block."""
    ra, dec = _field(seed=51)
    a = _sc(*_noisy(ra, dec, 51))
    ref = _sc(ra, dec)
    monkeypatch.setattr(ck, "measure_offset", lambda *args, **kw: _gross(1.02))
    v = ck._samestar_ref_grid(a, ref, max_off_mas=80.0)
    assert not v["clean"], v
    assert not v["measurable"], v          # NOT a measured absolute-frame offset
    assert "runner-up" in v["reason"], v["reason"]


def test_a_gross_tie_that_beats_its_runner_up_still_blocks(monkeypatch):
    """The regression direction: a real gross offset has one spot and no rival, so
    it still comes back measurable=True and still fails the field."""
    ra, dec = _field(seed=53)
    a = _sc(*_noisy(ra, dec, 53))
    ref = _sc(ra, dec)
    monkeypatch.setattr(ck, "measure_offset",
                        lambda *args, **kw: _gross(float("inf"), off_mas=20400.0))
    v = ck._samestar_ref_grid(a, ref, max_off_mas=80.0)
    assert v["measurable"] and not v["clean"], v
    assert "global tie" in v["reason"], v["reason"]


def test_a_gross_tie_from_a_record_with_no_margin_still_blocks(monkeypatch):
    """Fail-closed on the missing key: a result dict from an older writer carries
    no ``peak_margin``, and the absence of the diagnostic must not turn a gross
    reading into could-not-verify."""
    ra, dec = _field(seed=55)
    a = _sc(*_noisy(ra, dec, 55))
    ref = _sc(ra, dec)
    stale = _gross(1.02)
    del stale["peak_margin"]
    monkeypatch.setattr(ck, "measure_offset", lambda *args, **kw: stale)
    v = ck._samestar_ref_grid(a, ref, max_off_mas=80.0)
    assert v["measurable"] and not v["clean"], v
