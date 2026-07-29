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
