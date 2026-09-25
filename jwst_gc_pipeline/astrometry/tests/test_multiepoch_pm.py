"""Synthetic-field tests for the JWST x JWST affine-tie proper-motion path
(multiepoch_pm.affine_tie / build_pm_catalog_2epoch, PR #140).

No real catalogs needed: a random tangent-plane star field stands in for
``src``, and a linear (or non-linear, for the last test) transform of it
stands in for ``ref``. This is what the review on PR #140 flagged as
missing -- 834 lines with no coverage.
"""
import numpy as np
import pytest
from astropy.coordinates import SkyCoord, concatenate
import astropy.units as u

from jwst_gc_pipeline.astrometry.multiepoch_pm import (
    affine_tie, build_pm_catalog_2epoch, TieNotVerifiedError)

CENTER = SkyCoord(266.5 * u.deg, -28.5 * u.deg)
RNG = np.random.default_rng(20260918)


def _sc_from_xy(x, y, center=CENTER):
    """Inverse of tangent_xy: tangent-plane offsets (arcsec) -> SkyCoord."""
    ra = center.ra.deg + (x / 3600.0) / np.cos(center.dec.rad)
    dec = center.dec.deg + y / 3600.0
    return SkyCoord(ra * u.deg, dec * u.deg)


def _random_field(n, halfwidth_arcsec=60.0):
    x = RNG.uniform(-halfwidth_arcsec, halfwidth_arcsec, n)
    y = RNG.uniform(-halfwidth_arcsec, halfwidth_arcsec, n)
    mag = RNG.uniform(-10, -5, n)  # bright instrumental mags, like real usage
    return x, y, mag


def test_affine_tie_recovers_a_known_distortion():
    """A field distorted by a KNOWN offset+rotation+scale+shear should have
    that distortion recovered by affine_tie to within its own noise floor."""
    # Sparse enough that the true injected offset (~0.2") stays far below the
    # field's own mean nearest-neighbor spacing -- otherwise match_to_catalog_
    # sky's nearest-neighbor pairing confuses the true counterpart with a
    # random crowding neighbor, which is a statement about this synthetic
    # field's density, not about affine_tie.
    n = 1500
    sx, sy, mag = _random_field(n, halfwidth_arcsec=400.0)
    # affine_tie fits the DISPLACEMENT dx = rx - sx = A0 + A1*x + A2*y, so a
    # "scale ~1.0003" or "identity + shear" map means A1/B2 are the small
    # DEVIATIONS from identity (~1e-4), not ~1 -- an true A1 of 1.0003 would
    # inject a ~x-sized (hundreds of arcsec) offset, not a plate-scale-sized
    # one. A2/B1 are a genuine SHEAR term a shift+rotate+scale-only model
    # could not represent.
    A_true = np.array([0.150, 0.00030, -0.00040])   # offset(as), scale dev., shear
    B_true = np.array([-0.080, 0.00025, -0.00020])
    dx = A_true[0] + A_true[1] * sx + A_true[2] * sy
    dy = B_true[0] + B_true[1] * sx + B_true[2] * sy
    noise = 0.001  # 1 mas centroiding noise, both epochs' worth combined
    rx = sx + dx + RNG.normal(0, noise, n)
    ry = sy + dy + RNG.normal(0, noise, n)

    src_sc = _sc_from_xy(sx, sy)
    ref_sc = _sc_from_xy(rx, ry)
    _, diag = affine_tie(src_sc, mag, ref_sc, mag, magcut=0, match_radius=1.0)

    # All 1500 matched correctly (n_match == n here), and with NO real
    # outliers in this synthetic field, the robust (median+MAD) clip should
    # keep nearly all of them. This is a regression guard for a real bug this
    # test caught: the original plain-std clip (nsigma * std of the
    # shrinking "keep" subset, re-evaluated each of 3 rounds) kept only
    # ~40% here despite every match being genuinely good -- classic
    # iterative sigma-clip bias. Coefficients came out correct either way
    # (checked below), but n_kept/n_match is a real quality diagnostic
    # elsewhere in this codebase and was badly underestimating it.
    assert diag['n_match'] == n
    assert diag['n_kept'] > 0.95 * n
    A, B = np.array(diag['A']), np.array(diag['B'])
    # offset terms: mas-level tolerance; linear terms: per-arcsec, so need
    # tighter relative tolerance to mean the same absolute precision over the
    # ~60" field radius used here.
    assert abs(A[0] - A_true[0]) < 0.002 and abs(B[0] - B_true[0]) < 0.002
    assert abs(A[1] - A_true[1]) < 2e-4 and abs(A[2] - A_true[2]) < 2e-4
    assert abs(B[1] - B_true[1]) < 2e-4 and abs(B[2] - B_true[2]) < 2e-4
    assert diag['rms_resid_mas'] < 5 * noise * 1e3


def test_per_star_pm_survives_the_tie():
    """A random, star-specific (non-linear-in-position) proper motion is NOT
    a linear function of position, so it must survive the affine tie and
    come out of build_pm_catalog_2epoch close to its injected value."""
    pytest.importorskip('flystar')  # build_pm_catalog_2epoch imports it lazily
    n = 3000
    # Sparse enough that no two of the n random positions land within
    # match_radius of each other by chance -- crowding collisions would drop
    # rows from pm and break the direct index-aligned comparison below.
    sx, sy, mag = _random_field(n, halfwidth_arcsec=300.0)
    dt = 2.0
    # random, uncorrelated-with-position PM per star (mas/yr) -> arcsec over dt
    pm_ra_true = RNG.normal(0, 4.0, n)   # mas/yr
    pm_dec_true = RNG.normal(0, 4.0, n)
    rx = sx + pm_ra_true * dt / 1000.0
    ry = sy + pm_dec_true * dt / 1000.0
    # Also apply a linear distortion the tie must remove, so this is a
    # realistic combined case, not just a trivial no-op tie.
    rx = rx + 0.05 + 0.0002 * sx
    ry = ry + -0.03 + 0.0003 * sy

    src_sc, ref_sc = _sc_from_xy(sx, sy), _sc_from_xy(rx, ry)
    src = dict(sc=src_sc, ex=np.full(n, 0.003), ey=np.full(n, 0.003),
              mag=mag, flux=10 ** (-0.4 * mag), epoch=2024.0, n=n)
    ref = dict(sc=ref_sc, ex=np.full(n, 0.003), ey=np.full(n, 0.003),
              mag=mag, flux=10 ** (-0.4 * mag), epoch=2024.0 + dt, n=n)
    ref_sc_tied, _ = affine_tie(ref['sc'], ref['mag'], src['sc'], src['mag'],
                                magcut=0, match_radius=1.0)
    ref = dict(ref, sc=ref_sc_tied)

    pm = build_pm_catalog_2epoch(src, ref, match_radius=1.0, err_cap_mas=1e6)
    # A handful of the 3000 random positions can land within match_radius of
    # each other by chance (crowding), failing the mutual-NN check -- that is
    # a property of this random field, not of build_pm_catalog_2epoch.
    assert len(pm) >= 0.99 * n
    # Per-star recovery: with only 2 epochs and no position noise here,
    # recovery should be tight -- a few mas/yr from residual tie coupling.
    resid_ra = pm['pm_ra'] - pm_ra_true
    resid_dec = pm['pm_dec'] - pm_dec_true
    assert np.std(resid_ra) < 0.5 and np.std(resid_dec) < 0.5


def test_affine_tie_absorbs_a_coherent_linear_velocity_field():
    """Documents the caveat in affine_tie's docstring as an executable test:
    a BULK/rotational proper motion is linear in position to first order, so
    the tie removes it (by construction) same as a real plate-scale error --
    the catalog it feeds is relative, not absolute."""
    pytest.importorskip('flystar')  # build_pm_catalog_2epoch imports it lazily
    n = 3000
    sx, sy, mag = _random_field(n)
    dt = 2.0
    # A coherent solid-body-rotation-like field: linear in (x,y), no per-star
    # randomness, amplitude comparable to the median found on real Sgr B2 x
    # GC Treasury data (~5 mas/yr).
    omega = 5e-5  # arcsec/arcsec/yr -- rotation rate producing ~a few mas/yr
                  # of tangential PM at the ~60" field edge used here
    pm_ra_true = -omega * sy * 1000.0   # mas/yr, linear in y
    pm_dec_true = omega * sx * 1000.0   # mas/yr, linear in x
    rx = sx + pm_ra_true * dt / 1000.0
    ry = sy + pm_dec_true * dt / 1000.0
    # Positions stay EXACTLY noise-free on purpose -- the point of this test
    # is that a purely linear field is fully degenerate with the affine
    # model and gets absorbed to numerical precision. Only flux gets
    # realistic between-epoch noise (as any two independent reductions
    # would have): without it every pair's magnitude is bit-identical by
    # construction (same `mag` array reused), which would look exactly like
    # jwst-gc-pipeline#958's literal duplicate-row contamination to
    # affine_tie's duplicate-row check even though nothing here is
    # duplicated -- see test_affine_tie_refuses_literal_duplicate_row_contamination.
    ref_mag = mag + RNG.normal(0, 0.01, n)

    src_sc, ref_sc = _sc_from_xy(sx, sy), _sc_from_xy(rx, ry)
    ref_sc_tied, diag = affine_tie(ref_sc, ref_mag, src_sc, mag, magcut=0, match_radius=1.0)
    assert diag['dup_row_suspect_fraction'] < 0.02

    src = dict(sc=src_sc, ex=np.full(n, 0.003), ey=np.full(n, 0.003),
              mag=mag, flux=10 ** (-0.4 * mag), epoch=2024.0, n=n)
    ref = dict(sc=ref_sc_tied, ex=np.full(n, 0.003), ey=np.full(n, 0.003),
              mag=ref_mag, flux=10 ** (-0.4 * ref_mag), epoch=2024.0 + dt, n=n)
    pm = build_pm_catalog_2epoch(src, ref, match_radius=1.0, err_cap_mas=1e6)

    injected_amplitude = np.median(np.hypot(pm_ra_true, pm_dec_true))
    recovered_amplitude = np.median(pm['pm_tot'])
    assert injected_amplitude > 1.0  # sanity: the injected signal was real
    # The whole point: after the tie, almost none of a purely linear/coherent
    # field is left -- recovered amplitude is a small fraction of injected.
    assert recovered_amplitude < 0.1 * injected_amplitude


def test_affine_tie_at_1_0_radius_handles_the_offset_that_broke_at_0_3(): # noqa: E501
    """jwst-gc-pipeline PR #959: build_treasury_pm.py's tie-fitting call to
    affine_tie used match_radius=0.3" until a real Sgr B2 per-observation
    guide-star-lock offset of ~134 mas exceeded the verified-tie safety
    margin local_residual_map enforces (offset must be well inside
    match_radius -- specifically < match_radius/3, so per-star pairing
    stays unambiguous). Pin the fix directly with a ~150 mas offset:
    comfortably above the OLD radius's 100 mas floor (so it must still
    raise there, a regression guard against a future radius change quietly
    stopping this test from exercising the actual failure) and comfortably
    below the NEW radius's 333 mas floor (so it must succeed there).
    """
    n = 1200
    sx, sy, mag = _random_field(n, halfwidth_arcsec=200.0)
    offset_arcsec = 0.150  # ~134 mas real Sgr B2 offset, rounded up
    noise = 0.001
    rx = sx + offset_arcsec + RNG.normal(0, noise, n)
    ry = sy + RNG.normal(0, noise, n)
    rmag = mag + RNG.normal(0, 0.01, n)
    src_sc, ref_sc = _sc_from_xy(sx, sy), _sc_from_xy(rx, ry)

    with pytest.raises(TieNotVerifiedError):
        affine_tie(src_sc, mag, ref_sc, rmag, magcut=0, match_radius=0.3)

    _, diag = affine_tie(src_sc, mag, ref_sc, rmag, magcut=0, match_radius=1.0)
    assert abs(diag['A'][0] - offset_arcsec) < 0.005


def test_affine_tie_refuses_literal_duplicate_row_contamination():
    """jwst-gc-pipeline#958: nominally-independent GC Treasury observations
    turned out to share 30-62% bit-identical rows (same RA/Dec to <1 mas,
    same flux) -- a translation-only coherence check (measure_offset) cannot
    catch this, since duplicated rows pile up at the SAME true offset and
    satisfy it trivially. affine_tie must refuse instead of silently tying
    to a contaminated reference.

    Mixed sample: a MINORITY of pairs are genuinely independent (real
    per-star centroid + flux noise, correctly recovered by the affine fit,
    landing at the noise floor -- not exactly zero); a MAJORITY are literal
    copies (same position AND same flux, landing at exactly zero) -- the
    #958 signature.
    """
    n = 1000
    sx, sy, mag = _random_field(n, halfwidth_arcsec=200.0)
    A_true = np.array([0.05, 0.0001, -0.0001])
    B_true = np.array([-0.03, 0.00008, -0.00012])
    dx = A_true[0] + A_true[1] * sx + A_true[2] * sy
    dy = B_true[0] + B_true[1] * sx + B_true[2] * sy
    noise = 0.001  # 1 mas, same floor used elsewhere in this file
    rx = sx + dx + RNG.normal(0, noise, n)
    ry = sy + dy + RNG.normal(0, noise, n)
    rmag = mag + RNG.normal(0, 0.01, n)  # realistic between-epoch flux noise

    n_dup = int(0.6 * n)
    dup_idx = RNG.choice(n, n_dup, replace=False)
    rx[dup_idx] = sx[dup_idx]
    ry[dup_idx] = sy[dup_idx]
    rmag[dup_idx] = mag[dup_idx]

    src_sc, ref_sc = _sc_from_xy(sx, sy), _sc_from_xy(rx, ry)
    with pytest.raises(TieNotVerifiedError, match="duplicate-row"):
        affine_tie(src_sc, mag, ref_sc, rmag, magcut=0, match_radius=1.0)


def test_affine_tie_duplicate_check_requires_flux_agreement_too():
    """pr-reviewer, PR #959 round 2: every duplicate-row test so far paired
    coincident POSITION with coincident FLUX together, so the dmag half of
    the check was never independently exercised -- dropping it entirely
    still left every test passing. Same setup as the majority-contamination
    test above (60% of pairs land at exactly zero separation after the tie),
    but this time those coincident pairs have a deliberately large flux
    mismatch: position agreement alone is not the #958 signature, since a
    real star can legitimately sit near its formal position by chance while
    being a completely different, unrelated source in the other catalog. It
    must NOT be flagged.
    """
    n = 1000
    sx, sy, mag = _random_field(n, halfwidth_arcsec=200.0)
    A_true = np.array([0.05, 0.0001, -0.0001])
    B_true = np.array([-0.03, 0.00008, -0.00012])
    dx = A_true[0] + A_true[1] * sx + A_true[2] * sy
    dy = B_true[0] + B_true[1] * sx + B_true[2] * sy
    noise = 0.001
    rx = sx + dx + RNG.normal(0, noise, n)
    ry = sy + dy + RNG.normal(0, noise, n)
    rmag = mag + RNG.normal(0, 0.01, n)

    n_coincident = int(0.6 * n)
    coincident_idx = RNG.choice(n, n_coincident, replace=False)
    rx[coincident_idx] = sx[coincident_idx]
    ry[coincident_idx] = sy[coincident_idx]
    # Same position, but NOT the same flux -- a literal duplicate row would
    # have copied this too; a merely-coincident position would not.
    rmag[coincident_idx] = mag[coincident_idx] + 2.0

    src_sc, ref_sc = _sc_from_xy(sx, sy), _sc_from_xy(rx, ry)
    _, diag = affine_tie(src_sc, mag, ref_sc, rmag, magcut=0, match_radius=1.0)
    assert diag['dup_row_suspect_fraction'] < 0.02


def test_affine_tie_refuses_minority_duplicate_row_contamination():
    """pr-reviewer PR #140 round 7: the majority-contamination test above
    cannot tell "resid_mas reads near zero because duplicates DOMINATE the
    global bulk estimate" from "resid_mas reads near zero because a
    duplicate's raw separation is zero regardless of the bulk offset" -- only
    the second is the actual invariant a literal duplicate row has. Under a
    real, small, non-zero bulk tie, a MINORITY of duplicates sits at
    resid_mas ~= -(the real offset) after that offset is subtracted, not
    near zero at all -- checked directly against real o105-vs-o108 data,
    5-30% synthetic duplicate fractions under a 2-5 mas real offset read
    frac_dup=0.000 and did not raise before this test was added. Checking
    RAW (pre-bulk-correction) separation as well as the post-tie residual is
    what catches this case.
    """
    n = 1000
    sx, sy, mag = _random_field(n, halfwidth_arcsec=200.0)
    real_dx, real_dy = 0.003, -0.002  # a few mas -- small on purpose
    noise = 0.001
    rx = sx + real_dx + RNG.normal(0, noise, n)
    ry = sy + real_dy + RNG.normal(0, noise, n)
    rmag = mag + RNG.normal(0, 0.01, n)

    n_dup = int(0.3 * n)
    dup_idx = RNG.choice(n, n_dup, replace=False)
    # Literal copies: NO bulk offset applied, unlike the other 70%.
    rx[dup_idx] = sx[dup_idx]
    ry[dup_idx] = sy[dup_idx]
    rmag[dup_idx] = mag[dup_idx]

    src_sc, ref_sc = _sc_from_xy(sx, sy), _sc_from_xy(rx, ry)
    with pytest.raises(TieNotVerifiedError, match="duplicate-row"):
        affine_tie(src_sc, mag, ref_sc, rmag, magcut=0, match_radius=1.0)


def test_affine_tie_tolerates_ordinary_noisy_agreement():
    """Regression guard for the duplicate-row check itself: real per-star
    noise (no injected duplicates at all) must NOT trip it, even though a
    Rayleigh-distributed residual occasionally lands close to zero by
    chance. This is exactly the existing recovers-a-known-distortion fixture
    with no duplication injected."""
    n = 1500
    sx, sy, mag = _random_field(n, halfwidth_arcsec=400.0)
    A_true = np.array([0.150, 0.00030, -0.00040])
    B_true = np.array([-0.080, 0.00025, -0.00020])
    dx = A_true[0] + A_true[1] * sx + A_true[2] * sy
    dy = B_true[0] + B_true[1] * sx + B_true[2] * sy
    noise = 0.001
    rx = sx + dx + RNG.normal(0, noise, n)
    ry = sy + dy + RNG.normal(0, noise, n)
    rmag = mag + RNG.normal(0, 0.01, n)
    src_sc, ref_sc = _sc_from_xy(sx, sy), _sc_from_xy(rx, ry)
    _, diag = affine_tie(src_sc, mag, ref_sc, rmag, magcut=0, match_radius=1.0)
    assert diag['dup_row_suspect_fraction'] < 0.02


def test_isolation_survives_duplicate_observation_concatenation():
    """build_treasury_pm's normal ref is a concatenation of several
    independently-tied observations of the SAME field (see
    load_and_tie_ref_catalogs), so a star's true counterpart appears in ref
    once per observation that detected it, at very nearly the same position
    each time. Before the same_star_radius guard, those re-detections of the
    star ITSELF looked like a bright competing neighbor (comp_flux_ratio~1),
    and the isolation cut rejected essentially everything -- 100% on real
    Arches/Quintuplet data, the case this test reconstructs synthetically.
    """
    pytest.importorskip('flystar')
    n = 800
    sx, sy, mag = _random_field(n, halfwidth_arcsec=300.0)
    dt = 2.0
    pm_ra_true = RNG.normal(0, 4.0, n)
    pm_dec_true = RNG.normal(0, 4.0, n)
    rx = sx + pm_ra_true * dt / 1000.0
    ry = sy + pm_dec_true * dt / 1000.0

    src_sc = _sc_from_xy(sx, sy)
    src = dict(sc=src_sc, ex=np.full(n, 0.003), ey=np.full(n, 0.003),
              mag=mag, flux=10 ** (-0.4 * mag), epoch=2024.0, n=n)

    # Two "observations" of the SAME ref positions, each with its own tiny
    # centroid noise (a few tenths of a mas) -- exactly what
    # load_and_tie_ref_catalogs hands to build_pm_catalog_2epoch after tying
    # each observation independently.
    noise = 0.0002  # arcsec
    rx1 = rx + RNG.normal(0, noise, n); ry1 = ry + RNG.normal(0, noise, n)
    rx2 = rx + RNG.normal(0, noise, n); ry2 = ry + RNG.normal(0, noise, n)
    ref_sc = concatenate([_sc_from_xy(rx1, ry1), _sc_from_xy(rx2, ry2)])
    ref_flux = np.concatenate([10 ** (-0.4 * mag), 10 ** (-0.4 * mag)])
    ref_mag = np.concatenate([mag, mag])
    ref = dict(sc=ref_sc, ex=np.full(2 * n, 0.003), ey=np.full(2 * n, 0.003),
              mag=ref_mag, flux=ref_flux, epoch=2024.0 + dt, n=2 * n)

    pm = build_pm_catalog_2epoch(src, ref, match_radius=1.0, err_cap_mas=1e6)
    assert len(pm) >= 0.95 * n
    # This is the regression: without the same_star_radius guard, dominant
    # (and therefore trustworthy) is 0 here, same as it was on real data.
    assert pm['dominant'].sum() > 0.9 * len(pm)
    assert pm['trustworthy'].sum() > 0.9 * len(pm)


def test_affine_tie_local_correction_removes_a_module_seam():
    """pr-reviewer's per-cell PM-variance analysis (PR #959 review) found the
    excess scatter on Arches/Quintuplet/Brick/Cloud c is spatially COHERENT,
    not centroid noise; a follow-up per-cell median-PM map on real Cloud c
    data pinned a ~14 mas/yr DISCONTINUITY across one ~30" field-edge strip --
    a step function a global 6-parameter affine cannot represent by
    construction (it only fits a smooth linear plane). Inject the same
    signature synthetically (a real small affine trend PLUS a sharp step in
    dx at x=0, like a small relative offset between two NIRCam modules) and
    confirm the LOCAL correction, not the global affine alone, removes it.
    """
    n = 4000
    sx, sy, mag = _random_field(n, halfwidth_arcsec=90.0)
    A_true = np.array([0.02, 0.0001, -0.0001])
    B_true = np.array([-0.01, 0.00005, 0.0001])
    step_dx_mas = 15.0  # a discrete jump, not a linear trend
    step = np.where(sx > 0, step_dx_mas / 1000.0, 0.0)
    dx = A_true[0] + A_true[1] * sx + A_true[2] * sy + step
    dy = B_true[0] + B_true[1] * sx + B_true[2] * sy
    noise = 0.001
    rx = sx + dx + RNG.normal(0, noise, n)
    ry = sy + dy + RNG.normal(0, noise, n)
    rmag = mag + RNG.normal(0, 0.01, n)
    src_sc, ref_sc = _sc_from_xy(sx, sy), _sc_from_xy(rx, ry)

    _, diag_local = affine_tie(src_sc, mag, ref_sc, rmag, magcut=0, match_radius=1.0,
                               local_correction=True, local_cell_arcsec=30.0,
                               local_min_stars=15)
    _, diag_global = affine_tie(src_sc, mag, ref_sc, rmag, magcut=0, match_radius=1.0,
                                local_correction=False)

    assert diag_global['local_correction'] is None
    assert diag_local['local_correction'] is not None
    assert diag_local['local_correction']['n_valid_cells'] > 0
    # The global-only fit cannot represent the step. Its 3-sigma clip loop
    # rejects most of the offset half-field as "outliers" rather than fitting
    # it, so the surviving residual undershoots the naive half-step estimate
    # (measured: 2.08 mas vs a 0.63 mas no-step noise floor) -- still a clear,
    # reproducible excess over the floor. The local correction should bring
    # the residual down close to that noise floor.
    assert diag_global['rms_resid_mas'] > 1.5
    assert diag_local['rms_resid_mas'] < 0.5 * diag_global['rms_resid_mas']
    assert diag_local['rms_resid_mas'] < 2.0


def test_affine_tie_local_correction_false_matches_pre_feature_behavior():
    """local_correction=False must reproduce the pre-existing global-affine-
    only behavior exactly (regression pin for the opt-out) -- same fixture as
    test_affine_tie_recovers_a_known_distortion."""
    n = 1500
    sx, sy, mag = _random_field(n, halfwidth_arcsec=400.0)
    A_true = np.array([0.150, 0.00030, -0.00040])
    B_true = np.array([-0.080, 0.00025, -0.00020])
    dx = A_true[0] + A_true[1] * sx + A_true[2] * sy
    dy = B_true[0] + B_true[1] * sx + B_true[2] * sy
    noise = 0.001
    rx = sx + dx + RNG.normal(0, noise, n)
    ry = sy + dy + RNG.normal(0, noise, n)
    src_sc, ref_sc = _sc_from_xy(sx, sy), _sc_from_xy(rx, ry)
    _, diag = affine_tie(src_sc, mag, ref_sc, mag, magcut=0, match_radius=1.0,
                         local_correction=False)
    assert diag['local_correction'] is None
    assert diag['rms_resid_mas'] == diag['rms_resid_mas_before_local']
    A, B = np.array(diag['A']), np.array(diag['B'])
    assert abs(A[0] - A_true[0]) < 0.002 and abs(B[0] - B_true[0]) < 0.002
