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
from astropy.table import QTable, Table

from jwst_gc_pipeline.astrometry import multiepoch_pm
from jwst_gc_pipeline.astrometry.multiepoch_pm import (
    affine_tie, build_pm_catalog_2epoch, TieNotVerifiedError,
    load_jwst_m7, std_columns_to_arcsec)

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


def _field_with_src_companions(ratios, sep_arcsec=0.3):
    """src = n primaries plus, for each finite ``ratios[i]``, one src-only
    companion ``sep_arcsec`` away with flux ratios[i] x the primary's (a deeper
    src catalogue resolves stars the ref epoch does not); ref = the primaries
    only, moved by a small random PM. Primaries sit on a 5" grid so no two
    land inside one another's isolation beam by chance."""
    n = len(ratios)
    side = int(np.ceil(np.sqrt(n)))
    gx, gy = np.meshgrid(np.arange(side) * 5.0, np.arange(side) * 5.0)
    pick = RNG.permutation(side * side)[:n]
    sx, sy = gx.ravel()[pick] - 2.5 * side, gy.ravel()[pick] - 2.5 * side
    mag = RNG.uniform(-10, -5, n)
    flux = 10 ** (-0.4 * mag)
    has = np.isfinite(ratios)
    ang = RNG.uniform(0, 2 * np.pi, n)[has]
    cx, cy = sx[has] + sep_arcsec * np.cos(ang), sy[has] + sep_arcsec * np.sin(ang)
    cflux = ratios[has] * flux[has]
    m = n + int(has.sum())
    src = dict(sc=_sc_from_xy(np.concatenate([sx, cx]), np.concatenate([sy, cy])),
               ex=np.full(m, 0.003), ey=np.full(m, 0.003),
               mag=np.concatenate([mag, -2.5 * np.log10(cflux)]),
               flux=np.concatenate([flux, cflux]), epoch=2024.0, n=m)
    dt = 2.0
    rx = sx + RNG.normal(0, 4.0, n) * dt / 1000.0
    ry = sy + RNG.normal(0, 4.0, n) * dt / 1000.0
    ref = dict(sc=_sc_from_xy(rx, ry), ex=np.full(n, 0.003), ey=np.full(n, 0.003),
               mag=mag, flux=flux, epoch=2024.0 + dt, n=n)
    return src, ref


def test_src_isolation_ignores_faint_src_neighbours():
    """A src neighbour at contrast >= 10 (flux ratio <= flux_ratio_cap) does not
    disqualify a star; a comparable one does. With the previous rule
    (src_isolation='any') every src neighbour in the beam disqualified the
    star, so a deeper src catalogue (Arches v6fix, built on the faint-star
    stack) dropped trustworthy from 6,025 (v6) to 1,339."""
    pytest.importorskip('flystar')
    n = 300
    ratios = np.concatenate([np.full(n, 0.05), np.full(n, 0.5), np.full(n, np.nan)])
    src, ref = _field_with_src_companions(ratios)
    faint, bright, alone = np.arange(n), np.arange(n, 2 * n), np.arange(2 * n, 3 * n)

    pm = build_pm_catalog_2epoch(src, ref, match_radius=0.15, err_cap_mas=1e6)
    # build_pm_catalog_2epoch keeps src order for matched rows; every primary
    # matches and no companion does (0.3" > match_radius)
    assert len(pm) == 3 * n
    assert pm.meta['src_isolation'] == 'flux'
    assert pm['dominant'][faint].all() and pm['trustworthy'][faint].all()
    assert not pm['dominant'][bright].any()
    assert pm['dominant'][alone].all()
    np.testing.assert_allclose(pm['src_comp_flux_ratio'][faint], 0.05, rtol=1e-6)
    np.testing.assert_allclose(pm['src_comp_flux_ratio'][bright], 0.5, rtol=1e-6)
    assert (pm['src_comp_flux_ratio'][alone] == 0).all()
    assert (pm['n_src_within_beam'][faint] == 2).all()

    old = build_pm_catalog_2epoch(src, ref, match_radius=0.15, err_cap_mas=1e6,
                                  src_isolation='any')
    assert old.meta['src_isolation'] == 'any'
    assert not old['dominant'][faint].any() and not old['dominant'][bright].any()
    assert old['dominant'][alone].all()


def test_src_isolation_keeps_contrast_exactly_ten():
    """Contrast exactly 1/flux_ratio_cap (1.0 / 10.0 == 0.1 in IEEE doubles)
    is kept; a slightly brighter companion is not."""
    pytest.importorskip('flystar')
    n = 100
    src, ref = _field_with_src_companions(np.full(2 * n, 0.1))
    src['flux'][:2 * n] = 10.0
    src['flux'][2 * n:3 * n] = 1.0
    src['flux'][3 * n:] = 1.001
    pm = build_pm_catalog_2epoch(src, ref, match_radius=0.15, err_cap_mas=1e6)
    assert len(pm) == 2 * n
    assert (pm['src_comp_flux_ratio'][:n] == 0.1).all()
    assert pm['dominant'][:n].all()
    assert not pm['dominant'][n:].any()


def test_src_isolation_nonfinite_neighbour_flux_counts_as_competing():
    pytest.importorskip('flystar')
    n = 200
    src, ref = _field_with_src_companions(np.full(n, 0.05))
    src['flux'][n:] = np.nan
    pm = build_pm_catalog_2epoch(src, ref, match_radius=0.15, err_cap_mas=1e6)
    assert len(pm) == n
    assert not pm['dominant'].any()


def test_src_isolation_rejects_unknown_mode():
    with pytest.raises(ValueError, match='src_isolation'):
        build_pm_catalog_2epoch({}, {}, src_isolation='bright')


def test_ref_frame_position_is_the_untied_ref_position():
    """ra_ref/dec_ref must be the ref star's own (untied) position, which is
    where it sits in the Treasury HiPS, while ra0/dec0 live in the src frame
    after the tie."""
    pytest.importorskip('flystar')
    n = 2000
    sx, sy, mag = _random_field(n, halfwidth_arcsec=300.0)
    offx, offy = 0.10, -0.12   # 100/-120 mas pointing offset the tie removes
    # small noise keeps the tie's duplicate-row guard (#958) from firing
    rx = sx + offx + RNG.normal(0, 0.003, n)
    ry = sy + offy + RNG.normal(0, 0.003, n)
    mag_ref = mag + RNG.normal(0, 0.02, n)
    src_sc, ref_sc = _sc_from_xy(sx, sy), _sc_from_xy(rx, ry)
    src = dict(sc=src_sc, ex=np.full(n, 0.003), ey=np.full(n, 0.003),
              mag=mag, flux=10 ** (-0.4 * mag), epoch=2024.0, n=n)
    ref = dict(sc=ref_sc, ex=np.full(n, 0.003), ey=np.full(n, 0.003),
              mag=mag_ref, flux=10 ** (-0.4 * mag_ref), epoch=2026.0, n=n)
    tied, _ = affine_tie(ref['sc'], ref['mag'], src['sc'], src['mag'],
                         magcut=0, match_radius=1.0)
    ref = dict(ref, sc=tied, sc_raw=ref_sc, obs_index=np.zeros(n, dtype=int))
    pm = build_pm_catalog_2epoch(src, ref, match_radius=1.0, err_cap_mas=1e6)

    ra_ref = SkyCoord(pm['ra_ref'] * u.deg, pm['dec_ref'] * u.deg)
    ra0 = SkyCoord(pm['ra0'] * u.deg, pm['dec0'] * u.deg)
    idx, sep, _ = ra_ref.match_to_catalog_sky(ref_sc)
    assert np.all(sep.arcsec < 1e-6)
    assert np.median(ra_ref.separation(ra0).arcsec) == pytest.approx(np.hypot(offx, offy), abs=0.01)


def _fake_pm_table(with_ref_cols=True, ref_paths=None):
    """Minimal PM table for make_overlay_json (no flystar needed)."""
    import json
    from astropy.table import Table
    t = Table({'ra0': [266.5, 266.6], 'dec0': [-28.5, -28.4],
               'pm_ra': [1.0, -2.0], 'pm_dec': [0.5, 3.0], 'pm_tot': [1.1, 3.6],
               'pm_ra_err': [0.1, 0.1], 'pm_dec_err': [0.1, 0.1],
               'mag_src': [-8.0, -7.0], 'trustworthy': [True, True]})
    if with_ref_cols:
        t['ra_ref'] = [266.5001, 266.6001]
        t['dec_ref'] = [-28.5001, -28.4001]
    t.meta['epochs'] = [2024.0, 2026.0]
    t.meta['baseline_yr'] = 2.0
    if ref_paths is not None:
        t.meta['ref_paths_json'] = json.dumps(ref_paths)
    return t


def test_overlay_json_frame_selection_and_provenance(tmp_path):
    from jwst_gc_pipeline.astrometry.make_pm_overlay_json import make_overlay_json
    fits_path = tmp_path / 'pm.fits'
    _fake_pm_table(ref_paths=['a_o128.fits', 'a_o129.fits']).write(fits_path)
    doc = make_overlay_json(str(fits_path), str(tmp_path / 'o.json'), 'x', verbose=False)
    assert doc['meta']['position_frame'] == 'treasury-ref-epoch'
    assert doc['sources'][0]['ra'] == pytest.approx(266.5001)
    assert doc['meta']['ref_observations'] == ['a_o128.fits', 'a_o129.fits']
    doc_src = make_overlay_json(str(fits_path), str(tmp_path / 'o2.json'), 'x',
                                verbose=False, frame='src')
    assert doc_src['meta']['position_frame'] == 'src-tied'
    assert doc_src['sources'][0]['ra'] == pytest.approx(266.5)


def test_overlay_json_refuses_old_fits_without_ref_columns(tmp_path):
    from jwst_gc_pipeline.astrometry.make_pm_overlay_json import make_overlay_json
    fits_path = tmp_path / 'old.fits'
    _fake_pm_table(with_ref_cols=False).write(fits_path)
    with pytest.raises(ValueError, match='ra_ref'):
        make_overlay_json(str(fits_path), str(tmp_path / 'o.json'), 'x', verbose=False)
    doc = make_overlay_json(str(fits_path), str(tmp_path / 'o.json'), 'x', verbose=False,
                            frame='src', ref_observations=['o112', 'o118'])
    assert doc['meta']['ref_observations'] == ['o112', 'o118']


def _m7_table(std_ra, std_dec, filt='f200w'):
    n = len(std_ra)
    t = Table()
    t['skycoord_ref'] = SkyCoord((CENTER.ra.deg + np.arange(n) * 1e-4) * u.deg,
                                 np.full(n, CENTER.dec.deg) * u.deg)
    t[f'mag_ab_{filt}'] = np.full(n, 18.0)
    t[f'std_ra_{filt}'] = np.asarray(std_ra, float)
    t[f'std_dec_{filt}'] = np.asarray(std_dec, float)
    return t


def test_load_jwst_m7_std_columns_are_degrees(tmp_path):
    """std_ra/std_dec are degrees (merge_catalogs.py); 1e-6 deg is 3.6 mas."""
    path = tmp_path / 'm7.fits'
    _m7_table([1e-6, 2e-6], [1e-6, 3e-6]).write(path)
    jwst = load_jwst_m7(str(path))
    np.testing.assert_allclose(jwst['ex'], [3.6e-3, 7.2e-3])
    np.testing.assert_allclose(jwst['ey'], [3.6e-3, 10.8e-3])


def test_load_jwst_m7_zero_nan_and_missing_use_floor(tmp_path):
    path = tmp_path / 'm7.fits'
    _m7_table([0.0, np.nan, -1e-6], [0.0, np.nan, 1e-6]).write(path)
    jwst = load_jwst_m7(str(path))
    np.testing.assert_allclose(jwst['ex'], 0.005)
    np.testing.assert_allclose(jwst['ey'], [0.005, 0.005, 3.6e-3])
    # No std columns at all: every row takes the floor.
    t = _m7_table([1e-6], [1e-6])
    t.remove_columns(['std_ra_f200w', 'std_dec_f200w'])
    path2 = tmp_path / 'm7_nostd.fits'
    t.write(path2)
    jwst = load_jwst_m7(str(path2))
    np.testing.assert_allclose(jwst['ex'], 0.005)
    np.testing.assert_allclose(jwst['ey'], 0.005)


def test_std_columns_to_arcsec_cosdec_is_opt_in():
    t = _m7_table([1e-6], [1e-6])
    ex, ey = std_columns_to_arcsec(t, 'f200w')
    np.testing.assert_allclose([ex[0], ey[0]], 3.6e-3)
    ex, ey = std_columns_to_arcsec(t, 'f200w', dec_deg=-28.5, apply_cosdec=True)
    np.testing.assert_allclose(ex[0], 3.6e-3 * np.cos(np.deg2rad(-28.5)))
    np.testing.assert_allclose(ey[0], 3.6e-3)
    with pytest.raises(ValueError):
        std_columns_to_arcsec(t, 'f200w', apply_cosdec=True)
    # The default switch leaves the coordinate-RA convention unchanged.
    assert multiepoch_pm.STD_RA_APPLY_COSDEC is False


@pytest.mark.parametrize('apply_cosdec', [False, True])
def test_load_jwst_m7_and_treasury_loader_agree(tmp_path, monkeypatch, apply_cosdec):
    """build_treasury_pm._load_one and load_jwst_m7 share one conversion site."""
    from jwst_gc_pipeline.astrometry import build_treasury_pm as B
    monkeypatch.setattr(multiepoch_pm, 'STD_RA_APPLY_COSDEC', apply_cosdec)
    t = _m7_table([1e-6, 0.0, 2e-6], [2e-6, 1e-6, np.nan])
    t['skycoord_f200w'] = t['skycoord_ref']
    t['flux_f200w'] = np.full(3, 1e-3)
    t['flux_err_f200w'] = np.full(3, 1e-5)
    path = tmp_path / 'm7.fits'
    t.write(path)
    a = load_jwst_m7(str(path))
    b = B._load_one(str(path), 'f200w', 2023.7, 5.0, 0)
    np.testing.assert_allclose(a['ex'], b['ex'])
    np.testing.assert_allclose(a['ey'], b['ey'])
    if apply_cosdec:
        np.testing.assert_allclose(a['ex'][0], 3.6e-3 * np.cos(np.deg2rad(CENTER.dec.deg)))


def test_std_columns_to_arcsec_nonfinite_and_infinite_rows_are_nan():
    t = _m7_table([np.inf, np.nan, 0.0, -1e-6, 1e-6], [np.inf, np.nan, 0.0, -1e-6, 1e-6])
    ex, ey = std_columns_to_arcsec(t, 'f200w')
    assert np.isnan(ex[:4]).all() and np.isnan(ey[:4]).all()
    np.testing.assert_allclose([ex[4], ey[4]], 3.6e-3)


def test_std_columns_to_arcsec_bare_column_wins_over_suffixed():
    """A table with both bare and suffixed std columns is read from the bare pair."""
    t = _m7_table([5e-6], [5e-6])
    t['std_ra'] = [1e-6]
    t['std_dec'] = [2e-6]
    ex, ey = std_columns_to_arcsec(t, 'f200w')
    np.testing.assert_allclose([ex[0], ey[0]], [3.6e-3, 7.2e-3])
    # Without the bare pair the suffixed one is used.
    t.remove_columns(['std_ra', 'std_dec'])
    ex, ey = std_columns_to_arcsec(t, 'f200w')
    np.testing.assert_allclose([ex[0], ey[0]], 18e-3)


def test_std_columns_to_arcsec_masked_entries_are_nan():
    t = Table(masked=True)
    t['std_ra_f200w'] = np.ma.array([1e-6, 1e20, 2e-6], mask=[False, True, False])
    t['std_dec_f200w'] = np.ma.array([1e-6, 1e-6, 1e20], mask=[False, False, True])
    ex, ey = std_columns_to_arcsec(t, 'f200w')
    np.testing.assert_allclose(ex, [3.6e-3, np.nan, 7.2e-3])
    np.testing.assert_allclose(ey, [3.6e-3, 3.6e-3, np.nan])


def test_std_columns_to_arcsec_honours_column_units():
    """A unit-bearing column is converted from its unit, not read as degrees."""
    t = Table()
    t['std_ra_f200w'] = [3.6, 7.2]
    t['std_ra_f200w'].unit = u.mas
    t['std_dec_f200w'] = [1e-6, 2e-6]
    t['std_dec_f200w'].unit = u.deg
    ex, ey = std_columns_to_arcsec(t, 'f200w')
    np.testing.assert_allclose(ex, [3.6e-3, 7.2e-3])
    np.testing.assert_allclose(ey, [3.6e-3, 7.2e-3])
    q = QTable()
    q['std_ra_f200w'] = [3.6, 7.2] * u.mas
    q['std_dec_f200w'] = [1e-6, 2e-6] * u.deg
    ex, ey = std_columns_to_arcsec(q, 'f200w')
    np.testing.assert_allclose(ex, [3.6e-3, 7.2e-3])
    np.testing.assert_allclose(ey, [3.6e-3, 7.2e-3])
    # A non-angular unit is refused instead of being multiplied by 3600.
    t['std_ra_f200w'].unit = u.Jy
    with pytest.raises(u.UnitConversionError):
        std_columns_to_arcsec(t, 'f200w')


@pytest.mark.parametrize('missing', [['std_ra_f200w'], ['std_dec_f200w'],
                                     ['std_ra_f200w', 'std_dec_f200w']])
def test_treasury_loader_raises_when_std_columns_absent(tmp_path, missing):
    """_load_one keeps raising KeyError when either std column is absent."""
    from jwst_gc_pipeline.astrometry import build_treasury_pm as B
    t = _m7_table([1e-6], [1e-6])
    t['skycoord_f200w'] = t['skycoord_ref']
    t['flux_f200w'] = [1e-3]
    t['flux_err_f200w'] = [1e-5]
    t.remove_columns(missing)
    path = tmp_path / 'm7_nostd.fits'
    t.write(path)
    with pytest.raises(KeyError):
        B._load_one(str(path), 'f200w', 2023.7, 5.0, 0)


# ---------------------------------------------------------------------------
# assign='mincost' (multiepoch_pm._gated_mincost_pairs) and local_flux_noise
# ---------------------------------------------------------------------------

def _cat_at(x_arcsec, flux, ex=0.003, noise=0.0):
    """A catalog dict with stars on the y=0 line at ``x_arcsec`` (tangent arcsec)."""
    x = np.atleast_1d(np.asarray(x_arcsec, float))
    n = len(x)
    flux = np.broadcast_to(np.asarray(flux, float), (n,)).copy()
    return dict(sc=_sc_from_xy(x, np.zeros(n)), ex=np.full(n, ex), ey=np.full(n, ex),
                mag=-2.5 * np.log10(flux), flux=flux, flux_noise=np.full(n, noise),
                epoch=2024.0, n=n)


def _pairs(ia, ib):
    return set(zip(ia.tolist(), ib.tolist()))


def test_mincost_keeps_the_nearer_claimant_where_nearest_drops_both():
    """Two src stars 20 mas apart both call the one ref star between them
    nearest: _unique_nearest_pairs drops BOTH; the gated assignment keeps the
    closer one (3 mas sigma per axis -> a 25 mas position gate)."""
    src = _cat_at([0.0, 0.020], 1.0)
    ref = _cat_at([0.008], 1.0)
    ia, ib = multiepoch_pm._unique_nearest_pairs(src['sc'], ref['sc'], 0.15)
    assert len(ia) == 0
    ia, ib, tier = multiepoch_pm._gated_mincost_pairs(src, ref)
    assert _pairs(ia, ib) == {(0, 0)}
    assert tier.tolist() == [1]


def test_mincost_flux_consistency_beats_proximity():
    """The nearer src star is 3x brighter than the ref star, the farther one
    matches it: the flux gate gives the ref star to the farther src star."""
    src = _cat_at([0.0, 0.020], [3.0, 1.0])
    ref = _cat_at([0.008], 1.0)
    ia, ib, tier = multiepoch_pm._gated_mincost_pairs(src, ref)
    assert _pairs(ia, ib) == {(1, 0)}
    # the same geometry without flux information pairs the nearer one
    src_nf, ref_nf = ({k: v for k, v in c.items() if k not in ('flux', 'flux_noise')}
                      for c in (src, ref))
    ia, ib, _ = multiepoch_pm._gated_mincost_pairs(src_nf, ref_nf)
    assert _pairs(ia, ib) == {(0, 0)}


def test_mincost_keeps_both_stars_of_a_chain():
    """a0 -- b0 -- a1 -- b1: a1 is nearer b0 (8 mas) than b1 (16 mas), so both
    src stars call b0 nearest and the nearest rule drops both.  Giving b0 to a0
    and b1 to a1 matches both stars, which is what the assignment does."""
    src = _cat_at([0.000, 0.020], 1.0)
    ref = _cat_at([0.012, 0.036], 1.0)
    na, nb = multiepoch_pm._unique_nearest_pairs(src['sc'], ref['sc'], 0.15)
    assert len(na) == 0
    ia, ib, tier = multiepoch_pm._gated_mincost_pairs(src, ref)
    assert _pairs(ia, ib) == {(0, 0), (1, 1)}
    assert (tier == 1).all()


def test_mincost_keeps_a_nearest_pair_outside_the_gate_as_tier_2():
    """A 60 mas displacement (a ~30 mas/yr mover) fails the 25 mas gate, but
    the nearest rule pairs it, so it stays, as tier 2."""
    src = _cat_at([0.0, 5.0], 1.0)
    ref = _cat_at([0.060, 5.004], 1.0)
    ia, ib, tier = multiepoch_pm._gated_mincost_pairs(src, ref)
    assert _pairs(ia, ib) == {(0, 0), (1, 1)}
    assert dict(zip(ia.tolist(), tier.tolist())) == {0: 2, 1: 1}


def test_mincost_never_adds_a_pair_outside_the_gate_that_nearest_lacks():
    """The leftovers of tier 1 are NOT re-paired by the nearest rule: a contested
    ref star whose claimants both fail the gate stays unmatched, as under the
    nearest rule."""
    src = _cat_at([0.0, 0.100], 1.0)
    ref = _cat_at([0.050], 1.0)       # 50 mas from each: outside the 25 mas gate, contested
    ia, ib, tier = multiepoch_pm._gated_mincost_pairs(src, ref)
    assert len(ia) == 0


def test_mincost_equals_nearest_on_a_sparse_field():
    """With no contested ref star every nearest pair inside the gate is kept, with
    the same partner, and nothing is added."""
    rng = np.random.default_rng(31)
    n = 600
    x = rng.uniform(-600, 600, n)
    y = rng.uniform(-600, 600, n)
    flux = 10 ** rng.uniform(-3, -1, n)
    dt = 2.0
    src = dict(sc=_sc_from_xy(x, y), ex=np.full(n, 0.003), ey=np.full(n, 0.003),
               mag=-2.5 * np.log10(flux), flux=flux, epoch=2024.0, n=n)
    ref = dict(src, sc=_sc_from_xy(x + rng.normal(0, 2.0, n) * dt / 1e3,
                                   y + rng.normal(0, 2.0, n) * dt / 1e3),
               flux=flux * np.exp(rng.normal(0, 0.03, n)), epoch=2024.0 + dt)
    na, nb = multiepoch_pm._unique_nearest_pairs(src['sc'], ref['sc'], 0.15)
    ia, ib, tier = multiepoch_pm._gated_mincost_pairs(src, ref)
    assert _pairs(ia, ib) == _pairs(na, nb)
    assert len(set(ia.tolist())) == len(ia) and len(set(ib.tolist())) == len(ib)


def test_mincost_recovers_contested_pairs_in_a_crowded_field():
    """Crowded synthetic field (4 stars / arcsec^2 is far denser than this; 1 is
    enough to contest ~10% of the stars): the assignment returns one-to-one
    pairs, at least as many as the nearest rule, almost all of them true."""
    rng = np.random.default_rng(32)
    n = 4000
    half = 40.0
    x = rng.uniform(-half, half, n)
    y = rng.uniform(-half, half, n)
    flux = 10 ** rng.uniform(-3, -1, n)
    sig = 0.003
    src = dict(sc=_sc_from_xy(x, y), ex=np.full(n, sig), ey=np.full(n, sig),
               mag=-2.5 * np.log10(flux), flux=flux, flux_noise=np.full(n, 1e-5),
               epoch=2024.0, n=n)
    keep = rng.uniform(size=n) > 0.1     # a tenth of the stars are missing in ref
    rx = x + rng.normal(0, sig, n)
    ry = y + rng.normal(0, sig, n)
    ref = dict(sc=_sc_from_xy(rx[keep], ry[keep]), ex=np.full(keep.sum(), sig),
               ey=np.full(keep.sum(), sig), mag=src['mag'][keep],
               flux=flux[keep] * np.exp(rng.normal(0, 0.02, keep.sum())),
               flux_noise=np.full(keep.sum(), 1e-5), epoch=2026.0, n=int(keep.sum()))
    truth = {int(i): j for j, i in enumerate(np.flatnonzero(keep))}
    na, nb = multiepoch_pm._unique_nearest_pairs(src['sc'], ref['sc'], 0.15)
    ia, ib, tier = multiepoch_pm._gated_mincost_pairs(src, ref)
    assert len(ia) >= len(na)
    assert len(set(ia.tolist())) == len(ia) and len(set(ib.tolist())) == len(ib)
    right = np.mean([truth.get(int(a)) == int(b) for a, b in zip(ia, ib)])
    right_nearest = np.mean([truth.get(int(a)) == int(b) for a, b in zip(na, nb)])
    assert right > 0.97 and right >= right_nearest - 0.01


def test_build_pm_catalog_2epoch_assign_option():
    pytest.importorskip('flystar')
    src = _cat_at([0.0, 0.020, 5.0], 1.0)
    ref = _cat_at([0.008, 5.004], 1.0)
    ref['epoch'] = 2026.0
    plain = build_pm_catalog_2epoch(src, ref, match_radius=0.15, err_cap_mas=1e6)
    assert len(plain) == 1 and 'assign_tier' not in plain.colnames
    assert plain.meta['assign'] == 'nearest'
    pm = build_pm_catalog_2epoch(src, ref, match_radius=0.15, err_cap_mas=1e6,
                                 assign='mincost')
    assert len(pm) == 2 and pm.meta['assign'] == 'mincost'
    assert pm['assign_tier'].tolist() == [1, 1]
    with pytest.raises(ValueError, match='assign'):
        build_pm_catalog_2epoch({}, {}, assign='hungarian')


def test_rescued_rows_are_matched_but_not_trustworthy_by_default():
    pytest.importorskip('flystar')
    n = 50
    side = 8
    gx, gy = np.meshgrid(np.arange(side) * 5.0, np.arange(side) * 5.0)
    x, y = gx.ravel()[:n], gy.ravel()[:n]
    flux = np.full(n, 1e-3)
    src = dict(sc=_sc_from_xy(x, y), ex=np.full(n, 0.003), ey=np.full(n, 0.003),
               mag=-2.5 * np.log10(flux), flux=flux, epoch=2024.0, n=n,
               err_rescued=np.arange(n) < 10)
    ref = dict(src, sc=_sc_from_xy(x + 0.004, y), epoch=2026.0, err_rescued=np.zeros(n, bool))
    pm = build_pm_catalog_2epoch(src, ref, match_radius=0.15, err_cap_mas=1e6)
    assert len(pm) == n
    assert pm['src_err_rescued'].sum() == 10 and not pm['ref_err_rescued'].any()
    assert pm['trustworthy'].sum() == n - 10
    pm = build_pm_catalog_2epoch(src, ref, match_radius=0.15, err_cap_mas=1e6,
                                 rescued_trustworthy=True)
    assert pm['trustworthy'].sum() == n


def test_local_flux_noise_ignores_rows_with_a_broken_flux_err():
    """The noise is the median flux_err of the S/N 5-20 rows, so a bright star
    whose flux_err is NaN, zero or inflated does not move it, and every row
    (including the broken ones) gets it."""
    rng = np.random.default_rng(33)
    n = 600
    ra = CENTER.ra.deg + rng.uniform(-40, 40, n) / 3600 / np.cos(CENTER.dec.rad)
    dec = CENTER.dec.deg + rng.uniform(-40, 40, n) / 3600
    ferr = np.full(n, 1e-6)
    flux = rng.uniform(6e-6, 19e-6, n)
    flux[:20] = 5e-5
    ferr[:5] = np.nan
    ferr[5:10] = 0.0
    ferr[10:20] = 2e-5       # S/N 2.5
    noise = multiepoch_pm.local_flux_noise(ra, dec, flux, ferr)
    np.testing.assert_allclose(noise, 1e-6)
    # too few S/N 5-20 rows to calibrate: unknown, not made up
    assert np.isnan(multiepoch_pm.local_flux_noise(ra[:5], dec[:5], flux[:5], ferr[:5])).all()


def test_local_flux_noise_follows_a_gradient():
    """Noise doubling across the field is tracked cell by cell."""
    rng = np.random.default_rng(34)
    n = 4000
    x = rng.uniform(-60, 60, n)
    y = rng.uniform(-60, 60, n)
    ra = CENTER.ra.deg + x / 3600 / np.cos(CENTER.dec.rad)
    dec = CENTER.dec.deg + y / 3600
    ferr = 1e-6 * np.where(x < 0, 1.0, 2.0)
    flux = ferr * rng.uniform(6, 18, n)
    noise = multiepoch_pm.local_flux_noise(ra, dec, flux, ferr)
    assert abs(np.median(noise[x < -20]) / 1e-6 - 1.0) < 0.05
    assert abs(np.median(noise[x > 20]) / 1e-6 - 2.0) < 0.1


def test_mincost_flux_gate_rejects_an_inconsistent_claimant():
    """The only src star inside the position gate is 3x brighter than the ref star
    it would take; the other claimant is 100 mas away.  Both call the ref star
    nearest, so the nearest rule drops both, and the flux gate keeps the
    assignment from pairing the brighter star with it."""
    src = _cat_at([0.0, 0.100], [3.0, 1.0])
    ref = _cat_at([0.008], 1.0)
    assert len(multiepoch_pm._unique_nearest_pairs(src['sc'], ref['sc'], 0.15)[0]) == 0
    ia, ib, tier = multiepoch_pm._gated_mincost_pairs(src, ref)
    assert len(ia) == 0


def test_mincost_never_pairs_a_ref_star_twice():
    """a1's nearest ref star b' fails the flux gate, so tier 1 gives b to a1 (flux
    consistent) although the nearest rule pairs b with a2 (nearer, flux off by
    1.3x): the nearest pair (a2, b) must not be kept beside (a1, b)."""
    src = _cat_at([-0.006, 0.015], [1.3, 1.0])         # a2, a1
    ref = _cat_at([0.0, 0.016], [1.0, 5.0])            # b, b'
    na, nb = multiepoch_pm._unique_nearest_pairs(src['sc'], ref['sc'], 0.15)
    assert _pairs(na, nb) == {(0, 0), (1, 1)}
    ia, ib, tier = multiepoch_pm._gated_mincost_pairs(src, ref)
    assert len(set(ib.tolist())) == len(ib) and len(set(ia.tolist())) == len(ia)
    assert _pairs(ia, ib) == {(1, 0)}


def test_local_flux_noise_fills_a_hole_from_neighbouring_cells():
    """A row in a cell with no S/N 5-20 rows takes the median of the 3x3 cells
    around it, not the global median (which here sits between the two halves)."""
    rng = np.random.default_rng(35)
    n = 8000
    x = rng.uniform(-60, 60, n)
    y = rng.uniform(-60, 60, n)
    hole = (np.abs(x + 30) < 8) & (np.abs(y) < 8)
    x, y = x[~hole], y[~hole]
    x = np.append(x, -30.0)
    y = np.append(y, 0.0)
    ferr = 1e-6 * np.where(x < 0, 1.0, 2.0)
    flux = ferr * rng.uniform(6, 18, len(x))
    flux[-1], ferr[-1] = 1e-4, np.nan                 # the broken row sits in the hole
    ra = CENTER.ra.deg + x / 3600 / np.cos(CENTER.dec.rad)
    dec = CENTER.dec.deg + y / 3600
    noise = multiepoch_pm.local_flux_noise(ra, dec, flux, ferr)
    assert abs(noise[-1] / 1e-6 - 1.0) < 0.1


def test_assign_tier_survives_a_fits_round_trip(tmp_path):
    """astropy writes an int8 column to FITS as logical, so tier 2 read back as True."""
    pytest.importorskip('flystar')
    src = _cat_at([0.0, 5.0], 1.0)
    ref = _cat_at([0.060, 5.004], 1.0)       # first pair is 60 mas apart: outside the position gate
    ref['epoch'] = 2026.0
    pm = build_pm_catalog_2epoch(src, ref, match_radius=0.15, err_cap_mas=1e6, assign='mincost')
    assert sorted(pm['assign_tier'].tolist()) == [1, 2]
    pm.write(tmp_path / 'pm.fits')
    back = Table.read(tmp_path / 'pm.fits')
    assert back['assign_tier'].dtype.kind == 'i'
    assert sorted(back['assign_tier'].tolist()) == [1, 2]
