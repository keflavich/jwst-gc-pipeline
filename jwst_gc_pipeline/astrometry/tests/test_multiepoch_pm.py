"""Synthetic-field tests for the JWST x JWST affine-tie proper-motion path
(multiepoch_pm.affine_tie / build_pm_catalog_2epoch, PR #140).

No real catalogs needed: a random tangent-plane star field stands in for
``src``, and a linear (or non-linear, for the last test) transform of it
stands in for ``ref``. This is what the review on PR #140 flagged as
missing -- 834 lines with no coverage.
"""
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
import astropy.units as u

from jwst_gc_pipeline.astrometry.multiepoch_pm import affine_tie, build_pm_catalog_2epoch

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

    src_sc, ref_sc = _sc_from_xy(sx, sy), _sc_from_xy(rx, ry)
    ref_sc_tied, diag = affine_tie(ref_sc, mag, src_sc, mag, magcut=0, match_radius=1.0)

    src = dict(sc=src_sc, ex=np.full(n, 0.003), ey=np.full(n, 0.003),
              mag=mag, flux=10 ** (-0.4 * mag), epoch=2024.0, n=n)
    ref = dict(sc=ref_sc_tied, ex=np.full(n, 0.003), ey=np.full(n, 0.003),
              mag=mag, flux=10 ** (-0.4 * mag), epoch=2024.0 + dt, n=n)
    pm = build_pm_catalog_2epoch(src, ref, match_radius=1.0, err_cap_mas=1e6)

    injected_amplitude = np.median(np.hypot(pm_ra_true, pm_dec_true))
    recovered_amplitude = np.median(pm['pm_tot'])
    assert injected_amplitude > 1.0  # sanity: the injected signal was real
    # The whole point: after the tie, almost none of a purely linear/coherent
    # field is left -- recovered amplitude is a small fraction of injected.
    assert recovered_amplitude < 0.1 * injected_amplitude
