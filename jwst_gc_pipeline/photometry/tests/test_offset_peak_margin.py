"""``peak_margin``: how far the offset histogram's winning bin beats its RUNNER-UP.

Issue #411.  ``contrast`` is peak / MEDIAN non-empty bin.  Over a +-3" window the
median non-empty bin holds ONE pair, so a histogram that is a LATTICE of dozens of
near-equal spots still reports a contrast in the hundreds and ``ok=True`` -- a
1.02:1 coin flip presented as a confident tie.  Measured on w51 F140M the pooled
peak said 378 mas at contrast 546, beating the true zero spot by 1.7%, while every
one of that filter's 64 frames is within 24 mas of the same catalogue.

These tests pin the discriminator on synthetic data: a rigid offset has one spot
and no rival, a replica lattice has many.
"""
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

from jwst_gc_pipeline.photometry.astrometry_offsets import measure_offset, _peak_margin

RA0, DEC0 = 266.4, -28.9
COSD = float(np.cos(np.radians(DEC0)))


def _field(n=900, seed=0, box_deg=0.05):
    rng = np.random.default_rng(seed)
    return (RA0 + rng.uniform(0, box_deg, n) / COSD,
            DEC0 + rng.uniform(0, box_deg, n))


def _sc(ra, dec):
    return SkyCoord(ra * u.deg, dec * u.deg)


def test_peak_margin_masks_only_the_neighbourhood_of_the_winner():
    """A single tall bin with its own spillover in the bins next to it must not be
    read as having a rival: the runner-up is taken OUTSIDE the winner's hill."""
    H = np.zeros((9, 9))
    H[4, 4] = 100.0
    H[4, 5] = H[3, 4] = 40.0        # spillover, adjacent
    assert not np.isfinite(_peak_margin(H, 4, 4))   # nothing else anywhere
    H[0, 0] = 20.0                                   # a real, separated rival
    assert np.isclose(_peak_margin(H, 4, 4), 5.0)


def test_the_exclusion_radius_follows_the_width_of_the_hill():
    """A fixed one-bin pad makes ``peak_margin`` a measure of peak WIDTH: a broad
    hill's own shoulders sit outside the pad and are divided by as if they were a
    rival.  The exclusion radius is twice the half-max radius, so it grows with
    the hill and the divisor stays the background."""
    # one hill, descending 8 per ring from 100 (half-max radius 7), and NOTHING
    # else anywhere on the plane
    H = np.zeros((41, 41))
    for di in range(-6, 7):
        for dj in range(-6, 7):
            H[20 + di, 20 + dj] = max(0.0, 100.0 - 8.0 * max(abs(di), abs(dj)))
    # what a FIXED one-bin pad would divide by: the hill itself, two bins out
    fixed = H.copy()
    fixed[19:22, 19:22] = 0.0
    assert fixed.max() == 84.0
    assert np.isclose(H.max() / fixed.max(), 100.0 / 84.0, rtol=1e-6)
    # with the adaptive radius nothing outside the hill holds anything
    assert not np.isfinite(_peak_margin(H, 20, 20))
    # ... and a genuinely separate rival, past the hill, is still seen
    H[2, 2] = 50.0
    assert np.isclose(_peak_margin(H, 20, 20), 2.0)


def test_rigid_offset_has_no_runner_up():
    """The state the gross branch is written for (brick-1182 v001 class): one
    physical shift, one spot in the histogram."""
    ra, dec = _field(seed=1)
    a = _sc(ra, dec)
    b = _sc(ra + 0.4 / 3600.0 / COSD, dec - 0.3 / 3600.0)
    r = measure_offset(a, b, maxsep=3.0 * u.arcsec, sweep=False)
    assert r["ok"] and r["contrast"] > 50, r
    assert r["peak_margin"] > 5.0, r


def test_a_replica_lattice_reads_high_contrast_and_a_margin_of_one():
    """The w51 shape.  Each reference star is detected several times at FIXED
    detector-frame separations -- the PSF's own repeated structure picked up by a
    peak finder -- so every reference star pairs with its star AND with that star's
    replicas.  The histogram is a lattice of near-equal spots; ``contrast`` cannot
    see that and ``peak_margin`` can."""
    ra, dec = _field(n=700, seed=2)
    rng = np.random.default_rng(7)
    # replica offsets in arcsec, one of them zero (the true position)
    lattice = [(0.0, 0.0), (0.25, -0.03), (-0.25, 0.03),
               (0.15, 0.19), (-0.15, -0.19), (0.09, -0.23), (-0.09, 0.23)]
    src_ra, src_dec = [], []
    for k, (dx, dy) in enumerate(lattice):
        # a handful of extra replicas on ONE non-zero lattice point, so the arg-max
        # lands there rather than on the truth -- a 2% edge, not a measurement
        n = len(ra) + (14 if k == 1 else 0)
        idx = np.arange(len(ra)) if n == len(ra) else np.concatenate(
            [np.arange(len(ra)), rng.choice(len(ra), 14, replace=False)])
        src_ra.append(ra[idx] + dx / 3600.0 / COSD
                      + rng.normal(0, 0.002, len(idx)) / 3600.0 / COSD)
        src_dec.append(dec[idx] + dy / 3600.0
                       + rng.normal(0, 0.002, len(idx)) / 3600.0)
    src = _sc(np.concatenate(src_ra), np.concatenate(src_dec))
    ref = _sc(ra, dec)

    r = measure_offset(src, ref, maxsep=3.0 * u.arcsec, sweep=False)
    assert r["ok"], r                      # it clears the contrast floor ...
    assert r["contrast"] > 50, r           # ... comfortably ...
    assert r["off"] > 100.0, r             # ... at an offset that is not in the data
    assert r["peak_margin"] < 1.25, r      # ... and the runner-up is right beside it


def test_a_broad_single_hill_keeps_its_margin():
    """The fail-open direction of the gate this number feeds.

    ``check_interframe_overlap._global_tie`` demotes a GROSS tie whose margin is
    below 1.25 to could-not-verify, which does not block.  A genuine gross offset
    measured against a reference of ordinary per-star precision is ONE hill with
    no rival anywhere, and it must keep a margin far above that floor at every
    scatter the reference can have -- VIRAC2's own precision is ~40 mas, against
    20 mas bins over a 3" window, so the hill is several bins wide.

    With the fixed one-bin pad this test's own 30/40/50/70 mas rows read
    1.13/1.10/1.06/1.00: the divisor was the hill's own shoulder, so a real 500
    mas offset stopped being reported as measured for no reason but its noise.
    """
    shift = 0.500                              # arcsec: gross, > the gate's 80 mas
    for scatter_mas in (5, 10, 20, 30, 40, 50, 70):
        rng = np.random.default_rng(3)
        n = 900
        ra = RA0 + rng.uniform(0, 0.05, n) / COSD
        dec = DEC0 + rng.uniform(0, 0.05, n)
        s = scatter_mas / 1000.0 / 3600.0
        ref = _sc(ra + rng.normal(0, s, n) / COSD, dec + rng.normal(0, s, n))
        src = _sc(ra + shift / 3600.0 / COSD + rng.normal(0, s, n) / COSD,
                  dec + rng.normal(0, s, n))
        r = measure_offset(src, ref, maxsep=3.0 * u.arcsec, sweep=False)
        assert r["off"] > 400.0, (scatter_mas, r)      # the shift IS recovered
        # 5x the 1.25 floor: this is headroom, not a boundary the noise can walk
        assert r["peak_margin"] > 6.25, (scatter_mas, r["peak_margin"], r["contrast"])
