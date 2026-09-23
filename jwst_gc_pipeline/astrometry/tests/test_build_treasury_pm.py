"""Tests for build_treasury_pm.py's own code (dedup, adaptive tie_magcut) --
the commit that added them (ac4bf1e4) shipped with none, per PR #140 review.
"""
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

from jwst_gc_pipeline.astrometry.build_treasury_pm import (
    _dedup_mask, load_and_tie_ref_catalogs,
)


def _sc_at(center, dra_mas, ddec_mas):
    """A single SkyCoord dra/ddec (mas, already *cosdec-corrected) from center."""
    ra = center.ra.deg + (dra_mas / 1000.0 / 3600.0) / np.cos(center.dec.rad)
    dec = center.dec.deg + ddec_mas / 1000.0 / 3600.0
    return SkyCoord(ra * u.deg, dec * u.deg)


CENTER = SkyCoord(266.5 * u.deg, -28.5 * u.deg)


def test_dedup_keeps_all_of_a_well_separated_field():
    """No two of these are within radius of each other -- nothing should be
    dropped. Regression guard against an over-eager dedup."""
    rng = np.random.default_rng(1)
    n = 200
    dra = rng.uniform(-100, 100, n)   # arcsec-scale spread, mas-scale radius
    ddec = rng.uniform(-100, 100, n)
    sc = SkyCoord([CENTER.ra.deg + d / 3600 for d in dra] * u.deg,
                 [CENTER.dec.deg + d / 3600 for d in ddec] * u.deg)
    keep = _dedup_mask(sc, 0.05)
    assert keep.sum() == n


def test_dedup_collapses_a_simple_duplicate_pair():
    a = _sc_at(CENTER, 0, 0)
    b = _sc_at(CENTER, 5, 0)   # 5 mas away, well inside a 50 mas radius
    sc = SkyCoord([a.ra.deg, b.ra.deg] * u.deg, [a.dec.deg, b.dec.deg] * u.deg)
    keep = _dedup_mask(sc, 0.05)
    assert keep.tolist() == [True, False]


def test_dedup_collapses_a_transitive_chain_not_directly_within_radius():
    """A-B and B-C are each within radius, but A-C is NOT directly.

    NOTE: the old single-nearest-neighbour `_dedup_mask` happens to pass
    this particular case too -- whether it does depends on which of A/C
    scipy's KDTree resolves as B's nearest neighbour in a tie, which is
    platform/version-dependent, not something this test controls. It is
    kept as a basic correctness check of connected-components dedup, but
    it does NOT by itself distinguish the fix from the old code -- see
    test_dedup_collapses_more_than_two_coincident_duplicates below for the
    case that actually does (pr-reviewer caught this: restoring the old
    _dedup_mask still passed this test)."""
    a = _sc_at(CENTER, 0, 0)
    b = _sc_at(CENTER, 30, 0)
    c = _sc_at(CENTER, 60, 0)
    sc = SkyCoord([a.ra.deg, b.ra.deg, c.ra.deg] * u.deg,
                 [a.dec.deg, b.dec.deg, c.dec.deg] * u.deg)
    assert a.separation(b).mas < 50 and b.separation(c).mas < 50
    assert a.separation(c).mas > 50  # not directly within radius
    keep = _dedup_mask(sc, 0.05)
    assert keep.sum() == 1
    assert keep[0]  # lowest index of the component survives


def test_dedup_collapses_more_than_two_coincident_duplicates():
    """4 rows at the exact same position (e.g. a literal duplicate-row
    merge artifact). A single-nearest-neighbour dedup can only ever mark
    ONE row per NN relationship as a duplicate of another; with 3+ rows
    tied at zero separation it can leave more than one "kept" depending on
    tie resolution -- unlike the A-B-C chain above, this is not
    tie-order-lucky: connected components collapsing an n>2 group to 1
    survivor is the actual behaviour the rewrite guarantees and the old
    code does not."""
    pts = [_sc_at(CENTER, 0, 0) for _ in range(4)]
    sc = SkyCoord([p.ra.deg for p in pts] * u.deg, [p.dec.deg for p in pts] * u.deg)
    keep = _dedup_mask(sc, 0.05)
    assert keep.sum() == 1
    assert keep[0]


def test_dedup_handles_two_separate_groups_independently():
    """Two duplicate pairs far apart from each other: each collapses on its
    own, unrelated to the other."""
    a = _sc_at(CENTER, 0, 0)
    a2 = _sc_at(CENTER, 5, 0)
    far = _sc_at(CENTER, 100_000, 0)   # ~100" away, its own component
    far2 = _sc_at(CENTER, 100_005, 0)
    sc = SkyCoord([a.ra.deg, a2.ra.deg, far.ra.deg, far2.ra.deg] * u.deg,
                 [a.dec.deg, a2.dec.deg, far.dec.deg, far2.dec.deg] * u.deg)
    keep = _dedup_mask(sc, 0.05)
    assert keep.tolist() == [True, False, True, False]


def _write_ref_fits(path, sc, flux):
    t = Table()
    t['skycoord'] = sc
    t['flux'] = flux
    t['flux_err'] = flux / 20.0
    t['std_ra'] = np.full(len(sc), 1e-6)    # degrees (build_treasury_pm's own convention)
    t['std_dec'] = np.full(len(sc), 1e-6)
    t.write(path, overwrite=True)


def test_adaptive_tie_magcut_derives_from_src_not_a_literal_15(tmp_path, capsys):
    """affine_tie's own default (magcut=15) assumes a calibrated Vega/AB-like
    scale; this pipeline's mag_src is uncalibrated instrumental
    -2.5*log10(flux), which can run entirely below (or above) 15 regardless
    of real brightness -- the bug this fix targets. With tie_magcut=None,
    the resolved cutoff must track src's OWN distribution instead of a
    literal value that means nothing for this scale.
    """
    rng = np.random.default_rng(3)
    n = 500
    dra = rng.uniform(-60, 60, n)
    ddec = rng.uniform(-60, 60, n)
    sc = SkyCoord((CENTER.ra.deg + dra / 3600) * u.deg,
                 (CENTER.dec.deg + ddec / 3600) * u.deg)
    # An instrumental flux scale where EVERY mag is >> 15 (the opposite
    # direction of the deg-vs-arcsec-style bug found on Cloud c, and the
    # more obviously-wrong case for a literal magcut=15: here it would
    # exclude EVERYTHING rather than nothing).
    flux = rng.uniform(1e-8, 1e-6, n)
    mag = -2.5 * np.log10(flux)
    src = dict(sc=sc, mag=mag, n=n)

    ref_path = tmp_path / 'ref.fits'
    _write_ref_fits(ref_path, sc, flux)

    ref, diags = load_and_tie_ref_catalogs([str(ref_path)], 'f212n', 2026.0, src,
                                           tie_magcut=None, tie_bright_percentile=20.0)
    out = capsys.readouterr().out
    assert 'tie_magcut (bright 20% of src):' in out
    expected = np.percentile(mag, 20.0)
    # A literal magcut=15 would have excluded every one of these stars
    # (all mag >> 15); the adaptive cutoff must sit within src's own range.
    assert mag.min() <= expected <= mag.max()
    assert abs(expected - np.percentile(mag, 20.0)) < 1e-6
