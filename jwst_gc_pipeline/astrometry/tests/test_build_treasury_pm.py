"""Tests for build_treasury_pm.py's own code (dedup, adaptive tie_magcut) --
the commit that added them (ac4bf1e4) shipped with none, per PR #140 review.
"""
import json

import numpy as np
import pytest
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

from jwst_gc_pipeline.astrometry.build_treasury_pm import (
    _dedup_mask, load_and_tie_ref_catalogs, _ref_pair_duplicate_fraction,
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


def _random_cat(rng, n, center=CENTER, halfwidth_arcsec=60.0):
    dra = rng.uniform(-halfwidth_arcsec, halfwidth_arcsec, n)
    ddec = rng.uniform(-halfwidth_arcsec, halfwidth_arcsec, n)
    sc = SkyCoord((center.ra.deg + dra / 3600) * u.deg,
                 (center.dec.deg + ddec / 3600) * u.deg)
    mag = rng.uniform(-10, -5, n)
    return dict(sc=sc, mag=mag, n=n), dra, ddec


def test_ref_pair_duplicate_fraction_detects_shared_rows():
    """jwst-gc-pipeline#958 (pr-reviewer, PR #959 round 1): nominally-
    independent ref observations sharing bit-identical rows internally is
    invisible to affine_tie's own src-ref duplicate-row gate, since each
    affine_tie call only ever compares src to ONE ref file, never two ref
    files to each other. This checks ref-vs-ref directly."""
    rng = np.random.default_rng(11)
    n = 200
    cat_a, dra, ddec = _random_cat(rng, n)
    # cat_b: the first half is a LITERAL copy of cat_a's rows (the #958
    # signature); the second half is genuinely independent data.
    n_dup = 100
    dra_b = np.concatenate([dra[:n_dup], rng.uniform(-60, 60, n - n_dup)])
    ddec_b = np.concatenate([ddec[:n_dup], rng.uniform(-60, 60, n - n_dup)])
    mag_b = np.concatenate([cat_a['mag'][:n_dup], rng.uniform(-10, -5, n - n_dup)])
    sc_b = SkyCoord((CENTER.ra.deg + dra_b / 3600) * u.deg,
                    (CENTER.dec.deg + ddec_b / 3600) * u.deg)
    cat_b = dict(sc=sc_b, mag=mag_b, n=n)

    frac = _ref_pair_duplicate_fraction(cat_a, cat_b)
    assert abs(frac - n_dup / n) < 0.01


def test_ref_pair_duplicate_fraction_zero_for_independent_catalogs():
    """Regression guard: two genuinely independent random catalogs must not
    read as duplicated just because they occupy the same field."""
    rng = np.random.default_rng(12)
    cat_a, _, _ = _random_cat(rng, 200)
    cat_b, _, _ = _random_cat(rng, 200)
    assert _ref_pair_duplicate_fraction(cat_a, cat_b) == 0.0


def test_ref_pair_duplicate_fraction_requires_flux_agreement_too():
    """pr-reviewer, PR #959 round 3: the new dmag test added for affine_tie's
    duplicate-row check covers multiepoch_pm.py's own copy of this logic,
    not _ref_pair_duplicate_fraction's separate implementation here -- same
    positions with a large flux mismatch must not read as a duplicate."""
    rng = np.random.default_rng(16)
    cat_a, _, _ = _random_cat(rng, 200)
    cat_b = dict(sc=cat_a['sc'], mag=cat_a['mag'] + 1.0, n=cat_a['n'])
    assert _ref_pair_duplicate_fraction(cat_a, cat_b) == 0.0


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

    # ref is a SEPARATE (synthetic) observation of the same field, not a
    # literal copy of src: small positional (few-mas) and flux (~1%) offsets,
    # well inside affine_tie's match radius. Without these, ref is bit-
    # identical to src -- position AND flux -- which is exactly the
    # jwst-gc-pipeline#958 duplicate-row signature affine_tie's own
    # duplicate-row check now (correctly) refuses to tie; this test is about
    # tie_magcut selectivity, not about that check, so it must not
    # accidentally exercise it.
    ref_dra = dra + rng.normal(0, 0.005, n)   # dra/ddec are arcsec; 5 mas noise
    ref_ddec = ddec + rng.normal(0, 0.005, n)
    ref_sc = SkyCoord((CENTER.ra.deg + ref_dra / 3600) * u.deg,
                      (CENTER.dec.deg + ref_ddec / 3600) * u.deg)
    ref_flux = flux * (1.0 + rng.normal(0, 0.01, n))

    ref_path = tmp_path / 'ref.fits'
    _write_ref_fits(ref_path, ref_sc, ref_flux)

    ref, diags = load_and_tie_ref_catalogs([str(ref_path)], 'f212n', 2026.0, src,
                                           tie_magcut=None, tie_bright_percentile=20.0)
    out = capsys.readouterr().out
    assert 'tie_magcut (bright 20% of src):' in out
    expected = np.percentile(mag, 20.0)
    # A literal magcut=15 would have excluded every one of these stars
    # (all mag >> 15); the adaptive cutoff must sit within src's own range.
    assert mag.min() <= expected <= mag.max()
    assert abs(expected - np.percentile(mag, 20.0)) < 1e-6


def test_load_and_tie_ref_catalogs_warns_on_ref_vs_ref_duplication(tmp_path, capsys):
    """End-to-end version of test_ref_pair_duplicate_fraction_detects_shared_
    rows: two ref FITS files that are literal copies of each other (the
    jwst-gc-pipeline#958 signature) must trigger the warning and be recorded
    in each observation's own diag dict, via load_and_tie_ref_catalogs
    itself -- not just the pure helper function.
    """
    rng = np.random.default_rng(14)
    n = 400
    dra = rng.uniform(-60, 60, n)
    ddec = rng.uniform(-60, 60, n)
    sc = SkyCoord((CENTER.ra.deg + dra / 3600) * u.deg,
                 (CENTER.dec.deg + ddec / 3600) * u.deg)
    flux = rng.uniform(1e-8, 1e-6, n)
    mag = -2.5 * np.log10(flux)
    src = dict(sc=sc, mag=mag, n=n)

    # A single real (independent) ref position, tied with small noise --
    # ref1 and ref2 are then LITERAL COPIES of it, exactly as o105/o108
    # were found to be bit-identical to each other for a majority of rows.
    ref_dra = dra + rng.normal(0, 0.005, n)
    ref_ddec = ddec + rng.normal(0, 0.005, n)
    ref_sc = SkyCoord((CENTER.ra.deg + ref_dra / 3600) * u.deg,
                      (CENTER.dec.deg + ref_ddec / 3600) * u.deg)
    ref_flux = flux * (1.0 + rng.normal(0, 0.01, n))

    ref1_path, ref2_path = tmp_path / 'ref1.fits', tmp_path / 'ref2.fits'
    _write_ref_fits(ref1_path, ref_sc, ref_flux)
    _write_ref_fits(ref2_path, ref_sc, ref_flux)

    ref, diags = load_and_tie_ref_catalogs(
        [str(ref1_path), str(ref2_path)], 'f212n', 2026.0, src,
        tie_magcut=None, tie_bright_percentile=20.0)
    out = capsys.readouterr().out
    assert 'jwst-gc-pipeline#958' in out
    assert diags[0]['ref_dup_fraction_max'] > 0.9
    assert diags[1]['ref_dup_fraction_max'] > 0.9


def test_load_and_tie_ref_catalogs_ties_at_the_offset_that_broke_the_old_radius(tmp_path):
    """pr-reviewer, PR #959 round 2: the radius fix (build_treasury_pm.py's
    own call site widened 0.3->1.0; affine_tie's function default is
    untouched) was pinned only by a test that calls affine_tie DIRECTLY with
    an explicit match_radius -- reverting the production call site back to
    0.3 left every existing test green, since none of them exercised
    load_and_tie_ref_catalogs itself at an offset that size. This does:
    a ~150 mas ref offset (the real Sgr B2 case, rounded up) must tie
    successfully through load_and_tie_ref_catalogs, not just through
    affine_tie called with the radius spelled out by the test.
    """
    rng = np.random.default_rng(15)
    n = 500
    dra = rng.uniform(-60, 60, n)
    ddec = rng.uniform(-60, 60, n)
    sc = SkyCoord((CENTER.ra.deg + dra / 3600) * u.deg,
                 (CENTER.dec.deg + ddec / 3600) * u.deg)
    flux = rng.uniform(1e-8, 1e-6, n)
    mag = -2.5 * np.log10(flux)
    src = dict(sc=sc, mag=mag, n=n)

    offset_mas = 150.0
    ref_dra = dra + offset_mas / 1000.0 + rng.normal(0, 0.005, n)
    ref_ddec = ddec + rng.normal(0, 0.005, n)
    ref_sc = SkyCoord((CENTER.ra.deg + ref_dra / 3600) * u.deg,
                      (CENTER.dec.deg + ref_ddec / 3600) * u.deg)
    ref_flux = flux * (1.0 + rng.normal(0, 0.01, n))
    ref_path = tmp_path / 'ref.fits'
    _write_ref_fits(ref_path, ref_sc, ref_flux)

    # No pytest.raises here -- this must SUCCEED at the production radius.
    # It fails loudly (an unhandled TieNotVerifiedError) if a future edit
    # reverts build_treasury_pm.py's call site back toward 0.3".
    ref, diags = load_and_tie_ref_catalogs([str(ref_path)], 'f212n', 2026.0, src,
                                           tie_magcut=None, tie_bright_percentile=20.0)
    assert diags[0]['n_kept'] > 0.5 * diags[0]['n_match']


# ---------------------------------------------------------------------------
# rescue of rows whose flux_err is broken (--rescue-inflated-err) and
# skipping refs that cannot be tied (--skip-unverified-refs)
# ---------------------------------------------------------------------------

def _write_rescue_fits(path, rng):
    """400 S/N 6-19 stars (the 1e-6 noise calibration), 10 healthy bright stars and,
    below that, rows with a broken flux_err: 5 inflated (S/N 2.5), 3 NaN, 2 zero
    -- all at 50x the noise -- plus 5 faint rows at 3x the noise."""
    n = 425
    ra = CENTER.ra.deg + rng.uniform(-40, 40, n) / 3600 / np.cos(CENTER.dec.rad)
    dec = CENTER.dec.deg + rng.uniform(-40, 40, n) / 3600
    flux = np.concatenate([rng.uniform(6e-6, 19e-6, 400), np.full(10, 1e-4), np.full(10, 5e-5),
                           np.full(5, 3e-6)])
    ferr = np.concatenate([np.full(410, 1e-6), np.full(5, 2e-5), np.full(3, np.nan), np.full(2, 0.0),
                           np.full(5, 1e-6)])
    t = Table()
    t['skycoord'] = SkyCoord(ra * u.deg, dec * u.deg)
    t['flux'] = flux
    t['flux_err'] = ferr
    t['std_ra'] = np.full(n, 1e-6)
    t['std_dec'] = np.full(n, 1e-6)
    t.write(path, overwrite=True)


def test_load_one_rescues_rows_whose_flux_err_fails_the_cut_but_whose_flux_is_high(tmp_path):
    from jwst_gc_pipeline.astrometry.build_treasury_pm import _load_one
    path = tmp_path / 'cat.fits'
    _write_rescue_fits(path, np.random.default_rng(21))
    plain = _load_one(str(path), 'f212n', 2024.7, 5.0, 0)
    assert plain['n'] == 410 and 'err_rescued' not in plain
    resc = _load_one(str(path), 'f212n', 2024.7, 5.0, 0, rescue_k=8.0)
    assert resc['n'] == 420 and resc['err_rescued'].sum() == 10
    assert resc['flux'][resc['err_rescued']].min() == 5e-5      # faint rows stay out
    np.testing.assert_allclose(resc['flux_noise'], 1e-6)
    # the healthy rows are identical with and without the option
    np.testing.assert_array_equal(resc['flux'][~resc['err_rescued']], plain['flux'])


def test_load_filter_catalog_carries_rescue_flags_through_the_dedup(tmp_path):
    from jwst_gc_pipeline.astrometry.build_treasury_pm import load_filter_catalog
    p1, p2 = tmp_path / 'a.fits', tmp_path / 'b.fits'
    _write_rescue_fits(p1, np.random.default_rng(22))
    _write_rescue_fits(p2, np.random.default_rng(23))
    cat = load_filter_catalog([str(p1), str(p2)], 'f212n', 2024.7, rescue_k=8.0)
    assert cat['n'] == len(cat['err_rescued']) == len(cat['flux_noise']) == len(cat['ex'])
    assert cat['err_rescued'].sum() > 0
    assert 'err_rescued' not in load_filter_catalog([str(p1), str(p2)], 'f212n', 2024.7)


def test_skip_unverified_drops_a_ref_with_no_overlap_and_keeps_obs_index(tmp_path):
    import pytest
    from jwst_gc_pipeline.astrometry.multiepoch_pm import TieNotVerifiedError
    rng = np.random.default_rng(24)
    n = 500
    dra = rng.uniform(-60, 60, n)
    ddec = rng.uniform(-60, 60, n)
    sc = SkyCoord((CENTER.ra.deg + dra / 3600) * u.deg, (CENTER.dec.deg + ddec / 3600) * u.deg)
    flux = rng.uniform(1e-8, 1e-6, n)
    src = dict(sc=sc, mag=-2.5 * np.log10(flux), n=n)
    ref_sc = SkyCoord((CENTER.ra.deg + (dra + rng.normal(0, 0.005, n)) / 3600) * u.deg,
                      (CENTER.dec.deg + (ddec + rng.normal(0, 0.005, n)) / 3600) * u.deg)
    far_sc = SkyCoord((CENTER.ra.deg + 0.5 + dra / 3600) * u.deg, (CENTER.dec.deg + ddec / 3600) * u.deg)
    far_flux = flux * (1.0 + rng.normal(0, 0.01, n))
    near, far = tmp_path / 'near.fits', tmp_path / 'far.fits'
    _write_ref_fits(far, far_sc, far_flux)
    _write_ref_fits(near, ref_sc, flux * (1.0 + rng.normal(0, 0.01, n)))
    paths = [str(far), str(near)]
    with pytest.raises(TieNotVerifiedError):
        load_and_tie_ref_catalogs(paths, 'f212n', 2026.0, src, verbose=False)
    ref, diags = load_and_tie_ref_catalogs(paths, 'f212n', 2026.0, src, verbose=False,
                                           skip_unverified=True)
    assert ref['unverified_paths'] == [str(far)]
    assert [d['ref_index'] for d in diags] == [1]
    assert set(ref['obs_index'].tolist()) == {1} and ref['n'] == n
    with pytest.raises(TieNotVerifiedError):
        load_and_tie_ref_catalogs([str(far)], 'f212n', 2026.0, src, verbose=False,
                                  skip_unverified=True)


def test_dedup_prefer_keeps_the_healthy_row_over_a_lower_indexed_rescued_one():
    """A star in both modules, the lower-indexed row only usable through the rescue:
    with prefer the healthy later row is kept, without it the lowest index is."""
    pts = [_sc_at(CENTER, 0, 0), _sc_at(CENTER, 5, 0), _sc_at(CENTER, 5_000, 0)]
    sc = SkyCoord([p.ra.deg for p in pts] * u.deg, [p.dec.deg for p in pts] * u.deg)
    assert _dedup_mask(sc, 0.05).tolist() == [True, False, True]
    assert _dedup_mask(sc, 0.05, prefer=[1, 0, 1]).tolist() == [False, True, True]
    # equal preference falls back to the lowest index, as without prefer
    assert _dedup_mask(sc, 0.05, prefer=[1, 1, 0]).tolist() == [True, False, True]
    # a chain A-B-C with the healthy row at the far end still collapses to one row
    a, b, c = (_sc_at(CENTER, 0, 0), _sc_at(CENTER, 30, 0), _sc_at(CENTER, 60, 0))
    sc = SkyCoord([a.ra.deg, b.ra.deg, c.ra.deg] * u.deg, [a.dec.deg, b.dec.deg, c.dec.deg] * u.deg)
    assert _dedup_mask(sc, 0.05, prefer=[1, 1, 0]).tolist() == [False, False, True]


def test_rescue_never_displaces_a_healthy_duplicate_in_load_filter_catalog(tmp_path):
    """Module A has a rescue-only row of a star that module B measures normally: the row
    kept is B's, so the rescue table's rows are a superset of the plain table's."""
    from jwst_gc_pipeline.astrometry.build_treasury_pm import load_filter_catalog
    rng = np.random.default_rng(25)
    pa, pb = tmp_path / 'a.fits', tmp_path / 'b.fits'
    _write_rescue_fits(pa, rng)
    # B repeats A's 10 inflated-err rows with a healthy flux_err and adds nothing else
    ta = Table.read(pa)
    tb = ta[410:415].copy()
    tb['flux_err'] = 1e-6
    tb['flux'] = 1e-4
    tb.write(pb, overwrite=True)
    plain = load_filter_catalog([str(pa), str(pb)], 'f212n', 2024.7)
    resc = load_filter_catalog([str(pa), str(pb)], 'f212n', 2024.7, rescue_k=8.0)
    assert plain['n'] == 415          # A's 410 healthy rows + B's 5 (their A twins fail the S/N cut)
    assert resc['n'] == 420           # + A's 5 NaN / zero flux_err rows; B keeps the 5 it measures normally
    assert resc['err_rescued'].sum() == 5
    # A's inflated rows (flux 5e-5, indices below B's) are NOT kept in place of B's healthy twins (1e-4)
    healthy_bright = resc['flux'] == 1e-4
    assert healthy_bright.sum() == 15 and not resc['err_rescued'][healthy_bright].any()
    only_a = resc['flux'] == 5e-5
    assert only_a.sum() == 5 and resc['err_rescued'][only_a].all()


def _synthetic_two_epoch_field(tmp_path, rng, n=500, companion_every=5, companion_flux=0.3):
    """src/ref FITS for one field: a star every ~arcsec, every ``companion_every``-th one with a
    companion 0.2" away at ``companion_flux`` x its flux in both epochs."""
    dra = rng.uniform(-60, 60, n)
    ddec = rng.uniform(-60, 60, n)
    flux = 10 ** rng.uniform(-6, -4, n)
    k = np.arange(0, n, companion_every)
    dra = np.concatenate([dra, dra[k] + 0.2])
    ddec = np.concatenate([ddec, ddec[k]])
    flux = np.concatenate([flux, flux[k] * companion_flux])

    def _write(path, ra_arcsec, dec_arcsec, fl):
        sc = SkyCoord((CENTER.ra.deg + ra_arcsec / 3600 / np.cos(CENTER.dec.rad)) * u.deg,
                      (CENTER.dec.deg + dec_arcsec / 3600) * u.deg)
        _write_ref_fits(path, sc, fl)

    n_all = len(flux)
    src_path, ref_path = tmp_path / 'src.fits', tmp_path / 'ref.fits'
    _write(src_path, dra, ddec, flux)
    _write(ref_path, dra + rng.normal(0, 0.004, n_all), ddec + rng.normal(0, 0.004, n_all),
           flux * (1.0 + rng.normal(0, 0.01, n_all)))
    return str(src_path), str(ref_path), n_all, len(k)


def test_build_end_to_end_defaults_and_the_legacy_options(tmp_path):
    """build() defaults to min-cost pairing, rescue K=8 and cap 0.5; the old flags give the old table."""
    pytest = __import__('pytest')
    pytest.importorskip('flystar')
    from jwst_gc_pipeline.astrometry import build_treasury_pm as btp
    src, ref, n_all, n_comp = _synthetic_two_epoch_field(tmp_path, np.random.default_rng(41))
    out = tmp_path / 'pm.fits'
    pm = btp.build([src], [ref], 'f212n', 2024.7, 2026.7, str(out), verbose=False)
    assert pm.meta['assign'] == btp.DEFAULT_ASSIGN == 'mincost'
    assert pm.meta['flux_ratio_cap'] == btp.DEFAULT_FLUX_RATIO_CAP == 0.5
    assert pm.meta['rescue_k'] == btp.DEFAULT_RESCUE_K == 8.0
    assert {'assign_tier', 'src_err_rescued', 'ref_err_rescued'} <= set(pm.colnames)
    legacy = btp.build([src], [ref], 'f212n', 2024.7, 2026.7, str(tmp_path / 'legacy.fits'), verbose=False,
                       assign='nearest', rescue_k=None, flux_ratio_cap=0.10)
    assert legacy.meta['assign'] == 'nearest' and legacy.meta['flux_ratio_cap'] == 0.10
    assert 'rescue_k' not in legacy.meta
    assert 'assign_tier' not in legacy.colnames and 'src_err_rescued' not in legacy.colnames
    assert len(legacy) > 0.95 * n_all
    # a 0.3 companion at 0.2" inside the 0.45" beam: dominant at the default cap 0.5, not at 0.10
    n_dom_legacy = int(np.asarray(legacy['dominant']).sum())
    assert n_dom_legacy < len(legacy) - 0.5 * n_comp
    assert int(np.asarray(pm['dominant']).sum()) > n_dom_legacy + 0.5 * n_comp
    assert len(pm) >= len(legacy)
    back = Table.read(out)
    assert len(back) == len(pm)


def test_build_signature_defaults_are_the_module_constants():
    import inspect
    from jwst_gc_pipeline.astrometry import build_treasury_pm as btp
    par = inspect.signature(btp.build).parameters
    assert par['assign'].default == btp.DEFAULT_ASSIGN
    assert par['rescue_k'].default == btp.DEFAULT_RESCUE_K
    assert par['flux_ratio_cap'].default == btp.DEFAULT_FLUX_RATIO_CAP


def test_cli_defaults_and_the_legacy_flags(monkeypatch):
    import sys
    from jwst_gc_pipeline.astrometry import build_treasury_pm as btp
    seen = {}
    monkeypatch.setattr(btp, 'build', lambda *a, **k: seen.update(k) or None)
    base = ['x', '--src', 's.fits', '--src-epoch', '2024.7', '--ref', 'r.fits', '--ref-epoch', '2026.7',
            '--filter', 'f212n', '--out', 'o.fits']
    monkeypatch.setattr(sys, 'argv', base)
    btp.main()
    assert seen['flux_ratio_cap'] == btp.DEFAULT_FLUX_RATIO_CAP == 0.5
    assert seen['assign'] == btp.DEFAULT_ASSIGN == 'mincost'
    assert seen['rescue_k'] == btp.DEFAULT_RESCUE_K == 8.0
    assert seen['rescued_trustworthy'] is False and seen['skip_unverified_refs'] is False
    # the three flags that reproduce the tables made before the defaults changed
    monkeypatch.setattr(sys, 'argv', base + ['--assign', 'nearest', '--rescue-inflated-err', '0',
                                             '--flux-ratio-cap', '0.10'])
    btp.main()
    assert seen['flux_ratio_cap'] == 0.10 and seen['assign'] == 'nearest' and seen['rescue_k'] is None
    monkeypatch.setattr(sys, 'argv', base + ['--flux-ratio-cap', '0.3', '--rescue-inflated-err', '5',
                                             '--rescued-trustworthy', '--skip-unverified-refs'])
    btp.main()
    assert seen['flux_ratio_cap'] == 0.3 and seen['assign'] == 'mincost' and seen['rescue_k'] == 5.0
    assert seen['rescued_trustworthy'] is True and seen['skip_unverified_refs'] is True


def test_build_treats_rescue_k_zero_as_off(tmp_path):
    """The CLI maps --rescue-inflated-err 0 to off; the library does the same.  Passed on to
    _load_one as 0 the threshold is 0 x local noise: every positive-flux row that fails the S/N cut
    is rescued, and the table gains the rescue columns."""
    pytest.importorskip('flystar')
    from jwst_gc_pipeline.astrometry import build_treasury_pm as btp
    src, ref, n_all, n_comp = _synthetic_two_epoch_field(tmp_path, np.random.default_rng(41))
    off = btp.build([src], [ref], 'f212n', 2024.7, 2026.7, str(tmp_path / 'off.fits'), verbose=False,
                    rescue_k=None)
    zero = btp.build([src], [ref], 'f212n', 2024.7, 2026.7, str(tmp_path / 'zero.fits'), verbose=False,
                     rescue_k=0)
    assert 'rescue_k' not in zero.meta
    assert 'src_err_rescued' not in zero.colnames and 'ref_err_rescued' not in zero.colnames
    assert zero.colnames == off.colnames and len(zero) == len(off)


@pytest.mark.parametrize('bad', [-1.0, -1e-9, float('nan')])
def test_build_rejects_a_negative_or_nan_rescue_threshold(bad):
    """The check runs before any catalog is read (the paths here do not exist)."""
    from jwst_gc_pipeline.astrometry import build_treasury_pm as btp
    with pytest.raises(ValueError, match='rescue_k'):
        btp.build(['no_such_src.fits'], ['no_such_ref.fits'], 'f212n', 2024.7, 2026.7, 'o.fits',
                  rescue_k=bad)


def test_cli_rejects_a_negative_rescue_threshold(monkeypatch):
    import sys
    import pytest
    from jwst_gc_pipeline.astrometry import build_treasury_pm as btp
    monkeypatch.setattr(btp, 'build', lambda *a, **k: pytest.fail('build must not run'))
    monkeypatch.setattr(sys, 'argv', ['x', '--src', 's.fits', '--src-epoch', '2024.7', '--ref', 'r.fits',
                                      '--ref-epoch', '2026.7', '--filter', 'f212n', '--out', 'o.fits',
                                      '--rescue-inflated-err', '-1'])
    with pytest.raises(SystemExit):
        btp.main()


# --- cross-tile duplicate averaging (average_ref_duplicates) ---------------------------------

def _ref_dict(rows):
    """A ref dict as load_and_tie_ref_catalogs returns it, from dicts with ``obs``, ``dx``, ``dy``
    (mas from CENTER) and optional ``flux``, ``ex``/``ey`` (mas), ``frames``, ``sat``, ``resc``."""
    n = len(rows)
    ra = CENTER.ra.deg + np.array([r['dx'] for r in rows]) / 1e3 / 3600 / np.cos(CENTER.dec.rad)
    dec = CENTER.dec.deg + np.array([r['dy'] for r in rows]) / 1e3 / 3600
    sc = SkyCoord(ra * u.deg, dec * u.deg)
    flux = np.array([r.get('flux', 1e-4) for r in rows])
    return dict(sc=sc, sc_raw=sc, flux=flux, mag=-2.5 * np.log10(flux),
                flux_noise=np.array([r.get('noise', 0.0) for r in rows]),
                ex=np.array([r.get('ex', 3.0) for r in rows]) / 1e3,
                ey=np.array([r.get('ey', r.get('ex', 3.0)) for r in rows]) / 1e3,
                sat=np.array([r.get('sat', False) for r in rows]),
                err_rescued=np.array([r.get('resc', False) for r in rows]),
                n_frames=np.array([r.get('frames', 2.0) for r in rows]),
                obs_index=np.array([r['obs'] for r in rows]), row=100 + np.arange(n),
                epoch=2026.7, n=n)


def _offset_mas(ref, i, j):
    dlon, dlat = ref['sc'][i].spherical_offsets_to(ref['sc'][j])
    return dlon.to_value(u.mas), dlat.to_value(u.mas)


def test_average_ref_duplicates_merges_a_pair_with_frame_weights():
    from jwst_gc_pipeline.astrometry.build_treasury_pm import average_ref_duplicates
    ref = _ref_dict([dict(obs=0, dx=0, dy=0, frames=2, ex=3.0, ey=4.0, flux=1.0e-4, noise=1e-6),
                     dict(obs=1, dx=6, dy=-3, frames=4, ex=2.0, ey=1.0, flux=1.2e-4, noise=2e-6),
                     dict(obs=1, dx=900, dy=0)])
    out, info = average_ref_duplicates(ref, 0.020)
    assert out['n'] == len(out['sc']) == 2 and info['n_groups'] == 1
    assert info['groups_by_size'] == {2: 1} and info['n_rows_in'] == 3 and info['n_rows_out'] == 2
    assert out['n_merged'].tolist() == [2, 1]
    # primary = the copy with most frames: its observation, catalogue row and untied position
    assert out['obs_index'].tolist() == [1, 1] and out['row'].tolist() == [101, 102]
    assert out['sc_raw'][0].separation(ref['sc_raw'][1]).mas < 1e-6
    # position = frame-weighted mean of the tied positions: 4/6 of the way from the 2-frame copy
    # to the 4-frame one
    dlon, dlat = ref['sc'][0].spherical_offsets_to(out['sc'][0])
    np.testing.assert_allclose([dlon.to_value(u.mas), dlat.to_value(u.mas)], [4.0, -2.0], atol=1e-4)
    # propagated error of the mean, in the unaveraged rows' own currency
    np.testing.assert_allclose(out['ex'][0] * 1e3, np.hypot(2 / 6 * 3.0, 4 / 6 * 2.0), rtol=1e-12)
    np.testing.assert_allclose(out['ey'][0] * 1e3, np.hypot(2 / 6 * 4.0, 4 / 6 * 1.0), rtol=1e-12)
    np.testing.assert_allclose(out['flux'][0], (2 * 1.0e-4 + 4 * 1.2e-4) / 6, rtol=1e-12)
    np.testing.assert_allclose(out['mag'][0], -2.5 * np.log10(out['flux'][0]), rtol=1e-12)
    np.testing.assert_allclose(out['flux_noise'][0], np.hypot(2 / 6 * 1e-6, 4 / 6 * 2e-6), rtol=1e-12)
    assert out['n_frames'].tolist() == [6.0, 2.0]
    # the input dict is not modified
    assert ref['n'] == 3 and len(ref['sc']) == 3 and ref['ex'][0] == 3.0e-3


def test_average_ref_duplicates_never_merges_two_rows_of_one_observation():
    from jwst_gc_pipeline.astrometry.build_treasury_pm import average_ref_duplicates
    ref = _ref_dict([dict(obs=0, dx=0, dy=0), dict(obs=0, dx=5, dy=0)])
    out, info = average_ref_duplicates(ref, 0.020)
    assert out['n'] == 2 and info['n_groups'] == 0 and out['n_merged'].tolist() == [1, 1]


def test_average_ref_duplicates_radius_and_flux_gates():
    from jwst_gc_pipeline.astrometry.build_treasury_pm import average_ref_duplicates
    ref = _ref_dict([dict(obs=0, dx=0, dy=0), dict(obs=1, dx=25, dy=0)])
    assert average_ref_duplicates(ref, 0.020)[0]['n'] == 2
    assert average_ref_duplicates(ref, 0.030)[0]['n'] == 1
    # a blend of two stars in one tile: three times the flux of the other tile's row
    ref = _ref_dict([dict(obs=0, dx=0, dy=0, flux=3e-4), dict(obs=1, dx=5, dy=0, flux=1e-4)])
    assert average_ref_duplicates(ref, 0.020)[0]['n'] == 2
    ref = _ref_dict([dict(obs=0, dx=0, dy=0, flux=1.05e-4), dict(obs=1, dx=5, dy=0, flux=1e-4)])
    assert average_ref_duplicates(ref, 0.020)[0]['n'] == 1


def test_average_ref_duplicates_pairs_only_mutual_nearest_rows():
    from jwst_gc_pipeline.astrometry.build_treasury_pm import average_ref_duplicates
    ref = _ref_dict([dict(obs=0, dx=0, dy=0), dict(obs=1, dx=5, dy=0), dict(obs=1, dx=-9, dy=0)])
    out, info = average_ref_duplicates(ref, 0.020)
    # the nearer copy joins the star; the other row of that observation stays as it is
    assert out['n'] == 2 and out['n_merged'].tolist() == [2, 1]
    np.testing.assert_allclose(_offset_mas(ref, 2, 0)[0], 9.0, atol=1e-3)
    # equal frames: the lower observation is the primary
    assert out['row'].tolist() == [100, 102] and out['obs_index'].tolist() == [0, 1]
    assert out['sc'][1].separation(ref['sc'][2]).mas < 1e-6
    # and the other way round: two rows of observation 0 near one row of observation 1
    ref = _ref_dict([dict(obs=0, dx=-9, dy=0), dict(obs=0, dx=0, dy=0), dict(obs=1, dx=5, dy=0)])
    out, info = average_ref_duplicates(ref, 0.020)
    assert out['n'] == 2 and out['n_merged'].tolist() == [1, 2] and out['row'].tolist() == [100, 101]
    assert out['sc'][0].separation(ref['sc'][0]).mas < 1e-6


def test_nearest_per_key_keeps_the_smallest_separation_of_each_key():
    from jwst_gc_pipeline.astrometry.build_treasury_pm import _nearest_per_key
    rng = np.random.default_rng(3)
    key = rng.integers(0, 40, 300)
    sep = rng.permutation(300).astype(float)
    mask = _nearest_per_key(key, sep)
    expected = np.zeros(300, bool)
    for k in np.unique(key):
        at = np.flatnonzero(key == k)
        expected[at[np.argmin(sep[at])]] = True
    np.testing.assert_array_equal(mask, expected)
    assert _nearest_per_key(np.zeros(0, np.int64), np.zeros(0)).shape == (0,)


def test_average_ref_duplicates_recovers_every_star_seen_by_several_observations():
    from jwst_gc_pipeline.astrometry.build_treasury_pm import average_ref_duplicates
    rng = np.random.default_rng(11)
    n_star = 400
    pos = rng.uniform(-60_000, 60_000, (n_star, 2))                  # mas; stars are >> 20 mas apart
    seen = rng.random((n_star, 3)) < 0.6
    rows, star_of = [], []
    for s in range(n_star):
        for obs in np.flatnonzero(seen[s]):
            jitter = rng.normal(0, 2.0, 2)
            rows.append(dict(obs=int(obs), dx=pos[s, 0] + jitter[0], dy=pos[s, 1] + jitter[1], frames=1 + (s + obs) % 5))
            star_of.append(s)
    perm = rng.permutation(len(rows))                                # row order independent of star and tile
    rows, star_of = [rows[k] for k in perm], np.asarray(star_of)[perm]
    ref = _ref_dict(rows)
    out, info = average_ref_duplicates(ref, 0.020)
    n_seen = seen.sum(axis=1)
    assert out['n'] == (n_seen > 0).sum() and info['n_components_multi'] == info['n_groups'] == (n_seen > 1).sum()
    primary_star = star_of[out['row'] - 100]
    assert len(np.unique(primary_star)) == out['n']                  # every star once
    np.testing.assert_array_equal(out['n_merged'], n_seen[primary_star])
    np.testing.assert_array_equal(out['n_frames'], [sum(ref['n_frames'][star_of == s]) for s in primary_star])


def test_average_ref_duplicates_merges_three_observations():
    from jwst_gc_pipeline.astrometry.build_treasury_pm import average_ref_duplicates
    ref = _ref_dict([dict(obs=0, dx=0, dy=0, frames=1), dict(obs=1, dx=6, dy=0, frames=2),
                     dict(obs=2, dx=-3, dy=4, frames=3)])
    out, info = average_ref_duplicates(ref, 0.020)
    assert out['n'] == 1 and out['n_merged'].tolist() == [3] and info['groups_by_size'] == {3: 1}
    assert out['obs_index'].tolist() == [2] and out['n_frames'].tolist() == [6.0]
    dlon, dlat = ref['sc'][0].spherical_offsets_to(out['sc'][0])
    np.testing.assert_allclose([dlon.to_value(u.mas), dlat.to_value(u.mas)],
                               [(2 * 6 + 3 * -3) / 6, (3 * 4) / 6], atol=1e-4)


def test_average_ref_duplicates_leaves_ambiguous_clusters_alone():
    from jwst_gc_pipeline.astrometry.build_treasury_pm import average_ref_duplicates
    # a chain A(0)-B(1)-C(2)-A'(0): two rows of observation 0 in one cluster
    chain = _ref_dict([dict(obs=0, dx=0, dy=0), dict(obs=1, dx=6, dy=0),
                       dict(obs=2, dx=12, dy=0), dict(obs=0, dx=18, dy=0)])
    out, info = average_ref_duplicates(chain, 0.020)
    assert out['n'] == 4 and info['n_groups'] == 0 and info['n_components_multi'] == 1
    # B and C, 24 mas apart, both close to A: the cluster's ends are not themselves a pair
    ends = _ref_dict([dict(obs=0, dx=0, dy=0), dict(obs=1, dx=-12, dy=0), dict(obs=2, dx=12, dy=0)])
    out, info = average_ref_duplicates(ends, 0.020)
    assert out['n'] == 3 and out['n_merged'].tolist() == [1, 1, 1] and info['n_components_multi'] == 1


def test_average_ref_duplicates_flags_saturation_and_rescue_like_the_copies():
    from jwst_gc_pipeline.astrometry.build_treasury_pm import average_ref_duplicates
    ref = _ref_dict([dict(obs=0, dx=0, dy=0, sat=True), dict(obs=1, dx=4, dy=0, sat=False)])
    assert average_ref_duplicates(ref, 0.020)[0]['sat'].tolist() == [True]
    # a rescued copy beside a healthy one is absorbed with no weight, however many frames it has
    ref = _ref_dict([dict(obs=0, dx=0, dy=0, frames=2, resc=False), dict(obs=1, dx=8, dy=0, frames=9, resc=True)])
    out, info = average_ref_duplicates(ref, 0.020)
    assert out['n'] == 1 and out['n_merged'].tolist() == [2] and info['n_rescued_absorbed'] == 1
    assert out['err_rescued'].tolist() == [False] and out['obs_index'].tolist() == [0]
    assert out['sc'][0].separation(ref['sc'][0]).mas < 1e-6 and out['n_frames'].tolist() == [2.0]
    ref = _ref_dict([dict(obs=0, dx=0, dy=0, sat=False), dict(obs=1, dx=8, dy=0, sat=True, resc=True)])
    assert average_ref_duplicates(ref, 0.020)[0]['sat'].tolist() == [False]   # the absorbed copy does not count
    # two rescued copies average like any two, and the row stays rescued
    ref = _ref_dict([dict(obs=0, dx=0, dy=0, resc=True), dict(obs=1, dx=8, dy=0, resc=True)])
    out, info = average_ref_duplicates(ref, 0.020)
    assert out['err_rescued'].tolist() == [True] and info['n_rescued_absorbed'] == 0
    np.testing.assert_allclose(_offset_mas(ref, 0, 1)[0] / 2, 4.0, atol=1e-3)


def test_average_ref_duplicates_without_frame_counts_weights_the_copies_equally():
    from jwst_gc_pipeline.astrometry.build_treasury_pm import average_ref_duplicates
    ref = _ref_dict([dict(obs=0, dx=0, dy=0), dict(obs=1, dx=10, dy=0)])
    del ref['n_frames']
    out, info = average_ref_duplicates(ref, 0.020)
    assert 'n_frames' not in out
    dlon, dlat = ref['sc'][0].spherical_offsets_to(out['sc'][0])
    np.testing.assert_allclose(dlon.to_value(u.mas), 5.0, atol=1e-4)
    # NaN or sub-1 counts (a catalogue without the column) weigh as one frame
    ref = _ref_dict([dict(obs=0, dx=0, dy=0, frames=np.nan), dict(obs=1, dx=10, dy=0, frames=0)])
    dlon, dlat = ref['sc'][0].spherical_offsets_to(average_ref_duplicates(ref, 0.020)[0]['sc'][0])
    np.testing.assert_allclose(dlon.to_value(u.mas), 5.0, atol=1e-4)


def test_average_ref_duplicates_leaves_every_other_row_bit_identical():
    from jwst_gc_pipeline.astrometry.build_treasury_pm import average_ref_duplicates
    rng = np.random.default_rng(7)
    rows = [dict(obs=k % 3, dx=x, dy=y, flux=10 ** rng.uniform(-5, -4), ex=rng.uniform(1, 5),
                 frames=int(rng.integers(1, 5)))
            for k, (x, y) in enumerate(rng.uniform(-30_000, 30_000, (300, 2)))]
    rows[3]['frames'] = 3     # row 3 is the primary of the pair it forms with the extra row below
    rows.append(dict(obs=1, dx=rows[3]['dx'] + 4, dy=rows[3]['dy'], flux=rows[3]['flux'], frames=2))
    ref = _ref_dict(rows)
    out, info = average_ref_duplicates(ref, 0.020)
    assert info['n_groups'] == 1 and out['n'] == 300
    others = np.ones(300, bool)
    others[3] = False   # the merged row sits at its primary's slot
    for k in ('ex', 'ey', 'flux', 'mag', 'sat', 'flux_noise', 'obs_index', 'row', 'err_rescued', 'n_frames'):
        np.testing.assert_array_equal(out[k][others], ref[k][:300][others], err_msg=k)
    np.testing.assert_array_equal(out['sc'].ra.deg[others], ref['sc'].ra.deg[:300][others])
    np.testing.assert_array_equal(out['sc'].dec.deg[others], ref['sc'].dec.deg[:300][others])
    assert out['n_merged'][others].tolist() == [1] * 299 and out['n_merged'][3] == 2
    assert out['row'][3] == 103 and out['n_frames'][3] == 5.0


def test_load_one_reads_the_frame_counts_when_the_catalog_has_them(tmp_path):
    from jwst_gc_pipeline.astrometry.build_treasury_pm import _load_one
    path = tmp_path / 'cat.fits'
    _write_rescue_fits(path, np.random.default_rng(21))
    assert np.isnan(_load_one(str(path), 'f212n', 2024.7, 5.0, 0)['n_frames']).all()
    t = Table.read(path)
    t = t[np.random.default_rng(1).permutation(len(t))]            # usable rows are not a prefix
    t['nmatch_good_f212n'] = np.arange(len(t)) % 5 + 1
    t.write(path, overwrite=True)
    cat = _load_one(str(path), 'f212n', 2024.7, 5.0, 0)
    assert 0 < cat['n'] < len(t) and cat['row'].max() >= cat['n']
    np.testing.assert_array_equal(cat['n_frames'], cat['row'] % 5 + 1)


def test_load_and_tie_ref_catalogs_carries_the_frame_counts_of_every_tile(tmp_path):
    rng = np.random.default_rng(5)
    n = 500
    dra, ddec = rng.uniform(-60, 60, n), rng.uniform(-60, 60, n)
    flux = rng.uniform(1e-8, 1e-6, n)
    src = dict(sc=SkyCoord((CENTER.ra.deg + dra / 3600) * u.deg, (CENTER.dec.deg + ddec / 3600) * u.deg),
               mag=-2.5 * np.log10(flux), n=n)
    paths, counts = [], []
    for k in range(2):
        sc = SkyCoord((CENTER.ra.deg + (dra + rng.normal(0, 0.005, n)) / 3600) * u.deg,
                      (CENTER.dec.deg + (ddec + rng.normal(0, 0.005, n)) / 3600) * u.deg)
        _write_ref_fits(tmp_path / f'ref{k}.fits', sc, flux * (1.0 + rng.normal(0, 0.01, n)))
        t = Table.read(tmp_path / f'ref{k}.fits')
        counts.append(rng.integers(1, 7, n))
        t['nmatch_good_f212n'] = counts[-1]
        t.write(tmp_path / f'ref{k}.fits', overwrite=True)
        paths.append(str(tmp_path / f'ref{k}.fits'))
    ref, diags = load_and_tie_ref_catalogs(paths, 'f212n', 2026.0, src, tie_magcut=None,
                                           tie_bright_percentile=20.0)
    assert len(ref['n_frames']) == ref['n'] == len(ref['sc'])
    for k in range(2):
        rows = ref['row'][ref['obs_index'] == k]
        np.testing.assert_array_equal(ref['n_frames'][ref['obs_index'] == k], counts[k][rows])


def _two_tile_field(tmp_path, rng):
    """src plus two ref 'tiles', each seeing a random part of the stars (independent few-mas noise,
    shuffled rows).  Returns src, the two ref paths and ``member[tile][star]``; a star's number is
    its row in the src file."""
    src, ref_all, n_all, n_comp = _synthetic_two_epoch_field(tmp_path, rng)
    t = Table.read(ref_all)
    sc, flux = t['skycoord'], np.asarray(t['flux'])
    member = [rng.random(n_all) < 0.9, rng.random(n_all) < 0.5]
    paths = []
    for k, m in enumerate(member):
        stars = rng.permutation(np.flatnonzero(m))
        extra = 0.003 * k                                             # the second tile is 3 mas noisier
        s = sc[stars]
        _write_ref_fits(tmp_path / f'tile{k}.fits',
                        SkyCoord(s.ra + rng.normal(0, extra, len(stars)) * u.arcsec / np.cos(s.dec.rad),
                                 s.dec + rng.normal(0, extra, len(stars)) * u.arcsec),
                        flux[stars] * (1.0 + rng.normal(0, 0.01 * k, len(stars))))
        paths.append(str(tmp_path / f'tile{k}.fits'))
    return src, paths, member, n_all


def test_build_averages_cross_tile_duplicates_only_when_asked(tmp_path):
    pytest.importorskip('flystar')
    from jwst_gc_pipeline.astrometry import build_treasury_pm as btp
    src, refs, member, n_all = _two_tile_field(tmp_path, np.random.default_rng(43))
    n_both = int((member[0] & member[1]).sum())
    kw = dict(verbose=False, emit_rows=True)
    plain = btp.build([src], refs, 'f212n', 2024.7, 2026.7, str(tmp_path / 'plain.fits'), **kw)
    assert 'ref_n_merged' not in plain.colnames and 'ref_dup_avg_json' not in plain.meta
    avg = btp.build([src], refs, 'f212n', 2024.7, 2026.7, str(tmp_path / 'avg.fits'),
                    ref_average_duplicates=0.030, **kw)
    assert avg.meta['ref_average_duplicates'] == 0.030
    info = json.loads(avg.meta['ref_dup_avg_json'])
    assert info['radius_arcsec'] == 0.030
    assert info['n_groups'] > 0.9 * n_both and info['n_rows_out'] == info['n_rows_in'] - info['n_groups']
    # a pair's ref row stands for as many catalogue rows as the tiles hold that star
    expected = member[0][avg['src_row']].astype(int) + member[1][avg['src_row']]
    assert (np.asarray(avg['ref_n_merged']) == expected).mean() > 0.97
    assert (expected == 2).sum() > 0.8 * n_both
    # ... and names a real row of the first tile that has it (the primary: equal frames, lowest observation)
    first = np.where(member[0][avg['src_row']], 0, 1)
    assert (np.asarray(avg['ref_obs_index']) == first).mean() > 0.97
    assert Table.read(tmp_path / 'avg.fits')['ref_n_merged'].tolist() == avg['ref_n_merged'].tolist()
    # a merged pair's PM error loses the ref half of its variance: ex/sqrt(2) (std columns: 1e-6 deg)
    both = np.intersect1d(plain['src_row'], avg['src_row'])
    p = plain[np.isin(plain['src_row'], both)]
    a = avg[np.isin(avg['src_row'], both)]
    p, a = p[np.argsort(p['src_row'])], a[np.argsort(a['src_row'])]
    merged = np.asarray(a['ref_n_merged']) == 2
    assert merged.sum() > 0.8 * n_both
    dt = 2026.7 - 2024.7
    np.testing.assert_allclose(np.asarray(p['pm_ra_err'])[merged] ** 2 - np.asarray(a['pm_ra_err'])[merged] ** 2,
                               (3.6 / dt) ** 2 / 2, rtol=0.01)
    assert (np.asarray(p['pm_ra_err'])[~merged] == np.asarray(a['pm_ra_err'])[~merged]).mean() > 0.97


def test_build_rejects_a_bad_duplicate_radius_before_reading_anything():
    from jwst_gc_pipeline.astrometry import build_treasury_pm as btp
    for bad in (0.0, -0.01, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='ref_average_duplicates'):
            btp.build(['no_such_src.fits'], ['no_such_ref.fits'], 'f212n', 2024.7, 2026.7, 'o.fits',
                      ref_average_duplicates=bad)


def test_cli_ref_average_duplicates_flag(monkeypatch):
    import sys
    from jwst_gc_pipeline.astrometry import build_treasury_pm as btp
    seen = {}
    monkeypatch.setattr(btp, 'build', lambda *a, **k: seen.update(k) or None)
    base = ['x', '--src', 's.fits', '--src-epoch', '2024.7', '--ref', 'r.fits', '--ref-epoch', '2026.7',
            '--filter', 'f212n', '--out', 'o.fits']
    monkeypatch.setattr(sys, 'argv', base)
    btp.main()
    assert seen['ref_average_duplicates'] is None
    monkeypatch.setattr(sys, 'argv', base + ['--ref-average-duplicates', '0.02'])
    btp.main()
    assert seen['ref_average_duplicates'] == 0.02
    monkeypatch.setattr(btp, 'build', lambda *a, **k: pytest.fail('build must not run'))
    monkeypatch.setattr(sys, 'argv', base + ['--ref-average-duplicates', '0'])
    with pytest.raises(SystemExit):
        btp.main()


# --- tile-frame alignment of the duplicates (tile_offsets, anchor) ----------------------------

def _write_offsets(path, fields, tiles=(0, 1, 2), x0=-100.0, y0=-100.0, pix=1.0, **over):
    """A tile-frame-error file: ``fields`` (n_tiles, 2, ny, nx) mas about CENTER."""
    kw = dict(fields=np.asarray(fields, float), tiles=np.asarray(tiles), origin_deg=np.array([CENTER.ra.deg, CENTER.dec.deg]),
              x0=x0, y0=y0, pixel_arcsec=pix)
    kw.update(over)
    np.savez(path, **{k: v for k, v in kw.items() if v is not None})
    return str(path)


def _const_fields(per_tile, ny=200, nx=200):
    """Per-tile constant (east, north) mas fields on the default grid."""
    return np.broadcast_to(np.asarray(per_tile, float)[:, :, None, None], (len(per_tile), 2, ny, nx)).copy()


def test_load_tile_frame_offsets_reads_the_fields_and_refuses_bad_files(tmp_path):
    import hashlib
    from jwst_gc_pipeline.astrometry.build_treasury_pm import load_tile_frame_offsets
    good = _write_offsets(tmp_path / 'offs.npz', _const_fields([(0, 0), (2, -1), (-1, 3)]))
    off = load_tile_frame_offsets(good)
    assert off['fields'].shape == (3, 2, 200, 200) and off['tiles'].tolist() == [0, 1, 2]
    assert off['file'] == 'offs.npz'
    assert off['sha256'] == hashlib.sha256((tmp_path / 'offs.npz').read_bytes()).hexdigest()[:16]
    assert off['x0'] == -100.0 and off['pixel_arcsec'] == 1.0
    for key in ('fields', 'tiles', 'origin_deg', 'x0', 'y0', 'pixel_arcsec'):          # each of the six is required
        with np.load(good) as z:
            kept = {k: z[k] for k in z.files if k != key}
        np.savez(tmp_path / f'lacks_{key}.npz', **kept)
        with pytest.raises(ValueError, match=f'lacks.*{key}'):
            load_tile_frame_offsets(str(tmp_path / f'lacks_{key}.npz'))
    with pytest.raises(ValueError, match='fields must be'):                             # more fields than tiles
        load_tile_frame_offsets(_write_offsets(tmp_path / 'b.npz', _const_fields([(0, 0)] * 3), tiles=(0, 1)))
    with pytest.raises(ValueError, match='fields must be'):                             # fewer fields than tiles
        load_tile_frame_offsets(_write_offsets(tmp_path / 'b2.npz', _const_fields([(0, 0)] * 2), tiles=(0, 1, 2)))
    with pytest.raises(ValueError, match='fields must be'):                             # an empty grid
        load_tile_frame_offsets(_write_offsets(tmp_path / 'b3.npz', _const_fields([(0, 0)] * 3, ny=0, nx=5)))
    with pytest.raises(ValueError, match='fields must be'):
        load_tile_frame_offsets(_write_offsets(tmp_path / 'c.npz', _const_fields([(0, 0)] * 3)[:, :1]))
    with pytest.raises(ValueError, match='duplicate tile'):
        load_tile_frame_offsets(_write_offsets(tmp_path / 'd.npz', _const_fields([(0, 0)] * 3), tiles=(0, 1, 1)))
    bad = _const_fields([(0, 0)] * 3)
    bad[1, 0, 5, 5] = np.nan
    with pytest.raises(ValueError, match='non-finite'):
        load_tile_frame_offsets(_write_offsets(tmp_path / 'e.npz', bad))
    with pytest.raises(ValueError, match='pixel_arcsec'):
        load_tile_frame_offsets(_write_offsets(tmp_path / 'f.npz', _const_fields([(0, 0)] * 3), pix=0.0))


def _sc_arr(dx_mas, dy_mas):
    ra = CENTER.ra.deg + np.asarray(dx_mas, float) / 1e3 / 3600 / np.cos(CENTER.dec.rad)
    return SkyCoord(ra * u.deg, (CENTER.dec.deg + np.asarray(dy_mas, float) / 1e3 / 3600) * u.deg)


def test_tile_frame_shift_is_the_field_difference_between_the_tiles(tmp_path):
    from jwst_gc_pipeline.astrometry.build_treasury_pm import load_tile_frame_offsets, tile_frame_shift
    ny = nx = 200
    xc = -100.0 + (np.arange(nx) + 0.5)
    F = _const_fields([(1.0, -2.0), (0.0, 0.0), (0.0, 0.0)])
    F[1, 0] = 0.1 * xc[None, :]                     # tile 1: east error grows 0.1 mas per arcsec of east
    F[1, 1] = 0.05 * xc[:, None]                    # ... and north error 0.05 mas per arcsec of north (rows = y)
    off = load_tile_frame_offsets(_write_offsets(tmp_path / 'offs.npz', F))
    x, y = np.array([-12.3, 0.0, 7.77, 30.0]), np.array([5.0, -40.0, 0.25, 12.5])
    sc = _sc_arr(x * 1e3, y * 1e3)
    n = len(x)
    e, nn = tile_frame_shift(off, sc, np.full(n, 1), np.full(n, 0))     # tile 1 -> tile 0: F_0 - F_1
    np.testing.assert_allclose(e, 1.0 - 0.1 * x, atol=2e-3)
    np.testing.assert_allclose(nn, -2.0 - 0.05 * y, atol=2e-3)
    e2, n2 = tile_frame_shift(off, sc, np.full(n, 0), np.full(n, 1))    # the opposite move
    np.testing.assert_allclose([e2, n2], [-e, -nn], atol=1e-12)
    e3, n3 = tile_frame_shift(off, sc, np.full(n, 1), np.full(n, 1))    # same tile: nothing moves
    assert (e3 == 0).all() and (n3 == 0).all()
    # copies of one star in different tiles each move with their own tile pair
    e4, n4 = tile_frame_shift(off, sc[:2], np.array([1, 2]), np.array([0, 0]))
    np.testing.assert_allclose(e4, [1.0 - 0.1 * x[0], 1.0], atol=2e-3)
    # beyond the grid the edge value holds (east edge pixel centre at +99.5")
    far = _sc_arr([150e3], [0.0])
    np.testing.assert_allclose(tile_frame_shift(off, far, np.array([1]), np.array([0]))[0], 1.0 - 0.1 * 99.5, atol=2e-3)
    with pytest.raises(ValueError, match='covers observations'):
        tile_frame_shift(off, sc[:1], np.array([1]), np.array([7]))


def test_tile_frame_shift_reads_a_sloped_field_on_an_asymmetric_grid_by_tile_id(tmp_path):
    """x0 != y0, nx != ny, a 2" pixel and a tile list that is not 0, 1, 2 in order: the grid origin, the pixel scale, the
    half pixel, the axis order and the tile-id lookup each change the answer.  The fields are linear in east and north,
    which the bilinear read reproduces exactly inside the grid."""
    from jwst_gc_pipeline.astrometry.build_treasury_pm import load_tile_frame_offsets, tile_frame_shift
    x0, y0, pix, nx, ny = -40.0, -70.0, 2.0, 60, 100                      # east -40..80", north -70..130"
    coef = {5: ((0.30, 0.020, -0.010), (-0.50, 0.015, 0.025)),            # (east, north) error = a + b * east" + c * north"
            2: ((-0.10, 0.005, 0.004), (0.20, -0.012, 0.008)),
            7: ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))}

    def field(tile, x, y):
        return [a + b * x + c * y for a, b, c in coef[tile]]

    X, Y = np.meshgrid(x0 + pix * (np.arange(nx) + 0.5), y0 + pix * (np.arange(ny) + 0.5))      # (ny, nx)
    tiles = (7, 5, 2)
    off = load_tile_frame_offsets(_write_offsets(tmp_path / 'sloped.npz', np.array([field(t, X, Y) for t in tiles]),
                                                 tiles=tiles, x0=x0, y0=y0, pix=pix))
    px, py = np.array([-25.3, 12.0, 60.5, -38.0, 0.0]), np.array([3.7, 88.4, -41.2, 125.0, 0.0])
    sc = CENTER.spherical_offsets_by(px * u.arcsec, py * u.arcsec)
    for a, b in ((5, 7), (7, 5), (2, 5), (5, 2), (2, 7)):
        e, n = tile_frame_shift(off, sc, np.full(len(px), a), np.full(len(px), b))
        np.testing.assert_allclose([e, n], np.subtract(field(b, px, py), field(a, px, py)), atol=1e-5)
    ft, tt = np.array([5, 2, 7, 5, 2]), np.array([7, 5, 5, 2, 7])                  # several tile pairs in one call
    e, n = tile_frame_shift(off, sc, ft, tt)
    want = [[field(b, x, y)[ax] - field(a, x, y)[ax] for a, b, x, y in zip(ft, tt, px, py)] for ax in (0, 1)]
    np.testing.assert_allclose([e, n], want, atol=1e-5)


def _three_stars(F):
    """A (tiles 0, 1; the tile-1 copy has more frames), B (tile 2 alone), C (tiles 0, 1, 2, equal frames), each
    seen at its true position plus the tile's constant frame error F[t] (mas); returns rows and the true positions."""
    x = [(0.0, 0.0), (10_000.0, 0.0), (0.0, 10_000.0)]
    rows = [dict(obs=0, dx=x[0][0] + F[0][0], dy=x[0][1] + F[0][1], frames=2),
            dict(obs=1, dx=x[0][0] + F[1][0], dy=x[0][1] + F[1][1], frames=4),
            dict(obs=2, dx=x[1][0] + F[2][0], dy=x[1][1] + F[2][1])]
    rows += [dict(obs=t, dx=x[2][0] + F[t][0], dy=x[2][1] + F[t][1], frames=3) for t in (0, 1, 2)]
    return rows, _sc_arr([p[0] for p in x], [p[1] for p in x])


def _offset_from_truth(truth, sc, i):
    dlon, dlat = truth[i].spherical_offsets_to(sc[i])
    return dlon.to_value(u.mas), dlat.to_value(u.mas)


def test_average_ref_duplicates_with_tile_offsets_averages_in_the_anchor_frame(tmp_path):
    from jwst_gc_pipeline.astrometry.build_treasury_pm import average_ref_duplicates, load_tile_frame_offsets
    F = [(0.0, 0.0), (2.0, -1.0), (-1.0, 3.0)]
    off = load_tile_frame_offsets(_write_offsets(tmp_path / 'offs.npz', _const_fields(F)))
    rows, truth = _three_stars(F)
    ref = _ref_dict(rows)
    plain, _ = average_ref_duplicates(ref, 0.020)
    # no alignment: the frame-weighted mean of the tiles' frames (A: 2:4 frames, C: equal)
    np.testing.assert_allclose(_offset_from_truth(truth, plain['sc'], 0), [4 / 6 * 2.0, -4 / 6 * 1.0], atol=2e-3)
    np.testing.assert_allclose(_offset_from_truth(truth, plain['sc'], 2), [1 / 3, 2 / 3], atol=2e-3)
    out, info = average_ref_duplicates(ref, 0.020, tile_offsets=off)
    assert out['n'] == 3 and out['n_merged'].tolist() == [2, 1, 3]
    # default anchor = the primary copy: A -> tile 1 (most frames), C -> tile 0 (equal frames, lowest observation)
    np.testing.assert_allclose(_offset_from_truth(truth, out['sc'], 0), [2.0, -1.0], atol=2e-3)
    np.testing.assert_allclose(_offset_from_truth(truth, out['sc'], 2), [0.0, 0.0], atol=2e-3)
    # the row without duplicates is not touched (bit for bit)
    assert out['sc'].ra.deg[1] == ref['sc'].ra.deg[2] and out['sc'].dec.deg[1] == ref['sc'].dec.deg[2]
    # everything but the position is the plain average's
    for k in ('ex', 'ey', 'flux', 'mag', 'sat', 'obs_index', 'row', 'n_frames', 'n_merged'):
        np.testing.assert_array_equal(out[k], plain[k], err_msg=k)
    ti = info['tile_offsets']
    assert ti['file'] == 'offs.npz' and ti['anchor'] == 'primary copy' and ti['n_copies_shifted'] == 3
    assert ti['anchor_tile_counts'] == {0: 1, 1: 1}
    # shifted copies: A's tile-0 copy by F_1 - F_0 = (2, -1); C's tile-1 and tile-2 copies by F_0 - F_1 = (-2, 1), F_0 - F_2 = (1, -3)
    np.testing.assert_allclose(ti['rms_shift_mas'], [np.sqrt((4 + 4 + 1) / 3), np.sqrt((1 + 1 + 9) / 3)], rtol=1e-6)
    json.dumps(info)
    assert 'tile_offsets' not in average_ref_duplicates(ref, 0.020)[1]


def test_average_ref_duplicates_anchor_flags_choose_the_frame(tmp_path):
    from jwst_gc_pipeline.astrometry.build_treasury_pm import average_ref_duplicates, load_tile_frame_offsets
    F = [(0.0, 0.0), (2.0, -1.0), (-1.0, 3.0)]
    off = load_tile_frame_offsets(_write_offsets(tmp_path / 'offs.npz', _const_fields(F)))
    rows, truth = _three_stars(F)
    ref = _ref_dict(rows)
    flag = np.zeros(len(rows), bool)
    flag[0] = True                      # A: the tile-0 copy, although the tile-1 copy is the primary
    flag[5] = True                      # C: the tile-2 copy
    out, info = average_ref_duplicates(ref, 0.020, tile_offsets=off, anchor=flag)
    np.testing.assert_allclose(_offset_from_truth(truth, out['sc'], 0), [0.0, 0.0], atol=2e-3)
    np.testing.assert_allclose(_offset_from_truth(truth, out['sc'], 2), [-1.0, 3.0], atol=2e-3)
    ti = info['tile_offsets']
    assert ti['anchor'] == 'flagged copy, else primary'
    assert (ti['n_groups_one_flagged_copy'], ti['n_groups_several_flagged_copies'], ti['n_groups_no_flagged_copy']) == (2, 0, 0)
    assert ti['anchor_tile_counts'] == {0: 1, 2: 1}
    # several flagged copies: the first in primary order (A: the tile-1 copy); C unflagged: its primary
    flag = np.zeros(len(rows), bool)
    flag[[0, 1]] = True
    out, info = average_ref_duplicates(ref, 0.020, tile_offsets=off, anchor=flag)
    np.testing.assert_allclose(_offset_from_truth(truth, out['sc'], 0), [2.0, -1.0], atol=2e-3)
    np.testing.assert_allclose(_offset_from_truth(truth, out['sc'], 2), [0.0, 0.0], atol=2e-3)
    ti = info['tile_offsets']
    assert (ti['n_groups_one_flagged_copy'], ti['n_groups_several_flagged_copies'], ti['n_groups_no_flagged_copy']) == (0, 1, 1)
    # nothing flagged: every group falls back to its primary copy
    out, info = average_ref_duplicates(ref, 0.020, tile_offsets=off, anchor=np.zeros(len(rows), bool))
    np.testing.assert_allclose(_offset_from_truth(truth, out['sc'], 0), [2.0, -1.0], atol=2e-3)
    assert info['tile_offsets']['n_groups_no_flagged_copy'] == 2
    with pytest.raises(ValueError, match='needs tile_offsets'):
        average_ref_duplicates(ref, 0.020, anchor=flag)


def test_average_ref_duplicates_reads_the_tile_fields_at_each_copys_own_position(tmp_path):
    from jwst_gc_pipeline.astrometry.build_treasury_pm import average_ref_duplicates, load_tile_frame_offsets
    F = _const_fields([(0, 0), (0, 0), (0, 0)])
    F[1, 0] = 20.0 * (-100.0 + (np.arange(200) + 0.5))[None, :]       # tile 1: 20 mas of east error per arcsec of east
    off = load_tile_frame_offsets(_write_offsets(tmp_path / 'slope.npz', F))
    ref = _ref_dict([dict(obs=0, dx=750.0, dy=0.0, frames=3), dict(obs=1, dx=765.0, dy=0.0, frames=3)])
    out, _ = average_ref_duplicates(ref, 0.020, tile_offsets=off)
    assert out['n'] == 1 and out['n_merged'].tolist() == [2]
    # the tile-0 copy (primary, anchor) stays; the tile-1 copy at 765 mas east moves by F_0 - F_1 at ITS position, -15.3 mas
    # (the primary's position would give -15.0 mas)
    east = CENTER.spherical_offsets_to(out['sc'])[0].to_value(u.mas)
    np.testing.assert_allclose(east, [(750.0 + 765.0 - 15.3) / 2], atol=0.02)


def test_average_ref_duplicates_zero_tile_offsets_change_nothing(tmp_path):
    from jwst_gc_pipeline.astrometry.build_treasury_pm import average_ref_duplicates, load_tile_frame_offsets
    off = load_tile_frame_offsets(_write_offsets(tmp_path / 'zero.npz', _const_fields([(0, 0)] * 3)))
    rng = np.random.default_rng(11)
    rows = [dict(obs=k % 3, dx=x, dy=y, flux=10 ** rng.uniform(-5, -4), frames=int(rng.integers(1, 5)))
            for k, (x, y) in enumerate(rng.uniform(-30_000, 30_000, (200, 2)))]
    for k in (3, 10, 17):
        rows.append(dict(obs=(k + 1) % 3, dx=rows[k]['dx'] + 4, dy=rows[k]['dy'] - 2, flux=rows[k]['flux'], frames=2))
    ref = _ref_dict(rows)
    plain, info0 = average_ref_duplicates(ref, 0.020)
    zero, info1 = average_ref_duplicates(ref, 0.020, tile_offsets=off)
    assert info0['n_groups'] == 3
    np.testing.assert_array_equal(zero['sc'].ra.deg, plain['sc'].ra.deg)
    np.testing.assert_array_equal(zero['sc'].dec.deg, plain['sc'].dec.deg)
    assert info1['tile_offsets']['n_copies_shifted'] == 3 and info1['tile_offsets']['max_abs_shift_mas'] == 0.0


def test_average_ref_duplicates_tile_offsets_without_duplicates_and_with_a_missing_tile(tmp_path):
    from jwst_gc_pipeline.astrometry.build_treasury_pm import average_ref_duplicates, load_tile_frame_offsets
    off = load_tile_frame_offsets(_write_offsets(tmp_path / 'offs.npz', _const_fields([(0, 0), (2, 0)]), tiles=(0, 1)))
    ref = _ref_dict([dict(obs=0, dx=0, dy=0), dict(obs=1, dx=900, dy=0)])
    out, info = average_ref_duplicates(ref, 0.020, tile_offsets=off)
    assert out['n'] == 2 and info['tile_offsets']['n_copies_shifted'] == 0 and info['n_groups'] == 0
    np.testing.assert_array_equal(out['sc'].ra.deg, ref['sc'].ra.deg)
    # a duplicate group that uses a tile the file lacks is refused rather than left unaligned
    ref = _ref_dict([dict(obs=0, dx=0, dy=0), dict(obs=2, dx=4, dy=0)])
    with pytest.raises(ValueError, match='covers observations'):
        average_ref_duplicates(ref, 0.020, tile_offsets=off)


def test_read_anchor_keys_and_ref_anchor_mask(tmp_path):
    from jwst_gc_pipeline.astrometry.build_treasury_pm import read_anchor_keys, ref_anchor_mask
    ref = _ref_dict([dict(obs=0, dx=0, dy=0), dict(obs=1, dx=4, dy=0), dict(obs=1, dx=900, dy=0), dict(obs=2, dx=-900, dy=0)])
    Table(dict(ref_obs_index=np.array([1, 2, 1, -1, 0, 2, -1]),
               ref_row=np.array([101, 103, 101, -1, 100, -1, 103]))).write(tmp_path / 'rec.fits')
    keys = read_anchor_keys(str(tmp_path / 'rec.fits'))
    assert len(keys) == 3                          # (-1, -1), (2, -1) and (-1, 103) dropped (a pair needs both), (1, 101) once
    assert ref_anchor_mask(ref, keys).tolist() == [True, True, False, True]
    Table(dict(ref_obs_index=np.array([2]), ref_row=np.array([999]))).write(tmp_path / 'other.fits')
    with pytest.raises(ValueError, match='none of the anchor table'):
        ref_anchor_mask(ref, read_anchor_keys(str(tmp_path / 'other.fits')))
    Table(dict(ref_row=np.array([1]))).write(tmp_path / 'bad.fits')
    with pytest.raises(ValueError, match='lacks'):
        read_anchor_keys(str(tmp_path / 'bad.fits'))


def test_build_aligns_cross_tile_duplicates_with_tile_offsets_only_when_asked(tmp_path):
    pytest.importorskip('flystar')
    from jwst_gc_pipeline.astrometry import build_treasury_pm as btp
    src, refs, member, n_all = _two_tile_field(tmp_path, np.random.default_rng(43))
    kw = dict(verbose=False, emit_rows=True, ref_average_duplicates=0.030)
    path = lambda name: str(tmp_path / name)      # noqa: E731
    zero = _write_offsets(tmp_path / 'zero.npz', _const_fields([(0, 0), (0, 0)]), tiles=(0, 1))
    shift = _write_offsets(tmp_path / 'shift.npz', _const_fields([(0, 0), (2.0, 0.0)]), tiles=(0, 1))
    plain = btp.build([src], refs, 'f212n', 2024.7, 2026.7, path('plain.fits'), verbose=False, emit_rows=True)
    base = btp.build([src], refs, 'f212n', 2024.7, 2026.7, path('base.fits'), **kw)
    with0 = btp.build([src], refs, 'f212n', 2024.7, 2026.7, path('zero.fits'), ref_dup_tile_offsets=zero, **kw)
    for c in ('src_row', 'ref_row', 'ref_obs_index', 'pm_ra', 'pm_dec', 'pm_ra_err', 'trustworthy'):
        assert np.asarray(with0[c]).tolist() == np.asarray(base[c]).tolist(), c
    assert json.loads(with0.meta['ref_dup_avg_json'])['tile_offsets']['file'] == 'zero.npz'
    assert 'tile_offsets' not in json.loads(base.meta['ref_dup_avg_json'])
    # tile 1's frame is 2 mas east of tile 0's: with the primary (equal frames: tile 0) as the anchor the tile-1 copy moves
    # 2 mas west and the mean 1 mas west, i.e. pm_ra changes by -1/dt mas/yr for a merged star and not for the others
    dt = 2026.7 - 2024.7
    prim = btp.build([src], refs, 'f212n', 2024.7, 2026.7, path('prim.fits'), ref_dup_tile_offsets=shift, **kw)
    assert (np.asarray(prim['src_row']) == np.asarray(base['src_row'])).all()
    d_ra = np.asarray(prim['pm_ra']) - np.asarray(base['pm_ra'])
    merged = np.asarray(base['ref_n_merged']) == 2
    assert merged.sum() > 100
    np.testing.assert_allclose(d_ra[merged], -1.0 / dt, atol=0.02 * 1.0 / dt)
    assert np.abs(d_ra[~merged]).max() < 1e-9
    # an east-only shift leaves pm_dec alone up to the round-off of positions held in float64 degrees (~2e-7 mas)
    np.testing.assert_allclose(np.asarray(prim['pm_dec']), np.asarray(base['pm_dec']), atol=1e-6)
    # the anchor table: a star whose plain pairing used a tile-1 copy is aligned to tile 1 (the mean moves 1 mas east)
    anch = btp.build([src], refs, 'f212n', 2024.7, 2026.7, path('anch.fits'), ref_dup_tile_offsets=shift,
                     ref_dup_anchor_table=path('plain.fits'), **kw)
    assert (np.asarray(anch['src_row']) == np.asarray(base['src_row'])).all()
    d_anch = np.asarray(anch['pm_ra']) - np.asarray(base['pm_ra'])
    first = dict(zip(np.asarray(plain['src_row']).tolist(), np.asarray(plain['ref_obs_index']).tolist()))
    used1 = np.array([first.get(s, -1) == 1 for s in np.asarray(base['src_row']).tolist()])
    in_plain = np.array([s in first for s in np.asarray(base['src_row']).tolist()])
    sel = merged & in_plain
    assert (sel & used1).sum() > 20 and (sel & ~used1).sum() > 20
    assert np.abs(d_anch[sel & used1] - 1.0 / dt).max() < 0.02 / dt
    assert np.abs(d_anch[sel & ~used1] + 1.0 / dt).max() < 0.02 / dt
    info = json.loads(anch.meta['ref_dup_avg_json'])['tile_offsets']
    assert info['anchor_table'] == 'plain.fits' and info['file'] == 'shift.npz'
    # each recorded hash is that of its own file (the loader test checks _file_sha256 against hashlib)
    assert info['anchor_table_sha256'] == btp._file_sha256(path('plain.fits'))
    assert info['sha256'] == btp._file_sha256(shift) != info['anchor_table_sha256']
    assert info['n_groups_one_flagged_copy'] > 0


def test_build_rejects_tile_offsets_without_duplicate_averaging_before_reading_anything():
    from jwst_gc_pipeline.astrometry import build_treasury_pm as btp
    args = (['no_such_src.fits'], ['no_such_ref.fits'], 'f212n', 2024.7, 2026.7, 'o.fits')
    with pytest.raises(ValueError, match='need ref_average_duplicates'):
        btp.build(*args, ref_dup_tile_offsets='no_such_offsets.npz')
    with pytest.raises(ValueError, match='need ref_average_duplicates'):
        btp.build(*args, ref_dup_anchor_table='no_such_table.fits')
    with pytest.raises(ValueError, match='needs ref_dup_tile_offsets'):
        btp.build(*args, ref_average_duplicates=0.02, ref_dup_anchor_table='no_such_table.fits')


def test_cli_ref_dup_tile_offsets_flags(monkeypatch):
    import sys
    from jwst_gc_pipeline.astrometry import build_treasury_pm as btp
    seen = {}
    monkeypatch.setattr(btp, 'build', lambda *a, **k: seen.update(k) or None)
    base = ['x', '--src', 's.fits', '--src-epoch', '2024.7', '--ref', 'r.fits', '--ref-epoch', '2026.7',
            '--filter', 'f212n', '--out', 'o.fits']
    monkeypatch.setattr(sys, 'argv', base)
    btp.main()
    assert seen['ref_dup_tile_offsets'] is None and seen['ref_dup_anchor_table'] is None
    monkeypatch.setattr(sys, 'argv', base + ['--ref-average-duplicates', '0.02', '--ref-dup-tile-offsets', 'f.npz'])
    btp.main()
    assert seen['ref_dup_tile_offsets'] == 'f.npz' and seen['ref_dup_anchor_table'] is None
    monkeypatch.setattr(sys, 'argv', base + ['--ref-average-duplicates', '0.02', '--ref-dup-tile-offsets', 'f.npz',
                                             '--ref-dup-anchor-table', 't.fits'])
    btp.main()
    assert seen['ref_dup_anchor_table'] == 't.fits'
    monkeypatch.setattr(btp, 'build', lambda *a, **k: pytest.fail('build must not run'))
    for extra in (['--ref-dup-tile-offsets', 'f.npz'], ['--ref-dup-anchor-table', 't.fits'],
                  ['--ref-average-duplicates', '0.02', '--ref-dup-anchor-table', 't.fits']):
        monkeypatch.setattr(sys, 'argv', base + extra)
        with pytest.raises(SystemExit):
            btp.main()
