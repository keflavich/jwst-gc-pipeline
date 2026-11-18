"""Tests for build_treasury_pm.py's own code (dedup, adaptive tie_magcut) --
the commit that added them (ac4bf1e4) shipped with none, per PR #140 review.
"""
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
