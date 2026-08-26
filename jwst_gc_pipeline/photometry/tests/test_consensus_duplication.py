"""A duplicated visit consensus must say so -- and a merely CROWDED one must not.

Issue #484.  ``build_visit_consensus`` associates member exposures to the seed
with a ONE-WAY, non-exclusive nearest-neighbour query at ``match_radius``, and
appends every member star that finds no partner as a NEW consensus row.  An
exposure displaced by about ``match_radius`` therefore contributes a second
copy of stars that are already in the seed, and the consensus goes bimodal
silently.

gc2211 o023 F200W is the measured case: exposure 1 sits 156-201 mas from its
siblings, straddling the 0.2" radius.  ``measure_offset``'s unconstrained
``H.argmax()`` then locked onto the wrong mode on three of eight detectors and
blamed the two GOOD exposures -- fabricating a 74 mas per-detector spread.

Counting close pairs does NOT find that state: these fields are crowded enough
that a clean catalog already has a large population inside 0.2" (27 of the 75
consensus catalogs on disk exceed 5%, and o023's own 13.0% is what a Poisson
field of its density predicts).  What separates them is DIRECTION -- shadow
copies come from one rigid exposure displacement and share an axis, crowding
blends do not.  These tests pin both halves: the crowded-but-clean case stays
quiet, and the axis-aligned case speaks up.
"""
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from jwst_gc_pipeline.photometry.visit_consensus import (
    CONSENSUS_DUPLICATION_WARN_ALIGNED_FRAC, consensus_duplication)

RADIUS = 0.2 * u.arcsec
CENTRE = (266.5, -28.9)


def _field(n=4000, seed=7, width_arcsec=120.0):
    """A Poisson field.  ``n`` and ``width_arcsec`` set the density, which is
    the thing the old raw-fraction statistic was really measuring."""
    rng = np.random.default_rng(seed)
    half = width_arcsec / 3600.0 / 2.0
    ra = CENTRE[0] + rng.uniform(-half, half, n) / np.cos(np.radians(CENTRE[1]))
    dec = CENTRE[1] + rng.uniform(-half, half, n)
    return SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')


def _crowded(seed=7):
    """A CLEAN field at the density of the real archive: p50 NN near 0.29",
    which is where brick F200W o004 (0.272") and sgrb2 F210M o001 (0.290") sit.
    Nothing is duplicated in it."""
    return _field(n=9500, seed=seed, width_arcsec=60.0)


def _shadow_copies(coords, frac, offset_mas, pa_deg=0.0, seed=11, isotropic=False):
    """Append copies of ``frac`` of the stars, displaced by ``offset_mas``.

    ``isotropic=True`` scatters the copies' position angles uniformly instead of
    putting them all on ``pa_deg`` -- the control for "extra close pairs, no
    common direction", which is what crowding looks like.
    """
    rng = np.random.default_rng(seed)
    k = int(round(frac * len(coords)))
    pick = rng.choice(len(coords), size=k, replace=False)
    pa = (rng.uniform(0, 2 * np.pi, k) if isotropic
          else np.full(k, np.radians(pa_deg)))
    ddec = offset_mas * np.cos(pa) / 3.6e6
    dra = offset_mas * np.sin(pa) / 3.6e6 / np.cos(np.radians(CENTRE[1]))
    ra = np.concatenate([coords.ra.deg, coords.ra.deg[pick] + dra])
    dec = np.concatenate([coords.dec.deg, coords.dec.deg[pick] + ddec])
    return SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')


def test_a_clean_sparse_consensus_is_quiet():
    dup = consensus_duplication(_field(), match_radius=RADIUS)
    assert dup['n'] == 4000
    assert dup['nn_p50_arcsec'] > 0.2
    assert dup['aligned_frac'] < CONSENSUS_DUPLICATION_WARN_ALIGNED_FRAC


def test_a_clean_but_CROWDED_consensus_is_quiet_though_its_raw_fraction_is_large():
    """The regression this statistic was re-derived for.

    A clean catalog at archive density has a fifth to a quarter of its rows
    inside the 0.2" association radius purely from crowding -- 27 of the 75
    consensus catalogs on disk do, brick F200W o004 at 24.1% among them.  A
    trigger on that raw fraction fires on all of them.  The axis statistic must
    stay quiet, because none of those pairs share a direction.
    """
    dup = consensus_duplication(_crowded(), match_radius=RADIUS)
    assert dup['nn_p50_arcsec'] == pytest.approx(0.29, abs=0.03), (
        'the synthetic crowded field must actually reach archive density')
    assert dup['frac_within_match_radius'] > 0.2, (
        'a clean field of this density really does put a fifth of its rows '
        'inside the association radius -- that is the point')
    assert dup['aligned_frac'] < CONSENSUS_DUPLICATION_WARN_ALIGNED_FRAC
    # and it is quiet because it is ISOTROPIC, not because the pairs are absent
    assert dup['n_close_pairs'] > 1000
    assert dup['close_pair_axis_alignment'] == pytest.approx(
        dup['close_pair_axis_isotropic'], abs=0.02)


def test_shadow_copies_in_a_crowded_field_are_detected():
    """The o023 shape, on top of the crowding that hides it from a raw count."""
    coords = _shadow_copies(_crowded(), frac=0.02, offset_mas=120.0, pa_deg=63.0)
    dup = consensus_duplication(coords, match_radius=RADIUS)
    assert dup['aligned_frac'] > CONSENSUS_DUPLICATION_WARN_ALIGNED_FRAC
    assert dup['close_pair_axis_alignment'] > 5 * dup['close_pair_axis_isotropic'], (
        'the alignment has to stand well clear of the isotropic floor; o023 '
        'reads 0.251 against a floor of 0.012')


def test_the_same_number_of_ISOTROPIC_extra_pairs_stays_quiet():
    """Direction, not count, is the discriminator.

    Identical star count, identical number of extra pairs, identical separation
    -- only the position angles differ.  A count-based statistic cannot tell
    these two catalogs apart; this one must.
    """
    clean = _crowded()
    aligned = consensus_duplication(
        _shadow_copies(clean, frac=0.02, offset_mas=120.0, pa_deg=63.0),
        match_radius=RADIUS)
    scattered = consensus_duplication(
        _shadow_copies(clean, frac=0.02, offset_mas=120.0, isotropic=True),
        match_radius=RADIUS)
    assert aligned['n'] == scattered['n']
    assert scattered['frac_within_match_radius'] == pytest.approx(
        aligned['frac_within_match_radius'], abs=0.01), (
        'the two catalogs must be indistinguishable to a count-based statistic')
    assert aligned['aligned_frac'] > CONSENSUS_DUPLICATION_WARN_ALIGNED_FRAC
    assert scattered['aligned_frac'] < CONSENSUS_DUPLICATION_WARN_ALIGNED_FRAC


def test_the_displacement_axis_and_scale_are_reported():
    """A reader has to be able to tell an exposure shift from a blend, so the
    statistic reports WHICH axis and at WHAT separation, not just a flag."""
    coords = _shadow_copies(_crowded(), frac=0.02, offset_mas=120.0, pa_deg=63.0)
    dup = consensus_duplication(coords, match_radius=RADIUS)
    assert dup['axis_pa_deg'] == pytest.approx(63.0, abs=6.0)
    assert dup['aligned_sep_p50_arcsec'] == pytest.approx(0.120, abs=0.02)


def test_a_small_catalog_returns_nan_rather_than_a_meaningless_fraction():
    dup = consensus_duplication(_field(n=40), match_radius=RADIUS)
    assert dup['n'] == 40
    assert np.isnan(dup['frac_within_match_radius'])
    assert np.isnan(dup['nn_p50_arcsec'])
    assert np.isnan(dup['aligned_frac'])


def test_too_few_close_pairs_returns_nan_rather_than_a_noisy_axis():
    """With a handful of pairs the resultant length is dominated by shot noise,
    so the axis terms refuse rather than report."""
    dup = consensus_duplication(_field(n=300, width_arcsec=600.0),
                                match_radius=RADIUS)
    assert dup['n_close_pairs'] < 50
    assert np.isfinite(dup['frac_within_match_radius'])
    assert np.isnan(dup['aligned_frac'])
    assert np.isnan(dup['close_pair_axis_alignment'])


def test_the_radius_is_honoured():
    """The statistic is about the ASSOCIATION radius, so a build run at a
    different radius must be judged against that radius."""
    coords = _shadow_copies(_crowded(), frac=0.02, offset_mas=180.0, pa_deg=63.0)
    wide = consensus_duplication(coords, match_radius=RADIUS)
    narrow = consensus_duplication(coords, match_radius=0.1 * u.arcsec)
    assert narrow['frac_within_match_radius'] < wide['frac_within_match_radius']
    assert narrow['match_radius_arcsec'] == pytest.approx(0.1)
    # 180 mas copies fall OUTSIDE a 0.1" radius, so the narrow view cannot see
    # them and must not claim to
    assert wide['aligned_frac'] > CONSENSUS_DUPLICATION_WARN_ALIGNED_FRAC
    assert narrow['aligned_frac'] < CONSENSUS_DUPLICATION_WARN_ALIGNED_FRAC
    # a bare float is accepted as arcsec, as the builder's own helpers do
    assert consensus_duplication(coords, match_radius=0.2)['match_radius_arcsec'] \
        == pytest.approx(0.2)


def test_the_raw_close_pair_fraction_is_still_reported_as_context():
    """It is not the trigger any more, but it is the number a reader compares
    against the p50 NN separation, so it stays in the record."""
    dup = consensus_duplication(_crowded(), match_radius=RADIUS)
    for key in ('frac_within_match_radius', 'frac_within_half_radius',
                'nn_p50_arcsec', 'n_close_pairs'):
        assert np.isfinite(dup[key]), key


def test_the_builder_reports_it_and_the_checkpoint_records_it():
    """Pins the wiring.  The unit tests above pass even if nothing calls the
    helper, which is the whole point of adding it."""
    import inspect

    from jwst_gc_pipeline.photometry import astrometry_checkpoint as ac
    from jwst_gc_pipeline.photometry import visit_consensus as vc

    build_src = inspect.getsource(vc.build_visit_consensus)
    assert 'consensus_duplication(' in build_src, (
        'build_visit_consensus no longer measures its own duplication')
    assert 'CONSENSUS_DUPLICATION_WARN_ALIGNED_FRAC' in build_src, (
        'the duplication is measured but never surfaced in the log')
    assert 'frac_within_match_radius' not in build_src, (
        'the raw close-pair fraction is a DENSITY statistic (27 of 75 archive '
        'catalogs exceed 5%); it must not be the warn trigger again')

    ck_src = inspect.getsource(ac.run_visit_checkpoint)
    assert 'duplication=' in ck_src, (
        'the m2 record no longer carries the consensus duplication, so a '
        'bimodal consensus is invisible after the run')
