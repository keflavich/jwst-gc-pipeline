"""A duplicated visit consensus must say so.

Issue #484.  ``build_visit_consensus`` associates member exposures to the seed
with a ONE-WAY, non-exclusive nearest-neighbour query at ``match_radius``, and
appends every member star that finds no partner as a NEW consensus row.  An
exposure displaced by about ``match_radius`` therefore contributes a second
copy of stars that are already in the seed, and the consensus goes bimodal
silently.

gc2211 o023 F200W is the measured case: exposure 1 sits 156-201 mas from its
siblings, straddling the 0.2" radius, and the consensus came out with 16.9% of
its 39,297 rows having a neighbour within 0.25" against a p50 NN of 0.441".
``measure_offset``'s unconstrained ``H.argmax()`` then locked onto the wrong
mode on three of eight detectors and blamed the two GOOD exposures --
fabricating a 74 mas per-detector spread.

``consensus_duplication`` is the cheap, direct tell for that state, and it is
independent of any offset measurement.  It reports; it decides nothing.
"""
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from jwst_gc_pipeline.photometry.visit_consensus import (
    CONSENSUS_DUPLICATION_WARN_FRAC, consensus_duplication)


def _field(n=4000, seed=7, centre=(266.5, -28.9), width_arcsec=120.0):
    """A plausible crowded-field catalog: random, so its NN distribution has the
    usual Poisson shape and essentially nothing below 0.2"."""
    rng = np.random.default_rng(seed)
    half = width_arcsec / 3600.0 / 2.0
    ra = centre[0] + rng.uniform(-half, half, n) / np.cos(np.radians(centre[1]))
    dec = centre[1] + rng.uniform(-half, half, n)
    return SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')


def _duplicate(coords, frac, offset_mas, seed=11):
    """Append shadow copies of ``frac`` of the stars, displaced by
    ``offset_mas`` in Dec -- what a displaced exposure appends."""
    rng = np.random.default_rng(seed)
    k = int(round(frac * len(coords)))
    pick = rng.choice(len(coords), size=k, replace=False)
    ra = np.concatenate([coords.ra.deg, coords.ra.deg[pick]])
    dec = np.concatenate([coords.dec.deg,
                          coords.dec.deg[pick] + offset_mas / 3.6e6])
    return SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')


def test_a_clean_consensus_reads_far_below_the_warn_fraction():
    dup = consensus_duplication(_field(), match_radius=0.2 * u.arcsec)
    assert dup['n'] == 4000
    assert dup['nn_p50_arcsec'] > 0.2, (
        'the synthetic field must be sparse enough that a clean NN '
        'distribution really does start above the association radius')
    assert dup['frac_within_match_radius'] < CONSENSUS_DUPLICATION_WARN_FRAC


def test_shadow_copies_at_the_association_radius_are_detected():
    """The o023 shape: a displaced exposure duplicates part of the seed at a
    separation just inside ``match_radius``."""
    coords = _duplicate(_field(), frac=0.17, offset_mas=180.0)
    dup = consensus_duplication(coords, match_radius=0.2 * u.arcsec)
    assert dup['frac_within_match_radius'] > CONSENSUS_DUPLICATION_WARN_FRAC
    # both members of each duplicated pair are within the radius, so the
    # fraction tracks 2 x the duplicated fraction rather than 1 x
    assert dup['frac_within_match_radius'] == pytest.approx(0.29, abs=0.05)


def test_the_separation_scale_is_reported_not_just_the_count():
    """0.18" duplicates land inside the radius but outside half of it; that is
    what distinguishes an exposure-displacement duplicate from a genuine close
    blend, and a reader needs both numbers."""
    coords = _duplicate(_field(), frac=0.17, offset_mas=180.0)
    dup = consensus_duplication(coords, match_radius=0.2 * u.arcsec)
    assert dup['frac_within_half_radius'] < 0.02
    assert dup['frac_within_match_radius'] > 0.2


def test_a_small_catalog_returns_nan_rather_than_a_meaningless_fraction():
    dup = consensus_duplication(_field(n=40), match_radius=0.2 * u.arcsec)
    assert dup['n'] == 40
    assert np.isnan(dup['frac_within_match_radius'])
    assert np.isnan(dup['nn_p50_arcsec'])


def test_the_radius_is_honoured():
    """The statistic is about the ASSOCIATION radius, so a build run at a
    different radius must be judged against that radius."""
    coords = _duplicate(_field(), frac=0.17, offset_mas=180.0)
    wide = consensus_duplication(coords, match_radius=0.2 * u.arcsec)
    narrow = consensus_duplication(coords, match_radius=0.1 * u.arcsec)
    assert narrow['frac_within_match_radius'] < wide['frac_within_match_radius']
    assert narrow['match_radius_arcsec'] == pytest.approx(0.1)
    # a bare float is accepted as arcsec, as the builder's own helpers do
    assert consensus_duplication(coords, match_radius=0.2)['match_radius_arcsec'] \
        == pytest.approx(0.2)


def test_the_builder_reports_it_and_the_checkpoint_records_it():
    """Pins the wiring.  The unit tests above pass even if nothing calls the
    helper, which is the whole point of adding it."""
    import inspect

    from jwst_gc_pipeline.photometry import astrometry_checkpoint as ac
    from jwst_gc_pipeline.photometry import visit_consensus as vc

    build_src = inspect.getsource(vc.build_visit_consensus)
    assert 'consensus_duplication(' in build_src, (
        'build_visit_consensus no longer measures its own duplication')
    assert 'CONSENSUS_DUPLICATION_WARN_FRAC' in build_src, (
        'the duplication is measured but never surfaced in the log')

    ck_src = inspect.getsource(ac.run_visit_checkpoint)
    assert 'duplication=' in ck_src, (
        'the m2 record no longer carries the consensus duplication, so a '
        'bimodal consensus is invisible after the run')
