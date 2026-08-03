"""The JWST-internal reference: per-filter consensus and the reference filter."""
import os

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from jwst_gc_pipeline.photometry.consensus_catalog import (
    NoReferenceFilterError, consensus_path, pool_visit_consensi,
    promote_reference_filter, reference_consensus_path, reference_filter,
    reference_filter_rank, tie_to_reference_consensus, write_filter_consensus)


# --------------------------------------------------------------------------
# Which filter anchors the field.
# --------------------------------------------------------------------------

def test_the_intended_order_is_reproduced():
    """F212N > F210M > F187N > F182M > F200W > F150W.

    Note F210M beats F187N while F187N beats F200W: neither wavelength nor
    bandwidth alone produces that, so the two trade off.
    """
    intended = ['F212N', 'F210M', 'F187N', 'F182M', 'F200W', 'F150W']
    assert sorted(intended, key=reference_filter_rank) == intended


def test_bandwidth_alone_does_not_explain_the_order():
    """A rule that sorted on bandwidth first would put F187N above F210M."""
    assert reference_filter_rank('F210M') < reference_filter_rank('F187N')


def test_wavelength_alone_does_not_explain_the_order():
    """F200W is closer to Ks than F187N, and still ranks below it: F200W
    saturates the bright stars VIRAC2 measures."""
    assert reference_filter_rank('F187N') < reference_filter_rank('F200W')


@pytest.mark.parametrize('field,expected', [
    (['F182M', 'F187N', 'F212N', 'F405N', 'F410M', 'F466N'], 'F212N'),  # brick
    (['F115W', 'F200W', 'F356W', 'F444W'], 'F200W'),                    # w51
    (['F150W', 'F444W'], 'F150W'),
])
def test_the_field_picks_its_closest_match_to_virac2(field, expected):
    assert reference_filter(field) == expected


def test_the_second_intended_order_is_reproduced():
    """F277W > F140M > F115W: a long-wavelength filter can outrank a blue one.

    This is what forces LOG wavelength.  Linearly, F277W is 0.62 um from Ks and
    F140M is 0.75 -- so close that no positive long-wavelength penalty can put
    F277W first; in log, ln(2.77/2.15) < |ln(1.40/2.15)| with room to spare.
    """
    intended = ['F277W', 'F140M', 'F115W']
    assert sorted(intended, key=reference_filter_rank) == intended


def test_the_far_infrared_ranks_last():
    """No channel term is needed to get this -- log distance does it."""
    assert reference_filter_rank('F2550W') > reference_filter_rank('F1130W')
    assert reference_filter_rank('F1130W') > reference_filter_rank('F444W')


def test_a_field_with_no_filters_says_so():
    with pytest.raises(NoReferenceFilterError):
        reference_filter([])


# --------------------------------------------------------------------------
# Pooling the per-visit consensi into one per-filter catalog.
# --------------------------------------------------------------------------

def _consensus(ra, dec, mag=None):
    coords = SkyCoord(np.asarray(ra) * u.deg, np.asarray(dec) * u.deg)
    return dict(coords=coords,
                mag=np.asarray(mag if mag is not None else [20.0] * len(ra)))


def test_pooling_keeps_every_star_and_counts_the_visits():
    per_visit = {
        '001': _consensus([266.5, 266.6], [-28.7, -28.71]),
        '002': _consensus([266.8], [-28.9]),
    }
    pooled = pool_visit_consensi(per_visit)
    assert len(pooled) == 3
    assert set(pooled['n_visits']) == {1}
    assert pooled.meta['NVISITS'] == 2


def test_a_star_seen_in_two_visits_is_averaged_once():
    """Overlapping visits must not double a star in the reference catalog."""
    per_visit = {
        '001': _consensus([266.5], [-28.7]),
        '002': _consensus([266.5 + 1e-5], [-28.7]),   # ~0.03", same star
    }
    pooled = pool_visit_consensi(per_visit)
    assert len(pooled) == 1
    assert pooled['n_visits'][0] == 2
    assert 266.5 < pooled['RA'][0] < 266.5 + 1e-5


def test_two_stars_of_ONE_visit_are_never_merged():
    """Association is across visits.  Two genuinely close stars seen in the
    same exposure set are two stars, however near."""
    per_visit = {'001': _consensus([266.5, 266.5 + 1e-5], [-28.7, -28.7])}
    pooled = pool_visit_consensi(per_visit)
    assert len(pooled) == 2


def test_pooling_nothing_says_so():
    with pytest.raises(ValueError, match='no visit consensus'):
        pool_visit_consensi({'001': None})


# --------------------------------------------------------------------------
# On disk.
# --------------------------------------------------------------------------

def test_the_written_catalog_says_what_it_is(tmp_path):
    table = pool_visit_consensi({'001': _consensus([266.5], [-28.7])})
    path = write_filter_consensus(str(tmp_path), 'F212N', table)
    assert path == consensus_path(str(tmp_path), 'F212N')
    back = Table.read(path)
    assert back.meta['FILTER'] == 'F212N'
    assert back.meta['CONSTYPE'] == 'per-filter JWST consensus'
    assert 'skycoord' in back.colnames        # what load_reference_catalog reads


def test_promoting_the_reference_filter_names_the_filter_it_chose(tmp_path):
    table = pool_visit_consensi({'001': _consensus([266.5], [-28.7])})
    write_filter_consensus(str(tmp_path), 'F212N', table)
    chosen, path = promote_reference_filter(str(tmp_path),
                                            ['F405N', 'F212N', 'F182M'])
    assert chosen == 'F212N'
    assert path == reference_consensus_path(str(tmp_path))
    back = Table.read(path)
    assert back.meta['REFFILT'] == 'F212N'
    assert back.meta['CONSTYPE'] == 'JWST reference-filter consensus'


def test_promoting_without_the_reference_filters_catalog_refuses(tmp_path):
    """Tying every filter to a silently-absent reference is the failure this
    ladder exists to prevent."""
    table = pool_visit_consensi({'001': _consensus([266.5], [-28.7])})
    write_filter_consensus(str(tmp_path), 'F405N', table)     # not the chosen one
    with pytest.raises(FileNotFoundError, match='F212N'):
        promote_reference_filter(str(tmp_path), ['F405N', 'F212N'])


# --------------------------------------------------------------------------
# Tying a filter to the reference.
# --------------------------------------------------------------------------

def test_a_filter_offset_from_the_reference_is_measured(tmp_path):
    rng = np.random.default_rng(42)
    ra = 266.5 + rng.uniform(-0.01, 0.01, 400)
    dec = -28.7 + rng.uniform(-0.01, 0.01, 400)
    reference = SkyCoord(ra * u.deg, dec * u.deg)
    shift_deg = 20.0 / 3600.0 / 1000.0            # 20 mas in Dec
    moved = SkyCoord(ra * u.deg, (dec + shift_deg) * u.deg)
    result = tie_to_reference_consensus(moved, reference, context='test')
    assert np.isclose(result['off_mas'], 20.0, atol=5.0)
