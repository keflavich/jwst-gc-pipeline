"""sickle's NIRCam observation is 007, and only 007.

The registry listed `nircam: ['001', '002', '007']`.  001 and 002 are the MIRI
observations, so every consumer that asks "which NIRCam observations does this
field have" got three, expected the five NIRCam filters from each, and found
none for two of them.  The campaign monitor rendered exactly that: three
observation cards, five filters each, ten of the fifteen chips reading missing
on a field whose NIRCam half has 528 m7 per-frame catalogs and three filters
merged through m7.

Counted from disk on 2026-08-19:

    F187N  96 _cal   F210M  96      all obs 007
    F335M  24        F470N  24      F480M 24
    F770W   5        F1130W  5      F1500W 5   under obs 001, 002 AND 003

`run_sickle_3958_o007.sh` says the same in prose: "NIRCam is obs 007.  The 3958
MIRI data are obs 001 and are NOT covered here."

MIRI 003 was missing from the registry for the same reason the NIRCam list was
wrong -- nobody had compared the entry with the tree.
"""
import pytest

from jwst_gc_pipeline import fields


def _sickle():
    for fld in fields.FIELDS:
        if fld.name == 'sickle':
            for obs in fld.observations:
                if obs.proposal == '3958':
                    return obs
    raise AssertionError('sickle/3958 is not registered')


def test_nircam_is_observation_007_only():
    assert _sickle().obsids.get('nircam') == ('007',)


@pytest.mark.parametrize('bad', ['001', '002'])
def test_the_miri_observations_are_not_listed_as_nircam(bad):
    """The specific regression: listing a MIRI observation under `nircam` makes
    every reader expect NIRCam filters that cannot exist."""
    assert bad not in _sickle().obsids.get('nircam', ())


def test_miri_carries_all_three_of_its_observations():
    assert _sickle().obsids.get('miri') == ('001', '002', '003')


def test_the_joint_miri_token_still_names_only_the_joint_run():
    """001 and 002 are cataloged together and 003 is not, so widening the obsid
    list must not widen the joint token -- that would name a catalog run that
    never happened."""
    assert _sickle().joint_obsids.get('miri') == ('001-002',)


def test_every_nircam_filter_of_sickle_belongs_to_the_one_observation():
    """The filter list is per-proposal and mixes both instruments; what this
    pins is that the NIRCam obsid list is a single observation, so a consumer
    pairing filters with observations cannot produce a NIRCam/MIRI cross
    product."""
    obs = _sickle()
    assert len(obs.obsids.get('nircam', ())) == 1
    for filt in ('f187n', 'f210m', 'f335m', 'f470n', 'f480m'):
        assert filt in obs.filters
