"""m4's two observations must not share per-frame catalog names.

m4 is proposal 1979 observations 002 and 003 -- two pointings 174" apart in one
tree that reuse the SAME (visit001, vgroup02101, exp00001-00003) in both
F150W2 and F322W2.  `perframe_obs_token` returned '' for both, so their
per-frame catalog names were equal and the second writer silently overwrote the
first.  Measured on disk 2026-09-03:

    filter   _cal o002   _cal o003   total   per-frame m2 catalogs   deficit
    F150W2       96          24       120             96               24
    F322W2       24           6        30             24                6

The deficit equals the o003 count exactly, in both filters.
"""
import pytest

from jwst_gc_pipeline.photometry.naming import (
    MULTIOBS_PROPOSALS, PER_OBS_PERFRAME_FIELDS, SHARED_TREE_PROPOSALS,
    perframe_obs_token)


@pytest.mark.parametrize('field,expected', [('002', '_o002'), ('003', '_o003')])
def test_m4s_two_observations_get_distinct_tokens(field, expected):
    assert perframe_obs_token('1979', field) == expected


def test_the_two_tokens_actually_differ():
    """The whole point: one name per observation, not one shared name."""
    assert perframe_obs_token('1979', '002') != perframe_obs_token('1979', '003')


def test_ngc6397_is_untouched():
    """ngc6397 is 1979 observation 001, in its own tree with no collision.

    This is why the exception is keyed per (proposal, observation) rather than
    by adding 1979 to MULTIOBS_PROPOSALS -- tokening ngc6397 would orphan every
    per-frame catalog it already has, and it was mid-chain when this landed.
    """
    assert perframe_obs_token('1979', '001') == ''
    assert '1979' not in MULTIOBS_PROPOSALS
    assert '1979' not in SHARED_TREE_PROPOSALS


@pytest.mark.parametrize('proposal,field,expected', [
    ('1334', '001', ''),          # m92
    ('2221', '001', ''),          # brick, 6-band half
    ('1182', '004', ''),          # brick, 4-band half
    ('4147', '012', ''),          # sgrc
    ('6778', '001', '_j6778'),    # ngc6334 shared tree
    ('7213', '001', '_j7213'),
    ('9438', '006', '_o006'),     # crowded_l3, a real MULTIOBS proposal
    ('2211', '028', '_o028'),
])
def test_no_other_field_moves(proposal, field, expected):
    assert perframe_obs_token(proposal, field) == expected


def test_the_exception_list_names_only_m4():
    assert PER_OBS_PERFRAME_FIELDS == (('1979', '002'), ('1979', '003'))


def test_the_token_is_zero_padded():
    """`--field 2` must not glob `_o2` while the writer stamps `_o002`."""
    assert perframe_obs_token('1979', '2') == '_o002'
