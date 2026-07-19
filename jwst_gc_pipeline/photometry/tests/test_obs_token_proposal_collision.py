"""Regression: ngc6334 shares one target dir between TWO proposals (7213 + 6778)
that also share filters (F200W, F470N) with the SAME obs number (001) and the SAME
(visit, vgroup, exp) tuples.  Without a per-proposal token the per-frame catalog
tables collide and the second cataloging run overwrites the first (6778 clobbered
7213's F200W/F470N catalogs, 2026-07-09).  ``obs_token`` must return a distinct
``_j{proposal}`` token for those proposals while leaving every other target's
tokens (and the existing prop-2211 multi-obs case) unchanged.
"""
from jwst_gc_pipeline.photometry.catalog_long import obs_token


def test_ngc6334_proposals_get_distinct_proposal_token():
    # the two proposals that share the ngc6334 target -> distinct tokens
    assert obs_token('7213', '001') == '_j7213'
    assert obs_token('6778', '001') == '_j6778'
    assert obs_token('7213', '001') != obs_token('6778', '001')
    # token does not depend on field (both use obs 001 -> field can't disambiguate)
    assert obs_token('7213', None) == '_j7213'
    assert obs_token('6778', '') == '_j6778'


def test_prop2211_multiobs_unchanged():
    # existing multi-obs (same-proposal) disambiguation is untouched
    assert obs_token('2211', '023') == '_o023'
    assert obs_token('2211', '046') == '_o046'
    assert obs_token('2211', '') == ''   # no field -> no token


def test_other_targets_get_no_token():
    # single-proposal-per-basepath targets keep the empty token (unchanged names)
    for prop in ('2221', '1182', '4147', '1334', '1979', '3523', '1905'):
        assert obs_token(prop, '001') == ''
        assert obs_token(prop, None) == ''
