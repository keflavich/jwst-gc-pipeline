"""The merge's per-frame input glob spells the token the per-frame writer
stamps -- one function, not two derivations (issue #316, item 1).

A glob that disagrees with the writer does not raise.  It matches nothing,
``merge_individual_frames`` raises ``No tables found``, and the CLI's handler
reports "no per-frame catalogs for this filter; skipping".  So the property
has to be tested directly: for every (proposal, field) the pipeline runs, the
glob token and the writer token are the same string.
"""

import inspect
import os

import pytest

from jwst_gc_pipeline.photometry import naming
from jwst_gc_pipeline.photometry import merge_catalogs as MC


#: (proposal, field) pairs covering every branch: single-obs with and without a
#: field, the two per-obs-merged proposals padded and unpadded, and the shared
#: ngc6334 tree.
CASES = [
    ('6151', '001'),    # w51            -- single observation per basepath
    ('6151', None),
    ('4147', '012'),    # sgrc
    ('2092', '005'),    # cloudef control field
    ('1182', '001'),    # brick
    ('2211', '023'),    # gc2211         -- multi-obs, padded
    ('2211', '23'),     # gc2211         -- multi-obs, UNPADDED
    ('10678', '1'),     # gc-treasury    -- 139 tiles, unpadded
    ('10678', '139'),
    ('7213', '001'),    # ngc6334        -- two proposals, one tree
    ('6778', '001'),
]


def _writer_token(proposal, field):
    """What the per-frame catalog writer actually stamps."""
    from jwst_gc_pipeline.photometry.crowdsource_catalogs_long import obs_token
    return obs_token(proposal, field)


@pytest.mark.parametrize('proposal,field', CASES)
def test_the_glob_token_is_the_writer_token(proposal, field):
    assert naming.perframe_obs_token(proposal, field) == _writer_token(proposal, field)


def test_a_single_observation_proposal_is_never_tokened_by_a_field():
    """w51/6151 with --field 001: the writer stamps nothing, so globbing
    `_o001` matched none of its per-frame catalogs and the merge reported the
    filter as having none."""
    assert naming.perframe_obs_token('6151', '001') == ''
    assert naming.merged_catalog_module_token('6151', '001') == ''


def test_an_unpadded_field_still_globs_the_padded_name():
    """`--field 23` writes `_o023`; a raw f-string globbed `_o23`."""
    assert naming.perframe_obs_token('2211', '23') == '_o023'
    assert naming.merged_catalog_module_token('2211', '23') == '_o023'


def test_the_shared_ngc6334_tree_is_tagged_by_proposal():
    assert naming.perframe_obs_token('7213', '001') == '_j7213'
    assert naming.merged_catalog_module_token('7213', '001') == '_j7213'


@pytest.mark.parametrize('proposal,field', CASES)
def test_the_merged_name_reader_and_the_merge_output_agree(proposal, field):
    """``cataloging.merged_catalog_path`` reads back what the merge writes."""
    from jwst_gc_pipeline.photometry.cataloging import merged_catalog_path
    tok = naming.merged_catalog_module_token(proposal, field)
    path = merged_catalog_path('/bp', 'm5', 'nrca', 'F182M', proposal, field)
    assert f'/catalogs/f182m_nrca{tok}_indivexp_merged' in path


@pytest.mark.parametrize('proposal,field,target', [
    ('6151', '001', 'w51'),
    ('2211', '23', 'gc2211_o023'),
    ('7213', '001', 'ngc6334'),
])
def test_the_pattern_merge_actually_globs_carries_the_writer_token(
        proposal, field, target, monkeypatch, tmp_path):
    """Behavioural: capture the glob patterns ``merge_individual_frames``
    builds and check the token in them is the one the writer stamps.

    The function raises ``No tables found`` once the globs come back empty,
    which is exactly the point -- that is what a disagreeing token produces,
    and it is what the CLI swallows."""
    seen = []

    def _spy(pattern, *a, **kw):
        seen.append(pattern)
        return []

    monkeypatch.setattr(MC.glob, 'glob', _spy)
    with pytest.raises(ValueError, match='No tables found'):
        MC.merge_individual_frames(
            module='nrca', filtername='f182m', progid=proposal, target=target,
            method='dao', suffix='_basic', field=field,
            basepath=str(tmp_path), max_visitid=1, exposure_numbers=[1])

    want = naming.perframe_obs_token(proposal, field)
    marker = '_visit'
    heads = [os.path.basename(p).split(marker)[0] for p in seen
             if marker in p and 'f182m_nrc' in p]
    assert heads, f'no per-frame glob was built (patterns: {seen[:3]})'
    if want:
        assert all(h.endswith(want) for h in heads), heads[:3]
    else:
        # nothing may sit between the detector and the visit
        slots = [h.split('f182m_', 1)[1] for h in heads]
        assert not any('_o' in s or '_j' in s for s in slots), slots[:3]


def test_merge_individual_frames_derives_both_tokens_from_naming():
    """Reverting to a hand-rolled ``f'_o{field}'`` here fails, even though the
    token functions above still agree with themselves."""
    code = '\n'.join(line for line in
                     inspect.getsource(MC.merge_individual_frames).splitlines()
                     if not line.lstrip().startswith('#'))
    assert 'perframe_obs_token(progid, field)' in code
    assert 'merged_catalog_module_token(progid, field)' in code
    # the hand-rolled spelling, in code rather than in the comment that records it
    assert "f'_o{field}'" not in code
