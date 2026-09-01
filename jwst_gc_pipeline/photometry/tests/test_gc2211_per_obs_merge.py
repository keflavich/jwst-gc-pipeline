"""gc2211's five observations merge per-observation, not pooled.

They are five DIFFERENT targets -- different parts of the sky, observed at
different times -- that happen to share a parent observation id and therefore a
directory.  Pooling them into one merged catalog mixes unrelated sky.

The mismatch this fixes: the per-observation split (#469) renamed the targets to
``gc2211_o023`` .. ``gc2211_o050``, while ``merge_individual_frames`` still
branched on the literal target name ``'gc2211'``.  Under the new names that test
is False, so the merge fell through to the untokened ``else`` and globbed
``f200w_nrca1_visit001_...`` -- which matches none of the ``_o023``-stamped
per-frame tables the writer produces.  Every gc2211 m12 finalize died::

    ValueError: No tables found matching
    /orange/adamginsburg/jwst/gc2211_o023//F200W/f200w_nrca...._dao_basic.fits

after its 8 fan-out shards had written 192 per-frame tables (2026-08-22).

Keying off the running PROPOSAL rather than a target spelling is what keeps this
working when a field is renamed or split again.
"""
import re

import pytest

from jwst_gc_pipeline.photometry.naming import (
    PER_OBS_MERGED_PROPOSALS, merge_field_for_proposal,
    merged_catalog_obs_token)
from jwst_gc_pipeline.photometry.naming import ObservationFieldError


GC2211_OBS = ('023', '028', '046', '049', '050')


def test_2211_is_registered_as_per_obs_merged():
    assert '2211' in PER_OBS_MERGED_PROPOSALS


@pytest.mark.parametrize('obs', GC2211_OBS)
def test_each_observation_gets_its_own_token(obs):
    assert merged_catalog_obs_token('2211', obs) == f'_o{obs}'
    assert merge_field_for_proposal('2211', obs) == obs


def test_the_five_tokens_are_all_different():
    """The property that matters: no two of the five can name the same merged
    catalog, which is what pooling amounted to."""
    toks = {merged_catalog_obs_token('2211', o) for o in GC2211_OBS}
    assert len(toks) == len(GC2211_OBS)


def test_an_unscoped_2211_merge_refuses():
    """Pooling is now unreachable rather than merely unused: a 2211 merge with
    no field raises at the point the observation is decided, instead of quietly
    globbing `_o*` across all five pointings."""
    with pytest.raises(ObservationFieldError):
        merge_field_for_proposal('2211', None)
    with pytest.raises(ObservationFieldError):
        merge_field_for_proposal('2211', '')


def test_other_proposals_are_untouched():
    # 1182/2221 (brick) left this list in #590: they image one field with
    # disjoint filter sets and were overwriting a single untokened merged
    # catalog, so they are per-obs-merged now and asserted below instead.
    for pid, field in (('4147', '012'), ('5365', '001'), ('3958', '007')):
        assert merged_catalog_obs_token(pid, field) == ''
        assert merge_field_for_proposal(pid, field) is None


def test_bricks_two_proposals_are_per_obs_merged():
    for pid, field, tok in (('1182', '004', '_o004'), ('2221', '001', '_o001')):
        assert merged_catalog_obs_token(pid, field) == tok
        assert merge_field_for_proposal(pid, field) == field


def test_10678_still_per_obs():
    assert merged_catalog_obs_token('10678', '001') == '_o001'


# --------------------------------------------------------------------------
# the glob itself
# --------------------------------------------------------------------------

def test_the_merge_glob_carries_the_obs_token(tmp_path):
    """Drives the real glob construction against a real tree.

    A per-frame table is written under the name the WRITER produces
    (``f200w_nrca1_o023_visit001_vgroup02201_exp00001_m2_daophot_basic.fits``),
    and the merge must find it.  Asserting on the token alone would pass even
    if the glob still ignored it.
    """
    import glob as _glob

    d = tmp_path / 'F200W'
    d.mkdir(parents=True)
    for obs, vg in (('023', '02201'), ('046', '04201')):
        (d / f'f200w_nrca1_o{obs}_visit001_vgroup{vg}_exp00001'
             f'_m2_daophot_basic.fits').write_bytes(b'')

    tok = merged_catalog_obs_token('2211', '023')
    pat = (f"{d}/f200w_nrca1{tok}_visit001_vgroup*_exp00001"
           f"_m2_daophot_basic.fits")
    hits = _glob.glob(pat)
    assert len(hits) == 1, f'{pat} matched {hits}'
    assert '_o023_' in hits[0]
    assert '_o046_' not in hits[0], 'the glob reached another observation'


def test_the_untokened_glob_would_have_found_nothing(tmp_path):
    """The failure as it happened, so the fix is not mistaken for a no-op."""
    import glob as _glob

    d = tmp_path / 'F200W'
    d.mkdir(parents=True)
    (d / 'f200w_nrca1_o023_visit001_vgroup02201_exp00001'
         '_m2_daophot_basic.fits').write_bytes(b'')

    untokened = (f"{d}/f200w_nrca1_visit001_vgroup*_exp00001"
                 f"_m2_daophot_basic.fits")
    assert _glob.glob(untokened) == []


# --------------------------------------------------------------------------
# the literals that broke
# --------------------------------------------------------------------------

def test_merge_no_longer_branches_on_the_literal_target_name():
    """A target-name literal silently stops matching when a field is renamed,
    and falls through to a branch that finds nothing.  The proposal decides."""
    import inspect

    from jwst_gc_pipeline.photometry import merge_catalogs

    for fn in (merge_catalogs.merge_individual_frames,
               merge_catalogs.merge_daophot):
        src = inspect.getsource(fn)
        code = '\n'.join(ln for ln in src.split('\n')
                         if not ln.lstrip().startswith('#'))
        assert not re.search(r"target\s*==\s*['\"]gc2211['\"]", code), (
            f'{fn.__name__} still branches on the literal target name; the '
            'per-observation split renames it to gc2211_o023..')


def test_the_pooling_branch_is_gone():
    """`glob_obs_ = '_o*'` pooled all five pointings into one untokened
    catalog."""
    import inspect

    from jwst_gc_pipeline.photometry import merge_catalogs

    src = inspect.getsource(merge_catalogs.merge_individual_frames)
    code = '\n'.join(ln for ln in src.split('\n')
                     if not ln.lstrip().startswith('#'))
    assert "'_o*'" not in code, 'the all-obs pooling glob is still reachable'
