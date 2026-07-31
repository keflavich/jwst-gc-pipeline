"""The field registry: does it say what the pipeline used to say, and does the
order of the YAML file stay irrelevant?

Two kinds of test here.

*Equivalence* — each view reproduces the dictionary it replaced. Where it
deliberately differs, the difference is written out below, so changing one of
them is a decision rather than a diff nobody reads.

*Order independence* — the whole point of the YAML file is that you can write
an entry wherever it reads best. Shuffling the file must change nothing,
especially not ``merge_jobs``, which is the SLURM array: entry *n* is task *n*,
so a reordering would run each task on a different filter.
"""
import copy
import random

import pytest
import yaml

from jwst_gc_pipeline import fields as F


# --------------------------------------------------------------------------
# What the registry says, pinned.
# --------------------------------------------------------------------------

def test_every_field_has_a_root_the_roots_block_defines():
    for field in F.FIELDS:
        assert field.root in F.ROOTS, field.name


def test_the_roots_are_the_only_absolute_paths():
    """A field's directory is built from `roots`, so redirecting the pipeline
    to another disk is one edit."""
    with open(F.REGISTRY_PATH) as fh:
        raw = fh.read()
    body = raw.split('fields:', 1)[1]
    assert '/orange/' not in body and '/blue/' not in body


def test_basepath_matches_the_root_it_names():
    assert F.basepath('brick') == '/blue/adamginsburg/adamginsburg/jwst/brick/'
    assert F.basepath('sgrc') == '/orange/adamginsburg/jwst/sgrc/'


def test_an_unregistered_target_gets_the_blue_tree():
    """What the `if target in (...)` branches this replaced did in their else."""
    assert F.basepath('nosuchfield') == '/blue/adamginsburg/adamginsburg/jwst/nosuchfield/'


def test_wd1_and_wd2_are_on_orange():
    """The catalog and merge drivers disagreed about these two: the catalog
    stage wrote to /orange and the merge read from /blue."""
    assert F.basepath('wd1').startswith('/orange/')
    assert F.basepath('wd2').startswith('/orange/')


# --------------------------------------------------------------------------
# Equivalence with what the pipeline used to hold.
# --------------------------------------------------------------------------

#: The (target, proposal) pairs whose obsid the registry supplies and the old
#: `project_obsnum` did not, plus the one it drops.  Each is a decision.
OBSNUM_CHANGES = {
    # Was absent, so a cloudc 2526 merge raised KeyError('2526').  The
    # observation is real: 2526 obs 021, MIRI F770W, target 'G0'.
    ('cloudc', '2526'): '021',
    # w51 has no 1182 data on disk and no 1182 filters, so no merge could ever
    # reach this; it was never observed.
    ('w51', '1182'): None,
}


def test_obs_filters_lists_the_same_filters_as_before():
    todays = {
        'brick': {'2221': ['f410m', 'f212n', 'f466n', 'f405n', 'f187n', 'f182m', 'f2550w'],
                  '1182': ['f444w', 'f356w', 'f200w', 'f115w']},
        'cloudc': {'2221': ['f410m', 'f212n', 'f466n', 'f405n', 'f187n', 'f182m', 'f2550w'],
                   '2526': ['f770w']},
        'sgra': {'1939': ['f115w', 'f212n', 'f405n']},
        'arches': {'2045': ['f212n', 'f323n']},
        'quintuplet': {'2045': ['f212n', 'f323n']},
        'gc2211': {'2211': ['f150w', 'f200w', 'f277w']},
        'm92': {'1334': ['f090w', 'f150w', 'f277w', 'f444w']},
    }
    view = F.obs_filters()
    for target, per_proposal in todays.items():
        for proposal, filters in per_proposal.items():
            assert sorted(view[target][proposal]) == sorted(filters), (target, proposal)


def test_project_obsnum_matches_apart_from_the_listed_changes():
    todays = {'brick': {'2221': '001', '1182': '004'},
              'cloudc': {'2221': '002'},
              'sickle': {'3958': '*'},
              'cloudef': {'2092': '*'},
              'sgrc': {'4147': '012'},
              'sgrb2': {'5365': '001'},
              'arches': {'2045': '001'},
              'quintuplet': {'2045': '003'},
              'sgra': {'1939': '001'},
              'gc2211': {'2211': '*'},
              'wd1': {'1905': '001'},
              'wd2': {'3523': '005'},
              'w51': {'6151': '001', '1182': '002'},
              'm92': {'1334': '001'},
              'ngc6397': {'1979': '001'},
              'm4': {'1979': '002'},
              'ngc6334': {'7213': '001', '6778': '001'}}
    view = F.project_obsnum()
    for target, per_proposal in todays.items():
        for proposal, obsid in per_proposal.items():
            expected = OBSNUM_CHANGES.get((target, proposal), obsid)
            assert view.get(target, {}).get(proposal) == expected, (target, proposal)


def test_nvisits_is_the_transpose_and_keeps_its_values():
    view = F.nvisits()
    assert view['2221'] == {'brick': 2, 'cloudc': 2}
    assert view['1979'] == {'ngc6397': 1, 'm4': 1}
    assert view['6778']['ngc6334'] == 3
    assert view['1905']['wd1'] == 3


def test_no_view_emits_none_where_a_glob_is_built():
    """`jw0{proposal}-o{obsid}_*` with obsid None matches nothing and says so
    to no one; a missing entry must be missing, not None."""
    for target, per_proposal in F.project_obsnum().items():
        for proposal, obsid in per_proposal.items():
            assert obsid is not None, (target, proposal)
    for target, per_proposal in F.obs_filters().items():
        for proposal, filters in per_proposal.items():
            assert all(f is not None for f in filters), (target, proposal)


def test_an_observation_with_no_nircam_data_is_absent_from_project_obsnum():
    """cloudc/2526 is MIRI only, so it has no NIRCam observation number."""
    assert '2526' not in F.project_obsnum().get('cloudc', {})


# --------------------------------------------------------------------------
# Per-instrument observation numbers.
# --------------------------------------------------------------------------

def test_proposal_2221_numbers_nircam_and_miri_opposite_to_each_other():
    """Both are right; the products on disk agree with each.  One number per
    (target, proposal) could only ever be right for one of them."""
    assert F.glob_obsid('brick', '2221', 'nircam') == '001'
    assert F.glob_obsid('brick', '2221', 'miri') == '002'
    assert F.glob_obsid('cloudc', '2221', 'nircam') == '002'
    assert F.glob_obsid('cloudc', '2221', 'miri') == '001'


def test_field_to_reg_mapping_is_per_instrument():
    assert F.field_to_reg_mapping('2221', 'nircam') == {'001': 'brick', '002': 'cloudc'}
    assert F.field_to_reg_mapping('2221', 'miri') == {'002': 'brick', '001': 'cloudc'}


def test_a_joint_observation_resolves_to_its_one_field():
    """Sgr B2's MIRI observations 002 and 998 are cataloged together."""
    assert F.target_for_obsid('5365', '002-998', 'miri') == 'sgrb2'


def test_an_unregistered_observation_says_so():
    with pytest.raises(KeyError, match='not in fields.yaml'):
        F.target_for_obsid('9999', '001')


def test_two_fields_cannot_claim_one_observation(monkeypatch):
    clash = F.Field('clash', root='orange', observations=(
        F.Obs(proposal='2221', obsids={'nircam': ('001',)}),))
    monkeypatch.setattr(F, 'FIELDS', F.FIELDS + (clash,))
    with pytest.raises(F.FieldRegistryError, match='claimed by both'):
        F.field_to_reg_mapping('2221', 'nircam')


# --------------------------------------------------------------------------
# Order independence.
# --------------------------------------------------------------------------

#: The merge job list for the Brick, written out.  Comparing against a view
#: would make this test `view == view`; the point is to pin the array indices
#: to something a person wrote down.
BRICK_MERGE_JOBS = [
    ('1182', 'f115w'), ('1182', 'f200w'), ('1182', 'f356w'), ('1182', 'f444w'),
    ('2221', 'f182m'), ('2221', 'f187n'), ('2221', 'f212n'), ('2221', 'f405n'),
    ('2221', 'f410m'), ('2221', 'f466n'), ('2221', 'f2550w'),
]


def test_the_merge_job_order_is_pinned():
    assert F.merge_jobs('brick') == BRICK_MERGE_JOBS


def test_an_unregistered_target_has_no_merge_jobs_and_says_so():
    """An empty job list would make a typo'd target a silent no-op."""
    with pytest.raises(KeyError, match='nothing to merge'):
        F.merge_jobs('nosuchfield')


def test_filters_come_back_in_wavelength_order():
    """f2550w after f466n, which alphabetical order would get wrong."""
    brick = F.obs_filters()['brick']['2221']
    assert brick == ['f182m', 'f187n', 'f212n', 'f405n', 'f410m', 'f466n', 'f2550w']


def test_proposals_come_back_numerically():
    assert list(F.obs_filters()['brick']) == ['1182', '2221']


def _reload_from(tmp_path, raw):
    path = tmp_path / 'fields.yaml'
    with open(path, 'w') as fh:
        yaml.safe_dump(raw, fh)
    return F._load(str(path))


def test_shuffling_the_file_changes_nothing(tmp_path):
    """The property the YAML file is for: write entries wherever they read
    best."""
    with open(F.REGISTRY_PATH) as fh:
        raw = yaml.safe_load(fh)

    shuffled = copy.deepcopy(raw)
    rng = random.Random(20260731)
    names = list(shuffled['fields'])
    rng.shuffle(names)
    shuffled['fields'] = {n: shuffled['fields'][n] for n in names}
    for spec in shuffled['fields'].values():
        proposals = list(spec.get('observations') or {})
        rng.shuffle(proposals)
        spec['observations'] = {p: spec['observations'][p] for p in proposals}
        for obs in spec['observations'].values():
            for key in ('filters', 'niriss_filters'):
                if obs.get(key):
                    rng.shuffle(obs[key])

    straight_dir = tmp_path / 'straight'
    tumbled_dir = tmp_path / 'tumbled'
    straight_dir.mkdir()
    tumbled_dir.mkdir()
    _, straight = _reload_from(straight_dir, raw)
    _, tumbled = _reload_from(tumbled_dir, shuffled)
    assert straight == tumbled


def test_an_unknown_instrument_is_rejected(tmp_path):
    raw = {'roots': {'blue': '/b'},
           'fields': {'x': {'root': 'blue',
                            'observations': {'1': {'obsids': {'nirspec': ['001']}}}}}}
    with pytest.raises(F.FieldRegistryError, match='unknown instrument'):
        _reload_from(tmp_path, raw)


def test_a_root_the_roots_block_does_not_define_is_rejected(tmp_path):
    raw = {'roots': {'blue': '/b'}, 'fields': {'x': {'root': 'green'}}}
    with pytest.raises(F.FieldRegistryError, match='the roots block defines'):
        _reload_from(tmp_path, raw)


# --------------------------------------------------------------------------
# The registry is the only copy.
# --------------------------------------------------------------------------

@pytest.mark.parametrize('module,name', [
    ('jwst_gc_pipeline/photometry/merge_catalogs.py', 'obs_filters = {'),
    ('jwst_gc_pipeline/photometry/merge_catalogs.py', 'project_obsnum = {'),
    ('jwst_gc_pipeline/photometry/crowdsource_catalogs_long.py', 'nvisits = {'),
    ('jwst_gc_pipeline/photometry/crowdsource_catalogs_long.py', 'field_to_reg_mapping = {'),
    ('jwst_gc_pipeline/reduction/PipelineRerunNIRCAM-LONG.py', 'refnames = {'),
    ('jwst_gc_pipeline/reduction/PipelineRerunNIRCAM-LONG.py', 'fov_regname = {'),
    ('jwst_gc_pipeline/reduction/PipelineRerunNIRCAM-LONG.py', 'field_to_reg_mapping = {'),
    ('jwst_gc_pipeline/reduction/PipelineMIRI.py', 'fov_regname = {'),
    ('jwst_gc_pipeline/reduction/PipelineMIRI.py', 'field_to_reg_mapping = {'),
])
def test_no_module_keeps_its_own_copy(module, name):
    """A second copy is how these drifted apart in the first place."""
    import os
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(os.path.join(root, module)) as fh:
        assert name not in fh.read(), f'{module} still defines {name}'
