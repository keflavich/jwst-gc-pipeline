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

#: How the NIRCam `project_obsnum` view differs from the old dictionary.  Each
#: entry is a decision, not a diff nobody read.  cloudc/2526 is absent from this
#: list on purpose: it is MIRI-only, so it has no NIRCam number -- the merge
#: reaches it through `glob_obsid(..., 'miri')` instead.
OBSNUM_CHANGES = {
    # w51 has no 1182 data on disk and no 1182 filters, so no merge could ever
    # reach this; it was never observed.
    ('w51', '1182'): None,
    # omegacen is in the reduce driver's map and had no merge entry at all.
    ('omegacen', '8322'): '001',
    ('omegacen', '12587'): '001',
    # gc-treasury post-dates the old dictionary; its obsids are the wildcard,
    # so the glob token is '*'.
    ('gc-treasury', '10678'): '*',
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

    # And nothing appeared that neither the old dictionary nor the list above
    # accounts for -- checking only the listed pairs would let a spurious extra
    # entry through.
    extra = {(t, p) for t, d in view.items() for p in d
             if p not in todays.get(t, {})} - set(OBSNUM_CHANGES)
    assert not extra, f'undeclared project_obsnum entries: {sorted(extra)}' 


def test_nvisits_is_the_transpose_and_keeps_its_values():
    view = F.nvisits()
    assert view['2221'] == {'brick': 2, 'cloudc': 2}
    assert view['1979'] == {'ngc6397': 1, 'm4': 1}
    assert view['6778']['ngc6334'] == 3
    assert view['1905']['wd1'] == 3


def test_no_view_emits_none_where_a_glob_is_built():
    """`jw{proposal:05d}-o{obsid}_*` with obsid None matches nothing and says so
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


# --------------------------------------------------------------------------
# Findings from the adversarial review of this change.
# --------------------------------------------------------------------------

def test_the_registry_import_is_not_shadowed_by_a_local_variable():
    """Every driver has a local `fields` (the --field list).  Importing the
    module under that name made `fields.field_to_reg_mapping` an
    AttributeError on a list, at startup, in all three reduce drivers."""
    import ast
    import os
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for module in ('jwst_gc_pipeline/reduction/PipelineRerunNIRCAM-LONG.py',
                   'jwst_gc_pipeline/reduction/PipelineMIRI.py',
                   'jwst_gc_pipeline/reduction/PipelineRerunNIRISS.py',
                   'jwst_gc_pipeline/photometry/crowdsource_catalogs_long.py',
                   'jwst_gc_pipeline/photometry/merge_catalogs.py'):
        tree = ast.parse(open(os.path.join(root, module)).read())
        imported = {alias.asname or alias.name
                    for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom) and node.module == 'jwst_gc_pipeline'
                    for alias in node.names if alias.name == 'fields'}
        assigned = {t.id for node in ast.walk(tree)
                    if isinstance(node, ast.Assign)
                    for t in node.targets if isinstance(t, ast.Name)}
        assert not (imported & assigned), (
            f'{module} imports the registry as {sorted(imported & assigned)} '
            f'and also assigns that name')


def test_cloudef_keeps_its_miri_observations():
    """Present in the catalog driver's copy of the map, absent from the reduce
    driver's -- which is the copy this registry was generated from."""
    assert F.field_to_reg_mapping('2092', 'miri') == {
        '004': 'cloudef', '006': 'cloudef', '008': 'cloudef'}


@pytest.mark.parametrize('proposal,token,target', [
    ('5365', '002-998', 'sgrb2'),
    ('3958', '001-002', 'sickle'),
])
def test_joint_observations_survive(proposal, token, target):
    """Both halves tile one field; cataloging either alone builds half a
    mosaic."""
    assert F.field_to_reg_mapping(proposal, 'miri')[token] == target
    assert F.default_field_token(target, proposal, 'miri') == token


def test_the_default_field_prefers_the_joint_token():
    """It used to come from whichever key an inverted dict happened to keep."""
    assert F.default_field_token('sgrb2', '5365', 'miri') == '002-998'
    assert F.default_field_token('sickle', '3958', 'nircam') == '001'


#: Every (proposal, obsid) that PipelineRerunNIRCAM-LONG.py listed before the
#: reference catalogs moved into the registry, with the file it named.  Written
#: out so the move is checkable rather than asserted.
NIRCAM_REFERENCE_CATALOGS = {
    ('2221', '001'): 'gaia_virac2_refcat_epoch2022.70.fits',
    ('1182', '004'): 'gaia_virac2_refcat_epoch2022.70.fits',
    ('3958', '007'): 'gaia_virac2_refcat_epoch2024.64.fits',
    ('5365', '001'): 'gaia_virac2_refcat_epoch2024.68.fits',
    ('6151', '001'): 'gaia_refcat.fits',
    ('2092', '002'): 'gaia_virac2_refcat_epoch2023.21.fits',
    ('2092', '005'): 'gaia_virac2_refcat_epoch2023.21.fits',
    ('4147', '012'): 'gaia_virac2_refcat_epoch2023.72.fits',
    ('2045', '001'): 'gaia_virac2_refcat_epoch2023.64.fits',
    ('2045', '003'): 'gaia_virac2_refcat_epoch2024.62.fits',
    ('1939', '001'): 'gaia_virac2_refcat_epoch2022.72.fits',
    ('2211', '023'): 'gaia_virac2_refcat_epoch2023.71.fits',
    ('2211', '028'): 'gaia_virac2_refcat_epoch2023.71_o028.fits',
    ('1905', '001'): 'gaia_refcat.fits',
    ('3523', '005'): 'gaia_refcat.fits',
    ('1334', '001'): 'gaia_refcat.fits',
    ('1979', '001'): 'gaia_refcat.fits',
    ('7213', '001'): 'gaia_virac2_refcat_epoch2026.30.fits',
    ('6778', '001'): 'gaia_virac2_refcat_epoch2024.68.fits',
}


@pytest.mark.parametrize('key,filename', sorted(NIRCAM_REFERENCE_CATALOGS.items()))
def test_the_registry_names_the_catalog_the_driver_used_to(key, filename):
    proposal, obsid = key
    assert F.reference_catalog_path(proposal, obsid).endswith(filename)


def test_the_reference_catalog_is_per_instrument():
    """Proposal 2221 observation 001 is brick under NIRCam and cloudc under
    MIRI; each ties to its own catalog."""
    nircam = F.reference_catalog_path('2221', '001', instrument='nircam')
    miri = F.reference_catalog_path('2221', '001', instrument='miri')
    assert '/brick/' in nircam and '/cloudc/' in miri


def test_miri_registers_candidates_and_nircam_registers_one():
    """MIRI takes the first candidate present on disk; NIRCam has exactly one."""
    assert len(F.reference_catalog_candidates('3958', '001', instrument='miri')) == 2
    assert len(F.reference_catalog_candidates('3958', '007')) == 1


def test_the_per_filter_override_survives():
    """1182 F115W was anchored directly to the Gaia-tied seed."""
    assert F.reference_catalog_path(
        '1182', '004', filtername='F115W').endswith(
            'gaia_virac2_refcat_epoch2022.70.fits')


def test_an_observation_with_no_catalog_names_the_block_to_add(monkeypatch):
    """A new field hits this before anything else, so the error has to say
    which file to edit and what to put in it."""
    lonely = F.Field('lonely', root='orange', observations=(
        F.Obs(proposal='4242', obsids={'nircam': ('001',)}),))
    monkeypatch.setattr(F, 'FIELDS', F.FIELDS + (lonely,))
    monkeypatch.setitem(F.BY_NAME, 'lonely', lonely)
    with pytest.raises(F.FieldRegistryError) as problem:
        F.reference_catalog_candidates('4242', '001')
    message = str(problem.value)
    assert 'fields.yaml' in message and 'reference_catalog:' in message


def test_a_proposal_cannot_have_two_reference_frames(monkeypatch):
    """The frame token names the offsets table, which is per proposal, so two
    fields sharing a proposal must agree rather than one winning silently."""
    # 2045 is arches and quintuplet, so this keeps the case the registry really
    # has: two fields declaring the same proposal, which must agree.
    other = F.Field('other', root='orange', observations=(
        F.Obs(proposal='2045', reference_frame='Gaia'),))
    monkeypatch.setattr(F, 'FIELDS', F.FIELDS + (other,))
    with pytest.raises(F.FieldRegistryError, match='more than one reference frame'):
        F.reference_frame('2045')


def test_a_registry_loaded_from_elsewhere_uses_its_own_roots(tmp_path):
    raw = {'roots': {'orange': '/somewhere/else'},
           'fields': {'x': {'root': 'orange'}}}
    _, loaded = _reload_from(tmp_path, raw)
    assert loaded[0].basepath == '/somewhere/else/x/'


def test_an_obs_built_by_hand_has_working_defaults():
    """The dict fields defaulted to (), so glob_obsid raised AttributeError on
    any Obs not built by the loader."""
    assert F.Obs(proposal='9999').glob_obsid() is None


def test_the_offsets_table_follows_the_basepath_the_caller_is_using():
    """Built from the registry's own root, it pointed at the released tree even
    when GC_BASEPATH_OVERRIDE had sent the run somewhere else."""
    assert F.offsets_table_path('brick', '1182', basepath='/scratch/brick/') == (
        '/scratch/brick/offsets/Offsets_JWST_Brick1182_F444ref.csv')
    assert F.offsets_table_relpath('brick', '1182') == (
        'offsets/Offsets_JWST_Brick1182_F444ref.csv')
    assert F.offsets_table_relpath('brick', '2221') is None


# --------------------------------------------------------------------------
# The obsid wildcard (program 10678, the GC Treasury; issue #413).
# --------------------------------------------------------------------------
# 10678 lands its observations (139 visits, ~1668 planned observations) as the
# campaign executes, so fields.yaml cannot enumerate them ahead of time; its
# block claims them with `obsids: '*'` and every lookup resolves a concrete
# obsid through that wildcard.

def test_the_treasury_field_is_registered():
    """The name must match data-qa's ``mast_monitor.TREASURY_FIELD``, which
    routes every 10678 observation to this field."""
    assert 'gc-treasury' in F.BY_NAME
    assert F.basepath('gc-treasury').startswith('/blue/')
    assert F.obs_filters()['gc-treasury']['10678'] == ['f212n', 'f480m', 'f770w']


def test_a_concrete_obsid_resolves_through_the_wildcard():
    """The reduce drivers hard-index ``mapping[obsid]`` at startup, so the
    mapping itself has to answer for observation numbers the registry never
    listed -- the crash the wildcard exists to prevent."""
    for instrument in ('nircam', 'miri'):
        mapping = F.field_to_reg_mapping('10678', instrument)
        assert mapping, f'{instrument}: empty mapping'
        assert mapping['037'] == 'gc-treasury'
        assert mapping.get('105') == 'gc-treasury'
        assert '042' in mapping
        assert F.target_for_obsid('10678', '037', instrument) == 'gc-treasury'


def test_a_wildcard_free_mapping_still_raises_on_a_missing_obsid():
    """The fallback belongs to the wildcard owner alone; every other proposal
    keeps its exact-key behaviour."""
    mapping = F.field_to_reg_mapping('2221', 'nircam')
    with pytest.raises(KeyError):
        mapping['037']
    assert '037' not in mapping
    assert mapping.get('037') is None


def test_an_explicit_obsid_wins_over_the_wildcard(monkeypatch):
    """A field naming an observation outright is more specific than a field
    claiming everything, so the explicit claim wins for that obsid and the
    wildcard keeps the rest."""
    special = F.Field('special', root='blue', observations=(
        F.Obs(proposal='10678', obsids={'nircam': ('042',)}),))
    monkeypatch.setattr(F, 'FIELDS', F.FIELDS + (special,))
    mapping = F.field_to_reg_mapping('10678', 'nircam')
    assert mapping['042'] == 'special'
    assert mapping['043'] == 'gc-treasury'


def test_two_fields_cannot_both_hold_the_wildcard(monkeypatch):
    """'*' claims every observation, which only one field can do."""
    rival = F.Field('rival', root='blue', observations=(
        F.Obs(proposal='10678', obsids={'nircam': ('*',)}),))
    monkeypatch.setattr(F, 'FIELDS', F.FIELDS + (rival,))
    with pytest.raises(F.FieldRegistryError, match='wildcard'):
        F.field_to_reg_mapping('10678', 'nircam')


def test_the_wildcard_resolves_only_obsid_shaped_keys():
    """A catch-all that answers for ANYTHING absorbs typos.  'nrcb', 'F212N'
    and '0042' are not observation numbers, and a mapping that says they are
    turns a misspelling into a confident wrong field name instead of the
    KeyError the caller can act on."""
    mapping = F.field_to_reg_mapping('10678', 'nircam')
    for good in ('001', '042', '139', '001-002'):
        assert mapping[good] == 'gc-treasury', good
        assert good in mapping
    for bad in ('nrcb', 'F212N', '0042', '42', '', 'merged', None):
        assert bad not in mapping, bad
        assert mapping.get(bad) is None, bad
        with pytest.raises(KeyError):
            mapping[bad]
    with pytest.raises(KeyError):
        F.target_for_obsid('10678', 'not-an-obsid')


def test_a_copy_of_a_wildcard_map_still_resolves():
    """``dict.copy`` returns a plain dict, which drops the fallback and
    restores the KeyError this class exists to prevent."""
    mapping = F.field_to_reg_mapping('10678', 'nircam')
    clone = mapping.copy()
    assert isinstance(clone, F.WildcardObsidMap)
    assert clone.wildcard_target == 'gc-treasury'
    assert clone['037'] == 'gc-treasury'
    assert copy.copy(mapping)['037'] == 'gc-treasury'


def test_a_wildcard_filter_counts_as_more_than_one_observation():
    """``filter_observation_count`` feeds the m2 foreign-observation filter,
    which runs only when the count is > 1.  ``len(('*',)) == 1`` read as "one
    observation images this filter", switching the filter OFF for the one
    field that most needs it: every 10678 tile writes its per-frame catalogs
    into the same <basepath>/<filter>/ tree, all of them visit001, so the
    single-observation branch would collapse different tiles' catalogs onto
    one identity and drop the rest."""
    assert F.filter_observation_count('gc-treasury', 'F212N') > 1
    assert F.filter_observation_count('gc-treasury', 'F480M') > 1
    assert F.filter_observation_count('gc-treasury', 'F770W') > 1
    # ... and the counts every other field reports are untouched.
    assert F.filter_observation_count('gc2211', 'F200W') == 5
    assert F.filter_observation_count('ngc6334', 'F090W') == 1
    assert F.filter_observation_count('sgrb2', 'F770W') == 3


def test_a_scalar_obsid_list_is_refused(tmp_path):
    """'*' is the only supported scalar.  `nircam: '001'` would load as
    ('0', '0', '1') -- three observations that do not exist -- and the
    wildcard makes a bare string look like a supported spelling."""
    raw = {'roots': {'blue': '/b'},
           'fields': {'x': {'root': 'blue',
                            'observations': {'10678': {
                                'obsids': {'nircam': '001'}}}}}}
    with pytest.raises(F.FieldRegistryError, match='scalar'):
        _reload_from(tmp_path, raw)


def test_the_default_reference_catalog_answers_for_any_observation():
    """10678 has no exact reference_catalog keys, so every obsid falls through
    to the default -- reaching stage 1 with a catalog rather than a raise."""
    for obsid in ('001', '037', '139'):
        path = F.reference_catalog_path('10678', obsid)
        assert path == ('/blue/adamginsburg/adamginsburg/jwst/gc-treasury/'
                        'catalogs/gaia_virac2_refcat_epoch2026.65.fits'), obsid
    miri = F.reference_catalog_path('10678', '105', instrument='miri')
    assert miri.endswith('gaia_virac2_refcat_epoch2026.65.fits')


def test_an_exact_reference_catalog_key_wins_over_the_default(monkeypatch):
    """The default is a fallback: an observation with its own key keeps it, so
    a special-epoch refcat can still be pinned per observation."""
    keyed = F.Field('keyed', root='orange', observations=(
        F.Obs(proposal='4242', obsids={'nircam': ('001', '002')},
              reference_catalogs={'001': ('catalogs/exact.fits',)},
              default_reference_catalog=('catalogs/fallback.fits',)),))
    monkeypatch.setattr(F, 'FIELDS', F.FIELDS + (keyed,))
    monkeypatch.setitem(F.BY_NAME, 'keyed', keyed)
    assert F.reference_catalog_candidates('4242', '001')[0].endswith(
        'catalogs/exact.fits')
    assert F.reference_catalog_candidates('4242', '002')[0].endswith(
        'catalogs/fallback.fits')


def test_no_key_and_no_default_still_raises(monkeypatch):
    """The default must never turn a genuinely unregistered catalog into a
    silent empty answer; the raise (and its edit-this-file message) stays."""
    bare = F.Field('bare', root='orange', observations=(
        F.Obs(proposal='4242', obsids={'nircam': ('001',)}),))
    monkeypatch.setattr(F, 'FIELDS', F.FIELDS + (bare,))
    monkeypatch.setitem(F.BY_NAME, 'bare', bare)
    with pytest.raises(F.FieldRegistryError, match='no reference catalog'):
        F.reference_catalog_candidates('4242', '001')


def test_the_wildcard_survives_a_reload(tmp_path):
    """The loader turns the scalar '*' into the wildcard tuple, whatever file
    it reads."""
    raw = {'roots': {'blue': '/b'},
           'fields': {'x': {'root': 'blue',
                            'observations': {'10678': {
                                'obsids': {'nircam': '*'},
                                'default_reference_catalog': 'catalogs/r.fits',
                            }}}}}
    _, loaded = _reload_from(tmp_path, raw)
    obs = loaded[0].observation('10678')
    assert obs.obsids['nircam'] == ('*',)
    assert obs.default_reference_catalog == ('catalogs/r.fits',)


#: `field_to_reg_mapping` for every (proposal, instrument) pair that existed
#: before the wildcard was added, captured from the registry as it stood
#: (2026-08-16).  The wildcard machinery must leave every one of them alone.
FIELD_MAPS_BEFORE_THE_WILDCARD = {
    ('1182', 'nircam'): {'004': 'brick'},
    ('1334', 'nircam'): {'001': 'm92'},
    ('1905', 'nircam'): {'001': 'wd1', '003': 'wd1'},
    ('1939', 'nircam'): {'001': 'sgra'},
    ('1979', 'nircam'): {'001': 'ngc6397', '002': 'm4', '003': 'm4'},
    ('2045', 'nircam'): {'001': 'arches', '003': 'quintuplet'},
    ('2092', 'nircam'): {'002': 'cloudef', '005': 'cloudef'},
    ('2211', 'nircam'): {'023': 'gc2211', '028': 'gc2211', '046': 'gc2211',
                         '049': 'gc2211', '050': 'gc2211'},
    ('2221', 'nircam'): {'001': 'brick', '002': 'cloudc'},
    ('3523', 'nircam'): {'003': 'wd2', '005': 'wd2'},
    ('3958', 'nircam'): {'001': 'sickle', '002': 'sickle', '007': 'sickle'},
    ('4147', 'nircam'): {'012': 'sgrc'},
    ('5365', 'nircam'): {'001': 'sgrb2'},
    ('6151', 'nircam'): {'001': 'w51'},
    ('6778', 'nircam'): {'001': 'ngc6334'},
    ('7213', 'nircam'): {'001': 'ngc6334'},
    ('8322', 'nircam'): {'001': 'omegacen'},
    ('12587', 'nircam'): {'001': 'omegacen'},
    ('2092', 'miri'): {'004': 'cloudef', '006': 'cloudef', '008': 'cloudef'},
    ('2221', 'miri'): {'001': 'cloudc', '002': 'brick'},
    ('2526', 'miri'): {'021': 'cloudc'},
    ('3958', 'miri'): {'001': 'sickle', '001-002': 'sickle', '002': 'sickle',
                       '003': 'brick'},
    ('5365', 'miri'): {'001': 'sgrb2', '002': 'sgrb2', '002-998': 'sgrb2',
                       '998': 'sgrb2'},
    ('6151', 'miri'): {'001': 'w51', '002': 'w51'},
    ('4147', 'niriss'): {'012': 'sgrc'},
}


def test_every_preexisting_proposal_maps_exactly_as_before():
    """The no-regression sweep for the wildcard change: a pre-existing proposal
    has no wildcard, so its mapping must compare equal to the plain dict it
    used to be -- same keys, same targets, no fallback."""
    for (proposal, instrument), expected in FIELD_MAPS_BEFORE_THE_WILDCARD.items():
        got = F.field_to_reg_mapping(proposal, instrument)
        assert dict(got) == expected, (proposal, instrument)
        assert got.wildcard_target is None, (proposal, instrument)

    # And the sweep itself is complete: nothing beyond the snapshot and the
    # treasury answers with a non-empty mapping.
    proposals = sorted({o.proposal for f in F.FIELDS for o in f.observations},
                       key=int)
    populated = {(p, inst) for inst in F.INSTRUMENTS for p in proposals
                 if F.field_to_reg_mapping(p, inst)}
    new = populated - set(FIELD_MAPS_BEFORE_THE_WILDCARD)
    assert new == {('10678', 'nircam'), ('10678', 'miri')}, sorted(new)
