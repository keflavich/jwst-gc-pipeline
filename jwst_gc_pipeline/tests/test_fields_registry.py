"""The registry must reproduce today's dictionaries exactly.

That is what makes the migration safe: call sites can move one at a time,
because each view is byte-for-byte what the dictionary it replaces holds now.
Where a view deliberately differs, the difference is written down here.
"""
import ast
import pathlib
import re

import pytest

from jwst_gc_pipeline import fields
from jwst_gc_pipeline.photometry import merge_catalogs as MC

# Today's dictionaries disagree in two places, in two different ways.  The
# registry holds each fact once, so it cannot reproduce either -- which is the
# point.  Both are bugs, not conventions.
#
#   cloudc/2526  in obs_filters, absent from project_obsnum.  The registry
#                carries it with obsid=None; merge_catalogs.py:1482 and :1701
#                index project_obsnum unguarded, so that pair raises KeyError.
INCOMPLETE = {('cloudc', '2526')}
#   w51/1182     in project_obsnum only.  Nothing iterates it -- the job list
#                comes from obs_filters -- so it is simply dead, and the
#                registry drops it.
DROPPED = {('w51', '1182')}

# The SLURM array index is derived from each target's proposal order, and the
# cross-band merged catalogs' column order from each filter list.  WRITTEN OUT,
# not recomputed: the tests below must keep meaning something after migration
# step 2 deletes `MC.obs_filters` and points it at the view, at which point any
# comparison against that name becomes `view == view`.
EXPECTED_JOB_ORDER = {
    'arches': (
        ('2045', ('f212n', 'f323n')),
    ),
    'brick': (
        ('2221', ('f410m', 'f212n', 'f466n', 'f405n', 'f187n', 'f182m', 'f2550w')),
        ('1182', ('f444w', 'f356w', 'f200w', 'f115w')),
    ),
    'cloudc': (
        ('2221', ('f410m', 'f212n', 'f466n', 'f405n', 'f187n', 'f182m', 'f2550w')),
        ('2526', ('f770w',)),
    ),
    'cloudef': (
        ('2092', ('f162m', 'f210m', 'f360m', 'f480m', 'f770w', 'f2100w')),
    ),
    'gc2211': (
        ('2211', ('f150w', 'f200w', 'f277w')),
    ),
    'm4': (
        ('1979', ('f150w2', 'f322w2')),
    ),
    'm92': (
        ('1334', ('f090w', 'f150w', 'f277w', 'f444w')),
    ),
    'ngc6334': (
        ('7213', ('f115w', 'f162m', 'f182m', 'f200w', 'f356w', 'f405n', 'f444w', 'f470n')),
        ('6778', ('f090w', 'f187n', 'f200w', 'f277w', 'f335m', 'f470n')),
    ),
    'ngc6397': (
        ('1979', ('f150w2', 'f322w2')),
    ),
    'quintuplet': (
        ('2045', ('f212n', 'f323n')),
    ),
    'sgra': (
        ('1939', ('f115w', 'f212n', 'f405n')),
    ),
    'sgrb2': (
        ('5365', ('f150w', 'f182m', 'f187n', 'f210m', 'f212n', 'f300m', 'f360m', 'f405n', 'f410m', 'f466n', 'f480m', 'f770w', 'f1280w', 'f2550w')),
    ),
    'sgrc': (
        ('4147', ('f115w', 'f162m', 'f182m', 'f212n', 'f360m', 'f405n', 'f470n', 'f480m')),
    ),
    'sickle': (
        ('3958', ('f187n', 'f210m', 'f335m', 'f470n', 'f480m', 'f770w', 'f1130w', 'f1500w')),
    ),
    'w51': (
        ('6151', ('f140m', 'f162m', 'f182m', 'f187n', 'f210m', 'f335m', 'f360m', 'f405n', 'f410m', 'f480m', 'f770w', 'f1280w', 'f2100w')),
    ),
    'wd1': (
        ('1905', ('f115w', 'f150w', 'f164n', 'f187n', 'f200w', 'f212n', 'f277w', 'f323n', 'f405n', 'f444w', 'f466n')),
    ),
    'wd2': (
        ('3523', ('f115w', 'f150w', 'f162m', 'f164n', 'f182m', 'f187n', 'f200w', 'f212n', 'f250m', 'f277w', 'f300m', 'f323n', 'f335m', 'f405n', 'f410m', 'f444w', 'f466n')),
    ),
}


def _literal_nvisits():
    """`nvisits` lives inside main(), so it can only be read as source."""
    src = (pathlib.Path(fields.__file__).parent / 'photometry'
           / 'crowdsource_catalogs_long.py').read_text()
    match = re.search(r'^\s*nvisits = \{', src, re.M)
    start = match.end() - 1
    depth = 0
    for end in range(start, len(src)):
        depth += (src[end] == '{') - (src[end] == '}')
        if depth == 0:
            break
    return ast.literal_eval(src[start:end + 1])


def test_obs_filters_view_matches_todays_dict():
    assert fields.obs_filters() == MC.obs_filters


@pytest.mark.parametrize('target', sorted(EXPECTED_JOB_ORDER))
def test_view_preserves_proposal_and_filter_ORDER(target):
    """Order is not cosmetic, and this is pinned to a literal on purpose.

    `individual_frame_merge_jobs` builds the SLURM array index from each
    target's proposal order, so swapping two observations inside one field
    sends every array task at a different filter.  Filter order sets the column
    order of the cross-band merged catalogs.  Value equality sees neither.
    """
    view = fields.obs_filters()[target]
    assert tuple((p, tuple(fs)) for p, fs in view.items()) == \
        EXPECTED_JOB_ORDER[target], (
            f'{target}: order changed -- SLURM array indices and cross-band '
            f'column order would move')


def test_the_written_out_order_still_matches_todays_dict():
    """Guards the literal above against drifting from reality WHILE both exist.

    This is the only test here that may legitimately be deleted at step 2.
    """
    today = {t: tuple((p, tuple(fs)) for p, fs in d.items())
             for t, d in MC.obs_filters.items()}
    assert today == EXPECTED_JOB_ORDER


@pytest.mark.parametrize('target', sorted(EXPECTED_JOB_ORDER))
def test_the_job_list_built_from_the_view_is_identical(target):
    """End-to-end: the (proposal, filter) pairs an array job would dispatch."""
    from_view = [(p, f) for p, fs in fields.obs_filters()[target].items()
                 for f in fs]
    expected = [(p, f) for p, fs in EXPECTED_JOB_ORDER[target] for f in fs]
    assert from_view == expected


def test_offsets_view_covers_every_registered_proposal():
    """Today's dict omits 1905, 3523 and 2526, so `offsets_tables[progid]`
    raises KeyError for every wd1/wd2 per-filter merge."""
    view = fields.offsets_table_paths()
    needed = {p for d in MC.obs_filters.values() for p in d}
    assert needed <= set(view), sorted(needed - set(view))


def test_the_brick_offsets_path_is_exactly_todays():
    """The one real table in the pipeline, compared against the literal in
    merge_catalogs rather than by substring -- appending '.BOGUS' to the
    registry path used to pass."""
    src = (pathlib.Path(fields.__file__).parent / 'photometry'
           / 'merge_catalogs.py').read_text()
    literal = re.search(r"'1182': Table\.read\(f'([^']+)'\)", src).group(1)
    assert fields.offsets_table_paths()['1182'] == literal
    assert pathlib.Path(literal).exists(), literal


def test_offsets_view_is_paths_not_tables():
    """Naming matters here: today's dict holds a read Table, and a str passes
    the `is not None` guard before raising TypeError on ['Visit']."""
    assert not hasattr(fields, 'offsets_tables')
    value = fields.offsets_table_paths()['1182']
    assert isinstance(value, str)


def test_a_shared_proposal_cannot_silently_drop_a_table():
    """2221 is brick+cloudc, 2045 arches+quintuplet, 1979 m4+ngc6397.  Keyed by
    proposal alone downstream, so a per-field table would vanish."""
    shared = fields.Field('x', root='blue', observations=(
        fields.Obs('2221', offsets_table='/a.csv'),))
    other = fields.Field('y', root='blue', observations=(
        fields.Obs('2221', offsets_table='/b.csv'),))
    original = fields.FIELDS
    try:
        fields.FIELDS = (shared, other)
        with pytest.raises(ValueError, match='shared by more than one field'):
            fields.offsets_table_paths()
    finally:
        fields.FIELDS = original


def test_project_obsnum_view_matches_todays_dict():
    view, today = fields.project_obsnum(), MC.project_obsnum
    assert set(view) == set(today)
    for target in today:
        expected = dict(today[target])
        got = dict(view[target])
        for name, proposal in INCOMPLETE | DROPPED:
            if name == target:
                expected.pop(proposal, None)
                got.pop(proposal, None)
        assert got == expected, target


def test_nvisits_view_matches_todays_dict():
    assert fields.nvisits() == _literal_nvisits()


def _orange_targets_from_the_branch():
    """Parse the tuple out of merge_catalogs rather than retyping it.

    A hand-copied list in the test would be one more duplicate of the very
    thing this registry exists to remove.
    """
    src = (pathlib.Path(fields.__file__).parent / 'photometry'
           / 'merge_catalogs.py').read_text()
    start = src.index("    if target in ('sickle'")
    end = src.index('):', start)
    return set(re.findall(r"'([a-z0-9]+)'", src[start:end]))


def test_basepath_matches_the_branch_it_replaces():
    orange = _orange_targets_from_the_branch()
    assert orange, 'could not parse the branch'
    for target in fields.BY_NAME:
        expected = (f'/orange/adamginsburg/jwst/{target}/' if target in orange
                    else f'/blue/adamginsburg/adamginsburg/jwst/{target}/')
        assert fields.basepath(target) == expected, target


@pytest.mark.parametrize('unknown', ['ngc3603', 'not_a_field', ''])
def test_an_unregistered_target_gets_the_blue_tree(unknown):
    """The branch has no membership test on its else -- anything unknown gets
    blue.  Raising KeyError instead would break every unregistered caller."""
    assert fields.basepath(unknown) == (
        f'/blue/adamginsburg/adamginsburg/jwst/{unknown}/')


# --- what the registry buys: these cannot be written down inconsistently -----

def test_every_observation_is_complete():
    """The failure the scattered dictionaries allow, caught in one place."""
    incomplete = [(f.name, o.proposal) for f in fields.FIELDS
                  for o in f.observations
                  if o.obsid is None or o.nvisits is None or not o.filters]
    unexpected = set(incomplete) - INCOMPLETE
    assert not unexpected, (
        f'observations missing an obsid, nvisits or filters: {sorted(unexpected)}')


def test_the_incomplete_set_is_still_exactly_that():
    """If someone fixes cloudc/2526, this fails and says to update the list."""
    incomplete = {(f.name, o.proposal) for f in fields.FIELDS
                  for o in f.observations
                  if o.obsid is None or o.nvisits is None or not o.filters}
    assert incomplete == INCOMPLETE, (
        f'the set of incomplete observations changed: {sorted(incomplete)}.  '
        f'If you fixed one, remove it from INCOMPLETE.')


def test_the_dropped_entries_are_still_absent_from_obs_filters():
    """w51/1182 is dead because the job list is built from obs_filters."""
    for target, proposal in DROPPED:
        assert proposal not in MC.obs_filters.get(target, {}), (
            f'{target}/{proposal} is now in obs_filters, so it is no longer '
            f'dead -- add it to the registry and drop it from DROPPED.')


@pytest.mark.parametrize('name', sorted(fields.BY_NAME))
def test_no_field_is_empty(name):
    assert fields.BY_NAME[name].observations, f'{name} has no observations'
