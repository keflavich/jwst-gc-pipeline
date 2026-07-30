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


def test_basepath_matches_the_branch_it_replaces():
    orange = ('sickle', 'cloudef', 'sgrc', 'sgrb2', 'arches', 'quintuplet',
              'sgra', 'gc2211', 'w51', 'm92', 'ngc6397', 'm4', 'ngc6334')
    for target in fields.BY_NAME:
        expected = (f'/orange/adamginsburg/jwst/{target}/' if target in orange
                    else f'/blue/adamginsburg/adamginsburg/jwst/{target}/')
        assert fields.basepath(target) == expected, target


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
