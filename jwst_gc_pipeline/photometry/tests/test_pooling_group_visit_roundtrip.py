"""The writer and the reader must agree on what a visit is.

``_record_pooling`` writes the correction's visit into each pooled group, and
that value has been through ``resolve_full_visit_id``, so it is the full
``jwPPPPPOOOVVV``.  The exposure key on the other side of the round trip carries
the frame's bare ``VISIT`` metadatum -- ``'1'``, ``'2'``.

A reader that compares them as strings evaluates ``'1' == 'jw02221001001'`` and
matches nothing.  That is worse than not writing the field at all: with no
visit, ``measure_pooling_population`` falls back to a candidate sweep that
reconstructs every group, and the moment the field is populated the sweep is
skipped and every group is discarded instead.  Measured against the live tree
on 2026-08-19: 1758 of 1758 groups reconstruct through ``_same_visit``, 0 of
1758 through a string compare.

These tests pin both halves -- that the writer emits the full id, and that the
reader's comparison spans the two spellings -- because either one alone passes
while the round trip is broken.
"""
import importlib.util
import os

import pytest


REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


def _population_module():
    """Import ``reports/measure_pooling_population.py`` by path.

    It lives outside the package, and importing it must not run the survey --
    it is ``__main__``-guarded, so a plain spec load is safe.
    """
    path = os.path.join(REPO, 'reports', 'measure_pooling_population.py')
    spec = importlib.util.spec_from_file_location('_pool_pop', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize('key_visit,group_visit,same', [
    # the round trip this exists for: bare key, full group value
    ('1', 'jw02221001001', True),
    ('2', 'jw02221001002', True),
    # a different visit of the same observation is NOT the same visit
    ('2', 'jw02221001001', False),
    ('1', 'jw02221001002', False),
    # both bare, the legacy shape
    ('1', '1', True),
    ('1', '2', False),
    # both full, and disagreeing on the observation
    ('jw02211023001', 'jw02211046001', False),
    ('jw02211023001', 'jw02211023001', True),
])
def test_the_two_visit_spellings_compare_as_one_visit(key_visit, group_visit,
                                                      same):
    mod = _population_module()
    assert mod._same_visit(key_visit, group_visit) is same


def test_a_bare_key_is_not_narrowed_away_by_an_observation_it_cannot_carry():
    """The exposure key has no observation field, so requiring the observations
    to match would discard every group written by the new writer.

    This is the same loose rule ``_match_rows`` applies to a table whose rows
    mix the two forms: compare the observation only when both sides have one.
    """
    mod = _population_module()
    assert mod._same_visit('1', 'jw02211023001')
    assert mod._same_visit('1', 'jw02211046001')
    # ...and the visit number still has to agree
    assert not mod._same_visit('3', 'jw02211023001')


def _by_key(visit='1', vgroup='02201', exposure=1):
    """Two detectors of one module, keyed the way the records key them:
    ``(visit, exposure, module, filter, vgroup)`` with a BARE visit."""
    return {
        (visit, exposure, 'nrca1', 'F212N', vgroup): (1.0, 2.0),
        (visit, exposure, 'nrca2', 'F212N', vgroup): (3.0, 4.0),
    }


def test_a_group_carrying_the_full_visit_id_still_finds_its_exposures():
    """The end-to-end round trip, which is what actually broke.

    Asserting on `_same_visit` alone leaves the call site free to compare the
    two spellings as strings: that mutation passes every test above while
    reconstructing 0 of 1758 groups against the live tree.
    """
    mod = _population_module()
    by_key = _by_key(visit='1')
    group = dict(visit='jw02221001001', exposure=1, vgroup='02201', n=2,
                 dra_onsky_mas=2.0, ddec_onsky_mas=3.0)   # medians of the two

    members = mod._members_of(by_key, group, ('nrca1', 'nrca2'))
    assert members is not None, (
        'the group was discarded: the reader could not match its full visit '
        'id against the exposure keys\' bare visit')
    assert len(members) == 2
    assert sorted(members) == [(1.0, 2.0), (3.0, 4.0)]


def test_a_group_from_the_other_visit_is_still_not_matched():
    """The normalisation must not become "any visit" -- pooling across visits
    is the defect the visit key was added to prevent."""
    mod = _population_module()
    by_key = _by_key(visit='2')
    group = dict(visit='jw02221001001', exposure=1, vgroup='02201', n=2,
                 dra_onsky_mas=2.0, ddec_onsky_mas=3.0)
    assert mod._members_of(by_key, group, ('nrca1', 'nrca2')) is None


def test_the_writer_puts_the_full_visit_id_on_the_group():
    """``_record_pooling`` copies the correction's visit, and corrections carry
    the resolved full id.  A bare number here would make the reader's loose
    rule the only thing holding the match together."""
    from jwst_gc_pipeline.photometry.cataloging import _record_pooling

    corrections = [dict(visit='jw02221001001', exposure=1, module='nrca',
                        filtername='F212N', vgroup='02201',
                        pooled_from=('nrca1', 'nrca2'), pooled_n=2,
                        pooled_stat='median', dra_onsky_mas=1.0,
                        ddec_onsky_mas=2.0)]
    rec = {}
    # no `record_path`, so nothing is written to disk; the group dicts are
    # built either way and they are what this pins.
    _record_pooling(rec, corrections, 2, '/x/offsets.csv')
    groups = (rec.get('pooling') or {}).get('groups') or []
    assert groups, 'nothing was recorded'
    assert groups[0]['visit'] == 'jw02221001001', (
        f"the group carries {groups[0]['visit']!r}; a reader matching it "
        f"against the exposure key's bare visit needs the full id to resolve "
        f"the observation")
