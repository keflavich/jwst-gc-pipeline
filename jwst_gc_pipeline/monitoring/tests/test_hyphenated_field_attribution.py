"""A field whose NAME contains a hyphen is attributable from its job names.

`gc-treasury` is the first registered field with a hyphen in it, and the head
pattern (`[a-z][a-z0-9_]*`) had no hyphen, so `gc-treasury10678-o040-m4-finalize`
matched a head of `gc`, resolved to no field, fell through every shape and was
reported as belonging to no registered field.  All 292 of 10678's queued jobs
at once -- which the overview page then printed BY NAME, together with whatever
else the account happened to be running.

Both halves are pinned here: the names attribute, and the page does not list
them.
"""
import pytest

from jwst_gc_pipeline.monitoring import jobs, render

#: Enough of the registry to exercise the longest-first rule without reading it.
TARGETS = ('cloudef_controlfield', 'gc-treasury', 'cloudef', 'sgrb2', 'brick',
           'arches', 'sgrb', 'w51')


@pytest.mark.parametrize('name,obsid,stage', [
    ('gc-treasury10678-o040-m4-finalize', '040', 'm4'),
    ('gc-treasury10678-o041-m12-finalize', '041', 'm12'),
    ('gc-treasury10678-o127-m7-fanout', '127', 'm7'),
    # The submitters disagree with the registry about the hyphen, and both
    # spellings are real 10678 jobs seen in the queue.
    ('gctreasury10678-o061-reduce', '061', 'reduce'),
    ('gctreasury10678-o040-regen', '040', 'regen'),
])
def test_a_hyphenated_field_attributes(name, obsid, stage):
    got = jobs.parse_job_name(name, TARGETS)
    assert got is not None, f'{name} was not attributed'
    assert got['target'] == 'gc-treasury'
    assert got['proposal'] == '10678'
    assert got['obsid'] == obsid
    assert got['stage'] == stage


def test_the_filter_survives_the_hyphenated_head():
    got = jobs.parse_job_name('gc-treasury10678-o127-cat-F212N', TARGETS)
    assert got['obsid'] == '127'
    assert got['stage'] == 'cat'
    assert got['filter'] == 'F212N'


@pytest.mark.parametrize('name', [
    'brick2221-o001-m12-fanout',
    'sgrb25365-o001-m12-finalize',
    'arches-001-m12-fanout',
    'pf_sgrb2_m12_s3',
    'brick-catalog',
    'w51-9filt-reseed-gaiafix',
    'cloudef_controlfield-o004-m12-fanout',
    'brick',
])
def test_the_shapes_that_already_worked_still_do(name):
    """The registry-first path must not cost the un-hyphenated names."""
    assert jobs.parse_job_name(name, TARGETS) is not None


@pytest.mark.parametrize('name', [
    'data-qa-board-sync',
    'c2d-spx',
    'interactive',
    # `sgrb` is registered; `sgrbfoo` is not, and must not be filed under it.
    'sgrbfoo-o001-m12',
])
def test_an_unrelated_job_is_still_unattributed(name):
    """Widening the match must not start claiming other people's jobs."""
    assert jobs.parse_job_name(name, TARGETS) is None


def test_longest_first_survives(monkeypatch):
    """`sgrb2` must win over `sgrb`, and `cloudef_controlfield` over `cloudef`."""
    assert jobs.parse_job_name('sgrb25365-o001-m12', TARGETS)['target'] == 'sgrb2'
    got = jobs.parse_job_name('cloudef_controlfield-o004-m12', TARGETS)
    assert got['target'] == 'cloudef_controlfield'


def _entry(target, obsid, proposal='10678'):
    return {'run': {'target': target, 'proposal': proposal, 'obsid': obsid,
                    'per_filter': {}},
            'tally': {}, 'worst': 'info', 'anchor': f'f-{target}-{obsid}',
            'newest_mtime': None, 'jobs': []}


def test_the_page_does_not_list_unattributed_job_names():
    """A queue is not all one campaign; its names are not this page to publish."""
    page = render.render_page(
        [_entry('brick', '001')],
        unattributed_jobs=[{'name': 'c2d-spx'},
                           {'name': 'some-private-thing'}],
        include_skyview=False, include_detail=False, include_schedule=False)
    assert 'c2d-spx' not in page
    assert 'some-private-thing' not in page
    assert 'could not be attributed' not in page


def test_the_prefix_split_requires_a_name_boundary():
    """`_split_registered_prefix` is correct on its own, not only via its caller.

    `parse_job_name` would reject `sgrbfoo-o001-m12` anyway, because the
    remainder parser needs a leading separator.  That makes the boundary check
    here a SECOND guard, and a second guard that nothing exercises is one a
    later edit deletes as dead.  Asserted directly so it is not.
    """
    assert jobs._split_registered_prefix('sgrbfoo-o001-m12', TARGETS) == (
        None, '', 'sgrbfoo-o001-m12')
    assert jobs._split_registered_prefix('sgrb-o001-m12', TARGETS) == (
        'sgrb', '', '-o001-m12')
    assert jobs._split_registered_prefix('gc-treasury10678-o040-m4', TARGETS) == (
        'gc-treasury', '10678', '-o040-m4')
