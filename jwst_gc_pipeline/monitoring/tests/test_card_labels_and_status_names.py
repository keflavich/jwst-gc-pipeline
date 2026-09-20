"""Every observation is findable on the overview page, and by a distinct name.

Two ways an observation went missing:

* **Indistinguishable.** 10678 renders one card per observation, and all 45 of
  them were labelled `gc-treasury`.  The observation was on the card, in the
  smaller `proposal/oNNN` chip, but a grid is scanned by its names.
* **Absent.** o062 is GC_62, status `Implementation`: no data, so no card; not
  scheduled, so no schedule row; and its status was rendered as a bare count,
  so the number appeared nowhere.  "Is o062 in this survey" had no answer.
"""
import re

from jwst_gc_pipeline.monitoring import render, skyview


def _entry(target, obsid, proposal='10678'):
    return {'run': {'target': target, 'proposal': proposal, 'obsid': obsid,
                    'per_filter': {}},
            'tally': {}, 'worst': 'info', 'anchor': f'f-{target}-{obsid}',
            'newest_mtime': None, 'jobs': []}


def test_a_multi_observation_field_names_the_observation():
    entries = [_entry('gc-treasury', '040'), _entry('gc-treasury', '062'),
               _entry('gc-treasury', '135')]
    labels = render.card_labels(entries)
    assert sorted(labels.values()) == ['gc-treasury-040', 'gc-treasury-062',
                                       'gc-treasury-135']


def test_a_single_observation_field_keeps_its_bare_name():
    """Appending an obsid to a field that has only one is noise, not detail."""
    entries = [_entry('brick', '001', '2221'), _entry('gc-treasury', '040')]
    assert sorted(render.card_labels(entries).values()) == ['brick',
                                                            'gc-treasury']


def test_the_rendered_cards_carry_the_distinct_names():
    """Matched against the NAME span, not the page.

    `'gc-treasury-040' in page` passes on the anchor (`f-gc-treasury-040`)
    whatever the label says, so it went on passing with the labels reverted.
    """
    entries = [_entry('gc-treasury', '040'), _entry('gc-treasury', '135')]
    page = render.render_page(entries, include_skyview=False,
                              include_detail=False, include_schedule=False)
    shown = re.findall(r'<span class="gcm-card-name">([^<]*)</span>', page)
    assert shown == ['gc-treasury-040', 'gc-treasury-135']


def test_an_entry_with_no_obsid_does_not_get_a_trailing_dash():
    entries = [_entry('gc-treasury', None), _entry('gc-treasury', '040')]
    assert set(render.card_labels(entries).values()) == {'gc-treasury',
                                                         'gc-treasury-040'}


def _footprints(**counts):
    planned = []
    for status, numbers in counts.items():
        for n in numbers:
            planned.append({'target': f'GC_{n}', 'number': str(n),
                            'status': status, 'ra': 266.0, 'dec': -29.0,
                            'nircam': [], 'miri': []})
    return {'program': '10678', 'status_source': 'stsci-visit-status',
            'planned': planned, 'observed': [],
            'status_counts': {k: len(v) for k, v in counts.items()}}


def test_an_implementation_tile_is_named_not_just_counted():
    """o062's case: counted, drawn, and previously nameless."""
    html = skyview.section(_footprints(Implementation=[62, 14],
                                       Scheduled=[44]))
    assert 'Implementation 2' in html
    assert '>62<' in html


def test_the_named_tail_is_folded_away_by_default():
    """59 numbers with no date between them is a wall if always shown."""
    html = skyview.section(_footprints(Implementation=list(range(1, 60))))
    assert '<details class="gcm-sky-more">' in html
    # <details> with no `open` attribute: present in the document, collapsed.
    start = html.index('<details class="gcm-sky-more">')
    assert 'open' not in html[start:start + len('<details class="gcm-sky-more">')]


def test_a_status_with_no_nameable_tiles_stays_a_count():
    """Report the count rather than inventing a list the data cannot support."""
    fp = _footprints(Implementation=[62])
    fp['status_counts']['Withdrawn'] = 3      # counted, but no tile carries it
    html = skyview.section(fp)
    assert 'Withdrawn 3' in html
