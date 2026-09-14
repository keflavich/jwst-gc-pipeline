"""The 39 lines that put the panel on the page, which had no tests.

The panel itself was well covered; how it reaches the page was not.  These pin
the three wiring decisions, each of which is invisible until it regresses:

* the panel is on the FRONT page and off the per-field pages -- it is about the
  survey, not about one field, and 18 copies of it is 18 chances to disagree;
* ``schedule.json`` is published alongside the page, or the provenance the
  panel cites cannot be checked by a reader;
* ``--schedule-program ''`` turns it off, which is the documented escape hatch
  when STScI's format changes under a running cron.
"""
import datetime
import json
import os

from jwst_gc_pipeline.monitoring import __main__ as cli
from jwst_gc_pipeline.monitoring import render, report
from jwst_gc_pipeline.monitoring import schedule as S
from jwst_gc_pipeline.monitoring.tests.test_schedule import REPORT


#: A visit whose start is in the future, so it renders in the `sched` state.
#:
#: Both visits parsed out of REPORT are in the past and come out `wait`, so
#: before this the "every state is linked" test compared 2 against 2 and never
#: rendered a scheduled badge at all -- unlinking `sched` in `_state_badge`
#: left the whole file green.  `sched` is the state the linking rule is most
#: arguable for, which makes it the one that has to be exercised.
def _future_visit():
    when = (datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(days=3))
    return {'program': '10678', 'visit_id': '10678:99:1', 'observation': '099',
            'target': 'GC_99', 'instrument': 'NIRCam Imaging',
            'duration': '01:00:00', 'parallels': [],
            'start': when.strftime('%Y-%m-%dT%H:%M:%SZ')}


def _sched(with_future=True):
    visits = [v for v in S.parse_report(REPORT) if v['program'] == '10678']
    if with_future:
        visits = visits + [_future_visit()]
    return {'program': '10678', 'stale': False, 'fetched': 'x',
            'weeks': [{'week': '20260817', 'generated': '20260814', 'url': 'u',
                       'n_visits': 3, 'n_program': 2}],
            'visits': visits}


def test_the_front_page_carries_the_panel():
    page = render.render_page([], schedule=_sched(), include_skyview=False)
    assert 'id="schedule"' in page


def test_a_per_field_page_does_not():
    page = render.render_page([], schedule=_sched(), include_skyview=False,
                              include_schedule=False)
    assert 'id="schedule"' not in page


def test_no_schedule_means_no_panel_and_no_crash():
    assert 'id="schedule"' not in render.render_page(
        [], schedule=None, include_skyview=False)


def test_schedule_json_is_published_beside_the_page(tmp_path):
    """The panel cites its provenance; a reader has to be able to open it.

    Exercised through publish() rather than by reading the source, so a
    refactor that keeps the name and drops the link still fails.
    """
    outdir = tmp_path / 'out'
    outdir.mkdir()
    (outdir / 'monitor.html').write_text('<html></html>')
    S.write_json(str(outdir), _sched())
    pub = tmp_path / 'pub'
    report.publish(str(outdir), str(pub))
    assert (pub / S.SCHEDULE_JSON).exists()
    assert json.load(open(pub / S.SCHEDULE_JSON))['program'] == '10678'


def test_the_cli_defaults_to_the_treasury_and_can_be_switched_off():
    parser = cli.build_parser()
    assert parser.parse_args(['--outdir', '/tmp/x']).schedule_program == \
        S.DEFAULT_PROGRAM
    off = parser.parse_args(['--outdir', '/tmp/x', '--schedule-program', ''])
    assert off.schedule_program == ''


def test_write_report_with_no_program_fetches_nothing(tmp_path, monkeypatch):
    """`--schedule-program ''` is the escape hatch if STScI changes the format
    under a running cron; it has to actually skip the network."""
    called = []
    monkeypatch.setattr(S, 'load', lambda *a, **k: called.append(a) or {})
    monkeypatch.setattr(report, 'build_entries', lambda *a, **k: ([], []))
    monkeypatch.setattr(report, 'collect_cutouts', lambda *a, **k: [])
    out = report.write_report(outdir=str(tmp_path), schedule_program=None,
                              per_field=False, with_cutouts=False)
    assert called == []
    assert out['schedule'] is None


def test_write_report_writes_the_json_when_a_program_is_given(tmp_path,
                                                              monkeypatch):
    monkeypatch.setattr(S, 'load', lambda *a, **k: _sched())
    monkeypatch.setattr(report, 'build_entries', lambda *a, **k: ([], []))
    monkeypatch.setattr(report, 'collect_cutouts', lambda *a, **k: [])
    report.write_report(outdir=str(tmp_path), schedule_program='10678',
                        per_field=False, with_cutouts=False)
    path = os.path.join(str(tmp_path), S.SCHEDULE_JSON)
    assert os.path.exists(path)
    assert json.load(open(path))['program'] == '10678'


# --- the state badge is where the archive link went --------------------------

def _rows():
    from jwst_gc_pipeline.monitoring import schedule_section
    return schedule_section.section(_sched())


def test_the_state_badge_links_to_the_archive():
    """The archive link used to sit on the footprints in the sky map, where a
    click meant "select this tile" or "leave the site" depending on a status
    the reader could not see under the cursor.  Here the row already names the
    observation and states what is known about it."""
    import re
    html = _rows()
    badges = re.findall(
        r'<a class="gcm-sch-mast" href="([^"]+)"[^>]*>'
        r'<span class="gcm-sch-badge is-(\w+)">', html)
    assert badges, 'no state badge is linked'
    for url, _state in badges:
        assert url.startswith('https://mast.stsci.edu')
        assert 'program_id=10678' in url
        assert 'observtn=' in url


def test_every_state_is_linked_including_the_ones_with_nothing_there_yet():
    """A badge that is a link on some rows and a bare span on others is the
    ambiguity this moved away from.  The query is well formed for a scheduled
    visit too; the tooltip says to expect an empty result."""
    import re
    html = _rows()
    states = re.findall(r'<span class="gcm-sch-badge is-(\w+)"', html)
    # the fixture has to actually contain the state the rule is arguable for,
    # or this compares a number against itself
    assert 'sched' in states, states
    linked = len(re.findall(r'<a class="gcm-sch-mast"', html))
    assert states and linked == len(states)


def test_the_scheduled_badge_in_particular_is_a_link():
    """Named separately from the count above: a count can be satisfied by a
    fixture that never renders this state, which is how it was satisfied."""
    import re
    html = _rows()
    assert re.search(
        r'<a class="gcm-sch-mast"[^>]*>\s*<span class="gcm-sch-badge is-sched"',
        html), 'the scheduled badge is not linked'


def test_the_tooltip_says_what_to_expect_from_each_state():
    from jwst_gc_pipeline.monitoring import schedule_section as ss
    assert set(ss._MAST_TITLE) == {'ok', 'wait', 'sched'}
    assert 'expect an empty result' in ss._MAST_TITLE['sched']
    assert 'already scanned data' in ss._MAST_TITLE['ok']
    assert 'not scanned data' in ss._MAST_TITLE['wait']


def test_a_visit_with_an_unusable_observation_number_keeps_a_plain_badge():
    """`mast_url` returns None rather than a broken query; the badge still has
    to render, because the state is what the column is for."""
    from jwst_gc_pipeline.monitoring import schedule_section as ss
    plain = ss._state_badge('ok', 'on disk', '10678', 'GC_3')
    assert plain.startswith('<span class="gcm-sch-badge')
    assert 'href' not in plain


def test_the_badge_keeps_its_state_colour_through_the_link():
    """Recolouring it as a link would destroy the state it encodes."""
    from jwst_gc_pipeline.monitoring import schedule_section as ss
    assert '.gcm-sch-mast { text-decoration: none; }' in ss.CSS
    linked = ss._state_badge('sched', 'scheduled', '10678', '3')
    assert 'class="gcm-sch-badge is-sched"' in linked


def test_a_forgotten_program_argument_is_a_typeerror_not_a_silent_unlink():
    """`_row(..., program=None)` used to default, and a caller that omitted it
    produced a table where every badge was a bare span -- the 65-of-65 linking
    property resting on one call site remembering an optional argument."""
    import datetime as _dt
    import pytest
    from jwst_gc_pipeline.monitoring import schedule_section as ss
    now = _dt.datetime.now(_dt.timezone.utc)
    with pytest.raises(TypeError):
        ss._row(_future_visit(), now, set())
