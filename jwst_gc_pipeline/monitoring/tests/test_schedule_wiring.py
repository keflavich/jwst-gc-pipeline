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
import json
import os

from jwst_gc_pipeline.monitoring import __main__ as cli
from jwst_gc_pipeline.monitoring import render, report
from jwst_gc_pipeline.monitoring import schedule as S
from jwst_gc_pipeline.monitoring.tests.test_schedule import REPORT


def _sched():
    return {'program': '10678', 'stale': False, 'fetched': 'x',
            'weeks': [{'week': '20260817', 'generated': '20260814', 'url': 'u',
                       'n_visits': 3, 'n_program': 2}],
            'visits': [v for v in S.parse_report(REPORT)
                       if v['program'] == '10678']}


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
