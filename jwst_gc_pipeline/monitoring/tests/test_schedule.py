"""Reading the published observing schedule, and what the page may claim from it.

The report is fixed-width text with two shapes that break naive parsing, and one
claim the panel must never make.

Shapes:

* a **continuation line** with a blank visit id is the COORDINATED PARALLEL of
  the visit above -- 10678 attaches a MIRI parallel to every NIRCam prime, so
  treating it as its own visit doubles every pointing and puts a MIRI row on the
  page with no start time;
* the KEYWORDS column is comma-and-space-separated free text
  (``Circumstellar clouds, Circumstellar disks``), and TARGET NAME can contain
  spaces, so splitting on whitespace shifts every column right of them.  The
  file carries its own column ruler; use it.

Claim: a schedule is a PLAN.  STScI's own note is that executed observations can
differ from those scheduled.  So a past visit whose data is not on disk is "not
seen", never "missing" -- the page cannot establish a fault from a plan.
"""
import datetime
import json
import os

import pytest

from jwst_gc_pipeline.monitoring import schedule as S
from jwst_gc_pipeline.monitoring import schedule_section as SS

#: Trimmed from 20260817_report_20260814.txt -- the real column widths, one
#: 10678 prime with its MIRI parallel, and a neighbouring program whose KEYWORDS
#: contain the commas that broke every delimiter-guessing version.
REPORT = """Visit Information for OP Package 2622808f01

VISIT ID       PCS MODE    VISIT TYPE                     SCHEDULED START TIME  DURATION     SCIENCE INSTRUMENT AND MODE                         TARGET NAME                      CATEGORY                        KEYWORDS
-------------  ----------  -----------------------------  --------------------  -----------  --------------------------------------------------  -------------------------------  ------------------------------  --------------------------------
7306:11:1      FINEGUIDE   PRIME TARGETED FIXED           2026-08-17T04:17:38Z  00/00:32:00  MIRI Medium Resolution Spectroscopy                 IRAS14562-5406-SKY               Star                            Circumstellar clouds, Circumstellar disks
10678:1:1      FINEGUIDE   PRIME TARGETED FIXED           2026-08-17T08:10:36Z  00/00:39:27  NIRCam Imaging                                      GC_1                             Unidentified                    Infrared sources
                           COORDINATED PARALLEL                                              MIRI Imaging
10678:20:1     FINEGUIDE   PRIME TARGETED FIXED           2026-08-17T08:59:11Z  00/00:39:27  NIRCam Imaging                                      GC_20                            Unidentified                    Infrared sources
                           COORDINATED PARALLEL                                              MIRI Imaging
"""

NOW = datetime.datetime(2026, 8, 16, 12, 0, tzinfo=datetime.timezone.utc)


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def test_a_coordinated_parallel_is_folded_into_its_prime():
    visits = S.parse_report(REPORT)
    assert len(visits) == 3, 'the parallel lines must not become visits'
    gc1 = [v for v in visits if v['visit_id'] == '10678:1:1'][0]
    assert [p['instrument'] for p in gc1['parallels']] == ['MIRI Imaging']


def test_the_visit_id_is_split_into_program_observation_visit():
    gc20 = [v for v in S.parse_report(REPORT)
            if v['visit_id'] == '10678:20:1'][0]
    assert (gc20['program'], gc20['observation'], gc20['visit']) \
        == ('10678', '20', '1')


def test_a_keywords_column_with_commas_does_not_shift_the_others():
    """The failure of every delimiter-guessing version: `Circumstellar clouds,
    Circumstellar disks` became two columns and pushed TARGET into CATEGORY."""
    row = [v for v in S.parse_report(REPORT) if v['program'] == '7306'][0]
    assert row['target'] == 'IRAS14562-5406-SKY'
    assert row['category'] == 'Star'
    assert row['keywords'].startswith('Circumstellar clouds,')


def test_a_target_name_is_read_whole():
    gc1 = [v for v in S.parse_report(REPORT) if v['visit_id'] == '10678:1:1'][0]
    assert gc1['target'] == 'GC_1'
    assert gc1['instrument'] == 'NIRCam Imaging'


def test_a_report_with_no_column_ruler_is_refused():
    with pytest.raises(S.ScheduleFormatError):
        S.parse_report('VISIT ID  START\n1:1:1  now\n')


@pytest.mark.parametrize('text,seconds', [
    ('00/00:39:27', 2367), ('00/01:50:40', 6640), ('01/00:00:01', 86401),
    ('', None), ('garbage', None), ('00/1:2', None)])
def test_duration_parsing(text, seconds):
    assert S.duration_seconds(text) == seconds


def test_a_row_with_no_start_time_has_no_place_on_a_timeline():
    assert S.start_datetime({'start': ''}) is None
    assert S.start_datetime({'start': '2026-08-17T08:10:36'}) is None  # no Z
    assert S.start_datetime({'start': '2026-08-17T08:10:36Z'}).hour == 8


# ---------------------------------------------------------------------------
# which weekly reports to read
# ---------------------------------------------------------------------------

def test_a_reissued_week_keeps_only_the_newest_generation():
    """A week is republished when the plan changes.  Reading both shows the same
    pointing twice at two different times with nothing to say which is live."""
    html = ('<a href="/x/_documents/20260817_report_20260814.txt">a</a>'
            '<a href="/x/_documents/20260817_report_20260816.txt">b</a>'
            '<a href="/x/_documents/20260810_report_20260806.txt">c</a>')
    urls = S.report_urls(html, base='https://www.stsci.edu')
    assert [u[0] for u in urls] == ['20260817', '20260810']
    assert urls[0][1] == '20260816'
    assert urls[0][2].startswith('https://www.stsci.edu/x/_documents/')


def test_the_index_is_not_required_to_be_absolute():
    urls = S.report_urls(
        '<a href="https://elsewhere/_documents/20260817_report_20260814.txt">a</a>')
    assert urls[0][2].startswith('https://elsewhere/')


# ---------------------------------------------------------------------------
# offline behaviour -- this runs on a compute node under scrontab
# ---------------------------------------------------------------------------

def test_offline_with_no_cache_reports_itself_rather_than_raising(tmp_path):
    """A monitor that disappears when the network blinks teaches people to
    distrust the parts of it that are still right."""
    out = S.load(str(tmp_path), offline=True)
    assert out['visits'] == []
    assert out['stale'] is True
    assert 'no schedule index' in out['note']


def test_offline_reads_the_cache_and_says_it_is_stale(tmp_path):
    cache = tmp_path / S.CACHE_SUBDIR
    cache.mkdir()
    (cache / 'index.html').write_text(
        '<a href="/d/_documents/20260817_report_20260814.txt">x</a>')
    (cache / '20260817_report_20260814.txt').write_text(REPORT)
    out = S.load(str(tmp_path), program='10678', offline=True)
    assert [v['visit_id'] for v in out['visits']] == ['10678:1:1', '10678:20:1']
    assert out['stale'] is True
    assert out['weeks'][0]['n_program'] == 2
    assert out['weeks'][0]['n_visits'] == 3


def test_visits_are_sorted_by_start_time(tmp_path):
    cache = tmp_path / S.CACHE_SUBDIR
    cache.mkdir()
    (cache / 'index.html').write_text(
        '<a href="/d/_documents/20260817_report_20260814.txt">x</a>')
    (cache / '20260817_report_20260814.txt').write_text(REPORT)
    out = S.load(str(tmp_path), program='10678', offline=True)
    starts = [v['start'] for v in out['visits']]
    assert starts == sorted(starts)


def test_write_and_read_back_the_json(tmp_path):
    sched = {'program': '10678', 'visits': [], 'weeks': [], 'stale': False}
    path = S.write_json(str(tmp_path), sched)
    assert os.path.basename(path) == S.SCHEDULE_JSON
    assert S.load_json(path)['program'] == '10678'
    assert S.load_json(str(tmp_path / 'nope.json')) is None


# ---------------------------------------------------------------------------
# the summary and the panel
# ---------------------------------------------------------------------------

def _sched():
    return {'program': '10678', 'stale': False, 'fetched': '2026-08-16T12:00:00Z',
            'weeks': [{'week': '20260817', 'generated': '20260814',
                       'url': 'u', 'n_visits': 3, 'n_program': 2}],
            'visits': [v for v in S.parse_report(REPORT) if v['program'] == '10678']}


def test_the_summary_counts_upcoming_against_a_given_now():
    s = S.summarize(_sched(), now=NOW)
    assert (s['n_visits'], s['n_upcoming'], s['n_past']) == (2, 2, 0)
    assert s['next']['visit_id'] == '10678:1:1'
    assert s['n_targets'] == 2
    assert s['hours'] == pytest.approx(1.3, abs=0.05)


def test_the_summary_moves_a_visit_to_past_once_now_is_after_it():
    # 08:30 is after GC_1 (08:10:36) and before GC_20 (08:59:11).
    between = datetime.datetime(2026, 8, 17, 8, 30, tzinfo=datetime.timezone.utc)
    s = S.summarize(_sched(), now=between)
    assert (s['n_past'], s['n_upcoming']) == (1, 1)
    assert s['next']['visit_id'] == '10678:20:1'


def test_the_panel_leads_with_the_time_to_the_next_visit():
    out = SS.section(_sched(), entries=(), now=NOW)
    assert 'in 20 h 10 m' in out
    assert '10678:1:1' in out and 'GC_1' in out


def test_a_past_visit_with_no_data_is_NOT_SEEN_not_missing():
    """The page cannot establish a fault from a plan: the plan may have changed,
    the data may be in its proprietary window, or it may just not be downloaded."""
    later = datetime.datetime(2026, 8, 17, 9, 0, tzinfo=datetime.timezone.utc)
    out = SS.section(_sched(), entries=(), now=later)
    assert 'is-wait">not seen<' in out
    # The prose explains the vocabulary and so contains the word; no BADGE may.
    assert 'is-wait">missing<' not in out
    assert 'is-fail' not in out


def test_a_past_visit_whose_observation_the_scan_knows_reads_as_on_disk():
    later = datetime.datetime(2026, 8, 17, 9, 0, tzinfo=datetime.timezone.utc)
    entries = [{'run': {'proposal': '10678', 'obsid': 'o001'}}]
    out = SS.section(_sched(), entries=entries, now=later)
    assert 'is-ok">on disk<' in out


def test_another_programs_observation_number_does_not_count_as_on_disk():
    """`o001` exists for half the archive.  Matching it without the program
    would mark 10678's first visit delivered because the Brick has an o001."""
    later = datetime.datetime(2026, 8, 17, 9, 0, tzinfo=datetime.timezone.utc)
    entries = [{'run': {'proposal': '2221', 'obsid': 'o001'}}]
    out = SS.section(_sched(), entries=entries, now=later)
    assert 'is-ok">on disk<' not in out


def test_the_panel_says_the_schedule_is_a_plan():
    out = SS.section(_sched(), entries=(), now=NOW)
    assert 'plan' in out.lower()
    assert 'differ from those scheduled' in out


def test_a_stale_fetch_is_declared_on_the_page():
    sched = dict(_sched(), stale=True, note='could not reach the index')
    out = SS.section(sched, entries=(), now=NOW)
    assert 'stale' in out
    assert 'could not reach the index' in out


def test_no_visits_still_renders_a_section_saying_so():
    out = SS.section({'program': '10678', 'visits': [], 'weeks': [],
                      'stale': False, 'fetched': 'x'}, now=NOW)
    assert 'id="schedule"' in out
    assert 'No visits for program 10678' in out


def test_no_schedule_at_all_renders_nothing():
    assert SS.section(None) == ''


def test_the_upcoming_list_is_capped_and_says_how_many_it_hid():
    many = dict(_sched())
    base = many['visits'][0]
    many['visits'] = [dict(base, visit_id=f'10678:{i}:1', observation=str(i),
                           start=f'2026-08-{17 + i // 24:02d}T{i % 24:02d}:00:00Z')
                      for i in range(SS.UPCOMING_SHOWN + 5)]
    out = SS.section(many, entries=(), now=NOW)
    assert 'further scheduled visit(s) not shown' in out


def test_html_in_a_target_name_cannot_reach_the_page():
    sched = _sched()
    sched['visits'][0]['target'] = '<script>alert(1)</script>'
    out = SS.section(sched, entries=(), now=NOW)
    assert '<script>alert(1)</script>' not in out
    assert '&lt;script&gt;' in out
