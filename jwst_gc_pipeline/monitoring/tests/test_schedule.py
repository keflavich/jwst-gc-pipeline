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


def test_offline_reads_the_cache_and_says_it_is_stale(tmp_path, monkeypatch):
    """An index too old to be current, which offline mode cannot refresh.

    A cache a few minutes old is the normal second-pass path and is NOT a
    degradation; only one that could not be refreshed is.
    """
    cache = tmp_path / S.CACHE_SUBDIR
    cache.mkdir()
    idx = cache / 'index.html'
    idx.write_text('<a href="/d/_documents/20260817_report_20260814.txt">x</a>')
    (cache / '20260817_report_20260814.txt').write_text(REPORT)
    monkeypatch.setattr(S, '_now_s',
                        lambda: os.path.getmtime(idx) + S.INDEX_MAX_AGE_S + 1)
    out = S.load(str(tmp_path), program='10678', offline=True)
    assert [v['visit_id'] for v in out['visits']] == ['10678:1:1', '10678:20:1']
    assert out['stale'] is True
    assert 'could not be refreshed' in out['note']
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
    assert 'degraded' in out
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



# ---------------------------------------------------------------------------
# degradation must never become a claim about the survey (PR #404 review)
# ---------------------------------------------------------------------------

HEADER = REPORT.splitlines()[2]
RULER = REPORT.splitlines()[3]


def _continuation(visit_type, instrument='', target=''):
    """A continuation line with its cells in the columns the ruler defines.

    Hand-spacing a fixture is how a test ends up asserting about a column the
    parser never reads.
    """
    spans = S._columns(RULER)
    line = [' '] * (spans[-1][0] + 40)
    for idx, value in ((2, visit_type), (5, instrument), (6, target)):
        if value:
            line[spans[idx][0]:spans[idx][0] + len(value)] = value
    return ''.join(line).rstrip()


def test_an_empty_middle_column_is_read_as_empty(tmp_path):
    """The discriminator the ruler actually buys.

    A two-or-more-space split reproduces the real file's numbers exactly,
    because KEYWORDS is comma-and-SINGLE-space separated -- so the test that
    claimed to pin the ruler was passed by the mutation.  What a delimiter
    split cannot do is an EMPTY intermediate column: the report's
    `ATTACHED TO PRIME` rows have a blank DURATION, and splitting puts the
    instrument into the duration cell.
    """
    text = '\n'.join([
        'Visit Information', '', HEADER, RULER,
        '12551:41:1     NONE        PARALLEL DARK CALIBRATION      '
        '^ATTACHED TO PRIME^                NIRSpec Dark',
    ])
    row = S.parse_report(text)[0]
    assert row['duration'] == ''
    assert row['instrument'] == 'NIRSpec Dark'
    assert row['start'] == '^ATTACHED TO PRIME^'


def test_a_decoy_ruler_in_the_preamble_does_not_zero_the_panel():
    """One horizontal rule added to STScI's preamble used to zero the panel
    with no exception and no trace, and the page then said the programme was
    not on the schedule."""
    text = '\n'.join(['Visit Information', '--------------------', '',
                      HEADER, RULER,
                      REPORT.splitlines()[5], REPORT.splitlines()[6]])
    visits = S.parse_report(text)
    assert [v['visit_id'] for v in visits] == ['10678:1:1']
    assert visits[0]['target'] == 'GC_1'


def test_a_ruler_with_no_VISIT_ID_header_above_it_is_refused():
    with pytest.raises(S.ScheduleFormatError):
        S.parse_report('preamble\n------  -----\nx  y\n')


def test_a_continuation_line_that_is_not_a_parallel_is_not_one():
    """Line 61 of the real report is a second TARGET for 12782:2:1, not a
    second instrument on the pointing."""
    text = '\n'.join([
        'Visit Information', '', HEADER, RULER,
        REPORT.splitlines()[5],
        _continuation('TARGET ACQUISITION', target='TA_RJ0018'),
    ])
    visit = S.parse_report(text)[0]
    assert visit['parallels'] == []
    assert visit.get('extra_targets') == ['TA_RJ0018']


def test_an_unreadable_cached_report_is_declared_rather_than_read_as_absence(
        tmp_path):
    cache = tmp_path / S.CACHE_SUBDIR
    cache.mkdir()
    (cache / 'index.html').write_text(
        '<a href="/d/_documents/20260817_report_20260814.txt">x</a>')
    (cache / '20260817_report_20260814.txt').write_text('')     # zero-byte
    out = S.load(str(tmp_path), program='10678', offline=True)
    assert out['visits'] == []
    assert out['stale'] is True
    assert out['note'], 'a zero-byte cached report left no trace at all'


def test_a_cached_report_that_no_longer_parses_is_declared(tmp_path):
    cache = tmp_path / S.CACHE_SUBDIR
    cache.mkdir()
    (cache / 'index.html').write_text(
        '<a href="/d/_documents/20260817_report_20260814.txt">x</a>')
    (cache / '20260817_report_20260814.txt').write_text('truncated preamble\n')
    out = S.load(str(tmp_path), program='10678', offline=True)
    assert out['stale'] is True and out['note']


def test_a_degraded_read_does_not_claim_the_programme_is_unscheduled():
    """The one claim the panel must never make, in its other form."""
    out = SS.section({'program': '10678', 'visits': [], 'weeks': [],
                      'stale': True, 'note': 'could not reach the index',
                      'fetched': None}, now=NOW)
    assert 'says nothing about whether it is on it' in out
    assert 'not on the published schedule yet' not in out


def test_a_clean_read_with_no_visits_still_says_so_plainly():
    out = SS.section({'program': '99999', 'visits': [], 'weeks': [],
                      'stale': False, 'note': '', 'fetched': 'x'}, now=NOW)
    assert 'not on the published schedule yet' in out


def test_no_local_path_or_username_reaches_the_note(tmp_path):
    """The note is rendered onto a PUBLIC page."""
    cache = tmp_path / S.CACHE_SUBDIR
    cache.mkdir()
    (cache / 'index.html').write_text('<a href="/d/_documents/'
                                      '20260817_report_20260814.txt">x</a>')
    (cache / '20260817_report_20260814.txt').write_text('')
    out = S.load(str(tmp_path), program='10678', offline=True)
    assert str(tmp_path) not in out['note']
    assert '/' not in out['note']


def test_the_cache_write_is_atomic(tmp_path):
    """A partial file used to be cached permanently, and by the note bug that
    loss was invisible."""
    target = tmp_path / 'r.txt'
    assert S._write_cache(str(target), 'content')
    assert target.read_text() == 'content'
    assert not list(tmp_path.glob('*.tmp'))


def test_http_layer_failures_are_caught_by_the_fetch_tuple():
    """`IncompleteRead` is a subclass of neither OSError nor ValueError, so a
    truncated HTTPS response propagated out and killed the monitor build."""
    import http.client
    assert issubclass(http.client.IncompleteRead, S.FETCH_ERRORS)
    assert issubclass(http.client.HTTPException, S.FETCH_ERRORS)


def test_ties_sort_numerically_not_lexicographically(tmp_path):
    sched = _sched()
    for v, vid in zip(sched['visits'], ['10678:10:1', '10678:2:1']):
        v['visit_id'], v['start'] = vid, '2026-08-17T08:10:36Z'
    cache = tmp_path / S.CACHE_SUBDIR
    cache.mkdir()
    (cache / 'index.html').write_text('<a href="/d/_documents/'
                                      '20260817_report_20260814.txt">x</a>')
    (cache / '20260817_report_20260814.txt').write_text(REPORT)
    out = S.load(str(tmp_path), program='10678', offline=True)
    assert out['visits'][0]['visit_id'] == '10678:1:1'


def test_the_panel_names_the_survey_and_expands_its_vocabulary():
    """This page is public; `visit`, `tile` and `+ MIRI Imaging` shipped bare
    and the page never named the survey."""
    out = SS.section(_sched(), entries=(), now=NOW)
    assert 'Treasury' in out
    assert 'coordinated parallel' in out
    assert 'tiles' in out


def test_a_fresh_cached_index_is_reused_without_refetching(tmp_path, monkeypatch):
    """One refresh runs the generator twice (field pass, cutout pass), so
    without this every hourly refresh hit stsci.edu twice for a weekly page."""
    cache = tmp_path / S.CACHE_SUBDIR
    cache.mkdir()
    (cache / 'index.html').write_text(
        '<a href="/d/_documents/20260817_report_20260814.txt">x</a>')
    (cache / '20260817_report_20260814.txt').write_text(REPORT)
    calls = []
    monkeypatch.setattr(S, '_get', lambda url, timeout=30: calls.append(url))
    # min_rows=1: this fixture is a two-visit stub, well under the production
    # plausibility floor.  The floor itself is exercised in both directions by
    # the two tests at the end of this file.
    out = S.load(str(tmp_path), program='10678', min_rows=1)
    assert calls == [], 'a fresh cached index must not be re-fetched'
    assert out['stale'] is False, 'reusing a fresh cache is not a degradation'
    assert len(out['visits']) == 2


def test_a_stale_cached_index_is_refetched(tmp_path, monkeypatch):
    cache = tmp_path / S.CACHE_SUBDIR
    cache.mkdir()
    idx = cache / 'index.html'
    idx.write_text('<a href="/d/_documents/20260817_report_20260814.txt">x</a>')
    (cache / '20260817_report_20260814.txt').write_text(REPORT)
    monkeypatch.setattr(S, '_now_s',
                        lambda: os.path.getmtime(idx) + S.INDEX_MAX_AGE_S + 1)
    calls = []

    def _fake(url, timeout=30):
        calls.append(url)
        return idx.read_text()

    monkeypatch.setattr(S, '_get', _fake)
    S.load(str(tmp_path), program='10678')
    assert calls and calls[0] == S.SCHEDULE_INDEX


# ---------------------------------------------------------------------------
# The six mutations that survived review round N, plus the two new anomalies.
# Each of these was written against a specific mutation that left 208 green.
# ---------------------------------------------------------------------------

def test_an_index_with_no_report_LINKS_does_not_claim_the_program_is_unscheduled(
        tmp_path, monkeypatch):
    """The forbidden sentence, reachable with NO local failure.

    An STScI 200 that is a maintenance page, a changed href pattern, or a
    truncated cached index all give a non-empty index that yields zero report
    URLs.  Nothing else fires, and the page then says the program "is not on the
    published schedule yet" -- a claim about the PROGRAM made on the strength of
    a page we failed to read.
    """
    cache = tmp_path / S.CACHE_SUBDIR
    cache.mkdir()
    (cache / 'index.html').write_text(
        '<html><body><h1>Scheduled maintenance</h1></body></html>')
    monkeypatch.setattr(S, '_get', lambda url, timeout=30: '')
    out = S.load(str(tmp_path), program='10678')
    assert out['stale'] is True, out
    assert out['note'], 'a zero-link index must say so'
    assert 'no weekly report links' in out['note'], out['note']


def test_a_TRUNCATED_report_is_refused_rather_than_undercounted(tmp_path,
                                                               monkeypatch):
    """A body truncated in transit parses cleanly -- ruler and leading rows
    intact -- and published a confident undercount with stale=False.  Measured
    on the real 20260817 report: 20% of it gives "5 visits, 3.3 h"; 90% gives
    "30 visits, 19.7 h".  Both silent."""
    cache = tmp_path / S.CACHE_SUBDIR
    cache.mkdir()
    (cache / 'index.html').write_text(
        '<a href="/d/_documents/20260817_report_20260814.txt">x</a>')
    (cache / '20260817_report_20260814.txt').write_text(REPORT)
    monkeypatch.setattr(S, '_get', lambda url, timeout=30: '')
    out = S.load(str(tmp_path), program='10678', min_rows=50)
    assert out['stale'] is True, out
    assert 'plausibility floor' in (out['note'] or ''), out['note']


def test_the_plausibility_floor_does_not_fire_on_a_full_report(tmp_path,
                                                              monkeypatch):
    """The other direction: a report at or above the floor must stay clean, or
    the floor is just an unconditional refusal."""
    cache = tmp_path / S.CACHE_SUBDIR
    cache.mkdir()
    (cache / 'index.html').write_text(
        '<a href="/d/_documents/20260817_report_20260814.txt">x</a>')
    (cache / '20260817_report_20260814.txt').write_text(REPORT)
    monkeypatch.setattr(S, '_get', lambda url, timeout=30: '')
    out = S.load(str(tmp_path), program='10678', min_rows=1)
    assert out['stale'] is False, out
    assert not out['note'], out['note']


def test_the_cache_write_survives_a_SECOND_WRITER(tmp_path):
    """`path + '.tmp'` is atomic against a kill and NOT against a second
    process: both writers share the temp inode, so the loser's os.replace moves
    it into place while the winner still holds a descriptor and the winner's
    remaining bytes land INSIDE the published file.  Measured on the fixed name:
    a 100000-byte cache reading BBBBB...GGGGG, permanently cached.

    Asserting temp-then-replace is not enough -- the previous test asserted only
    that the content landed and no `.tmp` remained, which a plain
    `open(path, 'w')` also satisfies.
    """
    import multiprocessing
    target = str(tmp_path / 'cache.txt')

    def _writer(ch):
        S._write_cache(target, ch * 50000)

    procs = [multiprocessing.Process(target=_writer, args=(c,))
             for c in ('B', 'G')]
    for p in procs:
        p.start()
    for p in procs:
        p.join(30)
    got = open(target).read()
    assert len(got) == 50000, f'interleaved write: {len(got)} bytes'
    assert set(got) in ({'B'}, {'G'}), (
        f'published file mixes two writers: head {got[:5]} tail {got[-5:]}')
    assert not [f for f in os.listdir(tmp_path) if f.endswith('.tmp')]


def test_the_cache_write_uses_a_UNIQUE_temp_name(tmp_path):
    """The property the interleaving test rests on, asserted directly so it
    cannot regress into a fixed name that merely happens to pass."""
    import inspect
    src = inspect.getsource(S._write_cache)
    assert 'mkstemp' in src, (
        "a fixed `path + '.tmp'` is shared by concurrent writers")
    assert "f'{path}.tmp'" not in src


# ---------------------------------------------------------------------------
# truncation detection, round 3: a ROW COUNT cannot detect truncation
# ---------------------------------------------------------------------------

def _big_report(n=60):
    """A report with n 10678 visits, above MIN_PLAUSIBLE_ROWS, ending in \\n --
    the shape of a real weekly file."""
    head = REPORT.split('\n')[:4]
    rows = []
    for i in range(1, n + 1):
        rows.append(
            f'10678:{i}:1      FINEGUIDE   PRIME TARGETED FIXED           '
            f'2026-08-17T{i % 24:02d}:10:36Z  00/00:39:27  NIRCam Imaging'
            f'                                      GC_{i}'
            + ' ' * max(1, 33 - len(f'GC_{i}'))
            + 'Unidentified                    Infrared sources')
    return '\n'.join(head + rows) + '\n'


def _staged(tmp_path, body, generated='20260814'):
    cache = tmp_path / S.CACHE_SUBDIR
    cache.mkdir(exist_ok=True)
    (cache / 'index.html').write_text(
        f'<a href="/d/_documents/20260817_report_{generated}.txt">x</a>')
    (cache / f'20260817_report_{generated}.txt').write_text(body)
    return cache


def test_a_report_cut_mid_line_is_refused_at_the_PRODUCTION_default(
        tmp_path, monkeypatch):
    """The row floor stopped firing at 17.6% of the real file, so 20%, 50% and
    90% truncations all published confident undercounts at the production
    default.  A truncation lands mid-line; the discriminator is the final
    newline, which all eight live reports have."""
    full = _big_report()
    for frac in (0.2, 0.5, 0.9):
        cut = full[:int(len(full) * frac)]
        assert not cut.endswith('\n'), 'fixture must model a mid-line cut'
        _staged(tmp_path, cut)
        monkeypatch.setattr(S, '_get', lambda url, timeout=30: '')
        out = S.load(str(tmp_path), program='10678')
        assert out['stale'] is True, (frac, out)
        assert 'does not end in a newline' in (out['note'] or ''), out['note']


def test_a_full_report_is_clean_at_the_PRODUCTION_default(tmp_path, monkeypatch):
    """Both floor tests passed `min_rows=` explicitly, so setting
    MIN_PLAUSIBLE_ROWS = 0 restored the pre-fix behaviour with the suite green.
    This one runs `load()` the way `report.py:169` does."""
    _staged(tmp_path, _big_report())
    monkeypatch.setattr(S, '_get', lambda url, timeout=30: '')
    out = S.load(str(tmp_path), program='10678')
    assert out['stale'] is False, out
    assert not out['note'], out['note']
    assert len(out['visits']) == 60


def test_a_SHORT_cached_report_is_re_fetched_and_self_heals(tmp_path, monkeypatch):
    """The unparseable-cache branch re-fetches; a cache that parses cleanly but
    SHORT did not, and the cache filename is keyed on week+generation, so the
    only escape was STScI reissuing the week.  Every hour after one killed write
    the public page repeated the same wrong number."""
    full = _big_report()
    _staged(tmp_path, full[:int(len(full) * 0.5)])
    monkeypatch.setattr(S, '_get', lambda url, timeout=30: full)
    out = S.load(str(tmp_path), program='10678')
    assert out['stale'] is False, out
    assert len(out['visits']) == 60, 'the re-fetch must replace the short body'


def test_a_short_cached_report_is_NOT_re_fetched_offline(tmp_path, monkeypatch):
    """Offline is a promise not to touch the network, and it outranks the
    repair."""
    full = _big_report()
    _staged(tmp_path, full[:int(len(full) * 0.5)])
    calls = []
    monkeypatch.setattr(S, '_get',
                        lambda url, timeout=30: calls.append(url) or full)
    out = S.load(str(tmp_path), program='10678', offline=True)
    assert calls == [], 'offline must not fetch'
    assert out['stale'] is True, out


def test_the_cache_write_uses_a_different_temp_name_every_time(tmp_path,
                                                              monkeypatch):
    """Behavioural, not textual.  The previous pin read `inspect.getsource` and
    matched the leading COMMENT, so replacing the whole body with
    `open(path, 'w')` while keeping the comment left the suite green."""
    seen = []
    real_replace = os.replace

    def _spy(src, dst):
        seen.append(src)
        return real_replace(src, dst)

    monkeypatch.setattr(os, 'replace', _spy)
    target = str(tmp_path / 'x.txt')
    assert S._write_cache(target, 'a')
    assert S._write_cache(target, 'b')
    assert len(seen) == 2, 'the write must go through a temp file and os.replace'
    assert seen[0] != seen[1], 'a FIXED temp name races a second writer'
    assert target + '.tmp' not in seen
    with open(target) as fh:
        assert fh.read() == 'b'


# ---------------------------------------------------------------------------
# the mutations that survived round 2 and were still surviving at round 3
# ---------------------------------------------------------------------------

def test_visits_sort_NUMERICALLY_within_a_start_time(tmp_path, monkeypatch):
    """Lexicographic on the visit id puts 10678:10:1 before 10678:2:1, so the
    upcoming list reads out of order for any program past nine observations --
    10678 plans 1668.  Exercised through `load()`, which is what orders them."""
    def _row(obs, start):
        name = f'GC_{obs}'
        return (f'10678:{obs}:1'.ljust(15) + 'FINEGUIDE   PRIME TARGETED FIXED'
                + ' ' * 11 + f'{start}  00/00:39:27  NIRCam Imaging'
                + ' ' * 38 + name + ' ' * max(1, 33 - len(name))
                + 'Unidentified                    Infrared sources')
    same = '2026-08-17T08:10:36Z'
    head = REPORT.split('\n')[:4]
    body = '\n'.join(head + [_row(o, same) for o in (20, 2, 10, 1)]
                     + [_row(o, '2026-08-18T08:10:36Z')
                        for o in range(30, 30 + 60)]) + '\n'
    _staged(tmp_path, body)
    monkeypatch.setattr(S, '_get', lambda url, timeout=30: '')
    out = S.load(str(tmp_path), program='10678')
    tied = [v['visit_id'] for v in out['visits'] if v['start'] == same]
    assert tied == ['10678:1:1', '10678:2:1', '10678:10:1', '10678:20:1'], tied


def test_a_continuation_line_that_is_not_a_PARALLEL_is_not_one(tmp_path):
    """A blank-visit-id continuation can also be a second TARGET for the same
    visit (line 61 of the 2026-08-17 report, TA_RJ0018 under 12782:2:1).
    Dropping the visit-type check puts a second instrument on a pointing that
    has one."""
    body = (REPORT.rstrip('\n') + '\n'
            + '                           SECOND TARGET                    '
              '                          NIRCam Imaging'
              '                                      TA_RJ0018\n')
    gc20 = [v for v in S.parse_report(body) if v['visit_id'] == '10678:20:1'][0]
    assert [p['instrument'] for p in gc20['parallels']] == ['MIRI Imaging'], (
        'a non-PARALLEL continuation must not become a parallel')


def test_the_unparseable_cached_report_is_re_fetched(tmp_path, monkeypatch):
    """A cached report that no longer parses is the killed-write case, and the
    cache is authoritative forever.  Without the re-fetch one bad write loses a
    week permanently."""
    full = _big_report()
    _staged(tmp_path, 'this is not a schedule report at all\n')
    calls = []

    def _fake(url, timeout=30):
        calls.append(url)
        return full

    monkeypatch.setattr(S, '_get', _fake)
    out = S.load(str(tmp_path), program='10678')
    assert calls, 'an unparseable cached report must be re-fetched'
    assert out['stale'] is False, out
    assert len(out['visits']) == 60


def test_a_degraded_note_never_carries_a_LOCAL_PATH_or_a_raw_exception(
        tmp_path, monkeypatch):
    """The note is published on a public page.  A filesystem path or a raw
    exception repr leaks the layout of the machine and reads as a fault in the
    schedule rather than in our read of it."""
    cache = tmp_path / S.CACHE_SUBDIR
    cache.mkdir()
    (cache / 'index.html').write_text(
        '<a href="/d/_documents/20260817_report_20260814.txt">x</a>')
    (cache / '20260817_report_20260814.txt').write_text('not a report\n')
    monkeypatch.setattr(S, '_get', lambda url, timeout=30: '')
    note = S.load(str(tmp_path), program='10678')['note'] or ''
    assert str(tmp_path) not in note, note
    assert 'Error' not in note and 'Traceback' not in note, note
