"""When STScI plans to observe the survey, read from the published schedule.

The monitor has always been able to say what is ON DISK.  Until a program has
delivered anything that is the whole story, and for 10678 -- the Treasury, 1668
planned pointings -- it said "none delivered yet" for months while the answer
people actually wanted was *when does it start*.

That answer is published: STScI posts a weekly observing schedule as a plain
text report, one row per visit, at

    https://www.stsci.edu/jwst/science-execution/observing-schedules

and the week beginning 2026-08-17 contains 10678's first visits::

    10678:1:1      FINEGUIDE   PRIME TARGETED FIXED   2026-08-17T08:10:36Z  00/00:39:27  NIRCam Imaging   GC_1   ...
                               COORDINATED PARALLEL                                      MIRI Imaging

Two things about that row shape drive this module:

* the **continuation line** (blank visit id) is the coordinated parallel, not a
  separate visit.  Counting it as one would double every 10678 pointing and put
  a MIRI visit on the page with no start time;
* the schedule is a **plan**.  STScI's own note: "it is possible that the actual
  executed observations will differ from those planned."  So nothing here is
  evidence that an observation happened -- that stays with the on-disk scan and
  MAST.  These are intentions with timestamps, and the page has to say so.

Parsing is by the report's own **column ruler** (the row of dashes under the
header) rather than by splitting on runs of whitespace.  Target names and
keywords contain single spaces, and the KEYWORDS column contains commas and
spaces both; every delimiter-guessing version of this split ``Circumstellar
clouds, Circumstellar disks`` into new columns and shifted everything right of
it.  The ruler is in the file, so use it.

Network
-------
This runs from a SLURM compute node under ``scrontab``, where outbound HTTPS may
not be available.  So every fetch is **cached to disk and falls back to the
cache**, and a fetch failure is reported as a stale-cache note rather than
raising -- the monitor's own convention: the section says what it could not
reach instead of vanishing.
"""
import datetime
import http.client
import json
import os
import tempfile
import re
import urllib.error
import urllib.request

#: The index page listing every weekly report.
SCHEDULE_INDEX = 'https://www.stsci.edu/jwst/science-execution/observing-schedules'

#: Reports are linked from the index as
#: ``.../_documents/20260817_report_20260814.txt``.  Matched rather than
#: constructed: the first date is the week the schedule STARTS and the second is
#: when it was generated, and only STScI knows the second.
REPORT_HREF_RE = re.compile(
    r'href="([^"]*/_documents/(\d{8})_report_(\d{8})\.txt)"')

#: ``10678:20:1`` -- program, observation, visit.
VISIT_ID_RE = re.compile(r'^(\d+):(\d+):(\d+)$')

#: The GC Treasury.  Everything here works for any program; this is the default
#: because it is the one the monitor exists to watch.
DEFAULT_PROGRAM = '10678'

#: Where reports are cached, relative to the monitor outdir.
CACHE_SUBDIR = 'schedule'

#: What the renderer reads.
SCHEDULE_JSON = 'schedule.json'

USER_AGENT = 'jwst-gc-pipeline monitor (schedule reader)'

#: How long a cached index stays authoritative without re-fetching.  One
#: refresh runs the generator TWICE -- the field pass and the probe-cutout pass
#: -- so without this every hourly refresh hit stsci.edu twice for a page that
#: changes weekly.  Reusing a fresh cache is not a degradation and does not set
#: ``stale``.
INDEX_MAX_AGE_S = 1800

#: Fewest visits a real published weekly report can carry.  Below this the body
#: was almost certainly truncated: the live 20260817 report has 100+ rows, and a
#: 2% truncation still parses to 1 row without raising.
MIN_PLAUSIBLE_ROWS = 20

#: What a failed fetch can raise.  ``http.client.HTTPException`` is in the list
#: because it is a subclass of NEITHER ``OSError`` nor ``ValueError``: a
#: truncated HTTPS response (``IncompleteRead``) propagated out of ``load``,
#: through ``write_report``, and killed the monitor build -- and
#: ``refresh_monitor.sh`` only fails the job above rc 1, so the crash was
#: indistinguishable from the normal "the archive has failing runs" exit and the
#: deploy shipped whatever stale pages were on disk.
FETCH_ERRORS = (urllib.error.URLError, http.client.HTTPException,
                OSError, ValueError)


class ScheduleFormatError(ValueError):
    """The report does not have the header/ruler this parser needs."""


def _now_s():
    """Wall clock, as its own function so a test can move it."""
    return datetime.datetime.now(datetime.timezone.utc).timestamp()


def _write_cache(path, text):
    """Cache ``text`` at ``path`` atomically.  True when it landed.

    ``open(path, 'w')`` truncates and then fills, so a kill or a full
    filesystem mid-write leaves a partial or zero-byte file -- which the reader
    then treated as authoritative forever.
    """
    # mkstemp, not a fixed `path + '.tmp'`: the fixed name is atomic against a
    # KILL and not against a SECOND WRITER.  Two concurrent refreshes share the
    # temp inode, so the loser's os.replace moves it into place while the winner
    # still holds a descriptor on it and its remaining bytes land INSIDE the
    # published file -- measured: a 100000-byte cache reading BBBBB...GGGGG,
    # permanently cached, with the loser also raising FileNotFoundError.
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or '.',
                               prefix=os.path.basename(path) + '.',
                               suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        return True
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return False


def _get(url, timeout=30):
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return fh.read().decode('utf-8', 'replace')


def _columns(ruler):
    """``[(name_slice_start, stop)]`` from the row of dashes under the header.

    ``stop`` is None for the last column, which runs to the end of the line --
    KEYWORDS is comma-separated free text and is not padded.
    """
    spans, start = [], None
    for i, ch in enumerate(ruler):
        if ch == '-' and start is None:
            start = i
        elif ch != '-' and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(ruler)))
    if not spans:
        raise ScheduleFormatError('no column ruler (row of dashes) in report')
    return [(a, b) for a, b in spans[:-1]] + [(spans[-1][0], None)]


def parse_report(text):
    """Every visit in one weekly report, parallels folded into their prime.

    Returns ``[{'visit_id', 'program', 'observation', 'visit', 'pcs_mode',
    'visit_type', 'start', 'duration', 'instrument', 'target', 'category',
    'keywords', 'parallels': [...]}]``.
    """
    lines = text.splitlines()
    ruler_at = None
    for i, line in enumerate(lines):
        if not line.strip() or set(line.strip()) != {'-', ' '}:
            continue
        # The ruler is the one UNDER THE HEADER, not the first row of dashes in
        # the file.  Taking the first meant one horizontal rule added to STScI's
        # preamble would zero the panel with no exception and no trace -- and
        # the page then states, in prose, that the programme is not on the
        # schedule.  That is a claim about the survey manufactured from a local
        # parse failure.
        if i and 'VISIT ID' in lines[i - 1].upper():
            ruler_at = i
            break
    if ruler_at is None:
        raise ScheduleFormatError(
            'no column ruler under a "VISIT ID" header line')
    spans = _columns(lines[ruler_at])

    def cell(line, idx):
        if idx >= len(spans):
            return ''
        a, b = spans[idx]
        return (line[a:b] if b is not None else line[a:]).strip()

    visits, current = [], None
    for line in lines[ruler_at + 1:]:
        if not line.strip():
            continue
        vid = cell(line, 0)
        if not vid:
            # A continuation line: the coordinated parallel of the visit above.
            # It has no start time of its own -- it runs with its prime -- so
            # attaching it rather than emitting it is what keeps the count of
            # pointings equal to the count of visits.
            # ...but only when it says so.  A continuation line can also be a
            # second TARGET for the same visit (line 61 of the 2026-08-17
            # report, `TA_RJ0018` under 12782:2:1); calling that a parallel
            # would put a second instrument on a pointing that has one.
            if current is not None:
                inst = cell(line, 5)
                vtype = cell(line, 2)
                if inst and 'PARALLEL' in vtype.upper():
                    current['parallels'].append(
                        {'visit_type': vtype, 'instrument': inst})
                elif cell(line, 6):
                    current.setdefault('extra_targets', []).append(cell(line, 6))
            continue
        m = VISIT_ID_RE.match(vid)
        if not m:
            continue
        current = {
            'visit_id': vid,
            'program': m.group(1),
            'observation': m.group(2),
            'visit': m.group(3),
            'pcs_mode': cell(line, 1),
            'visit_type': cell(line, 2),
            'start': cell(line, 3),
            'duration': cell(line, 4),
            'instrument': cell(line, 5),
            'target': cell(line, 6),
            'category': cell(line, 7),
            'keywords': cell(line, 8),
            'parallels': [],
        }
        visits.append(current)
    return visits


def duration_seconds(duration):
    """``00/00:39:27`` (days/hh:mm:ss) -> seconds, or None."""
    if not duration:
        return None
    days, _, clock = duration.partition('/')
    parts = clock.split(':')
    if len(parts) != 3:
        return None
    try:
        h, m, s = (int(p) for p in parts)
        return int(days or 0) * 86400 + h * 3600 + m * 60 + s
    except ValueError:
        return None


def start_datetime(visit):
    """The visit's scheduled start as an aware UTC datetime, or None.

    A visit with no start time is a parallel that got past the fold above, or a
    row the report left blank; either way it has no place on a timeline.
    """
    raw = (visit.get('start') or '').strip()
    if not raw.endswith('Z'):
        return None
    try:
        return datetime.datetime.strptime(
            raw, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc)
    except ValueError:
        return None


def report_urls(index_html, base='https://www.stsci.edu'):
    """``[(week_start, generated, url)]`` newest week first, deduplicated.

    A week can be REISSUED -- the same start date with a later generation date
    -- when the plan changes.  Keeping both would show a pointing twice, at two
    different times, with nothing to say which is current.  The newest
    generation for a given week wins.
    """
    best = {}
    for href, week, generated in REPORT_HREF_RE.findall(index_html):
        url = href if href.startswith('http') else base + href
        prev = best.get(week)
        if prev is None or generated > prev[1]:
            best[week] = (week, generated, url)
    return [best[w] for w in sorted(best, reverse=True)]


def load(outdir, program=DEFAULT_PROGRAM, weeks=8, offline=False, min_rows=MIN_PLAUSIBLE_ROWS):
    """Scheduled visits for ``program``, newest weeks first, with provenance.

    Returns ``{'program', 'visits', 'weeks', 'fetched', 'stale', 'note'}``.
    Never raises on a network problem: ``stale`` says the cache was used and
    ``note`` says why, because a monitor that disappears when the network blinks
    teaches people to distrust the parts that are still right.
    """
    cache = os.path.join(outdir, CACHE_SUBDIR)
    os.makedirs(cache, exist_ok=True)
    note, stale = '', False

    index_path = os.path.join(cache, 'index.html')
    index_html = ''
    fresh_cache = False
    if os.path.exists(index_path):
        try:
            age = max(0.0, _now_s() - os.path.getmtime(index_path))
            fresh_cache = age < INDEX_MAX_AGE_S
        except OSError:
            fresh_cache = False
    if fresh_cache:
        try:
            with open(index_path) as fh:
                index_html = fh.read()
        except OSError:
            index_html = ''
    if not index_html and not offline:
        # The fetch and the cache WRITE get separate handlers.  Sharing one
        # reported a local write failure as "could not reach the STScI schedule
        # index (...Permission denied: '/orange/adamginsburg/jwst/monitor/
        # schedule/index.html')" -- and that note is rendered onto a PUBLIC
        # page, so an internal storage layout and the username shipped under a
        # message that also blamed the wrong component.  Notes carry no paths.
        try:
            index_html = _get(SCHEDULE_INDEX)
        except FETCH_ERRORS:
            note = 'could not reach the STScI schedule index'
            stale = True
        if index_html and not _write_cache(index_path, index_html):
            note = note or 'fetched the schedule index but could not cache it'
            stale = True
    if not index_html and os.path.exists(index_path):
        # A cache older than INDEX_MAX_AGE_S that we could not refresh.  THAT
        # is the degradation -- reusing a fresh one a few minutes old is the
        # normal path and is not reported as one.
        try:
            with open(index_path) as fh:
                index_html = fh.read()
        except OSError:
            index_html = ''
        if index_html:
            stale = True
            note = note or 'using a cached schedule index that could not be refreshed'
    if not index_html:
        return {'program': program, 'visits': [], 'weeks': [], 'stale': True,
                'fetched': None,
                'note': note or 'no schedule index available, cached or live'}

    wanted = report_urls(index_html)[:weeks]
    if not wanted:
        # An index that is non-empty but yields NO report URLs is not a local
        # failure, so none of the other anomaly checks fire -- and the page then
        # prints "not on the published schedule yet", which is a claim about the
        # PROGRAM made on the strength of a page we failed to read.  Reachable
        # from an STScI 200 that is a maintenance page, a changed href pattern,
        # or a truncated cached index.
        return {'program': program, 'visits': [], 'weeks': [], 'stale': True,
                'fetched': None,
                'note': note or ('read the schedule index but found no weekly '
                                 'report links in it')}
    visits, weeks_seen = [], []
    for week, generated, url in wanted:
        path = os.path.join(cache, f'{week}_report_{generated}.txt')
        text, from_cache = '', False
        if os.path.exists(path):
            try:
                with open(path) as fh:
                    text = fh.read()
                from_cache = True
            except OSError:
                text = ''
        if not text and not offline:
            try:
                text = _get(url)
            except FETCH_ERRORS:
                note = note or f'could not fetch {os.path.basename(url)}'
                stale = True
            else:
                if not _write_cache(path, text):
                    note = note or 'fetched a weekly report but could not cache it'
                    stale = True
        if not text:
            # An empty or unreadable weekly report is a DEGRADATION, not an
            # absence of visits.  Left silent, the panel says in prose that the
            # programme is not on the schedule -- a claim about the survey
            # manufactured from a local read failure.
            note = note or f'{week}: the weekly report is empty or unreadable'
            stale = True
            continue
        try:
            parsed = parse_report(text)
        except ScheduleFormatError as exc:
            if from_cache and not offline:
                # A cached report that no longer parses is the truncated-write
                # case; the cache was authoritative forever, so one killed
                # write lost a whole week permanently and invisibly.  Re-fetch
                # once before believing it.
                try:
                    text = _get(url)
                except FETCH_ERRORS:
                    text = ''
                if text and _write_cache(path, text):
                    try:
                        parsed = parse_report(text)
                    except ScheduleFormatError as exc2:
                        note = note or f'{week}: {exc2}'
                        stale = True
                        continue
                else:
                    note = note or f'{week}: cached report unreadable ({exc})'
                    stale = True
                    continue
            else:
                note = note or f'{week}: {exc}'
                stale = True
                continue
        if len(parsed) < min_rows:
            # A plausibility FLOOR, not a zero check.  A body truncated in
            # transit parses cleanly -- the ruler and the leading rows are
            # intact -- and publishes a confident undercount with stale=False:
            # measured on the real 20260817 report, 20% of it gives 23 rows and
            # "5 visits, 3.3 h", 90% gives "30 visits, 19.7 h", both silent.
            # A real JWST week schedules far more than a handful of visits, so
            # anything under the floor is a short read rather than a quiet week.
            note = note or (f'{week}: the report parsed to only {len(parsed)} '
                            f'visit(s), below the {min_rows}-row '
                            f'plausibility floor -- probably a truncated read')
            stale = True
        mine = [v for v in parsed if v['program'] == str(program)]
        weeks_seen.append({'week': week, 'generated': generated, 'url': url,
                           'n_visits': len(parsed), 'n_program': len(mine)})
        for v in mine:
            v['week'] = week
        visits.extend(mine)

    def _order(v):
        # Lexicographic on the visit id put 10678:10:1 before 10678:2:1.
        parts = tuple(int(p) if p.isdigit() else 0
                      for p in str(v['visit_id']).split(':'))
        return (v.get('start') or '', parts)

    visits.sort(key=_order)
    return {'program': str(program), 'visits': visits, 'weeks': weeks_seen,
            'fetched': datetime.datetime.now(datetime.timezone.utc)
                               .strftime('%Y-%m-%dT%H:%M:%SZ'),
            'stale': stale, 'note': note}


def summarize(sched, now=None):
    """Counts and the next visit, for the section header.

    ``now`` is injectable so the summary is testable without freezing the clock.
    """
    now = now or datetime.datetime.now(datetime.timezone.utc)
    upcoming, past, total_s = [], [], 0
    for v in sched.get('visits') or []:
        when = start_datetime(v)
        total_s += duration_seconds(v.get('duration')) or 0
        if when is None:
            continue
        (upcoming if when >= now else past).append((when, v))
    upcoming.sort(key=lambda p: p[0])
    past.sort(key=lambda p: p[0])
    targets = {v.get('target') for v in (sched.get('visits') or [])
               if v.get('target')}
    return {
        'n_visits': len(sched.get('visits') or []),
        'n_upcoming': len(upcoming),
        'n_past': len(past),
        'n_targets': len(targets),
        'hours': round(total_s / 3600.0, 1),
        'next': upcoming[0][1] if upcoming else None,
        'next_at': upcoming[0][0] if upcoming else None,
        'last_at': past[-1][0] if past else None,
    }


def write_json(outdir, sched):
    path = os.path.join(outdir, SCHEDULE_JSON)
    with open(path, 'w') as fh:
        json.dump(sched, fh, indent=1, sort_keys=True)
    return path


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        try:
            return json.load(fh)
        except ValueError:
            return None
