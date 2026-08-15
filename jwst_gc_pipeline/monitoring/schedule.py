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
import json
import os
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


class ScheduleFormatError(ValueError):
    """The report does not have the header/ruler this parser needs."""


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
        if line.strip() and set(line.strip()) == {'-', ' '}:
            ruler_at = i
            break
    if ruler_at is None:
        raise ScheduleFormatError('no column ruler (row of dashes) in report')
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
            if current is not None:
                inst = cell(line, 5)
                if inst:
                    current['parallels'].append(
                        {'visit_type': cell(line, 2), 'instrument': inst})
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


def load(outdir, program=DEFAULT_PROGRAM, weeks=8, offline=False):
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
    if not offline:
        try:
            index_html = _get(SCHEDULE_INDEX)
            with open(index_path, 'w') as fh:
                fh.write(index_html)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            note = f'could not reach the STScI schedule index ({exc})'
            stale = True
    if not index_html and os.path.exists(index_path):
        with open(index_path) as fh:
            index_html = fh.read()
        stale = True
        note = note or 'using the cached schedule index'
    if not index_html:
        return {'program': program, 'visits': [], 'weeks': [], 'stale': True,
                'fetched': None,
                'note': note or 'no schedule index available, cached or live'}

    wanted = report_urls(index_html)[:weeks]
    visits, weeks_seen = [], []
    for week, generated, url in wanted:
        path = os.path.join(cache, f'{week}_report_{generated}.txt')
        text = ''
        if os.path.exists(path):
            with open(path) as fh:
                text = fh.read()
        elif not offline:
            try:
                text = _get(url)
                with open(path, 'w') as fh:
                    fh.write(text)
            except (urllib.error.URLError, OSError, ValueError) as exc:
                note = note or f'could not fetch {os.path.basename(url)} ({exc})'
                stale = True
        if not text:
            continue
        try:
            parsed = parse_report(text)
        except ScheduleFormatError as exc:
            note = note or f'{os.path.basename(url)}: {exc}'
            continue
        mine = [v for v in parsed if v['program'] == str(program)]
        weeks_seen.append({'week': week, 'generated': generated, 'url': url,
                           'n_visits': len(parsed), 'n_program': len(mine)})
        for v in mine:
            v['week'] = week
        visits.extend(mine)

    visits.sort(key=lambda v: (v.get('start') or '', v['visit_id']))
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
