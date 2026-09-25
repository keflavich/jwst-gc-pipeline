"""The "Scheduled" panel: what STScI plans to observe, and whether it landed.

Kept out of ``render.py`` for the same reason ``skyview`` is: that module is
already the page's largest, and this panel has its own data source, its own
failure mode (the network) and its own caveat (a plan is not an observation).

The panel answers three questions in that order, because that is the order they
get asked:

1. **when is the next visit** -- the headline.  For months the monitor's answer
   about 10678 was "nothing delivered yet", which is true and useless.  The
   schedule answers it when there is an answer, and says so when there is not:
   the 20260817 week was REISSUED on 2026-08-17 with 10678 replanned out of it,
   so the panel currently reads "no visits for program 10678 in the weekly
   schedules read" -- a statement about the published plan, from the published
   plan.
2. **what is coming this week** -- the timeline, one row per visit.
3. **did the ones that have passed actually arrive** -- each past visit is
   marked against the observations the monitor already scanned, so a planned
   visit whose data never appeared is visible rather than assumed.

(3) is deliberately a WEAK match: the schedule names a target (``GC_20``) and
the archive names an observation (``o020``), and the only thing tying them
together is the observation number in the visit id.  So the panel says "on
disk" only where an observation with that number exists for the program, and
says "not seen" otherwise -- never "missing", which would assert a fault the
page cannot actually establish.
"""
import datetime
import html

from .fold import fold as _fold

from . import schedule as _schedule
from .skyview import mast_url as _mast_url


def esc(text):
    """Same escaping as ``render.esc``, defined here rather than imported.

    ``render`` imports THIS module to place the panel, so importing ``esc``
    back from it is a cycle that fails at import time.  A three-line duplicate
    is cheaper than an indirection layer for one function.
    """
    return html.escape('' if text is None else str(text), quote=True)

#: How many upcoming visits the timeline shows before it collapses the rest.
UPCOMING_SHOWN = 24


def _fmt_dt(when):
    return when.strftime('%Y-%m-%d %H:%M') + 'Z' if when else '—'


def _fmt_delta(seconds):
    """``in 2 d 4 h`` / ``4 h 12 m ago`` -- the reader wants the gap, not a date."""
    past = seconds < 0
    s = int(abs(seconds))
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m = s // 60
    if d:
        text = f'{d} d {h} h'
    elif h:
        text = f'{h} h {m} m'
    else:
        text = f'{m} m'
    return f'{text} ago' if past else f'in {text}'


def _when_cell(when, delta):
    """The gap to a visit, kept current by the browser.

    A schedule is the one table where a stale relative time is actively
    misleading: "in 2 h" rendered at build time still says "in 2 h" after the
    visit has run.  The server's text is the no-JavaScript answer;
    `data-epoch` is what the page's own updater recomputes from, and the style
    tells it to use this table's wording rather than the compact one.
    """
    if delta is None or when is None:
        return '—'
    return (f'<span class="gcm-ago" data-epoch="{int(when.timestamp())}" '
            f'data-style="delta">{esc(_fmt_delta(delta))}</span>')


def observed_observations(entries, program):
    """``{observation number}`` the archive scan already knows for ``program``.

    Reads the entries the monitor built anyway rather than re-scanning: the
    point of the cross-check is to relate two things the page already has.
    """
    seen = set()
    for entry in entries or ():
        run = entry.get('run') or {}
        if str(run.get('proposal') or run.get('proposal_id') or '') != str(program):
            continue
        obs = str(run.get('obsid') or run.get('field') or '').lstrip('o')
        if obs:
            seen.add(obs.lstrip('0') or '0')
    return seen


#: What the state badge says when it is a link, per state.  Spelled out
#: because a badge that navigates has to say where before it is clicked: the
#: three states differ in whether MAST is expected to have anything.
_MAST_TITLE = {
    'ok': 'search MAST for observation %s of program %s — this monitor has '
          'already scanned data for it',
    'wait': 'search MAST for observation %s of program %s — this monitor has '
            'not scanned data for it; it may be proprietary, not yet '
            'delivered, or simply not downloaded here',
    'sched': 'search MAST for observation %s of program %s — it has not run '
             'yet, so expect an empty result until it does',
}


def _state_badge(state, label, program, observation):
    """The state badge, linked to the archive search for its observation.

    The archive link used to live on the footprints in the sky map, where a
    click sometimes selected a tile and sometimes left the site depending on a
    status the reader could not see under the cursor.  Here the row already
    names the observation and states what is known about it, so the badge has
    somewhere honest to point from.

    Linked for every state, ``scheduled`` included, with the tooltip saying
    what to expect: the query is well formed either way, and a badge that is a
    link on some rows and not others is the ambiguity this moved away from.
    """
    url = _mast_url(program, observation)
    badge = f'<span class="gcm-sch-badge is-{state}">{label}</span>'
    if not url:
        return badge
    title = _MAST_TITLE.get(state, 'search MAST for observation %s of program %s')
    return (f'<a class="gcm-sch-mast" href="{esc(url)}" target="_blank" '
            f'rel="noopener" title="{esc(title % (observation, program))}">'
            f'{badge}</a>')


def _row(visit, now, on_disk, program):
    # `program` is positional and required.  It was defaulted to None, and a
    # caller that forgot it got a table where every badge was a bare span --
    # the 65-of-65 linking property resting on one call site remembering an
    # optional argument.  `section` is the only caller, so requiring it is free.
    when = _schedule.start_datetime(visit)
    delta = (when - now).total_seconds() if when else None
    dur = _schedule.duration_seconds(visit.get('duration'))
    parallels = ', '.join(p['instrument'] for p in (visit.get('parallels') or [])
                          if p.get('instrument'))
    obs = str(visit.get('observation') or '').lstrip('0') or '0'
    if when and when >= now:
        state, label = 'sched', 'scheduled'
    elif obs in on_disk:
        state, label = 'ok', 'on disk'
    else:
        # NOT "missing": the plan may have changed, the data may still be in
        # the archive's proprietary window, or it may simply not have been
        # downloaded yet.  The page can establish "we have not seen it".
        state, label = 'wait', 'not seen'
    return f"""<tr class="gcm-sch-{state}">
  <td class="gcm-mono">{esc(visit.get('visit_id'))}</td>
  <td class="gcm-mono">{esc(visit.get('target') or '—')}</td>
  <td class="gcm-mono">{esc(_fmt_dt(when))}</td>
  <td class="gcm-mono">{_when_cell(when, delta)}</td>
  <td class="gcm-mono">{f'{dur // 60} m' if dur else '—'}</td>
  <td>{esc(visit.get('instrument') or '—')}
      {f'<span class="gcm-sch-par">+ {esc(parallels)}</span>' if parallels else ''}</td>
  <td>{_state_badge(state, label, program, obs)}</td>
</tr>"""


CSS = """
.gcm-sch-table { width: 100%; border-collapse: collapse; font-size: .82rem; }
.gcm-sch-table th { text-align: left; font-family: var(--mono); font-size: .7rem;
  text-transform: uppercase; letter-spacing: .06em; color: var(--text-faint);
  border-bottom: 1px solid var(--rule); padding: .3rem .5rem; }
.gcm-sch-table td { padding: .3rem .5rem; border-bottom: 1px solid var(--rule-soft);
  vertical-align: baseline; }
.gcm-sch-table .gcm-mono { font-family: var(--mono); font-variant-numeric: tabular-nums; }
.gcm-sch-wrap { overflow-x: auto; }
.gcm-sch-par { color: var(--text-faint); font-size: .75rem; }
.gcm-sch-badge { font-family: var(--mono); font-size: .68rem; padding: .05rem .4rem;
  border-radius: 2px; border: 1px solid var(--rule); white-space: nowrap; }
.gcm-sch-badge.is-sched { color: var(--accent); border-color: var(--accent); }
.gcm-sch-badge.is-ok { color: var(--ok); border-color: var(--ok); }
.gcm-sch-badge.is-wait { color: var(--warn); border-color: var(--warn); }
/* The badge is the archive link.  No underline and no link color: it already
   reads as a control, and recoloring it would destroy the state it encodes. */
.gcm-sch-mast { text-decoration: none; }
.gcm-sch-mast:hover .gcm-sch-badge { background: rgba(255,255,255,.09); }
.gcm-sch-mast:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.gcm-sch-next { font-family: var(--mono); font-size: 1.35rem; font-weight: 600;
  color: var(--accent); }
.gcm-sch-head { display: flex; flex-wrap: wrap; gap: 1.5rem; align-items: baseline;
  margin: .4rem 0 1rem; }
"""


def section(sched, entries=(), now=None):
    """The panel, or a note saying why there is nothing to show."""
    if not sched:
        return ''
    now = now or datetime.datetime.now(datetime.timezone.utc)
    program = sched.get('program') or _schedule.DEFAULT_PROGRAM
    summary = _schedule.summarize(sched, now=now)
    on_disk = observed_observations(entries, program)

    weeks = sched.get('weeks') or []
    span = (f'{weeks[-1]["week"]}–{weeks[0]["week"]}' if weeks else 'none')
    prov = (f'{len(weeks)} weekly report(s) read ({span}); '
            f'fetched {esc(sched.get("fetched") or "—")}')
    # Rendered whenever it is non-empty, not only under `stale`.  Three
    # degradations reached zero visits without setting `stale`, and the panel
    # then stated in prose that the programme was not on the schedule -- a
    # claim about the survey manufactured from a local parse failure.
    if sched.get('note'):
        label = 'degraded' if sched.get('stale') else 'note'
        prov += f' — <strong>{label}</strong>: {esc(sched.get("note"))}'
    elif sched.get('stale'):
        prov += ' — <strong>degraded</strong>: cached copy used'

    if not sched.get('visits'):
        if sched.get('stale') or sched.get('note'):
            # Something went wrong locally.  "The programme is not on the
            # schedule" would be a statement about the survey with no evidence
            # behind it.
            claim = (f'Could not read the schedule for program {esc(program)}, '
                     f'so this panel says nothing about whether it is on it.')
        else:
            claim = (f'No visits for program {esc(program)} in the weekly '
                     f'schedules read. Either the program is not on the '
                     f'published schedule yet, or its weeks are outside the '
                     f'window this reads.')
        return f"""<section class="gcm-sec" id="schedule"><h2>Scheduled — program {esc(program)}</h2>
{_fold(f'<p class="gcm-note">{claim} {prov}.</p>', 'why this panel is empty')}</section>"""

    nxt = summary['next']
    if nxt is not None:
        delta = (summary['next_at'] - now).total_seconds()
        headline = (f'<div><div class="gcm-sch-next">{esc(_fmt_delta(delta))}</div>'
                    f'<div class="gcm-note" style="margin:0">next: '
                    f'<code>{esc(nxt["visit_id"])}</code> {esc(nxt.get("target") or "")} '
                    f'at {esc(_fmt_dt(summary["next_at"]))}</div></div>')
    else:
        headline = ('<div><div class="gcm-sch-next">—</div>'
                    '<div class="gcm-note" style="margin:0">nothing upcoming in '
                    'the weeks read</div></div>')

    rows, hidden, n_upcoming_shown = [], 0, 0
    for visit in sched['visits']:
        when = _schedule.start_datetime(visit)
        upcoming = bool(when and when >= now)
        if upcoming and n_upcoming_shown >= UPCOMING_SHOWN:
            hidden += 1
            continue
        rows.append(_row(visit, now, on_disk, program))
        n_upcoming_shown += 1 if upcoming else 0
    more = (_fold(f'<p class="gcm-note">{hidden} further scheduled visit(s) '
                  f'not shown; the full list is in <code>schedule.json</code>.</p>',
                  'visits not shown') if hidden else '')

    return f"""<style>{CSS}</style>
<section class="gcm-sec" id="schedule"><h2>Scheduled — program {esc(program)}</h2>
<div class="gcm-sch-head">
  {headline}
  <div class="gcm-tallies" style="margin-left:0">
    <span class="gcm-tally"><b>{summary['n_visits']}</b> visits</span>
    <span class="gcm-tally"><b>{summary['n_targets']}</b> tiles</span>
    <span class="gcm-tally"><b>{summary['hours']}</b> h</span>
    <span class="gcm-tally is-ok"><b>{summary['n_past']}</b> elapsed</span>
  </div>
</div>
<details class="gcm-fold"><summary>about this program and schedule</summary>
<p class="gcm-note">JWST program {esc(program)} is the <strong>Galactic Center
Treasury</strong> survey; each row is one scheduled <em>visit</em> — a single
telescope pointing — and <em>tiles</em> such as <code>GC_1</code> are the survey's
named sky positions. <code>NIRCam Imaging + MIRI Imaging</code> is a
coordinated parallel: a second instrument observing a nearby patch of sky at the
same time. <em>dur</em> is the visit's planned length.</p>
<p class="gcm-note">Read from the STScI weekly observing schedule, which is a
<strong>plan</strong> — STScI's own note is that executed observations can
differ from those scheduled. Nothing here is evidence that an observation
happened; “on disk” means this monitor has scanned data for that observation
number, and “not seen” means only that it has not. {prov}.</p></details>
<div class="gcm-sch-wrap"><table class="gcm-sch-table">
<thead><tr><th>visit</th><th>target</th><th>start (UTC)</th><th>when</th>
<th>dur</th><th>instrument</th><th>state</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
{more}
</section>"""
