"""Render a scan into a self-contained HTML monitoring page.

Two outputs, one renderer:

* ``render_page(...)`` -- the aggregate page: an overview grid of every field,
  then a detail section per field.
* ``render_page(..., standalone=True)`` -- the same markup wrapped in a full HTML
  document, for the per-field files written to disk.

No external requests: the CSS and the small amount of JS are inlined, there are
no webfonts, and the only images are drawn with CSS.  The page is styled through
custom properties so the viewer's light/dark preference AND an explicit
``data-theme`` toggle both work.

Design notes
------------
The accent is the ``F212N`` cyan of the CMZ house two-colour scheme (blue = 2.12
um, red = 4.05/4.80 um), and filter chips are tinted along that same
wavelength ramp -- so a filter's colour on the page means the same thing it means
in the survey's own imagery.  Warm hues are left to the severity system
(fail / warn) so status never competes with the accent.
"""
import html
import os
import time

# --------------------------------------------------------------------------
# Wavelength -> hue, so a filter chip is tinted by what it actually observes.
# --------------------------------------------------------------------------

_FILTER_RE = __import__('re').compile(r'^F(\d{3,4})')


def filter_micron(filt):
    """``'F212N' -> 2.12``; ``'F2550W' -> 25.50``; ``'F150W2' -> 1.50``.

    Only the digits that FOLLOW the F encode the wavelength.  The trailing ``2``
    of the wide pairs ``F150W2``/``F322W2`` is part of the filter's name, not the
    number -- reading every digit turns 1.50 um into 15.02 and paints both
    globular-cluster filters at the far red end of the ramp.
    """
    m = _FILTER_RE.match(str(filt).upper())
    return int(m.group(1)) / 100.0 if m else None


def filter_hue(filt):
    """Hue in degrees along the SW-cyan -> LW-amber ramp (1.1 um .. 5.0 um)."""
    micron = filter_micron(filt)
    if micron is None:
        return 205
    lo, hi = 1.1, 5.0
    frac = min(1.0, max(0.0, (micron - lo) / (hi - lo)))
    return 200 - 172 * frac          # 200 (cyan) -> 28 (amber)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

_SEV_LABEL = {'fail': 'fail', 'warn': 'warn', 'info': 'ok', 'skip': 'n/a'}


def esc(text):
    return html.escape('' if text is None else str(text), quote=True)


def ago(mtime, now=None):
    """``'3h ago'`` / ``'12d ago'`` / ``'--'``."""
    if not mtime:
        return '—'
    delta = (now or time.time()) - mtime
    if delta < 0:
        return 'just now'
    for size, unit in ((86400, 'd'), (3600, 'h'), (60, 'm')):
        if delta >= size:
            return f'{int(delta // size)}{unit} ago'
    return f'{int(delta)}s ago'


def num(value):
    return '—' if value is None else f'{value:,}'


# --------------------------------------------------------------------------
# CSS
# --------------------------------------------------------------------------

CSS = """
:root {
  /* Neutrals carry a slight cyan bias so they read as chosen next to the accent. */
  --ground:      #f3f6f7;
  --surface:     #ffffff;
  --surface-2:   #e9eef0;
  --rule:        #ccd8dc;
  --rule-soft:   #dee7ea;
  --text:        #16232a;
  --text-dim:    #5a6d76;
  --text-faint:  #8698a1;
  --accent:      #12798f;
  --accent-soft: #d7ecf1;
  --fail:        #b8362c;
  --fail-soft:   #f7dcd9;
  --warn:        #a5701a;
  --warn-soft:   #f8ecd4;
  --ok:          #2f7a4e;
  --ok-soft:     #dcefe2;
  --skip:        #7d8d96;
  --skip-soft:   #e6ebed;
  --shadow:      0 1px 2px rgba(20, 40, 48, .08), 0 8px 24px -16px rgba(20, 40, 48, .35);

  --mono: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas,
          "Liberation Mono", monospace;
  --sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto,
          "Helvetica Neue", Arial, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root {
    --ground:      #0b1418;
    --surface:     #111e24;
    --surface-2:   #16262d;
    --rule:        #24393f;
    --rule-soft:   #1b2c32;
    --text:        #dce8ec;
    --text-dim:    #93a8b1;
    --text-faint:  #6b828c;
    --accent:      #46bcd6;
    --accent-soft: #0f3540;
    --fail:        #ef7a6e;
    --fail-soft:   #3a1a17;
    --warn:        #e0ac52;
    --warn-soft:   #382a12;
    --ok:          #64c58c;
    --ok-soft:     #12301f;
    --skip:        #7d939c;
    --skip-soft:   #1a2930;
    --shadow:      0 1px 2px rgba(0, 0, 0, .4), 0 10px 28px -18px rgba(0, 0, 0, .9);
  }
}
:root[data-theme="dark"] {
  --ground: #0b1418; --surface: #111e24; --surface-2: #16262d;
  --rule: #24393f; --rule-soft: #1b2c32;
  --text: #dce8ec; --text-dim: #93a8b1; --text-faint: #6b828c;
  --accent: #46bcd6; --accent-soft: #0f3540;
  --fail: #ef7a6e; --fail-soft: #3a1a17;
  --warn: #e0ac52; --warn-soft: #382a12;
  --ok: #64c58c; --ok-soft: #12301f;
  --skip: #7d939c; --skip-soft: #1a2930;
  --shadow: 0 1px 2px rgba(0,0,0,.4), 0 10px 28px -18px rgba(0,0,0,.9);
}
:root[data-theme="light"] {
  --ground: #f3f6f7; --surface: #ffffff; --surface-2: #e9eef0;
  --rule: #ccd8dc; --rule-soft: #dee7ea;
  --text: #16232a; --text-dim: #5a6d76; --text-faint: #8698a1;
  --accent: #12798f; --accent-soft: #d7ecf1;
  --fail: #b8362c; --fail-soft: #f7dcd9;
  --warn: #a5701a; --warn-soft: #f8ecd4;
  --ok: #2f7a4e; --ok-soft: #dcefe2;
  --skip: #7d8d96; --skip-soft: #e6ebed;
  --shadow: 0 1px 2px rgba(20,40,48,.08), 0 8px 24px -16px rgba(20,40,48,.35);
}

.gcm { background: var(--ground); color: var(--text); font-family: var(--sans);
       font-size: 15px; line-height: 1.5; padding: 0 0 5rem; }
.gcm *, .gcm *::before, .gcm *::after { box-sizing: border-box; }
.gcm-wrap { max-width: 1180px; margin: 0 auto; padding: 0 1.25rem; }

/* --- masthead ---------------------------------------------------------- */
.gcm-head { border-bottom: 1px solid var(--rule); background: var(--surface);
            position: sticky; top: 0; z-index: 20; }
.gcm-head-in { display: flex; flex-wrap: wrap; align-items: baseline;
               gap: .5rem 1.5rem; padding: .9rem 1.25rem; max-width: 1180px;
               margin: 0 auto; }
.gcm-title { font-family: var(--mono); font-size: 1.05rem; font-weight: 600;
             letter-spacing: -.01em; margin: 0; }
.gcm-title span { color: var(--accent); }
.gcm-sub { font-family: var(--mono); font-size: .74rem; color: var(--text-faint);
           letter-spacing: .04em; text-transform: uppercase; }
.gcm-tallies { display: flex; gap: 1rem; margin-left: auto;
               font-family: var(--mono); font-size: .78rem; }
.gcm-tally b { font-size: 1.05rem; font-weight: 600; font-variant-numeric: tabular-nums; }
.gcm-tally { color: var(--text-dim); display: flex; align-items: baseline; gap: .35rem; }
.gcm-tally.is-fail b { color: var(--fail); }
.gcm-tally.is-warn b { color: var(--warn); }
.gcm-tally.is-ok   b { color: var(--ok); }

/* --- section furniture ------------------------------------------------- */
.gcm-sec { margin-top: 2.75rem; }
.gcm-sec > h2 { font-family: var(--mono); font-size: .8rem; font-weight: 600;
                letter-spacing: .12em; text-transform: uppercase;
                color: var(--text-dim); margin: 0 0 .25rem;
                padding-bottom: .5rem; border-bottom: 1px solid var(--rule); }
.gcm-note { color: var(--text-dim); font-size: .86rem; margin: .6rem 0 1rem;
            max-width: 68ch; }

/* --- overview grid ----------------------------------------------------- */
.gcm-grid { display: grid; gap: .75rem;
            grid-template-columns: repeat(auto-fill, minmax(268px, 1fr)); }
.gcm-card { background: var(--surface); border: 1px solid var(--rule-soft);
            border-left: 3px solid var(--skip); border-radius: 3px;
            padding: .7rem .8rem .75rem; box-shadow: var(--shadow);
            display: flex; flex-direction: column; gap: .5rem;
            text-decoration: none; color: inherit; }
.gcm-card:hover { border-color: var(--accent); }
.gcm-card:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.gcm-card.sev-fail { border-left-color: var(--fail); }
.gcm-card.sev-warn { border-left-color: var(--warn); }
.gcm-card.sev-info { border-left-color: var(--ok); }
.gcm-card-top { display: flex; align-items: baseline; gap: .5rem; }
.gcm-card-name { font-family: var(--mono); font-weight: 600; font-size: .98rem; }
.gcm-card-obs { font-family: var(--mono); font-size: .72rem; color: var(--text-faint); }
.gcm-card-when { margin-left: auto; font-family: var(--mono); font-size: .7rem;
                 color: var(--text-faint); }

/* --- stage ladder bar -------------------------------------------------- */
.gcm-ladder { display: flex; gap: 2px; }
.gcm-step { flex: 1; height: 16px; border-radius: 1px; background: var(--surface-2);
            border: 1px solid var(--rule-soft); position: relative; }
.gcm-step.done { background: var(--accent); border-color: var(--accent); }
.gcm-step.part { background: var(--accent-soft); border-color: var(--accent); }
.gcm-step.ambig { background: repeating-linear-gradient(45deg,
                  var(--warn-soft), var(--warn-soft) 3px,
                  var(--warn) 3px, var(--warn) 4px); border-color: var(--warn); }
.gcm-step.bad { background: var(--fail); border-color: var(--fail); }
.gcm-ladder-key { display: flex; gap: .5rem; font-family: var(--mono);
                  font-size: .62rem; color: var(--text-faint);
                  letter-spacing: .06em; }
.gcm-ladder-key span { flex: 1; text-align: center; }

/* --- chips ------------------------------------------------------------- */
.gcm-chips { display: flex; flex-wrap: wrap; gap: .3rem; }
.gcm-chip { font-family: var(--mono); font-size: .68rem; letter-spacing: .03em;
            padding: .1rem .38rem; border-radius: 2px; border: 1px solid;
            white-space: nowrap; }
.gcm-chip.filt { border-color: hsl(var(--h) 45% 45% / .45);
                 background: hsl(var(--h) 55% 50% / .13);
                 color: hsl(var(--h) 55% 32%); }
@media (prefers-color-scheme: dark) {
  .gcm-chip.filt { color: hsl(var(--h) 60% 72%); }
}
:root[data-theme="dark"] .gcm-chip.filt { color: hsl(var(--h) 60% 72%); }
:root[data-theme="light"] .gcm-chip.filt { color: hsl(var(--h) 55% 32%); }
.gcm-chip.fail { color: var(--fail); background: var(--fail-soft); border-color: var(--fail); }
.gcm-chip.warn { color: var(--warn); background: var(--warn-soft); border-color: var(--warn); }
.gcm-chip.info { color: var(--ok);   background: var(--ok-soft);   border-color: var(--ok); }
.gcm-chip.skip { color: var(--skip); background: var(--skip-soft); border-color: var(--skip); }
.gcm-chip.run  { color: var(--accent); background: var(--accent-soft); border-color: var(--accent); }

/* --- detail ------------------------------------------------------------ */
.gcm-field { margin-top: 2.25rem; scroll-margin-top: 4.5rem;
             background: var(--surface); border: 1px solid var(--rule-soft);
             border-radius: 3px; box-shadow: var(--shadow); overflow: hidden; }
.gcm-field-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: .75rem;
                  padding: .8rem 1rem; border-bottom: 1px solid var(--rule-soft);
                  background: var(--surface-2); }
.gcm-field-head h3 { font-family: var(--mono); font-size: 1.02rem; margin: 0;
                     font-weight: 600; }
.gcm-field-head .path { font-family: var(--mono); font-size: .7rem;
                        color: var(--text-faint); word-break: break-all; }
.gcm-field-body { padding: .9rem 1rem 1.1rem; display: flex;
                  flex-direction: column; gap: 1.1rem; }
.gcm-sub-h { font-family: var(--mono); font-size: .68rem; letter-spacing: .12em;
             text-transform: uppercase; color: var(--text-faint);
             margin: 0 0 .4rem; }

.gcm-scroll { overflow-x: auto; }
table.gcm-t { border-collapse: collapse; width: 100%; font-size: .8rem;
              font-variant-numeric: tabular-nums; }
table.gcm-t th { text-align: left; font-family: var(--mono); font-weight: 600;
                 font-size: .68rem; letter-spacing: .07em; text-transform: uppercase;
                 color: var(--text-faint); padding: .3rem .55rem;
                 border-bottom: 1px solid var(--rule); white-space: nowrap; }
table.gcm-t td { padding: .28rem .55rem; border-bottom: 1px solid var(--rule-soft);
                 white-space: nowrap; }
table.gcm-t td.n { font-family: var(--mono); text-align: right; }
table.gcm-t td.z { color: var(--text-faint); }
table.gcm-t tr:last-child td { border-bottom: 0; }
table.gcm-t td.filt { font-family: var(--mono); font-weight: 600; }

.gcm-checks { list-style: none; margin: 0; padding: 0;
              display: flex; flex-direction: column; gap: .3rem; }
.gcm-check { display: grid; grid-template-columns: 3.4rem 1fr; gap: .6rem;
             padding: .35rem .5rem; border-radius: 2px; background: var(--surface-2);
             border-left: 2px solid var(--skip); font-size: .84rem; }
.gcm-check.fail { border-left-color: var(--fail); }
.gcm-check.warn { border-left-color: var(--warn); }
.gcm-check.info { border-left-color: var(--ok); }
.gcm-check-sev { font-family: var(--mono); font-size: .64rem; letter-spacing: .1em;
                 text-transform: uppercase; padding-top: .18rem; }
.gcm-check.fail .gcm-check-sev { color: var(--fail); }
.gcm-check.warn .gcm-check-sev { color: var(--warn); }
.gcm-check.info .gcm-check-sev { color: var(--ok); }
.gcm-check.skip .gcm-check-sev { color: var(--skip); }
.gcm-check-detail { color: var(--text-dim); font-size: .78rem; margin-top: .15rem;
                    white-space: pre-wrap; word-break: break-word; }
.gcm-check-src { font-family: var(--mono); font-size: .68rem;
                 color: var(--text-faint); margin-top: .15rem; word-break: break-all; }

.gcm-empty { color: var(--text-faint); font-size: .82rem; font-style: italic; }
.gcm-foot { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--rule);
            color: var(--text-faint); font-size: .76rem; }
.gcm-foot code { font-family: var(--mono); }
a.gcm-back { font-family: var(--mono); font-size: .7rem; color: var(--accent);
             text-decoration: none; margin-left: auto; }
a.gcm-back:hover { text-decoration: underline; }
.gcm-wu { display: flex; flex-wrap: wrap; align-items: baseline; gap: .5rem;
          font-size: .8rem; }
.gcm-wu a { color: var(--accent); }
.gcm-wu-fig { font-weight: 600; }
/* --- expandable evidence ------------------------------------------------ */
.gcm-ev { margin-top: .45rem; }
.gcm-ev > summary { cursor: pointer; font-family: var(--mono); font-size: .68rem;
                    letter-spacing: .06em; text-transform: uppercase;
                    color: var(--accent); list-style: none; padding: .12rem 0;
                    width: max-content; }
.gcm-ev > summary::-webkit-details-marker { display: none; }
.gcm-ev > summary::before { content: "\25B8"; display: inline-block;
                            margin-right: .35rem; transition: transform .12s; }
.gcm-ev[open] > summary::before { transform: rotate(90deg); }
.gcm-ev > summary:hover { text-decoration: underline; }
.gcm-ev > summary:focus-visible { outline: 2px solid var(--accent);
                                  outline-offset: 2px; }
.gcm-ev-body { margin-top: .5rem; padding: .6rem .7rem; border-radius: 2px;
               background: var(--surface); border: 1px solid var(--rule-soft);
               display: flex; flex-direction: column; gap: .7rem; }
.gcm-cause { font-size: .82rem; color: var(--text); }
.gcm-draws { display: flex; flex-wrap: wrap; gap: 1rem; align-items: flex-start; }
.gcm-draw { margin: 0; max-width: 260px; }
.gcm-draw figcaption { font-size: .7rem; color: var(--text-faint);
                       margin-top: .3rem; }
.gcm-draw svg { background: var(--surface-2); }
.gcm-bars { display: flex; flex-direction: column; gap: .18rem; min-width: 190px; }
.gcm-bar-row { display: flex; align-items: center; gap: .4rem;
               font-family: var(--mono); font-size: .7rem; }
.gcm-bar-lab { width: 4.2rem; color: var(--text-dim); }
.gcm-bar { flex: 1; height: 8px; background: var(--surface-2); border-radius: 1px;
           overflow: hidden; }
.gcm-bar i { display: block; height: 100%; background: var(--fail); }
.gcm-bar-n { width: 4rem; text-align: right; color: var(--text-faint); }
.gcm-figs ul { margin: .2rem 0 0; padding-left: 1.1rem; font-size: .78rem; }
.gcm-figs a { color: var(--accent); }
@media (prefers-reduced-motion: reduce) { .gcm * { transition: none !important; } }
"""


# --------------------------------------------------------------------------
# Fragments
# --------------------------------------------------------------------------

#: The ladder shown on every card, in run order.  Keeping reduction and
#: cataloging in ONE bar is the point: a field with a full catalog ladder but no
#: reduced frames is a stale catalog, and that only reads at a glance if the two
#: halves sit side by side.
#: The bare ``_o<obs>_crf.fits`` is deliberately NOT a rung.  Most fields keep
#: only the destreaked/aligned working copy, so a bare-crf rung would read as a
#: hole in the middle of every healthy ladder.  The distinction still matters and
#: is still made -- it lives in the per-filter table and in the ``unreduced-<F>``
#: check, where "crf present, reduced absent" is the finding.
LADDER = (('uncal', 'unc'), ('cal', 'cal'), ('reduced', 'red'),
          ('i2d', 'i2d'), ('m12', 'm12'), ('m3', 'm3'), ('m4', 'm4'),
          ('m5', 'm5'), ('m6', 'm6'), ('m7', 'm7'), ('m8', 'm8'))

#: Columns of the per-filter table: the ladder plus the bare-crf column the
#: ladder omits.
TABLE_COLUMNS = (('uncal', 'unc'), ('cal', 'cal'), ('crf', 'crf'),
                 ('reduced', 'red'), ('i2d', 'i2d'), ('satstar', 'sat'),
                 ('m12', 'm12'), ('m3', 'm3'), ('m4', 'm4'), ('m5', 'm5'),
                 ('m6', 'm6'))


def _ladder_state(run):
    """``{step: ('done'|'part'|'ambig'|'')}`` rolled up over the run's filters.

    ``part`` means some filters have the product and some do not -- the common
    real state mid-run, and the one a boolean "done" would hide.
    """
    per_filter = run.get('per_filter') or {}
    cross = run.get('crossband') or {}
    n_filters = len(per_filter) or 1
    state = {}
    for step, _ in LADDER:
        if step in ('m7', 'm8'):
            row = cross.get(step) or {}
            if not row.get('n'):
                state[step] = ''
            elif row.get('scope') == 'ambiguous':
                state[step] = 'ambig'
            else:
                state[step] = 'done'
            continue
        have = [f for f, rows in per_filter.items() if (rows.get(step) or {}).get('n')]
        ambiguous = any((per_filter[f].get(step) or {}).get('scope') == 'ambiguous'
                        for f in have)
        if not have:
            state[step] = ''
        elif ambiguous:
            state[step] = 'ambig'
        elif len(have) == n_filters:
            state[step] = 'done'
        else:
            state[step] = 'part'
    return state


def _ladder_html(run, with_key=False):
    state = _ladder_state(run)
    cells = []
    for step, label in LADDER:
        cls = state.get(step, '')
        cells.append(f'<div class="gcm-step {cls}" title="{esc(step)}: '
                     f'{esc(cls or "absent")}"></div>')
    out = f'<div class="gcm-ladder">{"".join(cells)}</div>'
    if with_key:
        keys = ''.join(f'<span>{esc(label)}</span>' for _, label in LADDER)
        out += f'<div class="gcm-ladder-key">{keys}</div>'
    return out


def _filter_chips(filters):
    return ''.join(
        f'<span class="gcm-chip filt" style="--h:{filter_hue(f):.0f}">{esc(f)}</span>'
        for f in filters)


def _severity_chip(tally):
    parts = []
    for sev, cls in (('fail', 'fail'), ('warn', 'warn')):
        if tally.get(sev):
            parts.append(f'<span class="gcm-chip {cls}">{tally[sev]} {sev}</span>')
    if not parts:
        parts.append('<span class="gcm-chip info">clear</span>')
    return ''.join(parts)


def _card(entry):
    run = entry['run']
    tally = entry['tally']
    worst = entry['worst']
    jobs = entry.get('jobs') or []
    active = [j for j in jobs if j.get('state') in ('RUNNING', 'PENDING')]
    newest = entry.get('newest_mtime')
    anchor = entry['anchor']
    job_chip = (f'<span class="gcm-chip run">{len(active)} in queue</span>'
                if active else '')
    return f"""
<a class="gcm-card sev-{esc(worst)}" href="#{esc(anchor)}">
  <div class="gcm-card-top">
    <span class="gcm-card-name">{esc(run['target'])}</span>
    <span class="gcm-card-obs">{esc(run['proposal'])}/o{esc(run['obsid'])}</span>
    <span class="gcm-card-when">{esc(ago(newest))}</span>
  </div>
  {_ladder_html(run)}
  <div class="gcm-chips">{_severity_chip(tally)}{job_chip}</div>
  <div class="gcm-chips">{_filter_chips(sorted(run.get('per_filter') or {}))}</div>
</a>"""


def _stage_table(run):
    per_filter = run.get('per_filter') or {}
    if not per_filter:
        return '<p class="gcm-empty">No filter directories on disk.</p>'
    head = ''.join(f'<th>{esc(label)}</th>' for _, label in TABLE_COLUMNS)
    rows = []
    for filt in sorted(per_filter):
        cells = []
        for step, _label in TABLE_COLUMNS:
            row = per_filter[filt].get(step) or {}
            n = row.get('n', 0)
            cls = 'n' + ('' if n else ' z')
            mark = '' if row.get('scope') != 'ambiguous' else '*'
            cells.append(f'<td class="{cls}">{num(n) if n else "·"}{mark}</td>')
        variant = (per_filter[filt].get('reduced') or {}).get('variant') or ''
        cells.append(f'<td class="z">{esc(variant)}</td>')
        rows.append(f'<tr><td class="filt" style="color:hsl({filter_hue(filt):.0f} '
                    f'50% 40%)">{esc(filt)}</td>{"".join(cells)}</tr>')
    return f"""
<div class="gcm-scroll"><table class="gcm-t">
<thead><tr><th>filter</th>{head}<th>reduced&nbsp;as</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<p class="gcm-note" style="margin:.4rem 0 0;font-size:.74rem">
  <code>*</code> = the product name carries no <code>_o&lt;obs&gt;</code> token and this
  field has more than one observation, so the count cannot be attributed to
  {esc(run['proposal'])}/o{esc(run['obsid'])}.</p>"""


def _astrometry_table(run):
    astrom = run.get('astrometry') or {}
    if not astrom:
        return ('<p class="gcm-empty">No m2 checkpoint records — the field has not '
                'reached the m2 merge, or ran with ASTROM_CHECKPOINT=0.</p>')
    from .checks import LOCAL_CELL_TOL_MAS
    rows = []
    for filt, rec in sorted(astrom.items()):
        mis = rec.get('n_misaligned') or 0
        contrast = rec.get('min_contrast')
        for visit in (rec.get('visits') or [{}]) or [{}]:
            worst = visit.get('worst_tile_mas')
            tie = visit.get('tie_off_mas')
            hot = (worst or 0) > LOCAL_CELL_TOL_MAS
            rows.append(f"""<tr>
<td class="filt" style="color:hsl({filter_hue(filt):.0f} 50% 40%)">{esc(filt)}</td>
<td class="z">{esc(visit.get('visit') or '—')}</td>
<td class="n">{num(rec.get('n_exposures'))}</td>
<td class="n" style="{'color:var(--fail);font-weight:600' if mis else ''}">{num(mis)}</td>
<td class="n">{num(rec.get('n_swept'))}</td>
<td class="n">{'—' if contrast is None else f'{contrast:.0f}'}</td>
<td class="n">{'—' if tie is None else f'{tie:.2f}'}</td>
<td class="z">{esc(visit.get('tie_source') or '')}</td>
<td class="n">{visit.get('tiles_ok') if visit.get('tiles_ok') is not None else '—'}/{visit.get('tiles_total') if visit.get('tiles_total') is not None else '—'}</td>
<td class="n" style="{'color:var(--fail);font-weight:600' if hot else ''}">{'—' if worst is None else f'{worst:.1f}'}</td>
<td class="z">{esc(visit.get('worst_tile_cell') or '')}</td>
<td class="z">{esc(rec.get('date') or '')}</td>
</tr>""")
    return f"""
<div class="gcm-scroll"><table class="gcm-t">
<thead><tr><th>filter</th><th>visit</th><th>exposures</th><th>misaligned</th>
<th>swept</th><th>min contrast</th><th>bulk tie (mas)</th><th>tie from</th>
<th>tiles ok</th><th>worst tile (mas)</th><th>cell</th>
<th>checkpoint</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<p class="gcm-note" style="margin:.4rem 0 0;font-size:.74rem">
  <b>“tiles ok” is not a tolerance.</b> <code>measure_offset_grid</code> runs with no
  <code>max_off_mas</code>, and <code>astrometry_offsets</code> sets
  <code>off_ok=True</code> whenever that is <code>None</code> — so N/N counts tiles
  whose offset histogram had a coherent <em>peak</em>, however large the offset.
  The column that carries the gate is <b>worst tile</b>, against
  {LOCAL_CELL_TOL_MAS:g} mas.</p>"""


def _provenance_block(run):
    prov = run.get('provenance') or {}
    if not prov:
        return '<p class="gcm-empty">No <code>*.prov.json</code> sidecars.</p>'
    rows = []
    for phase, rec in sorted(prov.items()):
        tags = rec.get('tags') or {}
        listing = ', '.join(f'{esc(t)} ({n})' for t, n in
                            sorted(tags.items(), key=lambda kv: -kv[1]))
        colour = 'var(--fail)' if rec.get('n_distinct', 0) > 1 else 'inherit'
        rows.append(f'<tr><td class="filt">{esc(phase)}</td>'
                    f'<td class="n">{num(rec.get("n_sidecars"))}</td>'
                    f'<td style="color:{colour};white-space:normal">{listing}</td></tr>')
    return (f'<div class="gcm-scroll"><table class="gcm-t"><thead><tr><th>phase</th>'
            f'<th>products</th><th>pipeline tag(s)</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def _fmt_cell(value):
    if value is None:
        return '—'
    if isinstance(value, float):
        return f'{value:,.1f}'
    return esc(value)


def _evidence_table(rows):
    """The affected items themselves, capped and labelled when truncated."""
    if not rows or not rows.get('data'):
        return ''
    head = ''.join(f'<th>{esc(c)}</th>' for c in rows.get('columns') or [])
    body = ''.join(
        '<tr>' + ''.join(f'<td class="n">{_fmt_cell(c)}</td>' for c in row) + '</tr>'
        for row in rows['data'])
    total, shown = rows.get('total', len(rows['data'])), len(rows['data'])
    more = (f'<p class="gcm-note" style="margin:.3rem 0 0;font-size:.72rem">'
            f'showing {shown} of {total}</p>' if total > shown else '')
    return (f'<div class="gcm-scroll"><table class="gcm-t"><thead><tr>{head}</tr>'
            f'</thead><tbody>{body}</tbody></table></div>{more}')


def _detector_tally_html(tally):
    if not tally:
        return ''
    bars = []
    for det, bad, total in tally:
        frac = (bad / total) if total else 0
        bars.append(
            f'<div class="gcm-bar-row"><span class="gcm-bar-lab">{esc(det)}</span>'
            f'<span class="gcm-bar"><i style="width:{frac * 100:.0f}%"></i></span>'
            f'<span class="gcm-bar-n">{bad}/{total}</span></div>')
    return f'<div class="gcm-bars">{"".join(bars)}</div>'


def _figure_links(figs, figure_base='figures'):
    """Links to diagnostics already on disk.

    Relative, and resolved only where those files were published alongside the
    page — so the list says where it points rather than presenting dead links as
    if they were live.
    """
    if not figs:
        return ''
    items = ''.join(
        f'<li><a href="{esc(figure_base)}/{esc(f["name"])}">{esc(f["name"])}</a>'
        f' <span class="gcm-check-src">{esc(f["dir"])}</span></li>'
        for f in figs)
    return (f'<div class="gcm-figs"><div class="gcm-sub-h">Existing figures</div>'
            f'<ul>{items}</ul></div>')


def _writeup_links(wu):
    """Link into the field's diagnostic writeup, at the figure that shows THIS.

    The writeup carries a fixed D1..D8 figure set for every field, so a finding
    can point at the one that shows it rather than at a directory listing.
    """
    if not wu:
        return ''
    bits = []
    fig = wu.get('figure')
    if fig:
        bits.append(f'<a class="gcm-wu-fig" href="{esc(fig["href"])}">'
                    f'{esc(fig["name"].split("_")[0])} — {esc(fig["label"])}</a>')
    if wu.get('main'):
        bits.append(f'<a href="{esc(wu["main"])}">full diagnostic writeup (PDF)</a>')
    if not bits:
        return ''
    return (f'<div class="gcm-wu"><span class="gcm-sub-h">Diagnostic writeup</span>'
            f'{" · ".join(bits)}</div>')


def _evidence_block(v, figure_base='figures'):
    ev = v.get('evidence') or {}
    cause = (f'<div class="gcm-cause">{esc(v["cause"])}</div>'
             if v.get('cause') else '')
    drawings = []
    if ev.get('tile_map'):
        drawings.append(
            f'<figure class="gcm-draw">{ev["tile_map"]}'
            f'<figcaption>Per-tile residual across the mosaic. Outlined cells '
            f'exceed tolerance; the circled cell is the worst. Whether the bad '
            f'cells sit on the edge or in the interior is the diagnosis.'
            f'</figcaption></figure>')
    if ev.get('quiver'):
        drawings.append(
            f'<figure class="gcm-draw">{ev["quiver"]}'
            f'<figcaption>Per-exposure offset vectors, coloured by detector. '
            f'Solid = flagged misaligned. One colour pointing away means one '
            f'detector; everything fanning out means the frame moved.'
            f'</figcaption></figure>')
    if ev.get('detector_tally'):
        drawings.append(
            f'<div class="gcm-draw"><div class="gcm-sub-h">Misaligned by detector'
            f'</div>{_detector_tally_html(ev["detector_tally"])}</div>')
    draw_html = (f'<div class="gcm-draws">{"".join(drawings)}</div>'
                 if drawings else '')
    table = _evidence_table(ev.get('rows'))
    figs = _figure_links(ev.get('figures'), figure_base)
    wu = _writeup_links(ev.get('writeup'))
    if not (cause or draw_html or table or figs or wu):
        return ''
    return (f'<details class="gcm-ev"><summary>what is affected, and why</summary>'
            f'<div class="gcm-ev-body">{cause}{draw_html}{wu}{table}{figs}</div>'
            f'</details>')


def _checks_block(verdicts, show_skip=False, figure_base='figures'):
    items = [v for v in verdicts if show_skip or v['severity'] != 'skip']
    if not items:
        return '<p class="gcm-empty">Nothing to report.</p>'
    out = []
    for v in items:
        detail = (f'<div class="gcm-check-detail">{esc(v["detail"])}</div>'
                  if v.get('detail') else '')
        src = (f'<div class="gcm-check-src">{esc(v["source"])}</div>'
               if v.get('source') else '')
        out.append(f"""<li class="gcm-check {esc(v['severity'])}">
<span class="gcm-check-sev">{esc(_SEV_LABEL.get(v['severity'], v['severity']))}</span>
<div><div>{esc(v['summary'])}</div>{detail}{src}
{_evidence_block(v, figure_base)}</div></li>""")
    return f'<ul class="gcm-checks">{"".join(out)}</ul>'


def _jobs_block(jobs):
    if not jobs:
        return '<p class="gcm-empty">No jobs in the queue for this field.</p>'
    rows = []
    for job in sorted(jobs, key=lambda j: (j.get('state', ''), j.get('name', ''))):
        rows.append(f'<tr><td class="n">{esc(job.get("jobid"))}</td>'
                    f'<td class="filt">{esc(job.get("name"))}</td>'
                    f'<td>{esc(job.get("state"))}</td>'
                    f'<td class="n">{esc(job.get("elapsed"))}</td>'
                    f'<td class="z" style="white-space:normal">'
                    f'{esc(job.get("reason"))}</td></tr>')
    return (f'<div class="gcm-scroll"><table class="gcm-t"><thead><tr><th>job</th>'
            f'<th>name</th><th>state</th><th>elapsed</th><th>reason</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def _cutouts_block(cutouts):
    if not cutouts:
        return ''
    rows = []
    for cut in sorted(cutouts, key=lambda c: (c['target'], c['label'])):
        filters = ' '.join(f'{k}:{v}' for k, v in sorted(cut['filters'].items()))
        flags = []
        if cut.get('n_empty'):
            flags.append(f'<span class="gcm-chip fail">{cut["n_empty"]} zero-byte'
                         f'</span>')
        if cut.get('n_orphan_tmp'):
            flags.append(f'<span class="gcm-chip warn">{cut["n_orphan_tmp"]} orphan '
                         f'tmp</span>')
        rows.append(f"""<tr>
<td class="filt">{esc(cut['target'])}</td><td class="filt">{esc(cut['label'])}</td>
<td class="n">{num(cut['n_frames'])}</td><td class="n">{num(cut['n_catalogs'])}</td>
<td class="z" style="white-space:normal">{esc(filters)}</td>
<td class="z">{esc(ago(cut['mtime']))}</td>
<td style="white-space:normal">{''.join(flags)}</td></tr>""")
    return f"""
<section class="gcm-sec"><h2>Cutout runs</h2>
<p class="gcm-note">Every <code>--cutout-region</code> run on disk: the shared
<code>monitor5as</code> probes plus the hand-made experiment cutouts. A cutout run
stops after m6 — m7/m8 need more than one filter — so "catalogs" here counts
per-filter products only. A cutout tree is small enough to stat exhaustively, so
these rows also flag <em>zero-byte</em> products and orphan <code>tmp*</code>
files — a write that died mid-flight leaves both, and a count-based ladder reads
them as finished work.</p>
<div class="gcm-scroll"><table class="gcm-t">
<thead><tr><th>field</th><th>label</th><th>frames</th><th>catalogs</th>
<th>filters</th><th>touched</th><th>flags</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div></section>"""


def _paper_block(summary):
    """The astrometry paper's own validation numbers, read not recomputed.

    Shown for the field the paper covers.  The point of the table is the SHAPE of
    a bad row, which a single verdict line cannot carry: ``vs anchor`` far from
    zero means the filter does not sit in the same frame as its programme's
    anchor, and ``mode flip`` -- the p60 vs p90 bright-cut offsets disagreeing --
    means the catalog is a mixture of two vintages, which no single-cut number
    reveals.
    """
    if not summary:
        return ''
    bands = summary.get('bands') or {}
    if not bands:
        return '<p class="gcm-empty">No per-band validation records.</p>'
    fresh = {f'{r["program"]}/{r["band"]}': r for r in summary.get('freshness') or []}
    rows = []
    for key in sorted(bands):
        rec = bands[key]
        if 'status' in rec:
            rows.append(f'<tr><td class="filt">{esc(key)}</td>'
                        f'<td colspan="6" style="color:var(--fail)">'
                        f'{esc(rec["status"])}</td></tr>')
            continue
        flip = rec.get('mode_flip_mas')
        anchor = rec.get('vs_anchor')
        flags = []
        if (fresh.get(key) or {}).get('rewritten_since_verdict'):
            flags.append('<span class="gcm-chip fail">rewritten</span>')
        if (fresh.get(key) or {}).get('predates_min_catalog_date'):
            flags.append('<span class="gcm-chip warn">pre-min-date</span>')
        rows.append(f"""<tr>
<td class="filt">{esc(key)}</td>
<td class="n">{'—' if rec.get('vs_virac_p60') is None else f"{rec['vs_virac_p60']:.1f}"}</td>
<td class="n">{'—' if rec.get('vs_virac_p90') is None else f"{rec['vs_virac_p90']:.1f}"}</td>
<td class="n">{'—' if rec.get('contrast_p60') is None else f"{rec['contrast_p60']:.0f}"}</td>
<td class="n" style="{'color:var(--fail);font-weight:600' if (flip or 0) > 10 else ''}">{'—' if flip is None else f'{flip:.1f}'}</td>
<td class="n" style="{'color:var(--fail);font-weight:600' if (anchor or 0) > 30 else ''}">{'—' if anchor is None else f'{anchor:.1f}'}</td>
<td class="z">{esc((rec.get('mtime') or '')[:16])}</td>
<td style="white-space:normal">{''.join(flags)}</td></tr>""")

    cert = summary.get('certifiers') or {}
    cert_bits = ', '.join(f'{esc(k)}={esc(v)}' for k, v in sorted(cert.items())
                          if k != 'table') or 'none recorded'
    generated = summary.get('generated') or '?'
    return f"""
<p class="gcm-note" style="margin:.2rem 0 .5rem">
  From <code>{esc(os.path.basename(summary.get('postrecat_dir') or ''))}/summary.json</code>,
  written {esc(str(generated)[:16])} by the paper's own
  <code>post_recat_validation.py</code>. Its gates (vs-anchor &gt; 30 mas, mode flip
  &gt; 10 mas, degenerate-pair drift ≥ 0.10 mag) are applied there, with the
  sanctioned window-swept offset histogram over the full vetted catalogs —
  nothing on this page recomputes them.</p>
<div class="gcm-scroll"><table class="gcm-t">
<thead><tr><th>program/band</th><th>vs VIRAC p60 (mas)</th><th>p90 (mas)</th>
<th>contrast</th><th>mode flip (mas)</th><th>vs anchor (mas)</th>
<th>catalog written</th><th>flags</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<p class="gcm-note" style="margin:.4rem 0 0;font-size:.74rem">
  Photometric certifiers: {cert_bits}.</p>"""


def _field_section(entry, show_skip=False, figure_base='figures'):
    run = entry['run']
    anchor = entry['anchor']
    label = f"{run['target']} · {run['proposal']}/o{run['obsid']}"
    if run.get('cutout_label'):
        label += f" · cutout {run['cutout_label']}"
    return f"""
<section class="gcm-field sev-{esc(entry['worst'])}" id="{esc(anchor)}">
  <div class="gcm-field-head">
    <h3>{esc(label)}</h3>
    <span class="path">{esc(run['basepath'])}</span>
    <a class="gcm-back" href="#overview">back to overview</a>
  </div>
  <div class="gcm-field-body">
    <div><h4 class="gcm-sub-h">Stage ladder</h4>{_ladder_html(run, with_key=True)}</div>
    <div><h4 class="gcm-sub-h">Products per filter</h4>{_stage_table(run)}</div>
    <div><h4 class="gcm-sub-h">Astrometry — m2 checkpoint</h4>{_astrometry_table(run)}</div>
    <div><h4 class="gcm-sub-h">Provenance</h4>{_provenance_block(run)}</div>
    {f'<div><h4 class="gcm-sub-h">Astrometry paper — release validation</h4>'
     f'{_paper_block(entry.get("paper"))}</div>' if entry.get('paper') else ''}
    <div><h4 class="gcm-sub-h">Queue</h4>{_jobs_block(entry.get('jobs') or [])}</div>
    <div><h4 class="gcm-sub-h">Findings</h4>
      {_checks_block(entry['verdicts'], show_skip, figure_base)}</div>
  </div>
</section>"""


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

_JS = """
(function () {
  var root = document.documentElement;
  var btn = document.getElementById('gcm-theme');
  if (!btn) { return; }
  btn.addEventListener('click', function () {
    var now = root.getAttribute('data-theme');
    var dark = now ? now === 'dark'
      : window.matchMedia('(prefers-color-scheme: dark)').matches;
    root.setAttribute('data-theme', dark ? 'light' : 'dark');
  });
})();
"""


def render_page(entries, cutouts=(), title='JWST-GC pipeline monitor',
                subtitle='', standalone=False, show_skip=False,
                generated=None, unattributed_jobs=(), figure_base='figures'):
    """The whole page.

    ``entries`` is the list built by ``report.build_entries`` -- one per
    observation, already carrying its verdicts, jobs and roll-up.
    """
    generated = generated or time.time()
    total = len(entries)
    n_fail = sum(1 for e in entries if e['worst'] == 'fail')
    n_warn = sum(1 for e in entries if e['worst'] == 'warn')
    n_ok = total - n_fail - n_warn
    n_jobs = sum(len([j for j in (e.get('jobs') or [])
                      if j.get('state') in ('RUNNING', 'PENDING')]) for e in entries)

    cards = ''.join(_card(e) for e in entries)
    details = ''.join(_field_section(e, show_skip, figure_base)
                      for e in entries)

    unattr = ''
    if unattributed_jobs:
        names = ', '.join(sorted({j['name'] for j in unattributed_jobs}))
        unattr = (f'<p class="gcm-note">{len(unattributed_jobs)} queued job(s) could '
                  f'not be attributed to a registered field: <code>{esc(names)}</code>. '
                  f'They are listed here rather than folded into a field\'s count.</p>')

    stamp = time.strftime('%Y-%m-%d %H:%M %Z', time.localtime(generated))
    body = f"""
<div class="gcm">
<header class="gcm-head"><div class="gcm-head-in">
  <h1 class="gcm-title">jwst-gc <span>pipeline monitor</span></h1>
  <span class="gcm-sub">{esc(subtitle or stamp)}</span>
  <div class="gcm-tallies">
    <span class="gcm-tally"><b>{total}</b> runs</span>
    <span class="gcm-tally is-fail"><b>{n_fail}</b> failing</span>
    <span class="gcm-tally is-warn"><b>{n_warn}</b> flagged</span>
    <span class="gcm-tally is-ok"><b>{n_ok}</b> clear</span>
    <span class="gcm-tally"><b>{n_jobs}</b> queued</span>
    <button id="gcm-theme" class="gcm-chip skip" type="button"
            style="cursor:pointer;background:none">theme</button>
  </div>
</div></header>

<div class="gcm-wrap">
<section class="gcm-sec" id="overview"><h2>Overview</h2>
<p class="gcm-note">One card per registered observation. The bar is the stage
ladder in run order — reduction (unc·cal·red·i2d) then cataloging
(m12→m8). Solid = every filter has it, pale = some do, hatched = the product
name cannot be attributed to this observation. Click a card for the detail.</p>
{unattr}
<div class="gcm-grid">{cards}</div></section>

{_cutouts_block(cutouts)}

<section class="gcm-sec"><h2>Detail</h2>{details}</section>

<footer class="gcm-foot">
Generated {esc(stamp)} by <code>python -m jwst_gc_pipeline.monitoring</code>.
Thresholds are imported from the modules that enforce them
(<code>visit_consensus</code>, <code>astrometry_checkpoint</code>,
<code>astrometry_offsets</code>), so this page cannot drift away from the gates it
reports. Astrometric offsets are read from the pipeline's own m2 checkpoint
records — nothing here re-measures an offset.
</footer>
</div>
<script>{_JS}</script>
</div>"""

    if not standalone:
        # The publishing wrapper supplies <!doctype>/<head>/<body>, but takes the
        # page's name from a <title> in the content when one is present.
        return f'<title>{esc(title)}</title><style>{CSS}</style>{body}'
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title><style>{CSS}
html, body {{ margin: 0; padding: 0; background: var(--ground); }}</style>
</head><body>{body}</body></html>"""


def write_html(path, markup):
    """Write ``markup`` to ``path``, creating the directory."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    with open(path, 'w') as fh:
        fh.write(markup)
    return path
