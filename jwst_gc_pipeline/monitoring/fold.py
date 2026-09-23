"""Collapsed-by-default prose on the monitor page.

Every explanatory paragraph on the page sits behind a disclosure toggle, so
the page opens on its tables, cards and map and the prose is one click away.
The maintainer asked for all paragraphs to start hidden (2026-09-23).
"""

#: Shared by every section; render.CSS appends it.  The marker is drawn here
#: rather than left to the browser so it reads as ``>`` closed and ``v`` open
#: in every engine.
CSS = """
details.gcm-fold { margin: .35rem 0 .6rem; max-width: 68ch; }
details.gcm-fold > summary { cursor: pointer; list-style: none;
  color: var(--text-dim, #8b949e); font-size: .78rem; user-select: none; }
details.gcm-fold > summary::-webkit-details-marker { display: none; }
details.gcm-fold > summary::before { content: '>'; display: inline-block;
  width: 1.1em; font-family: var(--mono, monospace); }
details.gcm-fold[open] > summary::before { content: 'v'; }
details.gcm-fold > summary:hover { color: var(--text, inherit); }
details.gcm-fold > p { margin-top: .3rem; }
"""


def fold(inner, label):
    """Wrap ``inner`` (one or more ``<p>``) in a closed ``<details>``."""
    return (f'<details class="gcm-fold"><summary>{label}</summary>'
            f'{inner}</details>')
