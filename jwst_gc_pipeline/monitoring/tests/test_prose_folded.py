"""Every explanatory paragraph on the monitor starts folded away.

The maintainer asked for all paragraphs on the monitor page to be hidden by
default behind a ``> / v`` disclosure.  A paragraph added later without the
fold would sit open on the page and nothing else would notice, so this renders
each part of the page that carries prose and checks every ``gcm-note`` /
``gcm-sky-foot`` paragraph is inside a CLOSED ``details.gcm-fold``.
"""
import re

from jwst_gc_pipeline.monitoring import render, skyview
from jwst_gc_pipeline.monitoring.tests.test_card_labels_and_status_names import (
    _footprints)
from jwst_gc_pipeline.monitoring.tests.test_schedule_wiring import _sched

_PROSE = re.compile(r'<p class="gcm-(?:note|sky-foot)"')
_OPEN = re.compile(r'<details class="gcm-fold"( open)?>')


def _unfolded(page):
    """Offsets of prose paragraphs not inside a closed gcm-fold."""
    bad = []
    for m in _PROSE.finditer(page):
        before = page[:m.start()]
        opener = None
        for o in _OPEN.finditer(before):
            opener = o
        closed_since = (opener is None
                        or '</details>' in before[opener.end():])
        if closed_since or opener.group(1):
            # A paragraph holding only controls (the map's buttons) is the
            # one sanctioned exception: folding it would hide the buttons.
            tail = page[m.end():page.find('</p>', m.end())]
            if not tail.lstrip('>').lstrip().startswith('<button'):
                bad.append(page[m.start():m.start() + 80])
    return bad


def test_front_page_prose_is_folded():
    page = render.render_page([], schedule=_sched(), include_skyview=False,
                              standalone=True)
    assert _PROSE.search(page)
    assert _unfolded(page) == []


def test_sky_view_prose_is_folded_and_its_buttons_are_not():
    html = skyview.section(_footprints(Implementation=[62, 14]))
    assert _unfolded(html) == []
    assert 'id="gcm-sky-load"' in html
    assert html.index('id="gcm-sky-load"') > html.rindex('</details>', 0,
                                                         html.index('id="gcm-sky-load"'))


def test_the_fold_marker_reads_chevron_closed_and_v_open():
    assert "content: '>'" in render.CSS
    assert "[open] > summary::before { content: 'v'; }" in render.CSS
