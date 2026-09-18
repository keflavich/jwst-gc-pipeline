"""Relative times on the monitor are recomputed in the browser, not frozen."""
import datetime
import importlib
import json
import os
import re
import shutil
import subprocess

import pytest

render = importlib.import_module('jwst_gc_pipeline.monitoring.render')
schedule_section = importlib.import_module(
    'jwst_gc_pipeline.monitoring.schedule_section')


def test_a_relative_time_carries_the_instant_it_is_relative_to():
    """Without the epoch the browser has nothing to recompute from, and the
    page is left showing the age it had when it was built -- which for a page
    rebuilt hourly and then left open is wrong by however long the tab has
    been sitting there."""
    html = render.ago_html(1_758_000_000)
    assert 'data-epoch="1758000000"' in html
    assert 'class="gcm-ago"' in html
    # the server's own answer stays in the element as the no-JavaScript text
    assert render.ago(1_758_000_000) in html


def test_a_missing_timestamp_stays_a_dash_with_nothing_to_update():
    """`data-epoch=""` would parse as NaN on the client; an em dash with no
    element is what "never" should look like."""
    assert render.ago_html(None) == render.esc(render.ago(None))
    assert 'data-epoch' not in render.ago_html(0)


def test_the_schedule_cell_uses_the_schedule_wording():
    """`in 2 d 4 h` in one column and `2d ago` in another are the same
    quantity in two dialects; the updater is told which one this cell speaks
    rather than picking."""
    when = datetime.datetime(2026, 9, 20, 12, 0, tzinfo=datetime.timezone.utc)
    cell = schedule_section._when_cell(when, 7200)
    assert 'data-style="delta"' in cell
    assert f'data-epoch="{int(when.timestamp())}"' in cell
    assert 'in 2 h 0 m' in cell

    assert schedule_section._when_cell(None, None) == '—'


HARNESS = r"""
// The page's own script, run against a stub DOM with a clock we control.
const CELLS = __CELLS__;
const els = CELLS.map(function (c) {
  return {_text: c.text, _attrs: c.attrs, title: '',
          get textContent() { return this._text; },
          set textContent(v) { this._text = v; },
          getAttribute: function (k) { return this._attrs[k]; }};
});
let VISIBILITY = null;
globalThis.document = {
  hidden: false,
  querySelectorAll: function (sel) {
    return sel.indexOf('gcm-ago') >= 0 ? els : [];
  },
  addEventListener: function (name, fn) {
    if (name === 'visibilitychange') { VISIBILITY = fn; }
  },
  getElementById: function () { return null; },
  documentElement: {getAttribute: function () { return null; },
                    setAttribute: function () {}},
};
let NOW = __NOW__;
globalThis.Date = class extends global.Date {
  constructor(...args) { super(...(args.length ? args : [NOW * 1000])); }
  static now() { return NOW * 1000; }
};
const TIMERS = [];
globalThis.setInterval = function (fn, ms) { TIMERS.push(fn); return TIMERS.length; };

__SCRIPT__

function state() { return els.map(function (e) { return e.textContent; }); }
console.log(JSON.stringify(state()));
NOW += __ADVANCE__;
TIMERS.forEach(function (fn) { fn(); });
console.log(JSON.stringify(state()));
console.log(JSON.stringify(els.map(function (e) { return e.title; })));
"""


def _run(tmp_path, cells, now, advance):
    node = shutil.which('node')
    if node is None:
        pytest.skip('node is not available')
    script = re.search(r'\(function \(\) \{\n  // Relative times.*?\}\)\(\);',
                       render._JS, re.S)
    assert script, 'the relative-time block is no longer identifiable in _JS'
    js = (HARNESS.replace('__SCRIPT__', script.group(0))
                 .replace('__CELLS__', json.dumps(cells))
                 .replace('__NOW__', str(now))
                 .replace('__ADVANCE__', str(advance)))
    path = tmp_path / 'live.js'
    path.write_text(js)
    done = subprocess.run([node, str(path)], capture_output=True, text=True,
                          timeout=30)
    assert done.returncode == 0, done.stderr[-2000:]
    return [json.loads(line) for line in done.stdout.strip().split('\n')]


def test_the_browser_recomputes_the_age_and_keeps_recomputing(tmp_path):
    """The whole point: a tab left open must not keep reporting the age the
    server computed. Three hours later the same cell reads three hours older.
    """
    now = 1_758_000_000
    cells = [{'text': '2h ago', 'attrs': {'data-epoch': str(now - 7200),
                                          'data-style': 'ago'}}]
    first, after, _titles = _run(tmp_path, cells, now, 3 * 3600)
    assert first == ['2h ago']
    assert after == ['5h ago'], after


def test_a_scheduled_visit_becomes_a_past_one_without_a_rebuild(tmp_path):
    """A schedule is where a frozen relative time actively lies: "in 1 h"
    rendered at build time still says "in 1 h" after the visit has run."""
    now = 1_758_000_000
    cells = [{'text': 'in 1 h 0 m', 'attrs': {'data-epoch': str(now + 3600),
                                              'data-style': 'delta'}}]
    first, after, _titles = _run(tmp_path, cells, now, 2 * 3600)
    assert first == ['in 1 h 0 m']
    assert after == ['1 h 0 m ago'], after


def test_each_cell_gains_the_absolute_instant_as_a_tooltip(tmp_path):
    """A relative time with no way to reach the absolute one cannot be checked
    against anything."""
    now = 1_758_000_000
    cells = [{'text': '2h ago', 'attrs': {'data-epoch': str(now - 7200),
                                          'data-style': 'ago'}}]
    _first, _after, titles = _run(tmp_path, cells, now, 60)
    assert titles[0].endswith('Z')
    assert titles[0].startswith('2025-') or titles[0].startswith('20')


def test_a_cell_with_an_unreadable_epoch_is_left_alone(tmp_path):
    """Overwriting it with `NaN ago` would replace a correct server-rendered
    answer with a broken client-rendered one."""
    now = 1_758_000_000
    cells = [{'text': 'never', 'attrs': {'data-epoch': 'not-a-number',
                                         'data-style': 'ago'}}]
    first, after, _titles = _run(tmp_path, cells, now, 3600)
    assert first == ['never'] and after == ['never']
