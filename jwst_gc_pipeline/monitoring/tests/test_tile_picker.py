"""The 139-tile picker and the execution ledger on the monitor's sky view.

Program 10678 tiles the Galactic Center as GC_1..GC_139.  Two facts about it
could not be read off the page before: which tile is which, and which visits
have actually run.  The second is not a display detail -- the APT reports
``IMPLEMENTATION`` for all 139 visits even now that three have executed and one
was skipped, so a page sourced from the APT alone states 0 observed forever.
"""
import importlib.util
import json
import os
import re

import pytest

_BF = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..',
    'scripts', 'monitoring', 'build_footprints.py'))
_spec = importlib.util.spec_from_file_location('build_footprints_tp', _BF)
bf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bf)


# --- the visit-status table --------------------------------------------------

#: Shaped exactly like the real page, including the two things that broke a
#: naive parse: a header row whose cells survive tag-stripping, and a SKIPPED
#: row that carries six cells instead of eight because it never ran.
_PAGE = """
<table><tbody>
<tr><th>Observation</th><th>Visit</th><th>Status</th><th>Target(s)</th>
    <th>Templates</th><th>Hours</th><th>Start Time (UT)</th><th>End Time (UT)</th></tr>
<tr><td>139</td><td>1</td><td>Executed</td><td>GC_139</td><td>NIRCam Imaging</td>
    <td>1.03</td><td>Sep 11, 2026 06:35:20</td><td>Sep 11, 2026 07:53:48</td></tr>
<tr><td>137</td><td>1</td><td>Executed</td><td>GC_137</td><td>NIRCam Imaging</td>
    <td>0.94</td><td>Sep 11, 2026 11:27:27</td><td>Sep 11, 2026 12:19:33</td></tr>
<tr><td>136</td><td>1</td><td>Skipped</td><td>GC_136</td><td>NIRCam Imaging</td>
    <td>0.99</td></tr>
<tr><td>2</td><td>1</td><td>Flight Ready</td><td>GC_2</td><td>NIRCam Imaging</td>
    <td>1.00</td><td></td><td></td></tr>
</tbody></table>
"""


def _serve(page, monkeypatch):
    class _Resp:
        def read(self):
            return page.encode()
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
    monkeypatch.setattr(bf.urllib.request, 'urlopen',
                        lambda *a, **k: _Resp())


def test_status_table_is_parsed_including_the_short_skipped_row(monkeypatch):
    _serve(_PAGE, monkeypatch)
    got = bf.fetch_visit_status('10678')
    assert set(got) == {'139', '137', '136', '2'}
    assert got['139']['status'] == 'Executed'
    assert got['139']['start'] == 'Sep 11, 2026 06:35:20'
    # six cells: read by position with a length check, not unpacked
    assert got['136']['status'] == 'Skipped'
    assert 'start' not in got['136']
    assert got['136']['hours'] == '0.99'


def test_the_header_row_is_not_read_as_an_observation(monkeypatch):
    """Stripping tags from the header leaves 'Observation'/'Status' text, which
    a row-shaped parse happily stores as a pointing whose status is 'Status'."""
    _serve(_PAGE, monkeypatch)
    got = bf.fetch_visit_status('10678')
    assert all(k.isdigit() for k in got), sorted(got)
    assert 'Status' not in {v['status'] for v in got.values()}


def test_executed_wins_over_another_visit_of_the_same_pointing(monkeypatch):
    """A pointing with several visits: one executed visit means the tile has
    data, and that outranks a sibling still Scheduled."""
    page = _PAGE.replace(
        '</tbody></table>',
        '<tr><td>139</td><td>2</td><td>Scheduled</td><td>GC_139</td>'
        '<td>NIRCam Imaging</td><td>1.0</td><td>x</td><td>y</td></tr>'
        '</tbody></table>')
    _serve(page, monkeypatch)
    assert bf.fetch_visit_status('10678')['139']['status'] == 'Executed'


def test_skipped_outranks_scheduled_but_not_executed():
    r = bf._status_rank
    assert r('Executed') > r('Skipped') > r('Scheduled') > r('Flight Ready')
    assert r('nonsense') == -1


def test_a_failed_fetch_is_not_fatal_and_says_so(monkeypatch, capsys):
    """The footprint build must not depend on a website being up."""
    def _boom(*a, **k):
        raise OSError('no route to host')
    monkeypatch.setattr(bf.urllib.request, 'urlopen', _boom)
    assert bf.fetch_visit_status('10678') == {}
    assert 'could not fetch visit status' in capsys.readouterr().err


# --- the rendered panel ------------------------------------------------------

def _footprints(statuses=None, n=5):
    """A minimal footprint file: `statuses` maps tile number -> status."""
    statuses = statuses or {}
    planned, observed = [], []
    for i in range(1, n + 1):
        rec = {'target': f'GC_{i}', 'number': str(i),
               'ra': 266.0 + i * 0.01, 'dec': -29.0,
               'filters': {}, 'dithered': True,
               'nircam': [[[266.0 + i * 0.01, -29.0], [266.01 + i * 0.01, -29.0],
                           [266.01 + i * 0.01, -28.99], [266.0 + i * 0.01, -28.99]]],
               'miri': []}
        if str(i) in statuses:
            rec['status'] = statuses[str(i)]
        (observed if rec.get('status') == 'Executed' else planned).append(rec)
    counts = {}
    for v in statuses.values():
        counts[v] = counts.get(v, 0) + 1
    out = {'program': '10678', 'title': 'test', 'pa_v3': 87.0,
           'pa_v3_range': [79.0, 95.0], 'n_planned': len(planned),
           'n_observed': len(observed), 'aces': [],
           'planned': planned, 'observed': observed}
    if statuses:
        out['status_counts'] = counts
        out['status_source'] = 'stsci-visit-status'
    return out


def _section(fp):
    from jwst_gc_pipeline.monitoring import skyview
    return skyview.section(fp)


def test_every_tile_is_selectable_and_the_options_are_in_numeric_order():
    """String order puts 10 before 2, which for a 139-tile survey makes the
    picker unusable."""
    html = _section(_footprints(n=12))
    opts = re.findall(r'<option value="(\d+)">', html)
    assert opts == [str(i) for i in range(1, 13)]
    assert 'all 12 tiles' in html


def test_a_tile_carries_its_number_target_and_status_in_the_option():
    html = _section(_footprints({'3': 'Executed', '4': 'Skipped'}))
    assert '3 — GC_3 — Executed' in html
    assert '4 — GC_4 — Skipped' in html


def test_executed_and_skipped_tiles_are_listed_by_number():
    html = _section(_footprints({'3': 'Executed', '5': 'Executed',
                                 '4': 'Skipped'}))
    ledger = re.search(r'Execution</div><div class="gcm-sky-stat">(.*?)</div></div>',
                       html, re.S).group(1)
    assert 'Executed 2' in ledger and 'Skipped 1' in ledger
    assert re.search(r'data-goto="3"', ledger)
    assert re.search(r'data-goto="4"', ledger)
    assert re.search(r'data-goto="5"', ledger)


def test_an_executed_tile_is_still_in_the_picker_though_it_moved_layers():
    """`planned` and `observed` are a display split, not an identity -- a tile
    must stay findable by number whichever list it lands in."""
    html = _section(_footprints({'2': 'Executed'}))
    assert '<option value="2">2 — GC_2 — Executed</option>' in html


def test_with_no_status_source_the_ledger_says_unavailable_not_zero():
    """'Executed 0' asserts a measurement that was never made; the APT reports
    IMPLEMENTATION for every visit and cannot distinguish the two."""
    html = _section(_footprints())
    assert 'visit status unavailable' in html
    assert 'Executed 0' not in html


def test_the_page_names_the_source_it_actually_used():
    live = _section(_footprints({'1': 'Executed'}))
    assert "STScI's per-visit status table" in live
    apt = _section(_footprints())
    assert 'reports IMPLEMENTATION for every visit' in apt


def test_static_polygons_are_keyed_by_observation_number_not_index():
    """The picker highlights `g[data-obs=N]`.  Keying on the array index breaks
    the moment one pointing is undrawable, silently shifting every tile after
    it onto its neighbour's outline."""
    html = _section(_footprints(n=4))
    assert sorted(re.findall(r'<g data-obs="(\d+)">', html)) == \
        sorted(['1', '2', '3', '4'])


def test_the_highlight_overrides_the_layer_presentation_attributes():
    """Stroke and colour are set as presentation attributes on the parent <g>,
    which beat a plain class rule -- the highlight did nothing without
    !important."""
    from jwst_gc_pipeline.monitoring import skyview
    rule = re.search(r'\.gcm-tile-hi path \{([^}]*)\}', skyview.CSS).group(1)
    assert rule.count('!important') >= 2


def test_the_selected_tile_overlay_is_exempt_from_the_layer_toggles():
    """`pick` has no toggle, so `on.pick` is undefined and the generic
    show/hide loop would hide the highlight on the next toggle click."""
    html = _section(_footprints({'1': 'Executed'}))
    assert "if (k === 'pick') { return; }" in html
