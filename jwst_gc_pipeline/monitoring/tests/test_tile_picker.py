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
               # A real pointing always has BOTH: MIRI rides as the coordinated
               # parallel of every NIRCam observation.  An empty list here made
               # the MIRI layers draw nothing and the split look half-broken.
               'miri': [[[266.0 + i * 0.01, -28.9], [266.01 + i * 0.01, -28.9],
                         [266.01 + i * 0.01, -28.89], [266.0 + i * 0.01, -28.89]]]}
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
    """Both are listed and both are reachable, but by different routes:
    an executed tile links OUT to its data in MAST, a skipped one has no data
    to link to and instead selects itself on the map."""
    html = _section(_footprints({'3': 'Executed', '5': 'Executed',
                                 '4': 'Skipped'}))
    ledger = re.search(r'Execution</div><div class="gcm-sky-stat">(.*?)</div></div>',
                       html, re.S).group(1)
    assert 'Observed 2' in ledger and 'Skipped 1' in ledger
    assert re.search(r'observtn=3&', ledger)
    assert re.search(r'observtn=5&', ledger)
    assert re.search(r'data-goto="4"', ledger)
    # every listed tile is reachable one way or the other
    assert len(re.findall(r'<a [^>]*>', ledger)) == 3


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
    # Each tile appears once per instrument layer, so the SET is what identifies
    # the tiles; the count is checked by the partition test below.
    assert set(re.findall(r'<g data-obs="(\d+)">', html)) == {'1', '2', '3', '4'}


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


# --- status colouring on the map ---------------------------------------------

def _layer_counts(html):
    """{static layer id: number of pointings drawn in it}."""
    out = {}
    for gid in ('stat-nircam', 'stat-miri', 'stat-sch-nircam', 'stat-sch-miri',
                'stat-skp-nircam', 'stat-skp-miri',
                'stat-obs-nircam', 'stat-obs-miri'):
        m = re.search(r'<g id="%s"[^>]*>(.*?)(?=<g id=)' % gid, html, re.S)
        out[gid] = len(re.findall(r'<g data-obs=', m.group(1))) if m else 0
    return out


def test_scheduled_and_skipped_are_drawn_in_their_own_colours():
    html = _section(_footprints({'1': 'Scheduled', '2': 'Scheduled',
                                 '3': 'Skipped', '4': 'Executed'}, n=6))
    n = _layer_counts(html)
    assert n['stat-sch-nircam'] == 2 and n['stat-sch-miri'] == 2
    assert n['stat-skp-nircam'] == 1 and n['stat-skp-miri'] == 1
    assert n['stat-obs-nircam'] == 1
    # 5 and 6 have no status, so they stay in the plain planned layer
    assert n['stat-nircam'] == 2


def test_every_pointing_lands_in_exactly_one_nircam_layer():
    """The four layers partition the field.  Drawing a tile twice would double
    its outline's opacity and quietly misreport the counts beside the toggles."""
    html = _section(_footprints({'1': 'Scheduled', '2': 'Skipped',
                                 '3': 'Executed'}, n=9))
    n = _layer_counts(html)
    total = (n['stat-nircam'] + n['stat-sch-nircam']
             + n['stat-skp-nircam'] + n['stat-obs-nircam'])
    assert total == 9
    assert len(re.findall(r'<g data-obs="1"', html)) == 2   # NIRCam + MIRI, once each


def test_the_four_status_colours_are_distinct():
    from jwst_gc_pipeline.monitoring import skyview
    colours = {skyview.COLOR_NIRCAM_PLANNED, skyview.COLOR_SCHEDULED,
               skyview.COLOR_SKIPPED, skyview.COLOR_OBSERVED}
    assert len(colours) == 4


def test_an_unknown_or_absent_status_still_draws_as_planned():
    """A footprint file built before statuses existed carries none at all, and
    a status this code has not seen must not make a tile vanish from the map."""
    html = _section(_footprints({'1': 'Withdrawn', '2': ''}, n=3))
    n = _layer_counts(html)
    assert n['stat-nircam'] == 3
    assert n['stat-sch-nircam'] == 0 and n['stat-skp-nircam'] == 0


def test_the_status_layers_have_toggles_that_start_on():
    html = _section(_footprints({'1': 'Scheduled', '2': 'Skipped'}))
    for lid in ('lyr-scheduled', 'lyr-skipped'):
        assert re.search(r'class="gcm-sky-btn on" id="%s"' % lid, html), lid
    assert "scheduled: true" in html and "skipped: true" in html
    assert "scheduled: ['stat-sch-nircam', 'stat-sch-miri']" in html
    assert "skipped: ['stat-skp-nircam', 'stat-skp-miri']" in html


def test_the_counts_beside_the_toggles_come_from_what_was_drawn():
    """Recomputing them in `section` from the raw file is a second count of the
    same thing, and the two drift the first time the split rule changes."""
    html = _section(_footprints({'1': 'Scheduled', '2': 'Scheduled',
                                 '3': 'Skipped'}, n=8))
    assert re.search(r'id="lyr-scheduled".*?<span[^>]*>2</span>', html, re.S)
    assert re.search(r'id="lyr-skipped".*?<span[^>]*>1</span>', html, re.S)


def test_the_legend_says_what_scheduled_and_skipped_mean():
    """'skipped' and 'not yet observed' are easy to conflate and mean opposite
    things for coverage."""
    import html as H
    text = H.unescape(_section(_footprints({'1': 'Scheduled', '2': 'Skipped'})))
    assert 'scheduled — has a date, not yet run' in text
    assert 'skipped — was scheduled and did not run' in text


def test_aladin_routes_the_same_three_ways_as_the_static_map():
    html = _section(_footprints({'1': 'Scheduled'}))
    assert "scheduled: layer('JWST scheduled'" in html
    assert "skipped: layer('JWST skipped'" in html
    assert "(st === 'scheduled') ? 'scheduled'" in html


# --- MAST links on the executed tiles ----------------------------------------

def test_only_executed_tiles_link_to_the_archive():
    """A scheduled visit has no data in MAST and a skipped one never will
    without re-planning, so linking them sends a reader to an empty result --
    which reads as the archive having lost something rather than as the visit
    not having happened."""
    html = _section(_footprints({'1': 'Executed', '2': 'Scheduled',
                                 '3': 'Skipped'}, n=4))
    ledger = re.search(r'Execution</div><div class="gcm-sky-stat">(.*?)</div></div>',
                       html, re.S).group(1)
    assert len(re.findall(r'href="https://mast[^"]*"', ledger)) == 1
    # the skipped tile is still listed, still selects its tile, still no link
    assert 'data-goto="3"' in ledger
    assert not re.search(r'href="https://mast[^"]*observtn=3&', ledger)


def test_the_link_names_the_program_and_the_observation():
    from jwst_gc_pipeline.monitoring import skyview
    url = skyview.mast_url('10678', '137')
    assert 'program_id=10678' in url and 'observtn=137' in url
    # the executed-RESULTS route, not the blank search form: the bare
    # `#/jwst?...` form opens an unexecuted query
    assert '#/jwst/results?' in url and 'useStore=false' in url


def test_a_missing_or_junk_number_yields_no_link_rather_than_a_broken_one():
    from jwst_gc_pipeline.monitoring import skyview
    for prog, obs in (('10678', None), ('10678', ''), (None, '3'),
                      ('10678', 'GC_3')):
        assert skyview.mast_url(prog, obs) is None


def test_the_executed_footprints_themselves_are_clickable():
    """Both instruments of an executed tile, and nothing else."""
    html = _section(_footprints({'1': 'Executed', '2': 'Executed',
                                 '3': 'Scheduled'}, n=5))
    wrapped = re.findall(
        r'<a href="https://mast[^"]*observtn=(\d+)[^"]*" target="_blank" '
        r'rel="noopener"><g data-obs=', html)
    assert sorted(wrapped) == ['1', '1', '2', '2']   # NIRCam + MIRI each


def test_the_tile_picker_does_not_swallow_clicks_on_the_mast_links():
    """The ledger's numbers used to be goto shortcuts and the handler calls
    preventDefault; if it bound to every anchor it would stop the archive
    links from navigating."""
    html = _section(_footprints({'1': 'Executed'}))
    assert ".gcm-sky-stat a[data-goto]'" in html


# --- the controls panel is draggable -----------------------------------------

def test_the_panel_has_a_drag_handle():
    """The controls sit ON the map, so wherever they are they hide part of it.
    The header is the handle rather than the whole panel: dragging from the
    body would fight the buttons and the tile dropdown inside it."""
    html = _section(_footprints({'1': 'Executed'}))
    assert 'id="gcm-sky-grip"' in html
    assert 'drag to move this panel' in html
    from jwst_gc_pipeline.monitoring import skyview
    assert '.gcm-sky-ui h4 { cursor: move' in skyview.CSS


def test_dragging_clears_the_css_right_anchor():
    """The panel is anchored with `right` in CSS.  Setting `left` without
    clearing it pins BOTH edges, and the panel stretches instead of moving."""
    html = _section(_footprints({'1': 'Executed'}))
    assert "panel.style.right = 'auto'" in html


def test_the_panel_cannot_be_dragged_out_of_reach():
    """`.gcm-sky-wrap` is `overflow: hidden`, so a panel dropped past an edge
    is clipped, not scrolled to -- it could never be dragged back.  The clamp
    keeps a strip of it inside, and re-runs on resize so a narrowing window
    cannot strand it either."""
    html = _section(_footprints({'1': 'Executed'}))
    assert 'function clampTo' in html
    assert "addEventListener('resize'" in html


def test_a_touch_drag_moves_the_panel_rather_than_scrolling_the_page():
    from jwst_gc_pipeline.monitoring import skyview
    assert 'touch-action: none' in skyview.CSS


def test_the_map_does_not_pan_underneath_a_drag():
    """Aladin and the static map both pan on pointer drags of their own."""
    html = _section(_footprints({'1': 'Executed'}))
    assert 'ev.stopPropagation();' in html


# --- the status LIFECYCLE, not a single value --------------------------------

def test_a_visit_that_has_run_is_observed_whatever_stage_it_reached():
    """STScI walks a visit along Collecting -> Executed -> Archived.  Reading
    only 'Executed' demotes the ones that have progressed PAST it: the observed
    count went 12 -> 7 over two hours on 2026-09-12 with no visit un-running,
    because five had become Archived (data in MAST, the strongest state) or
    Collecting."""
    for status in ('Executed', 'Archived', 'Collecting',
                   'executed', 'ARCHIVED'):
        assert bf.has_run(status), status
    for status in ('Scheduled', 'Flight Ready', 'Skipped', '', None):
        assert not bf.has_run(status), status


def test_the_two_files_agree_on_which_statuses_mean_observed():
    """`build_footprints` decides which tiles go in the observed layer and
    `skyview` decides which get a MAST link; they are in different packages and
    would drift silently."""
    from jwst_gc_pipeline.monitoring import skyview
    assert set(skyview.RUN_STATUSES) == set(bf.OBSERVED_STATUSES)


def test_the_furthest_stage_wins_for_a_multi_visit_pointing():
    r = bf._status_rank
    assert r('Archived') > r('Executed') > r('Collecting') > r('Skipped') \
        > r('Scheduled') > r('Flight Ready')


def test_every_run_stage_is_listed_and_linked():
    html = _section(_footprints({'1': 'Archived', '2': 'Collecting',
                                 '3': 'Executed', '4': 'Skipped'}, n=6))
    ledger = re.search(r'Execution</div><div class="gcm-sky-stat">(.*?)</div></div>',
                       html, re.S).group(1)
    assert 'Observed 3' in ledger
    assert len(re.findall(r'href="https://mast', ledger)) == 3


def test_the_total_is_not_labelled_with_one_of_its_parts():
    """'Executed 12' when 3 of them are Archived reads as a count of Executed."""
    html = _section(_footprints({'1': 'Archived', '2': 'Executed'}))
    ledger = re.search(r'Execution</div><div class="gcm-sky-stat">(.*?)</div></div>',
                       html, re.S).group(1)
    assert 'Observed 2' in ledger
    assert '1 archived' in ledger and '1 executed' in ledger


def test_the_status_tail_does_not_re_count_the_observed_tiles():
    """Excluding only 'executed' from the tail listed Archived and Collecting
    again underneath the total, so the same tiles were counted twice on one
    line and the numbers did not add to the tile count."""
    import html as H
    page = _section(_footprints({'1': 'Archived', '2': 'Collecting',
                                 '3': 'Executed', '4': 'Skipped',
                                 '5': 'Scheduled'}, n=6))
    ledger = H.unescape(re.search(
        r'Execution</div><div class="gcm-sky-stat">(.*?)</div></div>',
        page, re.S).group(1))
    tail = ledger.split('Skipped')[-1]
    assert 'archived' not in tail.lower() and 'collecting' not in tail.lower()


# --- the two treasury imagery layers -----------------------------------------

def test_miri_is_the_background_and_nircam_rides_on_top():
    """They are drawn TOGETHER, not chosen between: the MIRI parallel sits
    ~7.5' from its NIRCam prime, so they are different sky and the useful view
    is both at once.  MIRI is the background (behind); NIRCam is an overlay
    image layer (in front)."""
    from jwst_gc_pipeline.monitoring import skyview
    name, url, note = skyview.SURVEYS[0]
    assert 'miri' in url.lower(), 'the MIRI layer is the background'
    assert skyview.TREASURY_NIRCAM_HIPS not in [u for _n, u, _t in skyview.SURVEYS], \
        'NIRCam must not also be a background CHOICE -- it is always on top'
    html = _section(_footprints({'1': 'Executed'}))
    assert 'setOverlayImageLayer' in html


def test_both_treasury_layers_are_on_by_default():
    html = _section(_footprints({'1': 'Executed'}))
    assert re.search(r'class="gcm-sky-btn survey on"[^>]*data-survey="[^"]*miri', html)
    assert 'class="gcm-sky-btn on" id="lyr-nircam-hips"' in html
    assert 'var nircamOn = true;' in html


def test_switching_the_background_does_not_lose_the_nircam_layer():
    """`setImageSurvey` replaces the base and can clear the overlay stack, so
    picking 2MASS would silently drop the survey's own data."""
    html = _section(_footprints({'1': 'Executed'}))
    switch = html.index('function applySurvey(')
    after = html.index('aladin.setImageSurvey(', switch)
    assert html.index('applyNircamHips();', after) < html.index('}', after) + 400


def test_an_aladin_without_overlay_support_does_not_break_the_panel():
    """The background, the footprints and the tile picker do not depend on the
    overlay; an unguarded throw here would take their wiring down with it."""
    html = _section(_footprints({'1': 'Executed'}))
    block = html[html.index('function applyNircamHips'):]
    assert 'try {' in block[:400] and 'catch' in block[:900]


def test_the_layers_are_served_from_a_cors_enabled_host():
    """Aladin reads tiles into a WebGL texture, so a host with no
    Access-Control-Allow-Origin renders nothing.  data.rc.ufl.edu also answers
    401 on this path, which is why both layers are published under avm_images
    rather than linked where they were built."""
    from jwst_gc_pipeline.monitoring import skyview
    for url in [skyview.TREASURY_NIRCAM_HIPS, skyview.SURVEYS[0][1]]:
        assert url.startswith('https://starformation.astro.ufl.edu/avm_images/')
        assert 'data.rc.ufl.edu' not in url
