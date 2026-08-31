"""RGPS footprints on the monitor's sky view, and the geometry behind them.

Two separable things are tested here:

* the **generator** (``scripts/monitoring/build_rgps_footprints.py``), which
  turns the RGPS survey definition's Galactic regions into ICRS rings;
* the **sky view**, which draws them as three context layers and no longer
  draws Roman GBTDS's autumn season.

The generator's box sampling is worth pinning in particular.  A Galactic box is
a curve in ICRS, so its edges are sampled rather than cornered, and the first
version of that code built the two latitude edges as ``(b, l)`` and swapped the
result back -- emitting ``(b, l)`` for half of every box.  It produced polygons
whose vertices alternated between RA 160 and RA 350 while still being the right
*number* of vertices, so nothing but a contiguity check would have caught it.
"""
import importlib.util
import json
import math
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
# .../jwst_gc_pipeline/monitoring/tests -> repo root
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
_GEN = os.path.join(_REPO, 'scripts', 'monitoring', 'build_rgps_footprints.py')


@pytest.fixture(scope='module')
def gen():
    spec = importlib.util.spec_from_file_location('build_rgps_footprints', _GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: A miniature survey definition in the real file's shape: the same region
#: repeated across filters (which is how the real one is written), one box and
#: one pointing.
_SPEC = {
    'wide_area': {
        'F129': [{'l': [-67.0, 50.1], 'b': [-2.0, 2.0], 'name': 'Disk'}],
        'F158': [{'l': [-67.0, 50.1], 'b': [-2.0, 2.0], 'name': 'Disk'}],
    },
    'time_domain': {
        'F146': [{'l': [0.06, 2.64], 'b': [-0.53, 0.28],
                  'name': 'TDS_Galactic_Center_Q1'}],
    },
    'deep_spec': {
        'F129': [{'pointing': [28.83, 3.54, 0.3], 'name': 'Deep_W40'}],
    },
}


def test_a_region_repeated_across_filters_is_one_region(gen):
    """The definition lists each region once per observing filter with identical
    geometry.  Drawing it once per filter would stack five copies of the same
    outline and report five regions where the survey has one."""
    out = gen.build(_SPEC)
    assert [r['name'] for r in out['wide_area']] == ['Disk']
    assert out['wide_area'][0]['filters'] == ['F129', 'F158']


def test_box_edges_are_sampled_not_cornered(gen):
    """A 117-degree Galactic box is a curve in ICRS; four corners would draw a
    straight line through sky the survey does not cover."""
    out = gen.build(_SPEC, edge_step=2.0)
    disk = out['wide_area'][0]
    assert disk['shape'] == 'box'
    # 117 deg of longitude at 2 deg steps, twice, plus the two short edges.
    assert len(disk['poly']) > 100


def test_every_polygon_is_contiguous(gen):
    """The coordinate-swap regression: no vertex may jump across the sky.

    Each step along a ring is bounded by the sampling step (plus a little for
    the ICRS projection); a swapped ``(b, l)`` vertex shows up here as a jump of
    tens of degrees, which is what this asserts cannot happen.
    """
    out = gen.build(_SPEC, edge_step=2.0)
    for component, regions in out.items():
        for region in regions:
            poly = region['poly']
            for i, here in enumerate(poly):
                there = poly[(i + 1) % len(poly)]
                dra = abs(here[0] - there[0])
                dra = min(dra, 360.0 - dra)
                step = math.hypot(dra * math.cos(math.radians(here[1])),
                                  here[1] - there[1])
                assert step < 8.0, (
                    '%s/%s: %.1f deg jump between consecutive vertices'
                    % (component, region['name'], step))


def test_the_galactic_centre_field_lands_on_the_galactic_centre(gen):
    """A conversion that silently kept Galactic degrees would still produce
    plausible-looking numbers; this pins that they are ICRS."""
    out = gen.build(_SPEC)
    tds = out['time_domain'][0]
    ras = [v[0] for v in tds['poly']]
    decs = [v[1] for v in tds['poly']]
    # Sgr A* is 266.417, -29.008; Q1 is the quadrant just east/north of it.
    assert 264.0 < min(ras) < 268.0, ras[:3]
    assert -31.0 < min(decs) < -26.0, decs[:3]


def test_a_region_with_no_geometry_is_skipped_not_guessed(gen):
    """``ready_for_use`` entries with neither a box nor a pointing exist in the
    real file; inventing a footprint for them would draw sky nobody surveyed."""
    out = gen.build({'wide_area': {'F129': [
        {'l': [], 'b': [], 'name': 'Placeholder'},
        {'l': [10.0, 12.0], 'b': [-1.0, 1.0], 'name': 'Real'},
    ]}})
    assert [r['name'] for r in out['wide_area']] == ['Real']


# ---- the sky view ----------------------------------------------------------

_FP = {'program': '10678', 'title': 't', 'n_planned': 1, 'n_observed': 0,
       'pa_v3': 87.0, 'pa_v3_range': [79.0, 95.0],
       'planned': [{'number': 1, 'target': 'x',
                    'nircam': [[[266.4, -29.1], [266.5, -29.1],
                                [266.5, -29.0], [266.4, -29.0]]]}]}

_RGPS = {'components': {
    'wide_area': [{'name': 'Disk', 'poly': [[266.3, -29.8], [266.9, -29.8],
                                            [266.9, -29.3]]}],
    'time_domain': [{'name': 'TDS', 'poly': [[266.4, -29.2], [266.7, -29.2],
                                             [266.7, -28.9]]}],
    'deep_spec': [{'name': 'Deep', 'poly': [[266.5, -29.5], [266.6, -29.5],
                                            [266.6, -29.4]]}],
}}


def test_each_rgps_component_is_its_own_layer():
    """Wide-area, time-domain and deep fields differ in cadence and depth, so
    they are three surveys rather than three drawings of one."""
    from jwst_gc_pipeline.monitoring import skyview
    svg, info = skyview.static_map(_FP, None, rgps=_RGPS)
    assert info['n_rgps'] == 3
    assert info['n_rgps_by'] == {'wide_area': 1, 'time_domain': 1,
                                 'deep_spec': 1}
    for group in ('stat-rgps-wide', 'stat-rgps-tds', 'stat-rgps-deep'):
        assert 'id="%s"' % group in svg


def test_rgps_toggles_reach_the_static_layers():
    """The Roman buttons were once wired only to Aladin, so they did nothing
    until 1.8 MB of script loaded.  RGPS must not repeat that."""
    from jwst_gc_pipeline.monitoring import skyview
    html = skyview.section(_FP, None, rgps=_RGPS)
    for lid in ('rgps-wide', 'rgps-tds', 'rgps-deep'):
        assert 'id="lyr-%s"' % lid in html          # the button exists
        assert "'stat-%s'" % lid in html            # ...and owns a static group
        assert "['lyr-%s', '%s']" % (lid, lid) in html   # ...and is wired


def test_rgps_is_off_by_default():
    """This is a JWST monitor; RGPS is context, on the same terms as Roman."""
    from jwst_gc_pipeline.monitoring import skyview
    html = skyview.section(_FP, None, rgps=_RGPS)
    state = html[html.index('var on = {'):]
    state = state[:state.index('}')]
    for lid in ('rgps-wide', 'rgps-tds', 'rgps-deep'):
        assert "'%s': false" % lid in state, lid


def test_a_missing_rgps_file_leaves_empty_layers_not_a_crash():
    """The file is generated separately and may not be there; the page must
    still render, with the toggles present and reading zero."""
    from jwst_gc_pipeline.monitoring import skyview
    svg, info = skyview.static_map(_FP, None, rgps=None)
    assert info['n_rgps'] == 0
    assert 'id="stat-rgps-wide"' in svg
    html = skyview.section(_FP, None, rgps=None)
    assert 'id="lyr-rgps-wide"' in html


def test_roman_autumn_is_no_longer_drawn():
    """Spring and autumn are the same tiles at two roll angles, so the second
    layer covered nearly the same sky twice for another colour and toggle."""
    from jwst_gc_pipeline.monitoring import skyview
    roman = {'tiles': {'T1': {
        'spring': [[[266.5, -29.6], [266.6, -29.6], [266.6, -29.5]]],
        'autumn': [[[266.7, -29.6], [266.8, -29.6], [266.8, -29.5]]]}}}
    svg, info = skyview.static_map(_FP, roman)
    assert info['n_roman'] == 1                     # spring only
    assert 'stat-autumn' not in svg
    html = skyview.section(_FP, roman)
    assert 'lyr-autumn' not in html
    assert 'autumn' not in html.lower()


def test_the_autumn_data_is_kept_even_though_it_is_not_drawn():
    """Dropping the layer is a display decision; it does not require the file to
    be regenerated, and reversing it should not either."""
    from jwst_gc_pipeline.monitoring import skyview
    roman = {'tiles': {'T1': {
        'spring': [[[266.5, -29.6], [266.6, -29.6], [266.6, -29.5]]],
        'autumn': [[[266.7, -29.6], [266.8, -29.6], [266.8, -29.5]]]}}}
    assert skyview._roman_polys(roman) == roman['tiles']['T1']['spring']
    assert roman['tiles']['T1']['autumn'], 'the source data must be untouched'
