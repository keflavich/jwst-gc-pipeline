"""The slow panner: what it visits, and what it refuses to pan across."""
import importlib.util
import json
import math
import os

import pytest

from jwst_gc_pipeline.quicklook import panner

_SCRIPTS = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', '..', 'scripts', 'quicklook'))


@pytest.fixture(scope='module')
def build():
    spec = importlib.util.spec_from_file_location(
        'build_panner', os.path.join(_SCRIPTS, 'build_panner.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _layers(root, obsids, flavour='vminmax'):
    for obs in obsids:
        d = root / f'GCTreasury_o{obs}_RGB_480-mean-212_{flavour}_hips'
        d.mkdir(parents=True)
        (d / 'properties').write_text('hips_order = 14\n')


def _footprints(path, entries):
    path.write_text(json.dumps({'observed': entries, 'planned': []}))


def test_only_pointings_with_imagery_are_visited(build, tmp_path):
    """Taking the list from the schedule instead would pan the viewer across
    tiles that are observed but not yet rendered, and blank sky in a view that
    moves on its own reads as a broken page."""
    hips = tmp_path / 'pngs'
    _layers(hips, ['127', '128'])
    # o129 is observed and has no rendered layer
    fp = tmp_path / 'footprints.json'
    _footprints(fp, [{'number': n, 'target': f'GC_{n}',
                      'ra': 266.5 + i * 0.01, 'dec': -28.7}
                     for i, n in enumerate(('127', '128', '129'))])

    out = tmp_path / 'site'
    build.main(['--hips-dir', str(hips), '--footprints', str(fp),
                '--out', str(out)])
    tour = json.loads((out / panner.DATA_FILE).read_text())
    assert sorted(s['id'] for s in tour['stops']) == ['o127', 'o128']


def test_a_layer_with_no_centre_is_named_not_dropped_silently(build, tmp_path,
                                                              capsys):
    hips = tmp_path / 'pngs'
    _layers(hips, ['127', '140'])
    fp = tmp_path / 'footprints.json'
    _footprints(fp, [{'number': '127', 'target': 'GC_127',
                      'ra': 266.5, 'dec': -28.7}])
    build.main(['--hips-dir', str(hips), '--footprints', str(fp),
                '--out', str(tmp_path / 'site')])
    assert 'o140' in capsys.readouterr().out


def test_a_far_pointing_is_cut_to_rather_than_panned(build, tmp_path):
    """10678 is not one contiguous block. Panning the ~55' gap to the o040
    group at 2"/s is 27 minutes of empty sky -- most of a loop showing
    nothing."""
    hips = tmp_path / 'pngs'
    _layers(hips, ['127', '128', '040'])
    fp = tmp_path / 'footprints.json'
    _footprints(fp, [
        {'number': '127', 'target': 'GC_127', 'ra': 266.50, 'dec': -28.70},
        {'number': '128', 'target': 'GC_128', 'ra': 266.52, 'dec': -28.70},
        {'number': '040', 'target': 'GC_40', 'ra': 266.39, 'dec': -29.15},
    ])
    out = tmp_path / 'site'
    build.main(['--hips-dir', str(hips), '--footprints', str(fp),
                '--out', str(out)])
    tour = json.loads((out / panner.DATA_FILE).read_text())
    by_id = {s['id']: s for s in tour['stops']}
    # the two adjacent tiles are panned between; the distant one is cut
    assert by_id['o040']['jump'] is True
    assert sum(1 for s in tour['stops'] if s['jump']) == 2
    near = [s for s in tour['stops'] if not s['jump']]
    assert near and all(s['id'] in ('o127', 'o128') for s in near)


def test_consecutive_stops_are_close(build, tmp_path):
    """Greedy nearest-neighbour ordering, asserted by its consequence: the pan
    crosses the gap between consecutive stops in real time, so a tour that
    zig-zags is minutes of blank sky per zig."""
    hips = tmp_path / 'pngs'
    obsids = ['120', '121', '122', '123']
    _layers(hips, obsids)
    fp = tmp_path / 'footprints.json'
    # deliberately out of order on the sky
    ras = {'120': 266.50, '121': 266.56, '122': 266.52, '123': 266.54}
    _footprints(fp, [{'number': n, 'target': f'GC_{n}', 'ra': ras[n],
                      'dec': -28.70} for n in obsids])
    out = tmp_path / 'site'
    build.main(['--hips-dir', str(hips), '--footprints', str(fp),
                '--out', str(out)])
    tour = json.loads((out / panner.DATA_FILE).read_text())
    order = [s['id'] for s in tour['stops']]
    assert order[:4] == ['o120', 'o122', 'o123', 'o121'], order


def test_the_page_points_at_the_fixed_cut_layer(build, tmp_path):
    """A per-field stretch would make a continuous pan read as brightness
    steps at every tile edge."""
    assert panner.SURVEY_URL.endswith('jwst_gc_treasury_vminmax_hips/')
    page = panner.render_page()
    assert 'aladin' in page.lower()
    assert panner.DATA_FILE in page


def test_the_default_field_of_view_is_near_the_pixel_scale():
    """`full zoom` has to mean the data's own resolution: a fov that magnifies
    past it shows interpolation and calls it detail."""
    hips_order, tile = 14, 512
    pixel_deg = 45 / tile / 2 ** hips_order
    across = panner.DEFAULT_FOV / pixel_deg
    assert 1000 <= across <= 2500, across      # a full-window view, 1:1-ish
