"""The slow panner: what it visits, and what it refuses to pan across."""
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess

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


@pytest.fixture(autouse=True)
def _no_served_hips(build, tmp_path, monkeypatch):
    """The build reads the served coadds' properties; left at its default it
    would read whatever is in the live web root of the machine running the
    tests.  A test that wants a served coadd passes --hips-root."""
    monkeypatch.setattr(build, 'DEFAULT_HIPS_ROOT',
                        str(tmp_path / 'no-served-hips'))


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


def test_a_half_written_layer_is_not_a_stop(build, tmp_path):
    """`reproject_to_hips` writes tiles first and `properties` last, so a layer
    mid-build is indistinguishable from a finished one by directory name.

    This is not hypothetical: on 2026-09-16 the coadd directory held 37
    matching layers and 36 with `properties`, because the hourly cron was
    rebuilding one. Panning into a half-written pyramid is exactly the blank
    sky the page exists to avoid.
    """
    hips = tmp_path / 'pngs'
    _layers(hips, ['127', '128'])
    half = hips / 'GCTreasury_o129_RGB_480-mean-212_vminmax_hips' / 'Norder3'
    half.mkdir(parents=True)                       # tiles, but no properties
    (half / 'Npix1.png').write_bytes(b'x')

    fp = tmp_path / 'footprints.json'
    _footprints(fp, [{'number': n, 'target': f'GC_{n}',
                      'ra': 266.5 + i * 0.01, 'dec': -28.7}
                     for i, n in enumerate(('127', '128', '129'))])
    out = tmp_path / 'site'
    build.main(['--hips-dir', str(hips), '--footprints', str(fp),
                '--out', str(out)])
    tour = json.loads((out / panner.DATA_FILE).read_text())
    assert sorted(s['id'] for s in tour['stops']) == ['o127', 'o128']


def test_a_tour_with_nothing_to_pan_is_refused_before_it_is_written(build,
                                                                    tmp_path):
    """A tour whose every leg is a cut has nothing to watch, and the page's
    skip loop would spin through it inside requestAnimationFrame. The builder
    used to write both files and THEN raise on the summary line, publishing a
    tour nobody could watch while looking like it had failed to make one."""
    hips = tmp_path / 'pngs'
    _layers(hips, ['127', '128'])
    fp = tmp_path / 'footprints.json'
    _footprints(fp, [{'number': '127', 'target': 'GC_127',
                      'ra': 266.50, 'dec': -28.70},
                     {'number': '128', 'target': 'GC_128',
                      'ra': 266.52, 'dec': -28.70}])
    out = tmp_path / 'site'
    with pytest.raises(SystemExit) as caught:
        build.main(['--hips-dir', str(hips), '--footprints', str(fp),
                    '--out', str(out), '--max-leg', '0'])
    assert 'max-leg' in str(caught.value)
    # nothing was published
    assert not (out / panner.DATA_FILE).exists()
    assert not (out / panner.PAGE_FILE).exists()


NODE_HARNESS = r"""
// Enough of a browser for step() to run: the script's three globals, plus a
// self-returning `chain` so the trailing fetch(...).then(...).catch(...) is
// syntactically whole without doing anything.
var calls = 0;
var chain = {then: function () { return chain; },
             catch: function () { return chain; }};
globalThis.fetch = function () { return chain; };
globalThis.document = {getElementById: function () {
  return {addEventListener: function () {}, textContent: '', innerHTML: '',
          classList: {toggle: function () {}}};
}};
globalThis.requestAnimationFrame = function () {};      // step() re-arms; ignore
globalThis.A = {init: chain};
var DATA_URL = 'tour.json';   // the page defines this beside the script

__SCRIPT__

// Every stop a cut: the state the builder now refuses to write, and the one
// the page must survive if a tour file is hand-edited.
TOUR = {survey: 'x', fov: 0.01, rate: 2,
        stops: [{ra: 0, dec: 0, label: 'a', jump: true},
                {ra: 0.1, dec: 0, label: 'b', jump: true},
                {ra: 0.2, dec: 0, label: 'c', jump: true}]};
aladin = {gotoRaDec: function () { calls++; }};
playing = true;
last = 0;
step(1000);                       // must RETURN, not spin
console.log('returned');
"""


def test_an_all_cut_tour_does_not_hang_the_page(tmp_path):
    """The page's skip loop advances until it finds a leg to pan. With every
    stop flagged there is none, and an unbounded loop spins inside
    requestAnimationFrame -- a frozen tab with nothing in the console.

    Run rather than read: the version of this test that asserted the ABSENCE
    of `while (ends[0].jump)` passed on `while (ends[0].jump && true)` with the
    bound gone and the explanatory text surviving in a comment.
    """
    node = shutil.which('node')
    if node is None:
        pytest.skip('node is not available')
    script = tmp_path / 'harness.js'
    script.write_text(NODE_HARNESS.replace('__SCRIPT__', panner._SCRIPT))
    done = subprocess.run([node, str(script)], capture_output=True, text=True,
                          timeout=20)
    assert done.returncode == 0, done.stderr[-2000:]
    assert 'returned' in done.stdout


def test_nothing_is_painted_over_the_viewer_controls():
    """The caption used to be a banner pinned to the top of the viewer.

    Aladin puts its fullscreen button and its coordinate readout in the top
    corners and offers no way to move them, so an overlay anchored to `top:0`
    covers controls the reader needs.  The rule is structural rather than a
    check for the old class name: any overlay the page adds belongs at the
    bottom, with the buttons.
    """
    blocks = re.findall(r'([^{}]+)\{([^{}]*)\}', panner.CSS)
    overlays = [(sel.strip(), body) for sel, body in blocks
                if 'position:absolute' in body.replace(' ', '')]
    assert overlays, 'no absolutely-positioned rule at all -- CSS restructured?'
    for sel, body in overlays:
        flat = body.replace(' ', '').replace('\n', '')
        if sel == '#sky':
            continue                      # the viewer itself, under everything
        assert 'bottom:0' in flat, f'{sel} is not anchored to the bottom'
        assert 'top:0' not in flat, f'{sel} still reaches the top of the viewer'


def test_the_caption_rides_the_control_bar():
    page = panner.render_page()
    bar = page[page.index('<div class=bar>'):]
    bar = bar[:bar.index('<script')]
    assert 'back to the release' in bar
    assert 'F212N + F480M' in bar
    assert 'class=head' not in page
    # the field menu the script fills lives there too, beside pause and skip
    assert '<select id=field' in bar


DOM_HARNESS = r"""
// A document that keeps its elements, so what the script writes into them can
// be read back and a change event can be fired at the one the reader uses.
var els = {}, gotos = [];
function el(id) {
  if (!els[id]) {
    els[id] = {id: id, innerHTML: '', textContent: '', value: '', on: {},
               addEventListener: function (ev, fn) { this.on[ev] = fn; },
               classList: {toggle: function () {}}};
  }
  return els[id];
}
var chain = {then: function () { return chain; },
             catch: function () { return chain; }};
globalThis.fetch = function () { return chain; };
globalThis.document = {getElementById: el, activeElement: null};
globalThis.requestAnimationFrame = function () {};
globalThis.A = {init: chain};
var DATA_URL = 'tour.json';

__SCRIPT__

TOUR = {survey: 'x', fov: 0.01, rate: 2, stops: [
  {id: 'o128', label: 'GC_128', ra: 266.52, dec: -28.70, jump: false},
  {id: 'o040', label: 'GC_40', ra: 266.39, dec: -29.15, jump: true},
  {id: 'o127', label: 'GC_127', ra: 266.50, dec: -28.70, jump: false}]};
aladin = {gotoRaDec: function (ra, dec) { gotos.push([ra, dec]); }};
fillFields();
console.log(JSON.stringify({options: els.field.innerHTML}));

__BODY__
"""


def _dom(tmp_path, body):
    node = shutil.which('node')
    if node is None:
        pytest.skip('node is not available')
    script = tmp_path / 'dom.js'
    script.write_text(DOM_HARNESS.replace('__SCRIPT__', panner._SCRIPT)
                      .replace('__BODY__', body))
    done = subprocess.run([node, str(script)], capture_output=True, text=True,
                          timeout=20)
    assert done.returncode == 0, done.stderr[-2000:]
    return [json.loads(line) for line in done.stdout.strip().splitlines()]


def test_the_field_menu_lists_every_stop(tmp_path):
    """37 fields and no way to reach a named one: the tour visits them in the
    order it pans, so finding o127 meant waiting for it or clicking skip until
    it came round."""
    out = _dom(tmp_path, "console.log(JSON.stringify({done: true}));")
    options = out[0]['options']
    assert options.count('<option') == 3
    for name in ('GC_127', 'GC_128', 'GC_40'):
        assert name in options
    # sorted by field id, not by the tour's spatial order (o128, o040, o127)
    assert options.index('GC_40') < options.index('GC_127') < options.index('GC_128')


def test_choosing_a_field_moves_the_pan_to_it(tmp_path):
    """The value carries the TOUR index, so a selection is the same operation
    the skip button performs -- picking o127 must leave the pan mid-tour at
    o127, not merely recentre the viewer while the leg counter says otherwise.
    """
    out = _dom(tmp_path, """
      var idx = /value="(\\d+)">GC_127/.exec(els.field.innerHTML)[1];
      els.field.value = idx;
      els.field.on.change.call(els.field);
      console.log(JSON.stringify({idx: Number(idx), leg: leg, along: along,
                                  went: gotos[gotos.length - 1],
                                  playing: playing}));
    """)
    got = out[1]
    assert got['leg'] == got['idx'] == 2
    assert got['along'] == 0
    assert got['went'] == [266.50, -28.70]
    assert got['playing'] is True


def test_choosing_a_cut_field_holds_there(tmp_path):
    """o040 sits ~55' from the rest and the leg out of it is a cut, crossed
    instantly.  Left playing, the view would slide off the field the reader
    just asked for inside a couple of frames, which reads as the menu not
    working."""
    out = _dom(tmp_path, """
      var idx = /value="(\\d+)">GC_40/.exec(els.field.innerHTML)[1];
      els.field.value = idx;
      els.field.on.change.call(els.field);
      console.log(JSON.stringify({playing: playing, leg: leg,
                                  went: gotos[gotos.length - 1],
                                  play_label: els.play.textContent}));
    """)
    got = out[1]
    assert got['playing'] is False, 'the pan left the field that was chosen'
    assert got['went'] == [266.39, -29.15]
    assert 'Play' in got['play_label'], 'the button still offers to pause'


def test_the_menu_follows_the_pan(tmp_path):
    """Read as well as write: a menu that shows o040 while the view is over
    o127 is worse than none."""
    out = _dom(tmp_path, """
      playing = true; leg = 0; along = 0; last = 0;
      step(1000);
      console.log(JSON.stringify({value: els.field.value,
                                  where: els.where.innerHTML}));
    """)
    got = out[1]
    assert got['value'] == '0', 'the menu did not track the leg being panned'
    assert 'h' in got['where'] and '"' in got['where']


def test_the_menu_is_not_rewritten_while_it_is_open(tmp_path):
    """Setting `value` under a reader who is choosing moves the highlight out
    from under them."""
    out = _dom(tmp_path, """
      els.field.value = '2';
      document.activeElement = els.field;
      playing = true; leg = 0; along = 0; last = 0;
      step(1000);
      console.log(JSON.stringify({value: els.field.value}));
    """)
    assert out[1]['value'] == '2'


def test_o138_and_o139_are_left_off_the_tour_by_default(build, tmp_path,
                                                        capsys):
    """The maintainer asked for the panner to skip o138 and o139, the same
    pair the survey mosaics exclude.  They are rendered, so only the exclude
    list keeps them off; ``--exclude`` with no values restores them."""
    hips = tmp_path / 'pngs'
    _layers(hips, ['127', '128', '138', '139'])
    fp = tmp_path / 'footprints.json'
    _footprints(fp, [{'number': n, 'target': f'GC_{n}',
                      'ra': 266.5 + i * 0.01, 'dec': -28.7}
                     for i, n in enumerate(('127', '128', '138', '139'))])

    out = tmp_path / 'site'
    build.main(['--hips-dir', str(hips), '--footprints', str(fp),
                '--out', str(out)])
    tour = json.loads((out / panner.DATA_FILE).read_text())
    assert sorted(s['id'] for s in tour['stops']) == ['o127', 'o128']
    assert 'excluded from the tour: o138, o139' in capsys.readouterr().out

    everything = tmp_path / 'all'
    build.main(['--hips-dir', str(hips), '--footprints', str(fp),
                '--out', str(everything), '--exclude'])
    tour = json.loads((everything / panner.DATA_FILE).read_text())
    assert sorted(s['id'] for s in tour['stops']) == ['o127', 'o128',
                                                       'o138', 'o139']


def _miri_layers(root, obsids, family='MIRI_F770W'):
    for obs in obsids:
        d = root / f'GCTreasury_o{obs}_{family}_hips'
        d.mkdir(parents=True)
        (d / 'properties').write_text('hips_order = 12\n')


def _square(ra, dec, half=0.02):
    return json.dumps([[[ra - half, dec - half], [ra - half, dec + half],
                        [ra + half, dec + half], [ra + half, dec - half]]])


def _with_miri(n, i, step=0.01):
    """A footprint entry whose MIRI parallel sits 7' north of the pointing,
    as the real ones sit several arcminutes from theirs."""
    ra, dec = 266.5 + i * step, -28.7
    return {'number': n, 'target': f'GC_{n}', 'ra': ra, 'dec': dec,
            'miri': _square(ra, dec + 7 / 60)}


def test_every_product_gets_its_own_tour(build, tmp_path):
    """MIRI is a parallel several arcminutes from its NIRCam pointing, so
    panning the MIRI layer along the NIRCam tour would show blank sky most of
    the way.  Each product's stops come from its own layers and footprint."""
    hips = tmp_path / 'pngs'
    _layers(hips, ['127', '128', '129'])
    _miri_layers(hips, ['127', '128'])
    fp = tmp_path / 'footprints.json'
    _footprints(fp, [_with_miri(n, i)
                     for i, n in enumerate(('127', '128', '129'))])
    out = tmp_path / 'site'
    build.main(['--hips-dir', str(hips), '--footprints', str(fp),
                '--out', str(out), '--products', 'nircam', 'miri'])
    data = json.loads((out / panner.DATA_FILE).read_text())
    by_key = {p['key']: p for p in data['products']}
    assert list(by_key) == ['nircam', 'miri']

    nircam, miri = by_key['nircam'], by_key['miri']
    assert sorted(s['id'] for s in miri['stops']) == ['o127', 'o128']
    for stop in miri['stops']:
        # the mean vertex direction of a square, a few mas off its middle
        assert stop['dec'] == pytest.approx(-28.7 + 7 / 60, abs=1e-4)
    assert miri['survey'].endswith('/jwst_gc_treasury_miri_hips/')
    # order 12 is shown 4x wider than order 14, and panned 4x faster, so the
    # sky crosses the screen at the same speed
    assert miri['fov'] == pytest.approx(4 * nircam['fov'])
    assert miri['rate'] / miri['fov'] == pytest.approx(
        nircam['rate'] / nircam['fov'])

    # the top level is the default product, which is what a page written
    # before the menu existed reads
    assert data['survey'] == nircam['survey'] == panner.SURVEY_URL
    assert data['stops'] == nircam['stops']
    assert data['fov'] == nircam['fov']


def test_a_product_with_nothing_to_show_is_left_off_the_menu(build, tmp_path,
                                                            capsys):
    hips = tmp_path / 'pngs'
    _layers(hips, ['127', '128'])
    fp = tmp_path / 'footprints.json'
    _footprints(fp, [_with_miri(n, i) for i, n in enumerate(('127', '128'))])
    out = tmp_path / 'site'
    build.main(['--hips-dir', str(hips), '--footprints', str(fp),
                '--out', str(out), '--products', 'nircam', 'miri'])
    data = json.loads((out / panner.DATA_FILE).read_text())
    assert [p['key'] for p in data['products']] == ['nircam']
    assert 'miri: left off the menu' in capsys.readouterr().out


def test_single_module_layers_count_as_imagery(build, tmp_path):
    """A tile with one module finished is rendered as
    GCTreasury_o063_nrca_... and is in the coadd like any other."""
    hips = tmp_path / 'pngs'
    _layers(hips, ['127'], flavour='residual_vminmax')
    d = hips / 'GCTreasury_o128_nrca_RGB_480-mean-212_residual_vminmax_hips'
    d.mkdir(parents=True)
    (d / 'properties').write_text('hips_order = 14\n')
    _layers(hips, ['127', '128'])
    fp = tmp_path / 'footprints.json'
    _footprints(fp, [_with_miri(n, i) for i, n in enumerate(('127', '128'))])
    out = tmp_path / 'site'
    build.main(['--hips-dir', str(hips), '--footprints', str(fp),
                '--out', str(out), '--products', 'nircam', 'nircam-starless'])
    data = json.loads((out / panner.DATA_FILE).read_text())
    starless = [p for p in data['products'] if p['key'] == 'nircam-starless']
    assert sorted(s['id'] for s in starless[0]['stops']) == ['o127', 'o128']


def _served(root, name, order, positions):
    """A served HiPS at ``root/name`` with a tile at each (ra, dec)."""
    astropy_healpix = pytest.importorskip('astropy_healpix')
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    d = root / name
    d.mkdir(parents=True)
    (d / 'properties').write_text(f'hips_order = {order}\nhips_frame = '
                                  'galactic\nhips_tile_format = png\n')
    for ra, dec in positions:
        at = SkyCoord(ra * u.deg, dec * u.deg).galactic
        ipix = int(astropy_healpix.lonlat_to_healpix(at.l, at.b, 2 ** order,
                                                     order='nested'))
        tile = (d / f'Norder{order}' / f'Dir{ipix // 10000 * 10000}'
                / f'Npix{ipix}.png')
        tile.parent.mkdir(parents=True, exist_ok=True)
        tile.write_bytes(b'x')


def _composite_setup(tmp_path, in_rgb, under_nircam):
    """Three MIRI pointings 3' apart, so each is on its own order-12 tile
    (~0.9'); the composite has tiles at ``in_rgb`` and the NIRCam coadd at
    ``under_nircam`` (indices into the three)."""
    hips = tmp_path / 'pngs'
    _layers(hips, ['127', '128', '129'])
    _miri_layers(hips, ['127', '128', '129'])
    entries = [_with_miri(n, i, step=0.05)
               for i, n in enumerate(('127', '128', '129'))]
    fp = tmp_path / 'footprints.json'
    _footprints(fp, entries)
    miri = [(266.5 + i * 0.05, -28.7 + 7 / 60) for i in range(3)]
    root = tmp_path / 'avm'
    _served(root, 'gctreasury_mosaic_RGB_770-480-212_hips', 12,
            [miri[i] for i in in_rgb])
    _served(root, 'jwst_gc_treasury_vminmax_hips', 14,
            [miri[i] for i in under_nircam])
    return hips, fp, root


def _composite_stops(build, tmp_path, hips, fp, root):
    out = tmp_path / 'site'
    build.main(['--hips-dir', str(hips), '--footprints', str(fp),
                '--out', str(out), '--hips-root', str(root),
                '--products', 'nircam', 'rgb'])
    data = json.loads((out / panner.DATA_FILE).read_text())
    composite = [p for p in data['products'] if p['key'] == 'rgb'][0]
    return sorted(s['id'] for s in composite['stops'])


def test_the_composite_visits_only_pointings_it_has_tiles_for(build, tmp_path,
                                                             capsys):
    """The MIRI + NIRCam composite has no per-field layers, and is rebuilt
    separately from the coadds, so a MIRI pointing newer than it has no
    imagery in it.  Its tour keeps a pointing only where the served HiPS has
    a tile."""
    hips, fp, root = _composite_setup(tmp_path, in_rgb=[0, 1],
                                      under_nircam=[0, 1, 2])
    assert _composite_stops(build, tmp_path, hips, fp, root) == ['o127', 'o128']
    assert 'o129' in capsys.readouterr().out


def test_the_composite_skips_parallels_with_no_nircam_under_them(build,
                                                                tmp_path,
                                                                capsys):
    """A MIRI parallel sits several arcminutes off its own NIRCam pointing,
    and where no other pointing's NIRCam covers it the composite shows its
    red channel alone.  On the served build of 2026-09-27 that was 33 of the
    64 stops."""
    hips, fp, root = _composite_setup(tmp_path, in_rgb=[0, 1, 2],
                                      under_nircam=[1])
    assert _composite_stops(build, tmp_path, hips, fp, root) == ['o128']
    out = capsys.readouterr().out
    assert 'nothing from jwst_gc_treasury_vminmax_hips under 2' in out
    assert 'o127, o129' in out


def test_the_zoom_follows_the_coadds_own_order(build, tmp_path):
    """The page shows the coadd, so a coadd built coarser than its per-field
    layers must be opened at its own order: Aladin asked for tiles deeper
    than a layer has can show nothing at all."""
    hips = tmp_path / 'pngs'
    _layers(hips, ['127', '128'])                 # per-field layers: order 14
    fp = tmp_path / 'footprints.json'
    _footprints(fp, [_with_miri(n, i) for i, n in enumerate(('127', '128'))])
    root = tmp_path / 'avm'
    (root / 'jwst_gc_treasury_vminmax_hips').mkdir(parents=True)
    (root / 'jwst_gc_treasury_vminmax_hips' / 'properties').write_text(
        'hips_order = 13\n')
    out = tmp_path / 'site'
    build.main(['--hips-dir', str(hips), '--footprints', str(fp),
                '--out', str(out), '--hips-root', str(root),
                '--products', 'nircam'])
    data = json.loads((out / panner.DATA_FILE).read_text())
    assert data['fov'] == pytest.approx(2 * panner.DEFAULT_FOV)

    # no coadd on this disk: the per-field order stands in
    out = tmp_path / 'site2'
    build.main(['--hips-dir', str(hips), '--footprints', str(fp),
                '--out', str(out), '--products', 'nircam'])
    data = json.loads((out / panner.DATA_FILE).read_text())
    assert data['fov'] == pytest.approx(panner.DEFAULT_FOV)


def test_every_served_product_is_on_the_menu():
    """The request was for the stars-subtracted layers, the MIRI layers, the
    MIRI + NIRCam composite and each stretch; the fixed-cut NIRCam layer
    stays the default."""
    keys = [p['key'] for p in panner.PRODUCTS]
    assert keys[0] == 'nircam'
    assert panner.HIPS_BASE + panner.PRODUCTS[0]['hips'] + '/' == panner.SURVEY_URL
    assert len(set(keys)) == len(keys)
    hips = {p['hips'] for p in panner.PRODUCTS}
    for wanted in ('jwst_gc_treasury_hips', 'jwst_gc_treasury_log_hips',
                   'jwst_gc_treasury_residual_vminmax_hips',
                   'jwst_gc_treasury_residual_log_hips',
                   'jwst_gc_treasury_miri_hips',
                   'jwst_gc_treasury_miri_bgmatch_hips',
                   'jwst_gc_treasury_miri_residual_hips',
                   'gctreasury_mosaic_RGB_770-480-212_hips'):
        assert wanted in hips
    page = panner.render_page()
    bar = page[page.index('<div class=bar>'):page.index('<script')]
    assert '<select id=product' in bar


PRODUCT_BODY = r"""
var sets = [], fovs = [];
A.HiPS = function (url, opts) { return {url: url, name: opts.name}; };
aladin.setBaseImageLayer = function (layer) { sets.push(layer.url); };
aladin.setFoV = function (fov) { fovs.push(fov); };
var nircamStops = TOUR.stops;
PRODUCTS = [
  {key: 'nircam', label: 'NIRCam, fixed cuts', note: 'N', survey: 'u/n/',
   fov: 0.01, rate: 2, stops: nircamStops},
  {key: 'nircam-log', label: 'NIRCam, log', note: 'L', survey: 'u/l/',
   fov: 0.01, rate: 2, stops: nircamStops.map(function (s) {
     return Object.assign({}, s); })},
  {key: 'miri', label: 'MIRI F770W', note: 'M', survey: 'u/m/', fov: 0.04,
   rate: 8, stops: [
     {id: 'o040', label: 'GC_40', ra: 266.40, dec: -29.05, jump: false},
     {id: 'o127', label: 'GC_127', ra: 266.51, dec: -28.58, jump: false}]}];
TOUR = PRODUCTS[0];
fillProducts();
"""


def test_choosing_a_product_swaps_the_layer_and_the_tour(tmp_path):
    """MIRI covers different sky from NIRCam: the menu has to change the
    layer, the zoom, and the stops together, and land on the MIRI tile
    nearest the view rather than back at the start of the tour."""
    out = _dom(tmp_path, PRODUCT_BODY + """
      here = [266.50, -28.60];                  // over o127's MIRI parallel
      els.product.value = 'miri';
      els.product.on.change.call(els.product);
      console.log(JSON.stringify({key: TOUR.key, sets: sets, fovs: fovs,
                                  leg: leg, went: gotos[gotos.length - 1],
                                  fields: els.field.innerHTML,
                                  what: els.what.innerHTML,
                                  products: els.product.innerHTML,
                                  hidden: els.productpick.hidden}));
    """)
    got = out[1]
    assert got['products'].count('<option') == 3
    assert got['hidden'] is False
    assert got['key'] == 'miri'
    assert got['sets'] == ['u/m/']
    assert got['fovs'] == [0.04]
    assert got['leg'] == 1, 'did not land on the MIRI tile nearest the view'
    assert got['went'] == [266.51, -28.58]
    assert got['fields'].count('<option') == 2
    assert got['what'] == 'M'


def test_changing_the_stretch_keeps_the_pan_where_it_is(tmp_path):
    """The NIRCam stretches are the same tiles, so switching between them is a
    change of look: the view must not jump back to a tile centre."""
    out = _dom(tmp_path, PRODUCT_BODY + """
      leg = 2; along = 0.3; var before = gotos.length;
      els.product.value = 'nircam-log';
      els.product.on.change.call(els.product);
      console.log(JSON.stringify({key: TOUR.key, sets: sets, leg: leg,
                                  along: along, moved: gotos.length - before,
                                  field: els.field.value}));
    """)
    got = out[1]
    assert got['key'] == 'nircam-log'
    assert got['sets'] == ['u/l/']
    assert (got['leg'], got['along'], got['moved']) == (2, 0.3, 0)
    assert got['field'] == '2'


def test_a_tour_file_from_before_the_menu_still_pans(tmp_path):
    """The page and the tour file are deployed separately; a new page reading
    an old file must still work, with nothing to choose between."""
    out = _dom(tmp_path, """
      PRODUCTS = productsOf({survey: 'u/n/', fov: 0.01, rate: 2,
                             stops: TOUR.stops});
      TOUR = PRODUCTS[0];
      fillProducts();
      console.log(JSON.stringify({n: PRODUCTS.length, survey: TOUR.survey,
                                  hidden: els.productpick.hidden}));
    """)
    assert out[1] == {'n': 1, 'survey': 'u/n/', 'hidden': True}


def test_a_layer_that_fails_to_load_leaves_the_page_on_the_old_one(tmp_path):
    """If Aladin refuses the new layer the old one stays on screen, so the
    menu, the field list, and the caption must go back to describing it."""
    out = _dom(tmp_path, PRODUCT_BODY + """
      A.HiPS = function () { throw new Error('no such survey'); };
      els.product.value = 'miri';
      els.product.on.change.call(els.product);
      console.log(JSON.stringify({key: TOUR.key, sets: sets,
                                  product: els.product.value,
                                  fields: els.field.innerHTML,
                                  what: els.what.innerHTML,
                                  where: els.where.textContent}));
    """)
    got = out[1]
    assert got['key'] == 'nircam'
    assert got['sets'] == []
    assert got['product'] == 'nircam'
    assert got['fields'].count('<option') == 3
    assert got['what'] == 'N'
    assert 'MIRI F770W failed to load' in got['where']
    assert 'no such survey' in got['where']
