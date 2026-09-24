"""Every generated Aladin Lite viewer shows the full control set.

Aladin Lite v3 hides the settings and coordinate-grid controls by default, and
the settings menu (Settings > Grid) is the only place a reader can change the
grid colour.  The viewers each passed their own hand-picked ``show*Control``
flags, so the pages differed from one another and the grid colour could be
changed on none of them.  They now all
build their options from ``jwst_gc_pipeline.aladin_controls``.
"""

import importlib.util
import json
import os
import subprocess

import pytest

from jwst_gc_pipeline.aladin_controls import ALADIN_CONTROLS, aladin_controls_js

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_every_control_is_on_and_the_layer_list_starts_folded():
    shown = {k: v for k, v in ALADIN_CONTROLS.items() if k.startswith('show')}
    assert shown and all(shown.values())
    assert ALADIN_CONTROLS['showCooGridControl'] is True
    assert ALADIN_CONTROLS['showSettingsControl'] is True  # grid colour
    # the layer list opens on click instead of covering the map at load
    assert ALADIN_CONTROLS['expandLayersControl'] is False


def test_controls_js_is_brace_free_and_rejects_unknown_names():
    js = aladin_controls_js()
    assert '{' not in js and '}' not in js
    assert 'showCooGridControl: true' in js
    assert 'showSettingsControl: true' in js
    assert 'showStatusBar: false' in aladin_controls_js(showStatusBar=False)
    with pytest.raises(KeyError):
        aladin_controls_js(showGridColourControl=True)


def _generators_building_aladin():
    """Tracked, non-test Python files that construct an Aladin view."""
    files = subprocess.run(['git', 'ls-files', '*.py'], cwd=REPO, check=True,
                           capture_output=True, text=True).stdout.split()
    out = []
    for rel in files:
        if '/tests/' in rel or rel.startswith('tests/'):
            continue
        with open(os.path.join(REPO, rel), encoding='utf-8') as fh:
            if 'A.aladin(' in fh.read():
                out.append(rel)
    return out


def test_every_aladin_generator_uses_the_shared_control_set():
    gens = _generators_building_aladin()
    # the five viewers at the time of writing; a new one joins automatically
    assert len(gens) >= 5, gens
    missing = [g for g in gens
               if 'aladin_controls' not in open(os.path.join(REPO, g),
                                                encoding='utf-8').read()]
    assert not missing, f'Aladin viewers without the shared controls: {missing}'


def _ctor_objects(page):
    """The JSON control set as a page embeds it (cmdview/panner style)."""
    return 'const ALADIN_CONTROLS = ' + json.dumps(ALADIN_CONTROLS) in page


def test_cmd_explorer_page_carries_the_controls():
    from jwst_gc_pipeline.quicklook import cmdview
    from jwst_gc_pipeline.quicklook.tests.test_cmdview import _data
    page = cmdview.render(_data())
    assert _ctor_objects(page)
    assert "A.aladin('#sky', Object.assign({}, ALADIN_CONTROLS" in page


def test_panner_page_carries_the_controls():
    from jwst_gc_pipeline.quicklook import panner
    page = panner.render_page()
    assert _ctor_objects(page)
    assert "A.aladin('#sky', Object.assign({}, ALADIN_CONTROLS" in page
    # the tour used to hide these; they are no longer special-cased
    assert 'showZoomControl: false' not in page


def test_release_index_overview_carries_the_controls():
    path = os.path.join(REPO, 'scripts', 'release', 'field_overview.py')
    spec = importlib.util.spec_from_file_location('field_overview_ctl', path)
    fo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fo)
    geom = {'field': 'brick', 'href': 'brick.html',
            'polys': [[(266.50, -28.75), (266.58, -28.75),
                       (266.58, -28.65), (266.50, -28.65)]]}
    out = fo.section([geom])
    ctor = out.split('A.aladin(')[1].split(');')[0]
    assert aladin_controls_js() in ctor
    assert 'showFullscreenControl: false' not in ctor
