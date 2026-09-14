"""The page: what it says, and the two bits of its drawing maths that are
easy to get backwards and invisible from Python."""
import json
import re

import pytest

from jwst_gc_pipeline.quicklook import cmdview


def _data(**over):
    data = {
        'built': '2026-09-14 12:00 UTC',
        'bands': {'blue': 'f212n', 'red': 'f480m'},
        'mag_system': 'vega',
        'match_arcsec': 0.1,
        'labels': {'x': 'F212N − F480M (Vega)', 'y': 'F212N (Vega)'},
        'grid': {'extent': [0, 5, 9, 22], 'nx': 60, 'radius': 0.00962},
        'all': {'cells': [[0, 0, 3]], 'max': 3, 'n': 3},
        'fields': [{'id': 'o132', 'label': 'GC_132', 'source': 'jicama',
                    'n': 3, 'max': 3, 'cells': [[0, 0, 3]],
                    'polys': [[[266.5, -28.7], [266.6, -28.7],
                               [266.6, -28.6], [266.5, -28.6]]],
                    'centre': [266.55, -28.65], 'files': {}}],
        'incomplete': {},
        'centre': [266.55, -28.65],
        'fov': 1.4,
    }
    data.update(over)
    return data


def test_the_page_points_at_its_own_data_file():
    html = cmdview.render(_data(), 'cmd_explorer_data.json')
    assert '"cmd_explorer_data.json"' in html
    assert 'fetch(DATA_URL)' in html


def test_the_page_says_the_magnitudes_are_vega():
    """Not every input catalog is in Vega, so the page states the system it
    plots rather than leaving the reader to assume one."""
    html = cmdview.render(_data())
    assert 'Vega' in html
    assert 'F212N &minus; F480M' in html
    assert 'Magnitudes are <b>Vega</b>' in html
    assert not re.search(r'\bAB\b', html), 'the page must not offer AB anywhere'


def test_pointings_short_a_band_are_named_on_the_page():
    html = cmdview.render(_data(incomplete={'o133': ['f480m']}))
    assert 'o133' in html and 'F480M' in html


def test_no_waiting_block_when_nothing_is_waiting():
    assert 'Waiting on a band' not in cmdview.render(_data())


def test_the_magnitude_axis_is_inverted():
    """A colour-magnitude diagram runs brighter-upward. ``toPix`` maps v to
    ``b.y + v * b.h`` -- if that ever gains a ``1 - v`` the giant branch is
    drawn upside down and nothing else in the suite would notice."""
    src = cmdview._SCRIPT
    body = src[src.index('function toPix'):src.index('function hexPath')]
    assert 'b.y + v * b.h' in body
    assert '1 - v' not in body


def test_the_field_overlay_is_drawn_after_the_background():
    """Grey underneath, viridis on top: reversing them hides the selected
    pointing behind the sample it is meant to be compared against."""
    src = cmdview._SCRIPT
    body = src[src.index('function drawCMD'):src.index('function show')]
    grey_at = body.index('DATA.all.cells')
    field_at = body.index('field.cells')
    assert grey_at < field_at


def test_the_hips_layers_are_the_same_urls_the_main_viewer_uses():
    html = cmdview.render(_data())
    for _, url, _ in cmdview.HIPS_LAYERS:
        assert url in html
    layers = json.loads(re.search(r'const HIPS = (\[.*?\]);', html, re.S).group(1))
    assert [L['id'] for L in layers] == [i for i, _, _ in cmdview.HIPS_LAYERS]


def test_a_file_url_gets_an_explanation_rather_than_a_blank_canvas():
    assert "location.protocol === 'file:'" in cmdview.render(_data())


# Only the keys ``render`` itself reads; the rest are the browser's, and a
# missing one of those shows up in test_builder_output_has_what_the_page_reads.
@pytest.mark.parametrize('key', ['grid', 'fields', 'bands'])
def test_render_needs_the_builder_to_have_supplied_the_whole_payload(key):
    data = _data()
    del data[key]
    with pytest.raises(KeyError):
        cmdview.render(data)
