"""The offsets table on the release page, and the recipe it publishes."""
import ast
import importlib.util
import json
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REL = os.path.normpath(os.path.join(_HERE, '..', '..', '..', 'scripts', 'release'))


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_REL, f'{name}.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def mw():
    import sys
    sys.path.insert(0, _REL)
    return _load('make_webpage')


def _summary(tmp_path, **over):
    data = {
        'table_file': 'Offsets_JWST_Brick10678_consensus.csv',
        'table_rows': 898, 'table_mtime': '2026-09-16T11:43Z',
        'per_filter': [{'observation': 'o127', 'filter': 'F212N', 'n': 18,
                        'dra_median_arcsec': -0.0312,
                        'ddec_median_arcsec': 0.0455,
                        'dra_span_mas': 31.0, 'ddec_span_mas': 12.0,
                        'stages': ['m2']}],
        'frames': {'totals': {'corrected': 1340, 'uncorrected': 838,
                              'has_row': 1038, 'no_row': 1140},
                   'per_obs': {}, 'unreadable': 0},
    }
    data.update(over)
    (tmp_path / 'offsets_summary.json').write_text(json.dumps(data))
    return tmp_path


def test_the_page_says_which_frames_already_carry_a_correction(mw, tmp_path):
    """`fix_alignment` bakes the applied shift into RAOFFSET/DEOFFSET and is
    idempotent on a frame that has one, so a release cut mid-pass holds both
    states. Telling a reader to apply the table to every frame double-corrects
    the ones already done -- by their full shift, the largest error available
    to make here."""
    html = mw._offsets_section('gc-treasury', _summary(tmp_path))
    assert 'Do not apply this table blindly' in html
    assert '1,340' in html and '838' in html
    assert 'RAOFFSET' in html and 'DEOFFSET' in html
    assert 'MINUS' in html


def test_the_page_says_how_many_frames_have_no_row_yet(mw, tmp_path):
    """Half the frames have no measured row on any given day. A reader who
    assumes full coverage gives those a neighbour's shift."""
    html = mw._offsets_section('gc-treasury', _summary(tmp_path))
    assert '1,038' in html and '1,140' in html
    assert 'not measured yet' in html


def test_a_release_without_a_summary_renders_nothing(mw, tmp_path):
    """The section is skipped rather than guessed at: a page that invents an
    offsets table is worse than one that omits it."""
    assert mw._offsets_section('gc-treasury', tmp_path) == ''


def test_the_recipe_uses_the_pipeline_row_matcher(mw):
    """Selecting a row is not a four-column lookup: Exposure and Module narrow
    only when more than one row still matches, Module matches `nrcb3` or
    `nrcb`, and Vgroup narrows ALWAYS -- a visit can dither across disjoint
    tiles with the exposure number restarting, so (visit, exposure) can name a
    different pointing and match exactly one row while doing it.

    A hand-rolled lookup in published instructions is the defect
    `locked_row_match`'s own docstring warns about, and the first draft of this
    recipe had it: tested against real frames it resolved 14 of 25, and its
    failures were silent `0 matching rows` rather than wrong shifts only by
    luck.
    """
    recipe = mw._OFFSETS_RECIPE
    assert 'locked_row_match' in recipe
    assert 'import' in recipe and 'unified_alignment' in recipe
    # no private re-implementation of the narrowing
    assert "tbl['Module'] ==" not in recipe
    assert "tbl['Exposure'] ==" not in recipe


def test_the_recipe_is_valid_python_and_is_idempotent_by_construction(mw):
    """It is published to be copied. It must parse, and it must write back the
    TOTAL it now carries rather than the delta it just applied -- otherwise a
    second run corrects a second time."""
    recipe = mw._OFFSETS_RECIPE
    ast.parse(recipe)
    assert re.search(r"setval\(fn, 'RAOFFSET', value=total_ra", recipe)
    assert re.search(r"setval\(fn, 'DEOFFSET', value=total_dec", recipe)
    assert 'total_ra - have_ra' in recipe
    # and it leaves an unmeasured frame alone rather than inventing a shift
    assert "return None" in recipe and 'no row yet' in recipe


def test_the_table_is_linked_through_globus_not_a_site_path(mw, tmp_path):
    """The release tree is not web-served: every file on this page is reached
    through its Globus HTTPS URL. A site-relative `field/astrometry/x.csv`
    404s, which is what the first version of this section shipped."""
    url = ('https://g-92a536.55ba.08cc.data.globus.org/releases/v1.8-2026.09/'
           'gc-treasury/astrometry/Offsets_JWST_Brick10678_consensus.csv')
    manifest = {'files': [{'kind': 'offsets_table', 'url': url}]}
    html = mw._offsets_section('gc-treasury', _summary(tmp_path), manifest)
    assert url in html
    assert "href='gc-treasury/astrometry" not in html

    # with no manifest entry it names the file without inventing a link
    plain = mw._offsets_section('gc-treasury', _summary(tmp_path), {'files': []})
    assert 'Offsets_JWST_Brick10678_consensus.csv' in plain
    assert 'href=' not in plain.split('is the authority')[0].split('<h2>')[-1]
