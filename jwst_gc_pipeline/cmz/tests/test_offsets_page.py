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
        'per_filter': [
            {'observation': 'o139', 'filter': 'F212N', 'n_exposure_rows': 30,
             'dra_arcsec': -0.4691, 'ddec_arcsec': -0.1615,
             'total_mas': 496.0, 'residual_rms_mas': 5.2,
             'source': 'm2 consensus->reference'},
            {'observation': 'o127', 'filter': 'F212N', 'n_exposure_rows': 35,
             'residual_rms_mas': 5.6},
        ],
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


def test_the_recipe_survives_an_association_style_filename(mw, tmp_path):
    """Every MIRI frame in the release carries
    `jw10678-o132_t001_miri_f770w_2_o132_crf.fits` as its FILENAME, not the
    per-exposure form. `int(parts[2])` on that raises ValueError, so a user
    running this over their download hit an unhandled traceback on 198 of
    2,178 files -- about one in eleven.

    The right answer for them is "no row": this table is NIRCam-only.
    """
    recipe = mw._OFFSETS_RECIPE
    assert "parts[2].isdigit()" in recipe
    guard = recipe.split("parts[2].isdigit()")[1].split('\n\n')[0]
    assert 'return None' in guard

    # the guard precedes the int() that would raise
    assert recipe.index("parts[2].isdigit()") < recipe.index("exposure=int(parts[2])")

    ns = {}
    exec(recipe.replace(
        "Table.read('Offsets_JWST_Brick10678_consensus.csv')",
        "type('T', (), {})()"), ns)
    assert 'owed' in ns


def test_miri_frames_are_reported_as_never_gaining_a_row(mw, tmp_path):
    """"1,140 have no row yet" promises a correction that is not coming for
    198 of them. A NIRCam frame gains a row when the stages reach it; a MIRI
    frame never will."""
    summary = _summary(tmp_path)
    data = json.loads((summary / 'offsets_summary.json').read_text())
    data['frames']['totals']['no_row_not_nircam'] = 198
    (summary / 'offsets_summary.json').write_text(json.dumps(data))
    html = mw._offsets_section('gc-treasury', summary)
    assert '198' in html
    assert 'never' in html and 'F770W' in html

    # without any MIRI the sentence does not appear at all
    plain = mw._offsets_section('gc-treasury', _summary(tmp_path))
    assert 'F770W' not in plain


# ---- the classifier behind the counts ----
@pytest.fixture(scope='module')
def builder():
    import sys
    sys.path.insert(0, _REL)
    return _load('build_offsets_summary')


def _frame(tmp_path, name, filename, filt='F212N', detector='NRCA1'):
    from astropy.io import fits
    import numpy as np
    hdu = fits.PrimaryHDU(np.zeros((2, 2), dtype='float32'))
    hdu.header['FILENAME'] = filename
    hdu.header['FILTER'] = filt
    hdu.header['DETECTOR'] = detector
    path = tmp_path / name
    fits.HDUList([hdu, fits.ImageHDU(np.zeros((2, 2), dtype='float32'))]).writeto(path)
    return str(path)


def _table(**over):
    from astropy.table import Table
    row = {'Visit': 'jw10678127001', 'Filter': 'F212N', 'Module': 'nrca1',
           'Exposure': 1, 'Vgroup': '2101',
           'dra (arcsec)': 0.01, 'ddec (arcsec)': -0.02}
    row.update(over)
    return Table([{k: [v] for k, v in row.items()}[k] for k in row],
                 names=list(row))


def test_row_state_classifies_a_miri_frame_as_never_covered(builder, tmp_path):
    """The page's "198 will never gain a row" sentence keys on this state. A
    classifier that returns plain `no_row` makes the sentence vanish for the
    real release while every rendering test still passes, because they write
    the count into a fixture by hand."""
    frame = _frame(tmp_path, 'miri.fits',
                   'jw10678-o132_t001_miri_f770w_2_o132_crf.fits',
                   filt='F770W', detector='MIRIMAGE')
    assert builder._row_state(_table(), frame) == 'no_row_not_nircam'


def test_row_state_classifies_nircam_frames_by_whether_the_table_has_them(
        builder, tmp_path):
    """The NIRCam half matters as much as the MIRI half: a classifier that
    returned `no_row_not_nircam` for everything would pass a MIRI-only test
    and report the whole release as permanently uncorrectable."""
    covered = _frame(tmp_path, 'covered.fits',
                     'jw10678127001_02101_00001_nrca1_destreak_o127_crf.fits')
    assert builder._row_state(_table(), covered) == 'has_row'

    # same frame, a table that does not describe it
    missing = _frame(tmp_path, 'missing.fits',
                     'jw10678139001_02101_00003_nrcb2_destreak_o139_crf.fits',
                     detector='NRCB2')
    assert builder._row_state(_table(), missing) == 'no_row'


def test_the_table_reports_the_bulk_tie_not_the_frame_scatter(mw, tmp_path):
    """The table pooled two different measurements and took a median over both.

    `m2 consensus->reference` rows (20 of them) are the bulk tie: how far a
    visit's whole pointing sits from the reference frame, 70-500 mas.
    `m2 visit-consensus` rows (878) are residuals about that consensus, a few
    mas. A median over both reports ~5 mas, swamped by the residuals -- two
    orders of magnitude below the real offset, on a page telling people how to
    correct their astrometry.
    """
    html = mw._offsets_section('gc-treasury', _summary(tmp_path))
    assert 'BULK TIE' in html
    assert '496' in html, 'the real offset must be the number shown'
    assert '5.2' in html, 'the frame-to-frame scatter is reported separately'
    # and never as one pooled number
    assert 'Median offset per observation' not in html


def test_a_pair_with_no_measured_tie_is_called_out_not_shown_as_small(mw,
                                                                     tmp_path):
    """17 of 32 visits have no tie to the reference frame at all. Showing them
    a median of their residuals says "your astrometry is good to 5 mas" about
    a pointing whose error has never been measured."""
    html = mw._offsets_section('gc-treasury', _summary(tmp_path))
    assert 'No measured tie yet' in html
    assert 'o127' in html
    assert 'unknown rather' in html


# ---- sorting ----
SORT_HARNESS = r"""
// Minimal DOM: enough for the sorter to run. jsdom is not installed and this
// needs three behaviours, not a browser.
function makeCell(text) {
  return {textContent: text, dataset: {}, style: {}};
}
function makeRow(cells) { return {cells: cells.map(makeCell)}; }

const ROWS = [
  ['o139', 'F212N', '-469.1', '-161.5', '496', '30', '5.2'],
  ['o098', 'F212N',  '-34.0', '-247.3', '250', '38', '4.1'],
  ['o127', 'F212N',       '',       '',    '',  '35', '5.6'],
  ['o112', 'F480M',  '-60.6',  '-38.9',  '72', '12', '3.3'],
].map(makeRow);

const handlers = {};
const head = {rows: [{cells: __HEADERS__.map(function (h, i) {
  const th = makeCell(h.label);
  th.dataset.sort = h.kind;
  th.addEventListener = function (_, fn) { handlers[i] = fn; };
  return th;
})}]};
const body = {rows: ROWS.slice(), appendChild: function (r) {
  const at = body.rows.indexOf(r);
  if (at !== -1) { body.rows.splice(at, 1); }
  body.rows.push(r);
}};
globalThis.document = {getElementById: function () {
  return {tBodies: [body], tHead: head};
}};

__SCRIPT__

function order() { return body.rows.map(function (r) { return r.cells[0].textContent; }); }
handlers[__COL__]();
console.log(JSON.stringify(order()));
handlers[__COL__]();
console.log(JSON.stringify(order()));
"""


def _sort(mw, tmp_path, column):
    import re as _re
    import shutil
    import subprocess
    node = shutil.which('node')
    if node is None:
        pytest.skip('node is not available')
    inner = _re.search(r'<script>(.*)</script>', mw._SORTABLE_SCRIPT, _re.S).group(1)
    headers = ('[{label:"Obs",kind:"obs"},{label:"Filter",kind:"text"},'
               '{label:"dRA",kind:"num"},{label:"dDec",kind:"num"},'
               '{label:"Total",kind:"num"},{label:"Rows",kind:"num"},'
               '{label:"RMS",kind:"num"}]')
    js = (SORT_HARNESS.replace('__SCRIPT__', inner)
                      .replace('__HEADERS__', headers)
                      .replace('__COL__', str(column)))
    path = tmp_path / f'sort{column}.js'
    path.write_text(js)
    done = subprocess.run([node, str(path)], capture_output=True, text=True,
                          timeout=20)
    assert done.returncode == 0, done.stderr[-1500:]
    return [json.loads(line) for line in done.stdout.strip().split('\n')]


def test_the_table_sorts_by_observation_number_not_by_text(mw, tmp_path):
    """"Sorted by obs" means the number. As text `o098` sorts before `o112`
    only by accident of zero-padding, and the release calls them o98 and o112
    in prose -- a text sort puts o98 after o139 the moment the padding goes."""
    first, second = _sort(mw, tmp_path, 0)
    assert first == ['o139', 'o127', 'o112', 'o098'], first
    assert second == ['o098', 'o112', 'o127', 'o139'], second


def test_sorting_a_number_column_keeps_unmeasured_rows_at_the_end(mw, tmp_path):
    """An empty cell is "not measured", not zero. Sorted as zero it lands in
    the middle of the real offsets and reads as a small one -- the same
    confusion this table was rewritten to remove."""
    desc, asc = _sort(mw, tmp_path, 4)
    assert desc[0] == 'o139' and desc[-1] == 'o127', desc
    assert asc[-1] == 'o127', asc
