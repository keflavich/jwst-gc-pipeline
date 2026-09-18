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

// `o98` unpadded on purpose: with every row zero-padded a plain text sort
// produces the identical order, so the numeric comparator the test claims to
// hold is not actually exercised.
const ROWS = [
  ['o139', 'F212N', '-469.1', '-161.5', '496', 'm2 tie', '30 / 48', '5.2'],
  ['o98',  'F212N',  '-34.0', '-247.3', '250', 'm2 tie', '38 / 48', '4.1'],
  ['o127', 'F212N',       '',       '',    '', 'm2 tie', '35 / 48', '5.6'],
  ['o112', 'F480M',  '-60.6',  '-38.9',  '72', 'm2 tie', '12 / 12', '3.3'],
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
               '{label:"Total",kind:"num"},{label:"Source",kind:"text"},'
               '{label:"Frames",kind:"frac"},{label:"RMS",kind:"num"}]')
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
    """"Sorted by obs" means the number. The harness carries `o98` unpadded
    because that is the only case that separates the two comparators: with
    every row padded to three digits a plain text sort gives the identical
    order, and a test over padded rows only passes whichever it is handed."""
    first, second = _sort(mw, tmp_path, 0)
    assert first == ['o139', 'o127', 'o112', 'o98'], first
    assert second == ['o98', 'o112', 'o127', 'o139'], second


def test_sorting_a_number_column_keeps_unmeasured_rows_at_the_end(mw, tmp_path):
    """An empty cell is "not measured", not zero. Sorted as zero it lands in
    the middle of the real offsets and reads as a small one -- the same
    confusion this table was rewritten to remove."""
    desc, asc = _sort(mw, tmp_path, 4)
    assert desc[0] == 'o139' and desc[-1] == 'o127', desc
    assert asc[-1] == 'o127', asc


def test_the_coverage_column_sorts_by_the_fraction_not_the_numerator(mw,
                                                                    tmp_path):
    """`parseFloat('12 / 12')` is 12, so a COMPLETE pair sorts below a
    three-quarters-done `35 / 48` on the column whose whole purpose is to show
    how complete each one is."""
    desc, asc = _sort(mw, tmp_path, 6)
    assert desc[0] == 'o112', desc      # 12/12 = 1.00 is the most complete
    assert asc[0] == 'o139', asc        # 30/48 = 0.63 the least
    assert desc == ['o112', 'o98', 'o127', 'o139'], desc



# ---- coverage: rows against frames ----
def _csv(tmp_path, rows):
    """A minimal offsets table on disk, in the columns `summarise_table` reads."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    head = ('Visit,Filter,Module,Exposure,Vgroup,dra (arcsec),ddec (arcsec),'
            'prov_source')
    body = '\n'.join(','.join(str(c) for c in r) for r in rows)
    path = tmp_path / 'offsets.csv'
    path.write_text(f'{head}\n{body}\n')
    return str(path)


def _resid(exposure, dra=0.001, ddec=-0.002):
    return ('jw10678114001', 'F212N', 'nrca1', exposure, 2101, dra, ddec,
            'm2 visit-consensus')


def test_a_scatter_of_fewer_than_three_rows_is_not_reported(builder, tmp_path):
    """o114 F212N has ONE per-exposure row. `np.std` of one value is 0.0, and
    the page rendered that as "0.0 mas frame-to-frame" -- perfect agreement
    between frames, from a single measurement, on the observation with the
    least evidence behind it. Two rows are no better: their std is half their
    separation whatever the real scatter is.
    """
    def rms(n):
        rows = [_resid(i + 1, dra=0.001 * (i + 1)) for i in range(n)]
        out, _, _ = builder.summarise_table(_csv(tmp_path / str(n), rows))
        assert len(out) == 1
        return out[0].get('residual_rms_mas')

    assert rms(1) is None
    assert rms(2) is None
    assert rms(3) is not None and rms(3) > 0


def test_the_row_count_is_reported_against_the_frames_it_covers(builder,
                                                                tmp_path):
    """The count varies 1 to 44 between observations that hold exactly the
    same 48 frames, so a bare count reads as something the field did
    differently. It is coverage: how much of the observation this table
    describes yet."""
    rows = [_resid(i + 1) for i in range(3)]
    out, _, _ = builder.summarise_table(
        _csv(tmp_path / 'cov', rows), frames={('o114', 'F212N'): 48})
    assert out[0]['n_exposure_rows'] == 3
    assert out[0]['n_frames'] == 48

    # and a key the frame scan never saw stays None rather than borrowing one
    plain, _, _ = builder.summarise_table(
        _csv(tmp_path / 'cov2', rows), frames={('o139', 'F212N'): 48})
    assert plain[0]['n_frames'] is None


def test_the_frame_scan_counts_frames_per_observation_and_filter(builder,
                                                                 tmp_path):
    """The denominator comes from the release tree, staged as
    `exposures/<obs>/<FILTER>/`. Keyed on anything else it never matches the
    table's (observation, filter) and every cell silently falls back to the
    bare count this column was rewritten to remove.
    """
    for obs, filt, n in (('o114', 'F212N', 3), ('o114', 'F480M', 2),
                         ('o139', 'F212N', 1)):
        d = tmp_path / 'exposures' / obs / filt
        d.mkdir(parents=True)
        for i in range(n):
            _frame(d, f'f{i}.fits',
                   f'jw10678{obs[1:]}001_02101_0000{i}_nrca1_destreak_crf.fits',
                   filt=filt)
    _, _, _, per_key = builder.scan_frames(str(tmp_path / 'exposures'))
    assert per_key == {('o114', 'F212N'): 3, ('o114', 'F480M'): 2,
                       ('o139', 'F212N'): 1}


def test_the_table_cell_shows_the_fraction_of_frames_measured(mw, tmp_path):
    """`30` and `48` in separate places on the page is not the same as `30 /
    48` in the cell: the reader compares the cell against its neighbours,
    which is exactly the comparison that misled."""
    summary = _summary(tmp_path)
    data = json.loads((summary / 'offsets_summary.json').read_text())
    for row in data['per_filter']:
        row['n_frames'] = 48
    (summary / 'offsets_summary.json').write_text(json.dumps(data))
    html = mw._offsets_section('gc-treasury', summary)
    assert '30 / 48' in html
    # the harness supplies its own header kinds, so the sorting test cannot see
    # this: the column has to DECLARE the fraction comparator to get it
    assert '<th data-sort=frac>Frames measured</th>' in html

    # without a denominator it shows the count rather than inventing one
    plain = mw._offsets_section('gc-treasury', _summary(tmp_path))
    assert '<td class=size>30 / 48</td>' not in plain
    assert '<td class=size>30</td>' in plain


# ---- which rows are bulk ties, and which measurement made them ------------
def _bulk(visit='jw10678040001', filt='F212N', module='all', exposure=-1,
          dra=-4.4896, ddec=-19.9078,
          source='offset-histogram swept+confirmed vs VIRAC2 (i2d mosaic)'):
    return (visit, filt, module, exposure, 2101, dra, ddec, source)


def _m2_bulk(visit='jw10678139001', filt='F212N'):
    return (visit, filt, 'all', -1, 2101, -0.4691, -0.1615,
            'm2 consensus->reference')


def test_a_bulk_row_is_found_by_its_source_and_by_its_module(builder,
                                                             tmp_path):
    """Two clauses select a bulk row, and each has to work alone.

    Drop the Module clause and a table written before `prov_source` existed
    loses every tie it has. Drop the source clause and a row whose Module is
    not literally `all` falls through to the residual branch, where an
    arcsecond-scale value enters a frame-to-frame RMS -- the same pooling in
    the other direction.
    """
    # source only: Module says something else
    by_source, _, measured = builder.summarise_table(_csv(tmp_path / 'src', [
        ('jw10678139001', 'F212N', 'nrca1', 1, 2101, -0.4691, -0.1615,
         'm2 consensus->reference')]))
    assert measured == 1, 'the source alone identifies a bulk tie'
    assert by_source[0]['n_exposure_rows'] == 0

    # Module only: an older table with no usable source string
    by_module, _, measured = builder.summarise_table(_csv(tmp_path / 'mod', [
        ('jw10678139001', 'F212N', 'all', -1, 2101, -0.4691, -0.1615, '')]))
    assert measured == 1, 'Module=all alone identifies a bulk tie'
    assert by_module[0]['total_mas'] == pytest.approx(496, abs=1)


def test_a_histogram_tie_is_kept_and_named_rather_than_pooled_or_dropped(
        builder, mw, tmp_path):
    """The live table carries four `offset-histogram swept+confirmed vs
    VIRAC2 (i2d mosaic)` rows: o040 at 20.4" and o041 at 6.4", the real
    displacements of the second block.

    They belong in this table -- they are the largest errors a downloader can
    hit -- and they are a different measurement from the m2 tie at a different
    scale. Dropped, the page hides a 20" error; pooled unlabelled under prose
    saying "tens to hundreds of milliarcseconds", it contradicts itself.
    """
    rows, _, measured = builder.summarise_table(
        _csv(tmp_path / 'hist', [_bulk(), _m2_bulk()]))
    assert measured == 2
    by_obs = {r['observation']: r for r in rows}
    assert by_obs['o040']['total_mas'] == pytest.approx(20407, abs=5)
    assert by_obs['o040']['source'].startswith('offset-histogram')
    assert by_obs['o139']['source'] == 'm2 consensus->reference'

    # and the arcsecond value never reaches the frame-to-frame scatter
    assert 'residual_rms_mas' not in by_obs['o040']

    assert mw._source_label(by_obs['o040']['source']) == \
        'histogram vs VIRAC2 (mosaic)'
    assert mw._source_label(by_obs['o139']['source']) == 'm2 tie'
    # an unrecognised provenance is shown, not relabelled
    assert mw._source_label('m9 something new') == 'm9 something new'


def test_the_page_names_the_measurement_behind_each_offset(mw, tmp_path):
    """Without it the table shows a 20" row beside a 70 mas row with nothing
    to say they were measured differently, under one heading that describes
    one of them."""
    summary = _summary(tmp_path)
    data = json.loads((summary / 'offsets_summary.json').read_text())
    data['per_filter'].append(
        {'observation': 'o040', 'filter': 'F212N', 'n_exposure_rows': 0,
         'dra_arcsec': -4.4896, 'ddec_arcsec': -19.9078, 'total_mas': 20407.8,
         'in_release': False,
         'source': 'offset-histogram swept+confirmed vs VIRAC2 (i2d mosaic)'})
    (summary / 'offsets_summary.json').write_text(json.dumps(data))
    html = mw._offsets_section('gc-treasury', summary)

    # in the CELLS -- the paragraph above the table names both methods, so a
    # bare substring test passes with the column deleted
    assert '<td>histogram vs VIRAC2 (mosaic)</td>' in html
    assert '<td>m2 tie</td>' in html
    assert '<th data-sort=text>Source</th>' in html
    assert '20408' in html, 'the arcsecond-scale tie is shown, not hidden'
    # the prose no longer claims a range the table contradicts
    assert 'It is <b>tens to hundreds of milliarcseconds</b>' not in html
    assert 'a few are arcseconds' in html


def test_an_observation_not_staged_in_this_release_says_so(mw, tmp_path):
    """o040 and o041 have rows in the programme-wide offsets table and no
    frames in this release. `0` in the coverage column reads as "measured
    nothing about it", which is a statement about the measurement; what is
    true is that the download does not contain it."""
    summary = _summary(tmp_path)
    data = json.loads((summary / 'offsets_summary.json').read_text())
    data['per_filter'].append(
        {'observation': 'o040', 'filter': 'F212N', 'n_exposure_rows': 0,
         'dra_arcsec': -4.4896, 'ddec_arcsec': -19.9078, 'total_mas': 20407.8,
         'in_release': False, 'source': 'm2 consensus->reference'})
    (summary / 'offsets_summary.json').write_text(json.dumps(data))
    assert 'not in this release' in mw._offsets_section('gc-treasury', summary)

    assert mw._coverage_cell({'n_exposure_rows': 0, 'in_release': False}) == \
        'not in this release'
    # a staged observation with no denominator yet still shows its count
    assert mw._coverage_cell({'n_exposure_rows': 3, 'in_release': True}) == '3'


def test_the_frame_scan_finds_the_observation_inside_the_miri_tree(builder,
                                                                   tmp_path):
    """MIRI stages as `exposures/MIRI/o132/F770W/` and NIRCam as
    `exposures/o127/F212N/`. Taking the first path segment filed every MIRI
    frame under an observation called `MIRI`, which matches no row in the
    table and no observation in the release."""
    nircam = tmp_path / 'exposures' / 'o127' / 'F212N'
    nircam.mkdir(parents=True)
    _frame(nircam, 'n.fits',
           'jw10678127001_02101_00001_nrca1_destreak_o127_crf.fits')
    miri = tmp_path / 'exposures' / 'MIRI' / 'o132' / 'F770W'
    miri.mkdir(parents=True)
    _frame(miri, 'm.fits', 'jw10678-o132_t001_miri_f770w_2_o132_crf.fits',
           filt='F770W', detector='MIRIMAGE')

    per_obs, _totals, _unreadable, per_key = builder.scan_frames(
        str(tmp_path / 'exposures'))
    assert set(per_key) == {('o127', 'F212N'), ('o132', 'F770W')}
    assert set(per_obs) == {'o127', 'o132'}


def test_the_scatter_is_a_sample_standard_deviation(builder, tmp_path):
    """`ddof=0` reads ~18% low at the three-row floor, in the direction
    `MIN_ROWS_FOR_RMS` exists to guard against: the point of the floor is that
    a small-n scatter understates, so the estimator should not add to it."""
    import numpy as np
    offsets = [0.000, 0.010, 0.020]
    rows = [_resid(i + 1, dra=d, ddec=0.0) for i, d in enumerate(offsets)]
    out, _, _ = builder.summarise_table(_csv(tmp_path / 'ddof', rows))
    assert out[0]['residual_rms_mas'] == pytest.approx(
        np.std(offsets, ddof=1) * 1000, rel=1e-6)


def test_the_builder_marks_which_observations_this_release_stages(builder,
                                                                  tmp_path):
    """The page's "not in this release" cell reads a flag the builder sets.
    Set to None for everything, every such row falls back to showing `0`, and
    the rendering tests keep passing because they write the flag into a
    fixture by hand."""
    rows, _, _ = builder.summarise_table(
        _csv(tmp_path / 'staged', [_bulk(), _m2_bulk()]),
        frames={('o139', 'F212N'): 48})
    by_obs = {r['observation']: r for r in rows}
    assert by_obs['o139']['in_release'] is True
    assert by_obs['o040']['in_release'] is False

    # with no frame scan at all, the question was not asked -- which is not the
    # same as answering "no"
    blind, _, _ = builder.summarise_table(_csv(tmp_path / 'blind', [_bulk()]))
    assert blind[0]['in_release'] is None
