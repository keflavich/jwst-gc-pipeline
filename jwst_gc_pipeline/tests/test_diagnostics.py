"""Tests for the per-field diagnostic write-up machinery.

These cover the parts that are easy to get quietly wrong: which of several
generations of a product gets picked, whether a column that does not exist
degrades a panel or kills the run, and whether the prose the generator writes
actually follows the numbers it was given.  The figure builders themselves
need real data products and are exercised by running the driver on a field.
"""

import json
import os

import numpy as np
import pytest
from astropy.table import Table

from jwst_gc_pipeline.diagnostics import inventory as inv_mod
from jwst_gc_pipeline.diagnostics import loaders, style, writeup
from jwst_gc_pipeline.diagnostics.figures import FigureResult
from jwst_gc_pipeline.diagnostics.inventory import FieldInventory


# --------------------------------------------------------------- inventory

def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        fh.write('x')


@pytest.fixture
def catdir(tmp_path):
    return str(tmp_path / 'catalogs')


def test_crossband_prefers_higher_stage(catdir):
    for name in ('basic_merged_indivexp_photometry_tables_merged_resbgsub_m7.fits',
                 'basic_merged_indivexp_photometry_tables_merged_resbgsub_m8.fits'):
        _touch(os.path.join(catdir, name))
    path, _match = inv_mod._best(catdir, inv_mod._CROSSBAND_RE)
    assert path.endswith('_m8.fits')


def test_crossband_prefers_dedup_at_m8(catdir):
    for name in ('basic_merged_indivexp_photometry_tables_merged_resbgsub_m8.fits',
                 'basic_merged_indivexp_photometry_tables_merged_resbgsub_m8_dedup.fits'):
        _touch(os.path.join(catdir, name))
    path, _match = inv_mod._best(catdir, inv_mod._CROSSBAND_RE)
    assert path.endswith('_m8_dedup.fits')


@pytest.mark.parametrize('token', ['2221', '1905', '6151', '10678'])
def test_derivatives_are_never_canonical(catdir, token):
    """A post-hoc filtered product at a HIGHER stage must still lose.

    Parametrized over the token because the quality-cut suffix carries each
    field's OWN proposal (wd1 writes ``_qualcuts_oksep1905``, w51  # noqa: qualcuts-token
    ``_qualcuts_oksep6151``); a test spelling only the Brick's 2221 would pass  # noqa: qualcuts-token
    while the rule silently held for one program.
    """
    _touch(os.path.join(
        catdir, 'basic_merged_indivexp_photometry_tables_merged_resbgsub_m7.fits'))
    _touch(os.path.join(
        catdir,
        'basic_merged_indivexp_photometry_tables_merged_resbgsub_m8'
        f'_qualcuts_oksep{token}.fits'))
    path, _match = inv_mod._best(catdir, inv_mod._CROSSBAND_RE)
    assert path.endswith('_m7.fits')


def test_per_filter_is_matched_by_filter(catdir):
    for filt in ('f182m', 'f212n'):
        _touch(os.path.join(
            catdir, f'{filt}_merged_indivexp_merged_resbgsub_m7_dao_basic.fits'))
    path, _match = inv_mod._best(catdir, inv_mod._PERFILTER_RE, filt='f212n')
    assert os.path.basename(path).startswith('f212n_')


def test_merged_module_beats_single_module(catdir):
    for module in ('nrca', 'merged'):
        _touch(os.path.join(
            catdir, f'f212n_{module}_indivexp_merged_resbgsub_m7_dao_basic.fits'))
    path, _match = inv_mod._best(catdir, inv_mod._PERFILTER_RE, filt='f212n')
    assert '_merged_indivexp_' in os.path.basename(path)


def test_synthetic_mosaics_are_rejected():
    """A model/residual i2d shares the suffix but is not the science mosaic."""
    for name in ('jw02221-o001_t001_nircam_clear-f182m-merged_model_i2d.fits',
                 'jw02221-o001_t001_nircam_clear-f182m-mirimage_residual_i2d.fits',
                 'jw03958-o001_t001_miri_clear-f770w-mirimage_'
                 'resbgsub_group_m6_daophot_basic_mergedcat_'
                 'residual_smoothed_bg_i2d.fits'):
        assert inv_mod._SYNTHETIC_MOSAIC_RE.search(name), name
    assert not inv_mod._SYNTHETIC_MOSAIC_RE.search(
        'jw02221-o001_t001_nircam_clear-f182m-merged_i2d.fits')


def test_unknown_field_names_the_known_ones():
    with pytest.raises(KeyError) as err:
        inv_mod.inventory('not-a-field')
    assert 'brick' in str(err.value)


# ----------------------------------------------------------------- loaders

def test_read_columns_resolves_skycoord_pair(tmp_path):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    tbl = Table({'flux': [1.0, 2.0]})
    tbl['skycoord'] = SkyCoord([1.0, 2.0] * u.deg, [3.0, 4.0] * u.deg)
    path = str(tmp_path / 'cat.fits')
    tbl.write(path)
    # On disk the mixin is a .ra/.dec pair; Table.read reassembles it.
    out = loaders.read_columns(path, ['flux', 'skycoord.ra', 'skycoord.dec'])
    assert 'skycoord' in out.colnames
    assert np.allclose(loaders.column(out, 'skycoord.ra'), [1.0, 2.0])


def test_missing_columns_warn_and_are_dropped(tmp_path):
    path = str(tmp_path / 'cat.fits')
    Table({'flux': [1.0]}).write(path)
    with pytest.warns(loaders.MissingColumnsWarning):
        out = loaders.read_columns(path, ['flux', 'mean_modelsub_bkg'])
    assert out.colnames == ['flux']
    # A column that is absent reads as all-NaN rather than raising, so the
    # panel that wanted it degrades instead of taking the document with it.
    assert np.all(np.isnan(loaders.column(out, 'mean_modelsub_bkg')))


def test_magnitudes_label_matches_the_calibration():
    flux = np.array([100.0, 10.0])
    _mag, label = loaders.magnitudes(flux, None)
    assert 'instrumental' in label
    mag, label = loaders.magnitudes(flux, (1e-9, 1e-6))
    assert label == 'Vega mag'
    # A factor of ten in flux is 2.5 magnitudes, whatever the zero-point.
    assert np.isclose(mag[1] - mag[0], 2.5)


def test_photometric_zeropoints_recovers_the_conversion(tmp_path):
    rng = np.random.default_rng(0)
    flux = rng.uniform(10, 1000, 500)
    conv, vega = 3.5e-9, 1.2e-6
    tbl = Table({'flux_f212n': flux,
                 'flux_jy_f212n': flux * conv,
                 'mag_vega_f212n': -2.5 * np.log10(flux * conv / vega)})
    path = str(tmp_path / 'cross.fits')
    tbl.write(path)
    got = loaders.photometric_zeropoints(path, ['f212n'])
    assert np.isclose(got['f212n'][0], conv, rtol=1e-6)
    assert np.isclose(got['f212n'][1], vega, rtol=1e-6)


# ------------------------------------------------------------------- style

def test_running_percentiles_suppresses_sparse_bins():
    x = np.concatenate([np.zeros(200), np.full(2, 10.0)])
    y = np.concatenate([np.ones(200), np.full(2, 99.0)])
    centres, pct = style.running_percentiles(x, y, bins=10, min_count=10)
    assert centres.size == 10
    # The two-point bin must not produce a "measured" median of 99.
    assert not np.any(pct[50] > 50)


def test_binned_median_image_respects_min_count():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 5000)
    y = rng.uniform(0, 1, 5000)
    img, extent = style.binned_median_image(x, y, np.full(x.size, 7.0),
                                            nbins=8, min_count=3)
    assert np.nanmax(img) == pytest.approx(7.0)
    assert extent[0] < extent[1] and extent[2] < extent[3]


def test_spearman_of_a_monotone_relation_is_one():
    x = np.arange(500.0)
    rho, _p = style.spearman(x, np.exp(x / 500.0))
    assert rho == pytest.approx(1.0)


# ----------------------------------------------------------------- writeup

def _inv():
    return FieldInventory(name='testfield', basepath='/tmp/testfield',
                          filters=('f182m', 'f212n'), proposals=('2221',))


def _result(key, section, measurements):
    return FigureResult(key, f'/tmp/figures/{key}.pdf', 'A caption.', section,
                        measurements)


def test_writeup_quotes_the_measurements_it_was_given(tmp_path):
    results = [
        _result('D2_astrometry_internal', 'astrometry',
                dict(floors_mas={'f182m': 2.5, 'f212n': 3.5},
                     n_zero_scatter={'f182m': 0, 'f212n': 0},
                     n_sources={'f182m': 10, 'f212n': 10}, crossband={})),
    ]
    doc = writeup.Writeup(_inv(), results, str(tmp_path))
    text = doc.render()
    assert '2.50' in text and '3.50' in text
    assert r'\end{document}' in text


def test_writeup_prose_follows_the_numbers(tmp_path):
    """A large propagated/formal ratio must change what the text says."""
    def render(ratio):
        results = [_result('D4_photometry_precision', 'photometry',
                           dict(depth={'f212n': 20.0}, err_ratio={'f212n': ratio}))]
        return writeup.Writeup(_inv(), results, str(tmp_path)).render()

    assert 'understate the real uncertainty' in render(3.0)
    assert 'understate the real uncertainty' not in render(1.0)
    assert 'noise model is' in render(1.0)


def test_writeup_reports_a_weak_background_correlation_as_such(tmp_path):
    def render(rho):
        results = [_result('D7_background_spatial', 'background',
                           dict(correlations={'f212n': dict(spearman=rho, n=1000)}))]
        return writeup.Writeup(_inv(), results, str(tmp_path)).render()

    assert 'measuring real extended emission' in render(0.85)
    assert 'not primarily tracking the extended' in render(0.1)


def test_span_reads_naturally_for_one_filter():
    """"from X (F1) to X (F1)" reads as a bug; a single filter gets one value."""
    assert writeup._span({'f322w2': 2.5}, 2, ' mag') == '2.50 mag in F322W2'
    two = writeup._span({'f182m': 1.0, 'f212n': 3.0}, 1)
    assert two == '1.0 in F182M to 3.0 in F212N'
    assert writeup._span({'f182m': np.nan}, 2) is None
    assert writeup._span({}, 2) is None


def test_dominant_scale_splits_mixed_magnitude_scales():
    """A band with no zero-point must not be averaged in with calibrated ones."""
    depth = {'f182m': 24.7, 'f405n': 19.7, 'f770w': -3.9}
    scales = {'f182m': 'Vega mag', 'f405n': 'Vega mag',
              'f770w': 'instrumental'}
    keep, label, excluded = writeup._dominant_scale(depth, scales)
    assert set(keep) == {'f182m', 'f405n'}
    assert label == 'Vega mag'
    assert excluded == ['f770w']


def test_depth_range_excludes_the_uncalibrated_band(tmp_path):
    results = [_result('D4_photometry_precision', 'photometry',
                       dict(depth={'f182m': 24.7, 'f770w': -3.9},
                            err_ratio={},
                            scales={'f182m': 'Vega mag',
                                    'f770w': 'instrumental'}))]
    text = writeup.Writeup(_inv(), results, str(tmp_path)).render()
    assert '-3.90' not in text
    assert 'F770W' in text and 'absent from the cross-band merge' in text


def test_weak_background_correlation_blames_the_subtraction_when_it_should(tmp_path):
    """A pre-subtracted stage explains a weak correlation; a flat field does not."""
    def render(presub):
        results = [_result('D7_background_spatial', 'background',
                           dict(correlations={'f212n': dict(spearman=0.15,
                                                            n=1000)},
                                background_presubtracted=presub))]
        return writeup.Writeup(_inv(), results, str(tmp_path)).render()

    assert 'what a successful subtraction looks like' in render(True)
    assert 'mosaic is close to flat' in render(False)


def test_crossband_separations_use_only_independent_detections(tmp_path):
    """A seeded position imported from another band measures the merge radius."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from jwst_gc_pipeline.diagnostics import astrometry_figs

    n = 400
    rng = np.random.default_rng(0)
    ra = 266.5 + rng.uniform(0, 0.01, n)
    dec = -28.7 + rng.uniform(0, 0.01, n)
    # First half: genuinely independent, agreeing to ~1 mas.  Second half: one band's
    # position is a seed displaced by 300 mas, and is flagged as not
    # independently detected.
    offset = np.where(np.arange(n) < n // 2, 1.0, 300.0) / 3.6e6
    indep = np.arange(n) < n // 2

    tbl = Table({'independently_detected_f182m': indep.astype(int),
                 'independently_detected_f212n': indep.astype(int)})
    tbl['skycoord_f182m'] = SkyCoord(ra * u.deg, dec * u.deg)
    tbl['skycoord_f212n'] = SkyCoord((ra + offset) * u.deg, dec * u.deg)
    path = str(tmp_path / 'cross.fits')
    tbl.write(path)

    inv = FieldInventory(name='t', basepath=str(tmp_path),
                         filters=('f182m', 'f212n'), crossband_catalog=path)
    got = astrometry_figs._crossband_separations(inv, ['f182m', 'f212n'])
    assert got['independent_only'] is True
    sep = got['median_sep_mas'][0, 1]
    # ~1 mas if the gate works; ~150 mas if the seeded half leaks in.
    assert sep < 5.0, sep
    assert got['n_pairs'][0, 1] == n // 2


def test_writeup_writes_measurements_json(tmp_path):
    results = [_result('D1_overview', 'overview',
                       dict(n_sources={'f182m': 5}, area_arcsec2={},
                            peak_density=np.float32(0.2), median_density=0.1,
                            turnover={}, richest_filter='f182m'))]
    writeup.Writeup(_inv(), results, str(tmp_path)).write()
    with open(os.path.join(str(tmp_path), 'measurements.json')) as fh:
        payload = json.load(fh)
    # numpy scalars must survive the round trip, not become "np.float32(0.2)".
    assert payload['D1_overview']['measurements']['peak_density'] == \
        pytest.approx(0.2, rel=1e-6)


def test_missing_figures_do_not_break_the_document(tmp_path):
    doc = writeup.Writeup(_inv(), [], str(tmp_path))
    text = doc.render()
    assert r'\begin{document}' in text and r'\end{document}' in text
    assert 'includegraphics' not in text


def test_latex_specials_in_filenames_are_escaped(tmp_path):
    inv = _inv()
    inv.crossband_catalog = '/tmp/basic_merged_m8_dedup.fits'
    text = writeup.Writeup(inv, [], str(tmp_path)).render()
    assert 'basic\\_merged\\_m8\\_dedup.fits' in text


# ------------------------------------------------- review-fix regressions

def test_crossband_without_independence_flag_is_an_upper_bound(tmp_path):
    """Finding 1: when there is no independent-detection flag the fallback
    counts seeded positions, so the number is a contaminated upper bound and
    must not be sold as a systematic floor."""
    def render(indep_only):
        cross = dict(filters=['f182m', 'f212n'],
                     median_sep_mas=np.array([[np.nan, 99.0], [99.0, np.nan]]),
                     p84_sep_mas=np.array([[np.nan, 380.0], [380.0, np.nan]]),
                     independent_only=indep_only)
        results = [_result('D2_astrometry_internal', 'astrometry',
                           dict(floors_mas={'f212n': 2.8}, crossband=cross,
                                n_zero_scatter={}, n_sources={}))]
        return writeup.Writeup(_inv(), results, str(tmp_path)).render()

    contaminated = render(False)
    assert 'upper bound' in contaminated
    assert 'bounds \nthe systematic floor' not in contaminated
    assert 'stringent internal check' not in contaminated
    clean = render(True)
    assert 'stringent internal check' in clean
    assert 'upper bound' not in clean


def test_tie_record_uses_residual_about_bulk_and_excludes_window_tiles():
    """Findings 2 and 3: the tile statistic is the residual about the bulk tie,
    and swept / window-edge / no-tie tiles are separated out."""
    from jwst_gc_pipeline.diagnostics import astrometry_figs as A

    result = dict(dra=60.0, ddec=0.0, contrast=20.0, ok=True, swept=False,
                  off=60.0, window_edge_fraction=0.1)

    def cell(dra, ddec, **kw):
        d = dict(dra=dra, ddec=ddec, off=float(np.hypot(dra, ddec)),
                 contrast=15.0, ok=True, swept=False, window_edge_fraction=0.0,
                 ix=0, iy=0)
        d.update(kw)
        return d

    cells = [cell(60, 0), cell(62, 0), cell(58, 0),          # 3 trustworthy
             cell(500, 0, window_edge_fraction=0.95),        # rides the window
             cell(700, 0, swept=True),                       # swept
             dict(ok=False)]                                 # no coherent tie
    rec = A._tie_record(result, dict(cells=cells))
    t = rec['tiles']
    assert (t['n_measured'], t['n_window_edge'], t['n_swept'], t['n_no_tie']) \
        == (3, 1, 1, 1)
    # residual about the 60 mas bulk is ~0, not ~60
    assert t['median_off_mas'] < 5 and t['worst_off_mas'] < 5
    # the raw (bulk-inclusive) offset is kept for reference
    assert t['raw_median_off_mas'] > 55


def test_implications_tracer_bullet_is_gated_on_presubtracted(tmp_path):
    """Finding 6: the 'usable as a diffuse-emission tracer' claim must not
    appear when the column is a subtraction residual."""
    def render(presub):
        results = [_result('D7_background_spatial', 'background',
                           dict(correlations={'f212n': dict(spearman=0.7, n=1000),
                                              'f182m': dict(spearman=0.7, n=1000)},
                                background_presubtracted=presub))]
        return writeup.Writeup(_inv(), results, str(tmp_path)).render()

    assert 'in its own right' in render(False)
    assert 'in its own right' not in render(True)


def test_qfit_fraction_is_not_the_rejected_fraction(tmp_path):
    """Finding 7: 'fraction above qfit=0.2' is not the fraction that fails
    vetting, and the text must say so and drop 'dominated by'."""
    results = [_result('D5_photometry_quality', 'photometry',
                       dict(qfit={'f212n': dict(median=0.1, frac_above_warn=0.78)},
                            census={}))]
    text = writeup.Writeup(_inv(), results, str(tmp_path)).render()
    assert 'qfit above 0.2' in text
    assert 'not the fraction rejected by vetting' in text
    assert 'dominated by sources whose profile' not in text


def test_build_notes_section_distinguishes_error_from_gap(tmp_path):
    """Finding 10: omitted figures are a visible section, and a build error is
    told apart from a missing product."""
    doc = writeup.Writeup(
        _inv(), [], str(tmp_path),
        failures={'D7_background_spatial': 'no applicable data products',
                  'D3_astrometry_absolute': 'NoConvergence: inverse WCS failed'})
    text = doc.render()
    assert 'Omitted figures' in text
    assert 'no applicable data products' in text
    assert 'build error' in text and 'NoConvergence' in text


def test_perfilter_regex_matches_proposal_tokened_module():
    """Finding 5: the _j<proposal> collision-fix products must be visible."""
    m = inv_mod._PERFILTER_RE.match(
        'f200w_nrca_j7213_indivexp_merged_resbgsub_m6_dao_basic.fits')
    assert m is not None
    assert m.group('module') == 'nrca_j7213'


def test_perfilter_regex_matches_obs_tokened_module():
    """A per-obs-merged proposal (10678) tags the module slot with _o<field>;
    without it the inventory cannot see a gc-treasury tile's merged catalog.

    The observation is captured on its own: it names a mosaic TILE, and a tile
    modelled as a module makes 139 of them look like 139 detector modules.
    """
    m = inv_mod._PERFILTER_RE.match(
        'f212n_nrcblong_o042_indivexp_merged_resbgsub_m6_dao_basic.fits')
    assert m is not None
    assert m.group('module') == 'nrcblong'
    assert m.group('obs') == '042'


def test_tiles_of_one_module_are_not_reported_as_missing_modules(catdir):
    """gc-treasury's 139 tiles share one tree and one module.

    With the observation folded into the module group, every other tile reads
    as a module the write-up did not combine: three tiles produced a "covers
    part of the field only" note naming the other two.
    """
    for obs in ('001', '002', '042'):
        _touch(os.path.join(
            catdir,
            f'f212n_nrcblong_o{obs}_indivexp_merged_resbgsub_m6_dao_basic.fits'))
    assert inv_mod._module_siblings(catdir, 'f212n', 'nrcblong') == set()


def test_crossband_regex_matches_an_observation_scoped_catalog():
    """``merge_daophot`` appends ``_o<obs>`` after the stage (gc2211's
    per-pointing m7, brick's per-proposal m8 copies).  Without the optional
    group 18 of the 61 cross-band catalogs on disk are invisible."""
    for name, obs in (
            ('basic_merged_indivexp_photometry_tables_merged_resbgsub_m7_o023'
             '.fits', '023'),
            ('basic_merged_indivexp_photometry_tables_merged_resbgsub_m8_dedup'
             '_o004.fits', '004'),
            ('basic_merged_indivexp_photometry_tables_merged_resbgsub_m7.fits',
             None)):
        m = inv_mod._CROSSBAND_RE.match(name)
        assert m is not None, name
        assert m.group('obs') == obs, name


def test_crossband_prefers_the_all_observation_product(catdir):
    """The write-up describes the FIELD, so one tile's catalogue must not win
    on mtime over the pooled product beside it."""
    pooled = 'basic_merged_indivexp_photometry_tables_merged_resbgsub_m7.fits'
    _touch(os.path.join(catdir, pooled))
    for obs in ('023', '049'):
        _touch(os.path.join(
            catdir,
            f'basic_merged_indivexp_photometry_tables_merged_resbgsub_m7_o'
            f'{obs}.fits'))
    path, _match = inv_mod._best(catdir, inv_mod._CROSSBAND_RE)
    assert os.path.basename(path) == pooled


def test_an_observation_scoped_catalog_is_still_found_when_it_is_all_there_is(
        catdir):
    """A gc-treasury tile writes no pooled sibling at all."""
    name = ('basic_merged_indivexp_photometry_tables_merged_resbgsub_m7_o042'
            '.fits')
    _touch(os.path.join(catdir, name))
    path, match = inv_mod._best(catdir, inv_mod._CROSSBAND_RE)
    assert os.path.basename(path) == name
    assert int(match.group('stage')) == 7


def test_perfilter_regex_matches_a_joint_observation():
    """Joint registrations are real: sgrb2's MIRI is ``002-998`` and sickle's
    ``001-002``, and ``naming.observation_field_token`` normalises that
    spelling part by part.  The cross-band regex has always matched it; this
    one used to stop at three digits, so the same product would be visible in
    one reader and invisible in the other."""
    m = inv_mod._PERFILTER_RE.match(
        'f770w_mirimage_o002-998_indivexp_merged_m6_dao_basic.fits')
    assert m is not None
    assert m.group('module') == 'mirimage'
    assert m.group('obs') == '002-998'


def _registered_field(monkeypatch, tmp_path, name='sgrc'):
    """A REAL registry entry, re-rooted under *tmp_path*.

    The entry carries more than the inventory reads (the reference catalogs it
    resolves per observation, for one), so a stand-in namespace stops standing
    in as soon as the inventory grows; this borrows a registered field and
    moves the root its basepath is built from.  Returns ``(name, catalogs
    dir)``.
    """
    import dataclasses
    entry = inv_mod._fields.BY_NAME[name]
    moved = dataclasses.replace(entry, roots={entry.root: str(tmp_path)})
    monkeypatch.setitem(inv_mod._fields.BY_NAME, name, moved)
    return name, os.path.join(moved.basepath, 'catalogs')


def test_inventory_says_when_its_crossband_catalog_is_one_observation(
        tmp_path, monkeypatch):
    """The rank's ``pooled`` term ties at 0 when every candidate is
    observation-scoped -- gc-treasury's shape, 139 tiles and no pooled sibling
    -- so mtime picks a tile and nothing else in the inventory names an
    observation.  The write-up would then describe one tile as the field."""
    name, catalogs = _registered_field(monkeypatch, tmp_path)
    _touch(os.path.join(catalogs,
                        'basic_merged_indivexp_photometry_tables_merged'
                        '_resbgsub_m7_o042.fits'))
    inv = inv_mod.inventory(name)
    assert inv.crossband_stage == 7
    assert any('observation 042 only' in n for n in inv.notes), inv.notes


def test_inventory_says_nothing_extra_about_a_pooled_crossband_catalog(
        tmp_path, monkeypatch):
    name, catalogs = _registered_field(monkeypatch, tmp_path)
    _touch(os.path.join(catalogs,
                        'basic_merged_indivexp_photometry_tables_merged'
                        '_resbgsub_m7.fits'))
    inv = inv_mod.inventory(name)
    assert inv.crossband_stage == 7
    assert not any('observation' in n for n in inv.notes), inv.notes


def test_module_siblings_flags_partial_field_coverage(catdir):
    """Finding 5: a single-module pick with other modules on disk is flagged."""
    _touch(os.path.join(catdir, 'f200w_nrca_indivexp_merged_m6_dao_basic.fits'))
    _touch(os.path.join(catdir, 'f200w_nrcb_indivexp_merged_m6_dao_basic.fits'))
    assert inv_mod._module_siblings(catdir, 'f200w', 'nrcb') == {'nrca'}
