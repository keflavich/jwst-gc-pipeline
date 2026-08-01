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


def test_derivatives_are_never_canonical(catdir):
    _touch(os.path.join(
        catdir, 'basic_merged_indivexp_photometry_tables_merged_resbgsub_m7.fits'))
    # A post-hoc filtered product at a HIGHER stage must still lose.
    _touch(os.path.join(
        catdir,
        'basic_merged_indivexp_photometry_tables_merged_resbgsub_m8_qualcuts_oksep2221.fits'))
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
