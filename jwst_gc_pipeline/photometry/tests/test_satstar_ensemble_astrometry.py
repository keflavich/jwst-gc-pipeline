"""Issue #925: the consolidated satstar catalog must publish the per-exposure
ensemble it measures, and ``replace_saturated`` must not fill integer astrometry
columns with NaN.

Delivered o132 evidence the tests here pin:

* ``nmatch_good`` is INT32_MIN (-2147483648) for 99.8% of F480M and 97.0% of
  F212N ``replaced_saturated`` rows, and 0.0% of the rest, because the append
  path filled every missing column with ``np.ones(n) * np.nan`` regardless of
  the target dtype.
* ``std_ra`` is NaN for 99.9% / 98.2% of the same rows against 77.9% / 25.4%
  for the rest (all rows of
  ``basic_merged_indivexp_photometry_tables_merged_resbgsub_m8_o132.fits``,
  295201 of them), though the per-exposure
  satstar fits behind them repeat to 3 mas (F480M, median 4 exposures of 12) and
  1 mas (F212N, median 3 of 48) -- measured, then discarded by the
  brightest-first dedup.
* The per-exposure catalogs are globbed across six pipeline iterations
  (``_m3`` .. ``_m12``) as well as exposures, so a naive row count would report
  N=6x the number of images.
"""
import os
import re

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u

from jwst_gc_pipeline.photometry import merge_catalogs as mc


RA0, DEC0 = 266.5, -28.9
ITERS = ('m3', 'm4', 'm5', 'm6', 'm7', 'm12')


def _rows(star_offsets_mas, exposures, iterations=ITERS, flux=1.0e4,
          flux_by_exposure=None):
    """One synthetic star: ``exposures`` images x ``iterations`` re-fits.

    ``star_offsets_mas`` gives the per-exposure (dra, ddec) displacement in mas,
    so the across-exposure scatter is known exactly and the per-iteration rows
    of one exposure sit on top of each other (which is what makes N a count of
    images rather than of rows).
    """
    ra, dec, fl, exp, it = [], [], [], [], []
    for e in range(exposures):
        dra, ddec = star_offsets_mas[e]
        for k in iterations:
            dec_e = DEC0 + ddec / 3.6e6
            ra.append(RA0 + (dra / 3.6e6) / np.cos(np.radians(DEC0)))
            dec.append(dec_e)
            fl.append(flux if flux_by_exposure is None else flux_by_exposure[e])
            exp.append(f'jw10678132001_02101_{e:05d}_nrcalong')
            it.append(k)
    return ra, dec, fl, exp, it


def _table(ra, dec, fl, exp, it, with_provenance=True):
    t = Table()
    t['skycoord_fit'] = SkyCoord(np.array(ra) * u.deg, np.array(dec) * u.deg,
                                 frame='icrs')
    t['flux_fit'] = np.array(fl, dtype=float)
    if with_provenance:
        t[mc._SATSTAR_EXPKEY_COL] = np.array(exp, dtype=str)
        t[mc._SATSTAR_ITER_COL] = np.array(it, dtype=str)
    return t


# --------------------------------------------------------------------------
# filename provenance
# --------------------------------------------------------------------------

def test_exposure_key_is_the_image_not_the_iteration():
    a = mc.satstar_exposure_key(
        'jw10678132001_02101_00001_nrcalong_destreak_o132_crf_m12_satstar_catalog.fits')
    b = mc.satstar_exposure_key(
        'jw10678132001_02101_00001_nrcalong_destreak_o132_crf_resbgsub_m7_satstar_catalog.fits')
    c = mc.satstar_exposure_key(
        'jw10678132001_02101_00002_nrcalong_destreak_o132_crf_m12_satstar_catalog.fits')
    d = mc.satstar_exposure_key(
        'jw10678132001_02101_00001_nrcblong_destreak_o132_crf_m12_satstar_catalog.fits')
    assert a == b == 'jw10678132001_02101_00001_nrcalong'
    # different exposure number and different detector are different images
    assert c != a
    assert d != a


def test_iteration_token_round_trip():
    assert mc.satstar_iteration_token(
        'jw1_2_3_nrca1_destreak_o1_crf_m12_satstar_catalog.fits') == 'm12'
    assert mc.satstar_iteration_token(
        'jw1_2_3_nrca1_destreak_o1_crf_resbgsub_m5_satstar_catalog.fits') == 'm5'
    assert mc.satstar_iteration_token('no_token_here.fits') == ''


def test_exposure_key_falls_back_to_basename_without_the_iteration():
    # cutout / hand-made products do not follow the jw convention; distinct
    # files must still read as distinct exposures
    one = mc.satstar_exposure_key('cutout_alpha_m7_satstar_catalog.fits')
    two = mc.satstar_exposure_key('cutout_beta_m7_satstar_catalog.fits')
    assert one == 'cutout_alpha'
    assert one != two


# --------------------------------------------------------------------------
# ensemble statistics
# --------------------------------------------------------------------------

def test_n_frames_counts_images_not_rows():
    offs = [(0.0, 0.0), (4.0, 0.0), (0.0, 4.0), (-4.0, -4.0)]
    out = mc._dedup_satstar_catalog(_table(*_rows(offs, 4)))
    assert len(out) == 1
    # 4 exposures x 6 iterations = 24 rows behind one star
    assert int(out['n_meas_fit'][0]) == 24
    assert int(out['n_frames_fit'][0]) == 4


def test_scatter_is_between_images_not_between_refits():
    # Exposures at +-d in RA only: the sample std of the four per-exposure
    # means is the quantity under test, and the six identical iterations of
    # each exposure must not shrink it.
    d = 10.0
    offs = [(-d, 0.0), (-d, 0.0), (d, 0.0), (d, 0.0)]
    out = mc._dedup_satstar_catalog(_table(*_rows(offs, 4)))
    std_ra_mas = float(out['std_ra_fit'][0]) * 3.6e6
    # ddof=1 std of (-d, -d, +d, +d) is d * sqrt(4/3)
    assert std_ra_mas == pytest.approx(d * np.sqrt(4.0 / 3.0), rel=1e-3)
    assert float(out['std_dec_fit'][0]) == pytest.approx(0.0, abs=1e-12)


def test_single_exposure_star_has_no_scatter_but_a_real_count():
    out = mc._dedup_satstar_catalog(_table(*_rows([(0.0, 0.0)], 1)))
    assert int(out['n_frames_fit'][0]) == 1
    assert int(out['n_meas_fit'][0]) == len(ITERS)
    # one image cannot yield a scatter -- NaN is correct here, and is why the
    # count column has to be readable separately
    assert not np.isfinite(out['std_ra_fit'][0])
    assert not np.isfinite(out['std_dec_fit'][0])


def test_adopted_position_is_the_ensemble_mean_not_the_brightest_row():
    # brightest exposure deliberately offset from the others
    offs = [(60.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
    flux_by_exposure = [9.0e4, 1.0e4, 1.0e4, 1.0e4]
    out = mc._dedup_satstar_catalog(
        _table(*_rows(offs, 4, flux_by_exposure=flux_by_exposure)))
    got = out['skycoord_fit'][0]
    mean_dra_mas = (got.ra.deg - RA0) * np.cos(np.radians(DEC0)) * 3.6e6
    # mean of (60, 0, 0, 0) = 15, not the brightest row's 60
    assert mean_dra_mas == pytest.approx(15.0, abs=0.5)
    # the representative row's own position is kept for traceability
    repr_dra_mas = ((out['skycoord_repr_fit'][0].ra.deg - RA0)
                    * np.cos(np.radians(DEC0)) * 3.6e6)
    assert repr_dra_mas == pytest.approx(60.0, abs=0.5)


def test_ensemble_position_can_be_switched_off(monkeypatch):
    monkeypatch.setenv('SATSTAR_ENSEMBLE_POSITION', '0')
    offs = [(60.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
    flux_by_exposure = [9.0e4, 1.0e4, 1.0e4, 1.0e4]
    out = mc._dedup_satstar_catalog(
        _table(*_rows(offs, 4, flux_by_exposure=flux_by_exposure)))
    dra_mas = ((out['skycoord_fit'][0].ra.deg - RA0)
               * np.cos(np.radians(DEC0)) * 3.6e6)
    assert dra_mas == pytest.approx(60.0, abs=0.5)
    assert 'skycoord_repr_fit' not in out.colnames
    # the statistics are published either way
    assert int(out['n_frames_fit'][0]) == 4


def test_flux_fit_is_left_alone_for_the_photometry_side():
    # brightest-of-N is a photometric-continuity decision (#925 items 4-5);
    # this change publishes the ensemble without moving the flux
    offs = [(0.0, 0.0)] * 4
    flux_by_exposure = [1.0e4, 1.1e4, 1.2e4, 5.0e4]
    out = mc._dedup_satstar_catalog(
        _table(*_rows(offs, 4, flux_by_exposure=flux_by_exposure)))
    assert float(out['flux_fit'][0]) == pytest.approx(5.0e4)
    # ...but the alternative is measured and available
    assert float(out['flux_med_fit'][0]) == pytest.approx(
        np.mean([1.0e4, 1.1e4, 1.2e4, 5.0e4]))
    assert float(out['std_flux_fit'][0]) > 0


def test_distinct_stars_keep_separate_ensembles():
    ra, dec, fl, exp, it = _rows([(0.0, 0.0)] * 3, 3)
    # a second star 5 arcsec away, seen in two exposures
    for e in range(2):
        for k in ITERS:
            ra.append(RA0 + (5.0 / 3600.0) / np.cos(np.radians(DEC0)))
            dec.append(DEC0)
            fl.append(2.0e4)
            exp.append(f'jw10678132001_02101_{e:05d}_nrcalong')
            it.append(k)
    out = mc._dedup_satstar_catalog(_table(ra, dec, fl, exp, it))
    assert len(out) == 2
    assert sorted(int(v) for v in out['n_frames_fit']) == [2, 3]


def test_legacy_table_without_provenance_still_gets_columns():
    # a cache written before this change carries no exposure key; the count
    # then degrades to a row count, which is documented, not a crash
    out = mc._dedup_satstar_catalog(
        _table(*_rows([(0.0, 0.0)] * 2, 2), with_provenance=False))
    assert int(out['n_frames_fit'][0]) == 2 * len(ITERS)
    assert int(out['n_meas_fit'][0]) == 2 * len(ITERS)


def test_empty_and_single_row_inputs_are_safe():
    empty = _table([], [], [], [], [])
    assert len(mc._dedup_satstar_catalog(empty)) == 0
    one = _table(*_rows([(0.0, 0.0)], 1, iterations=('m7',)))
    out = mc._dedup_satstar_catalog(one)
    assert len(out) == 1


# --------------------------------------------------------------------------
# the NaN-into-int32 fill
# --------------------------------------------------------------------------

def test_blank_for_dtype_never_produces_int32_min():
    assert mc._blank_for_dtype(np.int32, 3).tolist() == [0, 0, 0]
    assert mc._blank_for_dtype(np.int64, 2).tolist() == [0, 0]
    assert np.isnan(mc._blank_for_dtype(np.float64, 2)).all()
    assert mc._blank_for_dtype(np.bool_, 2).tolist() == [False, False]
    assert mc._blank_for_dtype(np.dtype('U4'), 2).tolist() == ['', '']
    for dt in (np.int32, np.int64):
        assert np.iinfo(dt).min not in mc._blank_for_dtype(dt, 4).tolist()


def test_appended_rows_take_counts_from_the_ensemble():
    cat = Table()
    cat['flux'] = np.array([1.0, 2.0])
    cat['nmatch'] = np.array([3, 4], dtype=np.int32)
    cat['nmatch_good'] = np.array([3, 4], dtype=np.int32)
    cat['std_ra'] = np.array([1e-7, 2e-7])
    cat['std_dec'] = np.array([1e-7, 2e-7])
    cat['some_flag'] = np.array([True, False])
    cat['iter_found'] = np.array([1, 2], dtype=np.int32)

    toadd = Table()
    toadd['flux'] = np.array([9.0])
    toadd['n_frames_fit'] = np.array([4], dtype=np.int32)
    toadd['n_meas_fit'] = np.array([24], dtype=np.int32)
    toadd['std_ra_fit'] = np.array([2.6e-7])
    # std_ra is fed the COORDINATE-convention column (see the RA-convention
    # tests below); std_ra_fit is the on-sky angle and goes to satstar_std_ra
    toadd['std_ra_coord_fit'] = np.array([3.0e-7])
    toadd['std_dec_fit'] = np.array([4.0e-7])

    mc._fill_satstar_added_columns(toadd, cat)

    assert int(toadd['nmatch'][0]) == 4
    assert int(toadd['nmatch_good'][0]) == 4
    assert toadd['nmatch_good'].dtype == np.int32
    assert float(toadd['std_ra'][0]) == pytest.approx(3.0e-7)
    assert float(toadd['std_dec'][0]) == pytest.approx(4.0e-7)
    # unmapped columns get a dtype-appropriate blank, NOT NaN-cast-to-int
    assert int(toadd['iter_found'][0]) == 0
    assert bool(toadd['some_flag'][0]) is False
    for col in ('nmatch', 'nmatch_good', 'iter_found'):
        assert int(toadd[col][0]) != np.iinfo(np.int32).min


def test_appended_rows_survive_a_missing_ensemble_column():
    # a satstar catalog built by an older consolidation has no n_frames_fit;
    # the count must still be a legal integer rather than INT32_MIN
    cat = Table()
    cat['flux'] = np.array([1.0])
    cat['nmatch_good'] = np.array([3], dtype=np.int32)
    toadd = Table()
    toadd['flux'] = np.array([9.0])
    mc._fill_satstar_added_columns(toadd, cat)
    assert int(toadd['nmatch_good'][0]) == 0


def test_nan_frames_do_not_become_int32_min():
    cat = Table()
    cat['nmatch_good'] = np.array([3], dtype=np.int32)
    toadd = Table()
    toadd['n_frames_fit'] = np.array([np.nan])
    mc._fill_satstar_added_columns(toadd, cat)
    assert int(toadd['nmatch_good'][0]) == 0


# --------------------------------------------------------------------------
# source guards
# --------------------------------------------------------------------------

def _source():
    return open(mc.__file__).read()


def test_the_blanket_nan_fill_is_gone_from_both_append_branches():
    src = _source()
    # both the crowdsource and the DAOPHOT branch appended satstar rows this
    # way; a mutation restoring either one must fail here
    assert 'np.ones(len(satstar_toadd)) * np.nan' not in src
    assert 'np.ones(len(satstar_toadd))*np.nan' not in src
    # count CALL sites only: the def line carries the same argument names
    n_sites = src.count('\n        _fill_satstar_added_columns(satstar_toadd, cat)\n')
    assert n_sites == 2, f'expected both append branches routed, found {n_sites}'
    assert src.count('def _fill_satstar_added_columns(') == 1


def test_dedup_returns_through_the_ensemble_attachment():
    src = _source()
    # the dedup must not return tbl[...] directly again: that is the shape that
    # dropped every absorbed row without measuring it
    assert 'return _attach_satstar_ensemble(' in src
    body = src.split('def _dedup_satstar_catalog(')[1].split('\ndef ')[0]
    assert 'return tbl[np.sort(kept)]' not in body


def test_dedup_algorithm_token_was_bumped():
    # the consolidated cache is keyed on this; without a bump the pre-#925
    # cache keeps being served and none of the above reaches a real catalog
    assert mc._SATSTAR_DEDUP_ALG != 'fp4'


def test_every_mapped_column_names_a_real_ensemble_column():
    for src_col in mc._SATSTAR_ASTROMETRY_FROM_ENSEMBLE.values():
        assert src_col in mc.SATSTAR_ENSEMBLE_COLUMNS


# --------------------------------------------------------------------------
# RA convention (reviewer finding on #926)
# --------------------------------------------------------------------------

def test_std_ra_fit_is_an_on_sky_angle():
    # exposures offset by +-d in TRUE ANGLE on the sky; _rows already divides
    # by cos(dec) when building RA, so this is the angular scatter
    d = 10.0
    offs = [(-d, 0.0), (-d, 0.0), (d, 0.0), (d, 0.0)]
    out = mc._dedup_satstar_catalog(_table(*_rows(offs, 4)))
    got = float(out['std_ra_fit'][0]) * 3.6e6
    assert got == pytest.approx(d * np.sqrt(4.0 / 3.0), rel=1e-3)


def test_std_ra_coord_fit_is_the_coordinate_convention():
    # combine_singleframe's std_ra is the scatter of the RA COORDINATE with no
    # cos(dec) factor, so the column fed from here has to match it or one
    # column holds two conventions
    d = 10.0
    offs = [(-d, 0.0), (-d, 0.0), (d, 0.0), (d, 0.0)]
    out = mc._dedup_satstar_catalog(_table(*_rows(offs, 4)))
    on_sky = float(out['std_ra_fit'][0])
    coord = float(out['std_ra_coord_fit'][0])
    assert coord == pytest.approx(on_sky / np.cos(np.radians(DEC0)), rel=1e-6)
    assert coord > on_sky          # cos(dec) < 1 at this declination


def test_std_ra_is_filled_in_the_coordinate_convention_not_the_on_sky_one():
    # the mapping must reach for std_ra_coord_fit; feeding std_ra_fit would put
    # a 12.5% smaller number in std_ra at dec -28.9 than a daophot row carries
    # for the same true scatter
    assert mc._SATSTAR_ASTROMETRY_FROM_ENSEMBLE['std_ra'] == 'std_ra_coord_fit'
    assert mc._SATSTAR_ASTROMETRY_FROM_ENSEMBLE['satstar_std_ra'] == 'std_ra_fit'

    cat = Table()
    cat['std_ra'] = np.array([1.0e-7])
    cat['satstar_std_ra'] = np.array([1.0e-7])
    toadd = Table()
    toadd['std_ra_fit'] = np.array([2.0e-7])
    toadd['std_ra_coord_fit'] = np.array([3.0e-7])
    mc._fill_satstar_added_columns(toadd, cat)
    assert float(toadd['std_ra'][0]) == pytest.approx(3.0e-7)
    assert float(toadd['satstar_std_ra'][0]) == pytest.approx(2.0e-7)


def test_dec_column_carries_no_cos_factor_in_either_convention():
    # only RA has the ambiguity; a mutation adding cos(dec) to dec must fail
    offs = [(0.0, -10.0), (0.0, -10.0), (0.0, 10.0), (0.0, 10.0)]
    out = mc._dedup_satstar_catalog(_table(*_rows(offs, 4)))
    got = float(out['std_dec_fit'][0]) * 3.6e6
    assert got == pytest.approx(10.0 * np.sqrt(4.0 / 3.0), rel=1e-3)
