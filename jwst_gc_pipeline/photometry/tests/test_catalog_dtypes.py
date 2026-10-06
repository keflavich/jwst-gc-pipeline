"""Unit tests for catalog_dtypes.downcast_catalog_dtypes.

Per-frame ``*_daophot_basic.fits`` / ``*_satstar_catalog.fits`` catalogs carry
``float64``/``int64`` columns at write time; the helper under test shrinks the
photometric-metric / counter columns to ``float32``/``int32`` while leaving
every pixel-position, pixel-position-error, and sky-coordinate column (plus
any column not on either allowlist) exactly as it was.
"""
import numpy as np
import pytest

from astropy.table import Table, MaskedColumn
from astropy.coordinates import SkyCoord
import astropy.units as u

from jwst_gc_pipeline.photometry.catalog_dtypes import (
    FLOAT32_COLUMNS, INT32_COLUMNS, downcast_catalog_dtypes)


def _sample_table(n=5):
    rng = np.random.default_rng(0)
    t = Table()
    # Position / position-error columns -- MUST stay float64.
    t['x_fit'] = rng.uniform(1, 2048, n).astype(np.float64)
    t['y_fit'] = rng.uniform(1, 2048, n).astype(np.float64)
    t['x_init'] = t['x_fit'] + 0.01
    t['y_init'] = t['y_fit'] + 0.01
    t['x_err'] = rng.uniform(1e-3, 0.5, n).astype(np.float64)
    t['y_err'] = rng.uniform(1e-3, 0.5, n).astype(np.float64)
    t['x_0'] = t['x_fit'].copy()
    t['y_0'] = t['y_fit'].copy()
    t['xcentroid'] = t['x_fit'].copy()
    t['ycentroid'] = t['y_fit'].copy()
    t['dra'] = rng.uniform(1e-4, 1.0, n).astype(np.float64)
    t['ddec'] = rng.uniform(1e-4, 1.0, n).astype(np.float64)
    t['sat_com_ra'] = rng.uniform(266.6, 266.7, n).astype(np.float64)
    t['sat_com_dec'] = rng.uniform(-28.6, -28.5, n).astype(np.float64)
    t['skycoord_centroid'] = SkyCoord(ra=rng.uniform(266.6, 266.7, n) * u.deg,
                                      dec=rng.uniform(-28.6, -28.5, n) * u.deg)

    # Photometric-metric columns -- should downcast to float32.
    t['local_bkg'] = rng.uniform(-10, 60, n).astype(np.float64)
    t['flux_init'] = rng.uniform(0, 2e4, n).astype(np.float64)
    t['flux_fit'] = rng.uniform(1, 3e4, n).astype(np.float64)
    t['flux_err'] = rng.uniform(1, 500, n).astype(np.float64)
    t['flux_fit_raw'] = t['flux_fit'].copy()
    t['qfit'] = rng.uniform(1e-3, 6, n).astype(np.float64)
    t['cfit'] = rng.uniform(-0.3, 0.3, n).astype(np.float64)
    t['reduced_chi2'] = rng.uniform(0.3, 2500, n).astype(np.float64)
    t['model_data_peak_ratio'] = rng.uniform(0.1, 3, n).astype(np.float64)
    t['local_bkg_resbgsub'] = t['local_bkg'].copy()
    t['modelsub_bkg'] = rng.uniform(-60, 360, n).astype(np.float64)
    t['modelsub_bkg_rms'] = rng.uniform(0.1, 500, n).astype(np.float64)
    t['sidelobe_resid_sigma'] = rng.uniform(-20, 2.6e7, n).astype(np.float64)
    t['ssr_ratio'] = rng.uniform(0.1, 40, n).astype(np.float64)
    t['wingcal_rmask'] = rng.uniform(0.5, 5, n).astype(np.float64)
    t['wingcal_ratio'] = np.ones(n, dtype=np.float64)
    t['sat_severity_floor'] = rng.uniform(0, 1, n).astype(np.float64)
    t['satstar_implied_peak'] = rng.uniform(1, 1e4, n).astype(np.float64)
    t['satstar_observed_peak'] = rng.uniform(1, 1e4, n).astype(np.float64)

    # Counter / bitmask columns -- should downcast to int32.
    t['id'] = np.arange(1, n + 1, dtype=np.int64)
    t['group_id'] = np.arange(1, n + 1, dtype=np.int64)
    t['group_size'] = np.ones(n, dtype=np.int64)
    t['n_pixels_fit'] = (np.arange(n, dtype=np.int64) + 5)
    t['flags'] = np.array([0, 1, 24, 25, 0][:n], dtype=np.int64)
    t['sat_area'] = (np.arange(n, dtype=np.int64) + 6)
    t['modelsub_bkg_npix'] = (np.arange(n, dtype=np.int64) + 3)

    # Columns not on either allowlist -- must be left completely alone.
    t['model_overshoot'] = np.zeros(n, dtype=bool)
    t['seed_kind'] = np.array(['fresh'] * n)
    t['some_future_column'] = rng.uniform(0, 1, n).astype(np.float64)
    return t


POSITION_COLUMNS = ('x_fit', 'y_fit', 'x_init', 'y_init', 'x_err', 'y_err',
                    'x_0', 'y_0', 'xcentroid', 'ycentroid')
UNTOUCHED_EXTRA = ('dra', 'ddec', 'sat_com_ra', 'sat_com_dec',
                   'model_overshoot', 'seed_kind', 'some_future_column')


def test_float_columns_downcast_to_float32():
    t = _sample_table()
    downcast_catalog_dtypes(t)
    for name in FLOAT32_COLUMNS:
        assert t[name].dtype == np.float32, name


def test_int_columns_downcast_to_int32():
    t = _sample_table()
    downcast_catalog_dtypes(t)
    for name in INT32_COLUMNS:
        assert t[name].dtype == np.int32, name


def test_positions_and_skycoords_untouched():
    t = _sample_table()
    before = {name: np.asarray(t[name]).copy() for name in POSITION_COLUMNS}
    before_ra = t['skycoord_centroid'].ra.deg.copy()
    before_dec = t['skycoord_centroid'].dec.deg.copy()
    downcast_catalog_dtypes(t)
    for name in POSITION_COLUMNS:
        assert t[name].dtype == np.float64, name
        np.testing.assert_array_equal(np.asarray(t[name]), before[name])
    np.testing.assert_array_equal(t['skycoord_centroid'].ra.deg, before_ra)
    np.testing.assert_array_equal(t['skycoord_centroid'].dec.deg, before_dec)


def test_columns_not_on_either_allowlist_untouched():
    t = _sample_table()
    before = {name: (np.asarray(t[name]).copy(), t[name].dtype)
              for name in UNTOUCHED_EXTRA}
    downcast_catalog_dtypes(t)
    for name, (arr, dtype) in before.items():
        assert t[name].dtype == dtype, name
        np.testing.assert_array_equal(np.asarray(t[name]), arr)


def test_round_trip_error_bounds_on_float_columns():
    """float64 -> float32 relative error stays near machine epsilon for
    float32 (~1.2e-7), far below any measurement uncertainty these columns
    carry."""
    t = _sample_table()
    originals = {name: np.asarray(t[name]).astype(np.float64).copy()
                 for name in FLOAT32_COLUMNS}
    downcast_catalog_dtypes(t)
    for name, orig in originals.items():
        new = np.asarray(t[name]).astype(np.float64)
        nonzero = orig != 0
        rel_err = np.abs((new[nonzero] - orig[nonzero]) / orig[nonzero])
        assert np.all(rel_err < 1e-6), (name, rel_err.max())


def test_unknown_table_with_no_allowlisted_columns_is_a_noop():
    t = Table()
    t['foo'] = np.array([1.0, 2.0], dtype=np.float64)
    downcast_catalog_dtypes(t)
    assert t['foo'].dtype == np.float64


def test_int_column_exceeding_int32_range_is_left_alone():
    t = Table()
    t['id'] = np.array([1, 2, 2**32], dtype=np.int64)
    downcast_catalog_dtypes(t)
    assert t['id'].dtype == np.int64


def test_masked_float_column_keeps_mask_after_downcast():
    t = Table()
    t['flux_fit'] = MaskedColumn([1.0, 2.0, 3.0], mask=[False, True, False],
                                  dtype=np.float64)
    downcast_catalog_dtypes(t)
    assert t['flux_fit'].dtype == np.float32
    assert list(t['flux_fit'].mask) == [False, True, False]


def test_returns_the_same_table_mutated_in_place():
    t = _sample_table()
    out = downcast_catalog_dtypes(t)
    assert out is t


def test_downcasts_big_endian_columns_from_a_fits_round_trip(tmp_path):
    """A table just read from FITS carries big-endian (">f8"/">i8") dtypes;
    the helper must still recognize and downcast them (a dtype == np.float64
    check is False for ">f8" and would silently no-op on every real catalog).
    """
    t = _sample_table()
    path = tmp_path / 'roundtrip.fits'
    t.write(str(path))
    t2 = Table.read(str(path))
    assert t2['flux_fit'].dtype.kind == 'f' and t2['flux_fit'].dtype.itemsize == 8
    assert t2['flux_fit'].dtype != np.float64  # confirms it is big-endian
    assert t2['id'].dtype != np.int64
    downcast_catalog_dtypes(t2)
    assert t2['flux_fit'].dtype == np.float32
    assert t2['id'].dtype == np.int32
    assert t2['x_fit'].dtype.kind == 'f' and t2['x_fit'].dtype.itemsize == 8


def test_empty_table_is_a_noop():
    t = _sample_table(n=0)
    downcast_catalog_dtypes(t)
    for name in FLOAT32_COLUMNS:
        if name in t.colnames:
            assert len(t[name]) == 0


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))


def test_fully_masked_int_column_is_cast_without_maskerror():
    """Regression: seeded fits can leave an int counter fully masked, and
    int(np.min(col)) on that raised MaskError inside save_photutils_results."""
    t = Table()
    t['group_id'] = MaskedColumn(np.array([1, 2, 3], dtype=np.int64),
                                 mask=[True, True, True])
    downcast_catalog_dtypes(t)
    assert t['group_id'].dtype == np.int32
    assert t['group_id'].mask.all()
