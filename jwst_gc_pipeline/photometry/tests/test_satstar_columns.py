"""satstar catalog column vintage: detector-frame x_fit/y_fit and a unique id (#949)."""
import logging

import numpy as np
from astropy.table import Table

from jwst_gc_pipeline.photometry.satstar_columns import (
    XYFRAME_KEY, finalize_satstar_columns, is_detector_frame)


def _old_vintage(n=5):
    """Stamp-local x_fit/y_fit, id 1 on every row, detector pixels in xcentroid."""
    x_cut = np.linspace(40.0, 82.5, n)
    y_cut = np.linspace(41.0, 80.0, n)
    x0 = np.linspace(100.0, 2040.0, n) - 40.0
    y0 = np.linspace(200.0, 1900.0, n) - 40.0
    return Table({
        'id': np.ones(n, dtype=int),
        'x_init': x_cut + 0.3, 'y_init': y_cut - 0.2,
        'x_fit': x_cut, 'y_fit': y_cut,
        'flux_fit': np.full(n, 1e4),
        'xcentroid': x_cut + x0, 'ycentroid': y_cut + y0,
    })


def test_old_vintage_is_rewritten():
    tbl = _old_vintage()
    orig = tbl.copy()
    out = finalize_satstar_columns(tbl)
    assert out is tbl
    assert len(set(tbl['id'])) == len(tbl)
    assert list(tbl['id']) == [1, 2, 3, 4, 5]
    assert tbl.colnames[0] == 'id'
    assert tbl['x_fit'].max() > 100
    np.testing.assert_array_equal(tbl['x_fit'], orig['xcentroid'])
    np.testing.assert_array_equal(tbl['y_fit'], orig['ycentroid'])
    np.testing.assert_array_equal(tbl['x_fit_cutout'], orig['x_fit'])
    np.testing.assert_array_equal(tbl['y_fit_cutout'], orig['y_fit'])
    np.testing.assert_array_equal(tbl['x_init_cutout'], orig['x_init'])
    np.testing.assert_array_equal(tbl['cutout_id'], orig['id'])
    # the init-to-fit offset is a property of the fit and survives the shift
    np.testing.assert_allclose(tbl['x_init'] - tbl['x_fit'],
                               orig['x_init'] - orig['x_fit'], atol=1e-9)
    np.testing.assert_allclose(tbl['y_init'] - tbl['y_fit'],
                               orig['y_init'] - orig['y_fit'], atol=1e-9)
    assert tbl.meta[XYFRAME_KEY] == 'detector'
    assert is_detector_frame(tbl)


def test_second_call_is_a_no_op():
    tbl = _old_vintage()
    finalize_satstar_columns(tbl)
    snapshot = tbl.copy()
    finalize_satstar_columns(tbl)
    assert tbl.colnames == snapshot.colnames
    for name in snapshot.colnames:
        np.testing.assert_array_equal(tbl[name], snapshot[name])


def test_table_without_detector_columns_is_untouched(caplog):
    tbl = _old_vintage().copy()
    tbl.remove_columns(['xcentroid', 'ycentroid'])
    snapshot = tbl.copy()
    with caplog.at_level(logging.WARNING):
        finalize_satstar_columns(tbl)
    assert 'stamp-local' in caplog.text
    assert tbl.colnames == snapshot.colnames
    for name in snapshot.colnames:
        np.testing.assert_array_equal(tbl[name], snapshot[name])
    assert not is_detector_frame(tbl)


def test_missing_optional_columns_are_skipped():
    tbl = _old_vintage()
    tbl.remove_columns(['x_init', 'y_init'])
    finalize_satstar_columns(tbl)
    assert 'x_init_cutout' not in tbl.colnames
    assert 'x_init' not in tbl.colnames
    assert 'x_fit_cutout' in tbl.colnames


def test_empty_table():
    tbl = _old_vintage()[:0]
    finalize_satstar_columns(tbl)
    assert len(tbl) == 0
    assert is_detector_frame(tbl)


def test_fits_round_trip_keeps_the_stamp(tmp_path):
    tbl = _old_vintage()
    finalize_satstar_columns(tbl)
    path = tmp_path / 'cat.fits'
    tbl.write(path)
    back = Table.read(path)
    assert is_detector_frame(back)
    snapshot = back.copy()
    finalize_satstar_columns(back)
    np.testing.assert_array_equal(back['x_fit'], snapshot['x_fit'])
    np.testing.assert_array_equal(back['id'], snapshot['id'])


def test_cache_hit_serves_the_detector_vintage(tmp_path, monkeypatch):
    """A satstar catalog already on disk (cache key carries no code version)
    is normalised when load_or_make_satstar_catalog serves it."""
    import jwst_gc_pipeline.photometry.crowdsource_catalogs_long as CL
    monkeypatch.setenv('SATSTAR_SEED_CORE_DQ', '0')
    tbl = _old_vintage()
    tbl.meta['SATRECOV'] = 'off'
    tbl.write(tmp_path / 'frame_satstar_catalog.fits')
    out = CL.load_or_make_satstar_catalog(
        str(tmp_path / 'frame.fits'), path_prefix=str(tmp_path),
        recovery_signature='off')
    assert is_detector_frame(out)
    assert len(set(out['id'])) == len(out)
    assert out['x_fit'].max() > 100
    np.testing.assert_array_equal(out['x_fit'], _old_vintage()['xcentroid'])
