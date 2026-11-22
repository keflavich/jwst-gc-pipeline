"""Residual-support guard for rows a frame's merged-catalog residual shows are
over-subtracted (build_mergedcat_residuals -> _resid_supported_shrink).

A single-detection row, or an uncovered replaced_saturated row, carries the
flux of the frame that detected it, and the render draws it at that flux in
every frame that covers it.  Where a frame's data show less, the render digs a
crater.  ``MERGE_ROW_RESID_SUPPORT_MIN`` (off by default) renders such a row at
the flux the frame's residual supports.
"""
import glob

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

import jwst_gc_pipeline.photometry.crowdsource_catalogs_long as ccl
from jwst_gc_pipeline.photometry.crowdsource_catalogs_long import (
    _resid_supported_shrink,
    _row_resid_support_params,
)
from jwst_gc_pipeline.photometry.tests import test_mergedcat_satstar_cover as cover
from jwst_gc_pipeline.photometry.tests.test_satrow_data_guard import _stamp_at

SHAPE = (80, 80)
PSF_SHAPE = (21, 21)


def _hole(rows, shape=SHAPE):
    """The residual left by rendering ``rows`` (x, y, flux) on a frame that shows nothing there."""
    out = np.zeros(shape)
    for x0, y0, flux in rows:
        out -= _stamp_at(0, shape[1], 0, shape[0], flux, x0, y0)
    return out


def _star(x0, y0, flux, shape=SHAPE):
    return _stamp_at(0, shape[1], 0, shape[0], flux, x0, y0)


def _shrink(resid, rows, min_frac=0.3, **kw):
    xs, ys, fs = (np.array(c, dtype=float) for c in zip(*rows))
    return _resid_supported_shrink(resid, xs, ys, fs, _stamp_at, PSF_SHAPE, min_frac, **kw)


def test_shrink_takes_back_a_row_the_data_do_not_show():
    corr, n = _shrink(_hole([(40.3, 41.6, 1000.0)]), [(40.3, 41.6, 1000.0)])
    assert n == 1 and corr[0] == pytest.approx(-1000.0, rel=1e-3)


def test_shrink_leaves_a_row_the_frame_shows_at_the_merged_flux():
    corr, n = _shrink(np.zeros(SHAPE), [(40.0, 40.0, 1000.0)])
    assert n == 0 and corr[0] == 0.0


def test_shrink_corrects_only_where_the_data_show_less_than_the_fraction():
    # the data show half of the merged flux: residual hole of 500
    resid = _hole([(40.0, 40.0, 500.0)])
    corr, n = _shrink(resid, [(40.0, 40.0, 1000.0)], min_frac=0.3)
    assert n == 0                                    # half >= 0.3: merged flux kept
    corr, n = _shrink(resid, [(40.0, 40.0, 1000.0)], min_frac=0.6)
    assert n == 1 and corr[0] == pytest.approx(-500.0, rel=1e-3)


def test_shrink_never_removes_more_than_the_flux():
    corr, n = _shrink(_hole([(40.0, 40.0, 3000.0)]), [(40.0, 40.0, 1000.0)])
    assert corr[0] == pytest.approx(-1000.0)


def test_shrink_ignores_a_background_offset():
    resid = _hole([(40.0, 40.0, 1000.0)]) + 25.0
    corr, n = _shrink(resid, [(40.0, 40.0, 1000.0)])
    assert corr[0] == pytest.approx(-1000.0, rel=1e-2)
    corr, n = _shrink(np.full(SHAPE, 25.0), [(40.0, 40.0, 1000.0)])
    assert n == 0


def test_shrink_ignores_a_neighbour_already_subtracted():
    # a bright neighbour 6 px away is in the residual only as its own (zero) remnant
    resid = _hole([(40.0, 40.0, 1000.0)])
    corr, n = _shrink(resid, [(40.0, 40.0, 1000.0), (46.0, 40.0, 50000.0)])
    assert n == 1 and corr[0] == pytest.approx(-1000.0, rel=1e-3) and corr[1] == 0.0


def test_two_rows_that_dig_one_hole_share_the_correction():
    # the data show 1000 of the 2000 the two rows render at one place
    rows = [(40.0, 40.0, 1000.0), (40.2, 40.0, 1000.0)]
    resid = _hole(rows) + _star(40.1, 40.0, 1000.0)
    corr, n = _shrink(resid, rows, min_frac=0.6)
    assert corr.sum() == pytest.approx(-1000.0, rel=0.05)
    # what is left is the 0.1 px offset of the data star from the row that kept its flux
    left = resid + sum(_hole([(x, y, c)]) for (x, y, _f), c in zip(rows, corr))
    assert np.abs(left).max() < 0.1 * _star(40.1, 40.0, 1000.0).max()


@pytest.mark.parametrize('where', ['core_all_nan', 'ring_all_nan', 'outside'])
def test_shrink_leaves_a_row_the_frame_cannot_test(where):
    resid = _hole([(40.0, 40.0, 1000.0)])
    x = 40.0
    if where == 'core_all_nan':
        resid[37:44, 37:44] = np.nan
    elif where == 'ring_all_nan':
        yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
        resid[(np.hypot(xx - 40, yy - 40) >= 5.5) & (np.hypot(xx - 40, yy - 40) <= 10.5)] = np.nan
    else:
        x = 400.0
    corr, n = _shrink(resid, [(x, 40.0, 1000.0)])
    assert n == 0 and corr[0] == 0.0


def test_shrink_tests_a_core_with_a_few_nan_pixels():
    resid = _hole([(40.0, 40.0, 1000.0)])
    resid[40, 40] = np.nan
    resid[39, 41] = np.nan
    corr, n = _shrink(resid, [(40.0, 40.0, 1000.0)])
    assert n == 1 and corr[0] == pytest.approx(-1000.0, rel=1e-2)


def test_shrink_tests_a_row_next_to_the_frame_edge():
    resid = _hole([(1.3, 40.0, 1000.0)])
    corr, n = _shrink(resid, [(1.3, 40.0, 1000.0)])
    assert n == 1 and corr[0] == pytest.approx(-1000.0, rel=5e-2)


def test_params_env(monkeypatch):
    monkeypatch.delenv('MERGE_ROW_RESID_SUPPORT_MIN', raising=False)
    monkeypatch.delenv('MERGE_ROW_RESID_NMATCH_MAX', raising=False)
    assert _row_resid_support_params() == (0.0, 99)
    monkeypatch.setenv('MERGE_ROW_RESID_SUPPORT_MIN', '0.4')
    monkeypatch.setenv('MERGE_ROW_RESID_NMATCH_MAX', '3')
    assert _row_resid_support_params() == (0.4, 3)
    monkeypatch.setenv('MERGE_ROW_RESID_SUPPORT_MIN', '  ')
    monkeypatch.setenv('MERGE_ROW_RESID_NMATCH_MAX', ' ')
    assert _row_resid_support_params() == (0.0, 99)


@pytest.mark.parametrize('name, value', [('MERGE_ROW_RESID_SUPPORT_MIN', 'abc'),
                                         ('MERGE_ROW_RESID_SUPPORT_MIN', 'nan'),
                                         ('MERGE_ROW_RESID_SUPPORT_MIN', '-0.1'),
                                         ('MERGE_ROW_RESID_SUPPORT_MIN', '1.5'),
                                         ('MERGE_ROW_RESID_NMATCH_MAX', '0'),
                                         ('MERGE_ROW_RESID_NMATCH_MAX', 'x'),
                                         ('MERGE_ROW_RESID_NMATCH_MAX', '1.5')])
def test_params_reject_a_value_out_of_range(monkeypatch, name, value):
    monkeypatch.delenv('MERGE_ROW_RESID_SUPPORT_MIN', raising=False)
    monkeypatch.delenv('MERGE_ROW_RESID_NMATCH_MAX', raising=False)
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        _row_resid_support_params()


# --- end to end through build_mergedcat_residuals ---------------------------
# The synthetic frame of test_mergedcat_satstar_cover: A is in the frame's
# satstar model, B is a replaced_saturated row this frame's satstar pass
# rejected, C is an ordinary row.  Here the DATA carry B and C at ``b_scale`` /
# ``c_scale`` times the merged flux (0 = a frame that shows nothing there), and
# the merged catalog carries ``nmatch`` for C (None = no nmatch column).

def _run(tmp_path, monkeypatch, b_scale, c_scale, support_min, nmatch_c=1,
         nmatch_max=None, instrument=None):
    for var in ('MERGE_SATROW_DATA_SUPPORT_MIN', 'MERGE_ROW_RESID_SUPPORT_MIN',
                'MERGE_ROW_RESID_NMATCH_MAX'):
        monkeypatch.delenv(var, raising=False)
    if support_min is not None:
        monkeypatch.setenv('MERGE_ROW_RESID_SUPPORT_MIN', support_min)
    if nmatch_max is not None:
        monkeypatch.setenv('MERGE_ROW_RESID_NMATCH_MAX', nmatch_max)
    real_setup = cover._e2e_setup

    def setup(tmp, grid):
        frame, merged = real_setup(tmp, grid)
        base = ccl._render_model_from_table(
            Table({'x_fit': [cover.E2E_B[0], cover.E2E_C[0]],
                   'y_fit': [cover.E2E_B[1], cover.E2E_C[1]],
                   'flux_fit': [b_scale * cover.E2E_FLUX['B'],
                                c_scale * cover.E2E_FLUX['C']]}),
            grid, cover.E2E_SHAPE, (21, 21))
        (path,) = glob.glob(str(tmp / 'F480M' / 'pipeline' / '*_daophot_basic_residual.fits'))
        fits.HDUList([fits.PrimaryHDU(),
                      fits.ImageHDU(base.astype('float32'),
                                    header=fits.getheader(path, 'SCI'),
                                    name='SCI')]).writeto(path, overwrite=True)
        if nmatch_c is not None:
            cat = Table.read(merged)
            cat['nmatch'] = [3, 3, nmatch_c]
            cat.write(merged, overwrite=True)
        return frame, merged
    monkeypatch.setattr(cover, '_e2e_setup', setup)
    return cover._run_e2e(tmp_path, monkeypatch, None, instrument=instrument)


def _c_peak():
    zero = np.zeros((1, 1))
    return cover.E2E_FLUX['C'] * float(cover._e2e_grid().evaluate(zero, zero, 1.0, 0.0, 0.0)[0, 0])


@pytest.mark.filterwarnings('ignore::jwst_gc_pipeline.frame_wcs.MissingGwcsWarning')
def test_e2e_default_leaves_the_crater_of_a_row_the_frame_does_not_show(tmp_path, monkeypatch, capsys):
    calls, resid, _model = _run(tmp_path, monkeypatch, 1.0, 0.0, None)
    assert resid[cover._peak_px(cover.E2E_C)] == pytest.approx(-_c_peak(), rel=1e-3)
    assert 'MERGE_ROW_RESID_SUPPORT_MIN' not in capsys.readouterr().out


@pytest.mark.filterwarnings('ignore::jwst_gc_pipeline.frame_wcs.MissingGwcsWarning')
def test_e2e_guard_leaves_no_crater_at_a_single_detection_row(tmp_path, monkeypatch, capsys):
    calls, resid, model = _run(tmp_path, monkeypatch, 1.0, 0.0, '0.3')
    assert np.abs(resid).max() < 1e-3 * _c_peak()
    out = capsys.readouterr().out
    assert '1 of 2 tested rows rendered at the flux this frame\'s residual supports' in out
    assert 'MERGE_ROW_RESID_SUPPORT_MIN' in out


@pytest.mark.filterwarnings('ignore::jwst_gc_pipeline.frame_wcs.MissingGwcsWarning')
def test_e2e_guard_leaves_a_row_the_frame_shows_unchanged(tmp_path, monkeypatch, capsys):
    on = _run(tmp_path / 'on', monkeypatch, 1.0, 1.0, '0.3')
    off = _run(tmp_path / 'off', monkeypatch, 1.0, 1.0, None)
    np.testing.assert_allclose(on[1], off[1], atol=1e-9)
    assert '0 of 2 tested rows' in capsys.readouterr().out


@pytest.mark.filterwarnings('ignore::jwst_gc_pipeline.frame_wcs.MissingGwcsWarning')
def test_e2e_guard_tests_a_row_matched_in_more_frames_unless_capped(tmp_path, monkeypatch):
    _c, resid, _m = _run(tmp_path / 'a', monkeypatch, 1.0, 0.0, '0.3', nmatch_c=3, nmatch_max='1')
    assert resid[cover._peak_px(cover.E2E_C)] == pytest.approx(-_c_peak(), rel=1e-3)
    _c, resid, _m = _run(tmp_path / 'b', monkeypatch, 1.0, 0.0, '0.3', nmatch_c=3)
    assert np.abs(resid).max() < 1e-3 * _c_peak()


@pytest.mark.filterwarnings('ignore::jwst_gc_pipeline.frame_wcs.MissingGwcsWarning')
def test_e2e_guard_tests_the_uncovered_saturated_rows_without_an_nmatch_column(tmp_path, monkeypatch, capsys):
    _c, resid, _m = _run(tmp_path, monkeypatch, 0.0, 0.0, '0.3', nmatch_c=None)
    # B (uncovered saturated) is corrected; C is not tested for want of nmatch
    assert resid[cover._peak_px(cover.E2E_B)] == pytest.approx(0.0, abs=1e-3 * _c_peak())
    assert resid[cover._peak_px(cover.E2E_C)] == pytest.approx(-_c_peak(), rel=1e-3)
    assert 'has no nmatch column' in capsys.readouterr().out


@pytest.mark.filterwarnings('ignore::jwst_gc_pipeline.frame_wcs.MissingGwcsWarning')
def test_e2e_guard_is_ignored_for_miri(tmp_path, monkeypatch, capsys):
    _run(tmp_path, monkeypatch, 1.0, 0.0, '0.3', instrument='MIRI')
    assert 'MERGE_ROW_RESID_SUPPORT_MIN' not in capsys.readouterr().out
