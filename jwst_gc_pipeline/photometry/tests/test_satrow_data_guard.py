"""Data-support guard for uncovered replaced_saturated rows in the merged-catalog
residual render (build_mergedcat_residuals -> _satrow_supported_flux).

A replaced_saturated row carries the flux of the frames whose satstar pass
fitted it.  A frame that did not renders the row from the merged catalog, and
when that frame's data show nothing at the position (a one-frame saturated
transient fitted as a star) the render digs a crater as deep as the model peak.
``MERGE_SATROW_DATA_SUPPORT_MIN`` (off by default) renders such a row at the
flux the frame's data show.
"""
import glob

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

import jwst_gc_pipeline.photometry.crowdsource_catalogs_long as ccl
from jwst_gc_pipeline.photometry.crowdsource_catalogs_long import (
    _satrow_data_flux,
    _satrow_support_min,
    _satrow_supported_flux,
)
from jwst_gc_pipeline.photometry.tests import test_mergedcat_satstar_cover as cover

SIGMA = 1.1


def _stamp_at(x_lo, x_hi, y_lo, y_hi, flux, x0, y0):
    yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
    g = np.exp(-0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / SIGMA ** 2)
    return flux * g / (2.0 * np.pi * SIGMA ** 2)


def _frame_with(stars, shape=(80, 80), sky=20.0):
    base = np.full(shape, sky)
    for x0, y0, flux in stars:
        base += _stamp_at(0, shape[1], 0, shape[0], flux, x0, y0)
    return base


def test_data_flux_recovers_the_injected_amplitude():
    base = _frame_with([(40.3, 41.6, 1000.0)])
    got = _satrow_data_flux(base, [40.3], [41.6], _stamp_at)
    assert got[0] == pytest.approx(1000.0, rel=1e-3)


def test_data_flux_is_zero_for_a_position_with_only_sky():
    base = _frame_with([])
    assert _satrow_data_flux(base, [40.0], [40.0], _stamp_at)[0] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize('what', ['nan_core', 'edge', 'no_ring', 'nonfinite_position'])
def test_data_flux_is_nan_where_the_frame_cannot_test_the_position(what):
    base = _frame_with([(40.0, 40.0, 1000.0)])
    x, y = 40.0, 40.0
    if what == 'nan_core':
        base[39, 40] = np.nan
    elif what == 'edge':
        x = 5.0
    elif what == 'no_ring':
        base[:] = np.nan
        base[35:46, 35:46] = 20.0
    else:
        x = np.nan
    assert np.isnan(_satrow_data_flux(base, [x], [y], _stamp_at)[0])


def test_supported_flux_changes_only_the_rows_the_data_contradict():
    stars = [(20.0, 20.0, 1000.0),     # as bright as the merged flux
             (20.0, 60.0, 500.0),      # half of it: above the 0.3 floor
             (60.0, 20.0, 100.0)]      # a tenth of it
    base = _frame_with(stars)
    base[58:63, 58:63] = np.nan        # (60, 60): a core the frame cannot test
    xs = [20.0, 20.0, 60.0, 60.0, 40.0]
    ys = [20.0, 60.0, 20.0, 60.0, 40.0]
    flux = [1000.0, 1000.0, 1000.0, 1000.0, 1000.0]    # the last row: sky only
    out, n_cut = _satrow_supported_flux(base, xs, ys, flux, _stamp_at, 0.3)
    assert out[0] == pytest.approx(1000.0)
    assert out[1] == pytest.approx(1000.0)          # 0.5 >= 0.3: merged flux kept
    assert out[2] == pytest.approx(100.0, rel=1e-3)  # rendered at what the data show
    assert out[3] == pytest.approx(1000.0)          # untestable: merged flux kept
    assert out[4] == pytest.approx(0.0, abs=1e-3)   # nothing there: not rendered
    assert n_cut == 2


def test_supported_flux_never_renders_a_negative_flux():
    base = _frame_with([(40.0, 40.0, -300.0)])
    out, n_cut = _satrow_supported_flux(base, [40.0], [40.0], [1000.0], _stamp_at, 0.3)
    assert out[0] == 0.0 and n_cut == 1


def test_support_min_env(monkeypatch):
    monkeypatch.delenv('MERGE_SATROW_DATA_SUPPORT_MIN', raising=False)
    assert _satrow_support_min() == 0.0
    monkeypatch.setenv('MERGE_SATROW_DATA_SUPPORT_MIN', '  ')
    assert _satrow_support_min() == 0.0
    monkeypatch.setenv('MERGE_SATROW_DATA_SUPPORT_MIN', '0.3')
    assert _satrow_support_min() == 0.3


@pytest.mark.parametrize('value', ['abc', 'nan', '-0.1', '1.5'])
def test_support_min_rejects_a_value_outside_zero_one(monkeypatch, value):
    monkeypatch.setenv('MERGE_SATROW_DATA_SUPPORT_MIN', value)
    with pytest.raises(ValueError, match='MERGE_SATROW_DATA_SUPPORT_MIN'):
        _satrow_support_min()


# --- end to end through build_mergedcat_residuals ---------------------------
# The synthetic frame of test_mergedcat_satstar_cover: A is in the frame's
# satstar model, B is a replaced_saturated row this frame's satstar pass
# rejected, C is an ordinary row.  Here the DATA carry B at ``b_scale`` times
# the merged flux (0 = a frame that shows nothing at B).

def _run(tmp_path, monkeypatch, b_scale, support_min, instrument=None):
    monkeypatch.delenv('MERGE_SATROW_DATA_SUPPORT_MIN', raising=False)
    if support_min is not None:
        monkeypatch.setenv('MERGE_SATROW_DATA_SUPPORT_MIN', support_min)
    real_setup = cover._e2e_setup

    def setup(tmp, grid):
        frame, merged = real_setup(tmp, grid)
        base = ccl._render_model_from_table(
            Table({'x_fit': [cover.E2E_B[0], cover.E2E_C[0]],
                   'y_fit': [cover.E2E_B[1], cover.E2E_C[1]],
                   'flux_fit': [b_scale * cover.E2E_FLUX['B'], cover.E2E_FLUX['C']]}),
            grid, cover.E2E_SHAPE, (21, 21))
        (path,) = glob.glob(str(tmp / 'F480M' / 'pipeline' / '*_daophot_basic_residual.fits'))
        fits.HDUList([fits.PrimaryHDU(),
                      fits.ImageHDU(base.astype('float32'),
                                    header=fits.getheader(path, 'SCI'),
                                    name='SCI')]).writeto(path, overwrite=True)
        return frame, merged
    monkeypatch.setattr(cover, '_e2e_setup', setup)
    return cover._run_e2e(tmp_path, monkeypatch, None, instrument=instrument)


@pytest.mark.filterwarnings('ignore::jwst_gc_pipeline.frame_wcs.MissingGwcsWarning')
def test_e2e_default_renders_the_absent_star_at_the_merged_flux(tmp_path, monkeypatch, capsys):
    calls, resid, _model = _run(tmp_path, monkeypatch, 0.0, None)
    assert calls == [[cover.E2E_C], [cover.E2E_B]]
    assert resid[cover._peak_px(cover.E2E_B)] == pytest.approx(-cover._b_peak(), rel=1e-3)
    assert 'MERGE_SATROW_DATA_SUPPORT_MIN' not in capsys.readouterr().out


@pytest.mark.filterwarnings('ignore::jwst_gc_pipeline.frame_wcs.MissingGwcsWarning')
def test_e2e_guard_leaves_no_crater_where_the_frame_shows_nothing(tmp_path, monkeypatch, capsys):
    calls, resid, model = _run(tmp_path, monkeypatch, 0.0, '0.3')
    assert np.abs(resid).max() < 1e-4 * cover._b_peak()
    # B is not drawn into the display model either
    sm = cover._model_with_star(cover.E2E_SHAPE, *cover.E2E_A)
    assert (model[cover._peak_px(cover.E2E_B)] - sm[cover._peak_px(cover.E2E_B)]
            == pytest.approx(0.0, abs=1e-4 * cover._b_peak()))
    out = capsys.readouterr().out
    assert '1 of 1 uncovered saturated rows rendered at the flux this frame\'s data show' in out
    assert 'MERGE_SATROW_DATA_SUPPORT_MIN' in out


@pytest.mark.filterwarnings('ignore::jwst_gc_pipeline.frame_wcs.MissingGwcsWarning')
def test_e2e_guard_leaves_a_star_the_frame_shows_unchanged(tmp_path, monkeypatch, capsys):
    on = _run(tmp_path / 'on', monkeypatch, 1.0, '0.3')
    off = _run(tmp_path / 'off', monkeypatch, 1.0, None)
    np.testing.assert_allclose(on[1], off[1], atol=1e-9)
    assert np.abs(on[1]).max() < 1e-4 * cover._b_peak()
    assert 'rendered at the flux this frame\'s data show' not in capsys.readouterr().out


@pytest.mark.filterwarnings('ignore::jwst_gc_pipeline.frame_wcs.MissingGwcsWarning')
def test_e2e_guard_renders_a_faint_counterpart_at_the_data_flux(tmp_path, monkeypatch):
    calls, resid, _model = _run(tmp_path, monkeypatch, 0.1, '0.3')
    assert np.abs(resid).max() < 1e-3 * cover._b_peak()


@pytest.mark.filterwarnings('ignore::jwst_gc_pipeline.frame_wcs.MissingGwcsWarning')
def test_e2e_guard_is_ignored_for_miri(tmp_path, monkeypatch, capsys):
    _run(tmp_path, monkeypatch, 0.0, '0.3', instrument='MIRI')
    assert 'MERGE_SATROW_DATA_SUPPORT_MIN' not in capsys.readouterr().out
