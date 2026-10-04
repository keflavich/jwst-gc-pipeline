"""Regression test for the satstar-finder DATA-value floor.

A DQ-SATURATED pixel sitting on FAINT data is a spurious flag (persistence /
JUMP mis-tag / bad pixel), not a real saturated star.  Without the floor the
finder invents a satstar there and extrapolates a huge flux (W51 F480M: a fake
at flux 104948 fit onto a 127-count DQ-SATURATED pixel).  The floor drops those
components while keeping (a) bright real saturated stars and (b) deep-saturated
cores that read low/NaN but have NaN-variance (unrecoverable) centres.
"""
import os

import numpy as np
import pytest
from astropy.io import fits
from jwst.datamodels import dqflags

from jwst_gc_pipeline.reduction.saturated_star_finding import (
    find_saturated_stars, _resolve_satstar_data_floor)

SAT = dqflags.pixel['SATURATED']


def _frame(sci, dq, var=None):
    hdul = fits.HDUList([fits.PrimaryHDU()])
    hdul.append(fits.ImageHDU(sci.astype(float), name='SCI'))
    hdul.append(fits.ImageHDU(dq.astype(np.uint32), name='DQ'))
    if var is not None:
        hdul.append(fits.ImageHDU(var.astype(float), name='VAR_POISSON'))
    return hdul


def _blob(arr, y, x, val, r=1):
    arr[y - r:y + r + 1, x - r:x + r + 1] = val


def test_floor_drops_spurious_keeps_bright():
    ny = nx = 60
    sci = np.full((ny, nx), 10.0)
    dq = np.zeros((ny, nx), np.uint32)
    var = np.ones((ny, nx))
    # bright REAL saturated star (wings ~5000) -> keep
    _blob(sci, 15, 15, 5000.0); _blob(dq, 15, 15, SAT)
    # spurious faint DQ-SATURATED pixel (data ~127) -> drop
    _blob(sci, 40, 40, 127.0); _blob(dq, 40, 40, SAT)
    hdul = _frame(sci, dq, var)

    # no floor: both components survive
    _, _, coms0, _ = find_saturated_stars(hdul, sat_data_floor=0.0)
    assert len(coms0) == 2

    # floor 1000: spurious (127) dropped, bright (5000) kept
    _, _, coms1, _ = find_saturated_stars(hdul, sat_data_floor=1000.0)
    assert len(coms1) == 1
    cy, cx = coms1[0]
    assert abs(cy - 15) < 2 and abs(cx - 15) < 2  # the bright one survived


def test_floor_keeps_unrecoverable_core():
    ny = nx = 60
    sci = np.full((ny, nx), 10.0)
    dq = np.zeros((ny, nx), np.uint32)
    var = np.ones((ny, nx))
    # deep-saturated core: low/zero data BUT NaN variance (unrecoverable) -> keep
    _blob(sci, 30, 30, 0.0); _blob(dq, 30, 30, SAT)
    var[29:32, 29:32] = np.nan
    hdul = _frame(sci, dq, var)
    _, _, coms, _ = find_saturated_stars(hdul, sat_data_floor=5000.0)
    assert len(coms) == 1  # kept despite faint data, via unrecoverable exemption


def test_resolver_precedence(monkeypatch):
    # explicit > env > per-filter default > 0
    assert _resolve_satstar_data_floor('f480m', explicit=1234.0) == 1234.0
    monkeypatch.setenv('SATSTAR_DATA_FLOOR', '777')
    assert _resolve_satstar_data_floor('f480m') == 777.0
    monkeypatch.delenv('SATSTAR_DATA_FLOOR')
    assert _resolve_satstar_data_floor('f480m') == 1000.0   # per-filter default
    assert _resolve_satstar_data_floor('f999z') == 0.0      # unlisted -> off


# --------------------------------------------------------------------------
# readout bound (#952): FRAC x S * PHOTMJSR / t_last lowers the table floor
# --------------------------------------------------------------------------

from jwst_gc_pipeline.reduction import saturated_star_finding as SSF  # noqa: E402

# wd2 F410M: SHALLOW4, 7 groups, t_last = (6 x 5 + 2.5) x 10.73677 = 348.9 s
WD2_F410M = {'INSTRUME': 'NIRCAM', 'FILTER': 'F410M', 'NGROUPS': 7,
             'NFRAMES': 4, 'GROUPGAP': 1, 'TFRAME': 10.73677}
# W51 F480M: SHALLOW2, 5 groups, t_last = (4 x 5 + 1.5) x 10.73677 = 230.8 s
W51_F480M = {'INSTRUME': 'NIRCAM', 'FILTER': 'F480M', 'NGROUPS': 5,
             'NFRAMES': 2, 'GROUPGAP': 3, 'TFRAME': 10.73677}


def test_readout_floor_values(monkeypatch):
    monkeypatch.delenv('SATSTAR_DATA_FLOOR_READOUT_FRAC', raising=False)
    wd2 = SSF.satstar_readout_data_floor(fits.Header(WD2_F410M), 0.86)
    w51 = SSF.satstar_readout_data_floor(fits.Header(W51_F480M), 1.969)
    assert wd2 == pytest.approx(0.5 * 50000 * 0.86 / 348.94, rel=1e-3)   # ~62
    assert w51 == pytest.approx(0.5 * 50000 * 1.969 / 230.84, rel=1e-3)  # ~213
    # both sit between the spurious flags (wing max 10-80 on wd2 and W51) and
    # the W51 fake of the original floor (127 MJy/sr) on the side it targets
    assert 127.0 < w51 < 1000.0
    assert wd2 < 139.0   # lowest wing max of a star-like wd2 component


def test_readout_floor_off_cases(monkeypatch):
    monkeypatch.delenv('SATSTAR_DATA_FLOOR_READOUT_FRAC', raising=False)
    miri = dict(WD2_F410M, INSTRUME='MIRI')
    assert SSF.satstar_readout_data_floor(fits.Header(miri), 0.86) is None
    missing = {k: v for k, v in WD2_F410M.items() if k != 'TFRAME'}
    assert SSF.satstar_readout_data_floor(fits.Header(missing), 0.86) is None
    assert SSF.satstar_readout_data_floor(fits.Header(WD2_F410M), None) is None
    assert SSF.satstar_readout_data_floor(fits.Header(WD2_F410M), 0.0) is None
    monkeypatch.setenv('SATSTAR_DATA_FLOOR_READOUT_FRAC', '0')
    assert SSF.satstar_readout_data_floor(fits.Header(WD2_F410M), 0.86) is None
    monkeypatch.setenv('SATSTAR_DATA_FLOOR_READOUT_FRAC', '0.25')
    assert SSF.satstar_readout_data_floor(fits.Header(WD2_F410M), 0.86) == \
        pytest.approx(0.25 * 50000 * 0.86 / 348.94, rel=1e-3)


def test_resolver_readout_only_lowers_a_table_entry(monkeypatch):
    monkeypatch.delenv('SATSTAR_DATA_FLOOR', raising=False)
    assert _resolve_satstar_data_floor('f410m', readout_floor=62.0) == 62.0
    # a readout bound above the table leaves the table value
    assert _resolve_satstar_data_floor('f480m', readout_floor=2400.0) == 1000.0
    # unlisted filters stay off
    assert _resolve_satstar_data_floor('f150w', readout_floor=62.0) == 0.0
    assert _resolve_satstar_data_floor('f410m', readout_floor=None) == 800.0
    # explicit and env still win
    assert _resolve_satstar_data_floor('f410m', explicit=500.0,
                                       readout_floor=62.0) == 500.0
    monkeypatch.setenv('SATSTAR_DATA_FLOOR', '800')
    assert _resolve_satstar_data_floor('f410m', readout_floor=62.0) == 800.0


def test_readout_floor_keeps_a_last_group_star():
    """wd2 F410M: a 15-16 mag star saturates only in the last groups, so its
    wings peak at ~140-290 MJy/sr, above S*P/t_last = 130 and below the 800
    table floor.  A spurious flag on 20 MJy/sr sky goes either way."""
    ny = nx = 60
    sci = np.full((ny, nx), 20.0)
    dq = np.zeros((ny, nx), np.uint32)
    var = np.ones((ny, nx))
    _blob(sci, 15, 15, 160.0); _blob(dq, 15, 15, SAT)   # last-group star
    _blob(dq, 40, 40, SAT)                               # flag on sky
    hdul = _frame(sci, dq, var)
    floor = _resolve_satstar_data_floor(
        'f410m', readout_floor=SSF.satstar_readout_data_floor(
            fits.Header(WD2_F410M), 0.86, frac=0.5))
    _, _, coms, _ = find_saturated_stars(hdul, sat_data_floor=floor)
    assert len(coms) == 1
    cy, cx = coms[0]
    assert abs(cy - 15) < 2 and abs(cx - 15) < 2
    _, _, coms800, _ = find_saturated_stars(hdul, sat_data_floor=800.0)
    assert len(coms800) == 0


def _crf(tmp_path, primary, photmjsr=0.86):
    fn = tmp_path / 'jw03523005001_16101_00001_nrcblong_align_o005_crf.fits'
    sci = fits.ImageHDU(np.zeros((8, 8), 'float32'), name='SCI')
    sci.header['PHOTMJSR'] = photmjsr
    fits.HDUList([fits.PrimaryHDU(header=fits.Header(primary)), sci]
                 ).writeto(fn, overwrite=True)
    return str(fn)


def test_signature_marks_a_readout_floor(tmp_path, monkeypatch):
    """A frame whose floor now comes from the readout is refit once."""
    for name in [n for n in os.environ if 'SATSTAR' in n]:
        monkeypatch.delenv(name, raising=False)
    fn = _crf(tmp_path, WD2_F410M)
    assert SSF.satstar_fit_switch_signature(fn) == 'dfr0.5'
    monkeypatch.setenv('SATSTAR_ERR_BKG_SCATTER', '1')
    assert SSF.satstar_fit_switch_signature(fn) == 'es_dfr0.5'
    monkeypatch.delenv('SATSTAR_ERR_BKG_SCATTER')
    monkeypatch.setenv('SATSTAR_DATA_FLOOR_READOUT_FRAC', '0.25')
    assert SSF.satstar_fit_switch_signature(fn) == 'dfr0.25'
    monkeypatch.setenv('SATSTAR_DATA_FLOOR_READOUT_FRAC', '0')
    assert SSF.satstar_fit_switch_signature(fn) == ''
    monkeypatch.delenv('SATSTAR_DATA_FLOOR_READOUT_FRAC')
    monkeypatch.setenv('SATSTAR_DATA_FLOOR', '800')
    assert SSF.satstar_fit_switch_signature(fn) == ''
    monkeypatch.delenv('SATSTAR_DATA_FLOOR')
    # unlisted filter: no floor either way
    assert SSF.satstar_fit_switch_signature(
        _crf(tmp_path, dict(WD2_F410M, FILTER='F150W'))) == ''
