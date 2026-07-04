"""Regression test for the satstar-finder DATA-value floor.

A DQ-SATURATED pixel sitting on FAINT data is a spurious flag (persistence /
JUMP mis-tag / bad pixel), not a real saturated star.  Without the floor the
finder invents a satstar there and extrapolates a huge flux (W51 F480M: a fake
at flux 104948 fit onto a 127-count DQ-SATURATED pixel).  The floor drops those
components while keeping (a) bright real saturated stars and (b) deep-saturated
cores that read low/NaN but have NaN-variance (unrecoverable) centres.
"""
import numpy as np
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
    _, _, coms0 = find_saturated_stars(hdul, sat_data_floor=0.0)
    assert len(coms0) == 2

    # floor 1000: spurious (127) dropped, bright (5000) kept
    _, _, coms1 = find_saturated_stars(hdul, sat_data_floor=1000.0)
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
    _, _, coms = find_saturated_stars(hdul, sat_data_floor=5000.0)
    assert len(coms) == 1  # kept despite faint data, via unrecoverable exemption


def test_resolver_precedence(monkeypatch):
    # explicit > env > per-filter default > 0
    assert _resolve_satstar_data_floor('f480m', explicit=1234.0) == 1234.0
    monkeypatch.setenv('SATSTAR_DATA_FLOOR', '777')
    assert _resolve_satstar_data_floor('f480m') == 777.0
    monkeypatch.delenv('SATSTAR_DATA_FLOOR')
    assert _resolve_satstar_data_floor('f480m') == 1000.0   # per-filter default
    assert _resolve_satstar_data_floor('f999z') == 0.0      # unlisted -> off
