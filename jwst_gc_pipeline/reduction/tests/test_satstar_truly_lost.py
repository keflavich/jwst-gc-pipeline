"""The satstar finder must seed on TRULY-LOST cores, not frame0-recovered pixels.

A pixel flagged DQ-SATURATED but with a good group 0 is recovered by the ramp fit
(valid rate, NO DO_NOT_USE); only 0-good-group pixels carry DO_NOT_USE.  The finder
restricts to SATURATED & DO_NOT_USE (the compact lost core) and drops scattered
sub-core fragments, so bright RECOVERED emission never seeds a phantom satstar.
Falls back to the raw SATURATED mask when no pixel carries DO_NOT_USE (synthetic
frames without per-group DQ), preserving legacy behaviour.
"""
import numpy as np
from astropy.io import fits
from jwst.datamodels import dqflags

from jwst_gc_pipeline.reduction.saturated_star_finding import find_saturated_stars

SAT = dqflags.pixel['SATURATED']
DNU = dqflags.pixel['DO_NOT_USE']


def _frame(dq, sci=None):
    ny, nx = dq.shape
    hdul = fits.HDUList([fits.PrimaryHDU()])
    hdul.append(fits.ImageHDU((np.ones((ny, nx)) * 100 if sci is None else sci).astype(float), name='SCI'))
    hdul.append(fits.ImageHDU(dq.astype(np.uint32), name='DQ'))
    hdul.append(fits.ImageHDU(np.ones((ny, nx)), name='VAR_POISSON'))
    return hdul


def _blob(dq, y, x, r, flag):
    yy, xx = np.mgrid[0:dq.shape[0], 0:dq.shape[1]]
    dq[(np.hypot(yy - y, xx - x) <= r)] |= flag


def test_recovered_saturation_not_seeded(monkeypatch):
    monkeypatch.setenv('SATSTAR_REQUIRE_DO_NOT_USE', '1')
    dq = np.zeros((80, 80), np.uint32)
    # a real truly-lost core elsewhere activates the restriction (real frames
    # always carry some DO_NOT_USE)
    _blob(dq, 15, 15, 3, SAT | DNU)
    # big SATURATED-but-RECOVERED blob (no DO_NOT_USE) -> must NOT seed
    _blob(dq, 55, 55, 8, SAT)
    _, _, coms = find_saturated_stars(_frame(dq))
    assert len(coms) == 1                    # only the truly-lost core
    cy, cx = coms[0]
    assert abs(cy - 15) < 2 and abs(cx - 15) < 2


def test_truly_lost_compact_core_seeded(monkeypatch):
    monkeypatch.setenv('SATSTAR_REQUIRE_DO_NOT_USE', '1')
    dq = np.zeros((60, 60), np.uint32)
    _blob(dq, 30, 30, 8, SAT)            # recovered wings
    _blob(dq, 30, 30, 3, SAT | DNU)     # compact truly-lost core (>=5px)
    _, _, coms = find_saturated_stars(_frame(dq))
    assert len(coms) == 1
    cy, cx = coms[0]
    assert abs(cy - 30) < 2 and abs(cx - 30) < 2


def test_scattered_fragments_not_seeded(monkeypatch):
    monkeypatch.setenv('SATSTAR_REQUIRE_DO_NOT_USE', '1')
    dq = np.zeros((60, 60), np.uint32)
    _blob(dq, 30, 30, 10, SAT)                       # big recovered blob
    for (y, x) in [(25, 25), (33, 34), (28, 36), (35, 27)]:
        dq[y, x] |= (SAT | DNU)                      # isolated 1px truly-lost fragments
    _, _, coms = find_saturated_stars(_frame(dq))
    assert len(coms) == 0                            # min-core gate drops all


def test_fallback_without_do_not_use(monkeypatch):
    # no DO_NOT_USE anywhere -> fallback to SATURATED mask, legacy behaviour
    monkeypatch.setenv('SATSTAR_REQUIRE_DO_NOT_USE', '1')
    dq = np.zeros((60, 60), np.uint32)
    _blob(dq, 30, 30, 5, SAT)
    _, _, coms = find_saturated_stars(_frame(dq))
    assert len(coms) == 1                            # seeded via fallback


def test_disable_via_env(monkeypatch):
    monkeypatch.setenv('SATSTAR_REQUIRE_DO_NOT_USE', '0')
    dq = np.zeros((60, 60), np.uint32)
    _blob(dq, 30, 30, 8, SAT)                        # recovered, but restriction OFF
    _, _, coms = find_saturated_stars(_frame(dq))
    assert len(coms) == 1                            # legacy: seeds on raw SATURATED
