"""Satstar SEEDING vs FIT-MASKING are decoupled.

A pixel flagged DQ-SATURATED but with a good group 0 is recovered by the ramp fit
(valid rate, NO DO_NOT_USE).  Such RECOVERED-saturated stars must still be SEEDED
(daofind cannot fit their saturated cores), so seeding uses the full SATURATED
mask by default.  The truly-lost (SATURATED & DO_NOT_USE) distinction is applied
to the FIT MASK, not seeding.  The stricter truly-lost SEED restriction remains
available opt-in via SATSTAR_SEED_REQUIRE_DO_NOT_USE=1.
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


def test_recovered_saturation_is_seeded_by_default():
    # a RECOVERED-saturated star (SATURATED, no DO_NOT_USE) must still be seeded
    dq = np.zeros((60, 60), np.uint32)
    _blob(dq, 30, 30, 6, SAT)
    _, _, coms, _ = find_saturated_stars(_frame(dq))
    assert len(coms) == 1
    cy, cx = coms[0]
    assert abs(cy - 30) < 2 and abs(cx - 30) < 2


def test_truly_lost_core_seeded_by_default():
    dq = np.zeros((60, 60), np.uint32)
    _blob(dq, 30, 30, 6, SAT)
    _blob(dq, 30, 30, 3, SAT | DNU)
    _, _, coms, _ = find_saturated_stars(_frame(dq))
    assert len(coms) == 1


def test_opt_in_restriction_drops_recovered(monkeypatch):
    # SATSTAR_SEED_REQUIRE_DO_NOT_USE=1: a real truly-lost core elsewhere activates
    # the restriction; the recovered-only blob is then NOT seeded.
    monkeypatch.setenv('SATSTAR_SEED_REQUIRE_DO_NOT_USE', '1')
    dq = np.zeros((80, 80), np.uint32)
    _blob(dq, 15, 15, 3, SAT | DNU)     # truly-lost core
    _blob(dq, 55, 55, 6, SAT)            # recovered-only -> dropped under restriction
    _, _, coms, _ = find_saturated_stars(_frame(dq))
    assert len(coms) == 1
    cy, cx = coms[0]
    assert abs(cy - 15) < 2 and abs(cx - 15) < 2


def test_opt_in_restriction_min_core(monkeypatch):
    # under the restriction, scattered <min-core truly-lost fragments are dropped
    monkeypatch.setenv('SATSTAR_SEED_REQUIRE_DO_NOT_USE', '1')
    dq = np.zeros((60, 60), np.uint32)
    _blob(dq, 30, 30, 8, SAT)                        # recovered blob
    for (y, x) in [(25, 25), (33, 34), (28, 36)]:
        dq[y, x] |= (SAT | DNU)                      # 1px fragments
    _, _, coms, _ = find_saturated_stars(_frame(dq))
    assert len(coms) == 0
