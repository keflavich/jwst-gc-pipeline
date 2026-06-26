"""Regression test for Fix #2 (unidentified saturated stars): a real saturated
star whose genuine NaN-variance core is engulfed by a LARGE spurious
DQ-SATURATED region (bright extended emission, e.g. the 2526 cloud-c filament's
14717-px component) must NOT be erased by the >edge_npix large-component
suppression.  Before the fix the whole blob was deleted and the star was never
seeded/fit.
"""
import numpy as np
from astropy.io import fits
from jwst.datamodels import dqflags

from jwst_gc_pipeline.reduction.saturated_star_finding import find_saturated_stars

SAT = dqflags.pixel['SATURATED']


def _make_two_blob_frame(shape=(300, 300)):
    """SCI/DQ/VAR_POISSON HDUList with TWO large DQ-SATURATED finite-emission
    blobs (both > edge_npix).  Empirically the large-component suppression
    removes the TOP-LEFT blob (label 1) for this layout, so we put the
    CORE-bearing blob there -- exercising genuine-core preservation on a
    component that is actually subject to suppression."""
    sci = np.full(shape, 300.0, dtype=float)
    dq = np.zeros(shape, dtype=np.int32)
    var = np.ones(shape, dtype=float)
    # blob 1 (WITH genuine core), top-left, 110x110 -> label 1 (suppressed)
    dq[10:120, 10:120] |= SAT; sci[10:120, 10:120] = 800.0
    var[63:68, 63:68] = np.nan; sci[63:68, 63:68] = 0.0          # genuine core
    # blob 2 (no core), bottom-right, 110x110 -> label 2
    dq[170:280, 170:280] |= SAT; sci[170:280, 170:280] = 800.0
    return fits.HDUList([fits.PrimaryHDU(),
                         fits.ImageHDU(sci, name='SCI'),
                         fits.ImageHDU(dq, name='DQ'),
                         fits.ImageHDU(var, name='VAR_POISSON')])


def test_large_component_preserves_genuine_core():
    hdul = _make_two_blob_frame()
    saturated, sources, coms = find_saturated_stars(hdul, edge_npix=10000)
    # blob 1 is suppressed; its genuine NaN-variance core (~65,65) MUST survive
    # while its finite spurious-DQ emission corner IS removed.
    assert saturated[65, 65], "genuine saturated core was erased with the blob"
    assert not saturated[30, 30], "spurious-DQ emission should be suppressed"


def test_small_saturated_star_unaffected():
    # a normal small saturated star (well under edge_npix) is untouched
    sci = np.full((100, 100), 300.0)
    dq = np.zeros((100, 100), dtype=np.int32)
    var = np.ones((100, 100))
    dq[48:53, 48:53] |= SAT
    var[49:52, 49:52] = np.nan
    sci[49:52, 49:52] = 0.0
    hdul = fits.HDUList([fits.PrimaryHDU(),
                         fits.ImageHDU(sci, name='SCI'),
                         fits.ImageHDU(dq, name='DQ'),
                         fits.ImageHDU(var, name='VAR_POISSON')])
    saturated, sources, coms = find_saturated_stars(hdul, edge_npix=10000)
    assert saturated[50, 50], "ordinary small saturated star must be kept"
