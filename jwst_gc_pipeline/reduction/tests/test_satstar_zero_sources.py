"""find_saturated_stars must not crash when a frame has NO saturated pixels.

Regression: a frame/cutout with zero saturated components reaches
``sizes = sum_labels(saturated, sources, np.arange(nsource)+1)`` with an empty
index, and scipy's reduction does ``np.amin`` over nothing ->
``ValueError: zero-size array to reduction operation minimum which has no
identity``.  Full-frame GC fields always have saturated stars so this never
fired; small cutouts / sparse fields do.  ``find_saturated_stars`` now
short-circuits to an empty result for nsource == 0.
"""
import numpy as np
import pytest

from astropy.io import fits

from jwst_gc_pipeline.reduction.saturated_star_finding import find_saturated_stars


def _hdul(dq):
    return fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(dq, name='DQ')])


def test_no_saturated_pixels_returns_empty_no_crash():
    dq = np.zeros((64, 64), dtype=np.uint32)   # nothing flagged SATURATED
    saturated, sources, coms, _kinds = find_saturated_stars(_hdul(dq))
    assert saturated.sum() == 0
    assert int(sources.max()) == 0
    assert list(coms) == []


def test_with_a_saturated_blob_still_labels_it():
    from jwst.datamodels import dqflags
    dq = np.zeros((64, 64), dtype=np.uint32)
    dq[30:35, 30:35] = dqflags.pixel['SATURATED']
    saturated, sources, coms, _kinds = find_saturated_stars(_hdul(dq))
    assert int(sources.max()) == 1
    assert len(coms) == 1
