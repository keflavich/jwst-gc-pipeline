"""Regression tests for the FORBIDDEN dense-nearest-neighbour-median astrometry
guard (``measure_offsets.assert_sparse_reference_for_nn_median``).

Nearest-neighbour / search-around-sky median offset estimation returns a SPURIOUS
shift against a DENSE reference (it collapses toward ~0 when the true offset
exceeds the reference NN spacing).  This silently corrupted brick-1182 astrometry
twice.  The guard must:

  * RAISE ``DenseNNMedianAstrometryError`` on a dense reference (median NN spacing
    below the 3" floor -- e.g. the brick VIRAC2 refcat at ~1.15"), and
  * PASS a sparse reference (e.g. the Gaia-only subset at ~5.7").

These tests pin that behaviour so a refactor cannot silently re-enable the method.
"""
import numpy as np
import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord

from jwst_gc_pipeline.photometry.measure_offsets import (
    assert_sparse_reference_for_nn_median, DenseNNMedianAstrometryError)


def _grid_catalog(n, box_arcsec, seed=0):
    """n random sources uniformly over a box_arcsec square near the GC."""
    rng = np.random.RandomState(seed)
    half = (box_arcsec / 3600.0) / 2.0
    ra = 266.5 + (rng.rand(n) - 0.5) * 2 * half
    dec = -28.7 + (rng.rand(n) - 0.5) * 2 * half
    return SkyCoord(ra, dec, unit='deg')


def test_dense_reference_raises():
    # ~900 sources over 60" -> median NN spacing ~1", like the brick VIRAC2 refcat.
    dense = _grid_catalog(900, box_arcsec=60.0)
    with pytest.raises(DenseNNMedianAstrometryError):
        assert_sparse_reference_for_nn_median(dense, 0.2 * u.arcsec, context="test-dense")


def test_sparse_reference_passes():
    # ~25 sources over 60" -> median NN spacing ~6", like the Gaia-only subset.
    sparse = _grid_catalog(25, box_arcsec=60.0)
    # must NOT raise
    assert_sparse_reference_for_nn_median(sparse, 0.2 * u.arcsec, context="test-sparse")


def test_tiny_catalog_is_noop():
    # Fewer than 3 sources: nothing to guard, must not raise.
    two = SkyCoord([266.5, 266.6], [-28.7, -28.6], unit='deg')
    assert_sparse_reference_for_nn_median(two, 0.2 * u.arcsec, context="test-tiny")


def test_small_match_radius_factor_also_triggers():
    # A moderately sparse reference (NN ~4") still trips the factor*match_radius arm
    # when the match radius is large (2" -> 3x = 6" > 4").
    mod = _grid_catalog(60, box_arcsec=60.0)  # ~4" spacing
    with pytest.raises(DenseNNMedianAstrometryError):
        assert_sparse_reference_for_nn_median(mod, 2.0 * u.arcsec, context="test-bigradius")
