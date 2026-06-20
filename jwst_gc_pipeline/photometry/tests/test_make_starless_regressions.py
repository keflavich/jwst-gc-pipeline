"""Regression tests for make_starless_image.py.

Pins intent that lived only in comments / constant tables:

* ``_max_r_for_source`` SNR tiers (make_starless_image.py:273-283).  The
  per-source mask search radius is chosen by a cascade of SNR thresholds;
  below SNR_SKIP a catalog source contributes radius 0 (skipped, only the
  --force-mask-reg path masks it).  Saturated sources always get the
  saturated radius regardless of SNR.

* ``nan_gaussian`` kernel-normalisation infill (make_starless_image.py:253-268).
  Smoothing must interpolate across NaN holes, and pixels too far from any
  real data (normalisation weight < 0.01) must come back NaN rather than 0.

This module has no heavy (jwst/webbpsf/crowdsource) imports, so it is cheap.
"""
import numpy as np
import pytest

from jwst_gc_pipeline.photometry import make_starless_image as M


class TestMaxRForSource:
    def test_saturated_always_saturated_radius(self):
        # Saturated wins even at trivially low SNR.
        assert M._max_r_for_source(flux=1.0, snr=0.0, is_saturated=True) == M.MAX_R_SATURATED

    def test_bright_tier(self):
        assert M._max_r_for_source(1.0, 250.0, False) == M.MAX_R_BRIGHT

    def test_medium_tier(self):
        assert M._max_r_for_source(1.0, 150.0, False) == M.MAX_R_MEDIUM

    def test_faint_tier(self):
        # Strictly above SNR_SKIP -> faint radius.
        assert M._max_r_for_source(1.0, M.SNR_SKIP + 1, False) == M.MAX_R_FAINT

    def test_below_skip_returns_zero(self):
        # At or below SNR_SKIP a catalog source is skipped (radius 0).
        assert M._max_r_for_source(1.0, M.SNR_SKIP, False) == 0
        assert M._max_r_for_source(1.0, 0.0, False) == 0

    def test_thresholds_are_strict_boundaries(self):
        # Boundary values fall to the *lower* tier (uses > not >=).
        assert M._max_r_for_source(1.0, 200.0, False) == M.MAX_R_MEDIUM
        assert M._max_r_for_source(1.0, 100.0, False) == M.MAX_R_FAINT


class TestNanGaussian:
    def test_interpolates_across_nan_hole(self):
        data = np.ones((21, 21))
        data[10, 10] = np.nan  # single hole surrounded by real data
        out = M.nan_gaussian(data, sigma=2.0)
        assert np.isfinite(out[10, 10])
        np.testing.assert_allclose(out[10, 10], 1.0, atol=1e-6)

    def test_all_nan_returns_all_nan(self):
        data = np.full((11, 11), np.nan)
        out = M.nan_gaussian(data, sigma=2.0)
        assert np.all(np.isnan(out))

    def test_far_from_data_returns_nan_not_zero(self):
        # One finite pixel; corners are many sigma away -> weight < 0.01 -> NaN.
        data = np.full((41, 41), np.nan)
        data[20, 20] = 5.0
        out = M.nan_gaussian(data, sigma=1.0)
        assert np.isnan(out[0, 0])
        assert np.isfinite(out[20, 20])
