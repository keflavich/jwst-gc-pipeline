"""Tests for incremental refit (reuse unchanged per-source fits across phases).

Two layers:
1. the pure classifier (dirty mask / seed match / reusability), and
2. the CORE INVARIANT it relies on -- an independent (group=False) single-source
   PSF fit is unchanged when the background changes only OUTSIDE the source's
   footprint. If this invariant held falsely, reuse would corrupt fluxes; the
   test fits a real photutils PSFPhotometry both ways and asserts bit-equality.
"""
import numpy as np
import pytest

from jwst_gc_pipeline.photometry.incremental_refit import (
    dirty_bg_mask, match_prev_seeds, classify_reusable_seeds, splice_reused_rows)


# --------------------------------------------------------------------------
# pure classifier
# --------------------------------------------------------------------------

def test_dirty_mask_thresholds_and_dilates():
    bg_prev = np.zeros((40, 40))
    bg_cur = np.zeros((40, 40))
    bg_cur[20, 20] = 10.0                      # one changed pixel
    d = dirty_bg_mask(bg_cur, bg_prev, bg_delta_thresh=1.0, dilate_pix=3)
    assert d[20, 20]
    assert d[20, 23] and d[23, 20]             # dilated by ~3
    assert not d[20, 25]                        # beyond dilation
    assert not d[0, 0]


def test_dirty_mask_nonfinite_is_dirty():
    bg_prev = np.zeros((10, 10))
    bg_cur = np.zeros((10, 10))
    bg_cur[5, 5] = np.nan
    d = dirty_bg_mask(bg_cur, bg_prev, bg_delta_thresh=1e9, dilate_pix=0)
    assert d[5, 5]


def test_seed_match_tolerance_is_tight():
    prev = np.array([[10.0, 10.0], [30.0, 30.0]])
    seeds = np.array([[10.001, 10.0],    # carried forward -> match
                      [30.2, 30.0],      # moved 0.2 px -> NO match (refit)
                      [50.0, 50.0]])     # new -> no match
    idx = match_prev_seeds(seeds, prev, match_tol_pix=0.01)
    assert idx[0] == 0
    assert idx[1] == -1
    assert idx[2] == -1


def test_classify_clean_source_reusable_dirty_source_refit():
    bg_prev = np.zeros((80, 80))
    bg_cur = np.zeros((80, 80))
    bg_cur[60, 60] = 5.0                         # bg change near the second source
    prev_xy = np.array([[20.0, 20.0], [60.0, 60.0]])
    seed_xy = np.array([[20.0, 20.0], [60.0, 60.0]])
    reusable, prev_index = classify_reusable_seeds(
        seed_xy, prev_xy, bg_cur, bg_prev,
        footprint_radius_pix=6.0, bg_delta_thresh=1.0, match_tol_pix=0.01)
    assert reusable[0] and prev_index[0] == 0     # far from change -> reuse
    assert not reusable[1]                          # bg changed in its footprint


def test_classify_new_seed_never_reusable():
    bg = np.zeros((40, 40))
    prev_xy = np.array([[10.0, 10.0]])
    seed_xy = np.array([[10.0, 10.0], [25.0, 25.0]])  # second is new
    reusable, idx = classify_reusable_seeds(
        seed_xy, prev_xy, bg, bg.copy(),
        footprint_radius_pix=5.0, bg_delta_thresh=1.0)
    assert reusable[0]
    assert not reusable[1] and idx[1] == -1


def test_splice_preserves_seed_order():
    from astropy.table import Table
    prev = Table({"x_fit": [1.0, 2.0, 3.0], "flux_fit": [10.0, 20.0, 30.0]})
    # seeds: [reuse prev0, refit, reuse prev2]
    reusable = np.array([True, False, True])
    prev_index = np.array([0, -1, 2])
    fit = Table({"x_fit": [99.0], "flux_fit": [999.0]})   # the one refit row
    out = splice_reused_rows(prev, fit, reusable, prev_index,
                             seed_order=np.arange(3))
    assert list(out["flux_fit"]) == [10.0, 999.0, 30.0]
    assert list(out["x_fit"]) == [1.0, 99.0, 3.0]


# --------------------------------------------------------------------------
# CORE INVARIANT: independent fit is unaffected by a far bg change
# --------------------------------------------------------------------------

def _psfphot():
    from photutils.psf import PSFPhotometry, CircularGaussianPRF
    from photutils.background import LocalBackground
    psf = CircularGaussianPRF(flux=1, fwhm=2.5)
    return PSFPhotometry(psf_model=psf, fit_shape=(5, 5), aperture_radius=5,
                         localbkg_estimator=LocalBackground(6, 10),
                         progress_bar=False), psf


def _render(psf, x, y, flux, shape):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    psf = psf.copy()
    psf.x_0, psf.y_0, psf.flux = x, y, flux
    return psf(xx, yy)


def test_far_bg_change_leaves_independent_fit_bit_identical():
    """The invariant reuse depends on: with no grouper, source A's fit is a
    function only of the data within its footprint. Change the background far from
    A (beyond localbkg_outer=10) and A's fit must not move."""
    rng = np.random.default_rng(0)
    phot, psf = _psfphot()
    shape = (80, 80)
    # source A at (20,20); an unrelated bright region far away at (60,60)
    base = (_render(psf, 20.0, 20.0, 500.0, shape)
            + _render(psf, 60.0, 60.0, 800.0, shape))
    noise = rng.normal(0, 0.5, shape)
    from astropy.table import Table
    init = Table({"x": [20.0], "y": [20.0], "flux": [500.0]})

    # phase N-1 data
    data_prev = base + noise
    rA_prev = phot(data_prev, init_params=init)

    # phase N: subtract a background that differs ONLY far from A (a blob at 60,60)
    bg_delta = _render(psf, 60.0, 60.0, 300.0, shape)   # all >~30 px from A
    data_cur = base + noise - bg_delta
    rA_cur = phot(data_cur, init_params=init)

    # A's fit must be identical to floating-point round-off
    assert rA_cur["flux_fit"][0] == pytest.approx(rA_prev["flux_fit"][0], rel=1e-9, abs=1e-6)
    assert rA_cur["x_fit"][0] == pytest.approx(rA_prev["x_fit"][0], abs=1e-9)
    assert rA_cur["y_fit"][0] == pytest.approx(rA_prev["y_fit"][0], abs=1e-9)


def test_near_bg_change_does_move_the_fit():
    """Control: a NON-uniform bg change inside A's fit window DOES change the fit
    (so reuse there would be wrong -- the classifier must refit it). A uniform
    pedestal would be absorbed by the LocalBackground annulus, so the change must
    be localized on the core, which the annulus cannot compensate."""
    rng = np.random.default_rng(1)
    phot, psf = _psfphot()
    shape = (60, 60)
    base = _render(psf, 30.0, 30.0, 500.0, shape)
    noise = rng.normal(0, 0.5, shape)
    from astropy.table import Table
    init = Table({"x": [30.0], "y": [30.0], "flux": [500.0]})
    r_prev = phot(base + noise, init_params=init)
    # a localized bump right on A's core (inside the 5x5 fit window) that the
    # annulus-based localbkg cannot absorb
    bump = _render(psf, 30.0, 30.0, 100.0, shape)
    r_cur = phot(base + noise + bump, init_params=init)
    assert abs(r_cur["flux_fit"][0] - r_prev["flux_fit"][0]) > 1.0
