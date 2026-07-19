"""Regression tests for the FWHM-scaled mergedcat render stamp.

The hardcoded (21,21) render stamp (+-10px) is NIRCam-calibrated; for the broad
MIRI long-wavelength PSFs it clips the diffraction wings (F2550W FWHM 7.3px ->
only ~1.4 FWHM covered), leaving bright stars under-subtracted beyond r~10px.
``_resolve_render_psf_shape`` grows the stamp to +-3*FWHM for broad PSFs while
leaving NIRCam / short-MIRI at the 21px default, capped by the PSF grid stamp.
"""
import os
import numpy as np
import pytest

from jwst_gc_pipeline.photometry.catalog_long import (
    _resolve_render_psf_shape)


class _Grid:
    """Minimal stand-in for a GriddedPSFModel (data + oversampling)."""
    def __init__(self, oversampled_stamp=404, oversampling=4):
        self.oversampling = np.array([oversampling, oversampling])
        self.data = np.zeros((16, oversampled_stamp, oversampled_stamp),
                             dtype=np.float32)


@pytest.fixture(autouse=True)
def _clear_env():
    for k in ('MERGE_RENDER_PSF_SHAPE', 'MERGE_RENDER_FWHM_MULT'):
        os.environ.pop(k, None)
    yield
    for k in ('MERGE_RENDER_PSF_SHAPE', 'MERGE_RENDER_FWHM_MULT'):
        os.environ.pop(k, None)


def test_nircam_narrow_psf_stays_at_default():
    """NIRCam FWHM ~2.2px -> 2*ceil(6.5)+1=15 < 21 -> keep the 21px default."""
    g = _Grid()
    assert _resolve_render_psf_shape(2.18, g, default=(21, 21)) == (21, 21)


def test_f770w_stays_at_default():
    """F770W FWHM 2.45px -> 15 < 21 -> unchanged (F770W was already fine)."""
    g = _Grid()
    assert _resolve_render_psf_shape(2.445, g, default=(21, 21)) == (21, 21)


def test_broad_miri_grows():
    """F2550W FWHM 7.3px, mult 3 -> 2*ceil(21.9)+1 = 45 (grid stamp 101 allows)."""
    g = _Grid(oversampled_stamp=404, oversampling=4)   # 101px stamp
    assert _resolve_render_psf_shape(7.3, g, default=(21, 21)) == (45, 45)


def test_f2100w_grows_intermediate():
    """F2100W FWHM 6.13px -> 2*ceil(18.38)+1 = 39."""
    g = _Grid()
    assert _resolve_render_psf_shape(6.127, g, default=(21, 21)) == (39, 39)


def test_capped_by_grid_stamp():
    """A tiny PSF grid (21px stamp) caps a broad-FWHM request back to 21."""
    g = _Grid(oversampled_stamp=84, oversampling=4)    # 21px stamp
    assert _resolve_render_psf_shape(7.3, g, default=(21, 21)) == (21, 21)


def test_result_is_odd():
    """Rendered stamp must be odd (centred on the source)."""
    g = _Grid()
    for fwhm in (3.5, 4.0, 5.5, 6.13, 7.3):
        n = _resolve_render_psf_shape(fwhm, g, default=(21, 21))[0]
        assert n % 2 == 1


def test_env_absolute_override():
    """MERGE_RENDER_PSF_SHAPE forces an absolute odd size, ignoring FWHM."""
    g = _Grid()
    os.environ['MERGE_RENDER_PSF_SHAPE'] = '31'
    assert _resolve_render_psf_shape(7.3, g, default=(21, 21)) == (31, 31)
    os.environ['MERGE_RENDER_PSF_SHAPE'] = '30'       # even -> bumped odd
    assert _resolve_render_psf_shape(7.3, g, default=(21, 21)) == (31, 31)


def test_env_mult_override():
    """MERGE_RENDER_FWHM_MULT scales the growth factor."""
    g = _Grid()
    os.environ['MERGE_RENDER_FWHM_MULT'] = '2.0'
    # F2550W 7.3px, mult 2 -> 2*ceil(14.6)+1 = 31
    assert _resolve_render_psf_shape(7.3, g, default=(21, 21)) == (31, 31)


def test_none_fwhm_returns_default():
    g = _Grid()
    assert _resolve_render_psf_shape(None, g, default=(21, 21)) == (21, 21)
    assert _resolve_render_psf_shape(0.0, g, default=(21, 21)) == (21, 21)


# ---- satstar render cap (fix #2: peaked re-render eats the MIRI pedestal) ----
from jwst_gc_pipeline.photometry.catalog_long import (
    _cap_render_to_pedestal)


def _star_on_pedestal(n=61, peak=2300.0, fwhm=7.0, pedestal=1000.0):
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(xx - n // 2, yy - n // 2)
    sigma = fwhm / 2.3548
    star = peak * np.exp(-0.5 * (r / sigma) ** 2)
    return star + pedestal, r


def test_cap_removes_pedestal_hole():
    """An over-flux peaked model (peak > true star) would drive the residual
    below the pedestal; the cap holds the core residual at the pedestal."""
    base, r = _star_on_pedestal(peak=2300.0, pedestal=1000.0)
    # over-predicting model: 25% too bright (mimics A: 2879 vs 2296)
    over = 1.25 * (base - 1000.0)                    # peaked, no pedestal
    resid_uncapped = base - over
    core = r < 2.5
    assert np.median(resid_uncapped[core]) < 800.0   # a hole below ~1000 bg
    capped = _cap_render_to_pedestal(over, base, bgbox=43)
    resid = base - capped
    # core residual restored to ~pedestal (no hole)
    assert abs(np.median(resid[core]) - 1000.0) < 60.0


def test_cap_keeps_faint_wings_subtracted():
    """Where the model is BELOW the above-pedestal data (wings), it is kept."""
    base, r = _star_on_pedestal(peak=2300.0, pedestal=1000.0)
    under = 0.6 * (base - 1000.0)                    # everywhere below data-bg
    capped = _cap_render_to_pedestal(under, base, bgbox=43)
    # nothing capped: model unchanged where model < data-bg
    assert np.allclose(capped, under, atol=1e-6)


def test_cap_never_increases_model():
    base, r = _star_on_pedestal()
    model = 2.0 * (base - 1000.0)
    capped = _cap_render_to_pedestal(model, base, bgbox=43)
    assert np.all(capped <= model + 1e-6)


def test_cap_leaves_nan_base_uncapped():
    base, r = _star_on_pedestal()
    base[r < 1.5] = np.nan                            # masked core
    model = np.full_like(base, 5000.0)
    capped = _cap_render_to_pedestal(model, base, bgbox=43)
    assert np.all(capped[r < 1.5] == 5000.0)          # uncapped where base NaN


def test_cap_core_mask_fills_undersub_shoulder():
    """A peaked model that UNDER-predicts a broad core (cloudc F2550W B: r3-9
    shoulder +110) is filled to the data inside core_mask -> residual = bg."""
    base, r = _star_on_pedestal(peak=2300.0, fwhm=8.0, pedestal=1000.0)
    # peaked model too NARROW: matches peak, under-predicts the shoulder
    narrow = 2300.0 * np.exp(-0.5 * (r / (4.0 / 2.3548)) ** 2)  # fwhm 4 vs data 8
    shoulder = (r >= 3) & (r <= 7)
    resid_no_mask = base - _cap_render_to_pedestal(narrow, base, bgbox=43)
    assert np.median(resid_no_mask[shoulder]) > 1100.0     # under-sub ring survives
    cm = r <= 9.0
    resid = base - _cap_render_to_pedestal(narrow, base, bgbox=43, core_mask=cm)
    # shoulder filled to the data -> residual ~ pedestal
    assert abs(np.median(resid[shoulder]) - 1000.0) < 60.0
