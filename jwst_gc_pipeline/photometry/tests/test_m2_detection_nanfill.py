"""m2 detects on a NaN-filled residual, like m1 and m3+ do.

``do_photometry_step_manual`` hands ``_build_manual_seed`` a detection image in
every manual phase.  m1 and m3-m6 pass ``ctx.nan_replaced_data`` (NaN pixels
already filled).  m2 passes the m1 residual, built from the pristine data, so
it carries the frame's NaN pixels.  DAOStarFinder (photutils 3.0.1.dev27)
returns no peak with a NaN pixel within ~4 px, so m2 found nothing beside the
NaN pixels of the frame: masked cores, the reference border, bad pixels.

These tests pin:

* the m2 detection image has no NaN, equals the residual away from the NaN
  pixels, and is filled at them with the kernel-weighted mean of the finite
  neighbours (``replace_nans`` with the frame's Gaussian kernel);
* the m1 detection image is still ``ctx.nan_replaced_data`` itself;
* the filled residual gives ``_build_manual_seed`` a seed for a star 3 px from
  a NaN pixel (skipped if the installed photutils already finds it unfilled,
  because then the fill is not what makes the star detectable).
"""
import types

import numpy as np
import pytest
from astropy.convolution import Gaussian2DKernel
from astropy.table import Table
from astropy.wcs import WCS

from jwst_gc_pipeline.photometry import cataloging as C
from jwst_gc_pipeline.photometry.nan_fill import replace_nans

FWHM = 2.5
SHAPE = (100, 100)
HOLE = (slice(40, 44), slice(50, 54))     # 4 x 4 NaN block, rows then columns


def _kernel():
    return Gaussian2DKernel(x_stddev=FWHM / 2.355)


def _modsky():
    """The m1 model sky: a ramp, so the residual differs from the frame everywhere."""
    return 0.25 + 0.01 * np.arange(SHAPE[1])[None, :] * np.ones(SHAPE)


def _satstar_model():
    """The saturated-star model subtracted before m1, finite everywhere."""
    return np.full(SHAPE, 0.1)


def _ctx(original):
    """The slice of the frame context the m12 branch reads."""
    return types.SimpleNamespace(
        im1=None, fwhm_pix=FWHM, satstar_rejected_path=None, handoff_xy=None,
        handoff_radius=0.0, satstar_table=None,
        nan_replaced_data=np.full(SHAPE, 0.5), mask=np.zeros(SHAPE, bool),
        err=np.ones(SHAPE), bad=np.zeros(SHAPE, bool), dao_psf_model=None,
        aperture_radius_pix=5.0, localbkg_inner=6, localbkg_outer=10,
        grouper=None, dqarr=None, satstar_model_subtracted=_satstar_model(),
        data=np.zeros(SHAPE), original_data=original, ww=None)


def _run_m12(monkeypatch, tmp_path, original):
    ctx = _ctx(original)
    seeds = []
    monkeypatch.setattr(C, '_prepare_frame_for_photometry', lambda *a, **k: ctx)

    def _fake_seed(**kw):
        seeds.append(kw)
        return Table()
    monkeypatch.setattr(C, '_build_manual_seed', _fake_seed)
    monkeypatch.setattr(C, '_manual_phot_pass',
                        lambda **kw: (Table(), _modsky(), None))
    monkeypatch.setattr(C, '_save_manual_pass', lambda **k: None)
    C.do_photometry_step_manual(
        types.SimpleNamespace(target='gc-treasury'), 'F212N', 'nrcb1', 'nrcb1',
        '078', str(tmp_path), 'f.fits', '10678', manual_phase='m12')
    return ctx, {kw['label']: kw for kw in seeds}


@pytest.fixture
def frame():
    rng = np.random.default_rng(3)
    original = rng.normal(0.0, 1.0, SHAPE)
    original[HOLE] = np.nan
    return original


def test_m2_detection_image_has_no_nan(monkeypatch, tmp_path, frame):
    assert np.isnan(frame).sum() == 16
    _, seeds = _run_m12(monkeypatch, tmp_path, frame)
    assert set(seeds) == {'m1', 'm2'}
    assert not np.isnan(seeds['m2']['detection_image']).any()


def test_m2_detection_image_is_residual_with_kernel_fill(monkeypatch, tmp_path,
                                                         frame):
    _, seeds = _run_m12(monkeypatch, tmp_path, frame)
    det = seeds['m2']['detection_image']
    finite = np.isfinite(frame)
    # the m1 residual: pristine frame less the satstar model less the m1 model sky
    residual = frame - _satstar_model() - _modsky()
    np.testing.assert_allclose(det[finite], residual[finite])
    expected = replace_nans(residual, _kernel())
    np.testing.assert_allclose(det[HOLE], expected[HOLE])
    # filled from the finite neighbours, not zeros or a constant
    assert np.ptp(det[HOLE]) > 0.0
    assert np.all(np.abs(det[HOLE] - residual[finite].mean()) < 3.0)


def test_m2_fill_leaves_the_input_residual_untouched(monkeypatch, tmp_path,
                                                     frame):
    before = frame.copy()
    ctx, _ = _run_m12(monkeypatch, tmp_path, frame)
    np.testing.assert_array_equal(ctx.original_data, before)   # NaNs still there


def test_m1_detection_image_is_still_nan_replaced_data(monkeypatch, tmp_path,
                                                       frame):
    ctx, seeds = _run_m12(monkeypatch, tmp_path, frame)
    assert seeds['m1']['detection_image'] is ctx.nan_replaced_data


# ---------------------------------------------------------------------------
# Seed level: the filled residual makes the star beside the NaN detectable
# ---------------------------------------------------------------------------

def _wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = [266.4, -28.9]
    w.wcs.crpix = [50.0, 50.0]
    w.wcs.cdelt = [-0.03 / 3600.0, 0.03 / 3600.0]
    return w


def _seed(detection_image, ww):
    return C._build_manual_seed(
        detection_image=detection_image,
        nan_replaced_data=np.nan_to_num(detection_image), mask=np.zeros(SHAPE, bool),
        ww=ww, fwhm_pix=FWHM, satstar_table=None, prev_catalog=None,
        local_snr_threshold=3.0, roundlo=-1.0, roundhi=1.0, sharplo=0.5,
        sharphi=1.0, label='m2')


def _star_beside_nan():
    yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1]]
    sig = FWHM / 2.355
    star = 60.0 * np.exp(-((xx - 30.0) ** 2 + (yy - 60.0) ** 2) / (2 * sig ** 2))
    img = np.random.default_rng(5).normal(0.0, 1.0, SHAPE) + star
    img[60, 33] = np.nan                     # one NaN pixel, 3 px from the peak
    return img


def _near(seed, x=30.0, y=60.0, radius=1.5):
    if not len(seed):
        return False
    return bool(np.any(np.hypot(np.asarray(seed['x_init']) - x,
                                np.asarray(seed['y_init']) - y) < radius))


def test_filled_residual_seeds_a_star_beside_a_nan_pixel():
    ww = _wcs()
    img = _star_beside_nan()
    if _near(_seed(img, ww)):
        pytest.skip('installed photutils finds the star beside a NaN pixel '
                    'without the fill')
    assert _near(_seed(replace_nans(img, _kernel()), ww))
