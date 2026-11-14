"""``forced_psf_photometry(bkg_sigma=...)`` and ``annulus_scatter_sigma``.

The closed-form fit has an analytic error, ``1/sqrt(sum(psf**2 / err**2))``, so
the local-background term (#1120) can be checked against it exactly.
"""
import numpy as np
import pytest
from astropy.table import Table

from jwst_gc_pipeline.photometry.psf_fitting import (
    annulus_scatter_sigma, forced_psf_photometry)

SIG = 1.5


class _MockPSF:
    def evaluate(self, x, y, flux, x0, y0):
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * SIG ** 2))
        return flux * g / (2 * np.pi * SIG ** 2)


def _scene(n_src=3, err=1.0, noise=0.0, seed=3):
    rng = np.random.default_rng(seed)
    ny = nx = 60
    yy, xx = np.mgrid[0:ny, 0:nx]
    xs = np.array([15.0, 30.3, 45.6])[:n_src]
    ys = np.array([20.0, 31.2, 40.0])[:n_src]
    image = rng.normal(0.0, noise, (ny, nx)) if noise else np.zeros((ny, nx))
    for x, y in zip(xs, ys):
        image = image + _MockPSF().evaluate(xx, yy, 400.0, x, y)
    seed_tbl = Table({'x_init': xs, 'y_init': ys})
    return image, np.full((ny, nx), err), seed_tbl, xs, ys


def _analytic_err(x, y, err_stamp):
    ix, iy = int(round(x)), int(round(y))
    dy, dx = np.mgrid[-2:3, -2:3]
    psf = _MockPSF().evaluate(ix + dx, iy + dy, 1.0, x, y)
    return 1.0 / np.sqrt(np.sum(psf ** 2 / err_stamp ** 2))


def test_bkg_sigma_added_in_quadrature():
    image, err, seed, xs, ys = _scene()
    sigma = np.array([0.0, 3.0, 8.0])
    res = forced_psf_photometry(image, _MockPSF(), seed, error=err,
                                bkg_sigma=sigma)
    for i in range(3):
        expect = _analytic_err(xs[i], ys[i],
                               np.full((5, 5), np.hypot(1.0, sigma[i])))
        assert res['flux_err'][i] == pytest.approx(expect, rel=1e-12)
    # a noiseless image recovers the flux whatever the weights
    np.testing.assert_allclose(res['flux_fit'], 400.0, rtol=1e-9)


def test_none_nan_and_nonpositive_sigma_leave_the_fit_unchanged():
    image, err, seed, *_ = _scene(noise=0.5)
    base = forced_psf_photometry(image, _MockPSF(), seed, error=err)
    none = forced_psf_photometry(image, _MockPSF(), seed, error=err,
                                 bkg_sigma=None)
    off = forced_psf_photometry(image, _MockPSF(), seed, error=err,
                                bkg_sigma=np.array([np.nan, 0.0, -1.0]))
    for other in (none, off):
        np.testing.assert_array_equal(other['flux_fit'], base['flux_fit'])
        np.testing.assert_array_equal(other['flux_err'], base['flux_err'])


def test_invalid_error_pixels_stay_excluded():
    image, err, seed, xs, ys = _scene(n_src=1)
    bad = err.copy()
    bad[int(round(ys[0])), int(round(xs[0]))] = np.nan
    plain = forced_psf_photometry(image, _MockPSF(), seed, error=bad)
    inflated = forced_psf_photometry(image, _MockPSF(), seed, error=bad,
                                     bkg_sigma=np.array([2.0]))
    assert plain['n_pixels_fit'][0] == 24
    assert inflated['n_pixels_fit'][0] == 24
    stamp = np.full((5, 5), np.hypot(1.0, 2.0))
    psf = _MockPSF().evaluate(*np.meshgrid(
        int(round(xs[0])) + np.arange(-2, 3), int(round(ys[0])) + np.arange(-2, 3)),
        1.0, xs[0], ys[0])
    keep = np.ones((5, 5), bool)
    keep[2, 2] = False
    assert inflated['flux_err'][0] == pytest.approx(
        1.0 / np.sqrt(np.sum(psf[keep] ** 2 / stamp[keep] ** 2)), rel=1e-12)


def test_bkg_sigma_needs_error_and_matching_length():
    image, err, seed, *_ = _scene()
    with pytest.raises(ValueError, match='error'):
        forced_psf_photometry(image, _MockPSF(), seed,
                              bkg_sigma=np.ones(3))
    with pytest.raises(ValueError, match='shape'):
        forced_psf_photometry(image, _MockPSF(), seed, error=err,
                              bkg_sigma=np.ones(2))


# ---------------------------------------------------------------------------
# annulus_scatter_sigma
# ---------------------------------------------------------------------------
def _noise(sigma=5.0, shape=(80, 80), seed=1):
    rng = np.random.default_rng(seed)
    return 20.0 + rng.normal(0.0, sigma, shape), np.ones(shape, bool)


def test_annulus_sigma_recovers_scatter_and_matches_scalar_call():
    img, valid = _noise()
    xy = np.array([[40.0, 40.0], [30.4, 50.7]])
    out = annulus_scatter_sigma(img, valid, xy[:, 0], xy[:, 1], 8.0, 15.0)
    np.testing.assert_allclose(out, 5.0, rtol=0.1)
    assert out[1] == annulus_scatter_sigma(img, valid, 30.4, 50.7, 8.0, 15.0)


def test_annulus_sigma_ignores_masked_nan_and_inner_pixels():
    img, valid = _noise(sigma=2.0)
    yy, xx = np.indices(img.shape)
    rr = np.hypot(xx - 40.0, yy - 40.0)
    img = img + 1e5 * np.exp(-rr ** 2 / 2.0)          # star inside the hole
    wild = (rr >= 8.0) & (rr < 15.0) & (xx > 40)
    img[wild] = 1e4 * (-1.0) ** np.arange(wild.sum())
    valid[wild] = False                               # masked half
    img[40, 52] = np.nan                              # and a NaN
    sigma = annulus_scatter_sigma(img, valid, 40.0, 40.0, 8.0, 15.0)
    assert sigma == pytest.approx(2.0, rel=0.2)    # ~260 pixels: MAD is good to ~8%


def test_annulus_sigma_nan_when_too_few_pixels_or_off_image():
    img, valid = _noise()
    valid[:] = False
    valid[38:42, 38:42] = True                        # all inside the hole
    assert np.isnan(annulus_scatter_sigma(img, valid, 40.0, 40.0, 8.0, 15.0))
    valid[:] = True
    assert np.isnan(annulus_scatter_sigma(img, valid, 500.0, 500.0, 8.0, 15.0))
    assert np.isnan(annulus_scatter_sigma(img, valid, np.nan, 40.0, 8.0, 15.0))
    # min_pixels is a parameter: the same annulus passes with a low bar
    few = np.zeros_like(valid)
    few[40, 49:55] = True
    assert np.isnan(annulus_scatter_sigma(img, few, 40.0, 40.0, 8.0, 15.0))
    assert np.isfinite(annulus_scatter_sigma(img, few, 40.0, 40.0, 8.0, 15.0,
                                             min_pixels=5))


def test_annulus_sigma_uses_the_part_of_the_annulus_on_the_image():
    img, valid = _noise(sigma=3.0)
    sigma = annulus_scatter_sigma(img, valid, 2.0, 2.0, 8.0, 15.0)
    assert sigma == pytest.approx(3.0, rel=0.25)


def test_flat_annulus_reads_zero():
    img = np.full((60, 60), 7.0)
    valid = np.ones(img.shape, bool)
    assert annulus_scatter_sigma(img, valid, 30.0, 30.0, 8.0, 15.0) == 0.0
