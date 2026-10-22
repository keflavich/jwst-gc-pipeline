"""`nan_fill.replace_nans` reproduces interpolate_replace_nans(convolve_fft).

`_prepare_frame_for_photometry` fills each frame's NaN pixels with
`replace_nans`, a direct kernel-weighted sum at the NaN pixels, in place of two
whole-frame FFT convolutions (#1223).  The two compute the same quantity and
differ only in rounding; these tests pin the semantics (normalization, edges,
non-finite pixels, empty windows, dtype) against astropy, and the values
against the same sum in extended precision.
"""
import inspect

import numpy as np
import pytest
from astropy.convolution import (Gaussian2DKernel, convolve_fft,
                                 interpolate_replace_nans)

from jwst_gc_pipeline.photometry import nan_fill


def _frame(seed, shape=(61, 73), nan_fraction=0.15, dtype=np.float32):
    """Sky + a few bright spikes, scattered NaNs, a 19 x 19 interior hole (its
    centre has no finite pixel inside the kernel), NaNs along two edges and in
    a corner, and two +-inf pixels."""
    rng = np.random.default_rng(seed)
    ny, nx = shape
    a = rng.normal(100, 30, shape)
    a[rng.uniform(size=shape) < 0.02] *= 1e3
    a[rng.uniform(size=shape) < nan_fraction] = np.nan
    a[ny // 3:ny // 3 + 19, nx // 3:nx // 3 + 19] = np.nan
    a[0, :9] = np.nan
    a[:, -1] = np.nan
    a[-3:, :3] = np.nan
    a[ny // 6, nx // 6] = np.inf
    a[2 * ny // 3, 5] = -np.inf
    return a.astype(dtype)


def _astropy(a, kernel):
    return interpolate_replace_nans(a, kernel, convolve=convolve_fft,
                                    allow_huge=True)


def _exact(a, kernel):
    """The weighted mean at every NaN pixel, summed in np.longdouble with
    convolve_fft's conventions spelled out one pixel at a time."""
    k = np.array(getattr(kernel, 'array', kernel), np.longdouble)
    k /= k.sum()
    ky, kx = k.shape
    cy, cx = ky // 2, kx // 2
    ny, nx = a.shape
    out = {}
    for r, c in zip(*np.nonzero(np.isnan(a))):
        num = den = np.longdouble(0)
        for i in range(ky):
            for j in range(kx):
                rr, cc = r + cy - i, c + cx - j
                if 0 <= rr < ny and 0 <= cc < nx:
                    if np.isfinite(a[rr, cc]):
                        num += k[i, j] * np.longdouble(a[rr, cc])
                        den += k[i, j]
                else:  # boundary='fill', fill_value=0: a zero of weight 1
                    den += k[i, j]
        out[r, c] = 0.0 if den < nan_fill.ZERO_WEIGHT else num / den
    return out


KERNELS = {
    'gauss-7': Gaussian2DKernel(0.82),
    'gauss-9': Gaussian2DKernel(0.92),
    'gauss-17': Gaussian2DKernel(2.0),
    'odd-asymmetric': np.arange(1.0, 16.0).reshape(3, 5) ** 1.5,
    'even': np.arange(1.0, 13.0).reshape(4, 3),
}


@pytest.mark.parametrize('name', sorted(KERNELS))
@pytest.mark.parametrize('dtype', [np.float32, np.float64, '>f4'])
def test_it_matches_interpolate_replace_nans(monkeypatch, name, dtype):
    monkeypatch.delenv('NAN_FILL_DIRECT', raising=False)
    kernel = KERNELS[name]
    a = _frame(3, dtype=dtype)
    ref = _astropy(a, kernel)
    new = nan_fill.replace_nans(a, kernel)
    assert new.dtype == ref.dtype == a.dtype
    isnan = np.isnan(a)
    # everything that is not NaN is copied bit for bit (+-inf included)
    assert np.array_equal(new[~isnan], a[~isnan])
    assert np.array_equal(ref[~isnan], a[~isnan])
    # every NaN pixel is filled; where astropy found no weight (the middle of
    # the hole) both give 0.  A pixel whose only weight is outside the frame
    # is an exact 0 here and an FFT rounding residue (~1e-14) in astropy.
    assert np.isfinite(new[isnan]).all() and np.isfinite(ref[isnan]).all()
    no_weight = isnan & (ref == 0)
    assert no_weight.any()
    assert (new[no_weight] == 0).all()
    # and agree to the FFT's rounding
    diff = np.abs(new[isnan].astype(float) - ref[isnan].astype(float))
    assert (diff <= 1e-6 * np.maximum(np.abs(ref[isnan]), 1.0)).all()


@pytest.mark.parametrize('name', ['gauss-7', 'odd-asymmetric', 'even'])
def test_it_is_the_exact_weighted_mean(monkeypatch, name):
    """float64 frame: within a few float64 ulps of the extended-precision sum
    at every NaN pixel, edges and empty windows included."""
    monkeypatch.delenv('NAN_FILL_DIRECT', raising=False)
    kernel = KERNELS[name]
    a = _frame(11, shape=(33, 41), dtype=np.float64)
    new = nan_fill.replace_nans(a, kernel)
    for (r, c), want in _exact(a, kernel).items():
        assert abs(new[r, c] - float(want)) <= 1e-13 * max(abs(float(want)), 1.0)


def test_chunking_does_not_change_a_bit(monkeypatch):
    monkeypatch.delenv('NAN_FILL_DIRECT', raising=False)
    a = _frame(5)
    whole = nan_fill.replace_nans(a, KERNELS['gauss-9'])
    monkeypatch.setattr(nan_fill, '_CHUNK', 7)
    chunked = nan_fill.replace_nans(a, KERNELS['gauss-9'])
    assert np.array_equal(whole.view(np.uint32), chunked.view(np.uint32))


def test_a_frame_without_nans_is_copied(monkeypatch):
    monkeypatch.delenv('NAN_FILL_DIRECT', raising=False)
    a = np.arange(30.0).reshape(5, 6)
    a[2, 2] = np.inf
    out = nan_fill.replace_nans(a, KERNELS['gauss-7'])
    assert out is not a
    assert np.array_equal(out, a)


@pytest.mark.parametrize('why, array, kernel', [
    ('switched off', None, None),
    ('1-d', np.array([1.0, np.nan, 3.0]), np.array([0.25, 0.5, 0.25])),
    ('3-d', np.where(np.eye(4)[None] > 0, np.nan, 1.0), np.ones((3, 3, 3))),
])
def test_anything_else_goes_through_astropy(monkeypatch, why, array, kernel):
    calls = []
    real = nan_fill._fft_replace_nans

    def _spy(a, k):
        calls.append(1)
        return real(a, k)

    monkeypatch.setattr(nan_fill, '_fft_replace_nans', _spy)
    if array is None:
        monkeypatch.setenv('NAN_FILL_DIRECT', '0')
        array, kernel = _frame(2), KERNELS['gauss-7']
    else:
        monkeypatch.delenv('NAN_FILL_DIRECT', raising=False)
    out = nan_fill.replace_nans(array, kernel)
    assert calls == [1]
    assert np.array_equal(out, _astropy(array, kernel), equal_nan=True)


def test_an_unnormalizable_kernel_raises_as_astropy_does(monkeypatch):
    monkeypatch.delenv('NAN_FILL_DIRECT', raising=False)
    zero_sum = np.array([[1.0, -1.0], [-1.0, 1.0]])
    with pytest.raises(Exception, match="can't be normalized"):
        nan_fill.replace_nans(_frame(1), zero_sum)


def test_prepare_frame_fills_through_replace_nans():
    from jwst_gc_pipeline.photometry import cataloging
    src = inspect.getsource(cataloging._prepare_frame_for_photometry)
    assert 'nan_replaced_data = replace_nans(data_, kernel)' in src
    assert '= interpolate_replace_nans(' not in src
