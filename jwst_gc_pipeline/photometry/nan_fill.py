"""Kernel-weighted NaN replacement computed at the NaN pixels only.

``interpolate_replace_nans(data, kernel, convolve=convolve_fft)`` fills each
NaN pixel with the kernel-weighted mean of its finite neighbours.  It does so
by convolving the WHOLE frame twice on a padded FFT grid (the data with
NaN -> 0, and the finite-pixel weight map) and keeping the result at the NaN
pixels.  `_prepare_frame_for_photometry` runs it on every frame at every
phase.  On a 2048 x 2048 sgrb2 F182M frame (1.2% NaN, 7 x 7 kernel) it took
1.2-9.6 s depending on node load, the largest single cost of preparing an m8
frame; the direct sum below took 0.09-0.10 s.  On an F405N frame (6.1% NaN,
9 x 9 kernel): 1.3-7.9 s against 0.6-1.0 s (#1223).

`replace_nans` computes the same weighted mean directly at the NaN pixels:
for each kernel element it gathers the shifted data and weight at those
pixels and accumulates them.  Its cost scales with (NaN pixels x kernel
elements) instead of with the padded frame, so it suits a frame whose NaNs
are a small fraction of its pixels (masked cores, bad pixels, the reference
border), and does not suit a mosaic that is mostly NaN outside its footprint.

It reproduces ``convolve_fft``'s defaults as ``interpolate_replace_nans``
calls it (``boundary='fill'``, ``fill_value=0``, ``normalize_kernel=True``,
``nan_treatment='interpolate'``, ``min_wt=0``):

- the kernel is divided by its sum, after zeroing non-finite elements (a
  kernel summing to less than ``1 / MAX_NORMALIZATION`` goes to astropy, which
  refuses it);
- a non-finite pixel (NaN or +-inf) contributes value 0 with weight 0;
- a pixel outside the frame contributes value 0 with weight 1;
- a NaN pixel whose summed weight is below ``10 * eps`` becomes 0;
- only NaN pixels are replaced; +-inf pixels and the dtype are kept.

The sums are exact up to float64 rounding.  ``convolve_fft``'s are not: an
FFT's rounding error is relative to the largest values in the frame, so it
differs from the direct sum where a NaN pixel's weight is small (a pixel deep
inside a masked core, filled from the kernel's tail).  On the two sgrb2
frames, at the 2000 NaN pixels where the two differ most, the direct result
equals float32 of an extended-precision sum at 1998 and 1994 of them, the FFT
result at 2 and 3; the FFT is off by up to 6.8e-4 where the weight is ~1e-9.

Only a plain 2-D `numpy.ndarray` takes the direct path.  A masked array (whose
masked pixels astropy treats as NaN), a Quantity, or any other input goes
through ``interpolate_replace_nans`` unchanged, as does everything when
``NAN_FILL_DIRECT=0``.
"""
import os

import numpy as np
from astropy.convolution import convolve_fft, core as _convolution_core
from astropy.convolution import interpolate_replace_nans

#: ``convolve_fft`` zeroes the result where the convolved weight is below
#: ``10 * np.finfo(bigimwt.dtype).eps`` with ``bigimwt`` complex128.
ZERO_WEIGHT = 10 * np.finfo(np.complex128).eps

#: ``convolve_fft`` refuses to normalize a kernel whose sum is below this.
MIN_KERNEL_SUM = 1.0 / getattr(_convolution_core, 'MAX_NORMALIZATION', 100)

#: NaN pixels per gather; bounds the temporary index and value arrays.
_CHUNK = 1 << 20


def _fft_replace_nans(array, kernel):
    return interpolate_replace_nans(array, kernel, convolve=convolve_fft,
                                    allow_huge=True)


def replace_nans(array, kernel):
    """``interpolate_replace_nans(array, kernel, convolve=convolve_fft)``,
    summed directly at the NaN pixels.

    It takes no ``convolve_fft`` keywords: it reproduces the defaults
    ``interpolate_replace_nans`` uses (``min_wt=0`` among them).

    Parameters
    ----------
    array : 2-D `numpy.ndarray`
        The frame.  NaN pixels are replaced; everything else is copied.
    kernel : `astropy.convolution.Kernel2D` or 2-D array
        The interpolation kernel, centred at index ``shape // 2`` on each
        axis as ``convolve_fft`` centres it (any size, odd or even).

    Returns
    -------
    `numpy.ndarray`
        A copy of ``array`` with its NaN pixels replaced.  Anything other
        than a plain 2-D ndarray, an unnormalizable kernel, or
        ``NAN_FILL_DIRECT=0`` goes through ``interpolate_replace_nans`` with
        ``convolve_fft``.
    """
    karr = np.array(getattr(kernel, 'array', kernel), dtype=float)
    if (os.environ.get('NAN_FILL_DIRECT', '1') == '0'
            or type(array) is not np.ndarray
            or array.ndim != 2 or karr.ndim != 2):
        return _fft_replace_nans(array, kernel)
    isnan = np.isnan(array)
    if not isnan.any():
        return array.copy()
    karr[~np.isfinite(karr)] = 0
    if not karr.sum() >= MIN_KERNEL_SUM:
        # let astropy raise its own unnormalizable-kernel error
        return _fft_replace_nans(array, kernel)
    karr /= karr.sum()

    ky, kx = karr.shape
    cy, cx = ky // 2, kx // 2
    ny, nx = array.shape
    # out[r, c] = sum_ij k[i, j] * a[r + cy - i, c + cx - j]; the row offset
    # cy - i reaches cy below and ky - 1 - cy above, so pad by that much.
    top, left = ky - 1 - cy, kx - 1 - cx
    py, px = ny + ky - 1, nx + kx - 1
    finite = np.isfinite(array)
    value = np.zeros((py, px))
    weight = np.ones((py, px))
    value[top:top + ny, left:left + nx][finite] = array[finite]
    weight[top:top + ny, left:left + nx] = finite
    value = value.ravel()
    weight = weight.ravel()

    rows, cols = np.nonzero(isnan)
    base = (rows + top) * px + (cols + left)
    del rows, cols
    offsets = [((cy - i) * px + (cx - j), karr[i, j])
               for i in range(ky) for j in range(kx) if karr[i, j] != 0]
    filled = np.empty(base.size)
    for start in range(0, base.size, _CHUNK):
        idx = base[start:start + _CHUNK]
        num = np.zeros(idx.size)
        den = np.zeros(idx.size)
        for off, k in offsets:
            at = idx + off
            num += k * value[at]
            den += k * weight[at]
        with np.errstate(divide='ignore', invalid='ignore'):
            out = num / den
        out[den < ZERO_WEIGHT] = 0.0
        filled[start:start + idx.size] = out

    result = array.copy()
    result[isnan] = filled
    return result
