"""Short-axis width of simulated stars, narrow cores, streaks and blended pairs, measured with the width gate's function.

usage: python widthsim.py [--code PATH]

``_minor_axis_sigma`` (the width gate of the bright roundness window) is applied to noisy Gaussian stars of the F212N and F480M PSF widths at peak S/N 100
and 30 (400 random sub-pixel positions each), to Gaussians narrower than the PSF (a hot-pixel-like core), to a one-pixel streak and a single pixel, and to
equal pairs of F212N stars 2, 3 and 4 pixels apart.  The widths are printed in units of the PSF standard deviation (FWHM / 2.355) unless a column says px.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = sys.argv[sys.argv.index('--code') + 1] if '--code' in sys.argv else os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, CODE)
os.environ.setdefault('GC_ALLOW_DEV', '1')
from jwst_gc_pipeline.photometry.cataloging import _minor_axis_sigma as width  # noqa: E402

rng = np.random.default_rng(1)
N = 31
yy, xx = np.mgrid[:N, :N]


def star(fwhm, x0, y0, amp=1000.0, noise=0.0):
    s = fwhm / 2.3548
    im = amp * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * s * s))
    return im + (rng.normal(0, noise, im.shape) if noise > 0 else 0.0)


for fwhm, name in ((2.34, 'F212N'), (2.57, 'F480M')):
    s = fwhm / 2.3548
    for snr in (np.inf, 100.0, 30.0):
        w = np.array([width(star(fwhm, 15 + rng.uniform(-0.5, 0.5), 15 + rng.uniform(-0.5, 0.5), 1000.0, 1000.0 / snr if np.isfinite(snr) else 0.0),
                            [15.0], [15.0])[0] for _ in range(400)])
        print(f'{name} PSF FWHM {fwhm} px, peak S/N {snr:g}: width / PSF sigma min {np.nanmin(w) / s:.2f}, 5th percentile {np.nanpercentile(w, 5) / s:.2f}, '
              f'median {np.nanmedian(w) / s:.2f}, max {np.nanmax(w) / s:.2f}')
for fwhm in (1.0, 1.4, 1.8):
    w = np.array([width(star(fwhm, 15 + rng.uniform(-0.5, 0.5), 15 + rng.uniform(-0.5, 0.5)), [15.0], [15.0])[0] for _ in range(400)])
    print(f'noiseless Gaussian of FWHM {fwhm} px: width px min {np.nanmin(w):.2f}, median {np.nanmedian(w):.2f}, max {np.nanmax(w):.2f}')
im = np.zeros((N, N))
im[15, 10:21] = 1000.0
print('horizontal one-pixel streak, width px:', width(im, [15.0], [15.0])[0])
im = np.zeros((N, N))
for k in range(-5, 6):
    im[15 + k, 15 + k] = 1000.0
print('diagonal one-pixel streak, width px:', width(im, [15.0], [15.0])[0])
im = np.zeros((N, N))
im[15, 15] = 1000.0
print('single pixel, width px:', width(im, [15.0], [15.0])[0])
s = 2.34 / 2.3548
for d in (2.0, 3.0, 4.0):
    im = star(2.34, 15 - d / 2, 15) + star(2.34, 15 + d / 2, 15)
    print(f'equal pair of F212N stars {d:g} px apart: width / PSF sigma {width(im, [15.0], [15.0])[0] / s:.2f}')
