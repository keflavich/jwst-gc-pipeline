"""Frame-to-frame MIRI sky offsets from pixel-wise overlap differences.

WHY THIS EXISTS
---------------
``SkyMatchStep(skymethod='match')`` estimates the sky of EACH image separately
inside an overlap and then differences those estimates.  In a MIRI field that
is half infrared-dark cloud and half bright nebula (10678 near the Brick), the
per-image pixel populations inside an overlap differ -- different DQ masks,
edge trims, saturated cores -- so the per-image 'mode' (or median) lands on
different peaks of a bimodal distribution.  On jw10678-o078 F770W the raw
frames agree to <=1.8 MJy/sr, and skymatch applied relative levels up to
130 MJy/sr (mode) or 34 MJy/sr (median); 'global+match' does the same.
outlier_detection then sees frames that disagree by tens of sigma, flags
41-65% of their pixels, and resample leaves NaN holes: the black patches in
the served MIRI mosaics.  o114, o071 and o066 failed the same way.

Differencing the frames PIXEL BY PIXEL on a common grid and taking the median
of that difference compares the same sky in both frames, so a bimodal scene
cancels.  The per-frame levels come from a least-squares solve over all
overlapping pairs and go to skymatch as ``skymethod='user'``.
"""
import os

import numpy as np
from astropy.io import fits
from astropy.nddata import block_reduce
from astropy.wcs import WCS

__all__ = ['pairwise_sky_levels', 'write_skylist', 'DNU_BIT']

#: DQ DO_NOT_USE.  Outlier-flagged pixels also carry it, so inputs are the
#: pre-outlier-detection frames (align/cal), where it is only real bad pixels
#: and the edge trim.
DNU_BIT = 1


def _load(fn):
    with fits.open(fn) as hdul:
        data = hdul['SCI'].data.astype(float)
        data[(hdul['DQ'].data & DNU_BIT) > 0] = np.nan
        wcs = WCS(hdul['SCI'].header)
    return data, wcs


def _bin(grid, bin_factor):
    if bin_factor <= 1:
        return grid
    ny = grid.shape[0] // bin_factor * bin_factor
    nx = grid.shape[1] // bin_factor * bin_factor
    return block_reduce(grid[:ny, :nx], bin_factor, func=np.nanmedian)


def pairwise_sky_levels(files, bin_factor=4, min_overlap=200):
    """Per-frame sky levels that make overlapping frames agree.

    Parameters
    ----------
    files : list of str
        Pre-outlier-detection frames (``*_align.fits``/``*_cal.fits``) with a
        SCI extension, a DQ extension and a celestial FITS WCS.
    bin_factor : int
        Block size for a nanmedian binning of the REPROJECTED grids, which
        also suppresses stars.  Binning happens after reprojection because
        a step-sliced ``astropy.wcs.WCS`` does not carry SIP distortion
        correctly: binning first misregistered o078's frames by enough to
        read 200 MJy/sr differences off real structure.
    min_overlap : int
        Minimum number of binned pixels for a pair to enter the solve.

    Returns
    -------
    levels : ndarray
        One level per file, in skymatch's ``match_down=False`` convention:
        the highest is 0 and the rest are <= 0, so subtracting them raises
        the fainter frames to the brightest.
    pairs : list of tuple
        ``(i, j, npix, median(frame_i - frame_j), post-solve residual)``.
    """
    from reproject import reproject_interp
    from reproject.mosaicking import find_optimal_celestial_wcs

    frames = [_load(fn) for fn in files]
    n = len(frames)
    if n < 2:
        return np.zeros(n), []
    wcs_out, shape_out = find_optimal_celestial_wcs(frames)
    grids = [_bin(reproject_interp(fr, wcs_out, shape_out=shape_out,
                                   roundtrip_coords=False)[0],
                  bin_factor)
             for fr in frames]

    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            m = np.isfinite(grids[i]) & np.isfinite(grids[j])
            npix = int(m.sum())
            if npix < min_overlap:
                continue
            rows.append((i, j, npix, float(np.median(grids[i][m] - grids[j][m]))))
    if not rows:
        return np.zeros(n), []

    # x_i - x_j = d_ij, weighted by sqrt(npix), plus sum(x) = 0 to pin the
    # otherwise free additive constant
    A = np.zeros((len(rows) + 1, n))
    y = np.zeros(len(rows) + 1)
    w = np.ones(len(rows) + 1)
    for k, (i, j, npix, d) in enumerate(rows):
        A[k, i], A[k, j], y[k], w[k] = 1.0, -1.0, d, np.sqrt(npix)
    A[-1, :] = 1.0
    w[-1] = w[:-1].max()
    x, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
    resid = y[:-1] - A[:-1] @ x
    levels = x - x.max()
    pairs = [(i, j, npix, d, float(r))
             for (i, j, npix, d), r in zip(rows, resid)]
    return levels, pairs


def write_skylist(files, levels, path):
    """Write ``(filename sky)`` rows for ``SkyMatchStep(skymethod='user')``,
    which matches them to models by filename stem."""
    with open(path, 'w') as fh:
        for fn, lv in zip(files, levels):
            fh.write(f'{os.path.basename(fn)} {lv:.6f}\n')
    return path
