"""Per-source local background from a small footprint on the STAR-SUBTRACTED residual.

Motivation
----------
The PSF fit already carries a local background: ``PSFPhotometry`` is built with
``LocalBackground(inner, outer)`` (an MMM-mode annulus, radii
``inner = max(6, 2*FWHM + 0.5*FWHM)``, ``outer = inner + max(4, FWHM)``) and
that estimator runs on **the array being fit** -- the NaN-interpolated,
optionally ``Background2D``-subtracted science frame with the *saturated*-star
model removed.  Two consequences: it still contains every unsaturated
neighbour, so in a crowded field the 6-10 px annulus is measuring other stars
as much as sky; and at the ``resbgsub`` stages it is measured on data whose
extended background has already been removed, so it does not describe the
diffuse emission at all.  It is also a single number per source per frame,
with no associated scatter.

This module adds an independent, complementary estimate, following the
convention Jay Anderson uses in JWST1PASS: the **mean and RMS of a small
(default 3x3 px) footprint centred on the source**, measured on the *residual*
image after the star-only model has been subtracted.  That residual is built
from the PRISTINE (pre-``Background2D``) data minus the satstar model minus the
star model, so it retains the extended emission while removing the point
sources -- which is exactly the quantity ``local_bkg`` cannot report.

Measured on brick F182M nrca1 (m7, 19 888 sources, one exposure):

    resbkg_mean   median 4.51   1-99% [2.34, 12.75]
    resbkg_rms    median 1.22
    local_bkg     median 0.44   (annulus, on the bg-SUBTRACTED data)
    median(local_bkg - resbkg_mean) = -4.04
    Pearson r(resbkg_mean, local_bkg) = 0.116

The ~4-unit offset is the extended background that ``Background2D`` had already
removed from the fitted array, and the near-zero correlation confirms the two
columns are measuring different things rather than duplicating each other.
Binning the same frame 8x8 across the detector, the cell medians of
``resbkg_mean`` span 3.02-7.22 with a cell-to-cell scatter of 0.82 against a
per-source MAD of 0.71 -- i.e. coherent spatial structure at the scale of the
extended emission, which is the expected behaviour.

The two estimates answer different questions and are both kept:

===============================  ==========================  ====================
                                 ``local_bkg`` (photutils)   ``resbkg_*`` (here)
===============================  ==========================  ====================
image                            data being fit (bg-sub'd)   pristine-minus-models residual
region                           annulus, r ~ 6-10 px        3x3 px on the source
statistic                        MMM mode                    mean, and pixel RMS
neighbour stars included         yes                         no (model removed)
used by the fit                  yes (subtracted)            no (diagnostic)
scatter reported                 no                          yes
===============================  ==========================  ====================

``resbkg_*`` is **diagnostic only** -- nothing here feeds back into the flux
fit.  Changing that would change every flux in the catalogue and is out of
scope for this module.

Per-frame vs merged
-------------------
:func:`measure_footprint_background` runs once per exposure and writes
``resbkg_mean`` / ``resbkg_rms`` / ``resbkg_npix`` into that exposure's
catalogue.  :func:`combine_frames` then reduces the per-exposure values to the
merged-catalogue entry: a **sigma-clipped, inverse-variance weighted average**
with the across-frame scatter, the propagated error, and the number of frames
that survived clipping.

Caveats
-------
* A 3x3 footprint is 9 pixels.  The per-frame ``resbkg_rms`` is a noisy
  estimate of the local pixel-to-pixel scatter (fractional uncertainty
  ~1/sqrt(2(N-1)) ~ 25%); it is useful as a weight and as a relative map, not
  as a precise noise value for a single frame.
* No clipping is applied *within* the footprint.  With 9 pixels a clip is
  unstable and would bias the mean low; masked and non-finite pixels are
  excluded instead.  Clipping happens across frames, where there are enough
  independent samples for it to mean something.
* The footprint sits exactly where the PSF model was subtracted, so a bad fit
  shows up here as a large ``|resbkg_mean|``.  That is a feature (it flags
  over/under-subtraction) but means the value is not a pure background estimate
  for poorly-fit sources -- read it together with ``qfit``.
"""
import warnings

import numpy as np

__all__ = ['FOOTPRINT_BOX', 'RESBKG_COLUMNS', 'MERGED_RESBKG_COLUMNS',
           'measure_footprint_background', 'combine_frames']

#: Default footprint size in pixels (Jay Anderson's JWST1PASS convention).
FOOTPRINT_BOX = 3

#: Per-exposure column names written by :func:`measure_footprint_background`.
RESBKG_COLUMNS = ('resbkg_mean', 'resbkg_rms', 'resbkg_npix')

#: Merged-catalogue column names written by :func:`combine_frames`.
MERGED_RESBKG_COLUMNS = ('resbkg_mean_avg', 'resbkg_mean_std',
                         'resbkg_mean_err', 'resbkg_rms_avg',
                         'resbkg_nframes')


def measure_footprint_background(residual, x, y, *, box=FOOTPRINT_BOX,
                                 mask=None):
    """Mean and RMS of a ``box`` x ``box`` footprint centred on each source.

    Parameters
    ----------
    residual : 2-D array
        The star-subtracted residual image -- ``data - star_model``, with the
        extended background retained.  Passing the *data* here instead would
        measure the star, not the background.
    x, y : array-like
        Source pixel positions (0-based, as ``x_fit``/``y_fit``).
    box : int, optional
        Footprint size in pixels; forced odd so the footprint is centred.
    mask : 2-D bool array, optional
        True where pixels must be excluded (bad DQ, saturated cores, ...).

    Returns
    -------
    mean, rms, npix : ndarray
        Per source: the mean of the valid footprint pixels, their standard
        deviation (``ddof=1``), and how many were valid.  ``mean``/``rms`` are
        NaN where no (or one) valid pixel was available; ``npix`` is 0/1 there,
        so a consumer can tell "off the edge" from "measured but noisy".
    """
    residual = np.asarray(residual, dtype=float)
    if residual.ndim != 2:
        raise ValueError(f"residual must be 2-D, got shape {residual.shape}")
    box = int(box)
    if box < 1:
        raise ValueError(f"box must be >= 1, got {box}")
    if box % 2 == 0:
        box += 1                      # keep the footprint centred on the pixel
    half = box // 2

    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    ny, nx = residual.shape
    n = x.size
    mean = np.full(n, np.nan)
    rms = np.full(n, np.nan)
    npix = np.zeros(n, dtype=int)

    # Non-finite residual pixels are excluded along with the caller's mask, so
    # a NaN-interpolated frame does not silently contribute interpolated values.
    bad = ~np.isfinite(residual)
    if mask is not None:
        bad = bad | np.asarray(mask, dtype=bool)

    # Cast only the finite positions: np.round(nan).astype(int) is undefined
    # behaviour and warns, and a warning per bad source floods a real run.
    finite_pos = np.isfinite(x) & np.isfinite(y)
    xc = np.zeros(n, dtype=int)
    yc = np.zeros(n, dtype=int)
    xc[finite_pos] = np.round(x[finite_pos]).astype(int)
    yc[finite_pos] = np.round(y[finite_pos]).astype(int)
    for i in range(n):
        if not finite_pos[i]:
            continue
        x0, x1 = max(xc[i] - half, 0), min(xc[i] + half + 1, nx)
        y0, y1 = max(yc[i] - half, 0), min(yc[i] + half + 1, ny)
        if x1 <= x0 or y1 <= y0:
            continue                  # centre lies off the array entirely
        cut = residual[y0:y1, x0:x1]
        good = ~bad[y0:y1, x0:x1]
        k = int(good.sum())
        npix[i] = k
        if k == 0:
            continue
        vals = cut[good]
        mean[i] = vals.mean()
        if k > 1:
            rms[i] = vals.std(ddof=1)
    return mean, rms, npix


def combine_frames(mean, rms, npix, *, sigma=3.0, maxiters=5, min_npix=2):
    """Reduce per-exposure footprint backgrounds to one value per source.

    Sigma-clips the per-frame means across frames, then takes the
    inverse-variance weighted average of the survivors.  The weight of a frame
    is ``npix / rms**2`` -- the inverse variance of *the mean* of that
    footprint, so a frame with more valid pixels or a quieter background counts
    for more.

    Parameters
    ----------
    mean, rms, npix : 2-D arrays, shape ``(n_sources, n_frames)``
        Per-exposure outputs of :func:`measure_footprint_background`, stacked.
        Non-detections must be NaN (``mean``/``rms``) / 0 (``npix``).
    sigma, maxiters : float, int
        Passed to ``astropy.stats.sigma_clip`` (``mad_std`` centre/scale), run
        across frames for each source.
    min_npix : int
        Frames with fewer valid footprint pixels than this are dropped before
        clipping -- an edge-clipped 1-pixel "footprint" is not a background.

    Returns
    -------
    dict
        ``resbkg_mean_avg``  weighted, clipped mean background
        ``resbkg_mean_std``  weighted scatter of the kept frames (the "RMS"
                             across exposures; NaN for a single frame, where
                             scatter is undefined rather than 0)
        ``resbkg_mean_err``  propagated error, ``1/sqrt(sum(w))``
        ``resbkg_rms_avg``   mean of the per-frame pixel RMS (local noise level)
        ``resbkg_nframes``   number of frames surviving the cuts and clipping
    """
    from astropy.stats import sigma_clip
    from astropy.utils.exceptions import AstropyUserWarning

    mean = np.atleast_2d(np.asarray(mean, dtype=float))
    rms = np.atleast_2d(np.asarray(rms, dtype=float))
    npix = np.atleast_2d(np.asarray(npix, dtype=float))
    if not (mean.shape == rms.shape == npix.shape):
        raise ValueError(f"mean/rms/npix shapes differ: {mean.shape}, "
                         f"{rms.shape}, {npix.shape}")

    usable = (np.isfinite(mean) & np.isfinite(rms) & (rms > 0)
              & (npix >= min_npix))

    with np.errstate(divide='ignore', invalid='ignore'):
        weights = np.where(usable, npix / rms ** 2, 0.0)
    weights = np.where(np.isfinite(weights), weights, 0.0)

    # Sigma-clip the per-frame means across frames.  Clipping is unweighted:
    # it is an outlier test on the values themselves (a frame where a
    # neighbouring star was badly subtracted), and weighting it would let one
    # high-weight outlier define the centre it is supposed to be tested against.
    masked = np.where(usable, mean, np.nan)
    with warnings.catch_warnings():
        # The NaNs are deliberate -- they mark frames where this source was not
        # measured.  astropy warns that it clipped them, which is the intent.
        warnings.simplefilter('ignore', AstropyUserWarning)
        clipped = sigma_clip(masked, sigma=sigma, maxiters=maxiters,
                             stdfunc='mad_std', cenfunc='median', axis=1,
                             masked=True)
    keep = usable & ~np.ma.getmaskarray(clipped)

    w = np.where(keep, weights, 0.0)
    sw = w.sum(axis=1)
    ok = sw > 0

    avg = np.full(mean.shape[0], np.nan)
    std = np.full(mean.shape[0], np.nan)
    err = np.full(mean.shape[0], np.nan)
    rms_avg = np.full(mean.shape[0], np.nan)
    nframes = keep.sum(axis=1).astype(int)

    vals = np.where(keep, mean, 0.0)
    avg[ok] = (w[ok] * vals[ok]).sum(axis=1) / sw[ok]
    with np.errstate(divide='ignore'):
        err[ok] = 1.0 / np.sqrt(sw[ok])

    # Weighted scatter across frames.  Undefined (NaN) for a single frame:
    # reporting 0 there would be false precision, the same trap as the
    # exactly-zero position scatter in merge_catalogs.
    multi = ok & (nframes > 1)
    if multi.any():
        dev = np.where(keep, mean - avg[:, None], 0.0)
        std[multi] = np.sqrt((w[multi] * dev[multi] ** 2).sum(axis=1)
                             / sw[multi])

    rsum = np.where(keep, rms, 0.0).sum(axis=1)
    rms_avg[ok] = rsum[ok] / nframes[ok]

    return {'resbkg_mean_avg': avg,
            'resbkg_mean_std': std,
            'resbkg_mean_err': err,
            'resbkg_rms_avg': rms_avg,
            'resbkg_nframes': nframes}
