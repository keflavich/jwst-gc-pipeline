"""Per-source local background from a small footprint on the STAR-SUBTRACTED residual.

Motivation
----------
The PSF fit already carries a local background: ``PSFPhotometry`` is built with
``LocalBackground(inner, outer)`` (radii ``inner = max(6, 2*FWHM + 0.5*FWHM)``,
``outer = inner + max(4, FWHM)`` -- 6 and 10 px for F182M), whose estimator is
photutils' default **sigma-clipped MedianBackground** (NOT MMM; verified from
``repr(LocalBackground(6, 10))`` and from the ``LOCAL_BK`` card written into the
catalogues).  That estimator runs on **the array being fit** -- the
NaN-interpolated frame, with the *saturated*-star model removed and whichever
background subtraction is in effect already applied.  Two consequences: it still
contains every unsaturated neighbour, so in a crowded field the 6-10 px annulus
measures other stars as much as sky; and at the ``resbgsub`` stages the extended
background has already been removed from that array, so it describes the diffuse
emission only weakly.  It is also a single number per source per frame, with no
associated scatter.

This module adds an independent, complementary estimate, following the
convention Jay Anderson uses in JWST1PASS: the **mean and RMS of a small
(default 3x3 px) footprint centred on the source**, measured on the *residual*
image after the star-only model has been subtracted.  That residual is built
from the PRISTINE (pre-background-subtraction) data minus the satstar model
minus the star model, so it retains the extended emission while removing the
point sources -- which is exactly what ``local_bkg`` cannot report.

Does it trace the extended emission?  Yes.  Measured on brick F182M nrca1 m7
(19 888 sources, one exposure), against the m6 smoothed-residual background map
that was actually subtracted from the fitted array, sampled at each source:

    Spearman r(resbkg_mean, subtracted bg map) = 0.86
    Spearman r(local_bkg,   subtracted bg map) = 0.32
    median bg-map value at sources = 4.08,  median resbkg_mean = 4.51,
    median local_bkg = 0.44,  median(local_bkg - resbkg_mean) = -4.04

The ~4-unit offset is the extended background that had already been removed from
the fitted array.  NOTE it is removed by the **iter3/m6 residual-smoothed
background** (``--use-iter3-residual-bg``, the ``_resbgsub`` token), not by
``Background2D`` -- at m7 ``options.bgsub`` is False and ``Background2D`` never
runs on these frames.

Spatial coherence, same frame binned 8x8 across the detector: cell medians span
3.02-7.22 with a cell-to-cell scatter of 0.822, against a standard error on a
cell median of 0.074 (~321 sources per cell).  That is structure at **11x** the
noise on it.  (Do not compare the cell scatter to the per-source MAD: the raw
MAD of 0.710 is 1.053 in sigma units, i.e. LARGER than the cell scatter, so that
comparison argues the wrong way.  The right comparison is against the error on
the cell statistic.)

The two columns are **not** independent -- Spearman r(resbkg_mean, local_bkg) =
0.51, since both are influenced by the same sky.  (A Pearson r of 0.116 on these
heavy-tailed distributions is misleading, and a low correlation would not have
demonstrated independence anyway.)  The case for keeping both is the 0.86-vs-0.32
comparison above, not their mutual correlation.

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

Scope: this describes the per-frame PSF-fitting path (``cataloging.py``, and
the identical setup in ``legacy/crowdsource_step.py`` and
``artificial_stars.py``).  The **saturated**-star channel in
``reduction/saturated_star_finding.py`` is different -- ``LocalBackground(25,
50)``, an adaptive annulus, or ``None`` with a median-filter background
subtracted beforehand for MIRI -- and is not covered by anything here.
Saturated stars appended by ``replace_saturated`` get ``resbkg_* = NaN``, so
they sit outside both estimators.

``resbkg_*`` is **diagnostic only** -- nothing here feeds back into the flux
fit.  Changing that would change every flux in the catalogue and is out of
scope for this module.

.. warning::

   **``resbkg_mean`` is not a pure sky measurement for bright sources.**  The
   footprint sits exactly where the PSF model was subtracted, so imperfect
   subtraction enters it directly.  On the frame above, binned by fitted flux:

       flux percentile        median resbkg_mean   median subtracted bg map
       faintest quartile            3.92                   3.65
       middle                       4.54                   4.29
       upper quartile               5.03                   4.48
       brightest 1%                14.27                   5.48

   So for a typical source the contamination is ~6% and harmless, but in the
   brightest percentile roughly **+9 units -- about 2.6x the true background --
   is subtraction residual, not sky**.  Spearman r(resbkg_mean, flux_fit) =
   0.32; the full range of resbkg_mean runs from -220 to +455.  ``resbkg_rms``
   is even more flux-correlated (r = 0.64), so the combining weight
   ``npix/rms**2`` is itself correlated with the value it weights.  Use
   ``resbkg_mean`` as a background tracer only with a flux or ``qfit`` cut; read
   large ``|resbkg_mean|`` on a bright star as a subtraction-quality flag.

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
* Footprint pixels are correlated -- lag-1 autocorrelation of the high-passed
  residual is 0.43 (y) / 0.45 (x), from destreaking, NaN interpolation, IPC and
  the model subtraction itself (these are detector-space ``_crf`` frames, never
  resampled, so this is not drizzle correlation).  Strictly, ``npix/rms**2`` is
  therefore not the inverse variance of the mean.  Empirically it is close: two
  errors cancel, since correlation makes ``rms`` under-estimate sigma while
  N_eff < npix makes the true variance of the mean larger.  Measured over 8 real
  exposures, ``median(resbkg_mean_std / resbkg_mean_err) = 1.06`` for sources
  with >=3 frames, i.e. the propagated error is right to ~10%.
* See the bright-source warning above: this is not a pure sky measurement.
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


def combine_frames(mean, rms, npix, *, sigma=3.0, maxiters=5, min_npix=2,
                   dtype=np.float32):
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

    # float32 throughout, and NO upcast of the caller's stacks.  These arrays
    # are (n_sources, n_frames): at brick F200W scale (4.4M x 192) each one is
    # ~3.4 GB in float32 and ~6.8 GB in float64, and combine_singleframe's whole
    # design (see its Phase 1/2 comments) is about keeping that budget bounded.
    # Every temporary below is freed as soon as it is dead for the same reason.
    mean = np.asarray(mean, dtype=dtype)
    rms = np.asarray(rms, dtype=dtype)
    npix = np.asarray(npix, dtype=dtype)
    # 2-D only: np.atleast_2d would silently read a per-SOURCE 1-D array as one
    # source observed in N frames and return a single meaningless value.
    if mean.ndim != 2:
        raise ValueError(
            f"mean/rms/npix must be 2-D (n_sources, n_frames); got ndim="
            f"{mean.ndim}. For a single source pass shape (1, n_frames).")
    if not (mean.shape == rms.shape == npix.shape):
        raise ValueError(f"mean/rms/npix shapes differ: {mean.shape}, "
                         f"{rms.shape}, {npix.shape}")

    usable = (np.isfinite(mean) & np.isfinite(rms) & (rms > 0)
              & (npix >= min_npix))

    with np.errstate(divide='ignore', invalid='ignore'):
        weights = np.where(usable, npix / rms ** 2, 0).astype(dtype, copy=False)
    weights[~np.isfinite(weights)] = 0

    # Sigma-clip the per-frame means across frames.  Clipping is unweighted:
    # it is an outlier test on the values themselves (a frame where a
    # neighbouring star was badly subtracted), and weighting it would let one
    # high-weight outlier define the centre it is supposed to be tested against.
    masked = np.where(usable, mean, np.nan).astype(dtype, copy=False)
    with warnings.catch_warnings():
        # The NaNs are deliberate -- they mark frames where this source was not
        # measured.  astropy warns that it clipped them, which is the intent.
        warnings.simplefilter('ignore', AstropyUserWarning)
        clipped = sigma_clip(masked, sigma=sigma, maxiters=maxiters,
                             stdfunc='mad_std', cenfunc='median', axis=1,
                             masked=True)
    del masked
    keep = usable & ~np.ma.getmaskarray(clipped)
    del clipped, usable

    w = np.where(keep, weights, 0).astype(dtype, copy=False)
    del weights
    sw = w.sum(axis=1)
    ok = sw > 0

    avg = np.full(mean.shape[0], np.nan)
    std = np.full(mean.shape[0], np.nan)
    err = np.full(mean.shape[0], np.nan)
    rms_avg = np.full(mean.shape[0], np.nan)
    nframes = keep.sum(axis=1).astype(int)

    vals = np.where(keep, mean, 0).astype(dtype, copy=False)
    avg[ok] = (w[ok] * vals[ok]).sum(axis=1) / sw[ok]
    del vals
    with np.errstate(divide='ignore'):
        err[ok] = 1.0 / np.sqrt(sw[ok])

    # Weighted scatter across frames.  Undefined (NaN) for a single frame:
    # reporting 0 there would be false precision, the same trap as the
    # exactly-zero position scatter in merge_catalogs.
    multi = ok & (nframes > 1)
    if multi.any():
        dev = np.where(keep, mean - avg[:, None], 0).astype(dtype, copy=False)
        # Population (not sample) weighted scatter, matching the existing
        # std_*_avg convention in merge_catalogs.  It is biased LOW for small N
        # -- 29% at N=2, ~10% at N=8 on real data -- which is acceptable for a
        # spread indicator but means it is not an unbiased sigma estimate.
        std[multi] = np.sqrt((w[multi] * dev[multi] ** 2).sum(axis=1)
                             / sw[multi])
        del dev

    # Weighted, for consistency with everything else here (an unweighted mean
    # would let an edge-clipped 2-pixel footprint count as much as a full one).
    rsum = np.where(keep, w * rms, 0).sum(axis=1)
    rms_avg[ok] = rsum[ok] / sw[ok]

    return {'resbkg_mean_avg': avg,
            'resbkg_mean_std': std,
            'resbkg_mean_err': err,
            'resbkg_rms_avg': rms_avg,
            'resbkg_nframes': nframes}
