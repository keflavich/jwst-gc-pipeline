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

    Spearman r(modelsub_bkg, subtracted bg map) = 0.86
    Spearman r(local_bkg,   subtracted bg map) = 0.32
    median bg-map value at sources = 4.08,  median modelsub_bkg = 4.51,
    median local_bkg = 0.44,  median(local_bkg - modelsub_bkg) = -4.04

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

The two columns are **not** independent -- Spearman r(modelsub_bkg, local_bkg) =
0.51, since both are influenced by the same sky.  (A Pearson r of 0.116 on these
heavy-tailed distributions is misleading, and a low correlation would not have
demonstrated independence anyway.)  The case for keeping both is the 0.86-vs-0.32
comparison above, not their mutual correlation.

The two estimates answer different questions and are both kept:

===============================  ==========================  ====================
                                 ``local_bkg`` (photutils)   ``modelsub_bkg`` (here)
===============================  ==========================  ====================
image                            data being fit              pristine minus models
region                           annulus, r ~ 6-10 px        3x3 px on the source
statistic                        sigma-clipped median        mean, and pixel RMS
point sources removed            no                          yes (model subtracted)
stage-dependent                  yes                         no
used by the fit                  yes (subtracted)            no (diagnostic)
scatter reported                 no                          yes
===============================  ==========================  ====================

**What actually produces the gain: the IMAGE, not the footprint.**  It would be
easy to read the table above as "a 3x3 box beats an annulus because an annulus
is contaminated by neighbours".  That is not what the data says, and the
neighbour argument is weak anyway -- ``LocalBackground``'s estimator is a
3-sigma-clipped median, which is exactly the statistic that survives a minority
of contaminated pixels.

Running photutils' own ``LocalBackground(6, 10)`` -- same class, same radii --
on the SAME star-subtracted residual, and scoring both against the m6
smoothed-bg map that was actually subtracted (brick F182M nrca1 m7, n = 19 865):

    Spearman vs bg map:  3x3 on residual      0.865
                         6-10 annulus on residual   0.891
                         6-10 annulus on fitted data (= local_bkg)  0.330
    median:              3x3 4.513 | annulus 4.193 | bg map 4.079
    scatter about map:   3x3 0.371 | annulus 0.293
    brightest 1%:        3x3 14.24 | annulus 4.99 | bg map 5.55

The annulus on the residual is **better on every metric** -- closer median,
21% tighter scatter, and essentially immune to the bright-source contamination
that is this column's worst failure mode (4.99 vs 14.24 against a truth of
5.55).  The whole of the 0.33 -> 0.87 improvement comes from measuring on the
star-subtracted, pre-background-subtraction residual; none of it requires the
3x3.

The 3x3 is kept because it is **Jay Anderson's JWST1PASS convention**, because
``modelsub_bkg_rms`` is meant to be a LOCAL pixel-scatter estimate (an annulus
RMS at r ~ 6-10 px is a different quantity), and because a compact footprint
stays local in a crowded field.  It costs accuracy relative to a wider
footprint on the same residual, and that is a deliberate trade.
``manual_residual_background_box`` / ``--residual-background-box`` change it.

One caveat on the comparison, in fairness: the reference is itself a *smoothed*
map, so a ~200-pixel aperture correlates with it better than a 9-pixel one
partly by construction.  That does not apply to the bright-1% row, which is the
decisive one.

Why not just sample the background map directly?
------------------------------------------------
At m5-m7 ``ctx.background_map`` IS the map that was subtracted, it is already
passed to ``save_photutils_results``, and ``_sample_background_map()`` would
read it at each source in three lines -- scoring 1.00 against itself, with no
bright-source contamination and no 9-pixel noise.  It is a fair question.

The answer is that they are different objects.  The map is a *model* built from
the previous iteration's residual, smoothed; ``modelsub_bkg`` is a *measurement*
on this iteration's residual at this source.  Sampling the map tells you what
the pipeline assumed; measuring the residual tells you what is actually left
there, including where the model was wrong.  ``modelsub_bkg_rms`` also has no
equivalent in the map.  At m1-m4 there is no map at all.

(A third option -- measuring the footprint on ``nan_replaced_data - modsky``
rather than on the pristine-based residual -- would give bg_true - bg_model,
the background-model ERROR, which is arguably the more actionable diagnostic.
The pristine basis was chosen instead because it makes the column
stage-invariant, so an m4 and an m7 value are directly comparable.  The
model-error version is recoverable as ``modelsub_bkg - <map at source>``.)

Naming
------
The merged columns deliberately do NOT follow this file's usual
``std_{key}_avg`` / ``nmatch`` / ``{err}_prop`` conventions: they are
``mean_modelsub_bkg``, ``mean_modelsub_bkg_std``, ``mean_modelsub_bkg_err``,
``modelsub_bkg_rms_avg``, ``modelsub_bkg_nframes``, ``modelsub_bkg_npix_avg``.
The names describe the physical quantity (a mean over images of a mean over a
footprint) rather than the merge mechanics.  A script that looks up
``std_<col>_avg`` will not find the scatter for this column.

Scope: this describes the per-frame PSF-fitting path (``cataloging.py``, and
the identical setup in ``legacy/crowdsource_step.py`` and
``artificial_stars.py``).  The **saturated**-star channel in
``reduction/saturated_star_finding.py`` is different -- ``LocalBackground(25,
50)``, an adaptive annulus, or ``None`` with a median-filter background
subtracted beforehand for MIRI -- and is not covered by anything here.
Saturated stars appended by ``replace_saturated`` get ``modelsub_bkg* = NaN``, so
they sit outside both estimators.  So do **m8 forced-fill** sources: the columns
are attached in ``_save_manual_pass``, which m8 does not call, so an m8-filled
band carries whatever the m7 cross-band table had for it (masked, for the band
being filled).

``modelsub_bkg*`` is **diagnostic only** -- nothing here feeds back into the flux
fit.  Changing that would change every flux in the catalogue and is out of
scope for this module.

.. warning::

   **Neither estimator is a clean sky measurement for bright sources, and they
   fail differently.**  Measured on brick F182M nrca1 m7 against the m6
   smoothed-residual background map that was actually subtracted (so the
   expected residual is 0 for ``local_bkg`` and the map value for
   ``modelsub_bkg``).  Systematic = median deviation, noise = MAD:

   ====================  ======  ===================  ===================
   flux bin                   n  modelsub_bkg         local_bkg
                                 (systematic / MAD)   (systematic / MAD)
   ====================  ======  ===================  ===================
   0-25 %                  4972  +0.28 / 0.27         +0.31 / 0.22
   25-50 %                 4972  +0.38 / 0.30         +0.38 / 0.25
   50-75 %                 4972  +0.44 / 0.34         +0.46 / 0.30
   75-90 %                 2983  +0.53 / 0.44         +0.57 / 0.31
   90-99 %                 1790  +1.12 / 1.32         +1.05 / 0.59
   brightest 1 %            198  **+8.33 / 8.02**     **+3.40 / 0.90**
   ====================  ======  ===================  ===================

   Two things to read off this:

   * Up to the 99th percentile the two estimators are biased by *almost
     exactly the same amount* (+0.28/+0.31, +0.38/+0.38, ... +1.12/+1.05).
     A common bias in an annulus at 6-10 px and in a 3x3 box on the source is
     unlikely to be a property of either aperture -- it points at PSF wings /
     halo flux that the model does not carry.
   * In the brightest percentile they diverge and both fail.
     ``modelsub_bkg`` acquires a systematic of +8.3 with a comparable MAD of
     8.0 -- i.e. **both a bias and 30x the faint-source noise**, because the
     3x3 sits in the core where subtraction error is largest and varies source
     to source.  ``local_bkg`` acquires a smaller but far more *coherent*
     systematic, +3.4 with a MAD of only 0.90 -- the annulus still contains the
     star's wings, biasing it high consistently rather than noisily.

   So: bright-star contamination is **systematic in both**, and additionally
   **stochastic in modelsub_bkg**.  Use either as a background tracer only with
   a flux or ``qfit`` cut.  A large ``|modelsub_bkg|`` on a bright star is best
   read as a subtraction-quality flag.  Spearman r(modelsub_bkg, flux_fit) =
   0.32; r(modelsub_bkg_rms, flux_fit) = 0.64, so the combining weight is
   itself correlated with the value it weights.  Full range: -220 to +455.

Stage dependence: local_bkg vs modelsub_bkg
-------------------------------------------
``local_bkg`` measures whatever ``LocalBackground`` saw, and that changes with
the stage; ``modelsub_bkg`` is built from the pristine frame every time and does
not.  Measured on brick F182M nrca1 exposure 1 (median over sources):

    stage                m1     m2     m3     m4    | m5     m6     m7
    local_bkg           4.43   4.45   4.40   4.42   | 0.42   0.45   0.44
    (basis)             raw    raw    raw    raw    | resbgsub ...

``modelsub_bkg`` reads ~4.5 at every stage.  So at m1-m4, ``local_bkg`` ~
``modelsub_bkg``, both measuring the same sky; from m5 the smoothed-residual
background has been removed from the fitted array and ``local_bkg`` collapses
toward 0 while ``modelsub_bkg`` stays put.  That is the intended behaviour of
both, but it means **the same column name held two different physical
quantities across stages** with nothing recording which.  Hence
:func:`local_bkg_column_name`: every catalogue now also carries
``local_bkg_raw`` / ``local_bkg_bgsub`` / ``local_bkg_resbgsub`` (and the
``LBKGBASE`` header card), alongside the unchanged ``local_bkg``.

Per-frame vs merged
-------------------
:func:`measure_footprint_background` runs once per exposure and writes
``modelsub_bkg`` / ``modelsub_bkg_rms`` / ``modelsub_bkg_npix`` into that exposure's
catalogue.  :func:`combine_frames` then reduces the per-exposure values to the
merged-catalogue entry: a **sigma-clipped, inverse-variance weighted average**
with the across-frame scatter, the propagated error, and the number of frames
that survived clipping.

Caveats
-------
* A 3x3 footprint is 9 pixels.  The per-frame ``modelsub_bkg_rms`` is a noisy
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
  exposures, ``median(modelsub_bkg_std / modelsub_bkg_err) = 1.06`` for sources
  with >=3 frames, i.e. the propagated error is right to ~10%.
* See the bright-source warning above: this is not a pure sky measurement.
* The residual is built on the PRISTINE array, so at a saturated core the
  clipped data minus the full extrapolated satstar model is hugely negative.
  ``ctx.mask`` excludes SATURATED pixels, but with ``--saturation-data-floor``
  > 0 only ABOVE-floor saturated pixels are masked, so sub-floor saturated
  pixels can still enter a neighbour's footprint.  This is the likely origin of
  the -220 tail.
"""
import warnings

import numpy as np
# Imported at module level, not inside combine_frames: a lazy import charges
# astropy.stats to the first call's memory peak, which made the memory guard
# measure the import rather than the algorithm (3.16x cold vs 1.99x warm).
from astropy.stats import sigma_clip
from astropy.utils.exceptions import AstropyUserWarning

__all__ = ['FOOTPRINT_BOX', 'RESBKG_COLUMNS', 'MERGED_RESBKG_COLUMNS',
           'measure_footprint_background', 'combine_frames',
           'local_bkg_column_name']

#: Default footprint size in pixels (Jay Anderson's JWST1PASS convention).
FOOTPRINT_BOX = 3

#: Per-exposure column names written by :func:`measure_footprint_background`.
RESBKG_COLUMNS = ('modelsub_bkg', 'modelsub_bkg_rms', 'modelsub_bkg_npix')

#: Merged-catalogue column names written by :func:`combine_frames`.
MERGED_RESBKG_COLUMNS = ('mean_modelsub_bkg', 'mean_modelsub_bkg_std',
                         'mean_modelsub_bkg_err', 'modelsub_bkg_rms_avg',
                         'modelsub_bkg_nframes')


def local_bkg_column_name(bkg_basis):
    """Descriptive name for photutils' ``local_bkg`` given what it was run on.

    ``local_bkg`` is whatever ``LocalBackground`` measured on the array being
    fit, and that array differs by stage: raw at m1-m4, smoothed-residual
    background already removed at m5-m7.  Measured on brick F182M nrca1 exp1
    the median goes 4.43 / 4.45 / 4.40 / 4.42 (m1-m4) to 0.42 / 0.45 / 0.44
    (m5-m7) -- the same column name for two different physical quantities, with
    nothing in the catalog to tell them apart.

    ``'raw'`` -> ``local_bkg_raw``; ``'bgsub'`` -> ``local_bkg_bgsub``;
    ``'resbgsub'`` -> ``local_bkg_resbgsub``; ``'bgsub+resbgsub'`` ->
    ``local_bkg_bgsub_resbgsub``.  The plain ``local_bkg`` column is always
    kept as well, so existing consumers are unaffected.
    """
    return 'local_bkg_' + str(bkg_basis).replace('+', '_')


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
        ``modelsub_bkg_avg``  weighted, clipped mean background
        ``modelsub_bkg_std``  weighted scatter of the kept frames (the "RMS"
                             across exposures; NaN for a single frame, where
                             scatter is undefined rather than 0)
        ``modelsub_bkg_err``  propagated error, ``1/sqrt(sum(w))``
        ``modelsub_bkg_rms_avg``   mean of the per-frame pixel RMS (local noise level)
        ``resbkg_nframes``   number of frames surviving the cuts and clipping
    """

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

    # dtype-matched outputs: float64 here would both double these columns on
    # disk (they propagate per band through the crossband merge) and force an
    # upcast in the `dev` expression below, building two full (n_src, n_frames)
    # float64 temporaries at the largest allocation site.
    avg = np.full(mean.shape[0], np.nan, dtype=dtype)
    std = np.full(mean.shape[0], np.nan, dtype=dtype)
    err = np.full(mean.shape[0], np.nan, dtype=dtype)
    rms_avg = np.full(mean.shape[0], np.nan, dtype=dtype)
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

    # Weighted by npix ONLY -- deliberately NOT by w = npix/rms**2.  The stated
    # rationale (don't let an edge-clipped 2-pixel footprint count as much as a
    # full one) is satisfied by npix; the 1/rms**2 factor weights a quantity by
    # its own inverse square and drives the result toward the smallest value.
    # Measured on 8 real brick F182M nrca1 m7 exposures: npix/rms**2 gave a
    # median of 1.03 against 1.4038 for npix-only and 1.4042 for a plain mean,
    # i.e. the advertised "mean of the per-frame pixel RMS" was ~26% low.
    wn = np.where(keep, npix, 0).astype(dtype, copy=False)
    swn = wn.sum(axis=1)
    okn = swn > 0
    rms_avg[okn] = (wn[okn] * np.where(keep, rms, 0)[okn]).sum(axis=1) / swn[okn]
    del wn

    return {'mean_modelsub_bkg': avg,
            'mean_modelsub_bkg_std': std,
            'mean_modelsub_bkg_err': err,
            'modelsub_bkg_rms_avg': rms_avg,
            'modelsub_bkg_nframes': nframes}
