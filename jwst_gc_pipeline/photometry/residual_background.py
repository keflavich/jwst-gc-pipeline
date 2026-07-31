r"""Per-source background from a small footprint on the star-subtracted residual.

Provides a per-source local background measured on the residual image after the
point-source model has been subtracted, following the convention used by Jay
Anderson's JWST1PASS: the mean and pixel-to-pixel RMS of a small (default
:math:`3\times3` px) footprint centred on each source.  It is written per
exposure and combined across exposures into the merged catalogue.  It is
strictly diagnostic; no value computed here feeds back into the flux fit.

It complements, and does not replace, the local background photutils already
subtracts during the fit (``local_bkg``).  The two are measured on different
images and answer different questions; see `Relationship to local_bkg`
below.

Notation
--------
For exposure :math:`i` and source :math:`s`:

:math:`D_i`
    the NaN-interpolated science frame, before any background subtraction
    (``ctx.original_data``, captured prior to both the ``Background2D`` and the
    smoothed-residual subtractions).
:math:`S_i`
    the fitted saturated-star model.
:math:`P_i`
    the fitted point-source (PSF) model, rendered with ``include_local_bkg=False``.
:math:`B_i`
    whichever background was subtracted from the array actually fitted at this
    stage: zero at m1-m4, the reprojected smoothed-residual mosaic at m5-m7, and
    the ``Background2D`` map when ``--bgsub`` is set.

Two distinct images follow:

.. math::

    A_i = D_i - S_i - B_i \qquad\text{(the array passed to the fitter)}

    R_i = D_i - S_i - P_i \qquad\text{(the residual measured here)}

:math:`A_i` retains the point sources and has the background removed;
:math:`R_i` removes the point sources and retains the background.  Because
:math:`R_i` is defined from :math:`D_i` rather than from the fitted array, it
carries no :math:`B_i` term and is therefore **independent of the pipeline
stage**: an m4 and an m7 value of the quantities below are directly comparable.

Per-exposure quantities
-----------------------
Let :math:`(\hat{x}_s, \hat{y}_s)` be the fitted position, :math:`b` the
(odd) box size and :math:`h=(b-1)/2`.  The footprint is the pixel set

.. math::

    \mathcal{F}_i(s) = \bigl\{(x,y) \;:\;
        |x - \operatorname{round}(\hat{x}_s)| \le h, \;
        |y - \operatorname{round}(\hat{y}_s)| \le h \bigr\} \cap \Omega_i,

with :math:`\Omega_i` the array bounds, and the valid subset excludes masked and
non-finite pixels:

.. math::

    \mathcal{V}_i(s) = \bigl\{p \in \mathcal{F}_i(s) \;:\;
        R_i(p) \ \text{finite} \ \wedge\ p \notin \mathcal{M}_i \bigr\},
    \qquad n_i(s) = \bigl|\mathcal{V}_i(s)\bigr|.

======================  ==========================================================
column                  definition
======================  ==========================================================
``modelsub_bkg``        :math:`b_i(s) = \dfrac{1}{n_i}\displaystyle\sum_{p \in \mathcal{V}_i(s)} R_i(p)`
``modelsub_bkg_rms``    :math:`\sigma_i(s) = \sqrt{\dfrac{1}{n_i-1}\displaystyle\sum_{p \in \mathcal{V}_i(s)} \bigl(R_i(p)-b_i\bigr)^2}`
``modelsub_bkg_npix``   :math:`n_i(s)`
======================  ==========================================================

No clipping is applied within the footprint: with nine pixels a clip is
unstable and biases the mean low.  :math:`b_i` and :math:`\sigma_i` are NaN when
:math:`n_i = 0` and :math:`n_i \le 1` respectively, so :math:`n_i` distinguishes
an edge-clipped footprint from one with no data at all.

Merged quantities
-----------------
Let :math:`w_i = n_i / \sigma_i^2`, the inverse variance of :math:`b_i` under
the assumption of independent pixels, and let :math:`\mathcal{K}(s)` be the set
of exposures surviving both the usability cut
(:math:`n_i \ge` ``min_npix``, :math:`\sigma_i > 0`, both values finite) and a
:math:`\kappa`-:math:`\sigma` clip of :math:`\{b_i\}` across exposures
(:math:`\kappa=3`, ``mad_std`` scale, median centre).  Write
:math:`W = \sum_{i \in \mathcal{K}} w_i` and :math:`N = |\mathcal{K}|`.

=========================  =====================================================
column                     definition
=========================  =====================================================
``mean_modelsub_bkg``      :math:`\bar{b} = W^{-1}\displaystyle\sum_{i \in \mathcal{K}} w_i b_i`
``mean_modelsub_bkg_std``  :math:`s_b = \sqrt{W^{-1}\displaystyle\sum_{i \in \mathcal{K}} w_i (b_i - \bar{b})^2}`
``mean_modelsub_bkg_err``  :math:`\epsilon_b = W^{-1/2}`
``modelsub_bkg_rms_avg``   :math:`\bar{\sigma} = \Bigl(\sum_{i \in \mathcal{K}} n_i\Bigr)^{-1}\displaystyle\sum_{i \in \mathcal{K}} n_i \sigma_i`
``modelsub_bkg_nframes``   :math:`N`
``modelsub_bkg_npix_avg``  :math:`\bar{n} = N^{-1}\displaystyle\sum_{i \in \mathcal{K}} n_i`
=========================  =====================================================

Three properties of these definitions are deliberate and easy to get wrong:

* :math:`\bar{\sigma}` is weighted by :math:`n_i` **alone**, not by :math:`w_i`.
  Weighting a quantity by its own inverse square drives the result toward the
  smallest value; measured on eight exposures, :math:`w_i`-weighting returned a
  median of 1.067 against 1.203 for both :math:`n_i`-weighting and an unweighted
  mean, i.e. 11% low.
* :math:`s_b` is the population (not sample) form, matching the ``std_*_avg``
  convention elsewhere in ``merge_catalogs``.  It is biased low by
  :math:`1-\sqrt{(N-1)/N}` — 29% at :math:`N=2`, 6.5% at :math:`N=8`.  It is
  NaN for :math:`N=1`, where scatter is undefined rather than zero.
* :math:`N` is stored as a float so that NaN survives ``replace_saturated``,
  which appends saturated-star rows with NaN in every column they lack.  NaN
  therefore means "never measured" and :math:`0` means "measured, no usable
  exposure".

Relationship to ``local_bkg``
-----------------------------
``PSFPhotometry`` is constructed with ``LocalBackground(r_{\rm in}, r_{\rm out})``,
:math:`r_{\rm in} = \max(6, \operatorname{round}(2.5\,\mathrm{FWHM}))`,
:math:`r_{\rm out} = r_{\rm in} + \max(4, \operatorname{round}(\mathrm{FWHM}))`
— 6 and 10 px for every NIRCam filter, larger for MIRI from F1000W up — using
photutils' default estimator, a :math:`3\sigma`-clipped ``MedianBackground``:

.. math::

    \ell_i(s) = \operatorname{median}_{3\sigma}
        \bigl\{ A_i(p) \;:\; r_{\rm in} \le |p - \hat{p}_s| < r_{\rm out} \bigr\}.

==========================  ========================  ==========================
                            ``local_bkg``             ``modelsub_bkg``
==========================  ========================  ==========================
image                       :math:`A_i`               :math:`R_i`
region                      annulus, 6-10 px          :math:`3\times3` px
statistic                   :math:`3\sigma`-clipped   mean, and pixel RMS
                            median
point sources present       yes                       no
background present          no (from m5)              yes
stage-dependent             yes                       no
subtracted by the fit       yes                       no
scatter reported            no                        yes
==========================  ========================  ==========================

Because :math:`A_i` contains :math:`-B_i`, ``local_bkg`` measures a different
physical quantity at different stages.  Median over sources, brick F182M nrca1
exposure 1:

=====================  ======  ======  ======  ======  ======  ======  ======
                       m1      m2      m3      m4      m5      m6      m7
=====================  ======  ======  ======  ======  ======  ======  ======
``local_bkg``          4.43    4.45    4.40    4.42    0.42    0.45    0.44
``modelsub_bkg``       4.26    4.04    4.12    4.07    4.28    4.27    4.51
=====================  ======  ======  ======  ======  ======  ======  ======

``local_bkg`` collapses toward zero once :math:`B_i \ne 0`; ``modelsub_bkg``
does not.  Since the column name alone does not record which regime a catalogue
belongs to, every catalogue additionally carries a basis-qualified alias —
``local_bkg_raw``, ``local_bkg_bgsub``, ``local_bkg_resbgsub`` or
``local_bkg_bgsub_resbgsub`` (see :func:`local_bkg_column_name`) — and an
``LBKGBASE`` header card.  The unqualified ``local_bkg`` column is unchanged.

Validation
----------
Against the smoothed-residual map that was subtracted at m5-m7, sampled at each
source (brick F182M nrca1 m7, :math:`n = 19\,865`; Spearman rank correlation):

=========================================  ======
quantity                                   :math:`\rho`
=========================================  ======
``modelsub_bkg``                           0.86
``local_bkg``                              0.32
=========================================  ======

with medians 4.51, 0.44 and 4.08 respectively.  Binning the same frame
:math:`8\times8` across the detector, the cell medians span 3.02-7.22 with a
cell-to-cell scatter of 0.822 against a standard error on a cell median of
0.074 — spatial structure at :math:`11\times` its own noise.

The improvement over ``local_bkg`` is attributable to the image, not to the
footprint.  Applying photutils' own ``LocalBackground(6, 10)`` to :math:`R_i`
instead of :math:`A_i` gives :math:`\rho = 0.89`, marginally better than the
:math:`3\times3`, with a 21% tighter scatter about the reference and far less
bright-source bias (4.99 against 14.24 at a reference value of 5.55 in the
brightest percentile).  The :math:`3\times3` footprint is retained because it is
Anderson's convention, because ``modelsub_bkg_rms`` is intended as a *local*
pixel-scatter estimate for which an annulus at 6-10 px is not a substitute, and
because a compact footprint remains local in a crowded field.  The reference is
itself a smoothed map, which favours larger apertures by construction; that
caveat does not apply to the bright-source comparison.  ``--residual-background-box``
changes the footprint size.

Limitations
-----------
**Bright sources.**  The footprint lies where the PSF model was subtracted, so
model error enters :math:`b_i` directly.  Deviations from the reference map, by
flux percentile (systematic = median deviation, noise = MAD):

====================  ======  ==================  ==================
flux percentile            n  ``modelsub_bkg``    ``local_bkg``
====================  ======  ==================  ==================
0-25                    4972  +0.28 / 0.27        +0.31 / 0.22
25-50                   4972  +0.38 / 0.30        +0.38 / 0.25
50-75                   4972  +0.44 / 0.34        +0.46 / 0.30
75-90                   2983  +0.53 / 0.44        +0.57 / 0.31
90-99                   1790  +1.12 / 1.32        +1.05 / 0.59
99-100                   198  +8.33 / 8.02        +3.40 / 0.90
====================  ======  ==================  ==================

Both estimators are biased, and by nearly the same amount below the 99th
percentile — a bias common to a 6-10 px annulus and a :math:`3\times3` box is
not a property of either aperture, and is consistent with PSF wing flux absent
from the model.  In the brightest percentile they diverge: ``modelsub_bkg``
acquires both a large systematic and thirty times the faint-source noise, since
the footprint sits in the core where subtraction error is largest and varies
between sources; ``local_bkg`` acquires a smaller but far more coherent bias
from the wings crossing the annulus.  Spearman
:math:`\rho(b, F_{\rm fit}) = 0.32` and
:math:`\rho(\sigma, F_{\rm fit}) = 0.64`, so :math:`w_i` is itself correlated
with the value it weights.  Use either column as a background tracer only with
a flux or ``qfit`` cut; read a large :math:`|b|` on a bright source as a
subtraction-quality indicator.  The full range of :math:`b` on this frame is
about :math:`-215` to :math:`+455`.

**Correlated pixels.**  :math:`w_i = n_i/\sigma_i^2` assumes independent
pixels.  The lag-1 autocorrelation of the high-passed residual is
:math:`\sim0.4`-:math:`0.6` (the exact value depends on the high-pass
definition), from destreaking, NaN interpolation, IPC and the model subtraction
— these are detector-space ``_crf`` frames, never resampled, so this is not
drizzle correlation.  Two errors partly cancel: correlation makes
:math:`\sigma_i` underestimate the true pixel scatter, while
:math:`n_{\rm eff} < n_i` makes the true variance of :math:`b_i` larger than
:math:`\sigma_i^2/n_i`.  Empirically
:math:`\operatorname{median}(s_b/\epsilon_b) = 1.06`-:math:`1.13` for sources
with :math:`N \ge 3`, so :math:`\epsilon_b` is accurate to roughly 10%.

**Precision of a single** :math:`\sigma_i`.  Nine pixels give a fractional
uncertainty of :math:`1/\sqrt{2(n_i-1)} \approx 25\%`.  ``modelsub_bkg_rms`` is
useful as a weight and as a relative map, not as a per-exposure noise value.

**Saturated cores.**  :math:`R_i` is built on :math:`D_i`, so at a saturated
core the clipped data minus the full extrapolated satstar model is strongly
negative.  The mask excludes SATURATED pixels, but with
``--saturation-data-floor`` above zero only above-floor saturated pixels are
masked, so sub-floor pixels may enter a neighbouring source's footprint.  This
is the likely origin of the negative tail.

Relation to the background map
------------------------------
At m5-m7 the map subtracted to form :math:`A_i` is available as
``ctx.background_map`` and could be sampled at each source directly.  That is a
different quantity: the map is a smoothed *model* derived from the previous
iteration's residual, whereas :math:`b_i` is a *measurement* on the current
residual at the source, and it exists at m1-m4 where no map does.  The map has
no analogue of :math:`\sigma_i`.  The difference
:math:`b_i - (\text{map at } s)` recovers the background-model error.

Scope
-----
The above describes the per-exposure PSF-fitting path in ``cataloging.py``, and
the identical estimator configuration in ``legacy/crowdsource_step.py`` and
``artificial_stars.py``.  The saturated-star channel in
``reduction/saturated_star_finding.py`` uses different settings
(``LocalBackground(25, 50)``, an adaptive annulus, or no estimator with a
median-filtered background removed beforehand for MIRI) and is not described
here.  Saturated stars appended by ``replace_saturated``, and sources filled by
the m8 forced cross-band fill, carry no ``modelsub_bkg*`` values: the columns
are written in ``_save_manual_pass``, which neither path calls.
"""
import warnings

import numpy as np
# Imported at module level, not inside combine_frames: a lazy import charges
# astropy.stats to the first call's memory peak, which made the memory guard
# measure the import rather than the algorithm (3.16x cold vs 1.99x warm).
from astropy.stats import sigma_clip
from astropy.utils.exceptions import AstropyUserWarning

__all__ = ['FOOTPRINT_BOX', 'RESBKG_COLUMNS', 'MERGED_RESBKG_COLUMNS',
           'MERGED_RESBKG_COLUMNS_COMBINE',
           'measure_footprint_background', 'combine_frames',
           'local_bkg_column_name']

#: Default footprint size in pixels (Jay Anderson's JWST1PASS convention).
FOOTPRINT_BOX = 3

#: Per-exposure column names.  :func:`measure_footprint_background` returns the
#: arrays; ``cataloging._attach_residual_background`` writes them under these
#: names.
RESBKG_COLUMNS = ('modelsub_bkg', 'modelsub_bkg_rms', 'modelsub_bkg_npix')

#: Merged-catalogue column names written by :func:`combine_frames`.
#: ``modelsub_bkg_npix_avg`` is added by merge_catalogs, not by combine_frames,
#: so it is listed here (this is the merged-catalogue contract) but is NOT in
#: combine_frames' return dict -- see MERGED_RESBKG_COLUMNS_COMBINE.
MERGED_RESBKG_COLUMNS = ('mean_modelsub_bkg', 'mean_modelsub_bkg_std',
                         'mean_modelsub_bkg_err', 'modelsub_bkg_rms_avg',
                         'modelsub_bkg_nframes', 'modelsub_bkg_npix_avg')

#: The subset :func:`combine_frames` itself returns.
MERGED_RESBKG_COLUMNS_COMBINE = MERGED_RESBKG_COLUMNS[:-1]


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
        ``mean_modelsub_bkg``      weighted, clipped mean background
        ``mean_modelsub_bkg_std``  weighted scatter of the kept frames (the
                                   "RMS" across exposures; NaN for a single
                                   frame, where scatter is undefined, not 0)
        ``mean_modelsub_bkg_err``  propagated error, ``1/sqrt(sum(w))``
        ``modelsub_bkg_rms_avg``   npix-weighted mean of the per-frame pixel
                                   RMS (the local noise level)
        ``modelsub_bkg_nframes``   frames surviving the cuts and clipping
                                   (float, so NaN survives replace_saturated)
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
    # float, NOT int: replace_saturated fills columns a satstar row lacks with
    # NaN and then add_row()s it, and `cannot convert float NaN to integer`
    # aborts the whole merge.  Float also separates "never measured" (NaN, e.g.
    # an appended satstar) from "measured, no usable frame" (0.0).
    nframes = keep.sum(axis=1).astype(dtype)

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
        # -- 29% at N=2, 6.5% at N=8 -- which is acceptable for a
        # spread indicator but means it is not an unbiased sigma estimate.
        std[multi] = np.sqrt((w[multi] * dev[multi] ** 2).sum(axis=1)
                             / sw[multi])
        del dev

    # Weighted by npix ONLY -- deliberately NOT by w = npix/rms**2.  The stated
    # rationale (don't let an edge-clipped 2-pixel footprint count as much as a
    # full one) is satisfied by npix; the 1/rms**2 factor weights a quantity by
    # its own inverse square and drives the result toward the smallest value.
    # Measured on 8 real brick F182M nrca1 m7 exposures, sky-matching sources
    # across them: npix/rms**2 gave a median of 1.067 against 1.203 for
    # npix-only and 1.203 for a plain mean -- the advertised "mean of the
    # per-frame pixel RMS" was ~11% low.  (A cruder row-index alignment put it
    # at ~26%; the sky-matched number is the one to quote.)
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
