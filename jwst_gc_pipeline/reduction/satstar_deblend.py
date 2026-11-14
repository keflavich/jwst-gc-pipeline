#!/usr/bin/env python
"""ZEROFRAME-primary deblender for merged saturated cores (gc2211).

Core idea: a saturated connected component in the cal/crf image can contain >1
star whose cores touched.  The ZEROFRAME (single first read in `_ramp.fits`)
saturates at far higher flux, so it usually shows the individual PSF peaks inside
the blob.  We detect those peaks (matched-filter / smoothed local maxima),
snap to the deduped daophot wing-detections for sub-pixel astrometry, merge
candidates closer than the saturated-core radius (kills single-star artifacts),
and return one centre per star.  Fallback: where the ZEROFRAME core is itself
saturated and yields no usable peak, return the blob bbox-centre (current
behaviour) so a star is never lost.

``build_deblended_source_records`` is the entry point used by
``get_saturated_stars``: it expands the one-seed-per-connected-component list into
one seed PER STAR.  Dev history + validation under ``scripts/satstar_deblend/``.
"""
import numpy as np
from scipy import ndimage
from scipy.optimize import minimize, nnls
from scipy.spatial import cKDTree

try:
    from skimage.feature import peak_local_max
    _HAVE_SKIMAGE = True
except Exception:
    _HAVE_SKIMAGE = False


def robust_zf_ceiling(zeroframe, phys_max=1e6):
    """Estimate the ZEROFRAME saturation pile-up value, or +inf if nothing
    saturates.  Saturated pixels concentrate at one DN ('flat top'); if the top
    0.01% is within 5% of the top 0.1% that flat top exists -> return it (×0.98).
    Otherwise the bright tail keeps rising (no saturation) -> +inf, so no pixel is
    flagged invalid (deep-well / shallow frames are handled by peak detection).
    Robust to a few corrupt hot pixels (the LW 9e8 outlier) via the percentile and
    the ``phys_max`` guard."""
    f = zeroframe[np.isfinite(zeroframe)]
    f = f[(f > 0) & (f < phys_max)]
    if f.size < 1000:
        return np.inf
    p999 = float(np.percentile(f, 99.9))
    p9999 = float(np.percentile(f, 99.99))
    if p999 > 0 and p9999 <= p999 * 1.05:
        return p999 * 0.98
    return np.inf


#: PSF-prediction veto (#1195).  Fewer fit pixels than this leave the amplitude
#: unanchored, so the veto falls back to the plain peak search.
PSF_VETO_MIN_FIT_PX = 30
#: Components above this saturated area, or models above this many stars, use
#: the plain peak search: the NNLS design matrix grows with both.
PSF_VETO_MAX_AREA = 5000
PSF_VETO_MAX_MODEL = 40
#: Fit pixels at or above this fraction of the ZEROFRAME ceiling are left out
#: (near-clip non-linearity).
PSF_VETO_CLIP_FRAC = 0.9
#: The model positions of the accepted stars are refined by up to this many
#: pixels (a saturated core's centroid is good to a few tenths of a pixel, which
#: leaves a dipole in the residual of a bright star), for at most this many stars.
PSF_VETO_REFINE_PX = 1.0
PSF_VETO_MAX_REFINE = 8


def _refine_centres(psf_stamp, centers, fit_y, fit_x, data, origin):
    """``centers`` (crop coordinates) moved, one star at a time, to the position
    within ``PSF_VETO_REFINE_PX`` that minimises the residual of the
    non-negative least-squares fit on the fit pixels."""
    y0, x0 = origin
    nfit = len(data)
    ones = np.ones(nfit)

    def stamp(cy, cx):
        img = np.asarray(psf_stamp(fit_y + y0, fit_x + x0, cy + y0, cx + x0),
                         dtype=float)
        return np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    cols = [stamp(cy, cx) for (cy, cx) in centers]

    def misfit(k, pos):
        if max(abs(pos[0] - centers[k][0]),
               abs(pos[1] - centers[k][1])) > PSF_VETO_REFINE_PX:
            return np.inf
        trial = cols[:k] + [stamp(pos[0], pos[1])] + cols[k + 1:]
        a = np.column_stack(trial + [ones, -ones])
        coef, rnorm = nnls(a, data)
        return rnorm

    out = list(centers)
    for k in range(len(out)):
        res = minimize(lambda p: misfit(k, p), out[k], method='Nelder-Mead',
                       options=dict(xatol=0.02, fatol=1e-6 * float(np.sum(data ** 2)),
                                    maxiter=60, initial_simplex=None))
        if np.all(np.isfinite(res.x)) and res.fun < misfit(k, out[k]):
            out[k] = (float(res.x[0]), float(res.x[1]))
            cols[k] = stamp(*out[k])
    return out


def _psf_veto_peaks(psf_stamp, psf_margin, zf, zf_filled, bg, invalid,
                    other, claimed, det_mask, fit, centers0, extra_centers,
                    thresh, sigma, min_sep, n_peaks, origin):
    """Secondary peaks that a PSF model of the accepted stars does not predict.

    The ring and spike beads of one bright star are local maxima of the
    smoothed ZEROFRAME, so a plain peak search reads them as companions.  Here
    the accepted stars (``centers0``, crop coordinates; ``extra_centers`` are
    modelled but never returned) are fitted with PSF stamps by non-negative
    least squares on ``fit`` pixels, plus a free constant.  A peak of the
    smoothed residual counts as a star when it exceeds both ``thresh`` and
    ``psf_margin`` times the smoothed model at that position (an allowance for
    the mismatch between the model PSF and the detector's).  The strongest such
    peak joins the model, the amplitudes are refitted, and the search repeats.

    Returns ``(peaks, n_vetoed)``: the accepted peaks (crop coordinates) and
    the number of residual peaks above ``thresh`` that the margin rejected in
    the last pass.  Returns a status string when the fit cannot run, so the
    caller keeps the plain peak search.
    """
    ny, nx = zf.shape
    Y, X = np.mgrid[0:ny, 0:nx]
    y0, x0 = origin
    data = (zf - bg)[fit]
    resid_search = det_mask & ~other & ~claimed

    def image(cy, cx):
        img = np.asarray(psf_stamp(Y + y0, X + x0, cy + y0, cx + x0), dtype=float)
        return np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    centers = list(centers0) + list(extra_centers)
    if len(centers) > PSF_VETO_MAX_MODEL:
        return 'fallback_model_size'
    if len(centers0) <= PSF_VETO_MAX_REFINE:
        try:
            centers = _refine_centres(psf_stamp, list(centers0), Y[fit], X[fit],
                                      data, origin) + list(extra_centers)
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            return 'fallback_fit_failed'
    imgs = [image(cy, cx) for (cy, cx) in centers]
    peaks = []
    n_vetoed = 0
    while True:
        cols = np.column_stack([im[fit] for im in imgs]
                               + [np.ones(int(fit.sum())), -np.ones(int(fit.sum()))])
        try:
            coef, _ = nnls(cols, data)
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            return 'fallback_fit_failed'
        amps = coef[:len(imgs)]
        const = coef[-2] - coef[-1]
        model = sum(a * im for a, im in zip(amps, imgs))
        resid = zf_filled - bg - const - model
        # inside the invalid core zf_filled is a nearest-ring copy, not data;
        # model minus that copy would make false peaks on its boundary
        resid = np.where(invalid | ~np.isfinite(resid), 0.0, resid)
        resid_sm = ndimage.gaussian_filter(resid, sigma)
        model_sm = ndimage.gaussian_filter(model, sigma)
        n_vetoed = 0
        best = None
        if len(peaks) < n_peaks and len(centers) < PSF_VETO_MAX_MODEL:
            pk = peak_local_max(resid_sm, min_distance=min_sep,
                                threshold_abs=thresh,
                                labels=resid_search.astype(int),
                                num_peaks=n_peaks)
            for (py, px) in pk:
                limit = max(thresh, psf_margin * max(model_sm[py, px], 0.0))
                if resid_sm[py, px] > limit:
                    if best is None or resid_sm[py, px] > resid_sm[best]:
                        best = (int(py), int(px))
                else:
                    n_vetoed += 1
        if best is None:
            return peaks, n_vetoed
        peaks.append((float(best[0]), float(best[1])))
        centers.append(best)
        imgs.append(image(*best))


def _is_compact(zf_sm, yy, xx, fwhm_pix, conc_min):
    """True if the smoothed-ZF peak at (yy,xx) is a compact point source (peak >=
    conc_min × local annulus median) rather than a smooth Airy-ring/spike ridge.
    Near an edge (cannot measure) -> True (keep)."""
    iy, ix = int(round(yy)), int(round(xx))
    ny, nx = zf_sm.shape
    r = max(3, int(round(2 * fwhm_pix)))
    if not (r < ix < nx - r and r < iy < ny - r):
        return True
    Y, X = np.mgrid[iy - r:iy + r + 1, ix - r:ix + r + 1]
    rad = np.hypot(X - ix, Y - iy)
    sub = zf_sm[iy - r:iy + r + 1, ix - r:ix + r + 1]
    peak = sub[r, r]
    ann = sub[(rad >= fwhm_pix) & (rad <= 2 * fwhm_pix)]
    if ann.size < 4 or not np.isfinite(peak):
        return True
    med = float(np.median(ann))
    return peak >= conc_min * abs(med) if med != 0 else True


def _fof_merge(centers, link, radii=None, radius_frac=0.6):
    """Friends-of-friends merge of (y,x) centres → group means.

    Two centres link when they are within ``link`` px, or, given ``radii``
    (the radius of the ZF-saturated core each centre was taken from, 0 for a
    peak), within ``radius_frac * max(r_i, r_j)`` px.
    """
    if len(centers) <= 1:
        return centers
    pts = np.asarray(centers, dtype=float)
    r = np.zeros(len(pts)) if radii is None else np.asarray(radii, dtype=float)
    tree = cKDTree(pts)
    parent = list(range(len(pts)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in tree.query_pairs(max(link, radius_frac * r.max())):
        if np.hypot(*(pts[a] - pts[b])) <= max(link, radius_frac * max(r[a], r[b])):
            parent[find(a)] = find(b)
    groups = {}
    for i in range(len(pts)):
        groups.setdefault(find(i), []).append(i)
    return [tuple(pts[idx].mean(axis=0)) for idx in groups.values()]


def _neighbour_claim(blob, other, fwhm_pix, pad):
    """Crop pixels that belong to saturated neighbours of ``blob``: their own
    pixels (``other``), plus every pixel nearer to them than to ``blob``
    (Euclidean; a tie stays with ``blob``) within ``reach`` of them in the
    taxicab metric of the dilation.

    ``reach`` = ceil(1 + 2 FWHM) is the smallest search radius of any
    component (r_sat >= 1), so the neighbour's own search covers the handed-
    over pixels.  It is capped at ``pad`` so that they also lie inside the
    neighbour's crop, which extends ``pad`` px beyond its bounding box.
    """
    if not other.any():
        return other
    reach = min(int(np.ceil(1.0 + 2 * fwhm_pix)), int(pad))
    d_own = ndimage.distance_transform_edt(~blob)
    d_other = ndimage.distance_transform_edt(~other)
    l1_other = ndimage.distance_transform_cdt(~other, metric='taxicab')
    return other | ((d_other < d_own) & (l1_other <= reach))


def deblend_blob_zeroframe(zeroframe, data, sources, label_id, sl, fwhm_pix,
                           daophot_xy=None, sat_ceiling=None,
                           peak_min_sep_frac=0.9, snap_frac=1.2,
                           prom_nsigma=4.0, dao_nsigma=2.5, seed_conc_min=2.5,
                           max_stars=6, pad=8, confirm_xy=None,
                           enable_secondary_peaks=True, verbose=False,
                           area_per_peak=50.0, psf_stamp=None, psf_margin=0.25):
    """Return a list of (y, x) star centres inside one saturated blob.

    Parameters
    ----------
    zeroframe, data : 2-D arrays
        ZEROFRAME (raw DN) and the cal/crf SCI image, full frame.
    sources : 2-D int array
        Labelled saturated-component image (scipy.ndimage.label output).
    label_id : int
        Which component to deblend.
    sl : tuple of slice
        The component's bounding-box slice (from find_objects).
    fwhm_pix : float
        PSF FWHM in pixels (for the matched-filter smoothing + min separation).
    daophot_xy : (N,2) array or None
        Deduped daophot is_saturated positions (full-frame x,y) to snap to.
    sat_ceiling : float or None
        ZEROFRAME saturation value; pixels >= this (or ==0) are treated as
        invalid core and filled before detection.  If None, inferred.
    max_stars : int
        Minimum number of ZEROFRAME peaks passed to ``peak_local_max``.
    area_per_peak : float or None
        The peak limit grows to one per ``area_per_peak`` saturated px of the
        component when that is larger than ``max_stars``.  ``None`` or 0 keeps
        the fixed ``max_stars`` limit.
    psf_stamp : callable or None
        ``psf_stamp(yy, xx, yc, xc)`` returns the unit-flux PSF of a star at
        detector position (``yc``, ``xc``) on the detector-pixel arrays
        ``yy``, ``xx``.  Given one, secondary peaks come from a residual
        search against a PSF model of the accepted stars (see
        ``_psf_veto_peaks``), which drops the ring and spike beads of a single
        bright star (#1195).  ``None`` keeps the plain peak search.
    psf_margin : float
        Fraction of the model at a residual peak that the peak must exceed; 0
        or less also keeps the plain peak search.
    """
    area = int((sources[sl] == label_id).sum())
    r_sat = max(1.0, np.sqrt(area / np.pi))
    y0 = max(0, sl[0].start - pad); y1 = min(data.shape[0], sl[0].stop + pad)
    x0 = max(0, sl[1].start - pad); x1 = min(data.shape[1], sl[1].stop + pad)
    zf = zeroframe[y0:y1, x0:x1].astype(float)
    blob = (sources[y0:y1, x0:x1] == label_id)
    blob_dil = ndimage.binary_dilation(blob, iterations=int(np.ceil(r_sat + 2 * fwhm_pix)))
    # Pixels of OTHER saturated components in the crop.  The dilation (r_sat +
    # 2 FWHM) reaches neighbouring components a few px away, and each of those
    # seeds its own star(s); a core or peak centred there is the neighbour's star.
    # Seeding it here as well fits that star twice: iterative subtraction splits
    # its flux between two rows at the same position (wd2 nrcb3: 42 such pairs).
    # Detection still runs on the full ``blob_dil`` and the foreign centres are
    # dropped afterwards; masking the map instead creates false maxima on the
    # mask edge, where the neighbour's wing rises toward its core.
    other = (sources[y0:y1, x0:x1] > 0) & ~blob
    # A centre off every SATURATED pixel but nearer to a neighbour's than to
    # this component's belongs to the neighbour as well.  On wd2 F150W such
    # peaks sat between a large component and a small one, 2.8-4.3 px from
    # the small one's star; the unlocked fit drifted onto that star and split
    # its flux between two rows (5 pairs on nrcb3, 2 on nrcb1).
    other = _neighbour_claim(blob, other, fwhm_pix, pad)

    if sat_ceiling is None:
        sat_ceiling = robust_zf_ceiling(zeroframe)
    invalid = (~np.isfinite(zf)) | (zf == 0) | (zf >= sat_ceiling)

    # local background from the crop edge (outside the dilated blob)
    edge = (~blob_dil) & (~invalid)
    bg = np.median(zf[edge]) if edge.sum() > 20 else np.nanmedian(zf[~invalid])
    mad = 1.4826 * np.median(np.abs(zf[edge] - bg)) if edge.sum() > 20 else \
        1.4826 * np.median(np.abs(zf[~invalid] - np.nanmedian(zf[~invalid])))
    mad = max(mad, 1e-6)

    sigma = max(0.6, fwhm_pix / 2.355)
    thresh = prom_nsigma * mad
    min_sep = max(2, int(round(peak_min_sep_frac * fwhm_pix)))

    # PRIMARY -- ZF-SATURATED CORES.  A star whose ZF core ALSO saturates is a solid
    # hole of invalid pixels.  Close + fill that mask so one core = one connected
    # cluster (raw invalid pixels speckle and fragment), label, take each cluster's
    # CENTROID.  One centre per ZF-saturated star, robust to ring/spike asymmetry
    # that would split a single big core if detected on the filled ring.  Two
    # touching cores stay separate while an unsaturated gap divides them (L168).
    # The union of clusters is the CLAIMED region: any ZF peak falling inside it is
    # this core star's own (filled) flux, not a new star.
    inv_blob = invalid & blob_dil
    inv_blob = ndimage.binary_fill_holes(
        ndimage.binary_closing(inv_blob, iterations=1))
    core_centers, core_radii = [], []
    claimed = np.zeros_like(inv_blob)
    if inv_blob.any():
        lab_ci, n_ci = ndimage.label(inv_blob)
        if n_ci > 0:
            cen = ndimage.center_of_mass(inv_blob, lab_ci, np.arange(n_ci) + 1)
            areas = ndimage.sum_labels(inv_blob, lab_ci, np.arange(n_ci) + 1)
            for kk, ((cyl, cxl), a) in enumerate(zip(cen, areas), start=1):
                if a >= 3:
                    # a neighbour's core still claims its region (its ring
                    # maxima are not new stars) but seeds no centre here
                    claimed |= (lab_ci == kk)
                    if other[int(round(cyl)), int(round(cxl))]:
                        continue
                    core_centers.append((float(cyl + y0), float(cxl + x0)))
                    core_radii.append(float(np.sqrt(a / np.pi)))
        claimed = ndimage.binary_dilation(claimed, iterations=min_sep)

    # UNIFIED PEAK DETECTION (handles BOTH regimes).  Nearest-valid fill of the
    # invalid holes fills each saturated core from its OWN ring (no bridging), so a
    # saturated core becomes a peak too; where nothing saturates (deep ZF wells,
    # e.g. dithered/shallow pointings) the ZF is already an unsaturated image of the
    # stars.  Smoothed local maxima then find every core in either regime.
    zf_clean = np.where(np.isfinite(zf), zf, np.nan)
    if invalid.any() and (~invalid).any():
        fi = ndimage.distance_transform_edt(invalid, return_distances=False,
                                            return_indices=True)
        zf_filled = zf_clean[tuple(fi)]
    else:
        zf_filled = np.where(invalid, bg, zf_clean)
    zf_filled = np.where(np.isfinite(zf_filled), zf_filled, bg)
    zf_sm = ndimage.gaussian_filter(zf_filled - bg, sigma)

    # The peak limit applies BEFORE the claimed-region filter below, and the
    # brightest maxima of a blend are its filled saturated cores.  A fixed limit
    # of 6 therefore leaves no secondary peak in a component holding 6 or more
    # cores (wd2 F277W nrcblong: the 34,700 px cluster-core component has 36 ZF
    # cores and ~150 dolphot stars).  Scale the limit with the saturated area.
    n_peaks = int(max_stars)
    if area_per_peak:
        n_peaks = max(n_peaks, int(area / float(area_per_peak)))
    peak_centers = []
    psf_veto = 'off'
    n_vetoed = 0
    det_mask = blob_dil & np.isfinite(zf_sm)
    if _HAVE_SKIMAGE and psf_stamp is not None and psf_margin > 0:
        psf_veto = 'fallback_area' if area > PSF_VETO_MAX_AREA else 'applied'
    if psf_veto == 'applied':
        # Peaks of the smoothed ZEROFRAME that the PSF model of the accepted
        # stars predicts (rings, spikes) are not stars.  Without a saturated
        # core the brightest peak is the blob's own star and seeds the model.
        init = [(cy - y0, cx - x0) for (cy, cx) in core_centers]
        first = []
        if not init:
            pk0 = peak_local_max(zf_sm, min_distance=min_sep, threshold_abs=thresh,
                                 labels=(det_mask & ~other).astype(int), num_peaks=1)
            first = [(float(py), float(px)) for (py, px) in pk0]
            init = first
        neighbours = []
        if other.any():
            lab_o, n_o = ndimage.label((sources[y0:y1, x0:x1] > 0) & ~blob)
            if n_o:
                neighbours = [tuple(c) for c in ndimage.center_of_mass(
                    lab_o > 0, lab_o, np.arange(n_o) + 1)]
        fit = (blob_dil & np.isfinite(zf) & ~invalid & ~other
               & (zf < PSF_VETO_CLIP_FRAC * sat_ceiling))
        if not init:
            psf_veto = 'fallback_no_peak'
        elif int(fit.sum()) < PSF_VETO_MIN_FIT_PX:
            psf_veto = 'fallback_few_px'
        else:
            res = _psf_veto_peaks(
                psf_stamp, psf_margin, zf, zf_filled, bg, invalid, other,
                claimed, det_mask, fit, init, neighbours, thresh, sigma,
                min_sep, n_peaks, (y0, x0))
            if isinstance(res, str):
                psf_veto = res
            else:
                peaks, n_vetoed = res
                for (py, px) in first + peaks:
                    peak_centers.append((py + y0, px + x0))
    if _HAVE_SKIMAGE and psf_veto not in ('applied',):
        pk = peak_local_max(zf_sm, min_distance=min_sep, threshold_abs=thresh,
                            labels=det_mask.astype(int), num_peaks=n_peaks)
        for (py, px) in pk:
            # ABSORB peaks inside a saturated core's claimed region: they are that
            # core star's own filled flux (and a big filled core can show >1 ring
            # maximum -> would split a single star, the L31 failure).  The core's
            # centroid already represents it.
            if claimed.any() and claimed[py, px]:
                continue
            if other[py, px]:
                continue
            peak_centers.append((float(py + y0), float(px + x0)))

    # SECONDARY PEAK CONFIRMATION.  An OUTSIDE-claimed peak becomes a star only if a
    # cataloged source (deduped daophot, passed quality cuts) lies within
    # snap_frac*FWHM, OR -- when no catalog is supplied -- it is COMPACT (peak well
    # above its local annulus, i.e. a point source, not a smooth Airy-ring/spike
    # ridge of a bright neighbour).  This is what separates a real unsaturated
    # companion from PSF structure.  Saturated-core centres are exempt (a hole IS a
    # star).  When ``enable_secondary_peaks`` is False only the core centres + a
    # single fallback peak survive (conservative).
    if peak_centers:
        conf = confirm_xy if confirm_xy is not None else daophot_xy
        cxy = np.asarray(conf, dtype=float) if (conf is not None and len(conf)) else None
        kept = []
        for (py, px) in peak_centers:
            ok = False
            if cxy is not None:
                d = np.hypot(cxy[:, 1] - py, cxy[:, 0] - px)
                ok = bool(d.min() <= snap_frac * fwhm_pix)
            if (not ok) and (cxy is None):
                ok = _is_compact(zf_sm, py - y0, px - x0, fwhm_pix, seed_conc_min)
            # a peak with NO saturated cores in the blob (regime 2) is the blob's
            # own star: keep the brightest such peak even if unconfirmed so a
            # genuine merged double in a deep-well frame is never lost.
            kept.append((py, px, ok))
        if not core_centers and kept:
            # ensure at least the brightest peak survives in regime 2
            br = max(range(len(kept)),
                     key=lambda i: zf_sm[int(kept[i][0]-y0), int(kept[i][1]-x0)])
            kept[br] = (kept[br][0], kept[br][1], True)
        peak_centers = [(py, px) for (py, px, ok) in kept
                        if (ok and enable_secondary_peaks)]

    centers = core_centers + peak_centers
    center_r = core_radii + [0.0] * len(peak_centers)
    if verbose:
        print(f"      n_core={len(core_centers)} core_radii={[round(r,1) for r in core_radii]} "
              f"n_peak_kept={len(peak_centers)} ceiling={sat_ceiling:.0f}", flush=True)
    if not centers:
        masked = np.where(blob_dil, zf_sm, -np.inf)
        py, px = np.unravel_index(np.argmax(masked), masked.shape)
        if (np.isfinite(masked[py, px]) and masked[py, px] > thresh
                and not other[py, px]):
            centers = [(float(py + y0), float(px + x0))]
            center_r = [0.0]

    # DAOPHOT only for ASTROMETRY: snap each ZF peak to the nearest deduped daophot
    # is_saturated position within snap_frac*FWHM (sub-pixel JWST centroid).  We do
    # NOT add daophot points as new stars -- a daophot detection sitting on a bright
    # star's diffraction spike (no independent ZF core) would spawn a false
    # companion (the L31 failure).  A genuine faint companion shows up as its own
    # ZF peak above thresh and is already counted.
    if daophot_xy is not None and len(daophot_xy) and centers:
        dxy = np.asarray(daophot_xy, dtype=float)
        in_box = ((dxy[:, 0] >= x0 - 2) & (dxy[:, 0] < x1 + 2) &
                  (dxy[:, 1] >= y0 - 2) & (dxy[:, 1] < y1 + 2))
        dnear = dxy[in_box]
        used = np.zeros(len(dnear), dtype=bool)
        snapped = []
        for (cy, cx) in centers:
            if len(dnear):
                d = np.hypot(dnear[:, 0] - cx, dnear[:, 1] - cy)
                j = int(np.argmin(d))
                if d[j] <= snap_frac * fwhm_pix and not used[j]:
                    used[j] = True
                    snapped.append((float(dnear[j, 1]), float(dnear[j, 0])))
                    if verbose:
                        print(f"      snap ({cx:.1f},{cy:.1f})->dao "
                              f"({dnear[j,0]:.1f},{dnear[j,1]:.1f}) d={d[j]:.1f}", flush=True)
                    continue
            snapped.append((cy, cx))
        centers = snapped

    # Final guard: merge centres closer than min_sep, or closer than 0.6 r_core
    # where r_core is the radius of the ZF-saturated core either centre was
    # taken from (paranoia against a residual double-detection of one core).
    # The radius is per pair.  One radius for the whole component (its largest
    # core's) merged distinct peaks far from that core: wd2 F277W nrcblong, a
    # 5.7 px link from a 9.6 px core joined 13 pairs, 8 of them two dolphot
    # stars.  r_sat of the whole component is worse: for a blend of many stars
    # it measures the blend, and friends-of-friends at 0.6 r_sat chains every
    # centre into one (wd2 F277W cluster core: r_sat 105 px, link 63 px, 42
    # centres -> 1 seed).  Peaks inside a core's claimed region are already
    # absorbed above, and unconfirmed secondary peaks must pass the
    # compactness test.
    centers = _fof_merge(centers, link=min_sep, radii=center_r)

    if not centers:
        # ultimate fallback: bbox centre (current production behaviour)
        cy = 0.5 * (sl[0].start + sl[0].stop - 1)
        cx = 0.5 * (sl[1].start + sl[1].stop - 1)
        centers = [(cy, cx)]

    if verbose:
        print(f"  L{label_id}: area={area} r_sat={r_sat:.1f} -> {len(centers)} star(s)",
              flush=True)
    return centers, dict(r_sat=r_sat, bg=bg, mad=mad, area=area,
                         crop=(y0, y1, x0, x1), zf_sm=zf_sm, blob_dil=blob_dil,
                         psf_veto=psf_veto, n_vetoed=n_vetoed)


def build_deblended_source_records(saturated, sources, coms, sizes_by_label,
                                   zeroframe, data, fwhm_pix,
                                   daophot_xy=None, confirm_xy=None,
                                   min_area=4, stats=None, **deblend_kwargs):
    """Expand per-connected-component satstar seeds into per-STAR seeds.

    Drop-in replacement for the one-record-per-``coms``-entry loop in
    ``get_saturated_stars``.  Each returned record has the SAME schema the caller
    already consumes::

        {'com': (cy, cx), 'label': component_label, 'forced': False,
         'sat_area': component_saturated_area}

    A component that is too small (``< min_area`` saturated pixels), has a
    non-finite centroid, or deblends to a single star yields exactly ONE record at
    its (refined) centroid -- identical to the previous behaviour.  A merged
    component yields one record PER resolved star, ALL carrying the same component
    label + saturated area, so the existing per-source mask / iterative-subtraction
    logic in ``get_saturated_stars`` masks the shared saturated blob and subtracts
    the brighter member before fitting the fainter one.

    Parameters mirror ``deblend_blob_zeroframe``.  ``zeroframe`` is the ZEROFRAME
    (frame zero) array from the matching ``_ramp.fits`` (raw DN).

    ``stats``, when a dict, is filled with ``n_failed`` (components whose
    deblend raised and kept the refined centroid, with the first error in
    ``first_error``) and ``psf_veto`` (components per PSF-veto status).
    """
    slices = ndimage.find_objects(sources)
    ceiling = robust_zf_ceiling(zeroframe)
    records = []
    for ii, com in enumerate(coms):
        label_id = ii + 1
        area = int(sizes_by_label[ii]) if ii < len(sizes_by_label) else 0
        sl = slices[ii] if ii < len(slices) else None
        finite_com = (com is not None and np.isfinite(com[0]) and np.isfinite(com[1]))
        if sl is None or area < min_area or not finite_com:
            records.append({'com': com, 'label': label_id, 'forced': False,
                            'sat_area': area})
            continue
        try:
            centers, info = deblend_blob_zeroframe(
                zeroframe, data, sources, label_id, sl, fwhm_pix,
                daophot_xy=daophot_xy, confirm_xy=confirm_xy,
                sat_ceiling=ceiling, **deblend_kwargs)
            if stats is not None:
                counts = stats.setdefault('psf_veto', {})
                counts[info['psf_veto']] = counts.get(info['psf_veto'], 0) + 1
        except (ValueError, np.linalg.LinAlgError, IndexError) as exc:
            # the component keeps its refined centroid, as before, but the
            # failure is counted and printed instead of passing silently
            centers = None
            print(f"satstar deblend: component {label_id} failed "
                  f"({type(exc).__name__}: {exc}); keeping its centroid",
                  flush=True)
            if stats is not None:
                stats['n_failed'] = stats.get('n_failed', 0) + 1
                stats.setdefault('first_error', f"{type(exc).__name__}: {exc}")
        if not centers:
            centers = [com]
        for c in centers:
            records.append({'com': (float(c[0]), float(c[1])), 'label': label_id,
                            'forced': False, 'sat_area': area})
    return records
