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

Standalone here for development/validation; the validated logic gets ported into
`jwst_gc_pipeline/reduction/saturated_star_finding.py`.
"""
import numpy as np
from scipy import ndimage
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


def _fof_merge(centers, link):
    """Friends-of-friends merge of (y,x) centres within `link` px → group means."""
    if len(centers) <= 1:
        return centers
    pts = np.asarray(centers, dtype=float)
    tree = cKDTree(pts)
    parent = list(range(len(pts)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in tree.query_pairs(link):
        parent[find(a)] = find(b)
    groups = {}
    for i in range(len(pts)):
        groups.setdefault(find(i), []).append(i)
    return [tuple(pts[idx].mean(axis=0)) for idx in groups.values()]


def deblend_blob_zeroframe(zeroframe, data, sources, label_id, sl, fwhm_pix,
                           daophot_xy=None, sat_ceiling=None,
                           peak_min_sep_frac=0.9, snap_frac=1.2,
                           prom_nsigma=4.0, dao_nsigma=2.5, seed_conc_min=2.5,
                           max_stars=6, pad=8, confirm_xy=None,
                           enable_secondary_peaks=True, verbose=False):
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
    """
    area = int((sources[sl] == label_id).sum())
    r_sat = max(1.0, np.sqrt(area / np.pi))
    y0 = max(0, sl[0].start - pad); y1 = min(data.shape[0], sl[0].stop + pad)
    x0 = max(0, sl[1].start - pad); x1 = min(data.shape[1], sl[1].stop + pad)
    zf = zeroframe[y0:y1, x0:x1].astype(float)
    blob = (sources[y0:y1, x0:x1] == label_id)
    blob_dil = ndimage.binary_dilation(blob, iterations=int(np.ceil(r_sat + 2 * fwhm_pix)))

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
                    core_centers.append((float(cyl + y0), float(cxl + x0)))
                    core_radii.append(float(np.sqrt(a / np.pi)))
                    claimed |= (lab_ci == kk)
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

    peak_centers = []
    if _HAVE_SKIMAGE:
        det_mask = blob_dil & np.isfinite(zf_sm)
        pk = peak_local_max(zf_sm, min_distance=min_sep, threshold_abs=thresh,
                            labels=det_mask.astype(int), num_peaks=max_stars)
        for (py, px) in pk:
            # ABSORB peaks inside a saturated core's claimed region: they are that
            # core star's own filled flux (and a big filled core can show >1 ring
            # maximum -> would split a single star, the L31 failure).  The core's
            # centroid already represents it.
            if claimed.any() and claimed[py, px]:
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
    if verbose:
        print(f"      n_core={len(core_centers)} core_radii={[round(r,1) for r in core_radii]} "
              f"n_peak_kept={len(peak_centers)} ceiling={sat_ceiling:.0f}", flush=True)
    if not centers:
        masked = np.where(blob_dil, zf_sm, -np.inf)
        py, px = np.unravel_index(np.argmax(masked), masked.shape)
        if np.isfinite(masked[py, px]) and masked[py, px] > thresh:
            centers = [(float(py + y0), float(px + x0))]

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

    # Final guard: merge any centres closer than the core radius (paranoia against
    # a residual double-detection of one core).
    centers = _fof_merge(centers, link=max(min_sep, 0.6 * r_sat))

    if not centers:
        # ultimate fallback: bbox centre (current production behaviour)
        cy = 0.5 * (sl[0].start + sl[0].stop - 1)
        cx = 0.5 * (sl[1].start + sl[1].stop - 1)
        centers = [(cy, cx)]

    if verbose:
        print(f"  L{label_id}: area={area} r_sat={r_sat:.1f} -> {len(centers)} star(s)",
              flush=True)
    return centers, dict(r_sat=r_sat, bg=bg, mad=mad, area=area,
                         crop=(y0, y1, x0, x1), zf_sm=zf_sm, blob_dil=blob_dil)
