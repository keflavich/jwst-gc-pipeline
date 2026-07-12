"""Incremental refit — decide which per-source fits can be REUSED unchanged from
the previous manual phase instead of re-fitting.

See ``INCREMENTAL_REFIT_PLAN.md``. Core invariant (for the default ``group=False``
independent single-source fit): a source's fit is a deterministic function of its
seed position, the (fixed) PSF, and the image data ``raw - bg`` within its
footprint (fit window + localbkg annulus, radius ``localbkg_outer``). ``raw`` is
constant across phases, so the fit is unchanged iff the background is unchanged
over that footprint. A new nearby seed does NOT change an incumbent's independent
fit (the neighbour flux is in the data window either way, never modelled), so
there is no group-membership condition when ``group=False``.

This module is PURE (arrays in, decisions out) so the reuse logic is unit-tested
in isolation from the fit machinery and the SLURM pipeline.
"""

import numpy as np

__all__ = [
    "dirty_bg_mask",
    "match_prev_seeds",
    "classify_reusable_seeds",
    "splice_reused_rows",
]


def dirty_bg_mask(bg_cur, bg_prev, bg_delta_thresh, dilate_pix):
    """Boolean mask of pixels whose background changed between phases by more than
    ``bg_delta_thresh``, DILATED by ``dilate_pix`` (a bg change at pixel p taints
    any source whose footprint reaches p, so grow the mask by the footprint
    radius).  Non-finite bg in either phase is treated as dirty (fail-safe).
    """
    bg_cur = np.asarray(bg_cur, dtype=float)
    bg_prev = np.asarray(bg_prev, dtype=float)
    if bg_cur.shape != bg_prev.shape:
        raise ValueError(f"bg shape mismatch {bg_cur.shape} vs {bg_prev.shape}")
    finite = np.isfinite(bg_cur) & np.isfinite(bg_prev)
    dbg = np.where(finite, np.abs(bg_cur - bg_prev), np.inf)
    dirty = (dbg > bg_delta_thresh) | ~finite
    d = int(round(dilate_pix))
    if d > 0 and dirty.any():
        from scipy.ndimage import binary_dilation
        # a disk-ish structuring element of radius d
        yy, xx = np.ogrid[-d:d + 1, -d:d + 1]
        struct = (xx * xx + yy * yy) <= d * d
        dirty = binary_dilation(dirty, structure=struct)
    return dirty


def match_prev_seeds(seed_xy, prev_xy, match_tol_pix):
    """For each current seed, the index of the coincident previous fitted source
    (within ``match_tol_pix``), or -1 if none.  Nearest match; ties broken by
    distance.  ``seed_xy`` / ``prev_xy`` are (N,2) / (M,2) arrays of (x, y) pix.

    Intended tolerance is TINY (<=0.01 px): a reusable seed is one CARRIED FORWARD
    at the same position, not one re-detected at a slightly different centroid (a
    moved centroid means the input changed -> refit).
    """
    seed_xy = np.asarray(seed_xy, dtype=float).reshape(-1, 2)
    prev_xy = np.asarray(prev_xy, dtype=float).reshape(-1, 2)
    out = np.full(len(seed_xy), -1, dtype=int)
    if len(seed_xy) == 0 or len(prev_xy) == 0:
        return out
    # small N per frame; a direct nearest search is fine and dependency-free
    from scipy.spatial import cKDTree
    tree = cKDTree(prev_xy)
    dist, idx = tree.query(seed_xy, k=1, distance_upper_bound=match_tol_pix)
    ok = np.isfinite(dist)
    out[ok] = idx[ok]
    return out


def classify_reusable_seeds(seed_xy, prev_xy, bg_cur, bg_prev, *,
                            footprint_radius_pix, bg_delta_thresh,
                            match_tol_pix=0.01, valid_mask=None):
    """Boolean array over ``seed_xy``: True where the phase-N fit can REUSE the
    phase-(N-1) fitted row for that source.

    Reusable iff ALL of:
      * the seed matches a previous fitted source within ``match_tol_pix``;
      * the seed's footprint disk (radius ``footprint_radius_pix``) touches NO
        pixel whose background changed by > ``bg_delta_thresh`` (i.e. the data
        ``raw - bg`` over the footprint is unchanged);
      * the footprint lies on valid data (if ``valid_mask`` given, all-valid).

    Returns ``(reusable, prev_index)``: ``prev_index[i]`` is the matched previous
    row (or -1); ``reusable[i]`` gates whether to copy it.
    """
    seed_xy = np.asarray(seed_xy, dtype=float).reshape(-1, 2)
    prev_index = match_prev_seeds(seed_xy, prev_xy, match_tol_pix)
    reusable = prev_index >= 0
    if not reusable.any():
        return reusable, prev_index

    dirty = dirty_bg_mask(bg_cur, bg_prev, bg_delta_thresh,
                          dilate_pix=footprint_radius_pix)
    ny, nx = dirty.shape
    r = int(np.ceil(footprint_radius_pix))

    for i in np.nonzero(reusable)[0]:
        xc, yc = seed_xy[i]
        xi, yi = int(round(xc)), int(round(yc))
        x0, x1 = max(0, xi - r), min(nx, xi + r + 1)
        y0, y1 = max(0, yi - r), min(ny, yi + r + 1)
        if x0 >= x1 or y0 >= y1:
            reusable[i] = False       # footprint entirely off-frame -> refit
            continue
        # footprint disk within the bbox
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = (xx - xc) ** 2 + (yy - yc) ** 2 <= footprint_radius_pix ** 2
        # dirty already dilated by the footprint radius, so testing the seed's own
        # disk is belt-and-suspenders; keep it explicit for clarity.
        if dirty[y0:y1, x0:x1][disk].any():
            reusable[i] = False
            continue
        if valid_mask is not None:
            vm = np.asarray(valid_mask)
            if not vm[y0:y1, x0:x1][disk].all():
                reusable[i] = False
    return reusable, prev_index


def splice_reused_rows(prev_table, fit_table, reusable, prev_index, seed_order,
                       phase_label=None):
    """Assemble the phase-N per-frame catalog from REUSED previous rows + freshly
    FIT rows, preserving the seed order.

    Parameters
    ----------
    prev_table : Table
        Phase-(N-1) per-frame catalog (rows indexed by ``prev_index``).
    fit_table : Table
        The freshly-fit rows for the non-reusable seeds, in the order of
        ``seed_order[~reusable]``.
    reusable, prev_index : arrays from :func:`classify_reusable_seeds`, length =
        number of seeds (in seed order).
    seed_order : array
        Row order to emit (identity ``range(n_seed)`` unless the caller reordered).
    phase_label : optional
        If given and the table has an ``iter_detected`` / phase column, the reused
        rows keep their ORIGINAL detection provenance (they were not re-detected);
        this hook exists for callers that stamp a phase column.

    Returns
    -------
    Table with one row per seed, in ``seed_order``.
    """
    from astropy.table import Table, vstack

    n = len(reusable)
    reused_rows = prev_table[prev_index[reusable]] if reusable.any() else prev_table[:0]
    # map each seed position -> its source row
    out_rows = [None] * n
    ri = 0
    fi = 0
    for k in range(n):
        if reusable[k]:
            out_rows[k] = reused_rows[ri]
            ri += 1
        else:
            out_rows[k] = fit_table[fi]
            fi += 1
    # Build via row indices to avoid per-row Table overhead where possible.
    # (Columns must be compatible; the caller guarantees both come from the same
    # PSFPhotometry schema.)
    if n == 0:
        return fit_table[:0]
    reused_part = reused_rows
    fit_part = fit_table
    combined = vstack([t for t in (reused_part, fit_part) if len(t) > 0],
                      metadata_conflicts="silent")
    # reorder to seed order
    order = np.empty(n, dtype=int)
    ri = 0
    fi = len(reused_part)
    for k in range(n):
        if reusable[k]:
            order[k] = ri
            ri += 1
        else:
            order[k] = fi
            fi += 1
    return combined[order]
