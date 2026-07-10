"""Sanctioned astrometric-offset measurement — offset-histogram stacking.

This is the ONE public, guarded entry point for measuring the bulk (and per-tile)
on-sky offset between two source lists.  It is **density-immune**: it histograms
ALL pairwise offsets within a window and takes the peak, so it stays correct no
matter how large the shift.  This is the method that must be used instead of the
FORBIDDEN dense-nearest-neighbour-median (`match_to_catalog_sky(...).median()`),
which collapses toward ~0 in a crowded field and has repeatedly corrupted the
brick-1182 / prop-2221 astrometry.  See CLAUDE.md and
``reduction/ASTROMETRY_WCS_CORRECTION_FLOW.md``.

Do NOT write ad-hoc NN-median matching.  Call ``measure_offset`` here.
"""

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky


# Contrast (peak / median of the pair-offset histogram) below this = NO coherent
# tie (scattered pairs), i.e. the two frames are NOT registered at this scale.
DEFAULT_MIN_CONTRAST = 5.0


class NoCoherentTieError(RuntimeError):
    """Raised when the offset-histogram peak contrast is too low to trust — the two
    source lists have no coherent astrometric tie (broken/untied WCS)."""


def _hist_peak(dra_arcsec, ddec_arcsec, maxsep_arcsec, bin_arcsec):
    """2-D histogram peak of a cloud of pair offsets.  Returns
    (dra_mas, ddec_mas, off_mas, npairs, contrast)."""
    n = len(dra_arcsec)
    m = maxsep_arcsec
    bins = np.arange(-m, m + bin_arcsec, bin_arcsec)
    H, xe, ye = np.histogram2d(dra_arcsec, ddec_arcsec, bins=[bins, bins])
    i, j = np.unravel_index(H.argmax(), H.shape)
    bg = float(np.median(H[H > 0])) if (H > 0).any() else 0.0
    dra0 = (xe[i] + xe[i + 1]) / 2.0
    ddec0 = (ye[j] + ye[j + 1]) / 2.0
    # refine on the pairs within one bin of the peak
    near = (np.abs(dra_arcsec - dra0) < bin_arcsec) & (np.abs(ddec_arcsec - ddec0) < bin_arcsec)
    if near.sum() >= 5:
        dra0 = float(np.median(dra_arcsec[near]))
        ddec0 = float(np.median(ddec_arcsec[near]))
    contrast = float(H.max() / bg) if bg else float("inf")
    return (dra0 * 1000.0, ddec0 * 1000.0, float(np.hypot(dra0, ddec0) * 1000.0),
            int(n), contrast)


# Windows (arcsec) the sweep escalates through when a narrow window shows no
# coherent tie.  A LARGE rigid offset (brick-1182 v001 was ~20") has ZERO true
# pairs inside a 3" window -> the peak is noise (low contrast, reference-dependent)
# and looks exactly like "broken/no tie".  Widening the window recovers the real
# peak.  This is the trap that made a 20" offset read as ~2"/incoherent.
DEFAULT_SWEEP_WINDOWS = (3.0, 10.0, 30.0, 60.0)


def _measure_at_window(a, b, maxsep_arcsec, bin_arcsec, min_pairs):
    """Single-window histogram-peak offset.  Returns a result dict or None."""
    ia, ib, _, _ = search_around_sky(a, b, maxsep_arcsec * u.arcsec)
    if len(ia) < min_pairs:
        return None
    cosd = np.cos(np.radians(a[ia].dec.value))
    dra = (b[ib].ra - a[ia].ra).to(u.arcsec).value * cosd
    ddec = (b[ib].dec - a[ia].dec).to(u.arcsec).value
    # keep ~150 bins across the window so the peak stays resolved as we widen
    bw = max(bin_arcsec, maxsep_arcsec / 150.0)
    dra_mas, ddec_mas, off_mas, npairs, contrast = _hist_peak(
        dra, ddec, maxsep_arcsec, bw)
    return dict(dra=dra_mas, ddec=ddec_mas, off=off_mas, npairs=npairs,
                contrast=contrast, window_arcsec=maxsep_arcsec)


def measure_offset(a, b, maxsep=3.0 * u.arcsec, bin_arcsec=0.02, min_pairs=30,
                   min_contrast=None, raise_on_low_contrast=False, context="",
                   sweep=True, sweep_windows=None):
    """Bulk on-sky offset to move ``a`` onto ``b`` (mas), via offset-histogram
    stacking of all pairs within ``maxsep``.  Density-immune; the sanctioned
    replacement for NN-median.

    Parameters
    ----------
    a, b : SkyCoord
        Source lists.  Offset is (b - a) at the histogram peak.
    maxsep : Quantity
        Initial pair-search window.
    bin_arcsec : float
        Base histogram bin size (arcsec); the bin widens with the window to keep
        the peak resolved.
    min_contrast : float or None
        Peak/median contrast floor for a "real" tie (default
        ``DEFAULT_MIN_CONTRAST``).  Below it the tie is flagged (``ok=False``).
    raise_on_low_contrast : bool
        If True, raise ``NoCoherentTieError`` when NO window yields a coherent tie.
    sweep : bool
        If the initial window shows no coherent tie (contrast < ``min_contrast``),
        AUTO-ESCALATE the search window through ``sweep_windows`` and keep the
        highest-contrast result.  This is ON by default: it is the guard against a
        large offset masquerading as "no tie" (the brick-1182 v001 trap).  A single
        contrast threshold at a fixed narrow window is NOT trustworthy without it.
    sweep_windows : iterable of float or None
        Windows (arcsec) to escalate through (default ``DEFAULT_SWEEP_WINDOWS``).

    Returns
    -------
    dict or None
        ``dict(dra, ddec, off, npairs, contrast, ok, window_arcsec, swept)`` (mas),
        or None if too few pairs at every window.  ``ok`` is False when NO window
        reaches ``min_contrast``.  ``window_arcsec`` is the window the reported peak
        came from -- a value >> your expected offset is the tell that the tie was
        only found after widening (investigate: the frame is grossly shifted).
    """
    min_contrast = DEFAULT_MIN_CONTRAST if min_contrast is None else min_contrast
    maxsep_arcsec = maxsep.to(u.arcsec).value if hasattr(maxsep, "to") else float(maxsep)

    windows = [maxsep_arcsec]
    if sweep:
        extra = list(sweep_windows) if sweep_windows is not None else list(DEFAULT_SWEEP_WINDOWS)
        for w in extra:
            if w > maxsep_arcsec:
                windows.append(float(w))

    # Evaluate every window and take the GLOBAL max-contrast peak. We must not stop
    # early on a moderate peak at an intermediate window: a window still smaller than
    # the true offset contains only noise, which can clear a contrast floor by chance
    # (that noise peak is what made v001's 20" offset read as ~1.6"). Only a window
    # that actually CONTAINS the offset produces the real, dominant peak.
    best = None
    for w in windows:
        res = _measure_at_window(a, b, w, bin_arcsec, min_pairs)
        if res is None:
            continue
        if best is None or res["contrast"] > best["contrast"]:
            best = res

    if best is None:
        return None
    best["ok"] = best["contrast"] >= min_contrast
    # swept = the measured offset EXCEEDS the initial window, i.e. the narrow window
    # could not have contained it and the sweep is what found it. A True here on a
    # tie you expected to be small means the frame is grossly shifted -- investigate.
    best["swept"] = (best["off"] / 1000.0) > maxsep_arcsec
    if not best["ok"] and raise_on_low_contrast:
        raise NoCoherentTieError(
            f"No coherent astrometric tie ({context or 'unspecified'}) at any window "
            f"up to {windows[-1]:.0f}\": best contrast {best['contrast']:.1f} < "
            f"{min_contrast} ({best['npairs']} pairs). The two source lists are not "
            f"registered (broken/untied WCS). Do NOT trust a NN-median 'agreement'.")
    return best


def measure_offset_grid(a, b, nx=6, ny=6, ra_bounds=None, dec_bounds=None,
                        maxsep=3.0 * u.arcsec, bin_arcsec=0.02, min_pairs=30,
                        min_contrast=None, context=""):
    """Per-tile offset map — measure ``measure_offset`` in an ``nx`` x ``ny`` grid
    over the field.  A bulk offset ~0 can HIDE a half-mosaic that is untied
    (brick-1182 visit-001), so ALWAYS map per tile before signing off.

    Returns
    -------
    dict
        ``dict(cells=[...], n_ok, n_total, worst_off_mas, min_contrast_seen,
        clean)``.  ``clean`` is True only if every covered cell has ok=True.
    """
    min_contrast = DEFAULT_MIN_CONTRAST if min_contrast is None else min_contrast
    ra = a.ra.deg
    dec = a.dec.deg
    r0, r1 = ra_bounds if ra_bounds else (float(ra.min()), float(ra.max()))
    d0, d1 = dec_bounds if dec_bounds else (float(dec.min()), float(dec.max()))
    re = np.linspace(r0, r1, nx + 1)
    de = np.linspace(d0, d1, ny + 1)
    cells = []
    for i in range(nx):
        for j in range(ny):
            sel = (ra >= re[i]) & (ra < re[i + 1]) & (dec >= de[j]) & (dec < de[j + 1])
            if sel.sum() < min_pairs:
                continue
            res = measure_offset(a[sel], b, maxsep=maxsep, bin_arcsec=bin_arcsec,
                                 min_pairs=min_pairs, min_contrast=min_contrast,
                                 context=f"{context} tile[{i},{j}]")
            if res is None:
                continue
            res.update(ix=i, iy=j)
            cells.append(res)
    covered = [c for c in cells]
    n_ok = sum(1 for c in covered if c["ok"])
    return dict(cells=cells, n_ok=n_ok, n_total=len(covered),
                worst_off_mas=max((c["off"] for c in covered), default=float("nan")),
                min_contrast_seen=min((c["contrast"] for c in covered), default=float("nan")),
                clean=bool(covered) and n_ok == len(covered))


def agree_across_references(a, ref_a, ref_b, tol_mas=100.0, label_a="refA",
                           label_b="refB", **kwargs):
    """Measure the offset of ``a`` against TWO independent references and check they
    AGREE.  A spurious histogram peak (e.g. a large offset only partly captured, or
    a crowding artefact) is REFERENCE-DEPENDENT -- it moves when you swap a dense
    catalogue (VIRAC2) for a sparse one (Gaia-only).  A real rigid tie gives the
    same peak against both.  brick-1182 v001 measured at a narrow window gave
    VIRAC 2.7" vs Gaia 2.0" with dRA differing ~700 mas: that DISAGREEMENT was the
    tell it was not a clean tie (the true offset was ~20", outside the window).

    Returns
    -------
    dict
        ``dict(a=<measure_offset result>, b=<...>, ddra_mas, dddec_mas, sep_mas,
        agree)``.  ``agree`` is False if either measurement failed/low-contrast or
        the two peaks differ by more than ``tol_mas``.
    """
    ra = measure_offset(a, ref_a, context=label_a, **kwargs)
    rb = measure_offset(a, ref_b, context=label_b, **kwargs)
    out = dict(a=ra, b=rb)
    if ra is None or rb is None or not ra.get("ok") or not rb.get("ok"):
        out.update(ddra_mas=float("nan"), dddec_mas=float("nan"),
                   sep_mas=float("nan"), agree=False)
        return out
    ddra = ra["dra"] - rb["dra"]
    dddec = ra["ddec"] - rb["ddec"]
    sep = float(np.hypot(ddra, dddec))
    out.update(ddra_mas=ddra, dddec_mas=dddec, sep_mas=sep, agree=sep <= tol_mas)
    return out
