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


def measure_offset(a, b, maxsep=3.0 * u.arcsec, bin_arcsec=0.02, min_pairs=30,
                   min_contrast=None, raise_on_low_contrast=False, context=""):
    """Bulk on-sky offset to move ``a`` onto ``b`` (mas), via offset-histogram
    stacking of all pairs within ``maxsep``.  Density-immune; the sanctioned
    replacement for NN-median.

    Parameters
    ----------
    a, b : SkyCoord
        Source lists.  Offset is (b - a) at the histogram peak.
    maxsep : Quantity
        Pair-search window.  Should comfortably exceed the largest plausible shift.
    bin_arcsec : float
        Histogram bin size (arcsec).
    min_contrast : float or None
        Peak/median contrast floor for a "real" tie (default
        ``DEFAULT_MIN_CONTRAST``).  Below it the tie is flagged (``ok=False``).
    raise_on_low_contrast : bool
        If True, raise ``NoCoherentTieError`` instead of returning ``ok=False``.

    Returns
    -------
    dict or None
        ``dict(dra, ddec, off, npairs, contrast, ok)`` (mas), or None if too few
        pairs.  ``ok`` is False when contrast < ``min_contrast`` (broken tie).
    """
    min_contrast = DEFAULT_MIN_CONTRAST if min_contrast is None else min_contrast
    maxsep_arcsec = maxsep.to(u.arcsec).value if hasattr(maxsep, "to") else float(maxsep)
    ia, ib, _, _ = search_around_sky(a, b, maxsep_arcsec * u.arcsec)
    if len(ia) < min_pairs:
        return None
    cosd = np.cos(np.radians(a[ia].dec.value))
    dra = (b[ib].ra - a[ia].ra).to(u.arcsec).value * cosd
    ddec = (b[ib].dec - a[ia].dec).to(u.arcsec).value
    dra_mas, ddec_mas, off_mas, npairs, contrast = _hist_peak(
        dra, ddec, maxsep_arcsec, bin_arcsec)
    ok = contrast >= min_contrast
    if not ok and raise_on_low_contrast:
        raise NoCoherentTieError(
            f"No coherent astrometric tie ({context or 'unspecified'}): "
            f"histogram peak contrast {contrast:.1f} < {min_contrast} "
            f"({npairs} pairs). The two source lists are not registered at this "
            f"scale (broken/untied WCS). Do NOT trust a NN-median 'agreement' here.")
    return dict(dra=dra_mas, ddec=ddec_mas, off=off_mas,
                npairs=npairs, contrast=contrast, ok=ok)


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
