"""Reference-free inter-frame overlap registration check (JWST-internal).

WHY THIS EXISTS
---------------
The #1 recurring GC-astrometry corruption is two overlapping exposures (two
visits / pointings / detectors) that sit >1 pixel apart on the sky.  Where they
overlap, the drizzle stacks BOTH contributions, so every star there is doubled
or smeared and its position is biased -- while the field-average offset vs a
reference catalog still reads ~0, because each frame is individually ~ok vs the
reference and only their MUTUAL disagreement in the overlap is wrong.

This is exactly the brick-1182 F200W seam failure (2026-07-12): visit-001 carried
a ~90 mas field-dependent residual near the y=0.5 seam (a single rigid per-visit
shift removed the 20" gross error but left a rotation/scale/distortion residual),
so in the overlap the visit-001 image landed ~90 mas from the visit-002 image and
every star drizzled into a ~0.09" double.  Bulk / coarse-grid / vs-reference QC
all passed it: the whole-field histogram peak averaged the two visits to ~50 mas,
the 4x4 grid diluted the thin overlap strip inside a 67" tile, and
``registration_failsafes`` searched only +-2.5" vs the reference (not frame-vs-
frame) and averaged the overlap away.

The ONLY check that sees it is REFERENCE-FREE and PAIRWISE: for every pair of
exposure-groups that overlap on the sky, histogram-stack their mutual offset and
require it below ``tol_mas`` (default 30 mas).  Reference-free because the failure
is the two frames disagreeing with EACH OTHER, which a common reference hides.

Sanctioned measurement only: this uses ``measure_offset`` (offset-histogram
stacking, window-swept).  It NEVER nearest-neighbour-matches (that fabricates
false agreement in a crowded field -- see CLAUDE.md).  The sweep is mandatory: a
>2.5" overlap offset has zero true pairs in a narrow window and reads as "no
overlap" rather than FAIL; sweeping the window recovers it.

USAGE
-----
Build a ``groups`` mapping ``{label: SkyCoord}`` of per-exposure-group source
lists (one per visit, or per visit x module/detector -- finer groups localise the
culprit).  Then::

    from jwst_gc_pipeline.photometry.interframe_overlap import assert_overlaps_registered
    assert_overlaps_registered(groups, tol_mas=30.0)   # raises on any bad overlap

Detections should come from the per-exposure ``_crf`` frames (each on its own
corrected GWCS), NOT from the drizzled mosaic (which has already merged the
frames and doubled the stars).
"""

import itertools

import numpy as np
from astropy import units as u

from jwst_gc_pipeline.photometry.astrometry_offsets import (
    measure_offset, measure_offset_grid)


# Overlapping same-instrument frames should co-register to well under a NIRCam
# short-wave pixel (~31 mas).  30 mas is the CLAUDE.md sign-off tolerance.
DEFAULT_OVERLAP_TOL_MAS = 30.0


class OverlapMisregistrationError(RuntimeError):
    """Raised when two overlapping exposure-groups are misregistered vs each
    other by more than the tolerance -- the drizzle overlap is corrupted
    (doubled/smeared stars) even if each frame is fine vs a reference."""


def _n_close(a, b, sep_arcsec):
    """Number of ``a`` sources with a ``b`` source within ``sep_arcsec`` — a cheap
    'do these two frames overlap on the sky at all' gate."""
    if len(a) == 0 or len(b) == 0:
        return 0
    idx, sep2d, _ = a.match_to_catalog_sky(b)
    return int((sep2d.to(u.arcsec).value <= sep_arcsec).sum())


def _footprint_intersection(a, b, pad_arcsec=0.0):
    """GEOMETRIC footprint-intersection gate (2026-07-12).

    The original proximity gate ('any stars within 60"') called two ADJACENT
    but DISJOINT groups "overlapping" -- the NIRCam module gap (~44"), or two
    different fields sharing a directory.  ``measure_offset`` on two disjoint
    star fields then produces a STRUCTURAL cross-correlation peak at their
    geometric separation (real brick F405N run: every pair "offset" ~59.8",
    contrast 8-31 -- pure geometry, zero misregistration), a guaranteed false
    FAIL.  Footprints are the right discriminator: a genuinely gross (20"-
    class) misregistration barely moves an arcminute-scale footprint, so the
    intersection is preserved and the swept histogram still catches it --
    while a module gap / different field has NO intersection and is correctly
    skipped.

    Returns ``(bounds, n_a, n_b)``: the intersection box as
    ``(ra_lo, ra_hi, dec_lo, dec_hi)`` in deg (or None), and how many sources
    of each group fall inside it (with ``pad_arcsec`` of margin).
    """
    if len(a) == 0 or len(b) == 0:
        return None, 0, 0
    dec_mid = float(np.median(np.concatenate([a.dec.deg, b.dec.deg])))
    cosd = max(np.cos(np.radians(dec_mid)), 1e-6)
    pad_ra = pad_arcsec / 3600.0 / cosd
    pad_dec = pad_arcsec / 3600.0
    ra_lo = max(a.ra.deg.min(), b.ra.deg.min()) - pad_ra
    ra_hi = min(a.ra.deg.max(), b.ra.deg.max()) + pad_ra
    dec_lo = max(a.dec.deg.min(), b.dec.deg.min()) - pad_dec
    dec_hi = min(a.dec.deg.max(), b.dec.deg.max()) + pad_dec
    if ra_lo >= ra_hi or dec_lo >= dec_hi:
        return None, 0, 0
    n_a = int(((a.ra.deg >= ra_lo) & (a.ra.deg <= ra_hi)
               & (a.dec.deg >= dec_lo) & (a.dec.deg <= dec_hi)).sum())
    n_b = int(((b.ra.deg >= ra_lo) & (b.ra.deg <= ra_hi)
               & (b.dec.deg >= dec_lo) & (b.dec.deg <= dec_hi)).sum())
    return (ra_lo, ra_hi, dec_lo, dec_hi), n_a, n_b


def _in_bounds(coords, bounds):
    ra_lo, ra_hi, dec_lo, dec_hi = bounds
    return ((coords.ra.deg >= ra_lo) & (coords.ra.deg <= ra_hi)
            & (coords.dec.deg >= dec_lo) & (coords.dec.deg <= dec_hi))


def pairwise_overlap_offsets(groups, tol_mas=DEFAULT_OVERLAP_TOL_MAS,
                             maxsep=3.0 * u.arcsec, min_overlap_pairs=40,
                             overlap_gate_arcsec=60.0, sweep=True, context=""):
    """Measure the reference-free mutual offset of every OVERLAPPING pair of
    exposure-groups.

    Parameters
    ----------
    groups : dict[str, SkyCoord]
        ``{label: source_list}`` -- one entry per exposure-group (visit, or
        visit x detector).  Source lists must come from the per-exposure
        corrected frames, not the merged mosaic.
    tol_mas : float
        A pair is FLAGGED (``ok=False``) if it has a coherent mutual tie whose
        magnitude exceeds this.
    maxsep : Quantity
        Initial pair-search window for ``measure_offset`` (it sweeps wider on a
        weak tie, so a gross >window offset is still found).
    min_overlap_pairs : int
        Minimum sources of EACH group inside the footprint-intersection box
        for the pair to be considered overlapping (and minimum pairs for the
        measurement itself).  Fewer -> ``overlap=False`` (not a FAIL).
    overlap_gate_arcsec : float
        DEPRECATED (2026-07-12), ignored.  The overlap gate is now GEOMETRIC
        (footprint-intersection, ``_footprint_intersection``): the old
        star-proximity gate called the disjoint NIRCam module gap
        "overlapping" and produced guaranteed false FAILs at the structural
        cross-field separation.  Kept only for call compatibility.

    Returns
    -------
    list[dict]
        One dict per compared pair:
        ``dict(a, b, overlap, n_overlap, off_mas, dra_mas, ddec_mas, contrast,
        ok, swept, window_arcsec)``.  ``ok`` is True for a non-overlapping pair
        (nothing to check) and for an overlapping pair tied within ``tol_mas``.
    """
    labels = list(groups)
    results = []
    for la, lb in itertools.combinations(labels, 2):
        a, b = groups[la], groups[lb]
        # GEOMETRIC overlap gate: footprint-intersection box with enough
        # sources of BOTH groups inside it.  A star-proximity gate here called
        # the disjoint module gap "overlapping" and measure_offset then
        # returned the structural cross-field separation as a false FAIL.
        bounds, n_a_in, n_b_in = _footprint_intersection(a, b)
        n_close = min(n_a_in, n_b_in)
        if bounds is None or n_close < min_overlap_pairs:
            results.append(dict(a=la, b=lb, overlap=False, n_overlap=n_close,
                                off_mas=None, dra_mas=None, ddec_mas=None,
                                contrast=None, ok=True, swept=False,
                                window_arcsec=None))
            continue
        # measure on the intersection populations only (sources far outside
        # the shared footprint can only contribute noise pairs)
        m = measure_offset(a[_in_bounds(a, bounds)], b[_in_bounds(b, bounds)],
                           maxsep=maxsep, min_pairs=min_overlap_pairs,
                           sweep=sweep, context=f"{context} overlap {la}|{lb}")
        if m is None:
            results.append(dict(a=la, b=lb, overlap=False, n_overlap=n_close,
                                off_mas=None, dra_mas=None, ddec_mas=None,
                                contrast=None, ok=True, swept=False,
                                window_arcsec=None))
            continue
        # A coherent tie (contrast ok) whose magnitude exceeds tol => misregistered.
        # A low-contrast result means we could not measure it (report, don't pass
        # silently): treat as NOT ok so it forces a look.
        coherent = bool(m["ok"])
        within = m["off"] <= tol_mas
        ok = coherent and within
        results.append(dict(a=la, b=lb, overlap=True, n_overlap=n_close,
                            off_mas=float(m["off"]), dra_mas=float(m["dra"]),
                            ddec_mas=float(m["ddec"]), contrast=float(m["contrast"]),
                            ok=bool(ok), swept=bool(m.get("swept", False)),
                            window_arcsec=m.get("window_arcsec")))
    return results


def overlap_offset_grid(groups, tol_mas=DEFAULT_OVERLAP_TOL_MAS, nx=12, ny=12,
                        maxsep=3.0 * u.arcsec, min_overlap_pairs=40,
                        overlap_gate_arcsec=60.0, context=""):
    """Per-TILE reference-free overlap check — the localised version of
    :func:`pairwise_overlap_offsets`.

    A per-visit residual is SPATIALLY VARYING (brick-1182 visit-001 wandered
    14-103 mas across the field), so a single field-pooled offset per group pair
    can average BELOW ``tol_mas`` while a thin seam strip is ~90 mas off. This runs
    ``measure_offset_grid`` (with the offset-magnitude gate) on each overlapping
    group pair, so a LOCAL misregistration in any tile fails even when the
    field-average is small.

    Returns
    -------
    list[dict]
        Per compared pair: ``dict(a, b, overlap, worst_off_mas, worst_off_cell,
        n_ok, n_total, clean, ok)``. ``ok`` is ``clean`` for overlapping pairs and
        True for non-overlapping pairs.
    """
    labels = list(groups)
    results = []
    for la, lb in itertools.combinations(labels, 2):
        a, b = groups[la], groups[lb]
        # geometric footprint-intersection gate (see pairwise_overlap_offsets)
        bounds, n_a_in, n_b_in = _footprint_intersection(a, b)
        if bounds is None or min(n_a_in, n_b_in) < min_overlap_pairs:
            results.append(dict(a=la, b=lb, overlap=False, worst_off_mas=None,
                                worst_off_cell=None, n_ok=0, n_total=0,
                                clean=True, ok=True))
            continue
        # grid ONLY the intersection box -- tiles outside the shared footprint
        # have no true pairs and can only produce structural noise peaks
        g = measure_offset_grid(a[_in_bounds(a, bounds)], b[_in_bounds(b, bounds)],
                                nx=nx, ny=ny, maxsep=maxsep,
                                ra_bounds=(bounds[0], bounds[1]),
                                dec_bounds=(bounds[2], bounds[3]),
                                max_off_mas=tol_mas, min_pairs=min_overlap_pairs,
                                context=f"{context} {la}|{lb}")
        results.append(dict(a=la, b=lb, overlap=True,
                            worst_off_mas=g["worst_off_mas"],
                            worst_off_cell=g["worst_off_cell"],
                            n_ok=g["n_ok"], n_total=g["n_total"],
                            clean=bool(g["clean"]), ok=bool(g["clean"])))
    return results


def assert_overlaps_registered(groups, tol_mas=DEFAULT_OVERLAP_TOL_MAS,
                               raise_on_fail=True, per_tile=False, grid=(12, 12),
                               **kwargs):
    """Run the reference-free overlap check and raise
    ``OverlapMisregistrationError`` if any overlapping pair fails.  Returns the
    full results list.

    ``per_tile=False`` (default) uses the field-pooled
    :func:`pairwise_overlap_offsets` (one offset per pair -- cheap).
    ``per_tile=True`` uses :func:`overlap_offset_grid` with an ``nx x ny`` = ``grid``
    map, which catches a LOCAL (spatially varying) residual that field-pooling
    would average away -- use this for the brick-1182 seam class of failure.

    Set ``raise_on_fail=False`` to get the results without raising.
    """
    if per_tile:
        results = overlap_offset_grid(groups, tol_mas=tol_mas, nx=grid[0],
                                      ny=grid[1], **kwargs)
        bad = [r for r in results if r["overlap"] and not r["ok"]]
        if bad and raise_on_fail:
            lines = [
                f"  {r['a']} | {r['b']}: worst tile off={_fmt(r['worst_off_mas'])} mas "
                f"(cell {r['worst_off_cell']}), {r['n_ok']}/{r['n_total']} tiles ok"
                for r in bad
            ]
            raise OverlapMisregistrationError(
                f"{len(bad)} overlapping exposure-group pair(s) have a LOCAL "
                f"misregistration > {tol_mas:.0f} mas in at least one tile "
                f"(reference-free, per-tile). A field-average can hide this; the "
                f"drizzle overlap is corrupted (doubled/smeared stars). This is the "
                f"brick-1182 F200W seam failure mode.\n" + "\n".join(lines))
        return results

    results = pairwise_overlap_offsets(groups, tol_mas=tol_mas, **kwargs)
    bad = [r for r in results if r["overlap"] and not r["ok"]]
    if bad and raise_on_fail:
        lines = [
            f"  {r['a']} | {r['b']}: off={_fmt(r['off_mas'])} mas "
            f"(dRA={_fmt(r['dra_mas'])}, dDec={_fmt(r['ddec_mas'])}), "
            f"contrast={_fmt(r['contrast'], 1)}, n={r['n_overlap']}"
            + (f"  [swept to {r['window_arcsec']:.0f}\" -- GROSS offset]"
               if r.get("swept") else "")
            for r in bad
        ]
        raise OverlapMisregistrationError(
            f"{len(bad)} overlapping exposure-group pair(s) misregistered vs each "
            f"other by > {tol_mas:.0f} mas (reference-free). The drizzle overlap is "
            f"corrupted -- stars there are doubled/smeared even if each frame is fine "
            f"vs a reference catalog. This is the brick-1182 F200W seam failure mode.\n"
            + "\n".join(lines))
    return results


def _fmt(x, d=0):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"
