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
``registration_failsafes`` compared each frame only against the reference
catalog, within +-2.5", and averaged the overlap away.

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

Detections must come from the per-exposure ``_crf`` frames, each on its own
corrected GWCS.  The drizzled mosaic has already merged the frames and doubled
the stars, so it can no longer show the disagreement.
"""

import itertools

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky

# NOT measure_offset_grid.  This module used to gate per-tile with it and
# stopped in 0fb1958 (2026-07-13): a per-tile histogram over a 40-250-star tile
# produces a noise peak that clears any contrast floor and lands anywhere in the
# window.  The import outlived the call, and left the module reading as though
# it still used the estimator CLAUDE.md named as the gate (issue #392).
from jwst_gc_pipeline.photometry.astrometry_offsets import (
    confirm_peak_windows, local_residual_map, measure_offset)


# Overlapping same-instrument frames should co-register to well under a NIRCam
# short-wave pixel (~31 mas).  30 mas is the CLAUDE.md sign-off tolerance.
DEFAULT_OVERLAP_TOL_MAS = 30.0


class OverlapMisregistrationError(RuntimeError):
    """Raised when two overlapping exposure-groups are misregistered vs each
    other by more than the tolerance -- the drizzle overlap is corrupted
    (doubled/smeared stars) even if each frame is fine vs a reference."""


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

    The box is an axis-aligned RA/Dec bounding-box intersection — a
    CONSERVATIVE PROXY for the true (possibly rotated / L-shaped) footprint
    polygon.  Two tiles sharing only a bbox corner (not real sky) can get a
    small spurious box; the "≥ min sources of BOTH groups inside" requirement
    downstream is what makes that harmless (a corner box holds few sources of
    either group → skipped).

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


def _confirm_tie(a, b, cand_dra_mas, cand_ddec_mas, min_pairs, n_close):
    """Confirm a candidate pooled offset at NARROW window.

    A swept wide-window candidate can be a structural/chance peak: at a 60"
    window the histogram bin is ~0.4" fat, so n_peak counts thousands of
    chance pairs and contrast ~5 can be noise (real-data null test: unrelated
    populations produced a 628 mas 'tie' with n_peak 4505).  A REAL tie
    reproduces when the candidate shift is removed and the histogram is
    re-run at a 0.3" window with 0.02" bins: the true pairs collapse into a
    tight coherent peak near zero holding a real fraction of the population;
    a noise candidate dissolves.  Returns the confirming narrow-window
    result, or None."""
    dec_mid = float(np.median(a.dec.deg))
    cosd = max(np.cos(np.radians(dec_mid)), 1e-6)
    a_shift = SkyCoord(ra=a.ra + (cand_dra_mas / 3.6e6 / cosd) * u.deg,
                       dec=a.dec + (cand_ddec_mas / 3.6e6) * u.deg,
                       frame="icrs")
    m2 = measure_offset(a_shift, b, maxsep=0.3 * u.arcsec, bin_arcsec=0.02,
                        min_pairs=min_pairs, sweep=False)
    if m2 is None or not m2["ok"]:
        return None
    if m2["off"] > 100.0:   # candidate did not land near zero -> not confirmed
        return None
    n_peak_floor = max(30, int(0.05 * n_close))
    if m2.get("n_peak", 0) < n_peak_floor:
        return None
    return m2


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
        corrected frames; the merged mosaic has already doubled the stars.
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
            # same schema as every other result dict (n_peak/measurable
            # included): consumers index r["measurable"] unconditionally.
            # n_a_in/n_b_in are carried for parity with overlap_offset_grid
            # (issue #402): min(n_a_in, n_b_in) alone cannot tell a
            # 29-and-53 strip from a 0-and-64 one.
            results.append(dict(a=la, b=lb, overlap=False, n_overlap=n_close,
                                n_a_in=int(n_a_in), n_b_in=int(n_b_in),
                                off_mas=None, dra_mas=None, ddec_mas=None,
                                contrast=None, n_peak=0, measurable=False,
                                ok=True, swept=False, window_arcsec=None,
                                window_consistent=None,
                                window_edge_fraction=None,
                                alias_rejected=False))
            continue
        # measure on the intersection populations only (sources far outside
        # the shared footprint can only contribute noise pairs)
        m = measure_offset(a[_in_bounds(a, bounds)], b[_in_bounds(b, bounds)],
                           maxsep=maxsep, min_pairs=min_overlap_pairs,
                           sweep=sweep, context=f"{context} overlap {la}|{lb}")
        if m is None:
            results.append(dict(a=la, b=lb, overlap=False, n_overlap=n_close,
                                n_a_in=int(n_a_in), n_b_in=int(n_b_in),
                                off_mas=None, dra_mas=None, ddec_mas=None,
                                contrast=None, n_peak=0, measurable=False,
                                ok=True, swept=False, window_arcsec=None,
                                window_consistent=None,
                                window_edge_fraction=None,
                                alias_rejected=False))
            continue
        # A verdict needs BOTH a coherent peak (contrast) AND a peak holding a
        # non-trivial fraction of the shared population (n_peak) -- a stripey
        # interleave's structural peak clears the contrast floor but holds only
        # a sliver of the stars.  Not measurable != passing: ok=False so it
        # forces a look (the caller decides fail-closed semantics).
        coherent = bool(m["ok"])
        conf = (_confirm_tie(a[_in_bounds(a, bounds)], b[_in_bounds(b, bounds)],
                             m["dra"], m["ddec"], min_overlap_pairs, n_close)
                if coherent else None)
        if conf is not None:
            m = dict(m, dra=m["dra"] + conf["dra"], ddec=m["ddec"] + conf["ddec"])
            m["off"] = float(np.hypot(m["dra"], m["ddec"]))
        measurable = conf is not None
        within = m["off"] <= tol_mas
        ok = measurable and within
        # WINDOW CONFIRMATION (issue #158), run LAZILY.  A real rigid offset reads
        # the same value at every window large enough to hold it; a pair-density
        # ridge slides with the window.  Only a pair about to be called grossly
        # misregistered needs that discriminator, so confirm exactly there:
        # ``measure_offset(confirm_windows=True)`` would probe EVERY swept peak at
        # 1.4x and 2.2x its window, i.e. two extra wide-window histograms per
        # pair, for pairs whose verdict cannot change.
        if measurable and not within and m.get("swept"):
            _conf_w = confirm_peak_windows(
                a[_in_bounds(a, bounds)], b[_in_bounds(b, bounds)], m,
                min_pairs=min_overlap_pairs)
            m = dict(m, window_consistent=_conf_w["consistent"])
        results.append(dict(a=la, b=lb, overlap=True, n_overlap=n_close,
                            n_a_in=int(n_a_in), n_b_in=int(n_b_in),
                            off_mas=float(m["off"]), dra_mas=float(m["dra"]),
                            ddec_mas=float(m["ddec"]), contrast=float(m["contrast"]),
                            n_peak=int(m.get("n_peak", 0)),
                            measurable=bool(measurable),
                            ok=bool(ok), swept=bool(m.get("swept", False)),
                            window_arcsec=m.get("window_arcsec"),
                            window_consistent=m.get("window_consistent"),
                            window_edge_fraction=m.get("window_edge_fraction"),
                            alias_rejected=bool(m.get("alias_rejected", False))))
    return results


def overlap_offset_grid(groups, tol_mas=DEFAULT_OVERLAP_TOL_MAS, nx=12, ny=12,
                        maxsep=3.0 * u.arcsec, min_overlap_pairs=40,
                        overlap_gate_arcsec=60.0, margin_factor=3.0,
                        match_radius=0.3 * u.arcsec, nsigma=3.0, context=""):
    """Per-TILE reference-free overlap check — the localised version of
    :func:`pairwise_overlap_offsets`.

    MATCHED-PAIR fine layer (2026-07-13, v8 redesign): per-tile HISTOGRAM
    verdicts are statistically unsound at fine scale — a 40-250-star tile's
    noise peak can clear any contrast/population floor and land anywhere in
    the window (real-data ladder: ~59.8" -> ~2.9" -> ~1.5" false tiles as the
    floors tightened).  Once the POOLED swept tie of the pair is verified
    small, ``local_residual_map`` is the correct fine instrument: residuals
    come from real matched pairs, cells carry standard errors, and a cell is
    only flagged when its offset is BOTH above tolerance AND significant —
    the noise-peak failure mode does not exist.

    Pair pipeline:
      1. geometric footprint-intersection gate (skip disjoint pairs);
      2. POOLED swept histogram on the intersection populations (measurability
         = contrast + n_peak population-fraction floor):
         - not measurable -> could_not_verify (caller/release path owns it);
         - measurable and off > tol_mas -> FAIL (bulk misregistration —
           the brick-1182 v001 gross class and the 51 mas inter-visit class);
         - measurable, small -> 3. ``local_residual_map`` over the
           intersection at the grid's cell scale: any significant cell above
           ``tol_mas`` -> FAIL (the 30-100 mas seam class).

    GEOMETRY decides ``overlap``; SOURCE COUNTS decide measurability (issue
    #402).  A pair whose footprints DO intersect but which holds fewer than
    ``min_overlap_pairs`` sources of the sparser side inside the intersection
    used to be recorded as ``overlap=False, clean=True, ok=True,
    could_not_verify=False`` -- indistinguishable, in the record and in the
    gate's verdict line, from a pair with nothing to check.  That merged
    "checked and clean" with "too thin to look at" for exactly the population
    most likely to be misregistered: sickle F770W's two MIRI visits intersect
    over a strip holding 29 and 53 sources, carry a 37 mas swept candidate
    against a 30 mas tolerance, and are the field's ONLY pair, so the filter
    reported "0 overlapping pairs" and passed with nothing examined.  Such a
    pair is now ``overlap=True, could_not_verify=True, clean=False``, which is
    the fail-closed route everything else unmeasurable takes and is what makes
    it visible to the reference arbiter.  Only ``bounds is None`` -- footprints
    that do not intersect at all -- stays ``overlap=False``.

    Every record carries ``n_a_in``/``n_b_in``, the source counts inside the
    intersection box, so a consumer can tell 29-and-53 from 0-and-64.

    Returns
    -------
    list[dict]
        Per compared pair: ``dict(a, b, overlap, n_a_in, n_b_in,
        pooled_off_mas, pooled, worst_off_mas, worst_off_cell, n_ok, n_total,
        n_no_coverage, could_not_verify, clean, ok, fail_reason)``.
    """
    labels = list(groups)
    results = []
    for la, lb in itertools.combinations(labels, 2):
        a, b = groups[la], groups[lb]
        bounds, n_a_in, n_b_in = _footprint_intersection(a, b)
        if bounds is None:
            # geometrically disjoint -- there is genuinely nothing to check
            results.append(dict(a=la, b=lb, overlap=False,
                                n_a_in=int(n_a_in), n_b_in=int(n_b_in),
                                pooled_off_mas=None,
                                pooled=None, worst_off_mas=None,
                                worst_off_cell=None, n_ok=0, n_total=0,
                                n_no_coverage=0, could_not_verify=False,
                                clean=True, ok=True, fail_reason=None))
            continue
        if min(n_a_in, n_b_in) < min_overlap_pairs:
            # footprints DO intersect, too few sources to measure the pair:
            # could-not-verify, not "not overlapping" (issue #402)
            results.append(dict(a=la, b=lb, overlap=True,
                                n_a_in=int(n_a_in), n_b_in=int(n_b_in),
                                pooled_off_mas=None,
                                pooled=None, worst_off_mas=None,
                                worst_off_cell=None, n_ok=0, n_total=0,
                                n_no_coverage=0, could_not_verify=True,
                                clean=False, ok=True,
                                fail_reason=(
                                    f"thin overlap: {n_a_in} / {n_b_in} "
                                    f"sources inside the footprint "
                                    f"intersection, below the "
                                    f"min_overlap_pairs={min_overlap_pairs} "
                                    f"floor -- NOT measured")))
            continue
        a_in = a[_in_bounds(a, bounds)]
        b_in = b[_in_bounds(b, bounds)]
        base = dict(a=la, b=lb, overlap=True, n_a_in=int(n_a_in),
                    n_b_in=int(n_b_in), worst_off_mas=None,
                    worst_off_cell=None, n_ok=0, n_total=0, n_no_coverage=0,
                    could_not_verify=False, clean=False, ok=False,
                    fail_reason=None)

        # 2. pooled swept tie with measurability floor
        m = measure_offset(a_in, b_in, maxsep=maxsep,
                           min_pairs=min_overlap_pairs, sweep=True,
                           context=f"{context} {la}|{lb} pooled")
        n_close = min(n_a_in, n_b_in)
        conf = None
        if m is not None and m["ok"]:
            conf = _confirm_tie(a_in, b_in, m["dra"], m["ddec"],
                                min_overlap_pairs, n_close)
            if conf is not None:
                # refine the pooled offset with the narrow-window confirmation
                m = dict(m, dra=m["dra"] + conf["dra"],
                         ddec=m["ddec"] + conf["ddec"])
                m["off"] = float(np.hypot(m["dra"], m["ddec"]))
        measurable = conf is not None
        base["pooled"] = None if m is None else {
            k: m.get(k) for k in ("dra", "ddec", "off", "contrast", "n_peak",
                                  "swept", "window_arcsec")}
        base["pooled_off_mas"] = None if m is None else float(m["off"])
        if not measurable:
            base.update(could_not_verify=True, ok=True)
            results.append(base)
            continue
        if m["off"] > tol_mas:
            base.update(ok=False, clean=False,
                        fail_reason=f"pooled offset {m['off']:.0f} mas"
                                    + (" (swept -- GROSS)" if m.get("swept")
                                       else ""))
            results.append(base)
            continue

        # 3. matched-pair fine map at the grid cell scale
        radius_arcsec = match_radius.to(u.arcsec).value \
            if hasattr(match_radius, "to") else float(match_radius)
        # CHANCE-ASSOCIATION GUARD: matched-pair residuals are only meaningful
        # when the nearest matches ARE the counterparts.  Real registered
        # frames match at ~the (verified-small) tie distance; two populations
        # WITHOUT true counterparts match by chance at ~0.7*radius.  Gate on
        # the median nearest separation; beyond radius/3 the fine layer cannot
        # measure this pair (could_not_verify, owned by the reference map).
        _ia, _ib, _sep, _ = search_around_sky(a_in, b_in,
                                              radius_arcsec * u.arcsec)
        if len(_ia) == 0:
            base.update(could_not_verify=True, ok=True)
            results.append(base)
            continue
        _order = np.lexsort((_sep.arcsec, _ia))
        _first = np.concatenate(([True], _ia[_order][1:] != _ia[_order][:-1]))
        _med_sep_mas = float(np.median(_sep.arcsec[_order][_first]) * 1000.0)
        if _med_sep_mas > radius_arcsec * 1000.0 / 3.0:
            base.update(could_not_verify=True, ok=True,
                        fail_reason=None)
            results.append(base)
            continue
        ra_lo, ra_hi, dec_lo, dec_hi = bounds
        dec_mid = 0.5 * (dec_lo + dec_hi)
        cosd = max(np.cos(np.radians(dec_mid)), 1e-6)
        extent = max((ra_hi - ra_lo) * cosd, dec_hi - dec_lo) * 3600.0
        cell_arcsec = max(extent / max(nx, ny), 2.0)
        lrm = local_residual_map(a_in, b_in, m, cell_arcsec=cell_arcsec,
                                 match_radius=match_radius,
                                 min_stars=max(10, min_overlap_pairs // 4),
                                 tol_mas=tol_mas, nsigma=nsigma,
                                 context=f"{context} {la}|{lb} fine")
        flagged = [c for c in lrm["cells"] if c["flagged"]]
        worst = max(lrm["cells"], key=lambda c: c["off_mas"], default=None)
        base.update(
            worst_off_mas=None if worst is None else float(worst["off_mas"]),
            worst_off_cell=None if worst is None else dict(
                ix=worst["ix"], iy=worst["iy"], off_mas=worst["off_mas"],
                n=worst["n"], sem=float(np.hypot(worst["dra_sem"],
                                                 worst["ddec_sem"]))),
            n_ok=lrm["n_measured"] - len(flagged), n_total=lrm["n_measured"],
            n_no_coverage=0,
            could_not_verify=not lrm["cells"],
            clean=bool(lrm["cells"]) and not flagged,
            ok=bool((lrm["cells"] and not flagged) or not lrm["cells"]),
            fail_reason=(f"{len(flagged)} significant fine cell(s) > "
                         f"{tol_mas:.0f} mas" if flagged else None))
        results.append(base)
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
