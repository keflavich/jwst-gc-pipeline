"""Per-visit consistency of a ``RECORDED_BULK`` field's recorded bulks (#857).

The m7 cross-filter gate demands that every filter of a visit sits in one
frame.  A ``RECORDED_BULK`` field applies a hand-measured constant per
``(visit, filter)``, so two filters of one visit whose recorded bulks differ
are separated by exactly that difference after ``fix_alignment``, provided
their raw ``assign_wcs`` frames agree.  m4 (1979/002) records F150W2 and
F322W2 bulks 548 mas apart; the gate then reads the same 548 mas and
refuses.  :func:`bulk_spread_by_visit` computes the differential from the
configuration alone, so a bulk table can be checked before any frame is
regenerated.

No product is read here.  Re-measuring each bulk against Gaia needs the crf
frames and is a separate, heavier audit.
"""
import numpy as np

from jwst_gc_pipeline.reduction import alignment_config as ac


def _visit_filters(cfg):
    """``{visit: [filter, ...]}`` named by the entry's ``recorded_bulk`` keys.

    ``ANY`` keys carry no visit or filter of their own and are skipped: they
    resolve to one value for every frame they cover, so they cannot split a
    visit.
    """
    visits = {}
    for visit, filt in cfg.recorded_bulk:
        if visit == ac.ANY or filt == ac.ANY:
            continue
        visits.setdefault(visit, []).append(filt)
    return visits


def applied_bulk_onsky_mas(cfg, visit, filtername):
    """The bulk ``fix_alignment`` applies to ``(visit, filtername)``, as
    ``(dra_mas, ddec_mas)`` on-sky, resolved through
    :func:`alignment_config.lookup_recorded_bulk`.

    ``(nan, nan)`` when no key matches (the frame stays at the raw frame).
    """
    dra, ddec, found = ac.lookup_recorded_bulk(cfg, visit, filtername)
    if not found:
        return float('nan'), float('nan')
    dec_ref = cfg.dec_ref_deg if cfg.dec_ref_deg is not None else 0.0
    return dra * 1000.0 * np.cos(np.radians(dec_ref)), ddec * 1000.0


def bulk_spread_by_visit(cfg, anchor_filter=None):
    """Per-visit bulk differential of every filter against the anchor filter.

    Returns ``{visit: {'anchor': F, 'filters': {F: (dra_mas, ddec_mas)},
    'differential_mas': {F: (dra, ddec)}, 'max_differential_mas': float}}``.
    ``differential_mas`` is filter minus anchor, on-sky; the m7 gate reports
    anchor minus filter, so its two components carry the opposite sign.
    ``anchor_filter`` defaults to ``pick_reference_anchor_filter`` over the
    visit's filters, the band the m7 gate uses as its anchor.
    """
    from jwst_gc_pipeline.photometry.visit_consensus import (
        pick_reference_anchor_filter)
    out = {}
    for visit, filters in _visit_filters(cfg).items():
        anchor = anchor_filter or pick_reference_anchor_filter(filters)
        bulks = {f: applied_bulk_onsky_mas(cfg, visit, f) for f in sorted(filters)}
        a_dra, a_ddec = bulks[anchor]
        diff = {f: (b[0] - a_dra, b[1] - a_ddec) for f, b in bulks.items()}
        out[visit] = dict(
            anchor=anchor, filters=bulks, differential_mas=diff,
            max_differential_mas=float(max(np.hypot(*d) for d in diff.values())))
    return out
