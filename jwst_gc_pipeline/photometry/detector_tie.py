"""Per-(visit, detector) translation tie from the cross-detector overlap network.

Targets the residual the GDC distortion-swap experiment isolated (2026-07):
after the DVA inter-detector correction, cross-module (A/B seam) same-star
offsets of 2.7-5.4 mas remain that are detector-PAIR- and filter-dependent
(brick F212N nrca3-nrcb4 = -5.4 mas Dec; nrca4-nrcb3 = +2.5/-2.5 mas) while
the within-detector affine residual is <1 mas rms.  That is inter-detector
rigid PLACEMENT (SIAF-class, static 1-2.5 mas per detector per the network
self-calibration), not distortion -- and neither the module-locked tables
(one shift for all detectors) nor the per-exposure jitter rows (per-frame
2 mas gate at a per-frame ~1-2 mas noise floor) can express it.

WHY the reference is the OVERLAP NETWORK and NOT the visit consensus
--------------------------------------------------------------------
The obvious measurement -- each detector's pooled stars vs the ALL-detector
visit consensus -- SELF-CANCELS.  NIRCam dithers are small compared to a
detector, so an interior star is observed by the SAME detector in every
exposure: its consensus position is built almost entirely from that
detector's own measurements and carries the detector's placement error.
Measured on brick F212N (2026-07-23, validate_detector_tie.py): every
per-detector vs-consensus offset read 0.3-0.9 mas while the true seam terms
are 2.7-5.4 mas -- a ~15-20% diluted echo carried only by the seam-strip
stars.  This is the same self-referential trap class as
``registration_failsafes`` matching a mosaic against its own catalog
(CLAUDE.md release-gate blind spots) and the brick F182M 15 mas
self-cancellation.  The placement term lives ONLY in cross-detector
information, so it is measured there: same-star offsets of every
overlapping frame PAIR from different detectors, solved as a small
least-squares NETWORK for one rigid shift per detector (gauge: the
per-component MEDIAN shift is zero, so one bad detector cannot drag the
gauge and the visit-bulk/absolute tie is untouched by construction).

Measurement properties:

* per-pair measurement is the sanctioned histogram-detection + same-star
  refinement pair (``measure_offset`` swept -> giant-cell
  ``local_residual_map``; refinement is legal only after a verified small
  un-swept histogram tie).  NEVER NN-median (CLAUDE.md rule #1);
* a grossly shifted frame pair (swept / >90 mas) is SKIPPED -- gross
  misalignment is the per-exposure consensus machinery's job, not a
  placement trim;
* a detector below the floor (connectivity, n_pairs, sem, significance) is
  REFUSED: left uncorrected with a loud warning, never guessed;
* VIRAC2 is only a GROSS, non-blocking cross-check on the DIFFERENTIAL term
  (GC rule: Gaia is the frame, VIRAC2 the catalog, and neither may veto a
  coherent internal tie at the mas level);
* translation-only (v1): the measured seam terms are translations; a full
  6-param affine needs a new GWCS apply mechanism and is deferred (see
  ``DETECTOR_TIE_DESIGN.md``, section 2).

Gating: default OFF; ``ASTROM_M2_PER_DETECTOR_TIE=1`` enables the hook in
``astrometry_checkpoint.run_visit_checkpoint`` at correcting (m2) stages
only.  Corrections are per-detector offsets-table rows
(Exposure = BULK_EXPOSURE, Module = detector token) applied through the
existing consensus-table + m2 stop/regenerate flow.
"""
import itertools
import os

import numpy as np
from astropy import units as u

from .astrometry_offsets import (
    GlobalTieNotVerifiedError, KDTreeReference, local_residual_map,
    measure_offset,
)
from .interframe_overlap import _footprint_intersection, _in_bounds

# Environment flag that enables the per-detector tie at the m2 checkpoint.
PER_DETECTOR_TIE_ENV = "ASTROM_M2_PER_DETECTOR_TIE"

# --- refusal ladder (justifications in DETECTOR_TIE_DESIGN.md, section 2) ---
# Minimum combined same-star pairs over a detector's network measurements:
# below ~50 the sem of a ~5 mas-scatter population is >~0.7 mas and a 2 mas
# term is barely 3 sigma.
DETECTOR_TIE_MIN_PAIRS = 50
# Maximum standard error of the adopted tie: keeps every applied correction
# >= 2 sigma and the applied noise <= 1/4 of the smallest real (2 mas) term.
DETECTOR_TIE_MAX_SEM_MAS = 1.0
# Do not correct below this magnitude (sub-mas placements are below the SIAF
# static floor and below what re-drizzling can use), nor below 3 sigma.
DETECTOR_TIE_APPLY_MIN_MAS = 1.0
DETECTOR_TIE_NSIGMA = 3.0
# GROSS internal-vs-VIRAC2 cross-check tolerance.  Far above VIRAC2's ~5-10
# mas local wander so the reference's own systematics can never veto the
# internal measurement; catches only a spurious solve / wrong grouping.
DETECTOR_REF_GROSS_TOL_MAS = 15.0

# Per-pair floors/caps for the network measurements.
PAIR_MIN_OVERLAP_STARS = 25   # stars of BOTH frames inside the overlap box
PAIR_MIN_SAMESTAR = 10        # matched pairs for a usable pair measurement
PAIR_MAX_OFF_MAS = 90.0       # beyond this the pair is a gross-misalignment case
MAX_PAIRS_PER_DETPAIR = 40    # runtime cap, deterministic subsample
PAIR_SEM_FLOOR_MAS = 0.05     # weight cap: one razor-sharp pair must not dominate

SAMESTAR_MATCH_RADIUS = 0.3 * u.arcsec


def per_detector_tie_enabled():
    """True when the opt-in env flag is set (default OFF)."""
    return os.environ.get(PER_DETECTOR_TIE_ENV, "").strip() == "1"


def _samestar_refine(coords, reference, hist_result, min_pairs=PAIR_MIN_SAMESTAR,
                     context=""):
    """Same-star (matched-pair) refinement of a verified small histogram tie:
    single giant-cell ``local_residual_map``.  Returns
    (dra_mas, ddec_mas, dra_sem, ddec_sem, n_pairs) of the TOTAL tie
    (histogram + residual) or None when refinement is impossible."""
    ref_coords = reference.coords if isinstance(reference, KDTreeReference) \
        else reference
    try:
        lrm = local_residual_map(
            coords, ref_coords, hist_result, cell_arcsec=1e9,
            match_radius=SAMESTAR_MATCH_RADIUS, min_stars=min_pairs,
            context=context)
    except GlobalTieNotVerifiedError:
        return None
    if not lrm["cells"]:
        return None
    c = max(lrm["cells"], key=lambda cc: cc["n"])
    return (float(hist_result["dra"] + c["dra_mas"]),
            float(hist_result["ddec"] + c["ddec_mas"]),
            float(c["dra_sem"]), float(c["ddec_sem"]), int(c["n"]))


def group_frames_by_detector(exposure_entries):
    """Group per-frame entries by detector token.

    ``exposure_entries``: iterable of ``(key, coords, hist_result_or_None)``
    where ``key`` is the ``exposure_key`` tuple
    ``(visit, exposure, module, filter)`` (``module`` is the DETECTOR token
    of a per-frame catalog), ``coords`` the frame's reliable-star SkyCoord,
    and ``hist_result`` the frame's already-measured vs-consensus
    ``measure_offset`` result when available (recorded as the
    self-cancellation diagnostic; see the module docstring).  Returns
    detector -> list of ``(coords, hist_result_or_None)``.
    """
    by_det = {}
    for key, coords, hist in exposure_entries:
        by_det.setdefault(str(key[2]).lower(), []).append((coords, hist))
    return by_det


def measure_pair_samestar(a, b, context=""):
    """Same-star offset (b - a, mas) of one overlapping frame pair, or None.

    Sanctioned two-step: swept histogram detection, then giant-cell
    matched-pair refinement.  A pair whose tie is swept or >
    ``PAIR_MAX_OFF_MAS`` is NOT a placement measurement (gross misalignment,
    the per-exposure machinery's job) and returns None.
    """
    m = measure_offset(a, b, min_pairs=PAIR_MIN_OVERLAP_STARS, sweep=True,
                       context=context)
    if m is None or not m.get("ok") or m.get("swept") \
            or m["off"] > PAIR_MAX_OFF_MAS:
        return None
    refined = _samestar_refine(a, b, m, context=context + " same-star")
    if refined is None:
        return None
    dra, ddec, dra_sem, ddec_sem, n = refined
    return dict(dra_mas=dra, ddec_mas=ddec, dra_sem=dra_sem,
                ddec_sem=ddec_sem, n=n, contrast=float(m["contrast"]))


def measure_cross_detector_pairs(frames_by_detector,
                                 max_pairs_per_detpair=MAX_PAIRS_PER_DETPAIR,
                                 min_overlap_stars=PAIR_MIN_OVERLAP_STARS,
                                 context=""):
    """Same-star offsets for overlapping frame pairs of DIFFERENT detectors.

    Returns a list of measurement dicts
    ``dict(det_a, det_b, dra_mas, ddec_mas, dra_sem, ddec_sem, n, contrast)``
    with the offset in the ``det_b - det_a`` sense.  Frame pairs are found by
    footprint-box intersection (the ``interframe_overlap`` geometry gate: a
    module gap / disjoint tiles produce NO box, so no structural false peak)
    and capped per detector pair with a deterministic subsample.
    """
    candidates = {}
    detectors = sorted(frames_by_detector)
    for da, db in itertools.combinations(detectors, 2):
        cand = []
        for (ca, _ha), (cb, _hb) in itertools.product(frames_by_detector[da],
                                                      frames_by_detector[db]):
            bounds, n_a, n_b = _footprint_intersection(ca, cb)
            if bounds is None or min(n_a, n_b) < min_overlap_stars:
                continue
            cand.append((ca, cb, bounds))
        if cand:
            candidates[(da, db)] = cand
    rng = np.random.default_rng(1182)
    measurements = []
    for (da, db), cand in sorted(candidates.items()):
        if len(cand) > max_pairs_per_detpair:
            idx = rng.choice(len(cand), max_pairs_per_detpair, replace=False)
            cand = [cand[i] for i in sorted(idx)]
        for k, (ca, cb, bounds) in enumerate(cand):
            a_in = ca[_in_bounds(ca, bounds)]
            b_in = cb[_in_bounds(cb, bounds)]
            rec = measure_pair_samestar(a_in, b_in,
                                        context=f"{context} {da}|{db} #{k}")
            if rec is None:
                continue
            rec.update(det_a=da, det_b=db)
            measurements.append(rec)
    return measurements


def _connected_components(detectors, edges):
    parent = {d: d for d in detectors}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra_, rb_ = find(a), find(b)
        if ra_ != rb_:
            parent[ra_] = rb_
    comps = {}
    for d in detectors:
        comps.setdefault(find(d), []).append(d)
    return list(comps.values())


def solve_detector_network(measurements, detectors=None):
    """Weighted least-squares rigid-shift-per-detector solve of the pairwise
    same-star offsets, per connected component, MEDIAN gauge.

    Model: a pair measurement is ``D_ab = p_b - p_a + noise`` where ``p_d``
    is detector ``d``'s placement displacement.  The solve recovers ``p`` up
    to a per-component additive constant (the network null space); the gauge
    fixes the per-component per-axis MEDIAN of ``p`` to zero -- robust (one
    bad detector cannot drag the gauge, mirroring the consensus median
    re-centering) and bulk-preserving (the correction ``-p`` then has ~zero
    net effect on the visit's absolute tie).

    Weights ``1/sem^2`` with a ``PAIR_SEM_FLOOR_MAS`` floor; per-detector
    standard errors come from the pseudo-inverse covariance scaled by the
    reduced chi-square (never deflated below 1).

    Returns detector -> ``dict(dra_mas, ddec_mas, dra_sem_mas, ddec_sem_mas,
    sem_mas, n_pairs, n_measurements, component, component_size)`` where
    ``dra_mas/ddec_mas`` is the CORRECTION to add (``-p_d`` after gauge);
    detectors with no usable measurement are absent.
    """
    if not measurements:
        return {}
    dets = sorted(set(detectors or [])
                  | {m["det_a"] for m in measurements}
                  | {m["det_b"] for m in measurements})
    edges = [(m["det_a"], m["det_b"]) for m in measurements]
    out = {}
    for ci, comp in enumerate(_connected_components(dets, edges)):
        comp = sorted(comp)
        if len(comp) < 2:
            continue
        index = {d: i for i, d in enumerate(comp)}
        ms = [m for m in measurements
              if m["det_a"] in index and m["det_b"] in index]
        if not ms:
            continue
        n_obs, n_det = len(ms), len(comp)
        A = np.zeros((n_obs, n_det))
        for k, m in enumerate(ms):
            A[k, index[m["det_b"]]] = 1.0
            A[k, index[m["det_a"]]] = -1.0
        sol = {}
        for axis, sem_key in (("dra_mas", "dra_sem"), ("ddec_mas", "ddec_sem")):
            y = np.array([m[axis] for m in ms])
            sem = np.maximum(np.array([m[sem_key] for m in ms]),
                             PAIR_SEM_FLOOR_MAS)
            w = 1.0 / sem ** 2
            aw = A * np.sqrt(w)[:, None]
            yw = y * np.sqrt(w)
            cov0 = np.linalg.pinv(aw.T @ aw)
            p = cov0 @ (aw.T @ yw)
            resid = yw - aw @ p
            dof = max(n_obs - (n_det - 1), 1)
            scale = max(float(resid @ resid) / dof, 1.0)
            p = p - np.median(p)          # median gauge
            sol[axis] = (p, np.sqrt(np.maximum(np.diag(cov0) * scale, 0.0)))
        n_pairs = {d: 0 for d in comp}
        n_meas = {d: 0 for d in comp}
        for m in ms:
            for d in (m["det_a"], m["det_b"]):
                n_pairs[d] += int(m["n"])
                n_meas[d] += 1
        p_ra, sem_ra = sol["dra_mas"]
        p_de, sem_de = sol["ddec_mas"]
        for d in comp:
            i = index[d]
            out[d] = dict(
                dra_mas=float(-p_ra[i]), ddec_mas=float(-p_de[i]),
                dra_sem_mas=float(sem_ra[i]), ddec_sem_mas=float(sem_de[i]),
                sem_mas=float(np.hypot(sem_ra[i], sem_de[i])),
                n_pairs=n_pairs[d], n_measurements=n_meas[d],
                component=ci, component_size=len(comp))
    return out


def measure_visit_detector_ties(frames_by_detector, consensus_coords=None,
                                refcat=None, visit_bulk=None,
                                min_pairs=DETECTOR_TIE_MIN_PAIRS,
                                apply_min_mas=DETECTOR_TIE_APPLY_MIN_MAS,
                                nsigma=DETECTOR_TIE_NSIGMA,
                                ref_gross_tol_mas=DETECTOR_REF_GROSS_TOL_MAS,
                                max_pairs_per_detpair=MAX_PAIRS_PER_DETPAIR,
                                context=""):
    """Measure the per-detector tie for every detector of one (visit, filter)
    from the cross-detector overlap network.

    Parameters
    ----------
    frames_by_detector : dict
        detector token (e.g. ``'nrca3'``, ``'nrcalong'``) -> list of
        ``(coords, hist_result_or_None)`` per exposure
        (``group_frames_by_detector`` output).  The optional per-frame
        ``hist_result`` (frame vs the visit consensus, from
        ``build_visit_consensus``) is NOT used for the tie -- it is recorded
        as the ``vs_consensus_diag`` self-cancellation diagnostic (module
        docstring): its magnitude is the DILUTED echo of the placement term
        and must stay well below the network tie.
    consensus_coords : SkyCoord or KDTreeReference or None
        Not used by the network solve (kept for call-site symmetry; the
        consensus is deliberately NOT the tie reference -- see the module
        docstring).
    refcat : dict or None
        ``load_reference_catalog`` output.  When given, each applying
        detector's tie is GROSS cross-checked against VIRAC2
        (``refcat['all']``); Gaia is never used here and nothing fine-gates
        (gc-gaia-frame-not-catalog).
    visit_bulk : tuple or None
        ``(dra_mas, ddec_mas)`` of the visit -> VIRAC2 tie, subtracted from
        each detector's VIRAC2 offset so the cross-check compares
        detector-DIFFERENTIAL terms.  Measured here (per-frame same-star,
        pair-weighted over ALL frames) when None and ``refcat`` is given.

    Returns
    -------
    dict
        detector -> ``dict(detector, dra_mas, ddec_mas, sem_mas, n_pairs,
        n_measurements, component, component_size, apply, refuse_reason,
        vs_consensus_diag, vs_reference)``.  ``apply`` True only when the
        full refusal ladder passes; a refused detector must be left
        uncorrected by the caller.
    """
    del consensus_coords  # documented no-op (self-cancellation; docstring)
    measurements = measure_cross_detector_pairs(
        frames_by_detector, max_pairs_per_detpair=max_pairs_per_detpair,
        context=context)
    solved = solve_detector_network(measurements,
                                    detectors=sorted(frames_by_detector))

    # --- VIRAC2 visit bulk (for the differential cross-check) --------------
    ref_all = refcat.get("all") if refcat is not None else None
    if ref_all is not None and visit_bulk is None:
        vals = []
        for det, frames in sorted(frames_by_detector.items()):
            for coords, _hist in frames:
                m = measure_offset(coords, ref_all, sweep=True,
                                   context=f"{context} {det} vs ref (bulk)")
                if m is None or not m.get("ok") or m.get("swept"):
                    continue
                refined = _samestar_refine(coords, ref_all, m,
                                           context=f"{context} {det} ss (bulk)")
                if refined is not None:
                    vals.append(refined)
        if vals:
            v = np.array([(d, e, n) for d, e, _s1, _s2, n in vals])
            w = v[:, 2] / v[:, 2].sum()
            visit_bulk = (float(np.sum(w * v[:, 0])),
                          float(np.sum(w * v[:, 1])))

    ties = {}
    for det, frames in sorted(frames_by_detector.items()):
        rec = solved.get(det)
        if rec is None:
            rec = dict(dra_mas=float("nan"), ddec_mas=float("nan"),
                       dra_sem_mas=float("nan"), ddec_sem_mas=float("nan"),
                       sem_mas=float("nan"), n_pairs=0, n_measurements=0,
                       component=-1, component_size=1)
        rec = dict(rec)
        rec["detector"] = str(det)
        rec["apply"] = False
        rec["refuse_reason"] = None
        rec["vs_reference"] = None

        # self-cancellation diagnostic: pair-weighted mean of the provided
        # per-frame vs-consensus histogram offsets (NOT a tie measurement)
        diag = [(h["dra"], h["ddec"], h.get("n_peak", 1) or 1)
                for _c, h in frames
                if h is not None and h.get("ok") and not h.get("swept")]
        if diag:
            dv = np.array(diag)
            dw = dv[:, 2] / dv[:, 2].sum()
            rec["vs_consensus_diag"] = dict(
                dra_mas=float(np.sum(dw * dv[:, 0])),
                ddec_mas=float(np.sum(dw * dv[:, 1])), n_frames=len(diag))
        else:
            rec["vs_consensus_diag"] = None

        if rec["n_measurements"] == 0:
            rec["refuse_reason"] = ("no measurable cross-detector overlap "
                                    "(isolated detector or all pairs "
                                    "swept/gross)")
            ties[str(det)] = rec
            continue
        off = float(np.hypot(rec["dra_mas"], rec["ddec_mas"]))
        if rec["n_pairs"] < min_pairs:
            rec["refuse_reason"] = f"n_pairs {rec['n_pairs']} < {min_pairs}"
        elif not np.isfinite(rec["sem_mas"]) \
                or rec["sem_mas"] > DETECTOR_TIE_MAX_SEM_MAS:
            rec["refuse_reason"] = (f"sem {rec['sem_mas']:.2f} mas > "
                                    f"{DETECTOR_TIE_MAX_SEM_MAS} mas floor")
        elif off < apply_min_mas:
            rec["refuse_reason"] = (f"|tie| {off:.2f} mas < apply floor "
                                    f"{apply_min_mas} mas (no correction "
                                    f"needed)")
        elif off < nsigma * rec["sem_mas"]:
            rec["refuse_reason"] = (f"|tie| {off:.2f} mas < {nsigma} x sem "
                                    f"({rec['sem_mas']:.2f} mas) -- not "
                                    f"significant")
        else:
            rec["apply"] = True

        # VIRAC2 GROSS cross-check (diagnostic + spurious-solve catch; can
        # only REFUSE, never substitute its own value, never fine-gate).
        if rec["apply"] and ref_all is not None and visit_bulk is not None:
            vals = []
            for coords, _hist in frames:
                m = measure_offset(coords, ref_all, sweep=True,
                                   context=f"{context} {det} vs ref")
                if m is None or not m.get("ok") or m.get("swept"):
                    continue
                refined = _samestar_refine(coords, ref_all, m,
                                           context=f"{context} {det} vs ref "
                                                   f"same-star")
                if refined is not None:
                    vals.append(refined)
            if vals:
                v = np.array([(d, e, n) for d, e, _s1, _s2, n in vals])
                w = v[:, 2] / v[:, 2].sum()
                # Sign: measure_offset(frame, ref) = (ref - frame) = B - p_d
                # (B = the visit's true bulk offset to the reference, p_d the
                # detector displacement), and visit_bulk = B - mean(p).  So
                # (o_d - bulk) = mean(p) - p_d = the CORRECTION the reference
                # implies for detector d (same sense as the network's -p_d,
                # up to the mean-vs-median gauge difference).
                ref_dra = float(np.sum(w * v[:, 0])) - visit_bulk[0]
                ref_ddec = float(np.sum(w * v[:, 1])) - visit_bulk[1]
                split = float(np.hypot(ref_dra - rec["dra_mas"],
                                       ref_ddec - rec["ddec_mas"]))
                rec["vs_reference"] = dict(
                    dra_mas=ref_dra, ddec_mas=ref_ddec,
                    n_pairs=int(v[:, 2].sum()), n_frames=len(vals),
                    split_mas=split)
                if split > ref_gross_tol_mas:
                    rec["apply"] = False
                    rec["refuse_reason"] = (
                        f"network tie ({rec['dra_mas']:+.2f},"
                        f"{rec['ddec_mas']:+.2f}) vs reference differential "
                        f"({ref_dra:+.2f},{ref_ddec:+.2f}) split "
                        f"{split:.1f} mas > gross tol {ref_gross_tol_mas} mas "
                        f"-- spurious solve or wrong grouping; refusing")
        ties[str(det)] = rec
    return ties


def detector_tie_corrections(ties, visit, filtername, dec_deg, stage="m2"):
    """Offsets-table correction dicts (``update_offsets_table`` /
    ``seed_offsets_table_from_consensus`` schema) for the APPLYING detectors.

    Each correction carries ``exposure=None`` + ``module=<detector>`` -- the
    per-detector row kind (Exposure = BULK_EXPOSURE, Module = detector token)
    that applies to every exposure of that (visit, filter, detector).  The
    sign: ``dra_mas/ddec_mas`` is already the on-sky shift to ADD to the
    detector's frames (the negated, median-gauged network displacement) --
    identical in sense to the per-exposure jitter convention.
    """
    corrections = []
    for det, rec in sorted(ties.items()):
        if not rec.get("apply"):
            continue
        corrections.append(dict(
            visit=visit, exposure=None, module=str(det),
            filtername=filtername,
            dra_onsky_mas=float(rec["dra_mas"]),
            ddec_onsky_mas=float(rec["ddec_mas"]),
            dec_deg=float(dec_deg),
            source=f"{stage} per-detector tie "
                   f"(n={rec['n_pairs']}, sem={rec['sem_mas']:.2f}mas)"))
    return corrections
