"""Per-(visit, detector) translation tie against the visit-wide consensus.

Targets the residual the GDC distortion-swap experiment isolated (2026-07):
after the DVA inter-detector correction, cross-module (A/B seam) same-star
offsets of 2.7-5.4 mas remain that are detector-PAIR- and filter-dependent
(brick F212N nrca3-nrcb4 = -5.4 mas Dec; nrca4-nrcb3 = +2.5/-2.5 mas) while
the within-detector affine residual is <1 mas rms.  That is inter-detector
rigid PLACEMENT (SIAF-class, static 1-2.5 mas per detector per the network
self-calibration), not distortion -- and neither the module-locked tables
(one shift for all detectors) nor the per-exposure jitter rows (per-frame
2 mas gate at a per-frame ~1-2 mas noise floor) can express it.

Design doc: ``DETECTOR_TIE_DESIGN.md`` (read it before changing tolerances or
the refusal ladder).  Key properties:

* the tie is combined over ALL exposures of a (visit, filter, detector), each
  measured same-star against the INTERNAL visit-wide consensus (all
  detectors, ``build_visit_consensus``) -> 10^3-10^5 total pairs -> sem well
  below the 2-5 mas term;
* per-frame measurement is the sanctioned histogram-detection + same-star
  refinement pair (``measure_offset`` swept -> giant-cell
  ``local_residual_map``; refinement is legal only after the verified small
  un-swept histogram tie).  NEVER NN-median (CLAUDE.md rule #1).  Refinement
  runs per FRAME, not on a pooled multi-exposure cloud: pooling duplicates
  every star ~n_exp times, which makes matched pairs ambiguous by
  construction (``local_residual_map``'s partner-uniqueness guard rejects
  them);
* VIRAC2 is only a GROSS, non-blocking cross-check on the DIFFERENTIAL term
  (GC rule: Gaia is the frame, VIRAC2 the catalog, and neither may veto a
  coherent internal tie at the mas level);
* a detector below the floor (n_pairs, sem, contrast, significance) is
  REFUSED: left uncorrected with a loud warning, never guessed;
* translation-only (v1): the measured seam terms are translations; a full
  6-param affine needs a new GWCS apply mechanism and is deferred (see the
  design doc, section 2).

Gating: default OFF; ``ASTROM_M2_PER_DETECTOR_TIE=1`` enables the hook in
``astrometry_checkpoint.run_visit_checkpoint`` at correcting (m2) stages
only.  Corrections are per-detector offsets-table rows
(Exposure = BULK_EXPOSURE, Module = detector token) applied through the
existing consensus-table + m2 stop/regenerate flow.
"""
import os

import numpy as np

from .astrometry_offsets import (
    GlobalTieNotVerifiedError, KDTreeReference, local_residual_map,
    measure_offset,
)
from astropy import units as u

# Environment flag that enables the per-detector tie at the m2 checkpoint.
PER_DETECTOR_TIE_ENV = "ASTROM_M2_PER_DETECTOR_TIE"

# --- refusal ladder (justifications in DETECTOR_TIE_DESIGN.md, section 2) ---
# Minimum combined same-star matched pairs: below ~50 the sem of a ~5
# mas-scatter population is >~0.7 mas and a 2 mas term is barely 3 sigma.
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
# internal measurement; catches only a spurious peak / wrong grouping.
DETECTOR_REF_GROSS_TOL_MAS = 15.0

# Per-frame floor for the giant-cell refinement (the detector-level floor is
# DETECTOR_TIE_MIN_PAIRS on the combined pairs).
FRAME_MIN_PAIRS = 10

SAMESTAR_MATCH_RADIUS = 0.3 * u.arcsec


def per_detector_tie_enabled():
    """True when the opt-in env flag is set (default OFF)."""
    return os.environ.get(PER_DETECTOR_TIE_ENV, "").strip() == "1"


def _samestar_refine(coords, reference, hist_result, min_pairs=FRAME_MIN_PAIRS,
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


def measure_detector_tie(frames, consensus_ref, min_pairs=DETECTOR_TIE_MIN_PAIRS,
                         context=""):
    """Translation tie of one detector onto the visit consensus, combined over
    the detector's exposures.

    Each frame is measured same-star against the consensus (swept histogram
    detection, then giant-cell ``local_residual_map`` refinement -- the
    sanctioned pair; per FRAME so matched pairs stay unambiguous, see the
    module docstring), then the per-frame ties are combined with pair-count
    weights.  The reported ``sem_mas`` is the LARGER of the propagated
    within-frame error and the frame-to-frame scatter of the per-frame ties
    (the latter absorbs real per-exposure jitter, which is a genuine
    uncertainty on a static placement term).

    Parameters
    ----------
    frames : list of (coords, hist_result_or_None)
        One entry per exposure of this (visit, filter, detector):
        the frame's reliable-star SkyCoord and, optionally, its
        already-measured ``measure_offset(frame, consensus)`` result (e.g.
        ``vs_consensus`` from ``build_visit_consensus``) -- measured here
        when None.
    consensus_ref : SkyCoord or KDTreeReference
        The visit-wide (ALL-detector) consensus.

    Returns
    -------
    dict
        ``dict(dra_mas, ddec_mas, sem_mas, dra_sem_mas, ddec_sem_mas,
        n_pairs, n_frames, n_frames_refined, contrast, swept, measurable,
        refuse_reason, frame_ties)``.  ``measurable`` False (with
        ``refuse_reason``) when the tie cannot be measured to the module's
        standard; the caller must then leave the detector uncorrected.
    """
    if not isinstance(consensus_ref, KDTreeReference):
        consensus_ref = KDTreeReference(consensus_ref)
    out = dict(dra_mas=float("nan"), ddec_mas=float("nan"),
               sem_mas=float("nan"), dra_sem_mas=float("nan"),
               ddec_sem_mas=float("nan"), n_pairs=0, n_frames=len(frames),
               n_frames_refined=0, contrast=float("nan"), swept=False,
               measurable=False, refuse_reason=None, frame_ties=[])
    per_frame = []
    contrasts = []
    for i, (coords, hist) in enumerate(frames):
        fctx = f"{context} frame {i}"
        if hist is None:
            hist = measure_offset(coords, consensus_ref, sweep=True,
                                  context=f"{fctx} detection")
        if hist is None or not hist.get("ok"):
            continue
        contrasts.append(float(hist["contrast"]))
        if hist.get("swept"):
            # a grossly misplaced frame is the per-exposure machinery's
            # problem (im0 alignment error), not a placement trim
            out["swept"] = True
            out["refuse_reason"] = (
                f"frame {i} tie only found by window sweep "
                f"({hist['off']:.0f} mas) -- gross misalignment, not a "
                f"placement term")
            return out
        refined = _samestar_refine(coords, consensus_ref, hist,
                                   context=f"{fctx} same-star")
        if refined is None:
            continue
        dra, ddec, dra_sem, ddec_sem, n = refined
        per_frame.append((dra, ddec, dra_sem, ddec_sem, n))
        out["frame_ties"].append(dict(frame=i, dra_mas=dra, ddec_mas=ddec,
                                      dra_sem=dra_sem, ddec_sem=ddec_sem, n=n))
    if contrasts:
        out["contrast"] = float(np.median(contrasts))
    if not per_frame:
        out["refuse_reason"] = ("no frame could be same-star tied to the "
                                "consensus")
        return out
    arr = np.array([(d, e, s1, s2, n) for d, e, s1, s2, n in per_frame])
    n_tot = int(arr[:, 4].sum())
    w = arr[:, 4] / n_tot
    dra = float(np.sum(w * arr[:, 0]))
    ddec = float(np.sum(w * arr[:, 1]))
    # within-frame propagated error of the weighted mean
    dra_sem_w = float(np.sqrt(np.sum((w * arr[:, 2]) ** 2)))
    ddec_sem_w = float(np.sqrt(np.sum((w * arr[:, 3]) ** 2)))
    # frame-to-frame scatter term (>=2 frames): absorbs per-exposure jitter
    nf = len(per_frame)
    if nf >= 2:
        dra_sc = float(np.std(arr[:, 0], ddof=1) / np.sqrt(nf))
        ddec_sc = float(np.std(arr[:, 1], ddof=1) / np.sqrt(nf))
    else:
        dra_sc = ddec_sc = 0.0
    dra_sem = max(dra_sem_w, dra_sc)
    ddec_sem = max(ddec_sem_w, ddec_sc)
    out.update(dra_mas=dra, ddec_mas=ddec, dra_sem_mas=dra_sem,
               ddec_sem_mas=ddec_sem,
               sem_mas=float(np.hypot(dra_sem, ddec_sem)),
               n_pairs=n_tot, n_frames_refined=nf)
    if n_tot < min_pairs:
        out["refuse_reason"] = f"n_pairs {n_tot} < {min_pairs}"
        return out
    if not np.isfinite(out["sem_mas"]) or out["sem_mas"] > DETECTOR_TIE_MAX_SEM_MAS:
        out["refuse_reason"] = (f"sem {out['sem_mas']:.2f} mas > "
                                f"{DETECTOR_TIE_MAX_SEM_MAS} mas floor")
        return out
    out["measurable"] = True
    return out


def group_frames_by_detector(exposure_entries):
    """Group per-frame entries by detector token.

    ``exposure_entries``: iterable of ``(key, coords, hist_result_or_None)``
    where ``key`` is the ``exposure_key`` tuple
    ``(visit, exposure, module, filter)`` (``module`` is the DETECTOR token
    of a per-frame catalog), ``coords`` the frame's reliable-star SkyCoord,
    and ``hist_result`` the frame's already-measured vs-consensus
    ``measure_offset`` result when available.  Returns
    detector -> list of ``(coords, hist_result_or_None)``.
    """
    by_det = {}
    for key, coords, hist in exposure_entries:
        by_det.setdefault(str(key[2]).lower(), []).append((coords, hist))
    return by_det


def measure_visit_detector_ties(frames_by_detector, consensus_coords,
                                refcat=None, visit_bulk=None,
                                min_pairs=DETECTOR_TIE_MIN_PAIRS,
                                apply_min_mas=DETECTOR_TIE_APPLY_MIN_MAS,
                                nsigma=DETECTOR_TIE_NSIGMA,
                                ref_gross_tol_mas=DETECTOR_REF_GROSS_TOL_MAS,
                                context=""):
    """Measure the per-detector tie for every detector of one (visit, filter).

    Parameters
    ----------
    frames_by_detector : dict
        detector token (e.g. ``'nrca3'``, ``'nrcalong'``) -> list of
        ``(coords, hist_result_or_None)`` per exposure
        (``group_frames_by_detector`` output).
    consensus_coords : SkyCoord or KDTreeReference
        Visit-wide consensus positions (ALL detectors,
        ``build_visit_consensus``).
    refcat : dict or None
        ``load_reference_catalog`` output.  When given, each applying
        detector's tie is GROSS cross-checked against VIRAC2
        (``refcat['all']``); Gaia is never used here and nothing fine-gates
        (gc-gaia-frame-not-catalog).
    visit_bulk : tuple or None
        ``(dra_mas, ddec_mas)`` of the visit consensus -> VIRAC2 tie.
        Subtracted from the detector's VIRAC2 offset so the cross-check
        compares detector-DIFFERENTIAL terms.  Measured here (same-star, from
        the consensus) when None and ``refcat`` is given.

    Returns
    -------
    dict
        detector -> the ``measure_detector_tie`` record plus ``detector``,
        ``apply`` (bool -- passed the full refusal ladder incl.
        magnitude/significance/reference gross check) and the
        ``vs_reference`` diagnostic.
    """
    consensus_ref = consensus_coords if isinstance(consensus_coords, KDTreeReference) \
        else KDTreeReference(consensus_coords)

    ref_all = None
    if refcat is not None and refcat.get("all") is not None:
        ref_all = refcat["all"]
        if visit_bulk is None:
            bulk_hist = measure_offset(consensus_ref.coords, ref_all, sweep=True,
                                       context=f"{context} visit bulk vs ref")
            if bulk_hist is not None and bulk_hist.get("ok") \
                    and not bulk_hist.get("swept"):
                refined = _samestar_refine(
                    consensus_ref.coords, ref_all, bulk_hist,
                    min_pairs=min_pairs,
                    context=f"{context} visit bulk same-star")
                if refined is not None:
                    visit_bulk = (refined[0], refined[1])

    ties = {}
    for det, frames in sorted(frames_by_detector.items()):
        dctx = f"{context} {det}"
        rec = measure_detector_tie(frames, consensus_ref, min_pairs=min_pairs,
                                   context=dctx)
        rec["detector"] = str(det)
        rec["vs_reference"] = None
        rec["apply"] = False
        if rec["measurable"]:
            off = float(np.hypot(rec["dra_mas"], rec["ddec_mas"]))
            if off < apply_min_mas:
                rec["refuse_reason"] = (f"|tie| {off:.2f} mas < apply floor "
                                        f"{apply_min_mas} mas (no correction "
                                        f"needed)")
            elif off < nsigma * rec["sem_mas"]:
                rec["refuse_reason"] = (f"|tie| {off:.2f} mas < {nsigma} x sem "
                                        f"({rec['sem_mas']:.2f} mas) -- not "
                                        f"significant")
            else:
                rec["apply"] = True

        # VIRAC2 GROSS cross-check (diagnostic + spurious-peak catch; can only
        # REFUSE, never substitute its own value, never fine-gate).  Measured
        # per frame same-star against the reference, combined pair-weighted --
        # the same no-pooled-duplicates rule as the internal tie.
        if rec["apply"] and ref_all is not None and visit_bulk is not None:
            ref_frames = []
            for coords, _ in frames:
                rhist = measure_offset(coords, ref_all, sweep=True,
                                       context=f"{dctx} vs reference")
                if rhist is None or not rhist.get("ok") or rhist.get("swept"):
                    continue
                refined = _samestar_refine(coords, ref_all, rhist,
                                           context=f"{dctx} vs reference "
                                                   f"same-star")
                if refined is not None:
                    ref_frames.append(refined)
            if ref_frames:
                rarr = np.array([(d, e, n) for d, e, _s1, _s2, n in ref_frames])
                rw = rarr[:, 2] / rarr[:, 2].sum()
                ref_dra = float(np.sum(rw * rarr[:, 0])) - visit_bulk[0]
                ref_ddec = float(np.sum(rw * rarr[:, 1])) - visit_bulk[1]
                split = float(np.hypot(ref_dra - rec["dra_mas"],
                                       ref_ddec - rec["ddec_mas"]))
                rec["vs_reference"] = dict(
                    dra_mas=ref_dra, ddec_mas=ref_ddec,
                    n_pairs=int(rarr[:, 2].sum()), n_frames=len(ref_frames),
                    split_mas=split)
                if split > ref_gross_tol_mas:
                    rec["apply"] = False
                    rec["refuse_reason"] = (
                        f"internal tie ({rec['dra_mas']:+.2f},"
                        f"{rec['ddec_mas']:+.2f}) vs reference differential "
                        f"({ref_dra:+.2f},{ref_ddec:+.2f}) split "
                        f"{split:.1f} mas > gross tol {ref_gross_tol_mas} mas "
                        f"-- spurious peak or wrong grouping; refusing")
        ties[str(det)] = rec
    return ties


def detector_tie_corrections(ties, visit, filtername, dec_deg, stage="m2"):
    """Offsets-table correction dicts (``update_offsets_table`` /
    ``seed_offsets_table_from_consensus`` schema) for the APPLYING detectors.

    Each correction carries ``exposure=None`` + ``module=<detector>`` -- the
    per-detector row kind (Exposure = BULK_EXPOSURE, Module = detector token)
    that applies to every exposure of that (visit, filter, detector).  The
    sign: the tie is (consensus - detector), i.e. the on-sky shift to ADD to
    the detector's frames -- identical to the per-exposure jitter convention.
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
