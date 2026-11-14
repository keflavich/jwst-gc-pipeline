"""Trailed-exposure reading for the m2 checkpoint (issue #733).

The visit checkpoint grades an exposure only as an offset against its visit
consensus.  A trailed exposure (spacecraft drift during the integration) does
not carry a rigid offset: its stars are smeared along one sky direction, the
fitted centroids move with the smear, and the result reads as a displacement of
the whole frame.  gc2211 o023 is the case that named this (#484, #493): four
exposures with tracking errors, a ~150 mas per-exposure "split", and a
correction the offsets table could not express.  The offset estimators have no
way to say "this exposure is bad data", so this module measures the smear
directly from the pixels.

Two halves:

* :func:`measure_exposure_shape` -- per detector, the median adaptive
  Gaussian-weighted second moments of the brightest isolated well-fit stars:
  sigma, axis ratio and the major-axis position angle on the sky.
* :func:`grade_visit_trails` -- compares each exposure with its siblings of the
  same visit, visit group, filter and detector.  An exposure is flagged when the
  excess over its siblings is over threshold on at least half of its measured
  detectors AND the sky position angle agrees across them: a drift is common to
  the focal plane, an optical or single-detector defect is not.

Adaptive weighted moments are used because plain box moments truncate a trail.
``reduction/exposure_exclusions.py`` records that 19x19 and 41x41 px box moments
could not separate gc2211 o023 exposure 4 from a clean control.

STATUS: RECORD ONLY.  The thresholds below are provisional and have not been
checked against the o023 ``_cal`` frames, o049 exposure 4 or o028 exposure 2.
The verdict is written to the checkpoint record's per-visit ``trail`` block and
printed; it gates nothing.  Turning it into a named refusal, and the check
against the filter's nominal PSF that would see a visit in which every exposure
is trailed, follow once the distribution over the fields on disk has set the
thresholds.  Discarding an exposure stays a manual edit of
``reduction/exposure_exclusions.py``; nothing here writes to it.
"""
import math
import os

import numpy as np
from astropy.io import fits

from .column_utils import _best_available_xy
from .visit_consensus import exposure_key, select_reliable_stars
from ..frame_wcs import frame_wcs
from ..reduction.exposure_exclusions import exposure_stem

#: Provisional thresholds (see the module docstring).  The issue's own
#: measurement put the excess sigma of the trailed o023 exposures at 0.93-1.57
#: px; these sit below that and above the exposure-to-exposure scatter expected
#: of a clean visit.
TRAIL_EXCESS_SIGMA_PX = 0.5
#: Axis ratio b/a of a smeared exposure is lower than its siblings' by at least
#: this much.  Requiring it separates a trail from a plain defocus.
TRAIL_AXIS_RATIO_DROP = 0.05
#: Length of the mean of exp(2i * PA) over the over-threshold detectors; 1 means
#: the major axes are parallel on the sky.
TRAIL_PA_COHERENCE_MIN = 0.8
#: A detector is measured only from at least this many stars.
TRAIL_MIN_STARS = 30
#: A comparison needs at least this many exposures of the detector, so that an
#: exposure has two siblings to be compared with.
TRAIL_MIN_EXPOSURES = 3
#: At least this many detectors must be over threshold to call a drift: one
#: detector cannot show that the smear is common to the focal plane.
TRAIL_MIN_FLAGGED_DETECTORS = 2

#: Stamp half-size in pixels (stamps are 2 * half + 1 square).
STAMP_HALF_PX = 14

#: Exposures a person has inspected and kept despite a flag, ``{stem: reason}``.
#: The stem is ``reduction.exposure_exclusions.exposure_stem`` of the frame.  An
#: entry needs a reason that names what was seen in the pixels, in the style of
#: ``m2_correction_floors.py``.  Empty until the calibration run says which
#: exposures the check flags.
TRAIL_REVIEWED = {}


def adaptive_moments(stamp, x0, y0, sigma0=2.0, n_iter=40, tol=1e-4):
    """Adaptive Gaussian-weighted second moments of one background-subtracted stamp.

    The weight is a Gaussian with the covariance the iteration is converging to
    (Bernstein & Jarvis 2002), so the window follows the star's own size and
    shape rather than cutting it with a box.  Returns ``(cx, cy, Mxx, Myy,
    Mxy)`` in stamp pixel coordinates, or ``None`` when the iteration does not
    settle on a positive flux and a positive-definite covariance.
    """
    ny, nx = stamp.shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    cx, cy = float(x0), float(y0)
    mxx = myy = float(sigma0) ** 2
    mxy = 0.0
    for _ in range(n_iter):
        det = mxx * myy - mxy ** 2
        if not np.isfinite(det) or det <= 0:
            return None
        dx, dy = xx - cx, yy - cy
        q = (myy * dx * dx - 2.0 * mxy * dx * dy + mxx * dy * dy) / det
        wi = np.exp(-0.5 * q) * stamp
        flux = wi.sum()
        if not np.isfinite(flux) or flux <= 0:
            return None
        cx_new = (wi * xx).sum() / flux
        cy_new = (wi * yy).sum() / flux
        dx, dy = xx - cx_new, yy - cy_new
        # A Gaussian of covariance C seen through a Gaussian window of the same
        # covariance has weighted second moment C / 2.
        mxx_new = 2.0 * (wi * dx * dx).sum() / flux
        myy_new = 2.0 * (wi * dy * dy).sum() / flux
        mxy_new = 2.0 * (wi * dx * dy).sum() / flux
        change = max(abs(mxx_new + myy_new - mxx - myy) / max(mxx + myy, 1e-12),
                     math.hypot(cx_new - cx, cy_new - cy) * 1e-2)
        cx, cy, mxx, myy, mxy = cx_new, cy_new, mxx_new, myy_new, mxy_new
        if change < tol:
            break
    else:
        return None
    det = mxx * myy - mxy ** 2
    if not np.isfinite(det) or det <= 0:
        return None
    return cx, cy, mxx, myy, mxy


def _shape_from_moments(mxx, myy, mxy):
    """``(sigma, axis_ratio, theta)`` of a second-moment matrix.

    ``sigma`` is the root of the mean eigenvalue, ``axis_ratio`` is b/a (at
    most 1) and ``theta`` the major-axis angle from +x towards +y in radians.
    """
    half_sum = 0.5 * (mxx + myy)
    root = math.hypot(0.5 * (mxx - myy), mxy)
    lam_major, lam_minor = half_sum + root, half_sum - root
    if lam_minor <= 0:
        return None
    theta = 0.5 * math.atan2(2.0 * mxy, mxx - myy)
    return math.sqrt(half_sum), math.sqrt(lam_minor / lam_major), theta


def _sky_position_angle_deg(wcs, x, y, theta, step=1.0):
    """Position angle (east of north, modulo 180 deg) of a pixel-frame axis."""
    p0 = wcs.pixel_to_world(x, y)
    p1 = wcs.pixel_to_world(x + step * math.cos(theta), y + step * math.sin(theta))
    return float(p0.position_angle(p1).deg % 180.0)


def _axial_mean_deg(angles_deg, weights=None):
    """``(mean angle modulo 180, resultant length)`` of axial data."""
    a = np.radians(2.0 * np.asarray(angles_deg, dtype=float))
    w = np.ones_like(a) if weights is None else np.asarray(weights, dtype=float)
    if not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones_like(a)
    c, s = (w * np.cos(a)).sum() / w.sum(), (w * np.sin(a)).sum() / w.sum()
    return float((np.degrees(math.atan2(s, c)) / 2.0) % 180.0), float(math.hypot(c, s))


def _select_stars(table, half, n_max):
    """Row indices and pixel positions of isolated, bright, reliable stars."""
    x, y = _best_available_xy(table)
    x, y = np.asarray(x, float), np.asarray(y, float)
    finite = np.isfinite(x) & np.isfinite(y)
    keep = select_reliable_stars(table) & finite
    if not keep.any():
        return np.array([], dtype=int), x, y
    # Isolation is judged against EVERY catalogued source, not only the
    # reliable ones: a faint neighbour inside the stamp still biases a moment.
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([x[finite], y[finite]]))
    isolated = np.zeros(len(table), dtype=bool)
    for i in np.flatnonzero(keep):
        # Two entries within the stamp radius: the star itself and a neighbour.
        isolated[i] = len(tree.query_ball_point([x[i], y[i]], r=half)) <= 1
    cand = np.flatnonzero(keep & isolated)
    if "flux_fit" in table.colnames:
        flux = np.asarray(table["flux_fit"], dtype=float)[cand]
        cand = cand[np.argsort(-flux)]
    return cand[:n_max], x, y


def measure_exposure_shape(table, frame_path=None, n_max=400, half=STAMP_HALF_PX,
                           min_stars=TRAIL_MIN_STARS):
    """Median star shape of one per-frame (one-detector) catalog.

    ``frame_path`` defaults to ``table.meta['FILENAME']`` and may name a crf or
    a ``_cal`` frame.  Returns a JSON-able dict with ``status`` ``"ok"`` and
    ``sigma`` (px), ``axis_ratio``, ``pa_sky_deg``, ``pa_coherence`` and
    ``n_stars``, or ``status`` ``"cannot_measure"`` with a ``reason``.  A frame
    that is missing or unreadable is a reason, not an exception: the reading is
    advisory and must not stop the checkpoint that carries it.
    """
    frame_path = frame_path or table.meta.get("FILENAME") or table.meta.get("filename")
    if not frame_path:
        return dict(status="cannot_measure", reason="no FILENAME in the catalog meta")
    frame_path = str(frame_path)
    if not os.path.exists(frame_path):
        return dict(status="cannot_measure", reason=f"frame not on disk: {frame_path}")
    try:
        cand, x, y = _select_stars(table, half, n_max)
    except KeyError as ex:
        return dict(status="cannot_measure", reason=f"no pixel positions: {ex}")
    if len(cand) < min_stars:
        return dict(status="cannot_measure", n_stars=int(len(cand)),
                    reason=f"{len(cand)} isolated reliable stars (< {min_stars})")
    try:
        with fits.open(frame_path, memmap=False) as hdul:
            data = np.asarray(hdul["SCI"].data, dtype=float)
            wcs = frame_wcs(hdul)
    except (OSError, KeyError, ValueError) as ex:
        return dict(status="cannot_measure",
                    reason=f"could not read {frame_path}: {type(ex).__name__}: {ex}")

    ny, nx = data.shape
    sigmas, ratios, thetas, xs, ys = [], [], [], [], []
    for i in cand:
        ix, iy = int(round(x[i])), int(round(y[i]))
        if ix - half < 0 or iy - half < 0 or ix + half >= nx or iy + half >= ny:
            continue
        stamp = data[iy - half:iy + half + 1, ix - half:ix + half + 1]
        if not np.isfinite(stamp).all():
            continue
        edge = np.concatenate([stamp[:2].ravel(), stamp[-2:].ravel(),
                               stamp[2:-2, :2].ravel(), stamp[2:-2, -2:].ravel()])
        res = adaptive_moments(stamp - np.median(edge), x[i] - ix + half,
                               y[i] - iy + half)
        if res is None:
            continue
        shape = _shape_from_moments(*res[2:])
        if shape is None:
            continue
        sigmas.append(shape[0])
        ratios.append(shape[1])
        thetas.append(shape[2])
        xs.append(x[i])
        ys.append(y[i])
    if len(sigmas) < min_stars:
        return dict(status="cannot_measure", n_stars=int(len(sigmas)),
                    reason=f"{len(sigmas)} stars with a converged moment (< {min_stars})")

    ratios, thetas = np.asarray(ratios), np.asarray(thetas)
    # Weight by elongation: a round star has no defined axis.
    theta_mean, coherence = _axial_mean_deg(np.degrees(thetas), 1.0 - ratios + 1e-6)
    xc, yc = float(np.median(xs)), float(np.median(ys))
    pa = _sky_position_angle_deg(wcs, xc, yc, math.radians(theta_mean))
    return dict(status="ok", sigma=float(np.median(sigmas)),
                axis_ratio=float(np.median(ratios)), pa_sky_deg=pa,
                pa_coherence=coherence, n_stars=int(len(sigmas)))


def _exposure_id(key):
    """``(group, exposure)`` of an ``exposure_key``: the group is visit, visit
    group and filter, the unit whose exposures are compared with one another."""
    visit, exposure, _module, filtername, vgroup = key
    return (visit, vgroup, filtername), exposure


def grade_visit_trails(measured, reviewed=None):
    """Grade exposures against their siblings.

    ``measured`` is a list of ``dict(key=exposure_key, stem=str|None,
    shape=<measure_exposure_shape result>)``, one per detector frame.  Returns a
    list with one dict per (visit, visit group, filter, exposure): ``verdict``
    is ``"ok"``, ``"flagged"``, ``"reviewed"`` (flagged, and the stem is in
    ``reviewed``) or ``"cannot_grade"``, with a ``reason`` and the per-detector
    numbers behind it.
    """
    reviewed = TRAIL_REVIEWED if reviewed is None else reviewed
    by_exposure = {}
    for rec in measured:
        group, exposure = _exposure_id(rec["key"])
        by_exposure.setdefault(group, {}).setdefault(exposure, []).append(rec)

    out = []
    for group in sorted(by_exposure, key=str):
        exposures = by_exposure[group]
        # detector -> {exposure: shape} for the readings that measured something
        per_detector = {}
        for exposure, recs in exposures.items():
            for rec in recs:
                if rec["shape"].get("status") == "ok":
                    per_detector.setdefault(rec["key"][2], {})[exposure] = rec["shape"]
        for exposure in sorted(exposures, key=str):
            recs = exposures[exposure]
            stems = sorted({r["stem"] for r in recs if r.get("stem")})
            detectors = []
            for rec in recs:
                module = rec["key"][2]
                shape = rec["shape"]
                row = dict(module=module, status=shape.get("status"),
                           n_stars=shape.get("n_stars"))
                siblings = [s for e, s in per_detector.get(module, {}).items()
                            if e != exposure]
                if shape.get("status") != "ok":
                    row["reason"] = shape.get("reason")
                elif len(siblings) + 1 < TRAIL_MIN_EXPOSURES:
                    row.update(status="cannot_grade",
                               reason=f"{len(siblings) + 1} exposure(s) measured on "
                                      f"this detector (< {TRAIL_MIN_EXPOSURES})")
                else:
                    sib_sigma = float(np.median([s["sigma"] for s in siblings]))
                    sib_ratio = float(np.median([s["axis_ratio"] for s in siblings]))
                    excess = math.sqrt(max(shape["sigma"] ** 2 - sib_sigma ** 2, 0.0))
                    drop = sib_ratio - shape["axis_ratio"]
                    row.update(sigma=shape["sigma"], sibling_sigma=sib_sigma,
                               excess_sigma=excess, axis_ratio=shape["axis_ratio"],
                               sibling_axis_ratio=sib_ratio, axis_ratio_drop=drop,
                               pa_sky_deg=shape["pa_sky_deg"],
                               over_threshold=bool(excess >= TRAIL_EXCESS_SIGMA_PX
                                                   and drop >= TRAIL_AXIS_RATIO_DROP))
                detectors.append(row)

            graded = [d for d in detectors if "over_threshold" in d]
            over = [d for d in graded if d["over_threshold"]]
            entry = dict(visit=group[0], vgroup=group[1], filtername=group[2],
                         exposure=exposure, stems=stems, detectors=detectors,
                         n_graded=len(graded), n_over_threshold=len(over),
                         pa_coherence=None)
            if len(graded) < TRAIL_MIN_FLAGGED_DETECTORS:
                entry.update(verdict="cannot_grade",
                             reason=f"{len(graded)} detector(s) could be compared "
                                    f"with siblings (< {TRAIL_MIN_FLAGGED_DETECTORS})")
            elif len(over) < max(TRAIL_MIN_FLAGGED_DETECTORS, math.ceil(len(graded) / 2)):
                entry.update(verdict="ok",
                             reason=f"{len(over)} of {len(graded)} graded detectors "
                                    f"over threshold")
            else:
                _, coherence = _axial_mean_deg([d["pa_sky_deg"] for d in over])
                entry["pa_coherence"] = coherence
                if coherence < TRAIL_PA_COHERENCE_MIN:
                    entry.update(verdict="ok",
                                 reason=f"{len(over)} of {len(graded)} detectors over "
                                        f"threshold but the sky position angles "
                                        f"disagree (coherence {coherence:.2f} < "
                                        f"{TRAIL_PA_COHERENCE_MIN})")
                else:
                    why = [reviewed[s] for s in stems if s in reviewed]
                    if why:
                        entry.update(verdict="reviewed", reason="; ".join(why))
                    else:
                        entry.update(verdict="flagged",
                                     reason=f"{len(over)} of {len(graded)} detectors "
                                            f"over threshold with a common sky "
                                            f"position angle (coherence "
                                            f"{coherence:.2f})")
            out.append(entry)
    return out


def assess_visit_trails(tables, measure=measure_exposure_shape, reviewed=None):
    """The checkpoint's per-visit ``trail`` block for a list of per-frame catalogs.

    ``measure`` is injectable so that tests need no pixels.  The block is
    JSON-able and says that it is advisory.
    """
    measured = []
    for tbl in tables:
        frame = tbl.meta.get("FILENAME") or tbl.meta.get("filename")
        measured.append(dict(key=exposure_key(tbl),
                             stem=exposure_stem(frame) if frame else None,
                             shape=measure(tbl)))
    exposures = grade_visit_trails(measured, reviewed=reviewed)
    return dict(
        mode="record_only",
        thresholds=dict(excess_sigma_px=TRAIL_EXCESS_SIGMA_PX,
                        axis_ratio_drop=TRAIL_AXIS_RATIO_DROP,
                        pa_coherence_min=TRAIL_PA_COHERENCE_MIN,
                        min_stars=TRAIL_MIN_STARS,
                        min_exposures=TRAIL_MIN_EXPOSURES,
                        min_flagged_detectors=TRAIL_MIN_FLAGGED_DETECTORS,
                        provisional=True),
        n_measured=sum(1 for m in measured if m["shape"].get("status") == "ok"),
        n_frames=len(measured),
        flagged=[e["stems"] or [e["exposure"]] for e in exposures
                 if e["verdict"] == "flagged"],
        exposures=exposures)
