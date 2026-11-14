"""Reliability gate for ZEROFRAME saturated-star deblending.

Context (10678 gc-treasury, 2026-10): data-qa measured that most RAW deblend
candidates proposed at the seed stage are not real. Scanning every DQ-SATURATED
blob in all 4040 10678 crf frames (no PSF fit), ~65% of F480M blobs and ~31% of
F212N blobs get split into >=2 centres, but only ~0.5-1.3% of those proposed
centres are confirmed by an INDEPENDENT signal (a second group-0
GROUPDQ-SATURATED core, or a second VIRAC2 source, inside the blob). A
synthetic-single-star test (``deblend_blob_zeroframe`` run on an stpsf PSF with
no second star present) reproduces the dominant ~8-10 px secondary separation
seen in the real data at a high rate once the saturated area is large enough --
that separation is the PSF's own diffraction-spike/sidelobe structure, not a
second star.

PSF-fitting already discards most of this before a split reaches the shipped
catalog: production m12 catalogs keep ~2 (F212N) / ~16 (F480M) DQ-SAT splits
per frame, versus ~23 / ~322 raw candidates per frame at the seed stage.

Reliability of the SHIPPED survivors, independent measurement (2026-10-15):
an earlier precision/recall table in this module compared the gate's own
criterion (g0-core count) against a ground truth that itself included the
g0-core count -- that comparison is circular and has been removed. The
numbers below instead use two signals that share no code or criterion with
this gate: cross-dither repeatability of a shipped child's sky position
across a DIFFERENT exposure (0.05", a real star should reappear; a PSF-sidelobe
artifact should not), and cross-filter confirmation of an F480M child against
the independently-built F212N catalog (0.1", checked against a
rotated-90-degree control that reproduces the same field-crowding baseline
without any real association). Applied to the shipped m12 dqsat split
children of 19 production tiles, split by each child's OWN parent blob's
defect-filtered group-0-core count:

  (a) >=2 defect-filtered group-0 cores (this gate's accept criterion)
  (b) <2 defect-filtered group-0 cores (this gate's reject criterion)

Result (19 tiles, 45-frame/filter random subsample, 774 matched groups; see
``docs/reports/SATSTAR_DEBLEND_GATE_REPORT.md`` for the full table and
methodology): F480M is well-powered (1382 bucketed children; the
cross-filter rotated-90 control confirms at 0.2-0.7%, i.e. an effectively
clean false-positive floor) and bucket (a) and bucket (b) confirm as real at
nearly the same rate -- 98.8% vs 93.1% (and 94.0% vs 94.5% at a looser >=1
threshold) -- so the group-0-core COUNT has essentially no discriminating
power there, and the gate (bucket b) would discard 80.4% of the F480M
children independently confirmed real. F212N's own bucket (a) has only 3
groups in this subsample, too few to read a confirmation rate; the
directional result (41 of 42 independently-confirmed-real F212N children
fall in bucket b) is consistent with F480M's finding but not independently
conclusive at this sample size.

A SEPARATE, better-powered F212N check (prodD shipped m7 catalog, deblend ON
with no gate at all, vs the deblend-OFF v1.9 release, same tile) shows real
damage from ungated deblending that this gate's bucket measurement does not
speak to directly: bad-fit (qfit>1) close "secondaries" among satstar sources
rise from ~7-14% to ~55-61% of all close pairs, and 5 of 30 bright
(2MASS Ks<9.5) benchmark stars pick up a spurious, badly-fit (qfit 0.56-4.55)
companion 0.19-0.35" away that corrupts their photometry by 3-9 mag. F480M
shows none of this in the same tile. See
``docs/reports/SATSTAR_DEBLEND_GATE_REPORT.md`` Section 6 for the full
comparison and Section 7 for how the two checks fit together: this gate's
own group-0-core signal is not shown to discriminate real from fake F480M
splits, but F212N's shipped catalog is shown to need some form of quality
control when deblending runs ungated. ``SATSTAR_DEBLEND_GATE`` stays
default-on pending the recommended follow-up (re-catalog a tile through m7
with the gate on, to check whether it recovers the F212N secondaries/bright-
star damage, and separate that from this campaign's concurrent ePSF/
background changes).

The gate: count distinct connected clusters of group-0 GROUPDQ-SATURATED
pixels (the raw first-read saturation mask; see
``saturated_star_finding._find_group0_saturation_for``, already computed for
the unrelated fit-anchor use and passed into ``get_saturated_stars`` as
``zeroframe_group0_saturated`` -- no new file I/O) overlapping the (dilated)
saturated blob. This is INDEPENDENT of the deblender's own smoothed-ZEROFRAME
peak search: a real multi-star blend saturates >=2 separate cores in the raw
group-0 read; a PSF-sidelobe/diffraction-spike artifact of a single star does
not independently re-saturate at the sidelobe location in group 0.

Two refinements over the first version of this gate:

* **Defect filtering.** A raw group-0-saturated cluster can itself be a
  recurring detector hot/fixed-pattern pixel rather than a real second star's
  core -- data-qa's ``g0_clean.py`` found 97% of raw F212N g0>=2 components
  are exactly this (4989 raw -> 165 after filtering). ``group0_core_positions``
  drops a cluster whose centroid falls in a (detector, 4-px bin) that recurs
  in >=5 distinct frames of the 10678 scan (``_DEFECT_REGISTRY_PATH``, built
  once from data-qa's read-only scan and shipped as a static per-detector
  table -- see ``scripts/reduction/build_satstar_g0_defect_registry.py``).
  Without this, F212N in particular would accept many splits confirmed only
  by a hot pixel.
* **Per-centre matching.** The gate now accepts or rejects each PROPOSED
  centre individually: a centre is kept only if it has its own distinct,
  unmatched group-0 core within ``match_radius_px`` (nearest 1-to-1
  assignment via ``scipy.optimize.linear_sum_assignment``, so one core cannot
  confirm two centres). A component with N>=3 proposed centres and only 2
  confirmed cores now keeps the 2 matched centres and drops the rest, rather
  than keeping all N once the raw core count alone clears the floor.

``group0_saturated is None`` (``SATSTAR_ZEROFRAME_FIT`` off, a missing ramp,
or a GROUPDQ read failure) fails closed -- every proposed split on that frame
collapses to the single blob centroid -- and is counted and logged
per-frame/per-call (``n_gate_unavailable`` in the optional ``gate_diagnostics``
dict passed to ``build_deblended_source_records``), so a run where this
signal is silently absent does not pass as indistinguishable from a clean
"gate ran and rejected" run.
"""
import json
import os

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

#: minimum number of independent, defect-filtered group-0-saturated cores
#: required to accept a >=2-centre deblend split; below this, the split
#: collapses to the single blob centroid (the pre-deblend /
#: zeroframe_deblend=False behaviour).
MIN_GROUP0_CORES = 2

#: dilation (pixels) applied to the blob mask before counting group-0 cores,
#: so a core that saturates in group-0 just outside the (smaller) final-group
#: DQ-SATURATED footprint still counts towards it.
DILATE_PIX = 2

#: minimum area (pixels) for a group-0 cluster to count as a core, matching
#: data-qa's scan convention (``scan_satblend.py``); drops single/double-pixel
#: noise spikes in the group-0 read that are not a real saturated core.
MIN_CORE_AREA = 3

#: default per-detector 4-px-bin fixed-pattern defect registry, built once
#: (read-only) from data-qa's full 4040-frame 10678 seed-stage scan -- see
#: ``scripts/reduction/build_satstar_g0_defect_registry.py``. A (detector,
#: ybin, xbin) entry recurred as a group-0-saturated core position in >=5
#: distinct frames, i.e. it is the SAME detector pixel saturating every visit
#: regardless of dither -- a cosmetic/hot-pixel defect, not a star.
_DEFECT_REGISTRY_PATH = os.path.join(
    os.path.dirname(__file__), 'data', 'satstar_g0_defect_registry.json')
DEFECT_BIN_PX = 4

_ON = {"1", "true", "yes", "on"}
_OFF = {"0", "false", "no", "off"}

_registry_cache = {}


def _env_switch(name, default, env=None):
    """Minimal on/off env-var reader, duplicated from
    ``saturated_star_finding._env_switch`` rather than imported: that module
    imports THIS one (``build_deblended_source_records`` -> gate), so an
    import the other way would be circular. Same semantics: unset/blank ->
    ``default``; 1/true/yes/on or 0/false/no/off (case-insensitive) -> bool;
    anything else raises, so a typo is never silently read as either state."""
    env = os.environ if env is None else env
    raw = env.get(name)
    if raw is None or not str(raw).strip():
        return bool(default)
    val = str(raw).strip().lower()
    if val in _ON:
        return True
    if val in _OFF:
        return False
    raise ValueError(
        f"{name}={raw!r} is not a recognised on/off value; use one of "
        f"{'/'.join(sorted(_ON))} or {'/'.join(sorted(_OFF))} "
        f"(case-insensitive), or leave it unset for the default "
        f"({'on' if default else 'off'})")


def deblend_gate_enabled(env=None):
    """``SATSTAR_DEBLEND_GATE`` (default ON): require independently-matched
    group-0-saturated cores before accepting a multi-centre deblend split.
    Turning this OFF restores the pre-gate behaviour (every proposed split is
    accepted as-is) and is meant only for a deliberate, justified A/B
    comparison -- see the jwst-gc-pipeline CLAUDE.md override policy for the
    astrometry checkpoints, which this follows."""
    return _env_switch('SATSTAR_DEBLEND_GATE', True, env)


def load_defect_registry(path=None, env=None):
    """Load (and cache) the per-detector fixed-pattern defect registry.

    Returns ``{detector: set-of-(ybin, xbin)}``, or ``{}`` if no registry
    file is found (fails OPEN on a missing registry -- i.e. no defect
    filtering, same as before this existed -- never fails closed, since a
    missing registry must not itself suppress real splits).

    ``SATSTAR_G0_DEFECT_REGISTRY`` overrides the default packaged path.
    """
    env = os.environ if env is None else env
    path = path or env.get('SATSTAR_G0_DEFECT_REGISTRY') or _DEFECT_REGISTRY_PATH
    if path in _registry_cache:
        return _registry_cache[path]
    registry = {}
    if os.path.exists(path):
        with open(path) as fh:
            raw = json.load(fh)
        registry = {det: {tuple(b) for b in bins} for det, bins in raw.items()}
    _registry_cache[path] = registry
    return registry


def _is_defect_position(detector, y, x, registry, bin_px=DEFECT_BIN_PX):
    if not registry or detector is None:
        return False
    bins = registry.get(str(detector).lower())
    if not bins:
        return False
    ybin = int(np.floor(y / bin_px))
    xbin = int(np.floor(x / bin_px))
    return (ybin, xbin) in bins


def group0_core_positions(group0_saturated, blob_mask, sl, dilate_px=DILATE_PIX,
                          min_area=MIN_CORE_AREA, detector=None, registry=None):
    """Return the full-frame (y, x) centroid of every distinct group-0
    GROUPDQ-SATURATED cluster overlapping a dilated version of ``blob_mask``
    within slice ``sl``, after dropping any cluster at a known fixed-pattern
    defect position (see ``load_defect_registry``).

    Parameters
    ----------
    group0_saturated : ndarray[bool] or None
        Full-frame group-0 GROUPDQ SATURATED mask, same shape as the frame
        that ``blob_mask`` is drawn from (``_find_group0_saturation_for``).
        ``None`` means the independent signal is unavailable for this frame.
    blob_mask : ndarray[bool]
        Full-frame boolean mask of the saturated connected component being
        tested (e.g. ``sources == label_id``).
    sl : tuple(slice, slice) or None
        The component's bounding-box slice (an ``ndimage.find_objects`` entry);
        counting is restricted to this box. ``None`` -> return ``None``.
    detector, registry : see ``_is_defect_position``. ``registry=None`` loads
        the default packaged registry; pass ``{}`` explicitly to disable
        defect filtering.

    Returns
    -------
    list[(float, float)] or None
        Full-frame (y, x) centroids of the surviving (non-defect) group-0
        clusters, or ``None`` if ``group0_saturated`` or ``sl`` is
        unavailable.
    """
    if group0_saturated is None or sl is None:
        return None
    sub_blob = np.asarray(blob_mask[sl], dtype=bool)
    if not sub_blob.any():
        return []
    dil = ndimage.binary_dilation(sub_blob, iterations=dilate_px)
    sub_g0 = np.asarray(group0_saturated[sl], dtype=bool) & dil
    if not sub_g0.any():
        return []
    lbl, n = ndimage.label(sub_g0)
    if n == 0:
        return []
    areas = ndimage.sum_labels(sub_g0, lbl, np.arange(1, n + 1))
    coms = ndimage.center_of_mass(sub_g0, lbl, np.arange(1, n + 1))
    y0 = sl[0].start
    x0 = sl[1].start
    reg = load_defect_registry() if registry is None else registry
    positions = []
    for area, (cy, cx) in zip(areas, coms):
        if area < min_area:
            continue
        fy, fx = cy + y0, cx + x0
        if _is_defect_position(detector, fy, fx, reg):
            continue
        positions.append((float(fy), float(fx)))
    return positions


def count_group0_cores(group0_saturated, blob_mask, sl, dilate_px=DILATE_PIX,
                       detector=None, registry=None):
    """Number of (non-defect) group-0-saturated cores -- see
    ``group0_core_positions``, which this wraps. Kept for backward
    compatibility / direct count-only callers (including existing tests)."""
    positions = group0_core_positions(group0_saturated, blob_mask, sl,
                                      dilate_px=dilate_px, detector=detector,
                                      registry=registry)
    return None if positions is None else len(positions)


def _match_centers_to_cores(centers, core_positions, radius):
    """1-to-1 nearest assignment of proposed ``centers`` to detected
    ``core_positions`` (globally optimal, ``linear_sum_assignment``, so one
    core cannot be used to confirm two centres). Returns a boolean array,
    one per input centre, True where that centre matched a distinct core
    within ``radius`` pixels."""
    n_c, n_g = len(centers), len(core_positions)
    keep = np.zeros(n_c, dtype=bool)
    if n_c == 0 or n_g == 0:
        return keep
    c = np.asarray(centers, dtype=float)
    g = np.asarray(core_positions, dtype=float)
    dist = np.hypot(c[:, 0, None] - g[None, :, 0], c[:, 1, None] - g[None, :, 1])
    row_ind, col_ind = linear_sum_assignment(dist)
    for r, cidx in zip(row_ind, col_ind):
        if dist[r, cidx] <= radius:
            keep[r] = True
    return keep


def gate_split_centers(centers, com, blob_mask, sl, group0_saturated,
                       min_cores=MIN_GROUP0_CORES, env=None, detector=None,
                       registry=None, fwhm_pix=None, match_radius_px=None):
    """Accept or reject each centre of a proposed >=2-centre deblend split.

    Each proposed centre is matched 1-to-1 against the independently-detected,
    defect-filtered group-0-saturated cores overlapping the blob (see
    ``group0_core_positions``): a centre survives only if it has its own
    distinct core within ``match_radius_px`` (default ~1 FWHM, floor 1.5 px).
    Centres with no matching core are dropped. If fewer than ``min_cores``
    centres survive the match (i.e. < 2 remain), the whole split collapses to
    the single blob centroid ``com`` -- a deblend is attempted only when it is
    both needed (``len(centers) >= 2``) and reliable (independently
    confirmed, per-centre), per the maintainer's request.

    When the independent signal is unavailable for this frame
    (``group0_saturated is None``), the split is REJECTED (fails closed) and
    collapses to ``com``; the caller is expected to count/log this via
    ``n_g0_cores is None`` (see ``build_deblended_source_records``'s
    ``gate_diagnostics``), since a silently-absent signal must stay visible.

    A single-centre result (``len(centers) < 2``, i.e. no split was proposed)
    is returned unchanged -- there is nothing to gate.

    Returns
    -------
    (centers, n_g0_cores) : (list, int or None)
        The (possibly collapsed) centre list, and the measured
        (defect-filtered) group-0-core count (``None`` when the independent
        signal was unavailable), for the caller to log.
    """
    if len(centers) < 2:
        return centers, None
    if not deblend_gate_enabled(env):
        return centers, None
    core_positions = group0_core_positions(
        group0_saturated, blob_mask, sl, detector=detector, registry=registry)
    if core_positions is None:
        return [com], None
    n_g0 = len(core_positions)
    if n_g0 < min_cores:
        return [com], n_g0
    radius = match_radius_px
    if radius is None:
        radius = max(1.5, 0.5 * float(fwhm_pix)) if fwhm_pix else 3.0
    matched = _match_centers_to_cores(centers, core_positions, radius)
    kept = [c for c, m in zip(centers, matched) if m]
    if len(kept) < min_cores:
        return [com], n_g0
    return kept, n_g0
