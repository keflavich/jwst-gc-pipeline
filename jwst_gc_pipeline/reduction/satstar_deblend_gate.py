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
per frame, versus ~23 / ~322 raw candidates per frame at the seed stage. An
independent check of those SHIPPED survivors (cross-dither repeatability
across different exposures, and cross-filter confirmation of F480M splits
against the F212N catalog, both validated against a rotated-90-degree control
that rules out field-crowding coincidence) finds ~78-91% of them positionally
real. So the PSF fit is already the dominant filter; this module adds a
cheap, independently-validated gate that catches the remaining unconfirmed
minority, and -- more importantly -- stops the deblender from attempting a
split at all when nothing independent supports it, instead of relying on the
PSF fit to clean up afterwards.

The gate: count distinct connected clusters of group-0 GROUPDQ-SATURATED
pixels (the raw first-read saturation mask; see
``saturated_star_finding._find_group0_saturation_for``, already computed for
the unrelated fit-anchor use and passed into ``get_saturated_stars`` as
``zeroframe_group0_saturated`` -- no new file I/O) overlapping the (dilated)
saturated blob. This is INDEPENDENT of the deblender's own smoothed-ZEROFRAME
peak search: a real multi-star blend saturates >=2 separate cores in the raw
group-0 read; a PSF-sidelobe/diffraction-spike artifact of a single star does
not independently re-saturate at the sidelobe location in group 0.

Measured precision/recall (19 production 10678 tiles, against an independent
g0-core(>=2, defect-filtered)-OR-VIRAC2(>=2) ground truth; see
``gate_eval.py``/``REPORT.md`` in the deblendopt scratch dir for the full
table): g0-core>=2 alone reaches 100% precision (0% false-positive rate on the
defect-filtered ground truth) at 68.4% (F212N) / 52.4% (F480M) recall of the
independently-confirmable population, versus 0.17%/1.29% precision with no
gate at all. VIRAC2 as a second, OR'd signal raises F480M recall to 92.1% at
the same 100% precision; that is a follow-up (it needs a per-obsid VIRAC2
cache load that ``get_saturated_stars`` does not currently have in scope) and
is intentionally NOT part of this gate, which is scoped to the
zero-extra-I/O group-0 signal only.
"""
import os

import numpy as np
from scipy import ndimage

#: minimum number of independent group-0-saturated cores required to accept
#: a >=2-centre deblend split; below this, the split collapses to the single
#: blob centroid (the pre-deblend / zeroframe_deblend=False behaviour).
MIN_GROUP0_CORES = 2

#: dilation (pixels) applied to the blob mask before counting group-0 cores,
#: so a core that saturates in group-0 just outside the (smaller) final-group
#: DQ-SATURATED footprint still counts towards it.
DILATE_PIX = 2

_ON = {"1", "true", "yes", "on"}
_OFF = {"0", "false", "no", "off"}


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
    """``SATSTAR_DEBLEND_GATE`` (default ON): require >= ``MIN_GROUP0_CORES``
    independent group-0-saturated cores before accepting a multi-centre
    deblend split. Turning this OFF restores the pre-gate behaviour (every
    proposed split is accepted as-is) and is meant only for a deliberate,
    justified A/B comparison -- see the jwst-gc-pipeline CLAUDE.md override
    policy for the astrometry checkpoints, which this follows."""
    return _env_switch('SATSTAR_DEBLEND_GATE', True, env)


def count_group0_cores(group0_saturated, blob_mask, sl, dilate_px=DILATE_PIX):
    """Count distinct connected group-0-GROUPDQ-SATURATED clusters overlapping
    a dilated version of ``blob_mask`` within slice ``sl``.

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

    Returns
    -------
    int or None
        Number of distinct group-0-saturated clusters overlapping the
        (dilated) blob, or ``None`` if ``group0_saturated`` or ``sl`` is
        unavailable.
    """
    if group0_saturated is None or sl is None:
        return None
    sub_blob = np.asarray(blob_mask[sl], dtype=bool)
    if not sub_blob.any():
        return 0
    dil = ndimage.binary_dilation(sub_blob, iterations=dilate_px)
    sub_g0 = np.asarray(group0_saturated[sl], dtype=bool) & dil
    if not sub_g0.any():
        return 0
    _, n = ndimage.label(sub_g0)
    return int(n)


def gate_split_centers(centers, com, blob_mask, sl, group0_saturated,
                       min_cores=MIN_GROUP0_CORES, env=None):
    """Accept or reject a proposed >=2-centre deblend split.

    A split is accepted only when at least ``min_cores`` independent group-0
    GROUPDQ-saturated cores overlap the blob. When the independent signal is
    unavailable for this frame (``group0_saturated is None``), or it reports
    fewer cores than the number of proposed centres requires, the split is
    REJECTED and collapses to the single blob centroid ``com``: a deblend is
    attempted only when it is both needed (``len(centers) >= 2``) and
    reliable (independently confirmed), per the maintainer's request.

    A single-centre result (``len(centers) < 2``, i.e. no split was proposed)
    is returned unchanged -- there is nothing to gate.

    Returns
    -------
    (centers, n_g0_cores) : (list, int or None)
        The (possibly collapsed) centre list, and the measured group-0-core
        count (``None`` when the independent signal was unavailable), for the
        caller to log.
    """
    if len(centers) < 2:
        return centers, None
    if not deblend_gate_enabled(env):
        return centers, None
    n_g0 = count_group0_cores(group0_saturated, blob_mask, sl)
    if n_g0 is not None and n_g0 >= min_cores:
        return centers, n_g0
    return [com], n_g0
