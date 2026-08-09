"""Stage astrometry checkpoints — measure, verify, and (early only) correct.

Implements the failsafe ladder around the cataloging iterations:

* **m2 checkpoint** (after the m12 merge — the first per-frame catalogs):
  build the per-(visit, filter) consensus (``visit_consensus``), re-measure
  every exposure's bulk offset against it, and tie the consensus to the
  absolute reference (VIRAC2/Gaia) with multiple independent checks.  A
  per-exposure disagreement > ``EXPOSURE_CONSENSUS_TOL_MAS`` (2 mas) or a
  verified reference correction means the first-pass ("im0") alignment is
  WRONG: the offsets table is corrected (with provenance), the stale merged
  ``_i2d`` mosaics are tagged ``*_im0_badastrom.fits``, and the affected
  frames must be regenerated from ``_cal`` (fix_alignment re-applies the
  corrected table — the ONLY sanctioned way to change a baked ``RAOFFSET``;
  see ``ASTROMETRY_WCS_CORRECTION_FLOW.md``).

* **m3..m6 checkpoints**: the SAME measurement, but the astrometric solution
  must not move any more — positions come from the same crf GWCS, so a shift
  at these stages means a real defect (centroiding systematics, a seed that
  dragged fits, a stale frame).  Any exposure- or reference-level shift above
  tolerance raises ``AstrometryRegressionError`` (blocking; override only via
  ``ALLOW_LATE_STAGE_ASTROM_SHIFT=1``).

* **cross-filter checkpoint** (at the m7 cross-band merge): the filter closest
  in wavelength to VIRAC2 Ks anchors the absolute frame; every other filter
  must agree with the anchor to < ``CROSSFILTER_TOL_MAS`` (5 mas) bulk, and no
  ``LOCAL_CELL_SIZE_ARCSEC`` (2") cell may carry a significant local offset >
  ``LOCAL_CELL_TOL_MAS`` (15 mas) — significance REQUIRED (error bars; one
  star is not a measurement).  Alongside those two GATES it also MEASURES the
  coherent, position-dependent part of each filter-to-anchor residual
  (``measure_residual_field``), which neither gate can see and which is the
  scale of the field's astrometric floor: on the Brick, per-component rms of
  0.51 mas (F405N/F466N), 1.40 (F212N/F182M), 2.50 (F212N/F405N), 3.42
  (F212N/F200W) and 4.47 (F182M/F115W), at a per-cell SEM of 0.02–0.13 mas.
  The amplitude does NOT simply track the SW/LW split -- two same-channel,
  same-detector pairs exceed it -- so bandpass separation is the better
  predictor.  Recorded, printed, never gates.

  The 2" cell gate is BLIND on a dense field, not merely insensitive: the
  reliability cut leaves ~1.2 stars per 2" cell against
  ``LOCAL_CELL_MIN_STARS = 10`` (measured on brick F212N/F182M), so the map
  returns ``n_cells = 0``, and an injection sweep on Brick geometry never trips
  it at any amplitude up to 30 mas/arcmin (issue #296).  That silence used to
  score as a pass; an empty or near-empty map is now reported as UNVERIFIED --
  never as a failure, since measuring nothing is a coverage fact rather than
  evidence of a misalignment -- via the record's ``unverified`` /
  ``all_verified``.

Every checkpoint writes a machine-readable record under
``{basepath}/astrometry_checkpoints/`` so the release gate can audit the full
ladder.  Nothing here ever edits ``_cal.fits`` or pokes a mosaic GWCS.
"""
import collections
import glob
import json
import os
import re
from datetime import datetime, timezone

import numpy as np
from astropy import units as u
from astropy.table import Table

from .visit_consensus import (
    EXPOSURE_CONSENSUS_TOL_MAS, ConsensusBuildError, DuplicateExposureError,
    build_visit_consensus,
    catalog_coords, detect_module_antisymmetry, load_reference_catalog,
    measure_reference_tie, pick_reference_anchor_filter, select_reliable_stars,
)
from .astrometry_offsets import measure_offset, local_residual_map
from .consensus_catalog import (pool_visit_consensi,
                                 write_filter_consensus)
from ..atomic_io import atomic_write, keep_a_copy, locked

# Stages at which a measured shift is EXPECTED to be possible and is CORRECTED
# (the first checkpoint after the first per-frame photometry).  At every later
# stage the solution must be stable and a shift is a defect.
CORRECTION_STAGES = ("m1", "m2", "m12")

# A reference correction is only APPLIED when it exceeds this (below it the
# im0 solution already agrees with the reference at the measurement floor).
REFERENCE_APPLY_MIN_MAS = 2.0

# Late-stage (m3+) stability tolerance: the astrometric solution must not move.
STAGE_STABILITY_TOL_MAS = 2.0

# Cross-filter agreement tolerances (m7 checkpoint).
CROSSFILTER_TOL_MAS = 5.0
LOCAL_CELL_TOL_MAS = 15.0
LOCAL_CELL_SIZE_ARCSEC = 2.0
LOCAL_CELL_MIN_STARS = 10
#: Below this many populated cells the local map has checked too little of
#: the field for a pass to mean anything -- reported as unverified, never as
#: a failure.  One cell out of thousands is not coverage.
LOCAL_CELL_MIN_CELLS = 4

# Cross-filter residual FIELD (measurement only, never gates).  The 2"/15 mas
# local map above is a blend/gross-patch detector and at GC densities it does
# not merely fail to reach significance -- it returns NO CELLS AT ALL (~1 star
# per 2" cell against LOCAL_CELL_MIN_STARS = 10), and the caller reads only
# n_flagged, so an empty map is indistinguishable from a clean one.  The field
# measurement uses cells large enough to hold hundreds of stars, so a smooth
# position-dependent filter-to-filter difference becomes visible.  On the Brick
# it is 0.51-4.47 mas rms per component at a median cell SEM of 0.02-0.13 mas
# -- 7-45 sigma, and invisible to both the 5 mas bulk gate and the cell gate.
CROSSFILTER_FIELD_CELL_ARCSEC = 45.0
CROSSFILTER_FIELD_MIN_STARS = 40

STALE_TAG = "_im0_badastrom.fits"


class AstrometryCorrectionRequiredError(RuntimeError):
    """The m2 checkpoint measured a real misalignment: the im0 (first-pass)
    alignment is wrong and the affected frames must be regenerated from
    ``_cal`` with the corrected offsets table BEFORE cataloging continues —
    every catalog position derives from the (stale) crf GWCS, so continuing
    would propagate the error."""


class AstrometryRegressionError(RuntimeError):
    """A late-stage (m3+) checkpoint measured an astrometric shift.  The
    solution is supposed to be frozen after the m2 checkpoint; a shift here is
    a real defect and MUST be investigated, not re-corrected over."""


class CrossFilterAstrometryError(RuntimeError):
    """The cross-filter (m7) checkpoint failed: a filter disagrees with the
    anchor filter beyond tolerance, or a local cell carries a significant
    offset.  Blocking."""


class OffsetsTableUpdateError(RuntimeError):
    """The offsets-table correction could not be applied safely."""


def _utcnow_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _env_flag(name):
    return os.environ.get(name, "").strip() == "1"


#: Env override restoring the pre-#312 meaning of ``passed`` (failures only).
#: Named like ALLOW_LATE_STAGE_ASTROM_SHIFT: an explicit, greppable decision to
#: proceed on a field the checkpoint could not verify, not a default.
ALLOW_UNVERIFIED_ENV = "ALLOW_UNVERIFIED_ASTROM_CHECKPOINT"


def _checkpoint_passed(failures, unverified_blocking):
    """``passed`` for a checkpoint record.

    ``unverified`` is TWO different things in one list, and only one of them is
    evidence of a problem:

      * COULD NOT MEASURE -- an unbuildable consensus (two exposures, almost no
        stars), an isolated footprint with no tie, a local cell map with one or
        two populated cells.  Nothing was measured, so nothing is being ignored.
        This is deliberately not fatal and is pinned by seven tests
        (``test_unbuildable_consensus_is_unverified_not_fatal`` among them).
        Failing here would stop a field for having too little data to check it.

      * MEASURED AND REFUSED -- m2 measured a gross consensus->reference offset,
        or read equal-and-opposite offsets across an exposure's modules, and
        declined to apply anything.  A number exists and says the field is
        misaligned; the checkpoint simply cannot act on it.

    Only the second blocks (#312).  cloudc F410M/nrcblong/visit002 is the case
    that named this: m2 MEASURED 731.47 mas, over
    REFERENCE_CROSSCHECK_GROSS_MAS, set ``apply_ok=False``, filed 8 exposures
    unverified and reported ``passed=True``.  Every iteration since 2026-08-04
    recorded the identical value with ``ncorr=0`` and the retie loop declared
    convergence -- because corrections had STOPPED, not because the visit was
    aligned.  Those 8 exposures drizzle 4.06" out of place.

    A gross offset is precisely the case m2 refuses to correct, so making the
    refusal invisible to the gate left the loudest evidence of misalignment as
    the one thing that could not stop the pipeline.
    """
    if failures:
        return False
    if unverified_blocking and not _env_flag(ALLOW_UNVERIFIED_ENV):
        return False
    return True


# A per-exposure/per-visit tie correction is mas-scale by construction: it
# removes guide-star jitter and the consensus->reference residual, not a gross
# pointing error.  Anything larger is an upstream measurement failure (a
# window-limited or spurious offset-histogram peak) and must never be baked
# into the table: cloudef accumulated a +102" ddec correction this way on
# 2026-07-28, compounding across re-tie iterations (13.5" -> 19.2" -> 27.1" in
# dra) until the table was unusable, with no guard firing.
#
# seed_offsets_table_from_consensus already gates the RESULTING consensus row
# at this limit (a consensus table holds nothing but jitter, so its absolute
# value is bounded).  The curated VIRAC2locked tables carry legitimate
# arcsec-scale BULK offsets in their rows, so they cannot be gated on absolute
# value -- they are gated here, on the CORRECTION.
MAX_CORRECTION_ARCSEC = 0.5

# The per-VISIT BULK tie (consensus -> reference; exposure=None AND module=None)
# is a DIFFERENT quantity from per-exposure jitter and is legitimately large.
# Early-Cycle JWST visits that acquired the wrong guide star are really offset by
# arcseconds -- brick-1182 visit-001 by ~17-20", and ~4" / ~13" cases exist across
# the programme.  Correcting those IS the job, so the bulk tie is bounded only
# well above any real guide-star failure, at the sweep ceiling of measure_offset
# (60"): past that the "measurement" cannot even have come from a swept peak.
MAX_BULK_CORRECTION_ARCSEC = 60.0


def _positive_env_float(name, default):
    """Read a positive float from the environment, else ``default``.

    A blank/whitespace value means "unset" (mirrors ``_env_flag`` above, which
    strips).  A non-positive or unparseable value is a configuration error, not
    a licence to refuse every correction: ``ASTROM_MAX_CORRECTION_ARCSEC=0``
    would otherwise make a 1 mas correction raise.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    try:
        val = float(raw)
    except ValueError:
        raise OffsetsTableUpdateError(
            f"{name}={raw!r} is not a number (expected arcsec, e.g. 30)")
    if not (val > 0) or not np.isfinite(val):
        raise OffsetsTableUpdateError(
            f"{name}={raw!r} must be a positive, finite number of arcsec")
    return val


def _is_bulk_correction(corr):
    """A per-visit bulk consensus->reference tie carries no exposure AND no
    module (see BULK_EXPOSURE / BULK_MODULE); anything else is per-exposure."""
    return corr.get("exposure") is None and corr.get("module") is None


def _assert_correction_magnitudes(corrections, offsets_path):
    """Bound each correction by its KIND.

    Per-exposure jitter is mas-scale by construction, so it is held to
    ``MAX_CORRECTION_ARCSEC``.  That is the gate cloudef needed: its +102"
    runaway was written to a per-EXPOSURE row (F162M exposure 8).

    A per-visit BULK tie is held to the much looser
    ``MAX_BULK_CORRECTION_ARCSEC``, because a wrong-guide-star visit really is
    arcseconds off and auto-correcting it is legitimate.

    Both limits can be overridden with ``ASTROM_MAX_CORRECTION_ARCSEC`` /
    ``ASTROM_MAX_BULK_CORRECTION_ARCSEC`` when a deliberate re-authoring needs
    to go further.
    """
    limit = _positive_env_float("ASTROM_MAX_CORRECTION_ARCSEC",
                                MAX_CORRECTION_ARCSEC)
    bulk_limit = _positive_env_float("ASTROM_MAX_BULK_CORRECTION_ARCSEC",
                                     MAX_BULK_CORRECTION_ARCSEC)
    big, nonfinite = [], []
    for corr in corrections:
        dra_as = float(corr["dra_onsky_mas"]) / 1000.0
        ddec_as = float(corr["ddec_onsky_mas"]) / 1000.0
        is_bulk = _is_bulk_correction(corr)
        ident = (corr.get("visit"), corr.get("filtername"),
                 "BULK" if is_bulk else corr.get("exposure"), corr.get("module"))
        # NaN must be caught EXPLICITLY: abs(nan) > limit is False, so a
        # non-finite correction would sail through the ceiling and poison the
        # row -- and assert_offsets_table_sane cannot catch it either, since its
        # collapse comparisons against NaN are all False.  That is the very
        # failure class this gate exists to stop.  (inf is caught by the
        # magnitude test, but check it here too rather than rely on that.)
        if not (np.isfinite(dra_as) and np.isfinite(ddec_as)):
            nonfinite.append(ident + (dra_as, ddec_as))
            continue
        lim = bulk_limit if is_bulk else limit
        if abs(dra_as) > lim or abs(ddec_as) > lim:
            big.append(ident + (round(dra_as, 4), round(ddec_as, 4),
                                f"limit={lim}\""))
    if nonfinite:
        raise OffsetsTableUpdateError(
            f"{len(nonfinite)} non-finite correction(s) will NOT be applied to "
            f"{os.path.basename(offsets_path)} -- a NaN/inf offset is a failed "
            f"upstream measurement, and writing it silently destroys the row "
            f"(no downstream guard compares true against NaN).  "
            f"(visit, filter, exposure, module, dra\", ddec\"): {nonfinite}")
    if big:
        raise OffsetsTableUpdateError(
            f"{len(big)} correction(s) exceed their magnitude limit and will "
            f"NOT be applied to {os.path.basename(offsets_path)}.  A "
            f"per-exposure tie correction is mas-scale (limit {limit}\"); a "
            f"per-visit BULK tie may be arcseconds (limit {bulk_limit}\") but "
            f"not this large.  A correction over its limit means the upstream "
            f"measurement is wrong -- a window-limited or spurious peak -- not "
            f"that the frame is really that far off.  (visit, filter, exposure, "
            f"module, dra\", ddec\", limit): {big}.  For a deliberate gross "
            f"re-authoring, raise ASTROM_MAX_CORRECTION_ARCSEC / "
            f"ASTROM_MAX_BULK_CORRECTION_ARCSEC with written justification.")


# ---------------------------------------------------------------------------
# im0 invalidation: stale-tagging merged mosaics
# ---------------------------------------------------------------------------

def mark_i2d_stale(i2d_paths, reason, record_dir=None):
    """Tag stale first-pass merged mosaics: ``*_i2d.fits`` ->
    ``*_i2d_im0_badastrom.fits`` (rename, never delete/overwrite), and drop a
    sidecar JSON documenting why.  Returns the list of (old, new) renames."""
    renames = []
    for path in i2d_paths:
        if not os.path.exists(path):
            continue
        if path.endswith(STALE_TAG):
            continue
        if not path.endswith(".fits"):
            raise OffsetsTableUpdateError(f"refusing to stale-tag non-FITS {path}")
        new = path[:-len(".fits")] + STALE_TAG
        n = 1
        while os.path.exists(new):
            new = path[:-len(".fits")] + STALE_TAG.replace(".fits", f".{n}.fits")
            n += 1
        os.rename(path, new)
        sidecar = new + ".why.json"
        with open(sidecar, "w") as fh:
            json.dump(dict(original=path, renamed_to=new, reason=reason,
                           date=_utcnow_iso()), fh, indent=2)
        renames.append((path, new))
    if record_dir and renames:
        os.makedirs(record_dir, exist_ok=True)
        with open(os.path.join(record_dir, "stale_i2d_renames.json"), "a") as fh:
            for old, new in renames:
                fh.write(json.dumps(dict(old=old, new=new, reason=reason,
                                         date=_utcnow_iso())) + "\n")
    return renames


def find_i2d_for_filter(basepath, filtername, extra_globs=()):
    """Locate the merged first-pass (im0) ``_i2d.fits`` mosaics for a filter."""
    pats = [
        f"{basepath}/{filtername.upper()}/pipeline/*-{filtername.lower()}-*_i2d.fits",
        f"{basepath}/{filtername.upper()}/pipeline/*_{filtername.lower()}_*_i2d.fits",
    ]
    pats.extend(extra_globs)
    out = []
    for pat in pats:
        out.extend(p for p in glob.glob(pat) if not p.endswith(STALE_TAG))
    return sorted(set(out))


# ---------------------------------------------------------------------------
# offsets-table correction (the ONLY authoring channel for the tie)
# ---------------------------------------------------------------------------

def _assert_module_granularity(corrections, tbl, offsets_path):
    """Refuse per-module corrections a Module-less table cannot express.

    Corrections are keyed ``(visit, exposure, module)``, but the apply loop
    skips the module narrowing when the table has no ``Module`` column.  Every
    detector's correction for one exposure then lands on the SAME row,
    additively: 8 detectors x +0.4" each -> +3.2" on the row, with every
    individual correction legal under the magnitude ceiling.

    That is the mechanism behind the cloudef runaway -- jw02092002001 exp 8
    F162M went 7302 -> 51166 -> 73111 mas of accumulated prov across successive
    writes on a Module-less table.  The magnitude ceiling alone cannot see it,
    because the over-correction is a SUM of legal parts.

    Mirrors the existing "per-exposure correction on a per-visit table"
    refusal: the fix is to rebuild the table with ``--per-module`` so the
    corrections match its rows 1:1.

    NOTE this is the COARSE half of the check.  Having a ``Module`` column is
    not the same as having it at the corrections' granularity -- a table whose
    rows are module FAMILIES (``nrca``/``nrcb``/``nrcalong``/``nrcblong``) still
    collapses four DETECTOR corrections (``nrca1``..``nrca4``) onto one row.
    That case is caught row-wise by ``_assert_one_correction_per_row`` below.
    """
    if "Module" in tbl.colnames:
        return
    per_key = {}
    for corr in corrections:
        if corr.get("module") is None:
            continue        # bulk tie: module-independent by construction
        key = (str(corr.get("visit")), str(corr.get("filtername")),
               corr.get("exposure"))
        per_key.setdefault(key, set()).add(str(corr["module"]))
    clashes = {k: sorted(v) for k, v in per_key.items() if len(v) > 1}
    if clashes:
        raise OffsetsTableUpdateError(
            f"{os.path.basename(offsets_path)} has no Module column, but "
            f"{len(clashes)} (visit, filter, exposure) key(s) carry corrections "
            f"for MORE THAN ONE module -- they would all match the same row and "
            f"be summed into an N-fold over-correction (this is how cloudef ran "
            f"away).  Rebuild the table per-module "
            f"(build_virac2_offsets --per-module) so corrections map 1:1, or "
            f"pool the modules into a single correction first.  "
            f"Offending keys: {sorted(clashes.items())[:4]}")


def _assert_one_correction_per_row(corrections, tbl, offsets_path):
    """Refuse a correction set that is FINER than the table's rows.

    The general form of the module/vgroup granularity refusals above, checked
    against the rows the apply loop would actually touch rather than against a
    key the table may or may not carry.  Two corrections landing on one row are
    SUMMED, and every part can be legal under the per-correction magnitude
    ceiling -- so the sum is invisible to every other guard.

    This is the sgrc/cloudc/cloudef divergence of 2026-07-30..08-01.  Their
    tables DO carry a ``Module`` column, so ``_assert_module_granularity``
    returned early -- but the column holds module FAMILIES while the m2
    visit-consensus emits one correction per DETECTOR.  Measured on the live
    checkpoint records: sgrc F115W put 45 detector corrections onto 12 rows (4
    per row), cloudc F182M put 64 onto 32 (up to 11 per row).  Each re-tie
    iteration therefore added roughly the SUM of four detectors' SIAF-class
    residuals to one row instead of their common part, and the tables ran away
    (sgrc accumulated 185.7 -> 525.7 -> 1678.5 mas over three iterations).

    The fix at the CALLER is ``pool_corrections_to_table_granularity`` -- a
    module-family row can only express the module-COMMON shift, so the
    per-detector residuals must be pooled before they are applied, not summed.
    """
    hits = {}
    for corr in corrections:
        if _is_bulk_correction(corr):
            continue        # see _is_bulk_correction: broad BY DESIGN
        idx = _match_rows(corr, tbl)
        for i in idx:
            hits.setdefault(int(i), []).append(corr)
    over = {i: v for i, v in hits.items() if len(v) > 1}
    if not over:
        return
    worst = sorted(over.items(), key=lambda kv: -len(kv[1]))[:3]
    detail = []
    for i, cs in worst:
        row = tbl[i]
        who = sorted({str(c.get("module")) for c in cs})
        detail.append(
            f"row {i} (visit={row['Visit']} filt={row['Filter']} "
            f"exp={row['Exposure'] if 'Exposure' in tbl.colnames else '-'} "
            f"mod={row['Module'] if 'Module' in tbl.colnames else '-'}) <- "
            f"{len(cs)} corrections from modules {who}")
    raise OffsetsTableUpdateError(
        f"{len(over)} row(s) of {os.path.basename(offsets_path)} would receive "
        f"MORE THAN ONE correction in a single write, and they are SUMMED -- an "
        f"N-fold over-correction whose parts are each legal under the magnitude "
        f"ceiling.  The correction set is finer-grained than the table's rows "
        f"(typically per-DETECTOR corrections against module-FAMILY rows).  "
        f"Pool them to the table's granularity first -- pass pool=True to "
        f"update_offsets_table, or --pool to "
        f"scripts/reduction/apply_m2_checkpoint_corrections.py / "
        f"run_astrometry_checkpoint.py -- or rebuild the table at the "
        f"corrections' granularity.  {'; '.join(detail)}")


def pool_corrections_to_table_granularity(corrections, offsets_path,
                                          tbl=None, stat="median"):
    """Collapse corrections that share a table row into one, robustly.

    A module-FAMILY offsets row cannot express a per-DETECTOR shift: the four
    detectors of a NIRCam module sit at fixed SIAF positions within it, so their
    individual residuals are a distortion/DVA-class systematic the row has no
    freedom to remove.  What the row CAN express is the part they share.  Take
    the median (not the sum, and not the mean -- one bad detector should not
    move it) of every correction that lands on the same row.

    Returns a NEW list; corrections that own their row are passed through
    unchanged.  Pooled entries carry the member modules and count in ``source``
    so the provenance names what was collapsed.

    Applying this BEFORE the actionability floor is what makes the re-tie loop
    converge: four detector residuals of a few mas that largely cancel pool to a
    sub-floor module shift, and the checkpoint passes instead of writing their
    sum.  Summing them is what made sgrc diverge.

    Pooling is deliberately NARROW.  It only ever collapses detectors of ONE
    module family, and only when each contributes at most one correction:

    * ACROSS families is refused.  "the four detectors sit at fixed SIAF
      positions within it" is the whole justification, and it does not extend
      to medianing module A against module B -- the A/B seam is a systematic
      this project tracks separately.  Refusing also keeps
      ``_assert_module_granularity``'s Module-less refusal intact: on a table
      with no Module column every module lands on the same row, so the group
      spans families and pooling stops rather than quietly applying an
      A/B-averaged shift.  (sgrb2's VIRAC2locked table is exactly this shape
      and is a live ``locked`` field.)
    * REPEATED modules within a group are refused.  Two corrections for the
      same module on one row are not detectors of that module, they are two
      physically distinct things the table cannot tell apart -- e.g. sgrb2's
      records carry corrections with no vgroup at all against a Vgroup-less
      table, so two pointings collide.  Pooling must not absorb what the
      vgroup guard exists to stop.
    """
    if tbl is None:
        tbl = Table.read(offsets_path)
        if "Visit" not in tbl.colnames:
            # every granularity guard below narrows on Visit; without it they die
            # on KeyError, which is the wrong class to escape a guarded writer.
            raise OffsetsTableUpdateError(
                f"{offsets_path} has no Visit column ({tbl.colnames}) -- a "
                f"correction cannot be matched to a row without one")
    corrections = list(corrections)
    # Magnitude ceiling BEFORE the median.  Pooling cannot inflate a correction
    # past the ceiling (median <= max), so the risk runs the other way: a
    # detector whose measurement blew up is averaged out of existence and the
    # operator never learns the measurement failed.  Check the MEMBERS.
    _assert_correction_magnitudes(corrections, offsets_path)
    groups = {}
    order = []
    bulk = []
    for corr in corrections:
        if _is_bulk_correction(corr):
            bulk.append(corr)
            continue
        key = tuple(sorted(int(i) for i in _match_rows(corr, tbl)))
        if key not in groups:
            order.append(key)
        groups.setdefault(key, []).append(corr)

    # Overlapping-but-unequal row sets cannot be pooled: one correction would
    # have to be split across groups.  Refuse rather than guess -- this is the
    # 'nrcalong' variant leaking onto an 'nrca' row, which a granularity fix in
    # the matcher (not a pooling rule) has to resolve.
    seen = {}
    for key in order:
        for i in key:
            if i in seen and seen[i] != key:
                raise OffsetsTableUpdateError(
                    f"cannot pool corrections for "
                    f"{os.path.basename(offsets_path)}: row {i} is matched by "
                    f"two DIFFERENT row-sets {seen[i]} and {key} -- the "
                    f"correction modules overlap partially (e.g. an LW "
                    f"'nrcalong' correction matching an SW 'nrca' row).  Fix "
                    f"the module granularity of the corrections or the table; "
                    f"pooling cannot resolve a partial overlap.")
            seen[i] = key

    if stat not in _POOL_STATS:
        # `agg = np.median if stat == "median" else np.mean` silently degraded a
        # typo to the LESS robust statistic, and the statistic is the whole
        # point: members 1,1,1,100 give 1.0 as "median" and 25.75 as "medain".
        raise ValueError(f"pool stat must be one of {sorted(_POOL_STATS)}, "
                         f"got {stat!r}")
    agg = _POOL_STATS[stat]

    out = list(bulk)
    for key in order:
        members = groups[key]
        if len(members) == 1:
            out.append(members[0])
            continue
        mods = sorted(str(c.get("module")) for c in members)
        _assert_poolable(members, mods, key, tbl, offsets_path)
        dra = float(agg([float(c["dra_onsky_mas"]) for c in members]))
        ddec = float(agg([float(c["ddec_onsky_mas"]) for c in members]))
        # Dispersion, so a bimodal group is visible rather than pooling to a
        # meaningless middle with no trace.  Peak-to-peak of the 2-D residual
        # magnitudes; carried in `source` AND returned on the dict for the
        # checkpoint record (`source` is truncated to 64 chars on write).
        mags = [float(np.hypot(c["dra_onsky_mas"], c["ddec_onsky_mas"]))
                for c in members]
        spread = float(np.ptp(mags))
        _assert_pool_spread(spread, members, mods, offsets_path)
        pooled = dict(members[0])
        pooled["dra_onsky_mas"] = dra
        pooled["ddec_onsky_mas"] = ddec
        pooled["dec_deg"] = float(np.mean([float(c["dec_deg"]) for c in members]))
        pooled["module"] = _pooled_module_label(mods, tbl, key)
        pooled["pooled_from"] = mods
        pooled["pooled_n"] = len(members)
        pooled["pooled_spread_mas"] = spread
        pooled["pooled_stat"] = stat
        pooled["source"] = (f"{members[0].get('source', 'astrometry_checkpoint')}"
                            f" [{stat} of {len(members)}, ptp {spread:.2f}mas: "
                            f"{','.join(mods)}]")
        out.append(pooled)
    return out


# dra and ddec are aggregated INDEPENDENTLY, which is the component-wise median
# and not the geometric (2-D) median.  For the N<=4 groups this pooler is built
# for the two differ negligibly, and the component-wise form has the property
# that matters here -- it cannot exceed the component-wise max, so it can never
# sum.  Revisit if groups ever get large.
_POOL_STATS = {"median": np.median, "mean": np.mean}

# Refuse a group whose members disagree by more than this; they are not one
# shift seen four times, and their middle means nothing.  Generous by default:
# real per-detector SIAF/DVA spread is a few mas, and the sgrb2 groups measured
# on 2026-08-01 ran 1.7-3.4 mas peak-to-peak.
MAX_POOL_SPREAD_MAS = 50.0


def _assert_poolable(members, mods, row_key, tbl, offsets_path):
    """Refuse a group pooling cannot legitimately collapse.  See the pooler."""
    families = {_module_family(m) for m in mods if m and m != "None"}
    if len(families) > 1:
        raise OffsetsTableUpdateError(
            f"cannot pool corrections for {os.path.basename(offsets_path)}: "
            f"{len(members)} corrections spanning module families "
            f"{sorted(families)} land on the same row(s) {row_key} "
            f"(modules {mods}).  Pooling collapses the DETECTORS of one module, "
            f"whose fixed SIAF positions within it make their spread a "
            f"distortion-class systematic -- that argument does not extend to "
            f"medianing module A against module B, and the A/B seam is tracked "
            f"separately.  A table with no Module column always lands here, "
            f"which is correct: rebuild it per-module "
            f"(build_virac2_offsets --per-module) so corrections map 1:1.")
    bare = {m for m in mods if m and _module_family(m) == m}
    if bare and len(set(mods)) > 1:
        # A BARE module token beside a more specific one is the same physical
        # hardware twice, not several detectors of one module (issue #298).
        # Pooling would MEDIAN a frame against itself: sgrb2's F360M table has
        # no Module column, so `nrcb` and `nrcblong` -- same family, distinct
        # tokens -- passed the family check above and were silently blended
        # (10 and -4 mas pooled to 3.0).  That is worse than the aliasing this
        # refuses at write time, because nothing downstream can see it happened.
        raise OffsetsTableUpdateError(
            f"cannot pool corrections for {os.path.basename(offsets_path)}: "
            f"module(s) {sorted(bare)} appear beside more specific spellings "
            f"of the same hardware {sorted(set(mods) - bare)} on row(s) "
            f"{row_key}.  A bare module token is not a detector -- pooling "
            f"these medians one physical frame against itself.  The upstream "
            f"cause is a stale bare-module per-frame catalog ingested next to "
            f"its numbered/`long` counterpart (issue #298).")
    if len(set(mods)) != len(mods):
        dupes = sorted({m for m in mods if mods.count(m) > 1})
        raise OffsetsTableUpdateError(
            f"cannot pool corrections for {os.path.basename(offsets_path)}: "
            f"module(s) {dupes} contribute MORE THAN ONE correction to the same "
            f"row(s) {row_key}.  Two corrections for one module are not its "
            f"detectors -- they are two physically distinct things the table "
            f"cannot tell apart (typically two visit groups against a "
            f"Vgroup-less table).  Pooling must not absorb what the vgroup "
            f"guard exists to stop; extend the table to carry Vgroup.")


def _assert_pool_spread(spread, members, mods, offsets_path):
    limit = _positive_env_float("ASTROM_MAX_POOL_SPREAD_MAS", MAX_POOL_SPREAD_MAS)
    if spread <= limit:
        return
    raise OffsetsTableUpdateError(
        f"cannot pool corrections for {os.path.basename(offsets_path)}: "
        f"{len(members)} corrections for modules {mods} disagree by "
        f"{spread:.1f} mas peak-to-peak (limit {limit} mas, "
        f"ASTROM_MAX_POOL_SPREAD_MAS).  That is not one shift measured several "
        f"times, so their middle is not a measurement of anything -- one "
        f"detector's tie has probably failed.  Inspect the checkpoint record.")


def _pooled_module_label(mods, tbl, row_key):
    """The module token the pooled correction should carry.

    Prefer the table's own value for the row(s) it lands on -- that is by
    construction the granularity the table expresses.  Fall back to the shared
    family of the pooled detector names, which ``_assert_poolable`` has already
    established is unique (so this can no longer mislabel a cross-family pool
    with one arbitrary member's name).
    """
    if "Module" in tbl.colnames and row_key:
        vals = sorted({str(tbl["Module"][i]) for i in row_key})
        if len(vals) == 1:
            return vals[0]
    families = sorted({_module_family(m) for m in mods if m and m != "None"})
    return families[0] if families else None


def _module_family(module):
    """``nrca1``/``nrca``/``nrcalong`` -> ``nrca``; the channel-free family."""
    m = str(module).strip("1234")
    return m[:-4] if m.endswith("long") else m


def _apply_module_rows(corr_module, present):
    """Which of a filter's ``Module`` row values a correction may be added to.

    WRITE-direction module matching, deliberately NOT ``_module_variants``.
    That helper implements READ semantics -- "which row do I look up for this
    frame" -- where matching a family row in addition to your own is correct
    and harmless.  In the write direction the same permissiveness fans ONE
    correction across several rows and, worse, across CHANNELS: it maps
    ``nrcalong -> {'nrcalong', 'nrca'}``, so an LW correction is also added to
    the SW ``nrca`` row.

    Resolve against the values the table actually carries FOR THIS FILTER
    instead of against a hardcoded LW filter list: a filter's rows are all one
    channel (SW filters carry ``nrca``/``nrcb``, LW filters ``nrcalong``/
    ``nrcblong``), so the table itself says which token this correction's
    family means here.  An exact row value always wins, so a table rebuilt at
    detector granularity keeps 1:1 matching with no change here.

    The single-channel-per-filter assumption is ENFORCED here rather than
    merely documented: no table on disk currently mixes them, but
    ``_module_family('nrcalong') == 'nrca'``, so if one ever did, a bare
    ``'nrca'`` correction would fan onto the ``nrcalong`` row -- the mirror
    image of the LW->SW leak this function exists to close.
    """
    long_rows = {p for p in present if str(p).endswith("long")}
    if long_rows and long_rows != set(present):
        raise OffsetsTableUpdateError(
            f"offsets rows for one filter mix LW and SW module tokens "
            f"({sorted(present)}).  Write-direction matching resolves a "
            f"correction's family against the row values present for its "
            f"filter, which is only unambiguous while a filter's rows are all "
            f"one channel; here a bare 'nrca' correction could not be told "
            f"from an 'nrcalong' one.  Split the filter's rows by channel.")
    m = str(corr_module)
    if m in present:
        return {m}
    fam = _module_family(m)
    return {p for p in present if _module_family(p) == fam}


#: ``jw`` + proposal(5) + observation(3) + visit(3).
_VISIT_ID_RE = re.compile(r'jw(\d{5})(\d{3})(\d{3})\s*$', re.IGNORECASE)


class AmbiguousVisitMatchError(ValueError):
    """A correction's visit cannot be told from another observation's."""


def visit_obs_key(value):
    """``(observation, visit)`` from a JWST visit id; ``(None, visit)`` if bare.

    Every narrowing site used to key on ``int(str(visit)[-3:])`` -- the visit
    number alone.  That is unique for a field whose observations each contain
    one visit, which is every field here except gc2211: its FIVE observations
    are all visit 001, so the last three digits are ``001`` for all of them.

        jw02211023001  jw02211028001  jw02211046001  jw02211049001  jw02211050001
                 ^^^ observation                              ^^^ visit == 001

    A correction measured on one of those observations therefore matched the
    rows of all five and was added to every one -- which is how gc2211's table
    came to carry a single ``prov_*`` pair across five pointings 0.3-17.6
    arcmin apart, in five measurably different astrometric states (#284).
    Keying on (observation, visit) separates them and is a no-op for every
    single-observation field.
    """
    s = str(value).strip()
    m = _VISIT_ID_RE.match(s)
    if m:
        return m.group(2), int(m.group(3))
    return None, int(s[-3:])


#: ``jw`` + proposal(5) + observation(3) + visit(3), anywhere in a path.
_VISIT_ID_IN_NAME_RE = re.compile(r'jw(\d{5})(\d{3})(\d{3})_', re.IGNORECASE)


def resolve_full_visit_id(tables, bare_visit):
    """Upgrade a bare visit number to ``jwPPPPPOOOVVV`` from the frames' names.

    A per-frame catalog's ``VISIT`` metadatum is the visit NUMBER (``1``); the
    observation lives only in the source frame's name, which the catalog carries
    as ``FILENAME``::

        VISIT    = 1
        FILENAME = '.../jw02211050001_02201_00001_nrcb1_destreak_o050_crf.fits'
                          ^^^^^ proposal
                               ^^^ observation
                                  ^^^ visit

    A correction built from the bare number cannot say WHICH observation it was
    measured on, so `_match_rows` refuses it against a multi-observation table
    (`AmbiguousVisitMatchError`) -- which is every gc2211 finalize: all five of
    its observations are visit 001, so ``'1'`` names all of them at once and the
    correction would broadcast (#284).  Recovering the observation from the
    frame name is what makes the correction addressable.

    ``FILENAME`` is read first-table-wins: a group where the FIRST table lacks
    provenance keeps the bare visit even if the rest agree.  That is the
    conservative direction and is deliberately NOT the same rule as the
    mixed-group check below, which is all-or-nothing.

    Only upgrades when EVERY table in the group agrees on one (proposal,
    observation, visit); a group whose frames disagree is contaminated with
    another observation's exposures (the class #352 fixed), and silently
    picking one of them would attach the correction to the wrong pointing.
    Such a group keeps the bare visit, so `_match_rows` still refuses it -- an
    error that names the real problem rather than a plausible wrong answer.
    """
    ids = set()
    for tbl in tables:
        fn = None
        for key in ("FILENAME", "filename"):
            if key in getattr(tbl, "meta", {}):
                fn = str(tbl.meta[key])
                break
        if not fn:
            return bare_visit
        m = _VISIT_ID_IN_NAME_RE.search(os.path.basename(fn))
        if not m:
            return bare_visit
        ids.add(m.group(0).rstrip('_').lower())
    if len(ids) != 1:
        return bare_visit
    full = ids.pop()
    # The name must agree with the metadatum it is replacing; a mismatch means
    # one of the two is describing a different frame, and neither can be
    # trusted to address a table row.
    if int(full[-3:]) != int(str(bare_visit)[-3:]):
        return bare_visit
    return full


def detector_sibling_alias_keys(exposures):
    """Exposure keys whose OWN DETECTOR says their peak is a footprint ridge.

    gc2211 o023's F200W visit, nrcb1 -- four exposures, one search window, one
    detector::

        contrast 8, edge 1.00, reproduced=False  ->  alias_rejected
        contrast 7, edge 1.00, reproduced=False  ->  alias_rejected
        contrast 8, edge 0.99, reproduced=False  ->  alias_rejected
        contrast 5, edge 0.93, reproduced=True   ->  ACCEPTED

    The accepted one has the LOWEST contrast of the four -- exactly
    `DEFAULT_MIN_CONTRAST`, clearing the floor by nothing -- and differs from
    its siblings only in that one confirmation probe reproduced its peak.  On a
    detector whose pair count is down ~10x (5.4k-17k against 60k-272k elsewhere
    in the same visit) while its source count is normal, that is the #158
    footprint ridge: a pair-density feature of the DETECTOR's geometry, which
    every exposure of it shares.  A probe reproducing it is not evidence
    against that -- the ridge is there at the other window too.

    So: when a swept, window-edge measurement's siblings on the same detector,
    at the same window, were rejected as aliases, it is rejected with them.
    The rule uses only evidence already in the record and needs no new
    threshold; what it asks is whether this detector produced a tie ANYWHERE in
    the visit, and here it did not.

    Deliberately narrow -- ALL of:

      * the measurement is ``swept`` and sits at/over ``WINDOW_EDGE_FRACTION``
        (a clean mas-scale tie has edge ~1e-4 and is untouched)
      * it shares its detector and window with >= 2 rejected siblings, so a
        single unlucky neighbour cannot condemn it
      * a MAJORITY of the detector's measurements at that window were rejected

    A detector that ties cleanly in most exposures keeps its odd one out --
    that one is a real per-exposure problem and must stay visible.
    """
    from .astrometry_offsets import WINDOW_EDGE_FRACTION

    by_detector = collections.defaultdict(list)
    for exp in exposures:
        res = exp.get("vs_consensus")
        key = tuple(exp.get("key") or ())
        if res is None or len(key) < 3:
            continue
        by_detector[(key[2], res.get("window_arcsec"))].append((key, exp, res))

    flagged = set()
    for (_det, _win), group in by_detector.items():
        rejected = [g for g in group if g[2].get("alias_rejected")]
        if len(rejected) < 2 or len(rejected) * 2 <= len(group):
            continue
        for key, _exp, res in group:
            if res.get("alias_rejected"):
                continue
            if not res.get("swept"):
                continue
            edge = res.get("window_edge_fraction")
            if edge is None or edge < WINDOW_EDGE_FRACTION:
                continue
            flagged.add(key)
    return flagged


def _gross_per_exposure_offset(res):
    """Why this per-exposure measurement is too gross to correct, or None.

    ``_assert_correction_magnitudes`` already refuses an arcsecond-scale
    per-exposure correction -- but it refuses the whole BATCH, at the point of
    writing the table, so one bad exposure discards every good correction
    measured alongside it.  This is the same limit asked at the point of
    MEASUREMENT, where the exposure can be refused on its own.

    The limit tracks ``ASTROM_MAX_CORRECTION_ARCSEC`` so the two cannot drift:
    anything this returns non-None for is something the writer would have
    rejected anyway.
    """
    off = res.get("off")
    if off is None or not np.isfinite(off):
        return None
    limit_arcsec = _positive_env_float("ASTROM_MAX_CORRECTION_ARCSEC",
                                       MAX_CORRECTION_ARCSEC)
    if limit_arcsec <= 0 or off <= limit_arcsec * 1000.0:
        return None
    return (f"{off / 1000.0:.2f}\" off the visit consensus "
            f"(> the {limit_arcsec:g}\" per-exposure limit)")


def _table_visit_obs(tbl):
    """Per-row ``(observation, visit)`` keys for an offsets table."""
    obs, vis = [], []
    for v in tbl["Visit"]:
        o, n = visit_obs_key(v)
        obs.append(o)
        vis.append(n)
    return np.array(obs, dtype=object), np.array(vis)


def _match_rows(corr, tbl):
    """Row indices of ``tbl`` a single correction would be ADDED to.

    Factored out of ``update_offsets_table`` so the granularity guard and the
    pooling helper narrow EXACTLY the way the apply loop does -- a guard that
    re-implements the narrowing is a guard that drifts away from what it guards.
    Unlike the apply loop this never raises; callers decide what an empty or
    over-full match means.

    Raises ``AmbiguousVisitMatchError`` -- the ONE thing it does raise -- when
    the correction names a bare visit number and the table spans more than one
    observation, because there is then no way to tell which observation's rows
    it belongs to and matching them all is the #284 broadcast.
    """
    corr_obs, visit = visit_obs_key(corr["visit"])
    row_obs, row_visit = _table_visit_obs(tbl)
    known = set(o for o in row_obs if o is not None)
    if corr_obs is None and len(known) > 1:
        raise AmbiguousVisitMatchError(
            f"correction names visit {corr['visit']!r} with no observation, but "
            f"the table spans observations {sorted(known)}. Matching on the "
            f"visit number alone would add this correction to every one of "
            f"them -- the gc2211 #284 broadcast.\n"
            f"Give the correction its full jwPPPPPOOOVVV visit id (e.g. "
            f"'jw02211{sorted(known)[0]}001'). scripts/reduction/"
            f"step0_bulk_offset.py builds `str(--visit).zfill(3)`, which is a "
            f"bare visit; pass the full id to --visit instead -- zfill leaves "
            f"an already-full id unchanged, so nothing else needs to know.")
    match = (row_visit == visit) & (tbl["Filter"] == corr["filtername"])
    if corr_obs is not None:
        # Rows whose Visit is a bare number carry no observation to compare, so
        # they stay eligible; a table that mixes the two forms is matched as
        # loosely as its least-specific rows allow, not silently narrowed to none.
        match &= np.array([o is None or o == corr_obs for o in row_obs])
    if corr.get("exposure") is not None and "Exposure" in tbl.colnames:
        match &= tbl["Exposure"] == int(corr["exposure"])
    if corr.get("module") is not None and "Module" in tbl.colnames:
        present = {str(m) for m in tbl["Module"][match]}
        allowed = _apply_module_rows(corr["module"], present)
        match &= np.array([str(m) in allowed for m in tbl["Module"]])
    wanted_vgroup = vgroup_key(corr.get("vgroup"))
    if "Vgroup" in tbl.colnames and wanted_vgroup:
        match &= np.array([vgroup_row_matches(g, wanted_vgroup)
                           for g in tbl["Vgroup"]])
    return np.where(match)[0]


def _assert_vgroup_granularity(corrections, tbl, offsets_path):
    """Refuse per-vgroup corrections a Vgroup-less table cannot express.

    A visit can dither across several visit groups, and the exposure number
    restarts in each, so ``(visit, exposure)`` is ambiguous.  The consensus key
    is vgroup-aware (see ``visit_consensus.exposure_key``) and therefore emits a
    SEPARATE correction per vgroup -- but the offsets tables carry no ``Vgroup``
    column and ``update_offsets_table`` matches on
    ``(visit, filter, exposure[, module])``, so both corrections would land on
    the same row and be summed.

    That is the same accumulation hazard as the Module-less case, and making the
    key vgroup-aware makes it MORE reachable, not less: previously the two
    vgroups were (wrongly) blended into one consensus entry and produced one
    correction.  cloudc has 2 visit groups, gc2211 has 6.

    Refuse, mirroring the existing per-exposure-on-a-per-visit-table refusal.
    The durable fix is to extend the builder and tables to carry vgroup.
    """
    if "Vgroup" in tbl.colnames:
        return
    per_key = {}
    for corr in corrections:
        vg = vgroup_key(corr.get("vgroup"))
        if not vg:
            continue
        key = (str(corr.get("visit")), str(corr.get("filtername")),
               corr.get("exposure"), str(corr.get("module")))
        per_key.setdefault(key, set()).add(str(vg))
    clashes = {k: sorted(v) for k, v in per_key.items() if len(v) > 1}
    if clashes:
        raise OffsetsTableUpdateError(
            f"{os.path.basename(offsets_path)} has no Vgroup column, but "
            f"{len(clashes)} (visit, filter, exposure, module) key(s) carry "
            f"corrections from MORE THAN ONE visit group -- they would all "
            f"match the same row and be summed.  The exposure number restarts "
            f"per vgroup, so the table cannot tell these exposures apart; "
            f"extend the builder and table to carry Vgroup, or pool the "
            f"vgroups into a single correction first.  "
            f"Offending keys: {sorted(clashes.items())[:4]}")


def vgroup_key(value):
    """Canonical dict-key form of a visit-group id; ``""`` for "no vgroup".

    A CSV round-trip mangles this column twice over: a digit column is inferred
    as int64 (so "06201" returns as 6201), and the BULK rows' empty cell returns
    as a MASKED value whose ``str()`` is ``'--'``.  Keying on the raw value
    therefore fails to match an existing bulk row on the second upsert and
    inserts a duplicate sentinel instead of accumulating onto it.
    """
    if value is None or isinstance(value, np.ma.core.MaskedConstant):
        return ""
    s = str(value).strip()
    if s in ("", "--", "nan", "None", "N/A"):
        return ""
    return str(int(s)) if s.isdigit() else s


def same_vgroup(a, b):
    """Compare two visit-group ids tolerantly.

    Vgroups are zero-padded digit strings ("06201"), but a CSV round-trip makes
    astropy infer an int64 column and the leading zero is lost -- so a table read
    back from disk holds 6201 while the correction still says "06201".  Compare
    numerically when both sides are digits, textually otherwise.
    """
    sa, sb = str(a).strip(), str(b).strip()
    if sa.isdigit() and sb.isdigit():
        return int(sa) == int(sb)
    return sa == sb


#: A JWST visit token is ``jw`` + proposal(5) + observation(3) + visit(3).
#: ``fix_alignment`` / ``_apply_consensus_offsets_table`` derive a frame's key as
#: ``os.path.basename(fn).split('_')[0]``, which always has this shape, so a table
#: row whose ``Visit`` does not can never be matched by anything.
VISIT_TOKEN_RE = re.compile(r"^jw\d{11}$")


def assert_visit_token(token, context):
    """Refuse a ``Visit`` value no frame filename can ever equal.

    The reachable failure is a JOINT multi-observation run: cataloging is invoked
    with ``--field 002-998`` (sgrb2 MIRI obs 002 + the obs 998 "redo" combined),
    and ``seed_offsets_table_from_consensus`` interpolates that straight into
    ``jw0{proposal}{field}{visit:03d}`` -> ``jw05365002-998001``.  Every frame of
    that run keys as ``jw05365002001`` or ``jw05365998001``, so NOTHING matches:
    ``lookup_consensus_offset`` returns ``(0.0, 0.0)`` for every exposure and the
    re-tie loop re-measures the identical residual forever while reporting that it
    wrote corrections.  A silent zero is exactly the failure mode this checkpoint
    exists to eliminate, so the malformed token is refused at BOTH ends: when it
    would be written, and if an already-written table is read back.
    """
    tok = str(token)
    if VISIT_TOKEN_RE.match(tok):
        return tok
    raise OffsetsTableUpdateError(
        f"{context}: visit token {tok!r} is not a JWST visit id "
        f"(jw<5-digit proposal><3-digit obs><3-digit visit>), so no frame "
        f"filename can ever match it and every lookup would silently return "
        f"(0, 0).  A JOINT multi-observation run (field like '002-998') produces "
        f"exactly this; a consensus table is per-observation, so seed/apply one "
        f"observation at a time (--field 002, then --field 998) or give the "
        f"corrections their real per-frame observation.")


def _finite_float(value, default=0.0):
    """``float(value)`` for a table cell that may be missing or masked."""
    if value is None or isinstance(value, np.ma.core.MaskedConstant):
        return default
    return float(value)


def vgroup_row_matches(row_value, wanted):
    """Does an offsets-table row whose ``Vgroup`` cell is ``row_value`` apply to
    visit group ``wanted``?

    An EMPTY cell means "visit group UNKNOWN", not "visit group nothing": it is
    what a row written before this column existed (or a row preserved from
    another filter by the builder's field-safe merge, which fills missing columns
    with '') reads back as.  Such a row must keep applying exactly as it did
    before the column was added -- narrowing it away would SILENTLY drop a
    correction a previous iteration had already accumulated onto it, which is the
    same class of failure as the curation collapse the checkpoints exist to
    prevent.  So an empty cell is a WILDCARD.

    The ambiguity that creates (an unknown-vgroup row AND a real-vgroup row for
    the same exposure both match) is caught loudly downstream: both
    ``lookup_consensus_offset`` and ``fix_alignment`` raise on a >1 match.
    """
    if vgroup_key(row_value) == "":
        return True
    return same_vgroup(row_value, wanted)


def _module_variants(module):
    """Match semantics of shift_individual_catalog: a detector-level module
    matches its own row or the module-family row."""
    m = str(module)
    if m.endswith("a") or m.endswith("b"):
        m = m + "long"
    return {m, m.strip("1234"), m.replace("long", "")}


#: The two column conventions an offsets table can carry, most-authoritative
#: first.  ``dra``/``ddec`` is generate_offsets_table's; ``dra (arcsec)``/
#: ``ddec (arcsec)`` is what the VIRAC2locked tables carry and what
#: ``unified_alignment`` actually reads.  ``build_virac2_offsets`` writes both,
#: the second as a COPY of the first, so a builder-shaped table starts with them
#: equal and they are two names for one quantity thereafter.
_DRA_COLUMN_PAIRS = (("dra (arcsec)", "ddec (arcsec)"), ("dra", "ddec"))

#: A disagreement this large between the two pairs is a real divergence rather
#: than float round-trip through CSV.  0.1 mas; the tightest gate in the tree is
#: 2 mas.
COLUMN_PAIR_TOL_ARCSEC = 1e-4


def _column_pairs(tbl):
    """Every ``(dra, ddec)`` column pair this table actually carries."""
    return [(d, c) for d, c in _DRA_COLUMN_PAIRS
            if d in tbl.colnames and c in tbl.colnames]


#: How closely ``(arcsec) - plain`` must match the accumulated ``prov_*`` for the
#: divergence to count as EXPLAINED.  What this absorbs is float round-trip
#: through CSV, nothing physical: measured across all ten live locked tables the
#: two agree to 0.000000 mas, so 0.5 mas is six orders of margin over the only
#: error source there is.
PROV_EXPLAINS_TOL_MAS = 0.5

#: Lower bound on cos(dec) over the fields this runs on -- all Galactic Centre or
#: nearer the equator, so |dec| < 30 deg.  Used to BOUND the RA-axis check: the
#: apply loop divides on-sky mas by cos(dec) and dec_deg is not stored per row,
#: so the exact factor is unrecoverable but confined to [COS_DEC_MIN, 1].
COS_DEC_MIN = np.cos(np.radians(30.0))


def _row_label(tbl, i):
    for col in ("Visit", "Filter"):
        if col not in tbl.colnames:
            return f"row {i}"
    return f"row {i} ({tbl['Visit'][i]} {tbl['Filter'][i]})"


def _heal_column_pairs(tbl, offsets_path, rows=None):
    """Re-sync a table's duplicate columns, or refuse if the gap is unexplained.

    Only the ``(arcsec)`` pair was ever written, so on every table on disk
    ``dra``/``ddec`` is the AS-BUILT value and ``dra (arcsec)``/``ddec (arcsec)``
    is as-built + everything applied.  That is not a guess: across all ten live
    ``*_VIRAC2locked.csv`` tables,

        max | ((arcsec) - plain)*1000 - prov_*_added_mas |  =  0.000000 mas

    so the gap is exactly the recorded provenance.  When that identity holds the
    plain pair carries no information the ``(arcsec)`` pair lacks, and the two can
    be re-synced by proof rather than by assumption -- loudly, and with the
    caller's usual ``.pre_<stage>`` backup taken before anything is written.

    Refusing instead would stop the m2 checkpoint on EVERY locked-channel field:
    all ten live tables diverge (brick x2, cloudc, cloudef, gc2211, quintuplet,
    sgra, sgrb2, sgrc, sickle), no caller catches OffsetsTableUpdateError, and it
    would take the campaign down rather than protect it.

    A gap the provenance does NOT explain is a different thing -- something
    edited one pair by hand, or applied a correction outside this function -- and
    that still refuses, because healing it would assert the ``(arcsec)`` pair is
    right when nothing on record says so.

    Both axes are checked before anything is written, because both are written.
    Dec is exact (prov and ddec are both on-sky); RA is bounded, since the apply
    loop divided by a cos(dec) this function cannot recover exactly.

    ``rows``: restrict to these row indices (the ones a correction will touch).
    A field with one stale filter and ten clean ones must be able to recover
    filter by filter; a table-wide refusal blocks the eleven together and breaks
    the natural recovery path, since ``build_virac2_offsets`` merges field-safely
    and rebuilding one filter re-equalises only its own rows.
    """
    pairs = _column_pairs(tbl)
    if len(pairs) < 2:
        return 0
    (da, ca), (db, cb) = pairs[0], pairs[1]
    d_gap = np.asarray(tbl[da], dtype=float) - np.asarray(tbl[db], dtype=float)
    c_gap = np.asarray(tbl[ca], dtype=float) - np.asarray(tbl[cb], dtype=float)
    diverged = ((np.abs(d_gap) > COLUMN_PAIR_TOL_ARCSEC)
                | (np.abs(c_gap) > COLUMN_PAIR_TOL_ARCSEC))
    if rows is not None:
        scope = np.zeros(len(tbl), dtype=bool)
        scope[np.asarray(list(rows), dtype=int)] = True
        diverged &= scope
    bad = np.where(diverged)[0]
    if not len(bad):
        return 0

    prov_d = (np.asarray(tbl["prov_dra_added_mas"], dtype=float)
              if "prov_dra_added_mas" in tbl.colnames else np.zeros(len(tbl)))
    prov_c = (np.asarray(tbl["prov_ddec_added_mas"], dtype=float)
              if "prov_ddec_added_mas" in tbl.colnames else np.zeros(len(tbl)))
    # DEC is exact: prov is on-sky mas and ddec is an on-sky arcsec offset, no
    # cos(dec) between them.
    dec_bad = np.abs(c_gap * 1000.0 - prov_c) > PROV_EXPLAINS_TOL_MAS

    # RA needs a BOUND rather than an equality.  The apply loop divides the
    # on-sky mas by cos(dec) to get the coordinate offset, and dec_deg is not
    # stored per row, so the exact factor is unrecoverable -- but it is confined
    # to [cos(dec_max), 1], which for these fields is a ~14% window.  That is far
    # tighter than needed to reject a hand-edited RA gap against a recorded zero,
    # and the heal WRITES this column, so it has to be checked: without it a row
    # whose Dec gap is explained and whose RA gap is not gets its dra silently
    # overwritten.
    lo = np.minimum(prov_d / 1000.0, prov_d / 1000.0 / COS_DEC_MIN)
    hi = np.maximum(prov_d / 1000.0, prov_d / 1000.0 / COS_DEC_MIN)
    slack = PROV_EXPLAINS_TOL_MAS / 1000.0
    ra_bad = (d_gap < lo - slack) | (d_gap > hi + slack)

    rogue = np.where(diverged & (dec_bad | ra_bad))[0]
    if len(rogue):
        i = int(rogue[0])
        if dec_bad[i]:
            axis, gap_i, prov_i, col_a, col_b, expect = (
                "Dec", c_gap[i], prov_c[i], ca, cb,
                f"prov_ddec_added_mas {prov_c[i]:.2f} mas")
        else:
            axis, gap_i, prov_i, col_a, col_b, expect = (
                "RA", d_gap[i], prov_d[i], da, db,
                f"prov_dra_added_mas {prov_d[i]:.2f} mas, i.e. a coordinate gap "
                f"in [{lo[i] * 1000:.2f}, {hi[i] * 1000:.2f}] mas after the "
                f"cos(dec) the apply loop divided by")
        raise OffsetsTableUpdateError(
            f"{os.path.basename(offsets_path)}: {len(rogue)} row(s) have "
            f"'{col_a}' disagreeing with '{col_b}' by more than the recorded "
            f"provenance explains on the {axis} axis -- {_row_label(tbl, i)}: "
            f"gap {gap_i * 1000:.2f} mas vs {expect}. Something changed one pair "
            f"outside update_offsets_table, so which one is right is not on "
            f"record. NOT writing.")

    worst = int(bad[np.argmax(np.abs(c_gap[bad]))])
    print(f"  {os.path.basename(offsets_path)}: re-syncing '{db}'/'{cb}' from "
          f"'{da}'/'{ca}' on {len(bad)} row(s) -- only the '(arcsec)' pair was "
          f"ever written, and the gap matches prov_*_added_mas exactly, so the "
          f"plain pair is the as-built value and carries nothing the other lacks. "
          f"Worst {_row_label(tbl, worst)}: {c_gap[worst] * 1000:.1f} mas.",
          flush=True)
    tbl[db][bad] = np.asarray(tbl[da], dtype=float)[bad]
    tbl[cb][bad] = np.asarray(tbl[ca], dtype=float)[bad]
    return int(len(bad))


def update_offsets_table(offsets_path, corrections, stage, out_path=None,
                         backup=True, pool=False):
    """Apply measured on-sky corrections to an offsets table, with provenance.

    ``corrections``: list of dicts with keys
      ``visit`` (int or 'jw...NNN'), ``exposure`` (int or None = whole visit),
      ``module`` (detector or family, or None = all), ``filtername``,
      ``dra_onsky_mas``/``ddec_onsky_mas`` (correction to ADD, on-sky),
      ``dec_deg`` (for the cos(dec) Δα conversion), ``source`` (free text).

    Table convention (generate_offsets_table.py):
      ``dra`` is the Δα COORDINATE in arcsec ->
      ``dra_new = dra + (dra_onsky_mas/1000)/cos(dec)``;
      ``ddec_new = ddec + ddec_onsky_mas/1000``.

    The corrected table is validated with ``assert_offsets_table_sane``
    (collapsed-visit guard) before it is written.  The original is kept as a
    ``.pre_<stage>_<timestamp>`` backup.  Every corrected row gets provenance
    columns (``prov_stage``, ``prov_date``, ``prov_dra_added_mas``,
    ``prov_ddec_added_mas``, ``prov_source``).

    Returns the corrected Table.  Raises ``OffsetsTableUpdateError`` when a
    correction matches no row or the corrected table fails validation.
    Read-modify-write, under a lock on ``offsets_path``: the table is shared by
    every filter of a proposal, so two filters' checkpoints correcting it at
    once would each read the original, and the second write would drop the
    first's correction.  The lock is taken in place rather than by a wrapper
    delegating to a private function -- moving that boundary lets another
    branch's new parameter and the block that reads it end up in different
    functions, with no merge conflict to say so.
    """
    with locked(offsets_path):
        from ..reduction.validate_offsets_table import (
            CollapsedOffsetsTableError, DivergedColumnPairError,
            assert_offsets_table_sane)

        # materialise first: the checks below iterate `corrections` before the apply
        # loop does, and a generator would be consumed by them -- leaving the update
        # a silent no-op that still writes a table and a backup.
        corrections = list(corrections)

        # magnitude ceiling FIRST: fail before touching the table at all, so a
        # spurious measurement cannot be half-applied or leave a backup behind.
        _assert_correction_magnitudes(corrections, offsets_path)

        tbl = Table.read(offsets_path)
        if "Visit" not in tbl.colnames:
            # every granularity guard below narrows on Visit; without it they die
            # on KeyError, which is the wrong class to escape a guarded writer.
            # Three real tables carry both column pairs and no Visit column
            # (brick's _average / _F405ref_average / _VVV_average).
            raise OffsetsTableUpdateError(
                f"{offsets_path} has no Visit column ({tbl.colnames}) -- a "
                f"correction cannot be matched to a row without one")
        # `pool=True` performs the collapse the guard below names.  Off by default,
        # so this function stays strict for every existing caller; the m2 checkpoint
        # pools explicitly before the actionability floor (it needs the pooled
        # magnitudes to decide whether to stop at all), and the recovery scripts
        # opt in with --pool.  Without this the guard's remedy named a function no
        # script called.
        if pool:
            corrections = pool_corrections_to_table_granularity(
                corrections, offsets_path, tbl=tbl)
        # all three granularity checks: a correction set must map 1:1 onto the
        # table's rows in EVERY dimension it is keyed by, or legal-sized corrections
        # sum.  The first two name the missing COLUMN (the actionable diagnosis);
        # the third is the general row-wise backstop that also catches a column
        # present at the WRONG granularity (family rows vs detector corrections).
        # Every one of these narrows through _match_rows, which is the only
        # place AmbiguousVisitMatchError comes from.  Caught HERE, at the first
        # site that can reach it, for the reason #331 gave when it wrapped the
        # validation errors three commits ago: callers and
        # run_field_retie_loop.sh are written around OffsetsTableUpdateError, so
        # anything else silently changes this function's error contract.
        #
        # The narrowing happens ONCE, here, for every correction -- BULK
        # INCLUDED -- and the result is reused below.  Both
        # `_assert_one_correction_per_row` and
        # `pool_corrections_to_table_granularity` `continue` on
        # `_is_bulk_correction`, so a bulk correction is never narrowed by the
        # guards; narrowing only inside them left bulk reaching `_match_rows`
        # for the first time ten lines lower, outside this try, and escaping as
        # a bare AmbiguousVisitMatchError.  Bulk is exactly what
        # `scripts/reduction/step0_bulk_offset.py` emits, with the bare
        # `zfill(3)` visit this error is about -- so the one shape the message
        # names was the one shape the wrap missed.
        try:
            _assert_module_granularity(corrections, tbl, offsets_path)
            _assert_vgroup_granularity(corrections, tbl, offsets_path)
            _assert_one_correction_per_row(corrections, tbl, offsets_path)
            _rows_for = [(corr, _match_rows(corr, tbl)) for corr in corrections]
        except AmbiguousVisitMatchError as ex:
            raise OffsetsTableUpdateError(
                f"cannot match a correction to a row; NOT writing:\n{ex}") from ex
        # both column conventions exist: 'dra'/'ddec' (generate_offsets_table) and
        # 'dra (arcsec)'/'ddec (arcsec)' (the VIRAC2locked tables fix_alignment
        # reads).  A builder-shaped table carries BOTH, and every pair present is
        # written -- see the apply loop.
        pairs = _column_pairs(tbl)
        if not pairs:
            raise OffsetsTableUpdateError(
                f"{offsets_path} has no dra/ddec columns ({tbl.colnames})")
        for col, fill in (("prov_stage", ""), ("prov_date", ""), ("prov_source", "")):
            if col not in tbl.colnames:
                tbl[col] = np.full(len(tbl), fill, dtype="U64")
        for col in ("prov_dra_added_mas", "prov_ddec_added_mas"):
            if col not in tbl.colnames:
                tbl[col] = np.zeros(len(tbl))
        # An INT offset column truncates every fractional correction to zero and,
        # once one pair is float and the other is not, locks the table into a
        # permanent disagreement no write can clear.  Coerce before anything is
        # compared or applied.  (The truncation itself predates this: both pairs
        # truncated identically, so nothing noticed.)
        for _dc, _cc in pairs:
            for _col in (_dc, _cc):
                if tbl[_col].dtype.kind != "f":
                    tbl[_col] = np.asarray(tbl[_col], dtype=float)

        now = _utcnow_iso()
        # Re-sync the duplicate columns on the rows these corrections will touch,
        # BEFORE applying anything -- adding the same increment to both pairs
        # preserves an existing gap rather than closing it.  Scoped to touched
        # rows so a field with one stale filter and ten clean ones recovers filter
        # by filter instead of being blocked as a whole.
        # Reuses the single narrowing pass above rather than repeating it, so
        # there is no second site that can raise and no way for the two to
        # disagree about which rows a correction touches.
        _touched = set()
        for _corr, _idx in _rows_for:
            _touched.update(int(i) for i in _idx)
        _heal_column_pairs(tbl, offsets_path, rows=_touched)

        for corr in corrections:
            if corr.get("exposure") is not None and "Exposure" not in tbl.colnames:
                # a per-VISIT (module-locked) table cannot express a single-exposure
                # correction -- applying it to the visit row would shift EVERY
                # exposure of the visit.  Refuse; the table must first be extended
                # to per-exposure rows (build_virac2_locked_perexp-style).
                raise OffsetsTableUpdateError(
                    f"correction for exposure {corr['exposure']} of visit "
                    f"{corr['visit']} cannot be applied to the per-visit table "
                    f"{offsets_path} (no Exposure column) -- extend the table to "
                    f"per-exposure rows first")
            # VGROUP narrowing (and everything else) lives in _match_rows, so the
            # guards above narrow EXACTLY the way this loop does.  ``vgroup_key`` --
            # NOT ``is not None`` -- because exposure_key stringifies a missing
            # VGROUP meta to the literal "None", which would otherwise narrow
            # against a token no row can ever carry ("matches NO row").
            idx = _match_rows(corr, tbl)
            match = np.zeros(len(tbl), dtype=bool)
            match[idx] = True
            wanted_vgroup = vgroup_key(corr.get("vgroup"))
            if (not wanted_vgroup and "Vgroup" in tbl.colnames
                    and corr.get("exposure") is not None and match.sum() > 1):
                # a per-EXPOSURE correction that does not know its vgroup, on a table
                # that does: the shift would be ADDED to every group's row.  That is
                # the accumulation _assert_vgroup_granularity refuses in the mirror
                # case (table cannot express it); refuse it here too.
                spans = {vgroup_key(g) for g in tbl["Vgroup"][match]}
                if len(spans - {""}) > 1:
                    raise OffsetsTableUpdateError(
                        f"correction {corr} carries NO visit group but matches rows "
                        f"from {sorted(spans)} in {offsets_path} -- applying it would "
                        f"add the same shift to every group's row.  The exposure "
                        f"number restarts per visit group, so the correction must "
                        f"name its group (visit_consensus.exposure_key carries it as "
                        f"key[4]).")
            if match.sum() == 0:
                raise OffsetsTableUpdateError(
                    f"correction {corr} matches NO row in {offsets_path} -- refusing "
                    f"a partial application (this is how silent curation errors start)")
            cosd = max(np.cos(np.radians(float(corr["dec_deg"]))), 1e-6)
            dra_add = (float(corr["dra_onsky_mas"]) / 1000.0) / cosd
            ddec_add = float(corr["ddec_onsky_mas"]) / 1000.0
            # Write EVERY column pair the table carries, not just the one the
            # reducer reads.  A builder-shaped table holds both conventions --
            # build_virac2_offsets ends with `t['dra (arcsec)'] = t['dra']`, a
            # COPY -- and correcting only `dra (arcsec)` leaves `dra` frozen at
            # the pre-correction value.  They then disagree silently and without
            # limit: cloudc and cloudef reached ~7.9 and ~7.3 ARCSEC of
            # divergence across their bulk repairs, on 95 and 96 of their rows.
            # fix_alignment reads the `(arcsec)` pair so the reductions were
            # right, but the plain pair is the validator's fallback and the first
            # thing a person reads, and two columns holding one quantity with
            # only one maintained is how curation errors start.
            for _dc, _cc in _column_pairs(tbl):
                tbl[_dc][idx] = np.asarray(tbl[_dc][idx], dtype=float) + dra_add
                tbl[_cc][idx] = np.asarray(tbl[_cc][idx], dtype=float) + ddec_add
            tbl["prov_stage"][idx] = str(stage)
            tbl["prov_date"][idx] = now
            tbl["prov_dra_added_mas"][idx] = (
                np.asarray(tbl["prov_dra_added_mas"][idx], dtype=float)
                + float(corr["dra_onsky_mas"]))
            tbl["prov_ddec_added_mas"][idx] = (
                np.asarray(tbl["prov_ddec_added_mas"][idx], dtype=float)
                + float(corr["ddec_onsky_mas"]))
            tbl["prov_source"][idx] = str(corr.get("source", "astrometry_checkpoint"))[:64]

        # CUMULATIVE drift bound.  The per-correction ceiling cannot see creep that
        # accumulates across successive calls -- five legal 0.4" corrections over
        # five re-tie iterations is 2" of silent drift, and cloudef reached 105" this
        # way.  The prov_* columns already accumulate, so the check is nearly free.
        # Bounded at the BULK limit: a row may legitimately carry a whole-visit
        # guide-star fix of arcseconds, but never more than a swept peak could mean.
        drift_limit = _positive_env_float("ASTROM_MAX_BULK_CORRECTION_ARCSEC",
                                          MAX_BULK_CORRECTION_ARCSEC)
        drift = np.hypot(np.asarray(tbl["prov_dra_added_mas"], dtype=float),
                         np.asarray(tbl["prov_ddec_added_mas"], dtype=float)) / 1000.0
        over = np.where(drift > drift_limit)[0]
        if len(over):
            worst = [(str(tbl["Visit"][i]), str(tbl["Filter"][i]),
                      round(float(drift[i]), 3)) for i in over[:6]]
            raise OffsetsTableUpdateError(
                f"{len(over)} row(s) of {os.path.basename(offsets_path)} have "
                f"accumulated more than {drift_limit}\" of correction across all "
                f"writes (prov_dra/ddec_added_mas) -- runaway feedback, not a "
                f"measurement.  NOT writing.  (visit, filter, |accumulated|\"): "
                f"{worst}")

        # collapsed-visit / sanity validation BEFORE anything is written.  A table
        # WE just corrected must not carry the collapse signature -- raise, don't warn
        # (that signature is exactly the curation failure this checkpoint exists to
        # prevent from ever being applied again).
        try:
            assert_offsets_table_sane(tbl, context=os.path.basename(offsets_path),
                                      raise_on_issue=True)
        # DivergedColumnPairError is caught alongside the collapse for the same
        # reason the collapse is: every caller and the retie loop are written
        # around OffsetsTableUpdateError, so anything else leaves this function
        # as a bare ValueError and silently changes its error contract.  It can
        # only be reached via OFFSETS_TABLE_DIVERGENCE_RAISE=1 (raise_on_issue
        # deliberately does NOT escalate a divergence), but an opt-in switch
        # that changes the exception TYPE a caller sees is still a trap.
        except (CollapsedOffsetsTableError, DivergedColumnPairError) as ex:
            raise OffsetsTableUpdateError(
                f"corrected offsets table failed validation; NOT writing:\n{ex}") from ex

        out_path = out_path or offsets_path
        if backup and os.path.exists(out_path):
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            # A COPY.  Moving the table aside and then rebuilding it leaves a
            # window with no table at all, and a reader in that window resolves
            # its shift from the no-table branch -- which means (0, 0).
            keep_a_copy(out_path, f"{out_path}.pre_{stage}_{stamp}")
        with atomic_write(out_path) as tmp_path:
            tbl.write(tmp_path, overwrite=True)
        return tbl


BULK_EXPOSURE = -1     # sentinel Exposure for the per-visit consensus->reference row
BULK_MODULE = "all"    # sentinel Module for the per-visit bulk row


def lookup_consensus_offset(tbl, visit, exposure, module, filtername, vgroup=None):
    """Return ``(dra_arcsec, ddec_arcsec)`` to apply to ONE exposure: the SUM of
    its per-exposure jitter row and the per-visit BULK (consensus->reference) row.

    The consensus table carries two kinds of row for a (visit, filter):

    * per-exposure JITTER rows (``Exposure``>=1, real ``Module``) -- SPARSE, only
      exposures that exceeded the 2 mas consensus tolerance get one.  Narrowing by
      Exposure+Module is UNCONDITIONAL: a lone jitter row belongs to some OTHER
      exposure, and applying it here would spuriously shift an already-aligned
      frame.  (Differs from the brick VIRAC2locked block, where a single row IS a
      per-visit bulk for every exposure and the ``sum()>1`` guard is correct.)
    * the per-visit BULK row (``Exposure``==BULK_EXPOSURE, ``Module``==BULK_MODULE)
      -- the ``consensus->reference`` tie (whole visit onto VIRAC2).  It applies
      to EVERY exposure of the visit/filter identically (one shift, no per-exposure
      reference noise -- the same policy as brick's per-visit bulk).

    Each frame therefore gets jitter (tie to the internal consensus) + bulk (tie
    that consensus to the absolute reference) = a direct tie to the reference.
    Exposures with neither row return ``(0.0, 0.0)``.  Raises ValueError if a
    jitter or bulk match is ambiguous (>1 row)."""
    # A row nothing can ever match is indistinguishable, at this call, from an
    # exposure that legitimately needed no correction -- both are (0, 0).  Refuse
    # the table instead of returning the zero (see assert_visit_token).
    bad = sorted({str(v) for v in tbl["Visit"] if not VISIT_TOKEN_RE.match(str(v))})
    if bad:
        raise OffsetsTableUpdateError(
            f"consensus table carries {len(bad)} Visit token(s) {bad[:4]} that are "
            f"not JWST visit ids (jw<5-digit proposal><3-digit obs><3-digit "
            f"visit>), so no frame filename can ever match them and every lookup "
            f"against them silently returns (0, 0).  A JOINT multi-observation "
            f"cataloging run (field like '002-998') writes exactly this; re-seed "
            f"one observation at a time.")

    vf = (tbl["Visit"] == visit) & (tbl["Filter"] == filtername)
    dra = ddec = 0.0

    # per-visit bulk (consensus->reference), applied to every exposure
    bulk = vf & (tbl["Exposure"] == BULK_EXPOSURE) & (tbl["Module"] == BULK_MODULE)
    nb = int(bulk.sum())
    if nb > 1:
        raise ValueError(f"consensus BULK match={nb} for visit={visit} "
                         f"filt={filtername}; expected <=1 row")
    if nb == 1:
        r = tbl[bulk]
        dra += float(r["dra (arcsec)"][0]); ddec += float(r["ddec (arcsec)"][0])

    # per-exposure jitter (exclude the bulk sentinel from the module variants).
    # _module_variants gives the SAME semantics as shift_individual_catalog
    # (including the 'long' family variant for LW modules) -- a previous inline
    # set here omitted the 'long' variants and could miss e.g. an 'nrcalong'
    # row when looking up module 'nrca'.
    variants = _module_variants(module)
    jit = (vf & (tbl["Exposure"] == int(exposure))
           & np.array([str(m) in variants for m in tbl["Module"]]))
    # exposure numbers restart per visit group, so a Vgroup-carrying table must be
    # narrowed by it -- otherwise two disjoint pointings collide on one exposure
    # number and the lookup below raises "match=2".  A row whose Vgroup cell is
    # EMPTY predates the column and still applies (vgroup_row_matches); a row that
    # names a DIFFERENT group does not.
    if vgroup_key(vgroup) and "Vgroup" in tbl.colnames:
        jit &= np.array([vgroup_row_matches(g, vgroup) for g in tbl["Vgroup"]])
    nj = int(jit.sum())
    if nj > 1:
        raise ValueError(
            f"consensus jitter match={nj} for visit={visit} exp={exposure} "
            f"mod={module} filt={filtername} vgroup={vgroup}; expected <=1 row"
            + ("" if "Vgroup" in tbl.colnames else
               "  (table has no Vgroup column; if this visit dithers across "
               "several visit groups, rebuild it with build_virac2_offsets)"))
    if nj == 1:
        r = tbl[jit]
        dra += float(r["dra (arcsec)"][0]); ddec += float(r["ddec (arcsec)"][0])

    return dra, ddec


def seed_offsets_table_from_consensus(basepath, proposal_id, field, corrections,
                                      stage="m2", out_path=None,
                                      base_stamp_for=None):
    """Create OR merge a per-exposure consensus offsets table (UPSERT).

    ``update_offsets_table`` can only *edit* existing rows -- a correction that
    matches no row hard-fails there.  For a field whose m2 checkpoint measured
    per-exposure misalignment but that has no brick/cloudc VIRAC2locked table
    (sgrc, cloudef, ...) that means: (1) the FIRST iteration has nowhere to
    record the fix, and (2) LATER iterations flag a slightly different set of
    exposures (as the applied tie removes the bulk jitter, exposures churn across
    the 2 mas line), so a newly-flagged exposure hard-fails update_offsets_table.
    Both break the re-tie loop.  This function UPSERTS instead:

      * no table yet  -> create it from the corrections (the seed);
      * table exists  -> for each correction ADD its on-sky shift to the matching
        (visit, filter, exposure, module) row (cumulative -- the correction is
        the RESIDUAL after the previous tie), or INSERT a new row when that
        exposure was not previously flagged.

    Each row's ``dra (arcsec)``/``ddec (arcsec)`` shift its exposure ONTO the
    dense internal consensus (removing the raw guide-star per-exposure jitter);
    exposures never flagged simply have no row (fix_alignment applies 0).

    Written in the ``dra (arcsec)`` Δα-coordinate convention ``fix_alignment``
    reads, keyed (Visit, Filter, Exposure, Module), at
    ``{basepath}/offsets/Offsets_JWST_Brick{proposal_id}_consensus.csv``.
    ``corrections`` uses the SAME dict schema as ``update_offsets_table``.
    Optional ``base_stamp_for`` maps (visit_tok, filter, exposure, module) ->
    ``{'calver':..,'crds_ctx':..,'dvacorr':..}`` for the genlock guard.  Returns
    the written path; raises OffsetsTableUpdateError on empty input or a table
    that fails the sparse-consensus sanity checks."""
    if not corrections:
        raise OffsetsTableUpdateError(
            "seed_offsets_table_from_consensus: no corrections to seed from")
    # The resulting-row check below bounds the ACCUMULATED value; this bounds
    # each individual correction, so a large shift cannot slip through by
    # partially cancelling an existing row.
    _assert_correction_magnitudes(corrections, out_path or "consensus table")
    out_path = out_path or os.path.join(
        basepath, "offsets",
        f"Offsets_JWST_Brick{proposal_id}_consensus.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    now = _utcnow_iso()
    # Same shared-table read-modify-write as update_offsets_table, and the
    # lock is taken in place for the same reason.
    with locked(out_path):

        # index any existing rows by key so corrections ADD-or-INSERT (upsert)
        existed = os.path.exists(out_path)
        bykey = {}
        if existed:
            for r in Table.read(out_path):
                row = {c: r[c] for c in r.colnames}
                # normalise the round-tripped cell ONCE (masked/'--'/int64 -> canonical
                # string) so the key, the migration below and the written column all
                # agree on one representation.
                row["Vgroup"] = vgroup_key(row.get("Vgroup"))
                key = (str(r["Visit"]), str(r["Filter"]), int(r["Exposure"]),
                       str(r["Module"]), row["Vgroup"])
                bykey[key] = row

        # Resolve every correction's identity ONCE: its upsert key, and the key the
        # SAME physical exposure carried before the Vgroup column existed.
        prepared = []
        for corr in corrections:
            visit = int(str(corr["visit"])[-3:])
            visit_tok = assert_visit_token(
                f"jw0{proposal_id}{field}{visit:03d}",
                f"seed_offsets_table_from_consensus({os.path.basename(out_path)})")
            # A consensus->reference correction is the per-VISIT bulk tie (whole
            # visit onto VIRAC2) -- it carries exposure=None AND module=None.  Store
            # it under the sentinel (BULK_EXPOSURE, BULK_MODULE) row so fix_alignment
            # applies it to EVERY exposure of the visit/filter (lookup_consensus_
            # offset sums bulk + per-exposure jitter).  Writing it to a real
            # exposure/module would either miss most frames or double-shift one.
            is_bulk = corr.get("exposure") is None and corr.get("module") is None
            exposure = BULK_EXPOSURE if is_bulk else int(corr["exposure"])
            module = BULK_MODULE if is_bulk else str(corr["module"])
            # VGROUP is part of the identity (exposure numbers restart per group);
            # BULK rows are visit-wide and carry the sentinel "" instead.  Canonicalise
            # here so a missing meta ("None" from exposure_key) cannot be written into
            # the column as a literal token nothing will ever match.
            vgroup = "" if is_bulk else vgroup_key(corr.get("vgroup"))
            key = (visit_tok, corr["filtername"], exposure, module, vgroup)
            prepared.append((corr, visit_tok, exposure, module, vgroup, key))

        # MIGRATION of pre-Vgroup rows.  A row written before this column existed
        # keys as "" (no vgroup), while the correction for the same physical exposure
        # now carries a real one -- so the exact key MISSES and the upsert would
        # INSERT a second row, silently orphaning whatever the old row had already
        # accumulated (the arches consensus table, the only one on disk, is exactly
        # this case: 85 per-exposure rows with no Vgroup).  Adopt the old row and
        # backfill its Vgroup instead.  If TWO groups would claim the same legacy row
        # it is a genuine blend of two pointings that cannot be split -- refuse
        # rather than guess which one inherits the accumulated shift.
        claims = {}
        for _c, visit_tok, exposure, module, vgroup, key in prepared:
            if not vgroup or key in bykey:
                continue
            legacy = (visit_tok, key[1], exposure, module, "")
            if legacy in bykey:
                claims.setdefault(legacy, set()).add(vgroup)
        for legacy, vgs in sorted(claims.items()):
            if len(vgs) > 1:
                raise OffsetsTableUpdateError(
                    f"{os.path.basename(out_path)} row {legacy[:4]} was written "
                    f"before the Vgroup column existed and now has corrections from "
                    f"{sorted(vgs)} -- it BLENDED those visit groups into one row, so "
                    f"there is no way to say which group its accumulated shift "
                    f"belongs to.  Rebuild the table (or move it aside and re-seed) "
                    f"before applying per-vgroup corrections.")
            vgroup = vgs.pop()
            row = bykey.pop(legacy)
            row["Vgroup"] = vgroup
            bykey[legacy[:4] + (vgroup,)] = row
            kept = tuple(_finite_float(row.get(c))
                         for c in ("prov_dra_added_mas", "prov_ddec_added_mas"))
            print(f"[consensus] migrated pre-Vgroup row {legacy[:4]} -> "
                  f"Vgroup={vgroup} (keeps its accumulated "
                  f"{kept[0]:+.2f},{kept[1]:+.2f} mas)", flush=True)

        for corr, visit_tok, exposure, module, vgroup, key in prepared:
            cosd = max(np.cos(np.radians(float(corr["dec_deg"]))), 1e-6)
            dra_add = (float(corr["dra_onsky_mas"]) / 1000.0) / cosd
            ddec_add = float(corr["ddec_onsky_mas"]) / 1000.0
            if key in bykey:
                row = bykey[key]
                row["dra (arcsec)"] = float(row["dra (arcsec)"]) + dra_add
                row["ddec (arcsec)"] = float(row["ddec (arcsec)"]) + ddec_add
                row["prov_dra_added_mas"] = (float(row.get("prov_dra_added_mas", 0.0))
                                             + float(corr["dra_onsky_mas"]))
                row["prov_ddec_added_mas"] = (float(row.get("prov_ddec_added_mas", 0.0))
                                              + float(corr["ddec_onsky_mas"]))
                row["prov_stage"] = str(stage)
                row["prov_date"] = now
                row["prov_source"] = str(corr.get("source", "m2 visit-consensus"))[:64]
                # REFRESH the genlock base stamp on upsert: the cumulative row's shift
                # is applied to THIS iteration's crf generation, so the base must track
                # it.  Keeping the first iteration's stamp would make the genlock guard
                # compare a later shift against a stale base if a jwst/CRDS bump landed
                # mid-loop (safe only while the generation is constant across the loop).
                if base_stamp_for is not None:
                    stamp = base_stamp_for.get(
                        (visit_tok, corr["filtername"], exposure, module)) or {}
                    for k in ("calver", "crds_ctx", "dvacorr"):
                        row[f"base_{k}"] = str(stamp.get(k, ""))
            else:
                row = {
                    "Filter": corr["filtername"], "Module": module, "Visit": visit_tok,
                    "Exposure": exposure, "Vgroup": vgroup,
                    "dra (arcsec)": dra_add, "ddec (arcsec)": ddec_add,
                    "prov_stage": str(stage), "prov_date": now,
                    "prov_dra_added_mas": float(corr["dra_onsky_mas"]),
                    "prov_ddec_added_mas": float(corr["ddec_onsky_mas"]),
                    "prov_source": str(corr.get("source", "m2 visit-consensus seed"))[:64],
                }
                if base_stamp_for is not None:
                    stamp = base_stamp_for.get(
                        (visit_tok, corr["filtername"], exposure, module)) or {}
                    for k in ("calver", "crds_ctx", "dvacorr"):
                        row[f"base_{k}"] = str(stamp.get(k, ""))
                bykey[key] = row

        rows = list(bykey.values())
        # NB: the visit-collapse guard (assert_offsets_table_sane / flag_collapsed_
        # visits) does NOT apply here.  It compares per-visit MEDIAN offsets against a
        # 20 mas tol to catch the brick-1182 curation signature (a visit's real ~arcsec
        # BULK offset overwritten by another's).  Consensus shifts are mas-scale, so
        # any two visits agree within 20 mas by construction -- flagging that would be
        # a category error.  A sparse per-exposure consensus table has two failure
        # modes worth guarding instead:
        keys = [(str(r["Visit"]), str(r["Filter"]), int(r["Exposure"]), str(r["Module"]),
                 vgroup_key(r.get("Vgroup", ""))) for r in rows]
        dups = sorted({k for k in keys if keys.count(k) > 1})
        if dups:
            # duplicate (visit,filter,exposure,module) -> lookup_consensus_offset
            # would raise; refuse to write an ambiguous table.
            raise OffsetsTableUpdateError(
                f"consensus table {os.path.basename(out_path)} has duplicate "
                f"(visit,filter,exposure,module) rows: {dups}")
        # A family COLLISION is as fatal as an exact duplicate and the exact-key
        # check above cannot see it: ('...','F360M',1,'nrcb','2101') and
        # ('...','F360M',1,'nrcblong','2101') are distinct keys, but
        # unified_alignment._read_consensus resolves a frame through
        # _module_variants -- which for a DETECTOR spelling adds its family and
        # for a BARE spelling adds `long` -- so both match one frame and it
        # refuses to reduce the field (issue #298).
        #
        # Only a BARE family token aliases.  A frame `nrcb3` matches rows
        # `nrcb3` and `nrcb`; `nrcblong` matches `nrcblong` and `nrcb`; but
        # `nrcb3` and `nrcb4` match nothing of each other's.  Grouping on the
        # family alone would refuse every legitimate per-detector table.
        #
        # An EMPTY row Vgroup is a wildcard at read time (vgroup_row_matches),
        # so it aliases across vgroups too -- arches once carried 85 such legacy
        # rows.  Grouping on the vgroup exactly would put them in a different
        # bucket and miss the collision entirely.
        #
        # SCOPE: refuse only for keys THIS write touches, and merely report a
        # pre-existing collision elsewhere in the table.  A table-wide refusal
        # would mean one filter's legacy rows hard-block every other filter's
        # seed with no escape hatch -- ASTROM_CHECKPOINT_WARN_ONLY is consulted
        # after this call, not before -- so cloudef's F360M rows would brick
        # F162M, F210M and F480M as well.  Use
        # scripts/reduction/unwind_alias_module_rows.py on the pre-existing ones.
        def _alias_bucket(k):
            return (k[0], k[1], k[2], _module_family(k[3]))

        # `prepared` is exactly what this write created or updated.
        touched = {_alias_bucket(k) for *_rest, k in prepared}
        fam = {}
        for k in keys:
            fam.setdefault(_alias_bucket(k), []).append(k)
        alias = []
        for bucket, members in sorted(fam.items()):
            mods = {m[3] for m in members}
            if len(mods) < 2 or bucket[3] not in mods:
                continue
            # vgroups must be able to collide: equal, or either one empty
            vgs = {m[4] for m in members}
            if len(vgs) > 1 and all(v for v in vgs):
                continue
            alias.append((bucket, tuple(sorted(mods)), tuple(sorted(vgs))))
        new_alias = [a for a in alias if a[0] in touched]
        if new_alias:
            raise OffsetsTableUpdateError(
                f"consensus table {os.path.basename(out_path)} would carry the "
                f"SAME frame under aliasing module spellings, which "
                f"unified_alignment resolves to both rows and refuses to read: "
                f"{new_alias[:5]}{'...' if len(new_alias) > 5 else ''} "
                f"({len(new_alias)} collision(s)).  This means the checkpoint "
                f"ingested one physical exposure twice under two module "
                f"tokens -- check for stale bare-module per-frame catalogs "
                f"next to their `long` counterparts (issue #298).")
        for a in alias:
            print(f"WARNING: consensus table {os.path.basename(out_path)} "
                  f"already carries {a[0]} under aliasing spellings {a[1]} "
                  f"(vgroups {a[2]}).  This write does not touch it, but any "
                  f"frame it describes CANNOT be read -- repair with "
                  f"scripts/reduction/unwind_alias_module_rows.py (issue #298).",
                  flush=True)
        big = [(k, r["dra (arcsec)"], r["ddec (arcsec)"]) for k, r in zip(keys, rows)
               if abs(float(r["dra (arcsec)"])) > 0.5 or abs(float(r["ddec (arcsec)"])) > 0.5]
        if big:
            # a consensus (internal-jitter) fix is mas-scale; > 0.5" means the
            # upstream per-exposure measurement is wrong -- do NOT bake it in.
            raise OffsetsTableUpdateError(
                f"consensus table {os.path.basename(out_path)} has |offset| > "
                f"0.5\" (mas-scale expected): {big}")
        tbl = Table(rows)
        if existed:
            # A COPY, as in update_offsets_table: the table must never be
            # absent, or a concurrent reader aligns at (0, 0).
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            keep_a_copy(out_path, f"{out_path}.pre_{stage}_{stamp}")
        with atomic_write(out_path) as tmp_path:
            tbl.write(tmp_path, overwrite=True)
        return out_path


# ---------------------------------------------------------------------------
# provenance header stamping (used by fix_alignment at re-apply time)
# ---------------------------------------------------------------------------

#: Words that mean "someone meant to fill this in".  A tripwire value belongs in
#: source, where it can stop a run; stamped into a product it becomes the
#: provenance, and it outlives the run that wrote it.
PLACEHOLDER_WORDS = frozenset(
    ('BUG', 'TODO', 'FIXME', 'XXX', 'PLACEHOLDER', 'UNKNOWN'))

#: Sentinels are written ``THIS_IS_A_BUG_IF_YOU_USE_THIS``, so the words have to
#: be found between underscores.  Splitting on non-alphanumerics does that, and
#: keeps ``DEBUG`` (one token, not the word ``BUG``) legitimate.
_WORD_SEPARATOR = re.compile(r'[^A-Za-z0-9]+')


def looks_like_placeholder(value):
    """Whether a value reads as "fill this in later" rather than as data."""
    return bool(PLACEHOLDER_WORDS.intersection(
        token.upper() for token in _WORD_SEPARATOR.split(str(value)) if token))


def assert_not_placeholder(value, what):
    """Raise unless ``value`` is something a product can record.

    Called before anything is written, so a tripwire that reaches a product
    stops the run at the start rather than between two writes.
    """
    if looks_like_placeholder(value):
        raise ValueError(
            f"refusing to record {value!r} as {what}: it reads as a "
            f"placeholder.  Give the frame the exposure was tied to, from the "
            f"resolved shift (AlignmentShift.reference_frame, declared in "
            f"reduction/alignment_config.py), or 'n/a'.")


def provenance_header_cards(stage, dra_onsky_mas, ddec_onsky_mas, method,
                            references, table_name):
    """FITS header cards recording WHY the current RAOFFSET/DEOFFSET are what
    they are.  ``fix_alignment`` stamps these when it (re-)applies a corrected
    offsets table — the header of every aligned frame then carries the full
    provenance of its astrometric fix."""
    assert_not_placeholder(references, 'the astrometric reference frame '
                                       '(APROVRF)')
    return [
        ("APROVST", str(stage), "astrometry-fix stage (checkpoint)"),
        ("APROVMT", str(method)[:48], "offset measurement method"),
        ("APROVDR", float(dra_onsky_mas), "[mas] applied on-sky dRA correction"),
        ("APROVDD", float(ddec_onsky_mas), "[mas] applied on-sky dDec correction"),
        ("APROVRF", str(references)[:48], "reference catalogs used"),
        ("APROVTB", os.path.basename(str(table_name))[:48], "offsets table"),
        ("APROVDT", _utcnow_iso(), "astrometry-fix date (UTC)"),
    ]


# ---------------------------------------------------------------------------
# the per-stage checkpoint
# ---------------------------------------------------------------------------

def _group_by_visit_filter(tables):
    groups = {}
    for tbl in tables:
        meta_visit = None
        meta_filter = None
        for key in ("VISIT", "Visit", "visit"):
            if key in tbl.meta:
                meta_visit = str(tbl.meta[key])
                break
        for key in ("FILTER", "filter"):
            if key in tbl.meta:
                meta_filter = str(tbl.meta[key])
                break
        groups.setdefault((meta_visit, meta_filter), []).append(tbl)
    return groups


def _record_name(stage, filtername, obs_token=""):
    """Single source of truth for a checkpoint record's base name.

    ``obs_token`` disambiguates observations that SHARE a record directory
    (issue #281).  cloudef's 2092 obs 002 and 005 both write to
    ``cloudef/astrometry_checkpoints/``, so without it the second run's
    ``checkpoint_m2_F360M_latest.json`` silently REPLACES the first's -- and
    every frozen-stage reader then compares o002's exposures against o005's
    baseline, which is not a movement measurement of anything.  The per-filter
    consensus catalog already carries the token for exactly this reason
    (``consensus_catalog.consensus_path``); the checkpoint records did not.

    The WRITER keys on ``run_visit_checkpoint``'s ``filtername`` argument, which
    is ``None`` for a mixed-filter run -> the record is stored under ``_all``.
    The READERS are handed the per-group ``filt`` parsed from table metadata (a
    real filter name, never None), so ``_record_name`` alone is not enough to
    close the gap -- see ``_m2_record_path`` for the fallback the readers use.
    """
    token = str(obs_token or '')
    return f"checkpoint_{stage}_{filtername or 'all'}{token}"


def _filter_is_obs_ambiguous(record_dir, filtername):
    """True when more than one observation of this field images ``filtername``.

    Only then is an untokened record genuinely ambiguous.  Brick's two
    observations use disjoint filter sets, so its untokened records are safe to
    read; cloudef's o002/o005, gc2211's five and ngc6334's two proposals share
    their filter lists, and those are the unsafe set.

    Fail-CLOSED: if the field cannot be determined, treat the filter as
    ambiguous.  Reading the wrong observation's baseline is a silent wrong
    answer; refusing is a loud unverified.
    """
    try:
        from ..monitoring.scan import shared_filters
        from .. import fields as _fields
    except ImportError:
        return True
    # The record dir is `<basepath>/astrometry_checkpoints`, so the field is
    # the path component NEAREST it.  First-in-dict-order resolved
    # `.../jwst/brick/scratch/cloudef/astrometry_checkpoints` to 'brick' and,
    # brick having no shared filters, read as unambiguous -- a silent wrong
    # read, the one outcome this function exists to prevent.  Longest-match
    # does not fix it either ('arches' and 'sickle' tie on length).
    #
    # Path-sniffing is the wrong instrument regardless; the caller knows the
    # field.  Until it is threaded through, walk from the record dir outward
    # and take the first component that names a field, which is deterministic
    # and right for every real layout.
    known = set(getattr(_fields, "BY_NAME", {}))
    target = None
    for part in reversed([p for p in str(record_dir).split(os.sep) if p]):
        if part in known:
            target = part
            break
    if target is None:
        return True
    try:
        # BOTH instruments: sgrb2 registers nircam ['001'] but miri
        # ['001','002','998'], so the nircam default returned False for all 14
        # of its genuinely shared filters.
        shared = set()
        for instrument in ("nircam", "miri"):
            shared |= {str(f).upper() for f in shared_filters(target, instrument)}
        return str(filtername).upper() in shared
    except (KeyError, TypeError, ValueError):
        return True


def _m2_refusal_reason(record_dir, filtername, obs_token=""):
    """Why there is no m2 baseline, when the reason is a REFUSAL.

    A refused untokened record and a genuinely absent one both reach the
    caller as ``None``, and the caller then emits the frozen-stage movement
    failure -- which asserts that the solution moved.  That is a false
    statement about the data: nothing was measured to have moved, there was
    simply nothing to compare against.  Naming the refusal lets the message
    say what happened.
    """
    if not (record_dir and obs_token):
        return None
    tokened = os.path.join(
        record_dir, f"{_record_name('m2', filtername, obs_token)}_latest.json")
    if os.path.exists(tokened):
        return None
    for legacy in (os.path.join(record_dir,
                                f"{_record_name('m2', filtername)}_latest.json"),
                   os.path.join(record_dir,
                                f"{_record_name('m2', None)}_latest.json")):
        if os.path.exists(legacy) and _filter_is_obs_ambiguous(record_dir,
                                                               filtername):
            return (f"the untokened m2 record {os.path.basename(legacy)} was "
                    f"REFUSED for {filtername}{obs_token}: more than one "
                    f"observation of this field images this filter and an "
                    f"untokened record body carries no observation identity "
                    f"(issue #281).  Re-run m2 to write a tokened record")
    return None


def _m2_record_path(record_dir, filtername, obs_token=""):
    """Resolve the latest m2 record path for a per-group filter, tolerating the
    writer/reader spelling gap.

    The three m2 baseline readers receive the per-group ``filt`` (e.g. 'F212N'),
    while the writer may have keyed the record on a ``None`` ``filtername``
    argument and stored it under ``checkpoint_m2_all``.  A bare
    ``checkpoint_m2_{filt}`` lookup then MISSES -- and a missed m2 baseline reads
    as "no m2 record", which at a frozen stage fails closed and stops a healthy
    field.  Try the exact-filter spelling first; fall back to the ``_all``
    spelling with a LOUD line so the fallback is never silent.  Returns ``None``
    when neither exists (a genuine no-baseline).
    """
    if not record_dir:
        return None
    # Tokened spelling first.  The untokened LEGACY spelling is accepted only
    # where it CANNOT be another observation's: an untokened record body
    # carries no observation identity at all (`visit` is "1" for both
    # jw02092002001 and jw02092005001), so on a filter that more than one
    # observation images, falling back is not a degraded read -- it is the
    # exact hazard this function exists to stop, with a warning attached.
    # Verified: cloudef's untokened checkpoint_m2_{F162M,F210M,F360M,F480M}
    # records are on disk now, nothing deletes them, and o002's m3 read o005's
    # baseline through this path.
    #
    # Where the filter is unambiguous (brick's two observations use disjoint
    # filter sets) the legacy record is this run's own and is read.
    for _tok in ([obs_token, ""] if obs_token else [""]):
        _p = os.path.join(record_dir,
                          f"{_record_name('m2', filtername, _tok)}_latest.json")
        if not os.path.exists(_p):
            continue
        if obs_token and not _tok:
            # A `None` filtername means this IS the mixed-filter `_all`
            # lookup.  Ambiguity is a property of a FILTER, so with no filter
            # to ask about there is no answer and it fails closed.
            if filtername is None or _filter_is_obs_ambiguous(record_dir,
                                                              filtername):
                print(f"astrom checkpoint: REFUSING the untokened m2 record "
                      f"{os.path.basename(_p)} for {filtername}{obs_token}: "
                      f"more than one observation of this field images this "
                      f"filter, and an untokened record body carries no "
                      f"observation identity, so it may be the other one's "
                      f"(issue #281).  Re-run m2 to write a tokened record.",
                      flush=True)
                # BREAK, not `return None`.  Refusing the per-filter legacy
                # record says nothing about this run's OWN `_all` record, and
                # returning here let a legacy untokened file on disk hide a
                # perfectly good tokened mixed-filter one -- failing closed
                # against the wrong file.
                break
            print(f"astrom checkpoint: no tokened m2 record for "
                  f"{filtername}{obs_token}; falling back to the untokened "
                  f"{os.path.basename(_p)}.  Only one observation of this "
                  f"field images this filter, so it is unambiguous.",
                  flush=True)
        return _p
    # `exact` is named only for the message below.  It is NOT re-checked: the
    # loop above already tried it as its first iteration and would have
    # returned it, so an `if os.path.exists(exact): return exact` here is
    # unreachable -- verified by replacing it with `raise AssertionError`.
    exact = os.path.join(record_dir,
                         f"{_record_name('m2', filtername, obs_token)}_latest.json")
    # The writer keys the mixed-filter record with the token too, so the
    # reader must look for it there or a tokened run can never find it.
    allpath = os.path.join(
        record_dir, f"{_record_name('m2', None, obs_token)}_latest.json")
    if not os.path.exists(allpath):
        allpath = os.path.join(
            record_dir, f"{_record_name('m2', None)}_latest.json")
        # The SAME refusal as the per-filter branch above, which was missing
        # here.  `run_astrometry_checkpoint.py`'s `--filter` defaults to None,
        # so ONE filterless invocation creates an untokened `_all` record that
        # every observation of the field would then read as its own -- the
        # per-filter gate closed while the `_all` door stood open.  A `None`
        # filtername cannot be tested for ambiguity at all, so it fails closed.
        if obs_token and os.path.exists(allpath) and (
                filtername is None
                or _filter_is_obs_ambiguous(record_dir, filtername)):
            print(f"astrom checkpoint: REFUSING the untokened mixed-filter m2 "
                  f"record {os.path.basename(allpath)} for "
                  f"{filtername}{obs_token}: it carries no observation "
                  f"identity, and more than one observation of this field "
                  f"images this filter (issue #281).  Re-run m2 to write a "
                  f"tokened record.", flush=True)
            return None
    if filtername and os.path.exists(allpath):
        print(f"astrom checkpoint: no m2 baseline for filter {filtername!r} at "
              f"{os.path.basename(exact)}; falling back to "
              f"{os.path.basename(allpath)} (the m2 record was written for a None "
              f"filtername argument -- a mixed-filter run)", flush=True)
        return allpath
    return None


def _visit_entry_matches(v, visit, filtername):
    """Does a record's per-(visit, filter) entry match this reader's request?

    Match on visit AND on ``filtername`` when the entry carries one.  An ``_all``
    (mixed-filter) m2 record holds a SEPARATE entry per filter for the same visit
    (``run_visit_checkpoint`` groups by ``(visit, filter)`` and stamps each entry
    with ``filtername``), so a bare visit match would hand filter X's reader
    filter Y's numbers -- the alphabetically-first entry for that visit.  Legacy
    entries with no ``filtername`` field fall back to visit-only matching, which
    is the pre-existing behaviour for a per-filter record (all its entries are the
    requested filter anyway).
    """
    if str(v.get("visit")) != str(visit):
        return False
    vf = v.get("filtername")
    return vf is None or str(vf) == str(filtername)


def _m2_reference_tie_baseline(record_dir, filtername, visit, obs_token=""):
    """(dra_mas, ddec_mas) of the m2-frozen consensus->reference tie for this
    (filter, visit), from the latest m2 record; None when unavailable.

    Reads the REPORTED bulk (``reference_tie['dra_mas']/['ddec_mas']``) -- the
    SAME quantity the frozen-stage (m3+) delta gate compares against
    (``ref_tie['dra_mas']``).  That reported bulk is the same-star refined offset
    when available; comparing the current same-star tie against ``vs_full`` (the
    histogram check A) instead would compute a spurious ~several-mas "movement"
    equal to the histogram-vs-same-star method difference -- a FALSE regression at
    every frozen stage (observed: brick F182M m3 "MOVED 5.86 mas", m2 vs_full
    (+6.70,-7.54) [histogram] vs m3 same-star (+1.11,-5.77), 2026-07-19).  Falls
    back to ``vs_full`` only for legacy records that predate the reported-bulk
    field.

    Returns ``(None, reason)``-style information via the second element of the
    tuple: ``(baseline, m2_rejected)``.  ``m2_rejected`` is True when m2 measured
    a tie but REFUSED it (``apply_ok`` False -- no coherent dense peak, a gross
    sparse-Gaia split, or a failed per-tile/same-star gate).  A refused tie is a
    rejected measurement, not a freeze point: nothing was applied, so there is no
    frozen value for a later stage to have moved away from.  Reading it as a
    baseline turns an IMPROVEMENT into a regression -- w51 F140M (2026-08-02): m2
    rejected a 7827 mas swept-histogram peak (``apply_ok=False``,
    ``per_tile clean=False``, ``swept=True``) and said so in ``unverified``; m3
    then measured a clean 32 mas SAME-STAR tie (``apply_ok=True``,
    ``swept=False``) and raised "consensus->reference MOVED 7794.98 mas since the
    m2 freeze", blocking the field because the measurement got better.
    """
    path = _m2_record_path(record_dir, filtername, obs_token)
    if path is None:
        return None, False
    with open(path) as fh:
        rec = json.load(fh)
    for v in rec.get("visits", []):
        if not _visit_entry_matches(v, visit, filtername):
            continue
        rt = v.get("reference_tie") or {}
        rejected = rt.get("apply_ok") is False
        dra, ddec = rt.get("dra_mas"), rt.get("ddec_mas")
        if rejected:
            # A REFUSED tie is still a MEASUREMENT.  Hand the numbers back with
            # the rejected flag so the caller can compare against them and demote
            # only on failure -- discarding them throws away a stability result
            # that exists and passes.  sgra F212N: m2 refused a 48.49 mas tie
            # (independent checks disagreed) and m3 lands 0.41 mas away, which is
            # the strongest evidence the solution did not move; an earlier
            # revision of this function returned None here and turned that
            # verified PASS into UNVERIFIED.
            if dra is not None and ddec is not None \
                    and np.isfinite(dra) and np.isfinite(ddec):
                return (float(dra), float(ddec)), True
            return None, True
        if dra is not None and ddec is not None \
                and np.isfinite(dra) and np.isfinite(ddec):
            return (float(dra), float(ddec)), False
        vf = rt.get("vs_full") or {}   # legacy record without the reported bulk
        if "dra" in vf and "ddec" in vf:
            return (float(vf["dra"]), float(vf["ddec"])), False
    return None, False


def _m2_exposure_baseline(record_dir, filtername, visit, obs_token=""):
    """Map exposure-key tuple -> (dra_mas, ddec_mas) of the m2 per-exposure
    vs-consensus offset, from the latest m2 record; ``{}`` when unavailable.

    The frozen-stage per-exposure gate is a MOVEMENT check (mirror of
    ``_m2_reference_tie_baseline`` for the consensus->reference tie): an exposure
    fails only when its vs-consensus offset MOVED since the m2 freeze, not when
    its absolute vs-consensus offset merely exceeds ``EXPOSURE_CONSENSUS_TOL_MAS``.
    That absolute magnitude is intrinsic per-exposure centroid scatter -- already
    present and tolerated at the m2 (correcting) stage -- so re-checking it
    absolutely at every frozen stage re-trips on noise that never moved (observed:
    brick F115W m3, 14 exposures 2.0-3.0 mas off consensus reading the SAME
    2.0-3.3 mas at m2; the bluest/sparsest filter's intrinsic scatter sits in the
    dead-zone between the m2 correction floor (4 mas) and this 2 mas tol, so it
    could NEVER pass a frozen stage, 2026-07-20).
    """
    out = {}
    path = _m2_record_path(record_dir, filtername, obs_token)
    if path is None:
        return out
    with open(path) as fh:
        rec = json.load(fh)
    for v in rec.get("visits", []):
        if not _visit_entry_matches(v, visit, filtername):
            continue
        for e in v.get("exposures", []) or []:
            key = tuple(e.get("key", []) or [])
            dra, ddec = e.get("dra"), e.get("ddec")
            if key and dra is not None and ddec is not None \
                    and np.isfinite(dra) and np.isfinite(ddec):
                out[key] = (float(dra), float(ddec))
    return out


def _m2_skipped_exposures(record_dir, filtername, visit, obs_token=""):
    """Set of exposure-key tuples m2 DELIBERATELY left out of its consensus.

    ``build_visit_consensus`` drops an exposure with too few reliable stars and
    records it in ``consensus['skipped']``.  Such an exposure never received a
    frozen solution, so at a frozen stage it has no ``_m2_exposure_baseline``
    entry -- indistinguishable, from the baseline map alone, from a frame that
    appeared out of nowhere after the freeze.  The two need opposite verdicts:
    an unexplained new frame is a REGRESSION (the solution was supposed to be
    frozen), while an m2-skipped one is a known, recorded exclusion whose first
    measurement happens at m3 and therefore cannot have "moved" since m2.

    Observed on arches F212N (2026-08-02): a snowball storm in exposure 4 (JUMP_DET
    1.2% -> 7.6%, 261 blobs >100 px vs 9) cut its source count ~31% on all eight
    detectors, so m2 skipped all eight; m3 then measured them 12-18 mas off the
    consensus and raised ``AstrometryRegressionError``, killing the m4-m8 chain
    over a data-quality defect m2 had already found, reported, and worked around.
    """
    path = _m2_record_path(record_dir, filtername, obs_token)
    if path is None:
        return set()
    with open(path) as fh:
        rec = json.load(fh)
    out = set()
    for v in rec.get("visits", []):
        if not _visit_entry_matches(v, visit, filtername):
            continue
        cons = v.get("consensus") or {}
        for key in cons.get("skipped", []) or []:
            out.add(tuple(key))
    return out


def _m2_consensus_stars(record_dir, basepath, filtername, obs_token=""):
    """(SkyCoord of the m2 consensus stars, path) for the same-star gate.

    The path comes from the m2 RECORD's own ``consensus_catalog`` field where
    it has one, not from recomputing the token.  Recomputing keyed the star
    list and the baseline differently: the consensus catalog is obs-tokenised
    while the checkpoint record was not, so on cloudef -- where
    ``obs_token`` returns '' for proposal 2092 and the o002/o005 m2 runs
    interleave into one record file -- an o002 run could restrict against
    o005's star list, 104 mas away, while comparing against a baseline written
    by either.  Reading the path the m2 run itself recorded ties the two to the
    same run for nothing.

    Returns ``(None, path)`` when the catalog is absent -- a field whose m2
    predates the per-filter consensus catalog, or a filter m2 could not pool.
    The caller says so loudly and falls back: a missing baseline is not
    evidence the solution moved.
    """
    from .consensus_catalog import consensus_path
    path = None
    # NB _m2_record_path grows an obs_token parameter in PR #306; this call
    # deliberately does not pass one so the two branches stay independent.
    rec_path = _m2_record_path(record_dir, filtername) if record_dir else None
    if rec_path:
        try:
            with open(rec_path) as fh:
                path = json.load(fh).get("consensus_catalog")
        except (OSError, ValueError) as ex:
            print(f"astrom checkpoint: could not read consensus_catalog from "
                  f"{rec_path} ({type(ex).__name__}: {ex})", flush=True)
    if not path and basepath:
        path = consensus_path(basepath, filtername, obs_token=obs_token)
    if not (filtername and path and os.path.exists(path)):
        return None, path
    try:
        tbl = Table.read(path)
    except (OSError, ValueError) as ex:
        print(f"astrom checkpoint: m2 consensus catalog {path} unreadable "
              f"({type(ex).__name__}: {ex}); same-star gate disabled", flush=True)
        return None, path
    if not len(tbl):
        return None, path
    coords = catalog_coords(tbl)
    finite = np.isfinite(coords.ra.deg) & np.isfinite(coords.dec.deg)
    return (coords[finite] if finite.any() else None), path


def run_visit_checkpoint(exposure_tables, stage, refcat=None, filtername=None,
                         basepath=None, record_dir=None, context="",
                         consensus_kwargs=None, obs_token=""):
    """Run the per-(visit, filter) consensus checkpoint over per-frame catalogs.

    Parameters
    ----------
    exposure_tables : list of Table
        Per-frame catalogs (one per exposure/detector) of ONE filter, any
        number of visits — grouped internally by (visit, filter).
    stage : str
        Merge stage token ('m2' for the m12 merge, 'm3'..'m6').
    refcat : dict or None
        ``load_reference_catalog`` output (keys ``all``, ``sparse``, ``mag``).
        When None the reference tie is skipped (consensus-only checkpoint).
    record_dir : str or None
        Where to write the checkpoint record
        (default ``{basepath}/astrometry_checkpoints``).
    obs_token : str
        Per-observation filename disambiguator (``crowdsource_catalogs_long.obs_token``)
        for the per-filter consensus catalog this writes at m2.  Fields that
        share a target directory across proposals or obsids (ngc6334 6778/7213,
        cloudef 2092 obs 002/005) MUST pass it or the second run overwrites the
        first field's reference catalog.

    Returns
    -------
    dict — the full checkpoint record, with:
      ``visits``: per-(visit, filter) results (consensus, per-exposure offsets,
      reference tie);
      ``corrections``: the offsets-table corrections implied (empty at a late
      stage unless it ALSO raised);
      ``passed``: True when nothing moved beyond tolerance.

    Raises
    ------
    AstrometryRegressionError
        At a late stage (m3+) when any exposure or the reference tie moved
        beyond ``STAGE_STABILITY_TOL_MAS`` (unless
        ``ALLOW_LATE_STAGE_ASTROM_SHIFT=1``).
    """
    stage = str(stage)
    correcting = stage in CORRECTION_STAGES
    record_dir = record_dir or (os.path.join(basepath, "astrometry_checkpoints")
                                if basepath else None)
    consensus_kwargs = dict(consensus_kwargs or {})

    visits = []
    # The consensus catalogs THEMSELVES, keyed by visit.  `visits` below carries
    # a JSON-able SUMMARY of each (star count, median scatter, ...) for the
    # record file; the summary has no positions in it, so pooling must read
    # these instead.
    consensus_by_visit = {}
    corrections = []
    failures = []      # MEASURED shifts -- blocking at a late stage
    unverified = []    # could-not-verify -- loud warnings, audited by the gate
    # The subset that is MEASURED-AND-REFUSED rather than could-not-measure.
    # Only this blocks the gate; see _checkpoint_passed.
    unverified_blocking = []
    # SAME-STAR restriction at a frozen stage (issue #285).  A later stage fits
    # on a background-subtracted image and detects a DIFFERENT star set, so its
    # rebuilt consensus sits a few mas from m2's even when no frame moved --
    # and the gate, which compares (exposure - consensus) against m2's, then
    # attributes that consensus movement to every exposure in the visit.
    # Freezing the star LIST to m2's (positions still come from this stage)
    # removes the population change from the comparison and leaves the
    # movement.  At a CORRECTING stage there is nothing to freeze against and
    # the full star set is the right one.
    m2_stars, m2_stars_source = (None, None)
    if not correcting and basepath:
        m2_stars, m2_stars_source = _m2_consensus_stars(
            record_dir, basepath, filtername, obs_token)
        if m2_stars is None:
            print(f"astrom checkpoint [{stage}] {filtername}: no m2 consensus "
                  f"catalog at {m2_stars_source} -- the stage-stability check "
                  f"falls back to the FULL star set, so a population change "
                  f"between stages can still read as movement (issue #285)",
                  flush=True)
        else:
            print(f"astrom checkpoint [{stage}] {filtername}: same-star gate "
                  f"against {len(m2_stars)} m2 consensus stars "
                  f"({os.path.basename(m2_stars_source)})", flush=True)

    for (visit, filt), tables in sorted(_group_by_visit_filter(exposure_tables).items()):
        vctx = f"{context} {filt} visit {visit} [{stage}]"
        # The visit NUMBER groups and reports; the corrections this emits must
        # carry the full jwPPPPPOOOVVV id, or a multi-observation offsets table
        # cannot tell which of its pointings they belong to (see
        # resolve_full_visit_id).  Grouping stays on the bare number so record
        # keys and frozen-stage baselines are unchanged.
        corr_visit = resolve_full_visit_id(tables, visit)
        try:
            cons = build_visit_consensus(tables, context=vctx,
                                         restrict_to=m2_stars,
                                         **consensus_kwargs)
        except DuplicateExposureError as ex:
            # malformed INPUTS, not a sparse field.  Recording this as merely
            # "unverified" would let a duplicated exposure silently delete the
            # gate for this visit/filter -- the same shape as the defect that
            # motivated the check.  Fail.
            visits.append(dict(visit=visit, filtername=filt, consensus=None,
                               error=str(ex), error_kind="duplicate_exposure"))
            failures.append(f"{vctx}: duplicate exposure identity: {ex}")
            continue
        except ConsensusBuildError as ex:
            visits.append(dict(visit=visit, filtername=filt, consensus=None,
                               error=str(ex), error_kind="consensus_build"))
            unverified.append(f"{vctx}: consensus build failed: {ex}")
            continue

        # ---- per-exposure vs consensus ------------------------------------
        # At a frozen stage the per-exposure gate is a MOVEMENT check vs the m2
        # baseline (see _m2_exposure_baseline), NOT an absolute vs-consensus
        # magnitude check -- the latter re-trips on intrinsic per-exposure
        # scatter that m2 already tolerated.
        exp_baseline = ({} if correcting
                        else _m2_exposure_baseline(record_dir, filt, visit,
                                                  obs_token))
        # An exposure m2 deliberately skipped has no baseline BY CONSTRUCTION;
        # that absence is not evidence the frozen solution moved.
        m2_skipped = (set() if correcting
                      else _m2_skipped_exposures(record_dir, filt, visit,
                                                 obs_token))
        # issue #158 backstop: an ALIAS reads antisymmetric across the modules of
        # an exposure, where real jitter is common-mode.  Never emit corrections
        # from an antisymmetric set -- they are the footprint geometry, not a
        # misalignment (and they are above the appliable ceiling anyway, so this
        # costs no capability; it replaces an opaque stop with a diagnosis).
        antisym = detect_module_antisymmetry(cons["exposures"])
        if antisym["detected"]:
            ex = antisym["examples"][0]
            # BLOCKING: a number was measured and refused.  The message itself
            # says the consensus "should be rebuilt/investigated" -- that is not
            # a pass.
            unverified_blocking.append(
                f"{vctx}: MODULE-ANTISYMMETRIC offsets on "
                f"{antisym['n_antisymmetric']}/{antisym['n_pairs_tested']} "
                f"exposure(s) -- module {ex['module_a']} reads "
                f"({ex['dra_a_mas']:+.0f},{ex['ddec_a_mas']:+.0f}) mas and module "
                f"{ex['module_b']} reads ({ex['dra_b_mas']:+.0f},"
                f"{ex['ddec_b_mas']:+.0f}) mas, i.e. equal and OPPOSITE at "
                f"{ex['separation_mas'] / 1000.0:.1f}\" apart.  Real per-exposure "
                f"jitter is common-mode across an exposure's detectors, so this "
                f"is a wide-sweep/footprint-geometry ALIAS, not a misalignment "
                f"(issue #158).  NOT correcting; the affected exposures are "
                f"UNVERIFIED and the visit consensus for this filter should be "
                f"rebuilt/investigated")
            unverified.append(unverified_blocking[-1])
        antisym_keys = antisym["keys"]
        # A swept, window-edge peak on a detector whose OTHER exposures were all
        # rejected as footprint-ridge aliases is the same ridge (#158/#347).
        sibling_alias_keys = detector_sibling_alias_keys(cons["exposures"])
        exp_records = []
        for exp in cons["exposures"]:
            res = exp["vs_consensus"]
            rec = dict(key=list(exp["key"]), n_reliable=exp["n_reliable"],
                       raoffset_meta=exp["raoffset_meta"],
                       deoffset_meta=exp["deoffset_meta"],
                       component=exp.get("component", 0),
                       internal_tie=exp.get("internal_tie", True),
                       unverified=exp.get("unverified", False),
                       alias_suspect=bool(tuple(exp["key"]) in antisym_keys),
                       gross_diagnostic=_jsonable(exp.get("gross_diagnostic")),
                       misaligned=exp["misaligned"])
            if res is not None:
                rec.update({k: res.get(k) for k in
                            ("dra", "ddec", "off", "npairs", "contrast", "ok",
                             "swept", "window_arcsec", "dra_err", "ddec_err",
                             "n_peak", "window_edge_fraction",
                             "window_consistent", "alias_rejected")})
            exp_records.append(rec)
            if exp.get("unverified"):
                gd = exp.get("gross_diagnostic")
                extra = ""
                if gd is not None:
                    extra = (f"  Wide-sweep diagnostic: peak {gd['off'] / 1000.0:.1f}\" "
                             f"at the {gd['window_arcsec']:.0f}\" window "
                             f"(contrast {gd['contrast']:.1f}, off/window="
                             f"{gd.get('window_edge_fraction', float('nan')):.2f}, "
                             f"reproduced at an independent window: "
                             f"{gd.get('window_consistent')}) -- recorded, NOT "
                             f"applied; a per-exposure tie is mas-scale, so a gross "
                             f"frame belongs to the per-visit bulk path.")
                unverified.append(
                    f"{vctx}: exposure {exp['key']} has no measurable tie to the "
                    f"visit consensus (isolated footprint / too few overlap "
                    f"stars) -- internally UNVERIFIED; the reference tie is its "
                    f"only check.{extra}")
            if exp["misaligned"] and tuple(exp["key"]) in sibling_alias_keys:
                # Rejected with its siblings, and reported the same way they
                # are: this detector produced no tie anywhere in the visit, so
                # the exposure is UNVERIFIED, not misaligned.
                msg = (f"{vctx}: exposure {exp['key']} peak "
                       f"{res['off'] / 1000.0:.1f}\" at the "
                       f"{res.get('window_arcsec')}\" window "
                       f"(contrast {res.get('contrast')}, off/window="
                       f"{res.get('window_edge_fraction'):.2f}) is rejected "
                       f"with its DETECTOR's sibling exposures, which were "
                       f"alias-rejected at the same window -- a footprint "
                       f"ridge is a property of the detector's geometry, and "
                       f"reproducing at a second window does not distinguish "
                       f"it (issue #347).  NOT correcting.")
                print(f"ASTROM CHECKPOINT [{stage}] ALIAS (not correcting): "
                      f"{msg}", flush=True)
                unverified.append(msg)
            elif exp["misaligned"] and tuple(exp["key"]) in antisym_keys:
                # antisymmetric alias: recorded above at the visit level, never
                # corrected, never a late-stage regression (the number it would
                # be compared against is not a measurement of anything).
                print(f"ASTROM CHECKPOINT [{stage}] ALIAS (not correcting): "
                      f"{vctx} exposure {exp['key']} "
                      f"({res['dra']:+.0f},{res['ddec']:+.0f}) mas is "
                      f"module-antisymmetric -- see the MODULE-ANTISYMMETRIC "
                      f"note above (issue #158)", flush=True)
            elif exp["misaligned"]:
                msg = (f"{vctx}: exposure {exp['key']} is "
                       f"{res['off']:.2f} mas off the visit consensus "
                       f"(dra={res['dra']:.2f}±{res.get('dra_err', float('nan')):.2f}, "
                       f"ddec={res['ddec']:.2f}±{res.get('ddec_err', float('nan')):.2f}, "
                       f"swept={res.get('swept')})")
                gross = _gross_per_exposure_offset(res)
                if correcting and gross is not None:
                    # A per-exposure tie is mas-scale.  An arcsecond-scale one is
                    # the wide-sweep/footprint-geometry regime, which the
                    # UNVERIFIED path above already refuses to apply -- but an
                    # exposure only reaches that path when it has no measurable
                    # tie at all.  One that produced a peak, at the same
                    # arcsecond scale, fell through to here and was emitted as a
                    # correction; `_assert_correction_magnitudes` then rejected
                    # the whole batch, so ONE such exposure took every valid
                    # correction in the visit down with it (o023, 38963501: 24
                    # mas-scale corrections lost to a single -9.28" nrcb1 peak
                    # whose three sibling exposures were rejected as #158
                    # aliases).  Refuse it here instead: blocking-unverified, so
                    # the run still stops for it, and the visit's real
                    # corrections are still written.
                    unverified_blocking.append(
                        f"{vctx}: exposure {exp['key']} measured {gross} "
                        f"-- a per-exposure tie is mas-scale, so this is the "
                        f"wide-sweep/footprint-geometry regime, not a "
                        f"per-exposure misalignment.  NOT corrected; a gross "
                        f"frame belongs to the per-visit BULK path.  "
                        f"(dra={res['dra'] / 1000.0:+.3f}\", "
                        f"ddec={res['ddec'] / 1000.0:+.3f}\", "
                        f"contrast={res.get('contrast')}, "
                        f"off/window={res.get('window_edge_fraction')}, "
                        f"reproduced at an independent window: "
                        f"{res.get('window_consistent')})")
                    unverified.append(unverified_blocking[-1])
                    print(f"ASTROM CHECKPOINT [{stage}] GROSS (not correcting): "
                          f"{unverified_blocking[-1]}", flush=True)
                elif correcting:
                    dec_mid = float(np.median(cons["coords"].dec.deg))
                    corrections.append(dict(
                        visit=corr_visit, exposure=exp["key"][1],
                        module=exp["key"][2], filtername=filt,
                        # vgroup (key[4]) is carried so the WRITE path can tell
                        # two same-numbered exposures in different visit groups
                        # apart.  The offsets tables have no Vgroup column, so
                        # they cannot express the distinction -- update_offsets_
                        # table refuses rather than summing both onto one row.
                        vgroup=(exp["key"][4] if len(exp["key"]) > 4 else None),
                        dra_onsky_mas=res["dra"], ddec_onsky_mas=res["ddec"],
                        dec_deg=dec_mid,
                        source=f"{stage} visit-consensus"))
                    print(f"ASTROM CHECKPOINT [{stage}] CORRECT: {msg}", flush=True)
                else:
                    # FROZEN stage: flag only a MOVEMENT since the m2 freeze, not
                    # a nonzero absolute offset (intrinsic per-exposure scatter
                    # that m2 already tolerated -- else the bluest filters can
                    # never pass; brick F115W m3, 2026-07-20).
                    base = exp_baseline.get(tuple(exp["key"]))
                    if base is not None:
                        delta = float(np.hypot(res["dra"] - base[0],
                                               res["ddec"] - base[1]))
                        if delta > STAGE_STABILITY_TOL_MAS:
                            failures.append(
                                f"{vctx}: exposure {exp['key']} MOVED "
                                f"{delta:.2f} mas since the m2 freeze "
                                f"(m2=({base[0]:+.2f},{base[1]:+.2f}), now="
                                f"({res['dra']:+.2f},{res['ddec']:+.2f}) mas)")
                        else:
                            print(f"ASTROM CHECKPOINT [{stage}] STABLE: {vctx} "
                                  f"exposure {exp['key']} unchanged since m2 "
                                  f"(delta {delta:.2f} mas <= "
                                  f"{STAGE_STABILITY_TOL_MAS}; absolute "
                                  f"{res['off']:.2f} mas is intrinsic scatter)",
                                  flush=True)
                    elif tuple(exp["key"]) in m2_skipped:
                        # m2 EXCLUDED this exposure from its consensus (too few
                        # reliable stars -- a data-quality defect m2 found and
                        # recorded).  It never got a frozen solution, so its
                        # first vs-consensus measurement lands here and cannot
                        # be a movement.  Report it as UNVERIFIED so the release
                        # record's all_verified goes false and item 0b of
                        # RELEASE_DEPLOYMENT_CHECKLIST.md asks a human to check
                        # it -- NOTE that is a manual step, not a mechanism:
                        # all_verified has no non-test reader and
                        # stage_release.py never opens astrometry_checkpoints/.
                        # Automating it is its own PR (it refuses 12 of 14
                        # fields as things stand, so the triage is the work).
                        # Report it, and let the
                        # frozen chain run: raising here re-punishes a defect
                        # that is already handled (arches F212N exposure 4).
                        unverified.append(
                            msg + " [m2 SKIPPED this exposure from its consensus"
                            " (too few reliable stars); no frozen baseline"
                            " exists, so this is its first measurement, not a"
                            " movement -- the exposure's own data quality is"
                            " the thing to investigate]")
                        print(f"ASTROM CHECKPOINT [{stage}] UNVERIFIED "
                              f"(m2-skipped): {msg}", flush=True)
                    else:
                        # No m2 baseline for this exposure (new/renamed frame at
                        # a frozen stage): the solution was supposed to be frozen
                        # -- fall back to the absolute-offset failure.
                        _refused = _m2_refusal_reason(record_dir, filt,
                                                      obs_token)
                        if _refused:
                            # NOT a movement.  The frozen-stage text asserts
                            # the solution moved; nothing here was measured to
                            # have moved, there was simply nothing to compare
                            # against.  Still blocking -- an unverifiable
                            # frozen stage is not a pass, and `all_verified`
                            # has no non-test reader -- but the message must
                            # not claim a measurement that was never made.
                            failures.append(
                                f"{vctx}: exposure {exp['key']} CANNOT BE "
                                f"CHECKED against the m2 freeze -- {_refused}. "
                                f"This is a MISSING BASELINE, not a measured "
                                f"movement: its current vs-consensus offset is "
                                f"{res['off']:.2f} mas and no frozen value "
                                f"exists to compare it to.")
                        else:
                            failures.append(
                                msg + " [no m2 per-exposure baseline: "
                                "frozen-stage exposure absent from the m2 "
                                "record]")

        # ---- consensus vs absolute reference ------------------------------
        ref_tie = None
        if refcat is not None:
            ref_tie = measure_reference_tie(
                cons["coords"], refcat["all"], refcat["sparse"],
                filtername=filt, consensus_mag=cons.get("mag"),
                ref_mag=refcat.get("mag"), dense=refcat.get("dense", True),
                context=vctx)
            off = ref_tie["off_mas"]
            if np.isfinite(off) and off > REFERENCE_APPLY_MIN_MAS:
                if ref_tie["apply_ok"]:
                    if correcting:
                        dec_mid = float(np.median(cons["coords"].dec.deg))
                        corrections.append(dict(
                            visit=corr_visit, exposure=None, module=None,
                            filtername=filt,
                            dra_onsky_mas=ref_tie["dra_mas"],
                            ddec_onsky_mas=ref_tie["ddec_mas"],
                            dec_deg=dec_mid,
                            source=f"{stage} consensus->reference"))
                        print(f"ASTROM CHECKPOINT [{stage}] CORRECT: {vctx} "
                              f"consensus is {off:.2f} mas off VIRAC2 "
                              f"(coherent dense tie, per-tile clean, no gross "
                              f"sparse-Gaia split)", flush=True)
                    else:
                        # FROZEN stage: regression = the tie MOVED since the
                        # m2 freeze (> STAGE_STABILITY_TOL_MAS), not a nonzero
                        # absolute residual -- m2 legitimately PASSes with an
                        # unactionable (could-not-verify / sub-floor) residual,
                        # which every later stage necessarily re-measures
                        # (brick V12 F182M: m2 10.09 mas PASS, m3 10.31 mas ->
                        # false REGRESSION, 2026-07-16).
                        base, m2_rejected = _m2_reference_tie_baseline(
                            record_dir, filt, visit, obs_token)
                        if base is not None:
                            delta = float(np.hypot(ref_tie["dra_mas"] - base[0],
                                                   ref_tie["ddec_mas"] - base[1]))
                            if delta <= STAGE_STABILITY_TOL_MAS:
                                # Stable against the m2 measurement, whether or
                                # not m2 chose to APPLY it.  What m2 froze is the
                                # crf GWCS, physically; `apply_ok: false` says the
                                # ABSOLUTE tie is uncertified, not that the
                                # consensus was free to move.  Two measurements of
                                # the same quantity agreeing is evidence either
                                # way, so this keeps sgra F212N's verified pass.
                                note = ("; m2 apply_ok=False, absolute tie still "
                                        "uncertified" if m2_rejected else "")
                                print(f"ASTROM CHECKPOINT [{stage}] STABLE: {vctx} "
                                      f"tie unchanged since m2 (delta "
                                      f"{delta:.2f} mas <= "
                                      f"{STAGE_STABILITY_TOL_MAS}{note})",
                                      flush=True)
                            elif m2_rejected:
                                unverified.append(
                                    f"{vctx}: consensus->reference moved {delta:.2f} mas "
                                    f"from a tie m2 MEASURED but REFUSED "
                                    f"(m2=({base[0]:+.2f},{base[1]:+.2f}) apply_ok=False, "
                                    f"now=({ref_tie['dra_mas']:+.2f},"
                                    f"{ref_tie['ddec_mas']:+.2f}) mas). A refused tie is "
                                    f"not a frozen solution, so this is not a regression "
                                    f"-- the field's ABSOLUTE tie is what needs "
                                    f"investigating (bulk_source="
                                    f"{ref_tie.get('bulk_source')}, "
                                    f"swept={ref_tie.get('swept')})")
                                print(f"ASTROM CHECKPOINT [{stage}] UNVERIFIED "
                                      f"(moved {delta:.2f} mas from an m2-REFUSED "
                                      f"tie): {vctx}", flush=True)
                            else:
                                failures.append(
                                    f"{vctx}: consensus->reference MOVED "
                                    f"{delta:.2f} mas since the m2 freeze "
                                    f"(m2=({base[0]:+.2f},{base[1]:+.2f}), now="
                                    f"({ref_tie['dra_mas']:+.2f},"
                                    f"{ref_tie['ddec_mas']:+.2f}) mas)")
                        elif m2_rejected:
                            # m2 measured a tie and REFUSED it as untrustworthy.
                            # Nothing was frozen, so nothing can have moved; this
                            # is the first trustworthy measurement of the tie.
                            # UNVERIFIED, not a regression -- see w51 F140M
                            # above.  all_verified goes false, which today is a
                            # MANUAL checklist item (0b), not an enforced gate.
                            unverified.append(
                                f"{vctx}: consensus->reference offset {off:.2f} mas -- m2 "
                                f"MEASURED but REFUSED its own tie (untrustworthy), so no "
                                f"frozen baseline exists and this is the first trustworthy "
                                f"measurement, not a movement. The field's ABSOLUTE tie is "
                                f"what needs investigating, not a late-stage shift "
                                f"(bulk_source={ref_tie.get('bulk_source')}, "
                                f"swept={ref_tie.get('swept')})")
                            print(f"ASTROM CHECKPOINT [{stage}] UNVERIFIED "
                                  f"(m2 refused its own tie): {vctx} "
                                  f"consensus->reference {off:.2f} mas", flush=True)
                        else:
                            failures.append(
                                f"{vctx}: consensus->reference offset {off:.2f} mas at a "
                                f"LATE stage (solution was supposed to be frozen; "
                                f"no m2 baseline record found)")
                else:
                    # apply_ok is False only for a genuinely bad tie now: no
                    # coherent dense peak, a GROSS sparse split (spurious peak),
                    # or -- per reference regime -- the gating check failed. Point
                    # the investigator at the check that actually gated: per-tile
                    # cleanliness for a DENSE reference, the same-star refinement
                    # for a Gaia-only one (where per-tile is noise, not a signal).
                    if ref_tie.get("reference_dense", True):
                        gate = f"per-tile clean={ref_tie['per_tile'].get('clean')}"
                    else:
                        gate = ("same-star refined="
                                f"{ref_tie.get('same_star') is not None} "
                                "[Gaia-only ref: per-tile map is noise, not gating]")
                    # BLOCKING: m2 MEASURED the offset and declined to apply it.
                    # This is the cloudc F410M/nrcblong case in #312 -- 731 mas,
                    # over REFERENCE_CROSSCHECK_GROSS_MAS, reported as a pass.
                    unverified_blocking.append(
                        f"{vctx}: consensus->reference offset {off:.2f} mas but the "
                        f"tie is not trustworthy "
                        f"(cross-ref sep={ref_tie['cross_reference'].get('sep_mas'):.1f} mas, "
                        f"gross_ok={ref_tie.get('cross_reference_gross_ok')}, "
                        f"{gate}, "
                        f"swept={ref_tie.get('swept')}) -- NOT applying; investigate")
                    unverified.append(unverified_blocking[-1])

        consensus_by_visit[visit] = cons
        visits.append(dict(
            visit=visit, filtername=filt,
            consensus=dict(
                n_stars=int(len(cons["coords"])),
                anchor=list(cons["anchor_key"]),
                median_scatter_mas=float(np.median(cons["scatter_mas"]))
                if len(cons["scatter_mas"]) else float("nan"),
                consensus_ok=cons["consensus_ok"],
                skipped=[list(k) for k in cons["skipped"]],
                # The POPULATION change, recorded whether or not the same-star
                # gate is on.  A stage detecting FEWER stars than m2 is a
                # regression worth seeing (issue #285 asked why a catalog went
                # 3212 -> 1707); the same-star restriction stops that change
                # from being read as astrometric MOVEMENT, it does not make it
                # uninteresting.  `restricted` is the count entering the tie,
                # `unrestricted` what this stage detected before the m2 star
                # list was applied.
                # RECORD FORMAT: `same_star_gate` is a STRING
                # ("applied" / "refused" / "unavailable"), not a bool -- so
                # `if rec["same_star_gate"]:` is truthy for all three.  No
                # reader in the repo does that today; a new one must compare
                # the value.  Note also that `cons["exposures"]` holds only the
                # USABLE exposures, so a refusal on an exposure that landed in
                # `cons["skipped"]` is not counted here.
                #
                # What ACTUALLY happened, not whether a star list was found.
                # `m2_stars is not None` was true whenever a catalog existed,
                # so on cloudef the record asserted the same-star gate ran
                # against 40,124 stars while it was refused for all 16
                # exposures -- the record stated the opposite of the run.
                # ALL, not `any`.  `any(not refused)` reported "applied" when
                # one exposure of sixteen restricted, which was true of the
                # label and false of the run.  The restriction is now
                # all-or-nothing per visit (see build_visit_consensus), so
                # `all` is also the accurate spelling: "applied" means every
                # exposure in this consensus was restricted to the m2 stars.
                same_star_gate=(
                    "applied" if (m2_stars is not None and cons["exposures"]
                                  and all(not e.get("restrict_refused")
                                          for e in cons["exposures"]))
                    else ("refused" if m2_stars is not None else "unavailable")),
                same_star_refused=[
                    dict(key=list(e["key"]), reason=e["restrict_refused"])
                    for e in cons["exposures"] if e.get("restrict_refused")],
                n_same_star_refused=int(sum(
                    1 for e in cons["exposures"] if e.get("restrict_refused"))),
                # RECORDED, never gated (see build_visit_consensus): the
                # smallest fraction of the m2 star list falling inside any one
                # exposure's footprint.  A visit-wide list legitimately covers
                # more sky than one exposure, so there is no threshold -- but a
                # list that is half a DIFFERENT POINTING passes survival and
                # tie alike, because the foreign half simply never matches.
                # Computed in the consensus and previously never carried into
                # the record, which is the same defect as `restrict_refused`.
                restrict_list_coverage=(
                    min([c for c in (e.get("restrict_list_coverage")
                                     for e in cons["exposures"])
                         if c is not None], default=None)),
                n_reliable_restricted=int(sum(
                    e["n_reliable"] for e in cons["exposures"])),
                n_reliable_unrestricted=int(sum(
                    e.get("n_reliable_unrestricted", e["n_reliable"])
                    for e in cons["exposures"])),
                m2_consensus_stars=(int(len(m2_stars))
                                    if m2_stars is not None else None)),
            module_antisymmetry=dict(
                detected=antisym["detected"],
                n_pairs_tested=antisym["n_pairs_tested"],
                n_antisymmetric=antisym["n_antisymmetric"],
                min_mas=antisym["min_mas"],
                keys=[list(k) for k in sorted(antisym["keys"])],
                examples=[_jsonable(x) for x in antisym["examples"]]),
            exposures=exp_records,
            reference_tie=_jsonable(ref_tie)))

    # Persist the filter's consensus.  build_visit_consensus measures one
    # (visit, filter) at a time because detecting a misaligned exposure means
    # comparing it against its OWN visit's exposures; pooling those into one
    # per-filter catalog is what the rest of the pipeline ties to, and until
    # now it was discarded when this function returned.
    consensus_catalog_path = None
    if basepath and stage == 'm2' and consensus_by_visit:
        filt_for_file = filtername or next(
            (v.get('filtername') for v in visits if v.get('filtername')), None)
        try:
            pooled = pool_visit_consensi(consensus_by_visit, context=context)
            consensus_catalog_path = write_filter_consensus(
                basepath, filt_for_file, pooled, obs_token=obs_token)
            print(f"ASTROM CHECKPOINT [{stage}]: wrote per-filter consensus "
                  f"{consensus_catalog_path} ({len(pooled)} stars, "
                  f"{pooled.meta['NVISITS']} visits, worst inter-visit "
                  f"{pooled.meta['IVMAXMAS']:.1f} mas)", flush=True)
        except (ValueError, TypeError, OSError) as ex:
            # Not fatal: the catalog is a product of the checkpoint, not an
            # input to it, and the alignment decisions above are already
            # made.  Loud, because a missing one breaks the reference-filter
            # tie later, and recorded in `consensus_catalog_error` so the
            # release gate can see it without scraping stdout.
            consensus_catalog_path = None
            consensus_catalog_error = f"{type(ex).__name__}: {ex}"
            print(f"ASTROM CHECKPOINT [{stage}]: could NOT write the "
                  f"per-filter consensus: {consensus_catalog_error}",
                  flush=True)
        else:
            consensus_catalog_error = None
    else:
        consensus_catalog_error = None

    passed = _checkpoint_passed(failures, unverified_blocking)
    record = dict(stage=stage, filtername=filtername, context=context,
                  consensus_catalog=consensus_catalog_path,
                  consensus_catalog_error=consensus_catalog_error,
                  date=_utcnow_iso(), correcting=correcting, visits=visits,
                  corrections=corrections, failures=failures,
                  unverified=unverified, unverified_blocking=unverified_blocking,
                  passed=passed, all_verified=not unverified,
                  tolerances=dict(
                      exposure_consensus_tol_mas=EXPOSURE_CONSENSUS_TOL_MAS,
                      reference_apply_min_mas=REFERENCE_APPLY_MIN_MAS,
                      stage_stability_tol_mas=STAGE_STABILITY_TOL_MAS))
    if record_dir:
        _write_record(record_dir, _record_name(stage, filtername, obs_token),
                      record)

    for w in unverified:
        print(f"ASTROM CHECKPOINT [{stage}] COULD NOT VERIFY: {w}", flush=True)
    if unverified_blocking and not failures:
        # Say which way it went and how to proceed deliberately.  Before #312
        # this printed the same COULD NOT VERIFY lines and then reported a
        # pass, so the lines read as advisory.
        if _env_flag(ALLOW_UNVERIFIED_ENV):
            print(f"ASTROM CHECKPOINT [{stage}]: PASSING with "
                  f"{len(unverified_blocking)} blocking unverified item(s) because "
                  f"{ALLOW_UNVERIFIED_ENV}=1 -- the checkpoint could not "
                  f"confirm these are aligned.", flush=True)
        else:
            print(f"ASTROM CHECKPOINT [{stage}]: NOT A PASS -- "
                  f"{len(unverified_blocking)} item(s) were MEASURED and refused (see "
                  f"above).  A gross offset is exactly what m2 refuses to "
                  f"correct, so a refusal that still passed let the retie "
                  f"loop call a 4\" misalignment converged (#312).  Set "
                  f"{ALLOW_UNVERIFIED_ENV}=1 to proceed anyway.", flush=True)
    if failures and not correcting:
        msg = (f"ASTROMETRY REGRESSION at stage {stage}: the solution moved after "
               f"it was frozen --\n  " + "\n  ".join(failures))
        if _env_flag("ALLOW_LATE_STAGE_ASTROM_SHIFT"):
            print(f"WARNING (override ALLOW_LATE_STAGE_ASTROM_SHIFT=1): {msg}",
                  flush=True)
        else:
            raise AstrometryRegressionError(msg)
    return record


# ---------------------------------------------------------------------------
# the cross-filter (m7) checkpoint
# ---------------------------------------------------------------------------

def run_crossfilter_checkpoint(catalogs_by_filter, refcat=None, basepath=None,
                               record_dir=None, context="",
                               tol_mas=CROSSFILTER_TOL_MAS,
                               cell_arcsec=LOCAL_CELL_SIZE_ARCSEC,
                               cell_tol_mas=LOCAL_CELL_TOL_MAS,
                               cell_min_stars=LOCAL_CELL_MIN_STARS,
                               field_cell_arcsec=CROSSFILTER_FIELD_CELL_ARCSEC,
                               field_min_stars=CROSSFILTER_FIELD_MIN_STARS,
                               obs_token=""):
    """Cross-filter astrometry agreement at the cross-band merge.

    The filter closest in wavelength to VIRAC2 Ks anchors the absolute frame
    (checked against the reference with the full multi-check tie when
    ``refcat`` is given).  Every other filter must agree with the anchor to
    < ``tol_mas`` bulk (histogram + sweep), and the matched-pair local residual
    map must show no significant ``cell_arcsec`` cell above ``cell_tol_mas``.

    ``catalogs_by_filter``: dict filtername -> Table (vetted merged catalog).

    Raises ``CrossFilterAstrometryError`` on any failure (override only via
    ``ALLOW_CROSSFILTER_ASTROM_FAIL=1``).
    """
    if len(catalogs_by_filter) < 2:
        # Carry the new keys: an audit rule keyed on `all_verified is not True`
        # would otherwise refuse a legitimately single-filter field, or KeyError.
        return dict(passed=True, skipped="single filter", filters=[],
                    unverified=[], all_verified=True)
    record_dir = record_dir or (os.path.join(basepath, "astrometry_checkpoints")
                                if basepath else None)
    anchor_filter = pick_reference_anchor_filter(list(catalogs_by_filter))
    anchor_tbl = catalogs_by_filter[anchor_filter]
    anchor_keep = select_reliable_stars(anchor_tbl)
    anchor_coords = catalog_coords(anchor_tbl)[anchor_keep]

    anchor_tie = None
    if refcat is not None:
        anchor_tie = measure_reference_tie(
            anchor_coords, refcat["all"], refcat["sparse"],
            filtername=anchor_filter, ref_mag=refcat.get("mag"),
            dense=refcat.get("dense", True),
            context=f"{context} anchor {anchor_filter}")

    filters = []
    failures = []
    unverified = []   # measured nothing -- reported, never a pass on its own
    unverified_blocking = []
    if anchor_tie is not None:
        if not anchor_tie["vs_full"] or not anchor_tie["vs_full"].get("ok"):
            failures.append(f"anchor {anchor_filter}: no coherent tie to the reference")
        elif not anchor_tie.get("cross_reference_gross_ok", False):  # fail-closed default
            # Only a GROSS dense-vs-sparse split (spurious/window-limited VIRAC
            # peak) blocks. A fine ~5-10 mas Gaia-sparse split does NOT: in the GC
            # Gaia is the frame, not the reference catalog, and is too sparse to
            # tie a dense field (memory: gc-gaia-frame-not-catalog).
            failures.append(
                f"anchor {anchor_filter}: dense vs sparse reference GROSSLY DISAGREE "
                f"({anchor_tie['cross_reference'].get('sep_mas'):.1f} mas > "
                f"{anchor_tie.get('cross_reference_gross_tol_mas')} mas) -- "
                f"VIRAC tie likely a spurious/window-limited peak")
        elif not anchor_tie.get("per_tile_ok"):
            # DENSE reference: per-tile map D; Gaia-ONLY reference: same-star
            # refinement A' (measure_reference_tie picks the right one per regime).
            detail = ("per-tile reference map not clean" if anchor_tie.get("reference_dense", True)
                      else "same-star tie could not be refined (Gaia-only reference)")
            failures.append(f"anchor {anchor_filter}: {detail}")

    for filt, tbl in sorted(catalogs_by_filter.items()):
        if filt == anchor_filter:
            continue
        keep = select_reliable_stars(tbl)
        coords = catalog_coords(tbl)[keep]
        fctx = f"{context} {filt} vs anchor {anchor_filter}"
        bulk = measure_offset(coords, anchor_coords, sweep=True, context=fctx)
        frec = dict(filtername=filt, n_reliable=int(keep.sum()),
                    bulk=_jsonable(bulk), local=None, field=None)
        if bulk is None or not bulk.get("ok"):
            failures.append(f"{fctx}: NO coherent cross-filter tie ({bulk})")
        else:
            err = float(np.hypot(bulk.get("dra_err", 0.0) or 0.0,
                                 bulk.get("ddec_err", 0.0) or 0.0))
            if bulk["off"] > tol_mas and (not np.isfinite(err) or bulk["off"] > 3 * err):
                failures.append(
                    f"{fctx}: bulk offset {bulk['off']:.2f} mas > {tol_mas} mas "
                    f"(dra={bulk['dra']:.2f}±{bulk.get('dra_err', float('nan')):.2f}, "
                    f"ddec={bulk['ddec']:.2f}±{bulk.get('ddec_err', float('nan')):.2f})")
            if bulk.get("swept"):
                failures.append(f"{fctx}: tie only found by window SWEEP "
                                f"({bulk['off']:.0f} mas) -- grossly shifted")
            if bulk.get("swept") or bulk["off"] >= 100.0:
                # The local map is skipped on these paths, so NOTHING local was
                # checked.  A large-but-finite error bar can also suppress the
                # bulk failure above (`off > 3*err` false), leaving a record
                # that PASSED having measured nothing -- the same shape this
                # branch exists to stop.
                unverified.append(
                    f"{fctx}: bulk tie {bulk['off']:.1f} mas "
                    f"(swept={bulk.get('swept')}) skipped the local cell map "
                    f"entirely -- no local check ran")
            if not bulk.get("swept") and bulk["off"] < 100.0:
                local = local_residual_map(
                    coords, anchor_coords, bulk, cell_arcsec=cell_arcsec,
                    min_stars=cell_min_stars, tol_mas=cell_tol_mas,
                    context=fctx)
                frec["local"] = _jsonable_local(local)
                # Coherent position-dependent term (measurement only, never
                # gates -- see measure_residual_field).  The bulk gate above
                # and the 2" cell gate below are both blind to it.
                field = measure_residual_field(coords, anchor_coords, bulk,
                                               cell_arcsec=field_cell_arcsec,
                                               min_stars=field_min_stars,
                                               context=fctx)
                frec["field"] = _jsonable(field)
                if field is not None:
                    print(f"ASTROM CROSSFILTER FIELD: {fctx}: coherent "
                          f"{field['coherent_mas']:.2f} mas rms per component "
                          f"over {field['n_cells']}/{field['n_cells_in_bbox']} "
                          f"x {field_cell_arcsec:.0f}\" cells from "
                          f"{field['n_pairs']} pairs "
                          f"({100*field['matched_fraction']:.0f}% matched; "
                          f"cell SEM {field['median_sem_mas']:.2f} mas); a "
                          f"linear tie would leave "
                          f"{field['rms_after_affine_mas']:.2f} mas "
                          f"({100*field['affine_absorbed_adjusted']:.0f}% "
                          f"absorbed vs {100*field['affine_absorbed_chance']:.0f}% "
                          f"by chance, |J| {field['gradient_mas_per_arcmin']:.2f} "
                          f"mas/arcmin) -- MEASUREMENT ONLY, not a gate",
                          flush=True)
                npairs = local.get("n_pairs")
                if not local["n_cells"]:
                    # Name the cause the code CHECKED.  n_cells == 0 has three,
                    # and they are not interchangeable: no pair inside the match
                    # radius; pairs found but ALL ambiguous (the crowded-field
                    # case that crashed the brick F187N --refcat run,
                    # 2026-08-03); or pairs binned but every cell below
                    # min_stars.  Only the third is sparsity, and on a dense
                    # field the second is a MATCHING failure after a tie this
                    # run just certified.
                    if npairs == 0:
                        why = ("no matched pair survived the map's radius and "
                               "uniqueness filter -- on a dense field that is a "
                               "matching failure, not sparsity")
                    elif npairs is None:
                        why = "cause not recorded (local_residual_map predates n_pairs)"
                    else:
                        why = (f"{npairs} matched pairs binned, but no cell "
                               f"reached {cell_min_stars} of them")
                    # An EMPTY map is not a clean one.  At GC densities a
                    # `cell_arcsec` cell holds ~1 star against `cell_min_stars`,
                    # so local_residual_map skips every cell and returns
                    # n_cells = 0 -- and reading only `n_flagged` below then
                    # scores that as a pass.  An injection sweep on Brick
                    # geometry never trips this gate at ANY amplitude up to
                    # 30 mas/arcmin for exactly that reason (issue #296).
                    # Report it as UNVERIFIED, not as a failure: the map
                    # measured nothing, which is a coverage fact about the
                    # field, not evidence of a misalignment.
                    unverified.append(
                        f"{fctx}: local {cell_arcsec}\" cell map is EMPTY "
                        f"({why}) -- this filter pair got NO local check at "
                        f"all, and a pass here is silence rather than a "
                        f"verified result")
                elif local["n_flagged"]:
                    # A FLAGGED cell is checked FIRST, whatever the coverage.
                    # Ordering the thin-map test ahead of it turned a detection
                    # into a pass: a map with one or two populated cells, one of
                    # them significantly offset, reported "too little of the
                    # field is checked" and left `passed` True.  Reachable at
                    # the production defaults (2", min 10 stars) on any field
                    # with a couple of compact over-densities -- and it silenced
                    # exactly the detection this gate exists for.  Thin coverage
                    # is a reason to distrust a PASS, never a reason to discard
                    # a FAILURE.
                    worst = max((c for c in local["cells"] if c["flagged"]),
                                key=lambda c: c["off_mas"])
                    failures.append(
                        f"{fctx}: {local['n_flagged']} local {cell_arcsec}\" cell(s) "
                        f"with significant offset > {cell_tol_mas} mas (worst "
                        f"{worst['off_mas']:.1f}±{np.hypot(worst['dra_sem'], worst['ddec_sem']):.1f} "
                        f"mas from {worst['n']} stars at "
                        f"{worst['ra0']:.5f},{worst['dec0']:.5f})")
                    if local["n_cells"] < LOCAL_CELL_MIN_CELLS:
                        # Both are true and both are worth saying: the failure
                        # stands, and the coverage behind it is thin.
                        unverified.append(
                            f"{fctx}: the failure above rests on only "
                            f"{local['n_cells']} populated cell(s) (< "
                            f"{LOCAL_CELL_MIN_CELLS})")
                elif local["n_cells"] < LOCAL_CELL_MIN_CELLS:
                    # Nothing flagged, and too little of the field checked for
                    # the pass to mean anything.
                    unverified.append(
                        f"{fctx}: local {cell_arcsec}\" cell map has only "
                        f"{local['n_cells']} populated cell(s) (< "
                        f"{LOCAL_CELL_MIN_CELLS}) from {npairs} pairs -- too "
                        f"little of the field is checked for a pass to mean "
                        f"anything")
        filters.append(frec)

    passed = _checkpoint_passed(failures, unverified_blocking)
    record = dict(stage="m7-crossfilter", context=context, date=_utcnow_iso(),
                  anchor_filter=anchor_filter,
                  anchor_reference_tie=_jsonable(anchor_tie),
                  filters=filters, failures=failures, passed=passed,
                  unverified=unverified, unverified_blocking=unverified_blocking,
                  all_verified=not unverified,
                  tolerances=dict(crossfilter_tol_mas=tol_mas,
                                  local_cell_tol_mas=cell_tol_mas,
                                  local_cell_size_arcsec=cell_arcsec,
                                  local_cell_min_stars=cell_min_stars,
                                  field_cell_arcsec=field_cell_arcsec,
                                  field_min_stars=field_min_stars))
    if record_dir:
        # Tokened for the same reason the m2 records are (issue #281): brick's
        # 1182 and 2221 m7 runs write into one `astrometry_checkpoints/`, and
        # the untokened name meant 2221's verdict replaced 1182's.  This record
        # is write-only -- nothing reads it back, and the verdict itself is
        # raised in-memory as CrossFilterAstrometryError -- so what the
        # collision costs is the audit trail, not a wrong correction.  Its only
        # other identity field is `context`, which names the target, not the
        # observation.
        _write_record(record_dir, f"checkpoint_m7_crossfilter{obs_token}",
                      record)
    for w in unverified:
        print(f"ASTROM CHECKPOINT [m7-crossfilter] COULD NOT VERIFY: {w}",
              flush=True)
    if failures:
        msg = ("CROSS-FILTER ASTROMETRY FAILURE --\n  " + "\n  ".join(failures))
        if _env_flag("ALLOW_CROSSFILTER_ASTROM_FAIL"):
            print(f"WARNING (override ALLOW_CROSSFILTER_ASTROM_FAIL=1): {msg}",
                  flush=True)
        else:
            raise CrossFilterAstrometryError(msg)
    return record


# ---------------------------------------------------------------------------
# record serialization
# ---------------------------------------------------------------------------

def _jsonable(obj):
    """Strip non-serializable members (SkyCoord, arrays) from a result dict."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in ("cells",):
                out[k] = [_jsonable(c) for c in v]
            elif isinstance(v, dict):
                out[k] = _jsonable(v)
            elif isinstance(v, (str, int, float, bool, type(None))):
                out[k] = v
            elif isinstance(v, (np.integer,)):
                out[k] = int(v)
            elif isinstance(v, (np.floating,)):
                out[k] = float(v)
            elif isinstance(v, (np.bool_,)):
                out[k] = bool(v)
            elif isinstance(v, (list, tuple)):
                out[k] = [_jsonable(x) if isinstance(x, dict) else x
                          for x in v
                          if isinstance(x, (dict, str, int, float, bool, type(None)))]
            # SkyCoord / ndarray members are measurement inputs, not record data
        return out
    return obj


def _jsonable_local(local):
    if local is None:
        return None
    out = _jsonable({k: v for k, v in local.items() if k != "cells"})
    out["cells"] = [_jsonable(c) for c in local["cells"]]
    return out


#: A 6-parameter affine fitted to n cells has 2n observations, so a pure-noise
#: field still "absorbs" 6/(2n) of the variance.  Below this many cells that
#: chance level exceeds 25% and the absorbed fraction stops meaning anything.
CROSSFILTER_FIELD_MIN_CELLS = 12

#: Free parameters in the affine fit (2 translations + 4 linear terms).
_AFFINE_NPAR = 6

#: ``local_residual_map`` reports each cell's error as ``MAD*1.4826/sqrt(n)``,
#: which is the standard error of a MEAN.  The cells hold MEDIANS, whose
#: standard error is ~sqrt(pi/2) larger.  Without this the noise deconvolution
#: under-removes and every significance quoted from it is ~25% optimistic.
_MEDIAN_SEM_FACTOR = 1.2533


def measure_residual_field(coords, anchor_coords, bulk,
                           cell_arcsec=CROSSFILTER_FIELD_CELL_ARCSEC,
                           min_stars=CROSSFILTER_FIELD_MIN_STARS,
                           match_radius_arcsec=0.3,
                           min_cells=CROSSFILTER_FIELD_MIN_CELLS,
                           context=""):
    """The COHERENT, position-dependent part of a catalog-to-catalog residual.

    ``measure_offset`` reduces a whole field to two numbers, and the tie applied
    to the frames is that same rigid translation — so anything that varies
    across the FOV survives the tie by construction.  This measures what
    survives: same-star matched-pair residuals binned into ``cell_arcsec``
    cells (``local_residual_map``), bulk removed.

    **Every amplitude here is PER-COMPONENT** (per axis), including
    ``rms_mas``, ``median_sem_mas``, ``coherent_mas`` and
    ``rms_after_affine_mas``, and ``rms_convention`` records that in the record
    itself.  The only 2-D vector magnitude is ``max_cell_off_mas``, whose name
    says so.  Mixing the two conventions puts a silent factor √2 between
    quantities a reader will divide by each other.

    Keys:

    * ``rms_mas``            — cell-to-cell rms of the field, per component;
    * ``median_sem_mas``     — median per-cell standard error, per component,
      i.e. how much of ``rms_mas`` could be counting noise;
    * ``coherent_mas``       — ``sqrt(rms^2 - <sem^2>)``, the noise-deconvolved
      field amplitude.  This is the number to quote.  The per-cell SEM comes
      from ``local_residual_map``'s ``MAD·1.4826/√n``, which is the SEM of a
      *mean*, scaled here by ``_MEDIAN_SEM_FACTOR`` because the cells hold
      medians.  Without that scaling the deconvolution under-removes and every
      significance quoted from it is ~25% optimistic;
    * ``affine_*``           — a 6-parameter (translation+linear) fit over the
      FOV: the part a linear tie would remove.  ``affine_absorbed_fraction`` is
      raw, ``affine_absorbed_chance`` is what pure noise gives at this
      ``n_cells`` (``6/(2n)``), and ``affine_absorbed_adjusted`` corrects for
      it.  Quote the adjusted one, or the raw one beside the chance level;
    * ``gradient_mas_per_arcmin`` — the Frobenius norm of the fitted 2×2
      Jacobian, per arcmin.  For a field that is a pure ramp along one axis
      this equals that ramp; for a general field it is the quadrature sum of
      all four linear terms, NOT the peak directional gradient;
    * ``n_pairs`` / ``matched_fraction`` / ``match_radius_mas`` — how many
      stars the number rests on.  A low matched fraction means most of one
      list never entered the statistic, and anything displaced beyond the
      match radius is invisible to it by construction;
    * ``n_cells`` / ``n_cells_in_bbox`` / ``n_cells_dropped`` — coverage.  Cells
      below ``min_stars`` are silently skipped by ``local_residual_map``; the
      bounding-box count is the denominator that makes that visible.

    Diagnostic only: nothing here raises, and the caller must not gate on it.
    The point is that a coherent field of a few mas is presently invisible —
    it passes the 5 mas bulk gate (the bulk is ~0), and the 15 mas/2" cell gate
    cannot see it either -- at 2" a dense field leaves ~1.2 stars per cell
    against ``LOCAL_CELL_MIN_STARS = 10``, so that map comes back EMPTY.  The
    emptiness is now reported as UNVERIFIED rather than scored as a pass, but
    the cell gate still measures nothing there, which is why this exists.

    ``bulk`` is the ``measure_offset`` result used to pre-align the two
    catalogs; ``local_residual_map`` refuses to run without a verified small
    global tie, which is the guard that keeps this on the sanctioned side of
    ASTROMETRY RULE #1 (a real global tie already exists, so nearest-partner
    pairing is the right star).

    Returns ``None`` below ``min_cells`` populated cells.
    """
    local = local_residual_map(coords, anchor_coords, bulk,
                               cell_arcsec=cell_arcsec, min_stars=min_stars,
                               match_radius=float(match_radius_arcsec) * u.arcsec,
                               tol_mas=np.inf, context=context)
    cells = [c for c in local["cells"] if c["n"] >= min_stars]
    if len(cells) < min_cells:
        return None

    d = np.array([[c["dra_mas"], c["ddec_mas"]] for c in cells], dtype=float)
    # per-component SEM, to match every other amplitude reported here
    sem = np.array([float(np.hypot(c["dra_sem"], c["ddec_sem"])
                          * _MEDIAN_SEM_FACTOR / np.sqrt(2.0))
                    for c in cells])
    ra0 = np.array([float(c["ra0"]) for c in cells])
    dec0 = np.array([float(c["dec0"]) for c in cells])
    # The FIELD is what is left once the bulk is gone; the bulk is the tie's job.
    d = d - np.median(d, axis=0)
    off = np.hypot(d[:, 0], d[:, 1])

    cosd = float(np.cos(np.radians(dec0.mean())))
    x = (ra0 - ra0.mean()) * cosd * 3600.0
    y = (dec0 - dec0.mean()) * 3600.0
    n = len(cells)
    design = np.zeros((2 * n, _AFFINE_NPAR))
    design[:n, 0] = 1.0; design[:n, 1] = x; design[:n, 2] = y
    design[n:, 3] = 1.0; design[n:, 4] = x; design[n:, 5] = y
    obs = np.concatenate([d[:, 0], d[:, 1]])
    par, *_ = np.linalg.lstsq(design, obs, rcond=None)
    resid = obs - design @ par

    rms = float(np.sqrt((obs ** 2).mean()))          # per component
    rms_after = float(np.sqrt((resid ** 2).mean()))  # per component
    ms = float((sem ** 2).mean())
    raw = (float(1.0 - (resid ** 2).mean() / (obs ** 2).mean())
           if (obs ** 2).mean() else 0.0)
    chance = _AFFINE_NPAR / (2.0 * n)
    dof = (2.0 * n) / (2.0 * n - _AFFINE_NPAR)
    grad = 60.0 * float(np.hypot(np.hypot(par[1], par[2]),
                                 np.hypot(par[4], par[5])))

    ixs = [c["ix"] for c in cells]
    iys = [c["iy"] for c in cells]
    bbox = (max(ixs) - min(ixs) + 1) * (max(iys) - min(iys) + 1)
    npairs = int(sum(c["n"] for c in cells))

    return dict(
        n_cells=n, n_cells_in_bbox=int(bbox), n_cells_dropped=int(bbox - n),
        cell_arcsec=float(cell_arcsec), min_stars=int(min_stars),
        min_cells=int(min_cells),
        median_n_per_cell=float(np.median([c["n"] for c in cells])),
        n_pairs=npairs,
        matched_fraction=(float(npairs) / len(coords)) if len(coords) else float("nan"),
        match_radius_mas=float(match_radius_arcsec) * 1000.0,
        rms_convention="per-component",
        rms_mas=rms, median_sem_mas=float(np.median(sem)),
        coherent_mas=float(np.sqrt(max(rms ** 2 - ms, 0.0))),
        max_cell_off_mas=float(off.max()),
        rms_after_affine_mas=rms_after,
        affine_absorbed_fraction=raw,
        affine_absorbed_chance=float(chance),
        affine_absorbed_adjusted=float(1.0 - (1.0 - raw) * dof),
        gradient_mas_per_arcmin=grad,
    )


def _write_record(record_dir, name, record):
    os.makedirs(record_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(record_dir, f"{name}_{stamp}.json")
    with open(path, "w") as fh:
        json.dump(_jsonable_record(record), fh, indent=2, default=_json_default)
    latest = os.path.join(record_dir, f"{name}_latest.json")
    tmp = latest + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(_jsonable_record(record), fh, indent=2, default=_json_default)
    os.replace(tmp, latest)
    record["record_path"] = path
    return path


def _jsonable_record(record):
    return json.loads(json.dumps(record, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
