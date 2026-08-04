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
  star is not a measurement).

Every checkpoint writes a machine-readable record under
``{basepath}/astrometry_checkpoints/`` so the release gate can audit the full
ladder.  Nothing here ever edits ``_cal.fits`` or pokes a mosaic GWCS.
"""
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
    visit_numbers = np.array([int(str(v)[-3:]) for v in tbl["Visit"]])
    for corr in corrections:
        if _is_bulk_correction(corr):
            continue        # see _is_bulk_correction: broad BY DESIGN
        idx = _match_rows(corr, tbl, visit_numbers)
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
    corrections = list(corrections)
    # Magnitude ceiling BEFORE the median.  Pooling cannot inflate a correction
    # past the ceiling (median <= max), so the risk runs the other way: a
    # detector whose measurement blew up is averaged out of existence and the
    # operator never learns the measurement failed.  Check the MEMBERS.
    _assert_correction_magnitudes(corrections, offsets_path)
    visit_numbers = np.array([int(str(v)[-3:]) for v in tbl["Visit"]])
    groups = {}
    order = []
    bulk = []
    for corr in corrections:
        if _is_bulk_correction(corr):
            bulk.append(corr)
            continue
        key = tuple(sorted(int(i) for i in _match_rows(corr, tbl, visit_numbers)))
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


def _match_rows(corr, tbl, visit_numbers):
    """Row indices of ``tbl`` a single correction would be ADDED to.

    Factored out of ``update_offsets_table`` so the granularity guard and the
    pooling helper narrow EXACTLY the way the apply loop does -- a guard that
    re-implements the narrowing is a guard that drifts away from what it guards.
    Unlike the apply loop this never raises; callers decide what an empty or
    over-full match means.
    """
    visit = int(str(corr["visit"])[-3:])
    match = (visit_numbers == visit) & (tbl["Filter"] == corr["filtername"])
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
            CollapsedOffsetsTableError, assert_offsets_table_sane)

        # materialise first: the checks below iterate `corrections` before the apply
        # loop does, and a generator would be consumed by them -- leaving the update
        # a silent no-op that still writes a table and a backup.
        corrections = list(corrections)

        # magnitude ceiling FIRST: fail before touching the table at all, so a
        # spurious measurement cannot be half-applied or leave a backup behind.
        _assert_correction_magnitudes(corrections, offsets_path)

        tbl = Table.read(offsets_path)
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
        _assert_module_granularity(corrections, tbl, offsets_path)
        _assert_vgroup_granularity(corrections, tbl, offsets_path)
        _assert_one_correction_per_row(corrections, tbl, offsets_path)
        # both column conventions exist: 'dra'/'ddec' (generate_offsets_table) and
        # 'dra (arcsec)'/'ddec (arcsec)' (the VIRAC2locked tables fix_alignment reads)
        dra_col = "dra (arcsec)" if "dra (arcsec)" in tbl.colnames else "dra"
        ddec_col = "ddec (arcsec)" if "ddec (arcsec)" in tbl.colnames else "ddec"
        if dra_col not in tbl.colnames or ddec_col not in tbl.colnames:
            raise OffsetsTableUpdateError(
                f"{offsets_path} has no dra/ddec columns ({tbl.colnames})")
        for col, fill in (("prov_stage", ""), ("prov_date", ""), ("prov_source", "")):
            if col not in tbl.colnames:
                tbl[col] = np.full(len(tbl), fill, dtype="U64")
        for col in ("prov_dra_added_mas", "prov_ddec_added_mas"):
            if col not in tbl.colnames:
                tbl[col] = np.zeros(len(tbl))

        visit_numbers = np.array([int(str(v)[-3:]) for v in tbl["Visit"]])
        now = _utcnow_iso()
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
            idx = _match_rows(corr, tbl, visit_numbers)
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
            tbl[dra_col][idx] = np.asarray(tbl[dra_col][idx], dtype=float) + dra_add
            tbl[ddec_col][idx] = np.asarray(tbl[ddec_col][idx], dtype=float) + ddec_add
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
        except CollapsedOffsetsTableError as ex:
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


def _m2_reference_tie_baseline(record_dir, filtername, visit):
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
    if not record_dir:
        return None, False
    path = os.path.join(record_dir, f"checkpoint_m2_{filtername}_latest.json")
    if not os.path.exists(path):
        return None, False
    with open(path) as fh:
        rec = json.load(fh)
    for v in rec.get("visits", []):
        if str(v.get("visit")) != str(visit):
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


def _m2_exposure_baseline(record_dir, filtername, visit):
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
    if not record_dir:
        return out
    path = os.path.join(record_dir, f"checkpoint_m2_{filtername}_latest.json")
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        rec = json.load(fh)
    for v in rec.get("visits", []):
        if str(v.get("visit")) != str(visit):
            continue
        for e in v.get("exposures", []) or []:
            key = tuple(e.get("key", []) or [])
            dra, ddec = e.get("dra"), e.get("ddec")
            if key and dra is not None and ddec is not None \
                    and np.isfinite(dra) and np.isfinite(ddec):
                out[key] = (float(dra), float(ddec))
    return out


def _m2_skipped_exposures(record_dir, filtername, visit):
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
    if not record_dir:
        return set()
    path = os.path.join(record_dir, f"checkpoint_m2_{filtername}_latest.json")
    if not os.path.exists(path):
        return set()
    with open(path) as fh:
        rec = json.load(fh)
    out = set()
    for v in rec.get("visits", []):
        if str(v.get("visit")) != str(visit):
            continue
        cons = v.get("consensus") or {}
        for key in cons.get("skipped", []) or []:
            out.add(tuple(key))
    return out


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
    for (visit, filt), tables in sorted(_group_by_visit_filter(exposure_tables).items()):
        vctx = f"{context} {filt} visit {visit} [{stage}]"
        try:
            cons = build_visit_consensus(tables, context=vctx, **consensus_kwargs)
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
                        else _m2_exposure_baseline(record_dir, filt, visit))
        # An exposure m2 deliberately skipped has no baseline BY CONSTRUCTION;
        # that absence is not evidence the frozen solution moved.
        m2_skipped = (set() if correcting
                      else _m2_skipped_exposures(record_dir, filt, visit))
        # issue #158 backstop: an ALIAS reads antisymmetric across the modules of
        # an exposure, where real jitter is common-mode.  Never emit corrections
        # from an antisymmetric set -- they are the footprint geometry, not a
        # misalignment (and they are above the appliable ceiling anyway, so this
        # costs no capability; it replaces an opaque stop with a diagnosis).
        antisym = detect_module_antisymmetry(cons["exposures"])
        if antisym["detected"]:
            ex = antisym["examples"][0]
            unverified.append(
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
        antisym_keys = antisym["keys"]
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
            if exp["misaligned"] and tuple(exp["key"]) in antisym_keys:
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
                if correcting:
                    dec_mid = float(np.median(cons["coords"].dec.deg))
                    corrections.append(dict(
                        visit=exp["key"][0], exposure=exp["key"][1],
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
                        failures.append(
                            msg + " [no m2 per-exposure baseline: frozen-stage "
                            "exposure absent from the m2 record]")

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
                            visit=visit, exposure=None, module=None,
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
                            record_dir, filt, visit)
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
                    unverified.append(
                        f"{vctx}: consensus->reference offset {off:.2f} mas but the "
                        f"tie is not trustworthy "
                        f"(cross-ref sep={ref_tie['cross_reference'].get('sep_mas'):.1f} mas, "
                        f"gross_ok={ref_tie.get('cross_reference_gross_ok')}, "
                        f"{gate}, "
                        f"swept={ref_tie.get('swept')}) -- NOT applying; investigate")

        consensus_by_visit[visit] = cons
        visits.append(dict(
            visit=visit, filtername=filt,
            consensus=dict(
                n_stars=int(len(cons["coords"])),
                anchor=list(cons["anchor_key"]),
                median_scatter_mas=float(np.median(cons["scatter_mas"]))
                if len(cons["scatter_mas"]) else float("nan"),
                consensus_ok=cons["consensus_ok"],
                skipped=[list(k) for k in cons["skipped"]]),
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

    passed = not failures
    record = dict(stage=stage, filtername=filtername, context=context,
                  consensus_catalog=consensus_catalog_path,
                  consensus_catalog_error=consensus_catalog_error,
                  date=_utcnow_iso(), correcting=correcting, visits=visits,
                  corrections=corrections, failures=failures,
                  unverified=unverified, passed=passed,
                  all_verified=not unverified,
                  tolerances=dict(
                      exposure_consensus_tol_mas=EXPOSURE_CONSENSUS_TOL_MAS,
                      reference_apply_min_mas=REFERENCE_APPLY_MIN_MAS,
                      stage_stability_tol_mas=STAGE_STABILITY_TOL_MAS))
    if record_dir:
        _write_record(record_dir, f"checkpoint_{stage}_{filtername or 'all'}", record)

    for w in unverified:
        print(f"ASTROM CHECKPOINT [{stage}] COULD NOT VERIFY: {w}", flush=True)
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
                               cell_min_stars=LOCAL_CELL_MIN_STARS):
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
        return dict(passed=True, skipped="single filter", filters=[])
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
                    bulk=_jsonable(bulk), local=None)
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
            if not bulk.get("swept") and bulk["off"] < 100.0:
                local = local_residual_map(
                    coords, anchor_coords, bulk, cell_arcsec=cell_arcsec,
                    min_stars=cell_min_stars, tol_mas=cell_tol_mas,
                    context=fctx)
                frec["local"] = _jsonable_local(local)
                if local["n_flagged"]:
                    worst = max((c for c in local["cells"] if c["flagged"]),
                                key=lambda c: c["off_mas"])
                    failures.append(
                        f"{fctx}: {local['n_flagged']} local {cell_arcsec}\" cell(s) "
                        f"with significant offset > {cell_tol_mas} mas (worst "
                        f"{worst['off_mas']:.1f}±{np.hypot(worst['dra_sem'], worst['ddec_sem']):.1f} "
                        f"mas from {worst['n']} stars at "
                        f"{worst['ra0']:.5f},{worst['dec0']:.5f})")
        filters.append(frec)

    passed = not failures
    record = dict(stage="m7-crossfilter", context=context, date=_utcnow_iso(),
                  anchor_filter=anchor_filter,
                  anchor_reference_tie=_jsonable(anchor_tie),
                  filters=filters, failures=failures, passed=passed,
                  tolerances=dict(crossfilter_tol_mas=tol_mas,
                                  local_cell_tol_mas=cell_tol_mas,
                                  local_cell_size_arcsec=cell_arcsec,
                                  local_cell_min_stars=cell_min_stars))
    if record_dir:
        _write_record(record_dir, "checkpoint_m7_crossfilter", record)
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
