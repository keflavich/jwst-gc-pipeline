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
from datetime import datetime, timezone

import numpy as np
from astropy import units as u
from astropy.table import Table

from .visit_consensus import (
    EXPOSURE_CONSENSUS_TOL_MAS, ConsensusBuildError, build_visit_consensus,
    catalog_coords, load_reference_catalog, measure_reference_tie,
    pick_reference_anchor_filter, select_reliable_stars,
)
from .astrometry_offsets import measure_offset, local_residual_map

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

def _module_variants(module):
    """Match semantics of shift_individual_catalog: a detector-level module
    matches its own row or the module-family row."""
    m = str(module)
    if m.endswith("a") or m.endswith("b"):
        m = m + "long"
    return {m, m.strip("1234"), m.replace("long", "")}


def update_offsets_table(offsets_path, corrections, stage, out_path=None,
                         backup=True):
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
    """
    from ..reduction.validate_offsets_table import (
        CollapsedOffsetsTableError, assert_offsets_table_sane)

    tbl = Table.read(offsets_path)
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
        visit = int(str(corr["visit"])[-3:])
        match = (visit_numbers == visit) & (tbl["Filter"] == corr["filtername"])
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
        if corr.get("exposure") is not None and "Exposure" in tbl.colnames:
            match &= tbl["Exposure"] == int(corr["exposure"])
        if corr.get("module") is not None and "Module" in tbl.colnames:
            variants = _module_variants(corr["module"])
            match &= np.array([str(m) in variants for m in tbl["Module"]])
        if match.sum() == 0:
            raise OffsetsTableUpdateError(
                f"correction {corr} matches NO row in {offsets_path} -- refusing "
                f"a partial application (this is how silent curation errors start)")
        cosd = max(np.cos(np.radians(float(corr["dec_deg"]))), 1e-6)
        dra_add = (float(corr["dra_onsky_mas"]) / 1000.0) / cosd
        ddec_add = float(corr["ddec_onsky_mas"]) / 1000.0
        idx = np.where(match)[0]
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
        backup_path = f"{out_path}.pre_{stage}_{stamp}"
        os.replace(out_path, backup_path)
    tbl.write(out_path, overwrite=True)
    return tbl


BULK_EXPOSURE = -1     # sentinel Exposure for the per-visit consensus->reference row
BULK_MODULE = "all"    # sentinel Module for the per-visit bulk row


def lookup_consensus_offset(tbl, visit, exposure, module, filtername):
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

    # per-exposure jitter (exclude the bulk sentinel from the module variants)
    variants = {str(module), str(module).strip("1234"), str(module).replace("long", "")}
    jit = (vf & (tbl["Exposure"] == int(exposure))
           & np.array([str(m) in variants for m in tbl["Module"]]))
    nj = int(jit.sum())
    if nj > 1:
        raise ValueError(
            f"consensus jitter match={nj} for visit={visit} exp={exposure} "
            f"mod={module} filt={filtername}; expected <=1 row")
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
    out_path = out_path or os.path.join(
        basepath, "offsets",
        f"Offsets_JWST_Brick{proposal_id}_consensus.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    now = _utcnow_iso()

    # index any existing rows by key so corrections ADD-or-INSERT (upsert)
    existed = os.path.exists(out_path)
    bykey = {}
    if existed:
        for r in Table.read(out_path):
            key = (str(r["Visit"]), str(r["Filter"]), int(r["Exposure"]),
                   str(r["Module"]))
            bykey[key] = {c: r[c] for c in r.colnames}

    for corr in corrections:
        visit = int(str(corr["visit"])[-3:])
        visit_tok = f"jw0{proposal_id}{field}{visit:03d}"
        cosd = max(np.cos(np.radians(float(corr["dec_deg"]))), 1e-6)
        # A consensus->reference correction is the per-VISIT bulk tie (whole
        # visit onto VIRAC2) -- it carries exposure=None AND module=None.  Store
        # it under the sentinel (BULK_EXPOSURE, BULK_MODULE) row so fix_alignment
        # applies it to EVERY exposure of the visit/filter (lookup_consensus_
        # offset sums bulk + per-exposure jitter).  Writing it to a real
        # exposure/module would either miss most frames or double-shift one.
        is_bulk = corr.get("exposure") is None and corr.get("module") is None
        exposure = BULK_EXPOSURE if is_bulk else int(corr["exposure"])
        module = BULK_MODULE if is_bulk else str(corr["module"])
        dra_add = (float(corr["dra_onsky_mas"]) / 1000.0) / cosd
        ddec_add = float(corr["ddec_onsky_mas"]) / 1000.0
        key = (visit_tok, corr["filtername"], exposure, module)
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
                "Exposure": exposure,
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
    keys = [(str(r["Visit"]), str(r["Filter"]), int(r["Exposure"]), str(r["Module"]))
            for r in rows]
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
        # keep a backup of the pre-merge table (mirrors update_offsets_table)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        os.replace(out_path, f"{out_path}.pre_{stage}_{stamp}")
    tbl.write(out_path, overwrite=True)
    return out_path


# ---------------------------------------------------------------------------
# provenance header stamping (used by fix_alignment at re-apply time)
# ---------------------------------------------------------------------------

def provenance_header_cards(stage, dra_onsky_mas, ddec_onsky_mas, method,
                            references, table_name):
    """FITS header cards recording WHY the current RAOFFSET/DEOFFSET are what
    they are.  ``fix_alignment`` stamps these when it (re-)applies a corrected
    offsets table — the header of every aligned frame then carries the full
    provenance of its astrometric fix."""
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
    """
    if not record_dir:
        return None
    path = os.path.join(record_dir, f"checkpoint_m2_{filtername}_latest.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        rec = json.load(fh)
    for v in rec.get("visits", []):
        if str(v.get("visit")) != str(visit):
            continue
        rt = v.get("reference_tie") or {}
        dra, ddec = rt.get("dra_mas"), rt.get("ddec_mas")
        if dra is not None and ddec is not None \
                and np.isfinite(dra) and np.isfinite(ddec):
            return float(dra), float(ddec)
        vf = rt.get("vs_full") or {}   # legacy record without the reported bulk
        if "dra" in vf and "ddec" in vf:
            return float(vf["dra"]), float(vf["ddec"])
    return None


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


def run_visit_checkpoint(exposure_tables, stage, refcat=None, filtername=None,
                         basepath=None, record_dir=None, context="",
                         consensus_kwargs=None):
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
    corrections = []
    failures = []      # MEASURED shifts -- blocking at a late stage
    unverified = []    # could-not-verify -- loud warnings, audited by the gate
    for (visit, filt), tables in sorted(_group_by_visit_filter(exposure_tables).items()):
        vctx = f"{context} {filt} visit {visit} [{stage}]"
        try:
            cons = build_visit_consensus(tables, context=vctx, **consensus_kwargs)
        except ConsensusBuildError as ex:
            visits.append(dict(visit=visit, filtername=filt, consensus=None,
                               error=str(ex)))
            unverified.append(f"{vctx}: consensus build failed: {ex}")
            continue

        # ---- per-exposure vs consensus ------------------------------------
        # At a frozen stage the per-exposure gate is a MOVEMENT check vs the m2
        # baseline (see _m2_exposure_baseline), NOT an absolute vs-consensus
        # magnitude check -- the latter re-trips on intrinsic per-exposure
        # scatter that m2 already tolerated.
        exp_baseline = ({} if correcting
                        else _m2_exposure_baseline(record_dir, filt, visit))
        exp_records = []
        for exp in cons["exposures"]:
            res = exp["vs_consensus"]
            rec = dict(key=list(exp["key"]), n_reliable=exp["n_reliable"],
                       raoffset_meta=exp["raoffset_meta"],
                       deoffset_meta=exp["deoffset_meta"],
                       component=exp.get("component", 0),
                       internal_tie=exp.get("internal_tie", True),
                       unverified=exp.get("unverified", False),
                       misaligned=exp["misaligned"])
            if res is not None:
                rec.update({k: res.get(k) for k in
                            ("dra", "ddec", "off", "npairs", "contrast", "ok",
                             "swept", "window_arcsec", "dra_err", "ddec_err",
                             "n_peak")})
            exp_records.append(rec)
            if exp.get("unverified"):
                unverified.append(
                    f"{vctx}: exposure {exp['key']} has no measurable tie to the "
                    f"visit consensus (isolated footprint / too few overlap "
                    f"stars) -- internally UNVERIFIED; the reference tie is its "
                    f"only check")
            if exp["misaligned"]:
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
                ref_mag=refcat.get("mag"), context=vctx)
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
                        base = _m2_reference_tie_baseline(record_dir, filt, visit)
                        if base is not None:
                            delta = float(np.hypot(ref_tie["dra_mas"] - base[0],
                                                   ref_tie["ddec_mas"] - base[1]))
                            if delta > STAGE_STABILITY_TOL_MAS:
                                failures.append(
                                    f"{vctx}: consensus->reference MOVED "
                                    f"{delta:.2f} mas since the m2 freeze "
                                    f"(m2=({base[0]:+.2f},{base[1]:+.2f}), now="
                                    f"({ref_tie['dra_mas']:+.2f},"
                                    f"{ref_tie['ddec_mas']:+.2f}) mas)")
                            else:
                                print(f"ASTROM CHECKPOINT [{stage}] STABLE: {vctx} "
                                      f"tie unchanged since m2 (delta "
                                      f"{delta:.2f} mas <= "
                                      f"{STAGE_STABILITY_TOL_MAS})", flush=True)
                        else:
                            failures.append(
                                f"{vctx}: consensus->reference offset {off:.2f} mas at a "
                                f"LATE stage (solution was supposed to be frozen; "
                                f"no m2 baseline record found)")
                else:
                    # apply_ok is False only for a genuinely bad VIRAC tie now:
                    # no coherent dense peak, per-tile not clean, or a GROSS
                    # sparse split (spurious peak). A fine ~5-10 mas Gaia-sparse
                    # split no longer lands here (gc-gaia-frame-not-catalog).
                    unverified.append(
                        f"{vctx}: consensus->reference offset {off:.2f} mas but the "
                        f"VIRAC tie is not trustworthy "
                        f"(cross-ref sep={ref_tie['cross_reference'].get('sep_mas'):.1f} mas, "
                        f"gross_ok={ref_tie.get('cross_reference_gross_ok')}, "
                        f"per-tile clean={ref_tie['per_tile'].get('clean')}, "
                        f"swept={ref_tie.get('swept')}) -- NOT applying; investigate")

        visits.append(dict(
            visit=visit, filtername=filt,
            consensus=dict(
                n_stars=int(len(cons["coords"])),
                anchor=list(cons["anchor_key"]),
                median_scatter_mas=float(np.median(cons["scatter_mas"]))
                if len(cons["scatter_mas"]) else float("nan"),
                consensus_ok=cons["consensus_ok"],
                skipped=[list(k) for k in cons["skipped"]]),
            exposures=exp_records,
            reference_tie=_jsonable(ref_tie)))

    passed = not failures
    record = dict(stage=stage, filtername=filtername, context=context,
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
        elif not anchor_tie["per_tile"].get("clean"):
            failures.append(f"anchor {anchor_filter}: per-tile reference map not clean")

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
