#!/usr/bin/env python
"""Run an astrometry checkpoint from the command line.

Two modes:

* **visit** (default): per-(visit, filter) consensus checkpoint over per-frame
  catalogs — build the visit consensus, re-measure every exposure against it,
  tie the consensus to the reference with the multi-check ladder, and (at the
  m2 stage, with ``--apply``) correct the offsets table + stale-tag the im0
  ``_i2d`` mosaics.

* **crossfilter**: the m7 cross-band agreement check over vetted merged
  catalogs (anchor filter nearest VIRAC2 Ks; <5 mas bulk per filter; no
  significant 2" cell above 15 mas).

Examples
--------
Visit checkpoint (measure only)::

    python run_astrometry_checkpoint.py --stage m2 \
        --catalog-glob '/orange/adamginsburg/jwst/brick/F212N/f212n_*_visit*_exp*_m2_daophot_basic.fits' \
        --filter F212N \
        --refcat /orange/adamginsburg/jwst/brick/catalogs/gaia_virac2_refcat_epoch2022.70.fits \
        --basepath /orange/adamginsburg/jwst/brick

Apply corrections (m2 only)::

    ... --apply --offsets-table /orange/adamginsburg/jwst/brick/offsets/Offsets_JWST_Brick2221_VIRAC2locked.csv --mark-stale

Cross-filter checkpoint::

    python run_astrometry_checkpoint.py --crossfilter \
        --catalog F212N=/path/f212n_..._vetted.fits --catalog F405N=/path/f405n_..._vetted.fits \
        --refcat ... --basepath ...
"""
import argparse
import glob
import json
import sys

from astropy.table import Table

from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    CORRECTION_STAGES, find_i2d_for_filter, mark_i2d_stale,
    run_crossfilter_checkpoint, run_visit_checkpoint,
    seed_offsets_table_from_consensus, update_offsets_table,
)
from jwst_gc_pipeline.photometry.consensus_catalog import consensus_obs_token
from jwst_gc_pipeline.photometry.visit_consensus import load_reference_catalog


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="m2",
                   help="merge stage token (m2 = the m12 merge; m3..m6 = "
                        "stability checks; corrections only at "
                        f"{CORRECTION_STAGES})")
    p.add_argument("--filter", dest="filtername", default=None)
    p.add_argument("--catalog-glob", action="append", default=[],
                   help="glob(s) for per-frame catalogs (visit mode)")
    p.add_argument("--catalog", action="append", default=[],
                   help="FILTER=path vetted merged catalog (crossfilter mode)")
    p.add_argument("--crossfilter", action="store_true")
    p.add_argument("--refcat", default=None,
                   help="gaia+virac2 seed refcat FITS (build_gaia_virac2_refcat_byquery)")
    p.add_argument("--basepath", default=None)
    p.add_argument("--record-dir", default=None)
    p.add_argument("--proposal-id", default=None,
                   help="proposal id, for the per-observation filename token of "
                        "the m2 per-filter consensus catalog (see obs_token)")
    p.add_argument("--obsid", default=None,
                   help="observation id, ditto -- and with --mark-stale it "
                        "SCOPES the stale-tag to this observation's mosaics "
                        "(one <FILTER>/pipeline directory holds every "
                        "observation's; unscoped, one observation's correction "
                        "quarantines the others' good mosaics).  Padding does "
                        "not matter (3 == 003), but an obsid with no product "
                        "in the directory RAISES rather than tagging zero")
    p.add_argument("--offsets-table", default=None)
    p.add_argument("--apply", action="store_true",
                   help="apply implied corrections to --offsets-table "
                        "(correction stages only).  UPDATES existing rows: a "
                        "correction matching no row hard-fails, by design.  "
                        "For a filter the table has never carried, use --seed")
    p.add_argument("--seed", action="store_true",
                   help="with --apply: UPSERT instead of update -- create the "
                        "table if absent, add to a matching row, INSERT a row "
                        "for an exposure not previously flagged.  This is the "
                        "bootstrap path (seed_offsets_table_from_consensus), "
                        "the one the in-pipeline m2 checkpoint uses.  Needs "
                        "--proposal-id and --obsid")
    p.add_argument("--pool", action="store_true",
                   help="Pool per-detector corrections to the granularity of the offsets table before applying them (module-family rows cannot express a per-detector shift, and un-pooled corrections are SUMMED onto the shared row).  This is what the one-correction-per-row refusal asks for.  Not compatible with --seed: the seeder writes rows at the granularity of the corrections it is handed, so there is nothing to pool onto yet.")
    p.add_argument("--mark-stale", action="store_true",
                   help="with --apply: stale-tag the filter's im0 _i2d mosaics "
                        "(renamed *_im0_badastrom.fits)")
    p.add_argument("--brightness", default=None, metavar="CATALOG",
                   help="position-vs-brightness systematics check: measure the "
                        "matched-pair residual vs the reference as a function "
                        "of source magnitude (needs --refcat). A pointing "
                        "reference must show NO trend (saturation-core bias / "
                        "wing-fit substitution class). Exit 1 on a flagged bin "
                        "or a significant slope.")
    p.add_argument("--brightness-tol-mas", type=float, default=5.0)
    args = p.parse_args(argv)

    if args.seed and args.pool:
        # --pool exists because "module-family rows cannot express a
        # per-detector shift, and un-pooled corrections are SUMMED onto the
        # shared row".  `seed_offsets_table_from_consensus` takes no `pool`
        # argument, so --seed --pool would drop the pooling with no warning and
        # write per-detector corrections into rows keyed on
        # (Visit, Filter, Exposure, Module, Vgroup) -- several landing on one
        # row and summing, the exact failure --pool exists to prevent.  The
        # pair is incoherent whatever the run later discovers, so it is
        # rejected here rather than after a catalog load.
        p.error("--seed cannot be combined with --pool: the seeder writes rows "
                "at the granularity of the corrections it is handed, so there "
                "is nothing yet to pool onto.  Seed first, then re-run with "
                "--pool to correct the rows it created.")

    refcat = load_reference_catalog(args.refcat) if args.refcat else None

    if args.brightness:
        if refcat is None:
            p.error("--brightness needs --refcat")
        from astropy.coordinates import SkyCoord
        import numpy as np
        from jwst_gc_pipeline.photometry.astrometry_offsets import (
            measure_offset, residual_vs_magnitude)
        tbl = Table.read(args.brightness)
        colname = "skycoord" if "skycoord" in tbl.colnames else "skycoord_centroid"
        coords = SkyCoord(tbl[colname]).icrs
        fluxcol = next((c for c in ("flux", "flux_fit") if c in tbl.colnames), None)
        if fluxcol is None:
            p.error(f"no flux/flux_fit column in {args.brightness}")
        with np.errstate(divide="ignore", invalid="ignore"):
            mag = -2.5 * np.log10(np.where(
                np.asarray(tbl[fluxcol], float) > 0,
                np.asarray(tbl[fluxcol], float), np.nan))
        g = measure_offset(coords, refcat["all"], context="brightness/global")
        if g is None or not g.get("ok"):
            print(json.dumps(dict(passed=False,
                                  error="no verified global tie to the "
                                        "reference; fix the bulk registration "
                                        "first", global_tie=g)))
            return 1
        r = residual_vs_magnitude(coords, refcat["all"], mag, g,
                                  tol_mas=args.brightness_tol_mas,
                                  context="brightness")
        print(json.dumps(dict(
            passed=r["clean"], n_bins=r["n_bins"], n_flagged=r["n_flagged"],
            slope_dra_mas_per_mag=r["slope_dra_mas_per_mag"],
            slope_ddec_mas_per_mag=r["slope_ddec_mas_per_mag"],
            slope_significant=r["slope_significant"],
            worst_off_mas=r["worst_off_mas"],
            bins=[dict(mag=b["mag_mid"], n=b["n"], dra=round(b["dra_mas"], 2),
                       ddec=round(b["ddec_mas"], 2), off=round(b["off_mas"], 2),
                       flagged=b["flagged"]) for b in r["bins"]]), indent=2))
        return 0 if r["clean"] else 1

    if args.crossfilter:
        catalogs = {}
        for spec in args.catalog:
            filt, _, path = spec.partition("=")
            if not path:
                p.error(f"--catalog needs FILTER=path, got {spec!r}")
            catalogs[filt] = Table.read(path)
        record = run_crossfilter_checkpoint(
            catalogs, refcat=refcat, basepath=args.basepath,
            record_dir=args.record_dir, context="cli",
            obs_token=consensus_obs_token(args.proposal_id, args.obsid))
        print(json.dumps(dict(passed=record["passed"],
                              anchor=record.get("anchor_filter"),
                              failures=record.get("failures", [])), indent=2))
        return 0 if record["passed"] else 1

    paths = []
    for pat in args.catalog_glob:
        paths.extend(glob.glob(pat))
    if not paths:
        p.error("no per-frame catalogs matched --catalog-glob")
    tables = [Table.read(fn) for fn in sorted(set(paths))]
    print(f"loaded {len(tables)} per-frame catalogs", flush=True)

    record = run_visit_checkpoint(
        tables, args.stage, refcat=refcat, filtername=args.filtername,
        basepath=args.basepath, record_dir=args.record_dir, context="cli",
        obs_token=consensus_obs_token(args.proposal_id, args.obsid))

    corrections = record["corrections"]
    print(json.dumps(dict(passed=record["passed"],
                          n_corrections=len(corrections),
                          failures=record.get("failures", [])), indent=2))
    if corrections and args.apply:
        if args.stage not in CORRECTION_STAGES:
            print(f"REFUSING --apply at stage {args.stage}: corrections are only "
                  f"sanctioned at {CORRECTION_STAGES}", file=sys.stderr)
            return 1
        if not args.offsets_table:
            p.error("--apply needs --offsets-table")
        if args.seed:
            # UPSERT.  `update_offsets_table` can only EDIT existing rows, so a
            # field whose table has never carried this filter cannot be
            # bootstrapped with it -- the first iteration has nowhere to record
            # the fix and every correction hard-fails "matches NO row".  That
            # left the CLI unable to seed at all: the only seeding path was a
            # full cataloging m12 pass with ASTROM_CHECKPOINT_APPLY=1, which
            # seeds ONE filter and then raises to force regeneration, so a
            # 14-filter field needed 14 cataloging cycles (crowded_l3).
            if not (args.proposal_id and args.obsid):
                p.error("--seed needs --proposal-id and --obsid")
            if not args.basepath:
                p.error("--seed needs --basepath")
            out = seed_offsets_table_from_consensus(
                args.basepath, args.proposal_id, args.obsid, corrections,
                stage=args.stage, out_path=args.offsets_table)
            print(f"offsets table SEEDED (upsert): {out} "
                  f"({len(corrections)} corrections; backup kept)")
        else:
            update_offsets_table(args.offsets_table, corrections, args.stage,
                                 pool=args.pool)
            print(f"offsets table corrected: {args.offsets_table} "
                  f"({len(corrections)} corrections; backup kept)")
        if args.mark_stale and args.basepath and args.filtername:
            # --obsid scopes the stale-tag: one directory holds every
            # observation's mosaics, and unscoped this renames the other
            # observations' good ones too.  Without --obsid the lookup is
            # unscoped, as before.
            i2ds = find_i2d_for_filter(args.basepath, args.filtername,
                                       observation=args.obsid)
            renames = mark_i2d_stale(
                i2ds, reason=f"{args.stage} checkpoint corrected the offsets "
                             f"table; im0 mosaics are stale",
                record_dir=args.record_dir)
            for old, new in renames:
                print(f"stale-tagged: {old} -> {new}")
            print("REGENERATE the affected frames from _cal (destreak -> "
                  "fix_alignment with the corrected table -> Image3); never "
                  "re-apply on top of the stale shift.")
    elif corrections:
        print("corrections implied but NOT applied (pass --apply)")
    return 0 if record["passed"] or corrections else 1


if __name__ == "__main__":
    sys.exit(main())
