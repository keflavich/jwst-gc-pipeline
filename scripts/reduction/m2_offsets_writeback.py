#!/usr/bin/env python
"""Write m2's module-common per-exposure offsets into a module-locked offsets table.

Reads the field's m2 checkpoint record for each filter and, for every row of the
module-locked offsets table, writes the mean over the row's detectors of m2's
recorded vs-consensus offset.  The rules -- what is written, what is skipped and
why one pass converges -- are in ``jwst_gc_pipeline.photometry.m2_writeback``.

Dry run by default: prints every row's verdict and writes nothing.  ``--write``
applies the plan through ``update_offsets_table`` (backup
``<table>.pre_m2-writeback_<timestamp>``, rows stamped
``prov_stage='m2-writeback'``).

Only the ``locked`` channel is accepted.  A consensus-channel table keys its rows
per detector and creates rows on write, which is a different writer
(``seed_offsets_table_from_consensus``).

## Usage

    python m2_offsets_writeback.py --basepath /orange/adamginsburg/jwst/sgrb2 \\
        --proposal 5365 --field 001 --filter F212N --filter F480M          # dry run
    python m2_offsets_writeback.py ... --write

## After --write

The written rows reach the frames only through a regeneration: from ``_cal``
(destreak overwrite -> fix_alignment -> Image3), or a fix_alignment rerun with
``RAOFFSET_DISAGREE_TOL_ARCSEC`` at or below the value printed (its 0.05"
default treats a mas-scale change as current).  The script prints the
``submit_reduction.sbatch`` command for the filters with written rows, with
that tolerance exported.  Then rerun m2: the written rows should read ~0 and a
second pass should write nothing.
"""
import argparse
import json
import os
import sys

import numpy as np
from astropy.table import Table

from jwst_gc_pipeline.photometry.m2_writeback import (
    DEFAULT_MIN_MAS, M2WritebackError, apply_writeback, writeback_plan)


def _dec_deg(record):
    """Declination for the cos(dec) conversion: the record's own corrections
    carry the value m2 used; without any, the consensus catalog's median."""
    decs = [float(c["dec_deg"]) for c in record.get("corrections") or []
            if c.get("dec_deg") is not None]
    if decs:
        return float(np.median(decs))
    path = record.get("consensus_catalog")
    if not path or not os.path.exists(path):
        raise M2WritebackError(
            f"no correction carries dec_deg and the consensus catalog "
            f"{path!r} is not readable; cannot convert dRA to a coordinate offset")
    return float(np.median(np.asarray(Table.read(path)["DEC"], dtype=float)))


def _regen_command(basepath, proposal, field, filters, tol_arcsec):
    """The reduction that carries written rows into the frames: one array task
    per filter, this checkout pinned through PIPE_ROOT, and the stale-check
    tolerance lowered so frames that are not re-destreaked still realign."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    target = os.path.basename(os.path.normpath(basepath))
    return (f"sbatch --array=0-{len(filters) - 1} --qos=astronomy-dept-b "
            f"--job-name={target}{proposal}-o{field}-reduce-m2wb \\\n"
            f"    --export=ALL,PROPOSAL={proposal},FIELD={field},"
            f"FILTERS=\"{' '.join(filters)}\",PIPE_ROOT={root},"
            f"RAOFFSET_DISAGREE_TOL_ARCSEC={tol_arcsec:.2e} \\\n"
            f"    {os.path.join(here, 'submit_reduction.sbatch')}")


def _report(filt, record_path, plan, verbose):
    counts = plan.counts()
    print(f"{filt}: {os.path.basename(record_path)} -> "
          + ", ".join(f"{k} {v}" for k, v in counts.items() if v))
    rows = plan.written()
    if rows:
        vec = np.array([(r.dra_mas, r.ddec_mas) for r in rows])
        mag = np.hypot(*vec.T)
        print(f"  writing {len(rows)} row(s): |mean| median {np.median(mag):.2f}, "
              f"max {mag.max():.2f} mas; average written "
              f"({vec[:, 0].mean():+.2f},{vec[:, 1].mean():+.2f}) mas")
    for r in plan.rows:
        if r.status in ("write", "below_min") and not verbose:
            continue
        print(f"  {r.status:13s} {r.visit} exp {r.exposure:3d} {r.family:9s} "
              f"vg {r.vgroup} ({r.dra_mas:+.2f},{r.ddec_mas:+.2f}) mas"
              + (f"  -- {r.reason}" if r.reason else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--basepath", required=True,
                    help="field root holding offsets/ and astrometry_checkpoints/")
    ap.add_argument("--proposal", required=True)
    ap.add_argument("--field", required=True, help="observation, e.g. 001")
    ap.add_argument("--filter", dest="filters", action="append", required=True)
    ap.add_argument("--min-mas", type=float, default=DEFAULT_MIN_MAS,
                    help=f"smallest row mean written (default {DEFAULT_MIN_MAS})")
    ap.add_argument("--write", action="store_true",
                    help="apply the plan (default: dry run)")
    ap.add_argument("--verbose", action="store_true",
                    help="list written and below-minimum rows too")
    args = ap.parse_args(argv)

    from jwst_gc_pipeline.photometry.astrometry_checkpoint import _m2_record_path
    from jwst_gc_pipeline.photometry.consensus_catalog import consensus_obs_token
    from jwst_gc_pipeline.reduction.alignment_config import (
        offsets_channel, offsets_table_path)

    channel = offsets_channel(args.proposal, args.field)
    if channel != "locked":
        raise SystemExit(f"proposal {args.proposal} observation {args.field} is "
                         f"on the {channel!r} channel; only a module-locked "
                         f"table is written back")
    table = offsets_table_path(args.basepath, str(args.proposal), str(args.field))
    if not os.path.exists(table):
        raise SystemExit(f"no offsets table at {table}")
    record_dir = os.path.join(args.basepath, "astrometry_checkpoints")
    token = consensus_obs_token(args.proposal, args.field)

    tols, written = [], []
    for filt in args.filters:
        filt = filt.upper()
        record_path = _m2_record_path(record_dir, filt, token)
        if record_path is None:
            raise SystemExit(f"no m2 record for {filt}{token} in {record_dir}")
        with open(record_path) as fh:
            record = json.load(fh)
        plan = writeback_plan(record, Table.read(table), args.proposal,
                              args.field, _dec_deg(record),
                              min_mas=args.min_mas, record_path=record_path,
                              filtername=filt)
        _report(filt, record_path, plan, args.verbose)
        if args.write and plan.corrections:
            apply_writeback(table, plan)
            print(f"  WROTE {len(plan.written())} row(s) to {table}")
        if plan.realign_tol_arcsec() is not None:
            tols.append(plan.realign_tol_arcsec())
            written.append(filt)

    if not tols:
        print("nothing to write")
        return 0
    verb = "written" if args.write else "would be written"
    print(f"\nRows {verb}.  They reach the frames only on regeneration: from "
          f"_cal, or a fix_alignment rerun with "
          f"RAOFFSET_DISAGREE_TOL_ARCSEC={min(tols):.2e} (the 0.05\" default "
          f"treats these changes as current).  Then rerun m2.  Reduction:\n")
    print(_regen_command(args.basepath, args.proposal, args.field, written,
                         min(tols)))
    if not args.write:
        print("Dry run: nothing was written (--write to apply).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
