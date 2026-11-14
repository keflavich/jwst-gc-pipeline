#!/usr/bin/env python
"""Move superseded pre-obs-token checkpoint records out of the ``_latest`` glob.

Checkpoint records were first written without an observation token and then with
one:

    checkpoint_m2_F444W_latest.json         (pre-token)
    checkpoint_m2_F444W_o001_latest.json    (tokened)

Both end in ``_latest.json``, so any reader that globs the directory sees the
superseded record beside the current one (#827).  The release gate and
``astrometry_checkpoint.current_records`` already drop the untokened record where
a tokened sibling covers the same ``(stage, filter, module)``; this script takes
the file out of the directory listing so an ad-hoc ``glob`` stops finding it.

## Safety

* Dry run by default.  ``--execute`` acts.
* Moves with ``os.rename`` into ``astrometry_checkpoints/pre_obstoken_stale/``,
  a subdirectory that no non-recursive glob of the record directory reaches.
  Nothing is deleted.
* A file is moved only when ``current_records`` supersedes it: it carries no
  observation or proposal token AND a tokened record exists for the same
  ``(stage, filter, module)``.  A field that predates the token and has no
  replacement is left alone.  A module-less m7 cross-filter record whose only
  tokened siblings are per-module records (``..._nrcb_oNNN``) has a different
  key and is left alone too.
* Untokened timestamped history siblings of a superseded record move with it,
  because ``count_blocking_unverified.py --all-history`` and
  ``per_detector_offset_report.py`` read timestamped files and would otherwise
  keep counting them.
* ``pre_obstoken_stale/MANIFEST.json`` records each moved file's original name,
  mtime and the tokened record that supersedes it.  Reversal is
  ``--restore``, which renames the files listed there back.
* Each ``<base>/<field>/astrometry_checkpoints`` directory is processed on its
  own.  Brick has two live ones (``/blue/...`` for NIRCam, ``/orange/...`` for
  MIRI and older records); pass each root with ``--base``.

Usage::

    python quarantine_pre_obstoken_checkpoints.py --all                  # dry run
    python quarantine_pre_obstoken_checkpoints.py --base /blue/adamginsburg/adamginsburg/jwst --field brick
    python quarantine_pre_obstoken_checkpoints.py --field w51 --execute
    python quarantine_pre_obstoken_checkpoints.py --field w51 --restore --execute
"""
import argparse
import datetime
import glob
import json
import os
import sys

from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    current_records, parse_record_name, record_is_tokenised,
    record_supersession_key)

DEFAULT_BASE = os.environ.get("GC_BASEPATH_OVERRIDE",
                              os.environ.get("JWST_BASE",
                                             "/orange/adamginsburg/jwst"))
RECORD_SUBDIR = "astrometry_checkpoints"
STALE_SUBDIR = "pre_obstoken_stale"
MANIFEST = "MANIFEST.json"


def plan_for_dir(record_dir):
    """``(moves, counts)`` for one record directory.

    ``moves`` is ``[{'name', 'mtime', 'superseded_by'}]``; ``counts`` has the
    figures the dry run prints: untokened and total ``_latest`` records, and
    untokened timestamped history files.
    """
    current = current_records(record_dir)
    kept = set(current.values())
    # Which tokened records supersede an untokened key.
    superseders = {}
    for key, path in current.items():
        info = {'stage': key[0], 'filt': key[1], 'module': key[2],
                'obs': key[3], 'proposal': key[4]}
        if record_is_tokenised(info):
            superseders.setdefault(record_supersession_key(info), []).append(
                os.path.basename(path))

    n_latest = n_untokened_latest = n_untokened_history = 0
    moves = []
    for path in sorted(glob.glob(os.path.join(record_dir, "checkpoint_*.json"))):
        name = os.path.basename(path)
        info = parse_record_name(name)
        if info is None:
            continue
        is_latest = name.endswith("_latest.json")
        if is_latest:
            n_latest += 1
        if record_is_tokenised(info):
            continue
        if is_latest:
            n_untokened_latest += 1
        else:
            n_untokened_history += 1
        by = superseders.get(record_supersession_key(info))
        # `current_records` is the arbiter for a `_latest` file; a history file
        # follows the same key.  No tokened sibling -> nothing supersedes it.
        if not by or path in kept:
            continue
        moves.append({"name": name, "mtime": os.path.getmtime(path),
                      "superseded_by": sorted(by)})
    return moves, {"latest": n_latest,
                   "untokened_latest": n_untokened_latest,
                   "untokened_history": n_untokened_history}


def _read_manifest(stale_dir):
    path = os.path.join(stale_dir, MANIFEST)
    if not os.path.exists(path):
        return {"files": []}
    with open(path) as fh:
        return json.load(fh)


def _write_manifest(stale_dir, manifest):
    tmp = os.path.join(stale_dir, MANIFEST + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, os.path.join(stale_dir, MANIFEST))


def quarantine_dir(record_dir, execute=False):
    """Plan, and with ``execute`` perform, the moves for one directory.

    Returns ``(moves, counts, skipped)``; ``skipped`` lists names whose
    destination already exists (never overwritten).
    """
    moves, counts = plan_for_dir(record_dir)
    skipped = []
    if not execute or not moves:
        return moves, counts, skipped
    stale_dir = os.path.join(record_dir, STALE_SUBDIR)
    os.makedirs(stale_dir, exist_ok=True)
    manifest = _read_manifest(stale_dir)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    done = []
    for mv in moves:
        dest = os.path.join(stale_dir, mv["name"])
        if os.path.exists(dest):
            skipped.append(mv["name"])
            continue
        os.rename(os.path.join(record_dir, mv["name"]), dest)
        manifest["files"].append(dict(mv, moved_at=stamp))
        done.append(mv)
    _write_manifest(stale_dir, manifest)
    return done, counts, skipped


def restore_dir(record_dir, execute=False):
    """Rename the files listed in the manifest back into ``record_dir``.

    Returns ``(restored_names, skipped_names)``.  A file whose original name is
    occupied is skipped, never overwritten.
    """
    stale_dir = os.path.join(record_dir, STALE_SUBDIR)
    if not os.path.exists(os.path.join(stale_dir, MANIFEST)):
        return [], []
    manifest = _read_manifest(stale_dir)
    restored, skipped, remaining = [], [], []
    for ent in manifest["files"]:
        src = os.path.join(stale_dir, ent["name"])
        dest = os.path.join(record_dir, ent["name"])
        if not os.path.exists(src):
            continue
        if os.path.exists(dest):
            skipped.append(ent["name"])
            remaining.append(ent)
            continue
        restored.append(ent["name"])
        if execute:
            os.rename(src, dest)
    if execute:
        manifest["files"] = remaining
        _write_manifest(stale_dir, manifest)
    return restored, skipped


def record_dirs(bases, fields, all_fields):
    """``[(label, directory)]`` for every requested field under every base."""
    out = []
    for base in bases:
        if all_fields:
            pattern = os.path.join(base, "*", RECORD_SUBDIR)
            dirs = sorted(glob.glob(pattern))
        else:
            dirs = [os.path.join(base, f, RECORD_SUBDIR) for f in fields]
        for d in dirs:
            if os.path.isdir(d):
                out.append((f"{base}:{os.path.basename(os.path.dirname(d))}", d))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", action="append", default=None,
                    help="root holding <field>/astrometry_checkpoints; repeat "
                         f"for several (default {DEFAULT_BASE}).  Brick needs "
                         "both /blue/adamginsburg/adamginsburg/jwst and "
                         "/orange/adamginsburg/jwst")
    ap.add_argument("--field", action="append", default=[],
                    help="field name; repeat for several")
    ap.add_argument("--all", action="store_true", help="every field under each base")
    ap.add_argument("--execute", action="store_true",
                    help="move the files (default: dry run)")
    ap.add_argument("--restore", action="store_true",
                    help="reverse an earlier move using the manifest")
    args = ap.parse_args(argv)
    if not (args.field or args.all):
        ap.error("give --field NAME (repeatable) or --all")
    bases = args.base or [DEFAULT_BASE]
    dirs = record_dirs(bases, args.field, args.all)
    if not dirs:
        print(f"NOTHING FOUND under {bases}; check --base/--field.  This is "
              f"not a clean bill of health.")
        return 1

    mode = "EXECUTE" if args.execute else "dry run"
    tot_moved = 0
    for label, d in dirs:
        if args.restore:
            restored, skipped = restore_dir(d, execute=args.execute)
            if restored or skipped:
                print(f"{label}: restore {len(restored)} file(s)"
                      + (f", {len(skipped)} skipped (name occupied)"
                         if skipped else "") + f" [{mode}]")
            tot_moved += len(restored)
            continue
        moves, counts, skipped = quarantine_dir(d, execute=args.execute)
        print(f"{label}: {len(moves)} to quarantine "
              f"({counts['untokened_latest']} untokened of {counts['latest']} "
              f"_latest; {counts['untokened_history']} untokened timestamped) "
              f"[{mode}]  {d}")
        for mv in moves:
            print(f"    {mv['name']}  <- {', '.join(mv['superseded_by'])}")
        for name in skipped:
            print(f"    SKIPPED {name}: destination exists")
        tot_moved += len(moves)
    verb = "restored" if args.restore else "quarantined"
    print(f"{tot_moved} file(s) {verb if args.execute else 'planned'} "
          f"across {len(dirs)} director(y/ies) [{mode}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
