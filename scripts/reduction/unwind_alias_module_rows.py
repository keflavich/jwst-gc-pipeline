"""Remove per-exposure offsets rows written under an aliasing module spelling.

Issue #298.  The m2 checkpoint ingested a stale bare-module per-frame catalog
(``f360m_nrcb_visit...``) alongside its ``long`` counterpart
(``f360m_nrcblong_visit...``) and wrote per-exposure jitter rows under BOTH
tokens for the same physical frames.  ``unified_alignment._read_consensus``
resolves a frame through ``_module_variants`` (``nrcblong -> {nrcblong,
nrcb}``), matches two rows, and refuses to reduce the field.

**This is a removal, not a choice between two measurements.**  On cloudef
2092/002 the bare-module catalogs turned out to be observation 005's frames,
relabelled onto o002 because ``seed_offsets_table_from_consensus`` builds the
visit token from the RUN's field and the catalog's ``VISIT`` meta, which is 1
in both observations.  So the bare rows are not a second opinion about the
right frames; they are the wrong frames' corrections.

This script therefore removes the rows whose module token is NOT the detector
spelling, and only where an aliasing partner exists.  It refuses to guess in
any other situation:

* if only one spelling is present, nothing is done;
* if the surviving spelling is not the ``long`` one for an LW filter, it stops
  and asks -- an SW filter's bare-module row may be legitimate;
* if the two rows carry the SAME correction it still removes one, but says so,
  because a true duplicate is a different (and less alarming) defect.

Provenance: the original table is copied to ``<table>.pre_unwind298_<stamp>``
before anything is written, and a JSON receipt listing every removed row (with
its full contents) is written next to it.  Nothing is edited in place.

Usage::

    python unwind_alias_module_rows.py TABLE.csv --dry-run
    python unwind_alias_module_rows.py TABLE.csv --apply
"""
import argparse
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime, timezone

from astropy.table import Table


def module_family(module):
    """``nrcb1``/``nrcb``/``nrcblong`` -> ``nrcb``."""
    m = str(module).strip().lower().strip('1234')
    return m[:-4] if m.endswith('long') else m


def find_alias_groups(tbl):
    """``{(visit, filter, exposure, family, vgroup): {module: [row indices]}}``
    for every key genuinely carried under ALIASING module spellings.

    Only a BARE family token aliases.  ``unified_alignment._module_variants``
    maps a frame to ``{frame, frame-without-digits, frame-without-long}``, so a
    frame ``nrcb3`` matches rows ``nrcb3`` and ``nrcb``, and a frame
    ``nrcblong`` matches ``nrcblong`` and ``nrcb`` -- but ``nrcb3`` and
    ``nrcb4`` are different detectors and match nothing of each other's.
    Grouping on the family alone would flag every legitimate per-detector
    table, which is how this script's first draft "found" 9 non-collisions in
    cloudef's SW filters.
    """
    groups = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(tbl):
        mod = str(r['Module']).strip().lower()
        key = (str(r['Visit']), str(r['Filter']), str(r['Exposure']),
               module_family(mod), str(r['Vgroup']))
        groups[key][mod].append(i)
    return {k: v for k, v in groups.items()
            if len(v) > 1 and k[3] in v}


def choose_survivor(modules):
    """Which spelling to keep.  ``None`` when the answer is not obvious."""
    mods = sorted(modules)
    longs = [m for m in mods if m.endswith('long')]
    numbered = [m for m in mods if m[-1:] in '1234']
    if len(longs) == 1 and len(mods) == 2 and not numbered:
        # nrcblong (the detector) vs nrcb (the module spelling of the same
        # detector) -- keep the detector.
        return longs[0]
    if len(numbered) == len(mods) - 1 and len(longs) == 0:
        # bare module vs its numbered detectors -- keep the detectors, but that
        # is a different defect and not what this script is for.
        return None
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('table')
    ap.add_argument('--apply', action='store_true',
                    help='write the change (default is a dry run)')
    ap.add_argument('--dry-run', action='store_true', help='explicit no-op mode')
    args = ap.parse_args()
    if args.apply and args.dry_run:
        raise SystemExit('--apply and --dry-run are mutually exclusive')

    tbl = Table.read(args.table)
    groups = find_alias_groups(tbl)
    if not groups:
        print(f'{args.table}: no aliasing module spellings; nothing to do')
        return

    drop, receipts, refused = [], [], []
    for key, mods in sorted(groups.items()):
        keep = choose_survivor(mods)
        if keep is None:
            refused.append((key, sorted(mods)))
            continue
        same = len({(round(float(tbl[i]['dra (arcsec)']), 9),
                     round(float(tbl[i]['ddec (arcsec)']), 9))
                    for idxs in mods.values() for i in idxs}) == 1
        for mod, idxs in sorted(mods.items()):
            if mod == keep:
                continue
            for i in idxs:
                drop.append(i)
                receipts.append(dict(
                    key=dict(visit=key[0], filter=key[1], exposure=key[2],
                             family=key[3], vgroup=key[4]),
                    removed_module=mod, kept_module=keep,
                    identical_values=bool(same),
                    row={c: (float(tbl[i][c])
                             if hasattr(tbl[i][c], 'dtype')
                             and tbl[i][c].dtype.kind == 'f'
                             else str(tbl[i][c]))
                         for c in tbl.colnames}))

    print(f'{args.table}: {len(groups)} aliasing key(s), '
          f'{len(drop)} row(s) to remove, {len(refused)} refused')
    for key, mods in refused:
        print(f'  REFUSED {key}: spellings {mods} -- no obvious detector '
              f'spelling to keep; decide by hand and re-run')
    for rc in receipts:
        k = rc['key']
        print(f"  remove {k['filter']} {k['visit']} exp{k['exposure']} "
              f"vgroup{k['vgroup']}: module {rc['removed_module']!r} "
              f"(keeping {rc['kept_module']!r})"
              + ('  [identical values]' if rc['identical_values'] else ''))

    if not args.apply:
        print('\ndry run; pass --apply to write')
        return
    if not drop:
        print('nothing removable; not writing')
        return

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup = f'{args.table}.pre_unwind298_{stamp}'
    shutil.copy2(args.table, backup)
    receipt_path = f'{args.table}.unwind298_{stamp}.json'
    with open(receipt_path, 'w') as fh:
        json.dump(dict(table=os.path.abspath(args.table), backup=backup,
                       issue=298, date=stamp, removed=receipts,
                       refused=[dict(key=list(k), modules=m) for k, m in refused]),
                  fh, indent=2)
    keep_mask = [i not in set(drop) for i in range(len(tbl))]
    tbl[keep_mask].write(args.table, overwrite=True)
    print(f'\nwrote {args.table} ({len(tbl)} -> {sum(keep_mask)} rows)')
    print(f'backup  {backup}')
    print(f'receipt {receipt_path}')


if __name__ == '__main__':
    main()
