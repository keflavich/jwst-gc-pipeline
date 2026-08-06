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

Provenance and safety: this edits a LIVE offsets table, possibly while a retie
loop is running, so it uses the same protocol the pipeline's own writers do
(``jwst_gc_pipeline.atomic_io``):

* an exclusive ``locked()`` around the whole read-modify-write, so a concurrent
  ``update_offsets_table``/``seed_offsets_table_from_consensus`` cannot
  interleave;
* ``keep_a_copy`` to ``<table>.pre_unwind298_<stamp>`` -- a copy, never a move,
  because a reader in the gap takes the table-does-not-exist branch and aligns
  a frame at (0, 0);
* ``atomic_write``, so no reader ever sees a partial table;
* a JSON receipt listing every removed row in full;
* and a VERIFY pass that re-reads the written table and refuses to report
  success unless the surviving rows are exactly the ones intended.

Usage::

    python unwind_alias_module_rows.py TABLE.csv --dry-run
    python unwind_alias_module_rows.py TABLE.csv --apply
"""
import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone

from astropy.table import Table

from jwst_gc_pipeline.atomic_io import atomic_write, keep_a_copy, locked


class UnsupportedTableError(ValueError):
    """The table's schema cannot express the aliasing identity."""


#: Columns the aliasing identity is built from.  Many offsets tables on disk
#: carry a different schema (28 of 54 lack one or more of these), and a tool
#: whose job is careful operation on live tables must say so rather than
#: traceback with a raw KeyError.
REQUIRED_COLUMNS = ('Visit', 'Filter', 'Exposure', 'Module', 'Vgroup')


def module_family(module):
    """``nrcb1``/``nrcb``/``nrcblong`` -> ``nrcb``.

    Mirrors ``astrometry_checkpoint._module_family`` with the case/whitespace
    tolerance a CSV needs.
    """
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
    missing = [c for c in REQUIRED_COLUMNS if c not in tbl.colnames]
    if missing:
        raise UnsupportedTableError(
            f"table lacks {missing}; the aliasing identity needs "
            f"{list(REQUIRED_COLUMNS)}.  Many offsets tables carry a different "
            f"schema and cannot alias at all -- nothing to do here.")
    groups = defaultdict(lambda: defaultdict(list))
    vgroups = defaultdict(set)
    for i, r in enumerate(tbl):
        mod = str(r['Module']).strip().lower()
        # Group WITHOUT the vgroup: an EMPTY row Vgroup is a wildcard at read
        # time (`vgroup_row_matches`), so it aliases across vgroups too, and
        # grouping on the vgroup exactly puts such a row in its own bucket and
        # misses the collision entirely.  arches once carried 85 legacy rows
        # with no Vgroup.
        key = (str(r['Visit']), str(r['Filter']), str(int(r['Exposure'])),
               module_family(mod))
        groups[key][mod].append(i)
        vgroups[key].add(str(r['Vgroup']).strip())
    out = {}
    for k, mods in groups.items():
        if len(mods) < 2 or k[3] not in mods:
            continue
        vg = vgroups[k]
        # distinct non-empty vgroups genuinely cannot collide
        if len(vg) > 1 and all(v for v in vg):
            continue
        out[k] = mods
    return out


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
    try:
        groups = find_alias_groups(tbl)
    except UnsupportedTableError as ex:
        print(f'{args.table}: {ex}')
        return
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
                             family=key[3],
                             vgroups=sorted({str(tbl[j]['Vgroup']).strip()
                                             for idxs in mods.values()
                                             for j in idxs})),
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
              f"vgroup{'/'.join(k['vgroups'])}: module {rc['removed_module']!r} "
              f"(keeping {rc['kept_module']!r})"
              + ('  [identical values]' if rc['identical_values'] else ''))

    if not args.apply:
        print('\ndry run; pass --apply to write')
        return
    if not drop:
        print('nothing removable; not writing')
        return

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    keep_mask = [i not in set(drop) for i in range(len(tbl))]
    expected = [dict(zip(tbl.colnames, [str(v) for v in tbl[i]]))
                for i, k in enumerate(keep_mask) if k]

    with locked(args.table):
        # Re-read under the lock: the table may have changed between the dry
        # run above and now, and removing rows by INDEX from a stale read would
        # delete the wrong ones.
        current = Table.read(args.table)
        # Compare the IDENTITY of every row, not just the count.  A concurrent
        # writer that substituted a row -- update_offsets_table upserts in
        # place, so a correction changes a row without changing the length --
        # would pass a length check, and we drop rows by INDEX.
        def _ident(t):
            cols = [c for c in REQUIRED_COLUMNS if c in t.colnames]
            return [tuple(str(r[c]) for c in cols) for r in t]

        if len(current) != len(tbl) or _ident(current) != _ident(tbl):
            raise SystemExit(
                f'{args.table} changed under us between analysis and write '
                f'({len(tbl)} -> {len(current)} rows; row identities '
                f'{"differ" if len(current) == len(tbl) else "and count differ"}). '
                f'Rows are removed by index, so acting on the stale read would '
                f'delete the wrong ones.  Re-run.')
        backup = f'{args.table}.pre_unwind298_{stamp}'
        keep_a_copy(args.table, backup)
        receipt_path = f'{args.table}.unwind298_{stamp}.json'
        with open(receipt_path, 'w') as fh:
            json.dump(dict(table=os.path.abspath(args.table), backup=backup,
                           issue=298, date=stamp, removed=receipts,
                           refused=[dict(key=list(k), modules=m)
                                    for k, m in refused]),
                      fh, indent=2)
        with atomic_write(args.table) as tmp_path:
            current[keep_mask].write(tmp_path, overwrite=True)

        # VERIFY.  A backup is only useful if someone notices; check here.
        written = Table.read(args.table)
        got = [dict(zip(written.colnames, [str(v) for v in row]))
               for row in written]
        if len(got) != len(expected):
            raise SystemExit(
                f'{args.table}: wrote {len(got)} rows, expected '
                f'{len(expected)}.  The backup at {backup} is intact -- '
                f'restore it and investigate.')
        for want, have in zip(expected, got):
            for col in ('Visit', 'Filter', 'Exposure', 'Module', 'Vgroup'):
                if col in want and want[col] != have.get(col):
                    raise SystemExit(
                        f'{args.table}: surviving row mismatch on {col} '
                        f'({want[col]!r} != {have.get(col)!r}).  The backup at '
                        f'{backup} is intact -- restore it and investigate.')

    print(f'\nwrote {args.table} ({len(tbl)} -> {sum(keep_mask)} rows, verified)')
    print(f'backup  {backup}')
    print(f'receipt {receipt_path}')


if __name__ == '__main__':
    main()
