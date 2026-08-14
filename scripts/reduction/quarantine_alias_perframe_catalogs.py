"""Set aside a stale bare-module per-frame catalog that shadows its ``long`` twin.

Issue #298 has two halves and only one of them had a tool.

``unwind_alias_module_rows.py`` removes the offsets ROWS a collision produced.
It cannot stop the collision happening again, because the rows come from the
CATALOGS: an LW filter's per-frame catalog written under the bare family token
(``f360m_nrcb_visit001_...``) sits on disk next to the current one written under
the detector token (``f360m_nrcblong_visit001_...``), the m2 checkpoint ingests
BOTH as if they were two exposures, and ``seed_offsets_table_from_consensus``
refuses the write::

    OffsetsTableUpdateError: consensus table Offsets_JWST_Brick2092_consensus.csv
    would carry the SAME frame under aliasing module spellings ... (8 collisions).
    This means the checkpoint ingested one physical exposure twice under two
    module tokens -- check for stale bare-module per-frame catalogs next to their
    `long` counterparts (issue #298).

That is a 2-hour m12 finalize dying on its last step, on a field whose
astrometry had already passed.  It cost cloudef 2092/002 three campaign cycles
(2026-08-09 through 08-14) and sgrb2 5365/001 one, and it will cost them again
on the next run, because unwinding the rows leaves the files that wrote them.

**What counts as stale here is narrow and checkable.**  A file is set aside only
when ALL of:

* its filter is long-wavelength, so the detector token IS the ``long`` one and a
  bare family token cannot be the current spelling;
* a file exists with the identical name but the ``long`` token -- same filter,
  same visit, same vgroup, same exposure, same stage suffix;
* that twin is NEWER.

Anything else is left alone and reported.  In particular a bare-module file with
no ``long`` twin is NOT touched: on brick and cloudc some frames exist only under
the bare spelling, and those are products, not duplicates.

Set aside, not deleted: each file is renamed in place to
``<name>_stale_alias298_<stamp>``, which the pipeline's globs do not match, and
a JSON receipt records every rename so it can be undone with ``--restore``.

Usage::

    python quarantine_alias_perframe_catalogs.py --field cloudef --dry-run
    python quarantine_alias_perframe_catalogs.py --field cloudef --apply
    python quarantine_alias_perframe_catalogs.py --restore RECEIPT.json
"""
import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime, timezone

#: NIRCam long-wavelength filters.  For these the detector token is ``nrcalong``
#: / ``nrcblong``, so a bare ``nrca`` / ``nrcb`` cannot be the current spelling.
#: SW filters are deliberately absent: there a bare family token may be
#: legitimate, and guessing would set aside a product.
LW_FILTERS = frozenset({
    'f250m', 'f277w', 'f300m', 'f322w2', 'f323n', 'f335m', 'f356w', 'f360m',
    'f405n', 'f410m', 'f444w', 'f466n', 'f470n', 'f480m',
})

#: ``f360m_nrcb_visit001_vgroup02101_exp00001_m2_daophot_basic.fits``
PERFRAME_RE = re.compile(
    r'^(?P<filt>f[0-9]+[a-z0-9]*)_(?P<mod>nrc[ab])(?P<long>long)?'
    r'_(?P<frame>visit\d+_vgroup\d+_exp\d+)(?P<tail>.*)$')

SUFFIX = '_stale_alias298_'


class RestoreError(RuntimeError):
    """A receipt cannot be applied in reverse without losing something."""


def _index(field_root):
    """``{(filter, family, frame, tail): path}`` for bare and long spellings."""
    bare, long_ = {}, {}
    for d in sorted(glob.glob(os.path.join(field_root, 'F*'))):
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if SUFFIX in name:
                # already set aside by an earlier pass; the rename leaves the
                # frame/stage part of the name intact, so without this it comes
                # back as a bare-module file on every subsequent run and pads
                # the "left alone" report with its own past work.
                continue
            m = PERFRAME_RE.match(name)
            if not m or m.group('filt') not in LW_FILTERS:
                continue
            key = (m.group('filt'), m.group('mod'), m.group('frame'),
                   m.group('tail'))
            (long_ if m.group('long') else bare)[key] = os.path.join(d, name)
    return bare, long_


def plan(field_root):
    """``(to_quarantine, skipped)`` -- what is stale, and why the rest is not."""
    bare, long_ = _index(field_root)
    quarantine, skipped = [], []
    for key, path in sorted(bare.items()):
        twin = long_.get(key)
        if twin is None:
            skipped.append((path, 'no `long` twin -- this is the only copy'))
            continue
        if os.path.getmtime(twin) <= os.path.getmtime(path):
            skipped.append((path, 'the bare copy is NEWER than its `long` twin'
                                  ' -- that is not the stale-shadow shape'))
            continue
        quarantine.append((path, twin))
    return quarantine, skipped


def apply_plan(quarantine, receipt_path):
    moved = []
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    for path, twin in quarantine:
        dest = f'{path}{SUFFIX}{stamp}'
        os.rename(path, dest)
        moved.append({'from': path, 'to': dest, 'twin': twin})
    with open(receipt_path, 'w') as fh:
        json.dump({'stamp': stamp, 'issue': 298, 'moved': moved}, fh, indent=2)
    return moved


def restore(receipt_path):
    with open(receipt_path) as fh:
        receipt = json.load(fh)
    for row in receipt['moved']:
        if not os.path.exists(row['to']):
            raise RestoreError(f"{row['to']} is gone -- cannot restore "
                               f"{row['from']}")
        if os.path.exists(row['from']):
            raise RestoreError(f"{row['from']} exists again -- restoring would "
                               f"overwrite a file written since the quarantine")
    for row in receipt['moved']:
        os.rename(row['to'], row['from'])
    return len(receipt['moved'])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--field', help='target name under --root')
    ap.add_argument('--root', default='/orange/adamginsburg/jwst')
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--restore', metavar='RECEIPT.json')
    args = ap.parse_args(argv)

    if args.restore:
        n = restore(args.restore)
        print(f'restored {n} file(s) from {args.restore}')
        return 0
    if not args.field:
        ap.error('--field is required unless --restore')

    field_root = os.path.join(args.root, args.field)
    quarantine, skipped = plan(field_root)
    print(f'{field_root}: {len(quarantine)} stale shadow(s), '
          f'{len(skipped)} bare-module file(s) left alone')
    for path, twin in quarantine:
        print(f'  set aside {os.path.basename(path)}')
        print(f'       twin {os.path.basename(twin)}')
    for path, why in skipped:
        print(f'  keep      {os.path.basename(path)}: {why}')
    if not args.apply:
        print('\ndry run; pass --apply to rename')
        return 0
    if not quarantine:
        return 0
    receipt = os.path.join(
        field_root, f'quarantine298_'
        f'{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}.json')
    apply_plan(quarantine, receipt)
    print(f'\nreceipt {receipt}  (undo: --restore {receipt})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
