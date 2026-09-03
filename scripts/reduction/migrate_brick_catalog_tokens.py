#!/usr/bin/env python3
"""Give brick's per-filter merged catalogs the ``_o001`` / ``_o004`` token.

WHY THIS EXISTS
---------------
PR #597 added brick to ``naming.PER_OBS_MERGED_FIELDS`` so its two proposals
would stop overwriting one cross-band combined table (#590: brick's combined
catalog held one proposal's four bands instead of all eleven).  That membership
also drives the MODULE-slot token, which lands on every PER-FILTER catalog --
both where they are written and where the m7 crossband seed reads them.  Every
product on disk predates the merge, so the seed's glob now matches nothing and
brick's chain is dead (#625/#620)::

    on disk   f182m_merged_indivexp_merged_resbgsub_m6_dao_basic_vetted.fits
    expected  f182m_merged_o001_indivexp_merged_resbgsub_m6_dao_basic_vetted.fits
    matching the expected form: 0

The alternative -- splitting the token so per-filter names stay untokened -- was
attempted and abandoned (PR #628): one membership decision drives the write
name, the input glob, ``vetted_tok`` and ``merge_field_for_proposal``, and the
last two bind glob and output together deliberately.  Unpicking that risks
reopening #590 on a field that is mid-release.  Renaming the files is the
cheaper, reversible option: no code moves and every existing test still passes.

WHICH PROPOSAL OWNS A FILE
--------------------------
brick's two proposals have DISJOINT filters, so the filter in the name is an
unambiguous owner::

    1182 -> _o004    F115W F200W F356W F444W
    2221 -> _o001    F182M F187N F212N F405N F410M F466N

SCOPE
-----
Only current-generation per-filter merged catalogs -- those carrying an
``_m2``..``_m8`` iteration token.  Legacy ``LOCKED`` / ``XFILT`` /
``crowdsource`` / ``daoiterative`` products are left alone: they predate this
naming entirely and nothing reads them through the tokened path.

Writes a JSON manifest of every rename so the whole pass can be undone with
``--undo``.  Dry-run by default.
"""
import argparse
import datetime
import glob
import json
import os
import re
import sys

DEFAULT_CATALOGS = '/orange/adamginsburg/jwst/brick/catalogs'

#: filter -> observation token.  Disjoint by construction; see module docstring.
OWNER = {
    **{f: '_o004' for f in ('f115w', 'f200w', 'f356w', 'f444w')},      # 1182
    **{f: '_o001' for f in ('f182m', 'f187n', 'f212n',
                            'f405n', 'f410m', 'f466n')},               # 2221
}

#: `{filter}_{module}_indivexp_merged...<_mN>...` -- the current-generation
#: per-filter merged catalogs, and only those.
PATTERN = re.compile(
    r'^(?P<filt>f\d{3}[a-z0-9]+)_(?P<mod>merged|nrca|nrcb)_indivexp_merged'
    r'(?P<rest>.*_m[2-8](?:_|\.|$).*)$')


def plan(catalogs_dir):
    """``[(old, new), ...]`` for every file that should be tokened."""
    out, skipped = [], []
    for path in sorted(glob.glob(os.path.join(catalogs_dir, '*'))):
        name = os.path.basename(path)
        m = PATTERN.match(name)
        if not m:
            if re.match(r'^f\d{3}[a-z0-9]+_(merged|nrca|nrcb)_indivexp', name):
                skipped.append(name)      # legacy generation; leave alone
            continue
        token = OWNER.get(m.group('filt'))
        if token is None:
            skipped.append(name)
            continue
        if f"_{m.group('mod')}{token}_" in name:
            continue                       # already tokened
        new = name.replace(f"_{m.group('mod')}_indivexp",
                           f"_{m.group('mod')}{token}_indivexp", 1)
        out.append((path, os.path.join(catalogs_dir, new)))
    return out, skipped


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--catalogs', default=DEFAULT_CATALOGS,
                   help=f'brick catalogs directory (default {DEFAULT_CATALOGS})')
    p.add_argument('--apply', action='store_true',
                   help='perform the renames (default: dry run)')
    p.add_argument('--undo', metavar='MANIFEST',
                   help='reverse a previous run from its manifest')
    args = p.parse_args(argv)

    if args.undo:
        pairs = json.load(open(args.undo))
        done = 0
        for old, new in pairs:
            if os.path.exists(new) and not os.path.exists(old):
                os.rename(new, old)
                done += 1
        print(f'undid {done} of {len(pairs)} renames from {args.undo}')
        return 0

    pairs, skipped = plan(args.catalogs)
    clash = [n for _o, n in pairs if os.path.exists(n)]
    print(f'{len(pairs)} file(s) to rename; {len(skipped)} legacy file(s) left alone')
    for old, new in pairs[:5]:
        print(f'  {os.path.basename(old)}\n    -> {os.path.basename(new)}')
    if len(pairs) > 5:
        print(f'  ... and {len(pairs) - 5} more')
    if clash:
        print(f'REFUSING: {len(clash)} target name(s) already exist, e.g. '
              f'{os.path.basename(clash[0])}', file=sys.stderr)
        return 2
    if not args.apply:
        print('\ndry run -- pass --apply to perform these renames')
        return 0

    stamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%SZ')
    manifest = os.path.join(args.catalogs,
                            f'_rename_manifest_per_obs_token_{stamp}.json')
    json.dump(pairs, open(manifest, 'w'), indent=1)
    for old, new in pairs:
        os.rename(old, new)
    print(f'renamed {len(pairs)} file(s)')
    print(f'manifest (undo with --undo): {manifest}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
