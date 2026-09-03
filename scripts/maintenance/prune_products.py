#!/usr/bin/env python
"""Report -- and, only when told twice, remove -- products the retention policy
declares dead.

The policy itself lives in :mod:`jwst_gc_pipeline.retention`; this is the
operator front end.  It defaults to a dry run, it will not delete without a
manifest path, and it asks ``squeue`` which fields are busy before it plans
anything, because the fields with the most reclaimable bytes are exactly the
ones most likely to have a chain in flight.

Typical use, in the order a cleanup actually happens:

    # 1. see what the safe rules would take, field by field
    python scripts/maintenance/prune_products.py --target brick --target cloudc

    # 2. the same, with the two opt-in derivative rules and full detail
    python scripts/maintenance/prune_products.py --target brick \\
        --rule allcols --rule duplicate_table_format --verbose

    # 3. the named-dead directories, which are counted as whole trees
    python scripts/maintenance/prune_products.py --target brick --directories

    # 4. only after reading the manifest from step 1-3:
    python scripts/maintenance/prune_products.py --target brick \\
        --manifest /path/prune_brick_2026-09-02.json --apply

Nothing about this script is idempotent-by-luck: re-running step 4 with the same
manifest path overwrites the manifest, so use a dated name.
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..')))

from jwst_gc_pipeline import fields as field_registry   # noqa: E402
from jwst_gc_pipeline import retention                  # noqa: E402


def _human(nbytes):
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if abs(nbytes) < 1024 or unit == 'TB':
            return f'{nbytes:,.1f} {unit}' if unit != 'B' else f'{nbytes:,.0f} B'
        nbytes /= 1024.0


def _known_targets():
    try:
        return sorted(field_registry.BY_NAME)
    except AttributeError:                       # registry shape changed
        return []


def _roots_for(args):
    if args.root:
        return [os.path.abspath(r) for r in args.root]
    roots = []
    for target in args.target:
        bp = field_registry.fields_basepath(target)
        if not os.path.isdir(bp):
            print(f'skip {target}: {bp} is not a directory', file=sys.stderr)
            continue
        roots.append(os.path.realpath(bp.rstrip('/')))
    return roots


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--target', action='append', default=[],
                   help='Field name from the registry (repeatable).  Its '
                        'basepath is resolved from fields.yaml.')
    p.add_argument('--root', action='append', default=[],
                   help='Explicit directory to walk (repeatable).  Overrides '
                        '--target.')
    p.add_argument('--rule', action='append', default=[],
                   help='Enable exactly these rules (repeatable).  Default: '
                        'every rule marked default_on, which excludes the two '
                        'derivative-table rules.')
    p.add_argument('--list-rules', action='store_true',
                   help='Print the policy and exit.')
    p.add_argument('--directories', action='store_true',
                   help='Plan whole quarantine DIRECTORIES instead of files.')
    p.add_argument('--min-age-days', type=float, default=30.0,
                   help='Global age floor; a rule with a longer floor keeps '
                        'its own.  Default 30.')
    p.add_argument('--releases-root',
                   default='/orange/adamginsburg/jwst/releases',
                   help='Release tree whose symlink targets are protected.')
    p.add_argument('--protect', action='append', default=[],
                   help='Extra protect glob (repeatable).')
    p.add_argument('--assume-idle', action='store_true',
                   help='Skip the squeue check.  Only when you know the queue '
                        'is empty; a running chain that loses its inputs does '
                        'not fail cleanly.')
    p.add_argument('--manifest',
                   help='Where to write the JSON manifest.  Required for '
                        '--apply.')
    p.add_argument('--apply', action='store_true',
                   help='Actually delete.  Without this the script only '
                        'reports.')
    p.add_argument('--show-vetoed', action='store_true',
                   help='Include candidates the guard refused, with the '
                        'reason.  Useful for checking the guard works.')
    p.add_argument('--verbose', '-v', action='store_true',
                   help='List every candidate, not just the per-rule totals.')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.list_rules:
        for rule in retention.DEFAULT_RULES:
            state = 'on ' if rule.default_on else 'off'
            print(f'{state}  {rule.name:<24} floor {rule.min_age_days:>4.0f} d  '
                  f'{rule.description}')
        return 0

    if not args.target and not args.root:
        print('nothing to do: pass --target or --root', file=sys.stderr)
        return 2

    if args.apply and not args.manifest:
        print('--apply requires --manifest: the manifest is the only record '
              'of what was removed and why', file=sys.stderr)
        return 2

    roots = _roots_for(args)
    if not roots:
        print('no readable roots', file=sys.stderr)
        return 2

    try:
        guard = retention.guard_for(
            roots, _known_targets(), releases_root=args.releases_root,
            min_age_days=args.min_age_days, protect_globs=args.protect,
            assume_idle=args.assume_idle)
    except retention.RetentionError as ex:
        print(str(ex), file=sys.stderr)
        return 3

    if guard.busy_fields:
        print(f'busy fields (protected): {", ".join(sorted(guard.busy_fields))}')
    if guard.release_targets:
        print(f'release symlink targets protected: {len(guard.release_targets)}')

    enabled = set(args.rule) if args.rule else None
    if enabled:
        unknown = enabled - set(retention.POLICY)
        if unknown:
            print(f'unknown rule(s): {", ".join(sorted(unknown))}',
                  file=sys.stderr)
            return 2

    if args.directories:
        candidates = retention.plan_quarantine_directories(
            roots, guard=guard, include_vetoed=args.show_vetoed)
    else:
        candidates = retention.plan(roots, guard=guard, enabled=enabled,
                                    include_vetoed=args.show_vetoed)

    by_rule = {}
    for c in candidates:
        key = c.rule if c.deletable else f'{c.rule} (VETOED)'
        n, b = by_rule.get(key, (0, 0))
        by_rule[key] = (n + 1, b + c.size)

    print()
    for key, (n, b) in sorted(by_rule.items(), key=lambda kv: -kv[1][1]):
        print(f'{_human(b):>12}  {n:>7}  {key}')
    total = sum(c.size for c in candidates if c.deletable)
    print(f'{_human(total):>12}  {sum(1 for c in candidates if c.deletable):>7}'
          f'  TOTAL deletable')

    if args.verbose:
        print()
        for c in sorted(candidates, key=lambda c: -c.size):
            mark = ' ' if c.deletable else 'V'
            print(f'{mark} {_human(c.size):>10}  {c.path}')
            print(f'             {c.reason}')
            if c.vetoed_by:
                print(f'             VETOED: {c.vetoed_by}')

    summary = retention.apply(candidates, dry_run=not args.apply,
                              manifest_path=args.manifest)
    print()
    if args.apply:
        print(f"removed {summary['deleted']} of {summary['deletable']} "
              f"({_human(summary['bytes'])}); {summary['failed']} failed")
    else:
        print('DRY RUN -- nothing was removed.  Re-run with --manifest '
              '<path> --apply to act on this plan.')
    if args.manifest:
        print(f'manifest: {args.manifest}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
