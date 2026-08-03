"""CLI: build the monitoring pages, or submit the probe cutouts.

    python -m jwst_gc_pipeline.monitoring
    python -m jwst_gc_pipeline.monitoring --target brick --target w51
    python -m jwst_gc_pipeline.monitoring --cutout-label monitor5as
    python -m jwst_gc_pipeline.monitoring probe            # dry-run the matrix
    python -m jwst_gc_pipeline.monitoring probe --execute  # submit it
"""
import argparse
import json
import os
import sys

from . import probe as _probe, report as _report

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(
    os.path.join(__file__, '..'))))


def _add_common(parser):
    parser.add_argument('--target', action='append', default=None,
                        help='restrict to this field (repeatable)')
    parser.add_argument('--instrument', default='nircam')


def build_parser():
    parser = argparse.ArgumentParser(
        prog='python -m jwst_gc_pipeline.monitoring',
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_common(parser)
    parser.add_argument('--outdir', default=_report.DEFAULT_OUTDIR)
    parser.add_argument('--cutout-label', default=None,
                        help='scan the <base>/cutouts/<label> subtree instead of '
                             'the field itself')
    parser.add_argument('--show-skip', action='store_true',
                        help='include not-applicable checks in the findings list')
    parser.add_argument('--no-per-field', action='store_true',
                        help='write only the aggregate page')
    parser.add_argument('--log-dir', default=None)
    parser.add_argument('--json', dest='json_path', default=None,
                        help='also write the scan+verdicts as JSON')
    parser.add_argument('--publish-dir', default=None,
                        help='hardlink the generated pages into this directory '
                             '(symlink if it is on another filesystem), and point '
                             'index.html at the aggregate page. Safe to re-run: '
                             'relinking is what keeps the published copy correct '
                             'if the writer ever changes to an atomic rename.')

    sub = parser.add_subparsers(dest='command')
    probe_p = sub.add_parser('probe', help='plan / submit the 5-arcsec probe cutouts')
    _add_common(probe_p)
    probe_p.add_argument('--execute', action='store_true',
                         help='actually submit (default: print the commands)')
    probe_p.add_argument('--size-arcsec', type=float,
                         default=_probe.DEFAULT_PROBE_ARCSEC)
    probe_p.add_argument('--label', default=_probe.PROBE_LABEL)
    probe_p.add_argument('--repo-root', default=REPO_ROOT)
    probe_p.add_argument('--pipe-root', default=REPO_ROOT,
                         help='worktree pinned onto PYTHONPATH for the run')
    return parser


def cmd_probe(args):
    plans = _probe.plan_all(args.target, instrument=args.instrument,
                            size_arcsec=args.size_arcsec, label=args.label)
    rc = 0
    for plan in plans:
        if 'error' in plan:
            print(f"{plan['target']:<12s} SKIP  {plan['error']}")
            continue
        result = _probe.submit(plan, repo_root=args.repo_root,
                               pipe_root=args.pipe_root, execute=args.execute)
        if not args.execute:
            print(f"{plan['target']:<12s} {plan['filter']:<7s} "
                  f"{plan['cutout_region']:<28s} cover={plan['n_overlapping']}")
            print(f'  {result["command"]}')
        elif result['submitted']:
            print(f"{plan['target']:<12s} submitted {result['jobid']} "
                  f"({plan['job_name']})")
        else:
            rc = 1
            print(f"{plan['target']:<12s} FAILED to submit: {result.get('error')}")
    return rc


def cmd_report(args):
    out = _report.write_report(
        outdir=args.outdir, targets=args.target, instrument=args.instrument,
        cutout_label=args.cutout_label, show_skip=args.show_skip,
        per_field=not args.no_per_field, log_dir=args.log_dir)
    print(_report.summarize(out['entries']))
    print(f"\naggregate : {out['aggregate']}")
    print(f"fragment  : {out['fragment']}")
    if out['fields']:
        print(f"per-field : {len(out['fields'])} files under "
              f"{os.path.join(args.outdir, 'fields')}")
    if args.json_path:
        payload = {'generated': out['generated'],
                   'entries': [{'run': e['run'], 'verdicts': e['verdicts'],
                                'tally': e['tally'], 'worst': e['worst']}
                               for e in out['entries']],
                   'cutouts': out['cutouts']}
        os.makedirs(os.path.dirname(os.path.abspath(args.json_path)) or '.',
                    exist_ok=True)
        with open(args.json_path, 'w') as fh:
            json.dump(payload, fh, indent=1, default=str)
        print(f"json      : {args.json_path}")
    if args.publish_dir:
        linked = _report.publish(args.outdir, args.publish_dir)
        kinds = {}
        for kind in linked.values():
            kinds[kind or 'failed'] = kinds.get(kind or 'failed', 0) + 1
        detail = ', '.join(f'{n} {k}' for k, n in sorted(kinds.items()))
        print(f"published : {args.publish_dir} ({detail})")
        failed = [name for name, kind in linked.items() if kind is None]
        if failed:
            print(f"  could not link: {', '.join(failed[:8])}")
    # exit non-zero when something is failing, so a cron run is actionable
    return 1 if any(e['worst'] == 'fail' for e in out['entries']) else 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == 'probe':
        return cmd_probe(args)
    return cmd_report(args)


if __name__ == '__main__':
    sys.exit(main())
