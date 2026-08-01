#!/usr/bin/env python
"""Build the diagnostic figure set and LaTeX write-up for one field (or all).

    python scripts/analysis/make_diagnostic_writeup.py --field brick
    python scripts/analysis/make_diagnostic_writeup.py --all --skip-empty

Output goes to ``<field basepath>/diagnostic_writeup/``, which is initialised
as its own git repository so it can be pushed to Overleaf independently of
the pipeline repo.

Each figure is built in isolation: a builder that fails takes its own figure
out of the document and records why, rather than aborting the run.  With
seventeen fields at varying stages of reprocessing that is the difference
between a partial document and no document.
"""

import argparse
import os
import sys
import traceback
import warnings
from datetime import datetime

from jwst_gc_pipeline.diagnostics import (astrometry_figs, background_figs,
                                          overview_figs, photometry_figs,
                                          project, writeup)
from jwst_gc_pipeline.diagnostics.inventory import inventory, known_fields
from jwst_gc_pipeline.version import __version__ as PIPELINE_VERSION

# (key, callable) in the order they appear in the document.
BUILDERS = (
    ('D1_overview', overview_figs.overview),
    ('D2_astrometry_internal', astrometry_figs.internal_astrometry),
    ('D3_astrometry_absolute', astrometry_figs.absolute_astrometry),
    ('D4_photometry_precision', photometry_figs.photometric_precision),
    ('D5_photometry_quality', photometry_figs.photometric_quality),
    ('D6_background_distributions', background_figs.background_distributions),
    ('D7_background_spatial', background_figs.background_spatial),
    ('D8_color_diagrams', photometry_figs.color_diagrams),
)


def build_field(fieldname, outdir=None, only=None, verbose=True):
    """Build every figure and the write-up for *fieldname*.

    Returns ``(inventory, results, failures, outdir)``.
    """
    inv = inventory(fieldname)
    outdir = outdir or os.path.join(inv.basepath, 'diagnostic_writeup')
    os.makedirs(outdir, exist_ok=True)

    results, failures = [], {}
    for key, builder in BUILDERS:
        if only and key not in only:
            continue
        if verbose:
            print(f'  [{fieldname}] {key} ...', flush=True)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                result = builder(inv, outdir)
        except (ValueError, KeyError, IndexError, TypeError, OSError,
                RuntimeError, MemoryError, ArithmeticError) as exc:
            failures[key] = f'{type(exc).__name__}: {exc}'
            if verbose:
                print(f'      failed: {failures[key]}', file=sys.stderr)
                traceback.print_exc(limit=3, file=sys.stderr)
            continue
        if result is None:
            failures[key] = 'no applicable data products'
            if verbose:
                print('      skipped (no applicable data products)')
            continue
        for w in caught:
            result.notes.append(str(w.message))
        results.append(result)
        if verbose:
            print(f'      -> {os.path.basename(result.path)}')

    doc = writeup.Writeup(inv, results, outdir)
    tex = doc.write()
    if failures:
        with open(os.path.join(outdir, 'BUILD_NOTES.md'), 'w') as fh:
            fh.write(f'# Build notes for {fieldname}\n\n')
            fh.write(f'Generated {datetime.now().isoformat(timespec="seconds")} '
                     f'with pipeline {PIPELINE_VERSION}.\n\n')
            fh.write('Figures not included in the document, and why:\n\n')
            for key, why in failures.items():
                fh.write(f'- `{key}`: {why}\n')
    return inv, results, failures, tex


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--field', help='field name from the registry')
    group.add_argument('--all', action='store_true',
                       help='every field in the registry')
    parser.add_argument('--outdir', default=None,
                        help='override the output directory (single field only)')
    parser.add_argument('--only', nargs='+', default=None,
                        help='build only these figure keys')
    parser.add_argument('--no-git', action='store_true',
                        help='do not create or update the git repository')
    parser.add_argument('--skip-empty', action='store_true',
                        help='with --all, skip fields that produced no figures')
    parser.add_argument('--list', action='store_true',
                        help='list the registry field names and exit')
    args = parser.parse_args(argv)

    if args.list:
        print('\n'.join(known_fields()))
        return 0

    targets = known_fields() if args.all else (args.field,)
    if args.outdir and len(targets) > 1:
        parser.error('--outdir applies to a single field only')

    stamp = datetime.now().isoformat(timespec='seconds')
    summary = []
    for fieldname in targets:
        print(f'== {fieldname}', flush=True)
        inv, results, failures, tex = build_field(
            fieldname, outdir=args.outdir, only=args.only)
        outdir = os.path.dirname(tex)
        if not results and args.skip_empty:
            print('   no figures; skipping git scaffolding')
            summary.append((fieldname, 0, len(failures), 'skipped'))
            continue
        status = 'git disabled'
        if not args.no_git:
            project.scaffold(outdir, fieldname, inv.basepath, stamp,
                             PIPELINE_VERSION)
            status = project.init_and_commit(
                outdir,
                f'diagnostic write-up for {fieldname}: {len(results)} figures '
                f'({stamp}, pipeline {PIPELINE_VERSION})')
        print(f'   {len(results)} figures, {len(failures)} skipped -> {outdir} '
              f'[{status}]')
        summary.append((fieldname, len(results), len(failures), status))

    if len(summary) > 1:
        print('\n== summary')
        total = sum(s[1] for s in summary)
        for name, nfig, nskip, status in summary:
            print(f'   {name:12s} {nfig:2d} figures  {nskip:2d} skipped  {status}')
        print(f'   {"TOTAL":12s} {total:2d} figures')
    return 0


if __name__ == '__main__':
    sys.exit(main())
