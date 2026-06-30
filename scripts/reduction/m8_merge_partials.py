#!/usr/bin/env python3
"""Column-merge per-filter m8 partials into the combined m8 catalog (+ dedup).

The m8 forced cross-band fill (``--manual-start-phase=m8``) otherwise sweeps
every frame serially in one job and can overrun the wall (on sickle: F187N+F210M
alone = 192 frames, F470N/F480M never run).  Instead fan it out into one
per-filter job each run with ``--manual-m8-partial`` (see
``submit_cataloging_m8.sh``), so a job fills ONLY its band and writes
``..._resbgsub_m8_partial_<FILT>.fits``.  This script overlays those band columns
onto the m7 anchor to build ``..._resbgsub_m8.fits`` and then de-duplicates it.

Why an index-aligned overlay is safe: every partial is produced from the SAME m7
merged catalog, so all partials share row count + order; ``forced_fill_band``
writes back only band-specific columns (any colname containing the band's
lowercase token: ``flux_<f>``, ``mag_vega_<f>``, ``forced_filled_<f>`` ...).  The
band tokens (e.g. f187n/f210m/f335m/f470n/f480m) are mutually non-substring, so
"colname contains token" cleanly selects one band and never a shared column
(id, skycoord_ref, flags, ...).

SAFE: refuses to write a half-merged m8 if any partial is missing (it would look
complete but silently fall a band back to m7).  Pass ``--allow-missing`` to
override.  Dry-run by default; pass ``--execute`` to write.

Generic over field: pass the explicit ``--m7`` anchor path, or
``--catdir``+``--module`` to derive the default
``basic_<module>_indivexp_photometry_tables_merged_resbgsub_m7.fits``.
"""
import argparse
import os
import sys

import numpy as np
from astropy.table import Table


def _default_m7(catdir, module):
    return os.path.join(
        catdir,
        f'basic_{module}_indivexp_photometry_tables_merged_resbgsub_m7.fits')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--m7', default=None,
                    help='m7 merged catalog (anchor for row order + shared cols). '
                         'If omitted, derived from --catdir + --module.')
    ap.add_argument('--catdir', default=None,
                    help='catalogs/ dir (used with --module to derive --m7)')
    ap.add_argument('--module', default='nrcb',
                    help='module token for the derived --m7 (default nrcb)')
    ap.add_argument('--filters', required=True,
                    help='comma-separated filters to overlay, e.g. '
                         'F187N,F210M,F335M,F470N,F480M')
    ap.add_argument('--allow-missing', action='store_true',
                    help='write the merge even if some partials are absent (those '
                         'bands keep m7 values)')
    ap.add_argument('--no-dedup', dest='dedup', action='store_false', default=True,
                    help='skip the post-merge split-source de-duplication')
    ap.add_argument('--link-radius-arcsec', type=float, default=0.10,
                    help='dedup link radius (arcsec, default 0.10)')
    ap.add_argument('--execute', action='store_true',
                    help='actually write the combined m8 (default: dry-run report)')
    args = ap.parse_args(argv)

    if args.m7 is None:
        if not args.catdir:
            ap.error('provide --m7, or --catdir (+ --module) to derive it')
        args.m7 = _default_m7(args.catdir, args.module)
    if not os.path.exists(args.m7):
        ap.error(f'm7 anchor not found: {args.m7}')

    filters = [f.strip() for f in args.filters.split(',') if f.strip()]
    out_path = args.m7.replace('resbgsub_m7', 'resbgsub_m8')

    base = Table.read(args.m7)
    n = len(base)
    print(f"m7 anchor: {os.path.basename(args.m7)}  ({n} rows, {len(base.colnames)} cols)")

    missing = []
    plan = []  # (filt, partial_table, [cols])
    for filt in filters:
        tok = filt.lower()
        ppath = args.m7.replace('resbgsub_m7', f'resbgsub_m8_partial_{filt}')
        if not os.path.exists(ppath):
            missing.append(filt)
            print(f"  {filt}: MISSING partial {os.path.basename(ppath)}")
            continue
        pt = Table.read(ppath)
        if len(pt) != n:
            print(f"  {filt}: ROW COUNT MISMATCH {len(pt)} != {n} -> ABORT",
                  file=sys.stderr)
            sys.exit(2)
        cols = [c for c in pt.colnames if tok in c]
        nfill = int(np.asarray(pt[f'forced_filled_{tok}']).sum()) \
            if f'forced_filled_{tok}' in pt.colnames else -1
        print(f"  {filt}: {len(cols)} band cols, forced_filled={nfill}  "
              f"<- {os.path.basename(ppath)}")
        plan.append((filt, pt, cols))

    if missing and not args.allow_missing:
        print(f"\nERROR: {len(missing)} partial(s) missing ({','.join(missing)}); "
              f"refusing to write a half-merged m8.  Re-run those bands or pass "
              f"--allow-missing.", file=sys.stderr)
        sys.exit(1)

    if not args.execute:
        print(f"\n[DRY] would overlay {sum(len(c) for _, _, c in plan)} band cols "
              f"from {len(plan)} partials -> {os.path.basename(out_path)}"
              f"{' + dedup' if args.dedup else ''}  (re-run with --execute)")
        return

    for filt, pt, cols in plan:
        for c in cols:
            base[c] = pt[c]
    base.write(out_path, overwrite=True)
    print(f"\nWROTE {out_path}  ({n} rows, {len(base.colnames)} cols, "
          f"{len(plan)}/{len(filters)} bands overlaid"
          f"{'; MISSING ' + ','.join(missing) if missing else ''})")

    if args.dedup:
        import astropy.units as u
        from jwst_gc_pipeline.photometry.dedup_catalog import dedup_merged_catalog
        dedup_path = out_path.replace('_m8.fits', '_m8_dedup.fits')
        dedup_merged_catalog(out_path, dedup_path,
                             link_radius=args.link_radius_arcsec * u.arcsec)
        print(f"WROTE {dedup_path}  (de-duplicated; science-final)")


if __name__ == '__main__':
    main()
