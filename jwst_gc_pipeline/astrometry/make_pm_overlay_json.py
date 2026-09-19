#!/usr/bin/env python
"""Convert a build_treasury_pm output FITS catalog into the JSON overlay
consumed by jwst_gc_aladin.html (same 'sources: [{ra, dec, ...}]' shape as
the other CATALOGS entries there; see addCatalog() in that page).

Replaces two near-identical scratch copies (sgrb2/make_pm_overlay_json.py,
cloudef/make_pm_overlay_json.py) that existed only as ad hoc per-target
scripts outside version control.

Example:
    python -m jwst_gc_pipeline.astrometry.make_pm_overlay_json \\
        --pm-fits /orange/.../pm_sgrb2_treasury_f212n_affinetied.fits \\
        --out /orange/adamginsburg/web/public/avm_images/jwst_gc_cat_sgrb2_treasury_pm.json \\
        --name "Sgr B2 x GC Treasury proper motions" \\
        --mag-key mag_f212n_instr
"""
import argparse
import json
from astropy.table import Table


def make_overlay_json(pm_fits, out_path, name, mag_key='mag_instr',
                      mag_col='mag_src', verbose=True):
    t = Table.read(pm_fits)
    t = t[t['trustworthy']]
    if verbose:
        print(f'{len(t):,} trustworthy PMs -> {out_path}')

    sources = [
        {
            'ra': float(row['ra0']), 'dec': float(row['dec0']),
            'pm_ra': round(float(row['pm_ra']), 3),
            'pm_dec': round(float(row['pm_dec']), 3),
            'pm_tot': round(float(row['pm_tot']), 3),
            'pm_ra_err': round(float(row['pm_ra_err']), 3),
            'pm_dec_err': round(float(row['pm_dec_err']), 3),
            mag_key: round(float(row[mag_col]), 2),
        }
        for row in t
    ]
    doc = {
        'name': name,
        'meta': {
            'epochs': list(t.meta.get('EPOCHS', [])),
            'baseline_yr': float(t.meta.get('baseline_yr', 0)) or None,
            'frame': t.meta.get('FRAME', ''),
            'n_trustworthy': len(t),
        },
        'sources': sources,
    }
    with open(out_path, 'w') as f:
        # allow_nan=False: a non-finite value would otherwise emit a bare
        # NaN/Infinity token that JSON.parse rejects, taking out the whole
        # overlay for one bad source instead of failing here, at build time,
        # where it can be traced back to which row produced it.
        json.dump(doc, f, allow_nan=False)
    if verbose:
        print('wrote', out_path)
    return doc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pm-fits', required=True, help='build_treasury_pm output FITS')
    ap.add_argument('--out', required=True, help='output JSON path')
    ap.add_argument('--name', required=True, help='overlay display name')
    ap.add_argument('--mag-col', default='mag_src',
                    help="input FITS column to read the instrumental "
                         "magnitude from (build_treasury_pm always names it "
                         "mag_src; a differently-shaped PM catalog could not)")
    ap.add_argument('--mag-key', default='mag_instr',
                    help="JSON property name for the source's instrumental "
                         "magnitude, e.g. mag_f212n_instr")
    args = ap.parse_args()
    make_overlay_json(args.pm_fits, args.out, args.name,
                      mag_key=args.mag_key, mag_col=args.mag_col)


if __name__ == '__main__':
    main()
