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
import os
import subprocess
from astropy.table import Table


def _git_commit():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        return subprocess.check_output(['git', '-C', here, 'rev-parse', 'HEAD'],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def make_overlay_json(pm_fits, out_path, name, mag_key='mag_instr',
                      mag_col='mag_src', verbose=True, frame='ref',
                      ref_observations=None):
    """``frame='ref'`` places each source at its Treasury-frame (current-epoch)
    position (ra_ref/dec_ref) so it lands on the star in the Treasury HiPS;
    ``frame='src'`` uses ra0/dec0 (src frame after the affine tie).
    ``ref_observations`` overrides the Treasury ref catalog names, which
    otherwise come from the PM FITS (``ref_paths_json``, absent in files built
    before it was recorded).  Both end up in ``meta`` with the builder commit."""
    t = Table.read(pm_fits)
    ra_col, dec_col = 'ra0', 'dec0'
    if frame == 'ref':
        if 'ra_ref' not in t.colnames:
            raise ValueError(f'{pm_fits} has no ra_ref/dec_ref; rebuild with the '
                             f'raw-ref-position build_treasury_pm or pass frame="src"')
        ra_col, dec_col = 'ra_ref', 'dec_ref'
    if ref_observations is None and 'ref_paths_json' in t.meta:
        ref_observations = json.loads(t.meta['ref_paths_json'])
    t = t[t['trustworthy']]
    if verbose:
        print(f'{len(t):,} trustworthy PMs -> {out_path}')

    sources = [
        {
            'ra': float(row[ra_col]), 'dec': float(row[dec_col]),
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
            'position_frame': 'treasury-ref-epoch' if frame == 'ref' else 'src-tied',
            'n_trustworthy': len(t),
            'ref_observations': ref_observations,
            'builder_commit': _git_commit(),
        },
        'sources': sources,
    }
    # allow_nan=False: a non-finite value would otherwise emit a bare
    # NaN/Infinity token that JSON.parse rejects, taking out the whole
    # overlay for one bad source instead of failing here, at build time,
    # where it can be traced back to which row produced it. Serialize to a
    # string FIRST so that raise happens before anything is touched on disk
    # -- writing straight into out_path with 'w' truncates immediately, so a
    # NaN partway through would leave a truncated fragment at the live
    # overlay's path instead of failing cleanly, replacing a working file
    # with a broken one exactly as noisily as it was trying not to.
    payload = json.dumps(doc, allow_nan=False)
    tmp_path = out_path + '.tmp'
    with open(tmp_path, 'w') as f:
        f.write(payload)
    os.replace(tmp_path, out_path)
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
    ap.add_argument('--frame', choices=['ref', 'src'], default='ref',
                    help="position frame for ra/dec: 'ref' = Treasury frame at the "
                         "ref epoch (matches the HiPS), 'src' = src-tied ra0/dec0")
    ap.add_argument('--ref-observations', nargs='+', default=None,
                    help='Treasury ref catalog names to record in meta '
                         '(default: from the PM FITS, if it recorded them)')
    args = ap.parse_args()
    make_overlay_json(args.pm_fits, args.out, args.name,
                      mag_key=args.mag_key, mag_col=args.mag_col, frame=args.frame,
                      ref_observations=args.ref_observations)


if __name__ == '__main__':
    main()
