"""How replace_saturated paired satstars with merged-catalog rows, per variant.

For every reference run (field x seed x band) and phase m3..m7, the merged
basic catalog's ``replaced_saturated`` rows fall in three groups:

* tight   -- paired at the per-filter tight radius of replace_saturated
             (satstar_match_sep <= 0.05" short-wave, <= 0.1" long-wave)
* second  -- paired by the mutual-nearest second pass (satstar_match_sep
             beyond the tight radius).
             The overwritten row keeps its own ``flux_init`` (the seed flux of the
             star that row was fitting), so lr = log10(flux_init / satstar flux)
             tells whose row it was: |lr| < 0.2 'self' (the satstar's own,
             clipped row), lr < -0.5 'nbr' (a row seeded > 3x fainter: a
             neighbouring star), else 'mid'.
* appended -- satstar_match_sep NaN: the satstar had no partner and was added
             as a row of its own.  'dup' counts appended rows with a
             non-replaced row within 0.5" whose flux_init is within 0.2 dex of
             the satstar flux (its own clipped row left unpaired: a duplicate).

usage: python swap_census.py <variant> <out.fits>
"""
import glob
import re
import sys

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from jwst_gc_pipeline.photometry import reference_fields as RF

PH = ['m3', 'm4', 'resbgsub_m5', 'resbgsub_m6', 'resbgsub_m7']


def tight_radius(filt):
    """replace_saturated's first-pass radius: 0.05\" short-wave, 0.1\" long-wave
    NIRCam (with a small margin for float round-off)."""
    return 0.1001 if int(re.sub(r'\D', '', filt)) >= 250 else 0.0501


def census(m, tight):
    rep = np.asarray(m['replaced_saturated'], bool)
    # replace_saturated adds satstar_match_sep only when it paired a row;
    # without it every replaced row is an appended satstar
    s = (np.asarray(m['satstar_match_sep'], float) if 'satstar_match_sep' in m.colnames
         else np.full(len(m), np.nan))
    fl = np.asarray(m['flux'], float)
    fi = np.asarray(m['flux_init'], float) if 'flux_init' in m.colnames else np.full(len(m), np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        lr = np.log10(np.where((fi > 0) & (fl > 0), fi / fl, np.nan))
    sec = rep & np.isfinite(s) & (s > tight)
    app = rep & ~np.isfinite(s)
    out = dict(n_rows=len(m), tight=int((rep & np.isfinite(s) & (s <= tight)).sum()),
               second_self=int((sec & (np.abs(lr) < 0.2)).sum()),
               second_nbr=int((sec & (lr < -0.5)).sum()),
               second_mid=int((sec & ~(np.abs(lr) < 0.2) & ~(lr < -0.5)).sum()),
               second_nbr_sep_max=float(np.nanmax(s[sec & (lr < -0.5)])) if (sec & (lr < -0.5)).any() else np.nan,
               appended=int(app.sum()), dup=0)
    if app.any() and (~rep).any():
        sc = SkyCoord(m['skycoord'])
        ok = np.isfinite(sc.ra.deg) & np.isfinite(sc.dec.deg)
        other = np.flatnonzero(~rep & ok)
        ia = np.flatnonzero(app & ok)
        if len(other) and len(ia):
            i_o, i_a, _, _ = sc[ia].search_around_sky(sc[other], 0.5 * u.arcsec)
            dup = set()
            for jo, ja in zip(i_o, i_a):
                o, a = other[jo], ia[ja]
                if fi[o] > 0 and fl[a] > 0 and abs(np.log10(fi[o] / fl[a])) < 0.2:
                    dup.add(a)
            out['dup'] = len(dup)
    return out


def main(variant, outpath):
    _, fields = RF.load_config()
    rows = []
    for name, spec in fields.items():
        for seed in [0] + list(spec['seeds']):
            d = RF.run_dir(spec, variant, seed)
            for filt in spec['filters']:
                f = filt.lower()
                for ph in PH:
                    hits = glob.glob(f'{d}/catalogs/{f}_merged*_indivexp_merged_{ph}_dao_basic.fits')
                    if len(hits) != 1:
                        continue
                    m = Table.read(hits[0])
                    if 'replaced_saturated' not in m.colnames:
                        continue
                    rows.append(dict(field=name, variant=variant, seed=seed, filt=f, phase=ph, **census(m, tight_radius(f))))
    t = Table(rows=rows)
    t.write(outpath, overwrite=True)
    keys = ('tight', 'second_self', 'second_nbr', 'second_mid', 'appended', 'dup')
    print(f'variant {variant}: sums over seeds and phases m3..m7')
    print('| field / band | runs | ' + ' | '.join(keys) + ' | max nbr sep (") |')
    print('|---' * (len(keys) + 3) + '|')
    for name in dict.fromkeys(np.asarray(t['field']).astype(str)):
        for f in dict.fromkeys(np.asarray(t['filt'][np.asarray(t['field']).astype(str) == name]).astype(str)):
            b = t[(np.asarray(t['field']).astype(str) == name) & (np.asarray(t['filt']).astype(str) == f)]
            nr = len(set(np.asarray(b['seed']).tolist()))
            mx = np.nanmax(b['second_nbr_sep_max']) if np.isfinite(b['second_nbr_sep_max']).any() else np.nan
            print(f'| {name} {f.upper()} | {nr} | ' + ' | '.join(str(int(np.sum(b[k]))) for k in keys)
                  + f' | {mx:.3f} |')


if __name__ == '__main__':
    main(*sys.argv[1:])
