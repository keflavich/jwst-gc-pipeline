"""The own-band m6 rows under the 11 companion-cut leftover stars: flux, S/N and the dominating neighbours that cut them.

usage: python target_table.py LEFTOVERS_045.json LEFTOVERS_105.json [--root /orange/adamginsburg/jwst/gc-treasury]

The 11 targets are the stars of the control m7 residual (gc-treasury 10678 F212N, o045 5 and o105 6; (x, y) = mosaic pixels of the leftover list) that the
one-frame replays of the production m7 fit trace to the own-band companion cut: the star is an own-band m6 vetted row within 2.5 FWHM of a row more than 10x
brighter, so the m7 seed drops it.  For each target the row of either module within 0.05" of the leftover, and every row within 2.5 FWHM that is more
than 10x brighter: separation in FWHM, flux ratio, saturation flag.  The last column applies the exemption rule at S/N 50 and 1.75 FWHM.
"""
import json
import sys

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.table import Table

FW = 0.0720      # F212N PSF FWHM, arcsec (jwst_gc_pipeline/reduction/fwhm_table.ecsv)
TGT = {'045': [(6277, 2642), (6355, 3168), (4979, 3391), (7311, 2154), (6092, 3774)],
       '105': [(5188, 2780), (6137, 3868), (6588, 1271), (7097, 961), (11328, 2324), (9794, 833)]}
args = sys.argv[1:]
ROOT = '/orange/adamginsburg/jwst/gc-treasury'
if '--root' in args:
    i = args.index('--root')
    ROOT = args[i + 1]
    del args[i:i + 2]
LEFT = {'045': args[0], '105': args[1]}
CAT = f'{ROOT}/catalogs'
for tile in ('045', '105'):
    by = {(r['x'], r['y']): r for r in json.load(open(LEFT[tile]))}
    for mod in ('nrca', 'nrcb'):
        t = Table.read(f'{CAT}/f212n_{mod}_o{tile}_indivexp_merged_resbgsub_m6_dao_basic_vetted.fits')
        fl = np.asarray(t['flux'], float)
        t = t[np.isfinite(fl) & (fl > 0)]
        sc = t['skycoord']
        fl = np.asarray(t['flux'], float)
        fe = np.asarray(t['flux_err'], float)
        sat = np.asarray(t['is_saturated']) | np.asarray(t['replaced_saturated'])
        ia, ib, sep, _ = search_around_sky(sc, sc, 2.5 * FW * u.arcsec)
        m = (ia != ib) & (fl[ib] > fl[ia] / 0.1)
        ia, ib, sep = ia[m], ib[m], sep.arcsec[m] / FW
        for k in TGT[tile]:
            r = by[k]
            c = SkyCoord(r['ra'] * u.deg, r['dec'] * u.deg)
            d = c.separation(sc).arcsec
            hit = np.flatnonzero(d < 0.05)
            for h in hit:
                nb = ib[ia == h]
                nbs = sep[ia == h]
                desc = '; '.join(f'{s:.2f} FWHM, ratio {fl[h] / fl[b]:.3f}, {"SAT" if sat[b] else "unsat"}' for b, s in zip(nb, nbs)) or 'no dominating row in this module'
                snr = fl[h] / fe[h] if fe[h] > 0 else float('nan')
                keep = bool(len(nb)) and snr >= 50 and bool(np.all(nbs >= 1.75)) and not bool(np.any(sat[nb]))
                print(f'o{tile} {k} {mod}: row {d[h]:.3f}" from the leftover, flux {fl[h]:.0f}, S/N {snr:.0f}; dominated by: {desc}; exempt at S/N 50, 1.75 FWHM: {keep}')
