"""Own-band rows the companion cut removes on gc-treasury 10678 F212N, o045 and o105: how many a far-companion exemption would return.

usage: python companion_keep_table.py LEFTOVERS_045.json LEFTOVERS_105.json [--root /orange/adamginsburg/jwst/gc-treasury]

Catalog-level only (no seed builder, no fit): for every own-band m6 vetted row of both modules that has a source more than 10x brighter within 2.5 FWHM
(the dominated rows the cut removes), the table gives the S/N (flux / flux_err), the flux ratio to and the separation (in FWHM) from its
brightest dominating row, and whether that row is saturated (``is_saturated`` or ``replaced_saturated``).  Rows are counted per S/N floor, with
the leftover matches of the control m7 residual (within 0.072", one FWHM) at resid_peak >= 50 / 100 / 200 MJy/sr in parentheses.  The pool of
dominating rows here is the own-band catalog alone; the builder's pool also holds the cross-band positions, so the counts differ slightly from the
builder's (build_exempt_seed.py gives those).
"""
import json
import sys

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.table import Table

FW = 0.0720      # F212N PSF FWHM, arcsec (jwst_gc_pipeline/reduction/fwhm_table.ecsv)
args = sys.argv[1:]
ROOT = '/orange/adamginsburg/jwst/gc-treasury'
if '--root' in args:
    i = args.index('--root')
    ROOT = args[i + 1]
    del args[i:i + 2]
LEFT = {'045': args[0], '105': args[1]}
CAT = f'{ROOT}/catalogs'
R = {k: [] for k in ('tile', 'flux', 'snr', 'ratio', 'sep', 'nbsat', 'resid')}
for tile in ('045', '105'):
    left = json.load(open(LEFT[tile]))
    lsc = SkyCoord([r['ra'] for r in left] * u.deg, [r['dec'] for r in left] * u.deg)
    lres = np.array([r['resid_peak'] for r in left])
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
        ia, ib, sep = ia[m], ib[m], sep.arcsec[m]
        best = {}
        for a, b, s in zip(ia, ib, sep):
            if a not in best or fl[b] > fl[best[a][0]]:
                best[a] = (b, s)
        idx = np.array(sorted(best))
        nb = np.array([best[i][0] for i in idx])
        ss = np.array([best[i][1] for i in idx])
        idl, dl, _ = sc[idx].match_to_catalog_sky(lsc)
        on = dl.arcsec < 0.072
        R['tile'] += [tile] * len(idx)
        R['flux'] += list(fl[idx])
        R['snr'] += list(fl[idx] / fe[idx])
        R['ratio'] += list(fl[idx] / fl[nb])
        R['sep'] += list(ss / FW)
        R['nbsat'] += list(sat[nb])
        R['resid'] += list(np.where(on, lres[idl], 0.0))
A = {k: np.array(v) for k, v in R.items()}


def lo(sel):
    return '/'.join(f'{int((sel & (A["resid"] >= thr)).sum()):3d}' for thr in (50, 100, 200))


for tile in ('045', '105'):
    s0 = A['tile'] == tile
    print(f'== o{tile}: cut rows {int(s0.sum())}; unsaturated brightest neighbour {int((s0 & ~A["nbsat"]).sum())}')
    print('   S/N floor | saturated neighbour: rows (leftover >=50/100/200) | unsaturated neighbour: rows (leftover >=50/100/200)'
          '   [separation >= 1.75 FWHM, ratio >= 0.03]')
    for snr in (20, 30, 40, 50, 60, 80, 100):
        base = s0 & (A['snr'] >= snr) & (A['sep'] >= 1.75) & (A['ratio'] >= 0.03)
        cells = [f"{int(sel.sum()):5d} ({lo(sel)})" for sel in (base & A['nbsat'], base & ~A['nbsat'])]
        print(f'   {snr:5d}     | {cells[0]:34s} | {cells[1]}')
    for sep in (1.5, 1.75, 2.0):
        base = s0 & (A['snr'] >= 50) & (A['sep'] >= sep) & (A['ratio'] >= 0.03) & ~A['nbsat']
        print(f'   unsaturated, S/N >= 50, separation >= {sep}: {int(base.sum())} rows; leftover >=50/100/200: {lo(base)}')
    base = s0 & (A['snr'] >= 50) & (A['sep'] >= 1.75) & ~A['nbsat']
    for rmin in (0.0, 0.01, 0.02, 0.03, 0.05):
        b = base & (A['ratio'] >= rmin)
        print(f'   unsaturated, S/N >= 50, separation >= 1.75, ratio >= {rmin}: {int(b.sum())} rows; leftover >=50/100/200: {lo(b)}')
