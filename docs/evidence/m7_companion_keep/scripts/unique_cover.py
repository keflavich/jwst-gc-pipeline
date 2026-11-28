"""Distinct stars and distinct control leftovers covered by the far-companion exemption rows, both modules together.

usage: python unique_cover.py BUILD_OUTDIR LEFTOVERS_045.json LEFTOVERS_105.json OUT.json

BUILD_OUTDIR is the output directory of build_exempt_seed.py (the exempt_<module>.fits files of its o<T> subdirectories are the rows the first S/N floor
adds).  A leftover is covered when an exemption row lies within 0.072" (one FWHM).  Distinct positions: the exemption rows of both modules merged by
friends-of-friends at the same radius.
"""
import json
import sys

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.table import Table

BUILD, LEFT, OUT = sys.argv[1], {'045': sys.argv[2], '105': sys.argv[3]}, sys.argv[4]
R = 0.072 * u.arcsec
out = {}
for tile in ('045', '105'):
    left = json.load(open(LEFT[tile]))
    lsc = SkyCoord([r['ra'] for r in left] * u.deg, [r['dec'] for r in left] * u.deg)
    lres = np.array([r['resid_peak'] for r in left])
    rows = [SkyCoord(Table.read(f'{BUILD}/o{tile}/exempt_{mod}.fits')['skycoord']) for mod in ('nrca', 'nrcb')]
    n_mod = [len(r) for r in rows]
    allsc = SkyCoord(np.concatenate([r.ra.deg for r in rows]) * u.deg, np.concatenate([r.dec.deg for r in rows]) * u.deg)
    ia, ib, _, _ = search_around_sky(allsc, allsc, R)
    parent = np.arange(len(allsc))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in zip(ia, ib):
        ra_, rb_ = find(a), find(b)
        if ra_ != rb_:
            parent[ra_] = rb_
    n_star = len({find(i) for i in range(len(allsc))})
    il, ie, _, _ = search_around_sky(lsc, allsc, R)
    hit = np.zeros(len(left), dtype=bool)
    hit[il] = True
    res = dict(rows_by_module=n_mod, rows_total=int(sum(n_mod)), distinct_positions=int(n_star), leftovers_total=len(left))
    for thr in (50, 100, 200, 500):
        res[f'leftovers_ge{thr}'] = int((lres >= thr).sum())
        res[f'covered_ge{thr}'] = int((hit & (lres >= thr)).sum())
    out[tile] = res
    print(tile, json.dumps(res))
json.dump(out, open(OUT, 'w'), indent=1)
