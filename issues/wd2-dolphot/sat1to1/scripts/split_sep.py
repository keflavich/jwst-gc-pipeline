"""Complementary split pairs (disjoint finite bands, one row dolphot-matched) by separation, with satstar status.
usage: python split_sep.py ARM [R]"""
import sys
import numpy as np
from astropy.coordinates import search_around_sky
import astropy.units as u
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an

arm = sys.argv[1]
R = float(sys.argv[2]) if len(sys.argv) > 2 else 0.08
an.ZPWIN.update(an.zp_windows())
A = an.Arm(arm)
cat = A.cat
bands = [b for b in an.BANDS if f'mag_vega_f{b.lower()}' in cat.colnames]
F = np.array([np.isfinite(np.asarray(an.fl(cat[f'mag_vega_f{b.lower()}']), float)) for b in bands]).T
REP = np.array([np.ma.filled(cat[f'replaced_saturated_f{b.lower()}'], False).astype(bool)
                if f'replaced_saturated_f{b.lower()}' in cat.colnames else np.zeros(len(cat), bool) for b in bands]).T
mrow = A.idx[A.matched]
i1, i2, sep, _ = search_around_sky(A.sky[mrow], A.sky, R * u.arcsec)
keep = mrow[i1] != i2
i1, i2, sep = i1[keep], i2[keep], sep[keep]
disj = ~(F[mrow[i1]] & F[i2]).any(axis=1)
i1, i2, sep = i1[disj], i2[disj], sep[disj].to_value(u.mas)
best = {}
for a, b2, s in zip(i1, i2, sep):
    if a not in best or s < best[a][1]:
        best[a] = (b2, s)
reff = np.asarray(cat['skycoord_ref_filtername']).astype(str)
rows = [(mrow[a], p, s) for a, (p, s) in best.items()]
for lo, hi in [(0, 3), (3, 20), (20, 50), (50, 80)]:
    sel = [(r, p, s) for r, p, s in rows if lo <= s < hi]
    anyrep = sum(1 for r, p, s in sel if REP[r].any() or REP[p].any())
    nvals = sum(int(F[p].sum()) for r, p, s in sel)
    print(f'{arm} {lo:2d}-{hi:2d} mas: {len(sel)} pairs; with a satstar-replaced band in either row: {anyrep}; '
          f'partner band values: {nvals}')
