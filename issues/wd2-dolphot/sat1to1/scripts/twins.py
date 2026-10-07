"""Twin rows in an arm's m8 catalog: pairs of rows within R mas.  Reports how many, how many have disjoint
finite bands, and how many are matched to a dolphot star.  usage: python twins.py ARM [R_mas]"""
import sys
import numpy as np
from astropy.coordinates import search_around_sky
import astropy.units as u
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an

arm = sys.argv[1]
R = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
an.ZPWIN.update(an.zp_windows())
A = an.Arm(arm)
cat = A.cat
bands = [b for b in an.BANDS if f'mag_vega_f{b.lower()}' in cat.colnames]
F = np.array([np.isfinite(np.asarray(an.fl(cat[f'mag_vega_f{b.lower()}']), float)) for b in bands]).T
i1, i2, s, _ = search_around_sky(A.sky, A.sky, R * u.mas)
k = i1 < i2
i1, i2, s = i1[k], i2[k], s[k]
disj = ~(F[i1] & F[i2]).any(axis=1)
mrows = set(A.idx[A.matched].tolist())
mt = np.array([(a in mrows) or (b in mrows) for a, b in zip(i1, i2)], bool)
print(f'{arm}: {len(cat)} rows; pairs within {R} mas: {len(i1)}; disjoint bands: {disj.sum()}; '
      f'dolphot-matched: {mt.sum()}; disjoint & matched: {(disj & mt).sum()}')
nb1 = F[i1].sum(1); nb2 = F[i2].sum(1)
print('finite bands per twin (median row1/row2):', np.median(nb1), np.median(nb2))
np.save(f'/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/sat1to1/twins_{arm}.npy', np.array([i1, i2]))
