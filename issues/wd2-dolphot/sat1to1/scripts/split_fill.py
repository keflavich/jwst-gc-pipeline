"""How many matched dolphot stars lacking a band value in their matched row have that value in a complementary
split partner (row within R, disjoint finite bands).  Bright = brighter than the band's bright limit used in #1032
(dolphot mag < ZPWIN lower edge = satstar-replaced p95 + 1).  usage: python split_fill.py ARM [R]"""
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
mk = np.where(A.matched)[0]
mrow = A.idx[mk]
i1, i2, sep, _ = search_around_sky(A.sky[mrow], A.sky, R * u.arcsec)
keep = mrow[i1] != i2
i1, i2 = i1[keep], i2[keep]
disj = ~(F[mrow[i1]] & F[i2]).any(axis=1)
i1, i2 = i1[disj], i2[disj]
part = {}
for a, b2 in zip(i1, i2):
    part.setdefault(a, []).append(b2)
print(f'{arm}: matched stars with a complementary partner: {len(part)}')
print('| band | bright limit | bright matched without value | of which partner has value | all matched without value | of which partner has value |')
print('|---|---|---|---|---|---|')
for jj, b in enumerate(bands):
    lim = an.ZPWIN[b][0] if b in an.ZPWIN else 19
    ref = A.ref[b][mk]
    miss = np.isfinite(ref) & ~F[mrow, jj]
    br = miss & (ref < lim)
    fill = np.array([any(F[p, jj] for p in part.get(a, [])) for a in range(len(mk))])
    print(f'| F{b} | {lim:.1f} | {br.sum()} | {(br & fill).sum()} | {miss.sum()} | {(miss & fill).sum()} |')
