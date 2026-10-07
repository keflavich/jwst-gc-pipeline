"""A/B of the m7 cross-band merge, old code (mfs0) vs one-to-one satstar pairing (mfs1), scored against dolphot.
Needs matched_Q_s1t1_mfs0.fits / matched_Q_s1t1_mfs1.fits from compare_dolphot.py (see post.sh).
usage: python compare_ab.py [ARM0 ARM1]   (two existing analyze.py arms, for a smoke test)"""
import sys
import numpy as np
from astropy.coordinates import search_around_sky
import astropy.units as u
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an

Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
M7 = 'catalogs/basic_merged_indivexp_photometry_tables_merged_resbgsub_m7.fits'
for a in ('mfs0', 'mfs1'):
    an.PATH[a] = (f'{Q}/tree_{a}/{M7}', f'{an.D}/matched_Q_s1t1_{a}.fits')
if len(sys.argv) > 2:
    an.PATH['mfs0'], an.PATH['mfs1'] = an.PATH[sys.argv[1]], an.PATH[sys.argv[2]]
an.ZPWIN.update(an.zp_windows())
arms = {a: an.Arm(a) for a in ('mfs0', 'mfs1')}


def splits(A, R=0.08):
    cat = A.cat
    bands = [b for b in an.BANDS if f'mag_vega_f{b.lower()}' in cat.colnames]
    F = np.array([np.isfinite(np.asarray(an.fl(cat[f'mag_vega_f{b.lower()}']), float)) for b in bands]).T
    mrow = A.idx[A.matched]
    i1, i2, sep, _ = search_around_sky(A.sky[mrow], A.sky, R * u.arcsec)
    keep = mrow[i1] != i2
    i1, i2, sep = i1[keep], i2[keep], sep[keep]
    disj = ~(F[mrow[i1]] & F[i2]).any(axis=1)
    best = {}
    for a, b2, s in zip(i1[disj], i2[disj], sep[disj].to_value(u.mas)):
        if a not in best or s < best[a][1]:
            best[a] = (b2, s)
    return best, F


def twins(A, R=3):
    i1, i2, _, _ = search_around_sky(A.sky, A.sky, R * u.mas)
    return int((i1 < i2).sum())


L = ['| quantity | mfs0 (main 5434f5e7) | mfs1 (one-to-one) |', '|---|---|---|']
row = lambda name, f: L.append(f'| {name} | ' + ' | '.join(str(f(arms[a])) for a in ('mfs0', 'mfs1')) + ' |')
row('rows in m7 catalog', lambda A: len(A.cat))
row('row pairs within 3 mas', twins)
row('dolphot stars matched', lambda A: int(A.matched.sum()))
sp = {a: splits(arms[a]) for a in arms}
row('complementary split pairs (within 80 mas)', lambda A: len(sp[A.name][0]))
row('partner band values', lambda A: int(sum(sp[A.name][1][p].sum() for p, s in sp[A.name][0].values())))
row('band values on matched rows', lambda A: int(sum((A.matched & np.isfinite(A.our[b])).sum() for b in an.BANDS)))
print('\n'.join(L))

print('\n| band | ZP mfs0 | ZP mfs1 | matched with value mfs0 | mfs1 | bright no value mfs0 | mfs1 | '
      'replaced: N, med, f(abs dm>0.3) mfs0 | mfs1 |')
print('|---|---|---|---|---|---|---|---|---|')
for b in an.BANDS:
    cells = [f'{arms[a].zp[b]:+.3f}' for a in arms]
    cells += [str(int((arms[a].matched & np.isfinite(arms[a].our[b])).sum())) for a in arms]
    for a in arms:
        A = arms[a]
        lim = an.ZPWIN[b][0] if b in an.ZPWIN else 19
        cells.append(str(int((A.matched & np.isfinite(A.ref[b]) & (A.ref[b] < lim) & ~np.isfinite(A.our[b])).sum())))
    for a in arms:
        A = arms[a]
        d = A.dm(b)
        s = A.matched & np.isfinite(d) & A.rep[b]
        cells.append(f'{s.sum()}, {np.median(d[s]):+.3f}, {np.mean(np.abs(d[s]) > 0.3):.3f}' if s.sum() else '0')
    print(f'| F{b} | ' + ' | '.join(cells) + ' |')

# per dolphot star: value present in one arm only
A0, A1 = arms['mfs0'], arms['mfs1']
both = A0.matched & A1.matched
print(f'\ndolphot stars matched in both: {both.sum()}; matched only mfs0: {(A0.matched & ~A1.matched).sum()}; '
      f'only mfs1: {(A1.matched & ~A0.matched).sum()}')
g0 = g1 = n0 = n1 = 0
for b in an.BANDS:
    v0 = both & np.isfinite(A0.our[b]) & np.isfinite(A0.ref[b])
    v1 = both & np.isfinite(A1.our[b]) & np.isfinite(A1.ref[b])
    n0 += (v0 & ~v1).sum()
    n1 += (v1 & ~v0).sum()
    g0 += (v0 & ~v1 & (np.abs(A0.dm(b)) < 0.3)).sum()
    g1 += (v1 & ~v0 & (np.abs(A1.dm(b)) < 0.3)).sum()
    ch = v0 & v1 & (np.abs(A0.our[b] - A1.our[b]) > 0.01)
    if ch.sum():
        print(f'  F{b}: {ch.sum()} values change by > 0.01 mag; median abs dm mfs0 {np.median(np.abs(A0.dm(b)[ch])):.3f} '
              f'mfs1 {np.median(np.abs(A1.dm(b)[ch])):.3f}')
print(f'band values with a dolphot ref present only in mfs0: {n0} ({g0} within 0.3); only in mfs1: {n1} ({g1} within 0.3)')
