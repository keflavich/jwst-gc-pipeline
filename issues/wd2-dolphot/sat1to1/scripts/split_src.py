"""For complementary split pairs in a separation range: rows of the partner's base-creating band (vetted per-band
table) within 40 mas of the pair.  usage: python split_src.py ARM LO_mas HI_mas [N]"""
import sys
import numpy as np
from astropy.table import Table
from astropy.coordinates import search_around_sky, SkyCoord
import astropy.units as u
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an

arm, lo, hi = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
N = int(sys.argv[4]) if len(sys.argv) > 4 else 30
an.ZPWIN.update(an.zp_windows())
A = an.Arm(arm)
cat = A.cat
bands = [b for b in an.BANDS if f'mag_vega_f{b.lower()}' in cat.colnames]
F = np.array([np.isfinite(np.asarray(an.fl(cat[f'mag_vega_f{b.lower()}']), float)) for b in bands]).T
mrow = A.idx[A.matched]
i1, i2, sep, _ = search_around_sky(A.sky[mrow], A.sky, 0.08 * u.arcsec)
keep = mrow[i1] != i2
i1, i2, sep = i1[keep], i2[keep], sep[keep]
disj = ~(F[mrow[i1]] & F[i2]).any(axis=1)
i1, i2, sep = i1[disj], i2[disj], sep[disj].to_value(u.mas)
best = {}
for a, b2, s in zip(i1, i2, sep):
    if a not in best or s < best[a][1]:
        best[a] = (b2, s)
reff = np.asarray(cat['skycoord_ref_filtername']).astype(str)
T = f'/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/tree_{arm}/catalogs'
cache = {}
n = 0
for a, (p, s) in sorted(best.items(), key=lambda kv: kv[1][1]):
    if not (lo <= s < hi):
        continue
    r = mrow[a]
    for row, tag in ((r, 'row'), (p, 'partner')):
        bx = reff[row].lower()
        if bx not in cache:
            cache[bx] = Table.read(f'{T}/{bx}_merged_indivexp_merged_resbgsub_m7_dao_basic_vetted.fits')
    descs = []
    for row in (r, p):
        bx = reff[row].lower()
        t = cache[bx]
        c = SkyCoord(cat['skycoord_ref'][row])
        d = c.separation(t['skycoord']).to_value(u.mas)
        near = np.where(d < 40)[0]
        descs.append(f'{bx}: ' + '; '.join(
            f'{d[j]:.1f}mas rep={bool(t["replaced_saturated"][j])} msep={float(t["satstar_match_sep"][j])*1e3:.1f}'
            for j in near[np.argsort(d[near])]))
    ndup = sum(1 for x in descs if x.count('mas rep=True') >= 2 and x.split(': ')[1].split(';')[1].strip().startswith('0.0mas'))
    print(f'sep {s:5.1f} | dupband={ndup} | row  {descs[0]} | partner {descs[1]}')
    n += 1
    if n >= N:
        break
