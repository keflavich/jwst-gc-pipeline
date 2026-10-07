"""Compare the F212N per-band merge of main (pb0) and PR #1122 (pb1)."""
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

H = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
FN = 'catalogs/f212n_merged_indivexp_merged_resbgsub_m7_dao_basic.fits'
cats = {'prod (bd392ab8)': f'{H}/tree_mainfcbg/{FN}',
        'pb0 main 5434f5e7': f'{H}/sat1to1/pb/tree_pb0/{FN}',
        'pb1 PR #1122': f'{H}/sat1to1/pb/tree_pb1/{FN}'}


def dup_pairs(t, tol=1 * u.mas):
    c = t['skycoord']
    i, j, sep, _ = c.search_around_sky(c, tol)
    keep = i < j
    return i[keep], j[keep]


T = {}
print('| catalog | rows | replaced_saturated | duplicate pairs (<1 mas) | both replaced |')
print('|---|---|---|---|---|')
for k, fn in cats.items():
    t = Table.read(fn)
    T[k] = t
    i, j = dup_pairs(t)
    rs = np.asarray(t['replaced_saturated']).astype(bool) if 'replaced_saturated' in t.colnames else np.zeros(len(t), bool)
    print(f'| {k} | {len(t)} | {rs.sum()} | {len(i)} | {(rs[i] & rs[j]).sum()} |')

a, b = T['pb0 main 5434f5e7'], T['pb1 PR #1122']
ca, cb = a['skycoord'], b['skycoord']
idx, sep, _ = ca.match_to_catalog_sky(cb)
gone = sep > 1 * u.mas
print()
print(f'pb0 rows with no pb1 row within 1 mas: {gone.sum()}')
ia, ja = dup_pairs(a)
dup_rows = set(ia) | set(ja)
print(f'  of them in a pb0 duplicate pair: {sum(1 for x in np.where(gone)[0] if x in dup_rows)}')
idx2, sep2, _ = cb.match_to_catalog_sky(ca)
print(f'pb1 rows with no pb0 row within 1 mas: {(sep2 > 1 * u.mas).sum()}')
m = ~gone
fa = np.asarray(a['flux'])[m]
fb = np.asarray(b['flux'])[idx[m]]
same = (fa == fb) | (np.isnan(fa) & np.isnan(fb))
print(f'matched rows with identical flux: {same.sum()} of {m.sum()}')
if (~same).sum():
    d = -2.5 * np.log10(fb[~same] / fa[~same])
    print('  flux-changed rows dmag p0/p50/p100:', np.nanpercentile(d, [0, 50, 100]))
