"""Lost-star detail (only-A matches) and gone/new row census by distance to saturated stars.
usage: lost2.py A B"""
import sys
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
D = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark'
M8 = 'catalogs/basic_merged_indivexp_photometry_tables_merged_resbgsub_m8_dedup.fits'
A, B = sys.argv[1:3]
ma = Table.read(f'{D}/matched_Q_{A}.fits')
ca, cb = Table.read(f'{D}/Q_integ/tree_{A}/{M8}'), Table.read(f'{D}/Q_integ/tree_{B}/{M8}')
L = np.load(f'{D}/Q_integ/kf_lost/lost_{A}_{B}.npy')
ia = np.asarray(ma['our_idx'])
dm = []
for i in L:
    r = ca[ia[i]]
    f = r['skycoord_ref_filtername'].decode() if isinstance(r['skycoord_ref_filtername'], bytes) else str(r['skycoord_ref_filtername'])
    key = f[1:].upper()
    refm = float(ma[f'ref_{key}'][i]) if f'ref_{key}' in ma.colnames else np.nan
    ourm = float(r[f'mag_vega_{f}'])
    dm.append((f, refm, ourm, float(r[f'qfit_{f}']), int(r['n_real_bands'])))
dm = np.array([(x[1], x[2], x[3]) for x in dm])
d = dm[:, 1] - dm[:, 0]
okd = np.isfinite(d) & (dm[:, 0] < 90)
print(f'lost stars: detection-band dm (ours - dolphot) N={okd.sum()} median {np.median(d[okd]):+.2f}; '
      f'|dm|<0.3: {(np.abs(d[okd]) < 0.3).sum()}, 0.3-1: {((np.abs(d[okd]) >= 0.3) & (np.abs(d[okd]) < 1)).sum()}, >=1: {(np.abs(d[okd]) >= 1).sum()}; '
      f'dolphot has no value in that band: {(~okd).sum()}')
print('ref mag in detection band:', np.histogram(dm[okd, 0], [0, 15, 17, 19, 21, 23, 30])[0])
# gone/new rows
sa, sb = SkyCoord(ca['skycoord_ref']), SkyCoord(cb['skycoord_ref'])
ib, d_ab, _ = sa.match_to_catalog_sky(sb)
ia2, d_ba, _ = sb.match_to_catalog_sky(sa)
gone = d_ab > 0.05 * u.arcsec
new = d_ba > 0.05 * u.arcsec
bands = [c[len('replaced_saturated_'):] for c in cb.colnames if c.startswith('replaced_saturated_')]
def satmask(c):
    m = np.zeros(len(c), bool)
    for b in bands:
        m |= np.asarray(c[f'replaced_saturated_{b}']).astype(bool)
    return m
satA, satB = satmask(ca), satmask(cb)
_, dsa, _ = sa.match_to_catalog_sky(sa[satA], nthneighbor=1)
_, dsb, _ = sb.match_to_catalog_sky(sb[satB], nthneighbor=1)
dsa, dsb = dsa.to(u.arcsec).value, dsb.to(u.arcsec).value
edges = [0, 0.3, 1, 3, 10, 1e9]
def hist(x):
    return np.histogram(x, edges)[0]
print('\nbins of distance to nearest saturated row ("):', edges[:-1])
print(f'all {A} rows        ', hist(dsa[~satA]))
print(f'gone {A} rows       ', hist(dsa[gone & ~satA]), ' total', (gone & ~satA).sum(), ' satrows gone', (gone & satA).sum())
print(f'new {B} rows        ', hist(dsb[new & ~satB]), ' total', (new & ~satB).sum(), ' satrows new', (new & satB).sum())
for nm, c, sel in [('gone', ca, gone & ~satA), ('new', cb, new & ~satB)]:
    r = c[sel]
    print(f'{nm}: spike_artifact {np.sum(r["spike_artifact"])}, n_real_bands==1 {np.sum(r["n_real_bands"] == 1)}, '
          f'>=4 {np.sum(r["n_real_bands"] >= 4)}; detection band:',
          dict(sorted(zip(*np.unique([x.decode() if isinstance(x, bytes) else x for x in r['skycoord_ref_filtername']], return_counts=True)), key=lambda kv: -kv[1])[:8]))
