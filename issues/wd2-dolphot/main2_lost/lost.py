"""Dolphot stars matched in arm A but not in arm B: where they are, how bright,
what the nearest B row is, and how close a saturated star sits.
usage: lost.py A B  (Q arm names, e.g. mainfcbg mainfcbgkf)"""
import sys
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
D = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark'
M8 = 'catalogs/basic_merged_indivexp_photometry_tables_merged_resbgsub_m8_dedup.fits'
A, B = sys.argv[1:3]
ma, mb = Table.read(f'{D}/matched_Q_{A}.fits'), Table.read(f'{D}/matched_Q_{B}.fits')
ca, cb = Table.read(f'{D}/Q_integ/tree_{A}/{M8}'), Table.read(f'{D}/Q_integ/tree_{B}/{M8}')
assert np.allclose(ma['RA'], mb['RA'])
lost = ma['matched'] & ~mb['matched']
gain = ~ma['matched'] & mb['matched']
print(f'{A} matched {ma["matched"].sum()}, {B} matched {mb["matched"].sum()}; only {A} {lost.sum()}, only {B} {gain.sum()}')
ref = SkyCoord(ma['RA'] * u.deg, ma['DEC'] * u.deg)
# the compare applies a ref->ours offset; recover it from the matched pairs of A
sa, sb = SkyCoord(ca['skycoord_ref']), SkyCoord(cb['skycoord_ref'])
ia = np.asarray(ma['our_idx'])
ok = ma['matched']
dra = np.median((sa[ia[ok]].ra - ref[ok].ra).to(u.mas) * np.cos(ref[ok].dec))
ddec = np.median((sa[ia[ok]].dec - ref[ok].dec).to(u.mas))
refs = ref.spherical_offsets_by(dra, ddec)
print(f'offset dra {dra:.1f} ddec {ddec:.1f}')
idx, d2, _ = refs.match_to_catalog_sky(sb)
# saturated rows in B: any band replaced_saturated
bands = [c[len('replaced_saturated_'):] for c in cb.colnames if c.startswith('replaced_saturated_')]
satB = np.zeros(len(cb), bool)
for b in bands:
    satB |= np.asarray(cb[f'replaced_saturated_{b}']).astype(bool)
_, dsat, _ = refs.match_to_catalog_sky(sb[satB])
satA = np.zeros(len(ca), bool)
for b in bands:
    satA |= np.asarray(ca[f'replaced_saturated_{b}']).astype(bool)
_, dsatA, _ = refs.match_to_catalog_sky(sa[satA])
# B row taken by another dolphot star?
taken = np.zeros(len(cb), int) - 1
ib = np.asarray(mb['our_idx'])
taken[ib[mb['matched']]] = np.where(mb['matched'])[0]
def summarize(sel, name):
    n = sel.sum()
    d = d2[sel].to(u.arcsec).value
    print(f'\n== {name}: {n}')
    print('nearest B row sep ("): <0.08', (d < 0.08).sum(), ' 0.08-0.15', ((d >= 0.08) & (d < 0.15)).sum(),
          ' 0.15-0.3', ((d >= 0.15) & (d < 0.3)).sum(), ' >=0.3', (d >= 0.3).sum())
    tk = taken[idx[sel]] >= 0
    print('nearest B row already matched to another dolphot star:', tk.sum())
    ds = dsat[sel].to(u.arcsec).value
    dsa = dsatA[sel].to(u.arcsec).value
    print('dist to nearest B saturated row ("): <1', (ds < 1).sum(), ' 1-3', ((ds >= 1) & (ds < 3)).sum(),
          ' 3-10', ((ds >= 3) & (ds < 10)).sum(), ' >=10', (ds >= 10).sum())
    print('dist to nearest A saturated row ("): <1', (dsa < 1).sum(), ' 1-3', ((dsa >= 1) & (dsa < 3)).sum(),
          ' 3-10', ((dsa >= 3) & (dsa < 10)).sum(), ' >=10', (dsa >= 10).sum())
    f2 = np.asarray(ma['ref_200W'][sel], float)
    print('dolphot F200W:', np.histogram(f2[np.isfinite(f2) & (f2 < 90)], [0, 15, 17, 19, 21, 23, 30])[0],
          'no F200W', (~(np.isfinite(f2) & (f2 < 90))).sum())
    return np.where(sel)[0]
L = summarize(lost, f'only {A}')
G = summarize(gain, f'only {B}')
# A-row properties for lost stars
r = ca[ia[L]]
print('\nA rows of lost stars: spike_artifact', np.sum(r['spike_artifact']), ' n_real_bands median', np.median(r['n_real_bands']),
      ' n_merged median', np.median(r['n_merged']))
print('skycoord_ref_filtername:', dict(zip(*np.unique(np.asarray(r['skycoord_ref_filtername']), return_counts=True))))
np.save(f'{D}/Q_integ/kf_lost/lost_{A}_{B}.npy', L)
np.save(f'{D}/Q_integ/kf_lost/gain_{A}_{B}.npy', G)
out = Table({'i': L, 'RA': ma['RA'][L], 'DEC': ma['DEC'][L], 'F200W': ma['ref_200W'][L],
             'sepB': d2[L].to(u.arcsec), 'dsatB': dsat[L].to(u.arcsec), 'takenB': taken[idx[L]],
             'Arow': ia[L], 'Brow': idx[L]})
out.write(f'{D}/Q_integ/kf_lost/lost_{A}_{B}.ecsv', overwrite=True)
out.sort('dsatB'); out[:40].pprint(max_width=200, max_lines=60)
