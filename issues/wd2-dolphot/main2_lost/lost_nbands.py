"""For dolphot stars matched in A but not in B: how many dolphot bands carry a magnitude, compared with
stars matched in both that sit at the same distance from a saturated row. usage: lost_nbands.py A B"""
import sys
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

D = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark'
M8 = 'catalogs/basic_merged_indivexp_photometry_tables_merged_resbgsub_m8_dedup.fits'
A, B = sys.argv[1:3]
ma, mb = Table.read(f'{D}/matched_Q_{A}.fits'), Table.read(f'{D}/matched_Q_{B}.fits')
bands = [c[4:] for c in ma.colnames if c.startswith('ref_')]
nb = np.zeros(len(ma), int)
for b in bands:
    r = np.asarray(ma[f'ref_{b}'], float)
    nb += (np.isfinite(r) & (r < 90)).astype(int)
cb = Table.read(f'{D}/Q_integ/tree_{B}/{M8}')
sat = np.zeros(len(cb), bool)
for c in cb.colnames:
    if c.startswith('replaced_saturated_'):
        sat |= np.asarray(cb[c]).astype(bool)
ref = SkyCoord(ma['RA'] * u.deg, ma['DEC'] * u.deg)
_, dsat, _ = ref.match_to_catalog_sky(SkyCoord(cb['skycoord_ref'])[sat])
dsat = dsat.arcsec
lost = np.asarray(ma['matched']) & ~np.asarray(mb['matched'])
both = np.asarray(ma['matched']) & np.asarray(mb['matched'])
print(f'{A} only: {lost.sum()}')
for lab, lo, hi in (('<1"', 0, 1), ('1-3"', 1, 3), ('3-10"', 3, 10), ('>=10"', 10, 1e9)):
    s_l = lost & (dsat >= lo) & (dsat < hi)
    s_b = both & (dsat >= lo) & (dsat < hi)
    if s_l.sum() == 0 and s_b.sum() == 0:
        continue
    hl = np.bincount(nb[s_l], minlength=17)
    print(f'  dsat {lab:6s} lost N={s_l.sum():4d} median nbands={np.median(nb[s_l]) if s_l.any() else np.nan:4.1f} '
          f'frac nbands<=3 {np.mean(nb[s_l] <= 3) if s_l.any() else np.nan:.2f} | matched-both N={s_b.sum():5d} '
          f'median nbands={np.median(nb[s_b]):4.1f} frac<=3 {np.mean(nb[s_b] <= 3):.2f}')
# which dolphot bands carry the lost stars
cnt = {b: int(np.sum(lost & np.isfinite(np.asarray(ma[f'ref_{b}'], float)) & (np.asarray(ma[f'ref_{b}'], float) < 90))) for b in bands}
print('lost stars with a dolphot value per band:', cnt)
