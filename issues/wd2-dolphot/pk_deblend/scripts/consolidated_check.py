"""Where do pk3 stars with per-frame satstar rows at the star go in the consolidated satstar catalog (merge input)?"""
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
O = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/pk_deblend_trace'
G = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/f277w_gap/dbl'
ps = Table.read(f'{O}/per_star_ext.ecsv')
pf = Table.read(f'{O}/per_frame.ecsv')
star = SkyCoord(ps['ra'] * u.deg, ps['dec'] * u.deg)
for arm, tree in (('pk2', 'tree_dbl2'), ('pk3', 'tree_dbl3')):
    c = Table.read(f'{G}/{tree}/catalogs/f277w_consolidated_satstar_catalog.fits')
    print(arm, 'consolidated rows', len(c))
    if arm == 'pk3':
        print(c.colnames)
    sk = c['skycoord_fit']
    j, d, _ = star.match_to_catalog_sky(sk)
    d = d.arcsec
    s = pf[pf['arm'] == arm]
    sat008 = np.zeros(len(ps), bool)
    for r in s:
        sat008[r['idx']] |= r['sat_sep'] < 0.08
    mh = np.asarray(ps[f'{arm}_mhit'], bool)
    print(f'  stars with per-frame satstar <0.08": {sat008.sum()}; consolidated row <0.08": {(sat008 & (d < 0.08)).sum()}; '
          f'consolidated 0.08-0.5": {(sat008 & (d >= 0.08) & (d < 0.5)).sum()}')
    print(f'  of those, merged row <0.08": {(sat008 & mh).sum()}; not merged: {(sat008 & ~mh).sum()}')
    m = sat008 & ~mh
    if m.any():
        dl = np.asarray(c['flux_fit'], float)[j]
        print('  unmerged: nearest consolidated sep', np.round(d[m], 3), 'flux ratio to dolphot', np.round(dl[m] / ps['dol_flux'][m], 2))
        if 'n_meas_fit' in c.colnames:
            print('  n_meas_fit', np.asarray(c['n_meas_fit'])[j][m], 'n_frames_fit', np.asarray(c['n_frames_fit'])[j][m])
        if 'sat_area' in c.colnames:
            print('  sat_area of that consolidated row', np.asarray(c['sat_area'], float)[j][m])
    # within the 13 unmerged: per-frame rows within 0.5 of the nearest consolidated row
    if arm == 'pk3' and m.any():
        for i in np.where(m)[0]:
            cs = sk[j[i]]
            nbr = 0
            for e in (1, 2, 3, 4):
                f = Table.read(f'{G}/tree_dbl3/F277W/pipeline/jw03523005001_10101_0000{e}_nrcblong_align_o005_crf_resbgsub_m7_satstar_catalog.fits')
                fs = f['skycoord_fit']
                sep = cs.separation(fs).arcsec
                nbr += (sep < 0.8).sum()
            print(f"   star {i}: consolidated sep {d[i]:.3f} flux {np.asarray(c['flux_fit'],float)[j[i]]:.0f} (dolphot {ps['dol_flux'][i]:.0f}), per-frame rows within 0.8\" of it summed over 4 exp: {nbr}")
