"""c stars: dolphot neighbours and merged neighbours; is the merged value consistent with a blend (dolphot flux sum within R)? -> c_blend.ecsv"""
import sys, warnings
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
warnings.filterwarnings('ignore')
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
sys.path.insert(0, Q)
import analyze as an
an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2kfpk')
rs = SkyCoord(np.asarray(A.m['RA'], float)*u.deg, np.asarray(A.m['DEC'], float)*u.deg)
C = Table.read('c_detail.ecsv'); out = []
zp = {'277W': 24.128923927404763, '250M': 24.315577424137363, '300M': 23.970356429490028}
for b in ('277W', '250M', '300M'):
    ref = np.asarray(A.ref[b], float); ok = np.isfinite(ref); idxs = np.where(ok)[0]; sk = rs[ok]
    fin = Table.read(f'{Q}/tree_main2kfpk/catalogs/f{b.lower()}_merged_indivexp_merged_resbgsub_m7_dao_basic.fits')
    fc = fin['skycoord']; fm = -2.5*np.log10(np.asarray(fin['flux'], float)) + zp[b]
    for c in C[C['band'] == b]:
        p = rs[int(c['dolphot_idx'])]
        sep = sk.separation(p).arcsec
        row = dict(band=b, dolphot_idx=int(c['dolphot_idx']), ref=c['ref'], fin=c['fin_mag'], dm=c['dm_fin'])
        for R in (0.15, 0.3, 0.6):
            sel = (sep < R)
            fl = 10**(-0.4*ref[idxs[sel]])
            row[f'n_dol_{R}'] = int(sel.sum()); row[f'blend_mag_{R}'] = float(-2.5*np.log10(fl.sum()))
        nb = (sep > 0.01) & (sep < 0.6)
        if nb.any():
            k = np.argmin(np.where(nb, sep, 9)); row['nn_dol_sep'] = float(sep[k]); row['nn_dol_dm'] = float(ref[idxs[k]] - c['ref'])
        else: row['nn_dol_sep'] = np.nan; row['nn_dol_dm'] = np.nan
        sf = fc.separation(p).arcsec; nbm = (sf > 0.08) & (sf < 0.6)
        row['n_merged_nb_0.6'] = int(nbm.sum())
        out.append(row)
O = Table(rows=out); O.write('c_blend.ecsv', overwrite=True)
for r in O: print(r['band'], r['dolphot_idx'], f"ref={r['ref']:.2f} fin={r['fin']:.2f} dm={r['dm']:+.2f} blend0.15={r['blend_mag_0.15']:.2f}(n={r['n_dol_0.15']}) blend0.3={r['blend_mag_0.3']:.2f}(n={r['n_dol_0.3']}) blend0.6={r['blend_mag_0.6']:.2f}(n={r['n_dol_0.6']}) nn_sep={r['nn_dol_sep']:.2f} nn_dm={r['nn_dol_dm']:+.2f}")
