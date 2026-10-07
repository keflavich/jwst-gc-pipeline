"""Do the satstar-deblend arms (#1082) give a good row to the mainfcbg F277W
no-row stars?  Frame jw03523005001_10101_00001_nrcblong only.
Instrumental mag -2.5 log10(flux_fit) is put on the dolphot scale with the
median offset of isolated accepted rows (mainfcbg m7 satstar catalog) at 14-17 mag."""
import sys
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an
b = '277W'
base = 'jw03523005001_10101_00001_nrcblong_align_o005_crf_resbgsub_m7_satstar_catalog.fits'
S = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/S_f150w_offset/satrerun/bp'
cats = {'mainfcbg': f'{an.Q}/tree_mainfcbg/F{b}/pipeline/{base}'}
for a in ('x', 'xdb', 'xdbms', 'csn', 'csndb'):
    cats[a] = f'{S}/{a}/F{b}/pipeline/{base}'
an.ZPWIN.update(an.zp_windows())
A = an.Arm('mainfcbg')
rs_all = SkyCoord(np.asarray(A.m['RA'], float) * u.deg, np.asarray(A.m['DEC'], float) * u.deg)
ref = A.ref[b]
tr = Table.read(f'trace_mainfcbg_{b}.ecsv')
noro = np.asarray(tr['dolphot_idx'])
ctrl = np.where(A.matched & np.isfinite(A.our[b]) & (ref >= np.nanmin(tr['ref_mag'])) & (ref <= np.nanmax(tr['ref_mag'])))[0]
hasb = np.where(np.isfinite(ref))[0]
zp = None
for name, fn in cats.items():
    t = Table.read(fn)
    sk = t['skycoord_fit']
    ok = np.isfinite(sk.ra.deg) & (np.asarray(t['flux_fit'], float) > 0)
    t, sk = t[ok], sk[ok]
    mi = -2.5 * np.log10(np.asarray(t['flux_fit'], float))
    j, d, _ = sk.match_to_catalog_sky(rs_all[hasb])
    if zp is None:
        sel = (d.arcsec < 0.05) & (ref[hasb][j] > 14) & (ref[hasb][j] < 17)
        zp = np.median(ref[hasb][j][sel] - mi[sel])
        print(f'ZP from {name}: {zp:.3f} (N={sel.sum()})')
    m = mi + zp
    out = [f'{name}: {len(t)} rows']
    for g, idx in (('no-row', noro), ('control', ctrl)):
        jj, dd, _ = rs_all[idx].match_to_catalog_sky(sk)
        hit = dd.arcsec < 0.1
        dm = m[jj] - ref[idx]
        good = hit & (np.abs(dm) < 0.3)
        out.append(f'  {g} N={len(idx)}: row within 0.1" {hit.sum()}, |dm|<0.3 {good.sum()}, '
                   f'median dm {np.median(dm[hit]) if hit.any() else np.nan:+.2f}')
    print('\n'.join(out))
