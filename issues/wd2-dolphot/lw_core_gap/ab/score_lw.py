"""Score per-band LW m7 catalogs (unvetted) against dolphot; band-generic
version of score_dbl.py.
usage: python score_lw.py BAND NAME=PATH [NAME=PATH ...]   (BAND: 250M, 277W, 300M)"""
import sys
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an
b = sys.argv[1].upper().lstrip('F')
G = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/f277w_gap'
an.ZPWIN.update(an.zp_windows())
A = an.Arm('mainfcbg')
rs_all = SkyCoord(np.asarray(A.m['RA'], float) * u.deg, np.asarray(A.m['DEC'], float) * u.deg)
ref = A.ref[b]
tr = Table.read(f'{G}/trace_mainfcbg_{b}.ecsv')
noro = np.asarray(tr['dolphot_idx'])
lo, hi = np.nanmin(tr['ref_mag']), np.nanmax(tr['ref_mag'])
ctrl = np.where(A.matched & np.isfinite(A.our[b]) & (ref >= lo) & (ref <= hi))[0]
bright = np.where(np.isfinite(ref) & (ref < 18.6))[0]
mid = np.where(np.isfinite(ref) & (ref >= 18.6) & (ref < 21))[0]
core = SkyCoord(156.0 * u.deg, -57.757 * u.deg)
lines = [f'## F{b}']
for arg in sys.argv[2:]:
    name, path = arg.split('=', 1)
    t = Table.read(path)
    sk = t['skycoord']
    fl = np.asarray(t['flux'], float)
    ok = np.isfinite(sk.ra.deg) & (fl > 0)
    t, sk, fl = t[ok], sk[ok], fl[ok]
    mi = -2.5 * np.log10(fl)
    rep = np.asarray(t['replaced_saturated'], bool) if 'replaced_saturated' in t.colnames else np.zeros(len(t), bool)
    j, d, _ = rs_all[mid].match_to_catalog_sky(sk)
    sel = (d.arcsec < 0.05) & ~rep[j]
    zp = np.median(ref[mid][sel] - mi[j][sel])
    m = mi + zp
    lines.append(f'### {name}: {len(t)} rows ({rep.sum()} replaced_saturated), ZP {zp:.3f} from {sel.sum()} stars 18.6-21')
    for g, idx in ((f'no-row ({len(noro)})', noro), ('control', ctrl), (f'all dolphot F{b} < 18.6', bright),
                   (f'dolphot F{b} 18.6-21', mid)):
        jj, dd, _ = rs_all[idx].match_to_catalog_sky(sk)
        hit = dd.arcsec < 0.08
        dm = m[jj] - ref[idx]
        good = hit & (np.abs(dm) < 0.3)
        lines.append(f'  {g}: N={len(idx)}, row within 0.08" {hit.sum()}, |dm|<0.3 {good.sum()} '
                     f'({good.mean():.3f}), median dm {np.median(dm[hit]) if hit.any() else np.nan:+.3f}, '
                     f'replaced {np.sum(hit & rep[jj])}')
    incore = sk.separation(core).arcsec < 20
    j2, d2, _ = sk[incore].match_to_catalog_sky(rs_all)
    lines.append(f'  rows within 20" of the core: {incore.sum()}; with no dolphot source within 0.1": '
                 f'{np.sum(d2.arcsec > 0.1)}; of those brighter than 18.6: {np.sum((d2.arcsec > 0.1) & (m[incore] < 18.6))}')
print('\n'.join(lines))
