"""Do forced-refit rows that are fainter than dolphot sit off the dolphot
position?  dm and residual offset (after removing the median frac=0 offset)
for frac>=0.5 rows.  usage: python seedoff.py BAND"""
import sys
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an

b = sys.argv[1].upper().lstrip('F')
an.ZPWIN.update(an.zp_windows())
A = an.Arm('mainfcbg')
dsk = SkyCoord(np.asarray(A.m['RA'], float) * u.deg, np.asarray(A.m['DEC'], float) * u.deg)
ref = A.ref[b]
t = Table.read(f'{an.Q}/tree_mainfcbg/catalogs/f{b.lower()}_merged_indivexp_merged_resbgsub_m7_dao_basic.fits')
sk = t['skycoord']
fl = np.asarray(t['flux'], float)
ok = np.isfinite(sk.ra.deg) & (fl > 0)
t, sk, fl = t[ok], sk[ok], fl[ok]
mi = -2.5 * np.log10(fl)
rep = np.asarray(t['replaced_saturated'], bool)
ff = np.asarray(t['forced_refit_frac'], float)
have = np.where(np.isfinite(ref))[0]
j, d, _ = dsk[have].match_to_catalog_sky(sk)
hit = (d.arcsec < 0.08) & ~rep[j]
mid = hit & (ref[have] >= 18.6) & (ref[have] < 21) & (d.arcsec < 0.05)
zp = np.median(ref[have][mid] - mi[j][mid])
dm = mi[j] + zp - ref[have]
dra = (sk.ra.deg[j] - dsk.ra.deg[have]) * np.cos(np.deg2rad(dsk.dec.deg[have])) * 3.6e6
dde = (sk.dec.deg[j] - dsk.dec.deg[have]) * 3.6e6
z = hit & (ff[j] == 0)
ox, oy = np.median(dra[z]), np.median(dde[z])
r = np.hypot(dra - ox, dde - oy)
print(f'F{b}: median offset frac=0 ({ox:+.1f}, {oy:+.1f}) mas; residual offset median frac=0 {np.median(r[z]):.1f} mas')
for lab, s in (('frac=0', z), ('0<frac<0.5', hit & (ff[j] > 0) & (ff[j] < 0.5)), ('frac>=0.5', hit & (ff[j] >= 0.5))):
    for lo, hi in ((10, 18.6), (18.6, 21), (21, 24)):
        q = s & (ref[have] >= lo) & (ref[have] < hi)
        if q.sum() < 5:
            continue
        print(f'  {lab:11s} {lo}-{hi}: N={q.sum():5d} r med {np.median(r[q]):5.1f} mas p90 {np.percentile(r[q], 90):5.1f}; '
              f'dm med {np.median(dm[q]):+.3f}; corr(dm, r) {np.corrcoef(dm[q], r[q])[0, 1]:+.2f}')
q = hit & (ff[j] >= 0.5)
for lo, hi in ((0, 10), (10, 20), (20, 40), (40, 80)):
    s = q & (r >= lo) & (r < hi)
    if s.sum():
        print(f'  frac>=0.5, r {lo}-{hi} mas: N={s.sum()} dm med {np.median(dm[s]):+.3f}')
