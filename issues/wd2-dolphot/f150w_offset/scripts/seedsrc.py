"""Which band supplied each m7 cross-band seed position?  Seeds are the
highest-S/N member of a 30 mas cluster; F150W positions carry the DVA shift.
Classify each seed by its offset to dolphot: near the F150W band offset or
near the common offset of the other bands.  usage: python seedsrc.py"""
import glob
import sys
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an

T = f'{an.Q}/tree_mainfcbg'
an.ZPWIN.update(an.zp_windows())
A = an.Arm('mainfcbg')
dsk = SkyCoord(np.asarray(A.m['RA'], float) * u.deg, np.asarray(A.m['DEC'], float) * u.deg)
seedf = sorted(glob.glob(f'{T}/catalogs/crossband_seed_manual*.fits'))[0]
s = Table.read(seedf)
sk = s['skycoord']
j, d, _ = sk.match_to_catalog_sky(dsk)
ok = d.mas < 80
dra = (sk.ra.deg - dsk.ra.deg[j]) * np.cos(np.deg2rad(dsk.dec.deg[j])) * 3.6e6
dde = (sk.dec.deg - dsk.dec.deg[j]) * 3.6e6
OTHER = (-42.4, +5.4)
F150 = (-28.1, -12.7)
r_o = np.hypot(dra - OTHER[0], dde - OTHER[1])
r_f = np.hypot(dra - F150[0], dde - F150[1])
near_f = ok & (r_f < 6) & (r_o > 12)
near_o = ok & (r_o < 6) & (r_f > 12)
print(seedf, len(s), 'seeds;', ok.sum(), 'within 80 mas of a dolphot star')
print(f'seed at F150W offset: {near_f.sum()} ({near_f.sum() / ok.sum():.3f}); at other-band offset: {near_o.sum()} ({near_o.sum() / ok.sum():.3f})')
cf = np.asarray(s['confirming_filters']).astype(str)
has150 = np.char.find(cf, 'f150w') >= 0
print('seeds whose cluster includes F150W:', (ok & has150).sum(), '; of those at F150W offset:', (near_f & has150).sum())
for lo, hi in ((10, 16), (16, 18.6), (18.6, 21), (21, 24)):
    ref = A.ref['200W'][j]
    q = ok & (ref >= lo) & (ref < hi)
    print(f'  dolphot F200W {lo}-{hi}: N={q.sum()} at F150W offset {np.mean(near_f[q]):.3f} at other {np.mean(near_o[q]):.3f}')
