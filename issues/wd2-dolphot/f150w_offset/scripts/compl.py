"""Completeness of the m7 merged per-band catalogs against dolphot, per
dolphot magnitude bin, after removing each band's median offset.
usage: python compl.py"""
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
bins = [(10, 18), (18, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24, 25)]
print('band | rows | ' + ' | '.join(f'{lo}-{hi}' for lo, hi in bins))
for b in ('115W', '150W', '162M', '182M', '200W', '212N'):
    t = Table.read(f'{T}/catalogs/f{b.lower()}_merged_indivexp_merged_resbgsub_m7_dao_basic.fits')
    sk = t['skycoord']
    ok = np.isfinite(sk.ra.deg) & (np.asarray(t['flux'], float) > 0)
    sk = sk[ok]
    ref = A.ref[b]
    have = np.where(np.isfinite(ref))[0]
    j, d, _ = dsk[have].match_to_catalog_sky(sk)
    dra = (sk.ra.deg[j] - dsk.ra.deg[have]) * np.cos(np.deg2rad(dsk.dec.deg[have])) * 3.6e6
    dde = (sk.dec.deg[j] - dsk.dec.deg[have]) * 3.6e6
    close = d.mas < 60
    ox, oy = np.median(dra[close]), np.median(dde[close])
    hit = np.hypot(dra - ox, dde - oy) < 40
    cells = []
    for lo, hi in bins:
        q = (ref[have] >= lo) & (ref[have] < hi)
        cells.append(f'{hit[q].mean():.3f} ({q.sum()})')
    print(f'F{b} | {ok.sum()} | ' + ' | '.join(cells), flush=True)
