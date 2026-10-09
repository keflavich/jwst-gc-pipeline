"""Satstar (replaced-row) dm against dolphot by detector for main2, main2kf, main2kfh.

Each dolphot star replaced by a satstar in a band is assigned the detector of the
per-frame m7 satstar rows (tree_main2kfh) within 0.1 arcsec.  Writes det_split.md.
"""
import glob
import re
import sys
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an

Q = an.Q
an.PATH['main2kfh'] = (f'{Q}/tree_main2kfh/{an.M8}', f'{an.D}/matched_Q_main2kfh.fits')
an.ZPWIN.update(an.zp_windows())
ARMS = ['main2', 'main2kf', 'main2kfh']
A = {n: an.Arm(n) for n in ARMS}
BANDS = ['115W', '150W', '162M', '182M', '200W', '250M', '277W', '300M', '410M']


def mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


L = ['| band | det | N | ' + ' | '.join(f'{n} med / MAD' for n in ARMS) + ' | kfh - kf med | kfh closer / kf closer (>0.01) |',
     '|---|---|---|' + '---|' * len(ARMS) + '---|---|']
for b in BANDS:
    fr = sorted(glob.glob(f'{Q}/tree_main2kfh/F{b}/pipeline/*_m7_satstar_catalog.fits'))
    ra, dec, det = [], [], []
    for f in fr:
        t = Table.read(f)
        c = t['skycoord_fit']
        d = re.search(r'_(nrc[ab][1-4]|nrc[ab]long)_', f).group(1)
        ok = np.isfinite(c.ra.deg)
        ra.append(c.ra.deg[ok]); dec.append(c.dec.deg[ok]); det += [d] * int(ok.sum())
    sk = SkyCoord(np.concatenate(ra) * u.deg, np.concatenate(dec) * u.deg)
    det = np.array(det)
    a0 = A['main2kfh']
    sel = a0.matched & a0.rep[b] & np.isfinite(a0.ref[b])
    for n in ARMS:
        sel &= A[n].matched & np.isfinite(A[n].dm(b))
    tgt = a0.sky[a0.idx[sel]]
    j, i, _, _ = tgt.search_around_sky(sk, 0.1 * u.arcsec)
    sdet = np.full(sel.sum(), '', dtype='<U9')
    for ii, jj in zip(i, j):
        if sdet[ii] == '':
            sdet[ii] = det[jj]
    dms = {n: A[n].dm(b)[sel] for n in ARMS}
    for dname in ['all'] + sorted(set(sdet) - {''}):
        h = np.ones(len(sdet), bool) if dname == 'all' else sdet == dname
        if h.sum() < 10:
            continue
        cells = [f'{np.median(dms[n][h]):+.3f} / {mad(dms[n][h]):.3f}' for n in ARMS]
        dd = np.abs(dms['main2kfh'][h]) - np.abs(dms['main2kf'][h])
        L.append(f'| F{b} | {dname} | {h.sum()} | ' + ' | '.join(cells)
                 + f' | {np.median(dms["main2kfh"][h] - dms["main2kf"][h]):+.4f} | {(dd < -0.01).sum()} / {(dd > 0.01).sum()} |')
open(f'{Q}/kfhdet/det_split.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
