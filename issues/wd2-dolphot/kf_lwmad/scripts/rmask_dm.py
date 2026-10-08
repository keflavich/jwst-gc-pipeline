"""Uncapped (precap) and final dm against dolphot vs the satstar wing-fit mask radius (wingcal_rmask).
Stars replaced in both arms; per star the median over its m7 rows within 0.1".
usage: PYTHONPATH=$Q python kf_lwmad/rmask_dm.py BAND [BAND ...]"""
import glob
import sys
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord, concatenate
import astropy.units as u
import analyze as an

Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
an.ZPWIN.update(an.zp_windows())
ARMS = {'off': an.Arm('main2'), 'on': an.Arm('main2kf')}
TREE = {'off': 'main2', 'on': 'main2kf'}
RB = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 7), (7, 20)]
for b in sys.argv[1:]:
    A, B = ARMS['off'], ARMS['on']
    ok = A.matched & B.matched & A.rep[b] & B.rep[b] & np.isfinite(A.ref[b])
    print(f'\n#### F{b}: dm by wing-fit mask radius (px); cells: N, median final dm / median precap dm, median dolphot mag\n')
    print('| arm | ' + ' | '.join(f'{lo}–{hi} px' for lo, hi in RB) + ' |')
    print('|---|' + '---|' * len(RB))
    for arm, X in ARMS.items():
        sk, raw, pre, rm = [], [], [], []
        for fn in sorted(glob.glob(f'{Q}/tree_{TREE[arm]}/F{b}/pipeline/*_nrc*_align_o005_crf_resbgsub_m7_satstar_catalog.fits')):
            t = Table.read(fn)
            c = t['skycoord_fit']
            sk.append(c if isinstance(c, SkyCoord) else SkyCoord(c))
            raw.append(np.ma.filled(t['flux_fit_raw'], np.nan).astype(float))
            pre.append(np.ma.filled(t['flux_fit_precap'], np.nan).astype(float))
            rm.append(np.ma.filled(t['wingcal_rmask'], np.nan).astype(float))
        sk, raw, pre, rm = concatenate(sk), np.concatenate(raw), np.concatenate(pre), np.concatenate(rm)
        i, j, _, _ = sk.search_around_sky(X.sky[X.idx[ok]], 0.1 * u.arcsec)
        n = ok.sum()
        sh, r = np.full(n, np.nan), np.full(n, np.nan)
        for k in np.unique(i):
            jj = j[i == k]
            sh[k] = np.nanmedian(-2.5 * np.log10(pre[jj] / raw[jj]))
            r[k] = np.nanmedian(rm[jj])
        dm = X.dm(b)[ok]
        ref = X.ref[b][ok]
        cells = []
        for lo, hi in RB:
            s = (r >= lo) & (r < hi) & np.isfinite(dm) & np.isfinite(sh)
            cells.append(f'{s.sum()}: {np.median(dm[s]):+.3f} / {np.median(dm[s] + sh[s]):+.3f}, {np.median(ref[s]):.1f}'
                         if s.sum() >= 5 else (f'{s.sum()}' if s.sum() else '—'))
        print(f'| {arm} | ' + ' | '.join(cells) + ' |')
