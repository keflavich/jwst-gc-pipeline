"""Recovered-core cap tolerance counterfactual for main2 (KEEP_FINITE off) and main2kf (on).

Capped m7 satstar rows (flux_fit < 0.999 flux_fit_precap) have flux_fit = cap.  A cap with a
fractional tolerance tol gives flux = min(flux_fit_precap, cap * (1 + tol)); uncapped rows are
unchanged.  tol = inf is the cap switched off.  Per m8 star replaced in both arms, the
predicted dm is dm - median_frames 2.5 log10(flux_new / flux_fit), using the m7 rows within
0.1".  The cap ratio is read from flux_fit_raw (the capped fit before the wing
self-calibration, which divides it by wingcal_ratio), so the shift is
min(precap, raw * (1 + tol)) / raw on capped rows.  Effects on neighbours and residuals are not
modelled.
usage: PYTHONPATH=$Q python kf_lwmad/capoff_tol.py [BAND ...]   (default: the LW satstar bands)
"""
import glob
import sys
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord, concatenate
import astropy.units as u
import analyze as an

Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
an.ZPWIN.update(an.zp_windows())
ARMS = {'main2': an.Arm('main2'), 'main2kf': an.Arm('main2kf')}
LABEL = {'main2': 'off', 'main2kf': 'on'}
BANDS = sys.argv[1:] or ['250M', '277W', '300M', '323N', '410M']
BINS = [(10, 13), (13, 14), (14, 15), (15, 16), (16, 18)]
TOLS = [0.0, 0.03, 0.05, 0.10, np.inf]


def frame_rows(arm, band):
    sk, f, pre = [], [], []
    for fn in sorted(glob.glob(f'{Q}/tree_{arm}/F{band}/pipeline/*_nrc*_align_o005_crf_resbgsub_m7_satstar_catalog.fits')):
        t = Table.read(fn)
        c = t['skycoord_fit']
        sk.append(c if isinstance(c, SkyCoord) else SkyCoord(c))
        f.append(np.ma.filled(t['flux_fit_raw' if 'flux_fit_raw' in t.colnames else 'flux_fit'], np.nan).astype(float))
        pre.append(np.ma.filled(t['flux_fit_precap'], np.nan).astype(float))
    return concatenate(sk), np.concatenate(f), np.concatenate(pre)


def stat(x):
    x = x[np.isfinite(x)]
    return f'{np.median(x):+.3f} / {an.mad(x):.3f} / {np.mean(np.abs(x) > 0.3):.3f}' if len(x) else '—'


A, B = ARMS['main2'], ARMS['main2kf']
print('Predicted dm against dolphot for a recovered-core cap with tolerance tol (tol = inf: cap off).')
print('Stars replaced in both arms with m7 rows within 0.1" in both arms; bins are dolphot magnitude.')
print('Bin columns: median dm.  "all": median / MAD / f(|dm|>0.3).  tol = 0 is the current cap.\n')
for b in BANDS:
    ok = A.matched & B.matched & A.rep[b] & B.rep[b] & np.isfinite(A.ref[b])
    rows, have = {}, np.ones(ok.sum(), bool)
    for arm, X in ARMS.items():
        sk, f, pre = frame_rows(arm, b)
        i, j, _, _ = sk.search_around_sky(X.sky[X.idx[ok]], 0.1 * u.arcsec)
        capped = f[j] < 0.999 * pre[j]
        have &= np.bincount(i, minlength=ok.sum()) > 0
        rows[arm] = (i, f[j], pre[j], capped)
    ref = A.ref[b][ok]
    print(f'#### F{b} (N = {have.sum()} of {ok.sum()} replaced in both)\n')
    print('| arm | tol | ' + ' | '.join(f'{lo}–{hi}' for lo, hi in BINS) + ' | all |')
    print('|---|---|' + '---|' * (len(BINS) + 1))
    for arm, X in ARMS.items():
        i, f, pre, capped = rows[arm]
        dm0 = X.dm(b)[ok]
        for tol in TOLS:
            new = np.where(capped, np.minimum(pre, f * (1 + tol)), f)
            r = -2.5 * np.log10(new / f)
            sh = np.full(ok.sum(), np.nan)
            for k in np.unique(i):
                v = r[i == k]
                v = v[np.isfinite(v)]
                if len(v):
                    sh[k] = np.median(v)
            dm = dm0 + sh
            cells = []
            for lo, hi in BINS:
                s = have & (ref >= lo) & (ref < hi) & np.isfinite(dm)
                cells.append(f'{np.median(dm[s]):+.3f}' if s.sum() >= 3 else '—')
            print(f'| {LABEL[arm]} | {tol:g} | ' + ' | '.join(cells) + f' | {stat(dm[have])} |')
    n = [(have & (ref >= lo) & (ref < hi)).sum() for lo, hi in BINS]
    print(f'\nN per bin: {", ".join(map(str, n))}\n')
