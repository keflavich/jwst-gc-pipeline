"""Dither self-consistency: within each star, per-frame magnitude deviation against per-frame pixel-area deviation.
pred_f = 2.5 log10(area5 / raw5) is AREA at the star in that frame.  Slope 0 = flux independent of where the star
lands; slope +1 = reads faint by the full pixel-area term.  usage: python dither_gt.py"""
import numpy as np
from astropy.table import Table
from scipy.stats import theilslopes
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
SRC = {'main2': f'{Q}/apclosure/frames_{{b}}.ecsv', 'main2gt': f'{Q}/pam1151/frames_main2gt_{{b}}.ecsv'}
print('| band | arm | N frames | σ(Δpred) | ours psf | ap_area r5 | ap_raw r5 |')
print('|---|---|---|---|---|---|---|')
for band in ('115W', '150W', '162M', '182M', '200W', '212N', '164N', '187N', '250M', '277W', '300M', '323N', '335M', '405N', '410M', '466N'):
    for arm, fn in SRC.items():
        T = Table.read(fn.format(b=band))
        T = T[(T['src'] == 1) & (T['psf'] > 0) & (T['raw5'] > 0) & (T['area5'] > 0)]
        p = 2.5 * np.log10(T['area5'] / T['raw5'])
        m = {'psf': -2.5 * np.log10(T['psf']), 'area': -2.5 * np.log10(T['area5']), 'raw': -2.5 * np.log10(T['raw5'])}
        keyv = np.array([f'{i}_{d}' for i, d in zip(T['i'], T['det'])])  # same star, same detector
        u, inv, cnt = np.unique(keyv, return_inverse=True, return_counts=True)
        ok = cnt[inv] >= 2
        def dev(x):
            s = np.bincount(inv, x, len(u)) / cnt
            return (x - s[inv])[ok]
        dp = dev(np.asarray(p))
        out = []
        for k in ('psf', 'area', 'raw'):
            dm = dev(np.asarray(m[k]))
            sd = 1.4826 * np.median(np.abs(dm))
            kk = np.abs(dm) < 5 * sd
            s, i0, lo, hi = theilslopes(dm[kk], dp[kk], 0.95)
            out.append(f'{s:+.2f} [{lo:+.2f}, {hi:+.2f}]')
        print(f'| F{band} | {arm} | {ok.sum()} | {np.std(dp):.4f} | ' + ' | '.join(out) + ' |', flush=True)
