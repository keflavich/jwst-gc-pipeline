"""Running median dm against dolphot, final (capped) vs uncapped (flux_fit_precap), per arm.
Stars replaced in both arms; the per-star shift is the median over its m7 rows within 0.1" of
-2.5 log10(precap / flux_fit_raw).
usage: PYTHONPATH=$Q python kf_lwmad/fig_capoff.py OUT.png"""
import glob
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.coordinates import SkyCoord, concatenate
import astropy.units as u
import analyze as an

Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
an.ZPWIN.update(an.zp_windows())
ARMS = {'main2': an.Arm('main2'), 'main2kf': an.Arm('main2kf')}
STYLE = {'main2': ('#8c510a', 'off (main2)'), 'main2kf': ('#c51b7d', 'on (main2kf)')}
BANDS = ['250M', '300M', '277W', '323N', '200W', '150W']
EDGES = np.arange(12.0, 18.01, 0.5)


def shift(X, arm, b, ok):
    sk, raw, pre = [], [], []
    for fn in sorted(glob.glob(f'{Q}/tree_{arm}/F{b}/pipeline/*_nrc*_align_o005_crf_resbgsub_m7_satstar_catalog.fits')):
        t = Table.read(fn)
        c = t['skycoord_fit']
        sk.append(c if isinstance(c, SkyCoord) else SkyCoord(c))
        raw.append(np.ma.filled(t['flux_fit_raw'], np.nan).astype(float))
        pre.append(np.ma.filled(t['flux_fit_precap'], np.nan).astype(float))
    sk, raw, pre = concatenate(sk), np.concatenate(raw), np.concatenate(pre)
    i, j, _, _ = sk.search_around_sky(X.sky[X.idx[ok]], 0.1 * u.arcsec)
    out = np.full(ok.sum(), np.nan)
    r = -2.5 * np.log10(pre[j] / raw[j])
    for k in np.unique(i):
        v = r[i == k]
        v = v[np.isfinite(v)]
        if len(v):
            out[k] = np.median(v)
    return out


def runmed(x, y):
    xc, yc = [], []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        s = (x >= lo) & (x < hi) & np.isfinite(y)
        if s.sum() >= 8:
            xc.append(np.median(x[s]))
            yc.append(np.median(y[s]))
    return np.array(xc), np.array(yc)


fig, axs = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True, sharey=True)
A, B = ARMS['main2'], ARMS['main2kf']
for ax, b in zip(axs.flat, BANDS):
    ok = A.matched & B.matched & A.rep[b] & B.rep[b] & np.isfinite(A.ref[b])
    n = 0
    for arm, X in ARMS.items():
        sh = shift(X, arm, b, ok)
        dm = X.dm(b)[ok]
        ref = X.ref[b][ok]
        have = np.isfinite(sh)
        n = have.sum()
        col, lab = STYLE[arm]
        x, y = runmed(ref[have], dm[have])
        ax.plot(x, y, '-o', color=col, ms=4, lw=2, label=f'{lab}, final')
        x, y = runmed(ref[have], dm[have] + sh[have])
        ax.plot(x, y, '--s', color=col, ms=4, lw=1.5, mfc='none', label=f'{lab}, uncapped')
    ax.axhline(0, color='0.5', lw=0.8)
    ax.set_title(f'F{b} (N={n})', fontsize=11)
    ax.grid(alpha=0.25)
for ax in axs[1]:
    ax.set_xlabel('dolphot magnitude')
for ax in axs[:, 0]:
    ax.set_ylabel('dm = ours − dolphot − ZP (mag)')
axs[0, 0].set_ylim(-0.16, 0.16)
axs[0, 0].legend(fontsize=8, loc='lower right')
fig.suptitle('Satstar-replaced stars (replaced in both arms): final vs uncapped (flux_fit_precap), running median per 0.5 mag', fontsize=11)
fig.tight_layout()
fig.savefig(sys.argv[1], dpi=110)
