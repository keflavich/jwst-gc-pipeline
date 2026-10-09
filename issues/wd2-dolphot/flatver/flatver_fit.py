"""Regress unsaturated dm on pred, pred + (fl+ph), and pred + fl + ph; per-detector medians; figure flatver.png.
usage: python flatver_fit.py [tag] > flatver_fit.txt"""
import sys
import glob
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an

tag = sys.argv[1] if len(sys.argv) > 1 else 'jwst_1298'
HERE = f'{an.Q}/flatver'
an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2')
Pa = np.load(f'{an.Q}/photomver/areapred_main2.npz')
F = np.load(f'{HERE}/flatver_{tag}.npz')
dets = json.load(open(f'{HERE}/flatver_{tag}_dets.json'))


def rs(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def ols(y, X):
    X1 = np.c_[np.ones(len(y)), X]
    c, *_ = np.linalg.lstsq(X1, y, rcond=None)
    res = y - X1 @ c
    cov = np.linalg.inv(X1.T @ X1) * res.var(ddof=X1.shape[1])
    return c[1:], np.sqrt(np.diag(cov))[1:], rs(res)


rows = {}
print(f'old context {tag}')
print('| band | N | std(fl+ph) | TS slope dm~pred | std resid | OLS pred (alone) | OLS pred (with fl+ph) | coef fl+ph | std resid | OLS pred | coef fl | coef ph | std resid |')
print('|' + '---|' * 13)
for band in an.BANDS:
    if f'fl_{band}' not in F:
        continue
    dm, pred, fl, ph = A.dm(band), Pa[band], F[f'fl_{band}'], F[f'ph_{band}']
    lo, hi = an.ZPWIN.get(band, (0, 19))
    ok = A.matched & np.isfinite(dm) & ~A.rep[band] & ~A.sat[band] & (A.ref[band] >= lo) & (A.ref[band] < hi)
    ok &= np.isfinite(pred) & np.isfinite(fl) & np.isfinite(ph)
    d, p, f, h = dm[ok], pred[ok], fl[ok], ph[ok]
    s = f + h
    ts = stats.theilslopes(d, p)[0]
    c1, e1, r1 = ols(d, p[:, None])
    c2, e2, r2 = ols(d, np.c_[p, s])
    c3, e3, r3 = ols(d, np.c_[p, f, h]) if np.std(h) > 1e-6 and np.std(f) > 1e-6 else (np.full(3, np.nan), np.full(3, np.nan), np.nan)
    print(f'| F{band} | {ok.sum()} | {np.std(s):.4f} | {ts:.2f} | {r1:.4f} | {c1[0]:.2f}+-{e1[0]:.2f} | {c2[0]:.2f}+-{e2[0]:.2f} | {c2[1]:.2f}+-{e2[1]:.2f} | {r2:.4f} | '
          f'{c3[0]:.2f}+-{e3[0]:.2f} | {c3[1]:.2f}+-{e3[1]:.2f} | {c3[2]:.2f}+-{e3[2]:.2f} | {r3:.4f} |')
    rows[band] = dict(d=d, p=p, s=s, f=f, h=h, ok=ok)

print('\nPer-detector medians (mag): dm - pred, fl + ph (fl, ph separately), N')
print('| band | detector | N | med(dm-pred) | med(fl+ph) | med(fl) | med(ph) |')
print('|---|---|---|---|---|---|---|')
import glob as _g
from astropy.io import fits
from astropy.wcs import WCS
sky = A.sky[A.idx]
for band, R in rows.items():
    frames = sorted(_g.glob(f'{an.Q}/tree_main2/F{band}/pipeline/jw03523005001_*00001_*_align_o005_crf.fits'))
    detof = np.full(A.n, '', dtype=object)
    for fn in frames:
        with fits.open(fn, memmap=True) as fh:
            det = fh[0].header['DETECTOR']
            w = WCS(fh['SCI'].header)
            ny, nx = fh['SCI'].data.shape
        x, y = w.world_to_pixel(sky)
        inn = (x > 0) & (x < nx - 1) & (y > 0) & (y < ny - 1)
        detof[inn & (detof == '')] = det
    rows[band]['det'] = detof[R['ok']]
    for det in sorted(set(rows[band]['det']) - {''}):
        m = rows[band]['det'] == det
        if m.sum() < 20:
            continue
        print(f"| F{band} | {det} | {m.sum()} | {np.median(R['d'][m] - R['p'][m]):+.4f} | {np.median(R['s'][m]):+.4f} | {np.median(R['f'][m]):+.4f} | {np.median(R['h'][m]):+.4f} |")

# figure
bands = list(rows)
nb = len(bands)
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(5, 4)
for i, band in enumerate(bands[:16]):
    ax = fig.add_subplot(gs[i // 4 + 0, i % 4]) if i < 16 else None
    R = rows[band]
    y = R['d'] - R['p']
    ax.plot(R['s'], y - np.median(y), ',', color='k', alpha=0.3)
    # binned medians per detector
    for det in sorted(set(R['det']) - {''}):
        m = R['det'] == det
        if m.sum() >= 20:
            ax.plot(np.median(R['s'][m]), np.median(y[m]) - np.median(y), 'o', ms=5)
    ax.set_ylim(-0.3, 0.3)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_title(f'F{band}', fontsize=9)
    if i % 4 == 0:
        ax.set_ylabel('dm - pred (centred)')
    if i >= 12:
        ax.set_xlabel('fl + ph [mag]')
maps = sorted(glob.glob(f'{HERE}/map_{tag}_*.npy'))
for k, (want, pos) in enumerate([('NRCB1_200W', 0), ('NRCALONG_410M', 1)]):
    det, b = want.split('_')
    fn = f'{HERE}/map_{tag}_{det}_{b}.npy'
    ax = fig.add_subplot(gs[4, pos * 2:pos * 2 + 2])
    try:
        m = np.load(fn)
        im = ax.imshow(-2.5 * np.log10(1 / m), origin='lower', vmin=-0.05, vmax=0.05, cmap='RdBu_r')
        plt.colorbar(im, ax=ax, label='fl [mag]')
        ax.set_title(f'fl map {det} F{b}')
    except FileNotFoundError:
        ax.text(0.1, 0.5, f'{det} F{b}: flat identical in both contexts (fl = 0)', transform=ax.transAxes)
        ax.set_axis_off()
fig.tight_layout()
fig.savefig(f'{HERE}/flatver.png', dpi=100)
