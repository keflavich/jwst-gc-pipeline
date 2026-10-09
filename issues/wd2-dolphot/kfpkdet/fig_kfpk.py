"""main2kfpk vs main2kf figure: LW bright-end band-value fraction, dm of gained values, and F277W cutouts
(data, main2kf m7 residual, main2kfpk m7 residual) of stars that gain an F277W value."""
import sys
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.visualization import simple_norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an

Q = an.Q
an.ZPWIN.update(an.zp_windows())
K, P = an.Arm('main2kf'), an.Arm('main2kfpk')
fig = plt.figure(figsize=(17, 14))
gs = fig.add_gridspec(4, 8, width_ratios=[1, 1, 1, 0.15, 1, 1, 1, 0.15])
a = fig.add_subplot(gs[0, 0:3])
LW = ['250M', '277W', '300M', '335M', '410M']
E = [11, 13, 14, 15, 16, 17, 19]
for b, c in zip(LW, ['C0', 'C1', 'C2', 'C4', 'C5']):
    for A, ls, mk in ((K, '--', 'o'), (P, '-', 's')):
        fr = []
        for lo, hi in zip(E[:-1], E[1:]):
            s = A.matched & (A.ref[b] >= lo) & (A.ref[b] < hi)
            fr.append(np.isfinite(A.our[b][s]).mean() if s.sum() >= 10 else np.nan)
        a.plot(0.5 * (np.array(E[:-1]) + np.array(E[1:])), fr, color=c, ls=ls, marker=mk, ms=4,
               label=f'F{b} {"main2kf" if A is K else "main2kfpk"}')
a.set_xlabel('dolphot band mag'); a.set_ylabel('matched dolphot stars with a band value')
a.set_title('(a) band-value fraction; dashed main2kf, solid main2kfpk', fontsize=9); a.legend(fontsize=6, ncol=2)
a = fig.add_subplot(gs[0, 3:5])
both = K.matched & P.matched
for b, c in zip(LW, ['C0', 'C1', 'C2', 'C4', 'C5']):
    g = both & ~np.isfinite(K.dm(b)) & np.isfinite(P.dm(b)) & np.isfinite(P.ref[b])
    a.hist(np.clip(P.dm(b)[g], -0.6, 0.6), bins=np.linspace(-0.6, 0.6, 49), histtype='step', color=c, label=f'F{b} (N {g.sum()})')
a.set_xlabel('dm of gained values, main2kfpk (mag)'); a.set_title('(b) values gained by main2kfpk (main2kf NaN)', fontsize=9); a.legend(fontsize=7)
a = fig.add_subplot(gs[0, 5:8])
for b, c in zip(LW, ['C0', 'C1', 'C2', 'C4', 'C5']):
    g = both & ~np.isfinite(K.dm(b)) & np.isfinite(P.dm(b)) & np.isfinite(P.ref[b])
    a.plot(P.ref[b][g], P.dm(b)[g], '.', color=c, ms=3, label=f'F{b}')
a.axhline(0, color='gray', lw=0.7); a.set_ylim(-0.8, 0.8); a.set_xlabel('dolphot band mag'); a.set_ylabel('dm')
a.set_title('(c) gained values against dolphot magnitude', fontsize=9)
# cutouts
b = '277W'
g = np.where(both & ~np.isfinite(K.dm(b)) & np.isfinite(P.dm(b)) & np.isfinite(P.ref[b]))[0]
rng = np.random.default_rng(3)
pick = rng.choice(g, 6, replace=False)
pick = pick[np.argsort(P.ref[b][pick])]
pre = 'jw03523-o005_t001_nircam_clear-f277w-merged'
dat = fits.open(f'{Q}/tree_main2kfpk/F277W/pipeline/{pre}_i2d.fits')
im = dat['SCI'].data
w = WCS(dat['SCI'].header)
rk = fits.open(f'{Q}/tree_main2kf/F277W/pipeline/{pre}_resbgsub_m7_daophot_basic_mergedcat_residual_i2d.fits')
rp = fits.open(f'{Q}/tree_main2kfpk/F277W/pipeline/{pre}_resbgsub_m7_daophot_basic_mergedcat_residual_i2d.fits')
wr = WCS(rk['SCI'].header)
assert wr.pixel_shape == WCS(rp['SCI'].header).pixel_shape
for k, i in enumerate(pick):
    sk = P.sky[P.idx[i]]
    co = Cutout2D(im, sk, 3.0 * u.arcsec, wcs=w, mode='partial')
    ck = Cutout2D(rk['SCI'].data, sk, 3.0 * u.arcsec, wcs=wr, mode='partial')
    cp = Cutout2D(rp['SCI'].data, sk, 3.0 * u.arcsec, wcs=wr, mode='partial')
    v = co.data[np.isfinite(co.data)]
    nm = simple_norm(co.data, 'asinh', vmin=np.percentile(v, 1), vmax=np.percentile(v, 99.7))
    vr = np.nanpercentile(np.abs(cp.data), 99)
    row, col = 1 + k // 2, (k % 2) * 4
    for j, (cut, tt) in enumerate(((co, 'data'), (ck, 'main2kf resid'), (cp, 'main2kfpk resid'))):
        a = fig.add_subplot(gs[row, col + j])
        a.imshow(cut.data, origin='lower', cmap='gray', norm=nm if j == 0 else None,
                 vmin=None if j == 0 else -vr, vmax=None if j == 0 else vr)
        x, y = cut.wcs.world_to_pixel(sk)
        a.plot(x, y, 'o', mfc='none', mec='r', ms=14, mew=1.2)
        a.set_xticks([]); a.set_yticks([])
        if j == 0:
            a.set_title(f'dolphot F277W {P.ref[b][i]:.2f}', fontsize=8)
        elif j == 1:
            a.set_title(f'{tt}: no value', fontsize=8)
        else:
            a.set_title(f'{tt}: dm {P.dm(b)[i]:+.3f}', fontsize=8)
fig.suptitle('main2kfpk (#1140 peaks, DAOPHOT_HANDOFF_PEAK_MIN_AREA=50) vs main2kf. Cutouts 3" on the F277W merged i2d; residuals share the main2kfpk scale; red circle = dolphot star',
             fontsize=10)
fig.tight_layout()
fig.savefig(f'{Q}/kfpkdet/kfpk_ab.png', dpi=85)
