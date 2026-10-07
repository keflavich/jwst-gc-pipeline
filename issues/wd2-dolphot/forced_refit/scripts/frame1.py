"""Per-frame panels around (X, Y) on one F277W frame: crf SCI, SATURATED DQ,
m7 satstar model, m7 satstar residual (= daophot input), m7 daophot model and
residual; per-frame m7 rows marked (forced refits in magenta).
usage: python frame1.py X Y HALF [DET EXP BAND]"""
import sys
import numpy as np
from astropy.io import fits
from astropy.table import Table
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
X, Y, H = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
det = sys.argv[4] if len(sys.argv) > 4 else 'nrcblong'
ex = sys.argv[5] if len(sys.argv) > 5 else '1'
b = sys.argv[6] if len(sys.argv) > 6 else '277W'
bl = 'f' + b.lower()
T = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/tree_mainfcbg'
P = f'{T}/F{b}/pipeline'
fr = f'jw03523005001_10101_0000{ex}_{det}_align_o005_crf'
dp = f'jw03523-o005_t001_nircam_clear-{bl}-{det}_visit001_vgroup10101_exp0000{ex}_resbgsub_m7_daophot_basic'


def img2d(fn):
    with fits.open(fn) as h:
        for hdu in h:
            if hdu.data is not None and np.ndim(hdu.data) == 2:
                return np.asarray(hdu.data, float)
    raise ValueError(fn)


sl = (slice(Y - H, Y + H + 1), slice(X - H, X + H + 1))
crf = fits.open(f'{P}/{fr}.fits')
sci = crf['SCI'].data[sl]
sat = (crf['DQ'].data[sl] & 2) > 0
panels = [('crf SCI', sci), ('SATURATED DQ', sat.astype(float)),
          ('m7 satstar model', img2d(f'{P}/{fr}_resbgsub_m7_satstar_model.fits')[sl]),
          ('m7 satstar residual', img2d(f'{P}/{fr}_resbgsub_m7_satstar_residual.fits')[sl]),
          ('m7 daophot model', img2d(f'{P}/{dp}_model.fits')[sl]),
          ('m7 daophot residual', img2d(f'{P}/{dp}_residual.fits')[sl])]
t = Table.read(f'{T}/F{b}/{bl}_{det}_visit001_vgroup10101_exp0000{ex}_resbgsub_m7_daophot_basic.fits')
xs, ys = np.asarray(t['x_fit'], float), np.asarray(t['y_fit'], float)
fo = np.asarray(t['forced_refit'], bool)
inn = (np.abs(xs - X) <= H) & (np.abs(ys - Y) <= H)
fig, axes = plt.subplots(1, 6, figsize=(24, 4.6))
ref = panels[0][1]
vmin, vmax = np.nanpercentile(ref, [5, 99.5])
for ax, (lab, im) in zip(axes, panels):
    if lab == 'SATURATED DQ':
        ax.imshow(im, origin='lower', cmap='gray', vmin=0, vmax=1)
    elif 'residual' in lab:
        v = np.nanpercentile(np.abs(im), 90)
        ax.imshow(im, origin='lower', cmap='RdBu_r', vmin=-v, vmax=v)
    else:
        ax.imshow(np.arcsinh(im / max(vmax / 50, 1e-3)), origin='lower', cmap='gray',
                  vmin=np.arcsinh(vmin / max(vmax / 50, 1e-3)), vmax=np.arcsinh(vmax / max(vmax / 50, 1e-3)))
    ax.scatter(xs[inn & ~fo] - X + H, ys[inn & ~fo] - Y + H, marker='x', s=30, c='cyan', lw=0.9)
    ax.scatter(xs[inn & fo] - X + H, ys[inn & fo] - Y + H, marker='x', s=50, c='magenta', lw=1.2)
    ax.set_title(lab, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f'F{b} {det} exp{ex} around x={X} y={Y}; per-frame m7 rows: cyan x free fit, magenta x forced refit ({np.sum(inn & fo)} of {inn.sum()})')
fig.tight_layout()
fig.savefig(f'frame_{b}_{det}_{ex}_{X}_{Y}.png', dpi=70)
for k in np.where(inn & fo)[0]:
    print(f'forced row x={xs[k]:.1f} y={ys[k]:.1f} flux={t["flux_fit"][k]:.0f} init={t["flux_init"][k]:.0f} '
          f'sci={sci[int(round(ys[k])) - Y + H, int(round(xs[k])) - X + H]:.1f} sat={sat[int(round(ys[k])) - Y + H, int(round(xs[k])) - X + H]}')
