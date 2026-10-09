"""Cutouts of F150W forced-refit rows whose seed moved between dvx0 and dvx1 (#1128).

Rows are matched per frame by mutual nearest (x_fit, y_fit) within 1.5 px; a row is shown
if it is forced in both arms and its seed moved by > 0.2 px.  The panels show the crf SCI
cutout (asinh stretch) with the dvx0 seed (red x, production) and the dvx1 seed
(cyan +, DVA-consistent).  NaN pixels (saturated cores, DQ-masked) are drawn in gold.
Picks 12 rows spread over the dvx0 flux range.
"""
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.table import Table, vstack  # noqa: E402
from astropy.visualization import AsinhStretch, ImageNormalize  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

H = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/dvafix'
R = '/orange/adamginsburg/jwst/wd2/F150W/pipeline'
rows = []
for f0 in sorted(glob.glob(f'{H}/tree_dvx0/F150W/*_resbgsub_m7_daophot_basic.fits')):
    name = os.path.basename(f0)
    f1 = f'{H}/tree_dvx1/F150W/{name}'
    if not os.path.exists(f1):
        continue
    t0, t1 = Table.read(f0), Table.read(f1)
    p0, p1 = np.c_[t0['x_fit'], t0['y_fit']], np.c_[t1['x_fit'], t1['y_fit']]
    g0, g1 = np.all(np.isfinite(p0), 1), np.all(np.isfinite(p1), 1)
    ia, ib = np.flatnonzero(g0), np.flatnonzero(g1)
    d01, j01 = cKDTree(p1[g1]).query(p0[g0])
    _, j10 = cKDTree(p0[g0]).query(p1[g1])
    mut = (j10[j01] == np.arange(len(j01))) & (d01 < 1.5)
    a, b = t0[ia[mut]], t1[ib[j01[mut]]]
    F = np.asarray(a['forced_refit'], bool) & np.asarray(b['forced_refit'], bool)
    sh = np.hypot(a['x_init'] - b['x_init'], a['y_init'] - b['y_init']) > 0.2
    k = F & sh & (a['flux_fit'] > 0) & (b['flux_fit'] > 0)
    det, exp = name.split('_')[1], name.split('_')[4][-1]
    crf = f'{R}/jw03523005001_10101_0000{exp}_{det}_align_o005_crf.fits'
    rows.append(Table({'crf': [crf] * k.sum(), 'x0': a['x_init'][k], 'y0': a['y_init'][k],
                       'x1': b['x_init'][k], 'y1': b['y_init'][k],
                       'f0': a['flux_fit'][k], 'f1': b['flux_fit'][k]}))
T = vstack(rows)
T.sort('f0')
T.write(f'{H}/forced_moved.ecsv', overwrite=True)
dm = -2.5 * np.log10(T['f1'] / T['f0'])
print(f'N forced-in-both with moved seed: {len(T)}; dm(1-0) median {np.median(dm):+.3f}')
pick = np.unique(np.linspace(len(T) // 10, len(T) - 1, 12).astype(int))
fig, axes = plt.subplots(3, 4, figsize=(12, 9.5), constrained_layout=True)
hw = 8
for ax, i in zip(axes.flat, pick):
    r = T[i]
    sci = fits.getdata(r['crf'], 'SCI')
    xc, yc = int(round(r['x1'])), int(round(r['y1']))
    cut = sci[yc - hw:yc + hw + 1, xc - hw:xc + hw + 1]
    norm = ImageNormalize(cut, vmin=np.nanpercentile(cut, 5), vmax=np.nanpercentile(cut, 99.7),
                          stretch=AsinhStretch(0.05))
    ext = (xc - hw - 0.5, xc + hw + 0.5, yc - hw - 0.5, yc + hw + 0.5)
    cmap = plt.get_cmap('gray_r').copy()
    cmap.set_bad('gold')
    ax.imshow(cut, origin='lower', cmap=cmap, norm=norm, extent=ext)
    ax.plot(r['x0'], r['y0'], 'x', c='red', ms=11, mew=2, label='dvx0 seed (production)')
    ax.plot(r['x1'], r['y1'], '+', c='cyan', ms=13, mew=2, label='dvx1 seed (DVA-consistent)')
    d = np.hypot(r['x0'] - r['x1'], r['y0'] - r['y1'])
    ax.set_title(f"{os.path.basename(r['crf'])[26:31]} ({xc},{yc})  shift {d:.2f} px\n"
                 f"dvx1 - dvx0 = {-2.5 * np.log10(r['f1'] / r['f0']):+.2f} mag", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
axes.flat[0].legend(fontsize=7, loc='lower left')
fig.suptitle('F150W m7 forced refits: production seed vs DVA-consistent seed '
             f'(N={len(T)}, median dvx1 - dvx0 = {np.median(dm):+.2f} mag; gold = NaN pixels)', fontsize=11)
fig.savefig(f'{H}/forced_cutouts.png', dpi=90)
print('wrote forced_cutouts.png')
