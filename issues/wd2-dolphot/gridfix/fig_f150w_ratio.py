"""Detector maps and lA scatter of do = -2.5 log10(flux_fit_main2 / fold5) for F150W and F200W.

fold5: closure.py forced fit on the crf with the old (transposed) loader, 5x5 box.
Both fits use the same grid file and the same position; F200W shows the expected
agreement, F150W does not.
"""
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.table import Table  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an  # noqa: E402

D = f'{an.Q}/gridfix'
DETS = ['nrca1', 'nrcb1', 'nrcb3']


def load(b):
    F = Table.read(f'{D}/frames_{b}.ecsv')
    F = F[F['src'] == 1]
    cats = {}
    for fn in glob.glob(f'{an.Q}/tree_main2/F{b}/f{b.lower()}_*_daophot_basic.fits'):
        t = Table.read(fn)
        cats[os.path.basename(t.meta['FILENAME'])] = t
    out = {}
    for fr in np.unique(F['frame']):
        G = F[F['frame'] == fr]
        cat = cats[fr]
        ct = cKDTree(np.c_[np.asarray(cat['x_fit'], float), np.asarray(cat['y_fit'], float)])
        d, j = ct.query(np.c_[G['x'], G['y']])
        ok = d < 1e-3
        G, j = G[ok], j[ok]
        fl = np.asarray(cat['flux_fit'], float)[j]
        o = out.setdefault(G['det'][0], dict(x=[], y=[], lA=[], do=[], fl=[]))
        o['x'] += list(G['x']); o['y'] += list(G['y']); o['lA'] += list(G['lA'])
        o['do'] += list(-2.5 * np.log10(fl / np.asarray(G['fold5']))); o['fl'] += list(fl)
    return {k: {kk: np.array(vv) for kk, vv in v.items()} for k, v in out.items()}


fig, axes = plt.subplots(2, 2 * len(DETS), figsize=(4.0 * len(DETS) * 2 / 1.6, 7), constrained_layout=True)
for row, b in enumerate(['150W', '200W']):
    O = load(b)
    for k, det in enumerate(DETS):
        o = O[det]
        ok = np.isfinite(o['do'])
        med = np.median(o['do'][ok])
        v = o['do'][ok] - med
        ax = axes[row, 2 * k]
        sc = ax.scatter(o['x'][ok], o['y'][ok], c=v, s=5, cmap='RdBu_r', vmin=-0.03, vmax=0.03)
        ax.set_xlim(0, 2048); ax.set_ylim(0, 2048); ax.set_aspect('equal')
        ax.set_title(f'F{b} {det}: main2 / forced(old)', fontsize=9)
        ax.tick_params(labelsize=7)
        ax = axes[row, 2 * k + 1]
        bright = o['fl'][ok] > np.percentile(o['fl'][ok], 67)
        ax.scatter(o['lA'][ok][~bright], v[~bright], s=3, c='0.6', label='fainter 2/3')
        ax.scatter(o['lA'][ok][bright], v[bright], s=3, c='C0', label='brightest 1/3')
        ax.plot([-0.04, 0.04], [0.04, -0.04], 'k--', lw=1, label='slope -1')
        ax.set_ylim(-0.06, 0.06); ax.set_xlim(-0.04, 0.04)
        ax.set_xlabel('lA [mag]', fontsize=8); ax.set_ylabel('do - median [mag]', fontsize=8)
        ax.tick_params(labelsize=7)
        if row == 0 and k == 0:
            ax.legend(fontsize=7, markerscale=3)
cb = fig.colorbar(sc, ax=axes[:, ::2], fraction=0.02, pad=0.01)
cb.set_label('do - detector median [mag]')
fig.savefig(f'{D}/f150w_ratio.png', dpi=100)
print('wrote f150w_ratio.png')
