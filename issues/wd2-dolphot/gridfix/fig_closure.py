"""Detector maps of ours - dolphot before and after the grid-loader fix.

Top row: dm - pred (main2 as run, dolphot area double count removed).
Bottom row: dm + dfix - pred (new loader, closure.py 5x5 forced fit).
Each star is drawn once per detector at its first frame position; colour is
the per-star value minus the detector median, so the panels show the spatial
pattern; the detector median is printed in each title.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

D = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/gridfix'
PANELS = [('200W', 'nrca1'), ('200W', 'nrca3'), ('200W', 'nrcb2'), ('200W', 'nrcb4'),
          ('300M', 'nrcalong'), ('323N', 'nrcblong')]


def mad(x):
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x)))


fig, axes = plt.subplots(2, len(PANELS), figsize=(3.3 * len(PANELS), 7.2), constrained_layout=True)
for k, (b, det) in enumerate(PANELS):
    F = Table.read(f'{D}/frames_{b}.ecsv')
    S = Table.read(f'{D}/stars_{b}.ecsv')
    F = F[F['det'] == det]
    _, first = np.unique(F['i'], return_index=True)
    F = F[first]
    sidx = {int(i): j for j, i in enumerate(S['i'])}
    j = np.array([sidx[int(i)] for i in F['i']])
    dm, pr, df = np.asarray(S['dm'])[j], np.asarray(S['pred'])[j], np.asarray(S['dfix5'])[j]
    for row, (v, lab) in enumerate([(dm - pr, 'dm - pred (old loader)'),
                                    (dm + df - pr, 'dm + dfix - pred (new loader)')]):
        ax = axes[row, k]
        ok = np.isfinite(v)
        med = np.median(v[ok])
        sc = ax.scatter(F['x'][ok], F['y'][ok], c=v[ok] - med, s=7, cmap='RdBu_r', vmin=-0.05, vmax=0.05)
        ax.set_title(f'F{b} {det}\n{lab}\nrstd {mad(v):.4f}, median {med:+.3f}', fontsize=8.5)
        ax.set_xlim(0, 2048); ax.set_ylim(0, 2048); ax.set_aspect('equal')
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.set_ylabel('detector y [px]', fontsize=8)
        if row == 1:
            ax.set_xlabel('detector x [px]', fontsize=8)
cb = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.01)
cb.set_label('value - detector median [mag]  (positive: ours faint)')
fig.savefig(f'{D}/closure_maps.png', dpi=110)
print('wrote closure_maps.png')
