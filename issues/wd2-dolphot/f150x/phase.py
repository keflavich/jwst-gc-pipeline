"""Per-frame pixel-phase test for the F150W excess scatter.

Per frame row (closure stars): q = -2.5 log10(f_X / ap3), f_X from recentre_ab
(main2 'oi', new loader 'ni'), ap3 the r = 3 px aperture sum with the AREA map applied
(apclosure frames).  The per-star mean over its frames is removed (dq), so star-level
terms (dolphot, crowding, colour) cancel and dq holds the per-frame variation only.
Reports rstd(dq), the binned median of dq against the fitted pixel phase
(px, py = x_fit - round(x_fit), ...) and against r_phase = hypot(px, py), and a
least-squares fit dq = a0 + a1 r_phase^2.
usage: python phase.py -> phase.txt, phase.png
"""
import glob
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.table import Table, join, vstack  # noqa: E402

Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
OUT = f'{Q}/f150x'


def rs(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x)))


L = []
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
for ax, b in zip(axes, ['150W', '200W']):
    R = vstack([Table.read(f) for f in sorted(glob.glob(f'{Q}/gridfix/recentre/rc_{b}_*.ecsv'))])
    P = Table.read(f'{Q}/apclosure/frames_{b}.ecsv')
    P = P[(P['src'] == 1) & (P['area3'] > 0)]['i', 'frame', 'area3', 'area5']
    R = R['i', 'frame', 'det', 'x_fit', 'y_fit', 'f_oi', 'f_ni', 'forced']
    J = join(R, P, keys=['i', 'frame'])
    J = J[~np.asarray(J['forced'], bool)]
    px = np.asarray(J['x_fit'] - np.round(J['x_fit']), float)
    py = np.asarray(J['y_fit'] - np.round(J['y_fit']), float)
    rp = np.hypot(px, py)
    ids, inv = np.unique(J['i'], return_inverse=True)
    n = np.bincount(inv)
    L.append(f'## F{b}: {len(J)} frame rows (no forced refit), {len(ids)} stars')
    for v in ('oi', 'ni'):
        q = -2.5 * np.log10(np.asarray(J[f'f_{v}'], float) / np.asarray(J['area3'], float))
        ok = np.isfinite(q)
        qm = np.bincount(inv[ok], weights=q[ok], minlength=len(ids)) / np.maximum(np.bincount(inv[ok], minlength=len(ids)), 1)
        dq = q - qm[inv]
        g = ok & (n[inv] >= 3)
        g &= np.abs(dq) < 5 * rs(dq[g])
        A = np.c_[np.ones(g.sum()), rp[g] ** 2]
        a = np.linalg.lstsq(A, dq[g], rcond=None)[0]
        res = dq[g] - A @ a
        L.append(f'  {v}: rstd dq {rs(dq[g]):.4f}  fit dq = {a[0]:+.4f} + {a[1]:+.4f} r^2  '
                 f'(rstd after {rs(res):.4f}); N={g.sum()}')
        edges = np.linspace(0, 0.71, 8)
        cen = 0.5 * (edges[1:] + edges[:-1])
        med = [np.median(dq[g & (rp >= lo) & (rp < hi)]) if np.sum(g & (rp >= lo) & (rp < hi)) > 20 else np.nan
               for lo, hi in zip(edges[:-1], edges[1:])]
        L.append('    r_phase bins ' + ' '.join(f'{c:.2f}:{m:+.4f}' for c, m in zip(cen, med)))
        for name, ph in (('|px|', np.abs(px)), ('|py|', np.abs(py))):
            e2 = np.linspace(0, 0.5, 6)
            mm = [np.median(dq[g & (ph >= lo) & (ph < hi)]) for lo, hi in zip(e2[:-1], e2[1:])]
            L.append(f'    {name} bins ' + ' '.join(f'{0.5 * (lo + hi):.2f}:{m:+.4f}' for lo, hi, m in zip(e2[:-1], e2[1:], mm)))
        if v == 'ni':
            ax.scatter(rp[g], dq[g], s=2, c='0.7', lw=0)
            ax.plot(cen, med, 'ko-', label='binned median (new loader)')
            xx = np.linspace(0, 0.71, 50)
            ax.plot(xx, a[0] + a[1] * xx ** 2, 'r-', label=f'fit {a[1]:+.4f} r$^2$')
            ax.set_title(f'F{b}: per-frame dq vs pixel phase\nrstd dq {rs(dq[g]):.4f}', fontsize=10)
            ax.set_xlabel('pixel phase radius hypot(px, py) [px]')
            ax.legend(fontsize=8)
axes[0].set_ylabel('dq = -2.5 log10(f_PSF / ap3) - star mean [mag]')
axes[0].set_ylim(-0.04, 0.04)
fig.savefig(f'{OUT}/phase.png', dpi=100)
open(f'{OUT}/phase.txt', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
