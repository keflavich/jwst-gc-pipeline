"""Figure for the F200W scale-correction validation (#1137): pixel-frame scale
of (F200W - anchor) per detector with the shipped rotation only, rotation +
scale, and rotation + opposite scale, on wd2 (vs F212N, the field the table
was partly derived from) and NGC 6334 (vs F187N, independent)."""
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DETS = ['NRCA1', 'NRCA2', 'NRCA3', 'NRCA4', 'NRCB1', 'NRCB2', 'NRCB3', 'NRCB4']
# wd2 F150W - F212N, rotation-corrected (#1137 table 1), for reference
F150W_WD2 = [8, -1, 2, 4, 10, -7, -8, -2]


def read(fn):
    rows = [l.split() for l in open(fn) if re.match(r'^nrc', l)]
    return np.array([[float(r[3]), float(r[5]), float(r[7])] for r in rows])


panels = [('val_wd2.txt', 'wd2: F200W − F212N (exp 1–2, visit 1; table input)'),
          ('val_ngc6778.txt', 'NGC 6334: F200W(6778) − F187N(6778) (exp 1–4, 3 visits; independent)')]
x = np.arange(len(DETS))
fig, axes = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True, constrained_layout=True)
for ax, (fn, title) in zip(axes, panels):
    v = read(fn)
    ax.axhline(0, color='0.6', lw=0.8, zorder=0)
    ax.grid(axis='y', color='0.9', lw=0.6)
    ax.plot(x - 0.12, v[:, 2], 'x', color='0.6', ms=7, mew=1.5,
            label=f'rotation + opposite scale (rms {np.sqrt(np.mean(v[:, 2] ** 2)):.0f} ppm)')
    ax.plot(x - 0.04, v[:, 0], 'o', color='#D55E00', mfc='none', ms=8, mew=1.5,
            label=f'rotation only (rms {np.sqrt(np.mean(v[:, 0] ** 2)):.0f} ppm)')
    ax.plot(x + 0.04, v[:, 1], 'o', color='#0072B2', ms=8,
            label=f'rotation + scale (rms {np.sqrt(np.mean(v[:, 1] ** 2)):.0f} ppm)')
    if fn == 'val_wd2.txt':
        ax.plot(x + 0.12, F150W_WD2, 's', color='#009E73', ms=6, mfc='none', mew=1.2,
                label=f'F150W − F212N, rotation only (rms {np.sqrt(np.mean(np.square(F150W_WD2))):.0f} ppm)')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.set_title(title, loc='left', fontsize=10)
    ax.set_ylabel('pixel-frame scale  [ppm]')
    ax.set_ylim(-48, 82)
    ax.legend(frameon=False, fontsize=8.5, loc='upper right', ncol=2)
axes[1].set_xticks(x, DETS)
fig.savefig('fig_scale_val.png', dpi=130)
