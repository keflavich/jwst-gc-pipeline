"""Figure: bright-end NB-broad colour and per-band dm against dolphot on stars common to prod/main2/main2kf.
Top: median (our - dolphot) NB-broad colour per dolphot NB-mag bin.  Bottom: per-band dm minus each arm's own
15-18 mag median (solid = narrowband, dashed = broad/medium).  usage: python fig_matched_colour.py OUT.png"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.table import Table
import analyze as an
PAIRS = [('164N', '162M'), ('187N', '182M'), ('212N', '200W'), ('405N', '410M'), ('466N', '410M')]
ARMS = {'prod': 'tab:gray', 'main2': 'tab:brown', 'main2kf': 'tab:pink'}
EDGES = np.arange(10, 18.01, 0.5)
ms = {a: Table.read(an.PATH[a][1]) for a in ARMS}


def binmed(x, y, s):
    out = np.full(len(EDGES) - 1, np.nan)
    for i in range(len(EDGES) - 1):
        k = s & (x >= EDGES[i]) & (x < EDGES[i + 1])
        if k.sum() >= 5:
            out[i] = np.median(y[k])
    return out


xc = 0.5 * (EDGES[1:] + EDGES[:-1])
fig, axes = plt.subplots(2, len(PAIRS), figsize=(4 * len(PAIRS), 7), sharex=True, sharey='row')
for j, (nb, bb) in enumerate(PAIRS):
    r1, r2 = an.fl(ms['prod']['ref_' + nb]), an.fl(ms['prod']['ref_' + bb])
    o = {a: (an.fl(m['our_' + nb]), an.fl(m['our_' + bb])) for a, m in ms.items()}
    com = np.isfinite(r1) & np.isfinite(r2) & np.all([np.asarray(m['matched'], bool) & np.isfinite(o[a][0]) & np.isfinite(o[a][1])
                                                     for a, m in ms.items()], axis=0)
    for a, col in ARMS.items():
        o1, o2 = o[a]
        axes[0, j].plot(xc, binmed(r1, (o1 - o2) - (r1 - r2), com), '-o', ms=3, color=col, label=a)
        for oo, rr, ls in ((o1, r1, '-'), (o2, r2, '--')):
            d = oo - rr
            d = d - np.median(d[com & (rr >= 15) & (rr < 18)])
            axes[1, j].plot(xc, binmed(r1, d, com), ls, marker='o', ms=3, color=col)
    axes[0, j].set_title(f'F{nb} − F{bb}  (N={com.sum()})')
    axes[1, j].set_xlabel(f'dolphot F{nb} [Vega mag]')
    for ax in axes[:, j]:
        ax.axhline(0, color='0.5', lw=0.8)
        ax.grid(alpha=0.3)
axes[0, 0].set_ylabel('(our − dolphot) colour [mag]')
axes[1, 0].set_ylabel('dm − own 15–18 mag median [mag]\nsolid: NB, dashed: broad/medium')
axes[0, 0].legend(fontsize=9)
axes[0, 0].set_ylim(-0.15, 0.35)
axes[1, 0].set_ylim(-0.3, 0.15)
fig.tight_layout()
fig.savefig(sys.argv[1], dpi=110)
