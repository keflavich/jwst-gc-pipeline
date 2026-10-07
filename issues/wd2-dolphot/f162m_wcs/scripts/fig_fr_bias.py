"""dm vs dolphot by forced_refit_frac for F150W, F162M, F164N, F182M, F212N
(cache from fr_bias.py), with running medians per forced-fraction class."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = np.load('fr_bias_cache.npy', allow_pickle=True).item()
BANDS = ['150W', '162M', '164N', '182M', '212N']
CLS = [('free (frac = 0)', lambda f: f == 0, '0.65', 1.5),
       ('0 < frac < 0.5', lambda f: (f > 0) & (f < 0.5), '#3b7dd8', 4),
       ('frac >= 0.5', lambda f: f >= 0.5, '#d1495b', 9)]
edges = np.arange(15, 23.01, 1.0)
fig, axes = plt.subplots(1, len(BANDS), figsize=(19, 4.6), sharey=True)
for ax, b in zip(axes, BANDS):
    ref, dm, frac, ok = d[b]
    for lab, fn, c, s in CLS:
        sel = ok & fn(frac)
        ax.scatter(ref[sel], dm[sel], s=s, c=c, alpha=0.45 if c == '0.65' else 0.8, lw=0,
                   rasterized=True, label=f'{lab} ({sel.sum()})')
        mids, meds = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            q = sel & (ref >= lo) & (ref < hi)
            if q.sum() >= 8:
                mids.append(0.5 * (lo + hi)); meds.append(np.median(dm[q]))
        if c != '0.65':
            ax.plot(mids, meds, '-o', c=c, ms=4, lw=2, mec='white', mew=0.8)
    ax.axhline(0, c='k', lw=0.7)
    ax.set_xlim(15, 23); ax.set_ylim(-1.5, 1.5)
    ax.set_title(f'F{b}', fontsize=10)
    ax.set_xlabel('dolphot Vega mag')
    ax.legend(fontsize=7, markerscale=2, loc='upper left', frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
axes[0].set_ylabel('ours - dolphot - ZP (mag)')
fig.suptitle('wd2 Q mainfcbg m8 vs dolphot, split by forced_refit_frac (lines: 1-mag running medians)', fontsize=10)
fig.tight_layout()
fig.savefig('fig_fr_bias.png', dpi=120)
