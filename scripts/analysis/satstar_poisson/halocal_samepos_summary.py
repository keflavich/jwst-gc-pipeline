"""fig16: fraction of an exposure-to-exposure PSF change that a DIFFERENT star reproduces,
vs how close the two stars' detector positions are (halocal_samepos.py results).

    python halocal_samepos_summary.py <samepos.json> <out.png>
"""
import sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from halocal_samepos import ANN, SEPB

res = json.load(open(sys.argv[1]))['NRCBLONG']['test1']
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
xs = [np.sqrt(max(a, 1)*b) for a, b in SEPB]
cols = plt.cm.plasma(np.linspace(0, 0.85, len(ANN)))
for p, (part, title) in enumerate([('halo', 'between the spikes'), ('spike', 'on the spikes')]):
    for c, (lo, hi) in zip(cols, ANN):
        y = []; n = []
        for s0, s1 in SEPB:
            v = res.get(f'diffvisit_noanchor_{part}_{lo}-{hi}_{s0}-{s1}')
            y.append(np.nan if v is None else v['ratio']); n.append(0 if v is None else v['n'])
        if np.all(np.isnan(y)):
            continue
        ax[p].plot(xs, y, 'o-', color=c, lw=2, label=f'r = {lo}-{hi} px')
        for x_, y_, n_ in zip(xs, y, n):
            if np.isfinite(y_):
                ax[p].annotate(str(n_), (x_, y_), fontsize=7, xytext=(3, 3), textcoords='offset points', color=c)
    ax[p].set_xscale('log'); ax[p].axhline(0, color='k', lw=.5); ax[p].axhline(1, color='k', lw=.5, ls=':')
    ax[p].set_ylim(-0.1, 1.0)
    ax[p].set_xticks(xs); ax[p].set_xticklabels([f'{a}-{b}' for a, b in SEPB])
    ax[p].set_xlabel('distance between the two stars\' detector positions [px]  (both exposures)')
    ax[p].set_ylabel('corr(D_A, D_B) / noise ceiling')
    ax[p].set_title(f'NRCB5, {title}: how much of star A\'s change\nstar B reproduces (1 = identical, 0 = unrelated)')
    ax[p].legend(fontsize=9)
fig.text(0.5, 0.005, 'D = difference of two exposures\' residuals, in fractional units; same dither move for both stars; anchor exposure excluded; numbers = quadruples',
         ha='center', fontsize=8)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(sys.argv[2], dpi=80)
