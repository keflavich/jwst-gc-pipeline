"""nbdet.png: per band step per 0.5 mag bin (trend A, window 1.5) for nrcb1 / nrcb3 / other, ours and dolphot; plus ground-reference residual bins."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
nb = json.load(open('nbdet_nb_results.json'))
tr = json.load(open('thirdref_det_results.json'))
GROUPS = ['nrcb1', 'nrcb3', 'other']
COL = {'nrcb1': 'tab:blue', 'nrcb3': 'tab:red', 'other': 'tab:gray'}
fig, axs = plt.subplots(2, 3 + 1, figsize=(19, 8.5), squeeze=False)
for c, r in enumerate(nb):
    ed = np.array(r['edges'])
    xc = 0.5 * (ed[1:] + ed[:-1])
    for row, key in enumerate(('ours_bin', 'dol_bin')):
        ax = axs[row, c]
        for k, g in enumerate(GROUPS):
            a = np.array(r['results'][f'w1.5_A_{g}'][key], float)
            ax.errorbar(xc + (k - 1) * 0.04, a[:, 0], a[:, 1], fmt='o-', color=COL[g], capsize=2, label=g, ms=4)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_ylim(-0.3, 0.2)
        ax.set_xlabel(f"F{r['N']} (dolphot mag)")
        ax.set_ylabel('step in F%s-F%s (mag)' % (r['W'], r['N']))
        ax.set_title(f"{r['pair']} {'ours' if row == 0 else 'dolphot'} (linear trend, all-group fit)", fontsize=9)
        if c == 0 and row == 0:
            ax.legend(fontsize=8)
for row, cn in enumerate(('ours', 'dolphot')):
    ax = axs[row, 3]
    for tag, ls in (('F200W-Ks (H-Ks)', '-'), ('F150W-H (H-Ks)', ':')):
        for k, g in enumerate(GROUPS):
            rows = [x for x in tr[tag]['bins'][cn] if x[0] == k and x[2] >= 5 and np.isfinite(x[3])]
            if rows:
                lo = np.array([x[1] for x in rows]) + 0.25
                ax.errorbar(lo + (k - 1) * 0.03, [x[3] for x in rows], [x[4] for x in rows], fmt='o' + ls, color=COL[g], capsize=2, ms=3, label=f'{tag.split()[0]} {g}' if row == 0 else None)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylim(-0.5, 0.4)
    ax.set_xlabel('JWST mag (own catalogue)')
    ax.set_ylabel('saturated-star residual vs ground (mag)')
    ax.set_title(f'ground JHKs, {cn} (solid F200W-Ks, dotted F150W-H)', fontsize=9)
    if row == 0:
        ax.legend(fontsize=6, ncol=2)
fig.tight_layout()
fig.savefig('nbdet.png', dpi=75)
