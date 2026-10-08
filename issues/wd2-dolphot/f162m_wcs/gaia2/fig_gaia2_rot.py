"""Figure: m per detector and pooled beta, rows = wd2 unsaturated-only, wd2 incl. satstar rows (F150W), wd1 unsaturated-only."""
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
K, RAD = 62.0, 206264.806
DETS = [f'nrc{m}{i}' for m in 'ab' for i in range(1, 5)]
rows = [('wd2_q03s10', 'wd2, unsaturated rows only (qfit<=0.3, SNR>=10)', ['f212n', 'f187n', 'f164n']),
        ('wd2_q01s20_sat', 'wd2, satstar rows included in J fit (qfit<=0.1 SNR>=20 + satstar qfit<=0.6)', ['f212n', 'f150w', 'f164n']),
        ('wd1_q03s10', 'wd1, unsaturated rows only (qfit<=0.3, SNR>=10)', ['f212n', 'f187n', 'f164n', 'f115w', 'f200w'])]
cols = {'f212n': '#1f77b4', 'f187n': '#17becf', 'f182m': '#2ca02c', 'f150w': '#d95f02', 'f164n': '#e7298a', 'f115w': '#7570b3', 'f200w': '#a6761d'}
fig, axs = plt.subplots(3, 2, figsize=(14, 12.5), gridspec_kw=dict(width_ratios=[3, 1.3]))
for r, (tag, title, bands) in enumerate(rows):
    P = pickle.load(open(f'summary_{tag}.pkl', 'rb'))
    S, BETA, E = P['S'], P['BETA'], P['E']
    ax = axs[r, 0]
    n = len(bands); wd = 0.8 / n
    for k, b in enumerate(bands):
        for i, d in enumerate(DETS):
            if (b, d) not in S:
                continue
            s = S[(b, d)]
            lo, hi = np.percentile(s['mb'], [16, 84])
            x = i + (k - (n - 1) / 2) * wd
            ax.bar(x, s['m'], wd, color=cols[b], alpha=0.85, label=b.upper() if i == 0 else None)
            ax.errorbar(x, s['m'], yerr=[[max(s['m'] - lo, 0)], [max(hi - s['m'], 0)]], color='k', lw=1, capsize=2)
    for i, d in enumerate(DETS):
        if ('f150w', d) in E:
            ax.hlines(np.hypot(*E[('f150w', d)]) / K * RAD, i - 0.45, i + 0.45, color='k', ls='--', lw=1.5,
                      label='expected |ref(F150W) - ref(F212N)|' if i == 0 else None)
    ax.set_xticks(range(len(DETS))); ax.set_xticklabels([d.upper() for d in DETS])
    ax.set_ylim(0, 60); ax.set_ylabel('m [arcsec]')
    ax.set_title(title + ' (whiskers: bootstrap 16-84%)', fontsize=9)
    ax.legend(fontsize=7, loc='upper right', ncol=3)
    ax = axs[r, 1]
    labs, ys, es, cc = [], [], [], []
    for b in bands:
        for g in ['A2+A3', 'A1-A4']:
            if (b, g) in BETA:
                labs.append(f'{b.upper()}\n{g}'); ys.append(BETA[(b, g)][0]); es.append(BETA[(b, g)][1]); cc.append(cols[b])
    for j in range(len(ys)):
        ax.errorbar(j, ys[j], yerr=es[j], fmt='o', color=cc[j], capsize=3)
    ax.axhline(0, color='gray', lw=0.8); ax.axhline(1, color='#d95f02', ls=':', lw=1); ax.axhline(-1, color='#1f77b4', ls=':', lw=1)
    ax.set_ylim(-1.6, 2.0)
    ax.set_xticks(range(len(ys))); ax.set_xticklabels(labs, fontsize=6); ax.set_xlim(-0.5, len(ys) - 0.5)
    ax.set_ylabel('beta (projection onto E)'); ax.set_title('Module A pooled projection', fontsize=9)
    ax.text(0.02, 0.03, 'F212N-group bands: 0 = F212N refs right, -1 = F150W refs right\nF150W-group bands: +1 = F212N refs right, 0 = F150W refs right', transform=ax.transAxes, fontsize=5.5)
fig.tight_layout(); fig.savefig('fig_gaia2_rot.png', dpi=130)
