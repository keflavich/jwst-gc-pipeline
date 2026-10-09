"""main2 vs main2gt full-chain figure: (a) robust std of dm per band; (b,c) F300M dm vs pred, binned.
usage: python analyze_g.py gridfix/chain/fig_chain.py out.png"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import analyze as an
out = sys.argv[1]
an.ZPWIN.update(an.zp_windows())
A, B = an.Arm('main2'), an.Arm('main2gt')
P = np.load(f'{an.Q}/photomver/areapred_main2.npz')
PT = np.load(f'{an.Q}/psfsum/predT_main2.npz')
mad = an.mad


def sel(b):
    lo, hi = an.ZPWIN.get(b, (0, 19))
    ok = A.matched & B.matched & np.isfinite(P[b]) & np.isfinite(PT[b])
    for X in (A, B):
        ok &= np.isfinite(X.dm(b)) & ~X.rep[b] & ~X.sat[b]
    return ok & (A.ref[b] >= lo) & (A.ref[b] < hi)


rows = []
for b in an.BANDS:
    ok = sel(b)
    rows.append((b, mad(A.dm(b)[ok]), mad(B.dm(b)[ok]), mad(A.dm(b)[ok] - P[b][ok]), mad(B.dm(b)[ok] - P[b][ok])))
fig = plt.figure(figsize=(13, 4.6))
ax = fig.add_axes([0.05, 0.14, 0.42, 0.76])
x = np.arange(len(rows))
ax.plot(x, [r[1] for r in rows], 'o-', color='#8a8a8a', lw=1.5, ms=6, label='main2: dm')
ax.plot(x, [r[2] for r in rows], 'o-', color='#2a6fdb', lw=1.5, ms=6, label='main2gt (grid fix): dm')
ax.plot(x, [r[4] for r in rows], 's--', color='#d9822b', lw=1.5, ms=6, label='main2gt: dm − pred (dolphot area term removed)')
ax.set_xticks(x, [f'F{r[0]}' for r in rows], rotation=60, fontsize=8)
ax.set_ylabel('robust σ(ours − dolphot) [mag]')
ax.set_title('(a) unsaturated stars in the ZP window, matched in both arms', fontsize=9)
ax.set_ylim(0, None); ax.grid(axis='y', color='#e5e5e5'); ax.legend(fontsize=8, frameon=False, loc='upper left')
b = '300M'; ok = sel(b)
p = P[b][ok]
edges = np.quantile(p, np.linspace(0, 1, 13))
for i, (X, name, col) in enumerate(((A, 'main2', '#8a8a8a'), (B, 'main2gt', '#2a6fdb'))):
    axp = fig.add_axes([0.54 + i * 0.235, 0.14, 0.2, 0.76])
    d = X.dm(b)[ok]
    axp.scatter(p, d, s=2, color=col, alpha=0.25, rasterized=True)
    k = np.clip(np.digitize(p, edges) - 1, 0, 11)
    xm = [np.median(p[k == j]) for j in range(12)]; ym = [np.median(d[k == j]) for j in range(12)]
    axp.plot(xm, ym, 'o-', color='k', ms=4, lw=1.2, label='binned median')
    xx = np.linspace(p.min(), p.max(), 2); c0 = np.median(d) - np.median(p)
    axp.plot(xx, xx + c0, ':', color='#d9822b', lw=1.5, label='slope 1')
    axp.plot(xx, 2 * xx + np.median(d) - 2 * np.median(p), '--', color='#b0b0b0', lw=1, label='slope 2')
    axp.set_ylim(-0.15, 0.15); axp.set_xlabel('pred = 2.5 log10 pixel-area term [mag]')
    if i == 0:
        axp.set_ylabel('F300M dm [mag]')
    else:
        axp.set_yticklabels([])
    axp.set_title(f'({"bc"[i]}) F300M {name}: σ {mad(d):.4f}', fontsize=9)
    axp.legend(fontsize=7, frameon=False, loc='lower right'); axp.grid(color='#eeeeee')
fig.savefig(out, dpi=130)
for r in rows:
    print(f'F{r[0]} {r[1]:.4f} {r[2]:.4f} {r[3]:.4f} {r[4]:.4f}')
