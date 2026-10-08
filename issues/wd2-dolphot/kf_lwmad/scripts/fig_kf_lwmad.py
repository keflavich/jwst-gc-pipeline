"""dm vs dolphot mag for stars satstar-replaced in both main2 (KEEP_FINITE off) and main2kf (on), LW bands and F323N.
usage: python fig_kf_lwmad.py OUT.png"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import analyze as an
an.ZPWIN.update(an.zp_windows())
A, B = an.Arm('main2'), an.Arm('main2kf')
BANDS = ['250M', '277W', '300M', '410M', '323N']
EDGES = np.arange(11, 18.51, 0.5)
xc = 0.5 * (EDGES[1:] + EDGES[:-1])
fig, axes = plt.subplots(1, len(BANDS), figsize=(4 * len(BANDS), 4), sharey=True)
for ax, b in zip(axes, BANDS):
    s = A.matched & B.matched & A.rep[b] & B.rep[b] & np.isfinite(A.ref[b])
    for arm, col, lab in ((A, 'tab:brown', 'main2 (off)'), (B, 'tab:pink', 'main2kf (on)')):
        d = arm.dm(b)
        ax.plot(A.ref[b][s], d[s], '.', ms=2, alpha=0.25, color=col)
        med = [np.median(d[s & (A.ref[b] >= lo) & (A.ref[b] < hi)]) if (s & (A.ref[b] >= lo) & (A.ref[b] < hi)).sum() >= 5 else np.nan
               for lo, hi in zip(EDGES[:-1], EDGES[1:])]
        ax.plot(xc, med, '-o', ms=4, color=col, label=lab)
    ax.axhline(0, color='0.5', lw=0.8)
    ax.set_title(f'F{b}: replaced in both arms (N={s.sum()})')
    ax.set_xlabel(f'dolphot F{b} [Vega mag]')
    ax.grid(alpha=0.3)
axes[0].set_ylabel('dm − band ZP [mag]')
axes[0].set_ylim(-0.25, 0.25)
axes[0].legend(fontsize=9)
fig.tight_layout()
fig.savefig(sys.argv[1], dpi=110)
