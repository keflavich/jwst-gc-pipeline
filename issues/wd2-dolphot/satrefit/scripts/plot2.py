"""usage: python plot2.py OUT.png  -> 2x2 panels (F150W, F200W, F250M, F300M), running median dm per 0.5 mag.
Curves: final, uncapped, bgfree+v7b, uni+bgfree+v7b, best '+cap' variant (min |all-star median| + |trend| among non-wing '+cap' variants),
wing_b+cap (pipeline wing self-calibration emulated at the pre-ZF DQ SATURATED radius, gated as in the pipeline)."""
import pickle, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/out2'
bands = ['150W', '200W', '250M', '300M']
fig, axs = plt.subplots(2, 2, figsize=(13, 9.5))
best = {}
for ax, band in zip(axs.ravel(), bands):
    d = pickle.load(open(f'{Q}/score2_{band}.pkl', 'rb'))
    ref, have, dm, bins = d['ref'], d['have'], d['dm'], d['bins']
    lo, hi = bins[0][0], bins[-1][1]
    edges = np.arange(np.floor(lo * 2) / 2, min(hi, ref[have].max() + 0.5) + 0.01, 0.5)

    def rm(y, minn=8):
        xs, ys = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            s = have & (ref >= a) & (ref < b) & np.isfinite(y)
            if s.sum() >= minn:
                xs.append(np.median(ref[s]))
                ys.append(np.median(y[s]))
        return xs, ys

    def score(v):
        y = dm[v]
        meds = []
        for a, b in bins:
            s = have & (ref >= a) & (ref < b)
            if s.sum() >= 5:
                meds.append(np.median(y[s]))
        return abs(np.median(y[have])) + abs(meds[-1] - meds[0])
    cands = [k for k in dm if k.endswith('+cap') and 'wing' not in k]
    bv = min(cands, key=score)
    best[band] = bv
    ax.plot(*rm(dm['final']), 'k-o', ms=4, lw=2, label='final (capped)')
    ax.plot(*rm(dm['uncapped']), 'k--s', ms=4, mfc='none', lw=1.5, label='uncapped')
    ax.plot(*rm(dm['bgfree+v7b']), '-^', color='#1b9e77', ms=4, label='bgfree+v7b')
    ax.plot(*rm(dm['uni+bgfree+v7b']), '-v', color='#7570b3', ms=4, label='uni+bgfree+v7b')
    ax.plot(*rm(dm[bv]), '-D', color='#d95f02', ms=4, lw=2, label=f'best +cap: {bv}')
    ax.plot(*rm(dm['wing_b+cap']), ':', color='#e7298a', lw=2, label='wing_b+cap (gated)')
    ax.axhline(0, color='0.5', lw=0.8)
    ax.axhline(d['unsat_dm'], color='0.7', lw=0.8, ls='--')
    ax.set_title(f'F{band} (N={have.sum()})')
    ax.set_xlabel('dolphot magnitude')
    ax.set_ylabel('dm = ours - dolphot - ZP (mag)')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5)
fig.suptitle('Round 2: running median dm per 0.5 mag (gray dashed: unsaturated reference just below the satstar faint edge)', fontsize=10)
fig.tight_layout()
fig.savefig(sys.argv[1], dpi=110)
print(best)
