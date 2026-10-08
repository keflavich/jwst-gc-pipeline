"""usage: python plot3.py OUT.png -> row 1: running median dm per 0.5 mag; row 2: per-star amplitude ratio variant/base (running median) and rewritten fraction."""
import pickle, sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/out3'
bands = [b for b in ['150W', '200W', '250M', '300M'] if os.path.exists(f'{Q}/score3_{b}.pkl')]
fig, axs = plt.subplots(3, len(bands), figsize=(4.6 * len(bands), 12), squeeze=False)
for k, band in enumerate(bands):
    d = pickle.load(open(f'{Q}/score3_{band}.pkl', 'rb'))
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
    ax = axs[0, k]
    for nm, st in (('final', 'k-o'), ('uncapped', 'k--s'), ('bgfree', 'c-^'), ('bgfree+v7b+cap', 'g-D'), ('rw12', 'b-v'), ('rw12+bgfree', 'm-v'),
                   ('rw12+bgfree+cap', 'r-D'), ('rwinf+bgfree+cap', 'y:D')):
        ax.plot(*rm(dm[nm]), st, ms=4, label=nm)
    ax.axhline(0, color='0.5', lw=0.8)
    ax.axhline(d['unsat_dm'], color='0.7', lw=0.8, ls='--')
    ax.set_title(f'F{band} (N={have.sum()})')
    ax.set_ylabel('dm = ours - dolphot - ZP')
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.25)
    ax = axs[1, k]
    for nm, st in (('rw3', 'b-'), ('rw12', 'g-'), ('rwinf', 'k-'), ('rw12+bgfree', 'm--'), ('rwinf+bgfree', 'r--')):
        ax.plot(*rm(d['ratio'][nm]), st, label=nm)
    ax.axhline(1, color='0.5', lw=0.8)
    ax.set_ylabel('a_variant / a_base (uncapped)')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    ax = axs[2, k]
    for nm, st in (('rw3', 'b-'), ('rw6', 'c-'), ('rw12', 'g-'), ('rw20', 'y-'), ('rwinf', 'k-')):
        ax.plot(*rm(d['frac'][nm]), st, label=nm)
    ax.set_ylabel('fraction of fit pixels rewritten')
    ax.set_xlabel('dolphot magnitude')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
fig.suptitle('Round 3: charge-migration rewrite R(g0)*g0 within D px of DQ SATURATED (running medians per 0.5 mag)', fontsize=10)
fig.tight_layout()
fig.savefig(sys.argv[1], dpi=100)
