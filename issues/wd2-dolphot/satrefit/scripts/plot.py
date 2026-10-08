"""usage: python plot.py OUT.png BAND:v1,v2,v3 [BAND:...]"""
import pickle, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/out'
specs = [a.split(':') for a in sys.argv[2:]]
fig, axs = plt.subplots(1, len(specs), figsize=(6.2 * len(specs), 5), squeeze=False)
cols = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e']
for ax, (band, vs) in zip(axs[0], specs):
    d = pickle.load(open(f'{Q}/score_{band}.pkl', 'rb'))
    ref, have = d['ref'], d['have']
    lo, hi = d['bins'][0][0], d['bins'][-1][1]
    edges = np.arange(lo, hi + 0.01, 0.5)

    def rm(y):
        xs, ys = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            s = have & (ref >= a) & (ref < b) & np.isfinite(y)
            if s.sum() >= 8:
                xs.append(np.median(ref[s]))
                ys.append(np.median(y[s]))
        return xs, ys
    ax.plot(*rm(d['dm']['final']), 'k-o', ms=4, lw=2, label='final (capped)')
    ax.plot(*rm(d['dm']['uncapped']), 'k--s', ms=4, mfc='none', lw=1.5, label='baseline uncapped')
    for c, v in zip(cols, vs.split(',')):
        ax.plot(*rm(d['dm'][v]), '-^', color=c, ms=4, lw=1.5, label=v)
    ax.axhline(0, color='0.5', lw=0.8)
    ax.set_title(f'F{band} (N={have.sum()})')
    ax.set_xlabel('dolphot magnitude')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
axs[0, 0].set_ylabel('dm = ours - dolphot - ZP (mag)')
fig.suptitle('Satstar-replaced matched stars: running median dm per 0.5 mag, refit variants (main2)', fontsize=10)
fig.tight_layout()
fig.savefig(sys.argv[1], dpi=110)
