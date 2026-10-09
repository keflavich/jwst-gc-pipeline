"""fig_rstar.png: (R_header x g0 / flat) / cal for isolated unsaturated stars against peak group-0 level, per band and detector,
for the peak pixel, the PSF fit and the r < 5 px aperture; second row: PSF-fit ratio without the flat against the local flat."""
import sys
import glob
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/rstar'
BANDS = ['150W', '200W', '250M', '300M']
EDGES = np.array([0, 0.01, 0.02, 0.03, 0.045, 0.06, 0.08, 0.1, 0.15])
fig, axs = plt.subplots(2, 4, figsize=(17, 7.5))
for col, band in enumerate(BANDS):
    dets = ('nrcblong',) if band in ('250M', '300M') else ('nrcb1', 'nrcb3')
    ax, ax2 = axs[0, col], axs[1, col]
    for det, ls in zip(dets, ('-', '--')):
        R = []
        for fn in sorted(glob.glob(f'{D}/rstar_{band}_{det}_*.pkl')):
            d = pickle.load(open(fn, 'rb'))
            for r in d['rows']:
                r['Rh'] = d['meta']['Rh']
            R += d['rows']
        if not R:
            continue
        g = lambda k: np.array([r[k] for r in R], float)
        ok = (g('a_cal') > 0) & (g('a_gh') / g('ae_gh') > 30) & (g('ap5_cal') > 0)
        gfr = g('g0frac')
        Y = {'peak pixel': g('Rh') * g('peak_g0') / g('flat_pk') / g('peak_cal'), 'PSF fit': g('a_ghf') / g('a_cal'),
             'r < 5 px aperture': g('ap5_ghf') / g('ap5_cal')}
        for (lab, y), c in zip(Y.items(), ('C3', 'C0', 'C2')):
            xm, ym, ye = [], [], []
            for lo, hi in zip(EDGES[:-1], EDGES[1:]):
                s = ok & (gfr >= lo) & (gfr < hi)
                if s.sum() < 15:
                    continue
                xm.append(np.median(gfr[s]))
                ym.append(np.median(y[s]))
                ye.append(1.2533 * 1.4826 * np.median(np.abs(y[s] - np.median(y[s]))) / np.sqrt(s.sum()))
            ax.errorbar(xm, ym, ye, color=c, ls=ls, marker='o', ms=4, label=f'{lab}, {det}')
        fl = g('flat_w')
        rr = g('a_gh') / g('a_cal')
        ax2.scatter(fl[ok], rr[ok], s=2, alpha=0.3, color='C0' if ls == '-' else 'C1', label=det)
    xx = np.linspace(0.9, 1.08, 2)
    ax2.plot(xx, np.median(rr[ok] / fl[ok]) * xx, 'k:', label='ratio proportional to flat')
    ax.axhline(1, color='k', lw=0.8)
    ax.set_title(f'F{band}')
    ax.set_xlabel('peak group 0 / group-0 ceiling')
    ax.set_ylim(0.9, 1.06)
    ax2.set_xlim(0.9, 1.08)
    ax2.set_ylim(0.88, 1.1)
    ax2.set_xlabel('PSF$^2$-weighted flat over the fit pixels')
    ax.legend(fontsize=6.5, loc='lower right')
    ax2.legend(fontsize=7, loc='upper left', markerscale=4)
axs[0, 0].set_ylabel('(R_header g0 / flat) / cal')
axs[1, 0].set_ylabel('PSF fit (R_header g0) / cal, no flat')
fig.tight_layout()
fig.savefig(f'{D}/fig_rstar.png', dpi=105)
