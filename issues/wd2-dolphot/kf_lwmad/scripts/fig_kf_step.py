"""Per-frame KEEP_FINITE on/off satstar flux change vs dolphot mag, with and without the recovered-core cap
(flux_fit vs flux_fit_precap), and the capped fraction per arm.  Inputs: the replay tables of kf_step_diag.py /
kf_step_peak.py.  usage: python fig_kf_step.py OUT.png"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.table import Table
EDGES = np.arange(12, 17.51, 0.5)
xc = 0.5 * (EDGES[1:] + EDGES[:-1])


def binmed(x, y):
    return np.array([np.median(y[(x >= lo) & (x < hi)]) if ((x >= lo) & (x < hi)).sum() >= 10 else np.nan
                     for lo, hi in zip(EDGES[:-1], EDGES[1:])])


fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True, sharey='row', gridspec_kw={'height_ratios': [2, 1]})
for j, b in enumerate(['F250M', 'F277W']):
    s = Table.read(f'kf_step_stars_{b}_withref.fits')
    p = Table.read(f'kf_step_peak_{b}.fits')
    assert np.allclose(s['x'], p['x']) and np.allclose(s['y'], p['y']) and np.all(s['frame'] == p['frame'])
    ref = np.asarray(s['dmag_ref'], float)
    ok = np.isfinite(ref) & (p['f_off'] > 0) & (p['f_on'] > 0) & (p['pre_off'] > 0) & (p['pre_on'] > 0)
    r = ref[ok]
    fin = -2.5 * np.log10(np.asarray(p['f_on'])[ok] / np.asarray(p['f_off'])[ok])
    pre = -2.5 * np.log10(np.asarray(p['pre_on'])[ok] / np.asarray(p['pre_off'])[ok])
    ax = axes[0, j]
    ax.plot(xc, binmed(r, fin), '-o', color='tab:pink', label='flux_fit (cap applied)')
    ax.plot(xc, binmed(r, pre), '--s', color='tab:purple', label='flux_fit_precap (no cap)')
    ax.axhline(0, color='0.5', lw=0.8)
    ax.set_title(f'{b} (N={ok.sum()})')
    ax.grid(alpha=0.3)
    ax = axes[1, j]
    for col, fo, pr, lab in (('tab:brown', 'f_off', 'pre_off', 'off'), ('tab:pink', 'f_on', 'pre_on', 'on')):
        capped = (np.asarray(p[fo])[ok] < 0.999 * np.asarray(p[pr])[ok]).astype(float)
        ax.plot(xc, binmed(r, capped) if False else [capped[(r >= lo) & (r < hi)].mean() if ((r >= lo) & (r < hi)).sum() >= 10 else np.nan
                                                    for lo, hi in zip(EDGES[:-1], EDGES[1:])], '-o', color=col, label=f'KEEP_FINITE {lab}')
    ax.set_xlabel(f'dolphot {b} [Vega mag]')
    ax.grid(alpha=0.3)
axes[0, 0].set_ylabel('on − off [mag]')
axes[1, 0].set_ylabel('fraction capped')
axes[0, 0].legend(fontsize=8)
axes[1, 0].legend(fontsize=8)
fig.suptitle('Replay of nrcblong exp 1-4: satstar rows in both arms, KEEP_FINITE on vs off')
fig.tight_layout()
fig.savefig(sys.argv[1], dpi=110)
