"""fig3: median dm - reference against the depth of the cap trim, per band, for the hard cap, no cap and the 0.96 floor."""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/capbind')
from cb_lib import Band
from an4 import bind_info, REFV

TRIM = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.30)]
fig, axs = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
for ax, band in zip(axs, ('150W', '200W', '250M', '300M')):
    B = Band(band)
    cap0, _ = bind_info(B, 'cutH0')
    a = B.a_H_h0_bgfree
    c = cap0 * B.rcor
    c = np.where(np.isfinite(c), c, np.inf)
    acap = np.minimum(a, c)
    with np.errstate(invalid='ignore', divide='ignore'):
        trim = B.med_per_star(1 - acap / a)
    D = {'hard cap (round 7)': B.dm_of(acap), 'no cap': B.dm_of(a),
         'floor tau = 0.96': B.dm_of(np.maximum(acap, 0.96 * a))}
    have = B.have0 & np.isfinite(trim)
    for d in D.values():
        have &= np.isfinite(d)
    x = np.array([0.5 * (lo + hi) for lo, hi in TRIM])
    ns = [int((have & (trim >= lo) & (trim < hi)).sum()) for lo, hi in TRIM]
    for (lab, d), col, mk in zip(D.items(), ('C1', 'C0', 'C2'), ('o', 's', '^')):
        y = [np.median(d[have & (trim >= lo) & (trim < hi)]) - REFV[band] for lo, hi in TRIM]
        ax.plot(x, y, marker=mk, color=col, label=lab)
    for k, (xi, n) in enumerate(zip(x, ns)):
        ax.annotate(f'N={n}', (xi, 0.17 - 0.025 * (k % 2)), ha='center', fontsize=8)
    ax.axhline(0, color='k', ls=':')
    ax.axhspan(-0.02, 0.02, color='0.9', zorder=0)
    ax.set_title(f'F{band} (H+h0+bgfree)')
    ax.set_xlabel('per-star median cap trim, 1 - min(a, cap)/a')
axs[0].set_ylabel('median dm - unsaturated reference')
axs[0].set_ylim(-0.3, 0.19)
axs[0].legend(loc='lower left', fontsize=8)
fig.tight_layout()
fig.savefig(sys.argv[1], dpi=110)
