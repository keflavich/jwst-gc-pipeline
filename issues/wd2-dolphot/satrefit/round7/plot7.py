"""Round 7 figure (#1142 / #1148): median dm (ours - dolphot - ZP) per reference-magnitude bin of the satstar-refit
stars, for the header rim (H), the pedestal-aware rims (HBm: B from 500-2000 DN far field, HBs: B from the sky
window) and the wing / background terms, against the unsaturated-reference dm of each band (dashed)."""
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
VAR = [('final', 'pipeline final (main2)', '#808080', '-'),
       ('H+cap', 'H + cap (#1148)', '#2a78d6', '-'),
       ('HBm+cap', 'HBm + cap', '#eb6834', '--'),
       ('HBs+cap', 'HBs + cap', '#1baf7a', ':'),
       ('H+bgfree+cap', 'H + bgfree + cap', '#8a5cd6', '-'),
       ('H+h0+bgfree+cap', 'H + h0 + bgfree + cap', '#d6457a', '-')]
fig, ax = plt.subplots(1, 4, figsize=(16, 4.4), constrained_layout=True, sharey=True)
for a, band in zip(ax, ('150W', '200W', '250M', '300M')):
    d = pickle.load(open(f'{D}/out7/score7_{band}.pkl', 'rb'))
    ref, have = np.asarray(d['ref']), np.asarray(d['have'], bool)
    for key, lab, col, ls in VAR:
        dm = np.asarray(d['dm'][key])
        xs, ys = [], []
        for lo, hi in d['bins']:
            m = have & (ref >= lo) & (ref < hi) & np.isfinite(dm)
            if m.sum() >= 10:
                xs.append(0.5 * (lo + hi))
                ys.append(np.median(dm[m]))
        a.plot(xs, ys, color=col, ls=ls, lw=2, marker='o', ms=5, label=lab)
    a.axhline(d['unsat_dm'], color='k', ls='--', lw=1)
    a.axhline(0, color='0.7', lw=0.8)
    a.set_title(f'F{band}  ({int(have.sum())} stars)')
    a.set_xlabel('dolphot magnitude')
ax[0].set_ylabel('median dm = ours - dolphot - ZP [mag]')
ax[0].legend(fontsize=7.5, loc='lower right')
fig.savefig(D + '/out7/round7.png', dpi=110)
print('wrote', D + '/out7/round7.png')
