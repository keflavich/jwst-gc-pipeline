"""Figure for the #1148 reply: rewritten SATURATED-rim values on w51 LW (legacy destreak), cal scale, over the kept
SATURATED pixels' (crf + D) / (R_header g0), in group-0 bins; one thin line per frame, median of the frames thick.
Variants: measured curve, header rate (353ac7df), header rate minus the cal - crf level D (this commit)."""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
rows = [r for r in json.load(open(D + '/dcorr_merged.json')) if 'skip' not in r]
VAR = [('curve', 'measured curve', '#2a78d6'), ('hdr', 'header rate (353ac7df)', '#eb6834'),
       ('hdrD', 'header rate - cal-crf level', '#1baf7a')]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True, sharey=True)
for a, det in zip(ax, ('nrcalong', 'nrcblong')):
    for key, lab, col in VAR:
        per_bin = {}
        for r in rows:
            if r['field'] != 'w51' or r['det'] != det or r['sat_hdr'] == 0:
                continue
            xs, ys = [], []
            for c in r['cmp']:
                if c['n_rim'] < 10 or 'keptD' not in c:
                    continue
                x = np.sqrt(c['lo'] * c['hi'])
                y = (c[key] + c['D_rim']) / c['keptD']
                xs.append(x)
                ys.append(y)
                per_bin.setdefault(x, []).append(y)
            a.plot(xs, ys, color=col, lw=0.8, alpha=0.35)
        xb = sorted(per_bin)
        a.plot(xb, [np.median(per_bin[x]) for x in xb], color=col, lw=2.5, marker='o', ms=6, label=lab)
    a.axhline(1, color='0.5', lw=1)
    a.set_xscale('log')
    a.set_xlabel('group 0 of the rewritten SATURATED rim pixel [DN]')
    a.set_title(f'w51 {det}, F335M F360M F405N F410M F480M (exposure 1)')
ax[0].set_ylabel('(rim + D) / (kept SAT crf + D), same g0 bin')
ax[0].set_ylim(0.7, 1.8)
ax[0].legend(fontsize=8, loc='upper right')
fig.savefig(D + '/dcorr.png', dpi=110)
print('wrote', D + '/dcorr.png')
