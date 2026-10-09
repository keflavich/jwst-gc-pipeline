"""Histogram of the bright-end R used by the ZEROFRAME rim (measured curve) over R_header, per field, from
dryall.json; and the markdown table of rim / buffer changes for the #1148 reply (dryall.md)."""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
COL = {'brick': '#2a78d6', 'sgrb2': '#eb6834', 'sgrc': '#1baf7a', 'w51': '#8b5cf6', 'wd2': '#eda100'}
rows = [r for r in json.load(open(D + '/dryall_merged.json')) if 'skip' not in r and np.isfinite(r['hdr_ratio'])]
for r in rows:
    # final measured R (after the guard, satcheck and header fallback), from the [zeroframe R header] line
    r['R_used'] = r['hdr_ratio'] * r['Rh']

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
bins = np.arange(0.66, 1.12, 0.01)
a = ax[0]
for f, c in COL.items():
    v = [r['R_used'] / r['Rh'] for r in rows if r['field'] == f]
    if v:
        a.hist(np.clip(v, bins[0], bins[-1]), bins=bins, histtype='step', lw=2, color=c, label=f'{f} (N={len(v)})')
a.axvline(1, color='0.5', lw=1)
a.set_xlabel('bright-end R used by the curve / R_header')
a.set_ylabel('exposure-1 frames')
a.set_title('(a) rate the rim used before #1148 (header / this = rim change)')
a.legend(fontsize=8, loc='upper left')

a = ax[1]
for f, c in COL.items():
    xs, ys, ss = [], [], []
    for r in rows:
        if r['field'] != f or r['sat_on'] == 0 or not np.isfinite(r['sat_ratio_all']):
            continue
        xs.append(r['R_used'] / r['Rh'])
        ys.append(r['sat_ratio_all'])
        ss.append(10 + 3 * np.sqrt(r['sat_on']))
    if xs:
        a.scatter(xs, ys, s=ss, color=c, edgecolor='white', lw=1, label=f)
xx = np.linspace(0.66, 1.12, 10)
a.plot(xx, 1 / xx, color='0.5', lw=1, ls='--', label='1 / (R_used / R_header)')
a.set_xlabel('bright-end R used by the curve / R_header')
a.set_ylabel('SAT rim value, header / curve (median)')
a.set_title('(b) frames with rewritten SAT rim pixels (size ~ sqrt N)')
a.legend(fontsize=8)
fig.savefig(D + '/dryall.png', dpi=110)
print('wrote', D + '/dryall.png')


def cell(b):
    return '-' if b[1] == 0 else f'{b[0]:.3f} ({b[1]})'


with open(D + '/dryall.md', 'w') as fh:
    fh.write('| field | band | det | R_used / R_hdr | R_faint / R_hdr | B [DN] | SAT rim off / on | buffer off / on '
             '| buffer only-off / only-on | SAT on/off by g0 <2k / 2-5k / 5-15k / >15k (n) |\n')
    fh.write('|---|---|---|---|---|---|---|---|---|---|\n')
    for r in rows:
        fh.write(f"| {r['field']} | {r['band']} | {r['det']} | {r['R_used'] / r['Rh']:.3f} | {r['R_faint'] / r['Rh']:.3f} "
                 f"| {r['B']:+.0f} | {r['sat_off']} / {r['sat_on']} | {r['buf_off']} / {r['buf_on']} "
                 f"| {r['buf_only_off']} / {r['buf_only_on']} | {' / '.join(cell(b) for b in r['sat_ratio_bins'])} |\n")
for f in COL:
    v = np.array([r['R_used'] / r['Rh'] for r in rows if r['field'] == f])
    if len(v):
        print(f, len(v), 'R_used/R_hdr p5/p50/p95', np.round(np.percentile(v, [5, 50, 95]), 3),
              'min', v.min().round(3), 'max', v.max().round(3),
              'frames with SAT rim rewritten', sum(1 for r in rows if r['field'] == f and r['sat_on'] > 0))
