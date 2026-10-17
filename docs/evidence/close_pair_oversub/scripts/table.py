"""Markdown table of the S/N < -5 fraction per nearest-neighbour bin (nn_oversub.py outputs)."""
import glob
import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
GROUPS = [(0, 0.75), (0.75, 1.25), (1.25, 1.5), (1.5, 2), (2, 3), (3, 5), (5, float('inf'))]
lines = ['| field band (m7) | n vetted | ' + ' | '.join(f'{a}–{b:g}' for a, b in GROUPS)
         + ' | all | NN < 1.5 FWHM |', '|---|---|' + '---|' * len(GROUPS) + '---|---|']
files = sorted(glob.glob(os.path.join(OUT, '*.json')),
               key=lambda f: (json.load(open(f))['which'] != 'after', f))
for f in files:
    d = json.load(open(f))
    cells = []
    for a, b in GROUPS:
        rs = [r for r in d['bins'] if r['lo'] >= a and r['hi'] <= b]
        n = sum(r['n'] for r in rs)
        k = sum(r['n'] * (r['frac_lt_m5'] or 0) for r in rs)
        cells.append(f'{100 * k / n:.0f}% ({n})' if n >= 20 else f'– ({n})')
    lines.append(f"| {d['field']} {d['filt']} {d['which']} | {d['n_vetted']:,} | " + ' | '.join(cells)
                 + f" | {100 * d['frac_lt_m5_all']:.1f}% | {100 * d['frac_nn_lt_1p5fwhm']:.1f}% |")
open(os.path.join(OUT, 'table.md'), 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
