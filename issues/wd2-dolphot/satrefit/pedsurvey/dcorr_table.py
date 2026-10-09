"""Merge dcorr_<i>.json into dcorr_merged.json and write dcorr.md: per frame with rewritten SATURATED rim pixels,
the rim value / (R_header g0) under the curve, the header rate and the header rate minus D, against the far-field
crf / (R_header g0) and the kept SATURATED pixels' crf / (R_header g0), in group-0 bins."""
import glob
import json
import os

import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
rows = []
for f in sorted(glob.glob(D + '/dcorr_[0-9]*.json')):
    rows += json.load(open(f))
json.dump(rows, open(D + '/dcorr_merged.json', 'w'))
rows = [r for r in rows if 'skip' not in r]
print('frames', len(rows))


def f3(x):
    return '-' if x is None or not np.isfinite(x) else f'{x:.3f}'


lines = ['| field | band | det | D median / far [MJy/sr] | B / B+D [DN] | SAT rim n | buffer curve / hdr / hdr-D '
         '| g0 bin | n rim | curve | hdr | hdr-D | far crf | far crf+D | kept crf |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
for r in sorted(rows, key=lambda r: (r['field'], r['band'], r['det'])):
    if r['sat_hdr'] == 0:
        continue
    first = True
    for c in r['cmp']:
        if c['n_rim'] < 10:
            continue
        head = (f"| {r['field']} | {r['band']} | {r['det']} | {r['D_p'][2]:+.2f} / {r['D_far']:+.2f} "
                f"| {r['B']:+.0f} / {r['B_D']:+.0f} | {r['sat_hdr']} | {r['buf_curve']} / {r['buf_hdr']} / {r['buf_hdrD']} "
                if first else '| | | | | | | ')
        first = False
        lines.append(head + f"| {c['lo']}-{c['hi']} | {c['n_rim']} | {f3(c.get('curve'))} | {f3(c.get('hdr'))} "
                     f"| {f3(c.get('hdrD'))} | {f3(c.get('far'))} | {f3(c.get('farD'))} | {f3(c.get('kept'))} |")
open(D + '/dcorr.md', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))

print('\nbuffer-only frames (no SAT rim rewrite), buffer counts curve / hdr / hdr-D, B / B+D:')
for r in sorted(rows, key=lambda r: (r['field'], r['band'], r['det'])):
    if r['sat_hdr'] == 0 and max(r['buf_curve'], r['buf_hdr'], r['buf_hdrD']) >= 100:
        print(f"  {r['field']} {r['band']} {r['det']} D={r['D_p'][2]:+.2f} buf {r['buf_curve']} / {r['buf_hdr']} / "
              f"{r['buf_hdrD']}  B {r['B']:+.0f} / {r['B_D']:+.0f}")
