"""Tables of the leftover-driven SATURATED-neighbour check from satnbr_lo_<tile>_<band>.json (scan_satnbr_lo.py).

usage: python -I summarize_satnbr_lo.py OUTDIR > summary_lo.txt"""
import collections
import json
import sys

import numpy as np

OUT = sys.argv[1]
TILES = ['o045', 'o105', 'o118', 'o130', 'o138']
BANDS = ['F212N', 'F480M']
TAGS = ['optfin', 'optnm']
THR = [50, 100, 200, 500]
data = {}
for t in TILES:
    for b in BANDS:
        with open(f'{OUT}/satnbr_lo_{t}_{b}.json') as fh:
            data[(t, b)] = json.load(fh)
first = next(iter(data.values()))
print(f'finder tolerance max_unusable = {first["max_unusable"]}; floor {first["floor"]:g} MJy/sr; match radius {first["match_arcsec"]} arcsec; '
      f'variants {first["variants"]}\n')

print('## Table L1: blocked spikes per finder variant (A shipped tests, B no sigma test, C isolation 0.3, D both relaxed); n at excess >= 100 / >= 300')
print('| tile | band | A | B | C | D |')
print('|---|---|---|---|---|---|')
tot = {v: [0, 0] for v in 'ABCD'}
for (t, b), d in data.items():
    cells = []
    for v in 'ABCD':
        bl = d['blocked'][v]
        n1, n3 = len(bl), sum(x['excess'] >= 300 for x in bl)
        tot[v][0] += n1
        tot[v][1] += n3
        cells.append(f'{n1} / {n3}')
    print(f'| {t} | {b} | ' + ' | '.join(cells) + ' |')
print('| all | all | ' + ' | '.join(f'{tot[v][0]} / {tot[v][1]}' for v in 'ABCD') + ' |\n')


def klass(r):
    f = r['best']
    if f is None:
        return 'no covering frame'
    if f['dq_self'] & 2:
        return 'brightest pixel is SATURATED itself'
    if f['own']['A']:
        return 'spike the shipped finder returns'
    if f['clr']['A']:
        return 'blocked spike (variant A)'
    if any(f['clr'][v] for v in 'BCD'):
        return 'blocked only with relaxed tests (B, C or D)'
    if f['sat3']:
        return 'SATURATED pixel in the 3x3, not a spike'
    if f['sat7']:
        return 'SATURATED pixel in the 7x7 only'
    return 'no SATURATED pixel within 3 px'


ORDER = ['no covering frame', 'brightest pixel is SATURATED itself', 'spike the shipped finder returns', 'blocked spike (variant A)',
         'blocked only with relaxed tests (B, C or D)', 'SATURATED pixel in the 3x3, not a spike', 'SATURATED pixel in the 7x7 only',
         'no SATURATED pixel within 3 px']
print('## Table L2: leftovers by the frame pixel at their position (the brightest pixel of the 5x5 box in the frame where its excess is largest), all eight tile-bands')
for tag in TAGS:
    print(f'\n### {tag}')
    print('| class | ' + ' | '.join(f'>= {t}' for t in THR) + ' |')
    print('|---|' + '---|' * len(THR))
    cnt = {t: collections.Counter() for t in THR}
    for d in data.values():
        for r in d['leftovers']:
            if r['tag'] != tag:
                continue
            k = klass(r)
            for t in THR:
                if r['resid_peak'] >= t:
                    cnt[t][k] += 1
    for k in ORDER:
        print(f'| {k} | ' + ' | '.join(str(cnt[t][k]) for t in THR) + ' |')
    print('| total | ' + ' | '.join(str(sum(cnt[t].values())) for t in THR) + ' |')
print()

print('## Table L3: leftovers >= 100 of optfin whose brightest frame pixel has a SATURATED pixel in its 3x3 or is a blocked spike of any variant')
print('| tile | band | leftover (x, y) | residual peak | class of the leftover list | frame | pixel (x, y) | excess | sigma | brightest neighbour / excess | SATURATED in 3x3 | blob area | finder A own / cleared | any variant blocked |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
rows = []
for (t, b), d in data.items():
    for r in d['leftovers']:
        if r['tag'] == 'optfin' and r['resid_peak'] >= 100 and r['best'] is not None:
            f = r['best']
            if f['sat3'] or any(f['clr'][v] for v in 'ABCD'):
                rows.append((r['resid_peak'], t, b, r, f))
for pk, t, b, r, f in sorted(rows, key=lambda v: -v[0]):
    fr = f['frame'].replace('jw10678', '').replace('_destreak_' + t + '_crf.fits', '')
    print(f'| {t} | {b} | ({r["x"]}, {r["y"]}) | {pk:.0f} | {r["cls"]} | {fr} | ({f["x"]}, {f["y"]}) | {f["excess"]:.0f} | {f["nsigma"]:.0f} | {f["nbfrac"]:.2f} | '
          f'{f["sat3"]} | {f["sat_comp_area"]} | {f["own"]["A"]} / {f["clr"]["A"]} | {"".join(v for v in "ABCD" if f["clr"][v] and not f["own"][v])} |')
print()

print('## Table L4: fraction of the leftovers with any SATURATED pixel in the 7x7 around the brightest pixel in the best frame, by class of the leftover list (optfin >= 100, optnm >= 100)')
print('| render | class | n | SATURATED in 3x3 | SATURATED in 7x7 | no SATURATED |')
print('|---|---|---|---|---|---|')
for tag in TAGS:
    byc = collections.defaultdict(lambda: [0, 0, 0, 0])
    for d in data.values():
        for r in d['leftovers']:
            if r['tag'] != tag or r['resid_peak'] < 100 or r['best'] is None:
                continue
            f = r['best']
            c = byc[r['cls']]
            c[0] += 1
            c[1] += bool(f['sat3'])
            c[2] += bool(f['sat7'])
            c[3] += (not f['sat7'])
    for cls, c in sorted(byc.items()):
        print(f'| {tag} | {cls} | {c[0]} | {c[1]} | {c[2]} | {c[3]} |')
print()

print('## Table L5: blocked spikes of any variant matched to a leftover (within the match radius), per render and threshold')
print('| render | leftover >= | variant A | variant B | variant C | variant D |')
print('|---|---|---|---|---|---|')
for tag in TAGS:
    for thr in (50, 100, 200):
        cells = []
        for v in 'ABCD':
            n = 0
            for d in data.values():
                for r in d['leftovers']:
                    if r['tag'] == tag and r['resid_peak'] >= thr and r['blocked_variant'][v]:
                        n += 1
            cells.append(str(n))
        print(f'| {tag} | {thr} | ' + ' | '.join(cells) + ' |')
