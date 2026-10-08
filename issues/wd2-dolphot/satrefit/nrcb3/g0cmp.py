"""Mag-matched comparison of group-0 / ZEROFRAME core DN between nrcb1 and nrcb3 (uses g0_rows.npy from g0.py).
Usage: nice -19 python -u g0cmp.py -> g0cmp.json, g0cmp_tables.md"""
import json
import numpy as np
import pandas as pd

OUT = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/nrcb3'
R = np.load(f'{OUT}/g0_rows.npy', allow_pickle=True)
df = pd.DataFrame({'band': R[:, 0], 'det': R[:, 1], 'ref': R[:, 3].astype(float), 'g0': R[:, 4].astype(float), 'zf': R[:, 5].astype(float),
                   'sat': R[:, 6].astype(float), 'gsat': R[:, 7].astype(float), 'ng': R[:, 8].astype(float), 'glast': R[:, 9].astype(float), 'd01': R[:, 10].astype(float)})
rng = np.random.default_rng(5)
res, L = {}, ['### Group-0 and ZEROFRAME core DN, nrcb3 vs nrcb1 at matched dolphot magnitude (1-mag bins; ratio as 2.5 log10(b3/b1), positive = nrcb3 brighter in DN)\n',
              '| band | quantity | ' + ' | '.join(f'{m}-{m + 1}' for m in range(13, 19)) + ' | weighted mean +/- boot |', '|---|---|' + '---|' * 7]
for band in ['F150W', 'F162M', 'F182M', 'F200W']:
    d = df[(df.band == band) & df.det.isin(['nrcb1', 'nrcb3']) & (df.g0 > 0)].copy()
    d['mb'] = np.floor(d.ref)
    for q in ('g0', 'zf', 'd01'):
        def est(dd):
            vals, w = [], []
            for m, g in dd.groupby('mb'):
                a, b = g[g.det == 'nrcb3'][q], g[g.det == 'nrcb1'][q]
                if len(a) >= 5 and len(b) >= 5 and np.median(b) > 0 and np.median(a) > 0:
                    vals.append(2.5 * np.log10(np.median(a) / np.median(b)))
                    w.append(1 / (1 / len(a) + 1 / len(b)))
            return vals, w
        vals, w = est(d)
        cells = {}
        for m, g in d.groupby('mb'):
            a, b = g[g.det == 'nrcb3'][q], g[g.det == 'nrcb1'][q]
            if len(a) >= 5 and len(b) >= 5 and np.median(b) > 0 and np.median(a) > 0:
                cells[int(m)] = 2.5 * np.log10(np.median(a) / np.median(b))
        mean = np.average(vals, weights=w) if vals else np.nan
        bs = []
        b1, b3 = d[d.det == 'nrcb1'], d[d.det == 'nrcb3']
        for _ in range(200):
            r = pd.concat([b1.iloc[rng.integers(0, len(b1), len(b1))], b3.iloc[rng.integers(0, len(b3), len(b3))]])
            v, ww = est(r)
            bs.append(np.average(v, weights=ww) if v else np.nan)
        res[f'{band}_{q}'] = dict(cells=cells, mean=float(mean), err=float(np.nanstd(bs)))
        L.append(f'| {band} | {q} | ' + ' | '.join(f'{cells[m]:+.3f}' if m in cells else '-' for m in range(13, 19)) + f' | {mean:+.3f} +/- {np.nanstd(bs):.3f} |')
    # saturation depth
    for det in ('nrcb1', 'nrcb3'):
        g = d[d.det == det]
        res[f'{band}_{det}_gsat'] = dict(median=float(g.gsat.median()), frac_sat_lastgroup=float((g.glast >= 64000).mean()), ng=float(g.ng.median()))
L.append('\nFirst saturated group of the peak pixel (A/D or sat-ref criterion; 7 = never within 7 groups), median, and fraction at the A/D limit in the last group:\n')
L.append('| band | nrcb1 median gsat / frac last group >= 64000 | nrcb3 |\n|---|---|---|')
for band in ['F150W', 'F162M', 'F182M', 'F200W']:
    a, b = res[f'{band}_nrcb1_gsat'], res[f'{band}_nrcb3_gsat']
    L.append(f"| {band} | {a['median']:.0f} / {a['frac_sat_lastgroup']:.2f} | {b['median']:.0f} / {b['frac_sat_lastgroup']:.2f} |")
json.dump(res, open(f'{OUT}/g0cmp.json', 'w'), indent=1)
open(f'{OUT}/g0cmp_tables.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
