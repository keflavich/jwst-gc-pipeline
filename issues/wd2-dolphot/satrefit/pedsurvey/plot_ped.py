"""Figure for the #1148 survey reply: crf / group-0 pedestal B and the measured R(2440)/R_header.
(a) far-field crf / (R_header g0) against g0, with 1 + B/g0 (dashed) for selected frames;
(b) measured R(2440)/R_header against the pedestal prediction 1 + B/2440, every surveyed frame.
"""
import json
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit'
COL = {'brick': '#2a78d6', 'sgrb2': '#eb6834', 'sgrc': '#1baf7a', 'wd2': '#eda100'}

ped = json.load(open(D + '/pedpred.json'))
key = {(r['field'], r['band'], r['det']): r for r in ped}


def consistent(r):
    """B from the 500-1000 and 1000-2000 DN bins agree: a pedestal, not a frame whose far field is
    dominated by pixels with crf ~ 0 (sgrc F115W, brick F115W nrca1/nrca3)."""
    g, q = np.array(r['g']), np.array(r['r'])
    lo = (g >= 500) & (g < 1000)
    hi = (g >= 1000) & (g < 2000)
    if not lo.any() or not hi.any() or not np.isfinite(r['B']):
        return False
    b1, b2 = np.median((q[lo] - 1) * g[lo]), np.median((q[hi] - 1) * g[hi])
    return abs(b1 - b2) < max(30.0, 0.3 * abs(r['B']))


meas = {}
for r in json.load(open(Q + '/survey_fields/survey_fields.json')):
    meas[(r['field'], r['band'], r['det'])] = float(r['0']['R2440']) / float(r['Rhdr']) if '0' in r else \
        float(r[0]['R2440']) / float(r['Rhdr'])
for r in pickle.load(open(Q + '/out6/survey6.pkl', 'rb')):
    if r['exp'] == 1:
        meas[('wd2', 'F' + r['band'], r['det'])] = float(r[0]['R2440']) / float(r['Rhdr'])

fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
SEL = [('brick', 'F356W', 'nrcblong', '-'), ('brick', 'F444W', 'nrcblong', '--'), ('brick', 'F356W', 'nrcalong', ':'),
       ('sgrb2', 'F210M', 'nrca2', '-'), ('sgrb2', 'F150W', 'nrca2', '--'), ('wd2', 'F150W', 'nrcb3', '-'),
       ('wd2', 'F150W', 'nrcb1', '--')]
a = ax[0]
for f, b, d, ls in SEL:
    r = key.get((f, b, d))
    if r is None:
        continue
    g, q = np.array(r['g']), np.array(r['r'])
    a.plot(g, q, ls, color=COL[f], lw=2, marker='o', ms=4, label=f'{f} {b} {d}  B={r["B"]:+.0f} DN')
    gg = np.geomspace(300, 40000, 100)
    a.plot(gg, 1 + r['B'] / gg, ls, color=COL[f], lw=1, alpha=0.5)
a.axhline(1, color='0.5', lw=1)
a.axvline(2440, color='0.7', lw=1, ls=':')
a.set_xscale('log')
a.set_ylim(0.0, 1.6)
a.set_xlabel('group 0 [DN]')
a.set_ylabel('far-field crf / (R_header g0)')
a.set_title('(a) far field (>= 25 px from SATURATED); thin: 1 + B/g0')
a.legend(fontsize=7, loc='lower right')

a = ax[1]
for f in COL:
    xs, ys = [], []
    for k, m in meas.items():
        if k[0] != f or k not in key:
            continue
        r = key[k]
        if r['nB'] < 300 or not consistent(r):
            continue
        xs.append(1 + r['B'] / 2440.0)
        ys.append(m)
    a.plot(xs, ys, 'o', color=COL[f], ms=7, mec='white', mew=1.5, label=f'{f} (N={len(xs)})')
lim = [0.75, 1.1]
a.plot(lim, lim, color='0.5', lw=1)
a.set_xlim(lim)
a.set_ylim(lim)
a.set_xlabel('pedestal prediction 1 + B / 2440')
a.set_ylabel('measured R(2440) / R_header (pipeline curve)')
a.set_title('(b) exposure-1 frames with a consistent pedestal (see text)')
a.legend(fontsize=8, loc='upper left')
a.annotate('wd2 nrcb2-b4: excess above the\npedestal (charge migration)', (1.008, 1.075), (0.86, 1.06), fontsize=8,
           arrowprops=dict(arrowstyle='->', color='0.4'))
a.annotate('brick nrcblong:\nlegacy destreak removed\n~12 MJy/sr after cal', (0.80, 0.80), (0.82, 0.86), fontsize=8,
           arrowprops=dict(arrowstyle='->', color='0.4'))
fig.savefig(D + '/pedsurvey.png', dpi=110)
print('wrote', D + '/pedsurvey.png')
rows = []
for k, m in sorted(meas.items()):
    if k in key and key[k]['nB'] >= 300 and consistent(key[k]):
        rows.append((k, m, 1 + key[k]['B'] / 2440.0, key[k]['B'], key[k]['nB'], key[k]['destrkmd']))
with open(D + '/pedpred.md', 'w') as fh:
    fh.write('| field | band | det | B [DN] | n(500-2000) | 1 + B/2440 | measured R(2440)/R_hdr | diff | DESTRKMD |\n')
    fh.write('|---|---|---|---|---|---|---|---|---|\n')
    for k, m, p, B, n, dm in rows:
        fh.write(f'| {k[0]} | {k[1]} | {k[2]} | {B:+.0f} | {n} | {p:.4f} | {m:.4f} | {m - p:+.4f} | {dm} |\n')
d = np.array([m - p for k, m, p, B, n, dm in rows])
print('N', len(d), 'median |measured - predicted|', np.median(np.abs(d)), 'max', np.max(np.abs(d)))
