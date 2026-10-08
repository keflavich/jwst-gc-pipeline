"""Round 4: q vs distance for LW (F250M, F300M nrcblong), binned as wingmig_analyze.  -> out4/q_lw.md, out4/q_lw.pkl"""
import glob, pickle, sys
import numpy as np
import pandas as pd
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/wingmig')
from wingmig_collect import applyR
O = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/out4'
rng = np.random.default_rng(1)
dlab = ['1', '2', '3', '4-5', '6-8', '9-12', '13-20']
dbins = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 8), (9, 12), (13, 20)]
rb = [(1, 2), (2, 3), (3, 4), (4, 6), (6, 9), (3, 9)]
rl = [f'{a}-{b}' for a, b in rb]
rows, crow, meta = [], [], {}
for f in sorted(glob.glob(O + '/raw_F*.pkl')):
    o = pickle.load(open(f, 'rb'))
    tag = f.split('raw_')[1][:-4]
    band = tag.split('_')[0]
    cF, mF, _ = o['R']
    k = mF > 0.8 * np.median(mF)
    cF, mF = cF[k], mF[k]
    meta[tag] = (len(cF), cF[0], cF[-1], mF.copy())
    for i, s in enumerate(o['sats']):
        q = s['cal'] / (applyR(cF, mF, s['g0']) * s['g0'])
        db = np.ceil(s['d'] - 1e-9)
        for (a, b), lab in zip(dbins, dlab):
            m = (db >= a) & (db <= b) & np.isfinite(q)
            if m.sum():
                rows.append((tag, band, i, lab, s['flux'], s['sat_area'], np.median(q[m]), int(m.sum()), np.median(s['g0'][m])))
    for pk, c, g, z, r in o['ctl']:
        q = c / (applyR(cF, mF, g) * g)
        nn = r <= 1.01
        if nn.sum() < 2 or np.median(g[nn]) < 0.3 * pk or not (0.7 < np.nanmedian(q[nn]) < 1.3):
            continue
        for (a, b), lab in zip(rb, rl):
            m = (r > a) & (r <= b) & np.isfinite(q)
            if m.sum():
                crow.append((tag, band, pk, lab, np.median(q[m]), int(m.sum()), np.median(g[m])))
R = pd.DataFrame(rows, columns=['tag', 'band', 'i', 'dbin', 'flux', 'sat_area', 'q', 'n', 'g0med'])
Cc = pd.DataFrame(crow, columns=['tag', 'band', 'pk', 'rbin', 'q', 'n', 'g0med'])
R['fbin'] = R.groupby('band')['flux'].transform(lambda x: pd.qcut(x, 4, labels=['Q1 faint', 'Q2', 'Q3', 'Q4 bright']))


def boot(v, nb=500):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return np.nan, np.nan, len(v)
    return np.median(v), np.std([np.median(rng.choice(v, len(v))) for _ in range(nb)]), len(v)


L = ['### LW q = cal / (R(g0) g0) vs distance d from the DQ-SATURATED edge (wingmig method, g0 in the R curve range 200 DN - top bin)', '']
for tag, (nb, lo, hi, mv) in meta.items():
    L.append(f'- {tag}: {nb} R bins, g0 {lo:.0f}-{hi:.0f} DN, R {np.round(mv, 4).tolist()}')
res = {}
for band in sorted(R.band.unique()):
    L += ['', f'#### {band}: median q (bootstrap err; N star-frames)', '', '| group | ' + ' | '.join(dlab) + ' |', '|' + '---|' * (len(dlab) + 1)]
    sub = R[R.band == band]
    for key, g in [('all', sub)] + [(k, sub[sub.fbin == k]) for k in ['Q1 faint', 'Q2', 'Q3', 'Q4 bright']]:
        cells = []
        for lab in dlab:
            m, e, n = boot(g[g.dbin == lab].q)
            res[(band, key, lab)] = (m, e, n)
            cells.append(f'{m:.3f}+-{e:.3f} ({n})' if np.isfinite(m) else f'- ({n})')
        L.append(f'| {key} | ' + ' | '.join(cells) + ' |')
    L += ['', f'{band} control stars (unsaturated, peak g0 > 1500 DN), median q by radius from peak (px):', '', '| ' + ' | '.join(rl) + ' |', '|' + '---|' * len(rl)]
    cells = []
    for lab in rl:
        m, e, n = boot(Cc[(Cc.band == band) & (Cc.rbin == lab)].q)
        res[(band, 'ctl', lab)] = (m, e, n)
        cells.append(f'{m:.3f}+-{e:.3f} ({n})')
    L.append('| ' + ' | '.join(cells) + ' |')
pickle.dump(dict(res=res, dlab=dlab, rl=rl), open(O + '/q_lw.pkl', 'wb'))
open(O + '/q_lw.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
