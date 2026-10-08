import glob, pickle, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from wingmig_collect import applyR
rng = np.random.default_rng(1)
dedges = [(0,1),(1,2),(2,3),(3,5),(5,8),(8,12),(12,20)]
dlab = ['1','2','3','4-5','6-8','9-12','13-20']
def dbin(d):  # ceil of Euclid distance: d=1 -> (0,1], etc.
    return np.ceil(d - 1e-9)
dbins = [(1,1),(2,2),(3,3),(4,5),(6,8),(9,12),(13,20)]
files = sorted(glob.glob('raw_*.pkl'))
rows = []   # per (frame, star, dbin): median q, qz, n
frames = {}
ctl_rows = []
out = []
for f in files:
    o = pickle.load(open(f, 'rb')); tag = f[4:-4]; band = tag.split('_')[0]
    frames[tag] = o['meta'] | dict(Rn=len(o['R'][0]), Rlo=o['R'][0][0], Rhi=o['R'][0][-1], Rv=o['R'][1], Rzv=o['Rz'][1], Rcv=o['Rc'][1])
    cF, mF, _ = o['R']; cZ, mZ, _ = o['Rz']
    kf = mF > 0.8*np.median(mF); cF, mF = cF[kf], mF[kf]
    kz = mZ > 0.8*np.median(mZ); cZ, mZ = cZ[kz], mZ[kz]
    for i, s in enumerate(o['sats']):
        q = s['cal']/(applyR(cF, mF, s['g0'])*s['g0'])
        z = s['zf']; qz = np.full(z.shape, np.nan)
        okz = z > cZ[0]
        qz[okz] = s['cal'][okz]/(applyR(cZ, mZ, z[okz])*z[okz])
        db = dbin(s['d'])
        for (a, b), lab in zip(dbins, dlab):
            m = (db >= a) & (db <= b) & np.isfinite(q)
            if m.sum() >= 1:
                mz = m & np.isfinite(qz)
                rows.append((tag, band, i, lab, s['flux'], s['sat_area'], np.median(q[m]), np.median(qz[mz]) if mz.sum() else np.nan, int(m.sum()), np.median(s['g0'][m])))
    # control
    rb = [(1,2),(2,3),(3,4),(4,6),(6,9)]
    nrej = 0
    for pk, c, g, z, r in o['ctl']:
        q = c/(applyR(cF, mF, g)*g)
        nn = (r <= 1.01)
        if nn.sum() < 2 or np.median(g[nn]) < 0.3*pk or not (0.7 < np.nanmedian(q[nn]) < 1.3):
            nrej += 1; continue
        for (a, b) in rb:
            m = (r > a) & (r <= b) & np.isfinite(q)
            if m.sum(): ctl_rows.append((tag, band, pk, f'{a}-{b}', np.median(q[m]), m.sum(), np.median(g[m])))
import pandas as pd
R = pd.DataFrame(rows, columns=['tag','band','i','dbin','flux','sat_area','q','qz','n','g0med'])
C = pd.DataFrame(ctl_rows, columns=['tag','band','pk','rbin','q','n','g0med'])
print('ctl kept', len(ctl_rows)); R.to_pickle('rows_sat.pkl'); C.to_pickle('rows_ctl.pkl')
def boot(v, nb=500):
    v = np.asarray(v); v = v[np.isfinite(v)]
    if len(v) < 3: return np.nan, np.nan, len(v)
    m = np.median(v); bs = [np.median(rng.choice(v, len(v))) for _ in range(nb)]
    return m, np.std(bs), len(v)
# brightness bins by sat_area terciles across all (band-wise flux not comparable; use sat_area, and flux quartiles per band)
R['fbin'] = R.groupby('band')['flux'].transform(lambda x: pd.qcut(x, 4, labels=['Q1 faint','Q2','Q3','Q4 bright']))
R['abin'] = pd.cut(R.sat_area, [0, 50, 150, 400, 1e5], labels=['<50','50-150','150-400','>400'])
lines = []
def table(group, name, col='q'):
    lines.append(f'\n### {name} (median {col}, bootstrap err over star-frames, N star-frames)\n')
    lines.append('| group | ' + ' | '.join(dlab) + ' |'); lines.append('|' + '---|'*(len(dlab)+1))
    res = {}
    for key, g in R.groupby(group, observed=True):
        cells = []
        for lab in dlab:
            m, e, n = boot(g[g.dbin == lab][col])
            cells.append(f'{m:.3f}+-{e:.3f} ({n})' if np.isfinite(m) else f'- ({n})')
            res[(key, lab)] = (m, e, n)
        lines.append(f'| {key} | ' + ' | '.join(cells) + ' |')
    return res
allband = R.assign(all='all'); 
resA = table(['band'], 'By band')
resF = table(['band', 'fbin'], 'By band and flux_fit quartile')
resS = table(['abin'], 'By sat_area (px)')
resZ = table(['band'], 'By band, vs ZEROFRAME expectation', 'qz')
# g0 dependence at d=3-5
lines.append('\n### q vs pixel g0 median (star-bin level), d=4-5 and 6-8\n')
R['gbin'] = pd.cut(R.g0med, [200, 400, 800, 1600])
for lab in ['4-5', '6-8']:
    for k, g in R[R.dbin == lab].groupby('gbin', observed=True):
        m, e, n = boot(g.q); lines.append(f'- d={lab}, g0 in {k}: q={m:.3f}+-{e:.3f} (N={n})')
lines.append('\n### Control stars: q vs radius from peak (all control stars, per-star-bin medians)\n')
lines.append('| band | ' + ' | '.join(['1-2','2-3','3-4','4-6','6-9']) + ' |'); lines.append('|---|---|---|---|---|---|')
for b, g in C.groupby('band'):
    cells = []
    for rb in ['1-2','2-3','3-4','4-6','6-9']:
        m, e, n = boot(g[g.rbin == rb].q); cells.append(f'{m:.3f}+-{e:.3f} ({n})')
    lines.append(f'| {b} | ' + ' | '.join(cells) + ' |')
lines.append('\n### Control stars by peak g0 (r 4-6 px)\n')
C['pbin'] = pd.cut(C.pk, [1500, 3000, 6000, 12000, 1e6])
for (b, k), g in C[C.rbin == '4-6'].groupby(['band', 'pbin'], observed=True):
    m, e, n = boot(g.q); lines.append(f'- {b} peak g0 {k}: q={m:.3f}+-{e:.3f} (N={n})')
lines.append('\n### Matched-g0 comparison: satstar wings (d 4-8) vs control (r 4-9)\n')
for lo, hi in [(200,400),(400,800),(800,1600)]:
    a_ = R[R.dbin.isin(['4-5','6-8']) & (R.g0med>lo) & (R.g0med<=hi)].q
    c_ = C[C.rbin.isin(['4-6','6-9']) & (C.g0med>lo) & (C.g0med<=hi)].q
    ma, ea, na = boot(a_); mc, ec, nc = boot(c_)
    lines.append(f'- g0 {lo}-{hi}: satstar q={ma:.3f}+-{ea:.3f} (N={na}); control q={mc:.3f}+-{ec:.3f} (N={nc})')
lines.append('\n### Control stars matched to satstar g0 (r 4-6, g0med in same bins as satstar wings)\n')
for k, g in C[C.rbin.isin(['4-6'])].groupby(pd.cut(C[C.rbin=='4-6'].g0med, [200,400,800,1600]), observed=True):
    m, e, n = boot(g.q); lines.append(f'- control g0 {k}: q={m:.3f}+-{e:.3f} (N={n})')
open('tables.md', 'w').write('\n'.join(lines))
# fraction of wings with q>1 etc
# plot
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
x = np.arange(len(dlab)); cols = ['#1b9e77','#7570b3','#d95f02','#e7298a']
for a, band in zip(ax[:2], ['F150W', 'F200W']):
    for (fb, c) in zip(['Q1 faint','Q2','Q3','Q4 bright'], cols):
        y = [resF.get(((band, fb), l), (np.nan,)*3) for l in dlab]
        a.errorbar(x, [t[0] for t in y], [t[1] for t in y], color=c, marker='o', label=fb, capsize=2)
    a.axhline(1, color='k', lw=.8); a.set_title(f'{band} satstar wings'); a.set_xticks(x); a.set_xticklabels(dlab)
    a.set_xlabel('distance from nearest DQ-SATURATED px (px)'); a.legend(title='flux_fit quartile')
a = ax[2]
rl = ['1-2','2-3','3-4','4-6','6-9']
for b, c in zip(['F150W','F200W'], ['#1b9e77','#d95f02']):
    g = C[C.band == b]; y = [boot(g[g.rbin == r].q) for r in rl]
    a.errorbar(range(5), [t[0] for t in y], [t[1] for t in y], color=c, marker='s', label=b, capsize=2)
a.axhline(1, color='k', lw=.8); a.set_xticks(range(5)); a.set_xticklabels(rl); a.legend(title='control (unsat stars)')
a.set_xlabel('radius from peak (px)'); a.set_title('control'); ax[0].set_ylabel('q = cal / (R(g0) g0)')
plt.tight_layout(); plt.savefig('wingmig.png', dpi=130)
print('\n'.join(lines))
for t, m in frames.items(): print(t, m['readpatt'], m['nframes'], m['ngroups'], 'corr %.3f' % m['corr'], m['shape'], 'Rlo-hi %.0f-%.0f' % (m['Rlo'], m['Rhi']), np.round(m['Rv'], 4), np.round(m['Rcv'], 4))
