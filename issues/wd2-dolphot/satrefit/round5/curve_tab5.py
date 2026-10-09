import pickle
import numpy as np
O = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/out5'
res = pickle.load(open(O + '/curves5.pkl', 'rb'))
NL = [0, 2, 3, 5, 8, 12, 25, 40]
L = []
L.append('| band | det | exp | N | good px | bins raw->kept | R(2440) | R/wingmig | R/header | satcheck fac | flags |')
L.append('|---|---|---|---|---|---|---|---|---|---|---|')
flags = []
for r in res:
    for N in NL:
        x = r['res'][N]
        fl = []
        if x['ngood'] < 50:
            fl.append('COUNT<50 fallback')
        if x['ngood'] >= 50 and x['nkept'] <= 1:
            fl.append('guard->1 bin')
        if x['rebuilt']:
            fl.append('satcheck rebuilt')
        if fl:
            flags.append((r['band'], r['det'], r['exp'], N, ';'.join(fl)))
        L.append(f"| {r['band']} | {r['det']} | {r['exp']} | {N} | {x['ngood']} | {x['nraw']}->{x['nkept']} | {x['R2440']:.4f} | {x['R2440'] / r['Rwing']:.3f} | "
                 f"{(x['R2440'] / r['Rhdr']) if r['Rhdr'] else np.nan:.3f} | {x['satfac']:.3f} | {';'.join(fl)} |")
L.append('')
# summary medians by band/det and N
L.append('### R(2440 DN)/wingmig R, median over frames (min-max), by band and detector')
L.append('')
L.append('| band | det | frames | ' + ' | '.join(f'N={N}' for N in NL) + ' | wingmig R | header R | share good set within 3/5/12 px (N=0) |')
L.append('|---|---|---|' + '---|' * (len(NL) + 3))
keys = sorted({(r['band'], r['det']) for r in res})
for b, d in keys:
    rr = [r for r in res if r['band'] == b and r['det'] == d]
    cells = []
    for N in NL:
        v = np.array([r['res'][N]['R2440'] / r['Rwing'] for r in rr])
        cells.append(f'{np.nanmedian(v):.3f} ({np.nanmin(v):.3f}-{np.nanmax(v):.3f})')
    sh = {k: np.nanmedian([r['share'][k] for r in rr]) for k in (3, 5, 12)}
    L.append(f'| {b} | {d} | {len(rr)} | ' + ' | '.join(cells) + f" | {np.median([r['Rwing'] for r in rr]):.4f} | {np.median([r['Rhdr'] for r in rr]):.4f} | {sh[3]:.2f} / {sh[5]:.2f} / {sh[12]:.2f} |")
L.append('')
L.append('### Median R(2440)/header R by band and detector')
L.append('')
L.append('| band | det | ' + ' | '.join(f'N={N}' for N in NL) + ' |')
L.append('|---|---|' + '---|' * len(NL))
for b, d in keys:
    rr = [r for r in res if r['band'] == b and r['det'] == d]
    L.append(f'| {b} | {d} | ' + ' | '.join(f"{np.nanmedian([r['res'][N]['R2440'] / r['Rhdr'] for r in rr]):.3f}" for N in NL) + ' |')
L.append('')
L.append('### Median cal/g0 of the N=0 good set at g0 2000-3000 DN versus distance to SATURATED (median over frames; pixel count summed)')
L.append('')
bins = list(res[0]['Rdist'].keys())
L.append('| band | det | ' + ' | '.join(f'd {lo}-{hi if hi < 1e8 else "inf"}' for lo, hi in bins) + ' |')
L.append('|---|---|' + '---|' * len(bins))
for b, d in keys:
    rr = [r for r in res if r['band'] == b and r['det'] == d]
    cells = []
    for k in bins:
        n = sum(r['Rdist'][k][0] for r in rr)
        v = np.nanmedian([r['Rdist'][k][1] for r in rr])
        cells.append(f'{v:.4f} ({n})')
    L.append(f'| {b} | {d} | ' + ' | '.join(cells) + ' |')
L.append('')
L.append('### Flags')
L.append('')
if flags:
    for f in flags:
        L.append(f'- {f[0]} {f[1]} exp{f[2]} N={f[3]}: {f[4]}')
else:
    L.append('none')
L.append('')
L.append('### Minimum good-set count per N (over all frames)')
L.append('')
for N in NL:
    c = [(r['res'][N]['ngood'], r['band'], r['det'], r['exp']) for r in res]
    m = min(c)
    L.append(f'- N={N}: min {m[0]} ({m[1]} {m[2]} exp{m[3]}); median {int(np.median([x[0] for x in c]))}')
open(O + '/curves5.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L[L.index('### R(2440 DN)/wingmig R, median over frames (min-max), by band and detector'):]))
