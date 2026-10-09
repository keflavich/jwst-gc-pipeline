import pickle, numpy as np
S = pickle.load(open('out6/survey6.pkl', 'rb'))
L = ['| band | det | exp | R_header | R_N0 | R_N25 | N0/hdr | N25/hdr | N0 bins/state | N25 bins/state | flag (|hdr/N0-1| > 2 %) |', '|---|---|---|---|---|---|---|---|---|---|---|']
def st(r):
    return ('fallback' if r['fallback'] else f"{r['nraw']}->{r['nkept']}") + (' rebuilt' if r['rebuilt'] else '')
nflag = 0; ntot = 0
for s in S:
    if s['exp'] != 1:
        continue
    r0, r25, h = s[0], s[25], s['Rhdr']
    a, b = r0['R2440'] / h, r25['R2440'] / h
    fl = []
    if not np.isfinite(a):
        fl.append('no curve (fallback)')
    elif abs(1 / a - 1) > 0.02:
        fl.append(f'moves {100*(1/a-1):+.1f} %')
    if r0['rebuilt'] or r25['rebuilt']:
        fl.append('satcheck rebuild')
    if r0['nkept'] <= 1 and not r0['fallback']:
        fl.append('guard->1 bin')
    ntot += 1
    nflag += bool(fl and fl[0].startswith('moves'))
    L.append(f"| {s['band']} | {s['det']} | {s['exp']} | {h:.4f} | {r0['R2440']:.4f} | {r25['R2440']:.4f} | {a:.3f} | {b:.3f} | {st(r0)} | {st(r25)} | {'; '.join(fl)} |")
L.append('')
L.append(f'{nflag} of {ntot} exposure-1 frames move by more than 2 % under the header anchor.')
L.append('')
L.append('Median over detectors with a usable curve (N0 or N25 / header, nrcb1-4 and nrcblong only, exposure 1):')
for b in ('150W', '162M', '182M', '200W', '250M', '277W', '300M'):
    x = [(s[0]['R2440'] / s['Rhdr'], s[25]['R2440'] / s['Rhdr']) for s in S if s['band'] == b and s['exp'] == 1 and s['det'].startswith('nrcb')]
    x = np.array(x)
    L.append(f'- {b}: N0/hdr {np.nanmedian(x[:,0]):.3f} (min {np.nanmin(x[:,0]):.3f}, max {np.nanmax(x[:,0]):.3f}), N25/hdr {np.nanmedian(x[:,1]):.3f} (min {np.nanmin(x[:,1]):.3f}, max {np.nanmax(x[:,1]):.3f}), n={len(x)}')
L.append('')
L.append('### far-field (edt >= 25) median cal/g0 / R_header by g0 bin (DN); counts in brackets')
L.append('')
L.append('| band | det | exp | 200-500 | 500-1000 | 1000-2000 | 2000-4000 |')
L.append('|---|---|---|---|---|---|---|')
for s in S:
    if not ((s['band'] in ('150W', '200W') and s['det'] in ('nrcb1', 'nrcb3')) or (s['band'] in ('250M', '300M') and s['det'] in ('nrcblong', 'nrcalong'))):
        continue
    L.append(f"| {s['band']} | {s['det']} | {s['exp']} | " + ' | '.join(f"{v:.3f} ({n})" for (n, v) in s['far'].values()) + ' |')
L.append('')
L.append('median over exposures (nrcb1, nrcb3, nrcblong):')
L.append('')
L.append('| band | det | 200-500 | 500-1000 | 1000-2000 | 2000-4000 |')
L.append('|---|---|---|---|---|---|')
for b, d in (('150W', 'nrcb1'), ('150W', 'nrcb3'), ('200W', 'nrcb1'), ('200W', 'nrcb3'), ('250M', 'nrcblong'), ('250M', 'nrcalong'), ('300M', 'nrcblong'), ('300M', 'nrcalong')):
    ss = [s for s in S if s['band'] == b and s['det'] == d]
    cells = []
    for k in ss[0]['far']:
        v = [s['far'][k][1] for s in ss]
        cells.append(f'{np.nanmedian(v):.3f}')
    L.append(f'| {b} | {d} | ' + ' | '.join(cells) + ' |')
open('out6/survey6.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
