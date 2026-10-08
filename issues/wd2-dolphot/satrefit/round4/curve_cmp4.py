import glob, os, re
import numpy as np
O = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/out4'
rows = []
for f in sorted(glob.glob(f'{O}/*_pcurve.txt')):
    W, P, PR, sat, ceil = [], [], [], None, None
    for l in open(f):
        t = l.split()
        if l.startswith('W '):
            W.append((float(t[1]), float(t[2])))
        elif l.startswith('P '):
            P.append((float(t[1]), float(t[2])))
        elif l.startswith('PR '):
            PR.append((float(t[1]), float(t[2])))
        elif l.startswith('# ceiling'):
            ceil = l.strip()
        elif l.startswith('# satcheck'):
            sat = l.strip().split('curve/sat = ')[-1]
    W, P, PR = map(np.array, (W, P, PR))
    ratios = [p[1] / np.interp(p[0], W[:, 0], W[:, 1]) for p in P]
    b = os.path.basename(f)
    m = re.match(r'(\d+[WM])_.*_(\d{5})_(nrc\w+?)_align', b)
    rows.append((m.group(1), m.group(3), int(m.group(2)), P, ratios, W, len(PR), sat))
L = ['| band | det | exp | pipeline R (g0) kept bins | wingmig R at the same g0 | pipeline / wingmig | bins measured -> kept | wingmig R range (g0 max) | satcheck curve/sat |', '|---|---|---|---|---|---|---|---|---|']
for b, d, e, P, r, W, npr, sat in rows:
    wr = [np.interp(p[0], W[:, 0], W[:, 1]) for p in P]
    L.append(f'| {b} | {d} | {e} | ' + ', '.join(f'{p[1]:.4f} ({p[0]:.0f})' for p in P) + ' | ' + ', '.join(f'{x:.4f}' for x in wr) + ' | ' + ', '.join(f'{x:.3f}' for x in r) + f' | {npr} -> {len(P)} | {W[:, 1].min():.4f}-{W[:, 1].max():.4f} ({W[-1, 0]:.0f}) | {sat} |')
L += ['', 'Median pipeline/wingmig ratio at the first kept bin (g0 ~ 2400 DN) by band and detector:', '']
for b in sorted({r[0] for r in rows}):
    for d in sorted({r[1] for r in rows if r[0] == b}):
        v = [r[4][0] for r in rows if r[0] == b and r[1] == d]
        L.append(f'- {b} {d}: {np.median(v):.3f} (range {min(v):.3f}-{max(v):.3f}, {len(v)} frames)')
open(f'{O}/pcurve_compare.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
