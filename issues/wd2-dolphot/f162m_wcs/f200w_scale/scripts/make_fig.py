import re
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dets = [f'nrc{m}{i}' for m in 'ab' for i in range(1, 5)]
J = {f: np.load(f'rc_J_{f}.npz') for f in ('wd1', 'wd2')}


def scale_rot(f, band, det):
    # per-frame-median corrected scale as printed by rc_scale.py
    txt = open(f'rc_{f}.txt').read()
    sec = re.search(rf'### {band} - f212n.*?(?=\n###|\Z)', txt, re.S).group(0)
    rows = re.findall(r'corrected scale\s+([+-][\d.]+) ppm rot\s+([+-][\d.]+)"', sec)
    return [(float(a), float(b)) for a, b in rows]


tabA = {(f, b): scale_rot(f, b, None) for f in ('wd1', 'wd2') for b in ('f200w', 'f150w')}
tabC = {(f, b): scale_rot(f, b, None) for f in ('wd2',) for b in ('f187n', 'f182m')}
with open('tables_AC.txt', 'w') as fh:
    fh.write('A: corrected pixel-frame scale (ppm) / rot (")\ndet   ' + '  '.join(f'{f}-{b}' for f in ('wd2', 'wd1') for b in ('f200w', 'f150w')) + '\n')
    for i, d in enumerate(dets):
        fh.write(d + '  ' + '  '.join('%+6.1f/%+5.2f' % tabA[(f, b)][i] for f in ('wd2', 'wd1') for b in ('f200w', 'f150w')) + '\n')
    fh.write('C:\n')
    for i, d in enumerate(dets):
        fh.write(d + '  ' + '  '.join('%+6.1f/%+5.2f' % tabC[('wd2', b)][i] for b in ('f187n', 'f182m')) + '\n')

S = pickle.load(open('colour_split.pkl', 'rb'))
with open('tables_B.txt', 'w') as fh:
    for band in ('f200w', 'f150w'):
        fh.write(f'B {band}: scale ppm +- boot (n) per tercile; slope ppm/mag\n')
        sl = []
        for d in dets:
            r = S[(band, d)]
            fh.write(f'{d} all {r["all"]["scale"]:+6.1f}+-{r["all"]["escale"]:.1f} | ' + ' | '.join(
                f't{j} c={r["t%d" % j]["medcol"]:.2f} n={r["t%d" % j]["n"]} {r["t%d" % j]["scale"]:+6.1f}+-{r["t%d" % j]["escale"]:.1f}' for j in (1, 2, 3))
                + f' | slope {r["slope"][0]:+.2f}+-{r["slope"][1]:.2f}\n')
            sl.append(r['slope'])
        sl = np.array(sl)
        w = 1 / sl[:, 1] ** 2
        fh.write(f'  inverse-variance mean slope over 8 det: {np.sum(w * sl[:, 0]) / w.sum():+.2f} +- {w.sum() ** -0.5:.2f}; '
                 f'excluding nrcb3: {np.sum((w * sl[:, 0])[np.arange(8) != 6]) / w[np.arange(8) != 6].sum():+.2f} +- {w[np.arange(8) != 6].sum() ** -0.5:.2f}; '
                 f'chi2 about mean = {np.sum(w * (sl[:, 0] - np.sum(w * sl[:, 0]) / w.sum()) ** 2):.1f}\n')

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
x = np.arange(8)
sty = {('wd2', 'f200w'): ('C3', 'o', 'wd2 F200W'), ('wd1', 'f200w'): ('C3', 's', 'wd1 F200W'),
       ('wd2', 'f150w'): ('C0', 'o', 'wd2 F150W'), ('wd1', 'f150w'): ('C0', 's', 'wd1 F150W')}
for k, (c, m, lab) in sty.items():
    ax[0].plot(x + (0.12 if k[0] == 'wd1' else -0.12), [v[0] for v in tabA[k]], m, color=c, mfc=c if k[0] == 'wd2' else 'none', label=lab)
ax[0].axhline(0, color='k', lw=0.5)
ax[0].set_xticks(x, [d.upper() for d in dets], rotation=45)
ax[0].set_ylabel('corrected pixel-frame scale vs F212N (ppm)')
ax[0].set_title('wd2 vs wd1 (rotation-corrected)')
ax[0].legend(fontsize=8)
for j, d in enumerate(dets):
    r = S[('f200w', d)]
    cc = [r['t%d' % i]['medcol'] for i in (1, 2, 3)]
    ss = [r['t%d' % i]['scale'] for i in (1, 2, 3)]
    ee = [r['t%d' % i]['escale'] for i in (1, 2, 3)]
    ax[1].errorbar(cc, ss, ee, marker='o', ms=3, capsize=2, label=d.upper(), color=plt.cm.tab10(j))
ax[1].axhline(0, color='k', lw=0.5)
ax[1].set_xlabel('colour terciles: median -2.5 log10(F115W/F212N) flux ratio (mag)')
ax[1].set_ylabel('F200W pixel-frame scale (ppm)')
ax[1].set_title('wd2 F200W scale vs colour tercile (bootstrap 1-sigma)')
ax[1].legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig('fig_f200w_scale.png', dpi=130)
