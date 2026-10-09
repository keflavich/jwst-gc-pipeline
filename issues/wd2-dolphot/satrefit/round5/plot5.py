import pickle, re
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
O = 'out5'
NL = [0, 2, 3, 5, 8, 12, 25, 40]
rows = {}
sec = False
for ln in open(O + '/curves5.md'):
    if ln.startswith('### R(2440 DN)/wingmig'):
        sec = True; continue
    if sec and ln.startswith('###'):
        break
    if sec and ln.startswith('| ') and 'long' in ln or (sec and ln.startswith('| ') and 'nrcb' in ln):
        c = [x.strip() for x in ln.strip().strip('|').split('|')]
        rows[(c[0], c[1])] = [float(x.split()[0]) for x in c[3:11]]
S = {b: pickle.load(open(f'{O}/score5_{b}.pkl', 'rb')) for b in ('150W', '200W', '250M', '300M')}
fig, ax = plt.subplots(2, 4, figsize=(21, 9.5))
a = ax[0, 0]
for (b, d), v in rows.items():
    if d in ('nrcb1', 'nrcb3', 'nrcb2', 'nrcb4', 'nrcblong'):
        a.plot(range(8), v, marker='o', ms=3, ls='-' if b == '150W' else ('--' if b == '200W' else ':'),
               color={'nrcb1': 'C0', 'nrcb2': 'C2', 'nrcb3': 'C3', 'nrcb4': 'C4', 'nrcblong': 'k'}[d], label=f'{b} {d}', lw=1.2)
a.axhline(1, color='gray', lw=.7); a.set_xticks(range(8)); a.set_xticklabels(NL)
a.set_xlabel('exclusion distance N (px)'); a.set_ylabel('R(2440 DN) / wingmig R'); a.set_ylim(0.96, 1.13)
a.legend(fontsize=6, ncol=2); a.set_title('(a) R curve level vs N (exp 1 / median)')
def dmpanel(a, b, det, title):
    s = S[b]; bins = s['bins']; ref = s['ref']; h = s['have']
    if det:
        h = h & (s['star_det'] == det)
    xs = [(lo + hi) / 2 for lo, hi in bins]
    for v, c, ls in (('final', 'k', '-'), ('rw12h_e0+bgfree+cap', 'C1', '--'), ('RN25+cap', 'C0', '-'), ('RN25+bgfree+cap', 'C2', '-'),
                     ('RN25+h0+bgfree+cap', 'C3', '-'), ('RN5+cap', 'C0', ':')):
        y = [np.median(s['dm'][v][h & (ref >= lo) & (ref < hi)]) if (h & (ref >= lo) & (ref < hi)).sum() >= 5 else np.nan for lo, hi in bins]
        a.plot(xs, y, color=c, ls=ls, marker='o', ms=3, label=v)
    a.axhline(s['unsat_dm'], color='gray', lw=.7); a.axhline(0, color='gray', lw=.4, ls=':')
    a.set_title(title, fontsize=9); a.set_xlabel('ref mag'); a.set_ylabel('median dm')
dmpanel(ax[0, 1], '150W', 'nrcb1', '(b) F150W nrcb1'); dmpanel(ax[0, 2], '150W', 'nrcb3', '(c) F150W nrcb3')
dmpanel(ax[0, 3], '200W', 'nrcb3', '(d) F200W nrcb3 (nrcb1 in table)')
ax[0, 1].legend(fontsize=6)
dmpanel(ax[1, 0], '250M', None, '(e) F250M nrcblong'); dmpanel(ax[1, 1], '300M', None, '(f) F300M nrcblong')
a = ax[1, 2]
gE = [200, 400, 800, 1600, 3200, 6400, 12800, 1e9]; gc = np.sqrt(np.array(gE[:-2] + [12800]) * np.array(gE[1:-1] + [25600]))
for b, ls in (('150W', '-'), ('200W', '--')):
    p = S[b]['pix']
    for k, c, lab in (('r0', 'C3', 'pipeline N=0'), ('r5', 'C1', 'RN5'), ('rN', 'C0', 'RN25')):
        y = []
        for lo, hi in zip(gE[:-1], gE[1:]):
            m = (p['g0'] >= lo) & (p['g0'] < hi) & (p['cat'] == 0)
            y.append(np.median(p[k][m & (p['det'] == 3)]) / np.median(p[k][m & (p['det'] == 1)]))
        a.plot(range(len(y)), y, color=c, ls=ls, marker='o', ms=3, label=f'{b} {lab}')
a.axhline(1, color='gray', lw=.7); a.set_xticks(range(7)); a.set_xticklabels(['200', '400', '800', '1.6k', '3.2k', '6.4k', '12.8k+'])
a.set_xlabel('rim g0 (DN)'); a.set_ylabel('data/model, nrcb3 / nrcb1'); a.legend(fontsize=6); a.set_title('(g) rim-pixel data/model, nrcb3 over nrcb1')
a = ax[1, 3]
for b, ls in (('150W', '-'), ('200W', '--')):
    p = S[b]['pix']
    for k, c in (('r0', 'C3'), ('r5', 'C1'), ('rN', 'C0')):
        m = (p['cat'] == 0) & (p['r'] < 6)
        for d, mk in ((1, 'o'), (3, 's')):
            a.plot([0 if k == 'r0' else (1 if k == 'r5' else 2)], [np.median(p[k][m & (p['det'] == d)])], marker=mk, color=c, ls='', ms=7 if b == '150W' else 4, mfc=c if d == 3 else 'none')
a.axhline(1, color='gray', lw=.7); a.set_xticks([0, 1, 2]); a.set_xticklabels(['N=0', 'N=5', 'N=25']); a.set_ylabel('median data/model, r<6 px rim')
a.set_title('(h) filled=nrcb3, open=nrcb1; big=150W, small=200W', fontsize=8)
fig.tight_layout(); fig.savefig('../satrefit/satrefit5.png', dpi=90)
