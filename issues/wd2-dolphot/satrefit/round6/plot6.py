import pickle
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
O = 'out6'
S = pickle.load(open(O + '/survey6.pkl', 'rb'))
SC = {b: pickle.load(open(f'{O}/score6_{b}.pkl', 'rb')) for b in ('150W', '200W', '250M', '300M')}
fig, ax = plt.subplots(2, 4, figsize=(21, 9.5))
a = ax[0, 0]
bands = ['150W', '162M', '182M', '200W', '250M', '277W', '300M']
col = {'nrcb1': 'C0', 'nrcb2': 'C2', 'nrcb3': 'C3', 'nrcb4': 'C4', 'nrcblong': 'k', 'nrcalong': 'gray'}
for s in S:
    if s['exp'] != 1 or not (s['det'].startswith('nrcb') or s['det'] == 'nrcalong'):
        continue
    x = bands.index(s['band']) + {'nrcb1': -.3, 'nrcb2': -.1, 'nrcb3': .1, 'nrcb4': .3, 'nrcblong': 0, 'nrcalong': .2}[s['det']]
    for N, mk in ((0, 'o'), (25, 's')):
        y = s[N]['R2440'] / s['Rhdr']
        if np.isfinite(y):
            a.plot(x, y, mk, color=col[s['det']], mfc='none' if N == 25 else col[s['det']], ms=5)
a.axhspan(0.98, 1.02, color='gray', alpha=.15); a.axhline(1, color='gray', lw=.7)
a.set_xticks(range(7)); a.set_xticklabels(bands, fontsize=8); a.set_ylim(0.86, 1.12); a.set_ylabel('R(2440) / R_header')
a.set_title('(a) filled N=0, open N=25; blue b1, green b2, red b3, purple b4, black nrcblong, grey nrcalong', fontsize=7)
a = ax[0, 1]
GB = [(200, 500), (500, 1000), (1000, 2000), (2000, 4000)]
for (b, d), c, ls in ((('150W', 'nrcb1'), 'C0', '-'), (('150W', 'nrcb3'), 'C3', '-'), (('200W', 'nrcb1'), 'C0', '--'), (('200W', 'nrcb3'), 'C3', '--'),
                      (('250M', 'nrcblong'), 'k', '-'), (('300M', 'nrcblong'), 'k', '--')):
    ss = [s for s in S if s['band'] == b and s['det'] == d]
    y = [np.nanmedian([s['far'][k][1] for s in ss]) for k in GB]
    a.plot(range(4), y, color=c, ls=ls, marker='o', ms=4, label=f'F{b} {d}')
a.axhline(1, color='gray', lw=.7); a.set_xticks(range(4)); a.set_xticklabels(['200-500', '500-1k', '1k-2k', '2k-4k']); a.set_xlabel('g0 (DN)')
a.set_ylabel('far-field (edt>=25) cal/g0 / R_header'); a.legend(fontsize=6); a.set_title('(b) far-field level versus g0')
def dmpanel(a, b, det, title):
    s = SC[b]; bins = s['bins']; ref = s['ref']; h = s['have']
    if det:
        h = h & (s['star_det'] == det)
    xs = [(lo + hi) / 2 for lo, hi in bins]
    for v, c, ls in (('final', 'k', '-'), ('RN25+cap', 'C0', '--'), ('RN25+h0+bgfree+cap', 'C0', ':'), ('H+cap', 'C3', '-'), ('H+bgfree+cap', 'C2', '-'), ('H+h0+bgfree+cap', 'C1', '-')):
        y = [np.median(s['dm'][v][h & (ref >= lo) & (ref < hi)]) if (h & (ref >= lo) & (ref < hi)).sum() >= 5 else np.nan for lo, hi in bins]
        a.plot(xs, y, color=c, ls=ls, marker='o', ms=3, label=v)
    a.axhline(s['unsat_dm'], color='gray', lw=.7); a.set_title(title, fontsize=9); a.set_xlabel('ref mag'); a.set_ylabel('median dm')
dmpanel(ax[0, 2], '150W', 'nrcb1', '(c) F150W nrcb1'); dmpanel(ax[0, 3], '150W', 'nrcb3', '(d) F150W nrcb3')
ax[0, 2].legend(fontsize=6)
dmpanel(ax[1, 0], '200W', 'nrcb1', '(e) F200W nrcb1'); dmpanel(ax[1, 1], '200W', 'nrcb3', '(f) F200W nrcb3')
dmpanel(ax[1, 2], '250M', None, '(g) F250M nrcblong'); dmpanel(ax[1, 3], '300M', None, '(h) F300M nrcblong')
fig.tight_layout(); fig.savefig('satrefit6.png', dpi=90)
