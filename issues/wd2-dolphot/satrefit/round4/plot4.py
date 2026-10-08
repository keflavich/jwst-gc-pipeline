import pickle, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/out4'
fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))


def runmed(x, y, edges):
    c, m = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (x >= lo) & (x < hi)
        if s.sum() >= 8:
            c.append(0.5 * (lo + hi))
            m.append(np.median(y[s]))
    return np.array(c), np.array(m)


VARS = [('final', 'k'), ('rw12', 'C0'), ('rw12h_e0', 'C2'), ('rw12h_e1', 'C1'), ('norim+cap', 'C3')]
for a_, band, edges in ((ax[0, 0], '150W', np.arange(13.5, 19.6, 0.5)), (ax[0, 1], '200W', np.arange(13.5, 18.6, 0.5))):
    d = pickle.load(open(f'{O}/score4_{band}.pkl', 'rb'))
    ref = np.asarray(d['ref'], float)
    d4 = pickle.load(open(f'{O}/score4b_{band}.pkl', 'rb')) if os.path.exists(f'{O}/score4b_{band}.pkl') else None
    for v, col in VARS:
        for det, ls in (('nrcb1', '-'), ('nrcb3', '--')):
            if v == 'norim+cap':
                if d4 is None:
                    continue
                y, h, r_, sd = d4['dm'][v], d4['have'], np.asarray(d4['ref'], float), d4['star_det']
                s = h & (sd == det)
                x, m = runmed(r_[s], y[s], edges)
            else:
                s = d['have'] & (d['star_det'] == det)
                x, m = runmed(ref[s], d['dm'][v][s], edges)
            a_.plot(x, m, ls, color=col, marker='o', ms=3, label=f'{v} {det}' if band == '150W' else None)
    a_.axhline(0, color='gray', lw=0.5)
    a_.set_xlabel(f'dolphot F{band} mag')
    a_.set_ylabel('median dm (ours - dolphot - ZP)')
    a_.set_title(f'F{band} SW: solid nrcb1, dashed nrcb3')
    a_.set_ylim(-0.3, 0.1)
ax[0, 0].legend(fontsize=6, ncol=2)
# LW q
q = pickle.load(open(f'{O}/q_lw.pkl', 'rb'))
dl = q['dlab']
xs = np.arange(len(dl))
for band, ls in (('F250M', '-'), ('F300M', '--')):
    for g, col in (('all', 'k'), ('Q1 faint', 'C0'), ('Q4 bright', 'C3')):
        y = [q['res'][(band, g, k)][0] for k in dl]
        e = [q['res'][(band, g, k)][1] for k in dl]
        ax[0, 2].errorbar(xs, y, e, ls=ls, color=col, marker='o', ms=3, label=f'{band} {g}')
ax[0, 2].axhline(1, color='gray', lw=0.5)
ax[0, 2].set_xticks(xs)
ax[0, 2].set_xticklabels(dl)
ax[0, 2].set_xlabel('d from DQ-SATURATED (px)')
ax[0, 2].set_ylabel('q = cal / (R g0)')
ax[0, 2].set_title('LW nrcblong q (solid F250M, dashed F300M)')
ax[0, 2].legend(fontsize=6)
# per-pixel ratio
gE = np.array([200, 400, 800, 1600, 3200, 6400, 12800, 40000])
gc = np.sqrt(gE[:-1] * gE[1:])
for a_, band in ((ax[1, 0], '150W'), (ax[1, 1], '200W')):
    f = f'{O}/score4b_{band}.pkl'
    if not os.path.exists(f):
        continue
    p = pickle.load(open(f, 'rb'))['pix']
    for cat, ls, nm in ((0, '-', 'rim (rewritten)'), (1, '--', 'never-rewritten crf')):
        for det, col in ((1, 'C0'), (3, 'C3')):
            ys = []
            for lo, hi in zip(gE[:-1], gE[1:]):
                s = (p['cat'] == cat) & (p['det'] == det) & (p['g0'] >= lo) & (p['g0'] < hi) & (p['r'] < 6)
                ys.append(np.median(p['ratio'][s]) if s.sum() >= 20 else np.nan)
            a_.plot(gc, ys, ls, color=col, marker='o', ms=3, label=f'{nm} nrcb{det}')
    a_.set_xscale('log')
    a_.set_xlabel('g0 (DN)')
    a_.set_ylabel('median data / (F_dolphot PSF), r < 6 px')
    a_.set_title(f'F{band} per-pixel ratio')
    a_.set_ylim(0.8, 1.5)
    a_.legend(fontsize=6)
# norim a ratio
a_ = ax[1, 2]
for band, col in (('150W', 'C0'), ('200W', 'C1')):
    d4 = pickle.load(open(f'{O}/score4b_{band}.pkl', 'rb'))
    ref = np.asarray(d4['ref'], float)
    edges = np.arange(13.5, 19.6, 0.5)
    for det, ls in (('nrcb1', '-'), ('nrcb3', '--')):
        s = d4['have'] & (d4['star_det'] == det) & np.isfinite(d4['ratio']['norim'])
        x, m = runmed(ref[s], d4['ratio']['norim'][s], edges)
        a_.plot(x, m, ls, color=col, marker='o', ms=3, label=f'F{band} {det}')
a_.axhline(1, color='gray', lw=0.5)
a_.set_xlabel('dolphot mag')
a_.set_ylabel('a_norim / a_base (median)')
a_.set_title('norim amplitude ratio')
a_.legend(fontsize=6)
fig.tight_layout()
fig.savefig('/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/satrefit4.png', dpi=110)
