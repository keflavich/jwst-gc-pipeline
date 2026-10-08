"""Multi-panel figure nrcb3.png from env_results.json, det_variants.json, g0cmp.json, rcurve files.  Usage: nice -19 python -u fig.py"""
import glob
import json
import re
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/nrcb3'
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
env = json.load(open(f'{OUT}/env_results.json'))
dv = json.load(open(f'{OUT}/det_variants.json'))
g0 = json.load(open(f'{OUT}/g0cmp.json'))
C1, C3 = '#1f77b4', '#d62728'
bands = ['F150W', 'F162M', 'F182M', 'F200W']
fig, axs = plt.subplots(3, 4, figsize=(21, 15))
fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.08, hspace=0.5, wspace=0.28)

# A: pair test across the boundary
ax = axs[0, 0]
x = np.arange(4)
for k, (key, lab, c) in enumerate((('R20_mag', 'pairs <20", mag-matched', '#2ca02c'), ('R40_env', 'pairs <40", mag+density+bkg', '#9467bd'))):
    v = [env[b]['pairs'][key] for b in bands]
    ax.errorbar(x + 0.12 * (k - 0.5), [a[0] for a in v], [a[1] for a in v], fmt='o', color=c, label=lab, capsize=3)
    for xi, a in zip(x + 0.12 * (k - 0.5), v):
        ax.annotate(f'N={a[2]}', (xi, a[0]), textcoords='offset points', xytext=(0, -14 if k else 8), ha='center', fontsize=7, color=c)
raw = [env[b]['raw_diff'][0] for b in bands]
ax.plot(x, raw, 'k_', ms=18, mew=2, label='all stars, mag-matched')
ax.axhline(0, color='gray', lw=0.7)
ax.set_xticks(x)
ax.set_xticklabels(bands)
ax.set_ylabel('median dm(nrcb3) - dm(nrcb1)')
ax.set_title('A: no star has S rows on 2 detectors (N=0);\nnearest-pair test across the b1/b3 boundary', fontsize=10)
ax.legend(fontsize=7, loc='lower left')
ax.set_ylim(-0.12, 0.01)

# B summary: stratified estimates
ax = axs[0, 1]
names = ['mag+n2', 'mag+abkg', 'mag+n2+abkg', 'mag+n2+abkg+edge']
for k, nm in enumerate(names):
    ax.errorbar(x + 0.12 * (k - 1.5), [env[b]['strat'][nm][0] for b in bands], [env[b]['strat'][nm][1] for b in bands], fmt='o', label=nm, capsize=3)
ax.plot(x, [env[b]['control']['strat']['mag+n2+abkg'][0] for b in bands], 'ks', mfc='none', label='daophot control (mag+n2+abkg)')
ax.axhline(0, color='gray', lw=0.7)
ax.set_xticks(x)
ax.set_xticklabels(bands)
ax.set_title('B: nrcb3 - nrcb1 at matched environment', fontsize=10)
ax.set_ylabel('stratified median difference')
ax.legend(fontsize=7, loc='lower left')
ax.set_ylim(-0.1, 0.01)

# C: g0 ratio and R curves summary
ax = axs[0, 2]
for k, (q, lab) in enumerate((('g0', 'group 0 core peak'), ('zf', 'ZEROFRAME core peak'))):
    ax.errorbar(x + 0.1 * (k - 0.5), [g0[f'{b}_{q}']['mean'] for b in bands], [g0[f'{b}_{q}']['err'] for b in bands], fmt='o', label=lab, capsize=3)
ax.plot(x, [env[b]['raw_diff'][0] for b in bands], 'kx', label='satstar dm b3 - b1')
ax.axhline(0, color='gray', lw=0.7)
ax.set_xticks(x)
ax.set_xticklabels(bands)
ax.set_title('C: core DN at fixed dolphot mag, 2.5 log10(b3/b1)', fontsize=10)
ax.legend(fontsize=7)

# mag dependence of b3-b1 raw
ax = axs[0, 3]
for b, c in zip(bands, ['C0', 'C1', 'C2', 'C3']):
    z = env[b]
    ax.plot([0], [0], alpha=0)
ax.axis('off')
txt = ['C: F150W reference data', 'saturation ref (DN); linearity', 'correction at 50/70/90% sat;', 'gain 2.05 on all detectors']
ref = json.load(open(f'{OUT}/refdata.json'))
for det in ['nrcb1', 'nrcb2', 'nrcb3', 'nrcb4', 'nrca1', 'nrca3']:
    r = ref[f'F150W_{det}']
    txt.append(f"{det}: sat {r['sat_full']:.0f}\n   lin {r['lin_full'][0]:.3f}/{r['lin_full'][1]:.3f}/{r['lin_full'][2]:.3f}")
ax.text(0.0, 1.0, '\n'.join(txt), va='top', fontsize=8, family='monospace')

# B: dm vs density, vs background
def binplot(ax, band, var, xl, log=False):
    for det, c, key in (('nrcb1', C1, 'm1'), ('nrcb3', C3, 'm3')):
        bl = env[band]['bins'][var]
        xs = np.array([(b['lo'] + b['hi']) / 2 for b in bl])
        if var in ('abkg', 'lbkg'):
            xs = np.sqrt(np.array([max(b['lo'], 1e-3) * min(b['hi'], 40) for b in bl]))
        ys = [b[key] for b in bl]
        ns = [b['n1' if det == 'nrcb1' else 'n3'] for b in bl]
        ax.plot(xs, ys, '-o', color=c, label=det)
        for xx, yy, nn in zip(xs, ys, ns):
            ax.annotate(str(nn), (xx, yy), textcoords='offset points', xytext=(0, 6 if det == 'nrcb1' else -11), ha='center', fontsize=6.5, color=c)
    if log:
        ax.set_xscale('log')
    ax.set_xlabel(xl)
    ax.axhline(0, color='gray', lw=0.7)
    ax.set_ylabel('median dm (final)')
    ax.legend(fontsize=7, loc='lower right')
binplot(axs[1, 0], 'F150W', 'n2', 'F150W: dolphot neighbours within 2" (mag < own+3)')
binplot(axs[1, 1], 'F200W', 'n2', 'F200W: dolphot neighbours within 2" (mag < own+3)')
binplot(axs[1, 2], 'F150W', 'abkg', 'F150W: annulus background 1.5-2.5" (MJy/sr)', log=True)
binplot(axs[1, 3], 'F200W', 'abkg', 'F200W: annulus background 1.5-2.5" (MJy/sr)', log=True)
for a in axs[1]:
    a.set_ylim(-0.13, 0.02)
axs[1, 0].set_title('B: dm per bin, labels = N stars', fontsize=10)

# D: variants by detector
VARS = ['final', 'uncapped', 'bgfree', 'bgfree+cap', 'rw12', 'rw12+cap', 'rw12+bgfree', 'rw12+bgfree+cap', 'bgfree+v7b+cap']
for k, band in enumerate(['F150W', 'F200W']):
    ax = axs[2, k]
    t = dv[band]['table']
    xx = np.arange(len(VARS))
    y1 = [t[f'{v}|nrcb1|-99-99'][0] for v in VARS]
    y3 = [t[f'{v}|nrcb3|-99-99'][0] for v in VARS]
    e1 = [t[f'{v}|nrcb1|-99-99'][1] for v in VARS]
    e3 = [t[f'{v}|nrcb3|-99-99'][1] for v in VARS]
    ax.errorbar(xx - 0.07, y1, e1, fmt='o', color=C1, label='nrcb1', capsize=2)
    ax.errorbar(xx + 0.07, y3, e3, fmt='o', color=C3, label='nrcb3', capsize=2)
    ax.plot(xx, np.array(y3) - np.array(y1), 'k^', label='b3 - b1')
    ax.axhline(0, color='gray', lw=0.7)
    ax.set_xticks(xx)
    ax.set_xticklabels(VARS, rotation=40, ha='right', fontsize=7)
    ax.set_ylabel('median dm, all mags')
    ax.set_title(f'D: round-3 variants, {band} (refit frames)', fontsize=10)
    ax.legend(fontsize=7, loc='center left')
# R(g0)
for k, band in enumerate(['150W', '200W']):
    ax = axs[2, 2 + k]
    for fn in sorted(glob.glob(f'{Q}/satrefit/out3/{band}_*_rcurve.txt')):
        m = re.search(r'_(nrcb[13])_', os.path.basename(fn))
        a = np.loadtxt(fn)
        ax.plot(a[:, 0], a[:, 1], '-', color=C1 if m.group(1) == 'nrcb1' else C3, alpha=0.7, label=m.group(1) if fn == sorted(glob.glob(f'{Q}/satrefit/out3/{band}_*_rcurve.txt'))[0] or m.group(1) == 'nrcb3' and 'nrcb3' not in [l.get_label() for l in ax.lines] else None)
    ax.set_xscale('log')
    ax.set_xlabel('g0 (DN)')
    ax.set_ylabel('R(g0)')
    ax.set_title(f'D: R(g0) per frame, F{band}', fontsize=10)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, fontsize=7)
fig.savefig(f'{OUT}/nrcb3.png', dpi=100)
