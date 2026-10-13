"""#1151 check: does the jicama per-frame PSF flux on the MJy/sr crf track the pixel-area-corrected aperture flux?
Compares the pre-#1154 tree (main2, transposed grid; apclosure/frames_<band>.ecsv) with the fixed-grid tree
(main2gt; pam1151/frames_main2gt_<band>.ecsv).  Same isolated stars, same apertures (r = 5 px, local annulus).
Differences (mag): (e) psf - ap_area, ours against the area-corrected aperture (truth up to a constant);
(c) dolphot - ap_area.  Fit d = A*pred + B*predT + const, pred = 2.5 log10(AREA) at the star, predT = same at the
transposed pixel.  Correct pixel-area handling gives A = B = 0.
usage: python analyze_gt.py"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
sys.path.insert(0, f'{Q}/apclosure')
sys.argv = sys.argv[:1]
import importlib.util
spec = importlib.util.spec_from_file_location('ap', f'{Q}/apclosure/analyze_ap.py')
src = open(f'{Q}/apclosure/analyze_ap.py').read().replace('\nmain()\n', '\n')
ap = importlib.util.module_from_spec(spec)
exec(compile(src, spec.origin, 'exec'), ap.__dict__)
HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = {'main2 (pre-#1154 loader)': f'{Q}/apclosure/frames_{{b}}.ecsv',
        'main2gt (#1154 loader)': f'{HERE}/frames_main2gt_{{b}}.ecsv'}
R = 5


def load(arm, band):
    """ap.stars() reads <HERE>/frames_<band>.ecsv; gt/ holds links named that way to frames_main2gt_<band>.ecsv."""
    fn = ARMS[arm].format(b=band)
    if not os.path.exists(fn):
        return None
    ap.HERE = f'{Q}/apclosure' if 'apclosure' in fn else f'{HERE}/gt'
    rec = ap.stars(band)
    return None if rec is None or len(rec) < 30 else ap.diffs(band, rec, R)


res = {}
L = ['| band | arm | (e) ours − ap_area: A (pred) | B (predT) | Theil-Sen slope on pred − predT | robust σ | N |', '|---|---|---|---|---|---|---|']
Lc = ['| band | (c) dolphot − ap_area: A (pred) | B (predT) | N |', '|---|---|---|---|']
for band in ap.BANDS:
    for arm in ARMS:
        D = load(arm, band)
        if D is None:
            continue
        c, lo, hi, n, rho = ap.fit2(D['e'], D['pred'], D['predT'])
        s = ap.slope(D['pred'] - D['predT'], D['e'])
        res[(band, arm)] = (D, c, lo, hi, n)
        L.append(f'| F{band} | {arm} | {c[0]:+.2f} [{lo[0]:+.2f}, {hi[0]:+.2f}] | {c[1]:+.2f} [{lo[1]:+.2f}, {hi[1]:+.2f}] | {s[0]:+.2f} [{s[1]:+.2f}, {s[2]:+.2f}] | {s[3]:.4f} | {n} |')
    k = (band, 'main2gt (#1154 loader)')
    if k in res:
        D = res[k][0]
        c, lo, hi, n, rho = ap.fit2(D['c'], D['pred'], D['predT'])
        res[(band, 'dolphot')] = (D, c, lo, hi, n)
        Lc.append(f'| F{band} | {c[0]:+.2f} [{lo[0]:+.2f}, {hi[0]:+.2f}] | {c[1]:+.2f} [{lo[1]:+.2f}, {hi[1]:+.2f}] | {n} |')
open(f'{HERE}/pam1151.md', 'w').write('\n'.join(L) + '\n\n' + '\n'.join(Lc) + '\n')
print(open(f'{HERE}/pam1151.md').read())

# Figure 1: A and B per band, three arms
bands = [b for b in ap.BANDS if (b, 'main2gt (#1154 loader)') in res]
order = sorted(bands, key=lambda b: int(b[:3]))
arms = [('main2 (pre-#1154 loader)', '#b0b0b0', 'ours, pre-#1154 grid loader (main2)'),
        ('main2gt (#1154 loader)', '#1f77b4', 'ours, #1154 grid loader (main2gt)'),
        ('dolphot', '#d62728', 'dolphot (nircammask × AREA)')]
fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
x = np.arange(len(order))
for j, (ax, nm) in enumerate(zip(axs, ('A: coefficient on pred = 2.5 log10 AREA(star)', 'B: coefficient on predT = 2.5 log10 AREA(transposed pixel)'))):
    for k, (arm, col, lab) in enumerate(arms):
        v = np.array([res[(b, arm)][1][j] if (b, arm) in res else np.nan for b in order])
        lo = np.array([res[(b, arm)][2][j] if (b, arm) in res else np.nan for b in order])
        hi = np.array([res[(b, arm)][3][j] if (b, arm) in res else np.nan for b in order])
        ax.errorbar(x + (k - 1) * 0.22, v, [v - lo, hi - v], fmt='o', color=col, ms=6, capsize=2, label=lab)
    for yv in (-1, 1):
        ax.axhline(yv, color='0.75', ls=':', lw=1)
    ax.axhline(0, color='k', lw=1)
    ax.set_ylabel(nm.split(':')[0])
    ax.set_title(nm, fontsize=10, loc='left')
    ax.set_ylim(-2.6, 2.6)
axs[0].legend(fontsize=9, loc='upper left', ncol=3)
axs[1].set_xticks(x)
axs[1].set_xticklabels([f'F{b}' for b in order])
fig.suptitle('PSF magnitude − pixel-area-corrected aperture magnitude (r = 5 px, isolated stars): fit d = A·pred + B·predT + c\n'
             'correct pixel-area handling: A = B = 0', fontsize=11)
fig.savefig(f'{HERE}/pam1151_coef.png', dpi=110)

# Figure 2: binned d vs pred, four bands
show = [b for b in ('182M', '200W', '212N', '300M', '323N') if b in bands]
fig, axs = plt.subplots(1, len(show), figsize=(4.6 * len(show), 4.4), sharey=True, constrained_layout=True, squeeze=False)
for ax, band in zip(axs[0], show):
    for arm, col, lab in arms:
        D = res[(band, arm)][0]
        d = D['c'] if arm == 'dolphot' else D['e']
        d = d - np.median(d)
        edges = np.quantile(D['pred'], np.linspace(0, 1, 9))
        xm, ym, ye = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = (D['pred'] >= lo) & (D['pred'] <= hi)
            xm.append(np.median(D['pred'][s]))
            ym.append(np.median(d[s]))
            ye.append(ap.rs(d[s]) / np.sqrt(s.sum()) * 1.25)
        ax.errorbar(xm, ym, ye, fmt='o-', color=col, ms=4, label=lab)
    xs = np.array([-0.03, 0.04])
    ax.plot(xs, -xs, color='0.6', ls=':', lw=1, label='slope −1')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_title(f'F{band}')
    ax.set_xlabel('pred = 2.5 log10 AREA (mag)')
    ax.set_xlim(np.percentile(D['pred'], 0.5) - 0.004, np.percentile(D['pred'], 99.5) + 0.004)
axs[0][0].set_ylabel('PSF − area-corrected aperture (mag), median removed')
axs[0][0].legend(fontsize=7.5, loc='lower left')
fig.savefig(f'{HERE}/pam1151_binned.png', dpi=110)
print('wrote figures')
