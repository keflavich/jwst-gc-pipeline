"""Paired per-star comparison of the fullfield_residual.py fits (isolated catalog stars).

For each filter: median chi^2/pix in annuli around isolated catalog stars (no source
brighter than 10% of the star within 8 px; > 30 px from any saturated core), per
flux bin, for STPSF vs held-out ePSF vs hybrid, from the SAME frame and catalog;
plus a gallery of the brightest isolated stars (data, STPSF residual, ePSF residual,
same stretch in units of ERR).

    python fullfield_paired_fig.py <out.png> <label>:<o081_det.npz>:<fullfield_paired npz>:<clip sigma> [...]
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
DPI = int(os.environ.get('FIG_DPI', 150))  # 150: text legible when the figure is shown at page width
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

A = [(0, 1.5), (1.5, 3), (3, 6), (6, 10)]
PCT = ['0-50%', '50-80%', '80-95%', 'top 5%']
NG = 5


def main():
    out = sys.argv[1]
    sets = [a.split(':') for a in sys.argv[2:]]
    fig = plt.figure(figsize=(22, 6.8 * len(sets)))
    gs = fig.add_gridspec(len(sets), 1 + NG, width_ratios=[2.2] + [1] * NG, hspace=0.42, wspace=0.28)
    for row, (label, fn, pfn, clip) in enumerate(sets):
        clip = float(clip)
        z = np.load(fn); p = np.load(pfn)
        sel, f = p['sel'], p['f']
        qs = np.percentile(f, [0, 50, 80, 95, 100])
        ax = fig.add_subplot(gs[row, 0])
        rc = [0.5 * (a + b) for a, b in A]
        for b, col in zip(range(4), ['0.7', 'C0', 'C2', 'C3']):
            mm = (f >= qs[b]) & (f <= qs[b + 1])
            S = np.nanmedian(p['C_stpsf'][mm], 0); E = np.nanmedian(p['C_epsf'][mm], 0)
            ax.plot(rc, S, 'o--', color=col, mfc='none')
            ax.plot(rc, E, 'o-', color=col, label=f'{PCT[b]} (N={mm.sum()})')
        ax.axhline(1, color='k', lw=0.5)
        ax.set_yscale('log'); ax.set_xlabel('radius from star [px]'); ax.set_ylabel('median chi^2 / pixel')
        ax.set_title(f'{label}, {len(sel)} isolated stars\n'
                     f'dashed = STPSF, solid = held-out ePSF', fontsize=13)
        ax.legend(fontsize=11, title='flux percentile', title_fontsize=11)
        # gallery: bright isolated stars (top 10%, no masked pixel within 5 px), evenly spread in flux
        cand = []
        for t in np.argsort(f)[::-1][:max(NG, len(f) // 10)]:
            j = sel[t]; ix, iy = int(round(z['stpsf_x'][j])), int(round(z['stpsf_y'][j]))
            if z['good'][iy - 5:iy + 6, ix - 5:ix + 6].all():
                cand.append(t)
        top = [cand[k] for k in np.linspace(0, len(cand) - 1, NG).astype(int)]
        for q, t in enumerate(top):
            j = sel[t]
            x, y = z['stpsf_x'][j], z['stpsf_y'][j]
            ix, iy = int(round(x)), int(round(y))
            s = np.s_[iy - 12:iy + 13, ix - 12:ix + 13]
            e = z['err'][s]; gd = z['good'][s]
            sub = gs[row, 1 + q].subgridspec(3, 1, hspace=0.05)
            d = z['sci'][s] - z['stpsf_bkg'][s]
            a0 = fig.add_subplot(sub[0]); dv = np.arcsinh(d / np.nanmedian(e)); lo, hi = np.nanpercentile(dv, [1, 99.5])
            a0.imshow(dv, origin='lower', cmap='gray_r', vmin=lo, vmax=hi)
            cS, cE = p['C_stpsf'][t], p['C_epsf'][t]
            a0.set_title(f'({ix}, {iy})\nchi2(r<3) S {np.nanmean(cS[:2]):.0f} / E {np.nanmean(cE[:2]):.0f}', fontsize=11)
            for k, m in enumerate(['stpsf', 'epsf']):
                a = fig.add_subplot(sub[1 + k])
                a.imshow(np.where(gd, z[f'{m}_resid'][s] / e, np.nan), origin='lower', cmap='RdBu_r', vmin=-clip, vmax=clip)
                a.set_ylabel(f'{m}\n+/-{clip:.0f} sigma', fontsize=11)
            for a in fig.axes[-3:]:
                a.set_xticks([]); a.set_yticks([])
    fig.suptitle('o081 dither 1: same frame, same catalog, fluxes + positions refitted per PSF\n'
                 'gallery (bright isolated stars, top 10%): data (asinh) / STPSF residual / ePSF residual, in units of ERR', fontsize=15)
    fig.savefig(out, dpi=DPI, bbox_inches='tight')


if __name__ == '__main__':
    main()
