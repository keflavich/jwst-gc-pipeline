"""Report figures for the ePSF-vs-STPSF full-frame star subtraction (fullfield_residual.py).

    python fullfield_report_figs.py <outdir> <label>:<o081_det.npz>:<fullfield_paired npz> [...]

Writes per detector:
  <det>_stack.png   mean residual / flux of the brightest 20% isolated stars, STPSF vs
                    ePSF, and its azimuthal profile (the systematic PSF-shape error)
  <det>_zoom.png    two 160x160 px regions free of saturated cores: data, STPSF residual,
                    ePSF residual (same stretch), and where the ePSF is better
  <det>_budget.png  where the frame's excess chi^2 (sum of chi^2 - 1) lives, per pixel
                    class, for each model; plus annular chi^2 around stars compared with
                    the same annuli around random positions (the environment floor)
and <det>_report.json with the numbers.
"""
import json
import os
import sys

import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS = [('stpsf', 'STPSF'), ('epsf', 'held-out ePSF')]
H = 15          # stack half-size [px]
A = [(0, 1.5), (1.5, 3), (3, 6), (6, 10)]


def sat_mask(good, min_px=30, dist=30):
    bad = ~good
    lab, nl = ndimage.label(bad)
    sz = ndimage.sum(bad, lab, np.arange(1, nl + 1))
    big = np.isin(lab, np.nonzero(sz >= min_px)[0] + 1)
    return ndimage.distance_transform_edt(~big) <= dist


def stack_fig(z, p, label, out):
    sel, f = p['sel'], p['f']
    top = sel[f >= np.percentile(f, 80)]
    k = np.arange(-H, H + 1)
    S = {m: np.zeros((2 * H + 1, 2 * H + 1)) for m, _ in MODELS}; N = np.zeros((2 * H + 1, 2 * H + 1))
    for j in top:
        ix, iy = int(round(z['stpsf_x'][j])), int(round(z['stpsf_y'][j]))
        s = np.s_[iy - H:iy + H + 1, ix - H:ix + H + 1]
        g = z['good'][s]
        if g.shape != N.shape:
            continue
        for m, _ in MODELS:
            # residual in units of the star's own fitted flux (per model), so the stack is
            # the mean fractional PSF error of that model
            S[m] += np.where(g, z[f'{m}_resid'][s] / z[f'{m}_f'][j], 0)
        N += g
    rr = np.hypot(k[:, None], k[None])
    fig = plt.figure(figsize=(16, 4.6))
    gsp = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.05, 1.25], wspace=0.3)
    ax = [fig.add_subplot(gsp[0]), fig.add_subplot(gsp[1]), fig.add_subplot(gsp[3])]
    cax = fig.add_subplot(gsp[2])
    means = {m: S[m] / np.maximum(N, 1) for m, _ in MODELS}
    v = np.percentile(np.abs(means['stpsf'][rr <= 8]), 99)
    for q, (m, lab) in enumerate(MODELS):
        im = ax[q].imshow(means[m], origin='lower', cmap='RdBu_r', vmin=-v, vmax=v, extent=[-H - .5, H + .5, -H - .5, H + .5])
        ax[q].set_title(f'{lab}: mean residual / flux\nbrightest 20% of isolated stars (N={len(top)})', fontsize=10)
        ax[q].set_xlabel('px')
    plt.colorbar(im, cax=cax, label='fraction of stellar flux per pixel')
    rb = np.arange(0, 12.5, 0.5)
    prof = {}
    for m, lab in MODELS:
        prof[m] = [float(np.mean(means[m][(rr >= a) & (rr < b)])) for a, b in zip(rb[:-1], rb[1:])]
        ax[2].plot(0.5 * (rb[:-1] + rb[1:]), prof[m], 'o-', ms=3, label=lab)
    ax[2].set_yscale('symlog', linthresh=1e-4)
    ax[2].axhline(0, color='k', lw=.5); ax[2].set_xlabel('radius [px]'); ax[2].set_ylabel('azimuthal mean of (residual / flux)')
    ax[2].set_title('systematic PSF-shape error (0 = perfect model)', fontsize=10); ax[2].legend()
    fig.suptitle(f'{label}: what each PSF model leaves behind around bright isolated stars', fontsize=11)
    fig.savefig(out, dpi=75, bbox_inches='tight'); plt.close(fig)
    # summary: rms of the mean pattern within r <= 8 px, as a fraction of peak
    return {m: dict(rms_pattern_r8=float(np.sqrt(np.mean(means[m][rr <= 8] ** 2))), profile=prof[m]) for m, _ in MODELS} | {'n_stack': int(len(top))}


def zoom_fig(z, label, out, near_sat, nzoom=2, W=160):
    ny, nx = z['sci'].shape
    n = int(z['ncat'])
    # regions: most catalog stars with a clean (unmasked) core, no saturated core in the box
    x, y, fl = z['stpsf_x'][:n], z['stpsf_y'][:n], z['stpsf_f'][:n]
    bright = fl > np.percentile(fl[fl > 0], 70)
    step = W // 2
    best = []
    for y0 in range(0, ny - W, step):
        for x0 in range(0, nx - W, step):
            fsat = near_sat[y0:y0 + W, x0:x0 + W].mean()
            if fsat > 0.10:          # F480M (BRIGHT2) has no box entirely free of saturated cores
                continue
            c = np.sum(bright & (x > x0 + 8) & (x < x0 + W - 8) & (y > y0 + 8) & (y < y0 + W - 8))
            best.append((c, y0, x0))
    best.sort(reverse=True)
    picks = []
    for c, y0, x0 in best:
        if all(abs(y0 - a) >= W or abs(x0 - b) >= W for a, b in picks):
            picks.append((y0, x0))
        if len(picks) == nzoom:
            break
    fig, ax = plt.subplots(len(picks), 4, figsize=(17, 4.4 * len(picks)), squeeze=False)
    stats = []
    for r, (y0, x0) in enumerate(picks):
        s = np.s_[y0:y0 + W, x0:x0 + W]
        e = z['err'][s]; g = z['good'][s]
        d = z['sci'][s] - z['stpsf_bkg'][s]
        dv = np.arcsinh(d / np.nanmedian(e)); lo, hi = np.nanpercentile(dv, [1, 99.7])
        ax[r, 0].imshow(dv, origin='lower', cmap='gray_r', vmin=lo, vmax=hi)
        ax[r, 0].set_title(f'data - bkg (asinh), x={x0}..{x0+W}, y={y0}..{y0+W}\n(blank in residuals: < 30 px from a saturated core)', fontsize=9)
        g = g & ~near_sat[s]       # saturated-core surroundings shown blank: not a PSF comparison
        chi = {m: np.where(g, z[f'{m}_resid'][s] / e, np.nan) for m, _ in MODELS}
        for q, (m, lab) in enumerate(MODELS):
            ax[r, 1 + q].imshow(chi[m], origin='lower', cmap='RdBu_r', vmin=-5, vmax=5)
            ax[r, 1 + q].set_title(f'{lab} residual / ERR (+/-5)\nsum chi^2 = {np.nansum(chi[m]**2):.3g}', fontsize=9)
        imp = ndimage.gaussian_filter(np.nan_to_num(chi['stpsf'] ** 2 - chi['epsf'] ** 2), 1.0)
        vv = np.percentile(np.abs(imp), 99.5)
        ax[r, 3].imshow(imp, origin='lower', cmap='PuOr_r', vmin=-vv, vmax=vv)
        ax[r, 3].set_title('chi^2(STPSF) - chi^2(ePSF), smoothed 1 px\norange = ePSF better, purple = STPSF better', fontsize=9)
        for a in ax[r]:
            a.set_xticks([]); a.set_yticks([])
        stats.append(dict(frac_near_sat=float(near_sat[s].mean()), x0=x0, y0=y0, W=W, chi2_stpsf=float(np.nansum(chi['stpsf'] ** 2)), chi2_epsf=float(np.nansum(chi['epsf'] ** 2)),
                          npix=int(g.sum())))
    fig.suptitle(f'{label}: zooms with no saturated core (same catalog, per-model refit)', fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=72); plt.close(fig)
    return stats


def budget_fig(z, p, label, out, near_sat):
    good = z['good']; err = z['err']; ny, nx = good.shape
    n = int(z['ncat'])
    x, y, fl = z['stpsf_x'], z['stpsf_y'], z['stpsf_f']
    # pixel classes (mutually exclusive, in priority order)
    iso = np.zeros(good.shape, bool)
    ixs, iys = np.round(x[p['sel']]).astype(int), np.round(y[p['sel']]).astype(int)
    iso[iys, ixs] = True
    near_iso = ndimage.distance_transform_edt(~iso) <= 6
    allc = np.zeros(good.shape, bool)
    ok = (fl > 0)
    ix, iy = np.clip(np.round(x[ok]).astype(int), 0, nx - 1), np.clip(np.round(y[ok]).astype(int), 0, ny - 1)
    allc[iy, ix] = True
    near_any = ndimage.distance_transform_edt(~allc) <= 3
    classes = [('within 30 px of a saturated core', near_sat & good),
               ('within 6 px of an isolated catalog star', near_iso & good & ~near_sat),
               ('within 3 px of any other fitted source', near_any & good & ~near_sat & ~near_iso),
               ('elsewhere (background / confusion)', good & ~near_sat & ~near_iso & ~near_any)]
    res = {}
    for m, _ in MODELS:
        c2 = (z[f'{m}_resid'] / err) ** 2
        res[m] = [float(np.nansum(np.clip(c2[c] - 1, 0, None))) for _, c in classes]
    frac_px = [float(c.sum() / good.sum()) for _, c in classes]
    # annular chi^2: isolated stars vs random positions away from sources and saturation
    rng = np.random.default_rng(0)
    free = good & ~near_sat & ~near_any
    fy, fx = np.nonzero(free[20:-20, 20:-20]); pick = rng.choice(len(fy), 3000, replace=False)
    ry, rx = fy[pick] + 20, fx[pick] + 20
    k = np.arange(-10, 11); rr = np.hypot(k[:, None], k[None])
    rand = {m: np.zeros(len(A)) for m, _ in MODELS}
    for m, _ in MODELS:
        vals = [[] for _ in A]
        for yy, xx in zip(ry, rx):
            s = np.s_[yy - 10:yy + 11, xx - 10:xx + 11]
            c2 = (z[f'{m}_resid'][s] / err[s]) ** 2; u = good[s]
            for a, (r0, r1) in enumerate(A):
                w = u & (rr >= r0) & (rr < r1)
                if w.sum():
                    vals[a].append(c2[w].mean())
        rand[m] = [float(np.median(v)) for v in vals]
    f = p['f']; topm = f >= np.percentile(f, 80)
    star = {m: [float(v) for v in np.nanmedian(p[f'C_{m}'][topm], 0)] for m, _ in MODELS}
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
    yb = np.arange(len(classes))
    tot = {m: sum(res[m]) for m, _ in MODELS}
    for q, (m, lab) in enumerate(MODELS):
        ax[0].barh(yb + (0.2 if q == 0 else -0.2), np.array(res[m]) / tot['stpsf'], height=0.38, label=lab)
    ax[0].set_yticks(yb); ax[0].set_yticklabels([f'{c}\n({100*fp:.0f}% of pixels)' for (c, _), fp in zip(classes, frac_px)], fontsize=8)
    ax[0].set_xlabel('share of the frame\'s excess chi^2  sum(chi^2 - 1), STPSF total = 1')
    ax[0].set_title(f'where the residual lives: total STPSF {tot["stpsf"]:.3g}, ePSF {tot["epsf"]:.3g} '
                    f'({100*(1-tot["epsf"]/tot["stpsf"]):.1f}% lower)', fontsize=10)
    ax[0].legend(); ax[0].invert_yaxis()
    rc = [0.5 * (a + b) for a, b in A]
    for m, lab in MODELS:
        ls = '--' if m == 'stpsf' else '-'
        ax[1].plot(rc, star[m], 'o' + ls, color='C3', label=f'brightest 20% isolated stars, {lab}')
        ax[1].plot(rc, rand[m], 's' + ls, color='0.4', label=f'random source-free positions, {lab}')
    ax[1].axhline(1, color='k', lw=.5); ax[1].set_yscale('log'); ax[1].set_xlabel('radius [px]'); ax[1].set_ylabel('median chi^2 / pixel')
    ax[1].set_title('annular chi^2 around stars vs around random positions\n(the environment floor)', fontsize=10); ax[1].legend(fontsize=8)
    fig.suptitle(f'{label}: residual budget', fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=75); plt.close(fig)
    return dict(classes=[c for c, _ in classes], frac_pixels=frac_px, excess_chi2=res, annular_star_top20=star, annular_random=rand)


def main():
    outdir = sys.argv[1]
    for arg in sys.argv[2:]:
        label, fn, pfn = arg.split(':')
        det = os.path.basename(fn).replace('.npz', '').split('_')[-1]
        z = np.load(fn); p = np.load(pfn)
        near_sat = sat_mask(z['good'])
        rep = dict(label=label)
        rep['stack'] = stack_fig(z, p, label, os.path.join(outdir, f'{det}_stack.png'))
        rep['zoom'] = zoom_fig(z, label, os.path.join(outdir, f'{det}_zoom.png'), near_sat)
        rep['budget'] = budget_fig(z, p, label, os.path.join(outdir, f'{det}_budget.png'), near_sat)
        json.dump(rep, open(os.path.join(outdir, f'{det}_report.json'), 'w'), indent=1)
        print(label, json.dumps({k: v for k, v in rep.items() if k != 'stack'} | {'stack_rms': {m: rep['stack'][m]['rms_pattern_r8'] for m, _ in MODELS}}, indent=1))


if __name__ == '__main__':
    main()
