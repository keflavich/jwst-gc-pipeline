"""Figures + metrics for fullfield_residual.py outputs (ePSF vs STPSF star subtraction).

    python fullfield_figures.py <o081_<det>.npz> <label> <outprefix>

Writes <outprefix>_maps.png (full-frame residuals, same stretch, and a zoom),
<outprefix>_stats.png (chi^2/pix around stars vs flux and radius; mean stacked
residual of bright stars) and <outprefix>.json (the numbers).
"""
import json
import sys

import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS = [('stpsf', 'STPSF'), ('epsf', 'ePSF (held-out core + wing)'), ('hybrid', 'ePSF core + STPSF wing')]
RB = [0, 1.5, 3, 5, 8, 12, 20, 30]


def main():
    fn, label, outp = sys.argv[1:4]
    z = np.load(fn)
    sci, err, good = z['sci'], z['err'], z['good']
    ncat = int(z['ncat'])
    ny, nx = sci.shape
    names = [m for m, _ in MODELS if f'{m}_resid' in z.files]
    # pixels: exclude 20 px around any masked (saturated / DO_NOT_USE) pixel cluster of >= 5 px,
    # so saturated-star residuals (a different problem) do not enter the PSF comparison
    bad = ~good
    lab, nlab = ndimage.label(bad)
    sizes = ndimage.sum(bad, lab, np.arange(1, nlab + 1))
    big = np.isin(lab, np.nonzero(sizes >= 5)[0] + 1)
    near_sat = ndimage.distance_transform_edt(~big) < 20
    use = good & ~near_sat
    stats = dict(label=label, frame=fn, n_sources=int(len(z['x0'])), n_catalog=ncat)
    # the star part of the model (model image only; the background is separate)
    for m in names:
        r = z[f'{m}_resid']; chi2 = (r / err) ** 2
        starpx = use & (z[f'{m}_model'] > 3 * err)
        stats[m] = dict(chi2_all=float(np.nanmean(chi2[use])), chi2_median_all=float(np.nanmedian(chi2[use])),
                        chi2_starpx=float(np.nanmean(chi2[starpx])), n_starpx=int(starpx.sum()),
                        abs_resid_over_model_starpx=float(np.nansum(np.abs(r[starpx])) / np.nansum(z[f'{m}_model'][starpx])))
    # per-star annular chi^2 around catalog stars, by fitted flux (use the stpsf fit for binning)
    xs, ys = z['stpsf_x'][:ncat], z['stpsf_y'][:ncat]
    fl = z['stpsf_f'][:ncat]
    ix, iy = np.round(xs).astype(int), np.round(ys).astype(int)
    inside = (ix >= 30) & (ix < nx - 30) & (iy >= 30) & (iy < ny - 30) & (fl > 0)
    inside &= ~near_sat[np.clip(iy, 0, ny - 1), np.clip(ix, 0, nx - 1)]
    fb = np.nanpercentile(fl[inside], [0, 50, 80, 95, 99, 100])
    k = np.arange(-30, 31); ry = np.hypot(k[:, None], k[None, :])
    rbin = np.digitize(ry, RB) - 1
    ann = {m: np.zeros((len(fb) - 1, len(RB) - 1, 2)) for m in names}
    stack = {m: np.zeros((61, 61)) for m in names}; nstack = np.zeros((61, 61))
    for j in np.nonzero(inside)[0]:
        sl = np.s_[iy[j] - 30:iy[j] + 31, ix[j] - 30:ix[j] + 31]
        u = use[sl]
        b = np.searchsorted(fb, fl[j], side='right') - 1
        b = min(max(b, 0), len(fb) - 2)
        for m in names:
            c2 = (z[f'{m}_resid'][sl] / err[sl]) ** 2
            for q in range(len(RB) - 1):
                s = u & (rbin == q)
                ann[m][b, q, 0] += np.nansum(c2[s]); ann[m][b, q, 1] += s.sum()
            if b == len(fb) - 2:
                stack[m] += np.where(u, z[f'{m}_resid'][sl], 0) / fl[j]
        if b == len(fb) - 2:
            nstack += u
    stats['flux_bins'] = fb.tolist(); stats['radius_bins'] = RB
    for m in names:
        stats[m]['annular_chi2'] = (ann[m][..., 0] / np.maximum(ann[m][..., 1], 1)).tolist()
    json.dump(stats, open(outp + '.json', 'w'), indent=1)

    # ---------------------------------------------------------------- maps
    fig = plt.figure(figsize=(18, 12.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1])
    lo, hi = np.nanpercentile(z[f'{names[0]}_resid'][use], [2, 98])
    v = max(abs(lo), abs(hi))
    for p, m in enumerate(names):
        ax = fig.add_subplot(gs[0, p])
        ax.imshow(np.where(good, z[f'{m}_resid'], np.nan), origin='lower', cmap='RdBu_r', vmin=-v, vmax=v, interpolation='nearest')
        s = stats[m]
        ax.set_title(f'{dict(MODELS)[m]}\nmean chi2/pix: star px {s["chi2_starpx"]:.1f}, all {s["chi2_all"]:.1f}', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[0, 3])
    d = z['stpsf_model'] + z['stpsf_bkg'] - z['epsf_model'] - z['epsf_bkg'] if 'epsf_model' in z.files else None
    if d is not None:
        ax.imshow(d, origin='lower', cmap='RdBu_r', vmin=-v, vmax=v, interpolation='nearest')
        ax.set_title('model difference STPSF - ePSF\n(= residual ePSF - residual STPSF)', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    # zoom on the densest unsaturated bright-star box
    cnt = np.zeros((ny // 128, nx // 128))
    bright = inside & (fl > fb[-3])
    np.add.at(cnt, (iy[bright] // 128, ix[bright] // 128), 1)
    cnt[near_sat[::128, ::128][:cnt.shape[0], :cnt.shape[1]]] = 0
    jy, jx = np.unravel_index(np.argmax(cnt), cnt.shape)
    y0, x0 = int(np.clip(jy * 128 - 64, 0, ny - 256)), int(np.clip(jx * 128 - 64, 0, nx - 256))
    zs = np.s_[y0:y0 + 256, x0:x0 + 256]
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(np.arcsinh((sci[zs] - np.nanmedian(sci[zs])) / np.nanmedian(err[zs])), origin='lower', cmap='gray_r', interpolation='nearest')
    ax.set_title(f'data, zoom x={x0}..{x0+256}, y={y0}..{y0+256} (asinh)', fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    ez = np.nanmedian(err[zs])
    for p, m in enumerate(names):
        ax = fig.add_subplot(gs[1, p + 1])
        ax.imshow(np.where(good[zs], z[f'{m}_resid'][zs], np.nan), origin='lower', cmap='RdBu_r', vmin=-8 * ez, vmax=8 * ez, interpolation='nearest')
        ax.set_title(f'{dict(MODELS)[m]}: residual, +/-8 sigma', fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'{label}: star-subtracted frame, same catalog, independently fitted fluxes+positions per PSF (colour +/-{v:.2g} MJy/sr)', fontsize=12)
    fig.tight_layout(); fig.savefig(outp + '_maps.png', dpi=70)

    # ---------------------------------------------------------------- stats
    fig, ax = plt.subplots(1, 3 + len(names) - 1, figsize=(6 * (2 + len(names)) / 1.3, 5))
    rc = 0.5 * (np.array(RB[:-1]) + np.array(RB[1:]))
    for b, ls in zip(range(len(fb) - 1), [':', '-.', '--', '-', '-']):
        for m, col in zip(names, ['k', 'C3', 'C0']):
            a = np.array(stats[m]['annular_chi2'])[b]
            ax[0].plot(rc, a, ls, color=col, lw=1 + 0.5 * b, label=f'{m}' if b == len(fb) - 2 else None)
    ax[0].set_yscale('log'); ax[0].set_xlabel('radius from catalog star [px]'); ax[0].set_ylabel('mean chi^2 / pixel')
    ax[0].set_title('around catalog stars; line width = flux bin\n(thin: faintest 50%, thick: brightest 1%)', fontsize=10); ax[0].legend()
    for m, col in zip(names, ['k', 'C3', 'C0']):
        a = np.array(stats[m]['annular_chi2'])
        ax[1].plot(np.arange(len(fb) - 1), a[:, 0], 'o-', color=col, label=f'{m}, r<1.5')
        ax[1].plot(np.arange(len(fb) - 1), a[:, 3], 's--', color=col, label=f'{m}, 5-8 px')
    ax[1].set_xticks(range(len(fb) - 1)); ax[1].set_xticklabels(['0-50%', '50-80%', '80-95%', '95-99%', 'top 1%'])
    ax[1].set_yscale('log'); ax[1].set_xlabel('catalog-star flux percentile'); ax[1].legend(fontsize=8)
    ax[1].set_title('chi^2/pix vs brightness', fontsize=10)
    vv = np.nanpercentile(np.abs(stack[names[0]] / np.maximum(nstack, 1))[nstack > 0], 99)
    for p, m in enumerate(names):
        ax[2 + p].imshow(stack[m] / np.maximum(nstack, 1), origin='lower', cmap='RdBu_r', vmin=-vv, vmax=vv, extent=[-30.5, 30.5, -30.5, 30.5])
        ax[2 + p].set_title(f'{m}: mean residual / flux, brightest 1%\n({int(nstack.max())} stars)', fontsize=10)
    fig.tight_layout(); fig.savefig(outp + '_stats.png', dpi=70)
    print(json.dumps({m: {k: v for k, v in stats[m].items() if k != 'annular_chi2'} for m in names}, indent=1))


if __name__ == '__main__':
    main()
