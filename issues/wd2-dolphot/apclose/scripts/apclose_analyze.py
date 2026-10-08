"""Aggregate apclose/data/stars_*.fits into apclose.md and apclose.png."""
import os
import numpy as np
from astropy.table import Table
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

AP = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/apclose'
ARMS = ['main2kf', 'main2']
BANDS = ['F250M', 'F150W']
ANN = {'F250M': [(5, 8), (8, 12), (12, 18), (18, 26)], 'F150W': [(8, 14), (14, 22), (22, 34), (34, 50)]}
CIR = {'F250M': [3, 5, 8], 'F150W': [4, 6, 10]}
PIX = {'F250M': 0.063, 'F150W': 0.031}
rng = np.random.default_rng(1)
BINS = np.arange(12.0, 21.01, 0.5)


def boot_med(v, n=300):
    v = v[np.isfinite(v)]
    if v.size < 3:
        return np.nan, np.nan
    med = np.median(v)
    bs = np.median(v[rng.integers(0, v.size, (n, v.size))], axis=1)
    return med, bs.std()


def sel(t, kind, pre, k, iso=True):
    ok = (t['kind'] == kind) & (t[f'{pre}{k}_bad'] == 0) & np.isfinite(t[f'{pre}{k}_sum']) & np.isfinite(t['mag_dp'])
    if iso:
        ok &= t[f'{pre}{k}_iso'] == 1
    return ok


def binned(mag, R, bins=BINS, nmin=4):
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        k = (mag >= lo) & (mag < hi) & np.isfinite(R)
        if k.sum() >= nmin:
            m, e = boot_med(R[k])
            out.append((0.5 * (lo + hi), m, e, int(k.sum())))
    return np.array(out).reshape(-1, 4)


def rng_med(mag, R, lo, hi):
    k = (mag >= lo) & (mag < hi) & np.isfinite(R)
    if k.sum() < 4:
        return np.nan, np.nan, int(k.sum())
    m, e = boot_med(R[k])
    return m, e, int(k.sum())


def fmt(m, e, n):
    return 'n/a' if not np.isfinite(m) else f'{m:.3f}+-{e:.3f} ({n})'


def main():
    md = ['# Aperture closure (apclose)', '',
          'R = annulus_sum(data - bg) / flux_fit, normalised per annulus by the median R of unsaturated daophot stars in the reference window (so the unsaturated plateau is 1.0). '
          'Satstar R uses flux_fit_precap ("precap") and flux_fit ("final"). Cells show median +- bootstrap error (N). Magnitudes are dolphot mags (match within 0.1 arcsec). '
          'Plot shows the two innermost annuli only; tables cover all. Stars with NaN, DQ SATURATED or DO_NOT_USE pixels in the annulus, or with a neighbour (flux > 5 % of the target) within max(isolation radius, r_out + 2 px), are excluded.', '']
    md += ['Method notes:',
           '- Image = SCI of the arm tree crf file (SCI equals satstar residual + model wherever SCI is finite; max abs differences are a few tens to ~2000 MJy/sr in a handful of pixels, 107 or less in main2kf).',
           '- Saturated cores hold truncated DQ SATURATED values (SCI ~ 200 where the model is ~1e5) in both arms and are NaN in part of the core, so a circular aperture sum over the star is invalid for every satstar (0 satstars have a clean circle). Closure is therefore tested on annuli outside the saturated region, for satstars and unsaturated stars alike.',
           '- Satstar data = SCI; unsaturated daophot data = SCI minus satstar model. Background = sigma-clipped median of the satstar residual image in a 1.8-2.4 arcsec annulus (the spec annulus 2.5-3.5 arcsec was tested first; a closer one gave a flatter unsaturated plateau). Annulus sums of faint stars (> 18 mag LW) are limited by background systematics (R goes to 0 or negative), so the unsaturated reference window is 0.5 mag brighter than the satstar faint edge up to +1 mag, and the table shows the specified window value too.',
           '- Model-annulus column = sum of the satstar model image in the same annulus / flux_fit (same normalisation); data/model = annulus data sum / annulus model sum (independent of flux_fit and of the normalisation).',
           '']
    fig, axes = plt.subplots(len(BANDS), len(ARMS), figsize=(13, 9), sharey=True)
    summ = []
    for ib, band in enumerate(BANDS):
        for ia, arm in enumerate(ARMS):
            f = f'{AP}/data/stars_{arm}_{band}.fits'
            if not os.path.exists(f):
                continue
            t = Table.read(f)
            zlo, zhi = t.meta['ZPLO'], t.meta['ZPHI']  # p95(sat dolphot mag)+1 .. +3
            p95 = zlo - 1
            ref_primary = (p95 - 0.5, p95 + 1.0)
            ref_spec = (zlo, zhi)
            ax = axes[ib, ia]
            s = t[t['kind'] == 'sat']
            md += [f'## {band} {arm}', '',
                   f'satstar saturation faint edge (95th pct of dolphot mag of replaced satstars) = {p95:.2f}. '
                   f'Primary reference window for unsaturated stars: {ref_primary[0]:.2f}-{ref_primary[1]:.2f}; specified window (onset+1..+3): {ref_spec[0]:.2f}-{ref_spec[1]:.2f}.',
                   f'Satstar rows (all frames, pooled): {len(s)}; with dolphot match: {int(np.isfinite(s["mag_dp"]).sum())}.']
            # NaN / saturation census in the circular core
            kc = len(CIR[band]) - 1
            md.append(f'Satstars with any NaN in the largest circle (r={CIR[band][kc]} px): {int((s[f"ci{kc}_nnan"] > 0).sum())}; '
                      f'with any NaN/SAT/DNU pixel in it: {int((s[f"ci{kc}_bad"] > 0).sum())}; '
                      f'median equivalent saturated radius: {np.nanmedian(s["rsat_eq"]):.1f} px.')
            md.append('')
            colors = plt.cm.viridis(np.linspace(0, 0.9, len(ANN[band])))
            for k, (r1, r2) in enumerate(ANN[band]):
                d = t[sel(t, 'dao', 'an', k)]
                sp = t[sel(t, 'sat', 'an', k)]
                Rd = d['an%d_sum' % k] / d['flux']
                norm, nerr, nn = rng_med(d['mag_dp'], Rd, *ref_primary)
                norm_spec, _, nn_spec = rng_med(d['mag_dp'], Rd, *ref_spec)
                if not np.isfinite(norm) or norm <= 0:
                    md += [f'### annulus {r1}-{r2} px: no valid unsaturated reference (N={nn})', '']
                    continue
                Rp = sp['an%d_sum' % k] / sp['flux_fit_precap'] / norm
                Rf = sp['an%d_sum' % k] / sp['flux'] / norm
                Rm = sp['an%d_mod' % k] / sp['flux'] / norm
                Rd = Rd / norm
                Rdm = sp['an%d_sum' % k] / sp['an%d_mod' % k]
                bd, bp, bf, bm = (binned(d['mag_dp'], Rd), binned(sp['mag_dp'], Rp),
                                  binned(sp['mag_dp'], Rf), binned(sp['mag_dp'], Rm))
                md += [f'### {band} {arm} annulus {r1}-{r2} px ({r1 * PIX[band]:.2f}-{r2 * PIX[band]:.2f} arcsec)',
                       f'normalisation: median unsaturated R = {norm:.4f} +- {nerr:.4f} (N={nn}); specified-window value {norm_spec:.4f} (N={nn_spec})', '',
                       '| mag bin | unsat daophot R | satstar R precap | satstar R final | satstar model-annulus / flux_fit (norm.) | satstar data/model annulus (raw) |', '|---|---|---|---|---|---|']
                for lo in np.arange(12.0, 21.0, 1.0):
                    hi = lo + 1.0
                    cells = [fmt(*rng_med(d['mag_dp'], Rd, lo, hi)), fmt(*rng_med(sp['mag_dp'], Rp, lo, hi)),
                             fmt(*rng_med(sp['mag_dp'], Rf, lo, hi)), fmt(*rng_med(sp['mag_dp'], Rm, lo, hi)), fmt(*rng_med(sp['mag_dp'], Rdm, lo, hi))]
                    if all(c == 'n/a' for c in cells):
                        continue
                    md.append(f'| {lo:.0f}-{hi:.0f} | ' + ' | '.join(cells) + ' |')
                md.append('')
                # 13-15 vs 16-18 summary
                for lab, (lo, hi) in (('13-15', (13, 15)), ('16-18', (16, 18))):
                    summ.append((band, arm, f'{r1}-{r2}', lab, rng_med(sp['mag_dp'], Rp, lo, hi), rng_med(sp['mag_dp'], Rf, lo, hi),
                                 rng_med(d['mag_dp'], Rd, lo, hi), rng_med(sp['mag_dp'], Rdm, lo, hi)))
                # R vs wingcal_rmask / cap_psf_frac
                if k in (0, 1):
                    for col, edges in (('wingcal_rmask', [0, 0.5642, 2, 5, 1e9]), ('cap_psf_frac', [-1, 0.7, 0.9999, 1e9])):
                        v = np.asarray(sp[col], float)
                        md += [f'R vs {col} (satstars, this annulus):', '',
                               f'| {col} range | N | median R precap | median R final | median data/model | median dolphot mag |', '|---|---|---|---|---|---|']
                        for a, b in zip(edges[:-1], edges[1:]):
                            kk = np.isfinite(v) & (v >= a) & (v < b)
                            if kk.sum() >= 4:
                                md.append(f'| {a:.4g} to {b:.4g} | {int(kk.sum())} | {np.nanmedian(Rp[kk]):.3f} | {np.nanmedian(Rf[kk]):.3f} | {np.nanmedian(sp["an%d_sum" % k][kk] / sp["an%d_mod" % k][kk]):.3f} | {np.nanmedian(sp["mag_dp"][kk]):.2f} |')
                        md.append('')
                # plot: use annuli 0..: draw one panel line per annulus
                if k > 1:
                    continue
                c = colors[k]
                lab = f'{r1}-{r2}px'
                if len(bd):
                    ax.errorbar(bd[:, 0], bd[:, 1], bd[:, 2], color=c, ls='-', marker='o', ms=3, lw=1, label=f'unsat daophot {lab}' if k == 0 else None, alpha=0.9)
                if len(bp):
                    ax.errorbar(bp[:, 0], bp[:, 1], bp[:, 2], color=c, ls='--', marker='s', ms=4, lw=1.2, label=f'satstar precap {lab}' if k == 0 else None)
                if len(bf):
                    ax.errorbar(bf[:, 0], bf[:, 1], bf[:, 2], color=c, ls=':', marker='^', ms=4, lw=1.2, label=f'satstar final {lab}' if k == 0 else None)
            # sanity: unsaturated ratio flat over 3+ mag (annulus 0 and circles)
            md += [f'### Sanity check, unsaturated daophot stars, {band} {arm}: aperture_sum / flux_fit vs dolphot mag (median, N), raw units (not normalised)', '',
                   '| aperture | ' + ' | '.join(f'{lo:.0f}-{lo + 1:.0f}' for lo in np.arange(14, 20, 1.0)) + ' |', '|---|' + '---|' * 6]
            for pre, lst in (('an', ANN[band]), ('ci', CIR[band])):
                for k, a in enumerate(lst):
                    d = t[sel(t, 'dao', pre, k)]
                    R = d[f'{pre}{k}_sum'] / d['flux']
                    cells = [fmt(*rng_med(d['mag_dp'], R, lo, lo + 1.0)) for lo in np.arange(14, 20, 1.0)]
                    nm = f'annulus {a[0]}-{a[1]} px' if pre == 'an' else f'circle r={a} px'
                    md.append(f'| {nm} | ' + ' | '.join(cells) + ' |')
            md.append('')
            # circular apertures for satstars: how many are usable
            md.append(f'Satstar circular apertures with no NaN/SAT/DNU pixel and isolated: ' +
                      ', '.join(f'r={CIR[band][k]} px: {int(sel(t, "sat", "ci", k).sum())}' for k in range(len(CIR[band]))) + '.')
            md.append('')
            ax.axhline(1, color='gray', lw=0.5)
            ax.set_title(f'{band} {arm}')
            ax.set_xlabel('dolphot mag')
            ax.set_ylim(0.2, 2.5)
            ax.axvspan(ref_primary[0], ref_primary[1], color='gray', alpha=0.1)
            if ia == 0:
                ax.set_ylabel('R / R(unsat plateau)')
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, loc='upper left')
    # annulus colour key in the second panel
    for ib, band in enumerate(BANDS):
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(ANN[band])))
        for k, (r1, r2) in enumerate(ANN[band][:2]):
            axes[ib, 1].plot([], [], color=colors[k], label=f'annulus {r1}-{r2} px')
        axes[ib, 1].legend(fontsize=7, loc='upper left')
    fig.suptitle('Aperture closure: annulus sum / flux_fit, normalised to unsaturated daophot plateau (line styles: solid=unsat, dashed=satstar precap, dotted=satstar final)', fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{AP}/apclose.png', dpi=110)
    md += ['## Summary: normalised R of satstars at 13-15 and 16-18 mag', '',
           '| band | arm | annulus (px) | mag | satstar precap | satstar final | unsat daophot | satstar data/model |', '|---|---|---|---|---|---|---|---|']
    for b, a, an_, lab, p, f_, d_, dm_ in summ:
        md.append(f'| {b} | {a} | {an_} | {lab} | {fmt(*p)} | {fmt(*f_)} | {fmt(*d_)} | {fmt(*dm_)} |')
    md += ['', '## Bright/faint ratios of satstar R (13-15 mag over 16-18 mag; independent of the unsaturated normalisation)', '',
           '| band | arm | annulus (px) | precap | final | data/model |', '|---|---|---|---|---|---|']
    for i in range(0, len(summ) - 1, 2):
        b, a, an_, _, p1, f1, _, m1 = summ[i]
        _, _, an2, _, p2, f2, _, m2 = summ[i + 1]
        if an_ != an2 or not np.isfinite(p1[0]) or not np.isfinite(p2[0]):
            continue
        def rr(x, y):
            r = x[0] / y[0]
            return f'{r:.3f}+-{r * np.hypot(x[1] / x[0], y[1] / y[0]):.3f}'
        md.append(f'| {b} | {a} | {an_} | {rr(p1, p2)} | {rr(f1, f2)} | {rr(m1, m2)} |')
    with open(f'{AP}/apclose.md', 'w') as fh:
        fh.write('\n'.join(md) + '\n')


if __name__ == '__main__':
    main()
