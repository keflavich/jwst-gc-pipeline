"""Figures and a numeric summary for one detector's result (analyze_detector.py output).

    python figures.py --result result_nrcblong.npz --stars result_nrcblong_stars.npz \
        --label "NRCBLONG F480M" --prefix outdir/epsf --first 1

Writes <prefix><n>.png, n = first .. first+5, and <prefix>_summary.json.
Colours: data = blue, STPSF = orange (fixed across all figures); images use
viridis (log intensity) and RdBu_r (signed differences, zero = white).
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import epsf  # noqa: E402
from analyze_detector import bin_map, poly_smooth  # noqa: E402

CD, CS = '#1f5fbf', '#e07b00'   # data, STPSF
DPI = 90


def to_det(P, O):
    """Oversampled ePSF -> the detector-sampled image of a star centred on a pixel."""
    c = P.shape[0] // 2
    return P[c % O::O, c % O::O]


def interp_stpsf(Sc, spos, x, y):
    """Bilinear interpolation of the STPSF grid (regular ns x ns) at (x, y)."""
    ns = int(round(np.sqrt(len(spos))))
    c = np.unique(spos[:, 0])
    fx = np.interp(x, c, np.arange(ns)); fy = np.interp(y, c, np.arange(ns))
    i0 = min(int(fx), ns - 2); j0 = min(int(fy), ns - 2)
    tx, ty = fx - i0, fy - j0
    S = Sc.reshape(ns, ns, *Sc.shape[1:])
    return ((1 - tx) * (1 - ty) * S[j0, i0] + tx * (1 - ty) * S[j0, i0 + 1]
            + (1 - tx) * ty * S[j0 + 1, i0] + tx * ty * S[j0 + 1, i0 + 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result', required=True)
    ap.add_argument('--stars', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--first', type=int, default=1)
    a = ap.parse_args()
    r = np.load(a.result, allow_pickle=True)
    t = np.load(a.stars, allow_pickle=True)
    O, R, Rw = int(r['O']), int(r['R']), int(r['Rw'])
    g, gw = epsf.Grid(R, O), epsf.Grid(Rw, O)
    spos = r['spos']; ic = int(np.argmin(np.hypot(spos[:, 0] - 1024, spos[:, 1] - 1024)))
    Pw, Pwa, Pwb, Sw = r['Pw'], r['Pw_a'], r['Pw_b'], r['Sw']
    Swc = Sw[ic]
    P1, Sc = r['P1'], r['Sc']
    summ = dict(label=a.label, n_core_stars=int(len(t['x'])), n_frames=int(len(r['frames'])),
                n_wing_stars=int(len(r['w_x'])))
    n = a.first
    out = []

    # ------------------------------------------------ fig 1: ePSF, STPSF, difference
    fig, ax = plt.subplots(2, 4, figsize=(17, 8.6))
    pk = Pw.max()
    vmin = 1e-5
    ims = [(Pw / pk, 'data wing ePSF (global)'), (Swc / Swc.max(), 'STPSF ePSF (det. centre)')]
    for k, (im, ti) in enumerate(ims):
        h = ax[0, k].imshow(np.clip(im, vmin, None), norm=LogNorm(vmin, 1), origin='lower', cmap='viridis',
                            extent=[-Rw, Rw, -Rw, Rw])
        ax[0, k].set_title(ti + ' / peak')
    plt.colorbar(h, ax=ax[0, :2], shrink=0.8)
    diff = (Pw - Swc) / pk
    noise = (Pwa - Pwb) / 2 / pk
    for k, (im, ti) in enumerate([(diff, '(data - STPSF) / peak'), (noise, 'split-half (A - B)/2 / peak  [noise]')]):
        h = ax[0, 2 + k].imshow(im, vmin=-3e-3, vmax=3e-3, cmap='RdBu_r', origin='lower', extent=[-Rw, Rw, -Rw, Rw])
        ax[0, 2 + k].set_title(ti)
    plt.colorbar(h, ax=ax[0, 2:], shrink=0.8)
    # core, zoom, with sharper scales
    Scc = Sc[ic]
    ext = [-R, R, -R, R]
    h = ax[1, 0].imshow(np.clip(P1 / P1.max(), 1e-4, None), norm=LogNorm(1e-4, 1), origin='lower', cmap='viridis', extent=ext)
    ax[1, 0].set_title('data core ePSF (global) / peak')
    ax[1, 1].imshow(np.clip(Scc / Scc.max(), 1e-4, None), norm=LogNorm(1e-4, 1), origin='lower', cmap='viridis', extent=ext)
    ax[1, 1].set_title('STPSF core ePSF / peak')
    plt.colorbar(h, ax=ax[1, :2], shrink=0.8)
    dc = (P1 - Scc) / P1.max()
    h = ax[1, 2].imshow(dc, vmin=-0.03, vmax=0.03, cmap='RdBu_r', origin='lower', extent=ext)
    ax[1, 2].set_title(f'(data - STPSF)/peak, core; max |d|={np.abs(dc).max():.3f}')
    rel = np.where(Scc > 1e-3 * Scc.max(), P1 / Scc - 1, np.nan)
    h2 = ax[1, 3].imshow(rel, vmin=-0.3, vmax=0.3, cmap='RdBu_r', origin='lower', extent=ext)
    ax[1, 3].set_title('data / STPSF - 1 (where STPSF > 1e-3 peak)')
    plt.colorbar(h, ax=ax[1, 2], shrink=0.8); plt.colorbar(h2, ax=ax[1, 3], shrink=0.8)
    for x in ax.ravel():
        x.set_xlabel('dx [px]'); x.set_ylabel('dy [px]')
    fig.suptitle(f'{a.label}: empirical ePSF ({O}x oversampled, pixel-integrated) vs STPSF '
                 f'({summ["n_wing_stars"]} wing stars, {summ["n_core_stars"]} core stars, {summ["n_frames"]} frames)')
    fn = f'{a.prefix}{n}.png'; fig.savefig(fn, dpi=DPI, bbox_inches='tight'); plt.close(fig); out.append(fn); n += 1
    summ['core_peak_ratio_data_over_stpsf'] = float(P1.max() / Scc.max())
    summ['core_max_absdiff_over_peak'] = float(np.abs(dc).max())
    summ['core_rms_diff_over_peak_r<3'] = float(np.sqrt(np.mean(dc[g.ur <= 3] ** 2)))

    # ------------------------------------------------ fig 2: profiles, EE
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))
    rb = np.concatenate([np.arange(0, 3, 0.25), np.arange(3, 10, 0.5), np.arange(10, Rw + 0.1, 1.0)])
    rc = 0.5 * (rb[1:] + rb[:-1])
    pd = epsf.radial_profile(Pw, gw, rb); ps = epsf.radial_profile(Swc, gw, rb)
    pa = epsf.radial_profile(Pwa, gw, rb); pb_ = epsf.radial_profile(Pwb, gw, rb)
    ax[0].semilogy(rc, np.abs(pd), color=CD, lw=2, label='data (global wing ePSF)')
    ax[0].semilogy(rc, ps, color=CS, lw=2, label='STPSF')
    ax[0].set_xlabel('r [px]'); ax[0].set_ylabel('azimuthal mean ePSF [flux(r<10) / px]'); ax[0].legend()
    ax[0].set_title('radial profile')
    ratio = pd / ps; err = np.abs(pa - pb_) / 2 / ps
    ax[1].plot(rc, ratio, color=CD, lw=2)
    ax[1].fill_between(rc, ratio - err, ratio + err, color=CD, alpha=0.25, label='split-half noise')
    if 'Pw_bright' in r.files:
        pbr = epsf.radial_profile(r['Pw_bright'], gw, rb); pfa = epsf.radial_profile(r['Pw_faint'], gw, rb)
        ax[1].plot(rc, pbr / ps, color='#2a9d8f', lw=1.2, label='brightest third')
        ax[1].plot(rc, pfa / ps, color='#8e44ad', lw=1.2, label='faintest third')
        ax[0].semilogy(rc, np.abs(pbr), color='#2a9d8f', lw=1, label='data, brightest third')
        ax[0].semilogy(rc, np.abs(pfa), color='#8e44ad', lw=1, label='data, faintest third')
        ax[0].legend(fontsize=8)
        for r0, r1 in ((8, 12), (12, 20), (20, Rw)):
            sel_ = (rc > r0) & (rc < r1)
            summ[f'bright_over_faint_{r0}_{r1}'] = float(np.mean(pbr[sel_]) / np.mean(pfa[sel_]))
    ax[1].axhline(1, color='gray', lw=1)
    ax[1].set_ylim(0.0, 1.6); ax[1].set_xlabel('r [px]'); ax[1].set_ylabel('data / STPSF'); ax[1].legend()
    ax[1].set_title('profile ratio')
    radii = np.linspace(0.25, Rw, 120)
    eed = epsf.encircled(Pw, gw, radii); ees = epsf.encircled(Swc, gw, radii)
    ax[2].plot(radii, eed, color=CD, lw=2, label='data'); ax[2].plot(radii, ees, color=CS, lw=2, label='STPSF')
    ax[2].axvline(10, color='gray', ls=':', lw=1)
    ax[2].set_xlabel('r [px]'); ax[2].set_ylabel('EE (normalised to r=10 px)'); ax[2].legend()
    ax[2].set_title('encircled energy')
    eea = epsf.encircled(Pwa, gw, radii); eeb = epsf.encircled(Pwb, gw, radii)
    ax[3].plot(radii, eed - ees, color=CD, lw=2)
    ax[3].fill_between(radii, eed - ees - np.abs(eea - eeb) / 2, eed - ees + np.abs(eea - eeb) / 2, color=CD, alpha=0.25)
    ax[3].axhline(0, color='gray', lw=1)
    ax[3].set_xlabel('r [px]'); ax[3].set_ylabel('EE(data) - EE(STPSF)'); ax[3].set_title('EE difference')
    fig.suptitle(f'{a.label}: radial profile and encircled energy, data vs STPSF (both normalised to the flux within r = 10 px)')
    fn = f'{a.prefix}{n}.png'; fig.savefig(fn, dpi=DPI, bbox_inches='tight'); plt.close(fig); out.append(fn); n += 1
    for rr in (1, 2, 3, 5):
        summ[f'EE{rr}_data'] = float(np.interp(rr, radii, eed)); summ[f'EE{rr}_stpsf'] = float(np.interp(rr, radii, ees))
    summ[f'EE{Rw}_data'] = float(eed[-1]); summ[f'EE{Rw}_stpsf'] = float(ees[-1])
    for r0, r1 in ((3, 6), (6, 10), (10, 20), (20, Rw)):
        s = (rc > r0) & (rc < r1)
        summ[f'profile_ratio_{r0}_{r1}'] = float(np.mean(ratio[s]))

    # ------------------------------------------------ fig 3: 4x4 grid ePSFs, vs global and vs STPSF
    P4 = r['P4']; G = int(round(np.sqrt(len(P4))))
    cc = (np.arange(G) + 0.5) * 2048 / G
    fig, ax = plt.subplots(G, 3 * G + 2, figsize=(3 * G * 1.35 + 2, G * 1.45),
                           gridspec_kw=dict(width_ratios=[1] * G + [0.25] + [1] * G + [0.25] + [1] * G))
    for x in ax.ravel():
        x.set_xticks([]); x.set_yticks([])
    for x in ax[:, G]:
        x.axis('off')
    for x in ax[:, 2 * G + 1]:
        x.axis('off')
    zoom = 6 * O
    c0 = P1.shape[0] // 2
    sl = slice(c0 - zoom, c0 + zoom + 1)
    dmax = []
    for j in range(G):
        for i in range(G):
            P = P4[j * G + i]
            S = interp_stpsf(Sc, spos, cc[i], cc[j])
            row = G - 1 - j
            ax[row, i].imshow(np.clip(P[sl, sl] / P.max(), 1e-3, None), norm=LogNorm(1e-3, 1), cmap='viridis', origin='lower')
            d1 = (P - P1) / P1.max()
            ax[row, G + 1 + i].imshow(d1[sl, sl], vmin=-0.01, vmax=0.01, cmap='RdBu_r', origin='lower')
            d2 = (P - S) / P.max()
            ax[row, 2 * G + 2 + i].imshow(d2[sl, sl], vmin=-0.03, vmax=0.03, cmap='RdBu_r', origin='lower')
            dmax.append((np.abs(d1).max(), np.abs(d2).max()))
    ax[0, 0].set_title('cell ePSF (log)', loc='left', fontsize=9)
    ax[0, G + 1].set_title('cell - global, +/-1% of peak', loc='left', fontsize=9)
    ax[0, 2 * G + 2].set_title('cell - STPSF(cell), +/-3% of peak', loc='left', fontsize=9)
    fig.suptitle(f'{a.label}: {G}x{G} spatial ePSF grid (detector x right, y up; +/-6 px shown)', y=1.02)
    fn = f'{a.prefix}{n}.png'; fig.savefig(fn, dpi=DPI, bbox_inches='tight'); plt.close(fig); out.append(fn); n += 1
    dmax = np.array(dmax)
    summ['grid4_max_cell_minus_global'] = float(dmax[:, 0].max())
    summ['grid4_median_max_cell_minus_stpsf'] = float(np.median(dmax[:, 1]))
    # split-half noise of the best grid
    PGa, PGb = r['PGa'], r['PGb']
    summ['gridbest_splithalf_maxdiff'] = float(np.median(np.abs(PGa - PGb).max(axis=(1, 2)) / 2 / P1.max()))

    # ------------------------------------------------ fig 4: cross-validation, accuracy vs radius
    grids = r['grids']; rbins = r['rbins']; rcen = 0.5 * (rbins[1:] + rbins[:-1])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for bil, ls, lab in ((0, '-o', 'piecewise-constant cells'), (1, '--s', 'bilinear between cell centres')):
        xs, ys = [], []
        for G_ in grids:
            k = f'cv_{G_}_{bil}_core_metric'
            if k in r.files:
                xs.append(G_); ys.append(float(r[k]))
            elif bil == 1 and G_ == 1:
                xs.append(1); ys.append(float(r['cv_1_0_core_metric']))
        ax[0].plot(xs, ys, ls, color=CD if bil else '#6a6a6a', label=lab)
    ax[0].set_xlabel('grid G (G x G cells over the detector)'); ax[0].set_ylabel('held-out <(res/ERR)^2>, r <= 3 px')
    ax[0].set_title(f'cross-validation (20% of frames held out); best G={int(r["Gbest"])}'); ax[0].legend()
    Gb = int(r['Gbest']); bb = int(r['bilinear_best'])
    for G_, bil, col, lab in ((1, 0, '#6a6a6a', 'G=1 (constant)'), (Gb, bb, CD, f'best G={Gb}')):
        ax[1].semilogy(rcen, r[f'cv_{G_}_{bil}_rms'], 'o-', color=col, label=f'residual rms (robust), {lab}')
    ax[1].semilogy(rcen, r[f'cv_{Gb}_{bb}_phot'], 'k:', label='photon+read noise (ERR)')
    prof_c = np.array([P1[(g.ur >= a0) & (g.ur < a1)].mean() for a0, a1 in zip(rbins[:-1], rbins[1:])])
    ax[1].semilogy(rcen, 0.01 * prof_c, color='#c0392b', ls='-.', label='1% of the ePSF (azimuthal mean)')
    ax[1].semilogy(rcen, r[f'cv_{Gb}_{bb}_noise_tot'], 'k--', label='ERR + confusion (outer-stamp rms)')
    ax[1].set_xlabel('r [px]'); ax[1].set_ylabel('per-pixel rms [flux(r<10) units]'); ax[1].legend(fontsize=8)
    ax[1].set_title('held-out residual rms vs radius')
    for G_, bil, col, lab in ((1, 0, '#6a6a6a', 'G=1'), (Gb, bb, CD, f'G={Gb}')):
        ax[2].plot(rcen, r[f'cv_{G_}_{bil}_chi2_err'], 'o-', color=col, label=lab)
    ax[2].axhline(1, color='gray', lw=1)
    ax[2].set_yscale('log'); ax[2].set_xlabel('r [px]'); ax[2].set_ylabel('<(resid/ERR)^2>')
    ax[2].set_title('held-out chi^2 vs photon noise alone'); ax[2].legend()
    fig.suptitle(f'{a.label}: ePSF accuracy on held-out stars')
    fn = f'{a.prefix}{n}.png'; fig.savefig(fn, dpi=DPI, bbox_inches='tight'); plt.close(fig); out.append(fn); n += 1
    summ['cv'] = {f'G{G_}_{b}': float(r[f'cv_{G_}_{b}_core_metric']) for G_ in grids for b in (0, 1)
                  if f'cv_{G_}_{b}_core_metric' in r.files}
    summ['Gbest'] = Gb; summ['bilinear_best'] = bb
    summ['heldout_rms_best'] = dict(zip([f'{x:.2f}' for x in rcen], np.round(r[f'cv_{Gb}_{bb}_rms'], 6).tolist()))
    summ['heldout_phot'] = dict(zip([f'{x:.2f}' for x in rcen], np.round(r[f'cv_{Gb}_{bb}_phot'], 6).tolist()))
    summ['heldout_chi2err_best'] = dict(zip([f'{x:.2f}' for x in rcen], np.round(r[f'cv_{Gb}_{bb}_chi2_err'], 3).tolist()))

    # ------------------------------------------------ fig 5: maps
    ok = t['ok'] & np.isfinite(t['dsize_global']) & (t['chi2'] < np.nanpercentile(t['chi2'], 95))
    x, y = t['x'][ok], t['y'][ok]
    ev = (t['frame'][ok] % 2) == 0
    fw1 = float(r['met1'][()]['fwhm'])
    quant = [('dsize_global', 100 * fw1 / 100, 'FWHM change [%] (size mode)', 100.0, 3),
             ('de1_global', 1, 'delta e1', 1.0, 0.02), ('de2_global', 1, 'delta e2', 1.0, 0.02),
             ('dwing_global', 1, 'wing (3-10 px) excess [%]', 100.0, 10)]
    NB = 16
    shs = r['sh_stpsf']
    fig, ax = plt.subplots(4, 4, figsize=(19, 17))
    plt.rcParams.update({'axes.titlesize': 9})
    maps_summary = {}
    for q, (key, _, lab, scale, vr) in enumerate(quant):
        v = t[key][ok] * scale
        M, N = bin_map(x, y, v, NB)
        Ma, _ = bin_map(x[ev], y[ev], v[ev], NB); Mb, _ = bin_map(x[~ev], y[~ev], v[~ev], NB)
        noise = (Ma - Mb) / 2
        hp = M - poly_smooth(M, 2)
        hpa = Ma - poly_smooth(Ma, 2); hpb = Mb - poly_smooth(Mb, 2)
        okm = np.isfinite(hpa) & np.isfinite(hpb)
        rho = float(np.corrcoef(hpa[okm], hpb[okm])[0, 1]) if okm.sum() > 5 else np.nan
        med = np.nanmedian(M)
        ext = [0, 2048, 0, 2048]
        h = ax[q, 0].imshow(M - med, origin='lower', cmap='RdBu_r', vmin=-vr, vmax=vr, extent=ext)
        ax[q, 0].set_title(f'{lab} - median ({med:.3g})\n128-px cells, {int(np.median(N))} stars/cell')
        ax[q, 1].imshow(noise, origin='lower', cmap='RdBu_r', vmin=-vr, vmax=vr, extent=ext)
        ax[q, 1].set_title(f'split-half noise (A-B)/2\nrms {np.nanstd(noise):.3g}')
        ax[q, 2].imshow(hp, origin='lower', cmap='RdBu_r', vmin=-vr, vmax=vr, extent=ext)
        ax[q, 2].set_title(f'map - quadratic surface: rms {np.nanstd(hp):.3g}\nhalf-vs-half correlation {rho:.2f}')
        # STPSF prediction (5x5), same estimator and scale
        ns = int(round(np.sqrt(len(spos))))
        sv = (shs[:, q] * scale).reshape(ns, ns)
        ax[q, 3].imshow(sv - np.nanmedian(sv), origin='lower', cmap='RdBu_r', vmin=-vr, vmax=vr, extent=ext)
        ax[q, 3].set_title(f'STPSF prediction (5x5 grid)\nrange {np.nanmax(sv) - np.nanmin(sv):.3g}')
        plt.colorbar(h, ax=ax[q, :], shrink=0.8)
        maps_summary[key] = dict(map_rms=float(np.nanstd(M)), noise_rms=float(np.nanstd(noise)),
                                 highpass_rms=float(np.nanstd(hp)), highpass_splithalf_corr=rho,
                                 map_range_p5_p95=float(np.nanpercentile(M, 95) - np.nanpercentile(M, 5)),
                                 stpsf_range=float(np.nanmax(sv) - np.nanmin(sv)),
                                 smooth_range=float(np.nanmax(poly_smooth(M, 2)) - np.nanmin(poly_smooth(M, 2))),
                                 median_stars_per_cell=float(np.median(N)))
    for xx_ in ax.ravel():
        xx_.set_xlabel('x [px]'); xx_.set_ylabel('y [px]')
    fig.suptitle(f'{a.label}: PSF shape across the detector from {ok.sum()} star-exposures, '
                 'each fitted with the global ePSF + size/e1/e2 modes (FWHM change: size mode x 100%)', y=1.0)
    fn = f'{a.prefix}{n}.png'; fig.savefig(fn, dpi=DPI, bbox_inches='tight'); plt.close(fig); out.append(fn); n += 1
    summ['maps'] = maps_summary

    # ------------------------------------------------ fig 6: per-exposure (issue #993)
    fr = t['frame'][ok]
    nfr = int(fr.max()) + 1
    exp_t = np.array([np.nanmedian(t['expstart'][ok][fr == k]) if np.any(fr == k) else np.nan for k in range(nfr)])
    ds = [(np.nanmedian(t['dsize_global'][ok][fr == k]) * 100,
           1.2533 * np.nanstd(t['dsize_global'][ok][fr == k]) * 100 / np.sqrt(max((fr == k).sum(), 1)))
          if (fr == k).sum() > 10 else (np.nan, np.nan) for k in range(nfr)]
    ds = np.array(ds)
    wf, wnum, wden = r['w_frame'], r['wnum'], r['wden']
    ann = r['ann_rs']
    fig, ax = plt.subplots(3, 1, figsize=(15, 12), gridspec_kw=dict(hspace=0.35))
    order = np.argsort(exp_t)
    xo = np.arange(nfr)
    ax[0].errorbar(xo, ds[order, 0], ds[order, 1], fmt='o', ms=3, color=CD)
    ax[0].set_ylabel('median FWHM change [%]\n(core stars, vs global ePSF)')
    ax[0].axhline(0, color='gray', lw=1)
    ax[0].set_title(f'{a.label}: per-exposure PSF (frames in time order; 6 dithers per observation)')
    cols = ['#6a6a6a', CD, '#2a9d8f', '#c0392b']
    per_frame_w = []
    rng = np.random.default_rng(0)
    for kk in range(wnum.shape[1]):
        m = np.full(nfr, np.nan); e = np.full(nfr, np.nan)
        for k in range(nfr):
            sel = np.nonzero((wf == k) & np.isfinite(wnum[:, kk]))[0]
            if len(sel) < 8:
                continue
            ratio = wnum[sel, kk] / np.maximum(wden[sel, kk], 1e-12)
            med = np.median(ratio); mad = 1.4826 * np.median(np.abs(ratio - med))
            sel = sel[np.abs(ratio - med) < 5 * mad]
            m[k] = wnum[sel, kk].sum() / wden[sel, kk].sum()
            bs = [wnum[b_, kk].sum() / wden[b_, kk].sum() for b_ in (rng.choice(sel, len(sel)) for _ in range(100))]
            e[k] = np.std(bs)
        per_frame_w.append((m, e))
        if ann[kk][0] >= 15:      # beyond ~15 px the unsaturated-star wing is confusion-limited
            continue
        ax[1].errorbar(xo + 0.15 * kk, 100 * (m[order] - np.nanmedian(m)), 100 * e[order], fmt='o', ms=3,
                       color=cols[kk], label=f'{ann[kk][0]}-{ann[kk][1]} px')
    ax[1].axhline(0, color='gray', lw=1)
    ax[1].set_ylim(-30, 30)
    ax[1].set_ylabel('wing excess [%] per frame, minus median\n(stack of wing stars vs global wing ePSF)')
    ax[1].set_xlabel('frame (time order)'); ax[1].legend(ncol=4)
    # brighter-fatter / nonlinearity: size vs peak level
    sf = t['satfrac'][ok]; dsz = t['dsize_global'][ok] * 100
    bins = np.linspace(np.nanpercentile(sf, 1), np.nanpercentile(sf, 99.5), 12)
    bc = 0.5 * (bins[1:] + bins[:-1])
    med_ = [np.nanmedian(dsz[(sf >= b0) & (sf < b1)]) for b0, b1 in zip(bins[:-1], bins[1:])]
    err_ = [1.2533 * np.nanstd(dsz[(sf >= b0) & (sf < b1)]) / np.sqrt(max(((sf >= b0) & (sf < b1)).sum(), 1))
            for b0, b1 in zip(bins[:-1], bins[1:])]
    ax[2].errorbar(bc, med_, err_, fmt='o-', color=CD)
    ax[2].axhline(0, color='gray', lw=1)
    ax[2].set_xlabel('peak-pixel signal at the last group / CRDS saturation level')
    ax[2].set_ylabel('median FWHM change [%]')
    ax[2].set_title('size vs peak level (brighter-fatter / non-linearity check)')
    fn = f'{a.prefix}{n}.png'; fig.savefig(fn, dpi=DPI, bbox_inches='tight'); plt.close(fig); out.append(fn); n += 1
    summ['size_vs_satfrac'] = dict(satfrac=bc.round(3).tolist(), dfwhm_pct=np.round(med_, 3).tolist(), err=np.round(err_, 3).tolist())
    good = np.isfinite(ds[:, 0])
    summ['per_frame_fwhm_change_rms_pct'] = float(np.nanstd(ds[good, 0]))
    summ['per_frame_fwhm_change_err_median_pct'] = float(np.nanmedian(ds[good, 1]))
    for kk in range(wnum.shape[1]):
        m, e = per_frame_w[kk]
        gg = np.isfinite(m)
        summ[f'per_frame_wing_{ann[kk][0]}_{ann[kk][1]}_rms_pct'] = float(100 * np.nanstd(m[gg]))
        summ[f'per_frame_wing_{ann[kk][0]}_{ann[kk][1]}_err_median_pct'] = float(100 * np.nanmedian(e[gg]))
    # core metrics table
    m1 = r['met1'][()]; st = r['st_metrics'][ic]
    summ['fwhm_data_px'] = float(m1['fwhm']); summ['fwhm_stpsf_px'] = float(st['fwhm'])
    summ['e_data'] = float(m1['e']); summ['e_stpsf'] = float(st['e'])
    summ['theta_data'] = float(m1['theta']); summ['theta_stpsf'] = float(st['theta'])
    summ['wing_core_data'] = float(m1['wing_core']); summ['wing_core_stpsf'] = float(st['wing_core'])
    mg = r['metG']
    summ['grid_fwhm_range'] = float(np.ptp([m['fwhm'] for m in mg]))
    summ['stpsf_fwhm_range'] = float(np.ptp([m['fwhm'] for m in r['st_metrics']]))
    summ['grid_e_range'] = float(np.ptp([m['e'] for m in mg]))
    summ['stpsf_e_range'] = float(np.ptp([m['e'] for m in r['st_metrics']]))
    summ['figures'] = out
    with open(a.prefix + '_summary.json', 'w') as f:
        json.dump(summ, f, indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float))


if __name__ == '__main__':
    main()
