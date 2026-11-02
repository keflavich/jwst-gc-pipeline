"""Combined figures for several detectors (e.g. the 8 SW detectors in F212N).

    python sw_summary.py --scratch $S --dets nrca1,...,nrcb4 --prefix out/sw

Reads result_<det>.npz / result_<det>_stars.npz (analyze_detector.py) and writes
<prefix>1..5.png plus <prefix>_summary.json.
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
from analyze_detector import bin_map  # noqa: E402

plt.rcParams.update({'axes.titlesize': 9.5, 'axes.labelsize': 9})
DPI = 85
# NIRCam SW focal-plane layout as seen on the sky-ish grid (A: 2 4 / 1 3 ; B: 3 1 / 4 2)
LAYOUT = {'nrca2': (0, 0), 'nrca4': (0, 1), 'nrca1': (1, 0), 'nrca3': (1, 1),
          'nrcb3': (0, 2), 'nrcb1': (0, 3), 'nrcb4': (1, 2), 'nrcb2': (1, 3)}
COL = ['#1f5fbf', '#e07b00', '#2a9d8f', '#c0392b', '#8e44ad', '#6a6a6a', '#b8860b', '#17becf']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scratch', required=True)
    ap.add_argument('--dets', required=True)
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--filter', default='F212N')
    a = ap.parse_args()
    dets = a.dets.split(',')
    R_ = {d: np.load(os.path.join(a.scratch, f'result_{d}.npz'), allow_pickle=True) for d in dets}
    T_ = {d: np.load(os.path.join(a.scratch, f'result_{d}_stars.npz'), allow_pickle=True) for d in dets}
    summ = {}
    out = []

    def pos(d):
        return LAYOUT.get(d, (dets.index(d) // 4, dets.index(d) % 4))

    # ---------- 1: core ePSF and (data - STPSF)/peak per detector
    fig, ax = plt.subplots(4, 4, figsize=(15, 15.5))
    for x in ax.ravel():
        x.axis('off')
    for d in dets:
        r = R_[d]; O, R = int(r['O']), int(r['R']); g = epsf.Grid(R, O)
        spos = r['spos']; ic = int(np.argmin(np.hypot(spos[:, 0] - 1024, spos[:, 1] - 1024)))
        P1, S = r['P1'], r['Sc'][ic]
        i, j = pos(d)
        ext = [-R, R, -R, R]
        ax[2 * i, j].imshow(np.clip(P1 / P1.max(), 1e-4, None), norm=LogNorm(1e-4, 1), cmap='viridis', origin='lower', extent=ext)
        ax[2 * i, j].set_title(f'{d.upper()} ePSF (log), {len(T_[d]["x"])} stars')
        ax[2 * i + 1, j].imshow((P1 - S) / P1.max(), vmin=-0.03, vmax=0.03, cmap='RdBu_r', origin='lower', extent=ext)
        ax[2 * i + 1, j].set_title(f'{d.upper()} (data - STPSF)/peak, +/-3%')
        m1 = r['met1'][()]; st = r['st_metrics'][ic]
        summ[d] = dict(n_core=int(len(T_[d]['x'])), n_frames=int(len(r['frames'])), n_wing=int(len(r['w_x'])),
                       fwhm_data=float(m1['fwhm']), fwhm_stpsf=float(st['fwhm']),
                       peak_ratio=float(P1.max() / S.max()), Gbest=int(r['Gbest']), bil=int(r['bilinear_best']),
                       ee2_data=float(m1['ee'][1]), ee2_stpsf=float(st['ee'][1]),
                       e_data=float(m1['e']), e_stpsf=float(st['e']))
    fig.suptitle(f'{a.filter}: global core ePSF and difference from STPSF (detector centre), per SW detector')
    fn = f'{a.prefix}1.png'; fig.savefig(fn, dpi=DPI, bbox_inches='tight'); plt.close(fig); out.append(fn)

    # ---------- 2: profile ratio, EE difference, spikes
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
    for k, d in enumerate(dets):
        r = R_[d]; O, Rw = int(r['O']), int(r['Rw']); gw = epsf.Grid(Rw, O)
        spos = r['spos']; ic = int(np.argmin(np.hypot(spos[:, 0] - 1024, spos[:, 1] - 1024)))
        Pw, Sw = r['Pw'], r['Sw'][ic]
        rb = np.concatenate([np.arange(0, 3, 0.25), np.arange(3, 10, 0.5), np.arange(10, Rw + 0.1, 1.0)])
        rc = 0.5 * (rb[1:] + rb[:-1])
        ax[0].plot(rc, epsf.radial_profile(Pw, gw, rb) / epsf.radial_profile(Sw, gw, rb), color=COL[k], lw=1.3, label=d.upper())
        radii = np.linspace(0.25, Rw, 120)
        ax[1].plot(radii, epsf.encircled(Pw, gw, radii) - epsf.encircled(Sw, gw, radii), color=COL[k], lw=1.3)
        th = np.degrees(np.arctan2(gw.uy, gw.ux)) % 360
        tb = np.arange(0, 361, 3); tc = 0.5 * (tb[1:] + tb[:-1])
        annm = (gw.ur > 8) & (gw.ur < 25)
        ad = np.array([Pw[annm & (th >= a0) & (th < a1)].mean() for a0, a1 in zip(tb[:-1], tb[1:])])
        as_ = np.array([Sw[annm & (th >= a0) & (th < a1)].mean() for a0, a1 in zip(tb[:-1], tb[1:])])
        spk = as_ > np.percentile(as_, 85)
        summ[d]['spike_excess_ratio'] = float((ad[spk].mean() - ad[~spk].mean()) / (as_[spk].mean() - as_[~spk].mean()))
        for r0, r1 in ((3, 6), (6, 10), (10, 20)):
            s = (rc > r0) & (rc < r1)
            summ[d][f'profile_ratio_{r0}_{r1}'] = float(np.mean(epsf.radial_profile(Pw, gw, rb)[s] / epsf.radial_profile(Sw, gw, rb)[s]))
        pbr = epsf.radial_profile(r['Pw_bright'], gw, rb); pfa = epsf.radial_profile(r['Pw_faint'], gw, rb)
        s = (rc > 12) & (rc < 25)
        summ[d]['bright_over_faint_12_25'] = float(pbr[s].mean() / pfa[s].mean())
        if k == 0:
            ax[2].plot(tc, as_ / as_.max(), color='k', lw=2, label='STPSF')
        ax[2].plot(tc, ad / as_.max(), color=COL[k], lw=1)
    ax[0].axhline(1, color='gray'); ax[0].set_ylim(0, 1.6); ax[0].set_xlabel('r [px]'); ax[0].set_ylabel('data / STPSF (azimuthal mean)')
    ax[0].legend(ncol=2, fontsize=8); ax[0].set_title('profile ratio (normalised to flux within r = 10 px)')
    ax[1].axhline(0, color='gray'); ax[1].set_xlabel('r [px]'); ax[1].set_ylabel('EE(data) - EE(STPSF)'); ax[1].set_title('EE difference')
    ax[2].set_xlabel('position angle [deg]'); ax[2].set_ylabel('mean ePSF, 8-25 px (/ STPSF max)'); ax[2].legend()
    ax[2].set_title('spikes: azimuthal profile, 8-25 px')
    fig.suptitle(f'{a.filter}: radial profile, EE and spikes vs STPSF, all SW detectors')
    fn = f'{a.prefix}2.png'; fig.savefig(fn, dpi=DPI, bbox_inches='tight'); plt.close(fig); out.append(fn)

    # ---------- 3: maps (FWHM and e2 and e1) data and STPSF, focal-plane layout
    for q, (key, scale, lab, vr) in enumerate([('dsize_global', 100, 'FWHM change [%]', 1.5), ('de1_global', 1, 'delta e1', 0.015),
                                             ('de2_global', 1, 'delta e2', 0.015)]):
        fig, ax = plt.subplots(4, 4, figsize=(14, 14))
        for d in dets:
            t = T_[d]; r = R_[d]
            ok = t['ok'] & np.isfinite(t[key]) & (t['chi2'] < np.nanpercentile(t['chi2'], 95))
            M, N = bin_map(t['x'][ok], t['y'][ok], t[key][ok] * scale, 16)
            ev = (t['frame'][ok] % 2) == 0
            Ma, _ = bin_map(t['x'][ok][ev], t['y'][ok][ev], t[key][ok][ev] * scale, 16)
            Mb, _ = bin_map(t['x'][ok][~ev], t['y'][ok][~ev], t[key][ok][~ev] * scale, 16)
            i, j = pos(d)
            med = np.nanmedian(M)
            ax[2 * i, j].imshow(M - med, origin='lower', cmap='RdBu_r', vmin=-vr, vmax=vr, extent=[0, 2048, 0, 2048])
            ax[2 * i, j].set_title(f'{d.upper()} data: range(p5-p95) {np.nanpercentile(M, 95) - np.nanpercentile(M, 5):.3g}\nnoise {np.nanstd((Ma - Mb) / 2):.2g}, {int(np.median(N))} stars/cell')
            ns = int(round(np.sqrt(len(r['spos']))))
            sv = (r['sh_stpsf'][:, q] * scale).reshape(ns, ns)
            h = ax[2 * i + 1, j].imshow(sv - np.nanmedian(sv), origin='lower', cmap='RdBu_r', vmin=-vr, vmax=vr, extent=[0, 2048, 0, 2048])
            ax[2 * i + 1, j].set_title(f'{d.upper()} STPSF: range {np.nanmax(sv) - np.nanmin(sv):.3g}')
            summ[d][f'{key}_range_data'] = float(np.nanpercentile(M, 95) - np.nanpercentile(M, 5))
            summ[d][f'{key}_range_stpsf'] = float(np.nanmax(sv) - np.nanmin(sv))
            summ[d][f'{key}_noise'] = float(np.nanstd((Ma - Mb) / 2))
        for x in ax.ravel():
            x.set_xticks([]); x.set_yticks([])
        fig.colorbar(h, ax=ax, shrink=0.5, label=lab)
        fig.suptitle(f'{a.filter}: {lab} across each SW detector (128-px cells, minus median), data vs STPSF; '
                     'detectors in focal-plane order (A: 2 4 / 1 3, B: 3 1 / 4 2), each in its own x right / y up frame')
        fn = f'{a.prefix}{3 + q}.png'; fig.savefig(fn, dpi=DPI, bbox_inches='tight'); plt.close(fig); out.append(fn)

    # ---------- 6: CV, accuracy, per-exposure
    fig, ax = plt.subplots(1, 3, figsize=(19, 4.8))
    for k, d in enumerate(dets):
        r = R_[d]
        xs, ys = [], []
        for G in r['grids']:
            b = 1 if G > 1 else 0
            if f'cv_{G}_{b}_core_metric' in r.files:
                xs.append(G); ys.append(float(r[f'cv_{G}_{b}_core_metric']))
        ys = np.array(ys)
        ax[0].plot(xs, ys / ys[0], 'o-', color=COL[k], label=d.upper())
        summ[d]['cv'] = dict(zip([int(x) for x in xs], ys.round(3).tolist()))
        Gb, bb = int(r['Gbest']), int(r['bilinear_best'])
        rbins = r['rbins']; rcen = 0.5 * (rbins[1:] + rbins[:-1])
        ax[1].semilogy(rcen, r[f'cv_{Gb}_{bb}_rms'], '-', color=COL[k])
        ax[1].semilogy(rcen, r[f'cv_{Gb}_{bb}_phot'], ':', color=COL[k])
        summ[d]['rms_over_phot_r0'] = float(r[f'cv_{Gb}_{bb}_rms'][0] / r[f'cv_{Gb}_{bb}_phot'][0])
        summ[d]['rms_r0_over_peak'] = float(r[f'cv_{Gb}_{bb}_rms'][0] / r['P1'].max())
        t = T_[d]
        ok = t['ok'] & np.isfinite(t['dsize_global'])
        fr = t['frame'][ok]
        vals = []
        for f_ in np.unique(fr):
            s = fr == f_
            vals.append((np.median(t['expstart'][ok][s]), 100 * np.median(t['dsize_global'][ok][s]),
                         100 * 1.25 * np.std(t['dsize_global'][ok][s]) / np.sqrt(s.sum()), t['fname'][ok][s][0]))
        vals.sort()
        v = np.array([x[1] for x in vals]); e = np.array([x[2] for x in vals])
        ax[2].errorbar(np.arange(len(v)), v, e, fmt='o', ms=2, color=COL[k], alpha=0.8)
        summ[d]['per_frame_fwhm_rms_pct'] = float(np.std(v)); summ[d]['per_frame_fwhm_err_pct'] = float(np.median(e))
        summ[d]['frames_below_-1pct'] = [x[3][:26] for x in vals if x[1] < -1]
    ax[0].set_xlabel('grid G'); ax[0].set_ylabel('held-out core metric / G=1'); ax[0].legend(ncol=2, fontsize=8)
    ax[0].set_title('cross-validation (bilinear grids)')
    ax[1].set_xlabel('r [px]'); ax[1].set_ylabel('per-pixel rms [flux(r<10)]'); ax[1].set_title('held-out residual rms (solid) vs photon noise (dotted)')
    ax[2].axhline(0, color='gray'); ax[2].set_ylim(-7, 1.5); ax[2].set_xlabel('frame (time order)'); ax[2].set_ylabel('median FWHM change [%]')
    ax[2].set_title('per-exposure core width')
    fig.suptitle(f'{a.filter}: accuracy and per-exposure stability, all SW detectors')
    fn = f'{a.prefix}6.png'; fig.savefig(fn, dpi=DPI, bbox_inches='tight'); plt.close(fig); out.append(fn)
    summ['figures'] = out
    with open(a.prefix + '_summary.json', 'w') as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
