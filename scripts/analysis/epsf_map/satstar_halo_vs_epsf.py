"""Can the ePSF (or the hybrid PSF) model a saturated star's halo to better than 5%?

    python satstar_halo_vs_epsf.py <result_nrcblong.npz> <cutout dir> <brightness.json> <out prefix>

Issues #993 / #994: the leave-one-dither-out (LOO) empirical model of
``scripts/analysis/satstar_poisson`` leaves ~5% of the local halo intensity at
30-150 px around every saturated F480M star, because the halo changes between
dithers (o061: e3/e4 -17%, e5 +7% relative to e1, spikes unchanged).  This
script asks whether the ePSF of ``epsf_map`` would do better.  Three tests:

1. **Static STPSF fit** (beyond R1 = 12 px the hybrid of PR #1009 IS this STPSF).
   ``A_e * PSF + B`` per dither on the azimuthal-MEDIAN profile (field stars
   rejected), ``B`` the far-field level from brightness.json.  Data / model per
   annulus, for the target (o061, saturated core r_eq ~60 px) and o042 (the
   smallest core, r_eq ~22 px).
2. **Pure ePSF wing** (``Pw``, built out to Rw = 30 px): its profile relative to
   STPSF at 8-29 px, both normalised to unit flux inside r_norm.
3. **Does the ePSF track the dither change?**  Per-exposure field-star wing
   excess (``wnum / wden`` of analyze_detector.py, against the global wing
   ePSF) for o061's six dithers, against the saturated star's halo ratios.

The large-FOV STPSFs (801 px, F480M, NRCB5, the 10678-epoch WSS OPD) are built
at each star's detector position on the first run and cached as
``<out prefix>_psf_<star>.npy``.
"""
import json
import os
import sys

import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OPD = 'R2026091802-NRCA1_FP6-1.fits'
EDGES = np.array([12, 16, 20, 25, 30, 40, 50, 65, 80, 100, 120, 150, 200, 250, 300])
# o061 saturated-star halo relative to dither 1 (satstar_poisson README §3, fig 3)
SATSTAR_RATIO_061 = {1: 1.0, 2: 1.025, 3: 0.83, 4: 0.82, 5: 1.07, 6: 0.92}
STARS = {'tgt': dict(dithers=[1, 2, 3, 4, 5, 6], fit=(70, 250), label='o061 (target, core r$_{eq}$ 60 px)'),
         'trn_042': dict(dithers=[1, 2, 3, 6], fit=(30, 120), label='o042 (core r$_{eq}$ 22 px)')}


def stpsf_at(x, y, fn):
    if os.path.exists(fn):
        return np.load(fn)
    import stpsf
    nc = stpsf.NIRCam()
    nc.filter = 'F480M'
    nc.detector = 'NRCB5'
    nc.detector_position = (x, y)
    nc.load_wss_opd(OPD, plot=False, verbose=False, use_exact_wss_target_phase=False)
    p = nc.calc_psf(fov_pixels=801, oversample=2)['DET_DIST'].data
    np.save(fn, p)
    return p


def profile_fit(cutdir, star, dithers, rfit, B, outp):
    out = {}
    for e in dithers:
        z = np.load(f'{cutdir}/{star}_e{e}.npz')
        d, dq, xt, yt = z['d'], z['dq'], float(z['xt']), float(z['yt'])
        if e == dithers[0]:
            P = stpsf_at(int(z['X0']) + xt, int(z['Y0']) + yt, f'{outp}_psf_{star}.npy')
            c = (P.shape[0] - 1) / 2
        yy, xx = np.mgrid[0:d.shape[0], 0:d.shape[1]]
        r = np.hypot(xx - xt, yy - yt)
        M = ndimage.map_coordinates(P, [yy - yt + c, xx - xt + c], order=3, mode='constant', cval=0.0)
        good = np.isfinite(d) & ((dq & 3) == 0) & (r < c - 2)          # not DO_NOT_USE / SATURATED
        pd, pm = [], []
        for a, b in zip(EDGES[:-1], EDGES[1:]):
            m = good & (r >= a) & (r < b)
            pd.append(np.median(d[m] - B) if m.sum() > 30 else np.nan)
            pm.append(np.median(M[m]) if m.sum() > 30 else np.nan)
        pd, pm = np.array(pd), np.array(pm)
        rc = 0.5 * (EDGES[1:] + EDGES[:-1])
        f = (rc >= rfit[0]) & (rc <= rfit[1]) & np.isfinite(pd)
        A = np.sum(pd[f] * pm[f]) / np.sum(pm[f] ** 2)
        out[e] = dict(A=float(A), ratio=(pd / (A * pm)).tolist(), I_star=(A * pm).tolist())
    return out, P


def epsf_over_stpsf(res, P):
    Pw, O, rn = res['Pw'], int(res['O']), float(res['r_norm'])
    c = (Pw.shape[0] - 1) // 2
    u = (np.arange(Pw.shape[0]) - c) / O
    R = np.hypot(*np.meshgrid(u, u))
    cs = (P.shape[0] - 1) // 2
    k = np.arange(P.shape[0]) - cs
    RS = np.hypot(*np.meshgrid(k, k))
    fS = P[RS <= rn].sum()
    edges = [8, 12, 16, 20, 25, 29]
    return edges, [float(np.mean(Pw[(R >= a) & (R < b)]) / (np.mean(P[(RS >= a) & (RS < b)]) / fS))
                   for a, b in zip(edges[:-1], edges[1:])]


def wing_per_exposure(res, obs='061'):
    wn, wd, fn = res['wnum'], res['wden'], res['w_fname']
    ob = np.array([f[7:10] for f in fn])
    ex = np.array([int(f.split('_')[2]) for f in fn])
    rng = np.random.default_rng(1)
    out = {}
    for e in sorted(set(ex[ob == obs])):
        m = (ob == obs) & (ex == e) & np.all(np.isfinite(wn), 1)
        n, d = wn[m], wd[m]
        bs = [n[i].sum(0) / d[i].sum(0) for i in (rng.integers(0, m.sum(), m.sum()) for _ in range(200))]
        out[int(e)] = dict(n=int(m.sum()), excess=(n.sum(0) / d.sum(0)).tolist(), err=np.std(bs, 0).tolist())
    return out


def main():
    rfn, cutdir, bfn, outp = sys.argv[1:5]
    res = np.load(rfn, allow_pickle=True)
    bright = json.load(open(bfn))
    summary = dict(edges=EDGES.tolist(), stars={})
    psfs = {}
    for star, cfg in STARS.items():
        prof, psfs[star] = profile_fit(cutdir, star, cfg['dithers'], cfg['fit'], bright[star]['bkg'], outp)
        summary['stars'][star] = prof
    summary['epsf_over_stpsf_edges'], summary['epsf_over_stpsf'] = epsf_over_stpsf(res, psfs['trn_042'])
    summary['wing_per_exposure_061'] = wing_per_exposure(res)
    summary['wing_annuli_px'] = res['ann_rs'].tolist()
    json.dump(summary, open(outp + '.json', 'w'), indent=1)

    rc = 0.5 * (EDGES[1:] + EDGES[:-1])
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))
    for q, (star, cfg) in enumerate(STARS.items()):
        a = ax[q] if q == 0 else ax[1]
        for e, v in summary['stars'][star].items():
            a.plot(rc, v['ratio'], 'o-' if q == 0 else 's--', ms=4,
                   label=f'{cfg["label"].split(" ")[0]} e{e}  (A/A$_1$ = {v["A"] / summary["stars"][star][cfg["dithers"][0]]["A"]:.2f})')
    ax[0].axhspan(0.95, 1.05, color='C2', alpha=0.25, label='±5% (LOO model, #992/#998)')
    ax[0].set_xscale('log'); ax[0].set_ylim(0.5, 2.5); ax[0].set_xlim(30, 320)
    ax[0].set_xticks([30, 50, 100, 200, 300]); ax[0].set_xticklabels(['30', '50', '100', '200', '300'])
    ax[0].set_xlabel('radius [px]'); ax[0].set_ylabel('data / (A$_e$ STPSF + B), azimuthal median')
    ax[0].set_title(f'{STARS["tgt"]["label"]}: static STPSF\n(= the hybrid beyond 12 px), A$_e$ free per dither')
    ax[0].legend(fontsize=8, ncol=1)
    e_edges, e_ratio = summary['epsf_over_stpsf_edges'], summary['epsf_over_stpsf']
    erc = 0.5 * (np.array(e_edges[1:]) + np.array(e_edges[:-1]))
    ax[1].plot(erc, e_ratio, 'k^-', lw=2, label='pure ePSF wing / STPSF (same flux in 10 px)')
    ax[1].axhspan(0.95, 1.05, color='C2', alpha=0.25)
    ax[1].axhline(1, color='k', lw=0.5)
    # beyond ~50 px the o042 halo is fainter than the scene and the median profile measures the scene
    ax[1].set_xlim(8, 50); ax[1].set_ylim(0, 2.0)
    ax[1].set_xlabel('radius [px]'); ax[1].set_ylabel('ratio')
    ax[1].set_title(f'{STARS["trn_042"]["label"]}: data / STPSF (dashed)\nand the ePSF wing, which ends at 30 px')
    ax[1].legend(fontsize=8)
    w = summary['wing_per_exposure_061']
    ex = sorted(w)
    for k, (lab, mk) in enumerate((('2-4 px', 'o'), ('4-8 px', 's'))):
        v = np.array([w[e]['excess'][k] for e in ex]); s = np.array([w[e]['err'][k] for e in ex])
        ax[2].errorbar(np.array(ex) + 0.08 * (k - 0.5), 1 + v - v[0], s, fmt=mk + '-', capsize=3,
                       label=f'field-star wing {lab}, relative to e1 (N≈{w[ex[0]]["n"]}/exp)')
    ax[2].plot(ex, [SATSTAR_RATIO_061[e] for e in ex], 'k*-', ms=12, label='saturated-star halo ratio (30-150 px)')
    ax[2].set_xlabel('o061 dither'); ax[2].set_ylabel('relative to dither 1')
    ax[2].set_title('does the field-star ePSF track the per-dither halo change?')
    ax[2].legend(fontsize=8); ax[2].set_ylim(0.75, 1.15)
    fig.tight_layout()
    fig.savefig(outp + '.png', dpi=int(os.environ.get('FIG_DPI', 150)))


if __name__ == '__main__':
    main()
