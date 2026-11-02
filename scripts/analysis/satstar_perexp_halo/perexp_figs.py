"""Figures and the self-constrained tests for the per-exposure halo change.

    python perexp_figs.py <pxh.npz> <models.json> <out dir> [sigma_max] [epsf_map result dir]

Input: ``perexp_measure.py`` and ``perexp_models.py`` outputs.  Writes
``fig1_gallery.png`` ... ``fig8_models.png`` and ``tests.json`` with the numbers
quoted in the issue:

* ``pairs``: correlation of ``a`` between two stars of the SAME exposure, against
  their separation, with the control of two stars at the same dither index of
  DIFFERENT visits;
* ``inner_outer``: does the exposure's own inner halo (r = 15-40 px) predict its
  outer halo (40-80, 80-150 px)?  slope, correlation, fraction explained;
* ``split_half``: amplitude from the even vs the odd azimuth sectors (the
  noise-limited ceiling for any amplitude-only self-calibration);
* ``radial_pca``: principal modes of the per-annulus change ``prof``.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from perexp_models import good_scene_fit, cv

DPI = int(os.environ.get('FIG_DPI', 150))
DETS = ('NRCALONG', 'NRCBLONG')
S_INT = 0.08          # star-to-star scatter of the change; regression weights 1/(sigma^2 + S_INT^2)


def load(fin, smax):
    z = np.load(fin, allow_pickle=True)
    t = {k: z[k] for k in z.files}
    ok = np.isfinite(t['a']) & (t['ae'] > 0) & (t['ae'] < smax) & good_scene_fit(t)
    gs, cnt = np.unique(t['grp'][ok], return_counts=True)
    ok &= np.isin(t['grp'], gs[cnt >= 4])
    n = len(t['a'])
    sel = {k: (v[ok] if (np.ndim(v) and len(v) == n) else v) for k, v in t.items()}
    for g in np.unique(sel['grp']):
        m = sel['grp'] == g
        sel['a'][m] -= sel['a'][m].mean()
    return sel


def wm(x, s, axis=None):
    w = np.where(np.isfinite(x) & np.isfinite(s) & (s > 0), 1 / np.where(s > 0, s, 1) ** 2, 0)
    xs = np.where(w > 0, x, 0)
    W = w.sum(axis)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (w * xs).sum(axis) / W, 1 / np.sqrt(W)


def recentre(v, ve, grp):
    v = v.copy()
    for g in np.unique(grp):
        m = (grp == g) & np.isfinite(v)
        if m.sum():
            v[m] -= np.average(v[m], weights=1 / ve[m] ** 2)
    return v


def fig_gallery(t, out):
    """Polar maps of delta_e(r, theta) for the best-measured six-dither star-visits."""
    gs, cnt = np.unique(t['grp'], return_counts=True)
    six = gs[cnt == 6]
    score = np.array([np.median(t['ae'][t['grp'] == g]) for g in six])
    pick = six[np.argsort(score)[:6]]
    E = t['edges'].astype(float)
    th_e = np.radians(np.linspace(0, 360, t['delta'].shape[2] + 1))
    fig, ax = plt.subplots(len(pick), 6, figsize=(17, 2.95 * len(pick)), subplot_kw=dict(projection='polar'))
    for i, g in enumerate(pick):
        ii = np.flatnonzero(t['grp'] == g)
        ii = ii[np.argsort(t['expnum'][ii])]
        for j, k in enumerate(ii):
            a = ax[i, j]
            im = a.pcolormesh(th_e, np.log10(E), t['delta'][k], cmap='RdBu_r', vmin=-0.3, vmax=0.3, shading='flat')
            a.set_ylim(np.log10(E[0]), np.log10(E[-1])); a.set_yticklabels([]); a.set_xticklabels([])
            a.set_title(f'e{t["expnum"][k]}  a={t["a"][k]:+.3f}', fontsize=9)
            if j == 0:
                a.text(-0.35, 0.5, f'star {t["sid"][k]}\no{t["obs"][k]:03d} {t["det"][k][3:]}\n({t["xc"][k]:.0f},{t["yc"][k]:.0f})',
                       transform=a.transAxes, ha='center', va='center', fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.5, label=r'$\delta_e(r,\theta) = (H_e - \bar H)/I(r)$')
    fig.suptitle(r'Per-exposure halo change about each star, relative to its own dither mean; '
                 r'radius log-scaled 15-250 px (spike sectors masked = blank)', fontsize=12)
    fig.savefig(f'{out}/fig1_gallery.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def fig_dither(t, mods, out):
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.5), gridspec_kw=dict(width_ratios=[1, 1, 1.3]))
    res = {}
    for q, d in enumerate(DETS):
        m = t['det'] == d
        data = [t['a'][m & (t['expnum'] == e)] for e in range(1, 7)]
        ax[q].violinplot(data, positions=range(1, 7), showextrema=False)
        med = [np.median(x) for x in data]
        ste = [1.2533 * np.std(x) / np.sqrt(len(x)) for x in data]
        ax[q].errorbar(range(1, 7), med, ste, fmt='ko-', capsize=4, label='median ± s.e.')
        pd_ = mods['models']['dither'].get('per_dither', {}).get(d)
        if pd_ is not None:
            pdv = np.array(pd_) - np.mean(pd_)
            ax[q].plot(range(1, 7), pdv, 'r*--', ms=11, label='dither-only model (CV mean coef)')
        ax[q].axhline(0, color='k', lw=0.5)
        ax[q].set_ylim(-0.4, 0.4); ax[q].set_xlabel('dither (expnum)'); ax[q].set_ylabel('a (fractional halo change, r=15-80)')
        ax[q].set_title(f'{d}: {m.sum()} exposures'); ax[q].legend(fontsize=8)
        res[d] = dict(median=list(map(float, med)), se=list(map(float, ste)),
                      rms_within_dither=[float(np.std(x)) for x in data])
    # matrix: star-visit x dither, sorted by obs then detector
    gs = np.unique(t['grp'])
    M = np.full((len(gs), 6), np.nan)
    key = np.zeros(len(gs))
    for i, g in enumerate(gs):
        ii = np.flatnonzero(t['grp'] == g)
        M[i, t['expnum'][ii] - 1] = t['a'][ii]
        key[i] = t['obs'][ii[0]] * 10 + (t['det'][ii[0]] == 'NRCBLONG')
    o = np.argsort(key, kind='stable')
    im = ax[2].imshow(M[o], aspect='auto', cmap='RdBu_r', vmin=-0.25, vmax=0.25, interpolation='nearest',
                      extent=[0.5, 6.5, len(gs), 0])
    ob = (key[o] // 10).astype(int)
    tick = np.flatnonzero(np.r_[True, np.diff(ob) != 0])
    ax[2].set_yticks(tick[::4]); ax[2].set_yticklabels([f'o{ob[k]:03d}' for k in tick[::4]], fontsize=6)
    ax[2].set_xlabel('dither'); ax[2].set_title(f'every star-visit ({len(gs)}), sorted by observation')
    fig.colorbar(im, ax=ax[2], label='a')
    fig.tight_layout(); fig.savefig(f'{out}/fig2_dither.png', dpi=DPI); plt.close(fig)
    return res


def binned_map(x, y, v, nb=8):
    e = np.linspace(0, 2048, nb + 1)
    s, *_ = stats.binned_statistic_2d(x, y, v, 'median', bins=[e, e])
    c, *_ = stats.binned_statistic_2d(x, y, v, 'count', bins=[e, e])
    return np.where(c >= 5, s, np.nan).T, e


def fig_position(t, out):
    fig, ax = plt.subplots(2, 7, figsize=(22, 6.6))
    for i, d in enumerate(DETS):
        md = t['det'] == d
        for e in range(1, 7):
            m = md & (t['expnum'] == e)
            S, ed = binned_map(t['xc'][m], t['yc'][m], t['a'][m])
            im = ax[i, e - 1].imshow(S, origin='lower', extent=[0, 2048, 0, 2048], cmap='RdBu_r', vmin=-0.15, vmax=0.15)
            ax[i, e - 1].set_title(f'{d[3:]} e{e} (N={m.sum()})', fontsize=9)
            ax[i, e - 1].set_xticks([0, 1024, 2048]); ax[i, e - 1].set_yticks([0, 1024, 2048])
        S, _ = binned_map(t['xc'][md], t['yc'][md], np.abs(t['a'][md]))
        im2 = ax[i, 6].imshow(S, origin='lower', extent=[0, 2048, 0, 2048], cmap='viridis', vmin=0, vmax=0.15)
        ax[i, 6].set_title(f'{d[3:]} median |a|, all dithers', fontsize=9)
    fig.colorbar(im, ax=ax[:, :6], shrink=0.8, label='median a in 256-px cell')
    fig.colorbar(im2, ax=ax[:, 6], shrink=0.8, label='median |a|')
    fig.suptitle('Is the halo change a function of where the star is on the detector?  (cells with ≥5 exposures)')
    fig.savefig(f'{out}/fig3_position.png', dpi=DPI, bbox_inches='tight'); plt.close(fig)


def field_wing_vs_x(epsfdir, det, e):
    """Wing excess of the unsaturated field stars (epsf_map ``wnum/wden``, against the
    global wing ePSF) in detector-column bins, relative to its median."""
    fn = f'{epsfdir}/result_{det.lower()}.npz'
    if not (epsfdir and os.path.exists(fn)):
        return None
    z = np.load(fn, allow_pickle=True)
    wn, wd, x = z['wnum'], z['wden'], z['w_x']
    ok = np.all(np.isfinite(wn), 1) & np.all(np.isfinite(wd), 1)
    k = np.digitize(x, e) - 1
    out = {}
    for j, (lo, hi) in enumerate(z['ann_rs'].tolist()):
        v = np.array([wn[ok & (k == i), j].sum() / wd[ok & (k == i), j].sum() for i in range(len(e) - 1)])
        # wnum/wden is the excess over the ePSF; as a ratio to the column median
        out[f'{lo}-{hi}'] = ((1 + v) / np.median(1 + v) - 1).tolist()
    return out


def fig_xprofile(t, mods, out, epsfdir=None):
    """The detector-column dependence: binned (differenced) a and core area against
    x, and the fitted free profile f(x) of ``perexp_models`` (zeroed at its median)."""
    fig, ax = plt.subplots(1, 3, figsize=(20, 5.4))
    e = np.arange(0, 2049, 64)
    c = 0.5 * (e[1:] + e[:-1])
    res = {}
    for q, d in enumerate(DETS):
        m = t['det'] == d
        k = np.digitize(t['xc'][m], e) - 1
        ma = np.array([np.median(t['a'][m][k == i]) if (k == i).sum() > 10 else np.nan for i in range(len(c))])
        mr = np.array([np.median(t['area'][m][k == i] - 1) if (k == i).sum() > 10 else np.nan for i in range(len(c))])
        f = np.array(mods['models']['xprofile']['f'][d])
        f -= np.median(f)
        ax[q].plot(c, ma, 'ko-', ms=4, label='median a of stars in this column (differenced)')
        ax[q].plot(c, mr, 's-', color='C1', ms=4, label='median saturated-core area − 1 (raw DQ, independent)')
        ax[q].plot(mods['models']['xprofile']['knots'], f, 'C3-', lw=2.5, label='fitted halo f(x), CV-validated (zeroed at median)')
        e2 = np.arange(0, 2049, 128)
        fw = field_wing_vs_x(epsfdir, d, e2)
        if fw is not None:
            for key, sty in (('4-8', 'C2--'), ('8-15', 'C4--')):
                ax[q].plot(0.5 * (e2[1:] + e2[:-1]), fw[key], sty, lw=1.5,
                           label=f'UNSATURATED field stars: PSF wing {key} px vs column (epsf_map)')
        ax[q].axhline(0, color='k', lw=0.5)
        for xd in (-385, 385):
            ax[q].annotate('', xy=(1024 + xd, -0.42), xytext=(1024, -0.42), arrowprops=dict(arrowstyle='->', color='0.5'))
        ax[q].text(1024, -0.40, 'dither throw ±385 px', ha='center', color='0.4', fontsize=8)
        ax[q].set_ylim(-0.45, 0.4); ax[q].set_xlim(0, 2048)
        ax[q].set_xlabel('detector x (column) [px]'); ax[q].set_ylabel('fractional change')
        ax[q].set_title(f'{d}: halo amplitude vs detector column'); ax[q].legend(fontsize=8, loc='upper center')
        res[d] = dict(x=c.tolist(), median_a=ma.tolist(), median_area=mr.tolist(), f=f.tolist(), field_wing=fw)
    for q, d in enumerate(DETS):
        f = np.array(mods['models']['yprofile']['f'][d]); f -= np.median(f)
        ax[2].plot(mods['models']['yprofile']['knots'], f, f'C{q}-', lw=2, label=f'{d}: f(y)')
    ax[2].axhline(0, color='k', lw=0.5); ax[2].set_ylim(-0.45, 0.4); ax[2].set_xlim(0, 2048)
    ax[2].set_xlabel('detector y (row) [px]'); ax[2].set_title('same, against detector row (fitted f(y))'); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f'{out}/fig3b_xprofile.png', dpi=DPI); plt.close(fig)
    return res


def floor(t):
    """rms of what is left of the change, per annulus, after each correction.

    The change is modelled as a_hat * m(r), m(r) the regression of the annulus
    change on a (the mean radial shape), with a_hat from: nothing; the CV
    prediction of the column profile + core area (an a priori calibration plus
    an in-exposure observable); the exposure's own inner halo (r = 15-40 px,
    fitted in the data, which is what a per-exposure free amplitude in the LOO
    fit would do); the exposure's own a (15-80 px).  Measured on the azimuthal
    mean ``prof`` and on the 12-sector cells ``delta``."""
    rc = t['rc']
    P = np.array([recentre(t['prof'][:, j].astype(float), np.where(np.isfinite(t['profe'][:, j]), t['profe'][:, j], 1),
                           t['grp']) for j in range(len(rc))]).T
    D = t['delta'].astype(float)
    Dc = D - np.nanmean(D, 2, keepdims=True) + P[:, :, None]          # cells, with the annulus mean re-centred
    S = t['sigma'].astype(float)
    m = np.array([np.nansum(P[:, j] * t['a']) / np.nansum(np.where(np.isfinite(P[:, j]), t['a'] ** 2, 0)) for j in range(len(rc))])
    rng = np.random.default_rng(42)
    pred_xa, _ = cv(t, ('xprofile', 'area'), rng)
    ri = (rc >= 15) & (rc < 40)
    inner, _ = wm(D[:, ri, :].reshape(len(D), -1), S[:, ri, :].reshape(len(D), -1), axis=1)
    inner = recentre(inner, np.ones_like(inner), t['grp'])
    mi = np.nansum(m[ri] * np.isfinite(P[:, ri]), 1) / np.maximum(np.isfinite(P[:, ri]).sum(1), 1)
    ahat = dict(none=np.zeros(len(D)), column_plus_area=pred_xa, own_inner_halo=inner / np.where(mi > 0, mi, np.nan),
                own_amplitude=t['a'])
    out = dict(rc=rc.tolist(), m=m.tolist(), prof={}, cells={})
    rob = lambda x, ax: 1.4826 * np.nanmedian(np.abs(x), ax)
    models = {k: ah[:, None] * m[None] for k, ah in ahat.items()}
    # two free radial modes per exposure (global PCA modes of the first 8 annuli,
    # coefficients fitted to the exposure's own profile)
    k8 = 8
    good = np.all(np.isfinite(P[:, :k8]), 1)
    _, _, vt = np.linalg.svd(P[good, :k8], full_matrices=False)
    M2 = np.full_like(P, np.nan)
    for i in range(len(P)):
        f = np.isfinite(P[i, :k8])
        if f.sum() >= 3:
            c, *_ = np.linalg.lstsq(vt[:2, f].T, P[i, :k8][f], rcond=None)
            M2[i, :k8] = c @ vt[:2]
    models['own_two_modes'] = M2
    for k, mod in models.items():
        rp = P - mod
        rcl = Dc - mod[:, :, None]
        out['prof'][k] = np.sqrt(np.nanmean(rp ** 2, 0)).tolist()
        out['cells'][k] = rob(rcl, (0, 2)).tolist()
    out['prof']['noise'] = np.sqrt(np.nanmean(t['profe'].astype(float) ** 2, 0)).tolist()
    out['cells']['noise'] = np.sqrt(np.nanmedian(S ** 2, (0, 2))).tolist()
    out['n_per_annulus'] = np.isfinite(P).sum(0).tolist()
    return out


def fig_floor(fl, out):
    rc = np.array(fl['rc'])
    n = np.array(fl['n_per_annulus'])
    ok = n >= 50
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.6))
    lab = dict(none='no correction (static halo = LOO mean)', column_plus_area='a priori: f(x) column calibration + core area',
               own_inner_halo="self: amplitude from the exposure's own r=15-40 px halo",
               own_amplitude="self: the exposure's own a (r=15-80 px)",
               own_two_modes='self: two free radial modes per exposure (r<100 px)',
               noise='formal measurement noise (He)')
    sty = dict(none='k-o', column_plus_area='C0-s', own_inner_halo='C3-^', own_amplitude='C2-D', own_two_modes='C4-v',
               noise='k:')
    for q, (key, tt) in enumerate((('prof', 'azimuthal mean of the change (rms)'),
                                   ('cells', '30° sector x annulus cells (robust rms, 1.48 MAD)'))):
        for k, v in fl[key].items():
            ax[q].plot(rc[ok], np.array(v)[ok] * 100, sty[k], ms=4, label=lab[k])
        ax[q].axhline(5, color='C1', ls='--', lw=1, label='5% (the LOO residual of #992/#998)')
        ax[q].set_xscale('log'); ax[q].set_xlabel('radius [px]'); ax[q].set_ylabel('rms left, % of the local halo')
        ax[q].set_title(f'left after each correction:\n{tt}')
        ax[q].set_ylim(0, 25); ax[q].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f'{out}/fig9_floor.png', dpi=DPI); plt.close(fig)


def pair_stats(t, rng):
    """Correlation of a between two stars of the same exposure, vs separation; and the
    control of two stars at the same dither index of different visits (same detector)."""
    roots = np.unique(t['root'])
    I, J = [], []
    for r in roots:
        ii = np.flatnonzero(t['root'] == r)
        if len(ii) < 2:
            continue
        a_, b_ = np.triu_indices(len(ii), 1)
        I.append(ii[a_]); J.append(ii[b_])
    I, J = np.concatenate(I), np.concatenate(J)
    keep = t['sid'][I] != t['sid'][J]
    I, J = I[keep], J[keep]
    sep = np.hypot(t['xc'][I] - t['xc'][J], t['yc'][I] - t['yc'][J])
    bins = [0, 100, 200, 400, 700, 1000, 1500, 3000]
    out = dict(n_pairs=int(len(I)), bins=bins, r=[], n=[], r_err=[])
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (sep >= lo) & (sep < hi)
        if m.sum() < 10:
            out['r'].append(np.nan); out['n'].append(int(m.sum())); out['r_err'].append(np.nan); continue
        rr = np.corrcoef(t['a'][I[m]], t['a'][J[m]])[0, 1]
        out['r'].append(float(rr)); out['n'].append(int(m.sum())); out['r_err'].append(float((1 - rr ** 2) / np.sqrt(m.sum())))
    # control: same detector + same dither index, different visit, random partners
    ctl = []
    for d in DETS:
        for e in range(1, 7):
            ii = np.flatnonzero((t['det'] == d) & (t['expnum'] == e))
            jj = rng.permutation(ii)
            ok = t['obs'][ii] != t['obs'][jj]
            ctl.append(np.c_[ii[ok], jj[ok]])
    ctl = np.concatenate(ctl)
    out['control_same_dither_other_visit_r'] = float(np.corrcoef(t['a'][ctl[:, 0]], t['a'][ctl[:, 1]])[0, 1])
    out['control_n'] = int(len(ctl))
    noise = np.mean(t['ae'] ** 2)
    out['noise_ceiling_r'] = float(1 - noise / np.mean(t['a'] ** 2))
    return out, (I, J, sep)


def fig_pairs(t, ps, IJ, out):
    I, J, sep = IJ
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))
    m = sep < 200
    ax[0].hexbin(t['a'][I[m]], t['a'][J[m]], gridsize=50, extent=[-0.4, 0.4, -0.4, 0.4], mincnt=1, bins='log', cmap='magma_r')
    ax[0].plot([-0.4, 0.4], [-0.4, 0.4], 'k:', lw=0.8)
    ax[0].set_xlabel('a, star i'); ax[0].set_ylabel('a, star j (same exposure)')
    ax[0].set_title(f'pairs < 200 px apart: r = {np.corrcoef(t["a"][I[m]], t["a"][J[m]])[0, 1]:.2f} (N={m.sum()})')
    m = sep > 1000
    ax[1].hexbin(t['a'][I[m]], t['a'][J[m]], gridsize=50, extent=[-0.4, 0.4, -0.4, 0.4], mincnt=1, bins='log', cmap='magma_r')
    ax[1].plot([-0.4, 0.4], [-0.4, 0.4], 'k:', lw=0.8)
    ax[1].set_xlabel('a, star i'); ax[1].set_ylabel('a, star j (same exposure)')
    ax[1].set_title(f'pairs > 1000 px apart: r = {np.corrcoef(t["a"][I[m]], t["a"][J[m]])[0, 1]:.2f} (N={m.sum()})')
    b = np.array(ps['bins'], float)
    bc = 0.5 * (b[1:] + b[:-1])
    ax[2].errorbar(bc, ps['r'], ps['r_err'], fmt='ko-', capsize=3, label='two stars, same exposure')
    ax[2].axhline(ps['control_same_dither_other_visit_r'], color='C1', ls='--',
                  label=f'control: same detector+dither, other visit ({ps["control_same_dither_other_visit_r"]:+.2f})')
    ax[2].axhline(ps['noise_ceiling_r'], color='C2', ls=':', label=f'noise ceiling (identical signal) {ps["noise_ceiling_r"]:.2f}')
    ax[2].axhline(0, color='k', lw=0.5)
    ax[2].set_xlabel('separation of the two stars [px]'); ax[2].set_ylabel('correlation of a')
    ax[2].set_xscale('log'); ax[2].set_ylim(-0.2, 1.0); ax[2].legend(fontsize=8)
    ax[2].set_title('is the change a property of the exposure?')
    fig.tight_layout(); fig.savefig(f'{out}/fig4_pairs.png', dpi=DPI); plt.close(fig)


def fig_area(t, out):
    x = t['area'] - 1
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
    ax[0].hexbin(x, t['a'], gridsize=60, extent=[-0.25, 0.25, -0.4, 0.4], mincnt=1, bins='log', cmap='magma_r')
    w = 1 / (t['ae'] ** 2 + S_INT ** 2)
    beta = np.sum(w * x * t['a']) / np.sum(w * x * x)
    xx = np.linspace(-0.25, 0.25, 3)
    ax[0].plot(xx, beta * xx, 'c-', lw=2, label=f'a = {beta:.2f} (area-1)')
    r = np.corrcoef(x, t['a'])[0, 1]
    ax[0].set_xlabel('saturated-core area / star-visit mean − 1'); ax[0].set_ylabel('a')
    ax[0].set_title(f'all exposures: r = {r:.2f}'); ax[0].legend()
    # binned by brightness (mean n)
    gn = np.array([t['n'][t['grp'] == g].mean() for g in t['grp']])
    q = np.quantile(gn, [0, 0.25, 0.5, 0.75, 1])
    rs = []
    for lo, hi in zip(q[:-1], q[1:]):
        m = (gn >= lo) & (gn <= hi)
        rs.append((np.sqrt(np.median(gn[m]) / np.pi), np.corrcoef(x[m], t['a'][m])[0, 1],
                   np.sum(w[m] * x[m] * t['a'][m]) / np.sum(w[m] * x[m] ** 2)))
    rs = np.array(rs)
    ax[1].plot(rs[:, 0], rs[:, 1], 'ko-', label='correlation')
    ax[1].plot(rs[:, 0], rs[:, 2], 'cs-', label='slope β')
    ax[1].set_xlabel('saturated-core r$_{eq}$ [px] (brightness quartile)'); ax[1].legend()
    ax[1].set_title('core area as a predictor, by brightness')
    fig.tight_layout(); fig.savefig(f'{out}/fig5_area.png', dpi=DPI); plt.close(fig)
    return dict(beta=float(beta), r=float(r), by_quartile=rs.tolist())


def self_constrained(t):
    """Inner halo predicts outer?  And split-half azimuth for the noise ceiling."""
    rc = t['rc']
    d, s = t['delta'].astype(float), t['sigma'].astype(float)
    res = {}
    zones = dict(inner=(15, 40), mid=(40, 80), outer=(80, 150))
    Z = {}
    for k, (lo, hi) in zones.items():
        rr = (rc >= lo) & (rc < hi)
        v, ve = wm(d[:, rr, :].reshape(len(d), -1), s[:, rr, :].reshape(len(d), -1), axis=1)
        Z[k] = (recentre(v, ve, t['grp']), ve)
    for k in ('mid', 'outer'):
        x, xe = Z['inner']; y, ye = Z[k]
        m = np.isfinite(x) & np.isfinite(y)
        w = 1 / (ye[m] ** 2 + S_INT ** 2)
        slope = np.sum(w * x[m] * y[m]) / np.sum(w * x[m] ** 2)
        resid = y[m] - slope * x[m]
        var_y = np.mean(y[m] ** 2) - np.mean(ye[m] ** 2)
        res[f'inner_to_{k}'] = dict(n=int(m.sum()), slope=float(slope), r=float(np.corrcoef(x[m], y[m])[0, 1]),
                                    explained=float(1 - (np.mean(resid ** 2) - np.mean(ye[m] ** 2) - slope ** 2 * np.mean(xe[m] ** 2)) / var_y),
                                    rms_y=float(np.sqrt(np.mean(y[m] ** 2))), rms_resid=float(np.sqrt(np.mean(resid ** 2))),
                                    rms_noise_y=float(np.sqrt(np.mean(ye[m] ** 2))))
    # split-half azimuth at r = 15-80
    rr = (rc >= 15) & (rc < 80)
    ev, eve = wm(d[:, rr, 0::2].reshape(len(d), -1), s[:, rr, 0::2].reshape(len(d), -1), axis=1)
    od, ode = wm(d[:, rr, 1::2].reshape(len(d), -1), s[:, rr, 1::2].reshape(len(d), -1), axis=1)
    ev, od = recentre(ev, eve, t['grp']), recentre(od, ode, t['grp'])
    m = np.isfinite(ev) & np.isfinite(od)
    res['split_half'] = dict(n=int(m.sum()), r=float(np.corrcoef(ev[m], od[m])[0, 1]),
                             rms_diff_over_2=float(np.std(ev[m] - od[m]) / 2),
                             rms_noise_expected=float(np.sqrt(np.mean(eve[m] ** 2 + ode[m] ** 2)) / 2))
    return res, Z, (ev, od)


def radial(t):
    P, Pe = t['prof'].astype(float), t['profe'].astype(float)
    good = np.all(np.isfinite(P[:, :8]), 1)
    rc = t['rc']
    k = 8
    Xall = np.array([recentre(P[:, j], np.where(np.isfinite(Pe[:, j]), Pe[:, j], 1), t['grp']) for j in range(k)]).T
    X = Xall[good]
    # pairwise-complete correlation: the outer annuli are measurable for few stars
    C = np.array([[np.corrcoef(Xall[(f := np.isfinite(Xall[:, i]) & np.isfinite(Xall[:, j])), i], Xall[f, j])[0, 1]
                   for j in range(k)] for i in range(k)])
    npair = np.array([[int((np.isfinite(Xall[:, i]) & np.isfinite(Xall[:, j])).sum()) for j in range(k)] for i in range(k)])
    u, sv, vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    frac = sv ** 2 / np.sum(sv ** 2)
    return dict(rc=rc[:k].tolist(), corr=C.tolist(), corr_n=npair.tolist(), pca_frac=frac.tolist(), pca_modes=vt[:3].tolist(), n=int(good.sum()),
                rms_by_radius=np.sqrt(np.mean(X ** 2, 0)).tolist(),
                noise_by_radius=np.sqrt(np.nanmean(Pe[good][:, :k] ** 2, 0)).tolist())


def fig_shape(t, sc, Z, split, rad, out):
    fig, ax = plt.subplots(1, 4, figsize=(23, 5.4))
    x = Z['inner'][0]
    for k, c in (('mid', 'C0'), ('outer', 'C3')):
        y = Z[k][0]
        m = np.isfinite(x) & np.isfinite(y)
        ax[0].scatter(x[m], y[m], s=2, alpha=0.25, color=c,
                      label=f'{k}: slope {sc[f"inner_to_{k}"]["slope"]:.2f}, r={sc[f"inner_to_{k}"]["r"]:.2f}, '
                            f'explained {sc[f"inner_to_{k}"]["explained"]:.0%}')
    ax[0].plot([-0.4, 0.4], [-0.4, 0.4], 'k:', lw=0.8)
    ax[0].set_xlim(-0.4, 0.4); ax[0].set_ylim(-0.6, 0.6)
    ax[0].set_xlabel('inner halo change, r = 15-40 px'); ax[0].set_ylabel('outer halo change (40-80 / 80-150 px)')
    ax[0].set_title("self-constrained: does the star's own inner halo\npredict its outer halo, exposure by exposure?")
    ax[0].legend(fontsize=8, markerscale=4)
    ev, od = split
    m = np.isfinite(ev) & np.isfinite(od)
    ax[1].hexbin(ev[m], od[m], gridsize=60, extent=[-0.4, 0.4, -0.4, 0.4], mincnt=1, bins='log', cmap='magma_r')
    ax[1].plot([-0.4, 0.4], [-0.4, 0.4], 'c:', lw=1)
    ax[1].set_xlabel('a from even azimuth sectors'); ax[1].set_ylabel('a from odd sectors')
    ax[1].set_title(f'split-half azimuth (r=15-80): r = {sc["split_half"]["r"]:.2f}\n'
                    f'half-diff rms {sc["split_half"]["rms_diff_over_2"]:.3f} vs noise {sc["split_half"]["rms_noise_expected"]:.3f}')
    rc = np.array(rad['rc'])
    im = ax[2].imshow(np.array(rad['corr']), cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')
    ax[2].set_xticks(range(len(rc))); ax[2].set_xticklabels([f'{v:.0f}' for v in rc], fontsize=8)
    ax[2].set_yticks(range(len(rc))); ax[2].set_yticklabels([f'{v:.0f}' for v in rc], fontsize=8)
    ax[2].set_xlabel('annulus centre [px]'); ax[2].set_title(f'correlation of the change between radii\n(pairwise; N = {min(map(min, rad["corr_n"]))}-{max(map(max, rad["corr_n"]))})')
    fig.colorbar(im, ax=ax[2])
    for i, v in enumerate(rad['pca_modes']):
        ax[3].plot(rc, v, 'o-', label=f'mode {i + 1}: {rad["pca_frac"][i]:.0%} of variance')
    ax[3].plot(rc, rad['rms_by_radius'], 'k--', label='rms change')
    ax[3].plot(rc, rad['noise_by_radius'], 'k:', label='rms noise')
    ax[3].axhline(0, color='k', lw=0.5); ax[3].set_xscale('log'); ax[3].set_xlabel('radius [px]')
    ax[3].set_title(f'radial shape of the change (PCA of the azimuthal mean,\n{rad["n"]} exposures measured at all 8 radii)'); ax[3].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f'{out}/fig6_shape.png', dpi=DPI); plt.close(fig)


def fig_dipole(t, out):
    m = np.isfinite(t['cx']) & (t['dipe'] < 0.05)
    fig, ax = plt.subplots(2, 6, figsize=(22, 7.6))
    e = np.linspace(0, 2048, 9)
    res = {}
    for i, d in enumerate(DETS):
        for k in range(1, 7):
            mm = m & (t['det'] == d) & (t['expnum'] == k)
            cx = recentre(t['cx'], t['dipe'], t['grp'])[mm]
            cy = recentre(t['cy'], t['dipe'], t['grp'])[mm]
            sx, *_ = stats.binned_statistic_2d(t['xc'][mm], t['yc'][mm], cx, 'median', bins=[e, e])
            sy, *_ = stats.binned_statistic_2d(t['xc'][mm], t['yc'][mm], cy, 'median', bins=[e, e])
            cnt, *_ = stats.binned_statistic_2d(t['xc'][mm], t['yc'][mm], cx, 'count', bins=[e, e])
            sx[cnt < 4] = np.nan; sy[cnt < 4] = np.nan
            g = 0.5 * (e[1:] + e[:-1])
            X, Y = np.meshgrid(g, g, indexing='ij')
            a = ax[i, k - 1]
            a.quiver(X, Y, sx, sy, np.hypot(sx, sy), angles='xy', scale_units='xy', scale=0.1 / 256, cmap='plasma', clim=(0, 0.1))
            a.set_xlim(0, 2048); a.set_ylim(0, 2048); a.set_aspect('equal')
            a.set_title(f'{d[3:]} e{k}: median ({np.median(cx):+.3f},{np.median(cy):+.3f})', fontsize=9)
            res[f'{d}_e{k}'] = [float(np.median(cx)), float(np.median(cy)), int(mm.sum())]
    fig.suptitle(r'dipole of the change, $\delta(\theta) = c_0 + c_x\cos\theta + c_y\sin\theta$ at r = 20-65 px '
                 r'(arrow: 256 px = 0.10, $\theta$ in detector x,y)')
    fig.tight_layout(); fig.savefig(f'{out}/fig7_dipole.png', dpi=DPI); plt.close(fig)
    return res


def fig_models(mods, sc, out):
    names = list(mods['models'])
    ex = [mods['models'][n]['explained'] for n in names]
    names += ['self: inner→mid', 'self: inner→outer']
    ex += [sc['inner_to_mid']['explained'], sc['inner_to_outer']['explained']]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    col = ['C0' if not n.startswith('self') else 'C3' for n in names]
    ax.barh(range(len(names)), np.array(ex) * 100, color=col)
    for i, v in enumerate(ex):
        ax.text(v * 100 + (1 if v >= 0 else -1), i, f'{v:+.0%}', va='center', ha='left' if v >= 0 else 'right', fontsize=9)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.axvline(0, color='k', lw=0.5); ax.invert_yaxis()
    ax.set_xlabel('% of the (noise-subtracted) variance of the halo change explained, cross-validated by star')
    ax.set_title(f'what predicts the per-exposure halo change?  rms a = {mods["rms_a"]:.3f}, noise {mods["rms_noise"]:.3f}, '
                 f'{mods["n_exposures"]} exposures / {mods["n_stars"]} stars')
    fig.tight_layout(); fig.savefig(f'{out}/fig8_models.png', dpi=DPI); plt.close(fig)


def main():
    fin, fmod, out = sys.argv[1:4]
    smax = float(sys.argv[4]) if len(sys.argv) > 4 else 0.02
    epsfdir = sys.argv[5] if len(sys.argv) > 5 else None
    os.makedirs(out, exist_ok=True)
    t = load(fin, smax)
    mods = json.load(open(fmod))
    rng = np.random.default_rng(7)
    res = dict(n=int(len(t['a'])))
    fig_gallery(t, out)
    res['dither'] = fig_dither(t, mods, out)
    fig_position(t, out)
    res['xprofile'] = fig_xprofile(t, mods, out, epsfdir)
    ps, IJ = pair_stats(t, rng)
    res['pairs'] = ps
    fig_pairs(t, ps, IJ, out)
    res['area'] = fig_area(t, out)
    sc, Z, split = self_constrained(t)
    res['self_constrained'] = sc
    res['radial'] = radial(t)
    fig_shape(t, sc, Z, split, res['radial'], out)
    res['dipole'] = fig_dipole(t, out)
    fig_models(mods, sc, out)
    res['floor'] = floor(t)
    fig_floor(res['floor'], out)
    json.dump(res, open(f'{out}/tests.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k in ('n', 'area', 'self_constrained', 'floor')}, indent=1))


if __name__ == '__main__':
    main()
