"""Build position-dependent ePSFs for one detector, cross-validate the spatial grid,
compare with STPSF, and map the PSF shape across the detector.

    OMP_NUM_THREADS=1 python analyze_detector.py --detector nrcblong --scratch $SCRATCH/epsfmap \
        --stpsf $SCRATCH/epsfmap/stpsf_NRCB5_F480M.npz --out $SCRATCH/epsfmap/result_nrcblong.npz

Steps (see README.md for the reasoning):
 1. global WING ePSF from the brightest stars (stamp +/-Rw); its outer-annulus level is
    pinned to STPSF's (the only STPSF input -- it fixes the additive-constant
    degeneracy of any ePSF with a free per-star background);
 2. global CORE ePSF (+/-12 px), outer annulus pinned to the wing ePSF;
 3. G x G piecewise-constant spatial ePSF grids, G = 1..8, each cell initialised from
    the global ePSF; held-out (20% of the FRAMES) chi^2 and residual rms vs radius with
    (a) the cell ePSF, (b) bilinear interpolation between cell centres;
 4. the final grid (chosen by the held-out core chi^2) from all stars;
 5. per star: refit with the smooth model plus three shape modes (size, e1, e2) and a
    wing-excess ratio; binned into maps with split-half (even/odd frame) noise;
 6. the same metrics for the STPSF models on their grid;
 7. per-star measurement table (x, y, flux, background, RA/Dec from the GWCS, ...).
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import catalog  # noqa: E402
import epsf  # noqa: E402

LW = dict(O=4, fwhm0=2.6, r_norm=10.0, r_shape=6.0, r_core_chi=3.0, sigw=1.3)
SW = dict(O=4, fwhm0=2.2, r_norm=10.0, r_shape=6.0, r_core_chi=3.0, sigw=1.1)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def bilinear_weights(x, y, G, size=2048):
    """Weights of the G x G cell-centre nodes for positions (x, y): list over nodes."""
    cx = (x / size) * G - 0.5
    cy = (y / size) * G - 0.5
    cx = np.clip(cx, 0, G - 1); cy = np.clip(cy, 0, G - 1)
    i0 = np.minimum(np.floor(cx).astype(int), max(G - 2, 0)); j0 = np.minimum(np.floor(cy).astype(int), max(G - 2, 0))
    tx = cx - i0; ty = cy - j0
    W = np.zeros((G * G, len(x)))
    if G == 1:
        W[0] = 1
        return W
    for dj, wy in ((0, 1 - ty), (1, ty)):
        for di, wx in ((0, 1 - tx), (1, tx)):
            np.add.at(W, ((j0 + dj) * G + (i0 + di), np.arange(len(x))), wx * wy)
    return W


def cell_index(x, y, G, size=2048):
    i = np.clip((x / size * G).astype(int), 0, G - 1)
    j = np.clip((y / size * G).astype(int), 0, G - 1)
    return j * G + i


def build_grid(stars, grid, idx, P_global, G, par, edge_level, niter=4):
    cells = cell_index(stars.x[idx], stars.y[idx], G)
    Ps, nst = [], []
    for k in range(G * G):
        ii = idx[cells == k]
        if len(ii) < 50:
            Ps.append(P_global.copy()); nst.append(len(ii))
            continue
        P, _ = epsf.build(stars, grid, ii, P_global, r_norm=par['r_norm'], edge_level=edge_level, niter=niter)
        Ps.append(P); nst.append(len(ii))
    return np.array(Ps), np.array(nst)


def spatial_model(Ps, grid, G, stars, bilinear):
    eps = [epsf.EPSF(P, grid) for P in Ps]
    if G == 1:
        return eps[0]
    if bilinear:
        W = bilinear_weights(stars.x, stars.y, G)
    else:
        c = cell_index(stars.x, stars.y, G)
        W = np.array([(c == k).astype(float) for k in range(G * G)])
    return [(e, W[k]) for k, e in enumerate(eps)]


def residual_stats(stars, model, idx, rbins, trim=0.02):
    """Held-out metrics per radial bin (per pixel): robust residual rms (1.4826 MAD, in
    units of the star's flux within r_norm), the photon+read noise (ERR) and ERR +
    confusion rms in the same units, and chi^2 w.r.t. ERR alone (mean of (res/ERR)^2
    after trimming the top ``trim`` fraction -- failed fits and unmasked neighbours).
    ``core_metric``: the trimmed mean (res/ERR)^2 over r <= 3 px, the CV criterion."""
    epsf.fit_stars(stars, model, idx)
    ok = idx[stars.ok[idx]]
    nb = len(rbins) - 1
    rbin = np.digitize(stars.pr, rbins) - 1
    R_, E_, C_, Z_ = [[] for _ in range(nb)], [[] for _ in range(nb)], [[] for _ in range(nb)], [[] for _ in range(nb)]
    mean_res = np.zeros(stars.px.shape); ncount = np.zeros(stars.px.shape)
    ch = 3000
    for c0 in range(0, len(ok), ch):
        ii = ok[c0:c0 + ch]
        ux = stars.px[None] - stars.dx[ii][:, None, None]
        uy = stars.py[None] - stars.dy[ii][:, None, None]
        P, _, _ = epsf.model_and_grad(model, ux, uy, ii, grad=False)
        f = stars.f[ii][:, None, None]
        res = (stars.s[ii].astype(float) - f * P - stars.b[ii][:, None, None]) / f
        e = stars.e[ii].astype(float) / f
        good = ~stars.m[ii]
        var = e ** 2 + (stars.conf[ii][:, None, None] / f) ** 2
        mean_res += np.where(good, res, 0).sum(0); ncount += good.sum(0)
        for k in range(nb):
            sel = good & (rbin == k)[None]
            R_[k].append(res[sel]); E_[k].append(e[sel] ** 2); C_[k].append(var[sel]); Z_[k].append((res[sel] / e[sel]) ** 2)
    out = dict(rms=np.zeros(nb), phot=np.zeros(nb), noise_tot=np.zeros(nb), chi2_err=np.zeros(nb))
    core_num = core_den = 0.0
    for k in range(nb):
        rr = np.concatenate(R_[k]); z = np.sort(np.concatenate(Z_[k]))
        if len(rr) == 0:
            out['rms'][k] = out['phot'][k] = out['noise_tot'][k] = out['chi2_err'][k] = np.nan
            continue
        out['rms'][k] = 1.4826 * np.median(np.abs(rr - np.median(rr)))
        out['phot'][k] = np.sqrt(np.median(np.concatenate(E_[k])))
        out['noise_tot'][k] = np.sqrt(np.median(np.concatenate(C_[k])))
        zt = z[:max(1, int(len(z) * (1 - trim)))]
        out['chi2_err'][k] = zt.mean()
        if rbins[k + 1] <= 3.0 + 1e-9:
            core_num += zt.sum(); core_den += len(zt)
    out['core_metric'] = core_num / max(core_den, 1)
    out['core_chi2'] = float(np.median(stars.chi2[ok]))
    out['mean_res'] = mean_res / np.maximum(ncount, 1)
    out['nstar'] = len(ok)
    return out


def shape_fit(stars, model, idx, r_shape):
    """Per star: f, b and three shape modes (size, e1, e2) at fixed position, plus the
    fractional wing excess over the model in 3 < r < 10 px."""
    out = np.full((len(stars), 5), np.nan)
    inreg = stars.pr <= r_shape
    wreg = (stars.pr > 3) & (stars.pr < 10)
    ch = 2000
    for c0 in range(0, len(idx), ch):
        ii = idx[c0:c0 + ch]
        ux = stars.px[None] - stars.dx[ii][:, None, None]
        uy = stars.py[None] - stars.dy[ii][:, None, None]
        P, gx, gy = epsf.model_and_grad(model, ux, uy, ii)
        Bs = -(2 * P + ux * gx + uy * gy)
        B1 = -(ux * gx - uy * gy)
        B2 = -(uy * gx + ux * gy)
        d = stars.s[ii].astype(float)
        var = stars.e[ii].astype(float) ** 2 + stars.conf[ii][:, None, None] ** 2 + (0.02 * np.maximum(d, 0)) ** 2
        w = np.where(~stars.m[ii] & inreg[None], 1 / var, 0)
        f = stars.f[ii][:, None, None]
        J = np.stack([P, np.ones_like(P), f * Bs, f * B1, f * B2], axis=1)
        Jw = J * w[:, None]
        A = np.einsum('nkij,nlij->nkl', Jw, J)
        B = np.einsum('nkij,nij->nk', Jw, d)
        sol = epsf.safe_solve(A, B)
        # sol[:,0] = f, sol[:,1] = b, sol[:,2:] = a_s, a1, a2 (with f fixed in the basis scale)
        good = ~stars.m[ii] & wreg[None]
        num = np.where(good, d - stars.b[ii][:, None, None], 0).sum((1, 2)) / stars.f[ii]
        den = np.where(good, P, 0).sum((1, 2))
        out[ii, 0] = sol[:, 2]
        out[ii, 1] = 2 * sol[:, 3]
        out[ii, 2] = 2 * sol[:, 4]
        out[ii, 3] = num / den - 1
        out[ii, 4] = sol[:, 0]
    return out


def bin_map(x, y, v, nb, size=2048, stat='mean'):
    i = np.clip((x / size * nb).astype(int), 0, nb - 1)
    j = np.clip((y / size * nb).astype(int), 0, nb - 1)
    k = j * nb + i
    ok = np.isfinite(v)
    M = np.full(nb * nb, np.nan); N = np.zeros(nb * nb)
    for c in np.unique(k[ok]):
        vv = v[ok & (k == c)]
        N[c] = len(vv)
        if len(vv) >= 5:
            # 3-sigma clipped mean
            m, s = np.median(vv), 1.4826 * np.median(np.abs(vv - np.median(vv)))
            M[c] = vv[np.abs(vv - m) < 4 * max(s, 1e-12)].mean()
    return M.reshape(nb, nb), N.reshape(nb, nb)


def poly_smooth(M, order=2):
    """Least-squares 2-D polynomial surface through a map (NaNs ignored)."""
    nb = M.shape[0]
    yy, xx = np.mgrid[:nb, :nb] / (nb - 1) - 0.5
    terms = [xx ** i * yy ** j for i in range(order + 1) for j in range(order + 1 - i)]
    A = np.array([t.ravel() for t in terms]).T
    ok = np.isfinite(M.ravel())
    c, *_ = np.linalg.lstsq(A[ok], M.ravel()[ok], rcond=None)
    return (A @ c).reshape(nb, nb)


def epsf_metrics(P, grid, par, Pref=None):
    ee_r = np.array([1, 2, 3, 5, 8, 10])
    ee = epsf.encircled(P, grid, ee_r)
    fw = epsf.fwhm_halfmax(P, grid)
    sz, e1, e2, e, th = epsf.moments(P, grid, par['sigw'])
    wing = ee[5] - ee[2]          # fraction of the r<=10 flux beyond 3 px
    core = ee[1]
    return dict(fwhm=fw, size=sz, e1=e1, e2=e2, e=e, theta=th, ee=ee, ee_r=ee_r, wing_core=wing / core,
                peak=P.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', required=True)
    ap.add_argument('--scratch', required=True)
    ap.add_argument('--stpsf', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--grids', default='1,2,3,4,6,8')
    ap.add_argument('--niter', type=int, default=8)
    a = ap.parse_args()
    np.seterr(all='ignore')
    det = a.detector
    par = LW if 'long' in det else SW
    O = par['O']
    stampdir = os.path.join(a.scratch, 'stamps', det)

    st = np.load(a.stpsf)
    od = st['overdist'].astype(np.float64); spos = st['pos']
    ns = int(round(np.sqrt(len(spos))))
    ic = np.argmin(np.hypot(spos[:, 0] - 1024, spos[:, 1] - 1024))

    # ---------------- wing ePSF
    log('loading wing stars')
    ws, wtab, _ = catalog.load(stampdir, wing=True)
    Rw = ws.R
    gw = epsf.Grid(Rw, O)
    Sw = [epsf.normalise(epsf.stpsf_to_epsf(o, O, Rw), gw, par['r_norm']) for o in od]
    ann_w = (gw.ur > Rw - 1.5) & (gw.ur <= Rw)
    edge_w = Sw[ic][ann_w].mean()
    log(f'wing stars: {len(ws)}  R={Rw}  STPSF edge level {edge_w:.3g}')
    Pw, hw = epsf.build(ws, gw, np.arange(len(ws)), epsf.gaussian_epsf(gw, par['fwhm0']),
                        r_norm=par['r_norm'], edge_level=edge_w, niter=a.niter, verbose=True)
    # split-half wing ePSFs (even / odd frames) for the noise
    even = np.nonzero(wtab['frame'] % 2 == 0)[0]; odd = np.nonzero(wtab['frame'] % 2 == 1)[0]
    Pw_a, _ = epsf.build(ws, gw, even, Pw, r_norm=par['r_norm'], edge_level=edge_w, niter=3)
    Pw_b, _ = epsf.build(ws, gw, odd, Pw, r_norm=par['r_norm'], edge_level=edge_w, niter=3)
    wing_flux = ws.f.copy(); wing_b = ws.b.copy()
    # per-frame wing excess in the outer annulus (issue #993 link)
    ep_w = epsf.EPSF(Pw, gw)
    epsf.fit_stars(ws, ep_w, np.arange(len(ws)))
    ann_rs = [(2, 4), (4, 8), (8, 15), (15, Rw - 1)]
    wnum = np.full((len(ws), len(ann_rs)), np.nan); wden = np.full((len(ws), len(ann_rs)), np.nan)
    for c0 in range(0, len(ws), 500):
        ii = np.arange(c0, min(c0 + 500, len(ws)))
        ux = ws.px[None] - ws.dx[ii][:, None, None]; uy = ws.py[None] - ws.dy[ii][:, None, None]
        P = ep_w(ux, uy)
        d = (ws.s[ii].astype(float) - ws.b[ii][:, None, None]) / ws.f[ii][:, None, None]
        for k, (r0, r1) in enumerate(ann_rs):
            g_ = ~ws.m[ii] & ((ws.pr > r0) & (ws.pr <= r1))[None]
            wnum[ii, k] = np.where(g_, d - P, 0).sum((1, 2))
            wden[ii, k] = np.where(g_, P, 0).sum((1, 2))
    # confusion/selection test: wing ePSF from the brightest vs the faintest third
    amp = wtab['amp']
    lo, hi = np.percentile(amp, [33.3, 66.7])
    Pw_faint, _ = epsf.build(ws, gw, np.nonzero(amp <= lo)[0], Pw, r_norm=par['r_norm'], edge_level=edge_w, niter=3)
    Pw_bright, _ = epsf.build(ws, gw, np.nonzero(amp >= hi)[0], Pw, r_norm=par['r_norm'], edge_level=edge_w, niter=3)
    wing_res = dict(Pw=Pw, Pw_a=Pw_a, Pw_b=Pw_b, Sw=np.array(Sw), hist_w=hw, wnum=wnum, wden=wden, ann_rs=np.array(ann_rs),
                    Pw_faint=Pw_faint, Pw_bright=Pw_bright, amp_split=np.array([lo, hi]),
                    w_frame=wtab['frame'], w_expstart=wtab['expstart'], w_x=ws.x, w_y=ws.y, w_f=ws.f,
                    w_amp=wtab['amp'], w_fname=wtab['fname'])
    edge_c_ann = (gw.ur > 12 - 1.5) & (gw.ur <= 12)
    edge_c_wing = Pw[edge_c_ann].mean()
    edge_c = Sw[ic][edge_c_ann].mean()     # core ePSF pinned to STPSF at 10.5-12 px as well
    del ws

    # ---------------- core stars
    log('loading core stars')
    cs, tab, frames = catalog.load(stampdir)
    R = cs.R
    g = epsf.Grid(R, O)
    Sc = np.array([epsf.normalise(epsf.stpsf_to_epsf(o, O, R), g, par['r_norm']) for o in od])
    log(f'core stars: {len(cs)} in {len(frames)} frames; core edge level (STPSF) {edge_c:.3g}, wing ePSF there {edge_c_wing:.3g}')
    allidx = np.arange(len(cs))
    P1, h1 = epsf.build(cs, g, allidx, epsf.gaussian_epsf(g, par['fwhm0']), r_norm=par['r_norm'],
                        edge_level=edge_c, niter=a.niter, verbose=True)

    # ---------------- cross-validation of the spatial grid
    test = np.nonzero(tab['frame'] % 5 == 2)[0]
    train = np.nonzero(tab['frame'] % 5 != 2)[0]
    rbins = np.array([0, 0.5, 1.2, 1.5, 2.1, 3.01, 4, 5, 6, 8, 10, 12.01])
    cv = {}
    P1t, _ = epsf.build(cs, g, train, P1, r_norm=par['r_norm'], edge_level=edge_c, niter=3)
    grids = [int(x) for x in a.grids.split(',')]
    for G in grids:
        if G == 1:
            Ps = P1t[None]; nst = np.array([len(train)])
        else:
            Ps, nst = build_grid(cs, g, train, P1t, G, par, edge_c)
        for bil in ([False, True] if G > 1 else [False]):
            m = spatial_model(Ps, g, G, cs, bil)
            rs = residual_stats(cs, m, test, rbins)
            cv[(G, bil)] = rs
            log(f'CV G={G} bilinear={bil}: core metric <(res/ERR)^2>_r<=3 = {rs["core_metric"]:.4f}  '
                f'median chi2(fit weights)={rs["core_chi2"]:.4f} chi2/ERR by r: {rs["chi2_err"].round(2)} nstar/cell~{np.median(nst):.0f}')
    # choose: minimum held-out core chi^2 (r <= 3 px, w.r.t. photon noise)
    best = min(cv, key=lambda k: cv[k]['core_metric'])
    Gb = best[0]
    log(f'best grid: {best}')

    # ---------------- final grid from all stars
    if Gb == 1:
        PG = P1[None]; nG = np.array([len(cs)])
    else:
        PG, nG = build_grid(cs, g, allidx, P1, Gb, par, edge_c)
    # split halves of the final grid (noise of the cell ePSFs)
    ev = np.nonzero(tab['frame'] % 2 == 0)[0]; od_ = np.nonzero(tab['frame'] % 2 == 1)[0]
    PGa, _ = build_grid(cs, g, ev, P1, Gb, par, edge_c, niter=3) if Gb > 1 else (epsf.build(cs, g, ev, P1, par['r_norm'], edge_c, niter=3)[0][None], None)
    PGb, _ = build_grid(cs, g, od_, P1, Gb, par, edge_c, niter=3) if Gb > 1 else (epsf.build(cs, g, od_, P1, par['r_norm'], edge_c, niter=3)[0][None], None)
    # a fixed 4x4 grid always, for the figures (and its split halves)
    P4, n4 = build_grid(cs, g, allidx, P1, 4, par, edge_c) if Gb != 4 else (PG, nG)

    # ---------------- per-star shape parameters relative to the global ePSF
    ep1 = epsf.EPSF(P1, g)
    epsf.fit_stars(cs, ep1, allidx)
    sh_global = shape_fit(cs, ep1, allidx, par['r_shape'])
    # the same relative to the smooth spatial model (what is left after the grid)
    mG = spatial_model(PG, g, Gb, cs, True)
    epsf.fit_stars(cs, mG, allidx)
    fit_x, fit_y, fit_f, fit_b, fit_chi2, fit_ok = cs.x.copy(), cs.y.copy(), cs.f.copy(), cs.b.copy(), cs.chi2.copy(), cs.ok.copy()
    sh_grid = shape_fit(cs, mG, allidx, par['r_shape'])

    # ---------------- STPSF field dependence, same estimators
    st_metrics = [epsf_metrics(P, g, par) for P in Sc]
    # STPSF models as noiseless "stars" fitted with the central STPSF: shape modes
    fake = epsf.Stars(np.zeros((len(Sc), 2 * R + 1, 2 * R + 1)), np.ones((len(Sc), 2 * R + 1, 2 * R + 1)) * 1e-3,
                      np.zeros((len(Sc), 2 * R + 1, 2 * R + 1), bool), np.zeros(len(Sc)), np.zeros(len(Sc)),
                      np.zeros(len(Sc)), np.zeros(len(Sc)), R)
    epC = epsf.EPSF(Sc[ic], g)
    img = []
    for P in Sc:
        img.append(epsf.EPSF(P, g)(cs.px, cs.py))
    img = np.array(img); pk = img.max(axis=(1, 2))
    fake.s = (img / pk[:, None, None]).astype(np.float16)
    fake.e = (np.sqrt(img / pk[:, None, None] / 250 ** 2 + 1e-6)).astype(np.float16)
    fake.conf = np.zeros(len(Sc))
    epsf.fit_stars(fake, epC, np.arange(len(Sc)))
    sh_stpsf = shape_fit(fake, epC, np.arange(len(Sc)), par['r_shape'])
    # data grid metrics
    metG = [epsf_metrics(P, g, par) for P in PG]
    met4 = [epsf_metrics(P, g, par) for P in P4]
    met1 = epsf_metrics(P1, g, par)
    metGa = [epsf_metrics(P, g, par) for P in PGa]
    metGb = [epsf_metrics(P, g, par) for P in PGb]

    # ---------------- sky positions of the refined centroids (linear GWCS Jacobian)
    dxr = fit_x - tab['x0']; dyr = fit_y - tab['y0']
    jac = tab['jac']
    ra = tab['ra0'] + (jac[:, 0] * dxr + jac[:, 1] * dyr) / np.cos(np.deg2rad(tab['dec0']))
    dec = tab['dec0'] + jac[:, 2] * dxr + jac[:, 3] * dyr
    # flux in the frame's units (MJy/sr summed over pixels within r_norm) and background
    flux = fit_f * tab['amp']
    bkg = tab['bkg0'] + fit_b * tab['amp']

    table = dict(fname=tab['fname'], detector=tab['detector'], filter=tab['filter'], expstart=tab['expstart'],
                 obs=tab['obs'], expnum=tab['expnum'], x=fit_x, y=fit_y, x_peak_int=tab['ix'], y_peak_int=tab['iy'],
                 flux=flux, flux_dn_s=flux / tab['photmjsr'], bkg=bkg, bkg_coarse=tab['bkg0'], ra=ra, dec=dec,
                 peak_snr=tab['snr'], satfrac=tab['satfrac'], maskfrac=tab['maskfrac'], chi2=fit_chi2, ok=fit_ok,
                 dsize_global=sh_global[:, 0], de1_global=sh_global[:, 1], de2_global=sh_global[:, 2],
                 dwing_global=sh_global[:, 3], dsize_grid=sh_grid[:, 0], de1_grid=sh_grid[:, 1],
                 de2_grid=sh_grid[:, 2], dwing_grid=sh_grid[:, 3], conf=cs.conf, frame=tab['frame'])
    np.savez_compressed(a.out.replace('.npz', '_stars.npz'), **table)

    cvsave = {f'cv_{G}_{int(b)}_{k}': v for (G, b), d in cv.items() for k, v in d.items()}
    np.savez_compressed(
        a.out, P1=P1, PG=PG, nG=nG, PGa=PGa, PGb=PGb, P4=P4, n4=n4, Gbest=Gb, bilinear_best=best[1], Sc=Sc, spos=spos,
        R=R, Rw=Rw, O=O, r_norm=par['r_norm'], rbins=rbins, hist_core=h1, edge_c=edge_c, edge_c_wing=edge_c_wing, edge_w=edge_w,
        met1=met1, metG=metG, met4=met4, metGa=metGa, metGb=metGb, st_metrics=st_metrics, sh_stpsf=sh_stpsf,
        grids=np.array(grids), frames=np.array(frames), **cvsave, **wing_res)
    log('done', a.out)


if __name__ == '__main__':
    main()
