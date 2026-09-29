"""Is the per-exposure halo change a function of detector position?  (README §8)

Input: halocal_measure.py output (halo sector medians H[r, sector] for every
star x exposure).

1. Per star, the mean over its exposures H̄_s(r, sec) is removed: the scene is fixed on
   the sky and sampled identically in every dither (one roll), so ΔH = H - H̄_s is halo
   change + noise.
2. The star's halo amplitude A_s T(r, sec) is obtained from the rank-1 + offset
   factorisation H̄_s(r, sec) = B_s(sec) + A_s T(r, sec) over all stars of the detector
   (alternating least squares).  y = ΔH / (A_s T) is the fractional halo change.
3. Model, per detector and radial bin (azimuthal mean of the sector y's, or per sector):
       y_{s,e} = [α_e + g(x_{s,e}, y_{s,e})] - <α + g>_s + ε
   α_e : one term per exposure, common to every star in it
   g   : smooth function of the star's detector position, Legendre up to order LMAX
   <>_s: the same average over star s's exposures that defined H̄_s
4. Cross-validation by star (K folds): α and g are fitted without the held-out stars and
   predict their y.  That is exactly how a calibration would be applied to a star.

    python halocal_fit.py <meas.npz> <outprefix> [LMAX]
"""
import sys, json, warnings
warnings.simplefilter("ignore", RuntimeWarning)
import numpy as np
from numpy.polynomial import legendre as L

NFOLD = 5


def legendre_design(x, y, lmax):
    u = (np.asarray(x)-1024)/1024; v = (np.asarray(y)-1024)/1024
    cols = []; names = []
    for i in range(lmax+1):
        for j in range(lmax+1-i):
            if i == 0 and j == 0:
                continue
            ci = np.zeros(i+1); ci[i] = 1; cj = np.zeros(j+1); cj[j] = 1
            cols.append(L.legval(u, ci)*L.legval(v, cj)); names.append((i, j))
    return np.stack(cols, 1), names


def factorise(Hbar, niter=30):
    """Hbar [nstar, nr, nsec] (nan allowed) = B[s, sec] + A[s] T[r, sec]."""
    ns, nr, nsec = Hbar.shape
    ok = np.isfinite(Hbar)
    T = np.nanmedian(Hbar/np.nanmedian(Hbar[:, :3].reshape(ns, -1), 1)[:, None, None], 0)
    B = np.zeros((ns, nsec)); A = np.ones(ns)
    for _ in range(niter):
        # per star: A_s, B_s(sec) by least squares over (r, sec)
        for s in range(ns):
            m = ok[s]
            if m.sum() < 6:
                A[s] = np.nan; continue
            rows = np.nonzero(m)
            X = np.zeros((m.sum(), 1+nsec)); X[:, 0] = T[rows]
            X[np.arange(m.sum()), 1+rows[1]] = 1
            sol, *_ = np.linalg.lstsq(X, Hbar[s][m], rcond=None)
            A[s] = sol[0]; B[s] = sol[1:]
        good = np.isfinite(A) & (A > 0)
        # T(r, sec) given A, B
        R = (Hbar-B[:, None, :])/A[:, None, None]
        W = np.where(ok & good[:, None, None], A[:, None, None]**2, 0)
        T = np.nansum(np.where(W > 0, R*W, 0), 0)/np.maximum(W.sum(0), 1e-30)
        T /= np.nanmax(T[0])
    return A, B, T


def build_rows(z, det, rbin, per_sector=False):
    m = z['det'] == det
    sid = z['sid'][m]; H = z['H'][m]; He = z['He'][m]
    x = z['xc'][m]; y = z['yc'][m]; root = z['root'][m]
    ust, sidx = np.unique(sid, return_inverse=True)
    ns = len(ust)
    # star mean over its exposures (per r, sector), requiring >= 3 exposures
    nexp = np.bincount(sidx, minlength=ns)
    Hbar = np.full((ns,)+H.shape[1:], np.nan)
    for s in range(ns):
        ii = sidx == s
        if nexp[s] >= 3:
            Hbar[s] = np.nanmean(H[ii], 0)
    A, B, T = factorise(Hbar)
    amp = A[:, None, None]*T[None]             # star halo per (r, sec)
    Y = (H-Hbar[sidx])/amp[sidx]
    Ye = He/amp[sidx]
    return dict(sidx=sidx, nexp=nexp, A=A, T=T, Y=Y, Ye=Ye, x=x, y=y, root=root, sid=sid, n=z['n'][m])


def fit_predict(Yv, Yev, sidx, eidx, P, train, lam=1e-6, use_alpha=True, use_g=True):
    """Weighted LSQ of Yv = D θ, D = [exposure one-hot | P] demeaned within star (over
    that star's rows, all of which enter its mean).  Returns predictions for all rows."""
    ne = eidx.max()+1
    cols = []
    if use_alpha:
        E = np.zeros((len(Yv), ne)); E[np.arange(len(Yv)), eidx] = 1; cols.append(E)
    if use_g:
        cols.append(P)
    if not cols:
        return np.zeros_like(Yv), None
    D = np.concatenate(cols, 1)
    # demean within star (unweighted mean = the definition of Hbar)
    ns = sidx.max()+1
    cnt = np.bincount(sidx, minlength=ns)
    mean = np.zeros((ns, D.shape[1]))
    np.add.at(mean, sidx, D)
    D = D-mean[sidx]/cnt[sidx, None]
    w = 1/np.maximum(Yev, 1e-6)**2
    tr = train & np.isfinite(Yv)
    Dw = D[tr]*np.sqrt(w[tr])[:, None]
    G = Dw.T@Dw+lam*np.eye(D.shape[1])*np.trace(Dw.T@Dw)/D.shape[1]
    th = np.linalg.solve(G, Dw.T@(Yv[tr]*np.sqrt(w[tr])))
    return D@th, th


def main():
    fn, outp = sys.argv[1], sys.argv[2]
    lmax = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    z = np.load(fn)
    edges = z['edges']
    report = {}
    for det in np.unique(z['det']):
        R = build_rows(z, det, None)
        Y = R['Y']; Ye = R['Ye']
        _, eidx = np.unique(R['root'], return_inverse=True)
        P, names = legendre_design(R['x'], R['y'], lmax)
        rng = np.random.default_rng(1)
        fold = rng.integers(0, NFOLD, R['sidx'].max()+1)[R['sidx']]
        rep = []
        for ib in range(Y.shape[1]):
            yv = np.nanmean(Y[:, ib], 1)                   # azimuthal mean of the sectors
            ye = np.sqrt(np.nanmean(Ye[:, ib]**2, 1)/np.maximum(np.isfinite(Y[:, ib]).sum(1), 1))
            ok = np.isfinite(yv) & np.isfinite(ye) & (R['nexp'][R['sidx']] >= 3)
            if ok.sum() < 200:
                continue
            # empirical noise: sigma_y^2 = ye^2 + (tau/A)^2, tau (MJy/sr) = the dither-to-dither
            # sampling noise of a crowded-field median, fitted on the faintest half of the stars
            Aj = R['A'][R['sidx']]*np.nanmean(R['T'][ib])
            faint = ok & (Aj < np.nanmedian(Aj[ok]))
            tau2 = np.nanmedian(((yv**2-ye**2)*Aj**2)[faint])/0.4549      # median of chi2_1 = 0.4549
            ye = np.sqrt(ye**2+max(tau2, 0)/Aj**2)
            res = {}
            for name, ua, ug in [('none', 0, 0), ('alpha', 1, 0), ('g', 0, 1), ('alpha+g', 1, 1)]:
                pred = np.full(len(yv), np.nan)
                for f in range(NFOLD):
                    p, _ = fit_predict(np.where(ok, yv, np.nan), np.where(ok, ye, 1), R['sidx'], eidx, P,
                                       train=ok & (fold != f), use_alpha=ua, use_g=ug)
                    pred[fold == f] = p[fold == f]
                r = (yv-pred)[ok]; e = ye[ok]
                sig = (np.abs(yv[ok]) > 0) & (1/e > np.percentile(1/e, 75))       # the best-measured quarter
                res[name] = dict(chi2=float(np.mean((r/e)**2)), chi2_best=float(np.mean((r[sig]/e[sig])**2)),
                                 rms_best=float(np.sqrt(np.mean(r[sig]**2))))
            base = res['none']
            for k in res:
                res[k]['excess_removed_best'] = (base['chi2_best']-res[k]['chi2_best'])/max(base['chi2_best']-1, 1e-9)
            rep.append(dict(r=[int(edges[ib]), int(edges[ib+1])], n=int(ok.sum()), **res))
            print(f"{det} r={edges[ib]}-{edges[ib+1]} n={ok.sum()} tau={np.sqrt(max(tau2,0)):.2f}", ' | '.join(f"{k}: chi2={v['chi2']:.2f} best-quarter chi2={v['chi2_best']:.2f} rms={v['rms_best']:.4f} removed={v['excess_removed_best']:.0%}" for k, v in res.items()), flush=True)
        report[str(det)] = rep
    json.dump(report, open(outp+'_cv.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
