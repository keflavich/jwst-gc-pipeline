"""Rank-2 pattern model: static Q0 + variable template H with smooth per-exposure amplitude.

    d_e(p) = a_e * K_e*[Q0 + eta_e(p) H](q_e(p)+dq_e(p)) + c_e + g_e.(X,Y) + B_e h_e

eta_e(p): bicubic B-spline on the detector cutout (knot spacing ETA_SPACING px); eta of the
anchor exposure (e1) is fixed to 0, so Q0 = anchor pattern and H = the variable component.
"""
import numpy as np, time, sys, os
import scipy.sparse as sp
from scipy.sparse.linalg import cg, LinearOperator
from scipy import ndimage
import finufft
import patternfit as qfit3
from patternfit import (N, bm, KX, KY, POLY, to_F, from_G, proj, nufft2, nufft1, build, train_weight_maps)
from bandpsf import EPS, NTHREADS
from exposure import bkg_matrix

ETA_SPACING = int(os.environ.get('ETA_SPACING', 128))


def attach_eta(e):
    e.E = bkg_matrix(e.ex.n, ETA_SPACING, e.pix).tocsc()     # npix x neta (bicubic B-spline)
    e.eta = np.zeros(e.E.shape[1])
    return e


def eta_map(e):
    return e.E@e.eta


def solve_QH(qexps, lam, Q0, H0, maxiter=60, prox=0.03, nprox=2, verbose=True):
    etas = [eta_map(e) for e in qexps]
    cov0 = np.zeros(N*N); covH = np.zeros(N*N)
    for e, et in zip(qexps, etas):
        ix = np.clip(np.round(e.qx).astype(int)+N//2, 0, N-1); iy = np.clip(np.round(e.qy).astype(int)+N//2, 0, N-1)
        cov0 += np.bincount(iy*N+ix, weights=e.w*e.a**2, minlength=N*N)
        covH += np.bincount(iy*N+ix, weights=e.w*e.a**2*et**2, minlength=N*N)
    def fill(c):
        d2 = ndimage.uniform_filter(c.reshape(N, N), 3)
        d2 = np.where(d2 < 1e-6*np.median(d2[d2 > 0]), 0.0, d2)
        idx = ndimage.distance_transform_edt(d2 <= 0, return_distances=False, return_indices=True)
        return d2[idx[0], idx[1]].ravel()
    f0 = fill(cov0); fH = fill(covH)
    L0 = prox*f0+lam; LH = prox*fH+lam*1e-2
    D0 = ndimage.uniform_filter(f0.reshape(N, N), 5).ravel()+L0
    DH = ndimage.uniform_filter(fH.reshape(N, N), 5).ravel()+LH
    b0 = np.zeros((N, N), complex); bH = np.zeros((N, N), complex)
    for e, et in zip(qexps, etas):
        r = e.w*e.a*(e.d-e.add())
        b0 += nufft1(e.qx, e.qy, r)*e.mult(); bH += nufft1(e.qx, e.qy, r*et)*e.mult()
    b0 = from_G(b0); bH = from_G(bH)
    M = N*N
    def A(x):
        F0 = to_F(x[:M]); FH = to_F(x[M:])
        G0 = np.zeros((N, N), complex); GH = np.zeros((N, N), complex)
        for e, et in zip(qexps, etas):
            mu = e.mult()
            out = finufft.nufft2d2(2*np.pi*e.qy/N, 2*np.pi*e.qx/N, np.stack([F0*mu, FH*mu]), isign=1, eps=EPS, nthreads=NTHREADS).real
            v = e.a*(out[0]+et*out[1])
            wr = e.w*e.a*v
            g = finufft.nufft2d1(2*np.pi*e.qy/N, 2*np.pi*e.qx/N, np.stack([wr, wr*et]).astype(complex), (N, N), isign=-1, eps=EPS, nthreads=NTHREADS)
            G0 += g[0]*mu; GH += g[1]*mu
        return np.concatenate([from_G(G0)+proj(L0*proj(x[:M])), from_G(GH)+proj(LH*proj(x[M:]))])
    Pc = LinearOperator((2*M, 2*M), matvec=lambda v: np.concatenate([proj(proj(v[:M])/D0), proj(proj(v[M:])/DH)]))
    x = np.concatenate([proj(Q0.ravel()), proj(H0.ravel())])
    for ip in range(nprox):
        b = np.concatenate([b0+proj(L0*x[:M]), bH+proj(LH*x[M:])])
        it = [0]
        x1, info = cg(LinearOperator((2*M, 2*M), matvec=A), b, x0=x, rtol=1e-6, maxiter=maxiter, M=Pc,
                      callback=lambda xx: it.__setitem__(0, it[0]+1))
        if verbose:
            print(f'   prox {ip}: CG its={it[0]} |dQ0|/|Q0|={np.linalg.norm(x1[:M]-x[:M])/np.linalg.norm(x1[:M]):.2e} |dH|/|H|={np.linalg.norm(x1[M:]-x[M:])/max(np.linalg.norm(x1[M:]),1e-30):.2e}', flush=True)
        x = x1
    return x[:M].reshape(N, N), x[M:].reshape(N, N)


def fit_nuisance(e, Q0, H, anchor=False, niter=3):
    F = to_F(Q0); FHh = to_F(H)
    for _ in range(niter):
        et = eta_map(e)
        mu = e.mult()
        stack = np.stack([F*mu, F*mu*1j*KX, F*mu*1j*KY, -F*KX**2, -F*KY**2, -F*KX*KY,
                          FHh*mu, FHh*mu*1j*KX, FHh*mu*1j*KY])
        out = finufft.nufft2d2(2*np.pi*e.qy/N, 2*np.pi*e.qx/N, stack, isign=1, eps=EPS, nthreads=NTHREADS).real
        v0, gx0, gy0, qxx, qyy, qxy, vh, gxh, gyh = out
        v = v0+et*vh; gx = gx0+et*gxh; gy = gy0+et*gyh
        cols = [v, np.ones_like(v), e.X, e.Y]
        if not anchor:
            cols += [e.a*gx*e.phi[:, k] for k in range(len(POLY))]
            cols += [e.a*gy*e.phi[:, k] for k in range(len(POLY))]
            cols += [e.a*qxx, e.a*qyy, e.a*qxy]
        blocks = [sp.csc_matrix(np.stack(cols, 1))]
        if not anchor:
            blocks += [e.B, sp.diags(e.a*vh)@e.E]
        A = sp.hstack(blocks, format='csc')
        sw = np.sqrt(e.w)
        Aw = A.copy(); Aw.data *= sw[Aw.indices]
        colid = np.repeat(np.arange(Aw.shape[1]), np.diff(Aw.indptr))
        cn = np.sqrt(np.bincount(colid, weights=Aw.data**2, minlength=Aw.shape[1])); cn[cn == 0] = 1
        Aw.data /= cn[colid]
        rhs = sw*e.d
        if anchor:
            rhs = rhs-sw*e.a*v
        G = (Aw.T@Aw).toarray(); G[np.diag_indices_from(G)] += 1e-9
        sol = np.linalg.solve(G, Aw.T@rhs)/cn
        if not anchor:
            e.a = sol[0]
        e.c = sol[1]; e.g = sol[2:4].copy(); j = 4
        if not anchor:
            P = len(POLY)
            e.cx = e.cx+np.clip(sol[j:j+P], -0.3, 0.3); j += P
            e.cy = e.cy+np.clip(sol[j:j+P], -0.3, 0.3); j += P
            e.beta = e.beta+sol[j:j+3]; j += 3
            nb = e.B.shape[1]
            e.h = sol[j:j+nb]; j += nb
            e.eta = sol[j:]/e.a      # column was a*vh*E
    return e


def predict(e, Q0, H):
    et = eta_map(e); mu = e.mult()
    out = finufft.nufft2d2(2*np.pi*e.qy/N, 2*np.pi*e.qx/N, np.stack([to_F(Q0)*mu, to_F(H)*mu]), isign=1, eps=EPS, nthreads=NTHREADS).real
    return e.a*(out[0]+et*out[1])+e.add()


if __name__ == '__main__':
    import pickle
    tag = sys.argv[1]
    J1 = np.load('tgt_e1.npz')['J']
    exps = [attach_eta(e) for e in build('tgt', [f'tgt_e{i}.npz' for i in range(1, 7)], J1)]
    wt = np.median(np.concatenate([e.w for e in exps])); lam = 1e-4*wt
    Q0 = np.load(os.environ.get('QSTART', 'Q_q3.npy')); H = np.zeros((N, N))   # patternfit.py joint Q
    # initial amplitudes from the measured wing ratios (r = 30-110 px), constant over the cutout
    init = [0.0, 0.03, -0.19, -0.20, 0.07, -0.09]
    for e, v in zip(exps, init):
        e.eta[:] = v
    # first nuisance pass with the static Q (H=0) then alternate
    for k, e in enumerate(exps):
        qfit3.fit_nuisance(e, Q0, anchor=(k == 0))
    for it in range(int(os.environ.get('NJOINT', 3))):
        t0 = time.time()
        Q0, H = solve_QH(exps, lam, Q0, H, maxiter=60, nprox=2)
        for k, e in enumerate(exps):
            fit_nuisance(e, Q0, H, anchor=(k == 0))
        print(f'joint it{it} {time.time()-t0:.0f}s a=', [round(e.a, 4) for e in exps], ' eta mean=', [round(float(np.mean(eta_map(e))), 3) for e in exps], flush=True)
        np.save(f'Q0_{tag}.npy', Q0); np.save(f'H_{tag}.npy', H)
    pickle.dump([(e.a, e.cx, e.cy, e.beta, e.c, e.g, e.h, e.eta) for e in exps], open(f'nuis_{tag}.pkl', 'wb'))
    ks = [int(x) for x in os.environ.get('LOO', '1,2,3,4,5,6').split(',')]
    for k in ks:
        t0 = time.time()
        train = [e for j, e in enumerate(exps) if j != k-1]
        Q0k, Hk = solve_QH(train, lam, Q0, H, maxiter=60, nprox=3)
        e = exps[k-1]
        fit_nuisance(e, Q0k, Hk, anchor=False, niter=4)
        pred = predict(e, Q0k, Hk)
        wsum, cntf = train_weight_maps(train)
        ix = np.clip(np.round(e.qx).astype(int)+N//2, 0, N-1); iy = np.clip(np.round(e.qy).astype(int)+N//2, 0, N-1)
        varQ = e.a**2/np.maximum(wsum[iy, ix], 1e-30)
        n = e.ex.n
        out = {kk: np.full(n*n, np.nan) for kk in ['pred', 'cov', 'varQ', 'halo', 'eta']}
        out['pred'][e.pix] = pred; out['cov'][e.pix] = cntf[iy, ix]; out['varQ'][e.pix] = varQ
        out['halo'][e.pix] = e.B@e.h; out['eta'][e.pix] = eta_map(e)
        np.savez_compressed(f'{tag}_e{k}.npz', **{kk: v.reshape(n, n) for kk, v in out.items()}, a=e.a)
        chi = (e.d-pred)/np.sqrt(1/e.w+varQ); good = cntf[iy, ix] >= 3
        print(f'LOO e{k}: {time.time()-t0:.0f}s a={e.a:.5f} eta_mean={np.mean(eta_map(e)):.3f} chi2={np.mean(chi[good]**2):.3f} rstd={1.4826*np.median(np.abs(chi[good])):.3f} n={good.sum()}', flush=True)
    print('FINISHED', flush=True)
