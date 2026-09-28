"""Static pattern Q + per-exposure nuisance model, leave-one-dither-out.

    d_e(p) = a_e * [K_e * Q](q_e(p) + dq_e(p)) + c_e + g_e.(X,Y) + sum_s [B_{e,s} h_{e,s}](p)

  dq_e : cubic polynomial distortion correction in detector (X,Y)      (20 params)
  K_e  : first-order PSF-width kernel, FT = 1 - b_xx kx^2 - b_yy ky^2 - b_xy kx ky  (3)
  B_e,s: smooth halo about the target and the brightest saturated neighbours
         (cubic B-splines in log r x Fourier m<=mmax)
Q is band-limited (optical cutoff) on the ideal (V2V3) grid; solved by preconditioned CG
with a proximal term; nuisance by Gauss-Newton normal equations.

Q is everything that is fixed on the sky relative to the target: the saturated star's
PSF (all dithers share one roll angle, so the PSF is fixed relative to the sky too), its
neighbours, unresolved confusion and nebulosity.  Leave-one-dither-out (LOO) prediction
therefore tests whether the dithers are mutually consistent at the Poisson level once
distortion, flux scale, pedestal, PSF width, and smooth star-centred halos are allowed to
vary per exposure.  See README.md.

Usage (in a directory holding tgt_e{1..6}.npz from cutout.py):
    [QSTART=<initial Q .npy>] NJOINT=3 LOO=1,2,3,4,5,6 python patternfit.py <tag>
writes Q_<tag>.npy and <tag>_e<k>.npz (LOO prediction, coverage, model variance, halo).
Env: QN (grid size, default 1152), QH (grid spacing in px, default 1; <1 = super-resolved),
KMAX (band limit in cycles/px, default 0.45), QMASK (none|sat|satjump).
"""
import numpy as np, time, sys, os, pickle
import scipy.sparse as sp
from scipy.sparse.linalg import cg, LinearOperator
from scipy import ndimage
import finufft
from bandpsf import band_mask, EPS, NTHREADS
from exposure import Exposure
from halobasis import halo_basis

N = int(os.environ.get('QN', 1152))
HQ = float(os.environ.get('QH', 1.0))        # pattern-grid spacing in detector px (<1: super-resolved)
KMAXPX = float(os.environ.get('KMAX', 0.45))  # band limit in cycles per detector px
bm = band_mask(N, KMAXPX*HQ)
kv = np.arange(-N//2, N//2)*2*np.pi/N/HQ      # angular wavenumber per detector px
KX = kv[None, :]; KY = kv[:, None]
POLY = [(i, j) for i in range(4) for j in range(4) if i+j <= 3]      # 10 terms incl. const


def to_F(p):
    F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(p.reshape(N, N))))
    F[~bm] = 0
    return F/N**2


def from_G(G):
    G = G.copy(); G[~bm] = 0
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(G))).real.ravel()


def proj(v):
    return from_G(to_F(v)*N**2)


def nufft2(qx, qy, F):
    return finufft.nufft2d2(2*np.pi*qy/HQ/N, 2*np.pi*qx/HQ/N, F, isign=1, eps=EPS, nthreads=NTHREADS).real


def nufft1(qx, qy, r):
    return finufft.nufft2d1(2*np.pi*qy/HQ/N, 2*np.pi*qx/HQ/N, r.astype(complex), (N, N), isign=-1, eps=EPS, nthreads=NTHREADS)


def node(q):
    return np.clip(np.round(q/HQ).astype(int)+N//2, 0, N-1)


class QE:
    def __init__(self, ex, halo_centres, mmax=4, nknot=14):
        self.ex = ex
        pix, qx, qy, _ = ex.full_q(ex.xt0, ex.yt0)
        self.pix = pix; self.qx0 = qx; self.qy0 = qy
        self.d = ex.d.ravel()[pix]; self.w = ex.w.ravel()[pix]
        n = ex.n
        self.X = (pix % n - n/2)/(n/2); self.Y = (pix//n - n/2)/(n/2)
        self.phi = np.stack([self.X**i*self.Y**j for i, j in POLY], 1)
        self.cx = np.zeros(len(POLY)); self.cy = np.zeros(len(POLY))
        self.a = 1.0; self.c = 0.0; self.g = np.zeros(2); self.beta = np.zeros(3)
        blocks = []
        for (hx, hy, rmax, mm) in halo_centres:
            # halo centred on a detector position of this exposure
            dx = (pix % n)-hx; dy = (pix//n)-hy
            blocks.append(halo_basis(dx.astype(float), dy.astype(float), rmin=6. if rmax < 400 else 20., rmax=rmax, nknot=10 if rmax < 400 else nknot, mmax=mm))
        if int(os.environ.get('ROWCOL', 0)):
            # 1/f striping: an offset per (row, amplifier) and per column; the first of each
            # family is dropped (degenerate with the pedestal c)
            xdet = pix % n + ex.X0; row = pix//n; amp = np.clip(xdet//512, 0, 3)
            ra = row*4+amp; ura, ira = np.unique(ra, return_inverse=True)
            uc, ic = np.unique(pix % n, return_inverse=True)
            R = sp.csc_matrix((np.ones(len(pix)), (np.arange(len(pix)), ira)), shape=(len(pix), len(ura)))[:, 1:]
            C = sp.csc_matrix((np.ones(len(pix)), (np.arange(len(pix)), ic)), shape=(len(pix), len(uc)))[:, 1:]
            blocks += [R, C]
        self.B = sp.hstack(blocks, format='csc')
        self.h = np.zeros(self.B.shape[1])

    @property
    def qx(self):
        return self.qx0+self.phi@self.cx

    @property
    def qy(self):
        return self.qy0+self.phi@self.cy

    def mult(self):
        return 1.0-self.beta[0]*KX**2-self.beta[1]*KY**2-self.beta[2]*KX*KY

    def add(self):
        return self.c+self.g[0]*self.X+self.g[1]*self.Y+self.B@self.h

    def model_Q(self, F):
        return self.a*nufft2(self.qx, self.qy, F*self.mult())


def solve_Q(qexps, lam, Q0=None, maxiter=60, prox=0.03, nprox=2, verbose=True):
    cov = np.zeros(N*N)
    for e in qexps:
        ix = node(e.qx); iy = node(e.qy)
        cov += np.bincount(iy*N+ix, weights=e.w*e.a**2, minlength=N*N)
    d2 = ndimage.uniform_filter(cov.reshape(N, N), 3)
    d2 = np.where(d2 < 1e-6*np.median(d2[d2 > 0]), 0.0, d2)
    idx = ndimage.distance_transform_edt(d2 <= 0, return_distances=False, return_indices=True)
    dfill = d2[idx[0], idx[1]].ravel()
    L = prox*dfill+lam
    D = ndimage.uniform_filter(dfill.reshape(N, N), 5).ravel()+L
    b0 = np.zeros((N, N), complex)
    for e in qexps:
        b0 += nufft1(e.qx, e.qy, e.w*e.a*(e.d-e.add()))*e.mult()
    b0 = from_G(b0)
    def A(p):
        F = to_F(p)
        G = np.zeros((N, N), complex)
        for e in qexps:
            v = e.model_Q(F)
            G += nufft1(e.qx, e.qy, e.w*e.a*v)*e.mult()
        return from_G(G)+proj(L*proj(p))
    x = np.zeros(N*N) if Q0 is None else proj(Q0.ravel())
    for ip in range(nprox):
        b = b0+proj(L*x)
        it = [0]
        x1, info = cg(LinearOperator((N*N, N*N), matvec=A), b, x0=x, rtol=1e-6, maxiter=maxiter,
                      M=LinearOperator((N*N, N*N), matvec=lambda v: proj(proj(v)/D)), callback=lambda xx: it.__setitem__(0, it[0]+1))
        if verbose:
            print(f'   prox {ip}: CG its={it[0]} |dQ|/|Q|={np.linalg.norm(x1-x)/np.linalg.norm(x1):.2e}', flush=True)
        x = x1
    return x.reshape(N, N)


def fit_nuisance(e, Qp, anchor=False, niter=3):
    """Gauss-Newton on (a, c, g, dq-poly, beta, halo).  anchor=True: only c, g (defines frame)."""
    F = to_F(Qp)
    for _ in range(niter):
        Fm = F*e.mult()
        stack = np.stack([Fm, Fm*1j*KX, Fm*1j*KY, -F*KX**2, -F*KY**2, -F*KX*KY])
        out = finufft.nufft2d2(2*np.pi*e.qy/HQ/N, 2*np.pi*e.qx/HQ/N, stack, isign=1, eps=EPS, nthreads=NTHREADS).real
        v, gx, gy, qxx, qyy, qxy = out
        cols = [v, np.ones_like(v), e.X, e.Y]
        if not anchor:
            cols += [e.a*gx*e.phi[:, k] for k in range(len(POLY))]
            cols += [e.a*gy*e.phi[:, k] for k in range(len(POLY))]
            cols += [e.a*qxx, e.a*qyy, e.a*qxy]
        Dn = sp.csc_matrix(np.stack(cols, 1))
        A = sp.hstack([Dn, e.B], format='csc') if not anchor else Dn
        sw = np.sqrt(e.w)
        Aw = A.copy(); Aw.data *= sw[Aw.indices]
        colid = np.repeat(np.arange(Aw.shape[1]), np.diff(Aw.indptr))
        cn = np.sqrt(np.bincount(colid, weights=Aw.data**2, minlength=Aw.shape[1])); cn[cn == 0] = 1
        Aw.data /= cn[colid]
        rhs = sw*e.d
        if anchor:
            rhs = rhs-sw*e.a*v
        G = (Aw.T@Aw).toarray(); G[np.diag_indices_from(G)] += 1e-10
        sol = np.linalg.solve(G, Aw.T@rhs)/cn
        j = 0
        if not anchor:
            e.a = sol[0]
        e.c = sol[1]; e.g = sol[2:4].copy(); j = 4
        if not anchor:
            P = len(POLY)
            e.cx = e.cx+np.clip(sol[j:j+P], -0.3, 0.3); j += P
            e.cy = e.cy+np.clip(sol[j:j+P], -0.3, 0.3); j += P
            e.beta = e.beta+sol[j:j+3]; j += 3
            e.h = sol[j:]
    return e


def secondary_centres(ex, nsec):
    lab, n = ndimage.label((ex.dq & 2) > 0)
    sz = ndimage.sum(np.ones(lab.shape), lab, np.arange(1, n+1))
    com = ndimage.center_of_mass(np.ones(lab.shape), lab, np.arange(1, n+1))
    out = []
    for k in np.argsort(sz)[::-1]:
        cy, cx = com[k]
        if np.hypot(cx-ex.xt0, cy-ex.yt0) < 100 or sz[k] < 300:
            continue
        out.append((cx, cy, 250., 2))
        if len(out) >= nsec:
            break
    return out


def build(prefix, files, J1, nsec=4, crop=None):
    exps = []
    for f in files:
        ex = Exposure(f, J1)
        if crop:
            yy, xx = np.mgrid[:ex.n, :ex.n]
            ex.good0 &= (np.abs(xx-ex.xt0) < crop) & (np.abs(yy-ex.yt0) < crop); ex.set_mask(None)
        QMASK = os.environ.get('QMASK', 'none')
        if QMASK != 'none':
            ex.set_mask((ex.dq & {'sat': 2, 'satjump': 6}[QMASK]) > 0)
        cents = [(ex.xt0, ex.yt0, 720., 4)]+secondary_centres(ex, nsec)
        exps.append(QE(ex, cents))
    return exps


def train_weight_maps(train):
    wsum = np.zeros(N*N); cnt = np.zeros(N*N)
    for t in train:
        jx = node(t.qx); jy = node(t.qy)
        wsum += np.bincount(jy*N+jx, weights=t.w*t.a**2, minlength=N*N)
        m = np.zeros(N*N); m[jy*N+jx] = 1; cnt += m
    return wsum.reshape(N, N), ndimage.minimum_filter(cnt.reshape(N, N), 3)


if __name__ == '__main__':
    tag = sys.argv[1]
    J1 = np.load('tgt_e1.npz')['J']
    exps = build('tgt', [f'tgt_e{i}.npz' for i in range(1, 7)], J1)
    wt = np.median(np.concatenate([e.w for e in exps])); lam = 1e-4*wt
    Qp = np.load(os.environ.get('QSTART', 'Q_q2.npy'))
    for it in range(int(os.environ.get('NJOINT', 3))):
        t0 = time.time()
        for k, e in enumerate(exps):
            fit_nuisance(e, Qp, anchor=(k == 0))
        print(f'nuis it{it}', [(round(e.a, 4), np.round(e.cx[:3], 3).tolist(), np.round(e.beta, 3).tolist()) for e in exps], flush=True)
        Qp = solve_Q(exps, lam, Q0=Qp, maxiter=60, nprox=2)
        print(f'joint it{it} {time.time()-t0:.0f}s', flush=True)
        np.save(f'Q_{tag}.npy', Qp)
    if int(os.environ.get('NJOINT', 3)) == 0:
        for k, e in enumerate(exps):
            fit_nuisance(e, Qp, anchor=(k == 0))
            fit_nuisance(e, Qp, anchor=(k == 0))
    ks = [int(x) for x in os.environ.get('LOO', '1,2,3,4,5,6').split(',')]
    for k in ks:
        t0 = time.time()
        train = [e for j, e in enumerate(exps) if j != k-1]
        Qk = solve_Q(train, lam, Q0=Qp, maxiter=60, nprox=3)
        e = exps[k-1]
        fit_nuisance(e, Qk, anchor=False, niter=4)
        F = to_F(Qk)
        pred = e.model_Q(F)+e.add()
        wsum, cntf = train_weight_maps(train)
        ix = node(e.qx); iy = node(e.qy)
        varQ = e.a**2/np.maximum(wsum[iy, ix], 1e-30)
        n = e.ex.n
        out = {kk: np.full(n*n, np.nan) for kk in ['pred', 'cov', 'varQ', 'halo']}
        out['pred'][e.pix] = pred; out['cov'][e.pix] = cntf[iy, ix]; out['varQ'][e.pix] = varQ; out['halo'][e.pix] = e.B@e.h
        np.savez_compressed(f'{tag}_e{k}.npz', **{kk: v.reshape(n, n) for kk, v in out.items()}, a=e.a, cx=e.cx, cy=e.cy, beta=e.beta, c=e.c)
        chi = (e.d-pred)/np.sqrt(1/e.w+varQ); good = cntf[iy, ix] >= 3
        print(f'LOO e{k}: {time.time()-t0:.0f}s a={e.a:.5f} beta={np.round(e.beta,4)} chi2={np.mean(chi[good]**2):.3f} rstd={1.4826*np.median(np.abs(chi[good])):.3f} n={good.sum()}', flush=True)
    print('FINISHED', flush=True)
