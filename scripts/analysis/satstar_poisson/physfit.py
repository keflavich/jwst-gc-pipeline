"""Fit the physical-optics PSF (physpsf.py) to the static pattern Q of patternfit.py.

Two stages (README section 7):

1. `geometry`: pupil geometry (scale, rotation, V3/V2 anisotropy, segment size, strut width
   and angle), the star's position and the charge-diffusion width, by coordinate descent on
   the correlation of the high-passed Q with the high-passed geometric (OPD = 0) model on the
   diffraction spikes.  Spikes are pure pupil diffraction, so no wavefront is needed.
   Result for the demo star: scale 0.985, strut width 0.12 m, star at q = (+2.6, -0.6) px
   from the catalogue position; correlation 0.71 on the spikes and 0.62 between them.

2. `static`: phase retrieval of a free OPD map,

       Q(q) ~= f * D[ sum_lambda w_l |FFT(A_l exp(2 pi i OPD/lambda))|^2 ](q) + B(q) b

   D = pixel box x charge diffusion x IPC (band-limited, like Q); B = smooth 2-D B-spline
   background plus a star-centred smooth halo (log-r B-splines x m <= HALOM).  f and b are
   eliminated by weighted linear least squares at every evaluation (variable projection);
   the OPD, a (G, G) map over the pupil with a gradient penalty and a Gaussian prior, is
   fitted with L-BFGS using the exact adjoint.  Weights: noise plus a FLOOR relative error
   on the star light in the data.  Neighbours can be masked (compact positive residuals)
   and down-weighted with a Cauchy loss.
   On this star the retrieval does NOT converge to a physical solution: from any small
   random start it grows the OPD to 120-160 nm rms, i.e. it buys the observed halo (3-10x
   the pupil diffraction between the spikes) with fully developed speckle, whereas the
   observed halo is smooth (fine-structure contrast 17%, most of which is the geometric
   diffraction pattern).  It is kept for reference.

Usage (in a directory holding Q_<tag>.npy, tgt_e{1..6}.npz):
    python physfit.py weights                     # -> wq.npz, node weights of clean pixels
    python physfit.py geometry <Qfile> <out>      # -> <out>.npz (geometric static model)
    python physfit.py static <Qfile> <out>        # -> <out>.npz (OPD phase retrieval)
"""
import numpy as np, sys, os, time
from scipy import ndimage
from scipy.optimize import minimize
import scipy.sparse as sp
from physpsf import PhysPSF, f480m_band
from exposure import Exposure, bkg_matrix
from halobasis import halo_basis

N = 1152
RMIN, RMAX = float(os.environ.get('RMIN', 25)), float(os.environ.get('RMAX', 280))
FLOOR = float(os.environ.get('FLOOR', 0.05))
NLAM = int(os.environ.get('NLAM', 12))
G = int(os.environ.get('GOPD', 256))
MU = float(os.environ.get('MU', 1e-2))          # gradient penalty (per nm^2 of OPD difference)
SIGP = float(os.environ.get('SIGP', 100.))      # Gaussian prior rms of the OPD per grid node (nm)
HALOM = int(os.environ.get('HALOM', 6))         # smooth incoherent halo: log-r B-splines x m<=HALOM (-1: none)


def node_weights(files, J1):
    """sum of inverse variances of the clean (no SAT/JUMP) pixels landing on each node"""
    wqc = np.zeros(N*N); cnt = np.zeros(N*N)
    for f in files:
        ex = Exposure(f, J1)
        pix, qx, qy, _ = ex.full_q(ex.xt0, ex.yt0)
        ix = np.clip(np.round(qx).astype(int)+N//2, 0, N-1); iy = np.clip(np.round(qy).astype(int)+N//2, 0, N-1)
        cl = (ex.dq.ravel()[pix] & 6) == 0
        k = (iy*N+ix)[cl]
        wqc += np.bincount(k, weights=ex.w.ravel()[pix][cl], minlength=N*N)
        m = np.zeros(N*N); m[k] = 1; cnt += m
    return wqc.reshape(N, N), cnt.reshape(N, N)


def _wls(X, w, y):
    """weighted least squares with column normalisation (columns differ by ~1e10 in scale)"""
    Xw = X.multiply(np.sqrt(w)[:, None]).tocsc()
    cn = np.sqrt(np.asarray(Xw.multiply(Xw).sum(0))).ravel(); cn[cn == 0] = 1
    Xw = Xw@sp.diags(1/cn)
    A = (Xw.T@Xw).toarray(); A[np.diag_indices_from(A)] += 1e-10
    return np.linalg.solve(A, Xw.T@(np.sqrt(w)*y))/cn


class StaticFit:
    def __init__(self, Q, wqc, cnt, J1, geom=None, sources=((0.0, 0.0),)):
        self.Q = Q
        self.src = [tuple(map(float, s)) for s in sources]
        self.S = len(self.src)
        lams, wl = f480m_band(NLAM)
        self.ps = PhysPSF(N, J1, lams, wl, geom=geom, G=G)
        yy, xx = np.mgrid[:N, :N]-N//2
        self.r = np.hypot(xx, yy)
        self.base = (cnt >= 3) & (self.r >= RMIN) & (self.r <= RMAX) & (wqc > 0)
        if float(os.environ.get('SPIKEONLY', 0)) > 0:
            # only pixels within SPIKEONLY deg of the 8 spike directions (about the first source)
            xs, ys = sources[0]
            th = np.degrees(np.arctan2(yy-ys, xx-xs)) % 360
            dth = np.min([np.abs((th-a+180) % 360-180) for a in (0, 30, 90, 150, 180, 210, 270, 330)], 0)
            self.base &= dth < float(os.environ['SPIKEONLY'])
        self.var0 = np.where(wqc > 0, 1/np.where(wqc > 0, wqc, 1), np.inf)
        h = int(RMAX)+8
        self.c0 = N//2-h; self.nc = 2*h
        self.mask = np.zeros((N, N), bool)          # neighbours
        bg = np.median(Q[(self.r > 300) & (self.r < 450)])
        self.sig2 = self.var0+(FLOOR*np.clip(ndimage.median_filter(Q, 5)-bg, 1.0, None))**2
        self.rw = np.ones((N, N))                   # robust weights
        self.set_pixels()
        self.a = None
        self.f = np.full(self.S, 1.0/self.S)

    def set_pixels(self):
        sel = self.base & ~self.mask
        self.iy, self.ix = np.nonzero(sel)
        self.B = self.basis(self.iy, self.ix)
        self.y = self.Q[self.iy, self.ix]

    def basis(self, iy, ix):
        """smooth background (2-D B-splines, 64 px) + optional star-centred smooth halo"""
        loc = (iy-self.c0)*self.nc+(ix-self.c0)
        B = [bkg_matrix(self.nc, 64, loc)]
        if HALOM >= 0:
            xs, ys = self.src[0]
            B.append(halo_basis((ix-N//2-xs).astype(float), (iy-N//2-ys).astype(float), rmin=15., rmax=RMAX+10,
                                nknot=12, mmax=HALOM))
        return sp.hstack(B, format='csc')

    def weights(self, M=None):
        """fixed weights: noise + a relative error floor on the star's light in the DATA
        (Q minus the far background), so that they do not feed back on the model"""
        return self.rw[self.iy, self.ix]/self.sig2[self.iy, self.ix]

    def solve_lin(self, Mv, w):
        """Mv: (npix, S) detected unit-flux images of the sources.  Returns coefficients
        (fluxes..., background...) and the fitted values."""
        Mv = Mv.reshape(len(self.y), -1)
        X = sp.hstack([sp.csc_matrix(Mv), self.B], format='csc')
        c = _wls(X, w, self.y)
        return c, X@c

    def model(self, opd, amp=None):
        """per-source detected images (list of (N, N)) and the per-lambda fields"""
        I, E = self.ps.intensity(opd, amp=amp, return_fields=True)
        self.I = I
        return self.ps.detect_many(I, self.src), E

    def vec(self, Ms):
        return np.stack([m[self.iy, self.ix] for m in Ms], 1)

    def total(self, Ms, c):
        return sum(ci*m for ci, m in zip(c[:self.S], Ms))

    def reg(self, opd):
        o = opd.reshape(G, G)*1e9
        dx = np.diff(o, axis=1); dy = np.diff(o, axis=0)
        val = 0.5*MU*(np.sum(dx**2)+np.sum(dy**2))+0.5*np.sum((o/SIGP)**2)
        g = o/SIGP**2
        g[:, 1:] += MU*dx; g[:, :-1] -= MU*dx; g[1:, :] += MU*dy; g[:-1, :] -= MU*dy
        return val, g.ravel()*1e9

    def loss_grad(self, x):
        opd = x*1e-9
        Ms, E = self.model(opd)
        c, fit = self.solve_lin(self.vec(Ms), self.wfix)
        res = self.y-fit
        val = 0.5*np.sum(self.wfix*res**2)
        gM = np.zeros((N, N)); gM[self.iy, self.ix] = -self.wfix*res
        g = self.ps.grad_opd(self.ps.detect_T(gM, sources=self.src, fluxes=c[:self.S]), E, opd)*1e-9
        rv, rg = self.reg(opd)
        self.a = c[:self.S].sum(); self.f = c[:self.S]; self.last = (Ms, c)
        return val+rv, g+rg*1e-9

    def refresh_weights(self, opd, robust=True, remask=True):
        Ms, _ = self.model(opd)
        w = self.weights(self.vec(Ms)@(self.f*(self.a or 1.0)/self.f.sum()))
        c, fit = self.solve_lin(self.vec(Ms), w)
        self.a = c[:self.S].sum(); self.f = c[:self.S]
        M = self.total(Ms, c)
        # residual on the whole base region
        byy, bxx = np.nonzero(self.base)
        Bfull = self.basis(byy, bxx)
        fitb = M[byy, bxx]+Bfull@c[self.S:]
        res = np.zeros((N, N)); res[byy, bxx] = self.Q[byy, bxx]-fitb
        sig = np.sqrt(self.sig2)
        chi = np.where(self.base, res/sig, 0)
        if remask:
            sm = ndimage.gaussian_filter(np.where(self.base, res, 0), 1.0)/np.maximum(ndimage.gaussian_filter(self.base*1.0, 1.0), 1e-3)
            hot = self.base & (sm > float(os.environ.get('MASKSIG', 4))*sig)
            self.mask = ndimage.binary_dilation(hot, iterations=2) & self.base
        if robust:
            self.rw = 1/(1+(chi/3)**2)
        self.set_pixels()
        self.wfix = self.weights(M[self.iy, self.ix])
        self.chi = chi
        return chi

    def fit_opd(self, x0, maxiter=60):
        r = minimize(self.loss_grad, x0, jac=True, method='L-BFGS-B', options=dict(maxiter=maxiter, maxcor=20))
        return r.x, r.fun

    def geom_step(self, x, names=('rot', 'scale', 'aniso', 'strut_w', 'sigma'), steps=None, pos=True):
        """one Gauss-Newton step on geometry/detector parameters (central differences) and
        on the source positions (exact Fourier shift derivative)"""
        steps = dict(dict(rot=0.05, scale=0.002, aniso=0.002, strut_w=0.01, sigma=0.05, flat=0.005,
                          sm_d=0.02, strut_low=0.2), **(steps or {}))
        opd = x*1e-9
        Ms0, _ = self.model(opd); I0 = self.I
        f = self.f
        sw = np.sqrt(self.wfix)
        cols = []
        for nm in names:
            vals = []
            for sgn in (+1, -1):
                if nm == 'sigma':
                    old = self.ps.det['sigma']; self.ps.det['sigma'] = old+sgn*steps[nm]
                    Ms, _ = self.model(opd); self.ps.det['sigma'] = old
                else:
                    old = self.ps.geom[nm]; self.ps.set_geom({nm: old+sgn*steps[nm]})
                    Ms, _ = self.model(opd); self.ps.set_geom({nm: old})
                vals.append(self.vec(Ms)@f)
            cols.append((vals[0]-vals[1])/(2*steps[nm]))
        if pos:
            for s, (xs, ys) in enumerate(self.src):
                for d in ((0.05, 0), (0, 0.05)):
                    p = self.ps.detect(I0, shift=(xs+d[0], ys+d[1]))[self.iy, self.ix]
                    m = self.ps.detect(I0, shift=(xs-d[0], ys-d[1]))[self.iy, self.ix]
                    cols.append(f[s]*(p-m)/0.1)
        X = sp.hstack([sp.csc_matrix(np.concatenate([self.vec(Ms0), np.stack(cols, 1)], 1)), self.B], format='csc')
        sol = _wls(X, self.wfix, self.y)
        dp = sol[self.S:self.S+len(cols)]
        new = {}
        for nm, d in zip(names, dp):
            d = float(np.clip(d, -5*steps[nm], 5*steps[nm]))
            if nm == 'sigma':
                self.ps.det['sigma'] = max(0.0, self.ps.det['sigma']+d)
            else:
                new[nm] = self.ps.geom[nm]+d
        self.ps.set_geom(new)
        if pos:
            dq = np.clip(dp[len(names):], -1, 1).reshape(-1, 2)
            self.src = [(xs+d[0], ys+d[1]) for (xs, ys), d in zip(self.src, dq)]
        return dict(new, sigma=self.ps.det['sigma'], src=[(round(a, 3), round(b, 3)) for a, b in self.src])


def random_opd(G, half, rms_nm=30., slope=1.65, seed=0):
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(G, d=2*half/(G-1))
    kk = np.hypot(k[None, :], k[:, None]); kk[0, 0] = np.inf
    F = (rng.normal(size=(G, G))+1j*rng.normal(size=(G, G)))*np.maximum(kk, 1/6.6)**(-slope)
    F[0, 0] = 0
    o = np.fft.ifft2(F).real
    return (o/o.std()*rms_nm).ravel()


def chi_table(sf, opd, label):
    """robust residual / model in radial bins (relative misfit of the star model)"""
    Ms, _ = sf.model(opd)
    c, fit = sf.solve_lin(sf.vec(Ms), sf.wfix)
    star = sf.vec(Ms)@c[:sf.S]
    res = sf.y-fit; r = sf.r[sf.iy, sf.ix]
    out = []
    for lo, hi in [(25, 50), (50, 80), (80, 120), (120, 200), (200, 280)]:
        m = (r >= lo) & (r < hi)
        light = np.sqrt(np.maximum(sf.sig2[sf.iy, sf.ix]-sf.var0[sf.iy, sf.ix], 0))/FLOOR
        out.append(f'{lo}-{hi}: {1.4826*np.median(np.abs(res[m]))/np.median(light[m]):.3f}')
    print(label, 'f=', np.round(c[:sf.S], 1), 'rel.res', ' '.join(out), flush=True)
    return c


def _hp(a, w=15):
    return a-ndimage.uniform_filter(a, w)


def geometry_grid(Q, cnt, J1, out, xs=2.6, ys=-0.6):
    """Pupil geometry, source position and charge diffusion by coordinate descent on the
    correlation of the high-passed (15 px) Q with the high-passed geometric (OPD = 0) model,
    on pixels within 3 deg of the spikes (r = 60-280 px).  Spikes are pure pupil diffraction,
    so they fix the geometry without any wavefront; the Gauss-Newton geometry step of the
    OPD fit is unstable while the halo is misfit.  Also prints the between-spike correlation
    (r = 50-120 px, > 6 deg from the spikes)."""
    lams, wl = f480m_band(NLAM)
    geom = dict(rot=0.0, scale=1.0, aniso=1.0, flat=1.32, strut_w=0.08, strut_low=30.0)
    ps = PhysPSF(N, J1, lams, wl, geom=geom, G=G)
    yy, xx = np.mgrid[:N, :N]-N//2
    Qh = _hp(Q)

    def sels(x0, y0):
        r = np.hypot(xx-x0, yy-y0); th = np.degrees(np.arctan2(yy-y0, xx-x0)) % 360
        dth = np.min([np.abs((th-a+180) % 360-180) for a in (0, 30, 90, 150, 180, 210, 270, 330)], 0)
        return (cnt >= 3) & (r > 60) & (r < 280) & (dth < 3), (cnt >= 3) & (r > 50) & (r < 120) & (dth > 6)

    def corr(a, b):
        return np.corrcoef(np.clip(a, *np.percentile(a, [2, 98])), b)[0, 1]

    def score(I, x0, y0):
        Mh = _hp(ps.detect(I, shift=(x0, y0))); s1, s2 = sels(x0, y0)
        return corr(Qh[s1], Mh[s1]), corr(Qh[s2], Mh[s2])

    zero = np.zeros(G*G)
    for nm, vals in [('scale', [0.97, 0.98, 0.985, 0.99, 1.0, 1.01, 1.02]), ('rot', [-0.4, -0.2, 0, 0.2]),
                     ('aniso', [0.99, 0.995, 1.0, 1.005, 1.01]), ('flat', [1.30, 1.31, 1.32]),
                     ('strut_w', [0.05, 0.08, 0.10, 0.12, 0.14, 0.16]), ('strut_low', [29., 30., 31.])]:
        res = []
        for v in vals:
            ps.set_geom(dict(geom, **{nm: v})); res.append(score(ps.intensity(zero), xs, ys)+(v,))
        geom[nm] = max(res)[2]
        print(nm, [(round(a, 3), round(b, 3), v) for a, b, v in res], '->', geom[nm], flush=True)
    ps.set_geom(geom); I = ps.intensity(zero)
    best = None
    for x0 in (xs-0.3, xs, xs+0.3):
        for y0 in (ys-0.3, ys, ys+0.3):
            for sg in (0.1, 0.3, 0.5):
                ps.det['sigma'] = sg; c = score(I, x0, y0)
                if best is None or c[0] > best[0][0]:
                    best = (c, x0, y0, sg)
    (c1, c2), xs, ys, sg = best
    ps.det['sigma'] = sg
    print(f'geometry {geom} source ({xs:.2f}, {ys:.2f}) sigma {sg}: corr spikes {c1:.3f}, between spikes {c2:.3f}', flush=True)
    np.savez(out+'.npz', opd=zero, geom=geom, det=ps.det, f=np.array([1.0]), src=np.array([[xs, ys]]))


if __name__ == '__main__':
    J1 = np.load('tgt_e1.npz')['J']
    if sys.argv[1] == 'geometry':
        geometry_grid(np.load(sys.argv[2]), np.load('wq.npz')['cnt'], J1, sys.argv[3])
        sys.exit()
    if sys.argv[1] == 'weights':
        wqc, cnt = node_weights([f'tgt_e{i}.npz' for i in range(1, 7)], J1)
        np.savez('wq.npz', wqc=wqc, cnt=cnt)
        sys.exit()
    Qf, out = sys.argv[2], sys.argv[3]
    Q = np.load(Qf); z = np.load('wq.npz')
    # SRC: hand-set starting source positions for the `static` phase retrieval, in Q-grid px
    # from the grid centre (first = the target); geom_step refines them.  The retrieval is
    # kept for reference only and is not in the README recipe (it does not converge).
    src = [tuple(map(float, p.split(','))) for p in os.environ.get('SRC', '-4,0;9.5,0').split(';')]
    sf = StaticFit(Q, z['wqc'], z['cnt'], J1, geom=dict(rot=float(os.environ.get('ROT0', 0.0))), sources=src)
    x = random_opd(G, sf.ps.half, rms_nm=float(os.environ.get('RMS0', 20)))
    # hand-set initial total source flux in Q units, used only to form the first weights; the
    # fluxes are re-solved by variable projection in refresh_weights/loss_grad
    sf.a = 1.4e8; sf.f = np.full(sf.S, sf.a/sf.S)
    sf.refresh_weights(x*1e-9, robust=False, remask=False)
    chi_table(sf, x*1e-9, 'init')
    nouter = int(os.environ.get('NOUTER', 8))
    for it in range(nouter):
        t0 = time.time()
        x, f = sf.fit_opd(x, maxiter=int(os.environ.get('NINNER', 50)))
        chi_table(sf, x*1e-9, f'it{it} opd-fit f={f:.4g} rms={np.std(x[np.abs(x) > 0]):.1f}nm {time.time()-t0:.0f}s')
        if it >= int(os.environ.get('GEOM_FROM', 3)):
            g = sf.geom_step(x)
            print('   geom', g, flush=True)
        sf.refresh_weights(x*1e-9, robust=it >= 1, remask=it >= int(os.environ.get('MASK_FROM', 2)))
        print(f'   masked {sf.mask.sum()} nodes, npix {len(sf.y)}', flush=True)
        np.savez(out+'.npz', opd=x*1e-9, geom=sf.ps.geom, det=sf.ps.det, f=sf.f, src=np.array(sf.src), mask=sf.mask, rw=sf.rw)
    print('FINISHED', flush=True)
