"""Physical-optics (Fraunhofer) PSF of JWST + NIRCam LW, polychromatic, with exact adjoint.

The PSF is computed directly on the ideal (V2V3) pattern grid used by patternfit.py: node
(i, j) of an (N, N) grid is the offset q = (i - N/2, j - N/2) from the star, in units of
detector pixels of dither 1 (q = J1^-1 (v2v3 - v2v3_star), J1 = d(v2,v3)/d(x,y) at the star).

For one wavelength the focal-plane field is the discrete Fourier transform of the complex
pupil p = A exp(2 pi i OPD / lambda):

    E(q) = sum_m p(m) exp(-2 pi i m.q / N),    u(m) = lambda / (C N) J1^-T m     [metres]

so the pupil is sampled at lambda-dependent points u(m) (C = arcsec -> rad) and one FFT of
size N gives E exactly on the pattern grid.  The pupil geometry is analytic (18 hexagonal
segments with gaps, secondary obscuration, 3 struts, optional circular stop with shear) and
anti-aliased with a signed distance; the OPD lives on a fixed (G, G) grid in metres of
pupil and is bilinearly interpolated to every lambda's samples (sparse matrix, exact
transpose).  The detector response (pixel box, charge diffusion, IPC) multiplies the Fourier
transform of the intensity, which is then band-limited to |k| <= KMAX like the pattern.

Geometry: V2 is -x, V3 is +y on NRCB5 near the target.  Segments are flat-topped (edges
parallel to V2) so the six main spikes lie at 30, 90, 150 deg; the vertical strut (along
V3) makes the horizontal spike, the two lower struts run at +-30 deg from -V3.  The
dimensions are the nominal ones (flat-to-flat 1.32 m, 7 mm gaps, 0.74 m secondary); the
pupil could not be taken from STPSF (STScI hosts unreachable), so its fine details
(strut width and exact layout) are fitted.

    psf = PhysPSF(N, J1, lams, weights)
    I = psf.intensity(opd_grid)                # (N, N), sums to 1 per lambda
    M = psf.detect(I)                          # pixel/diffusion/IPC, band-limited
    g_opd = psf.grad_opd(g_I)                  # d loss / d opd_grid given d loss / d I
"""
import numpy as np
import scipy.fft as sfft
import scipy.sparse as sp

C = np.pi/180/3600
NWORK = 4


def fft2c(a):
    return sfft.fftshift(sfft.fft2(sfft.ifftshift(a), workers=NWORK))


def ifft2c(a):
    return sfft.fftshift(sfft.ifft2(sfft.ifftshift(a), workers=NWORK))


# ----------------------------------------------------------------------------- geometry
DEFAULT_GEOM = dict(rot=0.0,          # deg, rotation of the whole pupil (V2V3)
                    scale=1.0,        # multiplies every pupil dimension (= plate-scale/lambda error)
                    aniso=1.0,        # extra V3/V2 scale ratio of the pupil
                    flat=1.32, gap=0.007, sm_d=0.74,
                    strut_w=0.08, strut_low=30.0,   # lower struts: deg from -V3
                    stop_r=10.0, stop_dx=0.0, stop_dy=0.0,   # optional circular stop (m)
                    amp_gx=0.0, amp_gy=0.0)       # linear transmission gradient over the pupil


def segment_centres(pitch):
    c = [(0.0, 0.0)]
    out = []
    for ang in range(30, 360, 60):                 # ring 1 (flat-to-flat neighbours)
        a = np.radians(ang)
        out.append((pitch*np.cos(a), pitch*np.sin(a)))
    for ang in range(30, 360, 60):                 # ring 2, along the same directions
        a = np.radians(ang)
        out.append((2*pitch*np.cos(a), 2*pitch*np.sin(a)))
    for ang in range(0, 360, 60):                  # ring 2, between them
        a = np.radians(ang)
        out.append((np.sqrt(3)*pitch*np.cos(a), np.sqrt(3)*pitch*np.sin(a)))
    return np.array(out)


def hex_sd(x, y, apothem):
    """signed distance-ish (exact along the edge normals) to a flat-topped hexagon"""
    d = np.abs(y)
    for a in (np.radians(30), np.radians(150)):
        d = np.maximum(d, np.abs(x*np.cos(a)+y*np.sin(a)))
    return d-apothem


def pupil_amplitude(ux, uy, du, g):
    """anti-aliased transmission of the JWST+NIRCam LW pupil at points (ux, uy) [m, V2V3]."""
    s = g['scale']
    r = np.radians(g['rot'])
    x = (np.cos(r)*ux+np.sin(r)*uy)/s
    y = (-np.sin(r)*ux+np.cos(r)*uy)/(s*g['aniso'])
    dd = du/s
    pitch = g['flat']+g['gap']
    sd = np.full(x.shape, np.inf)
    for cx, cy in segment_centres(pitch):
        sd = np.minimum(sd, hex_sd(x-cx, y-cy, g['flat']/2))
    A = np.clip(0.5-sd/dd, 0, 1)
    # secondary obscuration
    A *= np.clip(0.5+(np.hypot(x, y)-g['sm_d']/2)/dd, 0, 1)
    # struts: one along +V3, two at +-strut_low from -V3
    w2 = g['strut_w']/2
    for ang in (90.0, 270.0-g['strut_low'], 270.0+g['strut_low']):
        a = np.radians(ang)
        t = x*np.cos(a)+y*np.sin(a); nrm = np.abs(-x*np.sin(a)+y*np.cos(a))
        cov = np.clip(0.5-(nrm-w2)/dd, 0, 1)*(t > 0)
        A *= 1-cov
    # optional NIRCam pupil stop (circular, sheared) in the unrotated V2V3 frame
    if g['stop_r'] < 5:
        rs = np.hypot(ux-g['stop_dx'], uy-g['stop_dy'])
        A *= np.clip(0.5-(rs-g['stop_r'])/du, 0, 1)
    if g['amp_gx'] or g['amp_gy']:
        A *= np.clip(1+g['amp_gx']*ux/3.3+g['amp_gy']*uy/3.3, 0, None)
    return A


def bilinear_matrix(ux, uy, G, half):
    """sparse (npts x G*G) bilinear interpolation from a G x G grid spanning [-half, half]"""
    h = 2*half/(G-1)
    fx = (ux+half)/h; fy = (uy+half)/h
    ix = np.clip(np.floor(fx).astype(int), 0, G-2); iy = np.clip(np.floor(fy).astype(int), 0, G-2)
    tx = fx-ix; ty = fy-iy
    rows = np.repeat(np.arange(len(ux)), 4)
    cols = np.stack([iy*G+ix, iy*G+ix+1, (iy+1)*G+ix, (iy+1)*G+ix+1], 1).ravel()
    vals = np.stack([(1-tx)*(1-ty), tx*(1-ty), (1-tx)*ty, tx*ty], 1).ravel()
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(ux), G*G))


def f480m_band(nlam=12, lo=4.63, hi=4.99):
    """top-hat F480M band sampled at nlam wavelengths [m]"""
    edges = np.linspace(lo, hi, nlam+1)
    return 0.5*(edges[1:]+edges[:-1])*1e-6, np.full(nlam, 1.0/nlam)


class PhysPSF:
    def __init__(self, N, J, lams, weights, geom=None, G=256, half=3.45):
        self.N = N; self.J = np.asarray(J, float)
        self.lams = np.asarray(lams, float); self.wl = np.asarray(weights, float)
        self.G = G; self.half = half
        self.geom = dict(DEFAULT_GEOM, **(geom or {}))
        m = np.arange(N)-N//2
        self.mx, self.my = [a.ravel() for a in np.meshgrid(m, m)]
        self.JinvT = np.linalg.inv(self.J).T
        self.set_geom(self.geom)
        k = np.arange(-N//2, N//2)/N
        self.kx = k[None, :]; self.ky = k[:, None]
        self.det = dict(sigma=0.3, ipc=0.006)

    def set_geom(self, geom):
        self.geom = dict(self.geom, **geom)
        self.pts = []
        for lam in self.lams:
            s = lam/(C*self.N)
            ux = s*(self.JinvT[0, 0]*self.mx+self.JinvT[0, 1]*self.my)
            uy = s*(self.JinvT[1, 0]*self.mx+self.JinvT[1, 1]*self.my)
            du = s*np.sqrt(abs(np.linalg.det(self.JinvT)))
            near = np.flatnonzero(np.hypot(ux, uy) < 3.7*self.geom['scale']+0.1)
            A = pupil_amplitude(ux[near], uy[near], du, self.geom)
            keep = A > 0
            idx = near[keep]
            self.pts.append(dict(idx=idx, A=A[keep], ux=ux[idx], uy=uy[idx], du=du,
                                 S=bilinear_matrix(ux[idx], uy[idx], self.G, self.half),
                                 norm=self.N**2*np.sum(A[keep]**2)))

    # ------------------------------------------------------------------ forward
    def fields(self, opd, amp=None):
        """per-lambda focal fields.  opd: (G*G,) metres; amp: optional (G*G,) multiplicative
        transmission map on the OPD grid."""
        out = []
        for lam, P in zip(self.lams, self.pts):
            ph = 2*np.pi/lam*(P['S']@opd)
            A = P['A'] if amp is None else P['A']*(P['S']@amp)
            p = np.zeros(self.N*self.N, complex)
            p[P['idx']] = A*np.exp(1j*ph)
            out.append(fft2c(p.reshape(self.N, self.N)))
        return out

    def intensity(self, opd, amp=None, return_fields=False):
        E = self.fields(opd, amp)
        I = np.zeros((self.N, self.N))
        for w, P, e in zip(self.wl, self.pts, E):
            I += w/P['norm']*(e.real**2+e.imag**2)
        return (I, E) if return_fields else I

    def mtf(self):
        s = self.det['sigma']; a = self.det['ipc']
        kx, ky = self.kx, self.ky
        return (np.sinc(kx)*np.sinc(ky)*np.exp(-2*np.pi**2*s**2*(kx**2+ky**2))
                * (1-4*a+2*a*(np.cos(2*np.pi*kx)+np.cos(2*np.pi*ky))))

    def shift_phase(self, x, y):
        """Fourier factor that moves an image by (+x, +y) nodes"""
        return np.exp(-2j*np.pi*(self.kx*x+self.ky*y))

    def detect(self, I, kmax=0.45, shift=None):
        """pixel/diffusion/IPC response, band limit, optional (x, y) shift of the source"""
        F = fft2c(I)*self.mtf()
        F[np.hypot(self.kx, self.ky) > kmax] = 0
        if shift is not None:
            F = F*self.shift_phase(*shift)
        return ifft2c(F).real

    def detect_many(self, I, sources, kmax=0.45):
        """one detected image per source position [(x, y), ...] (unit flux each)"""
        F = fft2c(I)*self.mtf()
        F[np.hypot(self.kx, self.ky) > kmax] = 0
        return [ifft2c(F*self.shift_phase(x, y)).real for x, y in sources]

    def detect_T(self, g, kmax=0.45, sources=None, fluxes=None):
        """transpose of sum_s f_s detect(., shift=s)"""
        F = fft2c(g)*self.mtf()
        F[np.hypot(self.kx, self.ky) > kmax] = 0
        if sources is not None:
            F = F*sum(f*np.conj(self.shift_phase(x, y)) for (x, y), f in zip(sources, fluxes))
        return ifft2c(F).real

    # ------------------------------------------------------------------ adjoint
    def grad_pupil(self, gI, E, opd, amp=None):
        """d loss / d (phase, A) per lambda at the pupil samples, given gI = d loss / d I."""
        out = []
        for lam, w, P, e in zip(self.lams, self.wl, self.pts, E):
            Gf = ifft2c(gI*e)*self.N**2*(w/P['norm'])          # T^H (g E), T = centred fft
            Gp = Gf.ravel()[P['idx']]
            ph = 2*np.pi/lam*(P['S']@opd)
            A = P['A'] if amp is None else P['A']*(P['S']@amp)
            ep = np.exp(1j*ph)
            p = A*ep
            gphi = 2*np.imag(Gp*np.conj(p))
            gA = 2*np.real(np.conj(Gp)*ep)
            out.append((gphi, gA))
        return out

    def grad_opd(self, gI, E, opd, amp=None, want_amp=False):
        g = np.zeros(self.G*self.G); ga = np.zeros(self.G*self.G) if want_amp else None
        for lam, P, (gphi, gA) in zip(self.lams, self.pts, self.grad_pupil(gI, E, opd, amp)):
            g += P['S'].T@(2*np.pi/lam*gphi)
            if want_amp:
                ga += P['S'].T@(gA*P['A'])
        return (g, ga) if want_amp else g

    def jvp(self, opd, E, dopd=None, damp=None, amp=None):
        """forward-mode derivative of the intensity: dI for an OPD change dopd (G*G, m) and/or
        a transmission change damp (G*G, multiplicative, relative) at the point opd."""
        dI = np.zeros((self.N, self.N))
        for lam, w, P, e in zip(self.lams, self.wl, self.pts, E):
            ph = 2*np.pi/lam*(P['S']@opd)
            A = P['A'] if amp is None else P['A']*(P['S']@amp)
            p = A*np.exp(1j*ph)
            dp = np.zeros(len(p), complex)
            if dopd is not None:
                dp += 1j*2*np.pi/lam*(P['S']@dopd)*p
            if damp is not None:
                dp += (P['S']@damp)*p
            full = np.zeros(self.N*self.N, complex); full[P['idx']] = dp
            de = fft2c(full.reshape(self.N, self.N))
            dI += w/P['norm']*2*np.real(np.conj(e)*de)
        return dI

    # ------------------------------------------------------------------ modal bases
    def grid_xy(self):
        t = np.linspace(-self.half, self.half, self.G)
        X, Y = np.meshgrid(t, t)
        return X.ravel(), Y.ravel()

    def zernike_basis(self, nmax=4, R=3.3):
        """Zernike polynomials (Noll-like ordering, piston/tip/tilt excluded) on the OPD grid
        over a circle of radius R; returns (nmode, G*G), unit rms over the circle."""
        from math import factorial
        X, Y = self.grid_xy(); rho = np.hypot(X, Y)/R; th = np.arctan2(Y, X)
        modes = []
        for n in range(2, nmax+1):
            for m in range(-n, n+1, 2):
                am = abs(m)
                Rn = np.zeros_like(rho)
                for k in range((n-am)//2+1):
                    Rn += (-1)**k*factorial(n-k)/(factorial(k)*factorial((n+am)//2-k)*factorial((n-am)//2-k))*rho**(n-2*k)
                z = Rn*(np.cos(am*th) if m >= 0 else np.sin(am*th))
                z[rho > 1.2] = 0
                modes.append(z/np.sqrt(np.mean(z[rho <= 1]**2)))
        return np.array(modes)

    def segment_basis(self):
        """piston, tip, tilt of each of the 18 segments on the OPD grid: (54, G*G), tip/tilt in
        metres of OPD per metre across the segment (unit slope)."""
        X, Y = self.grid_xy(); g = self.geom
        r = np.radians(g['rot'])
        x = (np.cos(r)*X+np.sin(r)*Y)/g['scale']; y = (-np.sin(r)*X+np.cos(r)*Y)/g['scale']
        pitch = g['flat']+g['gap']
        out = []
        for cx, cy in segment_centres(pitch):
            inside = hex_sd(x-cx, y-cy, pitch/2) <= 0
            out += [inside*1.0, inside*(x-cx), inside*(y-cy)]
        return np.array(out)


def _gradcheck():
    rng = np.random.default_rng(3)
    N = 256
    J = np.array([[-0.0628*4.5, 0.00001], [0.00009, 0.0631*4.5]])   # coarse pixels, quick test
    lams, w = f480m_band(3)
    ps = PhysPSF(N, J, lams, w, G=64)
    opd = rng.normal(size=64*64)*20e-9
    tgt = rng.normal(size=(N, N))
    def loss(o):
        return np.sum(tgt*ps.detect(ps.intensity(o)))
    I, E = ps.intensity(opd, return_fields=True)
    g = ps.grad_opd(ps.detect_T(tgt), E, opd)
    for k in rng.choice(np.flatnonzero(np.abs(g) > 0), 5):
        h = np.zeros_like(opd); h[k] = 1e-9
        print('grad', g[k], (loss(opd+h)-loss(opd-h))/2e-9)
    print('flux', I.sum())


if __name__ == '__main__':
    _gradcheck()
