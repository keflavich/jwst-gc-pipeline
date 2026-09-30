"""Band-limited PSF on a periodic grid, evaluated at arbitrary points with finufft.

p is an (N,N) real array of samples at integer ideal-pixel offsets q = j - N//2.
P(q) is the unique band-limited (periodic-sinc) interpolant, Nyquist row/col
zeroed so the interpolant is real.
"""
import numpy as np, finufft

NTHREADS = 4
EPS = 1e-9
# optical band limit: D_circ/lambda_min at F480M blue edge (6.6 m, 4.63 um) in cycles per
# 0.0629" ideal pixel = 0.4345.  Everything above is unphysical noise.
# The default KMAX=0.45 sits ~3.5% above that limit on purpose: the band mask is a hard cut on
# the discrete frequency grid (spacing 1/N), so a small margin guarantees the physical
# annulus just inside 0.4345 is never clipped.  The extra 0.4345-0.45 annulus carries only
# noise, which the proximal/Tikhonov term damps.  All README numbers use 0.45, and a 0.5 limit
# changed nothing (README section 3), so the margin does not affect the results.
import os
KMAX = float(os.environ.get("KMAX", 0.45))


def band_mask(N, kmax=KMAX):
    k = np.arange(-N//2, N//2)/N
    kk = np.hypot(k[None, :], k[:, None])
    m = kk <= kmax
    m[0, :] = False; m[:, 0] = False
    return m


class BandPSF:
    def __init__(self, p):
        self.set(p)

    @property
    def N(self):
        return self.p.shape[0]

    def set(self, p):
        self.p = np.asarray(p, float)
        N = self.N
        F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.p)))
        F[~band_mask(N)] = 0
        self.F = F / N**2
        self._grad = None

    def _pts(self, qx, qy):
        N = self.N
        return (2*np.pi*np.asarray(qy, float)/N, 2*np.pi*np.asarray(qx, float)/N)

    def __call__(self, qx, qy, F=None):
        ty, tx = self._pts(qx, qy)
        F = self.F if F is None else F
        return finufft.nufft2d2(ty, tx, F, isign=1, eps=EPS, nthreads=NTHREADS).real

    def project(self):
        """band-limited version of the sample array"""
        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(self.F*self.N**2))).real

    def grads(self):
        if self._grad is None:
            N = self.N
            k = np.arange(-N//2, N//2)*2*np.pi/N
            self._grad = (self.F*1j*k[None, :], self.F*1j*k[:, None])
        return self._grad

    def eval_with_grad(self, qx, qy):
        gx, gy = self.grads()
        ty, tx = self._pts(qx, qy)
        stack = np.stack([self.F, gx, gy])
        out = finufft.nufft2d2(ty, tx, stack, isign=1, eps=EPS, nthreads=NTHREADS).real
        return out[0], out[1], out[2]

    @staticmethod
    def adjoint(qx, qy, r, N):
        """Adjoint of p -> P(q) (real samples), returns (N,N) real array."""
        ty, tx = 2*np.pi*np.asarray(qy, float)/N, 2*np.pi*np.asarray(qx, float)/N
        G = finufft.nufft2d1(ty, tx, np.asarray(r, complex), (N, N), isign=-1,
                             eps=EPS, nthreads=NTHREADS)
        G[~band_mask(N)] = 0
        g = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(G))).real
        return g   # ifft2 includes 1/N^2 ; forward had /N^2 and fft2 (no norm)


def _selftest():
    rng = np.random.default_rng(1)
    N = 64
    p = rng.normal(size=(N, N))
    ps = BandPSF(p)
    # interpolant hits the samples
    q = np.arange(N) - N//2
    qx, qy = np.meshgrid(q, q)
    v = ps(qx.ravel(), qy.ravel()).reshape(N, N)
    # nyquist removal changes values; compare to p with nyquist removed
    Fp = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(p))); Fp[~band_mask(N)] = 0
    pn = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Fp))).real
    print('sample err', np.abs(v-pn).max())
    x = rng.uniform(-30, 30, 500); y = rng.uniform(-30, 30, 500); r = rng.normal(size=500)
    lhs = np.dot(ps(x, y), r)
    rhs = np.sum(BandPSF.adjoint(x, y, r, N)*p)
    print('dot test', lhs, rhs)
    a, gx, gy = ps.eval_with_grad(x, y)
    h = 1e-5
    print('grad test', np.abs((ps(x+h, y)-ps(x-h, y))/(2*h)-gx).max(), np.abs((ps(x, y+h)-ps(x, y-h))/(2*h)-gy).max())


if __name__ == '__main__':
    _selftest()
