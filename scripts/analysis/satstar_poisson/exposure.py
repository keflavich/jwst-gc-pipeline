"""A star-centred NIRCam cutout with per-pixel V2V3 (GWCS) coordinates, plus sparse
B-spline bases used as smooth nuisance terms."""
import numpy as np, scipy.sparse as sp
from scipy import ndimage
from scipy.interpolate import BSpline
from scipy.sparse.linalg import lsqr

DO_NOT_USE = 1


class Exposure:
    def __init__(self, fn, Jref, extra_mask=None):
        z = np.load(fn)
        self.fn = fn
        self.d = z['d']; self.var = z['var']; self.dq = z['dq']
        self.vf = z['vf']
        self.v2 = z['v2']; self.v3 = z['v3']
        self.n = self.d.shape[0]
        self.good = np.isfinite(self.d) & np.isfinite(self.var) & (self.var > 0) & ((self.dq & DO_NOT_USE) == 0)
        if extra_mask is not None:
            self.good &= ~extra_mask
        self.good0 = self.good.copy()
        self.w = np.where(self.good, 1.0/np.where(self.good, self.var, 1), 0.0)
        self.Jinv = np.linalg.inv(Jref)
        self.X0 = int(z['X0']); self.Y0 = int(z['Y0'])
        self.xt0 = float(z['xt']); self.yt0 = float(z['yt'])
        # jacobian maps (per-pixel) of v2v3 w.r.t. detector x,y
        self.dv2dx = np.gradient(self.v2, axis=1); self.dv2dy = np.gradient(self.v2, axis=0)
        self.dv3dx = np.gradient(self.v3, axis=1); self.dv3dy = np.gradient(self.v3, axis=0)
        self.idx = np.flatnonzero(self.good.ravel())
        self.B = None

    def set_mask(self, extra=None):
        self.good = self.good0.copy() if extra is None else (self.good0 & ~extra)
        self.w = np.where(self.good, 1.0/np.where(self.good, self.var, 1), 0.0)
        self.idx = np.flatnonzero(self.good.ravel())

    def Jloc(self, x, y):
        i = int(np.clip(round(y), 1, self.n-2)); j = int(np.clip(round(x), 1, self.n-2))
        return np.array([[self.dv2dx[i, j], self.dv2dy[i, j]], [self.dv3dx[i, j], self.dv3dy[i, j]]])

    def vpos(self, x, y):
        """v2v3 at fractional detector position via local linearisation."""
        i = int(np.clip(round(y), 0, self.n-1)); j = int(np.clip(round(x), 0, self.n-1))
        J = self.Jloc(x, y)
        return np.array([self.v2[i, j], self.v3[i, j]]) + J @ np.array([x-j, y-i])

    # ---------------- star geometry ----------------
    def full_q(self, x, y, pix=None):
        """ideal offsets for all good pixels (or pix subset) from star at (x,y)."""
        pix = self.idx if pix is None else pix
        vs = self.vpos(x, y)
        dv2 = self.v2.ravel()[pix]-vs[0]; dv3 = self.v3.ravel()[pix]-vs[1]
        qx = self.Jinv[0, 0]*dv2+self.Jinv[0, 1]*dv3
        qy = self.Jinv[1, 0]*dv2+self.Jinv[1, 1]*dv3
        dqds = self.Jinv @ self.Jloc(x, y)    # d q / d (x,y) of *pixel*; star derivative is minus
        return pix, qx, qy, dqds

    def stamp_q(self, x, y, R):
        n = self.n
        i0, j0 = int(round(y)), int(round(x))
        ii, jj = np.mgrid[max(0, i0-R):min(n, i0+R+1), max(0, j0-R):min(n, j0+R+1)]
        ii = ii.ravel(); jj = jj.ravel()
        pix = ii*n+jj
        g = self.good.ravel()[pix]
        pix = pix[g]; ii = ii[g]; jj = jj[g]
        A = self.Jinv @ self.Jloc(x, y)
        dx = jj-x; dy = ii-y
        qx = A[0, 0]*dx+A[0, 1]*dy; qy = A[1, 0]*dx+A[1, 1]*dy
        return pix, qx, qy, A


def bspline_basis(n, spacing):
    k = 3
    nint = int(np.ceil(n/spacing))
    t = np.concatenate([[0]*k, np.linspace(0, n-1, nint+1), [n-1]*k]).astype(float)
    x = np.arange(n, dtype=float)
    Bx = BSpline.design_matrix(x, t, k).tocsr()
    return Bx


def bkg_matrix(n, spacing, pix):
    B1 = bspline_basis(n, spacing)
    ii = pix//n; jj = pix % n
    By = B1[ii]; Bx = B1[jj]
    m = B1.shape[1]
    # row-wise kronecker
    By = By.tocoo(); Bx = Bx.tocsr()
    rows = []; cols = []; vals = []
    Byc = B1[ii].tocsr()
    # each row has 4 nnz in each -> 16
    yi = Byc.indices.reshape(-1, 4); yv = Byc.data.reshape(-1, 4)
    xi = Bx.indices.reshape(-1, 4); xv = Bx.data.reshape(-1, 4)
    cols = (yi[:, :, None]*m + xi[:, None, :]).reshape(len(pix), 16)
    vals = (yv[:, :, None]*xv[:, None, :]).reshape(len(pix), 16)
    rows = np.repeat(np.arange(len(pix)), 16)
    return sp.csr_matrix((vals.ravel(), (rows, cols.ravel())), shape=(len(pix), m*m))
