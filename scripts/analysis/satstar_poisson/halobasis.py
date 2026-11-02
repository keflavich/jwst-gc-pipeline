import numpy as np, scipy.sparse as sp
from scipy.interpolate import BSpline


def halo_basis(qx, qy, rmin=20., rmax=720., nknot=14, mmax=4):
    """sparse (npix x nb) basis: cubic B-splines in log r x [1, cos m th, sin m th],
    zero outside rmin<=r<=rmax (only in-range entries are stored)."""
    r = np.hypot(qx, qy)
    inside = np.flatnonzero((r >= rmin) & (r <= rmax))
    k = 3
    t = np.concatenate([[np.log(rmin)]*k, np.linspace(np.log(rmin), np.log(rmax), nknot), [np.log(rmax)]*k])
    nr = len(t)-k-1
    na = 1+2*mmax
    if len(inside) == 0:
        return sp.csr_matrix((len(r), nr*na))
    ri = r[inside]; th = np.arctan2(qy[inside], qx[inside])
    B = BSpline.design_matrix(np.log(ri), t, k).tocsr()
    ang = [np.ones_like(th)]
    for m in range(1, mmax+1):
        ang += [np.cos(m*th), np.sin(m*th)]
    ang = np.stack(ang, 1)
    bi = B.indices.reshape(-1, 4); bv = B.data.reshape(-1, 4)
    cols = (bi[:, :, None]*na+np.arange(na)[None, None, :]).reshape(len(ri), -1)
    vals = (bv[:, :, None]*ang[:, None, :]).reshape(len(ri), -1)
    rows = np.repeat(inside, 4*na)
    return sp.csr_matrix((vals.ravel(), (rows, cols.ravel())), shape=(len(r), nr*na))
