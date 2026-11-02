"""Anderson & King-style oversampled effective PSF (ePSF): build, fit, evaluate.

Conventions
-----------
* An ePSF here is the PIXEL-INTEGRATED PSF (optics x charge diffusion x IPC x pixel
  box) sampled on an offset grid of ``O`` nodes per native pixel: node k sits at
  offset u = (k - O*R) / O native px from the star centre, so the array is
  (2*O*R + 1)^2.  The model value of a detector pixel whose centre is offset u from
  the star is P(u) -- no further pixel integration.
* Normalisation: sum over nodes / O^2 = 1 within r <= R_norm, so a fitted flux is the
  flux within R_norm native px.
* A star in a stamp: data d_j = f P(x_j - xc, y_j - yc) + b; stamps are stored
  background-subtracted (coarse map) and divided by the peak amplitude, so f and b
  come out in those units and are converted back by the caller.
* P is evaluated with a cubic B-spline (scipy.ndimage.map_coordinates on the
  pre-filtered grid); its gradient by the same spline of np.gradient(P).

The build is the classic iteration: fit each star's (f, xc, yc, b) with the current
P, stack the normalised residuals (d - b)/f - P(u) onto the nearest node with weight
(f / sigma)^2 and a 3-sigma clip per node, add the stacked mean to P, re-centre (the
position/PSF-centroid degeneracy) and re-normalise.  The additive-constant degeneracy
(P + c with b - f c fits every star identically) is fixed by pinning the mean of P in
the outermost annulus of the stamp to a supplied value (``edge_level``).
"""
import numpy as np
from scipy import ndimage


class Grid:
    def __init__(self, R, O):
        self.R, self.O = R, O
        self.n = 2 * O * R + 1
        k = (np.arange(self.n) - O * R) / O
        self.uy, self.ux = np.meshgrid(k, k, indexing='ij')
        self.ur = np.hypot(self.ux, self.uy)


def spline(P):
    return ndimage.spline_filter(P, order=3, mode='constant')


def evaluate(coef, grid, ux, uy, order=3):
    """P at offsets (ux, uy) [native px] from spline coefficients ``coef``."""
    iy = uy * grid.O + grid.O * grid.R
    ix = ux * grid.O + grid.O * grid.R
    return ndimage.map_coordinates(coef, [iy.ravel(), ix.ravel()], order=order, prefilter=False,
                                   mode='constant', cval=0.0).reshape(np.shape(ux))


class EPSF:
    def __init__(self, P, grid):
        self.P, self.g = P, grid
        self.coef = spline(P)
        gy, gx = np.gradient(P)
        self.cgx, self.cgy = spline(gx * grid.O), spline(gy * grid.O)

    def __call__(self, ux, uy):
        return evaluate(self.coef, self.g, ux, uy)

    def grad(self, ux, uy):
        return evaluate(self.cgx, self.g, ux, uy), evaluate(self.cgy, self.g, ux, uy)


def gaussian_epsf(grid, fwhm):
    s = fwhm / 2.3548
    P = np.exp(-grid.ur ** 2 / (2 * s * s))
    return P / (P.sum() / grid.O ** 2)


def normalise(P, grid, r_norm):
    return P / (P[grid.ur <= r_norm].sum() / grid.O ** 2)


def recentre(P, grid, r_c=1.5, niter=3):
    """Shift P so that its intensity-weighted centroid within r_c is at the origin."""
    for _ in range(niter):
        w = np.where(grid.ur <= r_c, np.clip(P, 0, None), 0)
        cx = (w * grid.ux).sum() / w.sum()
        cy = (w * grid.uy).sum() / w.sum()
        if np.hypot(cx, cy) < 1e-4:
            break
        c = spline(P)
        P = evaluate(c, grid, grid.ux + cx, grid.uy + cy)
    return P


def pin_edge(P, grid, edge_level, width=1.5):
    """Remove the additive-constant degeneracy: mean of P in the outer annulus := edge_level."""
    ann = (grid.ur > grid.R - width) & (grid.ur <= grid.R)
    return P - (P[ann].mean() - edge_level)


# --------------------------------------------------------------------------- stars
class Stars:
    """Stamps (N, S, S) with per-pixel sigma and a bad-pixel mask, plus positions."""

    def __init__(self, s, e, m, xc, yc, ix, iy, R):
        self.s = np.asarray(s, dtype=np.float16)
        self.e = np.asarray(e, dtype=np.float16)
        self.m = (np.asarray(m, bool) | ~np.isfinite(self.s) | ~np.isfinite(self.e)
                  | (self.e <= 0) | (self.e > 1e4))
        self.s[self.m] = 0
        self.e[self.m] = 1e4
        self.ix, self.iy = ix.astype(np.float64), iy.astype(np.float64)
        self.R = R
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
        self.px, self.py = xx.astype(np.float64), yy.astype(np.float64)
        self.pr = np.hypot(self.px, self.py)
        N = len(s)
        self.dx = (xc - ix).astype(np.float64)       # star centre relative to stamp centre
        self.dy = (yc - iy).astype(np.float64)
        self.f = np.ones(N)
        self.b = np.zeros(N)
        self.ok = np.ones(N, bool)
        self.chi2 = np.zeros(N)
        # confusion noise: robust rms of the stamp beyond 8 px (unmodelled faint
        # neighbours dominate the per-pixel ERR there in this field), minus the ERR part
        out = (self.pr > min(8, R - 2))[None] & ~self.m
        conf = np.zeros(N)
        for c0 in range(0, N, 2000):
            sl = slice(c0, c0 + 2000)
            v = np.where(out[sl], self.s[sl].astype(np.float64), np.nan)
            mad = 1.4826 * np.nanmedian(np.abs(v - np.nanmedian(v, axis=(1, 2))[:, None, None]), axis=(1, 2))
            ee = np.nanmedian(np.where(out[sl], self.e[sl].astype(np.float64), np.nan), axis=(1, 2))
            conf[sl] = np.sqrt(np.maximum(mad ** 2 - ee ** 2, 0))
        self.conf = conf
        self.model_err = 0.02      # fractional model-error floor in the fit weights

    def __len__(self):
        return len(self.s)

    @property
    def x(self):
        return self.ix + self.dx

    @property
    def y(self):
        return self.iy + self.dy

    def subset(self, idx):
        new = object.__new__(Stars)
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray) and v.shape[:1] == (len(self),):
                setattr(new, k, v[idx])
            else:
                setattr(new, k, v)
        return new


def safe_solve(A, B):
    """Batched solve with a tiny ridge, so a fully-masked star gives 0 not an error."""
    k = A.shape[-1]
    tr = np.trace(A, axis1=1, axis2=2) / k
    A = A + np.eye(k)[None] * (1e-10 * tr + 1e-300)[:, None, None]
    return np.linalg.solve(A, B[..., None])[..., 0]


def fit_stars(stars, psf, idx=None, r_fit=None, niter=4, fit_pos=True, clip=5.0, r_pos=8.0):
    """Fit (f, dx, dy, b) for stars[idx] with ePSF ``psf`` (an EPSF, or a list of
    (EPSF, per-star weight) pairs = spatial interpolation).

    1. weighted Gauss-Newton for (f, dx, dy, b) on the pixels within ``r_pos`` (the
       position is set by the core; this keeps a 61x61 wing stamp cheap), with
       pixels beyond ``clip`` sigma dropped from the 3rd iteration on;
    2. linear re-fit of (f, b) at that position over all pixels within ``r_fit``
       (default: the whole stamp), again clipped.
    Weights: 1 / (ERR^2 + confusion^2 + (0.02 d)^2).
    """
    if idx is None:
        idx = np.arange(len(stars))
    R = stars.R
    rf = R if r_fit is None else r_fit
    hb = int(min(R, np.ceil(r_pos)))
    cs = slice(R - hb, R + hb + 1)
    pxc, pyc, prc = stars.px[cs, cs], stars.py[cs, cs], stars.pr[cs, cs]
    inreg = stars.pr <= rf
    inpos = prc <= min(r_pos, rf)
    chunk = _chunk(stars)
    for c0 in range(0, len(idx), chunk):
        ii = idx[c0:c0 + chunk]
        dfull = stars.s[ii].astype(np.float64)
        var_full = (stars.e[ii].astype(np.float64) ** 2 + stars.conf[ii][:, None, None] ** 2
                    + (stars.model_err * np.maximum(dfull, 0)) ** 2)
        wfull = np.where(stars.m[ii] | ~inreg[None], 0.0, 1.0 / var_full)
        d = dfull[:, cs, cs]
        w0 = np.where(inpos[None], wfull[:, cs, cs], 0.0)
        w = w0
        f, dx, dy, b = stars.f[ii].copy(), stars.dx[ii].copy(), stars.dy[ii].copy(), stars.b[ii].copy()
        for it in range(niter if fit_pos else 0):
            ux = pxc[None] - dx[:, None, None]
            uy = pyc[None] - dy[:, None, None]
            P, gx, gy = model_and_grad(psf, ux, uy, ii)
            res = d - (f[:, None, None] * P + b[:, None, None])
            if it >= 2 and clip:
                w = np.where(np.abs(res) * np.sqrt(w0) > clip, 0.0, w0)
            J = np.stack([P, -f[:, None, None] * gx, -f[:, None, None] * gy, np.ones_like(P)], axis=1)
            Jw = J * w[:, None]
            A = np.einsum('nkij,nlij->nkl', Jw, J)
            B = np.einsum('nkij,nij->nk', Jw, res)
            step = safe_solve(A, B)
            f += step[:, 0]
            dx += np.clip(step[:, 1], -0.3, 0.3)
            dy += np.clip(step[:, 2], -0.3, 0.3)
            b += step[:, 3]
        # linear (f, b) over the full fitting region at the fitted position
        ux = stars.px[None] - dx[:, None, None]
        uy = stars.py[None] - dy[:, None, None]
        P, _, _ = model_and_grad(psf, ux, uy, ii, grad=False)
        w = wfull
        for it in range(2):
            J = np.stack([P, np.ones_like(P)], axis=1)
            Jw = J * w[:, None]
            A = np.einsum('nkij,nlij->nkl', Jw, J)
            B = np.einsum('nkij,nij->nk', Jw, dfull)
            sol = safe_solve(A, B)
            f, b = sol[:, 0], sol[:, 1]
            res = dfull - (f[:, None, None] * P + b[:, None, None])
            if clip:
                w = np.where(np.abs(res) * np.sqrt(wfull) > clip, 0.0, wfull)
        nw = (w > 0).sum(axis=(1, 2))
        stars.chi2[ii] = (res ** 2 * w).sum(axis=(1, 2)) / np.maximum(nw - 4, 1)
        stars.f[ii], stars.dx[ii], stars.dy[ii], stars.b[ii] = f, dx, dy, b
        # stamps are divided by their peak pixel, so a sane fit has f * max(P) ~ 1
        pk = np.nanmax(P, axis=(1, 2))
        stars.ok[ii] = ((np.abs(dx) < 1.0) & (np.abs(dy) < 1.0) & np.isfinite(f)
                        & (f * pk > 0.5) & (f * pk < 2.0))
    return stars


def model_and_grad(psf, ux, uy, ii, grad=True):
    """Evaluate a single EPSF or a spatial model (list of (EPSF, weights[N_all]))."""
    if isinstance(psf, EPSF):
        P = psf(ux, uy)
        if not grad:
            return P, None, None
        gx, gy = psf.grad(ux, uy)
        return P, gx, gy
    P = np.zeros(ux.shape); gx = np.zeros(ux.shape); gy = np.zeros(ux.shape)
    for ep, wt in psf:
        wi = wt[ii]
        sel = np.nonzero(wi > 0)[0]
        if len(sel) == 0:
            continue
        P[sel] += wi[sel, None, None] * ep(ux[sel], uy[sel])
        if grad:
            a, c = ep.grad(ux[sel], uy[sel])
            gx[sel] += wi[sel, None, None] * a
            gy[sel] += wi[sel, None, None] * c
    return P, (gx if grad else None), (gy if grad else None)


def _chunk(stars):
    return max(200, int(2.5e6 / stars.s.shape[1] ** 2))


def _residual_chunks(stars, psf, grid, idx, weights=None):
    n, O, R = grid.n, grid.O, grid.R
    ch = _chunk(stars)
    for c0 in range(0, len(idx), ch):
        ii = idx[c0:c0 + ch]
        f = stars.f[ii][:, None, None]; b = stars.b[ii][:, None, None]
        ux = stars.px[None] - stars.dx[ii][:, None, None]
        uy = stars.py[None] - stars.dy[ii][:, None, None]
        P, _, _ = model_and_grad(psf, ux, uy, ii, grad=False)
        e = stars.e[ii].astype(np.float64)
        r = (stars.s[ii].astype(np.float64) - b) / f - P
        # equal weight per sample (plain clipped mean, as Anderson & King): the per-pixel
        # ERR is not the error budget in a confusion-limited field
        w = np.ones_like(r)
        if weights is not None:
            w = w * weights[ii][:, None, None]
        kx = np.round(ux * O).astype(int) + O * R
        ky = np.round(uy * O).astype(int) + O * R
        good = (~stars.m[ii]) & (kx >= 0) & (kx < n) & (ky >= 0) & (ky < n) & (w > 0)
        yield r[good], (ky * n + kx)[good], w[good]


def stack_residuals(stars, psf, grid, idx, clip=3.0, weights=None):
    """Clipped mean of the normalised residuals on the nearest ePSF node.  Two chunked
    passes: (1) mean and scatter per node from running sums, (2) mean of the samples
    within ``clip`` x scatter.  Returns (mean_residual[n,n], n_samples[n,n], var[n,n]).
    """
    nn = grid.n ** 2
    ws = np.zeros(nn); wv = np.zeros(nn); wv2 = np.zeros(nn); c = np.zeros(nn)
    for v, k, w in _residual_chunks(stars, psf, grid, idx, weights):
        ws += np.bincount(k, w, nn); wv += np.bincount(k, w * v, nn); wv2 += np.bincount(k, w * v * v, nn)
        c += np.bincount(k, None, nn)
    mu = wv / np.maximum(ws, 1e-300)
    s2 = np.maximum(wv2 / np.maximum(ws, 1e-300) - mu ** 2, 0)
    thr2 = clip ** 2 * np.maximum(s2, 1e-30)
    ws = np.zeros(nn); wv = np.zeros(nn); c = np.zeros(nn)
    for v, k, w in _residual_chunks(stars, psf, grid, idx, weights):
        keep = (v - mu[k]) ** 2 < thr2[k]
        ws += np.bincount(k[keep], w[keep], nn); wv += np.bincount(k[keep], (w * v)[keep], nn)
        c += np.bincount(k[keep], None, nn)
    mu = wv / np.maximum(ws, 1e-300)
    n = grid.n
    return mu.reshape(n, n), c.reshape(n, n), s2.reshape(n, n)


def build(stars, grid, idx, P0, r_norm, edge_level=0.0, niter=8, r_fit=None, weights=None, verbose=False,
          fit_first=True, min_samples=5):
    """Iterate fit / stack / update.  Returns (P, history)."""
    P = P0.copy()
    hist = []
    for it in range(niter):
        ep = EPSF(P, grid)
        if fit_first or it > 0:
            fit_stars(stars, ep, idx, r_fit=r_fit)
        use = idx[stars.ok[idx] & (stars.chi2[idx] < np.percentile(stars.chi2[idx][stars.ok[idx]], 95))]
        mu, cnt, s2 = stack_residuals(stars, ep, grid, use, weights=weights)
        mu = np.where(cnt >= min_samples, mu, 0.0)
        Pn = P + mu
        Pn = recentre(Pn, grid)
        Pn = pin_edge(Pn, grid, edge_level)
        Pn = normalise(Pn, grid, r_norm)
        dP = np.abs(Pn - P).max() / Pn.max()
        P = Pn
        hist.append(dict(it=it, dPmax=dP, nstar=len(use), med_chi2=float(np.median(stars.chi2[use]))))
        if verbose:
            print(hist[-1], flush=True)
    return P, hist


# --------------------------------------------------------------------------- STPSF
def stpsf_to_epsf(overdist, O, R):
    """STPSF OVERDIST (oversampled by O, distortion + charge diffusion + IPC applied) ->
    ePSF on this module's grid: convolve with the native-pixel box (exact, in Fourier
    space: sinc for a box O nodes wide) and crop to (2 O R + 1)^2 about the centre.

    STPSF's oversampled array has an even number of nodes (O * fov_pixels) with the
    star centred between the four central nodes when fov_pixels is odd... it is
    centred on a node boundary; we resample it onto node-centred offsets by a
    half-node spline shift before the crop.
    """
    n = overdist.shape[0]
    ft = np.fft.fft2(overdist)
    k = np.fft.fftfreq(n)
    box = np.sinc(k * O)
    conv = np.real(np.fft.ifft2(ft * box[:, None] * box[None, :]))
    # STPSF: centre of the array (n/2 - 0.5 in 0-based node index) is the star.
    c = (n - 1) / 2.0
    g = Grid(R, O)
    iy = g.uy * O + c
    ix = g.ux * O + c
    P = ndimage.map_coordinates(conv, [iy.ravel(), ix.ravel()], order=3, mode='constant').reshape(g.n, g.n)
    return P


# --------------------------------------------------------------------------- metrics
def radial_profile(P, grid, rbins):
    r = grid.ur
    return np.array([P[(r >= a) & (r < b)].mean() for a, b in zip(rbins[:-1], rbins[1:])])


def encircled(P, grid, radii):
    return np.array([P[grid.ur <= r].sum() / grid.O ** 2 for r in radii])


def fwhm_halfmax(P, grid, up=4):
    """FWHM from the area above half maximum (on a spline-upsampled grid)."""
    c = spline(P)
    h = 3.0
    k = np.arange(-h, h + 1e-9, 1.0 / (grid.O * up))
    uy, ux = np.meshgrid(k, k, indexing='ij')
    v = evaluate(c, grid, ux, uy)
    area = (v >= v.max() / 2).sum() * (1.0 / (grid.O * up)) ** 2
    return 2 * np.sqrt(area / np.pi)


def moments(P, grid, sigw):
    """Gaussian-weighted second moments -> (size, e1, e2, e, theta_deg)."""
    w = np.exp(-grid.ur ** 2 / (2 * sigw ** 2)) * P
    s = w.sum()
    mx = (w * grid.ux).sum() / s; my = (w * grid.uy).sum() / s
    qxx = (w * (grid.ux - mx) ** 2).sum() / s
    qyy = (w * (grid.uy - my) ** 2).sum() / s
    qxy = (w * (grid.ux - mx) * (grid.uy - my)).sum() / s
    t = qxx + qyy
    e1 = (qxx - qyy) / t; e2 = 2 * qxy / t
    return np.sqrt(t), e1, e2, np.hypot(e1, e2), 0.5 * np.degrees(np.arctan2(e2, e1))
