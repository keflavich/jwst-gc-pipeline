"""Transdimensional (RJMCMC) point-source sampler in the PCAT scheme.

This is a new Python 3 implementation of the probabilistic cataloger of
Portillo et al. (2017, AJ 154, 132; "Lion") and Feder et al. (2020, AJ 159,
163; PCAT-DE, https://github.com/RichardFeder/pcat-de, MIT).  It keeps PCAT's
proposal set and its parallel, checkerboard-region acceptance, renders sources
with the PCAT C kernel (``_vendor/pcat_lion_kernel.c``) and adds:

* acceptance ratios that satisfy detailed balance region by region:
  per-region birth density ``1/A_r`` and death choice ``1/(n_r+1)``, split /
  merge pair choice restricted to the region, the Jacobian of the asinh flux
  proposal, per-star flux-prior factors summed per region (PCAT's
  ``dlogP[regiony, regionx] += factor`` keeps one star per region when several
  share it), and moves that keep a star in its region (the reverse move then
  moves the same star set);
* exact area bookkeeping for partial regions, so the image size need not be a
  multiple of ``regsize`` and edges are not penalised;
* region acceptance applied sequentially, so the count-prior term uses the
  current number of sources, and an independent birth/death (split/merge)
  coin per region, so each region's kernel is a reversible mixture;
* Gibbs updates of the linear background (constant + optional Fourier modes,
  Gaussian amplitude prior) and, optionally, of the flux-function slope;
* several bands on different pixel grids (affine pixel maps with optional
  second-order terms) with Gaussian colour priors; optional colour-space
  star moves (``color_moves``), a split
  colour fraction scaled to the colour prior (``split_color_scale``) and a
  parsimony parameter count (``pen_nparams``) for joint fits of several
  exposures of one filter, where a tight colour prior ties the fluxes.

Coordinates are 0-based pixel coordinates of the pivot-band image (pixel
centres at integers).  Fluxes are total fluxes in image units x pixel (the
ePSF convention of :mod:`.psf`).

Target posterior
----------------
``P(N) ~ exp(-pen*N)`` with ``pen = (2 + nbands)/2 * alpha_parsimony`` (PCAT;
``pen = pen_nparams/2 * alpha_parsimony`` when ``pen_nparams`` is set), positions
uniform over the domain ``D`` (the image footprint, optionally padded by ``domain_pad``), pivot fluxes ``p(f) = (s-1)/fmin (f/fmin)^-s`` for
``f >= fmin = trueminf``, colours ``-2.5 log10(f_b/f_0) ~ N(mu_b, sig_b)`` and
a Gaussian likelihood with the given inverse-variance weights.
``count_prior='legacy'`` reproduces PCAT's count terms (``-pen`` per birth,
``-ln|D|`` per split, no region-count terms) for comparison.
"""
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence

import numpy as np
from scipy import stats

from . import kernel as _kernel
from .psf_poly import build_cf, load_epsf

__all__ = ['BandData', 'PCATConfig', 'PCATResult', 'PCATSampler', 'run_pcat',
           'fourier_templates']

_TYPES = ('movestar', 'birth_death', 'merge_split', 'bkg')


# ----------------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------------
@dataclass
class BandData:
    """One band: image, inverse-variance weight and PSF table.

    ``affine`` maps pivot pixel coordinates to this band's pixel coordinates,
    ``[x_b, y_b] = M @ [x, y] + t`` given as the 2x3 array ``[M | t]``; ``None``
    is the identity (required for the pivot band, index 0).  ``quad`` (optional,
    2x3) adds second-order terms, ``[x_b, y_b] += quad @ [x^2, x y, y^2]`` (pivot
    pixels), for detectors whose relative distortion is not affine over the image.
    """
    image: np.ndarray
    weight: np.ndarray
    cf: np.ndarray
    nc: int
    basis: str = 'quartic15'
    affine: Optional[np.ndarray] = None
    name: str = 'band'
    quad: Optional[np.ndarray] = None


@dataclass
class PCATConfig:
    trueminf: float = 1.0               # pivot-band flux floor (image units * pix)
    flux_slope: float = 2.0             # power-law slope s of p(f) ~ f^-s
    sample_slope: bool = False          # Gibbs-sample s (flat prior on slope_bounds)
    slope_bounds: tuple = (1.05, 4.0)
    alpha_parsimony: float = 1.0        # pen = (2+nbands)/2 * alpha
    count_prior: str = 'exact'          # 'exact' | 'legacy'
    regsize: int = 40
    margin: Optional[int] = None        # default: PSF half-width + 1 (pivot px)
    nsamp: int = 100
    nloop: int = 1000
    burnin_frac: float = 0.5
    max_nsrc: int = 20000
    init_catalog: Optional[dict] = None  # {'x','y','f' (nb, n) or (n,)}
    # defaults from the F182M cutout validation suite (runs/pcat/validation):
    # step factors 5/5 and birth/death weight 60 give movestar acceptance
    # ~0.3-0.5 and 2-5x shorter tau_int than 1/1 (recommended_config.json)
    move_weights: dict = field(default_factory=lambda: dict(
        movestar=60., birth_death=60., merge_split=30., bkg=5.))
    kickrange: float = 1.0              # split separation sigma (pivot px)
    err_f_divfac: float = 2.0
    pos_step_fac: float = 5.0
    flux_step_fac: float = 5.0
    noise_scale: object = 1.0           # sigma_true / sigma_weight (scalar or per band):
                                        # the sampler uses weight / noise_scale**2
    logdf: float = 0.01
    bkg: Optional[Sequence[float]] = None   # initial constant background per band
    float_bkg: bool = True
    bkg_prior_mu: Optional[Sequence[float]] = None
    bkg_prior_sig: Optional[Sequence[float]] = None   # None: flat
    fourier_order: int = 0              # n -> 4 n^2 templates per band
    fourier_prior_sig: float = 1.0      # Gaussian amplitude prior (image units)
    color_mus: Optional[Sequence[float]] = None
    color_sigs: Optional[Sequence[float]] = None
    split_color_sig: float = 0.05
    # multi-band / multi-exposure options (defaults reproduce the plain sampler exactly):
    # color_moves: movestar proposes (pivot flux, colours) instead of independent band
    #   fluxes; colour steps are color_step_fac * min(sig_b, 1.0857 sigma_fb / f_b) /
    #   sqrt(n_eff (2 + nbands)) and the pivot flux / position steps use the flux error of
    #   all bands combined (needed when a tight colour prior ties the band fluxes)
    # split_color_scale: split fraction of band b ~ N(u, split_color_scale * sig_b * u (1-u)
    #   / 1.0857) truncated to (0, 1), which keeps both children within ~split_color_scale
    #   sig_b of the parent colour; None uses split_color_sig
    # pen_nparams: parameters per source in the parsimony penalty (pen = pen_nparams / 2 *
    #   alpha_parsimony); None is PCAT's 2 + nbands (3 makes a joint fit of exposures tied
    #   by a tight colour prior carry the single-image penalty)
    color_moves: bool = False
    color_step_fac: float = 1.0
    split_color_scale: Optional[float] = None
    pen_nparams: Optional[float] = None
    domain_pad: float = 0.0
    seed: int = 0
    blas: str = 'auto'                  # 'auto' | 'openblas' | 'mkl' | 'none' | 'numpy'
    lib_dir: Optional[str] = None
    store_models: bool = True           # keep post-burn-in models (median model)
    check_leak: bool = False            # verify footprints stay inside margins
    verbose: int = 1
    # compiled engine (:class:`.fast_sampler.FastPCATSampler`, one band); the
    # defaults keep the target and the proposals of PCATSampler
    engine: str = 'python'              # 'python' | 'numba'
    stamp_tol: Optional[float] = None   # adaptive stamps: truncated-wing chi^2 per source
    stamp_sizes: tuple = (11, 15, 21, 31, 41)
    move_nstar: int = 0                 # stars moved per region per star move (0: all, PCAT)
    # numba engine: fraction w of births placed with probability proportional to the positive
    # residual (S/N)^2 of a pixel (the rest uniform); the death acceptance carries the reverse
    # density.  0 keeps PCAT's uniform births
    birth_resid_frac: float = 0.0
    region_cap: int = 512               # max sources per region (numba engine)
    burnin: Optional[int] = None        # burn-in samples (numba engine; overrides burnin_frac)
    # numba engine: burn-in = max(burnin, burnin_auto_factor * t_eq), t_eq = first sample at which
    # logL and N reach their second-half ranges (fast_sampler.equilibration_index)
    burnin_auto: bool = False
    burnin_auto_factor: float = 2.0
    target_ess: Optional[float] = None  # numba engine: stop once ESS(N), ESS(logL) after burn-in reach it
    ess_min_samples: int = 200
    ess_check_every: int = 50


@dataclass
class PCATResult:
    samples: dict
    accept: dict
    model_mean: list
    model_median: list
    residual_mean: list
    residual_median: list
    config: dict
    info: dict

    def post_burnin(self):
        """Indices of the samples after burn-in."""
        nb = int(np.floor(self.config['burnin_frac'] * self.samples['n'].size))
        return np.arange(nb, self.samples['n'].size)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def fourier_templates(ny, nx, order):
    """``4 order^2`` sin/cos product templates on an (ny, nx) grid.

    Formula after PCAT-DE ``fourier_bkg_modl.make_fourier_templates``
    (``sin/cos((i+1) pi x / nx) * sin/cos((j+1) pi y / ny)``), with the array
    axes in (y, x) order.  Returns ``(4 order^2, ny, nx)`` float64.
    """
    if order <= 0:
        return np.zeros((0, ny, nx))
    x = np.arange(nx) / nx
    y = np.arange(ny) / ny
    out = []
    for i in range(order):
        sx, cx = np.sin((i + 1) * np.pi * x), np.cos((i + 1) * np.pi * x)
        for j in range(order):
            sy, cy = np.sin((j + 1) * np.pi * y), np.cos((j + 1) * np.pi * y)
            out += [np.outer(sy, sx), np.outer(cy, sx), np.outer(sy, cx), np.outer(cy, cx)]
    return np.array(out)


def _asinh_fwd(f, lindf, logdf):
    return np.log(logdf * logdf * f + logdf * np.sqrt(lindf * lindf + logdf * logdf * f * f)) / logdf


def _asinh_inv(ff, lindf, logdf):
    return np.exp(-logdf * ff) * (-lindf * lindf * logdf * logdf + np.exp(2 * logdf * ff)) / (2 * logdf * logdf)


def flux_proposal(f0, fmin, lindf, logdf, rng, step_fac=1.0):
    """PCAT asinh-space flux proposal with reflection at ``fmin``.

    Returns ``(pf, log_q_ratio)`` with ``log_q_ratio = ln q(f0|pf) - ln q(pf|f0)``
    (the Jacobian of the transform; PCAT omits it).
    """
    ff = _asinh_fwd(f0, lindf, logdf)
    ffmin = _asinh_fwd(fmin, lindf, logdf)
    dff = rng.standard_normal(f0.shape) * step_fac
    above = ff - ffmin
    oob = -dff > above
    dff[oob] = -2 * above[oob] - dff[oob]
    pf = _asinh_inv(ff + dff, lindf, logdf)
    pf = np.maximum(pf, fmin)
    lq = 0.5 * np.log(lindf ** 2 + logdf ** 2 * pf ** 2) - 0.5 * np.log(lindf ** 2 + logdf ** 2 * f0 ** 2)
    return pf, lq


def _qterms(x, y):
    """second-order terms ``[x^2, x y, y^2]`` (3, n) of a quadratic pixel map."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    return np.vstack([x * x, x * y, y * y])


def _truncnorm_logpdf(u, mu, sig):
    """log density of N(mu, sig) truncated to (0, 1) at u."""
    a, b = (0.0 - mu) / sig, (1.0 - mu) / sig
    return stats.truncnorm.logpdf(u, a, b, loc=mu, scale=sig)


def _truncnorm_rvs(mu, sig, rng):
    a, b = (0.0 - mu) / sig, (1.0 - mu) / sig
    return stats.truncnorm.rvs(a, b, loc=mu, scale=sig, random_state=rng)


# ----------------------------------------------------------------------------
# sampler
# ----------------------------------------------------------------------------
class PCATSampler:
    def __init__(self, bands, config):
        self.cfg = cfg = config
        self.rng = np.random.default_rng(cfg.seed)
        self.bands = list(bands)
        self.nb = nb = len(self.bands)
        if cfg.count_prior not in ('exact', 'legacy'):
            raise ValueError(f'count_prior {cfg.count_prior!r}')
        lib = None
        if cfg.blas != 'numpy':
            lib = _kernel.load_kernel(cfg.blas, lib_dir=cfg.lib_dir)
        self.lib = lib
        b0 = self.bands[0]
        if b0.affine is not None and not np.allclose(b0.affine, [[1, 0, 0], [0, 1, 0]]):
            raise ValueError('pivot band affine must be the identity')
        self.ny0, self.nx0 = b0.image.shape
        dp = float(cfg.domain_pad)
        self.dpad = dp
        self.Lx = self.nx0 + 2 * dp
        self.Ly = self.ny0 + 2 * dp
        self.area = self.Lx * self.Ly
        npen = (2.0 + nb) if cfg.pen_nparams is None else float(cfg.pen_nparams)
        self.pen = npen / 2.0 * cfg.alpha_parsimony
        self.fmin = float(cfg.trueminf)
        self.slope = float(cfg.flux_slope)
        self.R = int(cfg.regsize)
        ns_ = np.broadcast_to(np.asarray(cfg.noise_scale, float), (nb,)).copy()
        if not np.all(np.isfinite(ns_) & (ns_ > 0)):
            raise ValueError(f'noise_scale must be positive, got {cfg.noise_scale!r}')
        self.noise_scale = ns_

        # ---- per-band setup
        self.renderers, self.pads, self.maps = [], [], []
        self.quads = []
        self.data, self.wts, self.model, self.resid = [], [], [], []
        self.unpad = []
        self.errf = []
        margin_req = 0.0
        for b, bd in enumerate(self.bands):
            terms = _kernel.basis_terms(bd.basis)
            r = _kernel.PSFRenderer(bd.cf, bd.nc, terms, lib=lib)
            self.renderers.append(r)
            if bd.affine is None:
                M = np.eye(2)
                t = np.zeros(2)
            else:
                A = np.asarray(bd.affine, dtype=np.float64)
                M, t = A[:, :2], A[:, 2]
            Minv = np.linalg.inv(M)
            self.maps.append((M, t, Minv))
            Q = None if bd.quad is None else np.asarray(bd.quad, dtype=np.float64).reshape(2, 3)
            if b == 0 and Q is not None:
                raise ValueError('pivot band quad must be None')
            self.quads.append(Q)
            ny, nx = bd.image.shape
            # extent of the pivot domain in band pixels
            cx = np.array([-0.5 - dp, self.nx0 - 0.5 + dp])
            cy = np.array([-0.5 - dp, self.ny0 - 0.5 + dp])
            corners = np.array([[x, y] for x in cx for y in cy]).T
            bc = M @ corners + t[:, None]
            if Q is not None:
                bc = bc + Q @ _qterms(corners[0], corners[1])
            ext = max(0.0, -bc[0].min(), -bc[1].min(), bc[0].max() - (nx - 1), bc[1].max() - (ny - 1))
            P = r.rad + 3 + int(np.ceil(ext))
            self.pads.append(P)
            img = np.zeros((ny + 2 * P, nx + 2 * P), np.float32)
            w = np.zeros_like(img)
            good = np.isfinite(bd.image) & np.isfinite(bd.weight) & (bd.weight > 0)
            img[P:P + ny, P:P + nx] = np.where(good, bd.image, 0.0)
            w[P:P + ny, P:P + nx] = np.where(good, bd.weight, 0.0) / self.noise_scale[b] ** 2
            self.data.append(img)
            self.wts.append(w)
            self.unpad.append((slice(P, P + ny), slice(P, P + nx)))
            # footprint half-width in pivot px (stamp +-(rad+1) band px)
            req = (r.rad + 1) * np.abs(Minv).sum(axis=1).max()
            if Q is not None:
                # local Jacobian of the quadratic map at the domain corners
                for xc, yc in corners.T:
                    J = M + np.array([[2 * Q[0, 0] * xc + Q[0, 1] * yc, Q[0, 1] * xc + 2 * Q[0, 2] * yc],
                                      [2 * Q[1, 0] * xc + Q[1, 1] * yc, Q[1, 1] * xc + 2 * Q[1, 2] * yc]])
                    req = max(req, (r.rad + 1) * np.abs(np.linalg.inv(J)).sum(axis=1).max())
            margin_req = max(margin_req, req)
            # flux error scale: sigma_f = sqrt(var / sum P^2) (PSF-fit noise)
            from .psf_poly import render_stamp_poly
            st = render_stamp_poly(bd.cf, bd.nc, terms, 0.0, 0.0)
            neff = 1.0 / np.sum(st ** 2)
            var = np.median(1.0 / w[w > 0]) if np.any(w > 0) else 1.0
            self.errf.append(np.sqrt(neff * var) / cfg.err_f_divfac)
        self.margin_req = int(np.ceil(margin_req - 1e-9))
        self.margin = int(cfg.margin) if cfg.margin is not None else self.margin_req
        if self.margin < self.margin_req:
            raise ValueError(f'margin {self.margin} < required {self.margin_req}')
        if 2 * self.margin > self.R:
            raise ValueError(f'regsize {self.R} < 2*margin {2*self.margin}')

        # pivot coordinates of every padded band pixel (for region labels)
        self.pix_piv = []
        for b, bd in enumerate(self.bands):
            P = self.pads[b]
            ny, nx = self.data[b].shape
            M, t, Minv = self.maps[b]
            qy, qx = np.mgrid[0:ny, 0:nx]
            bx = qx.ravel() - P
            by = qy.ravel() - P
            pv = Minv @ (np.vstack([bx, by]) - t[:, None])
            Q = self.quads[b]
            if Q is not None:
                # fixed-point inverse of the quadratic map (second-order terms << 1)
                for _ in range(4):
                    pv = Minv @ (np.vstack([bx, by]) - t[:, None] - Q @ _qterms(pv[0], pv[1]))
            sx = (pv[0] + 0.5 + dp).reshape(ny, nx)
            sy = (pv[1] + 0.5 + dp).reshape(ny, nx)
            self.pix_piv.append((sx, sy))

        # ---- background templates (unpadded band grids)
        self.ntemp = 1 + 4 * cfg.fourier_order ** 2
        self.T, self.TtWT = [], []
        for b, bd in enumerate(self.bands):
            ny, nx = bd.image.shape
            ft = fourier_templates(ny, nx, cfg.fourier_order)
            T = np.concatenate([np.ones((1, ny, nx)), ft], axis=0)
            Tm = T.reshape(self.ntemp, -1)
            w = self.wts[b][self.unpad[b]].ravel().astype(np.float64)
            self.T.append(Tm)
            self.TtWT.append((Tm * w) @ Tm.T)
        bk0 = np.zeros(nb) if cfg.bkg is None else np.broadcast_to(np.asarray(cfg.bkg, float), (nb,)).copy()
        self.beta = np.zeros((nb, self.ntemp))
        self.beta[:, 0] = bk0
        prec = np.zeros((nb, self.ntemp))
        mu = np.zeros((nb, self.ntemp))
        if cfg.bkg_prior_sig is not None:
            sig = np.broadcast_to(np.asarray(cfg.bkg_prior_sig, float), (nb,))
            prec[:, 0] = 1.0 / sig ** 2
            if cfg.bkg_prior_mu is not None:
                mu[:, 0] = np.broadcast_to(np.asarray(cfg.bkg_prior_mu, float), (nb,))
        if cfg.fourier_order > 0:
            prec[:, 1:] = 1.0 / cfg.fourier_prior_sig ** 2
        self.beta_prec, self.beta_mu = prec, mu

        # colour priors
        if nb > 1:
            if cfg.color_mus is None or cfg.color_sigs is None:
                raise ValueError('color_mus and color_sigs are required for nbands > 1')
            self.cmu = np.asarray(cfg.color_mus, float)
            self.csig = np.asarray(cfg.color_sigs, float)
        else:
            self.cmu = self.csig = np.zeros(0)
        if nb > 1 and (self.cmu.shape != (nb - 1,) or self.csig.shape != (nb - 1,)
                       or not np.all(self.csig > 0)):
            raise ValueError(f'color_mus / color_sigs need {nb - 1} entries with sig > 0')
        if cfg.split_color_scale is not None and not cfg.split_color_scale > 0:
            raise ValueError(f'split_color_scale must be positive, got {cfg.split_color_scale!r}')
        # pivot-flux error of all bands combined (colour moves): sigma^-2 = sum_b (r_b / errf_b)^2
        # with r_b = 10^(-0.4 mu_b) the prior-mean flux ratio of band b to the pivot
        self.rcol = np.concatenate([[1.0], 10 ** (-0.4 * self.cmu)])
        self.errf_eff = float(np.sum((self.rcol / np.asarray(self.errf)) ** 2) ** -0.5)
        self.color_moves = bool(cfg.color_moves) and nb > 1

        # ---- state
        if cfg.init_catalog is not None:
            ic = cfg.init_catalog
            self.x = np.asarray(ic['x'], float).copy()
            self.y = np.asarray(ic['y'], float).copy()
            f = np.asarray(ic['f'], float)
            self.f = (f[None, :] if f.ndim == 1 else f).copy()
            if self.f.shape[0] != nb:
                raise ValueError('init_catalog f must have one row per band')
            keep = self._in_domain(self.x, self.y) & (self.f[0] >= self.fmin) & np.all(self.f > 0, axis=0)
            self.x, self.y, self.f = self.x[keep], self.y[keep], self.f[:, keep]
        else:
            self.x = np.zeros(0)
            self.y = np.zeros(0)
            self.f = np.zeros((nb, 0))
        self.stats = {t: dict(nprop=0, nreg=0, nacc=0, time=0.0) for t in _TYPES}
        self.sub = {k: [0, 0] for k in ('birth', 'death', 'split', 'merge')}
        self.leak = 0.0
        self._rebuild_model()

    # ------------------------------------------------------------------ geometry
    def _in_domain(self, x, y):
        sx = x + 0.5 + self.dpad
        sy = y + 0.5 + self.dpad
        return (sx >= 0) & (sx < self.Lx) & (sy >= 0) & (sy < self.Ly)

    def _set_offset(self, ox, oy):
        R = self.R
        self.ox, self.oy = int(ox), int(oy)
        self.nregx = int(np.ceil((self.Lx + ox) / R))
        self.nregy = int(np.ceil((self.Ly + oy) / R))
        i = np.arange(self.nregx)
        j = np.arange(self.nregy)
        self.rx0 = np.maximum(i * R - ox, 0.0)
        self.rx1 = np.minimum((i + 1) * R - ox, self.Lx)
        self.ry0 = np.maximum(j * R - oy, 0.0)
        self.ry1 = np.minimum((j + 1) * R - oy, self.Ly)
        self.rarea = np.outer(self.ry1 - self.ry0, self.rx1 - self.rx0).ravel()
        self.nlab = self.nregx * self.nregy
        self.labels = {}
        m = self.margin
        for px in (0, 1):
            for py in (0, 1):
                labs = []
                for b in range(self.nb):
                    sx, sy = self.pix_piv[b]
                    col = self._ext_index(sx, ox, px, self.nregx, m)
                    row = self._ext_index(sy, oy, py, self.nregy, m)
                    lab = np.where((col >= 0) & (row >= 0), row * self.nregx + col, -1)
                    labs.append(np.ascontiguousarray(lab.astype(np.int32)))
                self.labels[(px, py)] = labs

    def _ext_index(self, s, off, par, nreg, m):
        R = self.R
        t = s + off
        i = np.floor(t / R).astype(np.int64)
        fr = t - i * R
        out = np.full(i.shape, -1, dtype=np.int64)
        on = (i % 2) == par
        out[on] = i[on]
        left = (~on) & (fr < m)          # within margin of region i-1
        out[left] = i[left] - 1
        right = (~on) & (fr >= R - m) & ~left
        out[right] = i[right] + 1
        out[(out < 0) | (out >= nreg)] = -1
        return out

    def _star_regions(self, x, y):
        sx = x + 0.5 + self.dpad
        sy = y + 0.5 + self.dpad
        col = np.floor((sx + self.ox) / self.R).astype(np.int64)
        row = np.floor((sy + self.oy) / self.R).astype(np.int64)
        return col, row

    def _active_regions(self, px, py):
        cols = np.arange(px, self.nregx, 2)
        rows = np.arange(py, self.nregy, 2)
        rr, cc = np.meshgrid(rows, cols, indexing='ij')
        return cc.ravel(), rr.ravel()

    # ------------------------------------------------------------------ model
    def _band_pos(self, b, x, y):
        M, t, _ = self.maps[b]
        P = self.pads[b]
        bx = M[0, 0] * x + M[0, 1] * y + t[0] + P
        by = M[1, 0] * x + M[1, 1] * y + t[1] + P
        Q = self.quads[b]
        if Q is not None:
            xx, xy, yy = x * x, x * y, y * y
            bx = bx + Q[0, 0] * xx + Q[0, 1] * xy + Q[0, 2] * yy
            by = by + Q[1, 0] * xx + Q[1, 1] * xy + Q[1, 2] * yy
        return bx, by

    def _bkg_image(self, b):
        ny, nx = self.bands[b].image.shape
        return (self.beta[b] @ self.T[b]).reshape(ny, nx)

    def _rebuild_model(self):
        self.model, self.resid = [], []
        for b in range(self.nb):
            m = np.zeros_like(self.data[b])
            if self.x.size:
                bx, by = self._band_pos(b, self.x, self.y)
                self.renderers[b].render(bx, by, self.f[b], m)
            m[self.unpad[b]] += self._bkg_image(b).astype(np.float32)
            self.model.append(m)
            self.resid.append((self.data[b] - m) * (self.wts[b] > 0))

    def _render_delta(self, xs, ys, fs):
        """dm per band for phonions (pivot x, y; fs (nb, n))."""
        out = []
        for b in range(self.nb):
            dm = np.zeros_like(self.data[b])
            if xs.size:
                bx, by = self._band_pos(b, xs, ys)
                self.renderers[b].render(bx, by, fs[b], dm)
            out.append(dm)
        return out

    def _dchi2(self, dms, labs):
        tot = np.zeros(self.nlab)
        for b in range(self.nb):
            out = np.zeros(self.nlab + 1)
            dm, res, w, lab = dms[b], self.resid[b], self.wts[b], labs[b]
            if self.lib is not None:
                self.lib.fast_dchi2_regions(dm.size, dm.ctypes.data, res.ctypes.data, w.ctypes.data,
                                            lab.ctypes.data, self.nlab, out.ctypes.data)
            else:
                nz = dm != 0
                l = lab[nz]
                d = dm[nz].astype(np.float64)
                v = w[nz] * (d * d - 2 * res[nz] * d)
                ok = l >= 0
                out[:self.nlab] = np.bincount(l[ok], weights=v[ok], minlength=self.nlab)
                out[self.nlab] = np.sum(np.abs(d[~ok]) * w[nz][~ok])
            tot += out[:self.nlab]
            self.leak += out[self.nlab]
        return tot

    def _apply(self, dms, labs, acc_lab):
        acc = np.ascontiguousarray(acc_lab.astype(np.int32))
        for b in range(self.nb):
            if self.lib is not None:
                self.lib.fast_apply_accept(dms[b].size, dms[b].ctypes.data, labs[b].ctypes.data,
                                           acc.ctypes.data, self.resid[b].ctypes.data,
                                           self.model[b].ctypes.data)
            else:
                lab = labs[b]
                sel = (lab >= 0)
                sel[sel] = acc[lab[sel]] > 0
                self.resid[b][sel] -= dms[b][sel]
                self.model[b][sel] += dms[b][sel]

    # ------------------------------------------------------------------ priors
    def _logp_flux(self, f):
        """log prior of pivot fluxes (power law above fmin)."""
        s = self.slope
        return np.log(s - 1.0) - np.log(self.fmin) - s * np.log(f / self.fmin)

    def _logp_colors(self, f):
        """sum over non-pivot bands of log p(f_b | f_0) (density in f_b)."""
        if self.nb == 1:
            return np.zeros(f.shape[1])
        c = -2.5 * np.log10(f[1:] / f[0][None, :])
        lp = (-0.5 * ((c - self.cmu[:, None]) / self.csig[:, None]) ** 2
              - np.log(self.csig[:, None] * np.sqrt(2 * np.pi))
              + np.log(2.5 / np.log(10.0)) - np.log(f[1:]))
        return lp.sum(axis=0)

    def _draw_fluxes(self, n):
        f = np.empty((self.nb, n))
        f[0] = self.fmin * np.exp(self.rng.exponential(1.0 / (self.slope - 1.0), size=n))
        for b in range(1, self.nb):
            c = self.rng.normal(self.cmu[b - 1], self.csig[b - 1], size=n)
            f[b] = f[0] * 10 ** (-0.4 * c)
        return f

    def _count_term_birth(self, n, n_r, area_r):
        """ln of the count/position factor for adding one source to region r."""
        if self.cfg.count_prior == 'legacy':
            return -self.pen
        return -self.pen + np.log((n + 1) / self.area) + np.log(area_r / (n_r + 1))

    # ------------------------------------------------------------------ proposals
    def _accept_regions(self, dlogL, factor, regs):
        lu = np.log(self.rng.uniform(size=regs.size))
        return lu < dlogL[regs] + factor

    def prop_move(self, px, py):
        x, y, f = self.x, self.y, self.f
        if x.size == 0:
            return 0, 0
        col, row = self._star_regions(x, y)
        act = ((col % 2) == px) & ((row % 2) == py)
        idx = np.flatnonzero(act)
        if idx.size == 0:
            return 0, 0
        n_eff = max(1.0, x.size * self.R ** 2 / self.area)
        cfg = self.cfg
        f0 = f[:, idx]
        pf = np.empty_like(f0)
        lfac = np.zeros(idx.size)
        if self.color_moves:
            # pivot flux (asinh step on the combined-band flux error) and colours
            # c_b' = c_b + N(0, sc_b); sc_b depends on max(f0, f0') only, so the colour step is
            # symmetric; q ratio in (f_0, f_b) space = q_f ratio x prod_b f_b' / f_b
            dnorm = np.sqrt(n_eff * (2 + self.nb))
            lindf = self.errf_eff / dnorm * cfg.flux_step_fac
            logdf = cfg.logdf / np.sqrt(n_eff)
            pf[0], lq = flux_proposal(f0[0], self.fmin, lindf, logdf, self.rng)
            lfac += lq
            fbig = np.maximum(f0[0], pf[0])
            c0 = -2.5 * np.log10(f0[1:] / f0[0][None, :])
            ferr = 1.0857 * (np.asarray(self.errf[1:])[:, None]
                             / (fbig[None, :] * self.rcol[1:, None]))
            sc = cfg.color_step_fac * np.minimum(self.csig[:, None], ferr) / dnorm
            c1 = c0 + self.rng.standard_normal(c0.shape) * sc
            pf[1:] = pf[0][None, :] * 10 ** (-0.4 * c1)
            lfac += np.sum(np.log(pf[1:] / f0[1:]), axis=0)
            errp = self.errf_eff
        else:
            for b in range(self.nb):
                lindf = self.errf[b] / np.sqrt(n_eff * (2 + self.nb)) * cfg.flux_step_fac
                logdf = cfg.logdf / np.sqrt(n_eff)
                fminb = self.fmin if b == 0 else 1e-6 * self.fmin
                pf[b], lq = flux_proposal(f0[b], fminb, lindf, logdf, self.rng)
                lfac += lq
            errp = self.errf[0]
        lfac += self._logp_flux(pf[0]) - self._logp_flux(f0[0])
        lfac += self._logp_colors(pf) - self._logp_colors(f0)
        dpos = (errp / np.sqrt(n_eff * (2 + self.nb)) / np.maximum(f0[0], pf[0])
                * cfg.pos_step_fac)
        dpos = np.maximum(dpos, 1e-3)
        nx_ = x[idx] + self.rng.standard_normal(idx.size) * dpos
        ny_ = y[idx] + self.rng.standard_normal(idx.size) * dpos
        c2, r2 = self._star_regions(nx_, ny_)
        stay = (c2 == col[idx]) & (r2 == row[idx]) & self._in_domain(nx_, ny_)
        nx_ = np.where(stay, nx_, x[idx])
        ny_ = np.where(stay, ny_, y[idx])
        xs = np.concatenate([x[idx], nx_])
        ys = np.concatenate([y[idx], ny_])
        fs = np.concatenate([-f0, pf], axis=1)
        dms = self._render_delta(xs, ys, fs)
        labs = self.labels[(px, py)]
        dlogL = -0.5 * self._dchi2(dms, labs)
        lab_star = row[idx] * self.nregx + col[idx]
        regs = np.unique(lab_star)
        fac = np.zeros(self.nlab)
        np.add.at(fac, lab_star, lfac)
        acc = self._accept_regions(dlogL, fac[regs], regs)
        acc_lab = np.zeros(self.nlab, bool)
        acc_lab[regs[acc]] = True
        self._apply(dms, labs, acc_lab)
        ok = acc_lab[lab_star]
        sel = idx[ok]
        self.x[sel] = nx_[ok]
        self.y[sel] = ny_[ok]
        self.f[:, sel] = pf[:, ok]
        return regs.size, int(acc.sum())

    def _sub(self, key, nprop, nacc):
        s = self.sub[key]
        s[0] += nprop
        s[1] += nacc

    def prop_birth_death(self, px, py):
        """Birth or death in every active region (independent coin per region).

        A per-region coin keeps every region's kernel a reversible birth/death
        mixture; a single coin for all regions (PCAT) composes birth-only (or
        death-only) kernels, which leaves the target invariant only to first
        order.
        """
        cc, rr = self._active_regions(px, py)
        regs_all = rr * self.nregx + cc
        col, row = self._star_regions(self.x, self.y)
        lab_star = row * self.nregx + col
        counts = (np.bincount(lab_star, minlength=self.nlab) if self.x.size
                  else np.zeros(self.nlab, int))
        labs = self.labels[(px, py)]
        coin = self.rng.uniform(size=regs_all.size) < 0.5
        # births
        breg = regs_all[coin]
        bcc, brr = cc[coin], rr[coin]
        nbirth = breg.size
        bx = self.rx0[bcc] + self.rng.uniform(size=nbirth) * (self.rx1[bcc] - self.rx0[bcc]) - 0.5 - self.dpad
        by = self.ry0[brr] + self.rng.uniform(size=nbirth) * (self.ry1[brr] - self.ry0[brr]) - 0.5 - self.dpad
        bf = self._draw_fluxes(nbirth)
        # deaths
        dreg = regs_all[~coin]
        dreg = dreg[counts[dreg] > 0]
        kill = np.empty(dreg.size, dtype=np.int64)
        for k, r in enumerate(dreg):
            members = np.flatnonzero(lab_star == r)
            kill[k] = members[self.rng.integers(members.size)]
        if nbirth + dreg.size == 0:
            return 0, 0
        xs = np.concatenate([bx, self.x[kill]])
        ys = np.concatenate([by, self.y[kill]])
        fs = np.concatenate([bf, -self.f[:, kill]], axis=1)
        dms = self._render_delta(xs, ys, fs)
        dlogL = -0.5 * self._dchi2(dms, labs)
        nprop = nbirth + dreg.size
        regs = np.concatenate([breg, dreg])
        isbirth = np.arange(nprop) < nbirth
        order = self.rng.permutation(nprop)
        lu = np.log(self.rng.uniform(size=nprop))
        acc = np.zeros(nprop, bool)
        n = self.x.size
        for k in order:
            r = regs[k]
            if isbirth[k]:
                if n >= self.cfg.max_nsrc:
                    continue
                fac = self._count_term_birth(n, counts[r], self.rarea[r])
            else:
                fac = -self._count_term_birth(n - 1, counts[r] - 1, self.rarea[r])
            if lu[k] < dlogL[r] + fac:
                acc[k] = True
                n += 1 if isbirth[k] else -1
        acc_lab = np.zeros(self.nlab, bool)
        acc_lab[regs[acc]] = True
        self._apply(dms, labs, acc_lab)
        bacc = acc[:nbirth]
        dacc = acc[nbirth:]
        self._sub('birth', nbirth, int(bacc.sum()))
        self._sub('death', dreg.size, int(dacc.sum()))
        if dacc.any():
            keep = np.ones(self.x.size, bool)
            keep[kill[dacc]] = False
            self.x, self.y, self.f = self.x[keep], self.y[keep], self.f[:, keep]
        if bacc.any():
            self.x = np.concatenate([self.x, bx[bacc]])
            self.y = np.concatenate([self.y, by[bacc]])
            self.f = np.concatenate([self.f, bf[:, bacc]], axis=1)
        return nprop, int(acc.sum())

    def _split_logfactor(self, n_before, nr_after, nbright_before, F, f1, f2, u, ub, log_invpairs):
        """ln acceptance factor (prior x proposal x Jacobian) of a split.

        ``n_before``: total sources before the split; ``nr_after``: sources in
        the region after it; ``nbright_before``: splittable sources in the
        region before it; ``F`` (nb,) parent fluxes; ``f1``, ``f2`` children;
        ``u`` pivot fraction; ``ub`` (nb-1,) non-pivot fractions;
        ``log_invpairs = ln[(1/W_1 + 1/W_2)/2]`` over the post-split region
        (computed in log space: for a distant pair ``W`` underflows).
        """
        kick = self.cfg.kickrange
        lf = (self._logp_flux(np.array([f1[0]]))[0] + self._logp_flux(np.array([f2[0]]))[0]
              - self._logp_flux(np.array([F[0]]))[0])
        if self.nb > 1:
            lf += (self._logp_colors(f1[:, None])[0] + self._logp_colors(f2[:, None])[0]
                   - self._logp_colors(F[:, None])[0])
            lf += np.sum(np.log(F[1:])) - np.sum(_truncnorm_logpdf(ub, u, self._split_sig(u)))
        lp = (lf + log_invpairs + np.log(2 * np.pi * kick * kick) + np.log(nbright_before)
              + np.log(1.0 - 2.0 * self.fmin / F[0]) + np.log(F[0]) - np.log(nr_after))
        if self.cfg.count_prior == 'legacy':
            return lp - self.pen - np.log(self.area) + np.log(nr_after)
        return lp - self.pen + np.log((n_before + 1) / self.area)

    def _split_sig(self, u):
        """sigma of the truncated-normal split fraction of each non-pivot band.

        ``split_color_sig`` (scalar) by default; with ``split_color_scale`` it is
        ``split_color_scale * sig_b * u (1 - u) / 1.0857``, which shifts the two
        children's colours by about ``split_color_scale * sig_b * (1-u)`` and
        ``... * u`` from the parent colour.  It depends on ``u`` only, which a split
        and its reverse merge share.
        """
        if self.cfg.split_color_scale is None:
            return self.cfg.split_color_sig
        return self.cfg.split_color_scale * self.csig * u * (1.0 - u) / 1.0857

    @staticmethod
    def _log_adjacency(xr, yr, i, kick):
        """ln of the merge-pair weights exp(-d^2 / 2 k^2) of source ``i`` to the
        other sources in ``xr, yr`` (-inf for ``i`` itself)."""
        d2 = (xr - xr[i]) ** 2 + (yr - yr[i]) ** 2
        lw = -d2 / (2 * kick * kick)
        lw[i] = -np.inf
        return lw

    @staticmethod
    def _log_invpairs(logW1, logW2):
        """ln[(1/W_1 + 1/W_2)/2] from ln W_1, ln W_2."""
        return np.log(0.5) + np.logaddexp(-logW1, -logW2)

    def prop_merge_split(self, px, py):
        """Split or merge in every active region (independent coin per region)."""
        if self.x.size == 0:
            return 0, 0
        cc, rr = self._active_regions(px, py)
        regs_all = rr * self.nregx + cc
        col, row = self._star_regions(self.x, self.y)
        lab_star = row * self.nregx + col
        labs = self.labels[(px, py)]
        kick = self.cfg.kickrange
        fmin = self.fmin
        coin = self.rng.uniform(size=regs_all.size) < 0.5
        splits, merges = [], []
        for r, c in zip(regs_all, coin):
            members = np.flatnonzero(lab_star == r)
            if c:
                # ---- split proposal
                bright = members[self.f[0, members] > 2 * fmin]
                if bright.size == 0:
                    continue
                s = bright[self.rng.integers(bright.size)]
                F = self.f[:, s].copy()
                u = fmin / F[0] + self.rng.uniform() * (1 - 2 * fmin / F[0])
                ssig = np.broadcast_to(self._split_sig(u), (self.nb - 1,))
                ub = np.array([_truncnorm_rvs(u, ssig[k], self.rng)
                               for k in range(self.nb - 1)])
                fr = np.concatenate([[u], ub])
                dx, dy = self.rng.normal(0, kick, size=2)
                x1, y1 = self.x[s] + (1 - u) * dx, self.y[s] + (1 - u) * dy
                x2, y2 = self.x[s] - u * dx, self.y[s] - u * dy
                xx = np.array([x1, x2])
                yy = np.array([y1, y2])
                c2, r2 = self._star_regions(xx, yy)
                if not (np.all(r2 * self.nregx + c2 == r) and np.all(self._in_domain(xx, yy))):
                    self._sub('split', 1, 0)
                    continue
                f1, f2 = fr * F, (1 - fr) * F
                others = members[members != s]
                xr = np.concatenate([self.x[others], xx])
                yr = np.concatenate([self.y[others], yy])
                i1, i2 = xr.size - 2, xr.size - 1
                lW1 = np.logaddexp.reduce(self._log_adjacency(xr, yr, i1, kick))
                lW2 = np.logaddexp.reduce(self._log_adjacency(xr, yr, i2, kick))
                splits.append((r, s, F, f1, f2, u, ub, x1, y1, x2, y2, members.size + 1,
                               bright.size, self._log_invpairs(lW1, lW2)))
            else:
                # ---- merge proposal
                if members.size < 2:
                    continue
                xr, yr = self.x[members], self.y[members]
                i = self.rng.integers(members.size)
                lwi = self._log_adjacency(xr, yr, i, kick)
                lWi = np.logaddexp.reduce(lwi)
                if not np.isfinite(lWi):
                    self._sub('merge', 1, 0)
                    continue
                pj = np.exp(lwi - lWi)
                j = self.rng.choice(members.size, p=pj / pj.sum())
                lWj = np.logaddexp.reduce(self._log_adjacency(xr, yr, j, kick))
                log_invpairs = self._log_invpairs(lWi, lWj)
                si, sj = members[i], members[j]
                fi, fj = self.f[:, si].copy(), self.f[:, sj].copy()
                F = fi + fj
                u = fi[0] / F[0]
                ub = fi[1:] / F[1:]
                xm = u * self.x[si] + (1 - u) * self.x[sj]
                ym = u * self.y[si] + (1 - u) * self.y[sj]
                nbright_after = int(np.sum(self.f[0, members] > 2 * fmin)
                                    - (fi[0] > 2 * fmin) - (fj[0] > 2 * fmin) + 1)
                merges.append((r, si, sj, F, fi, fj, u, ub, xm, ym, members.size,
                               nbright_after, log_invpairs))
        nsp, nmg = len(splits), len(merges)
        if nsp + nmg == 0:
            return 0, 0
        xs, ys, fs = [], [], []
        for (r, s, F, f1, f2, u, ub, x1, y1, x2, y2, *_ ) in splits:
            xs += [self.x[s], x1, x2]
            ys += [self.y[s], y1, y2]
            fs += [-F, f1, f2]
        for (r, si, sj, F, fi, fj, u, ub, xm, ym, *_ ) in merges:
            xs += [self.x[si], self.x[sj], xm]
            ys += [self.y[si], self.y[sj], ym]
            fs += [-fi, -fj, F]
        dms = self._render_delta(np.array(xs), np.array(ys), np.array(fs).T)
        dlogL = -0.5 * self._dchi2(dms, labs)
        props = splits + merges
        nprop = nsp + nmg
        order = self.rng.permutation(nprop)
        lu = np.log(self.rng.uniform(size=nprop))
        acc = np.zeros(nprop, bool)
        n = self.x.size
        for k in order:
            p = props[k]
            r = p[0]
            if k < nsp:
                (r, s, F, f1, f2, u, ub, x1, y1, x2, y2, nr_after, nbright, log_invpairs) = p
                if n >= self.cfg.max_nsrc:
                    continue
                fac = self._split_logfactor(n, nr_after, nbright, F, f1, f2, u, ub, log_invpairs)
            else:
                (r, si, sj, F, fi, fj, u, ub, xm, ym, nr_before, nbright_after, log_invpairs) = p
                # reverse split starts from n-1 sources with nbright_after splittable ones
                fac = -self._split_logfactor(n - 1, nr_before, nbright_after, F, fi, fj, u, ub,
                                             log_invpairs)
            if lu[k] < dlogL[r] + fac:
                acc[k] = True
                n += 1 if k < nsp else -1
        acc_lab = np.zeros(self.nlab, bool)
        acc_lab[[p[0] for p, a in zip(props, acc) if a]] = True
        self._apply(dms, labs, acc_lab)
        self._sub('split', nsp, int(acc[:nsp].sum()))
        self._sub('merge', nmg, int(acc[nsp:].sum()))
        newx, newy, newf, kill = [], [], [], []
        for k in np.flatnonzero(acc):
            p = props[k]
            if k < nsp:
                (r, s, F, f1, f2, u, ub, x1, y1, x2, y2, *_ ) = p
                self.x[s], self.y[s], self.f[:, s] = x1, y1, f1
                newx.append(x2)
                newy.append(y2)
                newf.append(f2)
            else:
                (r, si, sj, F, fi, fj, u, ub, xm, ym, *_ ) = p
                self.x[si], self.y[si], self.f[:, si] = xm, ym, F
                kill.append(sj)
        if kill:
            keep = np.ones(self.x.size, bool)
            keep[kill] = False
            self.x, self.y, self.f = self.x[keep], self.y[keep], self.f[:, keep]
        if newx:
            self.x = np.concatenate([self.x, newx])
            self.y = np.concatenate([self.y, newy])
            self.f = np.concatenate([self.f, np.array(newf).T], axis=1)
        return nprop, int(acc.sum())

    def gibbs_bkg(self):
        """Gibbs draw of the linear background (constant + Fourier) per band."""
        if not self.cfg.float_bkg:
            return 0, 0
        for b in range(self.nb):
            sl = self.unpad[b]
            w = self.wts[b][sl].ravel().astype(np.float64)
            # data minus the stars-only model
            r = (self.resid[b][sl].ravel().astype(np.float64)
                 + (self.beta[b] @ self.T[b]))
            Tm = self.T[b]
            prec = self.TtWT[b] + np.diag(self.beta_prec[b])
            rhs = Tm @ (w * r) + self.beta_prec[b] * self.beta_mu[b]
            L = np.linalg.cholesky(prec)
            mean = np.linalg.solve(prec, rhs)
            z = self.rng.standard_normal(self.ntemp)
            new = mean + np.linalg.solve(L.T, z)
            dbg = ((new - self.beta[b]) @ Tm).reshape(self.bands[b].image.shape).astype(np.float32)
            self.model[b][sl] += dbg
            self.resid[b][sl] -= dbg * (self.wts[b][sl] > 0)
            self.beta[b] = new
        return 1, 1

    def gibbs_slope(self):
        lo, hi = self.cfg.slope_bounds
        n = self.x.size
        S = np.sum(np.log(self.f[0] / self.fmin)) if n else 0.0
        if S <= 0:
            self.slope = self.rng.uniform(lo, hi)
            return
        g = stats.gamma(a=n + 1, scale=1.0 / S)
        a, b = g.cdf(lo - 1), g.cdf(hi - 1)
        if not b > a:
            return
        self.slope = 1.0 + g.ppf(self.rng.uniform(a, b))

    # ------------------------------------------------------------------ driver
    def loglike(self):
        chi2 = np.array([np.sum(self.wts[b].astype(np.float64) * self.resid[b].astype(np.float64) ** 2)
                         for b in range(self.nb)])
        return -0.5 * chi2.sum(), chi2

    def run(self):
        cfg = self.cfg
        names = [t for t in _TYPES if cfg.move_weights.get(t, 0) > 0]
        if not cfg.float_bkg and 'bkg' in names:
            names.remove('bkg')
        wts = np.array([cfg.move_weights[t] for t in names], float)
        wts /= wts.sum()
        fns = dict(movestar=self.prop_move, birth_death=self.prop_birth_death,
                   merge_split=self.prop_merge_split, bkg=lambda px, py: self.gibbs_bkg())
        ns = cfg.nsamp
        nmax_store = cfg.max_nsrc
        rec = dict(n=np.zeros(ns, int), logL=np.zeros(ns), chi2=np.zeros((ns, self.nb)),
                   beta=np.zeros((ns, self.nb, self.ntemp)), slope=np.zeros(ns),
                   time=np.zeros(ns))
        cat_x, cat_y, cat_f = [], [], []
        nburn = int(np.floor(cfg.burnin_frac * ns))
        msum = [np.zeros(self.bands[b].image.shape) for b in range(self.nb)]
        mstore = [[] for _ in range(self.nb)]
        t_start = time.time()
        for s in range(ns):
            t0 = time.time()
            self._set_offset(self.rng.integers(self.R), self.rng.integers(self.R))
            self._rebuild_model()
            types = self.rng.choice(len(names), size=cfg.nloop, p=wts)
            pars = self.rng.integers(2, size=(cfg.nloop, 2))
            for i in range(cfg.nloop):
                t = names[types[i]]
                ta = time.time()
                nreg, nacc = fns[t](int(pars[i, 0]), int(pars[i, 1]))
                st = self.stats[t]
                st['nprop'] += 1
                st['nreg'] += nreg
                st['nacc'] += nacc
                st['time'] += time.time() - ta
            if cfg.sample_slope:
                self.gibbs_slope()
            self._rebuild_model()
            lnl, chi2 = self.loglike()
            rec['n'][s] = self.x.size
            rec['logL'][s] = lnl
            rec['chi2'][s] = chi2
            rec['beta'][s] = self.beta
            rec['slope'][s] = self.slope
            rec['time'][s] = time.time() - t0
            cat_x.append(self.x.copy())
            cat_y.append(self.y.copy())
            cat_f.append(self.f.copy())
            if s >= nburn:
                for b in range(self.nb):
                    m = self.model[b][self.unpad[b]].astype(np.float64)
                    msum[b] += m
                    if cfg.store_models:
                        mstore[b].append(m.astype(np.float32))
            if cfg.verbose and (s % max(1, ns // 20) == 0 or s == ns - 1):
                acc = {t: (self.stats[t]['nacc'] / max(1, self.stats[t]['nreg'])) for t in names}
                print(f'[pcat] sample {s+1}/{ns} n={self.x.size} logL={lnl:.1f} '
                      f'chi2/pix={chi2.sum()/max(1, sum(int((w>0).sum()) for w in self.wts)):.3f} '
                      f'bkg={self.beta[:, 0].round(4).tolist()} slope={self.slope:.3f} '
                      f'dt={rec["time"][s]:.2f}s acc=' + ','.join(f'{k}:{v:.3f}' for k, v in acc.items()),
                      flush=True)
        nmax = max([c.size for c in cat_x] + [1])
        X = np.full((ns, nmax), np.nan)
        Y = np.full((ns, nmax), np.nan)
        Fl = np.full((ns, self.nb, nmax), np.nan)
        for s in range(ns):
            k = cat_x[s].size
            X[s, :k] = cat_x[s]
            Y[s, :k] = cat_y[s]
            Fl[s, :, :k] = cat_f[s]
        rec.update(x=X, y=Y, f=Fl)
        npost = ns - nburn
        model_mean = [m / max(1, npost) for m in msum]
        if cfg.store_models and npost > 0:
            model_median = [np.median(np.array(ms), axis=0) for ms in mstore]
        else:
            model_median = [None] * self.nb
        data = [np.where(self.wts[b][self.unpad[b]] > 0, self.data[b][self.unpad[b]], np.nan)
                for b in range(self.nb)]
        res_mean = [data[b] - model_mean[b] for b in range(self.nb)]
        res_med = [None if model_median[b] is None else data[b] - model_median[b] for b in range(self.nb)]
        acc = {t: dict(nprop=self.stats[t]['nprop'], nreg=self.stats[t]['nreg'],
                       nacc=self.stats[t]['nacc'],
                       frac=self.stats[t]['nacc'] / max(1, self.stats[t]['nreg']),
                       time_s=self.stats[t]['time'],
                       ms_per_prop=1e3 * self.stats[t]['time'] / max(1, self.stats[t]['nprop']))
               for t in names}
        for k, (npr, nac) in self.sub.items():
            acc[k] = dict(nreg=npr, nacc=nac, frac=nac / max(1, npr))
        info = dict(runtime_s=time.time() - t_start, margin=self.margin, margin_req=self.margin_req,
                    regsize=self.R, pen=self.pen, area=self.area, errf=list(map(float, self.errf)),
                    errf_eff=self.errf_eff, color_moves=self.color_moves,
                    pads=self.pads, nburn=nburn, leak=float(self.leak), ntemp=self.ntemp,
                    nbands=self.nb, blas=cfg.blas if self.lib is None else self.lib.blas, shape=[list(b.image.shape) for b in self.bands],
                    band_names=[b.name for b in self.bands],
                    noise_scale=self.noise_scale.tolist(),
                    sec_per_sample=float(np.mean(rec['time'])))
        cdict = asdict(cfg)
        cdict['init_catalog'] = None if cfg.init_catalog is None else 'given'
        return PCATResult(samples=rec, accept=acc, model_mean=model_mean, model_median=model_median,
                          residual_mean=res_mean, residual_median=res_med, config=cdict, info=info)


def run_pcat(image, weight, psf_os, oversample, config, nc=31, basis='quartic15', fit='nodes',
             extra_bands=None, fixed_model=None):
    """Run the sampler on one image (plus optional extra bands).

    Parameters
    ----------
    image, weight : (ny, nx) arrays (weight = inverse variance, 0 = masked).
    psf_os : 2D ePSF array (oversampled, convention of :mod:`.psf`) or a path.
    oversample : ePSF oversampling (ignored when ``psf_os`` is a path).
    config : :class:`PCATConfig`.
    extra_bands : list of :class:`BandData` for non-pivot bands.
    fixed_model : optional (ny, nx) pivot-band image held fixed during sampling
        (e.g. the PSF wings of bright stars outside the ``nc`` stamp, see
        :mod:`.wings`).  The sampler fits ``image - fixed_model``; the returned
        ``model_mean`` / ``model_median`` include ``fixed_model`` so that the
        residuals stay ``image - model``.

    Returns :class:`PCATResult` (``info['psf']`` holds the PSF table metadata).
    """
    if isinstance(psf_os, (str, bytes)) or hasattr(psf_os, '__fspath__'):
        psf_os, oversample, _ = load_epsf(psf_os)
    cf, pinfo = build_cf(psf_os, oversample, nc, basis=basis, fit=fit)
    image = np.asarray(image, float)
    if fixed_model is not None:
        fixed_model = np.asarray(fixed_model, float)
        if fixed_model.shape != image.shape:
            raise ValueError(f'fixed_model shape {fixed_model.shape} != image {image.shape}')
        image = image - fixed_model
    bands = [BandData(image=image, weight=np.asarray(weight, float), cf=cf,
                      nc=nc, basis=basis, name='pivot')]
    if extra_bands:
        bands += list(extra_bands)
    if config.engine == 'numba':
        from .fast_sampler import FastPCATSampler
        sampler = FastPCATSampler(bands, config)
    elif config.engine == 'python':
        sampler = PCATSampler(bands, config)
    else:
        raise ValueError(f'unknown engine {config.engine!r}')
    res = sampler.run()
    res.info['psf'] = pinfo
    if fixed_model is not None:
        res.model_mean[0] = res.model_mean[0] + fixed_model
        if res.model_median[0] is not None:
            res.model_median[0] = res.model_median[0] + fixed_model
        good = np.isfinite(fixed_model) & (np.asarray(weight) > 0)
        res.info['fixed_model'] = dict(sum=float(np.nansum(fixed_model[good])),
                                       max=float(np.nanmax(fixed_model)) if fixed_model.size else 0.0)
    return res
