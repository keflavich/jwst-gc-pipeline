"""Full-frame star-subtracted residuals: position-dependent ePSF vs STPSF.

One _cal frame, one existing catalog (the STScI level-3 source catalog of the
observation, ``*_cat.ecsv``; positions only), three PSF models on the SAME
representation (epsf.EPSF spline ePSFs, bilinear in detector position):

  stpsf   STPSF core (5x5 grid) + STPSF wing, both through stpsf_to_epsf
  epsf    held-out ePSF core (6x6 bilinear grid built from the other half of the
          frames: this frame is not in it) + held-out ePSF wing
  hybrid  held-out ePSF core + STPSF wing (the F480M ePSF wing is biased low by
          crowding beyond ~12 px, #1002)

For each model, independently: every catalog star's flux and position, plus a
smooth bilinear background (128-px nodes), are fitted simultaneously to the frame
(Gauss-Newton on a sparse weighted least-squares system, 1/ERR weights,
DO_NOT_USE/SATURATED pixels excluded).  The model image is then rendered with the
pipeline's own renderer (crowdsource_catalogs_long._render_model_from_table, the
function behind the *_mergedcat_model products) and residual = data - model - bkg.
Sky -> pixel uses the frame GWCS (frame_wcs, require_gwcs=True).

    PYTHONPATH=<repo with jwst_gc_pipeline> python fullfield_residual.py \
        --cal <_cal.fits> --cat <_cat.ecsv> --result <result_<det>.npz> \
        --stpsf <stpsf_<DET>_<FILT>.npz> --out <out.npz>
"""
import argparse
import ast
import os
import sys
import time

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy import ndimage, sparse
from scipy.sparse.linalg import lsqr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import epsf  # noqa: E402

from jwst_gc_pipeline.frame_wcs import frame_wcs  # noqa: E402

DO_NOT_USE, SATURATED = 1, 2
BKG_STEP = int(os.environ.get('BKG_STEP', 32))
SUPP_NSIG = float(os.environ.get('SUPP_NSIG', 8.0))


def load_pipeline_renderer():
    """_render_model_from_table from the pipeline module, without importing that module
    (it needs the jwst package at import time; the function itself is numpy only)."""
    import jwst_gc_pipeline.photometry as ph
    path = os.path.join(os.path.dirname(ph.__file__), 'crowdsource_catalogs_long.py')
    src = open(path).read()
    node = next(n for n in ast.parse(src).body
                if isinstance(n, ast.FunctionDef) and n.name == '_render_model_from_table')
    ns = {'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), ns)
    return ns['_render_model_from_table']


def bilinear_on_nodes(x, y, nodes_x, nodes_y):
    """Bilinear weights (Nnode, Nstar) on a regular node lattice (clamped outside)."""
    nx, ny = len(nodes_x), len(nodes_y)
    W = np.zeros((nx * ny, len(x)))
    if nx == 1 and ny == 1:
        W[0] = 1
        return W
    fx = np.clip(np.interp(x, nodes_x, np.arange(nx)), 0, nx - 1)
    fy = np.clip(np.interp(y, nodes_y, np.arange(ny)), 0, ny - 1)
    i0 = np.minimum(np.floor(fx).astype(int), nx - 2); j0 = np.minimum(np.floor(fy).astype(int), ny - 2)
    tx, ty = fx - i0, fy - j0
    idx = np.arange(len(x))
    for dj, wy in ((0, 1 - ty), (1, ty)):
        for di, wx in ((0, 1 - tx), (1, tx)):
            np.add.at(W, ((j0 + dj) * nx + (i0 + di), idx), wx * wy)
    return W


def bilinear_sparse(x, y, nodes_x, nodes_y):
    """The four (node index, weight) pairs per point, as (4, N) arrays."""
    nx, ny = len(nodes_x), len(nodes_y)
    fx = np.clip(np.interp(x, nodes_x, np.arange(nx)), 0, nx - 1)
    fy = np.clip(np.interp(y, nodes_y, np.arange(ny)), 0, ny - 1)
    i0 = np.minimum(np.floor(fx).astype(int), nx - 2); j0 = np.minimum(np.floor(fy).astype(int), ny - 2)
    tx, ty = fx - i0, fy - j0
    K = np.stack([j0 * nx + i0, j0 * nx + i0 + 1, (j0 + 1) * nx + i0, (j0 + 1) * nx + i0 + 1])
    V = np.stack([(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty])
    return K, V


def supplement_peaks(sci, err, good, xc, yc, fc, fwhm):
    """Model-independent extra sources: local maxima of the data matched-filtered with a
    Gaussian of the PSF FWHM, >= SUPP_NSIG sigma above a 4-FWHM median background, not
    within 2 FWHM of a catalog star, and not inside the halo (r < 8 FWHM) of the
    brightest 3% of the catalog (where PSF-mismatch ripples would be picked up)."""
    d = np.where(good, sci, np.nan)
    bg = ndimage.median_filter(np.nan_to_num(d, nan=np.nanmedian(d)), size=int(4 * fwhm) | 1)
    s = fwhm / 2.3548
    k = np.exp(-0.5 * ((np.arange(-3, 4)[:, None] ** 2 + np.arange(-3, 4)[None] ** 2) / s ** 2)); k /= k.sum()
    num = ndimage.convolve(np.nan_to_num(d - bg) * good, k)
    den = np.sqrt(ndimage.convolve(np.where(good, err, 0) ** 2, k ** 2))
    snr = num / np.maximum(den, 1e-12)
    pk = (snr == ndimage.maximum_filter(snr, size=3)) & (snr > SUPP_NSIG) & good
    yy, xx = np.nonzero(pk)
    near = np.zeros(sci.shape, bool)
    cat = np.zeros(sci.shape, bool)
    ix = np.clip(np.round(xc).astype(int), 0, sci.shape[1] - 1); iy = np.clip(np.round(yc).astype(int), 0, sci.shape[0] - 1)
    cat[iy, ix] = True
    near |= ndimage.distance_transform_edt(~cat) < 2 * fwhm
    bright = np.zeros(sci.shape, bool)
    b = fc > np.nanpercentile(fc, 97)
    bright[iy[b], ix[b]] = True
    near |= ndimage.distance_transform_edt(~bright) < 8 * fwhm
    keep = ~near[yy, xx]
    return xx[keep].astype(float), yy[keep].astype(float)


def composite(core, gcore, wing, gwing, r0=10.0, r1=12.0):
    """Core model inside r0, wing model outside r1, linear blend between (wing grid)."""
    off = (gwing.R - gcore.R) * gcore.O
    C = np.zeros_like(wing)
    C[off:off + gcore.n, off:off + gcore.n] = core
    t = np.clip((gwing.ur - r0) / (r1 - r0), 0, 1)
    return (1 - t) * C + t * wing


class SpatialPSF:
    """Bilinear mix of EPSF nodes; .evaluate() has the photutils (x, y, flux, x0, y0)
    signature the pipeline renderer calls."""

    def __init__(self, Ps, grid, nodes_x, nodes_y):
        self.eps = [epsf.EPSF(P, grid) for P in Ps]
        self.nx, self.ny, self.g = nodes_x, nodes_y, grid

    def weights(self, x, y):
        return bilinear_on_nodes(np.atleast_1d(x), np.atleast_1d(y), self.nx, self.ny)

    def evaluate(self, xx, yy, flux, x0, y0):
        w = self.weights(x0, y0)[:, 0]
        out = np.zeros(np.shape(xx))
        for k in np.nonzero(w > 0)[0]:
            out += w[k] * self.eps[k](xx - x0, yy - y0)
        return flux * out

    def stamps(self, ux, uy, x0, y0, grad=True):
        """P, dP/dux, dP/duy for a chunk of stars (ux, uy: (n, s, s))."""
        W = self.weights(x0, y0)
        P = np.zeros(ux.shape); gx = np.zeros(ux.shape); gy = np.zeros(ux.shape)
        for k, ep in enumerate(self.eps):
            sel = np.nonzero(W[k] > 0)[0]
            if len(sel) == 0:
                continue
            wk = W[k, sel][:, None, None]
            P[sel] += wk * ep(ux[sel], uy[sel])
            if grad:
                a, c = ep.grad(ux[sel], uy[sel])
                gx[sel] += wk * a; gy[sel] += wk * c
        return P, gx, gy


def build_models(res, st, frame_parity):
    O, R, Rw, rn = int(res['O']), int(res['R']), int(res['Rw']), float(res['r_norm'])
    g, gw = epsf.Grid(R, O), epsf.Grid(Rw, O)
    G = int(res['Gbest'])
    # held-out halves: PGa/Pw_a are built from EVEN frames, PGb/Pw_b from ODD ones
    PG = res['PGb'] if frame_parity == 0 else res['PGa']
    Pw = res['Pw_b'] if frame_parity == 0 else res['Pw_a']
    od = st['overdist'].astype(np.float64); spos = st['pos']
    Sc = [epsf.normalise(epsf.stpsf_to_epsf(o, O, R), g, rn) for o in od]
    Sw = [epsf.normalise(epsf.stpsf_to_epsf(o, O, Rw), gw, rn) for o in od]
    snx, sny = np.unique(np.round(spos[:, 0], 1)), np.unique(np.round(spos[:, 1], 1))
    # the STPSF list runs x fastest (spos[1] = (614, 205)) -- same order as the node index j*nx+i
    assert np.allclose(spos[1], [snx[1], sny[0]], atol=1)
    cnodes = (np.arange(G) + 0.5) * 2048 / G
    models = {
        'stpsf': SpatialPSF([epsf.normalise(composite(c, g, w, gw), gw, rn) for c, w in zip(Sc, Sw)], gw, snx, sny),
        'epsf': SpatialPSF([epsf.normalise(composite(P, g, Pw, gw), gw, rn) for P in PG], gw, cnodes, cnodes),
    }
    # hybrid: ePSF core per ePSF node + the STPSF wing interpolated to that node
    sw_model = SpatialPSF(Sw, gw, snx, sny)
    hyb = []
    for j, yc in enumerate(cnodes):
        for i, xc in enumerate(cnodes):
            w = sw_model.weights(xc, yc)[:, 0]
            wing = sum(w[k] * Sw[k] for k in np.nonzero(w > 0)[0])
            hyb.append(epsf.normalise(composite(PG[j * G + i], g, wing, gw), gw, rn))
    models['hybrid'] = SpatialPSF(hyb, gw, cnodes, cnodes)
    return models, R, Rw


def fit_frame(model, sci, err, good, x, y, radii, niter=3, verbose=True):
    """Simultaneous flux + position + bilinear-background fit.  Returns f, x, y, bkg image."""
    ny, nx = sci.shape
    w = np.where(good, 1.0 / np.where(good, err, 1.0), 0.0)
    rows_good = np.flatnonzero(good.ravel())
    rowmap = -np.ones(nx * ny, np.int64); rowmap[rows_good] = np.arange(len(rows_good))
    d = (sci.ravel() * w.ravel())[rows_good]
    # background basis: bilinear hats on BKG_STEP nodes
    bxn = np.arange(0, nx + BKG_STEP, BKG_STEP); byn = np.arange(0, ny + BKG_STEP, BKG_STEP)
    yy, xx = np.divmod(rows_good, nx)
    Kb, Vb = bilinear_sparse(xx.astype(float), yy.astype(float), bxn.astype(float), byn.astype(float))
    nb = len(bxn) * len(byn)
    rr = np.broadcast_to(np.arange(len(rows_good)), Kb.shape)
    Bmat = sparse.csr_matrix(((Vb * w.ravel()[rows_good][None]).ravel(), (rr.ravel(), Kb.ravel())),
                             shape=(len(rows_good), nb))
    del Kb, Vb, rr
    x = x.copy(); y = y.copy(); ns = len(x)
    f = np.zeros(ns)
    for it in range(niter):
        t0 = time.time()
        R_, C_, V_ = [], [], []
        for rad in np.unique(radii):
            sel = np.nonzero(radii == rad)[0]
            k = np.arange(-rad, rad + 1)
            for c0 in range(0, len(sel), max(50, int(1.5e6 / (2 * rad + 1) ** 2))):
                ii = sel[c0:c0 + max(50, int(1.5e6 / (2 * rad + 1) ** 2))]
                ix = np.round(x[ii]).astype(int); iy = np.round(y[ii]).astype(int)
                px = ix[:, None, None] + k[None, None, :]; py = iy[:, None, None] + k[None, :, None]
                px, py = np.broadcast_arrays(px, py)
                ux = px - x[ii][:, None, None]; uy = py - y[ii][:, None, None]
                P, gx, gy = model.stamps(ux, uy, x[ii], y[ii], grad=True)
                inside = (px >= 0) & (px < nx) & (py >= 0) & (py < ny) & (np.hypot(ux, uy) <= rad + 0.5)
                pix = np.where(inside, py * nx + px, 0)
                r = np.where(inside, rowmap[pix], -1)
                keep = r >= 0
                ww_ = w.ravel()[pix[keep]]
                col = np.broadcast_to(ii[:, None, None], P.shape)[keep]
                for q, A in enumerate((P, -gx, -gy)):
                    R_.append(r[keep]); C_.append(3 * col + q); V_.append(A[keep] * ww_)
        S = sparse.csr_matrix((np.concatenate(V_), (np.concatenate(R_), np.concatenate(C_))),
                              shape=(len(rows_good), 3 * ns))
        del R_, C_, V_
        A = sparse.hstack([S, Bmat]).tocsc()
        # column scaling for lsqr conditioning
        cn = np.sqrt(np.asarray(A.multiply(A).sum(0)).ravel()); cn[cn == 0] = 1
        sol = lsqr(A @ sparse.diags(1 / cn), d, atol=1e-7, btol=1e-7, iter_lim=400)[0] / cn
        a, b, c = sol[0:3 * ns:3], sol[1:3 * ns:3], sol[2:3 * ns:3]
        f = a
        ok = a > 0
        dx = np.where(ok, np.clip(b / np.where(ok, a, 1), -0.5, 0.5), 0)
        dy = np.where(ok, np.clip(c / np.where(ok, a, 1), -0.5, 0.5), 0)
        if it < niter - 1:
            x += dx; y += dy
        if verbose:
            print(f'  iter {it}: {ns} stars, nnz {A.nnz/1e6:.1f}M, median |dpos| {np.median(np.hypot(dx, dy)[ok]):.3f} px, '
                  f'neg flux {(~ok).sum()}, {time.time()-t0:.0f} s', flush=True)
    bcoef = sol[3 * ns:]
    yy, xx = np.mgrid[0:ny, 0:nx]
    Kb, Vb = bilinear_sparse(xx.ravel().astype(float), yy.ravel().astype(float), bxn.astype(float), byn.astype(float))
    bimg = (bcoef[Kb] * Vb).sum(0).reshape(ny, nx)
    return f, x, y, bimg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cal', required=True); ap.add_argument('--cat', required=True)
    ap.add_argument('--result', required=True); ap.add_argument('--stpsf', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--models', default='stpsf,epsf,hybrid')
    a = ap.parse_args()
    res = np.load(a.result, allow_pickle=True); st = np.load(a.stpsf)
    stem = os.path.basename(a.cal).replace('_cal.fits', '')
    frames = [str(s).replace('_stars.npz', '') for s in res['frames']]
    fidx = frames.index(stem)
    models, R, Rw = build_models(res, st, fidx % 2)
    print(f'{stem}: frame index {fidx} -> held-out ePSF half {"PGb (odd)" if fidx % 2 == 0 else "PGa (even)"}')
    with fits.open(a.cal) as h:
        sci = h['SCI'].data.astype(float); err = h['ERR'].data.astype(float); dq = h['DQ'].data
    good = np.isfinite(sci) & np.isfinite(err) & (err > 0) & ((dq & (DO_NOT_USE | SATURATED)) == 0)
    ww = frame_wcs(a.cal, require_gwcs=True)
    t = Table.read(a.cat)
    x, y = ww.world_to_pixel(t['sky_centroid'])
    m = np.isfinite(x) & np.isfinite(y) & (x > -3) & (x < sci.shape[1] + 2) & (y > -3) & (y < sci.shape[0] + 2)
    x, y = np.asarray(x[m], float), np.asarray(y[m], float)
    fwhm = float(os.environ.get('FWHM_PIX', 2.6 if 'long' in stem else 2.2))
    xs, ys = supplement_peaks(sci, err, good, x, y, np.asarray(t['aper_total_flux'][m], float), fwhm)
    print(f'{m.sum()} catalog stars on the frame + {len(xs)} supplementary >= {SUPP_NSIG} sigma peaks')
    ncat = len(x)
    x = np.concatenate([x, xs]); y = np.concatenate([y, ys])
    # stamp radius from the catalog flux (brighter -> larger); same radii for every model
    fl = np.asarray(t['aper_total_flux'][m], float); fl = np.where(np.isfinite(fl), fl, np.nanmin(fl))
    fl = np.concatenate([fl, np.full(len(xs), np.nanmin(fl))])
    q = np.nanpercentile(fl[:ncat], [80, 97])    # radii tiers from the catalog stars only
    radii = np.where(fl > q[1], Rw, np.where(fl > q[0], (R + Rw) // 2, R)).astype(int)
    print(f'{len(x)} sources fitted; stamp radii {dict(zip(*np.unique(radii, return_counts=True)))}')
    render = load_pipeline_renderer()
    out = dict(sci=sci.astype(np.float32), err=err.astype(np.float32), good=good, x0=x, y0=y, radii=radii, ncat=ncat)
    for name in a.models.split(','):
        print(f'== {name}', flush=True)
        f, xf, yf, bimg = fit_frame(models[name], sci, err, good, x, y, radii)
        model = np.zeros(sci.shape, np.float32)
        for rad in np.unique(radii):
            s = radii == rad
            tab = Table(dict(x_fit=xf[s], y_fit=yf[s], flux_fit=np.clip(f[s], 0, None)))
            model += render(tab, models[name], sci.shape, (2 * rad + 1, 2 * rad + 1))
        resid = sci - model - bimg
        chi = np.where(good, resid / err, np.nan)
        print(f'   chi rms (all good px) {np.nanstd(chi):.3f}; median chi^2 {np.nanmedian(chi**2):.3f}', flush=True)
        out.update({f'{name}_model': model, f'{name}_bkg': bimg.astype(np.float32), f'{name}_resid': resid.astype(np.float32),
                    f'{name}_f': f, f'{name}_x': xf, f'{name}_y': yf})
    np.savez_compressed(a.out, **out)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
