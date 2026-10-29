"""Compiled (numba) proposal loop for the PCAT sampler.

:class:`FastPCATSampler` targets the posterior of :class:`.pcat_sampler.PCATSampler`
(single band, ``count_prior`` 'exact' or 'legacy') with the same proposal set
(region-parallel star moves, birth/death, split/merge, Gibbs background), but
runs the ``nloop`` proposals of each sample inside one compiled function:

* stars live in fixed-capacity slot arrays (``max_nsrc``) with per-region
  member lists, so a proposal never scans the whole catalogue;
* each proposal renders its sources ("phonions") into a scratch image and
  evaluates the chi^2 change and the accepted update only on the pixels of
  their stamps (no full-image passes);
* a star move whose integer pixel and stamp size are unchanged is rendered as
  one stamp of the coefficient difference ``f' b(dx', dy') - f b(dx, dy)``
  (the PSF is linear in the basis coefficients);
* optionally (``stamp_tol``) a source is rendered on the smallest stamp of a
  ladder (``stamp_sizes``) whose truncated wing satisfies
  ``f^2 sum_{outside} P^2 <= stamp_tol * sigma_pix^2`` (at most ``stamp_tol``
  of chi^2 per source relative to the full ``nc`` stamp).  This changes the
  likelihood by the truncated wing; ``stamp_tol=None`` keeps the fixed ``nc``
  stamp of :class:`PCATSampler`;
* optionally (``move_nstar > 0``) a star move updates ``move_nstar`` randomly
  chosen stars per region instead of every star in the region (PCAT), with
  step sizes scaled to the number moved.  The target is unchanged; the
  proposal kernel differs.

Options are fields of :class:`.pcat_sampler.PCATConfig` (``engine='numba'``).
"""
import time
from dataclasses import asdict

import numpy as np
from numba import njit

from .pcat_sampler import PCATSampler, PCATResult, _TYPES
from .kernel import basis_terms

__all__ = ['FastPCATSampler', 'stamp_ladder', 'ess_geyer']

# stats rows
_MOVE, _BD, _MS, _BKG, _BIRTH, _DEATH, _SPLIT, _MERGE, _CAPREJ = range(9)
# proposal kinds
_K_MOVE, _K_BIRTH, _K_DEATH, _K_SPLIT, _K_MERGE = range(5)


# ----------------------------------------------------------------------------
# stamp ladder
# ----------------------------------------------------------------------------
def stamp_ladder(cf, nc, terms, sizes, tol, sigma2):
    """Flux thresholds for the adaptive stamp sizes.

    ``cf`` is the ``(k, nc*nc)`` table of the largest stamp; the table of a
    smaller odd size ``m`` is its central ``m x m`` crop.  A source of flux
    ``f`` uses the smallest ``m`` with ``f^2 W(m) <= tol * sigma2``, where
    ``W(m)`` is the largest (over a 9 x 9 grid of sub-pixel phases) sum of the
    squared stamp values outside the crop.  Returns ``(sizes, fmax)``:
    increasing sizes ending with ``nc`` and the largest flux each may carry
    (``inf`` for ``nc``).
    """
    from .psf_poly import render_stamp_poly
    sizes = sorted({int(s) for s in sizes if int(s) < nc and int(s) % 2 == 1}) + [int(nc)]
    rad = nc // 2
    ph = np.linspace(0.0, 1.0, 9, endpoint=False)
    stamps = [render_stamp_poly(cf, nc, terms, dx, dy) for dx in ph for dy in ph]
    fmax = []
    for m in sizes:
        if m == nc:
            fmax.append(np.inf)
            continue
        r = m // 2
        inner = np.zeros((nc, nc), bool)
        inner[rad - r:rad + r + 1, rad - r:rad + r + 1] = True
        W = max(float(np.sum(st[~inner] ** 2)) for st in stamps)
        fmax.append(np.sqrt(tol * sigma2 / W) if W > 0 else np.inf)
    fmax = np.maximum.accumulate(np.asarray(fmax, float))
    return np.asarray(sizes, np.int64), fmax


def separable_factors(Tm, ny, nx):
    """Rank-one factors of background templates: ``Tm[t] == outer(Vy[t], Ux[tx[t]])``.

    Returns ``(Vy, Ux, tx)`` with the distinct x factors ``Ux`` (unit max-abs),
    or ``None`` when some template is not a product of a y and an x function.
    The constant and the Fourier templates of :func:`.pcat_sampler.fourier_templates`
    are products, so the background costs ``2 * len(Ux)`` operations per pixel
    instead of ``2 * len(Tm)``.
    """
    nt = Tm.shape[0]
    Vy = np.zeros((nt, ny))
    tx = np.zeros(nt, np.int64)
    xs = []
    for t in range(nt):
        u = Tm[t].reshape(ny, nx)
        amp = np.abs(u).max()
        if amp == 0:
            return None
        i0, j0 = np.unravel_index(np.argmax(np.abs(u)), u.shape)
        ux = u[i0] / u[i0, j0]
        vy = u[:, j0].copy()
        if not np.allclose(np.outer(vy, ux), u, rtol=0, atol=1e-10 * amp):
            return None
        jm = np.argmax(np.abs(ux))
        sc = ux[jm]
        ux = ux / sc
        vy = vy * sc
        for a, xa in enumerate(xs):
            if np.allclose(xa, ux, rtol=0, atol=1e-12):
                tx[t] = a
                break
        else:
            tx[t] = len(xs)
            xs.append(ux)
        Vy[t] = vy
    return Vy, np.ascontiguousarray(np.array(xs)), tx


def ess_geyer(x):
    """Effective sample size of one chain (Geyer initial positive sequence)."""
    x = np.asarray(x, float)
    n = x.size
    if n < 4 or np.ptp(x) == 0:
        return float(n)
    x = x - x.mean()
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac /= ac[0]
    tau = -1.0
    for k in range(0, n - 1, 2):
        g = ac[k] + ac[k + 1]
        if g < 0:
            break
        tau += 2 * g
    return float(n / max(tau, 1.0))


# ----------------------------------------------------------------------------
# compiled kernels
# ----------------------------------------------------------------------------
@njit(cache=True)
def _seed(seed):
    np.random.seed(seed)


@njit(cache=True)
def _level(f, fthr):
    a = abs(f)
    for L in range(fthr.size):
        if a <= fthr[L]:
            return L
    return fthr.size - 1


@njit(cache=True)
def _coefs(bx, by, f, ta, tb, out, o):
    """basis coefficients f * dx^a dy^b of a source at band position (bx, by),
    written to out[o:o+k]; returns (ix, iy)."""
    ix = int(np.ceil(bx))
    iy = int(np.ceil(by))
    dx = ix - bx
    dy = iy - by
    for j in range(ta.size):
        out[o + j] = f * dx ** ta[j] * dy ** tb[j]
    return ix, iy


@njit(cache=True, fastmath=True)
def _render_add(img, ix, iy, lv, dd, o, cf3, ncl, ncw, tmp):
    """add the stamp of coefficients dd[o:o+k] (level lv, centre pixel (iy, ix)) to img.

    Table rows are ``ncw[lv]`` wide (a multiple of 8 with zero coefficients past
    ``ncl[lv]``) so the inner loop vectorises without a scalar remainder; img
    needs ``ncw - ncl`` spare columns on the right.  Column indices are unsigned,
    which removes numba's negative-index wraparound from the inner loop.
    """
    k = cf3.shape[1]
    nc = ncl[lv]
    W = ncw[lv]
    r = nc // 2
    y0 = iy - r
    x0 = ix - r
    cf = cf3[lv]
    if k == 15:
        a0 = np.float32(dd[o + 0])
        a1 = np.float32(dd[o + 1])
        a2 = np.float32(dd[o + 2])
        a3 = np.float32(dd[o + 3])
        a4 = np.float32(dd[o + 4])
        a5 = np.float32(dd[o + 5])
        a6 = np.float32(dd[o + 6])
        a7 = np.float32(dd[o + 7])
        a8 = np.float32(dd[o + 8])
        a9 = np.float32(dd[o + 9])
        a10 = np.float32(dd[o + 10])
        a11 = np.float32(dd[o + 11])
        a12 = np.float32(dd[o + 12])
        a13 = np.float32(dd[o + 13])
        a14 = np.float32(dd[o + 14])
        for i in range(nc):
            row = img[y0 + i]
            q = i * W
            for j in range(W):
                p = np.uint64(q + j)
                row[np.uint64(x0 + j)] += (a0 * cf[0, p] + a1 * cf[1, p] + a2 * cf[2, p] + a3 * cf[3, p] +
                                           a4 * cf[4, p] + a5 * cf[5, p] + a6 * cf[6, p] + a7 * cf[7, p] +
                                           a8 * cf[8, p] + a9 * cf[9, p] + a10 * cf[10, p] + a11 * cf[11, p] +
                                           a12 * cf[12, p] + a13 * cf[13, p] + a14 * cf[14, p])
        return
    npx = nc * W
    for p in range(npx):
        tmp[p] = 0.0
    for c in range(k):
        a = np.float32(dd[o + c])
        for p in range(npx):
            tmp[p] += a * cf[c, p]
    for i in range(nc):
        row = img[y0 + i]
        q = i * W
        for j in range(W):
            row[np.uint64(x0 + j)] += tmp[q + j]


@njit(cache=True)
def _slot_box(ph_ix, ph_iy, ph_lv, q0, q1, ncl, box):
    """bounding box of the stamps of phonions q0..q1-1 -> box[1:5] (y0, y1, x0, x1);
    box[0] = 1 when the box is smaller than the summed stamp area (box mode).
    Returns the number of pixels the chi^2 visits."""
    y0 = 1 << 30
    x0 = 1 << 30
    y1 = -(1 << 30)
    x1 = -(1 << 30)
    sa = 0
    for q in range(q0, q1):
        r = ncl[ph_lv[q]] // 2
        y0 = min(y0, ph_iy[q] - r)
        y1 = max(y1, ph_iy[q] + r + 1)
        x0 = min(x0, ph_ix[q] - r)
        x1 = max(x1, ph_ix[q] + r + 1)
        sa += (2 * r + 1) * (2 * r + 1)
    ba = (y1 - y0) * (x1 - x0)
    box[0] = 1 if ba <= sa else 0
    box[1] = y0
    box[2] = y1
    box[3] = x0
    box[4] = x1
    return min(ba, sa)


@njit(cache=True, fastmath=True)
def _slot_dchi2(dm, dm2, resid, wt, ph_ix, ph_iy, ph_lv, q0, q1, ncl, box):
    """chi^2 change of one region's phonions (rendered in dm).  Box mode reads
    the bounding box; stamp mode visits each stamp and moves dm to dm2 so that
    overlapping stamps count once."""
    s = 0.0
    if box[0] == 1:
        x0 = box[3]
        nx = box[4] - box[3]
        for i in range(box[1], box[2]):
            dmr = dm[i]
            rr = resid[i]
            wr = wt[i]
            for j in range(nx):
                jj = np.uint64(x0 + j)
                d = np.float64(dmr[jj])
                s += wr[jj] * (d * d - 2.0 * rr[jj] * d)
        return s
    for q in range(q0, q1):
        r = ncl[ph_lv[q]] // 2
        x0 = ph_ix[q] - r
        nx = 2 * r + 1
        for i in range(ph_iy[q] - r, ph_iy[q] + r + 1):
            dmr = dm[i]
            d2r = dm2[i]
            rr = resid[i]
            wr = wt[i]
            for j in range(nx):
                jj = np.uint64(x0 + j)
                d = dmr[jj]
                dd = np.float64(d)
                s += wr[jj] * (dd * dd - 2.0 * rr[jj] * dd)
                d2r[jj] += d
                dmr[jj] = 0.0
    return s


@njit(cache=True, fastmath=True)
def _slot_apply(dm, dm2, resid, model, ph_ix, ph_iy, ph_lv, q0, q1, ncl, box, accept):
    """resid -= change, model += change when accepted; clears the scratch."""
    if box[0] == 1:
        x0 = box[3]
        nx = box[4] - box[3]
        for i in range(box[1], box[2]):
            dmr = dm[i]
            if accept:
                rr = resid[i]
                mr = model[i]
                for j in range(nx):
                    jj = np.uint64(x0 + j)
                    d = dmr[jj]
                    rr[jj] -= d
                    mr[jj] += d
                    dmr[jj] = 0.0
            else:
                for j in range(nx):
                    dmr[np.uint64(x0 + j)] = 0.0
        return
    for q in range(q0, q1):
        r = ncl[ph_lv[q]] // 2
        x0 = ph_ix[q] - r
        nx = 2 * r + 1
        for i in range(ph_iy[q] - r, ph_iy[q] + r + 1):
            d2r = dm2[i]
            if accept:
                rr = resid[i]
                mr = model[i]
                for j in range(nx):
                    jj = np.uint64(x0 + j)
                    d = d2r[jj]
                    rr[jj] -= d
                    mr[jj] += d
                    d2r[jj] = 0.0
            else:
                for j in range(nx):
                    d2r[np.uint64(x0 + j)] = 0.0


@njit(cache=True)
def _asinh_fwd(f, lindf, logdf):
    return np.log(logdf * logdf * f + logdf * np.sqrt(lindf * lindf + logdf * logdf * f * f)) / logdf


@njit(cache=True)
def _asinh_inv(ff, lindf, logdf):
    return np.exp(-logdf * ff) * (-lindf * lindf * logdf * logdf + np.exp(2 * logdf * ff)) / (2 * logdf * logdf)


@njit(cache=True)
def _flux_prop(f0, fmin, lindf, logdf):
    ff = _asinh_fwd(f0, lindf, logdf)
    ffmin = _asinh_fwd(fmin, lindf, logdf)
    dff = np.random.standard_normal()
    above = ff - ffmin
    if -dff > above:
        dff = -2 * above - dff
    pf = _asinh_inv(ff + dff, lindf, logdf)
    if pf < fmin:
        pf = fmin
    lq = 0.5 * np.log(lindf * lindf + logdf * logdf * pf * pf) - 0.5 * np.log(lindf * lindf + logdf * logdf * f0 * f0)
    return pf, lq


@njit(cache=True)
def _logp_flux(f, slope, fmin):
    return np.log(slope - 1.0) - np.log(fmin) - slope * np.log(f / fmin)


@njit(cache=True)
def _logsumexp_excl(xs, ys, nm, xi, yi, iskip, x2, y2, use2, inv2k2):
    """ln sum_j exp(-d_ij^2/2k^2) over (xs[j], ys[j]) j != iskip, plus (x2, y2) if use2."""
    mx = -np.inf
    for j in range(nm):
        if j == iskip:
            continue
        v = -((xs[j] - xi) ** 2 + (ys[j] - yi) ** 2) * inv2k2
        if v > mx:
            mx = v
    v2 = -np.inf
    if use2:
        v2 = -((x2 - xi) ** 2 + (y2 - yi) ** 2) * inv2k2
        if v2 > mx:
            mx = v2
    if mx == -np.inf:
        return -np.inf
    s = 0.0
    for j in range(nm):
        if j == iskip:
            continue
        s += np.exp(-((xs[j] - xi) ** 2 + (ys[j] - yi) ** 2) * inv2k2 - mx)
    if use2:
        s += np.exp(v2 - mx)
    return mx + np.log(s)


@njit(cache=True)
def _logaddexp(a, b):
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    if a > b:
        return a + np.log1p(np.exp(b - a))
    return b + np.log1p(np.exp(a - b))


@njit(cache=True)
def _split_logfactor(n_before, nr_after, nbright_before, F, f1, f2, log_invpairs, par):
    fmin, slope, pen, area, kick, legacy = par[0], par[1], par[2], par[3], par[6], par[13]
    lf = _logp_flux(f1, slope, fmin) + _logp_flux(f2, slope, fmin) - _logp_flux(F, slope, fmin)
    lp = (lf + log_invpairs + np.log(2 * np.pi * kick * kick) + np.log(nbright_before)
          + np.log(1.0 - 2.0 * fmin / F) + np.log(F) - np.log(nr_after))
    if legacy > 0:
        return lp - pen - np.log(area) + np.log(nr_after)
    return lp - pen + np.log((n_before + 1) / area)


@njit(cache=True)
def _count_term_birth(n, n_r, area_r, par):
    pen, area, legacy = par[2], par[3], par[13]
    if legacy > 0:
        return -pen
    return -pen + np.log((n + 1) / area) + np.log(area_r / (n_r + 1))


@njit(cache=True)
def _region_of(x, y, par, geo):
    """(col, row) of pivot position (x, y) for the current offset."""
    R, dp = par[4], par[5]
    col = int(np.floor((x + 0.5 + dp + geo[0]) / R))
    row = int(np.floor((y + 0.5 + dp + geo[1]) / R))
    return col, row


@njit(cache=True)
def _in_domain(x, y, par):
    dp, Lx, Ly = par[5], par[11], par[12]
    sx = x + 0.5 + dp
    sy = y + 0.5 + dp
    return sx >= 0 and sx < Lx and sy >= 0 and sy < Ly


@njit(cache=True)
def _add_star(x, y, f, r, sx, sy, sf, sreg, spos, nst, rlist, rcnt):
    i = nst[0]
    sx[i] = x
    sy[i] = y
    sf[i] = f
    sreg[i] = r
    spos[i] = rcnt[r]
    rlist[r, rcnt[r]] = i
    rcnt[r] += 1
    nst[0] = i + 1


@njit(cache=True)
def _remove_star(i, sx, sy, sf, sreg, spos, nst, rlist, rcnt):
    """swap-remove star i (callers remove in decreasing index order)."""
    r = sreg[i]
    p = spos[i]
    last_m = rlist[r, rcnt[r] - 1]
    rlist[r, p] = last_m
    spos[last_m] = p
    rcnt[r] -= 1
    last = nst[0] - 1
    if i != last:
        sx[i] = sx[last]
        sy[i] = sy[last]
        sf[i] = sf[last]
        sreg[i] = sreg[last]
        spos[i] = spos[last]
        rlist[sreg[i], spos[i]] = i
    nst[0] = last


@njit(cache=True)
def _assign_regions(sx, sy, nst, sreg, spos, rlist, rcnt, par, geo):
    rcnt[:] = 0
    nregx = geo[2]
    cap = rlist.shape[1]
    for i in range(nst[0]):
        col, row = _region_of(sx[i], sy[i], par, geo)
        r = row * nregx + col
        if rcnt[r] >= cap:
            return False
        sreg[i] = r
        spos[i] = rcnt[r]
        rlist[r, rcnt[r]] = i
        rcnt[r] += 1
    return True


@njit(cache=True, fastmath=True)
def _bkg_add(model, resid, wpos, Ux, C, P, ny0, nx0, rowbuf, sub_resid):
    """model += sum_a C[:, a] (x) Ux[a] on the unpadded grid (and resid -= that * wpos)."""
    na = Ux.shape[0]
    for i in range(ny0):
        for j in range(nx0):
            rowbuf[j] = 0.0
        for a in range(na):
            c = C[i, a]
            ua = Ux[a]
            for j in range(nx0):
                rowbuf[j] += c * ua[j]
        mr = model[P + i]
        if sub_resid:
            rr = resid[P + i]
            wp = wpos[P + i]
            for j in range(nx0):
                d = np.float32(rowbuf[j])
                mr[P + j] += d
                rr[P + j] -= d * wp[P + j]
        else:
            for j in range(nx0):
                mr[P + j] += np.float32(rowbuf[j])


@njit(cache=True)
def _bkg_coef(beta, Vy, tx, C):
    """C[i, a] = sum_{t: tx[t] = a} beta[t] Vy[t, i]."""
    C[:, :] = 0.0
    for t in range(Vy.shape[0]):
        a = tx[t]
        bt = beta[t]
        for i in range(Vy.shape[1]):
            C[i, a] += bt * Vy[t, i]


@njit(cache=True)
def _rebuild(model, resid, data, wpos, sx, sy, sf, nst, cf3, ncl, ncw, fthr, ta, tb, tmp, dd,
             beta, Vy, Ux, tx, C, rowbuf, P, ny0, nx0):
    model[:, :] = 0.0
    for i in range(nst[0]):
        ix, iy = _coefs(sx[i] + P, sy[i] + P, sf[i], ta, tb, dd, 0)
        _render_add(model, ix, iy, _level(sf[i], fthr), dd, 0, cf3, ncl, ncw, tmp)
    _bkg_coef(beta, Vy, tx, C)
    _bkg_add(model, resid, wpos, Ux, C, P, ny0, nx0, rowbuf, False)
    ny, nx = model.shape
    for i in range(ny):
        for j in range(nx):
            resid[i, j] = (data[i, j] - model[i, j]) * wpos[i, j]


@njit(cache=True)
def _chi2(resid, wt):
    s = 0.0
    ny, nx = resid.shape
    for i in range(ny):
        for j in range(nx):
            r = np.float64(resid[i, j])
            s += wt[i, j] * r * r
    return s


@njit(cache=True, fastmath=True)
def _bkg_project(resid, wt, Ux, P, ny0, nx0, A, rowbuf):
    """A[i, a] = sum_j Ux[a, j] wt resid on the unpadded grid."""
    na = Ux.shape[0]
    for i in range(ny0):
        rr = resid[P + i]
        wr = wt[P + i]
        for j in range(nx0):
            rowbuf[j] = np.float64(wr[P + j]) * rr[P + j]
        for a in range(na):
            ua = Ux[a]
            s = 0.0
            for j in range(nx0):
                s += ua[j] * rowbuf[j]
            A[i, a] = s


@njit(cache=True)
def _gibbs_bkg(resid, model, wt, wpos, Vy, Ux, tx, TtWT, beta, prec_mu, Pinv, LinvT, P, ny0, nx0,
               rhs, A, C, rowbuf):
    """Gibbs draw of the template amplitudes given the stars (separable templates)."""
    nt = Vy.shape[0]
    _bkg_project(resid, wt, Ux, P, ny0, nx0, A, rowbuf)
    for t in range(nt):
        a = tx[t]
        s = 0.0
        for i in range(ny0):
            s += Vy[t, i] * A[i, a]
        for u in range(nt):
            s += TtWT[t, u] * beta[u]
        rhs[t] = s + prec_mu[t]
    new = Pinv @ rhs
    z = np.empty(nt)
    for t in range(nt):
        z[t] = np.random.standard_normal()
    new += LinvT @ z
    for t in range(nt):
        rhs[t] = new[t] - beta[t]
    _bkg_coef(rhs, Vy, tx, C)
    _bkg_add(model, resid, wpos, Ux, C, P, ny0, nx0, rowbuf, True)
    for t in range(nt):
        beta[t] = new[t]


@njit(cache=True)
def _loop(nloop, cumw, par, geo, rx0, rx1, ry0, ry1, rarea,
          resid, model, wt, wpos, dm, dm2,
          cf3, ncl, ncw, fthr, ta, tb, tmp,
          sx, sy, sf, sreg, spos, nst, rlist, rcnt,
          Vy, Ux, tx, TtWT, beta, prec_mu, Pinv, LinvT, rhs, Abuf, Cbuf, rowbuf,
          ph_ix, ph_iy, ph_lv, ph_dd,
          sl_r, sl_q0, sl_q1, sl_kind, sl_a, sl_b, sl_x, sl_y, sl_f, sl_fac, sl_dl, sl_box,
          mv_i, mv_x, mv_y, mv_f, idxbuf, order, xs_buf, ys_buf, stats, work):
    fmin, slope = par[0], par[1]
    R, dp = par[4], par[5]
    kick = par[6]
    errf, logdf0 = par[7], par[8]
    pos_fac, flux_fac = par[9], par[10]
    P = int(par[14])
    max_nsrc = int(par[15])
    move_k = int(par[16])
    area = par[3]
    nregx, nregy = geo[2], geo[3]
    k = ta.size
    cap = rlist.shape[1]
    inv2k2 = 1.0 / (2 * kick * kick)
    ny0, nx0 = int(par[17]), int(par[18])
    for it in range(nloop):
        u = np.random.random()
        typ = 0
        while typ < 3 and u >= cumw[typ]:
            typ += 1
        px = np.random.randint(2)
        py = np.random.randint(2)
        stats[typ, 0] += 1
        if typ == 3:
            _gibbs_bkg(resid, model, wt, wpos, Vy, Ux, tx, TtWT, beta, prec_mu, Pinv, LinvT, P, ny0, nx0,
                       rhs, Abuf, Cbuf, rowbuf)
            stats[3, 1] += 1
            stats[3, 2] += 1
            continue
        n = nst[0]
        nsl = 0
        nph = 0
        nmv = 0
        # ------------------------------------------------ build proposals
        if typ == 0:
            if n == 0:
                continue
            n_eff = max(1.0, n * R * R / area)
            for row in range(py, nregy, 2):
                for col in range(px, nregx, 2):
                    r = row * nregx + col
                    cnt = rcnt[r]
                    if cnt == 0:
                        continue
                    if move_k <= 0 or move_k >= cnt:
                        m = cnt
                        for a in range(cnt):
                            idxbuf[a] = rlist[r, a]
                    else:
                        m = move_k
                        for a in range(cnt):
                            idxbuf[a] = rlist[r, a]
                        for a in range(m):
                            b = a + np.random.randint(cnt - a)
                            t_ = idxbuf[a]
                            idxbuf[a] = idxbuf[b]
                            idxbuf[b] = t_
                    ne = n_eff if move_k <= 0 else float(m)
                    dnorm = np.sqrt(ne * 3.0)
                    lindf = errf / dnorm * flux_fac
                    logdf = logdf0 / np.sqrt(ne)
                    lfac = 0.0
                    sl_r[nsl] = r
                    sl_q0[nsl] = nph
                    sl_kind[nsl] = _K_MOVE
                    sl_a[nsl] = nmv
                    for a in range(m):
                        i = idxbuf[a]
                        f0 = sf[i]
                        pf, lq = _flux_prop(f0, fmin, lindf, logdf)
                        lfac += lq + _logp_flux(pf, slope, fmin) - _logp_flux(f0, slope, fmin)
                        dpos = errf / dnorm / max(f0, pf) * pos_fac
                        if dpos < 1e-3:
                            dpos = 1e-3
                        x1 = sx[i] + np.random.standard_normal() * dpos
                        y1 = sy[i] + np.random.standard_normal() * dpos
                        c2, r2 = _region_of(x1, y1, par, geo)
                        if not (c2 == col and r2 == row and _in_domain(x1, y1, par)):
                            x1 = sx[i]
                            y1 = sy[i]
                        mv_i[nmv] = i
                        mv_x[nmv] = x1
                        mv_y[nmv] = y1
                        mv_f[nmv] = pf
                        nmv += 1
                        # old and new source; one stamp of the coefficient difference
                        # when the pixel and stamp size are unchanged
                        ix0, iy0 = _coefs(sx[i] + P, sy[i] + P, -f0, ta, tb, ph_dd, nph * k)
                        lv0 = _level(f0, fthr)
                        ix1, iy1 = _coefs(x1 + P, y1 + P, pf, ta, tb, ph_dd, (nph + 1) * k)
                        lv1 = _level(pf, fthr)
                        if ix0 == ix1 and iy0 == iy1 and lv0 == lv1:
                            for c in range(k):
                                ph_dd[nph * k + c] += ph_dd[(nph + 1) * k + c]
                            ph_ix[nph] = ix0
                            ph_iy[nph] = iy0
                            ph_lv[nph] = lv0
                            nph += 1
                        else:
                            ph_ix[nph] = ix0
                            ph_iy[nph] = iy0
                            ph_lv[nph] = lv0
                            ph_ix[nph + 1] = ix1
                            ph_iy[nph + 1] = iy1
                            ph_lv[nph + 1] = lv1
                            nph += 2
                    sl_b[nsl] = nmv
                    sl_q1[nsl] = nph
                    sl_fac[nsl] = lfac
                    nsl += 1
        elif typ == 1:
            for row in range(py, nregy, 2):
                for col in range(px, nregx, 2):
                    r = row * nregx + col
                    if np.random.random() < 0.5:
                        x1 = rx0[col] + np.random.random() * (rx1[col] - rx0[col]) - 0.5 - dp
                        y1 = ry0[row] + np.random.random() * (ry1[row] - ry0[row]) - 0.5 - dp
                        f1 = fmin * np.exp(np.random.exponential(1.0 / (slope - 1.0)))
                        sl_kind[nsl] = _K_BIRTH
                        sl_x[nsl] = x1
                        sl_y[nsl] = y1
                        sl_f[nsl] = f1
                        ix0, iy0 = _coefs(x1 + P, y1 + P, f1, ta, tb, ph_dd, nph * k)
                        lv0 = _level(f1, fthr)
                    else:
                        cnt = rcnt[r]
                        if cnt == 0:
                            continue
                        i = rlist[r, np.random.randint(cnt)]
                        sl_kind[nsl] = _K_DEATH
                        sl_a[nsl] = i
                        ix0, iy0 = _coefs(sx[i] + P, sy[i] + P, -sf[i], ta, tb, ph_dd, nph * k)
                        lv0 = _level(sf[i], fthr)
                    sl_r[nsl] = r
                    sl_q0[nsl] = nph
                    ph_ix[nph] = ix0
                    ph_iy[nph] = iy0
                    ph_lv[nph] = lv0
                    nph += 1
                    sl_q1[nsl] = nph
                    nsl += 1
        else:
            if n == 0:
                continue
            for row in range(py, nregy, 2):
                for col in range(px, nregx, 2):
                    r = row * nregx + col
                    cnt = rcnt[r]
                    if np.random.random() < 0.5:
                        # ---- split
                        nbright = 0
                        for a in range(cnt):
                            if sf[rlist[r, a]] > 2 * fmin:
                                idxbuf[nbright] = rlist[r, a]
                                nbright += 1
                        if nbright == 0:
                            continue
                        s = idxbuf[np.random.randint(nbright)]
                        F = sf[s]
                        uu = fmin / F + np.random.random() * (1 - 2 * fmin / F)
                        ddx = np.random.standard_normal() * kick
                        ddy = np.random.standard_normal() * kick
                        x1 = sx[s] + (1 - uu) * ddx
                        y1 = sy[s] + (1 - uu) * ddy
                        x2 = sx[s] - uu * ddx
                        y2 = sy[s] - uu * ddy
                        c1, r1 = _region_of(x1, y1, par, geo)
                        c2, r2 = _region_of(x2, y2, par, geo)
                        if not (c1 == col and r1 == row and c2 == col and r2 == row
                                and _in_domain(x1, y1, par) and _in_domain(x2, y2, par)):
                            stats[_SPLIT, 1] += 1
                            continue
                        nm = 0
                        for a in range(cnt):
                            j = rlist[r, a]
                            if j != s:
                                xs_buf[nm] = sx[j]
                                ys_buf[nm] = sy[j]
                                nm += 1
                        lW1 = _logsumexp_excl(xs_buf, ys_buf, nm, x1, y1, -1, x2, y2, True, inv2k2)
                        lW2 = _logsumexp_excl(xs_buf, ys_buf, nm, x2, y2, -1, x1, y1, True, inv2k2)
                        sl_kind[nsl] = _K_SPLIT
                        sl_r[nsl] = r
                        sl_a[nsl] = s
                        sl_b[nsl] = nbright
                        sl_x[nsl] = x1
                        sl_y[nsl] = y1
                        sl_f[nsl] = uu * F
                        mv_x[nsl] = x2
                        mv_y[nsl] = y2
                        mv_f[nsl] = (1 - uu) * F
                        mv_i[nsl] = cnt + 1
                        sl_fac[nsl] = np.log(0.5) + _logaddexp(-lW1, -lW2)
                        sl_q0[nsl] = nph
                        ix0, iy0 = _coefs(sx[s] + P, sy[s] + P, -F, ta, tb, ph_dd, nph * k)
                        ph_ix[nph] = ix0
                        ph_iy[nph] = iy0
                        ph_lv[nph] = _level(F, fthr)
                        ix0, iy0 = _coefs(x1 + P, y1 + P, uu * F, ta, tb, ph_dd, (nph + 1) * k)
                        ph_ix[nph + 1] = ix0
                        ph_iy[nph + 1] = iy0
                        ph_lv[nph + 1] = _level(uu * F, fthr)
                        ix0, iy0 = _coefs(x2 + P, y2 + P, (1 - uu) * F, ta, tb, ph_dd, (nph + 2) * k)
                        ph_ix[nph + 2] = ix0
                        ph_iy[nph + 2] = iy0
                        ph_lv[nph + 2] = _level((1 - uu) * F, fthr)
                        nph += 3
                        sl_q1[nsl] = nph
                        nsl += 1
                    else:
                        # ---- merge
                        if cnt < 2:
                            continue
                        for a in range(cnt):
                            xs_buf[a] = sx[rlist[r, a]]
                            ys_buf[a] = sy[rlist[r, a]]
                        ii = np.random.randint(cnt)
                        lWi = _logsumexp_excl(xs_buf, ys_buf, cnt, xs_buf[ii], ys_buf[ii], ii, 0.0, 0.0,
                                              False, inv2k2)
                        if not np.isfinite(lWi):
                            stats[_MERGE, 1] += 1
                            continue
                        uu = np.random.random()
                        cum = 0.0
                        jj = -1
                        for a in range(cnt):
                            if a == ii:
                                continue
                            cum += np.exp(-((xs_buf[a] - xs_buf[ii]) ** 2 + (ys_buf[a] - ys_buf[ii]) ** 2)
                                          * inv2k2 - lWi)
                            jj = a
                            if uu < cum:
                                break
                        lWj = _logsumexp_excl(xs_buf, ys_buf, cnt, xs_buf[jj], ys_buf[jj], jj, 0.0, 0.0,
                                              False, inv2k2)
                        si = rlist[r, ii]
                        sj = rlist[r, jj]
                        fi = sf[si]
                        fj = sf[sj]
                        F = fi + fj
                        uu = fi / F
                        nb_after = 1
                        for a in range(cnt):
                            if sf[rlist[r, a]] > 2 * fmin:
                                nb_after += 1
                        if fi > 2 * fmin:
                            nb_after -= 1
                        if fj > 2 * fmin:
                            nb_after -= 1
                        xm = uu * sx[si] + (1 - uu) * sx[sj]
                        ym = uu * sy[si] + (1 - uu) * sy[sj]
                        sl_kind[nsl] = _K_MERGE
                        sl_r[nsl] = r
                        sl_a[nsl] = si
                        sl_b[nsl] = sj
                        sl_x[nsl] = xm
                        sl_y[nsl] = ym
                        sl_f[nsl] = F
                        mv_i[nsl] = cnt
                        mv_f[nsl] = nb_after
                        sl_fac[nsl] = np.log(0.5) + _logaddexp(-lWi, -lWj)
                        sl_q0[nsl] = nph
                        ix0, iy0 = _coefs(sx[si] + P, sy[si] + P, -fi, ta, tb, ph_dd, nph * k)
                        ph_ix[nph] = ix0
                        ph_iy[nph] = iy0
                        ph_lv[nph] = _level(fi, fthr)
                        ix0, iy0 = _coefs(sx[sj] + P, sy[sj] + P, -fj, ta, tb, ph_dd, (nph + 1) * k)
                        ph_ix[nph + 1] = ix0
                        ph_iy[nph + 1] = iy0
                        ph_lv[nph + 1] = _level(fj, fthr)
                        ix0, iy0 = _coefs(xm + P, ym + P, F, ta, tb, ph_dd, (nph + 2) * k)
                        ph_ix[nph + 2] = ix0
                        ph_iy[nph + 2] = iy0
                        ph_lv[nph + 2] = _level(F, fthr)
                        nph += 3
                        sl_q1[nsl] = nph
                        nsl += 1
        if nsl == 0:
            continue
        # ------------------------------------------------ render + chi^2 per region
        for q in range(nph):
            _render_add(dm, ph_ix[q], ph_iy[q], ph_lv[q], ph_dd, q * k, cf3, ncl, ncw, tmp)
            work[typ, 0] += ncl[ph_lv[q]] * ncl[ph_lv[q]]
        for a in range(nsl):
            work[typ, 1] += _slot_box(ph_ix, ph_iy, ph_lv, sl_q0[a], sl_q1[a], ncl, sl_box[a])
            sl_dl[a] = -0.5 * _slot_dchi2(dm, dm2, resid, wt, ph_ix, ph_iy, ph_lv, sl_q0[a], sl_q1[a], ncl,
                                          sl_box[a])
        # ------------------------------------------------ accept
        nacc = 0
        if typ == 0:
            for a in range(nsl):
                acc = np.log(np.random.random()) < sl_dl[a] + sl_fac[a]
                _slot_apply(dm, dm2, resid, model, ph_ix, ph_iy, ph_lv, sl_q0[a], sl_q1[a], ncl, sl_box[a], acc)
                if acc:
                    nacc += 1
                    for b in range(sl_a[a], sl_b[a]):
                        i = mv_i[b]
                        sx[i] = mv_x[b]
                        sy[i] = mv_y[b]
                        sf[i] = mv_f[b]
            stats[0, 1] += nsl
            stats[0, 2] += nacc
            continue
        for a in range(nsl):
            order[a] = a
        for a in range(nsl - 1, 0, -1):
            b = np.random.randint(a + 1)
            t_ = order[a]
            order[a] = order[b]
            order[b] = t_
        n = nst[0]
        for oo in range(nsl):
            a = order[oo]
            r = sl_r[a]
            kind = sl_kind[a]
            lu = np.log(np.random.random())
            ok = False
            if kind == _K_BIRTH:
                if n >= max_nsrc or rcnt[r] >= cap:
                    stats[_CAPREJ, 0] += 1
                else:
                    fac = _count_term_birth(n, rcnt[r], rarea[r], par)
                    ok = lu < sl_dl[a] + fac
            elif kind == _K_DEATH:
                fac = -_count_term_birth(n - 1, rcnt[r] - 1, rarea[r], par)
                ok = lu < sl_dl[a] + fac
            elif kind == _K_SPLIT:
                if n >= max_nsrc or rcnt[r] >= cap:
                    stats[_CAPREJ, 0] += 1
                else:
                    fac = _split_logfactor(n, mv_i[a], sl_b[a], sf[sl_a[a]], sl_f[a], mv_f[a], sl_fac[a], par)
                    ok = lu < sl_dl[a] + fac
            else:
                fac = -_split_logfactor(n - 1, mv_i[a], int(mv_f[a]), sl_f[a], sf[sl_a[a]], sf[sl_b[a]],
                                        sl_fac[a], par)
                ok = lu < sl_dl[a] + fac
            # reuse sl_fac as the accept flag from here on
            sl_fac[a] = 1.0 if ok else 0.0
            if ok:
                nacc += 1
                if kind == _K_BIRTH or kind == _K_SPLIT:
                    n += 1
                else:
                    n -= 1
        # apply, then update the catalogue: in-place edits, removals (decreasing index), additions
        nrem = 0
        for a in range(nsl):
            acc = sl_fac[a] > 0
            _slot_apply(dm, dm2, resid, model, ph_ix, ph_iy, ph_lv, sl_q0[a], sl_q1[a], ncl, sl_box[a], acc)
            kind = sl_kind[a]
            if kind == _K_BIRTH:
                stats[_BIRTH, 1] += 1
                stats[_BIRTH, 2] += acc
            elif kind == _K_DEATH:
                stats[_DEATH, 1] += 1
                stats[_DEATH, 2] += acc
            elif kind == _K_SPLIT:
                stats[_SPLIT, 1] += 1
                stats[_SPLIT, 2] += acc
            else:
                stats[_MERGE, 1] += 1
                stats[_MERGE, 2] += acc
            if not acc:
                continue
            if kind == _K_DEATH:
                idxbuf[nrem] = sl_a[a]
                nrem += 1
            elif kind == _K_SPLIT:
                s = sl_a[a]
                sx[s] = sl_x[a]
                sy[s] = sl_y[a]
                sf[s] = sl_f[a]
            elif kind == _K_MERGE:
                si = sl_a[a]
                sx[si] = sl_x[a]
                sy[si] = sl_y[a]
                sf[si] = sl_f[a]
                idxbuf[nrem] = sl_b[a]
                nrem += 1
        if nrem > 0:
            rem = np.sort(idxbuf[:nrem])
            for b in range(nrem - 1, -1, -1):
                _remove_star(rem[b], sx, sy, sf, sreg, spos, nst, rlist, rcnt)
        for a in range(nsl):
            if sl_fac[a] <= 0:
                continue
            kind = sl_kind[a]
            if kind == _K_BIRTH:
                _add_star(sl_x[a], sl_y[a], sl_f[a], sl_r[a], sx, sy, sf, sreg, spos, nst, rlist, rcnt)
            elif kind == _K_SPLIT:
                _add_star(mv_x[a], mv_y[a], mv_f[a], sl_r[a], sx, sy, sf, sreg, spos, nst, rlist, rcnt)
        stats[typ, 1] += nsl
        stats[typ, 2] += nacc


# ----------------------------------------------------------------------------
# sampler
# ----------------------------------------------------------------------------
class FastPCATSampler(PCATSampler):
    """Single-band PCAT sampler with a compiled proposal loop (see module doc)."""

    def __init__(self, bands, config):
        super().__init__(bands, config)
        cfg = self.cfg
        if self.nb != 1:
            raise NotImplementedError('FastPCATSampler supports one band')
        bd = self.bands[0]
        terms = basis_terms(bd.basis)
        self.ta = np.array([a for a, _ in terms], np.int64)
        self.tb = np.array([b for _, b in terms], np.int64)
        k = len(terms)
        nc = int(bd.nc)
        cf = np.asarray(bd.cf, np.float32).reshape(k, nc, nc)
        w = self.wts[0]
        sigma2 = float(np.median(1.0 / w[w > 0])) if np.any(w > 0) else 1.0
        if cfg.stamp_tol is None:
            sizes, fthr = np.array([nc], np.int64), np.array([np.inf])
        else:
            sizes, fthr = stamp_ladder(cf.reshape(k, nc * nc), nc, terms, cfg.stamp_sizes,
                                       float(cfg.stamp_tol), sigma2)
        self.stamp_sizes_used = sizes
        self.stamp_fmax = fthr
        rad = nc // 2
        # (level, k, m*W): level L holds its m x m central crop in rows of W = ncw[L]
        # entries (m rounded up to a multiple of 8, zero past m)
        ncw = (sizes + 7) // 8 * 8
        self.cf3 = np.zeros((sizes.size, k, int(np.max(sizes * ncw))), np.float32)
        for L, m in enumerate(sizes):
            r = m // 2
            W = int(ncw[L])
            blk = np.zeros((k, m, W), np.float32)
            blk[:, :, :m] = cf[:, rad - r:rad + r + 1, rad - r:rad + r + 1]
            self.cf3[L, :, :m * W] = blk.reshape(k, m * W)
        self.ncl = sizes.astype(np.int64)
        self.ncw = ncw.astype(np.int64)
        self.fthr = np.asarray(fthr, np.float64)
        self.tmp = np.zeros(self.cf3.shape[2], np.float32)
        # images, with spare zero-weight columns on the right for the padded stamp rows
        ny, nx = self.data[0].shape
        xpad = int(np.max(ncw - sizes))
        self.wt32 = np.zeros((ny, nx + xpad), np.float32)
        self.wt32[:, :nx] = self.wts[0]
        self.wpos = (self.wt32 > 0).astype(np.float32)
        self.data32 = np.zeros((ny, nx + xpad), np.float32)
        self.data32[:, :nx] = self.data[0]
        self.dm = np.zeros_like(self.data32)
        self.dm2 = np.zeros_like(self.data32)
        # background
        Tm = np.ascontiguousarray(self.T[0], np.float64)
        fac = separable_factors(Tm, self.ny0, self.nx0)
        if fac is None:
            raise NotImplementedError('FastPCATSampler needs separable background templates')
        self.Vy, self.Ux, self.tx = fac
        self.Abuf = np.zeros((self.ny0, self.Ux.shape[0]))
        self.Cbuf = np.zeros((self.ny0, self.Ux.shape[0]))
        self.rowbuf = np.zeros(self.nx0)
        self.TtWT0 = np.ascontiguousarray(self.TtWT[0])
        if cfg.float_bkg:
            prec = self.TtWT[0] + np.diag(self.beta_prec[0])
            L = np.linalg.cholesky(prec)
            self.Pinv = np.ascontiguousarray(np.linalg.inv(prec))
            self.LinvT = np.ascontiguousarray(np.linalg.inv(L.T))
        else:
            self.Pinv = np.zeros((self.ntemp, self.ntemp))
            self.LinvT = np.zeros((self.ntemp, self.ntemp))
        self.prec_mu = np.ascontiguousarray(self.beta_prec[0] * self.beta_mu[0])
        self.rhs = np.zeros(self.ntemp)
        self.beta0 = np.ascontiguousarray(self.beta[0].copy())
        # stars: fixed-capacity slots
        N = int(cfg.max_nsrc)
        if self.x.size > N:
            raise ValueError(f'initial catalogue {self.x.size} > max_nsrc {N}')
        self.sx = np.zeros(N)
        self.sy = np.zeros(N)
        self.sf = np.zeros(N)
        self.sreg = np.zeros(N, np.int64)
        self.spos = np.zeros(N, np.int64)
        self.nst = np.zeros(1, np.int64)
        n0 = self.x.size
        self.sx[:n0], self.sy[:n0], self.sf[:n0] = self.x, self.y, self.f[0]
        self.nst[0] = n0
        R = self.R
        self.nlab_max = (int(np.ceil(self.Lx / R)) + 1) * (int(np.ceil(self.Ly / R)) + 1)
        self.rcap = int(cfg.region_cap)
        self.rlist = np.zeros((self.nlab_max, self.rcap), np.int64)
        self.rcnt = np.zeros(self.nlab_max, np.int64)
        # per-proposal buffers
        nsl = self.nlab_max
        self.sl = dict(r=np.zeros(nsl, np.int64), q0=np.zeros(nsl, np.int64), q1=np.zeros(nsl, np.int64),
                       kind=np.zeros(nsl, np.int64), a=np.zeros(nsl, np.int64), b=np.zeros(nsl, np.int64),
                       x=np.zeros(nsl), y=np.zeros(nsl), f=np.zeros(nsl), fac=np.zeros(nsl), dl=np.zeros(nsl),
                       box=np.zeros((nsl, 5), np.int64))
        nph = 2 * N + 3 * nsl
        self.ph_ix = np.zeros(nph, np.int64)
        self.ph_iy = np.zeros(nph, np.int64)
        self.ph_lv = np.zeros(nph, np.int64)
        self.ph_dd = np.zeros(nph * k)
        nmv = max(N, nsl)
        self.mv_i = np.zeros(nmv, np.int64)
        self.mv_x = np.zeros(nmv)
        self.mv_y = np.zeros(nmv)
        self.mv_f = np.zeros(nmv)
        self.idxbuf = np.zeros(max(self.rcap, nsl) + 1, np.int64)
        self.order = np.zeros(nsl, np.int64)
        self.xs_buf = np.zeros(self.rcap + 2)
        self.ys_buf = np.zeros(self.rcap + 2)
        self.fstats = np.zeros((9, 3), np.int64)
        # pixels rendered / visited by the chi^2, per proposal type
        self.work = np.zeros((4, 2), np.int64)
        self.dd1 = np.zeros(k)
        legacy = 1.0 if cfg.count_prior == 'legacy' else 0.0
        ny0, nx0 = self.ny0, self.nx0
        self.par = np.array([self.fmin, self.slope, self.pen, self.area, float(R), self.dpad,
                             float(cfg.kickrange), float(self.errf[0]), float(cfg.logdf),
                             float(cfg.pos_step_fac), float(cfg.flux_step_fac), self.Lx, self.Ly,
                             legacy, float(self.pads[0]), float(cfg.max_nsrc), float(cfg.move_nstar),
                             float(ny0), float(nx0)])
        self.geo = np.zeros(4, np.int64)

    # ------------------------------------------------------------------ helpers
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
        self.geo[:] = (self.ox, self.oy, self.nregx, self.nregy)
        self.par[1] = self.slope
        if not _assign_regions(self.sx, self.sy, self.nst, self.sreg, self.spos, self.rlist, self.rcnt,
                               self.par, self.geo):
            raise RuntimeError(f'more than region_cap={self.rcap} sources in one region')

    def _sync_out(self):
        n = int(self.nst[0])
        self.x = self.sx[:n].copy()
        self.y = self.sy[:n].copy()
        self.f = self.sf[:n].copy()[None, :]
        self.beta[0] = self.beta0

    def _rebuild_fast(self):
        _rebuild(self.model32, self.resid32, self.data32, self.wpos, self.sx, self.sy, self.sf, self.nst,
                 self.cf3, self.ncl, self.ncw, self.fthr, self.ta, self.tb, self.tmp, self.dd1,
                 self.beta0, self.Vy, self.Ux, self.tx, self.Cbuf, self.rowbuf, int(self.pads[0]),
                 self.ny0, self.nx0)

    def run_loop(self, nloop, cumw):
        sl = self.sl
        _loop(int(nloop), cumw, self.par, self.geo, self.rx0, self.rx1, self.ry0, self.ry1, self.rarea,
              self.resid32, self.model32, self.wt32, self.wpos, self.dm, self.dm2,
              self.cf3, self.ncl, self.ncw, self.fthr, self.ta, self.tb, self.tmp,
              self.sx, self.sy, self.sf, self.sreg, self.spos, self.nst, self.rlist, self.rcnt,
              self.Vy, self.Ux, self.tx, self.TtWT0, self.beta0, self.prec_mu, self.Pinv, self.LinvT,
              self.rhs, self.Abuf, self.Cbuf, self.rowbuf, self.ph_ix, self.ph_iy, self.ph_lv, self.ph_dd,
              sl['r'], sl['q0'], sl['q1'], sl['kind'], sl['a'], sl['b'], sl['x'], sl['y'], sl['f'],
              sl['fac'], sl['dl'], sl['box'], self.mv_i, self.mv_x, self.mv_y, self.mv_f, self.idxbuf, self.order,
              self.xs_buf, self.ys_buf, self.fstats, self.work)

    # ------------------------------------------------------------------ driver
    def run(self):
        cfg = self.cfg
        names = [t for t in _TYPES if cfg.move_weights.get(t, 0) > 0]
        if not cfg.float_bkg and 'bkg' in names:
            names.remove('bkg')
        wv = np.array([cfg.move_weights.get(t, 0.0) if t in names else 0.0 for t in _TYPES], float)
        cumw = np.cumsum(wv / wv.sum())
        cumw[np.flatnonzero(wv > 0)[-1]:] = 1.0 + 1e-12
        _seed(int(cfg.seed) * 7919 + 17)
        self.model32 = np.zeros_like(self.data32)
        self.resid32 = np.zeros_like(self.data32)
        ns_max = int(cfg.nsamp)
        nburn_fixed = None if cfg.burnin is None else int(cfg.burnin)
        rec = dict(n=np.zeros(ns_max, int), logL=np.zeros(ns_max), chi2=np.zeros((ns_max, 1)),
                   beta=np.zeros((ns_max, 1, self.ntemp)), slope=np.zeros(ns_max), time=np.zeros(ns_max))
        cat_x, cat_y, cat_f = [], [], []
        b0 = self.bands[0]
        msum = np.zeros(b0.image.shape)
        mstore = []
        npost = 0
        t_start = time.time()
        sl0 = self.unpad[0]
        ess_hist = []
        s = 0
        stop_reason = 'nsamp'
        while s < ns_max:
            t0 = time.time()
            self._set_offset(self.rng.integers(self.R), self.rng.integers(self.R))
            self._rebuild_fast()
            self.run_loop(cfg.nloop, cumw)
            if cfg.sample_slope:
                self._sync_out()
                self.gibbs_slope()
                self.par[1] = self.slope
            self._rebuild_fast()
            chi2 = _chi2(self.resid32, self.wt32)
            n = int(self.nst[0])
            rec['n'][s] = n
            rec['logL'][s] = -0.5 * chi2
            rec['chi2'][s] = chi2
            rec['beta'][s, 0] = self.beta0
            rec['slope'][s] = self.slope
            rec['time'][s] = time.time() - t0
            cat_x.append(self.sx[:n].copy())
            cat_y.append(self.sy[:n].copy())
            cat_f.append(self.sf[:n].copy()[None, :])
            nburn = nburn_fixed if nburn_fixed is not None else int(np.floor(cfg.burnin_frac * ns_max))
            if s >= nburn:
                m = self.model32[sl0].astype(np.float64)
                msum += m
                npost += 1
                if cfg.store_models:
                    mstore.append(m.astype(np.float32))
            if cfg.verbose and (s % max(1, ns_max // 20) == 0 or s == ns_max - 1):
                acc = {t: self.fstats[_TYPES.index(t), 2] / max(1, self.fstats[_TYPES.index(t), 1])
                       for t in names}
                print(f'[pcat-fast] sample {s+1}/{ns_max} n={n} logL={-0.5*chi2:.1f} '
                      f'chi2/pix={chi2/max(1, int((self.wt32 > 0).sum())):.3f} '
                      f'bkg={self.beta0[0]:.4f} dt={rec["time"][s]:.3f}s acc='
                      + ','.join(f'{k}:{v:.3f}' for k, v in acc.items()), flush=True)
            s += 1
            if (cfg.target_ess is not None and s - nburn >= cfg.ess_min_samples
                    and (s - nburn) % cfg.ess_check_every == 0):
                post = slice(nburn, s)
                e = min(ess_geyer(rec['n'][post]), ess_geyer(rec['logL'][post]))
                ess_hist.append((s, e))
                if e >= cfg.target_ess:
                    stop_reason = 'target_ess'
                    break
        ns = s
        for key in rec:
            rec[key] = rec[key][:ns]
        self._sync_out()
        nmax = max([c.size for c in cat_x] + [1])
        X = np.full((ns, nmax), np.nan)
        Y = np.full((ns, nmax), np.nan)
        Fl = np.full((ns, 1, nmax), np.nan)
        for q in range(ns):
            kq = cat_x[q].size
            X[q, :kq] = cat_x[q]
            Y[q, :kq] = cat_y[q]
            Fl[q, :, :kq] = cat_f[q]
        rec.update(x=X, y=Y, f=Fl)
        nburn = nburn_fixed if nburn_fixed is not None else int(np.floor(cfg.burnin_frac * ns_max))
        nburn = min(nburn, ns)
        model_mean = [msum / max(1, npost)]
        model_median = [np.median(np.array(mstore), axis=0)] if (cfg.store_models and mstore) else [None]
        data = [np.where(self.wts[0][sl0] > 0, self.data[0][sl0], np.nan)]
        res_mean = [data[0] - model_mean[0]]
        res_med = [None if model_median[0] is None else data[0] - model_median[0]]
        acc = {}
        for t in names:
            r = _TYPES.index(t)
            acc[t] = dict(nprop=int(self.fstats[r, 0]), nreg=int(self.fstats[r, 1]),
                          nacc=int(self.fstats[r, 2]),
                          frac=self.fstats[r, 2] / max(1, self.fstats[r, 1]))
        for kname, r in (('birth', _BIRTH), ('death', _DEATH), ('split', _SPLIT), ('merge', _MERGE)):
            acc[kname] = dict(nreg=int(self.fstats[r, 1]), nacc=int(self.fstats[r, 2]),
                              frac=self.fstats[r, 2] / max(1, self.fstats[r, 1]))
        runtime = time.time() - t_start
        info = dict(runtime_s=runtime, margin=self.margin, margin_req=self.margin_req,
                    regsize=self.R, pen=self.pen, area=self.area, errf=list(map(float, self.errf)),
                    errf_eff=self.errf_eff, color_moves=False, pads=self.pads, nburn=nburn, leak=0.0,
                    ntemp=self.ntemp, nbands=1, blas='numba', shape=[list(b0.image.shape)],
                    band_names=[b0.name], noise_scale=self.noise_scale.tolist(),
                    sec_per_sample=float(np.mean(rec['time'])) if ns else 0.0,
                    ms_per_proposal=1e3 * float(np.mean(rec['time'])) / cfg.nloop if ns else 0.0,
                    engine='numba', stamp_sizes=self.stamp_sizes_used.tolist(),
                    stamp_fmax=[float(v) for v in self.stamp_fmax], move_nstar=int(cfg.move_nstar),
                    cap_rejections=int(self.fstats[_CAPREJ, 0]), nsamp_run=ns, stop_reason=stop_reason,
                    ess_history=ess_hist,
                    work_px={t: dict(render=int(self.work[_TYPES.index(t), 0]),
                                     chi2=int(self.work[_TYPES.index(t), 1]))
                             for t in _TYPES[:3]})
        cdict = asdict(cfg)
        cdict['init_catalog'] = None if cfg.init_catalog is None else 'given'
        # post_burnin() uses burnin_frac * nsamp_run
        cdict['burnin_frac'] = (nburn + 1e-6) / max(1, ns)
        return PCATResult(samples=rec, accept=acc, model_mean=model_mean, model_median=model_median,
                          residual_mean=res_mean, residual_median=res_med, config=cdict, info=info)
