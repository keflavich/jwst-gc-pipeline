"""Per-exposure scattered-halo change of every saturated F480M star in program 10678.

    python perexp_measure.py <meas.npz> <out.npz>

Input: ``meas.npz`` of ``halocal_measure.py`` -- for every saturated star in every
10678 F480M ``_cal`` (NRCALONG + NRCBLONG, 68 visits x 6 dithers, one roll), the
off-spike halo median H in 12 annuli (15-250 px) x 12 azimuth sectors about the
star's sky position, with its noise He.  One roll, so a star samples the same sky
in every dither: the scene is common to all its exposures and cancels in their
differences, which are therefore the halo's change.

For each star-visit with >= MIN_EXP exposures:

* reference ``Hbar(r, theta)`` = mean over the exposures;
* scene level ``B`` and halo profile ``I(r) = p(r) - B``, from a fit
  ``p(r) = B + A (r / 30)**-alpha`` to the sector-median reference profile;
* the fractional change of each exposure, cell by cell::

      delta_e(r, theta) = (H_e - Hbar) / I(r),     sigma = He_e / I(r)

  kept only where the halo is at least half the scene (``I > 0.5 B``, so that
  the uncertainty of ``B`` cannot scale ``delta``) and ``I > 5 He``.

Per exposure it also writes the summaries the analysis uses:

* ``a``: the amplitude, inverse-variance mean of delta over the kept cells at
  r = 15-80 px (5-sigma clipped);
* ``prof``: the azimuthal mean of delta in each annulus;
* ``dip``: the dipole ``delta(theta) = c0 + cx cos(theta) + cy sin(theta)`` fitted at
  r = 20-65 px;
* ``area``: saturated-core area relative to the star-visit mean (``n / <n>``).
"""
import sys

import numpy as np
from scipy import optimize

MIN_EXP = 4
R_AMP = (15, 80)
R_DIP = (20, 65)
CLIP = 5.0


def scene_fit(rc, p):
    ok = np.isfinite(p)
    if ok.sum() < 6:
        return None
    f = lambda x, B, A, a: B + A * (x / 30.) ** (-a)
    try:
        par, cov = optimize.curve_fit(f, rc[ok], p[ok], p0=[np.nanmin(p), max(p[ok][0] - np.nanmin(p), 1e-3), 3.],
                                      maxfev=4000)
    except (RuntimeError, ValueError):
        return None
    if not np.all(np.isfinite(par)) or par[1] <= 0:
        return None
    return par


def wmean(x, s):
    ok = np.isfinite(x) & np.isfinite(s) & (s > 0)
    if ok.sum() < 3:
        return np.nan, np.nan
    x, w = x[ok], 1 / s[ok] ** 2
    for _ in range(3):
        m = np.sum(w * x) / np.sum(w)
        keep = np.abs(x - m) * np.sqrt(w) < CLIP
        if keep.all() or keep.sum() < 3:
            break
        x, w = x[keep], w[keep]
    return np.sum(w * x) / np.sum(w), 1 / np.sqrt(np.sum(w))


def main():
    fin, fout = sys.argv[1:3]
    z = np.load(fin, allow_pickle=True)
    E = z['edges']
    rc = np.sqrt(E[1:] * E[:-1])
    nsec = z['H'].shape[2]
    th = np.radians((np.arange(nsec) + 0.5) * 360 / nsec)
    key = z['sid'] * 1000 + z['obs']
    order = np.argsort(key, kind='stable')
    bounds = np.flatnonzero(np.diff(key[order])) + 1
    rows = {k: [] for k in ('idx', 'grp', 'a', 'ae', 'c0', 'cx', 'cy', 'dipe', 'area', 'B', 'A30', 'alpha')}
    prof, profe, delta, sigma = [], [], [], []
    for g, ii in enumerate(np.split(order, bounds)):
        if len(ii) < MIN_EXP:
            continue
        H, He = z['H'][ii], z['He'][ii]
        Hbar = np.nanmean(H, 0)
        par = scene_fit(rc, np.nanmedian(Hbar, 1))
        if par is None:
            continue
        B, A, alpha = par
        I = A * (rc / 30.) ** (-alpha)                       # smooth halo profile, scene removed
        ok_r = (I > 0.5 * B)[None, :, None] & (I[None, :, None] > 5 * He)
        d = np.where(ok_r, (H - Hbar[None]) / I[None, :, None], np.nan)
        s = np.where(ok_r, He / I[None, :, None], np.nan)
        if not np.any(np.isfinite(d[:, (rc >= R_AMP[0]) & (rc < R_AMP[1])])):
            continue
        area = z['n'][ii] / z['n'][ii].mean()
        for j, i in enumerate(ii):
            ra = (rc >= R_AMP[0]) & (rc < R_AMP[1])
            a, ae = wmean(d[j, ra].ravel(), s[j, ra].ravel())
            pr = [wmean(d[j, k], s[j, k]) for k in range(len(rc))]
            rd = (rc >= R_DIP[0]) & (rc < R_DIP[1])
            x, w = d[j, rd], s[j, rd]
            T = np.broadcast_to(th, x.shape)
            m = np.isfinite(x) & np.isfinite(w)
            if m.sum() >= 8 and len(np.unique(T[m])) >= 6:
                M = np.c_[np.ones(m.sum()), np.cos(T[m]), np.sin(T[m])] / w[m][:, None]
                c, *_ = np.linalg.lstsq(M, x[m] / w[m], rcond=None)
                ce = np.sqrt(np.diag(np.linalg.pinv(M.T @ M)))
            else:
                c, ce = np.full(3, np.nan), np.full(3, np.nan)
            for k_, v in (('idx', i), ('grp', g), ('a', a), ('ae', ae), ('c0', c[0]), ('cx', c[1]), ('cy', c[2]),
                          ('dipe', np.hypot(ce[1], ce[2]) / np.sqrt(2)), ('area', area[j]), ('B', B), ('A30', A),
                          ('alpha', alpha)):
                rows[k_].append(v)
            prof.append([p[0] for p in pr]); profe.append([p[1] for p in pr])
            delta.append(d[j]); sigma.append(s[j])
    idx = np.array(rows['idx'])
    out = {k: np.array(v) for k, v in rows.items()}
    for k in ('sid', 'obs', 'expnum', 'expstart', 'xc', 'yc', 'det', 'n', 'root'):
        out[k] = z[k][idx]
    np.savez_compressed(fout, rc=rc, edges=E, theta=th, prof=np.array(prof), profe=np.array(profe),
                        delta=np.array(delta, dtype=np.float32), sigma=np.array(sigma, dtype=np.float32), **out)
    good = np.isfinite(out['a'])
    print(f'{fout}: {len(idx)} exposures of {len(np.unique(out["grp"]))} star-visits; '
          f'amplitude measured in {good.sum()}; median sigma(a) {np.nanmedian(out["ae"]):.3f}; '
          f'sigma(a) < 0.02 in {np.sum(out["ae"] < 0.02)}')


if __name__ == '__main__':
    main()
