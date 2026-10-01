"""Which model predicts the per-exposure halo change?  Cross-validated by star.

    python perexp_models.py <pxh.npz> <out.json> [sigma_max]

Input: ``perexp_measure.py`` output.  Every quantity is a change relative to the
star-visit's own mean (sum over the star's exposures is ~0), so every model is
written in the same differenced form: a predictor ``q_e`` enters as
``q_e - <q>_star``.  Models for the amplitude ``a_e`` (r = 15-80 px):

* ``position``: a smooth function of detector position, ``f_det(x, y)``,
  2-D Legendre of total degree <= DEG, one per detector;
* ``xprofile`` / ``yprofile``: a free function of detector column (row),
  piecewise linear on 64-px knots, one per detector -- the fine-scale version of
  ``position`` (the Legendre cannot follow the ~250 px NRCBLONG feature);
* ``dither``: one value per dither index (1-6), per detector.  The dither pattern
  is the same in every visit, so this is "which dither" (its time order, or the
  direction of the offset), not where on the detector;
* ``exposure``: one value per exposure, shared by all stars measured in it (the
  held-out star is predicted from the OTHER stars of the same exposure);
* ``area``: ``beta * (n_e / <n> - 1)``, the saturated-core area of the same
  exposure: an observable of the exposure itself;
* combinations.

Folds are by star (``sid``), so a star never informs its own prediction.  The
score is the fraction of the measured variance of ``a`` that is explained,
after subtracting the noise variance: ``1 - (<res^2> - <ae^2>) / (<a^2> - <ae^2>)``.
"""
import json
import sys

import numpy as np
from numpy.polynomial import legendre

DEG = 3
NFOLD = 5
DETS = ('NRCALONG', 'NRCBLONG')
# the change scatters by ~0.1 star to star while sigma(a) is ~0.003: weighting by
# 1/sigma(a) alone hands the fit to a few of the brightest stars.  Weight by the
# total scatter instead.
S_INT = 0.08
XSTEP = 64
LAM = float(__import__('os').environ.get('PEREXP_LAM', 3000.))   # curvature penalty of the profiles


def good_scene_fit(z):
    """Reject star-visits whose scene fit degenerated (B <= 0 or a flat 'halo', B ~ -A)."""
    return (z['B'] > 0) & (z['alpha'] > 0.5) & (z['alpha'] < 8)


def leg2d(x, y, deg=DEG, linear=True):
    u, v = x / 1023.5 - 1, y / 1023.5 - 1
    cols = []
    for i in range(deg + 1):
        for j in range(deg + 1 - i):
            if i == j == 0 or (not linear and i + j == 1):
                continue
            ci = np.zeros(i + 1); ci[-1] = 1
            cj = np.zeros(j + 1); cj[-1] = 1
            cols.append(legendre.legval(u, ci) * legendre.legval(v, cj))
    return np.array(cols).T


def hat(q, step=XSTEP):
    """Piecewise-linear (hat) basis on knots every ``step`` px over 0-2048; the
    first hat is dropped, since the hats sum to one and a constant cancels in the
    differenced form."""
    k = np.arange(0, 2048 + step, step)
    B = np.clip(1 - np.abs(q[:, None] - k[None]) / step, 0, None)
    return B[:, 1:]


def centre(Xraw, grp):
    """Subtract each star-visit's mean of the design rows (the differenced form)."""
    X = Xraw.copy()
    for g in np.unique(grp):
        m = grp == g
        X[m] -= X[m].mean(0)
    return X


def d2(nk):
    """Second differences of (0, c_1 .. c_nk): the first knot is pinned at 0."""
    D = np.zeros((nk - 1, nk))
    for i in range(nk - 1):
        if i > 0:
            D[i, i - 1] = 1
        D[i, i] += -2
        D[i, i + 1] += 1
    return D


def design(t, which, with_penalty=False):
    """Differenced design matrix; with ``with_penalty`` also the curvature penalty
    rows of the profile blocks.  The dither pattern (+-385 px in x) constrains
    f(x + 385) - f(x) only, so a free profile has null modes of period ~385/k px;
    the penalty picks the smoothest member."""
    det, grp = t['det'], t['grp']
    blocks, pen = [], []
    for name in which:
        if name == 'position':
            # The dither pattern moves every star by the same offset, so after
            # differencing the LINEAR terms are a function of the dither index
            # only: with 'dither' in the model they are degenerate with it.
            for d in DETS:
                B = leg2d(t['xc'], t['yc'], linear='dither' not in which) * (det == d)[:, None]
                blocks.append(B)
        elif name in ('xprofile', 'yprofile'):
            # a free function of detector column (row): 64-px knots, per detector
            q = t['xc'] if name == 'xprofile' else t['yc']
            for d in DETS:
                blocks.append(hat(q) * (det == d)[:, None])
                pen.append((len(blocks) - 1, d2(blocks[-1].shape[1])))
        elif name == 'dither':
            for d in DETS:
                # dither 1 is the reference: the six indicators sum to the detector
                # indicator, which the differencing removes, so all six are degenerate
                B = np.stack([(t['expnum'] == k) & (det == d) for k in range(2, 7)], 1).astype(float)
                blocks.append(B)
        elif name == 'area':
            blocks.append((t['area'] - 1)[:, None])
    X = centre(np.hstack(blocks), grp)
    if not with_penalty:
        return X
    off = np.cumsum([0] + [b.shape[1] for b in blocks])
    P = np.zeros((sum(p.shape[0] for _, p in pen), X.shape[1]))
    r = 0
    for i, D in pen:
        P[r:r + D.shape[0], off[i]:off[i + 1]] = D
        r += D.shape[0]
    return X, P


def exposure_pred(t, test):
    """Prediction for the held-out stars from the other stars of the same exposure."""
    pred = np.zeros(len(t['a']))
    w = 1 / (t['ae'] ** 2 + S_INT ** 2)
    for r in np.unique(t['root'][test]):
        m = t['root'] == r
        tr = m & ~test
        if tr.sum() == 0:
            continue
        pred[m & test] = np.sum(w[tr] * t['a'][tr]) / np.sum(w[tr])
    # differenced form within each held-out star-visit
    for g in np.unique(t['grp'][test]):
        m = (t['grp'] == g) & test
        pred[m] -= pred[m].mean()
    return pred


def cv(t, which, rng):
    sids = np.unique(t['sid'])
    fold_of = dict(zip(sids, rng.permutation(len(sids)) % NFOLD))
    fold = np.array([fold_of[s] for s in t['sid']])
    pred = np.zeros(len(t['a']))
    lin = [w for w in which if w != 'exposure']
    X, P = design(t, lin, with_penalty=True) if lin else (None, None)
    w = 1 / np.sqrt(t['ae'] ** 2 + S_INT ** 2)
    coefs = []
    for f in range(NFOLD):
        test, train = fold == f, fold != f
        p = np.zeros(test.sum())
        if X is not None:
            A = np.vstack([X[train] * w[train, None], np.sqrt(LAM) * P])
            y = np.r_[t['a'][train] * w[train], np.zeros(len(P))]
            c, *_ = np.linalg.lstsq(A, y, rcond=None)
            p += X[test] @ c
            coefs.append(c)
        if 'exposure' in which:
            p += exposure_pred(t, test)[test]
        pred[test] = p
    return pred, (np.mean(coefs, 0) if coefs else None)


def explained(a, ae, res):
    var_a = np.mean(a ** 2) - np.mean(ae ** 2)
    return float(1 - (np.mean(res ** 2) - np.mean(ae ** 2)) / var_a)


def main():
    fin, fout = sys.argv[1:3]
    smax = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02
    z = np.load(fin, allow_pickle=True)
    t = {k: z[k] for k in ('a', 'ae', 'area', 'xc', 'yc', 'det', 'grp', 'sid', 'expnum', 'root', 'obs')}
    ok = np.isfinite(t['a']) & (t['ae'] < smax) & (t['ae'] > 0) & good_scene_fit(z)
    # keep star-visits with >= 4 good exposures, and re-centre a on the kept ones
    gs, cnt = np.unique(t['grp'][ok], return_counts=True)
    ok &= np.isin(t['grp'], gs[cnt >= 4])
    t = {k: v[ok] for k, v in t.items()}
    for g in np.unique(t['grp']):
        m = t['grp'] == g
        t['a'][m] -= t['a'][m].mean()
    rng = np.random.default_rng(42)
    out = dict(n_exposures=int(len(t['a'])), n_star_visits=int(len(np.unique(t['grp']))),
               n_stars=int(len(np.unique(t['sid']))), sigma_max=smax,
               rms_a=float(np.sqrt(np.mean(t['a'] ** 2))), rms_noise=float(np.sqrt(np.mean(t['ae'] ** 2))),
               models={})
    models = [('position',), ('xprofile',), ('yprofile',), ('xprofile', 'yprofile'), ('dither',), ('exposure',),
              ('area',), ('position', 'area'), ('xprofile', 'area'), ('xprofile', 'yprofile', 'area'),
              ('dither', 'area'), ('position', 'dither'), ('position', 'dither', 'area'), ('exposure', 'area')]
    for which in models:
        name = '+'.join(which)
        pred, coef = cv(t, which, rng)
        res = t['a'] - pred
        out['models'][name] = dict(explained=explained(t['a'], t['ae'], res),
                                   rms_residual=float(np.sqrt(np.mean(res ** 2))))
        if which == ('area',):
            out['models'][name]['beta'] = float(coef[0])
        if which in (('xprofile',), ('yprofile',)):
            k = np.arange(0, 2048 + XSTEP, XSTEP)
            nk = len(k) - 1
            out['models'][name]['knots'] = k.tolist()
            out['models'][name]['f'] = {d: [0.0] + coef[nk * i:nk * i + nk].tolist() for i, d in enumerate(DETS)}
        if which == ('dither',):
            out['models'][name]['per_dither'] = {d: [0.0] + coef[5 * i:5 * i + 5].tolist() for i, d in enumerate(DETS)}
        print(f'{name:28s} explained {out["models"][name]["explained"]:+.3f}   rms res {out["models"][name]["rms_residual"]:.4f}')
    print(f'rms a {out["rms_a"]:.4f}  rms noise {out["rms_noise"]:.4f}  N={out["n_exposures"]} exposures, '
          f'{out["n_star_visits"]} star-visits, {out["n_stars"]} stars')
    json.dump(out, open(fout, 'w'), indent=1)


if __name__ == '__main__':
    main()
