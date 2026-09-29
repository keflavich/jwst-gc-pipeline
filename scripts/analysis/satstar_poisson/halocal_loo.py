"""Leave-one-dither-out of the target with the program-wide PSF calibration (README §8).

Identical to loo_crop.py (crop, grid, nuisance model, statistics) plus, optionally:

  PSFCAL=<halocal_psfcal.npz>  a fixed, zero-parameter correction: each exposure's
        predicted PSF change at its detector position, sum_k phi_k(x_e) P_k(q), is
        removed from the data before Q and the nuisance are fitted (SCALE=fit: one scalar
        fitted on the training dithers).
  MODES=<halocal_modes.npz>, KUSE=k   the first k empirical modes of the PSF change
        (learned from the other stars, target excluded) are appended to every exposure's
        nuisance basis, so each exposure -- the held-out one included, like its smooth
        halo terms -- gets k coefficients.  MODE_ROT=<deg> rotates the maps (control:
        same structure, wrong angular placement).

    [env] python halocal_loo.py <tag> <crop> <heldout e.g. 1,2,3,5> [--prefix tgt]
"""
import os, argparse
import numpy as np
import scipy.sparse as sp
from scipy.ndimage import map_coordinates
import patternfit as qfit3
from patternfit import to_F, build, solve_Q, fit_nuisance, train_weight_maps

ap = argparse.ArgumentParser()
ap.add_argument('tag'); ap.add_argument('crop', type=int); ap.add_argument('heldout')
ap.add_argument('--prefix', default='tgt')
args = ap.parse_args()
tag = args.tag; crop = args.crop; ks = [int(k) for k in args.heldout.split(',')]
dith = [i for i in range(1, 7) if os.path.exists(f'{args.prefix}_e{i}.npz')]
files = [f'{args.prefix}_e{i}.npz' for i in dith]
MINCOV = int(os.environ.get('MINCOV', 3))
SCALE = os.environ.get('SCALE', '1')
J1 = np.load(files[0])['J']
exps = build(args.prefix, files, J1, nsec=2, crop=crop)
for e in exps:
    z = np.load(e.ex.fn)
    e.pos = np.array([float(z['X0'])+float(z['xt']), float(z['Y0'])+float(z['yt'])])
    e.d_orig = e.d.copy(); e.corr = np.zeros_like(e.d)
print(tag, 'dithers', dith, 'positions', [tuple(np.round(e.pos)) for e in exps], flush=True)


def grid_sample(M, Ng, hq, qx, qy, rc):
    iy = qy/hq+Ng//2; ix = qx/hq+Ng//2
    v = np.stack([map_coordinates(M[:, j].reshape(Ng, Ng), [iy, ix], order=1, mode='constant') for j in range(M.shape[1])], 1)
    v[np.hypot(qx, qy) > rc] = 0
    return v


if os.environ.get('PSFCAL'):
    from halocal_psfcal import phi
    cal = np.load(os.environ['PSFCAL'])
    Ng = int(cal['Ng']); P = cal['P'].reshape(Ng*Ng, -1).astype(float)
    for e in exps:
        e.corr = grid_sample(P, Ng, float(cal['hq']), e.qx0, e.qy0, float(cal['rc']))@phi(e.pos[None], int(cal['lp']))[0]

if os.environ.get('MODES'):
    md = np.load(os.environ['MODES'])
    k = int(os.environ.get('KUSE', md['K'])); rot = float(os.environ.get('MODE_ROT', 0))
    Ng = int(md['Ng']); M = md['M'][:, :k].astype(float)
    for e in exps:
        c, s = np.cos(np.radians(rot)), np.sin(np.radians(rot))
        qx = c*e.qx0-s*e.qy0; qy = s*e.qx0+c*e.qy0
        img = grid_sample(M, Ng, float(md['hq']), qx, qy, float(md['rc']))
        e.B = sp.hstack([e.B, sp.csc_matrix(img)], format='csc'); e.h = np.zeros(e.B.shape[1])
    print(tag, f'{k} modes appended to every exposure (rotation {rot} deg)', flush=True)


def apply(s):
    for e in exps:
        e.d = e.d_orig-s*e.corr


s = 0.0 if SCALE == '0' or not os.environ.get('PSFCAL') else 1.0
apply(s)
wt = np.median(np.concatenate([e.w for e in exps])); lam = 1e-4*wt
Q = solve_Q(exps, lam, Q0=None, maxiter=80, nprox=3)
for it in range(3):
    for k_, e in enumerate(exps):
        fit_nuisance(e, Q, anchor=(k_ == 0))
    Q = solve_Q(exps, lam, Q0=Q, maxiter=60, nprox=2)
for k_, e in enumerate(exps):
    fit_nuisance(e, Q, anchor=(k_ == 0))
for k in ks:
    if k not in dith:
        continue
    j0 = dith.index(k)
    train = [e for j, e in enumerate(exps) if j != j0]
    sk = s
    if SCALE == 'fit' and os.environ.get('PSFCAL'):
        num = den = 0.0
        F = to_F(Q)
        for e in train:
            r = e.d_orig-(e.model_Q(F)+e.add())
            num += np.sum(e.w*r*e.corr); den += np.sum(e.w*e.corr**2)
        sk = num/den
        apply(sk)
    Qk = solve_Q(train, lam, Q0=Q, maxiter=60, nprox=3, verbose=False)
    e = exps[j0]; fit_nuisance(e, Qk, niter=4)
    pred = e.model_Q(to_F(Qk))+e.add()
    wsum, cntf = train_weight_maps(train)
    ix = qfit3.node(e.qx); iy = qfit3.node(e.qy)
    varQ = e.a**2/np.maximum(wsum[iy, ix], 1e-30)
    chi = (e.d-pred)/np.sqrt(1/e.w+varQ)
    R = np.hypot(e.qx, e.qy); dq = e.ex.dq.ravel()[e.pix]; ok = (cntf[iy, ix] >= MINCOV) & ((dq & 6) == 0)
    print(f'{tag} LOO e{k} s={sk:.3f}:', ' '.join(f'{lo}-{hi}:{1.4826*np.median(np.abs(chi[ok&(R>=lo)&(R<hi)])):.2f}'
                                                  for lo, hi in [(30, 50), (50, 80), (80, 120), (120, 200), (200, 300)]), flush=True)
    n = e.ex.n
    out = {kk: np.full(n*n, np.nan) for kk in ['pred', 'varQ', 'cov', 'corr']}
    out['pred'][e.pix] = pred+sk*e.corr; out['varQ'][e.pix] = varQ; out['cov'][e.pix] = cntf[iy, ix]; out['corr'][e.pix] = sk*e.corr
    np.savez_compressed(f'{tag}_e{k}.npz', **{kk: v.reshape(n, n) for kk, v in out.items()}, a=e.a, s=sk, ntrain=len(train), h=e.h)
    if SCALE == 'fit' and os.environ.get('PSFCAL'):
        apply(s)
print(tag, 'FINISHED', flush=True)
