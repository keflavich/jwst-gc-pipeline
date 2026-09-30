"""LOO on a crop around the target with configurable band limit (env KMAX), grid size (QN)
and grid spacing (QH).  python loo_crop.py <tag> <crop_halfwidth_px> <held-out list, e.g. 1,3>"""
import numpy as np, sys, os, time
import patternfit as qfit3
from patternfit import N, to_F, build, solve_Q, fit_nuisance, train_weight_maps
tag = sys.argv[1]; crop = int(sys.argv[2]); ks = [int(k) for k in sys.argv[3].split(',')]
files = [f'tgt_e{i}.npz' for i in range(1, 7)]
J1 = np.load(files[0])['J']
exps = build('tgt', files, J1, nsec=2, crop=crop)
wt = np.median(np.concatenate([e.w for e in exps])); lam = 1e-4*wt
Q = solve_Q(exps, lam, Q0=None, maxiter=80, nprox=3)
for it in range(3):
    for k, e in enumerate(exps):
        fit_nuisance(e, Q, anchor=(k == 0))
    Q = solve_Q(exps, lam, Q0=Q, maxiter=60, nprox=2)
    print(tag, 'joint it', it, flush=True)
for k, e in enumerate(exps):
    fit_nuisance(e, Q, anchor=(k == 0))
for k in ks:
    train = [e for j, e in enumerate(exps) if j != k-1]
    Qk = solve_Q(train, lam, Q0=Q, maxiter=60, nprox=3, verbose=False)
    e = exps[k-1]; fit_nuisance(e, Qk, niter=4)
    pred = e.model_Q(to_F(Qk))+e.add()
    wsum, cntf = train_weight_maps(train)
    ix = qfit3.node(e.qx); iy = qfit3.node(e.qy)
    varQ = e.a**2/np.maximum(wsum[iy, ix], 1e-30)
    chi = (e.d-pred)/np.sqrt(1/e.w+varQ)
    R = np.hypot(e.qx, e.qy); dq = e.ex.dq.ravel()[e.pix]; ok = (cntf[iy, ix] >= 3) & ((dq & 6) == 0)
    print(f'{tag} LOO e{k}:', ' '.join(f'{lo}-{hi}:{1.4826*np.median(np.abs(chi[ok&(R>=lo)&(R<hi)])):.2f}' for lo, hi in [(30, 50), (50, 80), (80, 120), (120, 200), (200, 300)]), flush=True)
    n = e.ex.n; out = np.full(n*n, np.nan); out[e.pix] = pred; v = np.full(n*n, np.nan); v[e.pix] = varQ
    np.savez_compressed(f'{tag}_e{k}.npz', pred=out.reshape(n, n), varQ=v.reshape(n, n))
