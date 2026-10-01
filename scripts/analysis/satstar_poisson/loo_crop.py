"""LOO on a crop around the star with configurable band limit (env KMAX), grid size (QN)
and grid spacing (QH).

    python loo_crop.py <tag> <crop_halfwidth_px> <held-out dithers, e.g. 1,3> [--prefix tgt]

reads <prefix>_e{1..6}.npz (cutout.py output; missing dithers, e.g. where the star fell
off the detector, are skipped) and writes <tag>_e<k>.npz with the LOO prediction, model
variance, training coverage and flux scale.  Held-out dithers are given by dither number.
MINCOV (env, default 3): minimum number of training dithers covering a pattern node for
a pixel to be scored."""
import numpy as np, os, argparse
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
print(tag, args.prefix, 'dithers', dith, flush=True)
J1 = np.load(files[0])['J']
exps = build(args.prefix, files, J1, nsec=2, crop=crop)
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
    if k not in dith:
        print(f'{tag}: dither {k} not available, skipped', flush=True)
        continue
    j0 = dith.index(k)
    train = [e for j, e in enumerate(exps) if j != j0]
    Qk = solve_Q(train, lam, Q0=Q, maxiter=60, nprox=3, verbose=False)
    e = exps[j0]; fit_nuisance(e, Qk, niter=4)
    pred = e.model_Q(to_F(Qk))+e.add()
    wsum, cntf = train_weight_maps(train)
    ix = qfit3.node(e.qx); iy = qfit3.node(e.qy)
    varQ = e.a**2/np.maximum(wsum[iy, ix], 1e-30)
    chi = (e.d-pred)/np.sqrt(1/e.w+varQ)
    R = np.hypot(e.qx, e.qy); dq = e.ex.dq.ravel()[e.pix]; ok = (cntf[iy, ix] >= MINCOV) & ((dq & 6) == 0)
    print(f'{tag} LOO e{k}:', ' '.join(f'{lo}-{hi}:{1.4826*np.median(np.abs(chi[ok&(R>=lo)&(R<hi)])):.2f}' for lo, hi in [(30, 50), (50, 80), (80, 120), (120, 200), (200, 300)]), flush=True)
    n = e.ex.n
    out = {kk: np.full(n*n, np.nan) for kk in ['pred', 'varQ', 'cov']}
    out['pred'][e.pix] = pred; out['varQ'][e.pix] = varQ; out['cov'][e.pix] = cntf[iy, ix]
    np.savez_compressed(f'{tag}_e{k}.npz', **{kk: v.reshape(n, n) for kk, v in out.items()}, a=e.a, ntrain=len(train))
print(tag, 'FINISHED', flush=True)
