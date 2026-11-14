"""LOO on a crop around the star with configurable band limit (env KMAX), grid size (QN)
and grid spacing (QH).

    python loo_crop.py <tag> <crop_halfwidth_px> <held-out dithers, e.g. 1,3> [--prefix tgt]

reads <prefix>_e{1..6}.npz (cutout.py output; missing dithers, e.g. where the star fell
off the detector, are skipped) and writes <tag>_e<k>.npz with the LOO prediction, model
variance, training coverage and flux scale.  Held-out dithers are given by dither number.
MINCOV (env, default 3): minimum number of training dithers covering a pattern node for
a pixel to be scored.
NOISEONLY (env, <seed>): calibration of the chi statistic on simulated data (issue #1008).
The cold-start joint fit runs on the real data (stage A); its model plus Gaussian noise of
variance 1/w becomes the data of every exposure.  All fitted quantities are then discarded
and the full cold-start fit and LOO are repeated on the simulated data (stage B).  The
simulated data are written as `d` to <tag>_e<k>.npz, which evaluate_loo.py scores in place
of the real data, and the per-bin sigma(chi) to <tag>_e<k>_chi.json."""
import numpy as np, os, argparse, json
import patternfit as qfit3
from patternfit import to_F, build, solve_Q, fit_nuisance, train_weight_maps
from noise_only import draw_noise_only, radial_report, BINS, BINS_NOISEONLY
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
NOISEONLY = os.environ.get('NOISEONLY')


def joint_fit(sim=None):
    """Cold-start Q solve, three joint iterations and a final nuisance fit.  sim: simulated
    data (one array per exposure) that replace the observed data before any fit."""
    exps = build(args.prefix, files, J1, nsec=2, crop=crop)
    if sim is not None:
        for e, d in zip(exps, sim):
            e.d = d
    wt = np.median(np.concatenate([e.w for e in exps])); lam = 1e-4*wt
    Q = solve_Q(exps, lam, Q0=None, maxiter=80, nprox=3)
    for it in range(3):
        for k, e in enumerate(exps):
            fit_nuisance(e, Q, anchor=(k == 0))
        Q = solve_Q(exps, lam, Q0=Q, maxiter=60, nprox=2)
        print(tag, 'joint it', it, flush=True)
    for k, e in enumerate(exps):
        fit_nuisance(e, Q, anchor=(k == 0))
    return exps, Q, lam


exps, Q, lam = joint_fit()
if NOISEONLY is not None:
    rng = np.random.default_rng(int(NOISEONLY))
    sim = [draw_noise_only(e.model_Q(to_F(Q))+e.add(), e.d, e.w, rng) for e in exps]
    print(tag, 'NOISEONLY seed', NOISEONLY, ': repeating the cold-start fit on simulated data', flush=True)
    # stage B: fresh exposures and Q0=None, so only the simulated pixel values carry over
    exps, Q, lam = joint_fit(sim)
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
    bins = BINS if NOISEONLY is None else BINS_NOISEONLY
    rep = radial_report(chi, R, ok, bins)
    print(f'{tag} LOO e{k}:', ' '.join(f'{b}:{v["sig"]:.2f}' for b, v in rep.items()), flush=True)
    n = e.ex.n
    out = {kk: np.full(n*n, np.nan) for kk in ['pred', 'varQ', 'cov']}
    out['pred'][e.pix] = pred; out['varQ'][e.pix] = varQ; out['cov'][e.pix] = cntf[iy, ix]
    extra = {}
    if NOISEONLY is not None:
        # the simulated data, so that evaluate_loo.py does not score the prediction against the real ones
        out['d'] = np.full(n*n, np.nan); out['d'][e.pix] = e.d
        extra = dict(noiseonly_seed=int(NOISEONLY))
        with open(f'{tag}_e{k}_chi.json', 'w') as fh:
            json.dump(dict(tag=tag, heldout=k, seed=int(NOISEONLY), crop=crop, mincov=MINCOV, bins=rep), fh, indent=1)
        print(f'{tag} LOO e{k} centred:', ' '.join(f'{b}:{v["sig_centred"]:.2f}' for b, v in rep.items()),
              ' chi2:', ' '.join(f'{b}:{v["chi2"]:.2f}' for b, v in rep.items()), flush=True)
    np.savez_compressed(f'{tag}_e{k}.npz', **{kk: v.reshape(n, n) for kk, v in out.items()}, a=e.a, ntrain=len(train), **extra)
print(tag, 'FINISHED', flush=True)
