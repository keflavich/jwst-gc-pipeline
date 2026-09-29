"""Per-exposure residual maps of every bright saturated star, for the position-dependent
PSF calibration (README §8).

For one star (all its good exposures from halocal_measure.py):
  1. convert its halocal_extract.py cutouts to the Exposure format of exposure.py
     (per-pixel V2V3 from the exposure's GWCS; centre at the star's adopted sky
     position mapped through that exposure's GWCS);
  2. fit the same model as the target's LOO (patternfit.py): a static sky-fixed pattern
     Q + per-exposure flux, pedestal/gradient, cubic distortion, PSF width and smooth
     star-centred halo -- here IN-SAMPLE, all exposures together;
  3. save each exposure's residual d - model at its good pixels (r <= RMAX, covered by
     >= 3 exposures), with the pixel's exact star-relative ideal-frame offset (qx, qy),
     its inverse variance, plus the star's detector position in that exposure.

The residual of exposure e is what a static PSF cannot describe: the per-exposure PSF
change minus its mean over the star's exposures (and minus what the smooth nuisance
absorbs).  halocal_psfcal.py fits it as a function of detector position.

    QN=<grid> NUFFT_THREADS=1 NUFFT_EPS=1e-6 python halocal_patterns.py <meas.npz> <extdir> <outdir> <nsat_min> <nsat_max> [nproc]
"""
import os, sys, json, tempfile, warnings
import numpy as np
import concurrent.futures as cf
import asdf
warnings.simplefilter('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RMAX = {128: 124, 256: 250}


def star_npz(z, rows, extdir, tmpdir):
    """Exposure-format npz files for one star; returns file names and detector positions."""
    from halocal_measure import v2v3_grid
    files = []; pos = []
    for i in rows:
        root = str(z['root'][i]); sidx = int(z['sidx'][i])
        c = np.load(os.path.join(extdir, root+'.npz'))
        d = c[f's{sidx}_sci'].astype(float); var = c[f's{sidx}_var'].astype(float); dq = c[f's{sidx}_dq'].astype(np.uint32)
        X0, Y0 = (int(v) for v in c[f's{sidx}_org'])
        n = d.shape[0]
        w = asdf.open(os.path.join(extdir, root+'_wcs.asdf'))['wcs']
        t = w.get_transform('detector', 'v2v3')
        xt, yt = (float(v) for v in w.invert(float(z['ra'][i]), float(z['dec'][i])))
        v2, v3, _, _ = v2v3_grid(t, X0, Y0, n)
        h = 0.5
        J = np.array([[(t(xt+h, yt)[0]-t(xt-h, yt)[0])/(2*h), (t(xt, yt+h)[0]-t(xt, yt-h)[0])/(2*h)],
                      [(t(xt+h, yt)[1]-t(xt-h, yt)[1])/(2*h), (t(xt, yt+h)[1]-t(xt, yt-h)[1])/(2*h)]])
        var = np.where(np.isfinite(var) & (var > 0), var, np.nan)
        fn = os.path.join(tmpdir, f'st_e{len(files)+1}.npz')
        np.savez(fn, d=d, dq=dq, var=var, vf=np.zeros_like(d), v2=v2, v3=v3, X0=X0, Y0=Y0,
                 xt=xt-X0, yt=yt-Y0, J=J, expstart=float(z['expstart'][i]), fn=root)
        files.append(fn); pos.append((xt, yt))
    return files, np.array(pos)


def run_star(args):
    measfn, extdir, outdir, sid, det = args
    out = os.path.join(outdir, f'{det}_s{sid}.npz')
    if os.path.exists(out):
        return sid, 'exists'
    import patternfit as pf
    z = np.load(measfn)
    rows = np.nonzero((z['sid'] == sid) & (z['det'] == det))[0]
    with tempfile.TemporaryDirectory(dir=outdir) as tmp:
        files, pos = star_npz(z, rows, extdir, tmp)
        J1 = np.load(files[0])['J']
        exps = pf.build('st', files, J1, nsec=2)
        wt = np.median(np.concatenate([e.w for e in exps])); lam = 1e-4*wt
        # a cheap Q suffices: its error is common to all of the star's exposures, and the
        # calibration regresses on position terms demeaned within the star
        Q = pf.solve_Q(exps, lam, Q0=None, maxiter=60, nprox=2, verbose=False)
        for it in range(1):
            for k, e in enumerate(exps):
                pf.fit_nuisance(e, Q, anchor=(k == 0))
            Q = pf.solve_Q(exps, lam, Q0=Q, maxiter=60, nprox=2, verbose=False)
        for k, e in enumerate(exps):
            pf.fit_nuisance(e, Q, anchor=(k == 0))
        F = pf.to_F(Q)
        hw = exps[0].ex.n//2
        rmax = RMAX.get(hw, hw-6)
        res = {}
        _, cnt = pf.train_weight_maps(exps)
        for k, e in enumerate(exps):
            pred = e.model_Q(F)+e.add()
            r = e.d-pred
            ix = pf.node(e.qx); iy = pf.node(e.qy)
            dq = e.ex.dq.ravel()[e.pix]
            rr = np.hypot(e.qx, e.qy)
            m = (rr <= rmax) & ((dq & 6) == 0) & (cnt[iy, ix] >= 3)
            res[f'e{k}_qx'] = e.qx[m].astype(np.float32)
            res[f'e{k}_qy'] = e.qy[m].astype(np.float32)
            res[f'e{k}_r'] = r[m].astype(np.float32)
            res[f'e{k}_w'] = e.w[m].astype(np.float32)
            res[f'e{k}_a'] = np.float64(e.a)
        # the static pattern itself (star + scene), for the star's flux normalisation
        np.savez_compressed(out, N=pf.N, pos=pos, rows=rows, root=z['root'][rows], expstart=z['expstart'][rows],
                            nsat=z['n'][rows], Q=Q.astype(np.float32), **res)
    return sid, f'{len(files)} exposures'


if __name__ == '__main__':
    measfn, extdir, outdir = sys.argv[1:4]
    nmin, nmax = int(sys.argv[4]), int(sys.argv[5])
    nproc = int(sys.argv[6]) if len(sys.argv) > 6 else 2
    os.makedirs(outdir, exist_ok=True)
    z = np.load(measfn)
    jobs = []
    for det in np.unique(z['det']):
        m = z['det'] == det
        for s in np.unique(z['sid'][m]):
            ii = m & (z['sid'] == s)
            ns = np.median(z['n'][ii])
            if ii.sum() >= 4 and nmin <= ns < nmax:
                jobs.append((ns, (measfn, extdir, outdir, int(s), str(det))))
    jobs = [j for _, j in sorted(jobs, key=lambda t: -t[0])]
    print(len(jobs), 'stars', flush=True)
    with cf.ProcessPoolExecutor(nproc) as ex:
        futs = {ex.submit(run_star, j): j for j in jobs}
        for i, f in enumerate(cf.as_completed(futs)):
            try:
                print(i, *f.result(), flush=True)
            except (ValueError, np.linalg.LinAlgError, IndexError) as err:
                print(i, 'FAILED', futs[f][3], futs[f][4], repr(err), flush=True)
