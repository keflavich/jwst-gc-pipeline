"""Position-dependent PSF-change maps from the program's saturated stars (README §8).

Model for the in-sample residual of star s in exposure e (halocal_patterns.py) at a pixel
with star-relative ideal-frame offset q:

    r_{s,e}(q) = F_s  sum_k [phi_k(x_{s,e}) - <phi_k>_{s}(q)] P_k(q)  + noise

  phi_k   : 2-D Legendre polynomials of the star's detector position (order <= LP,
            constant dropped -- a position-independent PSF is absorbed by Q)
  <>_s(q) : mean over the exposures of s that cover q (what the static Q absorbs)
  F_s     : the star's halo amplitude relative to the target (halocal_fit.factorise)
  P_k(q)  : unknown maps on a super-resolved grid (HQ px, default 0.5), fitted node by
            node from the normal equations accumulated over all calibration stars.

The target star is excluded, so its correction is an out-of-sample prediction.
Cross-validation: stars are split into NFOLD folds; each fold's residuals are
predicted by the maps from the other folds.

    python halocal_psfcal.py <meas.npz> <patdir> <det> <out.npz> [LP] [HQ] [RC]
"""
import os, sys, glob, json, warnings
import numpy as np
from numpy.polynomial import legendre as L
warnings.simplefilter('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import halocal_fit as hf

NFOLD = 3
TARGET_RADEC = (266.5090306896523, -28.95658817641266)
RBINS = [15, 30, 50, 80, 120, 200, 250]


def phi(pos, lp):
    u = (pos[:, 0]-1024)/1024; v = (pos[:, 1]-1024)/1024
    out = []
    for i in range(lp+1):
        for j in range(lp+1-i):
            if i+j == 0:
                continue
            ci = np.zeros(i+1); ci[i] = 1; cj = np.zeros(j+1); cj[j] = 1
            out.append(L.legval(u, ci)*L.legval(v, cj))
    return np.stack(out, 1)


def star_design(zs, lp, hq, rc):
    """Per-pixel (node, dphi, r, w, exposure, radius) for one star file."""
    ne = sum(1 for k in zs.files if k.endswith('_a'))
    P = phi(zs['pos'], lp)
    K = P.shape[1]
    # coverage on the 1-px grid: which exposures have a pixel at each integer offset
    M = int(2*rc+3)
    cnt = np.zeros((M, M)); sphi = np.zeros((M, M, K))
    per = []
    for e in range(ne):
        qx = zs[f'e{e}_qx'].astype(float); qy = zs[f'e{e}_qy'].astype(float)
        i1 = np.clip(np.round(qy).astype(int)+M//2, 0, M-1); j1 = np.clip(np.round(qx).astype(int)+M//2, 0, M-1)
        pres = np.zeros((M, M), bool); pres[i1, j1] = True
        cnt += pres; sphi += pres[..., None]*P[e][None, None, :]
        per.append((qx, qy, i1, j1))
    out = []
    Ng = int(2*rc/hq)+1
    for e in range(ne):
        qx, qy, i1, j1 = per[e]
        c = cnt[i1, j1]
        dphi = P[e][None, :]-sphi[i1, j1]/np.maximum(c, 1)[:, None]
        ii = np.round(qy/hq).astype(int)+Ng//2; jj = np.round(qx/hq).astype(int)+Ng//2
        rr = np.hypot(qx, qy)
        m = (c >= 3) & (rr <= rc) & (ii >= 0) & (ii < Ng) & (jj >= 0) & (jj < Ng)
        out.append(dict(node=(ii*Ng+jj)[m], dphi=dphi[m], r=zs[f'e{e}_r'][m].astype(float),
                        w=zs[f'e{e}_w'][m].astype(float), rad=rr[m], qx=qx[m], qy=qy[m]))
    return out, Ng, K


def amplitudes(meas, det):
    """F_s (halo amplitude relative to the target) keyed by star id, and the target sid."""
    R = hf.build_rows(meas, det, None)
    m = meas['det'] == det
    sids = np.unique(meas['sid'][m])          # build_rows' star order
    ra = meas['ra'][m]; dec = meas['dec'][m]
    d = np.hypot((ra-TARGET_RADEC[0])*np.cos(np.radians(TARGET_RADEC[1])), dec-TARGET_RADEC[1])*3600
    tsid = int(meas['sid'][m][np.argmin(d)]) if d.min() < 0.5 else -1
    A = dict(zip(sids.tolist(), R['A'].tolist()))
    At = A.get(tsid, np.nan)
    return {s: a/At for s, a in A.items()}, tsid, At


def accumulate(files, Fs, lp, hq, rc, nfold, tsid):
    G = None
    info = []
    rng = np.random.default_rng(3)
    for fn in files:
        sid = int(os.path.basename(fn).split('_s')[1].split('.')[0])
        if sid == tsid or not np.isfinite(Fs.get(sid, np.nan)) or Fs[sid] <= 0:
            continue
        zs = np.load(fn)
        rows, Ng, K = star_design(zs, lp, hq, rc)
        if G is None:
            G = np.zeros((nfold, Ng*Ng, K*(K+1)//2), np.float32); b = np.zeros((nfold, Ng*Ng, K), np.float32)
            iu = np.triu_indices(K)
        f = int(rng.integers(nfold))
        F = Fs[sid]
        for d in rows:
            w = d['w']
            X = F*d['dphi']
            np.add.at(G[f], d['node'], (w[:, None]*X[:, iu[0]]*X[:, iu[1]]).astype(np.float32))
            np.add.at(b[f], d['node'], (w[:, None]*X*d['r'][:, None]).astype(np.float32))
        info.append((fn, sid, f, F))
    return G, b, info, Ng, K


def solve(G, b, K, ridge=1e-3):
    iu = np.triu_indices(K)
    nn = G.shape[0]
    Gm = np.zeros((nn, K, K)); Gm[:, iu[0], iu[1]] = G; Gm[:, iu[1], iu[0]] = G
    tr = np.trace(Gm, axis1=1, axis2=2)/K
    good = tr > 0
    Gm[good] += ridge*tr[good, None, None]*np.eye(K)[None]
    P = np.zeros((nn, K))
    P[good] = np.linalg.solve(Gm[good], b[good][..., None])[..., 0]
    return P


def band_limit(Pmap, Ng, hq, kmax=0.45):
    """Low-pass each map at kmax cycles per detector px (the optical cutoff)."""
    k = np.fft.fftfreq(Ng, d=hq)
    kk = np.hypot(k[None, :], k[:, None])
    out = np.empty_like(Pmap)
    for j in range(Pmap.shape[-1]):
        F = np.fft.fft2(Pmap[..., j]); F[kk > kmax] = 0
        out[..., j] = np.fft.ifft2(F).real
    return out


def predict(Pmap, Ng, hq, d, F):
    from scipy.ndimage import map_coordinates
    iy = d['qy']/hq+Ng//2; ix = d['qx']/hq+Ng//2
    vals = np.stack([map_coordinates(Pmap[..., j], [iy, ix], order=1, mode='constant') for j in range(Pmap.shape[-1])], 1)
    return F*np.sum(d['dphi']*vals, 1)


def main():
    measfn, patdir, det, out = sys.argv[1:5]
    lp = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    hq = float(sys.argv[6]) if len(sys.argv) > 6 else 0.5
    rc = float(sys.argv[7]) if len(sys.argv) > 7 else 250.
    meas = np.load(measfn)
    Fs, tsid, At = amplitudes(meas, det)
    print(det, 'target sid', tsid, 'A_t', At, flush=True)
    files = sorted(glob.glob(os.path.join(patdir, f'{det}_s*.npz')))
    G, b, info, Ng, K = accumulate(files, Fs, lp, hq, rc, NFOLD, tsid)
    print(len(info), 'calibration stars, K =', K, 'grid', Ng, flush=True)
    # cross-validation
    cv = {f'{a}-{c}': dict(chi0=[], chi1=[], chi1s=[]) for a, c in zip(RBINS[:-1], RBINS[1:])}
    per_star = []
    for f in range(NFOLD):
        Pm = solve(G[[g for g in range(NFOLD) if g != f]].sum(0), b[[g for g in range(NFOLD) if g != f]].sum(0), K)
        Pm = band_limit(Pm.reshape(Ng, Ng, K), Ng, hq)
        for fn, sid, ff, F in info:
            if ff != f:
                continue
            rows, _, _ = star_design(np.load(fn), lp, hq, rc)
            for d in rows:
                p = predict(Pm, Ng, hq, d, F)
                sw = np.sqrt(d['w'])
                for a, c in zip(RBINS[:-1], RBINS[1:]):
                    m = (d['rad'] >= a) & (d['rad'] < c)
                    if m.sum() < 50:
                        continue
                    cv[f'{a}-{c}']['chi0'].append((F, float(np.sum(d['w'][m]*d['r'][m]**2)), int(m.sum()),
                                                   float(np.sum(d['w'][m]*(d['r'][m]-p[m])**2))))
            per_star.append(sid)
    summ = {}
    for key, v in cv.items():
        arr = np.array(v['chi0'])
        if len(arr) == 0:
            continue
        for fl, lab in [(0.0, 'all'), (0.05, 'F>0.05'), (0.15, 'F>0.15')]:
            m = arr[:, 0] > fl
            if m.sum() == 0:
                continue
            c0 = arr[m, 1].sum()/arr[m, 2].sum(); c1 = arr[m, 3].sum()/arr[m, 2].sum()
            summ[f'{key} {lab}'] = dict(nexp=int(m.sum()), chi2_before=c0, chi2_after=c1,
                                        excess_removed=(c0-c1)/max(c0-1, 1e-9))
            print(f'CV r={key:8s} {lab:7s} n={m.sum():5d}  chi2 {c0:7.3f} -> {c1:7.3f}   excess removed {(c0-c1)/max(c0-1,1e-9):6.1%}', flush=True)
    Pm = solve(G.sum(0), b.sum(0), K)
    Pm = band_limit(Pm.reshape(Ng, Ng, K), Ng, hq)
    np.savez_compressed(out, P=Pm.astype(np.float32), Ng=Ng, hq=hq, lp=lp, rc=rc, K=K, tsid=tsid, At=At,
                        cv=json.dumps(summ), stars=np.array([i[1] for i in info]), F=np.array([i[3] for i in info]))
    print('wrote', out, flush=True)


if __name__ == '__main__':
    main()
