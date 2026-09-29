"""Empirical modes of the per-exposure PSF change, learned from the program's saturated
stars (README §8).

Data: the in-sample residuals r_{s,e}(q) of halocal_patterns.py (static pattern +
per-exposure nuisance removed), at the exact star-relative ideal-frame offsets q of the
pixels.  Model:

    r_{s,e}(q) = F_s  sum_k a_{s,e,k} M_k(q) + noise

M_k are learned by a whitened PCA: every star-exposure's residual in S/N units,
sqrt(w) r, on the star-centred 1-px grid (nearest node); the modes are the leading
eigenvectors of the exposure x exposure Gram matrix, converted back to flux units with
the median per-node noise sigma/F_s, band-limited at the optical cutoff (KMAX) and
stored on an HQ-px grid (bicubic).  Coefficients are always fitted at the pixels' exact
positions with their own weights.
F_s is the star's halo amplitude relative to the target (halocal_fit.factorise), so the
maps are in the target's units.  The target star is excluded.

Cross-validation by star (NFOLD folds): modes learned without the fold; on the held-out
stars K coefficients are fitted per exposure and the excess chi^2 removed is reported by
radius, next to the same fit with CONTROL modes (the learned maps rotated by 15 deg:
same power spectrum and radial structure, wrong angular placement).  The control measures
what K free lambda/D-structured maps absorb generically.

    python halocal_modes.py <meas.npz> <det> <out.npz> <K> <patdir> [<patdir> ...]
"""
import os, sys, glob, json, warnings
import numpy as np
from scipy.ndimage import map_coordinates
warnings.simplefilter('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from halocal_psfcal import amplitudes

HQ = float(os.environ.get('MODE_HQ', 0.5))
RC = float(os.environ.get('MODE_RC', 124.))
KMAX = float(os.environ.get('KMAX', 0.45))
NFOLD = 3
NITER = int(os.environ.get('MODE_NITER', 15))
RBINS = [15, 30, 50, 80, 124, 160, 200, 250]
ROT = 15.0


def load_stars(patdirs, det, Fs, tsid, rc):
    Ng = int(2*rc/HQ)+1
    stars = []
    for pd in patdirs:
        for fn in sorted(glob.glob(os.path.join(pd, f'{det}_s*.npz'))):
            sid = int(os.path.basename(fn).split('_s')[1].split('.')[0])
            F = Fs.get(sid, np.nan)
            if sid == tsid or not np.isfinite(F) or F <= 0:
                continue
            z = np.load(fn)
            ne = sum(1 for k in z.files if k.endswith('_a'))
            exps = []
            for e in range(ne):
                qx = z[f'e{e}_qx'].astype(float); qy = z[f'e{e}_qy'].astype(float)
                rad = np.hypot(qx, qy); m = rad <= rc
                if m.sum() < 1000:
                    continue
                ii = np.round(qy[m]/HQ).astype(int)+Ng//2; jj = np.round(qx[m]/HQ).astype(int)+Ng//2
                exps.append(dict(node=ii*Ng+jj, qx=qx[m], qy=qy[m], r=z[f'e{e}_r'][m].astype(float),
                                 w=z[f'e{e}_w'][m].astype(float), rad=rad[m], pos=z['pos'][e]))
            if len(exps) >= 3:
                stars.append(dict(sid=sid, F=F, exps=exps))
    return stars, Ng


def band_limit(M, Ng, hq):
    k = np.fft.fftfreq(Ng, d=hq)
    kk = np.hypot(k[None, :], k[:, None])
    out = np.empty_like(M)
    for j in range(M.shape[1]):
        F = np.fft.fft2(M[:, j].reshape(Ng, Ng)); F[kk > KMAX] = 0
        out[:, j] = np.fft.ifft2(F).real.ravel()
    return out


def sample(M, Ng, e, rot=0.0):
    if rot:
        c, s = np.cos(np.radians(rot)), np.sin(np.radians(rot))
        qx = c*e['qx']-s*e['qy']; qy = s*e['qx']+c*e['qy']
    else:
        qx, qy = e['qx'], e['qy']
    iy = qy/HQ+Ng//2; ix = qx/HQ+Ng//2
    return np.stack([map_coordinates(M[:, j].reshape(Ng, Ng), [iy, ix], order=1, mode='constant') for j in range(M.shape[1])], 1)


def fit_coeffs(Xm, e, F):
    A = F*Xm
    G = (A*e['w'][:, None]).T@A
    G += 1e-8*np.trace(G)/max(len(G), 1)*np.eye(len(G))
    return np.linalg.solve(G, (A*e['w'][:, None]).T@e['r'])


def learn(stars, Ng, K):
    """Whitened PCA on the 1-px grid: X[e, n] = sqrt(w) r (nearest node, S/N units), modes
    = leading eigenvectors of the exposure x exposure Gram matrix, converted back to flux
    units (target's) with the median per-node noise sigma/F, then band-limited and
    resampled onto the HQ grid used by sample()."""
    N1 = int(2*RC)+1
    rows = []; sig = []
    for st in stars:
        for e in st['exps']:
            i = np.round(e['qy']).astype(int)+N1//2; j = np.round(e['qx']).astype(int)+N1//2
            idx = i*N1+j
            x = np.bincount(idx, weights=e['r']*np.sqrt(e['w']), minlength=N1*N1)
            c = np.bincount(idx, minlength=N1*N1)
            s_ = np.bincount(idx, weights=1/np.sqrt(e['w'])/st['F'], minlength=N1*N1)
            rows.append(np.where(c > 0, x/np.maximum(c, 1), 0).astype(np.float32))
            sig.append(np.where(c > 0, s_/np.maximum(c, 1), np.nan).astype(np.float32))
    X = np.array(rows)
    Gm = X.astype(np.float64)@X.T.astype(np.float64)
    ev, U = np.linalg.eigh(Gm)
    order = np.argsort(ev)[::-1][:K]
    V = (U[:, order].T@X)/np.sqrt(np.maximum(ev[order], 1e-30))[:, None]       # K x N1^2, unit norm
    u = np.nanmedian(np.array(sig), 0); u = np.nan_to_num(u, nan=0.0)
    M1 = (V*u[None, :]).T                                                       # N1^2 x K, flux units
    M1 = band_limit(M1, N1, 1.0)
    # resample onto the HQ grid (bilinear) so that sample() is grid-agnostic
    g = (np.arange(Ng)-Ng//2)*HQ+N1//2
    gy, gx = np.meshgrid(g, g, indexing='ij')
    M = np.stack([map_coordinates(M1[:, k].reshape(N1, N1), [gy.ravel(), gx.ravel()], order=3, mode='constant') for k in range(K)], 1)
    return M, ev[order]


def evaluate(stars, M, Ng, Ks, rot=0.0):
    """excess chi^2 per radial bin before / after fitting the first k modes per exposure."""
    acc = {k: np.zeros((len(RBINS)-1, 3)) for k in Ks}     # [sum w r^2 before, after, npix]
    for st in stars:
        for e in st['exps']:
            Xm = sample(M, Ng, e, rot)
            for k in Ks:
                if k == 0:
                    res = e['r']
                else:
                    a = fit_coeffs(Xm[:, :k], e, st['F'])
                    res = e['r']-st['F']*Xm[:, :k]@a
                for ib, (lo, hi) in enumerate(zip(RBINS[:-1], RBINS[1:])):
                    m = (e['rad'] >= lo) & (e['rad'] < hi)
                    acc[k][ib] += [np.sum(e['w'][m]*e['r'][m]**2), np.sum(e['w'][m]*res[m]**2), m.sum()]
    return acc


def main():
    measfn, det, out, K = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    patdirs = sys.argv[5:]
    meas = np.load(measfn)
    Fs, tsid, At = amplitudes(meas, det)
    stars, Ng = load_stars(patdirs, det, Fs, tsid, RC)
    print(det, len(stars), 'stars', sum(len(s['exps']) for s in stars), 'exposures; grid', Ng, 'K', K, flush=True)
    rng = np.random.default_rng(5)
    fold = rng.integers(0, NFOLD, len(stars))
    Ks = [0]+[k for k in (1, 2, 3, 5, 8, 12, 16, 24) if k <= K]
    tot = {k: np.zeros((len(RBINS)-1, 3)) for k in Ks}; totc = {k: np.zeros((len(RBINS)-1, 3)) for k in Ks}
    for f in range(NFOLD):
        tr = [s for s, g in zip(stars, fold) if g != f]; te = [s for s, g in zip(stars, fold) if g == f]
        M, _ = learn(tr, Ng, K)
        a = evaluate(te, M, Ng, Ks); c = evaluate(te, M, Ng, Ks, rot=ROT)
        for k in Ks:
            tot[k] += a[k]; totc[k] += c[k]
        print('fold', f, 'done', flush=True)
    summ = {}
    for ib, (lo, hi) in enumerate(zip(RBINS[:-1], RBINS[1:])):
        n = tot[0][ib, 2]
        if n == 0:
            continue
        c0 = tot[0][ib, 0]/n
        line = f'r={lo:3d}-{hi:3d} n={int(n):9d} chi2={c0:7.3f}'
        for k in Ks[1:]:
            c1 = tot[k][ib, 1]/n; cc = totc[k][ib, 1]/n
            summ[f'{lo}-{hi} K={k}'] = dict(chi2_0=c0, chi2_modes=c1, chi2_control=cc,
                                           removed_modes=(c0-c1)/max(c0-1, 1e-9), removed_control=(c0-cc)/max(c0-1, 1e-9))
            line += f' | K={k}: {(c0-c1)/max(c0-1,1e-9):5.1%} (ctl {(c0-cc)/max(c0-1,1e-9):5.1%})'
        print(line, flush=True)
    M, ev = learn(stars, Ng, K)
    print('eigenvalues / N_nodes:', np.round(ev/ (np.pi*RC**2), 2), flush=True)
    np.savez_compressed(out, M=M.astype(np.float32), ev=ev, Ng=Ng, hq=HQ, rc=RC, K=K, tsid=tsid, At=At, cv=json.dumps(summ))
    print('wrote', out, flush=True)


if __name__ == '__main__':
    main()
