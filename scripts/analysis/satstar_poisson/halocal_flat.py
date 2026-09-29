"""Detector-fixed multiplicative error (flat self-calibration) from the halo residuals of
the program's saturated stars (README §8).

If a gain error delta(x) is fixed to the detector pixel x, every star-exposure residual
satisfies r = delta(x) * I + (terms common to the star's exposures), with I the model
intensity at that pixel (a_e * Q at the pixel's star-relative offset).  delta is
estimated per detector pixel from all star-exposures covering it:

    delta(x) = sum w I r / sum w I^2

using each pixel's exact detector coordinates (the star-relative ideal offset q mapped
back through the exposure's GWCS: v2v3 = v_star + J0 q -> detector).  The stars are split
in two halves A and B; the test is whether delta_A and delta_B agree (correlation at
well-covered pixels, optionally after smoothing by S px), and whether delta_A lowers
chi^2 on B.

    python halocal_flat.py <meas.npz> <det> <out.npz> <patdir> [<patdir> ...]
"""
import os, sys, glob, json, warnings
import numpy as np
import asdf
from scipy import ndimage
warnings.simplefilter('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from halocal_measure import J0
from halocal_psfcal import amplitudes

EXT = None
RMIN = 20


def detector_coords(z, e, extdir):
    root = str(z['root'][e]); pos = z['pos'][e]
    w = asdf.open(os.path.join(extdir, root+'_wcs.asdf'))['wcs']
    t = w.get_transform('detector', 'v2v3')
    vs = np.array(t(float(pos[0]), float(pos[1])))
    qx = z[f'e{e}_qx'].astype(float); qy = z[f'e{e}_qy'].astype(float)
    v2 = vs[0]+J0[0, 0]*qx+J0[0, 1]*qy; v3 = vs[1]+J0[1, 0]*qx+J0[1, 1]*qy
    x, y = t.inverse(v2, v3)
    return np.round(x).astype(int), np.round(y).astype(int)


def main():
    measfn, det, out = sys.argv[1:4]
    patdirs = sys.argv[4:]
    extdir = os.path.join(os.path.dirname(measfn), 'ext')
    meas = np.load(measfn)
    Fs, tsid, _ = amplitudes(meas, det)
    rng = np.random.default_rng(11)
    acc = np.zeros((2, 2, 2048*2048))          # half, (num, den)
    items = []
    for pd in patdirs:
        for fn in sorted(glob.glob(os.path.join(pd, f'{det}_s*.npz'))):
            sid = int(os.path.basename(fn).split('_s')[1].split('.')[0])
            if sid == tsid:
                continue
            z = np.load(fn)
            N = int(z['N']); Q = z['Q'].astype(float)
            ne = sum(1 for k in z.files if k.endswith('_a'))
            h = int(rng.integers(2))
            for e in range(ne):
                qx = z[f'e{e}_qx'].astype(float); qy = z[f'e{e}_qy'].astype(float)
                rad = np.hypot(qx, qy); m = rad >= RMIN
                x, y = detector_coords(z, e, extdir)
                ok = m & (x >= 0) & (x < 2048) & (y >= 0) & (y < 2048)
                I = float(z[f'e{e}_a'])*Q[np.clip(np.round(qy).astype(int)+N//2, 0, N-1), np.clip(np.round(qx).astype(int)+N//2, 0, N-1)]
                r = z[f'e{e}_r'].astype(float); w = z[f'e{e}_w'].astype(float)
                ok &= I > 0
                idx = (y*2048+x)[ok]
                acc[h, 0] += np.bincount(idx, weights=(w*I*r)[ok], minlength=2048*2048)
                acc[h, 1] += np.bincount(idx, weights=(w*I*I)[ok], minlength=2048*2048)
                items.append((fn, e, h, idx.astype(np.int32), (w*I*I)[ok].astype(np.float32), (w*I*r)[ok].astype(np.float32), (w*r*r)[ok].astype(np.float32), rad[ok].astype(np.float32)))
    print(det, len(items), 'star-exposures', flush=True)
    dA = np.where(acc[0, 1] > 0, acc[0, 0]/np.maximum(acc[0, 1], 1e-30), np.nan)
    dB = np.where(acc[1, 1] > 0, acc[1, 0]/np.maximum(acc[1, 1], 1e-30), np.nan)
    # noise on delta is 1/sqrt(sum w I^2): compare where both halves are well measured
    sA = 1/np.sqrt(np.maximum(acc[0, 1], 1e-30)); sB = 1/np.sqrt(np.maximum(acc[1, 1], 1e-30))
    summ = {}
    for smooth in (0, 1, 2, 4):
        if smooth:
            def sm(num, den):
                return ndimage.gaussian_filter(num.reshape(2048, 2048), smooth).ravel(), ndimage.gaussian_filter(den.reshape(2048, 2048), smooth).ravel()
            nA, gA = sm(acc[0, 0], acc[0, 1]); nB, gB = sm(acc[1, 0], acc[1, 1])
            a = np.where(gA > 0, nA/np.maximum(gA, 1e-30), np.nan); b = np.where(gB > 0, nB/np.maximum(gB, 1e-30), np.nan)
            ea = 1/np.sqrt(np.maximum(gA*(4*np.pi*smooth**2), 1e-30)); eb = 1/np.sqrt(np.maximum(gB*(4*np.pi*smooth**2), 1e-30))
        else:
            a, b, ea, eb = dA, dB, sA, sB
        for lim in (0.02, 0.01, 0.005):
            m = np.isfinite(a) & np.isfinite(b) & (ea < lim) & (eb < lim)
            if m.sum() < 100:
                continue
            c = np.corrcoef(a[m], b[m])[0, 1]
            summ[f'smooth{smooth}_err<{lim}'] = dict(npix=int(m.sum()), corr=float(c), rmsA=float(np.std(a[m])), errA=float(np.median(ea[m])))
            print(f'smooth {smooth} px, both halves err<{lim}: npix={m.sum():8d} corr(A,B)={c:+.3f} rms(delta_A)={np.std(a[m]):.4f} median err={np.median(ea[m]):.4f}', flush=True)
    # chi2 change on each half when the other half's delta is applied
    for smooth in (0, 2):
        tot = np.zeros((2, 6, 2))
        rb = [20, 50, 80, 124, 200, 250, 1e9]
        if smooth:
            dd = []
            for h in range(2):
                n_, g_ = ndimage.gaussian_filter(acc[h, 0].reshape(2048, 2048), smooth).ravel(), ndimage.gaussian_filter(acc[h, 1].reshape(2048, 2048), smooth).ravel()
                dd.append(np.where(g_ > 0, n_/np.maximum(g_, 1e-30), 0))
        else:
            dd = [np.nan_to_num(dA), np.nan_to_num(dB)]
        for fn, e, h, idx, wII, wIr, wrr, rad in items:
            dlt = dd[1-h][idx]
            after = wrr-2*dlt*wIr+dlt**2*wII
            for ib in range(6):
                m = (rad >= rb[ib]) & (rad < rb[ib+1])
                tot[h, ib] += [wrr[m].sum(), after[m].sum()]
        for ib in range(6):
            b0 = tot[:, ib, 0].sum(); b1 = tot[:, ib, 1].sum()
            print(f'  smooth {smooth}: r={rb[ib]}-{rb[ib+1]}: sum w r^2 {b0:.4g} -> {b1:.4g} ({(b1-b0)/b0:+.2%})', flush=True)
    np.savez_compressed(out, dA=dA.reshape(2048, 2048).astype(np.float32), dB=dB.reshape(2048, 2048).astype(np.float32),
                        sA=sA.reshape(2048, 2048).astype(np.float32), sB=sB.reshape(2048, 2048).astype(np.float32), summary=json.dumps(summ))


if __name__ == '__main__':
    main()
