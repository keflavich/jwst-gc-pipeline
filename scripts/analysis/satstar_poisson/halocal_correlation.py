"""How far does the per-exposure halo change depend on detector position?  (README §8)

All 10678 visits use the same 6-point dither pattern, so two stars whose dither-1
positions differ by d have position sequences offset by d in every dither.  If the halo
change were a function of detector position, the per-dither changes of such stars would
agree for small d.  Two tests, for pairs of stars from different visits:

  halo : the 6-vector (over dithers) of the fractional change of the azimuthally averaged
         halo, halocal_fit.build_rows, bright quintile; correlation vs d
  maps : the in-sample residual maps (halocal_patterns.py), r*sqrt(w) on the star-centred
         1-px grid at r = 30-124 px, same dither; correlation vs d (full, and high-passed
         at 5 px so only the lambda/D-scale structure is compared)

    python halocal_correlation.py <meas.npz> <out.json> <patdir> [<patdir> ...]
"""
import sys, os, glob, json, warnings
import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage
warnings.simplefilter('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import halocal_fit as hf
from halocal_psfcal import amplitudes

SEPS = [(0, 50), (50, 100), (100, 200), (200, 400), (400, 700)]
RC = 124


def dither_offsets(R, ex):
    offs = {}
    for s in np.unique(R['sidx']):
        ii = np.nonzero(R['sidx'] == s)[0]; e = ex[ii]
        if 1 in e:
            i1 = ii[e == 1][0]
            for i in ii:
                offs.setdefault(int(ex[i]), []).append((R['x'][i]-R['x'][i1], R['y'][i]-R['y'][i1]))
    return {k: np.median(v, 0) for k, v in offs.items()}


def corr(u, w):
    u = u-u.mean(); w = w-w.mean()
    return float(np.sum(u*w)/np.sqrt(np.sum(u*u)*np.sum(w*w)))


def halo_test(z, det):
    R = hf.build_rows(z, det, None)
    m = z['det'] == det; ex = z['expnum'][m]; obs = z['obs'][m]
    offs = dither_offsets(R, ex)
    out = {}
    for lo, hi in [(3, 7), (7, 9)]:
        yv = np.nanmean(R['Y'][:, lo:hi], (1, 2))
        vec = []; p1 = []; A = []; ob = []
        for s in np.unique(R['sidx']):
            ii = np.nonzero(R['sidx'] == s)[0]
            if len(ii) < 5:
                continue
            v = np.full(6, np.nan); v[ex[ii]-1] = yv[ii]
            p1.append(np.mean([[R['x'][i]-offs[int(ex[i])][0], R['y'][i]-offs[int(ex[i])][1]] for i in ii], 0))
            vec.append(v); A.append(R['A'][s]); ob.append(obs[ii[0]])
        vec = np.array(vec); p1 = np.array(p1); A = np.array(A); ob = np.array(ob)
        g = A > np.nanpercentile(A, 80)
        vv, pp, oo = vec[g], p1[g], ob[g]
        pairs = cKDTree(pp).query_pairs(700, output_type='ndarray')
        pairs = pairs[oo[pairs[:, 0]] != oo[pairs[:, 1]]]
        d = np.hypot(*(pp[pairs[:, 0]]-pp[pairs[:, 1]]).T)
        cs = []
        for a, b in pairs:
            k = np.isfinite(vv[a]) & np.isfinite(vv[b])
            cs.append(corr(vv[a][k], vv[b][k]) if k.sum() >= 5 else np.nan)
        cs = np.array(cs)
        key = f'{int(z["edges"][lo])}-{int(z["edges"][hi])}'
        out[key] = [dict(sep=s_, n=int(np.isfinite(cs[(d >= s_[0]) & (d < s_[1])]).sum()),
                         mean=float(np.nanmean(cs[(d >= s_[0]) & (d < s_[1])]))) for s_ in SEPS]
        print(det, 'halo', key, [(o['sep'], o['n'], round(o['mean'], 3)) for o in out[key]], flush=True)
    return out, offs


def map_test(z, det, patdirs, offs):
    Fs, tsid, _ = amplitudes(z, det)
    N1 = 2*RC+1
    yy, xx = np.mgrid[:N1, :N1]; rr = np.hypot(xx-RC, yy-RC); ann = (rr >= 30) & (rr < RC)
    stars = []
    for pd in patdirs:
        for fn in sorted(glob.glob(os.path.join(pd, f'{det}_s*.npz'))):
            sid = int(os.path.basename(fn).split('_s')[1].split('.')[0]); F = Fs.get(sid, np.nan)
            if not np.isfinite(F) or F <= 0:
                continue
            p = np.load(fn); ne = sum(1 for k in p.files if k.endswith('_a'))
            ex = z['expnum'][p['rows']]; ob = z['obs'][p['rows']]
            maps = {}
            for e in range(ne):
                qx = p[f'e{e}_qx']; qy = p[f'e{e}_qy']; r = p[f'e{e}_r']; w = p[f'e{e}_w']
                m = np.hypot(qx, qy) <= RC
                idx = (np.round(qy[m]).astype(int)+RC)*N1+np.round(qx[m]).astype(int)+RC
                c = np.bincount(idx, minlength=N1*N1); x = np.bincount(idx, weights=(r*np.sqrt(w))[m], minlength=N1*N1)
                mp = np.where(c > 0, x/np.maximum(c, 1), np.nan).reshape(N1, N1)
                f = np.nan_to_num(mp)
                sm = ndimage.gaussian_filter(f, 5)/np.maximum(ndimage.gaussian_filter(np.isfinite(mp)*1., 5), 1e-3)
                maps[int(ex[e])] = (mp, mp-sm)
            if int(ex[0]) not in offs:
                continue
            p1 = np.mean([np.array(p['pos'][e])-offs[int(ex[e])] for e in range(ne)], 0)
            stars.append(dict(F=F, maps=maps, p1=p1, obs=int(ob[0])))
    P = np.array([s['p1'] for s in stars])
    pairs = cKDTree(P).query_pairs(700, output_type='ndarray')
    res = []
    for a, b in pairs:
        A, B = stars[a], stars[b]
        if A['obs'] == B['obs']:
            continue
        d = float(np.hypot(*(A['p1']-B['p1'])))
        for e in set(A['maps']) & set(B['maps']):
            for hp in (0, 1):
                u = A['maps'][e][hp]; v = B['maps'][e][hp]
                m = ann & np.isfinite(u) & np.isfinite(v)
                if m.sum() >= 2000:
                    res.append((d, hp, corr(u[m], v[m]), min(A['F'], B['F'])))
    res = np.array(res)
    out = {}
    for hp, lab in [(0, 'full'), (1, 'highpass5')]:
        for fl in (0.0, 0.1):
            key = f'{lab}_Fmin{fl}'
            out[key] = []
            for s_ in SEPS:
                k = (res[:, 1] == hp) & (res[:, 0] >= s_[0]) & (res[:, 0] < s_[1]) & (res[:, 3] > fl)
                out[key].append(dict(sep=s_, n=int(k.sum()), mean=float(res[k, 2].mean()) if k.any() else None))
            print(det, 'maps', key, [(o['sep'], o['n'], None if o['mean'] is None else round(o['mean'], 3)) for o in out[key]], flush=True)
    return out, len(stars)


if __name__ == '__main__':
    measfn, outfn = sys.argv[1:3]
    patdirs = sys.argv[3:]
    z = np.load(measfn)
    result = {}
    for det in ['NRCBLONG', 'NRCALONG']:
        h, offs = halo_test(z, det)
        mp, ns = map_test(z, det, patdirs, offs)
        result[det] = dict(halo=h, maps=mp, nstars_maps=ns, dither_offsets={k: v.tolist() for k, v in offs.items()})
    json.dump(result, open(outfn, 'w'), indent=1)
