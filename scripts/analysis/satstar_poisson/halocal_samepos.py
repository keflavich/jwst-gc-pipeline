"""Do two DIFFERENT stars that land on the SAME detector pixels show the SAME residual
structure?  And does the residual follow the star's brightness instead?  (README §8)

Test 1 -- same position, different star.
  Each star's in-sample residuals (halocal_patterns.py) are deviations from that star's
  own mean pattern Q, so a single exposure's residual depends on which other positions
  entered the mean.  That dependence cancels in an exposure DIFFERENCE: if star A's
  exposures i, k sit at the detector positions of star B's exposures j, l (within `tol`),
  then

      D_A = r_A,k/a_A,k - r_A,i/a_A,i        D_B = r_B,l/a_B,l - r_B,j/a_B,j

  are both "PSF(x_2) - PSF(x_1)" in fractional units if the PSF is a function of detector
  position alone, and D_A = D_B up to noise.  We measure, per annulus, the noise-weighted
  correlation of D_A and D_B, its noise ceiling sqrt(f_A f_B) (f = signal fraction of
  the map's power), and the fraction of D_A's excess power removed by subtracting D_B.
  Separation = the larger of the two coincidence distances; a large-separation control is
  in the same table.  Spike pixels (model > 3x the azimuthal median) and the halo between
  them are reported separately.

Test 2 -- brightness / saturation depth.
  The saturated core of a 4-group BRIGHT2 ramp is ringed by pixels that saturated in
  group 2, 3 or 4, whose slopes come from 1, 2 or 3 groups.  With rho = I/I_edge (model
  intensity relative to the group-1 saturation edge) those bands are rho in
  [1/2,1), [1/3,1/2), [1/4,1/3); rho < 1/4 is a full ramp.  An optical cause of the
  residual depends on radius r, not on rho; a ramp/nonlinearity cause depends on rho at
  fixed r.  Stars of different brightness put the same r at different rho, so the
  fractional residual power  P = (sum r^2 - sum 1/w)/sum I^2  on an (r, rho) grid
  separates the two.

    python halocal_samepos.py <meas.npz> <out.npz> <patdir> [<patdir> ...]
"""
import os, sys, glob, json, warnings
import numpy as np
from scipy.spatial import cKDTree
warnings.simplefilter('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from halocal_psfcal import amplitudes

RC = 124
ANN = [(15, 30), (30, 50), (50, 80), (80, 124)]
SEPB = [(0, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 300)]
TOL = 150.
QCUT = float(os.environ.get('QCUT', 30))     # drop stars whose median in-sample chi2 at r=50-124 exceeds this (failed fits)
RBINS = [15, 20, 25, 30, 40, 50, 65, 80, 100, 124]
RHOB = [0, 1/16, 1/8, 1/6, 1/5, 1/4, 1/3, 1/2, 1.0, 10]
# RHO_MODE=F: the second axis of test 2 is the star's halo amplitude F (relative to the
# target) instead of rho, which needs no saturation-edge estimate
RHO_MODE = os.environ.get('RHO_MODE', 'rho')
if RHO_MODE == 'F':
    RHOB = [0.005, 0.02, 0.04, 0.07, 0.12, 0.2, 0.35, 0.6, 1.5]
N1 = 2*RC+1
YY, XX = np.mgrid[:N1, :N1] - RC
RR = np.hypot(XX, YY)


SPK = [0, 1.5, 3, 1e9]          # model / azimuthal median: halo, intermediate, spike


def rho_accumulate(acc, P):
    nc, nr, nh = acc.shape[:3]
    ir = np.digitize(P[:, 0], RBINS)-1; ih = np.digitize(P[:, 1], RHOB)-1
    ic = np.digitize(P[:, 5], SPK)-1
    ok = (ir >= 0) & (ir < nr) & (ih >= 0) & (ih < nh) & (ic >= 0) & (ic < nc)
    cell = (ic[ok]*nr+ir[ok])*nh+ih[ok]
    for q, col in enumerate((2, 3, 4)):
        acc[..., q] += np.bincount(cell, weights=P[ok, col], minlength=nc*nr*nh).reshape(nc, nr, nh)
    acc[..., 3] += np.bincount(cell, minlength=nc*nr*nh).reshape(nc, nr, nh)


def load_star(fn, z, F, acc):
    p = np.load(fn)
    N = int(p['N']); Q = p['Q'].astype(float)
    ne = sum(1 for k in p.files if k.endswith('_a'))
    rows = p['rows']
    qy_, qx_ = np.mgrid[:N, :N]-N//2; rq = np.round(np.hypot(qx_, qy_)).astype(int)
    qmed = np.array([np.median(Q[rq == k]) if (rq == k).any() else np.nan for k in range(rq.max()+1)])
    mp = np.full((ne, N1, N1), np.nan, np.float32); vr = np.full((ne, N1, N1), np.nan, np.float32)
    Im = np.full((ne, N1, N1), np.nan, np.float32)
    chi2 = []
    for e in range(ne):
        qx = p[f'e{e}_qx'].astype(float); qy = p[f'e{e}_qy'].astype(float)
        r = p[f'e{e}_r'].astype(float); w = p[f'e{e}_w'].astype(float); a = float(p[f'e{e}_a'])
        rq_ = np.hypot(qx, qy); mq = (rq_ >= 50) & (rq_ < 124) & (w > 0)
        chi2.append(float(np.mean(r[mq]**2*w[mq])) if mq.any() else np.nan)
    if not np.nanmedian(chi2) <= QCUT:
        return None
    for e in range(ne):
        qx = p[f'e{e}_qx'].astype(float); qy = p[f'e{e}_qy'].astype(float)
        r = p[f'e{e}_r'].astype(float); w = p[f'e{e}_w'].astype(float); a = float(p[f'e{e}_a'])
        ix = np.round(qx).astype(int); iy = np.round(qy).astype(int)
        m = (np.abs(ix) <= RC) & (np.abs(iy) <= RC) & (w > 0)
        idx = (iy[m]+RC)*N1+ix[m]+RC
        sw = np.bincount(idx, weights=w[m]*a*a, minlength=N1*N1)
        sx = np.bincount(idx, weights=(w*r*a)[m], minlength=N1*N1)
        ok = sw > 0
        mp[e].flat[ok] = sx[ok]/sw[ok]; vr[e].flat[ok] = 1/sw[ok]
        # test 2: per-pixel model intensity, residual, variance, rho
        I = a*Q[np.clip(iy+N//2, 0, N-1), np.clip(ix+N//2, 0, N-1)]
        Iv = I[w > 0]
        Iedge = np.percentile(Iv, 99.5)
        rad = np.hypot(qx, qy)
        g = (w > 0) & (rad >= RBINS[0]) & (rad < RBINS[-1]) & (I > 0)
        spk = I[g]/np.maximum(a*qmed[np.clip(np.round(rad[g]).astype(int), 0, len(qmed)-1)], 1e-30)
        rho_accumulate(acc, np.stack([rad[g], np.full(g.sum(), F) if RHO_MODE == 'F' else I[g]/Iedge, r[g]**2, 1/w[g], I[g]**2, spk], 1))
        Im[e].flat[idx] = I[m]
    ex = z['expnum'][rows]; ob = z['obs'][rows]
    return dict(fn=os.path.basename(fn), F=F, pos=p['pos'].astype(float), ex=ex, obs=int(ob[0]),
                m=mp, v=vr, I=np.nanmean(Im, 0), a=np.array([float(p[f'e{e}_a']) for e in range(ne)]),
                nsat=p['nsat'] if 'nsat' in p.files else None)


def spike_mask(stars):
    """Spikes: model intensity > 3x its azimuthal median, from the brightest star."""
    s = max(stars, key=lambda s: s['F'])
    I = s['I']
    rb = np.round(RR).astype(int)
    med = np.zeros(RC+2)
    for k in range(RC+2):
        v = I[(rb == k) & np.isfinite(I)]
        med[k] = np.median(v) if len(v) else np.nan
    return np.isfinite(I) & (I > 3*med[np.clip(rb, 0, RC+1)]) & (RR >= 10)


def quads(stars):
    """(A, i, k, B, j, l, sep): A_i~B_j and A_k~B_l, i != k, different visits."""
    allpos = np.concatenate([s['pos'] for s in stars]); owner = np.concatenate([[n]*len(s['pos']) for n, s in enumerate(stars)])
    eidx = np.concatenate([np.arange(len(s['pos'])) for s in stars])
    t = cKDTree(allpos)
    pairs = t.query_pairs(TOL, output_type='ndarray')
    pairs = pairs[owner[pairs[:, 0]] != owner[pairs[:, 1]]]
    co = {}
    for u, v in pairs:
        A, B = owner[u], owner[v]
        if A > B:
            u, v, A, B = v, u, B, A
        co.setdefault((A, B), []).append((eidx[u], eidx[v], float(np.hypot(*(allpos[u]-allpos[v])))))
    out = []
    for (A, B), lst in co.items():
        for x in range(len(lst)):
            for y in range(x+1, len(lst)):
                i, j, d1 = lst[x]; k, l, d2 = lst[y]
                if i == k or j == l:
                    continue
                same = stars[A]['obs'] == stars[B]['obs']
                if same and (i != j or k != l):
                    continue
                out.append((int(A), int(i), int(k), int(B), int(j), int(l), max(d1, d2), same))
    return out


def wstats(DA, VA, DB, VB, m):
    """noise-weighted corr, ceiling, excess-power removed by subtracting D_B from D_A."""
    k = m & np.isfinite(DA) & np.isfinite(DB)
    if k.sum() < 200:
        return None
    da, db, va, vb = DA[k], DB[k], VA[k], VB[k]
    wt = 1/(va+vb)
    sab = np.sum(wt*da*db); saa = np.sum(wt*da*da); sbb = np.sum(wt*db*db)
    c = sab/np.sqrt(saa*sbb)
    fa = 1-np.sum(wt*va)/saa; fb = 1-np.sum(wt*vb)/sbb
    ceil = np.sqrt(max(fa, 0)*max(fb, 0))
    exA = np.sum(da*da/va)-k.sum()                       # excess chi2 of D_A alone
    exD = np.sum((da-db)**2/(va+vb))-k.sum()             # of D_A - D_B (noise of both removed)
    return dict(n=int(k.sum()), corr=float(c), ceil=float(ceil), fa=float(fa), fb=float(fb),
                chi2A=float(np.sum(da*da/va)/k.sum()), chi2D=float(np.sum((da-db)**2/(va+vb))/k.sum()),
                removed=float(1-exD/exA) if exA > 0 else np.nan)


def test1(stars, spk):
    Q = quads(stars)
    print(len(Q), 'quadruples within', TOL, 'px', flush=True)
    rows = []
    for A, i, k, B, j, l, sep, same in Q:
        sA, sB = stars[A], stars[B]
        DA = sA['m'][k]-sA['m'][i]; VA = sA['v'][k]+sA['v'][i]
        DB = sB['m'][l]-sB['m'][j]; VB = sB['v'][l]+sB['v'][j]
        ph = np.abs(((sA['pos'][[i, k]]-sB['pos'][[j, l]])+0.5) % 1-0.5).max()
        for ia, (lo, hi) in enumerate(ANN):
            ann = (RR >= lo) & (RR < hi)
            for part, msk in (('halo', ann & ~spk), ('spike', ann & spk)):
                st = wstats(DA, VA, DB, VB, msk)
                if st is not None:
                    rows.append(dict(same=bool(same), anchor=bool(0 in (i, k, j, l)), A=A, B=B, i=int(i), k=int(k), j=int(j), l=int(l), sep=sep, phase=float(ph),
                                     Fmin=float(min(sA['F'], sB['F'])), ann=ia, part=part, **st))
    return rows


def summarise1(rows):
    out = {}
    for same, noanc in ((False, False), (False, True), (True, False)):
      for part in ('halo', 'spike'):
        for ia, (lo, hi) in enumerate(ANN):
            for s0, s1 in SEPB:
                R = [r for r in rows if r['same'] == same and not (noanc and r['anchor']) and r['part'] == part and r['ann'] == ia and s0 <= r['sep'] < s1 and r['ceil'] > 0.2]
                if not R:
                    continue
                c = np.array([r['corr'] for r in R]); ce = np.array([r['ceil'] for r in R]); rm = np.array([r['removed'] for r in R])
                key = f'{"samevisit" if same else "diffvisit"}{"_noanchor" if noanc else ""}_{part}_{lo}-{hi}_{s0}-{s1}'
                out[key] = dict(n=len(R), corr=float(np.median(c)), ceil=float(np.median(ce)),
                                ratio=float(np.median(c/ce)), removed=float(np.median(rm)))
                print(f'{"SAME-visit" if same else "diff-visit"}{" no-anchor" if noanc else "          "} {part:5s} r={lo:3d}-{hi:3d} sep {s0:3d}-{s1:3d}: n={len(R):5d} corr={np.median(c):+.3f} '
                      f'ceiling={np.median(ce):.3f} corr/ceil={np.median(c/ce):+.3f} excess removed by D_B={np.median(rm):+.3f}', flush=True)
    return out


def test2(acc):
    nc, nr, nh = acc.shape[:3]
    Pw = np.where(acc[..., 3] > 500, (acc[..., 0]-acc[..., 1])/np.maximum(acc[..., 2], 1e-30), np.nan)
    chi = np.where(acc[..., 3] > 500, acc[..., 0]/np.maximum(acc[..., 1], 1e-30), np.nan)
    for c in range(nc):
        print(f'fractional residual rms sqrt(P) [%], pixels with model/azimuthal median in {SPK[c]}-{SPK[c+1]}; (r rows) x (rho cols)', flush=True)
        print('   r \\ rho ' + ' '.join(f'{RHOB[h]:.3f}-' for h in range(nh)))
        for i in range(nr):
            print(f'{RBINS[i]:4d}-{RBINS[i+1]:4d} ' + ' '.join('   -  ' if not np.isfinite(Pw[c, i, h]) else f'{100*np.sqrt(max(Pw[c, i, h], 0)):6.2f}' for h in range(nh)), flush=True)
    return acc, Pw, chi


def main():
    measfn, outfn = sys.argv[1:3]
    patdirs = sys.argv[3:]
    z = np.load(measfn)
    res = {}
    for det in ['NRCBLONG', 'NRCALONG']:
        Fs, tsid, _ = amplitudes(z, det)
        stars = []
        seen = set()
        acc = np.zeros((len(SPK)-1, len(RBINS)-1, len(RHOB)-1, 4))     # sum r^2, sum var, sum I^2, npix
        for pd in patdirs:
            for fn in sorted(glob.glob(os.path.join(pd, f'{det}_s*.npz'))):
                sid = int(os.path.basename(fn).split('_s')[1].split('.')[0])
                F = Fs.get(sid, np.nan)
                if sid in seen or not np.isfinite(F) or F <= 0:
                    continue
                seen.add(sid)
                st = load_star(fn, z, F, acc)
                if st is None:
                    print('  failed fit, skipped:', os.path.basename(fn), flush=True)
                    continue
                st['sid'] = sid; st['target'] = sid == tsid
                stars.append(st)
        print(det, len(stars), 'stars', flush=True)
        if not stars:
            continue
        spk = spike_mask(stars)
        rows = test1(stars, spk) if not os.environ.get('SKIP1') else []
        s1 = summarise1(rows)
        acc, Pw, chi = test2(acc)
        res[det] = dict(test1=s1, nstars=len(stars))
        np.savez_compressed(outfn.replace('.npz', f'_{det}.npz'), rows=json.dumps(rows), acc=acc, P=Pw, chi=chi, spk=spk,
                            SPK=SPK, sids=[s['sid'] for s in stars], F=[s['F'] for s in stars], RBINS=RBINS, RHOB=RHOB)
        del stars
    json.dump(res, open(outfn.replace('.npz', '.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
