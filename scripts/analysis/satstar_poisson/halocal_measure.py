"""Star-centred halo measurements for every saturated star in every 10678 F480M exposure
(input: halocal_extract.py output).  README §8.

For each extracted cutout:

1. **Link** the star across exposures: friends-of-friends on the GWCS sky positions of
   the SATURATED-region centroids (0.25" linking length).  This is a same-star identity
   match of isolated saturated cores (typical separation >> 1"), not an astrometric
   offset measurement.
2. **Centre** by point symmetry: the (x, y) that minimises the robust difference between
   the image and its point reflection, on pixels along the six diffraction spikes at
   r = 15-80 px.  The spikes are pure pupil diffraction, sharp and centrosymmetric.
3. **Geometry** in the star-centred ideal frame: per-pixel V2V3 from the exposure's
   GWCS, evaluated on a 17x17 grid and bicubic-interpolated (<0.1 mpix), then mapped to
   ideal pixels with one fixed Jacobian per detector.  Radii and angles are therefore
   sky angles, and the 1-2% plate-scale change across the detector does not leak into
   the profiles (a halo falling as r^-3.3 would otherwise change by ~5%).
4. **Measure** in annuli (EDGES) x 12 azimuth sectors, off the spikes (perpendicular
   distance > 4 px from all 8 spike lines) and away from other saturated stars:
     H   : median of the good pixels  [MJy/sr]
     He  : its Poisson+read-noise error, 1.2533*sqrt(<var>/n)
     S   : spike excess per annulus (median on-spike minus the local off-spike median)
   The sampled sky region is the same in every dither of a star (same roll), so the scene
   contributes the same H in every exposure; differences between exposures are halo.

    python halocal_measure.py <extdir> <out.npz> [nproc]
"""
import os, sys, glob, json, warnings
import numpy as np
import concurrent.futures as cf
from scipy import ndimage, optimize
from scipy.interpolate import RectBivariateSpline
from scipy.spatial import cKDTree
import asdf
warnings.simplefilter('ignore')

EDGES = np.array([15, 20, 25, 30, 40, 50, 65, 80, 100, 125, 160, 200, 250])
NSEC = 12
LINK_ARCSEC = 0.25
MAXDEV_MAS = 40.
# ideal-frame Jacobian d(V2,V3)/d(x,y) [arcsec/px]: NRCB5 value at the target; the same
# nominal scale is used for NRCA5 (only relative, per-detector, comparisons are made)
J0 = np.array([[-6.28220973e-02, 1.05499919e-05], [9.49585896e-05, 6.31152544e-02]])
J0inv = np.linalg.inv(J0)
# spike position angles in the ideal frame [deg]; measured by spike_angles() below on
# the brightest star of each detector and stored here
SPIKES = {'NRCBLONG': None, 'NRCALONG': None}


def v2v3_grid(t, X0, Y0, n):
    g = np.linspace(0, n-1, 17)
    gx, gy = np.meshgrid(g, g)
    v2, v3 = t(gx.ravel()+X0, gy.ravel()+Y0)
    s2 = RectBivariateSpline(g, g, v2.reshape(17, 17), kx=3, ky=3)
    s3 = RectBivariateSpline(g, g, v3.reshape(17, 17), kx=3, ky=3)
    a = np.arange(n, dtype=float)
    return s2(a, a), s3(a, a), s2, s3


def symmetric_centre(d, ok, x0, y0, spikes_det, rmin=15, rmax=80):
    """Point-symmetry centre on the spike pixels (detector frame of the cutout)."""
    n = d.shape[0]
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(xx-x0, yy-y0); th = np.degrees(np.arctan2(yy-y0, xx-x0))
    dth = np.min(np.abs(((th[..., None]-spikes_det[None, None, :])+180) % 360-180), -1)
    sel = ok & (r > rmin) & (r < rmax) & (r*np.sin(np.radians(np.minimum(dth, 90))) < 2.0)
    py, px = np.nonzero(sel)
    df = np.where(ok, d, 0.0); wf = ok.astype(float)
    val = d[py, px]
    def cost(c):
        rx = 2*c[0]-px; ry = 2*c[1]-py
        v = ndimage.map_coordinates(df, [ry, rx], order=1, mode='constant')
        w = ndimage.map_coordinates(wf, [ry, rx], order=1, mode='constant')
        m = w > 0.999
        if m.sum() < 50:
            return 1e30
        res = (val[m]-v[m])/np.maximum(np.abs(val[m])+np.abs(v[m]), 1e-3)
        return np.mean(np.abs(res))
    best = None
    for dx in np.arange(-1.5, 1.51, 0.5):
        for dy in np.arange(-1.5, 1.51, 0.5):
            c = cost((x0+dx, y0+dy))
            if best is None or c < best[0]:
                best = (c, x0+dx, y0+dy)
    r_ = optimize.minimize(cost, [best[1], best[2]], method='Nelder-Mead',
                           options=dict(xatol=0.01, fatol=1e-6, initial_simplex=[[best[1], best[2]], [best[1]+0.3, best[2]], [best[1], best[2]+0.3]]))
    return r_.x[0], r_.x[1], float(r_.fun), int(sel.sum())


NOMINAL_SPIKES = np.array([0.75, 29.25, 89.75, 149.25, 179.25, 209.25, 269.75, 329.75])
ABINS = np.arange(0, 360.01, 0.25)


def angular_profile(d, ok, xc, yc, rmin, rmax):
    n = d.shape[0]
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(xx-xc, yy-yc); th = np.degrees(np.arctan2(yy-yc, xx-xc)) % 360
    m = (r > rmin) & (r < rmax) & ok
    idx = np.digitize(th[m], ABINS)-1
    v = d[m]
    return np.array([np.median(v[idx == i]) if np.any(idx == i) else np.nan for i in range(len(ABINS)-1)])


def refine_peak(prof, a0, win=4.0):
    c = 0.5*(ABINS[1:]+ABINS[:-1])
    dd = ((c-a0)+180) % 360-180
    m = np.abs(dd) < win
    p = np.where(np.isfinite(prof), prof, -np.inf)
    i = np.argmax(np.where(m, p, -np.inf))
    return c[i]


def spike_angles(d, ok, xc, yc, rmin=70, rmax=130):
    n = d.shape[0]
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(xx-xc, yy-yc); th = np.degrees(np.arctan2(yy-yc, xx-xc)) % 360
    m = (r > rmin) & (r < rmax) & ok
    bins = np.arange(0, 360.5, 0.5); idx = np.digitize(th[m], bins)
    ang = np.array([np.nanmedian(d[m][idx == i]) if np.any(idx == i) else np.nan for i in range(1, len(bins))])
    ang = np.nan_to_num(ang, nan=np.nanmin(ang))
    sm = ndimage.maximum_filter1d(ang, 21, mode='wrap')
    pk = np.nonzero((ang == sm) & (ang > np.percentile(ang, 90)))[0]
    return bins[pk]+0.25


def _load(fn, sidx):
    z = np.load(fn)
    d = z[f's{sidx}_sci'].astype(float); var = z[f's{sidx}_var'].astype(float); dq = z[f's{sidx}_dq']
    X0, Y0 = (int(v) for v in z[f's{sidx}_org'])
    ok = np.isfinite(d) & np.isfinite(var) & (var > 0) & ((dq & 1) == 0) & ((dq & 2) == 0)
    return d, var, dq, X0, Y0, ok


def centre_one(args):
    """Pass 1: point-symmetry centre -> sky position (GWCS)."""
    fn, wcsfn, sidx, x, y, nsat, spikes = args
    d, var, dq, X0, Y0, ok = _load(fn, sidx)
    n = d.shape[0]
    rs = 1.3*np.sqrt(nsat/np.pi)          # radius of the saturated core
    xc, yc, sym, nsym = symmetric_centre(d, ok, x-X0, y-Y0, spikes, rmin=max(12, rs+3), rmax=min(n//2-6, rs+90))
    ra_c, dec_c = asdf.open(wcsfn)['wcs'](xc+X0, yc+Y0)
    return dict(xs=xc+X0, ys=yc+Y0, ra_s=float(ra_c), dec_s=float(dec_c), sym=sym, nsym=nsym)


def profile_one(args):
    """Pass 2: halo sector medians about the star's adopted sky position."""
    fn, wcsfn, sidx, ra_c, dec_c, others, spikes = args
    d, var, dq, X0, Y0, ok = _load(fn, sidx)
    n = d.shape[0]
    w = asdf.open(wcsfn)['wcs']
    xg, yg = w.invert(ra_c, dec_c)
    xc, yc = float(xg)-X0, float(yg)-Y0
    t = w.get_transform('detector', 'v2v3')
    v2, v3, s2, s3 = v2v3_grid(t, X0, Y0, n)
    vs = np.array(t(xc+X0, yc+Y0))
    qx = J0inv[0, 0]*(v2-vs[0])+J0inv[0, 1]*(v3-vs[1])
    qy = J0inv[1, 0]*(v2-vs[0])+J0inv[1, 1]*(v3-vs[1])
    r = np.hypot(qx, qy); th = np.degrees(np.arctan2(qy, qx)) % 360
    # the spike angles are measured in the detector frame; the ideal frame is the same
    # up to the (tiny) J0 skew, so they are reused
    dth = np.min(np.abs(((th[..., None]-spikes[None, None, :])+180) % 360-180), -1)
    dperp = r*np.sin(np.radians(np.minimum(dth, 90)))
    # mask the other saturated stars in the cutout (radius grows with their SAT size)
    yy, xx = np.mgrid[:n, :n]
    for ox, oy, on in others:
        rr = 6+2.5*np.sqrt(on)
        if abs(ox-X0-n/2) < n/2+rr and abs(oy-Y0-n/2) < n/2+rr:
            ok &= np.hypot(xx-(ox-X0), yy-(oy-Y0)) > rr
    off = ok & (dperp > 4)
    on_ = ok & (dperp < 1.2)
    rmax = min(n//2-4, EDGES[-1])
    nb = len(EDGES)-1
    H = np.full((nb, NSEC), np.nan); He = np.full((nb, NSEC), np.nan); Np = np.zeros((nb, NSEC), int)
    S = np.full(nb, np.nan); Se = np.full(nb, np.nan)
    rb = np.digitize(r, EDGES)-1
    sb = np.clip((th/(360/NSEC)).astype(int), 0, NSEC-1)
    for i in range(nb):
        if EDGES[i+1] > rmax:
            break
        ann = rb == i
        for s_ in range(NSEC):
            m = off & ann & (sb == s_)
            k = int(m.sum())
            if k >= 8:
                H[i, s_] = np.median(d[m]); He[i, s_] = 1.2533*np.sqrt(np.mean(var[m])/k); Np[i, s_] = k
        mo = on_ & ann; mf = off & ann
        if mo.sum() >= 5 and mf.sum() >= 20:
            S[i] = np.median(d[mo])-np.median(d[mf])
            Se[i] = 1.2533*np.sqrt(np.mean(var[mo])/mo.sum())
    return dict(xc=xc+X0, yc=yc+Y0, v2=vs[0], v3=vs[1], H=H, He=He, Np=Np, S=S, Se=Se)


def load_catalog(extdir):
    rows = []; regions = {}
    for fn in sorted(glob.glob(os.path.join(extdir, 'jw*_cal.npz'))):
        z = np.load(fn)
        meta = json.loads(str(z['meta']))
        regions[fn] = np.stack([z['x'], z['y'], z['n']], 1)
        cut = sorted(int(k[1:].split('_')[0]) for k in z.files if k.endswith('_sci'))
        for i in cut:
            rows.append(dict(fn=fn, sidx=i, x=float(z['x'][i]), y=float(z['y'][i]), n=int(z['n'][i]),
                             ra=float(z['ra'][i]), dec=float(z['dec'][i]), **meta))
    return rows, regions


def link(rows):
    ra = np.array([r['ra'] for r in rows]); dec = np.array([r['dec'] for r in rows])
    d0 = np.median(dec); r0 = np.median(ra)
    X = np.stack([(ra-r0)*np.cos(np.radians(d0))*3600, (dec-d0)*3600], 1)
    tree = cKDTree(X)
    pairs = tree.query_pairs(LINK_ARCSEC, output_type='ndarray')
    parent = np.arange(len(rows))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for a, b in pairs:
        ra_, rb_ = find(a), find(b)
        if ra_ != rb_:
            parent[ra_] = rb_
    root = np.array([find(i) for i in range(len(rows))])
    _, sid = np.unique(root, return_inverse=True)
    return sid


if __name__ == '__main__':
    extdir, out = sys.argv[1], sys.argv[2]
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    rows, regions = load_catalog(extdir)
    sid = link(rows)
    print(len(rows), 'cutouts,', sid.max()+1, 'stars', flush=True)
    # spike angles per detector: angular profile at r = rs+10 .. rs+60 stacked (median of
    # the normalised profiles) over the 25 brightest cutouts; the 8 peaks (6 hexagon + 2
    # strut) are searched within 4 deg of the nominal NRCB5 angles
    for det in SPIKES:
        cand = sorted([r for r in rows if r['detector'] == det], key=lambda r: -r['n'])[:25]
        profs = []
        for b in cand:
            z = np.load(b['fn']); d = z[f"s{b['sidx']}_sci"].astype(float); dq = z[f"s{b['sidx']}_dq"]
            X0, Y0 = z[f"s{b['sidx']}_org"]
            ok = np.isfinite(d) & ((dq & 1) == 0) & ((dq & 2) == 0)
            rs = 1.3*np.sqrt(b['n']/np.pi)
            p = angular_profile(d, ok, b['x']-X0, b['y']-Y0, rs+10, min(rs+60, d.shape[0]//2-5))
            profs.append(p/np.nanmedian(p))
        prof = np.nanmedian(profs, 0)
        SPIKES[det] = np.array([refine_peak(prof, a0) for a0 in NOMINAL_SPIKES])
        print(det, 'spikes', np.round(SPIKES[det], 2), flush=True)
    jobs = [(r['fn'], r['fn'].replace('.npz', '_wcs.asdf'), r['sidx'], r['x'], r['y'], r['n'], SPIKES[r['detector']]) for r in rows]
    with cf.ProcessPoolExecutor(nproc) as ex:
        cen = list(ex.map(centre_one, jobs, chunksize=8))
    print('pass 1 (centres) done', flush=True)
    # adopted sky position per (star, obs): median of the symmetry centres; exposures whose
    # own centre is > MAXDEV_MAS away, or whose SAT core size is off by > 40%, are flagged
    # (mis-links to a neighbour, or a failed centre) and not measured
    obs = np.array([int(r['obs']) for r in rows]); nn = np.array([r['n'] for r in rows])
    ras = np.array([c['ra_s'] for c in cen]); des = np.array([c['dec_s'] for c in cen])
    ra_ad = np.full(len(rows), np.nan); de_ad = np.full(len(rows), np.nan); good = np.zeros(len(rows), bool)
    key = sid*1000+obs
    for k in np.unique(key):
        ii = np.nonzero(key == k)[0]
        if len(ii) < 2:
            continue
        rm, dm_ = np.median(ras[ii]), np.median(des[ii]); nm = np.median(nn[ii])
        dev = np.hypot((ras[ii]-rm)*np.cos(np.radians(dm_)), des[ii]-dm_)*3.6e6
        jj = ii[(dev < MAXDEV_MAS) & (np.abs(nn[ii]/nm-1) < 0.4)]
        if len(jj) < 2:
            continue
        ra_ad[jj] = np.median(ras[jj]); de_ad[jj] = np.median(des[jj]); good[jj] = True
    print(f'{good.sum()} of {len(rows)} cutouts kept after centre/link consistency', flush=True)
    jobs = []
    for i, r in enumerate(rows):
        if not good[i]:
            continue
        oth = [(ox, oy, on) for ox, oy, on in regions[r['fn']] if np.hypot(ox-r['x'], oy-r['y']) > 1]
        jobs.append((r['fn'], r['fn'].replace('.npz', '_wcs.asdf'), r['sidx'], ra_ad[i], de_ad[i], oth, SPIKES[r['detector']]))
    with cf.ProcessPoolExecutor(nproc) as ex:
        res = list(ex.map(profile_one, jobs, chunksize=8))
    gi = np.nonzero(good)[0]
    keys = ['H', 'He', 'Np', 'S', 'Se']
    sel = lambda L: [L[i] for i in gi]
    np.savez_compressed(out, sid=sid[gi], **{k: np.array([r[k] for r in res]) for k in keys},
                        **{k: np.array([r[k] for r in res]) for k in ['xc', 'yc', 'v2', 'v3']},
                        **{k: np.array([cen[i][k] for i in gi]) for k in ['xs', 'ys', 'ra_s', 'dec_s', 'sym', 'nsym']},
                        ra=ra_ad[gi], dec=de_ad[gi],
                        n=nn[gi], det=np.array([r['detector'] for r in sel(rows)]),
                        obs=obs[gi], expnum=np.array([int(r['exposure']) for r in sel(rows)]),
                        expstart=np.array([r['expstart'] for r in sel(rows)]), root=np.array([r['root'] for r in sel(rows)]),
                        sidx=np.array([r['sidx'] for r in sel(rows)]), edges=EDGES,
                        spikes=json.dumps({k: v.tolist() for k, v in SPIKES.items()}))
    print('wrote', out, flush=True)
