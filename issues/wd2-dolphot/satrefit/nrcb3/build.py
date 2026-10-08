"""Build per-row tables (S rows and daophot control rows) with dm, star id, detector, exposure, environment.
Usage: nice -19 python -u build.py BAND  -> /orange/.../nrcb3/rows_<BAND>.npz
Read-only imports of analyze.py / ovlscale.py.  Reproduces the ovlscale_det.py matching exactly."""
import sys
import glob
import re
import numpy as np
from astropy.table import Table
from astropy.io import fits
from scipy.spatial import cKDTree
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/ovlscale')
band = sys.argv[1]
sys.argv = sys.argv[:1]
import analyze as an
import ovlscale as o

OUT = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/nrcb3'
rng = np.random.default_rng(1)


def annulus_bkg(img, x, y, rin, rout):
    """sigma-clipped median in annulus (pixel radii) around (x,y); NaN when too few pixels."""
    ny, nx = img.shape
    x0, y0 = int(round(x)), int(round(y))
    r = int(np.ceil(rout))
    xa, xb = max(x0 - r, 0), min(x0 + r + 1, nx)
    ya, yb = max(y0 - r, 0), min(y0 + r + 1, ny)
    if xb <= xa or yb <= ya:
        return np.nan
    yy, xx = np.mgrid[ya:yb, xa:xb]
    rr = np.hypot(xx - x, yy - y)
    v = img[ya:yb, xa:xb][(rr >= rin) & (rr < rout)]
    v = v[np.isfinite(v)]
    if v.size < 200:
        return np.nan
    for _ in range(6):
        med = np.median(v)
        sd = 1.4826 * np.median(np.abs(v - med))
        keep = np.abs(v - med) < 2.5 * sd
        if keep.all() or keep.sum() < 100:
            break
        v = v[keep]
    return float(np.median(v))


def main():
    an.ZPWIN.update(an.zp_windows())
    A = an.Arm('main2')
    m = A.m
    bb = band[1:]
    mra, mdec = np.asarray(m['RA'], float), np.asarray(m['DEC'], float)
    S, D, pixsr = o.load_band(band)
    pixas = np.sqrt(pixsr) * 206264.806
    ra0, dec0 = np.median(D['ra']), np.median(D['dec'])
    sxy = o.sky_xy(S['ra'], S['dec'], ra0, dec0)
    lab = o.fof(sxy, o.RADIUS)
    nst = lab.max() + 1
    cen = np.array([np.median(sxy[lab == k], axis=0) for k in range(nst)])
    fmed = np.array([np.nanmedian(S['flux'][lab == k]) for k in range(nst)])
    ref = A.ref[bb]
    ok = A.matched & np.isfinite(ref) & np.isfinite(A.our[bb])
    mxy = o.sky_xy(mra, mdec, ra0, dec0)
    tree = cKDTree(mxy[ok])
    okidx = np.where(ok)[0]
    dist, j = tree.query(cen)
    good = dist < 0.1
    star_m = okidx[j]
    rep = A.rep[bb][star_m] & good
    mS_star = o.mag(fmed, pixsr)
    cb = float(np.nanmedian(A.our[bb][star_m][rep] - mS_star[rep]))
    zp = A.zp[bb]
    rgood = good[lab]
    rref = ref[star_m[lab]]
    dm = {k: o.mag(S[v], pixsr) + cb - rref - zp for k, v in (('final', 'flux'), ('precap', 'precap'), ('raw', 'raw'))}
    sel = rgood & np.isfinite(dm['final']) & np.isfinite(rref)
    # environment: catalogue of all dolphot stars with finite mag in this band
    fin = np.isfinite(ref)
    ctree = cKDTree(mxy[fin])
    cmag = ref[fin]
    cxy = mxy[fin]

    def env(mi, own):
        """mi: index into m of the star; own: its dolphot mag.  returns n1, n2, nearest-brighter distance"""
        n = len(mi)
        n1 = np.zeros(n, int)
        n2 = np.zeros(n, int)
        dbr = np.full(n, np.nan)
        xy = mxy[mi]
        l2 = ctree.query_ball_point(xy, 2.0)
        for k in range(n):
            ids = np.asarray(l2[k], int)
            if ids.size == 0:
                continue
            dd = np.hypot(*(cxy[ids] - xy[k]).T)
            mg = cmag[ids]
            notself = dd > 0.02
            fo = notself & (mg < own[k] + 3)
            n2[k] = fo.sum()
            n1[k] = (fo & (dd < 1.0)).sum()
            br = notself & (mg < own[k])
            if br.any():
                dbr[k] = dd[br].min()
        return n1, n2, dbr
    # frame table
    dfiles = sorted(glob.glob(f'{o.TREE}/{band}/*_resbgsub_m7_daophot_basic.fits'))
    finfo = []
    for df in dfiles:
        mm = re.search(r'_(nrc[ab](?:[1-4]|long))_visit\d+_vgroup(\d+)_exp(\d+)_', os.path.basename(df)) if False else re.search(r'_(nrc[ab](?:[1-4]|long))_visit\d+_vgroup(\d+)_exp(\d+)_', df.split('/')[-1])
        finfo.append((mm.group(1), int(mm.group(2)), int(mm.group(3))))
    # S rows
    idx = np.where(sel)[0]
    n1, n2, dbr = env(star_m[lab[idx]], rref[idx])
    res = dict(band=band, cb=cb, zp=zp, nst=nst)
    srow = {k: v[idx] for k, v in dm.items()}
    srow.update(lab=lab[idx], det=S['det'][idx], frame=S['frame'][idx], x=S['x'][idx], y=S['y'][idx], ra=S['ra'][idx], dec=S['dec'][idx],
                ref=rref[idx], area=S['area'][idx], n1=n1, n2=n2, dbr=dbr, cap_binds=(S['raw'][idx] < 0.999 * S['precap'][idx]),
                starm=star_m[lab[idx]])
    srow['exp'] = np.array([finfo[f][2] for f in srow['frame']])
    srow['vg'] = np.array([finfo[f][1] for f in srow['frame']])
    srow['edge'] = np.minimum.reduce([srow['x'], srow['y'], 2047 - srow['x'], 2047 - srow['y']])
    lb = np.full(len(idx), np.nan)
    ann = np.full(len(idx), np.nan)
    # D control rows
    lo_z, hi_z = an.ZPWIN.get(bb, (0, 19))
    dxy = o.sky_xy(D['ra'], D['dec'], ra0, dec0)
    dd_, dj = tree.query(dxy)
    sm = okidx[dj]
    dref = ref[sm]
    unsat_star = ~A.rep[bb][sm] & ~A.sat[bb][sm]
    dsel = ((dd_ < 0.1) & (D['flags'] == 0) & ~D['forced'] & ~D['sat5'] & np.isfinite(D['flux']) & (D['flux'] > 0) & unsat_star
            & (dref >= lo_z) & (dref < hi_z) & np.isin(D['det'], ['nrcb1', 'nrcb3']))
    didx = np.where(dsel)[0]
    if len(didx) > 30000:
        didx = np.sort(rng.choice(didx, 30000, replace=False))
    dmD = o.mag(D['flux'], pixsr) + cb - dref - zp
    d1, d2, ddbr = env(sm[didx], dref[didx])
    drow = dict(dm=dmD[didx], det=D['det'][didx], frame=D['frame'][didx], x=D['x'][didx], y=D['y'][didx], ref=dref[didx], n1=d1, n2=d2, dbr=ddbr,
                starm=sm[didx])
    drow['edge'] = np.minimum.reduce([drow['x'], drow['y'], 2047 - drow['x'], 2047 - drow['y']])
    dann = np.full(len(didx), np.nan)
    # per-frame stored local_bkg and annulus
    rin, rout = 1.5 / pixas, 2.5 / pixas
    for fi, (det, vg, ex) in enumerate(finfo):
        si = np.where(srow['frame'] == fi)[0]
        di = np.where(drow['frame'] == fi)[0]
        if len(si) == 0 and len(di) == 0:
            continue
        crf = glob.glob(f'{o.TREE}/{band}/pipeline/*_{ex:05d}_{det}_align_o005_crf.fits')
        scat = glob.glob(f'{o.TREE}/{band}/pipeline/*_{ex:05d}_{det}_align_o005_crf_resbgsub_m7_satstar_catalog.fits')
        if len(si):
            s = Table.read(scat[0])
            # S rows of this frame in original order; reconstruct by matching xcentroid
            fr_all = np.where(S['frame'] == fi)[0]
            assert len(fr_all) == len(s) and np.allclose(S['x'][fr_all], np.asarray(s['xcentroid'], float))
            pos = {g: k for k, g in enumerate(fr_all)}
            lb[si] = [float(s['local_bkg'][pos[idx[q]]]) for q in si]
        with fits.open(crf[0], memmap=False) as h:
            img = np.asarray(h['SCI'].data, float)
        for q in si:
            ann[q] = annulus_bkg(img, srow['x'][q], srow['y'][q], rin, rout)
        for q in di:
            dann[q] = annulus_bkg(img, drow['x'][q], drow['y'][q], rin, rout)
        print(band, fi, det, ex, len(si), len(di), flush=True)
    srow['lbkg'] = lb
    srow['abkg'] = ann
    drow['abkg'] = dann
    np.savez(f'{OUT}/rows_{band}.npz', meta=np.array([cb, zp, pixas]), finfo=np.array(finfo, dtype=object),
             **{'S_' + k: v for k, v in srow.items()}, **{'D_' + k: v for k, v in drow.items()})
    print('done', band, len(idx), len(didx))


import os
if __name__ == '__main__':
    main()
