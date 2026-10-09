"""Point-source check of the group-0 rate scale.  usage: python run_rstar.py BAND DET EXP  -> rstar_<band>_<det>_<exp>.pkl
For isolated UNSATURATED dolphot stars, fit the same PSF + free constant, with the same pixels and weights, to
  cal  : the crf SCI image (= rate x PHOTMJSR / flat),
  gh   : R_header x g0             (what the H rim rewrite and the first-frame core replacement write),
  ghf  : R_header x g0 / flat      (the same with the flat applied),
  gf   : R_field(g0) x g0          (wingmig field curve, cal/g0 on pixels >= 25 px from SATURATED),
and record the ratios a_gh / a_cal etc. against magnitude, peak group-0 level and detector.
Aperture sums (r < 3, 5, 8 px, annulus 15-30 px median) give a PSF-model-free version of the same ratios."""
import os
import sys
import pickle
import numpy as np
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit')
import satrefit_core as C
import run_frames3 as R3
import run_frames5 as R5
from satrefit_core import S, fits, ndimage, Table
from scipy.optimize import minimize
from scipy.spatial import cKDTree

OUT = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/rstar'
CAT = '/orange/adamginsburg/jwst/wd2/catalogs/wd2_nircam_wf_mf_nf.ecsv'
CRDS = '/orange/adamginsburg/jwst/crds/references/jwst/nircam/'
HALF = 40
RFIT = {True: 10.0, False: 14.0}
RAP = (3.0, 5.0, 8.0)
BIN_, BOUT = 15, 30
GAIN = {True: 1.8, False: 2.0}
SNMIN = 20.0


def render_sub(psf, shape, x, y, rad):
    """unit-flux PSF on the window |dx|,|dy|<=rad around (x, y); returns (slices, array)."""
    ny, nx = shape
    xa, xb = max(0, int(x) - rad), min(nx, int(x) + rad + 1)
    ya, yb = max(0, int(y) - rad), min(ny, int(y) + rad + 1)
    if xb <= xa or yb <= ya:
        return None, None
    yy, xx = np.mgrid[ya:yb, xa:xb]
    return (slice(ya, yb), slice(xa, xb)), np.maximum(psf.evaluate(xx, yy, 1.0, x, y), 0.0)


def wfit(d, pu, w):
    """Weighted linear PSF amplitude + constant; returns (a, sigma_a, const)."""
    M = np.stack([pu, np.ones_like(pu)], axis=1)
    A = M.T @ (M * w[:, None])
    b = M.T @ (w * d)
    sol = np.linalg.solve(A, b)
    cov = np.linalg.inv(A)
    return float(sol[0]), float(np.sqrt(cov[0, 0])), float(sol[1])


def main(band, det, e):
    lw = band in ('250M', '300M')
    fn = R5.path_of(band, det, e)
    FL = R5.load_frame(fn)
    hdr = FL['hdr']
    Rh = float(FL['Rhdr'])
    g0 = FL['g0']
    cal = FL['data0']
    dq = FL['dq']
    with fits.open(fn, memmap=False) as fh:
        err = np.array(fh['ERR'].data, float)
        vp = np.asarray(fh['VAR_POISSON'].data, float)
        ww = S.frame_wcs(fh)
    err[~np.isfinite(err) | (err <= 0)] = 1e10
    flat = np.array(fits.getdata(CRDS + hdr['R_FLAT'].split('//')[1], 'SCI'), float)
    flat[~np.isfinite(flat) | (flat <= 0)] = np.nan
    cur0 = R5.pipeline_curve_N(fn, FL, 0)
    ceiling = float(cur0['ceiling'])
    g0s_all = S._find_group0_saturation_for(fn, do_not_use=True)
    g0flag = np.zeros(g0.shape, bool) if g0s_all is None else np.asarray(g0s_all, bool)
    sat = FL['sat']
    dnu = (dq & int(S.dqflags.pixel['DO_NOT_USE'])) != 0
    badq = sat | dnu | g0flag | ~np.isfinite(vp) | ~np.isfinite(g0) | ~np.isfinite(flat)
    edt_sat = FL['edt']
    # field R(g0) curve, wingmig construction (as capbind/stage1.py)
    gm = np.isfinite(FL['cal']) & np.isfinite(g0) & ~sat & ~((dq.astype(np.int64) & 1) != 0) & (g0 > 200) & (edt_sat >= 25) \
        & (FL['cal'] > 0) & ~g0flag
    cF, mF, _s, _n = R3.rcurve(g0[gm], (FL['cal'] / g0)[gm])
    RF = np.interp(np.log(np.clip(g0, 1, None)), np.log(cF), mF)
    img = dict(cal=cal, gh=Rh * g0, ghf=Rh * g0 / flat, gf=RF * g0)
    for v in img.values():
        v[~np.isfinite(v)] = 0.0
    grid, gf_ = C.load_grid(R5.TREE + '/psfs', hdr, lw)
    C.log('frame', os.path.basename(fn), 'Rh', Rh, 'ceiling', ceiling, 'field R at 2440/Rh', float(np.interp(np.log(2440.0), np.log(cF), mF)) / Rh)
    ny, nx = cal.shape
    cat = Table.read(CAT)
    mag = np.asarray(cat['MAG' + band], float)
    okm = np.isfinite(mag) & (mag > 0) & (mag < 40)
    ra, dec = np.asarray(cat['RA'], float), np.asarray(cat['DEC'], float)
    px, py = ww.world_to_pixel_values(ra, dec)
    infr = okm & (px > -30) & (px < nx + 30) & (py > -30) & (py < ny + 30)
    cx, cy, cm, cidx = px[infr], py[infr], mag[infr], np.nonzero(infr)[0]
    tree = cKDTree(np.c_[cx, cy])
    riso = 5.0 if lw else 8.0
    dmiso = 2.5 * np.log10(20.0)
    stat = dict(n_in=0, dq=0, sat_near=0, iso=0, sn=0, pos=0, fit=0)
    rows = []
    kg = Rh / GAIN[lw]
    for k in range(len(cx)):
        x, y = cx[k], cy[k]
        if not (HALF + 1 <= x < nx - HALF - 1 and HALF + 1 <= y < ny - HALF - 1):
            continue
        stat['n_in'] += 1
        xi, yi = int(round(x)), int(round(y))
        w3 = cal[yi - 1:yi + 2, xi - 1:xi + 2]
        j = np.unravel_index(np.argmax(w3), w3.shape)
        pxi, pyi = xi - 1 + j[1], yi - 1 + j[0]
        yy, xx = np.ogrid[pyi - 3:pyi + 4, pxi - 3:pxi + 4]
        disk = (xx - pxi) ** 2 + (yy - pyi) ** 2 <= 9
        if (badq[pyi - 3:pyi + 4, pxi - 3:pxi + 4] & disk).any() or cal[pyi, pxi] == 0:
            stat['dq'] += 1
            continue
        if edt_sat[pyi, pxi] < 25:
            stat['sat_near'] += 1
            continue
        nbr = [q for q in tree.query_ball_point([x, y], riso) if q != k]
        if any(cm[q] < cm[k] + dmiso for q in nbr):
            stat['iso'] += 1
            continue
        x0, y0 = pxi - HALF, pyi - HALF
        sl = (slice(y0, y0 + 2 * HALF + 1), slice(x0, x0 + 2 * HALF + 1))
        c0 = img['cal'][sl]
        shape = c0.shape
        yy, xx = np.indices(shape)
        rr0 = np.hypot(xx - HALF, yy - HALF)
        ann = (rr0 >= BIN_) & (rr0 < BOUT) & (c0 != 0)
        snr = (cal[pyi, pxi] - np.median(c0[ann])) / err[pyi, pxi]
        if snr < SNMIN:
            stat['sn'] += 1
            continue
        bad = badq[sl] | (c0 == 0)
        psf = S.psf_in_cutout_coords(grid, x0, y0)
        xs, ys = float(cx[k] - x0), float(cy[k] - y0)
        nbi = [q for q in tree.query_ball_point([cx[k], cy[k]], HALF * 1.5 + 12) if q != k and cm[q] < cm[k] + 8.0
               and -10 < cx[q] - x0 < shape[1] + 10 and -10 < cy[q] - y0 < shape[0] + 10]
        nbm = np.zeros(shape)
        for q in nbi:
            aq = 10 ** (-0.4 * (cm[q] - cm[k]))
            s2, p2 = render_sub(psf, shape, cx[q] - x0, cy[q] - y0, 40 if aq > 0.02 else 20)
            if s2 is not None:
                nbm[s2] += aq * p2
        s2, pk = render_sub(psf, shape, xs, ys, 3)
        a0 = max(float(c0[pyi - y0, pxi - x0]) / max(pk.max(), 1e-6), 1e-3)
        cut = c0 - a0 * nbm

        def core_chi(p, rad=4):
            if np.hypot(p[0] - xs, p[1] - ys) > 3.0:
                return 1e30
            s3, pu3 = render_sub(psf, shape, p[0], p[1], rad)
            d = cut[s3]
            m = ~bad[s3]
            M = np.stack([pu3[m], np.ones(m.sum())], axis=1)
            sol, *_ = np.linalg.lstsq(M, d[m], rcond=None)
            return float(np.sum((d[m] - M @ sol) ** 2))

        r = minimize(core_chi, [xs, ys], method='Nelder-Mead', options=dict(xatol=0.01, fatol=1e-6, maxiter=80))
        xf, yf = float(r.x[0]), float(r.x[1])
        if np.hypot(xf - xs, yf - ys) > 1.5:
            stat['pos'] += 1
            continue
        pu = np.maximum(psf.evaluate(xx, yy, 1.0, xf, yf), 0.0)
        rr = np.hypot(xx - xf, yy - yf)
        extra = np.zeros(shape, bool)
        for q in nbi:
            if cm[q] < cm[k] + 4.25:
                extra |= np.hypot(xx - (cx[q] - x0), yy - (cy[q] - y0)) < 2.5
        use = ~bad & ~extra & (rr < RFIT[lw])
        anm = ~bad & ~extra & (rr >= BIN_) & (rr < BOUT)
        if use.sum() < 50 or anm.sum() < 50:
            stat['fit'] += 1
            continue
        # first pass on cal for the neighbour scale
        dcal = img['cal'][sl]
        ghc = img['gh'][sl]
        sigbg = 1.4826 * np.median(np.abs(ghc[anm] - np.median(ghc[anm])))
        w = 1.0 / (sigbg ** 2 + kg * np.maximum(ghc[use] - np.median(ghc[anm]), 0.0))
        a1, _, _ = wfit((dcal - a0 * nbm)[use], pu[use], w)
        if not np.isfinite(a1) or a1 <= 0:
            stat['fit'] += 1
            continue
        out = dict(frame=(band, det, e), k=int(cidx[k]), mag=float(cm[k]), x=float(xf + x0), y=float(yf + y0), snr=float(snr),
                   peak_cal=float(cal[pyi, pxi]), peak_g0=float(g0[pyi, pxi]), g0frac=float(g0[pyi, pxi] / ceiling),
                   flat_pk=float(flat[pyi, pxi]), flat_w=float(np.sum((pu * pu * flat[sl])[use]) / np.sum((pu * pu)[use])),
                   nnb=len(nbi), nuse=int(use.sum()), sigbg=float(sigbg))
        for name, im in img.items():
            d = im[sl] - a1 * nbm
            a, ae, cst = wfit(d[use], pu[use], w)
            out['a_' + name], out['ae_' + name], out['c_' + name] = a, ae, cst
            bk = float(np.median(d[anm]))
            for rap in RAP:
                ap = (rr < rap) & ~bad & ~extra
                out[f'ap{rap:.0f}_' + name] = float(np.sum(d[ap] - bk))
                out[f'apn{rap:.0f}'] = int(ap.sum())
            out['bk_' + name] = bk
        out['psfap3'] = float(np.sum(pu[(rr < 3.0)]))
        rows.append(out)
        if len(rows) % 200 == 0:
            C.log('rows', len(rows))
    meta = dict(band=band, det=det, exp=e, Rh=Rh, ceiling=ceiling, stat=stat, cF=cF, mF=mF, flat_file=hdr['R_FLAT'])
    pickle.dump(dict(meta=meta, rows=rows), open(f'{OUT}/rstar_{band}_{det}_{e}.pkl', 'wb'))
    C.log('stats', stat, 'wrote', len(rows))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
