"""Core-mask experiment on UNSATURATED isolated stars.  usage: python run_unsat.py BAND DET EXP  -> unsat_<band>_<det>_<exp>.pkl
For each selected star: refine position with a full-PSF fit, solve the amplitude (satrefit_core.solve) with no mask,
circular core masks and model-threshold masks, annulus-bkg and free-bkg modes."""
import os
import sys
import pickle
import numpy as np
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit')
import satrefit_core as C
import run_frames5 as R5
from satrefit_core import S, fits, ndimage, Table, SkyCoord, LocalBackground
from scipy.optimize import minimize

OUT = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/coremask'
CAT = '/orange/adamginsburg/jwst/wd2/catalogs/wd2_nircam_wf_mf_nf.ecsv'
R5.VG.update({'162M': '14101', '182M': '16101', '277W': '10101'})
HALF = 40
CIRC = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
THR = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
BIN_, BOUT = 15, 30
NLOW = 120


def render_sub(psf, shape, x, y, rad):
    """unit-flux PSF on the window |dx|,|dy|<=rad around (x, y); returns (slices, array)."""
    ny, nx = shape
    xa, xb = max(0, int(x) - rad), min(nx, int(x) + rad + 1)
    ya, yb = max(0, int(y) - rad), min(ny, int(y) + rad + 1)
    if xb <= xa or yb <= ya:
        return None, None
    yy, xx = np.mgrid[ya:yb, xa:xb]
    return (slice(ya, yb), slice(xa, xb)), np.maximum(psf.evaluate(xx, yy, 1.0, x, y), 0.0)


def main(band, det, e):
    lw = band in ('250M', '300M')
    fn = R5.path_of(band, det, e)
    rng = np.random.default_rng(1000 * e + len(band))
    with fits.open(fn, memmap=False) as fh:
        hdr = fh[0].header
        dq = np.array(S.correct_dq_first_group_saturation(fh['DQ'].data, fn, hdr.get('INSTRUME', '')))
        data = np.array(fh['SCI'].data, float)
        err = np.array(fh['ERR'].data, float)
        vp = np.asarray(fh['VAR_POISSON'].data, float)
        ww = S.frame_wcs(fh)
    data[np.isnan(vp)] = 0
    err[~np.isfinite(err) | (err <= 0)] = 1e10
    grid, gf = C.load_grid(R5.TREE + '/psfs', hdr, lw)
    sat = (dq & int(S.dqflags.pixel['SATURATED'])) != 0
    dnu = (dq & int(S.dqflags.pixel['DO_NOT_USE'])) != 0
    badq = sat | dnu
    edt_sat = ndimage.distance_transform_edt(~sat)
    good = np.isfinite(data) & ~ndimage.binary_dilation(sat, iterations=3) & ~dnu & np.isfinite(vp) & (data != 0)
    Lsat = float(np.percentile(data[good], 99.999))
    nb = ndimage.binary_dilation(sat, iterations=1) & ~sat & np.isfinite(data) & np.isfinite(vp)
    Lnb = float(np.percentile(data[nb], 99)) if nb.sum() > 20 else np.nan
    C.log('frame', os.path.basename(fn), 'L(p99.999)', Lsat, 'L(sat-neighbour p99)', Lnb)
    ny, nx = data.shape
    cat = Table.read(CAT)
    mcol = 'MAG' + band
    mag = np.asarray(cat[mcol], float)
    okm = np.isfinite(mag) & (mag > 0) & (mag < 40)
    ra, dec = np.asarray(cat['RA'], float), np.asarray(cat['DEC'], float)
    px, py = ww.world_to_pixel_values(ra, dec)
    infr = okm & (px > -30) & (px < nx + 30) & (py > -30) & (py < ny + 30)
    cx, cy, cm, cidx = px[infr], py[infr], mag[infr], np.nonzero(infr)[0]
    C.log('dolphot rows in frame', len(cx))
    from scipy.spatial import cKDTree
    tree = cKDTree(np.c_[cx, cy])
    riso = 5.0 if lw else 8.0
    dmiso = 2.5 * np.log10(20.0)
    cand = []
    stat = dict(n_in=0, edge=0, dq=0, sat_near=0, iso=0, sn=0)
    for k in range(len(cx)):
        x, y = cx[k], cy[k]
        if not (HALF + 1 <= x < nx - HALF - 1 and HALF + 1 <= y < ny - HALF - 1):
            continue
        stat['n_in'] += 1
        xi, yi = int(round(x)), int(round(y))
        w3 = data[yi - 1:yi + 2, xi - 1:xi + 2]
        j = np.unravel_index(np.argmax(np.where(np.isfinite(w3), w3, -np.inf)), w3.shape)
        pxi, pyi = xi - 1 + j[1], yi - 1 + j[0]
        yy, xx = np.ogrid[pyi - 3:pyi + 4, pxi - 3:pxi + 4]
        disk = (xx - pxi) ** 2 + (yy - pyi) ** 2 <= 9
        if (badq[pyi - 3:pyi + 4, pxi - 3:pxi + 4] & disk).any() or data[pyi, pxi] == 0:
            stat['dq'] += 1
            continue
        if edt_sat[pyi, pxi] < 25:
            stat['sat_near'] += 1
            continue
        nbr = [q for q in tree.query_ball_point([x, y], riso) if q != k]
        if any(cm[q] < cm[k] + dmiso for q in nbr):
            stat['iso'] += 1
            continue
        cand.append((k, pxi, pyi))
    C.log('stats', stat, 'candidates', len(cand))
    # S/N and peak fraction
    sel = []
    for k, pxi, pyi in cand:
        x0, y0 = pxi - HALF, pyi - HALF
        cut = data[y0:y0 + 2 * HALF + 1, x0:x0 + 2 * HALF + 1]
        yy, xx = np.indices(cut.shape)
        rr = np.hypot(xx - HALF, yy - HALF)
        ann = (rr >= BIN_) & (rr < BOUT) & (cut != 0)
        b0 = np.median(cut[ann])
        snr = (data[pyi, pxi] - b0) / err[pyi, pxi]
        if snr < 50:
            stat['sn'] += 1
            continue
        sel.append((k, pxi, pyi, data[pyi, pxi] / Lsat))
    fr = np.array([s[3] for s in sel])
    low = np.nonzero(fr < 0.1)[0]
    keep = set(np.nonzero(fr >= 0.1)[0].tolist())
    if len(low) > NLOW:
        low = rng.choice(low, NLOW, replace=False)
    keep |= set(int(i) for i in low)
    sel = [sel[i] for i in sorted(keep)]
    C.log('after S/N and subsample', len(sel), 'fail S/N', stat['sn'])
    rows = []
    stamps = []
    for n_, (k, pxi, pyi, frac) in enumerate(sel):
        x0, y0 = pxi - HALF, pyi - HALF
        sl = (slice(y0, y0 + 2 * HALF + 1), slice(x0, x0 + 2 * HALF + 1))
        cut0 = data[sl].copy()
        errc = err[sl]
        psf = S.psf_in_cutout_coords(grid, x0, y0)
        shape = cut0.shape
        bad = (cut0 == 0) | ~np.isfinite(cut0) | badq[sl]
        cut0[~np.isfinite(cut0)] = 0
        xs, ys = float(cx[k] - x0), float(cy[k] - y0)
        # neighbours from the dolphot catalogue within the cutout (+10 px)
        nbi = [q for q in tree.query_ball_point([cx[k], cy[k]], HALF * 1.5 + 12) if q != k and cm[q] < cm[k] + 8.0
               and -10 < cx[q] - x0 < shape[1] + 10 and -10 < cy[q] - y0 < shape[0] + 10]

        def neighbours(a_star):
            m = np.zeros(shape)
            for q in nbi:
                aq = a_star * 10 ** (-0.4 * (cm[q] - cm[k]))
                rad = 40 if aq > 0.02 * a_star else 20
                s2, p2 = render_sub(psf, shape, cx[q] - x0, cy[q] - y0, rad)
                if s2 is None:
                    continue
                m[s2] += aq * p2
            return m

        # initial amplitude from the peak
        s2, pk = render_sub(psf, shape, xs, ys, 3)
        a0 = max(float(cut0[pyi - y0, pxi - x0]) / max(pk.max(), 1e-6), 1e-3)
        cut = cut0 - neighbours(a0)

        def core_chi(p, rad=4):
            if np.hypot(p[0] - xs, p[1] - ys) > 3.0:
                return 1e30
            s3, pu = render_sub(psf, shape, p[0], p[1], rad)
            d = cut[s3]
            m = ~bad[s3]
            M = np.stack([pu[m], np.ones(m.sum())], axis=1)
            sol, res, *_ = np.linalg.lstsq(M, d[m], rcond=None)
            return float(np.sum((d[m] - M @ sol) ** 2))

        r = minimize(core_chi, [xs, ys], method='Nelder-Mead', options=dict(xatol=0.01, fatol=1e-6, maxiter=80))
        xf, yf = float(r.x[0]), float(r.x[1])
        if np.hypot(xf - xs, yf - ys) > 1.5:
            continue
        yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
        pu = np.maximum(psf.evaluate(xx, yy, 1.0, xf, yf), 0.0)
        # setup
        st = C.Setup()
        st.errc = errc
        st.x_init, st.y_init = HALF, HALF
        st.bkg_in, st.bkg_out = BIN_, BOUT
        st.dist = None
        st.satmask = bad
        st.eff_buf = 0

        def prep(cutv, extra=None):
            st.cut = cutv
            msk = bad.copy()
            if extra is not None:
                msk |= extra
            st.mask = msk
            st.bkg = float(LocalBackground(BIN_, BOUT)(cutv, HALF, HALF, mask=msk))
            ef, sg = S.bkg_scatter_fit_error(errc, cutv, ~msk, HALF, HALF, BIN_, BOUT)
            st.sigma = sg

        prep(cut)
        a_full0 = C.solve(st, pu, xf, yf, {})
        if not np.isfinite(a_full0) or a_full0 <= 0:
            continue
        cut = cut0 - neighbours(a_full0)
        # mask residual-prone neighbour cores (brighter than 0.02x the star)
        extra = np.zeros(shape, bool)
        for q in nbi:
            if cm[q] < cm[k] + 4.25:
                extra |= np.hypot(xx - (cx[q] - x0), yy - (cy[q] - y0)) < 2.5
        prep(cut, extra)
        ptop = pu.max()
        rr = np.hypot(xx - xf, yy - yf)
        masks = {'full': np.zeros(shape, bool)}
        for rc in CIRC:
            masks['c%.1f' % rc] = rr < rc
        for f in THR:
            masks['t%.3f' % f] = pu > f * ptop
        out = dict(frame=(band, det, e), k=int(cidx[k]), ra=float(ra[cidx[k]]), dec=float(dec[cidx[k]]), mag=float(cm[k]),
                   x=float(cx[k]), y=float(cy[k]), xf=float(xf + x0), yf=float(yf + y0), peak=float(data[pyi, pxi]), frac=float(frac),
                   frac_nb=float(data[pyi, pxi] / Lnb) if np.isfinite(Lnb) else np.nan, bkg=st.bkg, sigma=st.sigma, nnb=len(nbi),
                   ppeak=float(ptop))
        base_mask = st.mask.copy()
        for name, mk in masks.items():
            st.mask = base_mask | mk
            out['n_' + name] = int((mk & ~base_mask).sum())
            out['a_' + name] = C.solve(st, pu, xf, yf, {})
            out['b_' + name] = C.solve(st, pu, xf, yf, {'bg': 'free'})
        st.mask = base_mask
        rows.append(out)
        if frac >= 0.5:
            h = 12
            xi_, yi_ = int(round(xf)), int(round(yf))
            if h <= xi_ < shape[1] - h and h <= yi_ < shape[0] - h:
                s3 = (slice(yi_ - h, yi_ + h + 1), slice(xi_ - h, xi_ + h + 1))
                stamps.append(dict(idx=len(rows) - 1, data=cut[s3].copy(), model=(out['a_full'] * pu + st.bkg)[s3].copy(),
                                   xf=xf - (xi_ - h), yf=yf - (yi_ - h), mask=(rr < 2.0)[s3].copy(), thr=(pu > 0.1 * ptop)[s3].copy(),
                                   mag=out['mag'], frac=frac))
        if n_ % 50 == 0:
            C.log('star', n_, len(sel), 'done', len(rows))
    meta = dict(band=band, det=det, exp=e, Lsat=Lsat, Lnb=Lnb, stat=stat, ncand=len(cand), nsel=len(sel))
    pickle.dump(dict(meta=meta, rows=rows, stamps=stamps), open(f'{OUT}/unsat_{band}_{det}_{e}.pkl', 'wb'))
    C.log('wrote', len(rows))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
