"""Round 5: pipeline R(g0) curve versus the exclusion distance N from DQ SATURATED, and a refit with the corrected curve.

usage:
  python run_frames5.py curve                       (all frames of the curve survey -> out5/curves5.pkl)
  python run_frames5.py refit BAND DET EXP NSTAR    (writes out5/<band>_<stem>_satrefit5.fits and out5/pix5_<band>_<stem>.npz)

The pipeline's measured R(g0) calibration set in zeroframe_recover_saturated is
    good = ~sat & finite(data) & finite(g0) & data > 0 & 2000 < g0 < ceiling          (N = 0)
This driver reproduces it exactly (same helpers S._rcurve_bins, guard, satcheck) and adds  edt >= N  where edt is the distance
(px) to the nearest DQ SATURATED pixel (DQ corrected for first-group saturation, as the pipeline).  Nothing in the pipeline code is edited.
"""
import copy
import os
import pickle
import sys
import numpy as np

sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit')
import satrefit_core as C
import run_frames3 as R3
from satrefit_core import S, fits, ndimage, Table, SkyCoord

NLIST = [0, 2, 3, 5, 8, 12, 25, 40]
TREE = C.Q + '/tree_main2'
OUT = C.Q + '/satrefit/out5'
VG = {'150W': '10101', '200W': '12101', '250M': '04101', '300M': '12101'}
G_REF = 2440.0


def stem_of(band, det, e):
    return f'jw03523005001_{VG[band]}_{e:05d}_{det}_align_o005_crf'


def path_of(band, det, e):
    return f'{TREE}/F{band}/pipeline/{stem_of(band, det, e)}.fits'


def load_frame(fn):
    with fits.open(fn, memmap=False) as fh:
        hdr = fh[0].header
        dq = np.array(S.correct_dq_first_group_saturation(fh['DQ'].data, fn, hdr.get('INSTRUME', '')))
        cal = np.array(fh['SCI'].data, float)
        vp = np.asarray(fh['VAR_POISSON'].data, float)
        photmjsr = fh['SCI'].header.get('PHOTMJSR', hdr.get('PHOTMJSR'))
    with fits.open(S._find_ramp_for(fn), memmap=True) as r:
        g0 = np.array(r['SCI'].data[0, 0], float)
    data0 = cal.copy()
    data0[np.isnan(vp)] = 0
    sat = (dq & S.dqflags.pixel['SATURATED']) != 0
    edt = ndimage.distance_transform_edt(~sat)
    return dict(hdr=hdr, dq=dq, cal=cal, data0=data0, g0=g0, sat=sat, edt=edt, Rhdr=S.zeroframe_header_R(hdr, photmjsr))


def pipeline_curve_N(fn, F, N):
    """Reproduction of the measured R(g0) path of zeroframe_recover_saturated with the extra cut edt >= N."""
    sw = S.satstar_fit_switches()
    g0, data0, dq, sat, edt = F['g0'], F['data0'], F['dq'], F['sat'], F['edt']
    fin = np.isfinite(g0)
    gs = g0[sat & fin & (g0 > 0)]
    ceiling = 0.9 * np.nanpercentile(gs, 99.0) if gs.size >= 10 else 0.9 * np.nanpercentile(g0[fin], 99.9)
    g0sat = S._find_group0_saturation_for(fn, do_not_use=True)
    clean_raw = fin & (g0 > 0) & (g0 < ceiling)
    if g0sat is not None and np.shape(g0sat) == g0.shape:
        clean_raw &= ~np.asarray(g0sat, bool)
    good = (~sat) & np.isfinite(data0) & fin & (data0 > 0) & (g0 > 2000.0) & (g0 < ceiling)
    ngood0 = int(good.sum())
    if N > 0:
        good = good & (edt >= N)
    res = dict(N=N, ngood=int(good.sum()), ngood0=ngood0, ceiling=float(ceiling), nraw=0, nkept=0, curve=None, raw=([], []),
               fallback=False, satfac=np.nan, rebuilt=False, notes=[])
    curve = None
    if int(good.sum()) >= 50:
        ctr, med = S._rcurve_bins(g0[good], data0[good] / g0[good], 2000.0, ceiling)
        res['nraw'] = len(ctr)
        res['raw'] = (list(ctr), list(med))
        if len(ctr) >= 2 and sw['rcurve_guard']:
            kept = 1
            for k in range(1, len(med)):
                a_, b_ = med[k - 1], med[k]
                if not (a_ > 0 and b_ > 0 and max(a_ / b_, b_ / a_) <= sw['rcurve_maxstep']):
                    break
                kept += 1
            ctr, med = ctr[:kept], med[:kept]
        res['nkept'] = len(ctr)
        if len(ctr) >= 2:
            curve = (np.array(ctr), np.array(med))
        elif len(ctr) == 1:
            curve = (np.array([ctr[0], ctr[0] * 1.01]), np.array([med[0], med[0]]))
    else:
        res['fallback'] = True
        res['notes'].append('fewer than 50 calibration pixels: the pipeline falls back (median / header R)')
    if curve is not None and sw['rcurve_satcheck']:
        dqi = dq.astype(np.int64)
        satcal = (sat & clean_raw & (g0 > 2000.0) & np.isfinite(data0) & (data0 > 0)
                  & ((dqi & (S.dqflags.pixel['DO_NOT_USE'] | S._RIM_BADPIX_BITS)) == 0))
        nsat = int(satcal.sum())
        res['nsat'] = nsat
        if nsat >= S._RCURVE_SATCHECK_MIN_PX:
            gsx, rsx = g0[satcal], data0[satcal] / g0[satcal]
            r_sat, g_sat = float(np.median(rsx)), float(np.median(gsx))
            r_cur = float(np.interp(np.log(g_sat), np.log(curve[0]), curve[1]))
            fac = r_cur / r_sat
            res['satfac'] = fac
            if max(fac, 1 / fac) > S._RCURVE_SATCHECK_MAX_RATIO:
                sc, sm = S._rcurve_bins(gsx, rsx, 2000.0, ceiling)
                curve = (np.array(sc), np.array(sm)) if len(sc) >= 2 else (np.array([g_sat, g_sat * 1.01]), np.array([r_sat, r_sat]))
                res['rebuilt'] = True
    res['curve'] = curve
    if curve is not None:
        res['R2440'] = float(np.interp(np.log(G_REF), np.log(curve[0]), curve[1]))
        res['R3600'] = float(np.interp(np.log(3600.0), np.log(curve[0]), curve[1]))
    else:
        res['R2440'] = res['R3600'] = np.nan
    return res


def wingmig_curve(F):
    g0, cal, sat, edt = F['g0'], F['cal'], F['sat'], F['edt']
    dnu = (F['dq'] & 1) != 0
    good = np.isfinite(cal) & np.isfinite(g0) & ~sat & ~dnu & (g0 > 200) & (edt >= 25) & (cal > 0)
    c, m, s, n = R3.rcurve(g0[good], cal[good] / g0[good])
    return c, m


def survey_frame(band, det, e):
    fn = path_of(band, det, e)
    F = load_frame(fn)
    c, m = wingmig_curve(F)
    Rw = float(np.interp(np.log(G_REF), np.log(c), m))
    out = dict(band=band, det=det, exp=e, Rwing=Rw, Rhdr=F['Rhdr'], res={}, nsat_px=int(F['sat'].sum()))
    for N in NLIST:
        out['res'][N] = pipeline_curve_N(fn, F, N)
    r0 = out['res'][0]
    # share of the N=0 good set near SATURATED
    g0, d0, sat, edt = F['g0'], F['data0'], F['sat'], F['edt']
    good0 = (~sat) & np.isfinite(d0) & np.isfinite(g0) & (d0 > 0) & (g0 > 2000.0) & (g0 < r0['ceiling'])
    ed = edt[good0]
    out['share'] = {k: float((ed <= k).mean()) if ed.size else np.nan for k in (3, 5, 12)}
    # median cal/g0 vs distance in the 2000-3000 DN range (diagnostic of the inflation)
    gg = g0[good0]
    rr = d0[good0] / gg
    sel = (gg > 2000) & (gg < 3000)
    out['Rdist'] = {}
    for lo, hi in ((1, 2), (2, 3), (3, 5), (5, 8), (8, 12), (12, 25), (25, 1e9)):
        s_ = sel & (ed >= lo) & (ed < hi)
        out['Rdist'][(lo, hi)] = (int(s_.sum()), float(np.median(rr[s_])) if s_.sum() >= 20 else np.nan)
    return out


def survey_list():
    L = []
    for band in ('150W', '200W'):
        for det in ('nrca1', 'nrca2', 'nrca3', 'nrca4', 'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4'):
            L.append((band, det, 1))
        for e in (2, 3, 4):
            for det in ('nrcb1', 'nrcb3'):
                L.append((band, det, e))
    for band in ('250M', '300M'):
        for det in ('nrcalong', 'nrcblong'):
            L.append((band, det, 1))
        for e in (2, 3, 4):
            L.append((band, 'nrcblong', e))
    return L


if __name__ == '__main__' and sys.argv[1] == 'curve':
    os.makedirs(OUT, exist_ok=True)
    res = []
    for b, d, e in survey_list():
        C.log('curve', b, d, e)
        res.append(survey_frame(b, d, e))
        pickle.dump(res, open(OUT + '/curves5.pkl', 'wb'))
    C.log('done curve survey', len(res))


# ---------------------------------------------------------------------------------------------------------------------------
# refit with the corrected rim curve
# ---------------------------------------------------------------------------------------------------------------------------
def curve_ratio(c_new, c_old, g):
    """R_new(g) / R_old(g), both interpolated in log g, flat beyond the ends (as np.interp in the pipeline)."""
    lg = np.log(np.clip(g, 1, None))
    return np.interp(lg, np.log(c_new[0]), c_new[1]) / np.interp(lg, np.log(c_old[0]), c_old[1])


def run_refit(band, det, e, nstar):
    import run_frames4 as R4
    lw = band in ('250M', '300M')
    fn = path_of(band, det, e)
    stem = stem_of(band, det, e)
    hdr = fits.getheader(fn)
    grid, gf = C.load_grid(TREE + '/psfs', hdr, lw)
    cat = Table.read(fn.replace('.fits', '') + '_resbgsub_m7_satstar_catalog.fits')
    P = C.prep_frame(fn)
    FL = load_frame(fn)
    sw = S.satstar_fit_switches()
    with fits.open(fn, memmap=False) as fh:
        rawerr = np.array(fh['ERR'].data, float)
        vp = np.asarray(fh['VAR_POISSON'].data, float)
    unrec = np.isnan(vp)
    errfloor = float(np.median(rawerr[np.isfinite(rawerr) & (rawerr > 0)]))
    cur = {N: pipeline_curve_N(fn, FL, N) for N in (0, 5, nstar)}
    c0 = cur[0]['curve']
    F4 = R4.frame_pieces4(fn, P, lw)
    # the N=0 reproduction must agree with run_frames4's pipeline curve
    assert np.allclose(c0[1], F4['pcurve'][1]) and np.allclose(c0[0], F4['pcurve'][0])
    g0 = FL['g0']
    rim = P['rim']
    var = {'RN%d' % nstar: nstar, 'RN5': 5}
    # validation of the reproduction at the rim
    gv = np.where(rim & np.isfinite(g0) & (g0 > 0), g0, np.nan)
    R0 = np.interp(np.log(np.clip(g0, 1, None)), np.log(c0[0]), c0[1])
    vrim = rim & np.isfinite(g0) & (g0 > 2000.0) & (g0 < cur[0]['ceiling'])
    chk = np.median(P['data'][vrim] / (R0[vrim] * g0[vrim])) if vrim.any() else np.nan
    C.log('rim check', chk, 'curves', {N: (cur[N]['ngood'], cur[N]['nkept'], cur[N]['R2440']) for N in cur}, 'floor', errfloor)
    fac = {}
    errs = {}
    fixrim = rim & ~np.isfinite(rawerr)
    for nm, N in var.items():
        f = np.ones(g0.shape)
        sel = rim & np.isfinite(g0)
        f[sel] = curve_ratio(cur[N]['curve'], c0, g0[sel])
        fac[nm] = f
        ev = P['err'].copy()
        newv = P['data'] * f
        ev[fixrim] = np.maximum(0.05 * np.abs(newv[fixrim]), errfloor if sw['rim_badpix'] else 0.0)
        errs[nm] = ev
    # check that the base rim error follows the same rule
    chk_e = np.nanmedian(P['err'][fixrim] / np.maximum(0.05 * np.abs(P['data'][fixrim]), errfloor)) if fixrim.any() else np.nan
    C.log('rim error rule check (should be 1)', chk_e)
    shape = P['data'].shape
    n = min(len(cat), int(os.environ.get('MAXROWS', len(cat))))
    ww = P['ww']
    labels = np.full(n, -1, int)
    for i in range(n):
        r = cat[i]
        if int(r['sat_area']) <= 0 or not np.isfinite(float(r['sat_com_ra'])):
            continue
        cx, cy = ww.world_to_pixel(SkyCoord(float(r['sat_com_ra']), float(r['sat_com_dec']), unit='deg'))
        d = np.hypot(P['comcen'][:, 0] - float(cx), P['comcen'][:, 1] - float(cy))
        j = int(np.nanargmin(d))
        if d[j] < 1.0:
            labels[i] = j + 1
    wins, models = [], []
    for i in range(n):
        r = cat[i]
        win = C.window_of(r, shape)
        wins.append(win)
        models.append(C.render(grid, win, float(r['x_fit']), float(r['y_fit']), float(r['flux_fit_raw']))[1])
    seq = P['data'].copy()
    rows = []
    pixrec = {k: [] for k in ('row', 'u0', 'uN', 'u5', 'g0', 'r', 'cat', 'peak')}
    nm_star = 'RN%d' % nstar
    for i in range(n):
        r = cat[i]
        y0, y1, x0, x1 = wins[i]
        xfit, yfit = float(r['x_fit']), float(r['y_fit'])
        out = dict(idx=i, label=labels[i], a_cat=float(r['flux_fit_precap']), a_raw=float(r['flux_fit_raw']), flux_fit=float(r['flux_fit']))
        if labels[i] > 0:
            sl = (slice(y0, y1), slice(x0, x1))
            cut_seq = seq[sl].copy()
            dat = P['data'][sl]
            st0 = C.make_setup(P, wins[i], r, cut_seq, labels[i])
            psf = S.psf_in_cutout_coords(grid, x0, y0)
            yy, xx = np.mgrid[0:y1 - y0, 0:x1 - x0]
            psf_unit = np.maximum(psf.evaluate(xx, yy, 1.0, xfit, yfit), 0.0)
            reg = C.fit_region(st0, psf_unit.shape)
            fitpx = reg & ~st0.mask & np.isfinite(st0.cut)
            rimc = rim[sl]
            out['nfit'] = int(fitpx.sum())
            out['nrim_fit'] = int((fitpx & rimc).sum())
            out['sat_area'] = int(r['sat_area'])
            # cap region as the pipeline builds it (own_cell for blended components omitted)
            satwin = (P['zf_deep'] if P['zf_deep'] is not None else P['saturated'])[sl]
            this = (P['sources'][sl] == labels[i]) & satwin
            this_exp = ndimage.binary_dilation(this, iterations=st0.eff_buf)
            region = S.recovered_cap_region(P['sources'][sl] == labels[i], this, this_exp)
            ur = unrec[sl]

            def cap_of(cutv):
                pk, lost, nrec = S.recovered_core_peak(cutv, region, ur, 0.2)
                if not np.isfinite(pk):
                    return np.nan
                cp, pf = S.recovered_cap_flux(cutv, region, ur, psf_unit, min_psf_frac=sw['cap_min_psf_frac'])
                return cp

            out['cap_base'] = cap_of(cut_seq)
            out['a_base'] = C.solve(st0, psf_unit, xfit, yfit, {}, kpois=P['kpois'])
            out['a_bgfree'] = C.solve(st0, psf_unit, xfit, yfit, {'bg': 'free'}, kpois=P['kpois'])
            # wing rewrite pixels (rw12h, D = 12), as in round 4
            cand = (rimc == 0) & np.isfinite(dat) & (dat != 0) & ~st0.mask & (F4['edt'][sl] <= R4.D)
            sel_h = cand & F4['valid'][sl]
            val_h = F4['val_h'][sl]
            cuts = {}
            for nm in var:
                cv = cut_seq.copy()
                cv[rimc] = cut_seq[rimc] + (dat[rimc] * fac[nm][sl][rimc] - dat[rimc])
                cuts[nm] = (cv, errs[nm][sl])
            cv = cuts[nm_star][0].copy()
            cv[sel_h] += (val_h - dat)[sel_h]
            cuts[nm_star + '+h0'] = (cv, errs[nm_star][sl])     # crf ERR kept at the wing pixels
            stv = {}
            for nm, (cv, ev) in cuts.items():
                P2 = dict(P)
                P2['err'] = R3.WinArr(ev)
                st = C.make_setup(P2, wins[i], r, cv, labels[i])
                stv[nm] = st
                out['a_' + nm] = C.solve(st, psf_unit, xfit, yfit, {}, kpois=P['kpois'])
                out['a_' + nm + '+bgfree'] = C.solve(st, psf_unit, xfit, yfit, {'bg': 'free'}, kpois=P['kpois'])
                out['cap_' + nm] = cap_of(cv)
            out['nrw_h'] = int((sel_h & fitpx).sum())
            # per-pixel records
            rr = np.hypot(xx - xfit, yy - yfit)
            pkv = psf_unit.max()
            ipk = np.unravel_index(np.argmax(psf_unit), psf_unit.shape)
            selp = reg & (rr <= 30) & (psf_unit >= 1e-3 * pkv) & np.isfinite(st0.cut) & (st0.cut != 0)
            cat_ = np.where(rimc & ~st0.mask, 0, np.where(~rimc & ~st0.mask, 1, np.where(rimc, 2, -1)))
            selp &= cat_ >= 0
            pk_flag = np.zeros(psf_unit.shape, bool)
            pk_flag[ipk] = True
            w = selp
            pixrec['row'].append(np.full(int(w.sum()), i, np.int32))
            pixrec['u0'].append(((st0.cut - st0.bkg)[w] / psf_unit[w]).astype(np.float32))
            pixrec['uN'].append(((stv[nm_star].cut - stv[nm_star].bkg)[w] / psf_unit[w]).astype(np.float32))
            pixrec['u5'].append(((stv['RN5'].cut - stv['RN5'].bkg)[w] / psf_unit[w]).astype(np.float32))
            pixrec['g0'].append(g0[sl][w].astype(np.float32))
            pixrec['r'].append(rr[w].astype(np.float32))
            pixrec['cat'].append(cat_[w].astype(np.int8))
            pixrec['peak'].append(pk_flag[w])
        rows.append(out)
        seq[y0:y1, x0:x1] -= models[i]
        if i % 100 == 0:
            C.log('row', i, n)
    tab = Table(rows=rows)
    sk = cat['skycoord_fit'][:n]
    sk = sk if isinstance(sk, SkyCoord) else SkyCoord(sk)
    tab['ra'] = sk.ra.deg
    tab['dec'] = sk.dec.deg
    tab.meta['NSTAR'] = nstar
    tab.meta['RIMCHK'] = float(chk)
    tab.meta['RIMERR'] = float(chk_e)
    outfn = f'{OUT}/{band}_{stem}_satrefit5.fits'
    tab.write(outfn, overwrite=True)
    np.savez_compressed(f'{OUT}/pix5_{band}_{stem}.npz', **{k: np.concatenate(v) for k, v in pixrec.items()})
    with open(f'{OUT}/{band}_{stem}_curves5.txt', 'w') as f:
        for N in cur:
            c = cur[N]['curve']
            f.write(f'N={N} ngood={cur[N]["ngood"]} nkept={cur[N]["nkept"]}\n')
            for a in zip(*c):
                f.write('  %.2f %.5f\n' % a)
    C.log('wrote', outfn)


if __name__ == '__main__' and sys.argv[1] == 'refit':
    run_refit(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
