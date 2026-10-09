"""Round 7: pedestal-aware rim rewrite, rim = Rh*(g0+B).  usage: python run_frames7.py refit BAND DET EXP
Variants H, H+h0 (as round 6), HBm, HBBs (B from 500<=g0<2000 / 150<=g0<500 far-field median of cal/Rh-g0), +h0 versions.  Writes out7/."""
import os
import sys
import numpy as np
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit')
import satrefit_core as C
import run_frames3 as R3
import run_frames4
import run_frames5 as R5
import run_frames6
from satrefit_core import S, fits, ndimage, Table, SkyCoord

OUT = C.Q + '/satrefit/out7'
R5.VG.update({'162M': '14101', '182M': '16101', '277W': '10101'})


def run_refit(band, det, e):
    R4 = run_frames4
    lw = band in ('250M', '300M')
    fn = R5.path_of(band, det, e)
    stem = R5.stem_of(band, det, e)
    hdr = fits.getheader(fn)
    grid, gf = C.load_grid(R5.TREE + '/psfs', hdr, lw)
    cat = Table.read(fn.replace('.fits', '') + '_resbgsub_m7_satstar_catalog.fits')
    P = C.prep_frame(fn)
    FL = R5.load_frame(fn)
    sw = S.satstar_fit_switches()
    with fits.open(fn, memmap=False) as fh:
        rawerr = np.array(fh['ERR'].data, float)
        vp = np.asarray(fh['VAR_POISSON'].data, float)
    unrec = np.isnan(vp)
    errfloor = float(np.median(rawerr[np.isfinite(rawerr) & (rawerr > 0)]))
    Rh = float(FL['Rhdr'])
    cur0 = R5.pipeline_curve_N(fn, FL, 0)
    c0 = cur0['curve']
    F4 = R4.frame_pieces4(fn, P, lw)
    assert np.allclose(c0[1], F4['pcurve'][1]) and np.allclose(c0[0], F4['pcurve'][0])
    g0 = FL['g0']
    rim = P['rim']
    R0 = np.interp(np.log(np.clip(g0, 1, None)), np.log(c0[0]), c0[1])
    vrim = rim & np.isfinite(g0) & (g0 > 2000.0) & (g0 < cur0['ceiling'])
    chk = np.median(P['data'][vrim] / (R0[vrim] * g0[vrim])) if vrim.any() else np.nan
    C.log('rim check', chk, 'R_header', Rh, 'R_N0(2440)', cur0['R2440'], 'ratio H/N0', Rh / cur0['R2440'])
    fixrim = rim & ~np.isfinite(rawerr)
    f = np.ones(g0.shape)
    sel = rim & np.isfinite(g0)
    f[sel] = Rh / R0[sel]
    far = np.isfinite(FL['cal']) & np.isfinite(g0) & ~FL['sat'] & ((FL['dq'] & 1) == 0) & (FL['edt'] >= 25) & (FL['cal'] != 0)
    ped = FL['cal'] / Rh - g0
    Bm = float(np.median(ped[far & (g0 >= 500) & (g0 < 2000)]))
    Bs = float(np.median(ped[far & (g0 >= 150) & (g0 < 500)]))
    C.log('pedestal B_mid', Bm, 'B_sky', Bs)
    fv = {}
    for nm, B in (('HBm', Bm), ('HBs', Bs)):
        fb = f.copy()
        okb = sel & (g0 + B > 0)
        fb[okb] = Rh * (g0[okb] + B) / (R0[okb] * g0[okb])
        fv[nm] = fb
    ev = P['err'].copy()
    newv = P['data'] * f
    ev[fixrim] = np.maximum(0.05 * np.abs(newv[fixrim]), errfloor if sw['rim_badpix'] else 0.0)
    evv = {'H': ev}
    for nm in fv:
        e2 = P['err'].copy()
        e2[fixrim] = np.maximum(0.05 * np.abs((P['data'] * fv[nm])[fixrim]), errfloor if sw['rim_badpix'] else 0.0)
        evv[nm] = e2
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
            cand = (rimc == 0) & np.isfinite(dat) & (dat != 0) & ~st0.mask & (F4['edt'][sl] <= R4.D)
            sel_h = cand & F4['valid'][sl]
            val_h = F4['val_h'][sl]
            cuts = {}
            cvs = {}
            for nm, ff in (('H', f), ('HBm', fv['HBm']), ('HBs', fv['HBs'])):
                cv = cut_seq.copy()
                cv[rimc] = cut_seq[rimc] + (dat[rimc] * ff[sl][rimc] - dat[rimc])
                cuts[nm] = (cv, evv[nm][sl])
                cvs[nm] = cv
            for nm in ('H', 'HBm', 'HBs'):
                cv2 = cvs[nm].copy()
                cv2[sel_h] += (val_h - dat)[sel_h]
                cuts[nm + '+h0'] = (cv2, evv[nm][sl])
            stv = {}
            for nm, (cv, e_) in cuts.items():
                P2 = dict(P)
                P2['err'] = R3.WinArr(e_)
                st = C.make_setup(P2, wins[i], r, cv, labels[i])
                stv[nm] = st
                out['a_' + nm] = C.solve(st, psf_unit, xfit, yfit, {}, kpois=P['kpois'])
                out['a_' + nm + '+bgfree'] = C.solve(st, psf_unit, xfit, yfit, {'bg': 'free'}, kpois=P['kpois'])
                out['cap_' + nm] = cap_of(cv)
            out['nrw_h'] = int((sel_h & fitpx).sum())
        rows.append(out)
        seq[y0:y1, x0:x1] -= models[i]
        if i % 100 == 0:
            C.log('row', i, n)
    tab = Table(rows=rows)
    sk = cat['skycoord_fit'][:n]
    sk = sk if isinstance(sk, SkyCoord) else SkyCoord(sk)
    tab['ra'] = sk.ra.deg
    tab['dec'] = sk.dec.deg
    tab.meta['RHDR'] = Rh
    tab.meta['RIMCHK'] = float(chk)
    tab.meta['BMID'] = Bm
    tab.meta['BSKY'] = Bs
    outfn = f'{OUT}/{band}_{stem}_satrefit7.fits'
    tab.write(outfn, overwrite=True)
    C.log('wrote', outfn)


if __name__ == '__main__' and sys.argv[1] == 'refit':
    run_refit(sys.argv[2], sys.argv[3], int(sys.argv[4]))
