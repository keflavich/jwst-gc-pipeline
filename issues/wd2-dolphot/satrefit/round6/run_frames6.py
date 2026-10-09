"""Round 6: header R (S.zeroframe_header_R = PHOTMJSR / (TFRAME (NFRAMES+1)/2)) as the rim-rewrite R.
usage:
  python run_frames6.py survey              curve survey (R_N0, R_N25 versus header R at 2440 DN) and far-field cal/g0/R_hdr -> out6/survey6.pkl
  python run_frames6.py refit BAND DET EXP  refit with R(g0) = R_header -> out6/<band>_<stem>_satrefit6.fits, out6/pix6_<band>_<stem>.npz
Reuses run_frames5 helpers (import only; nothing is edited).
"""
import os
import pickle
import sys
import numpy as np
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit')
import satrefit_core as C
import run_frames3 as R3
import run_frames5 as R5
from satrefit_core import S, fits, ndimage, Table, SkyCoord

OUT = C.Q + '/satrefit/out6'
R5.VG.update({'162M': '14101', '182M': '16101', '277W': '10101'})
SWB = ('150W', '162M', '182M', '200W')
LWB = ('250M', '277W', '300M')
GBINS = [(200, 500), (500, 1000), (1000, 2000), (2000, 4000)]


def survey_list():
    L = []
    for b in SWB:
        for det in ('nrca1', 'nrca2', 'nrca3', 'nrca4', 'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4'):
            L.append((b, det, 1))
    for b in LWB:
        for det in ('nrcalong', 'nrcblong'):
            L.append((b, det, 1))
    for b in ('150W', '200W'):          # extra exposures, far-field comparison only
        for e in (2, 3, 4):
            for det in ('nrcb1', 'nrcb3'):
                L.append((b, det, e))
    for b in ('250M', '300M'):
        for e in (2, 3, 4):
            L.append((b, 'nrcblong', e))
    return L


def survey_frame(band, det, e):
    fn = R5.path_of(band, det, e)
    F = R5.load_frame(fn)
    out = dict(band=band, det=det, exp=e, Rhdr=F['Rhdr'])
    for N in (0, 25):
        r = R5.pipeline_curve_N(fn, F, N)
        out[N] = dict(R2440=r['R2440'], ngood=r['ngood'], nkept=r['nkept'], nraw=r['nraw'], rebuilt=r['rebuilt'], fallback=r['fallback'], satfac=r['satfac'])
    g0, cal, sat, edt = F['g0'], F['cal'], F['sat'], F['edt']
    dnu = (F['dq'] & 1) != 0
    far = np.isfinite(cal) & np.isfinite(g0) & ~sat & ~dnu & (cal > 0) & (g0 > 200) & (edt >= 25)
    out['far'] = {}
    for lo, hi in GBINS:
        s = far & (g0 >= lo) & (g0 < hi)
        n = int(s.sum())
        out['far'][(lo, hi)] = (n, float(np.median(cal[s] / g0[s]) / F['Rhdr']) if n >= 50 else np.nan)
    return out


if __name__ == '__main__' and sys.argv[1] == 'survey':
    os.makedirs(OUT, exist_ok=True)
    res = []
    for b, d, e in survey_list():
        C.log('survey', b, d, e)
        res.append(survey_frame(b, d, e))
        pickle.dump(res, open(OUT + '/survey6.pkl', 'wb'))
    C.log('done survey', len(res))


def run_refit(band, det, e):
    import run_frames4 as R4
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
    ev = P['err'].copy()
    newv = P['data'] * f
    ev[fixrim] = np.maximum(0.05 * np.abs(newv[fixrim]), errfloor if sw['rim_badpix'] else 0.0)
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
    pixrec = {k: [] for k in ('row', 'u0', 'uH', 'g0', 'r', 'cat', 'peak')}
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
            cvH = cut_seq.copy()
            cvH[rimc] = cut_seq[rimc] + (dat[rimc] * f[sl][rimc] - dat[rimc])
            evH = ev[sl]
            cuts = {'H': (cvH, evH)}
            cv2 = cvH.copy()
            cv2[sel_h] += (val_h - dat)[sel_h]
            cuts['H+h0'] = (cv2, evH)
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
            pixrec['uH'].append(((stv['H'].cut - stv['H'].bkg)[w] / psf_unit[w]).astype(np.float32))
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
    tab.meta['RHDR'] = Rh
    tab.meta['RIMCHK'] = float(chk)
    outfn = f'{OUT}/{band}_{stem}_satrefit6.fits'
    tab.write(outfn, overwrite=True)
    np.savez_compressed(f'{OUT}/pix6_{band}_{stem}.npz', **{k: np.concatenate(v) for k, v in pixrec.items()})
    C.log('wrote', outfn)


if __name__ == '__main__' and sys.argv[1] == 'refit':
    run_refit(sys.argv[2], sys.argv[3], int(sys.argv[4]))
