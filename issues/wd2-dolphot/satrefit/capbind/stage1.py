"""capbind stage 1: per-frame region pixel records for the recovered-core cap + group-0 photometry (LW).
usage: python stage1.py BAND DET EXP     -> s1_<band>_<det>_<exp>.pkl
Per catalogue row (label > 0): cap region pixels with H and H+h0 cutout values, PSF, source DN, flags; group-0 PSF+const fits.
"""
import os
import pickle
import sys
import numpy as np
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit')
import satrefit_core as C
import run_frames3 as R3
import run_frames4 as R4
import run_frames5 as R5
from satrefit_core import S, fits, ndimage, Table, SkyCoord

OUT = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/capbind'
R5.VG.update({'162M': '14101', '182M': '16101', '277W': '10101'})
GAIN = {True: 1.8, False: 2.0}


def g0_fit(st, psf_unit, xfit, yfit, sbc, valid, sigbg, kg):
    """Weighted linear PSF + free constant on valid pixels of the group-0 surface-brightness cutout."""
    use = valid & np.isfinite(sbc)
    if int(use.sum()) < 30:
        return np.nan, np.nan, int(use.sum())
    d = sbc[use]
    var = sigbg ** 2 + kg * np.maximum(d, 0.0)
    w = 1.0 / var
    pu = psf_unit[use]
    M = np.stack([pu, np.ones_like(pu)], axis=1)
    A = M.T @ (M * w[:, None])
    b = M.T @ (w * d)
    sol = np.linalg.solve(A, b)
    cov = np.linalg.inv(A)
    return float(sol[0]), float(np.sqrt(cov[0, 0])), int(use.sum())


def run(band, det, e):
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
    Rh = float(FL['Rhdr'])
    cur0 = R5.pipeline_curve_N(fn, FL, 0)
    c0 = cur0['curve']
    ceiling = float(cur0['ceiling'])
    g0 = FL['g0']
    zf = S._find_zeroframe_for(fn)
    ff = S._find_first_frame_for(fn)
    g0s_all = S._find_group0_saturation_for(fn, do_not_use=True)
    g0s_sat = S._find_group0_saturation_for(fn, do_not_use=False)
    assert np.array_equal(np.nan_to_num(zf), np.nan_to_num(g0))
    sat_buf = ndimage.binary_dilation(FL['sat'], iterations=3)
    g0_eff, repl, kbr = S.first_frame_group0(g0, ff, sat_buf, lo=2000.0, hi=ceiling, group0_saturated=g0s_all)
    rim = P['rim']
    C.log('repl px', int(repl.sum()), 'repl in rim', int((repl & rim).sum()), 'rim', int(rim.sum()))
    ffp = ff[np.isfinite(ff) & (ff > 0)]
    FW_ff = float(ffp.max())
    FW_g0 = ceiling / 0.9
    R0 = np.interp(np.log(np.clip(g0, 1, None)), np.log(c0[0]), c0[1])
    f = np.ones(g0.shape)
    sel = rim & np.isfinite(g0)
    f[sel] = Rh / R0[sel]
    F4 = R4.frame_pieces4(fn, P, lw)
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
    sb = Rh * np.where(np.isfinite(g0), g0, np.nan)
    seqg = sb.copy()
    # field R(g0) curve (wingmig construction: >= 25 px from DQ SATURATED, g0 > 200), flat beyond the measured range
    dnu0 = (FL['dq'].astype(np.int64) & 1) != 0
    gm = np.isfinite(FL['cal']) & np.isfinite(g0) & ~FL['sat'] & ~dnu0 & (g0 > 200) & (FL['edt'] >= 25) & (FL['cal'] > 0) & ~(g0s_all if g0s_all is not None else False)
    cF, mF, _s, _n = R3.rcurve(g0[gm], (FL['cal'] / g0)[gm])
    RF = np.interp(np.log(np.clip(g0, 1, None)), np.log(cF), mF)
    sbf = RF * np.where(np.isfinite(g0), g0, np.nan)
    seqgf = sbf.copy()
    g0flag = np.zeros(shape, bool)
    if g0s_all is not None:
        g0flag |= g0s_all
    badbits = int(S.dqflags.pixel['DO_NOT_USE']) | int(S._RIM_BADPIX_BITS)
    dnu_cal = (FL['dq'].astype(np.int64) & int(S.dqflags.pixel['DO_NOT_USE'])) != 0
    dqbad = (FL['dq'].astype(np.int64) & int(S._RIM_BADPIX_BITS)) != 0
    kg = Rh / GAIN[lw]
    rows = []
    for i in range(n):
        r = cat[i]
        y0, y1, x0, x1 = wins[i]
        xfit, yfit = float(r['x_fit']), float(r['y_fit'])
        out = dict(idx=i, label=int(labels[i]))
        if labels[i] > 0:
            sl = (slice(y0, y1), slice(x0, x1))
            cut_seq = seq[sl].copy()
            dat = P['data'][sl]
            st0 = C.make_setup(P, wins[i], r, cut_seq, labels[i])
            psf = S.psf_in_cutout_coords(grid, x0, y0)
            yy, xx = np.mgrid[0:y1 - y0, 0:x1 - x0]
            psf_unit = np.maximum(psf.evaluate(xx, yy, 1.0, xfit, yfit), 0.0)
            regfit = C.fit_region(st0, psf_unit.shape)
            rimc = rim[sl]
            satwin = (P['zf_deep'] if P['zf_deep'] is not None else P['saturated'])[sl]
            this = (P['sources'][sl] == labels[i]) & satwin
            this_exp = ndimage.binary_dilation(this, iterations=st0.eff_buf)
            region = S.recovered_cap_region(P['sources'][sl] == labels[i], this, this_exp)
            ur = unrec[sl]
            cvH = cut_seq.copy()
            cvH[rimc] = cut_seq[rimc] + (dat[rimc] * f[sl][rimc] - dat[rimc])
            cand = (rimc == 0) & np.isfinite(dat) & (dat != 0) & ~st0.mask & (F4['edt'][sl] <= R4.D)
            sel_h = cand & F4['valid'][sl]
            cvH0 = cvH.copy()
            cvH0[sel_h] += (F4['val_h'][sl] - dat)[sel_h]

            def cap_pipe(cv):
                pk, lost, nrec = S.recovered_core_peak(cv, region, ur, 0.2)
                if not np.isfinite(pk):
                    return np.nan
                return S.recovered_cap_flux(cv, region, ur, psf_unit, min_psf_frac=sw['cap_min_psf_frac'])[0]

            out['cap_H_pipe'] = cap_pipe(cvH)
            out['cap_H0_pipe'] = cap_pipe(cvH0)
            iy, ix = np.nonzero(region)
            ipk = np.unravel_index(np.nanargmax(psf_unit), psf_unit.shape)
            pkidx = -1
            if region[ipk]:
                pkidx = int(np.nonzero((iy == ipk[0]) & (ix == ipk[1]))[0][0])
            sl_r = (iy + y0, ix + x0)
            isrepl = repl[sl_r]
            srcDN = np.where(isrepl, ff[sl_r], g0[sl_r])
            out['reg'] = dict(dy=iy - yfit, dx=ix - xfit, cutH=cvH[iy, ix], cutH0=cvH0[iy, ix], psf=psf_unit[iy, ix], ur=ur[iy, ix],
                              g0=g0[sl_r], ff=ff[sl_r], repl=isrepl, g0sat=(g0s_sat[sl_r] if g0s_sat is not None else np.zeros(len(iy), bool)),
                              g0flag=g0flag[sl_r], src=srcDN, rim=rimc[iy, ix], cal=dat[iy, ix], seqc=cut_seq[iy, ix])
            out['ppk'] = float(np.nanmax(psf_unit))
            out['pkidx'] = pkidx
            out['nreg'] = int(region.sum())
            # group-0 photometry (LW and SW alike)
            other = satwin & ~this
            out_g = {}
            for tag, frac, mode in (('g50', 0.5, 'h'), ('g30', 0.3, 'h'), ('g70', 0.7, 'h'), ('g20', 0.2, 'h'), ('g10', 0.1, 'h'),
                                    ('f50', 0.5, 'f'), ('f30', 0.3, 'f'), ('f20', 0.2, 'f'), ('f10', 0.1, 'f')):
                gc = g0[sl]
                valid = (regfit & np.isfinite(gc) & (gc < frac * ceiling) & ~g0flag[sl] & ~dqbad[sl] & ~dnu_cal[sl] & ~other)
                sbc = (seqg if mode == 'h' else seqgf)[sl]
                valid &= np.isfinite(sbc)
                a, ae, nu = g0_fit(st0, psf_unit, xfit, yfit, sbc, valid, st0.sigma if np.isfinite(st0.sigma) else 1.0, kg)
                out['a_' + tag] = a
                out['ae_' + tag] = ae
                out['n_' + tag] = nu
        rows.append(out)
        seq[y0:y1, x0:x1] -= models[i]
        seqg[y0:y1, x0:x1] -= models[i]
        seqgf[y0:y1, x0:x1] -= models[i]
        if i % 100 == 0:
            C.log('row', i, n)
    meta = dict(Rfield=(cF, mF), Rh=Rh, ceiling=ceiling, FW_ff=FW_ff, FW_g0=FW_g0, kbr=kbr, band=band, det=det, exp=e, stem=stem)
    pickle.dump(dict(meta=meta, rows=rows), open(f'{OUT}/s1_{band}_{det}_{e}' + ('_test' if os.environ.get('MAXROWS') else '') + '.pkl', 'wb'))
    C.log('wrote', band, det, e)


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2], int(sys.argv[3]))
