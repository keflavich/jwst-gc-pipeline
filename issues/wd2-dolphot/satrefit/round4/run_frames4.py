"""Round 4: charge-migration rewrite extended to the high-g0 near-core pixels.  usage: python run_frames4.py BAND DET EXP [MAXROWS]
(writes out4/<band>_<stem>_satrefit4.fits and out4/<band>_<stem>_pcurve.txt)

Variants (D = 12 px from DQ SATURATED, each solved with base and bgfree):
  rw12h_e1  R held flat above the top wingmig R bin up to the pipeline group-0 ceiling; propagated error (round-3 error model)
  rw12h_e0  same pixels and values, crf ERR kept
  rw12p_e1  g0 < 2000 DN: wingmig curve; g0 >= 2000 DN: the pipeline's own R(g0) curve (zeroframe_recover_saturated construction
            reproduced here: 8 log bins from R_g0_min=2000 to the ceiling, step guard, SATURATED-pixel check); propagated error.
Ceiling = 0.9 x p99 of positive group-0 at DQ SATURATED pixels (as zeroframe_recover_saturated).
Also recorded per star: pixels newly rewritten by rw12h relative to round-3 rw12 (g0 above the top wingmig bin) and their q = cal/(R g0).
"""
import os
import sys
import numpy as np

sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit')
import satrefit_core as C
import run_frames3 as R3
from satrefit_core import S, fits, ndimage, Table, SkyCoord

D = 12


def pipeline_curve(fn, data0, g0, dq):
    """Reproduce the measured R(g0) curve of zeroframe_recover_saturated (read-only reuse of its helpers)."""
    sw = S.satstar_fit_switches()
    sat = (dq & S.dqflags.pixel['SATURATED']) != 0
    fin = np.isfinite(g0)
    gs = g0[sat & fin & (g0 > 0)]
    ceiling = 0.9 * np.nanpercentile(gs, 99.0) if gs.size >= 10 else 0.9 * np.nanpercentile(g0[fin], 99.9)
    g0sat = S._find_group0_saturation_for(fn, do_not_use=True)
    clean_raw = fin & (g0 > 0) & (g0 < ceiling)
    if g0sat is not None and np.shape(g0sat) == g0.shape:
        clean_raw &= ~np.asarray(g0sat, bool)
    good = (~sat) & np.isfinite(data0) & fin & (data0 > 0) & (g0 > 2000.0) & (g0 < ceiling)
    notes = []
    curve = None
    if int(good.sum()) >= 50:
        ctr, med = S._rcurve_bins(g0[good], data0[good] / g0[good], 2000.0, ceiling)
        nraw = len(ctr)
        raw = (list(ctr), list(med))
        if len(ctr) >= 2 and sw['rcurve_guard']:
            kept = 1
            for k in range(1, len(med)):
                a_, b_ = med[k - 1], med[k]
                if not (a_ > 0 and b_ > 0 and max(a_ / b_, b_ / a_) <= sw['rcurve_maxstep']):
                    break
                kept += 1
            ctr, med = ctr[:kept], med[:kept]
        notes.append(f'measured {nraw} bins, kept {len(ctr)} after guard')
        if len(ctr) >= 2:
            curve = (np.array(ctr), np.array(med))
        elif len(ctr) == 1:
            curve = (np.array([ctr[0], ctr[0] * 1.01]), np.array([med[0], med[0]]))
    else:
        raw = ([], [])
        notes.append('fewer than 50 calibration pixels')
    if curve is not None and sw['rcurve_satcheck']:
        dqi = dq.astype(np.int64)
        satcal = (sat & clean_raw & (g0 > 2000.0) & np.isfinite(data0) & (data0 > 0)
                  & ((dqi & (S.dqflags.pixel['DO_NOT_USE'] | S._RIM_BADPIX_BITS)) == 0))
        nsat = int(satcal.sum())
        if nsat >= S._RCURVE_SATCHECK_MIN_PX:
            gsx, rsx = g0[satcal], data0[satcal] / g0[satcal]
            r_sat, g_sat = float(np.median(rsx)), float(np.median(gsx))
            r_cur = float(np.interp(np.log(g_sat), np.log(curve[0]), curve[1]))
            fac = r_cur / r_sat
            notes.append(f'satcheck {nsat} px: curve/sat = {fac:.3f}')
            if max(fac, 1 / fac) > S._RCURVE_SATCHECK_MAX_RATIO:
                sc, sm = S._rcurve_bins(gsx, rsx, 2000.0, ceiling)
                curve = (np.array(sc), np.array(sm)) if len(sc) >= 2 else (np.array([g_sat, g_sat * 1.01]), np.array([r_sat, r_sat]))
                notes.append('curve rebuilt from SATURATED pixels')
        else:
            notes.append(f'satcheck skipped ({nsat} px)')
    return curve, ceiling, sat, notes, raw


def frame_pieces4(fn, P, lw):
    F3 = R3.frame_rewrite(fn, P, lw)
    info = F3['info']
    with fits.open(fn, memmap=False) as fh:
        dq = S.correct_dq_first_group_saturation(fh['DQ'].data, fn, fh[0].header.get('INSTRUME', ''))
        cal = np.array(fh['SCI'].data, float)
        vp = np.asarray(fh['VAR_POISSON'].data, float)
    data0 = cal.copy()
    data0[np.isnan(vp)] = 0
    rf = S._find_ramp_for(fn)
    with fits.open(rf, memmap=True) as r:
        g0 = np.array(r['SCI'].data[0, 0], float)
    pcurve, ceiling, sat, notes, praw = pipeline_curve(fn, data0, g0, dq)
    dnu = (dq & 1) != 0
    c, m = info['ctr'], info['med']
    gain, rn0, s_flat = info['gain'], info['rn0'], info['s_flat']
    lo, gmax = info['lo_edge'], info['gmax']
    with np.errstate(invalid='ignore', divide='ignore'):
        lg = np.log(np.clip(g0, 1, None))
        Rh = np.interp(lg, np.log(c), m)           # flat above/below the end bins
        if pcurve is not None:
            Rpipe = np.interp(lg, np.log(pcurve[0]), pcurve[1])
        else:
            Rpipe = Rh
        Rp = np.where(g0 < 2000.0, Rh, Rpipe)
        base_ok = np.isfinite(g0) & ~sat & ~dnu & (g0 >= lo) & (g0 <= ceiling) & (g0 > R3.SNR_MIN * info['sig_low'])
        sd_dn = np.sqrt(rn0 ** 2 + np.clip(g0, 0, None) / gain)

        def errf(Rv):
            return np.sqrt((Rv * sd_dn) ** 2 + (s_flat * Rv * g0) ** 2)
        out = dict(edt=F3['edt'], g0=g0, Rh=Rh, Rp=Rp, val_h=Rh * g0, val_p=Rp * g0, err_h=errf(Rh), err_p=errf(Rp),
                   valid=base_ok & np.isfinite(Rh), new=base_ok & (g0 > gmax), info=info, sat=sat, ceiling=ceiling)
    # validate the reproduced curve against the pipeline's rewrite at rim pixels
    rim = P['rim'] & sat & np.isfinite(g0) & (g0 > 2000.0) & (g0 < ceiling)
    if rim.any() and pcurve is not None:
        ratio = P['data'][rim] / (Rpipe[rim] * g0[rim])
        notes.append(f'rim check ({int(rim.sum())} px): rewritten / (my curve x g0) median {np.median(ratio):.4f}, p16-84 {np.percentile(ratio, 16):.4f}-{np.percentile(ratio, 84):.4f}')
    out['pnotes'] = notes
    out['pcurve'] = pcurve
    out['praw'] = praw
    return out


def run_frame4(fn, grid, lw, max_rows=None):
    base = fn.replace('.fits', '')
    cat = Table.read(base + '_resbgsub_m7_satstar_catalog.fits')
    P = C.prep_frame(fn)
    F = frame_pieces4(fn, P, lw)
    info = F['info']
    C.log('pieces', 'ceiling', F['ceiling'], 'gmax', info['gmax'], F['pnotes'])
    shape = P['data'].shape
    n = len(cat) if not max_rows else min(len(cat), max_rows)
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
    vlist = [('rw12h_e1', 'h', 1), ('rw12h_e0', 'h', 0), ('rw12p_e1', 'p', 1)]
    for i in range(n):
        r = cat[i]
        y0, y1, x0, x1 = wins[i]
        xfit, yfit = float(r['x_fit']), float(r['y_fit'])
        out = dict(idx=i, label=labels[i], a_cat=float(r['flux_fit_precap']), a_raw=float(r['flux_fit_raw']))
        if labels[i] > 0:
            cut_seq = seq[y0:y1, x0:x1].copy()
            st0 = C.make_setup(P, wins[i], r, cut_seq, labels[i])
            psf = S.psf_in_cutout_coords(grid, x0, y0)
            yy, xx = np.mgrid[0:y1 - y0, 0:x1 - x0]
            psf_unit = np.maximum(psf.evaluate(xx, yy, 1.0, xfit, yfit), 0.0)
            reg = C.fit_region(st0, psf_unit.shape)
            fitpx = reg & ~st0.mask & np.isfinite(st0.cut)
            out['nfit'] = int(fitpx.sum())
            out['sat_area'] = int(r['sat_area'])
            out['a_base'] = C.solve(st0, psf_unit, xfit, yfit, {}, kpois=P['kpois'])
            dat = P['data'][y0:y1, x0:x1]
            sl = (slice(y0, y1), slice(x0, x1))
            cand = (P['rim'][sl] == 0) & np.isfinite(dat) & (dat != 0) & ~st0.mask & (F['edt'][sl] <= D)
            sel_r3 = cand & F['valid'][sl] & ~F['new'][sl]   # round-3 range (g0 <= top bin)
            sel_h = cand & F['valid'][sl]
            newpx = sel_h & F['new'][sl] & fitpx
            out['nrw_rw12'] = int((sel_r3 & fitpx).sum())
            out['nnew'] = int(newpx.sum())
            if out['nnew']:
                qn = dat[newpx] / (F['Rh'][sl][newpx] * F['g0'][sl][newpx])
                out['q_new_med'] = float(np.median(qn))
                out['q_new_sum'] = float(qn.sum())
            else:
                out['q_new_med'] = np.nan
                out['q_new_sum'] = 0.0
            # q of the round-3 range pixels in the same fit region, for comparison
            m3 = sel_r3 & fitpx
            if m3.any():
                out['q_old_med'] = float(np.median(dat[m3] / (F['Rh'][sl][m3] * F['g0'][sl][m3])))
            else:
                out['q_old_med'] = np.nan
            for nm, kind, errmode in vlist:
                if kind == 'h':
                    sel = sel_h
                    val, errw = F['val_h'][sl], F['err_h'][sl]
                else:
                    sel = cand & F['valid'][sl]
                    val, errw = F['val_p'][sl], F['err_p'][sl]
                cutv = cut_seq.copy()
                cutv[sel] += (val - dat)[sel]
                errv = P['err'][sl].copy()
                if errmode == 1:
                    errv[sel] = errw[sel]
                P2 = dict(P)
                P2['err'] = R3.WinArr(errv)
                st = C.make_setup(P2, wins[i], r, cutv, labels[i])
                out['nrw_' + nm] = int((sel & fitpx).sum())
                out['a_' + nm] = C.solve(st, psf_unit, xfit, yfit, {}, kpois=P['kpois'])
                out['a_' + nm + '+bgfree'] = C.solve(st, psf_unit, xfit, yfit, {'bg': 'free'}, kpois=P['kpois'])
        rows.append(out)
        seq[y0:y1, x0:x1] -= models[i]
        if i % 100 == 0:
            C.log('row', i, n)
    tab = Table(rows=rows)
    tab.meta['CEIL'] = float(F['ceiling'])
    tab.meta['GMAX'] = float(info['gmax'])
    sk = cat['skycoord_fit'][:n]
    sk = sk if isinstance(sk, SkyCoord) else SkyCoord(sk)
    tab['ra'] = sk.ra.deg
    tab['dec'] = sk.dec.deg
    return tab, F


if __name__ == '__main__':
    band, det, e = sys.argv[1], sys.argv[2], int(sys.argv[3])
    max_rows = int(sys.argv[4]) if len(sys.argv) > 4 else None
    lw = band in ('250M', '300M')
    vgroup = {'150W': '10101', '200W': '12101', '250M': '04101', '300M': '12101'}[band]
    tree = C.Q + '/tree_main2'
    stem = f'jw03523005001_{vgroup}_{e:05d}_{det}_align_o005_crf'
    fn = f'{tree}/F{band}/pipeline/{stem}.fits'
    tag = '_test' if max_rows else ''
    outfn = f'{C.Q}/satrefit/out4/{band}_{stem}_satrefit4{tag}.fits'
    os.makedirs(os.path.dirname(outfn), exist_ok=True)
    hdr = C.fits.getheader(fn)
    grid, gf = C.load_grid(tree + '/psfs', hdr, lw)
    t, F = run_frame4(fn, grid, lw, max_rows=max_rows)
    t.write(outfn, overwrite=True)
    info = F['info']
    with open(outfn.replace('_satrefit4', '_pcurve').replace('.fits', '.txt'), 'w') as f:
        f.write('# wingmig curve: g0_ctr R\n')
        for a in zip(info['ctr'], info['med']):
            f.write('W %.2f %.5f\n' % a)
        if F['pcurve'] is not None:
            f.write('# pipeline curve (kept bins): g0_ctr R\n')
            for a in zip(*F['pcurve']):
                f.write('P %.2f %.5f\n' % (a[0], a[1]))
        f.write('# pipeline raw bins (before guard): g0_ctr R\n')
        for a in zip(*F['praw']):
            f.write('PR %.2f %.5f\n' % (a[0], a[1]))
        f.write('# ceiling=%.1f gmax=%.1f\n' % (F['ceiling'], info['gmax']))
        for nn in F['pnotes']:
            f.write('# ' + nn + '\n')
    C.log('wrote', outfn)
