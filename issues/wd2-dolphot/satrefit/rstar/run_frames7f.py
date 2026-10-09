"""Round 7 H / H+h0 refit with and without the flat field on the rewritten pixels.  usage: python run_frames7f.py BAND DET EXP
The cal frame is rate x PHOTMJSR / flat, while the group-0 rewrite R x g0 carries no flat.  Variants:
  H, H+h0      : as run_frames7.py (rim -> R_header x g0, h0 -> field-curve x g0)
  Hf, Hf+h0f   : the same rewritten values divided by the pixel flat (h0 also multiplied by median(flat), since the field
                 curve is a median of cal / g0 and so already carries the typical 1 / flat)
  Hfs<s>+h0f   : Hf+h0f with every rewritten value scaled by s (SCALES), for the sensitivity of dm to the group-0 rate scale
Writes out7f/<band>_<stem>_satrefit7f.fits and a pickle of the cap-region pixel values per row."""
import os
import sys
import pickle
import numpy as np
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit')
import satrefit_core as C
import run_frames3 as R3
import run_frames4 as R4
import run_frames5 as R5
from satrefit_core import S, fits, ndimage, Table, SkyCoord

OUT = C.Q + '/satrefit/rstar/out7f'
CRDS = '/orange/adamginsburg/jwst/crds/references/jwst/nircam/'
SCALES = (0.97, 1.03)


def run_refit(band, det, e):
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
    flat = np.array(fits.getdata(CRDS + FL['hdr']['R_FLAT'].split('//')[1], 'SCI'), float)
    flat[~np.isfinite(flat) | (flat <= 0)] = 1.0
    fmed = float(np.median(flat[4:-4, 4:-4]))
    Rh = float(FL['Rhdr'])
    cur0 = R5.pipeline_curve_N(fn, FL, 0)
    c0 = cur0['curve']
    F4 = R4.frame_pieces4(fn, P, lw)
    g0 = FL['g0']
    rim = P['rim']
    R0 = np.interp(np.log(np.clip(g0, 1, None)), np.log(c0[0]), c0[1])
    fixrim = rim & ~np.isfinite(rawerr)
    f = np.ones(g0.shape)
    sel = rim & np.isfinite(g0)
    f[sel] = Rh / R0[sel]
    fflat = f.copy()
    fflat[sel] = f[sel] / flat[sel]
    C.log('flat file', FL['hdr']['R_FLAT'], 'median', fmed, 'rim flat p16/50/84', np.percentile(flat[sel], [16, 50, 84]) if sel.any() else None)
    ev = {}
    for nm, ff in (('H', f), ('Hf', fflat)):
        e2 = P['err'].copy()
        e2[fixrim] = np.maximum(0.05 * np.abs((P['data'] * ff)[fixrim]), errfloor if sw['rim_badpix'] else 0.0)
        ev[nm] = e2
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
    regs = []
    for i in range(n):
        r = cat[i]
        y0, y1, x0, x1 = wins[i]
        xfit, yfit = float(r['x_fit']), float(r['y_fit'])
        out = dict(idx=i, label=labels[i], a_cat=float(r['flux_fit_precap']), a_raw=float(r['flux_fit_raw']))
        reg_out = None
        if labels[i] > 0:
            sl = (slice(y0, y1), slice(x0, x1))
            cut_seq = seq[sl].copy()
            dat = P['data'][sl]
            flc = flat[sl]
            st0 = C.make_setup(P, wins[i], r, cut_seq, labels[i])
            psf = S.psf_in_cutout_coords(grid, x0, y0)
            yy, xx = np.mgrid[0:y1 - y0, 0:x1 - x0]
            psf_unit = np.maximum(psf.evaluate(xx, yy, 1.0, xfit, yfit), 0.0)
            rimc = rim[sl]
            satwin = (P['zf_deep'] if P['zf_deep'] is not None else P['saturated'])[sl]
            this = (P['sources'][sl] == labels[i]) & satwin
            this_exp = ndimage.binary_dilation(this, iterations=st0.eff_buf)
            region = S.recovered_cap_region(P['sources'][sl] == labels[i], this, this_exp)
            ur = unrec[sl]

            def cap_of(cutv):
                pk, lost, nrec = S.recovered_core_peak(cutv, region, ur, 0.2)
                if not np.isfinite(pk):
                    return np.nan
                return S.recovered_cap_flux(cutv, region, ur, psf_unit, min_psf_frac=sw['cap_min_psf_frac'])[0]

            cand = (rimc == 0) & np.isfinite(dat) & (dat != 0) & ~st0.mask & (F4['edt'][sl] <= R4.D)
            sel_h = cand & F4['valid'][sl]
            val_h = F4['val_h'][sl]
            cuts = {}
            for nm, ff in (('H', f), ('Hf', fflat)):
                cv = cut_seq.copy()
                cv[rimc] = cut_seq[rimc] + (dat[rimc] * ff[sl][rimc] - dat[rimc])
                cuts[nm] = (cv, ev[nm][sl])
                cv2 = cv.copy()
                vh = val_h if nm == 'H' else val_h * fmed / flc
                cv2[sel_h] += (vh - dat)[sel_h]
                cuts[nm + ('+h0' if nm == 'H' else '+h0f')] = (cv2, ev[nm][sl])
            # every group-0-rewritten value (rim and h0) scaled by s, flat applied
            for sc in SCALES:
                cv = cut_seq.copy()
                cv[rimc] = cut_seq[rimc] + (dat[rimc] * fflat[sl][rimc] * sc - dat[rimc])
                cv[sel_h] += (val_h * fmed / flc * sc - dat)[sel_h]
                cuts[f'Hfs{sc:.2f}+h0f'] = (cv, ev['Hf'][sl])
            iy, ix = np.nonzero(region)
            ipk = np.unravel_index(np.nanargmax(psf_unit), psf_unit.shape)
            pkidx = -1
            if region[ipk]:
                pkidx = int(np.nonzero((iy == ipk[0]) & (ix == ipk[1]))[0][0])
            reg_out = dict(psf=psf_unit[iy, ix], ur=ur[iy, ix], pkidx=pkidx, ppk=float(np.nanmax(psf_unit)), flat=flc[iy, ix],
                           rim=rimc[iy, ix])
            for nm, (cv, e_) in cuts.items():
                P2 = dict(P)
                P2['err'] = R3.WinArr(e_)
                st = C.make_setup(P2, wins[i], r, cv, labels[i])
                out['a_' + nm] = C.solve(st, psf_unit, xfit, yfit, {}, kpois=P['kpois'])
                out['a_' + nm + '+bgfree'] = C.solve(st, psf_unit, xfit, yfit, {'bg': 'free'}, kpois=P['kpois'])
                out['cap_' + nm] = cap_of(cv)
                reg_out['cut_' + nm] = cv[iy, ix]
            fitpx = C.fit_region(st0, psf_unit.shape) & ~st0.mask & np.isfinite(st0.cut)
            w2 = psf_unit ** 2
            out['flat_rimw'] = float(np.sum((w2 * flc)[fitpx & rimc]) / np.sum(w2[fitpx & rimc])) if (fitpx & rimc).any() else np.nan
            out['flat_fitw'] = float(np.sum((w2 * flc)[fitpx]) / np.sum(w2[fitpx])) if fitpx.any() else np.nan
            out['wrim'] = float(np.sum(w2[fitpx & rimc]) / np.sum(w2[fitpx])) if fitpx.any() else np.nan
            out['nrim_fit'] = int((fitpx & rimc).sum())
            out['nrw_h'] = int((sel_h & fitpx).sum())
        rows.append(out)
        regs.append(reg_out)
        seq[y0:y1, x0:x1] -= models[i]
        if i % 100 == 0:
            C.log('row', i, n)
    tab = Table(rows=rows)
    sk = cat['skycoord_fit'][:n]
    sk = sk if isinstance(sk, SkyCoord) else SkyCoord(sk)
    tab['ra'] = sk.ra.deg
    tab['dec'] = sk.dec.deg
    tab.meta['RHDR'] = Rh
    tab.meta['FLATMED'] = fmed
    outfn = f'{OUT}/{band}_{stem}_satrefit7f.fits'
    tab.write(outfn, overwrite=True)
    pickle.dump(regs, open(outfn.replace('.fits', '_reg.pkl'), 'wb'))
    C.log('wrote', outfn)


if __name__ == '__main__':
    run_refit(sys.argv[1], sys.argv[2], int(sys.argv[3]))
