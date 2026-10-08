"""Recovered-core cap vs core-fit amplitude, per satstar catalog row.
usage: python capcore_run.py BAND DET[,DET] [EXP ...]   (env CAPCORE_MAXROWS limits rows, for tests)
Writes capcore/out/<band>_<stem>_capcore.fits.  Reuses satrefit_core (read only) for the frame reconstruction; the cap is
reproduced from saturated_star_finding.recovered_core_peak / recovered_cap_flux / recovered_cap_region (read-only import).
"""
import os
import sys
import numpy as np
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit')
import satrefit_core as C  # noqa: E402
from satrefit_core import S, fits, Table, SkyCoord, ndimage  # noqa: E402

OUT = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/capcore/out'
MAXLOST = 0.2
MINPSF = 0.005


def labels_of(P, cat):
    ww = P['ww']
    lab = np.full(len(cat), -1, int)
    for i in range(len(cat)):
        r = cat[i]
        if int(r['sat_area']) <= 0 or not np.isfinite(float(r['sat_com_ra'])):
            continue
        cx, cy = ww.world_to_pixel(SkyCoord(float(r['sat_com_ra']), float(r['sat_com_dec']), unit='deg'))
        d = np.hypot(P['comcen'][:, 0] - float(cx), P['comcen'][:, 1] - float(cy))
        j = int(np.nanargmin(d))
        if d[j] < 1.0:
            lab[i] = j + 1
    return lab


def run(fn, band, lw, grid, maxrows=None):
    base = fn.replace('.fits', '')
    cat = Table.read(base + '_resbgsub_m7_satstar_catalog.fits')
    rej = Table.read(base + '_resbgsub_m7_satstar_rejected.fits')
    P = C.prep_frame(fn)
    shape = P['data'].shape
    fh = fits.open(fn, memmap=False)
    unrec = np.isnan(np.asarray(fh['VAR_POISSON'].data, float))
    sci_raw = np.asarray(fh['SCI'].data, float)
    photmjsr = float(fh['SCI'].header.get('PHOTMJSR', fh[0].header.get('PHOTMJSR')))
    hdr0 = fh[0].header
    fh.close()
    sw = S.satstar_fit_switches()
    zf = S._find_zeroframe_for(fn)
    ff = S._find_first_frame_for(fn) if (zf is not None and sw['first_frame']) else None
    g0s = None
    if zf is not None and (sw['g0_groupdq'] or ff is not None):
        g0s = S._find_group0_saturation_for(fn, do_not_use=(ff is not None))
    good = np.isfinite(P['data']) & ~P['saturated'] & ~P['rim']
    Lsat = float(np.percentile(P['data'][good], 99.999))
    satpix = P['saturated'] & np.isfinite(zf) & (zf > 0)
    g0_sat99 = float(np.percentile(zf[satpix], 99.0)) if satpix.sum() > 10 else np.nan
    R_hdr = S.zeroframe_header_R(hdr0, photmjsr)
    n = len(cat)
    if maxrows:
        n = min(n, maxrows)
    lab = labels_of(P, cat)
    labr = labels_of(P, rej)
    # seed (com) positions in full-frame (x, y) of all rows (catalog + rejected) by label, for the cap's own-cell split
    seeds = {}
    for tab, lb in ((cat, lab), (rej, labr)):
        for k in range(len(tab)):
            if lb[k] > 0:
                x0g = int(round(float(tab['xcentroid'][k]) - float(tab['x_fit'][k])))
                y0g = int(round(float(tab['ycentroid'][k]) - float(tab['y_fit'][k])))
                seeds.setdefault(int(lb[k]), []).append((float(tab['x_init'][k]) + x0g, float(tab['y_init'][k]) + y0g))
    wins, psfu, models = [], [], []
    for i in range(n):
        r = cat[i]
        win = C.window_of(r, shape)
        y0, y1, x0, x1 = win
        psf = S.psf_in_cutout_coords(grid, x0, y0)
        yy, xx = np.mgrid[0:y1 - y0, 0:x1 - x0]
        pu = np.clip(psf.evaluate(xx, yy, 1.0, float(r['x_fit']), float(r['y_fit'])), 0, None)
        wins.append(win)
        psfu.append(pu)
        models.append(float(r['flux_fit_raw']) * pu)
        if i % 100 == 0:
            C.log('render', i, n)
    seq = P['data'].copy()
    rcore = 2.0 if lw else 2.5
    cols = {k: [] for k in ('idx', 'label', 'cap', 'pfrac', 'drec', 'lostf', 'nrec', 'ncap_reg', 'a_core_c', 'a_core_f', 'n_core', 'cov_core',
                            'bkg', 'sigma', 'pk_val', 'pk_dx', 'pk_dy', 'pk_r', 'pk_psf_rel', 'pk_is_ipk', 'ipk_meas', 'ipk_val', 'ipk_model',
                            'pk_model', 'pk_g0', 'pk_ff', 'pk_g0sat', 'pk_rim', 'pk_crf', 'rim_ratio_med', 'rim_g0_med', 'n_rim_reg',
                            'n_g0sat_reg', 'n_unrec_core', 'ppk', 'a_core_c_cut', 'sibs')}
    for i in range(n):
        r = cat[i]
        y0, y1, x0, x1 = wins[i]
        xfit, yfit = float(r['x_fit']), float(r['y_fit'])
        row = {k: np.nan for k in cols}
        row['idx'] = i
        row['label'] = lab[i]
        if lab[i] > 0:
            cut = seq[y0:y1, x0:x1].copy()
            st = C.make_setup(P, wins[i], r, cut, lab[i])
            cutz = st.cut
            pu = psfu[i]
            # --- pipeline cap region (get_saturated_stars) ---
            eff_buf = S.compute_adaptive_mask_buffer(int(r['sat_area']), mask_buffer_min=2)
            satwin = (P['zf_deep'] if P['zf_deep'] is not None else P['saturated'])[y0:y1, x0:x1]
            comp = P['sources'][y0:y1, x0:x1] == lab[i]
            this = comp & satwin
            this_exp = ndimage.binary_dilation(this, iterations=eff_buf)
            own_x = float(r['x_init'])
            own_y = float(r['y_init'])
            sibs = [(sy - y0, sx - x0) for sx, sy in seeds.get(int(lab[i]), [])
                    if np.hypot(sx - x0 - own_x, sy - y0 - own_y) > 0.5]
            row['sibs'] = len(sibs)
            own_cell = S.nearest_seed_cell(this.shape, (own_y, own_x), sibs) if sibs else None
            reg = S.recovered_cap_region(comp, this, this_exp, own_cell=own_cell)
            un = unrec[y0:y1, x0:x1]
            drec, lostf, nrec = S.recovered_core_peak(cutz, reg, un, MAXLOST)
            row.update(drec=drec, lostf=lostf, nrec=nrec, ncap_reg=int(reg.sum()))
            cap, pfrac = np.nan, np.nan
            if np.isfinite(drec):
                cap, pfrac = S.recovered_cap_flux(cutz, reg, un, pu, min_psf_frac=MINPSF)
            row.update(cap=cap, pfrac=pfrac)
            # --- peak pixel diagnostics ---
            meas = reg & ~un & np.isfinite(cutz)
            rec = meas & (cutz > 0)
            ppk = float(np.nanmax(pu))
            row['ppk'] = ppk
            ipk = np.unravel_index(np.nanargmax(pu), pu.shape)
            row['ipk_meas'] = bool(meas[ipk])
            row['ipk_val'] = float(cutz[ipk])
            row['ipk_model'] = float(r['flux_fit_precap']) * ppk
            row['n_rim_reg'] = int((reg & P['rim'][y0:y1, x0:x1]).sum())
            if g0s is not None:
                row['n_g0sat_reg'] = int((reg & g0s[y0:y1, x0:x1]).sum())
            if rec.any():
                vals = np.where(rec, cutz, -np.inf)
                py, px = np.unravel_index(np.argmax(vals), vals.shape)
                gy, gx = py + y0, px + x0
                row.update(pk_val=float(cutz[py, px]), pk_dx=px - xfit, pk_dy=py - yfit, pk_r=float(np.hypot(px - xfit, py - yfit)),
                           pk_psf_rel=float(pu[py, px]) / ppk, pk_is_ipk=bool((py, px) == ipk),
                           pk_model=float(r['flux_fit_precap']) * float(pu[py, px]),
                           pk_g0=float(zf[gy, gx]) if zf is not None else np.nan,
                           pk_ff=float(ff[gy, gx]) if ff is not None else np.nan,
                           pk_g0sat=bool(g0s[gy, gx]) if g0s is not None else False,
                           pk_rim=bool(P['rim'][gy, gx]), pk_crf=float(sci_raw[gy, gx]))
                rr = rec & P['rim'][y0:y1, x0:x1] & np.isfinite(zf[y0:y1, x0:x1]) & (zf[y0:y1, x0:x1] > 0)
                if rr.any():
                    row['rim_ratio_med'] = float(np.median(cutz[rr] / zf[y0:y1, x0:x1][rr]))
                    row['rim_g0_med'] = float(np.median(zf[y0:y1, x0:x1][rr]))
            # --- core fit: measured/recovered pixels within rcore of (x_fit, y_fit) ---
            yy, xx = np.indices(cutz.shape)
            rad = np.hypot(xx - xfit, yy - yfit)
            inr = rad <= rcore
            use = inr & ~un & ~satwin & (cutz != 0) & np.isfinite(cutz) & (st.errc < 1e9)
            sigma = st.sigma
            err = st.errc.copy()
            if np.isfinite(sigma) and sigma > 0:
                err = np.sqrt(err ** 2 + sigma ** 2)
            row['bkg'] = st.bkg
            row['sigma'] = sigma
            row['n_unrec_core'] = int((inr & (un | satwin)).sum())
            nuse = int(use.sum())
            row['n_core'] = nuse
            if nuse >= 1:
                row['cov_core'] = float(pu[use].sum() / pu[inr].sum())
            if nuse >= 5:
                w = 1.0 / err[use] ** 2
                p = pu[use]
                d = cutz[use]
                row['a_core_c'] = float(np.sum(w * (d - st.bkg) * p) / np.sum(w * p * p))
                M = np.stack([p, np.ones_like(p)], axis=1)
                A = M.T @ (M * w[:, None])
                b = M.T @ (w * d)
                try:
                    row['a_core_f'] = float(np.linalg.solve(A, b)[0])
                except np.linalg.LinAlgError:
                    pass
        for k in cols:
            cols[k].append(row[k])
        seq[y0:y1, x0:x1] -= models[i]
        if i % 100 == 0:
            C.log('row', i, n)
    t = Table({k: np.asarray(v, dtype=float) for k, v in cols.items()})
    for k in ('pk_is_ipk', 'ipk_meas', 'pk_g0sat', 'pk_rim'):
        t[k] = np.nan_to_num(t[k], nan=-1).astype(int)
    for k in ('flux_fit_precap', 'flux_fit_raw', 'flux_fit', 'cap_psf_frac', 'local_bkg', 'x_fit', 'y_fit', 'sat_area', 'wingcal_ratio'):
        t['c_' + k] = np.asarray(cat[k][:n], float)
    sk = cat['skycoord_fit'][:n]
    sk = sk if isinstance(sk, SkyCoord) else SkyCoord(sk)
    t['ra'] = sk.ra.deg
    t['dec'] = sk.dec.deg
    t.meta.update(Lsat=Lsat, g0sat99=g0_sat99, photmjsr=photmjsr, rcore=rcore, Rhdr=float(R_hdr) if R_hdr else -1.0,
                  Zmax=float(np.nanmax(zf)) if zf is not None else -1.0, FFMAX=float(np.nanmax(ff)) if ff is not None else -1.0)
    return t


if __name__ == '__main__':
    band = sys.argv[1]
    dets = sys.argv[2].split(',')
    exps = [int(x) for x in sys.argv[3:]] or [1, 2, 3, 4]
    lw = band in ('250M', '300M')
    vgroup = {'150W': '10101', '200W': '12101', '250M': '04101', '300M': '12101'}[band]
    maxrows = int(os.environ.get('CAPCORE_MAXROWS', 0)) or None
    tree = C.Q + '/tree_main2'
    os.makedirs(OUT, exist_ok=True)
    for det in dets:
        grid = None
        for e in exps:
            stem = f'jw03523005001_{vgroup}_{e:05d}_{det}_align_o005_crf'
            fn = f'{tree}/F{band}/pipeline/{stem}.fits'
            outfn = f'{OUT}/{band}_{stem}_capcore{"_test" if maxrows else ""}.fits'
            if os.path.exists(outfn):
                print('exists', outfn, flush=True)
                continue
            if grid is None:
                grid, gf = C.load_grid(tree + '/psfs', C.fits.getheader(fn), lw)
                C.log('grid', gf)
            t = run(fn, band, lw, grid, maxrows)
            t.write(outfn, overwrite=True)
            C.log('wrote', outfn)
