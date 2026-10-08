"""Offline reproduction of the satstar masked-core amplitude fit (main2 arm) with fit-choice variants.

Reads the production pipeline code (read-only copy, commit 5434f5e7) to rebuild exactly what the
in-memory fit saw: ZEROFRAME-rewritten rim, zf deep-core mask, ERR with rim errors, sequential
neighbour subtraction.  Then solves the weighted linear amplitude at the catalog position.
"""
import os
import sys
import time
import numpy as np

os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
sys.dont_write_bytecode = True
# switches cataloging.py sets for the extended-emission NIRCam (wd2) run
os.environ['NIRCAM_SATSTAR_RECOVERED_CAP'] = '1'
os.environ['SATSTAR_ERR_BKG_SCATTER'] = '1'
os.environ['NIRCAM_SATSTAR_LOCK_POS'] = '1'
os.environ['NIRCAM_SATSTAR_LOCK_MIN_AREA_ARCSEC2'] = '0.5'
os.environ.pop('SATSTAR_ZF_KEEP_FINITE', None)

REPO = '/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-wd2main2'
sys.path.insert(0, REPO)
from astropy.io import fits  # noqa: E402
from astropy.table import Table  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402
from astropy.nddata.utils import overlap_slices  # noqa: E402
from astropy.stats import SigmaClip  # noqa: E402
from scipy import ndimage  # noqa: E402
from photutils.background import LocalBackground, MedianBackground  # noqa: E402
import jwst_gc_pipeline.reduction.saturated_star_finding as S  # noqa: E402

Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
PAD = 81
FIT_SHAPE = 81


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def prep_frame(fn):
    """Rebuild the in-memory arrays of get_saturated_stars for one crf frame."""
    fh = fits.open(fn, memmap=False)
    header = fh[0].header
    instr = header.get('INSTRUME', '')
    if 'DQ' in fh:
        fh['DQ'].data = S.correct_dq_first_group_saturation(fh['DQ'].data, fn, instr)
    sw = S.satstar_fit_switches()
    zf = S._find_zeroframe_for(fn) if S._zeroframe_fit_enabled() else None
    ff = S._find_first_frame_for(fn) if (zf is not None and sw['first_frame']) else None
    g0 = None
    if zf is not None and (sw['g0_groupdq'] or ff is not None):
        g0 = S._find_group0_saturation_for(fn, do_not_use=(ff is not None))
    data = fh['SCI'].data
    data[np.isnan(fh['VAR_POISSON'].data)] = 0
    photmjsr = fh['SCI'].header.get('PHOTMJSR', header.get('PHOTMJSR'))
    readout_floor = S.satstar_readout_data_floor(header, photmjsr)
    sat_floor = S._resolve_satstar_data_floor(header.get('FILTER', ''), explicit=None,
                                              readout_floor=readout_floor)
    sev_floor = S._resolve_satstar_severity_floor(header.get('FILTER', ''), explicit=None)
    saturated, sources, coms, kinds = S.find_saturated_stars(
        fh, min_sep_from_edge=5, edge_npix=10000, spike_merge_gap=0, spike_merge_ratio=3.0,
        sat_data_floor=sat_floor, severity_floor=sev_floor, partner_xy=None, sibling_xy=None)
    data, zf_deep, rim, _delta = S.zeroframe_fit_anchor(
        data, fh['DQ'].data, zf, group0_saturated=g0, first_frame=ff,
        R_header=S.zeroframe_header_R(header, photmjsr))
    err = np.array(fh['ERR'].data, dtype=float)
    if zf_deep is not None:
        err = S.zeroframe_rim_error(data, err, rim, floor=sw['rim_badpix'])
    err[~np.isfinite(err) | (err <= 0)] = 1e10
    vp = np.asarray(fh['VAR_POISSON'].data, float)
    sci = np.asarray(fh['SCI'].data, float)
    good = np.isfinite(vp) & np.isfinite(sci) & (sci > 50) & (vp > 0)
    kpois = float(np.median(vp[good] / sci[good])) if good.sum() > 100 else np.nan
    ww = S.frame_wcs(fh)
    pixscale = float(np.sqrt(fh['SCI'].header.get('PIXAR_A2', np.nan)))
    slices = ndimage.find_objects(sources)
    cen = np.full((len(slices), 2), np.nan)
    for i, sl in enumerate(slices):
        if sl is not None:
            cen[i] = (0.5 * (sl[1].start + sl[1].stop - 1), 0.5 * (sl[0].start + sl[0].stop - 1))
    P = dict(data=np.asarray(data, float), err=err, saturated=saturated, sources=sources,
             zf_deep=zf_deep, rim=rim, ww=ww, header=header, comcen=cen, kpois=kpois,
             pixscale=pixscale, nrim=int(rim.sum()), has_zf=zf is not None)
    fh.close()
    return P


def load_grid(psfdir, header, lw):
    det = header['DETECTOR'].lower()
    det = {'nrcalong': 'nrca5', 'nrcblong': 'nrcb5'}.get(det, det)
    filt = S.get_filtername(header).lower()
    fov = 1024 if lw else 512
    fn = f'{psfdir}/nircam_{det}_{filt}_fovp{fov}_samp2_npsf16.fits'
    g = S.to_griddedpsfmodel(fn)
    if isinstance(g, list):
        g = g[0]
    return g, fn


def window_of(row, shape):
    """Reconstruct the (y0,y1,x0,x1) fit cutout of a catalog row."""
    x0g = int(round(float(row['xcentroid']) - float(row['x_fit'])))
    y0g = int(round(float(row['ycentroid']) - float(row['y_fit'])))
    xc = int(round(float(row['x_init']) + x0g))
    yc = int(round(float(row['y_init']) + y0g))
    x0 = max(0, xc - PAD)
    y0 = max(0, yc - PAD)
    x1 = min(shape[1], xc + PAD)
    y1 = min(shape[0], yc + PAD)
    return y0, y1, x0, x1


def render(grid, win, xfit, yfit, flux, shape_cut=None):
    y0, y1, x0, x1 = win
    ny, nx = y1 - y0, x1 - x0
    yy, xx = np.mgrid[0:ny, 0:nx]
    psf = S.psf_in_cutout_coords(grid, x0, y0)
    return psf, np.maximum(psf.evaluate(xx, yy, flux, xfit, yfit), 0.0)


class Setup:
    """Per star, per neighbour mode: cutout, mask, background, scatter floor."""


def make_setup(P, win, row, cutnb, label):
    y0, y1, x0, x1 = win
    sat_area = int(row['sat_area'])
    eff_buf = S.compute_adaptive_mask_buffer(sat_area, mask_buffer_min=2)
    bin_, bout = S.compute_adaptive_bkg_annulus(sat_area)
    cut = cutnb.copy()
    cut[np.isnan(cut)] = 0.0
    errc = P['err'][y0:y1, x0:x1]
    satwin = (P['zf_deep'] if P['zf_deep'] is not None else P['saturated'])[y0:y1, x0:x1]
    this = (P['sources'][y0:y1, x0:x1] == label) & satwin
    this_exp = ndimage.binary_dilation(this, iterations=eff_buf)
    other = satwin & ~this
    satmask = this_exp | other
    mask = (cut == 0) | np.isnan(cut) | satmask
    x_init = float(row['x_init'])
    y_init = float(row['y_init'])
    lb = LocalBackground(bin_, bout)
    bkg = float(lb(cut, x_init, y_init, mask=mask))
    err_floor, sigma = S.bkg_scatter_fit_error(errc, cut, ~mask, x_init, y_init, bin_, bout)
    st = Setup()
    st.cut, st.errc, st.mask, st.satmask = cut, errc, mask, satmask
    st.bkg, st.sigma, st.err_floor = bkg, sigma, err_floor
    st.x_init, st.y_init, st.bkg_in, st.bkg_out = x_init, y_init, bin_, bout
    st.eff_buf = eff_buf
    st.dist = None
    return st


def fit_region(st, shape_cut):
    sl, _ = overlap_slices(shape_cut, (FIT_SHAPE, FIT_SHAPE), (st.y_init, st.x_init), mode='trim')
    reg = np.zeros(shape_cut, bool)
    reg[sl] = True
    return reg


def far_bkg(st, xfit, yfit, r_in, r_out):
    yy, xx = np.indices(st.cut.shape)
    rr = np.hypot(xx - xfit, yy - yfit)
    sel = (~st.mask) & np.isfinite(st.cut) & (rr >= r_in) & (rr < r_out)
    if sel.sum() < 50:
        return np.nan
    v = SigmaClip(sigma=3.0, maxiters=10)(st.cut[sel], masked=False)
    return float(np.median(v))


def solve(st, psf_unit, xfit, yfit, p, kpois=np.nan, a_pre=np.nan, base_dist=None):
    """Weighted linear amplitude.  p: dict of variant options.  Returns amplitude (nan on failure)."""
    shape_cut = st.cut.shape
    reg = fit_region(st, shape_cut)
    use = reg & ~st.mask & np.isfinite(st.cut)
    yy, xx = np.indices(shape_cut)
    rr = np.hypot(xx - xfit, yy - yfit)
    if p.get('rmax'):
        use &= rr < p['rmax']
    if p.get('guard') or p.get('wsig'):
        if st.dist is None:
            st.dist = ndimage.distance_transform_edt(~st.satmask)
    if p.get('guard'):
        use &= st.dist > p['guard']
    err = st.errc.copy()
    if p.get('wsig'):
        sig = float(p['wsig'])
        wprox = np.clip(1.0 - np.exp(-(st.dist ** 2) / (2.0 * sig ** 2)), 1e-3, 1.0)
        err = err / np.sqrt(wprox)
    sigma = st.sigma
    if np.isfinite(sigma) and sigma > 0:
        err = np.sqrt(err ** 2 + sigma ** 2)
    if p.get('uniform'):
        if np.isfinite(sigma) and sigma > 0:
            err = np.full(shape_cut, sigma)
        else:
            err = np.full(shape_cut, float(np.median(st.errc[use])))
    bmode = p.get('bg', 'ann')
    if bmode == 'far':
        bkg = far_bkg(st, xfit, yfit, *p['far'])
    elif bmode == 'none':
        bkg = 0.0
    else:
        bkg = st.bkg
    if not np.isfinite(bkg):
        return np.nan
    d = st.cut - (bkg if bmode != 'free' else 0.0)
    if int(use.sum()) < 10:
        return np.nan
    w = 1.0 / err[use] ** 2
    pu = psf_unit[use]
    du = d[use]
    if p.get('fill') and np.isfinite(a_pre) and np.isfinite(kpois):
        core = reg & st.satmask & ~use & np.isfinite(psf_unit)
        if p.get('rmax'):
            core &= rr < p['rmax']
        pc = psf_unit[core]
        dc = a_pre * pc
        ec2 = (sigma if np.isfinite(sigma) else 0.0) ** 2 + kpois * np.maximum(dc, 0.0)
        ec2 = np.maximum(ec2, 1e-12)
        pu = np.concatenate([pu, pc])
        du = np.concatenate([du, dc + (bkg if bmode != 'free' else 0.0) * 0.0])
        w = np.concatenate([w, 1.0 / ec2])
    if bmode == 'free':
        M = np.stack([pu, np.ones_like(pu)], axis=1)
        A = M.T @ (M * w[:, None])
        b = M.T @ (w * du)
        try:
            sol = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.nan
        return float(sol[0])
    num = np.sum(w * du * pu)
    den = np.sum(w * pu * pu)
    return float(num / den) if den > 0 else np.nan


def delta_psf(psf_unit, xfit, yfit, pixscale, dtab, rmax_arcsec, normalise):
    """P' = P (1 + Delta_unsat(r)); Delta interpolated in r (arcsec), zero beyond rmax_arcsec."""
    yy, xx = np.indices(psf_unit.shape)
    r = np.hypot(xx - xfit, yy - yfit) * pixscale
    rb, db = dtab
    keep = rb <= rmax_arcsec
    rb = np.concatenate([rb[keep], [rmax_arcsec]])
    db = np.concatenate([db[keep], [0.0]])
    dl = np.interp(r, rb, db, left=0.0, right=0.0)
    dl[r < rb[0]] = 0.0
    pp = psf_unit * (1.0 + dl)
    if normalise:
        tot, tot2 = float(np.sum(psf_unit)), float(np.sum(pp))
        if tot2 > 0:
            pp = pp * (tot / tot2)
    return pp


def run_frame(fn, band, variants, grid, dtab=None, rmax_arcsec=None, max_rows=None, out_nbr_check=None, wing_radii=None):
    """Return a Table with baseline and variant amplitudes for every catalog row of the frame."""
    base = fn.replace('.fits', '')
    cat = Table.read(base + '_resbgsub_m7_satstar_catalog.fits')
    t0 = time.time()
    P = prep_frame(fn)
    log('prep done', os.path.basename(fn), f'{time.time() - t0:.0f}s', 'rim px', P['nrim'], 'kpois', P['kpois'])
    shape = P['data'].shape
    n = len(cat)
    if max_rows:
        n = min(n, max_rows)
    ww = P['ww']
    # labels
    labels = np.full(n, -1, int)
    for i in range(n):
        r = cat[i]
        if int(r['sat_area']) <= 0 or not np.isfinite(float(r['sat_com_ra'])):
            continue
        c = SkyCoord(float(r['sat_com_ra']), float(r['sat_com_dec']), unit='deg')
        cx, cy = ww.world_to_pixel(c)
        d = np.hypot(P['comcen'][:, 0] - float(cx), P['comcen'][:, 1] - float(cy))
        j = int(np.nanargmin(d))
        if d[j] < 1.0:
            labels[i] = j + 1
    wins, models, psfs = [], [], []
    # own models at the final (capped, pre-wingcal) flux
    for i in range(n):
        r = cat[i]
        win = window_of(r, shape)
        wins.append(win)
        psf, m = render(grid, win, float(r['x_fit']), float(r['y_fit']), float(r['flux_fit_raw']))
        models.append(m)
    log('models rendered', n)
    total = np.zeros(shape)
    for w, m in zip(wins, models):
        total[w[0]:w[1], w[2]:w[3]] += m
    # check against the written model image
    chk = {}
    mimg_fn = base + '_resbgsub_m7_satstar_model.fits'
    if os.path.exists(mimg_fn):
        mim = fits.getdata(mimg_fn).astype(float)
        chk['model_sum_ratio'] = float(np.sum(total) / np.sum(mim))
        chk['model_maxabs_diff_rel'] = float(np.max(np.abs(total - mim)) / max(np.max(mim), 1e-30))
    log('model check', chk)
    wing = None
    if wing_radii is not None:
        wing = wing_emulation(P, base, grid, cat[:n], labels, wing_radii)
    seq = P['data'].copy()
    rows = []
    names = ['base'] + list(variants.keys())
    for i in range(n):
        r = cat[i]
        y0, y1, x0, x1 = wins[i]
        xfit, yfit = float(r['x_fit']), float(r['y_fit'])
        out = dict(idx=i, label=labels[i], a_cat=float(r['flux_fit_precap']), a_raw=float(r['flux_fit_raw']))
        if labels[i] > 0:
            own = models[i]
            cut_seq = seq[y0:y1, x0:x1].copy()
            cut_all = (P['data'][y0:y1, x0:x1] - (total[y0:y1, x0:x1] - own))
            cut_none = P['data'][y0:y1, x0:x1].copy()
            setups = {}
            for mode, cutnb in (('seq', cut_seq), ('all', cut_all), ('none', cut_none)):
                if mode != 'seq' and not any(v.get('nbr') == mode for v in variants.values()):
                    continue
                setups[mode] = make_setup(P, wins[i], r, cutnb, labels[i])
            psf = S.psf_in_cutout_coords(grid, x0, y0)
            yy, xx = np.mgrid[0:y1 - y0, 0:x1 - x0]
            psf_unit = np.maximum(psf.evaluate(xx, yy, 1.0, xfit, yfit), 0.0)
            allp = {'base': {}}
            allp.update(variants)
            out['nfit'] = int((fit_region(setups['seq'], psf_unit.shape) & ~setups['seq'].mask).sum())
            out['sat_area'] = int(r['sat_area'])
            out['sigma'] = setups['seq'].sigma
            out['bkg'] = setups['seq'].bkg
            out['nrim_win'] = int((P['rim'][y0:y1, x0:x1]).sum())
            for name in names:
                p = allp[name]
                st = setups[p.get('nbr', 'seq')]
                pu = psf_unit
                if p.get('psfcorr'):
                    pu = delta_psf(psf_unit, xfit, yfit, P['pixscale'], dtab, rmax_arcsec, p['psfcorr'] == 'a')
                out['a_' + name] = solve(st, pu, xfit, yfit, p, kpois=P['kpois'], a_pre=float(r['flux_fit_precap']))
                if p.get('psfcorr'):
                    out['pk_' + name] = float(psf_unit.max() / pu.max())
        else:
            for name in names:
                out['a_' + name] = np.nan
        rows.append(out)
        seq[y0:y1, x0:x1] -= models[i]
        if i % 100 == 0:
            log('row', i, n)
    tab = Table(rows=rows)
    tab.meta.update(chk)
    if wing is not None:
        for k, v in wing['cols'].items():
            tab[k] = v
        tab.meta['WC_L'] = wing['level']
        tab.meta['WC_CAL'] = wing['calstr']
        tab.meta['WC_NPK'] = wing['npk']
    tab['x_fit'] = np.asarray(cat['x_fit'][:n], float)
    tab['y_fit'] = np.asarray(cat['y_fit'][:n], float)
    sk = cat['skycoord_fit'][:n]
    sk = sk if isinstance(sk, SkyCoord) else SkyCoord(sk)
    tab['ra'] = sk.ra.deg
    tab['dec'] = sk.dec.deg
    return tab


def wing_emulation(P, base, grid, cat, labels, radii, level_pct=99.999):
    """Emulate apply_wing_selfcal on the in-memory data: calls the production _wing_selfcal (read-only import) with an explicit
    peak window 0.12 L .. 0.875 L, L = the frame's saturation level = level_pct percentile of finite, non-saturated, non-rim data.
    Returns per-row radii (a: wingcal_rmask, b: circular-equivalent DQ-SATURATED component radius before ZF recovery,
    c: remaining deep-core mask radius after ZF recovery) and C ratios (gated as in the pipeline, and ungated)."""
    from jwst_gc_pipeline.photometry.wingcal import (bucket_se, relative_scatter_floor, passes_se_gate,
                                                     interp_wingcal_ratio)
    d = P['data']
    sat = P['saturated']
    good = np.isfinite(d) & ~sat & ~P['rim']
    L = float(np.percentile(d[good], level_pct))
    mim = np.nan_to_num(fits.getdata(base + '_resbgsub_m7_satstar_model.fits').astype(float))
    data_sub = d - mim
    hdr = P['header']
    _, fwhm = S.get_fwhm(hdr, instrument_replacement='NIRCam')
    log('wing: level L', L, 'window', 0.12 * L, 0.875 * L, 'fwhm', fwhm)
    cal = S._wing_selfcal(data_sub, P['err'], sat, grid, list(radii), fwhm_pix=float(fwhm),
                          peak_lo=0.12 * L, peak_hi=0.875 * L)
    npk = max((v[1] for v in cal.values()), default=0)
    rs = np.array(sorted(cal))
    vs = np.array([cal[r][0] for r in rs])
    ns = np.array([cal[r][1] for r in rs])
    md = np.array([cal[r][2] for r in rs])
    gated_ok = bool(cal) and npk >= 8
    use = np.zeros(len(rs), bool)
    if gated_ok:
        se = bucket_se(md, ns, ratio=vs, rel_floor=relative_scatter_floor(rs, vs, md, ns))
        use = passes_se_gate(se, ratio=vs)
    n = len(cat)
    zd = P['zf_deep'] if P['zf_deep'] is not None else np.zeros_like(sat)
    src = P['sources']
    ra = np.asarray(cat['wingcal_rmask'], float)
    rb = np.full(n, np.nan)
    rc = np.full(n, np.nan)
    for i in range(n):
        if labels[i] > 0:
            comp = src == labels[i]
            rb[i] = np.sqrt((comp & sat).sum() / np.pi)
            rc[i] = np.sqrt((comp & zd).sum() / np.pi)
    cols = {'wc_pipe': np.asarray(cat['wingcal_ratio'], float) if 'wingcal_ratio' in cat.colnames else np.ones(n),
            'r_a': ra, 'r_b': rb, 'r_c': rc}
    for k, r in (('a', ra), ('b', rb), ('c', rc)):
        cols['C_' + k] = interp_wingcal_ratio(r, rs[use], vs[use]) if use.any() else np.ones(n)
        cols['Cu_' + k] = interp_wingcal_ratio(r, rs, vs) if len(rs) else np.ones(n)
    calstr = ';'.join(f'{int(r)}:{v:.4f}:{int(nn)}:{m:.4f}:{int(u)}' for r, v, nn, m, u in zip(rs, vs, ns, md, use))
    log('wing cal', calstr, 'gated_ok', gated_ok)
    return dict(cols=cols, level=L, calstr=calstr or 'none', npk=int(npk))
