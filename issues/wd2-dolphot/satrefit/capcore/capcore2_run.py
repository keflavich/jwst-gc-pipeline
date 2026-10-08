"""Modified recovered-core cap: ZEROFRAME-rewritten pixels (flagged group 0, value = k x first frame) with first frame > thr x S_ff are treated as
NOT measured for the cap only.  usage: python capcore2_run.py BAND DET [EXP...]  -> out2/<band>_<stem>_cap2.fits
Variant tags: a<thr>_gm (excluded pixels join the unrecoverable set incl. the lost-fraction gate), a<thr>_go (gate fraction from the original set),
b0.6_gm/go (S_ff = max first frame), B<thr>_gm (also unflagged group-0 rim pixels with g0/g0sat99 > thr), 1.0_gm (nothing excluded)."""
import os
import sys
import numpy as np
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit')
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/capcore')
import satrefit_core as C  # noqa: E402
from satrefit_core import S, fits, Table, SkyCoord, ndimage  # noqa: E402
from capcore_run import labels_of  # noqa: E402

OUT = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/capcore/out2'
THR = [0.5, 0.6, 0.7, 0.8]


def variants():
    v = []
    for t in THR:
        v.append((f'a{t}_gm', 'a', t, 'm'))
        v.append((f'a{t}_go', 'a', t, 'o'))
    for t in THR + [0.9]:
        v.append((f'b{t}_gm', 'b', t, 'm'))
        v.append((f'b{t}_go', 'b', t, 'o'))
    v += [('a1.0_gm', 'a', 1.0, 'm')]
    v += [(f'B{t}_gm', 'a', t, 'm', 'B') for t in THR]
    return v


def cap_of(cutz, reg, un_cap, un_gate, pu):
    """Pipeline cap: recovered_core_peak gate (on un_gate) then recovered_cap_flux (on un_cap)."""
    nreg = int(reg.sum())
    lost = int((reg & un_gate).sum()) / nreg if nreg else 1.0
    rec = reg & ~un_gate & np.isfinite(cutz) & (cutz > 0)
    if lost >= 0.2 or int(rec.sum()) < 3:
        return np.nan
    c, _ = S.recovered_cap_flux(cutz, reg, un_cap, pu, min_psf_frac=0.005)
    return c


def run(fn, lw, grid, maxrows=None):
    base = fn.replace('.fits', '')
    cat = Table.read(base + '_resbgsub_m7_satstar_catalog.fits')
    P = C.prep_frame(fn)
    shape = P['data'].shape
    fh = fits.open(fn, memmap=False)
    unrec = np.isnan(np.asarray(fh['VAR_POISSON'].data, float))
    fh.close()
    sw = S.satstar_fit_switches()
    zf = S._find_zeroframe_for(fn)
    ff = S._find_first_frame_for(fn) if (zf is not None and sw['first_frame']) else None
    g0s = S._find_group0_saturation_for(fn, do_not_use=(ff is not None))
    ffp = ff[np.isfinite(ff) & (ff > 0)]
    Sa = float(np.percentile(ffp, 99.9))
    Sb = float(ffp.max())
    satpix = P['saturated'] & np.isfinite(zf) & (zf > 0)
    g0sat99 = float(np.percentile(zf[satpix], 99.0))
    rim = P['rim']
    flagged = rim & g0s & np.isfinite(ff) & (ff > 0)
    direct = rim & ~g0s & np.isfinite(zf) & (zf > 0)
    n = len(cat)
    if maxrows:
        n = min(n, maxrows)
    lab = labels_of(P, cat)
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
    V = variants()
    cols = {'cap0': [], 'label': [], 'nflag_reg': [], 'nreg': []}
    for v in V:
        cols['cap_' + v[0]] = []
    seq = P['data'].copy()
    for i in range(n):
        r = cat[i]
        y0, y1, x0, x1 = wins[i]
        row = {k: np.nan for k in cols}
        row['label'] = lab[i]
        if lab[i] > 0:
            st = C.make_setup(P, wins[i], r, seq[y0:y1, x0:x1].copy(), lab[i])
            cutz = st.cut
            pu = psfu[i]
            eff_buf = S.compute_adaptive_mask_buffer(int(r['sat_area']), mask_buffer_min=2)
            satwin = (P['zf_deep'] if P['zf_deep'] is not None else P['saturated'])[y0:y1, x0:x1]
            comp = P['sources'][y0:y1, x0:x1] == lab[i]
            this = comp & satwin
            reg = S.recovered_cap_region(comp, this, ndimage.binary_dilation(this, iterations=eff_buf))
            un = unrec[y0:y1, x0:x1]
            row['cap0'] = cap_of(cutz, reg, un, un, pu)
            fl, di = flagged[y0:y1, x0:x1], direct[y0:y1, x0:x1]
            ffw, zfw = ff[y0:y1, x0:x1], zf[y0:y1, x0:x1]
            row['nreg'] = int(reg.sum())
            row['nflag_reg'] = int((reg & fl).sum())
            for v in V:
                tag, kind, t, gate = v[:4]
                ref = Sa if kind == 'a' else Sb
                ex = fl & (ffw > t * ref) if t < 1.0 else np.zeros_like(fl)
                if len(v) > 4:
                    ex = ex | (di & (zfw > t * g0sat99))
                un_new = un | ex
                row['cap_' + tag] = cap_of(cutz, reg, un_new, un_new if gate == 'm' else un, pu)
        for k in cols:
            cols[k].append(row[k])
        seq[y0:y1, x0:x1] -= models[i]
        if i % 100 == 0:
            C.log('row', i, n)
    t = Table({k: np.asarray(v, dtype=float) for k, v in cols.items()})
    for k in ('flux_fit_precap', 'flux_fit_raw', 'flux_fit'):
        t['c_' + k] = np.asarray(cat[k][:n], float)
    sk = cat['skycoord_fit'][:n]
    sk = sk if isinstance(sk, SkyCoord) else SkyCoord(sk)
    t['ra'] = sk.ra.deg
    t['dec'] = sk.dec.deg
    t.meta.update(Sa=Sa, Sb=Sb, g0sat99=g0sat99)
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
            outfn = f'{OUT}/{band}_{stem}_cap2{"_test" if maxrows else ""}.fits'
            if os.path.exists(outfn):
                continue
            if grid is None:
                grid, gf = C.load_grid(tree + '/psfs', C.fits.getheader(fn), lw)
            t = run(fn, lw, grid, maxrows)
            t.write(outfn, overwrite=True)
            C.log('wrote', outfn)
