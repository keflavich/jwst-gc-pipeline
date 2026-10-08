"""Round 3: charge-migration rewrite variant.  usage: python run_frames3.py BAND DET EXP [MAXROWS]
(writes out3/<band>_<stem>_satrefit3.fits and out3/<band>_<stem>_rcurve.txt)

Before the amplitude solve, non-saturated cutout pixels within D px of any DQ-SATURATED pixel are rewritten to R(g0)*g0
(g0 = ramp SCI[0,0]); the rewrite is applied as a delta on the neighbour-subtracted cutout, so the brighter-first model
subtraction is kept.  R(g0) = wingmig method (median of cal/g0 in 12 log bins from 200 DN to the g0 99.9th pct, over finite,
non-SATURATED, non-DO_NOT_USE pixels >= 25 px from any saturated pixel; bins with < 50 px dropped).  The pipeline's own curve in
zeroframe_recover_saturated starts at R_g0_min = 2000 DN (8 bins, step guard); this driver uses the 200 DN start of the q measurement.
"""
import os
import re
import sys
import numpy as np

sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit')
import satrefit_core as C
from satrefit_core import S, fits, ndimage, Table, SkyCoord

DLIST = [3, 6, 12, 20, np.inf]
GAIN_SW, GAIN_LW = 2.0, 1.8
SNR_MIN = 5.0


def dname(D):
    return 'inf' if not np.isfinite(D) else str(int(D))


def rcurve(g, r, lo=200., nb=12, minpx=50):
    hi = np.nanpercentile(g, 99.9)
    e = np.geomspace(lo, hi, nb + 1)
    c, m, s, n = [], [], [], []
    for k in range(nb):
        sel = (g >= e[k]) & (g < e[k + 1])
        if sel.sum() >= minpx:
            c.append(np.sqrt(e[k] * e[k + 1]))
            m.append(np.median(r[sel]))
            s.append(1.4826 * np.median(np.abs(r[sel] - np.median(r[sel]))) / np.median(r[sel]))
            n.append(int(sel.sum()))
    c, m, s, n = map(np.array, (c, m, s, n))
    # drop upper bins where R < 0.8 x the median (junk / hot-pixel bins)
    keep = m >= 0.8 * np.median(m)
    kk = len(m)
    for i in range(len(m)):
        if not keep[i] and i > len(m) // 2:
            kk = i
            break
    return c[:kk], m[:kk], s[:kk], n[:kk]


class WinArr:
    """Stand-in for the full-frame ERR array: any slice returns the prepared window."""
    def __init__(self, arr):
        self.arr = arr

    def __getitem__(self, key):
        return self.arr


def frame_rewrite(fn, P, lw):
    """Frame-level pieces: corrected DQ SATURATED mask, distance map, R(g0)*g0 rewrite values, errors, selection mask."""
    with fits.open(fn, memmap=False) as fh:
        dq = S.correct_dq_first_group_saturation(fh['DQ'].data, fn, fh[0].header.get('INSTRUME', ''))
        cal = np.array(fh['SCI'].data, float)
    rf = S._find_ramp_for(fn)
    with fits.open(rf, memmap=True) as r:
        g0 = np.array(r['SCI'].data[0, 0], float)
    sat = (dq & 2) != 0
    dnu = (dq & 1) != 0
    edt = ndimage.distance_transform_edt(~sat)
    good = np.isfinite(cal) & np.isfinite(g0) & ~sat & ~dnu & (g0 > 200) & (edt >= 25) & (cal > 0)
    c, m, s, n = rcurve(g0[good], cal[good] / g0[good])
    # group-0 read noise from the field: adjacent-pixel differences over the faintest 20 % of unsaturated pixels
    ok = np.isfinite(g0) & ~sat & ~dnu
    lowv = np.nanpercentile(g0[ok], [1, 20])
    sel = ok[:, 1:] & ok[:, :-1] & (g0[:, 1:] > lowv[0]) & (g0[:, 1:] < lowv[1]) & (g0[:, :-1] > lowv[0]) & (g0[:, :-1] < lowv[1])
    dif = (g0[:, 1:] - g0[:, :-1])[sel]
    sig_low = 1.4826 * np.median(np.abs(dif - np.median(dif))) / np.sqrt(2)
    gain = GAIN_LW if lw else GAIN_SW
    glow = float(np.median(g0[ok & (g0 > lowv[0]) & (g0 < lowv[1])]))
    rn2 = sig_low ** 2 - max(glow, 0) / gain
    rn0 = float(np.sqrt(rn2)) if rn2 > 0 else 0.0
    floor = max(SNR_MIN * sig_low, c[0])   # 5 x total field noise at the faint end; the curve's lowest bin centre
    lo_edge = max(SNR_MIN * sig_low, 200.0)
    # intrinsic R scatter: per-bin robust scatter minus expected noise (read + Poisson through g0)
    noise_frac = np.sqrt(rn0 ** 2 + c / gain) / c
    s_int = np.sqrt(np.maximum(s ** 2 - noise_frac ** 2, 0.0))
    brt = c >= 1000.
    s_flat = float(np.median(s_int[brt] if brt.any() else s_int))
    gmax = c[-1]
    with np.errstate(invalid='ignore', divide='ignore'):
        Rg = np.full(g0.shape, np.nan)
        inr = np.isfinite(g0) & (g0 >= lo_edge) & (g0 <= gmax)
        Rg[inr] = np.interp(np.log(g0[inr]), np.log(c), m)
        val = Rg * g0
        sd_dn = np.sqrt(rn0 ** 2 + np.clip(g0, 0, None) / gain)
        err = np.sqrt((Rg * sd_dn) ** 2 + (s_flat * val) ** 2)
    valid = inr & np.isfinite(val) & (g0 > SNR_MIN * sig_low) & ~sat & ~dnu
    info = dict(sig_low=sig_low, rn0=rn0, gain=gain, glow=glow, lo_edge=lo_edge, gmax=gmax, s_flat=s_flat, ctr=c, med=m, scat=s, n=n,
                s_int=s_int, npix_valid=int(valid.sum()))
    return dict(edt=edt, val=val, err=err, valid=valid, info=info, sat=sat)


def run_frame3(fn, grid, lw, max_rows=None):
    base = fn.replace('.fits', '')
    cat = Table.read(base + '_resbgsub_m7_satstar_catalog.fits')
    P = C.prep_frame(fn)
    F = frame_rewrite(fn, P, lw)
    info = F['info']
    C.log('rewrite pieces', {k: v for k, v in info.items() if k not in ('ctr', 'med', 'scat', 'n', 's_int')})
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
    vnames = ['base', 'bgfree'] + [f'rw{dname(D)}' for D in DLIST] + [f'rw{dname(D)}+bgfree' for D in DLIST]
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
            out['a_bgfree'] = C.solve(st0, psf_unit, xfit, yfit, {'bg': 'free'}, kpois=P['kpois'])
            dat = P['data'][y0:y1, x0:x1]
            cand = (F['valid'][y0:y1, x0:x1] & ~P['rim'][y0:y1, x0:x1] & np.isfinite(dat) & (dat != 0) & ~st0.mask
                    & np.isfinite(F['val'][y0:y1, x0:x1]))
            ed = F['edt'][y0:y1, x0:x1]
            delta_full = F['val'][y0:y1, x0:x1] - dat
            for D in DLIST:
                sel = cand & (ed <= D)
                cutv = cut_seq.copy()
                cutv[sel] += delta_full[sel]
                errv = P['err'][y0:y1, x0:x1].copy()
                errv[sel] = F['err'][y0:y1, x0:x1][sel]
                P2 = dict(P)
                P2['err'] = WinArr(errv)
                st = C.make_setup(P2, wins[i], r, cutv, labels[i])
                nm = f'rw{dname(D)}'
                out['nrw_' + nm] = int((sel & fitpx).sum())
                out['a_' + nm] = C.solve(st, psf_unit, xfit, yfit, {}, kpois=P['kpois'])
                out['a_' + nm + '+bgfree'] = C.solve(st, psf_unit, xfit, yfit, {'bg': 'free'}, kpois=P['kpois'])
        else:
            for nm in vnames:
                out['a_' + nm] = np.nan
        rows.append(out)
        seq[y0:y1, x0:x1] -= models[i]
        if i % 100 == 0:
            C.log('row', i, n)
    tab = Table(rows=rows)
    for k in ('sig_low', 'rn0', 'gain', 'glow', 'lo_edge', 'gmax', 's_flat'):
        tab.meta[k.upper()] = float(info[k])
    tab['x_fit'] = np.asarray(cat['x_fit'][:n], float)
    tab['y_fit'] = np.asarray(cat['y_fit'][:n], float)
    sk = cat['skycoord_fit'][:n]
    sk = sk if isinstance(sk, SkyCoord) else SkyCoord(sk)
    tab['ra'] = sk.ra.deg
    tab['dec'] = sk.dec.deg
    return tab, info


if __name__ == '__main__':
    band, det, e = sys.argv[1], sys.argv[2], int(sys.argv[3])
    max_rows = int(sys.argv[4]) if len(sys.argv) > 4 else None
    lw = band in ('250M', '277W', '300M', '323N', '335M', '405N', '410M', '444W', '466N')
    vgroup = {'150W': '10101', '200W': '12101', '250M': '04101', '300M': '12101'}[band]
    tree = C.Q + '/tree_main2'
    stem = f'jw03523005001_{vgroup}_{e:05d}_{det}_align_o005_crf'
    fn = f'{tree}/F{band}/pipeline/{stem}.fits'
    tag = '_test' if max_rows else ''
    outfn = f'{C.Q}/satrefit/out3/{band}_{stem}_satrefit3{tag}.fits'
    os.makedirs(os.path.dirname(outfn), exist_ok=True)
    hdr = C.fits.getheader(fn)
    grid, gf = C.load_grid(tree + '/psfs', hdr, lw)
    C.log('grid', gf)
    t, info = run_frame3(fn, grid, lw, max_rows=max_rows)
    t.write(outfn, overwrite=True)
    with open(outfn.replace('_satrefit3', '_rcurve').replace('.fits', '.txt'), 'w') as f:
        f.write('# g0_ctr  R  scatter_frac  npix  intrinsic_scatter\n')
        for a in zip(info['ctr'], info['med'], info['scat'], info['n'], info['s_int']):
            f.write('%.2f %.5f %.4f %d %.4f\n' % a)
        f.write('# sig_low=%.3f rn0=%.3f gain=%.2f lo_edge=%.1f gmax=%.0f s_flat=%.4f\n' % (
            info['sig_low'], info['rn0'], info['gain'], info['lo_edge'], info['gmax'], info['s_flat']))
    C.log('wrote', outfn)
