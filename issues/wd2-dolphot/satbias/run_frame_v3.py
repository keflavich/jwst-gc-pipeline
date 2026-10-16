"""usage: python run_frame_v3.py BAND STEM -> out/{BAND}_{STEM}_curve_v3.fits, out/{BAND}_{STEM}_unsat_v3.fits, _sat_v3.fits
1. per-pixel curves vs fill x = group0 / (t_g0/TFRAME) [first-frame-equivalent DN]:
     r_ff = (ZF/TFRAME)/(g0/t_g0),  g1/g0 and g2/g1 (normalised by the nominal time ratios),  g0/(rate*t_g0)  (all-groups-unflagged pixels)
2. ZF_corr = ZF / r_ff(ZF) pixel by pixel (curve from this frame, held at both ends), 5x5 PSF fits at the v1 positions."""
import sys
import time
import numpy as np
from zflib import *

EDGES = np.logspace(1.3, 4.7, 41)


def med(x):
    return float(np.median(x)) if len(x) else np.nan


def curves(fr, d):
    with fits.open(d + '_ramp.fits', memmap=False) as h:
        g1 = np.array(h['SCI'].data[0, 1], float)
        g2 = np.array(h['SCI'].data[0, 2], float)
        gd = np.array(h['GROUPDQ'].data[0])
    rate = fits.getdata(d + '_rate.fits').astype(float)
    nfr, tfr = fr.nframes, fr.tframe
    # group k averages frames k*(nfr+1)+1 .. +nfr (one dropped frame per group): centre (k*(nfr+1) + (nfr+1)/2) TFRAME
    tg = lambda k: tfr * (k * (nfr + 1) + (nfr + 1) / 2.0)
    t0, t1, t2 = tg(0), tg(1), tg(2)
    g1c = remove_streaks(np.where(((gd[1] & 3) == 0) & np.isfinite(g1), g1, np.nan))
    g2c = remove_streaks(np.where(((gd[2] & 3) == 0) & np.isfinite(g2), g2, np.nan))
    g0c = fr.g0_dn_clean
    x = g0c / (t0 / tfr)           # first-frame-equivalent DN from group 0
    z = fr.zf_dn_clean
    base = fr.g0valid & np.isfinite(z) & np.isfinite(x) & ((fr.dq & 1) == 0)
    ok1 = base & np.isfinite(g1c)
    ok2 = ok1 & np.isfinite(g2c)
    r12 = (g2c - g1c) / (t2 - t1)                      # DN/s from groups 1,2
    pred0 = g1c - r12 * (t1 - t0)                      # g0 expected from a straight line through g1,g2
    predz = g1c - r12 * (t1 - tfr)                     # first frame expected
    allok = base & np.all((gd & 3) == 0, axis=0) & np.isfinite(rate) & (rate > 0)
    rows = []
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        k = base & (x >= a) & (x < b)
        row = dict(lo=a, hi=b, n=int(k.sum()), x_med=med(x[k]), zf_med=med(z[k]), r_ff=np.nan, r_ff_pix=np.nan, g10=np.nan, n10=0, g21=np.nan,
                   n12=0, d0=np.nan, dz=np.nan, rt0=np.nan, slope=np.nan, nslope=0)
        if k.sum() >= 30:
            row['r_ff'] = row['zf_med'] / row['x_med']
            row['r_ff_pix'] = med(z[k] / x[k])
        k1 = ok1 & (x >= a) & (x < b)
        row['n10'] = int(k1.sum())
        if k1.sum() >= 30:
            row['g10'] = (med(g1c[k1]) / med(g0c[k1])) / (t1 / t0)
        k2 = ok2 & (x >= a) & (x < b)
        row['n12'] = int(k2.sum())
        if k2.sum() >= 30:
            row['g21'] = (med(g2c[k2]) / med(g1c[k2])) / (t2 / t1)
            row['d0'] = med(g0c[k2] - pred0[k2])
            row['dz'] = med(z[k2] - predz[k2])
            row['rt0'] = med(r12[k2] * t0)
        ks = allok & (x >= a) & (x < b)
        row['nslope'] = int(ks.sum())
        if ks.sum() >= 30:
            row['slope'] = med(g0c[ks]) / (med(rate[ks]) * t0)
        rows.append(row)
    return Table(rows=rows)


def make_corr(fr, ct):
    ok = np.isfinite(ct['r_ff']) & (ct['n'] >= 200)
    xz = np.array(ct['zf_med'])[ok]
    r = np.array(ct['r_ff'])[ok]
    o = np.argsort(xz)
    xz, r = xz[o], r[o]
    # monotone x only
    keep = np.concatenate([[True], np.diff(xz) > 0])
    xz, r = xz[keep], r[keep]
    zc = fr.zf_dn_clean
    rr = np.interp(zc, xz, r)           # held at both ends
    # alternative: above the last bin continue the log-linear trend of the last 4 bins, capped at 1.004 (docstring: 2.49/2.5)
    n4 = min(4, len(xz))
    sl_, ic_ = np.polyfit(np.log(xz[-n4:]), r[-n4:], 1)
    hi = zc > xz.max()
    rext = rr.copy()
    rext[hi] = np.clip(ic_ + sl_ * np.log(zc[hi]), r[-1], 1.004)
    return zc / rr, float(xz.max()), float(r[-1]), zc / rext


def run(band, stem):
    t0_ = time.time()
    d = f'{TREE}/F{band}/pipeline/{stem}'
    fr = Frame(band, stem, streak=True)
    ct = curves(fr, d)
    ct.write(f'out/{band}_{stem}_curve_v3.fits', overwrite=True)
    zcorr_dn, xmax, rmax, zext_dn = make_corr(fr, ct)
    zimg = zcorr_dn * fr.K / fr.tframe
    zimg_ext = zext_dn * fr.K / fr.tframe
    cat = Table.read(f'{TREE}/F{band}/pipeline/{stem}_align_o005_crf_resbgsub_m7_satstar_catalog.fits')
    grid, _ = C.load_grid(TREE + '/psfs', fr.hdr, band in LW)
    total, own = fr.neighbor_image(cat, grid)
    tu = Table.read(f'out/{band}_{stem}_unsat.fits')
    rows = []
    for r in tu:
        o = fit_star(zimg, grid, float(r['x']), float(r['y']), nb=total)
        rows.append(dict(zfe5=o['amp5'] if o else np.nan, zfe7=o['amp7'] if o else np.nan))
    Table(rows=rows).write(f'out/{band}_{stem}_unsat_v3.fits', overwrite=True)
    ts = Table.read(f'out/{band}_{stem}_sat.fits')
    rows = []
    for r in ts:
        i = int(r['i'])
        sl, m = own[i]
        total[sl] -= m
        o = fit_star(zimg, grid, float(r['x']), float(r['y']), nb=total)
        o2 = fit_star(zimg_ext, grid, float(r['x']), float(r['y']), nb=total)
        total[sl] += m
        xi, yi = int(round(float(r['x']))), int(round(float(r['y'])))
        w5 = (slice(yi - 2, yi + 3), slice(xi - 2, xi + 3))
        rows.append(dict(zfe5=o['amp5'] if o else np.nan, zfe7=o['amp7'] if o else np.nan, zfx5=o2['amp5'] if o2 else np.nan,
                         zpk_max=float(np.nanmax(fr.zf_dn_clean[w5])) if np.isfinite(fr.zf_dn_clean[w5]).any() else np.nan,
                         xmax_curve=xmax, r_at_max=rmax))
    Table(rows=rows).write(f'out/{band}_{stem}_sat_v3.fits', overwrite=True)
    print('done', time.time() - t0_, flush=True)


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2])
