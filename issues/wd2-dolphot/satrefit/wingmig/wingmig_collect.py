"""Collect per-pixel q = cal/(R(g0)*g0) for satstar wings and control stars."""
import sys, os, numpy as np, pickle
from astropy.io import fits
from scipy import ndimage as ndi
PIPE = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/tree_main2/%s/pipeline/'
SAT, DNU = 2, 1
def find_ramp(fn):
    import re
    stem = re.match(r'(jw\d+_\d+_\d+_[a-z0-9]+)', os.path.basename(fn)).group(1)
    d = os.path.dirname(fn)
    for rf in [f'{d}/{stem}_ramp.fits', f'{d}/pipeline/{stem}_ramp.fits']:
        if os.path.exists(rf): return rf
def rcurve(g, r, lo=200., nb=12, minpx=50):
    hi = np.nanpercentile(g, 99.9)
    e = np.geomspace(lo, hi, nb + 1)
    c, m, n = [], [], []
    for k in range(nb):
        s = (g >= e[k]) & (g < e[k+1])
        if s.sum() >= minpx:
            c.append(np.sqrt(e[k]*e[k+1])); m.append(np.median(r[s])); n.append(int(s.sum()))
    return np.array(c), np.array(m), np.array(n)
def applyR(c, m, g):
    out = np.full(g.shape, np.nan)
    ok = (g >= c[0]) & (g <= c[-1])
    out[ok] = np.interp(np.log(g[ok]), np.log(c), m)
    return out
def run(band, vg, exp, det):
    stem = f'jw03523005{vg}_{exp:05d}_{det}'
    fn = PIPE % band + f'jw03523005001_{vg}_{exp:05d}_{det}_align_o005_crf.fits'
    fn = PIPE % band + f'jw03523005001_{vg}_{exp:05d}_{det}_align_o005_crf.fits'
    rf = find_ramp(fn)
    with fits.open(fn, memmap=False) as h:
        cal = np.array(h['SCI'].data, float); dq = np.array(h['DQ'].data)
    with fits.open(rf, memmap=True) as r:
        hd = r[0].header
        meta = dict(readpatt=hd['READPATT'], nframes=hd['NFRAMES'], ngroups=hd['NGROUPS'], tgroup=hd['TGROUP'])
        g0 = np.array(r['SCI'].data[0, 0], float)
        zf = np.array(r['ZEROFRAME'].data[0], float)
    ok_corr = np.isfinite(cal) & np.isfinite(g0)
    corr = np.corrcoef(cal[ok_corr][::50], g0[ok_corr][::50])[0, 1]
    meta['corr'] = corr; meta['shape'] = cal.shape == g0.shape
    sat = (dq & SAT) != 0
    dnu = (dq & DNU) != 0
    edt, inds = ndi.distance_transform_edt(~sat, return_indices=True)
    lab, nl = ndi.label(sat, structure=np.ones((3, 3)))
    good = np.isfinite(cal) & np.isfinite(g0) & ~sat & ~dnu & (g0 > 200) & (edt >= 25)
    gg = g0[good]; rr = cal[good]/gg
    cF, mF, nF = rcurve(gg, rr)
    zgood = good & np.isfinite(zf) & (zf > 200)
    cZ, mZ, nZ = rcurve(zf[zgood], cal[zgood]/zf[zgood])
    # control stars
    mx = ndi.maximum_filter(g0, size=7)
    pk = (g0 == mx) & (edt > 25) & np.isfinite(g0) & ~dnu & (g0 > 1500)
    py, px = np.nonzero(pk)
    # radius map around control peaks (2-8 px)
    yy, xx = np.mgrid[-9:10, -9:10]; rad = np.hypot(yy, xx)
    ctl = []
    ctlmask = np.zeros(cal.shape, bool)
    for y, x in zip(py, px):
        if y < 10 or x < 10 or y > 2037 or x > 2037: continue
        sl = (slice(y-9, y+10), slice(x-9, x+10))
        c_ = cal[sl]; g_ = g0[sl]; z_ = zf[sl]
        m_ = (rad >= 1) & (rad <= 9) & ~sat[sl] & ~dnu[sl] & np.isfinite(c_) & np.isfinite(g_) & (g_ > 200)
        rid = len(ctl)
        ctl.append((g0[y, x], c_[m_], g_[m_], z_[m_], rad[m_]))
        ctlmask[sl] |= (rad >= 2) & (rad <= 8)
    # control R curve restricted to 2-8 px of peaks
    cm = ctlmask & good
    cC, mC, nC = rcurve(g0[cm], cal[cm]/g0[cm], minpx=30)
    # satstars
    cat = fits.getdata(fn.replace('.fits', '_resbgsub_m7_satstar_catalog.fits'))
    sats = []
    for row in cat:
        x, y = row['x_0'], row['y_0']
        if not (np.isfinite(x) and np.isfinite(y)) or not (0 <= x < 2048 and 0 <= y < 2048): continue
        xi, yi = int(round(x)), int(round(y))
        # component nearest the catalog position
        iy, ix = inds[0][yi, xi], inds[1][yi, xi]
        L = lab[iy, ix]
        if L == 0 or np.hypot(iy-y, ix-x) > 6: continue
        y0, y1, x0, x1 = max(0, yi-40), min(2048, yi+41), max(0, xi-40), min(2048, xi+41)
        sl = (slice(y0, y1), slice(x0, x1))
        near = (inds_lab := lab[inds[0][sl], inds[1][sl]]) == L   # nearest sat pixel belongs to this star
        m = near & (edt[sl] > 0) & (edt[sl] <= 20) & ~sat[sl] & ~dnu[sl] & np.isfinite(cal[sl]) & np.isfinite(g0[sl]) & (g0[sl] > 200)
        # clean: no other sat component within 25px of these pixels is guaranteed by nearest-sat assignment only
        sats.append(dict(flux=row['flux_fit'], fprecap=row['flux_fit_precap'], sat_area=row['sat_area'],
                         npix_sat=int((lab == L).sum()),
                         d=edt[sl][m], cal=cal[sl][m], g0=g0[sl][m], zf=zf[sl][m]))
    out = dict(meta=meta, R=(cF, mF, nF), Rz=(cZ, mZ, nZ), Rc=(cC, mC, nC), ctl=ctl, sats=sats)
    return out
if __name__ == '__main__':
    band, vg, exp, det = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    o = run(band, vg, exp, det)
    pickle.dump(o, open(f'raw_{band}_{exp}_{det}.pkl', 'wb'))
    print(band, exp, det, o['meta'], 'nsat', len(o['sats']), 'nctl', len(o['ctl']))
    print(' R', np.round(o['R'][0]), np.round(o['R'][1], 5), o['R'][2])
    print(' Rz', np.round(o['Rz'][1], 5)); print(' Rc', np.round(o['Rc'][0]), np.round(o['Rc'][1], 5))
