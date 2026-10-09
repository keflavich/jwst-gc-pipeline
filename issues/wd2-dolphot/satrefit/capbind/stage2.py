"""capbind stage 2: LW wing migration q = cal / (R_h g0) versus distance d to the nearest DQ SATURATED pixel, satstars and unsaturated controls.
usage: python stage2.py BAND DET EXP   -> s2_<band>_<det>_<exp>.pkl  (per-unit rows: one unit = star x frame x d bin, median over its pixels)"""
import os
import pickle
import sys
import numpy as np
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit')
import satrefit_core as C
import run_frames3 as R3
import run_frames5 as R5
from satrefit_core import S, fits, ndimage, Table

OUT = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/capbind'
EDGES = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 8), (9, 12), (13, 20)]


def dbin_of(d):
    c = np.ceil(d - 1e-9).astype(int)
    out = np.full(c.shape, -1)
    for k, (lo, hi) in enumerate(EDGES):
        out[(c >= lo) & (c <= hi)] = k
    return out


def run(band, det, e):
    fn = R5.path_of(band, det, e)
    FL = R5.load_frame(fn)
    cal, g0, sat, dq = FL['cal'], FL['g0'], FL['sat'], FL['dq'].astype(np.int64)
    Rh = float(FL['Rhdr'])
    cur0 = R5.pipeline_curve_N(fn, FL, 0)
    ceiling = float(cur0['ceiling'])
    dnu = (dq & 1) != 0
    g0s = S._find_group0_saturation_for(fn, do_not_use=True)
    g0flag = g0s if g0s is not None else np.zeros(g0.shape, bool)
    edt, inds = ndimage.distance_transform_edt(~sat, return_indices=True)
    lab, nl = ndimage.label(sat, structure=np.ones((3, 3)))
    c, m, s_, n_ = R3.rcurve(g0[np.isfinite(cal) & np.isfinite(g0) & ~sat & ~dnu & (g0 > 200) & (edt >= 25) & (cal > 0) & ~g0flag],
                            (cal / g0)[np.isfinite(cal) & np.isfinite(g0) & ~sat & ~dnu & (g0 > 200) & (edt >= 25) & (cal > 0) & ~g0flag])

    def qf_of(g, cl):
        out = np.full(g.shape, np.nan)
        ok = (g >= c[0]) & (g <= c[-1])
        out[ok] = cl[ok] / (np.interp(np.log(g[ok]), np.log(c), m) * g[ok])
        return out

    base = (np.isfinite(cal) & np.isfinite(g0) & ~sat & ~dnu & (g0 > 200) & (g0 < ceiling) & ~g0flag)
    cat = Table.read(fn.replace('.fits', '') + '_resbgsub_m7_satstar_catalog.fits')
    units = []
    seen = {}
    for i, row in enumerate(cat):
        x, y = float(row['x_0']), float(row['y_0'])
        if not (np.isfinite(x) and np.isfinite(y)) or not (0 <= x < 2048 and 0 <= y < 2048):
            continue
        xi, yi = int(round(x)), int(round(y))
        iy, ix = inds[0][yi, xi], inds[1][yi, xi]
        L = lab[iy, ix]
        if L == 0 or np.hypot(iy - y, ix - x) > 6:
            continue
        fl = float(row['flux_fit_precap'])
        if L in seen and seen[L][1] >= fl:
            continue
        seen[L] = (i, fl)
    for L, (i, fl) in seen.items():
        row = cat[i]
        xi, yi = int(round(float(row['x_0']))), int(round(float(row['y_0'])))
        y0, y1, x0, x1 = max(0, yi - 40), min(2048, yi + 41), max(0, xi - 40), min(2048, xi + 41)
        sl = (slice(y0, y1), slice(x0, x1))
        near = lab[inds[0][sl], inds[1][sl]] == L
        db = dbin_of(edt[sl])
        gg, cc, dd = g0[sl], cal[sl], db
        q = cc / (Rh * gg)
        mk0 = near & base[sl] & (edt[sl] > 0) & (edt[sl] <= 20)
        mk = mk0 & (q > 0.5) & (q < 2.0)
        qf = qf_of(gg, cc)
        for k in range(len(EDGES)):
            mm = mk & (dd == k)
            if mm.sum() >= 3:
                units.append(dict(kind='sat', idx=i, nbad=int((mk0 & (dd == k)).sum() - mm.sum()), flux=fl, sat_area=int(row['sat_area']), npix_sat=int((lab == L).sum()), dbin=k, n=int(mm.sum()),
                                  q=float(np.median(q[mm])), qf=float(np.nanmedian(qf[mm])) if np.isfinite(qf[mm]).any() else np.nan, g0=float(np.median(gg[mm]))))
    # controls
    mx = ndimage.maximum_filter(g0, size=7)
    okpk = (g0 == mx) & (edt > 25) & np.isfinite(g0) & ~dnu & ~g0flag & (g0 > 1500) & (g0 < ceiling)
    py, px = np.nonzero(okpk)
    yy, xx = np.mgrid[-20:21, -20:21]
    rad = np.hypot(yy, xx)
    ncl = 0
    for y, x in zip(py, px):
        if y < 21 or x < 21 or y > 2026 or x > 2026:
            continue
        nb = np.array([g0[y - 1, x], g0[y + 1, x], g0[y, x - 1], g0[y, x + 1]])
        qn = cal[y - 1:y + 2, x - 1:x + 2] / (Rh * g0[y - 1:y + 2, x - 1:x + 2])
        if np.median(nb) < 0.3 * g0[y, x] or not (0.7 < np.nanmedian(qn) < 1.4):
            continue
        sl = (slice(y - 20, y + 21), slice(x - 20, x + 21))
        db = dbin_of(rad)
        gg, cc = g0[sl], cal[sl]
        q = cc / (Rh * gg)
        mk0 = base[sl] & (rad >= 1) & (rad <= 20) & (edt[sl] > 20)
        mk = mk0 & (q > 0.5) & (q < 2.0)
        qf = qf_of(gg, cc)
        for k in range(len(EDGES)):
            mm = mk & (db == k)
            if mm.sum() >= 3:
                units.append(dict(kind='ctl', idx=-1, nbad=int((mk0 & (db == k)).sum() - mm.sum()), flux=float(g0[y, x]), sat_area=0, npix_sat=0, dbin=k, n=int(mm.sum()), q=float(np.median(q[mm])),
                                  qf=float(np.nanmedian(qf[mm])) if np.isfinite(qf[mm]).any() else np.nan, g0=float(np.median(gg[mm]))))
        ncl += 1
    C.log('units', len(units), 'sat comps', len(seen), 'controls', ncl)
    pickle.dump(dict(meta=dict(Rh=Rh, ceiling=ceiling, band=band, det=det, exp=e, Rfield=(c, m)), units=units), open(f'{OUT}/s2_{band}_{det}_{e}.pkl', 'wb'))


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2], int(sys.argv[3]))
