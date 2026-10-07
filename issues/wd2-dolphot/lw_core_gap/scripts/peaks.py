"""Offline test of a peak-based hand-off for unaccepted SATURATED components.
For one F277W frame: local maxima of the finite SCI inside SATURATED components,
classified by (a) whether the component holds an accepted satstar centre,
(b) distance to the nearest dolphot F277W star, and coverage of the no-row stars.
usage: python peaks.py ARM BAND FRAME_INDEX"""
import glob, sys
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u
from scipy import ndimage
from scipy.spatial import cKDTree
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an

arm, b, fi = sys.argv[1], sys.argv[2], int(sys.argv[3])
FWHM = {'277W': 1.48, '250M': 1.33, '300M': 1.58}[b]
P = f'{an.Q}/tree_{arm}/F{b}/pipeline'
f = sorted(glob.glob(f'{P}/jw03523005001_*_nrcblong_align_o005_crf.fits'))[fi]
tr = Table.read(f'trace_{arm}_{b}.ecsv')
an.ZPWIN.update(an.zp_windows())
A = an.Arm(arm)
rs_all = SkyCoord(np.asarray(A.m['RA'], float) * u.deg, np.asarray(A.m['DEC'], float) * u.deg)
sci = fits.getdata(f, 'SCI').astype(float); dq = fits.getdata(f, 'DQ'); err = fits.getdata(f, 'ERR')
w = WCS(fits.getheader(f, 'SCI'))
sat = (dq & 2) != 0
dnu = (dq & 1) != 0
lab, n = ndimage.label(sat)
sizes = np.bincount(lab.ravel())
acc = Table.read(f.replace('.fits', '_resbgsub_m7_satstar_catalog.fits'))
ax, ay = np.asarray(acc['xcentroid'], float), np.asarray(acc['ycentroid'], float)
ok = np.isfinite(ax) & np.isfinite(ay)
ax, ay = ax[ok], ay[ok]
acc_lab = set(lab[np.clip(np.rint(ay).astype(int), 0, sat.shape[0] - 1), np.clip(np.rint(ax).astype(int), 0, sat.shape[1] - 1)].tolist()) - {0}
atree = cKDTree(np.column_stack([ax, ay]))
# dolphot stars with an F277W value, frame pixels
hasb = np.isfinite(A.ref[b])
xd, yd = w.world_to_pixel(rs_all[hasb])
dtree = cKDTree(np.column_stack([xd, yd]))
magd = A.ref[b][hasb]
# no-row stars
xn, yn = w.world_to_pixel(rs_all[np.asarray(tr['dolphot_idx'])])
inside = (xn > 4) & (xn < sat.shape[1] - 4) & (yn > 4) & (yn < sat.shape[0] - 4)
xn, yn = xn[inside], yn[inside]
labn = lab[np.rint(yn).astype(int), np.rint(xn).astype(int)]
print(f'{f.split("/")[-1]}: no-row stars on frame {inside.sum()}; on a SAT pixel {np.sum(labn > 0)}; '
      f'in a comp holding an accepted satstar centre {np.sum(np.isin(labn, list(acc_lab)))}')
dacc_n, _ = atree.query(np.column_stack([xn, yn]))
print(f'  no-row distance to nearest accepted satstar centre (px): p10/50/90 {np.percentile(dacc_n, [10, 50, 90]).round(1)}')
# local maxima
s = np.where(np.isfinite(sci), sci, -np.inf)
for size in (3, 5):
    mx = ndimage.maximum_filter(s, size=size, mode='constant', cval=-np.inf)
    pk = (s == mx) & np.isfinite(sci) & sat
    py, px = np.nonzero(pk)
    # local median background in a 9x9 box
    med = ndimage.median_filter(np.where(np.isfinite(sci), sci, 0), size=9)
    prom = (sci[py, px] - med[py, px]) / np.maximum(err[py, px], 1e-3)
    ratio = sci[py, px] / np.maximum(med[py, px], 1e-3)
    dacc, _ = atree.query(np.column_stack([px, py]))
    in_acc = np.isin(lab[py, px], list(acc_lab))
    dd, jd = dtree.query(np.column_stack([px, py]))
    big = sizes[lab[py, px]] > 60
    print(f'size={size}: {len(px)} peaks on SAT pixels; {in_acc.sum()} in accepted-holding comps; '
          f'{np.sum(dacc <= 1.5 * FWHM)} within 1.5 FWHM of accepted; {big.sum()} in comps > 60 px')
    for name, sel in (('all', np.ones(len(px), bool)), ('dacc>1.5FWHM', dacc > 1.5 * FWHM),
                      ('dacc>1.5FWHM & ratio>=1.3', (dacc > 1.5 * FWHM) & (ratio >= 1.3)),
                      ('dacc>1.5FWHM & ratio>=1.5', (dacc > 1.5 * FWHM) & (ratio >= 1.5))):
        hit = dd[sel] <= 1.0
        print(f'   {name}: {sel.sum()} peaks, dolphot F{b} star within 1 px: {hit.sum()} ({hit.mean():.2f}); '
              f'matched mag p10/50/90 {np.percentile(magd[jd[sel][hit]], [10, 50, 90]).round(1) if hit.any() else "-"}')
        ptree = cKDTree(np.column_stack([px[sel], py[sel]]))
        dn, _ = ptree.query(np.column_stack([xn, yn]))
        print(f'      no-row stars with such a peak within 1 px: {np.sum(dn <= 1.0)} of {len(xn)}')
