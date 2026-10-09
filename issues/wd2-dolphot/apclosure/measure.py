"""Per-band aperture/PSF closure measurement. usage: python measure.py BAND  -> tab_<band>.ecsv"""
import sys, glob, os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
import astropy.units as u
from scipy.spatial import cKDTree
from photutils.aperture import CircularAperture
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an

band = sys.argv[1]
NMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
OUT = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/apclosure'
RADII = [3, 5, 8]
RIN, ROUT = 12, 18
an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2')
P = np.load(f'{an.Q}/photomver/areapred_main2.npz')[band]
ref = A.ref[band]
ra, dec = np.asarray(A.m['RA'], float), np.asarray(A.m['DEC'], float)
lo, hi = an.ZPWIN[band]
ok = A.matched & np.isfinite(P) & np.isfinite(A.dm(band)) & ~A.rep[band] & ~A.sat[band] & (ref >= lo) & (ref < hi) & np.isfinite(ra)
# isolation using dolphot positions and mags
has = np.isfinite(ra) & np.isfinite(ref)
sky = SkyCoord(ra * u.deg, dec * u.deg)
ra0, dec0 = np.nanmedian(ra), np.nanmedian(dec)
X = (ra - ra0) * np.cos(np.deg2rad(dec0)) * 3600; Y = (dec - dec0) * 3600
hid = np.nonzero(has)[0]
tree = cKDTree(np.c_[X[hid], Y[hid]])
iso = np.zeros(A.n, bool)
for i in np.nonzero(ok)[0]:
    nb = hid[tree.query_ball_point([X[i], Y[i]], 1.0)]
    nb = nb[nb != i]
    if len(nb) == 0:
        iso[i] = True; continue
    d = np.hypot(X[nb] - X[i], Y[nb] - Y[i])
    iso[i] = not (np.any(d < 0.5) or np.any(ref[nb] < ref[i] + 3))
sel = np.nonzero(ok & iso)[0]
nsel0 = len(sel)
rng = np.random.default_rng(1)
if len(sel) > NMAX:
    sel = np.sort(rng.choice(sel, NMAX, replace=False))
print(band, 'ok', ok.sum(), 'isolated', nsel0, 'used', len(sel), flush=True)
psky = SkyCoord(A.cat[f'skycoord_f{band.lower()}'])[np.where(A.idx >= 0, A.idx, 0)]  # our band position (dolphot RA/DEC is offset ~1.4 px SW)
tree_dir = f'{an.Q}/tree_main2/F{band}'
cats = {}
for fn in glob.glob(f'{tree_dir}/f{band.lower()}_*_daophot_basic.fits'):
    t = Table.read(fn)
    cats[os.path.basename(t.meta['FILENAME'])] = t
frames = sorted(glob.glob(f'{tree_dir}/pipeline/jw03523005001_*_align_o005_crf.fits'))
rows = []
HW = ROUT + 3
yy, xx = np.mgrid[-HW:HW + 1, -HW:HW + 1]


def bkg(img, good, cx, cy):
    r = np.hypot(xx - cx, yy - cy)
    m = (r >= RIN) & (r <= ROUT) & good
    if m.sum() < 50:
        return np.nan
    return sigma_clipped_stats(img[m], sigma=3, maxiters=5)[1]


for fn in frames:
    with fits.open(fn) as fh:
        sci = np.asarray(fh['SCI'].data, float); area = np.asarray(fh['AREA'].data, float); dq = np.asarray(fh['DQ'].data)
        h = fh['SCI'].header
    w = WCS(h)
    ny, nx = sci.shape
    x, y = w.world_to_pixel(psky[sel])
    cat = cats.get(os.path.basename(fn))
    if cat is not None:
        ct = cKDTree(np.c_[cat['x_fit'], cat['y_fit']])
        cx_, cy_, cf_ = np.asarray(cat['x_fit'], float), np.asarray(cat['y_fit'], float), np.asarray(cat['flux_fit'], float)
    bad = ((dq & 1) > 0) | ~np.isfinite(sci) | ~np.isfinite(area)
    sa = sci * area
    for k, i in enumerate(sel):
        if not (np.isfinite(x[k]) and HW + 1 < x[k] < nx - HW - 1 and HW + 1 < y[k] < ny - HW - 1):
            continue
        px, py, src, pf = x[k], y[k], 0, np.nan
        if cat is not None:
            d, j = ct.query([x[k], y[k]])
            if d < 0.5:
                px, py, src, pf = cx_[j], cy_[j], 1, cf_[j]
        ix, iy = int(round(px)), int(round(py))
        sl = (slice(iy - HW, iy + HW + 1), slice(ix - HW, ix + HW + 1))
        b = bad[sl]
        cxl, cyl = px - ix, py - iy
        rr = np.hypot(xx - cxl, yy - cyl)
        if np.any(b & (rr <= RADII[-1])):
            continue
        g = ~b
        rec = dict(i=i, frame=os.path.basename(fn), det=h.get('DETECTOR', ''), src=src, psf=pf)
        okk = True
        for img, tag in ((sci, 'raw'), (sa, 'area')):
            c = np.where(g, img[sl], 0.0)
            bk = bkg(c, g, cxl, cyl)
            if not np.isfinite(bk):
                okk = False; break
            for r in RADII:
                ap = CircularAperture((cxl + HW, cyl + HW), r)
                m = ap.to_mask(method='exact').to_image(c.shape)
                rec[f'{tag}{r}'] = np.sum((c - bk) * m)  # bad pixels (c=0) lie outside the aperture
        if okk:
            rows.append(rec)
T = Table(rows)
T.write(f'{OUT}/frames_{band}.ecsv', overwrite=True)
print(band, 'frame rows', len(T), flush=True)
