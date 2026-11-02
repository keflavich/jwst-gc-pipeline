"""Model-independent diagnostics used to locate the Poisson-limit floor (see README.md).

All star positions come from the GWCS (stdatamodels meta.wcs); no SIP WCS is used.
Each function prints its table; `python diagnostics.py <test>` runs one on the demo star.
"""
import sys, warnings
import numpy as np
from scipy import ndimage
from astropy.io import fits
# nan-heavy medians/ratios on masked pixels; other warnings (GWCS, I/O) stay visible
warnings.filterwarnings('ignore', category=RuntimeWarning)

RA, DEC = 266.5090306896523, -28.95658817641266         # the 10678 obs-061 target
CAL = '../data/jw10678061001_02101_{i:05d}_{det}_{suf}.fits'


def _dm(fn):
    import stdatamodels.jwst.datamodels as dm
    return dm.open(fn)


def differential_distortion(n=6, det='nrcblong'):
    """How far each dither's pixel grid departs from a pure translation of dither 1 around
    the target (px).  A detector-pixel PSF cannot be shared between dithers if this is >0.01."""
    W = [_dm(CAL.format(i=i, det=det, suf='cal')).meta.wcs for i in range(1, n+1)]
    x1, y1 = W[0].invert(RA, DEC)
    g = np.arange(-500, 501, 100.); gx, gy = np.meshgrid(g, g)
    ra, dec = W[0](x1+gx, y1+gy)
    for j in range(n):
        xt, yt = W[j].invert(RA, DEC); x, y = W[j].invert(ra, dec)
        dev = np.hypot((x-xt)-gx, (y-yt)-gy)
        print(f'e{j+1}: max departure from translation r<=200 {np.nanmax(dev[(abs(gx) <= 200) & (abs(gy) <= 200)]):.2f} px, r<=500 {np.nanmax(dev):.2f} px')


def dither_ratio(ref=1, others=(2, 3, 4, 5, 6), det='nrcblong', H=200):
    """e_j / e_ref after exact GWCS pixel->sky->pixel resampling of e_ref (cubic spline).
    Quadrant medians by radius: azimuthally uniform => flux-like change of the star's light."""
    M = {i: _dm(CAL.format(i=i, det=det, suf='cal')) for i in (ref,)+tuple(others)}
    d1 = M[ref].data.astype(float); bad = ~np.isfinite(d1) | ((M[ref].dq & 1) > 0)
    co = ndimage.spline_filter(np.where(bad, 0, d1), order=3)
    for j in others:
        xt, yt = M[j].meta.wcs.invert(RA, DEC)
        yy, xx = np.mgrid[int(yt)-H:int(yt)+H, int(xt)-H:int(xt)+H]
        yy = yy.clip(0, 2047); xx = xx.clip(0, 2047)
        ra, dec = M[j].meta.wcs(xx.astype(float), yy.astype(float)); x1, y1 = M[ref].meta.wcs.invert(ra, dec)
        v = ndimage.map_coordinates(co, [y1, x1], order=3, prefilter=False, mode='constant', cval=np.nan)
        b = ndimage.map_coordinates(bad.astype(float), [y1, x1], order=1, mode='constant', cval=1) > 0.01
        dj = M[j].data[yy, xx].astype(float); ok = ~b & ((M[j].dq[yy, xx] & 1) == 0) & np.isfinite(dj)
        ratio = np.where(ok, dj/v, np.nan); rr = np.hypot(xx-xt, yy-yt); th = np.arctan2(yy-yt, xx-xt)
        tab = [[np.nanmedian(ratio[ok & (rr >= lo) & (rr < hi) & (th >= -np.pi+q*np.pi/2) & (th < -np.pi/2+q*np.pi/2)])
                for q in range(4)] for lo, hi in [(30, 45), (45, 70), (70, 110), (110, 170)]]
        print(f'e{j}/e{ref} quadrant medians (rows r=30-45,45-70,70-110,110-170):\n', np.round(np.array(tab), 3))


def raw_ramp_wing(n=6, det='nrcblong'):
    """Median raw (_uncal) group differences in star-centred annuli: no pipeline step involved."""
    for i in range(1, n+1):
        m = _dm(CAL.format(i=i, det=det, suf='cal')); xt, yt = m.meta.wcs.invert(RA, DEC)
        u = fits.getdata(CAL.format(i=i, det=det, suf='uncal'), 'SCI').astype(float)
        yy, xx = np.mgrid[:u.shape[-2], :u.shape[-1]]; r = np.hypot(xx-xt, yy-yt)
        row = []
        for lo, hi in [(45, 70), (70, 110), (110, 170), (400, 600)]:
            mm = (r >= lo) & (r < hi)
            row.append(' '.join(f'{np.median((u[0, g+1]-u[0, g])[mm]):6.0f}' for g in range(u.shape[1]-1)))
        print(f'e{i}: DN/group  r45-70 [{row[0]}]  r70-110 [{row[1]}]  r110-170 [{row[2]}]  r400-600 [{row[3]}]')


def integration_consistency(i=1, det='nrcblong'):
    """int2/int1 wing ratio from _rateints: is anything changing *within* an exposure?"""
    m = _dm(CAL.format(i=i, det=det, suf='cal')); xt, yt = m.meta.wcs.invert(RA, DEC)
    h = fits.open(CAL.format(i=i, det=det, suf='rateints'))
    s = h['SCI'].data.astype(float); dq = h['DQ'].data
    yy, xx = np.mgrid[:s.shape[-2], :s.shape[-1]]; r = np.hypot(xx-xt, yy-yt)
    for lo, hi in [(30, 45), (45, 70), (70, 110), (110, 170), (400, 600)]:
        mm = (r >= lo) & (r < hi) & ((dq[0] & 1) == 0) & ((dq[1] & 1) == 0) & np.isfinite(s[0]) & np.isfinite(s[1])
        print(f'e{i} r {lo}-{hi}: int2/int1 = {np.median(s[1][mm]/s[0][mm]):.4f}')


def halo_index(files):
    """(between-spike - bkg)/(on-spike - bkg) at r=60-120 px for star-centred cutouts
    (cutout.py npz).  Spike angles measured on the first file."""
    def load(f):
        z = np.load(f); d = z['d']; yy, xx = np.mgrid[:d.shape[0], :d.shape[1]]
        return d, np.hypot(xx-z['xt'], yy-z['yt']), np.degrees(np.arctan2(yy-z['yt'], xx-z['xt'])) % 360, z
    d, r, th, z = load(files[0])
    m = (r > 70) & (r < 130) & np.isfinite(d)
    bins = np.arange(0, 360.5, 0.5); idx = np.digitize(th[m], bins)
    ang = np.array([np.nanmedian(d[m][idx == k]) for k in range(1, len(bins))])
    sm = ndimage.maximum_filter1d(ang, 21, mode='wrap')
    sp = bins[np.nonzero((ang == sm) & (ang > np.percentile(ang, 90)))[0]]+0.25
    for f in files:
        d, r, th, z = load(f)
        dth = np.min(np.abs(((th[..., None]-sp[None, None, :])+180) % 360-180), -1)
        ok = np.isfinite(d) & ((z['dq'] & 6) == 0)
        bk = np.nanmedian(d[ok & (r > 350) & (r < 450) & (dth > 10)])
        on = np.nanmedian(d[ok & (r > 60) & (r < 120) & (dth < 1.0)])-bk
        off = np.nanmedian(d[ok & (r > 60) & (r < 120) & (dth > 8)])-bk
        print(f'{f}: det=({z["X0"]+z["xt"]:6.0f},{z["Y0"]+z["yt"]:6.0f}) halo/spike={off/on:.4f}')


def residual_structure(tgt_npz, loo_npz):
    """LOO-residual autocorrelation (PSF-scale vs white) and int-free statistics."""
    sys.path.insert(0, '.')
    from exposure import Exposure
    ex = Exposure(tgt_npz, np.load(tgt_npz)['J'])
    z = np.load(loo_npz); chi = (ex.d-z['pred'])/np.sqrt(ex.var+z['varQ'])
    yy, xx = np.mgrid[:ex.n, :ex.n]; r = np.hypot(xx-ex.xt0, yy-ex.yt0)
    ok = ex.good & np.isfinite(chi) & ((ex.dq & 6) == 0)
    for lo, hi in [(60, 150), (300, 500)]:
        C = np.where(ok & (r >= lo) & (r < hi), chi, np.nan); out = []
        for dx, dy in [(1, 0), (0, 1), (2, 0), (3, 0), (5, 0)]:
            a = C[:ex.n-dy, :ex.n-dx]; b = C[dy:, dx:]; g = np.isfinite(a) & np.isfinite(b)
            out.append(f'({dx},{dy}):{np.corrcoef(np.clip(a[g], -10, 10), np.clip(b[g], -10, 10))[0, 1]:.2f}')
        print(f'r {lo}-{hi}:', ' '.join(out))


if __name__ == '__main__':
    t = sys.argv[1]
    {'distortion': differential_distortion, 'ratio': dither_ratio, 'raw': raw_ramp_wing,
     'ints': integration_consistency}[t]()
