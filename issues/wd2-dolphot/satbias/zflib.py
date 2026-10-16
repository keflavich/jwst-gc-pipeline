"""ZEROFRAME flux anchor: shared routines.  Writes only into zfanchor/.

Conversion of the ramp ZEROFRAME (first frame, superbias-subtracted DN) to the crf's MJy/sr:
    ZF_MJy = ZF_DN * K / T_ZF,   K = cal/rate (= PHOTMJSR / flat, per pixel),  T_ZF = TFRAME (nominal) x tscale
tscale is a free calibration constant, measured on unsaturated stars (zf_unsat.py).
"""
import os
import sys
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy import ndimage

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ZFC = os.path.dirname(HERE) + '/zfcore'
sys.path.insert(0, ZFC)
import zf_core as C  # noqa: E402  (imports the read-only kfpr pipeline checkout)
S = C.S

TREE = C.TREE
LW = ('277W', '300M', '250M', '335M', '410M', '405N', '323N', '466N')
FRAMES = {'150W': ('10101', ['nrcb3', 'nrcb1']), '200W': ('12101', ['nrcb3', 'nrcb1']), '162M': ('14101', ['nrcb3', 'nrcb1']),
          '277W': ('10101', ['nrcblong', 'nrcalong']), '300M': ('12101', ['nrcblong', 'nrcalong']),
          '405N': (None, ['nrcblong', 'nrcalong'])}
ZF_RAW_SAT = 60000.0     # raw (uncal) DN threshold; ramp ZEROFRAME is 0 where jwst flagged it
H = 22                   # cutout half-size for local fits
BKG_IN, BKG_OUT = 9.0, 17.0


def stems(band):
    """frame stems of the benchmark (SW: nrcb1+nrcb3 of the main visit group; LW: all 8 frames), from the satstar catalogs"""
    import glob
    fs = sorted(glob.glob(f'{TREE}/F{band}/pipeline/jw03523005001_*_nrc*_align_o005_crf_resbgsub_m7_satstar_catalog.fits'))
    out = [os.path.basename(f).split('_align_')[0] for f in fs]
    if band not in LW:
        vis = FRAMES[band][0]
        out = [o for o in out if o.split('_')[1] == vis and o.split('_')[-1] in ('nrcb1', 'nrcb3')]
    return out


class Frame:
    def __init__(self, band, stem, streak=True):
        self.band, self.stem = band, stem
        d = f'{TREE}/F{band}/pipeline/{stem}'
        self.crf_fn = d + '_align_o005_crf.fits'
        with fits.open(self.crf_fn, memmap=False) as h:
            self.hdr = h[0].header.copy()
            self.crf = np.array(h['SCI'].data, float)
            self.dq = np.array(h['DQ'].data)
            self.photmjsr = h['SCI'].header['PHOTMJSR']
        with fits.open(d + '_ramp.fits', memmap=False) as h:
            zf = np.array(h['ZEROFRAME'].data[0], float)
            self.groupdq0 = np.array(h['GROUPDQ'].data[0, 0])
            self.g0 = np.array(h['SCI'].data[0, 0], float)
            self.rh = h[0].header.copy()
        with fits.open(d + '_uncal.fits', memmap=False) as h:
            raw = np.array(h['ZEROFRAME'].data[0], float)
        cal = fits.getdata(d + '_cal.fits')
        rate = fits.getdata(d + '_rate.fits')
        self.tframe = float(self.hdr['TFRAME'])
        self.nframes = int(self.hdr['NFRAMES'])
        self.ngroups = int(self.hdr['NGROUPS'])
        self.readpatt = self.hdr['READPATT']
        # K = cal/rate = PHOTMJSR/flat; where rate is unusable use a running median of K
        ok = np.isfinite(cal) & np.isfinite(rate) & (np.abs(rate) > 0.3)
        K = np.full(cal.shape, np.nan)
        K[ok] = cal[ok] / rate[ok]
        Kmed = np.nanmedian(K)
        K[~np.isfinite(K) | (np.abs(K / Kmed - 1) > 0.3)] = np.nan
        Kf = K
        if np.isnan(Kf).any():
            fill = ndimage.median_filter(np.where(np.isfinite(K), K, Kmed), size=15)
            Kf = np.where(np.isfinite(K), K, fill)
        self.K = Kf
        self.zf_dn = zf
        self.raw = raw
        self.zvalid = np.isfinite(zf) & (zf != 0) & (raw < ZF_RAW_SAT)
        z = np.where(self.zvalid, zf, np.nan)
        if streak:
            z = remove_streaks(z)
        self.zf_dn_clean = z
        self.zf = z * self.K / self.tframe   # MJy/sr at nominal TFRAME
        # group 0 (mean of NFRAMES frames, t = TFRAME (NFRAMES+1)/2), used as a second short-exposure reference at high counts
        g0ok = np.isfinite(self.g0) & (self.g0 > 0) & ((self.groupdq0 & (2 | 1)) == 0)
        self.g0valid = g0ok
        g = np.where(g0ok, self.g0, np.nan)
        if streak:
            g = remove_streaks(g)
        self.g0_dn_clean = g
        self.tg0 = self.tframe * (self.nframes + 1) / 2.0
        self.g0img = g * self.K / self.tg0

    def neighbor_image(self, cat, grid):
        """sum of satstar catalog models (flux_fit_raw) over the frame"""
        total = np.zeros(self.crf.shape)
        own = []
        for i in range(len(cat)):
            r = cat[i]
            x, y = float(r['xcentroid']), float(r['ycentroid'])
            x0, y0 = int(round(x)) - 40, int(round(y)) - 40
            sl, m = model_stamp(grid, x, y, 40, float(r['flux_fit_raw']), self.crf.shape)
            if sl is None:
                own.append(None)
                continue
            total[sl] += m
            own.append((sl, m))
        return total, own


def remove_streaks(z, nout=4):
    """subtract per-row, per-amplifier robust median (1/f row noise + amp pedestal) from a ZEROFRAME in DN"""
    out = z.copy()
    ny, nx = z.shape
    w = nx // nout
    for k in range(nout):
        blk = z[:, k * w:(k + 1) * w]
        med = np.nanmedian(blk, axis=1)
        mad = 1.4826 * np.nanmedian(np.abs(blk - med[:, None]), axis=1)
        for _ in range(2):
            keep = np.abs(blk - med[:, None]) < 3 * mad[:, None]
            b2 = np.where(keep, blk, np.nan)
            med = np.nanmedian(b2, axis=1)
        out[:, k * w:(k + 1) * w] = blk - med[:, None]
    return out


def model_stamp(grid, x, y, h, flux, shape):
    """(slice, model) of grid PSF at frame position (x, y), half-size h, clipped to the frame"""
    xi, yi = int(round(x)), int(round(y))
    x0, y0 = xi - h, yi - h
    x1, y1 = xi + h + 1, yi + h + 1
    cx0, cy0 = max(x0, 0), max(y0, 0)
    cx1, cy1 = min(x1, shape[1]), min(y1, shape[0])
    if cx1 <= cx0 or cy1 <= cy0:
        return None, None
    psf = S.psf_in_cutout_coords(grid, cx0, cy0)
    yy, xx = np.mgrid[0:cy1 - cy0, 0:cx1 - cx0]
    m = np.maximum(psf.evaluate(xx, yy, flux, x - cx0, y - cy0), 0.0)
    return (slice(cy0, cy1), slice(cx0, cx1)), m


def unit_psf(grid, x, y, h, shape):
    return model_stamp(grid, x, y, h, 1.0, shape)


def fit_star(img, grid, x, y, nb=None, valid=None, shapes=(5, 7, 9), refine=False, bkg_clip=3.0):
    """Core-led amplitude fit with fixed (x, y) (frame pixels) and local background.

    img: 2-D image (MJy/sr), NaN = unusable.  nb: optional neighbour-model image subtracted first.
    Returns dict: amp{shape}, n{shape}, bkg, sig (per-pixel rms of the annulus), center_ok, peak (max of img in the 3x3 core)
    """
    sl, m1 = unit_psf(grid, x, y, H, img.shape)
    if sl is None:
        return None
    cut = img[sl].copy()
    if nb is not None:
        cut = cut - nb[sl]
    if valid is not None:
        cut = np.where(valid[sl], cut, np.nan)
    yy, xx = np.mgrid[sl]
    r = np.hypot(xx - x, yy - y)
    ann = (r >= BKG_IN) & (r < BKG_OUT) & np.isfinite(cut)
    if ann.sum() < 40:
        return None
    amp = 0.0
    bkg = np.nan
    for _ in range(3):
        v = (cut - amp * m1)[ann]
        med = np.median(v)
        sg = 1.4826 * np.median(np.abs(v - med))
        k = np.abs(v - med) < bkg_clip * max(sg, 1e-9)
        bkg = np.median(v[k])
        sig = 1.4826 * np.median(np.abs(v[k] - bkg))
        # amplitude estimate for the next wing correction: 5x5
        c5 = (np.abs(xx - round(x)) <= 2) & (np.abs(yy - round(y)) <= 2) & np.isfinite(cut)
        den = np.sum(m1[c5] ** 2)
        amp = np.sum(m1[c5] * (cut[c5] - bkg)) / den if den > 0 else 0.0
    out = dict(bkg=float(bkg), sig=float(sig))
    xr, yr = round(x), round(y)
    for s in shapes:
        hs = s // 2
        box = (np.abs(xx - xr) <= hs) & (np.abs(yy - yr) <= hs)
        use = box & np.isfinite(cut)
        den = np.sum(m1[use] ** 2)
        out[f'amp{s}'] = float(np.sum(m1[use] * (cut[use] - bkg)) / den) if (den > 0 and use.sum() >= 0.8 * s * s) else np.nan
        out[f'n{s}'] = int(use.sum())
        out[f'err{s}'] = float(sig / np.sqrt(den)) if den > 0 else np.nan
    c3 = (np.abs(xx - xr) <= 1) & (np.abs(yy - yr) <= 1)
    out['center_ok'] = bool(np.all(np.isfinite(cut[c3])) and c3.sum() == 9)
    out['peak'] = float(np.nanmax(np.where(c3, cut, np.nan))) if np.isfinite(cut[c3]).any() else np.nan
    out['mpeak'] = float(m1.max())
    return out


def refine_xy(img, grid, x, y, valid=None, nb=None):
    """sub-pixel (x, y) by minimising the 5x5 residual (amp analytic, bkg from annulus), starting at (x, y)"""
    from scipy.optimize import minimize
    sl, _ = unit_psf(grid, x, y, 6, img.shape)
    if sl is None:
        return x, y

    def chi(p):
        o = fit_star(img, grid, p[0], p[1], nb=nb, valid=valid, shapes=(5,))
        if o is None or not np.isfinite(o['amp5']):
            return 1e30
        sl2, m1 = unit_psf(grid, p[0], p[1], H, img.shape)
        cut = img[sl2] - o['bkg']
        yy, xx = np.mgrid[sl2]
        b = (np.abs(xx - round(p[0])) <= 2) & (np.abs(yy - round(p[1])) <= 2) & np.isfinite(cut)
        return float(np.sum((cut[b] - o['amp5'] * m1[b]) ** 2))
    res = minimize(chi, [x, y], method='Nelder-Mead', options=dict(xatol=0.01, fatol=1e-6, maxiter=30, initial_simplex=[[x, y], [x + 0.3, y], [x, y + 0.3]]))
    px, py = res.x
    if abs(px - x) > 1.0 or abs(py - y) > 1.0:
        return x, y
    return float(px), float(py)


def zeroframe_info(fr):
    return dict(tframe=fr.tframe, nframes=fr.nframes, ngroups=fr.ngroups, readpatt=fr.readpatt, photmjsr=fr.photmjsr)
