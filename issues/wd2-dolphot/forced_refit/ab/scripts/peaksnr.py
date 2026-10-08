"""Local peak S/N in the AB mosaic at class c / b / control rows (3x3 max minus annulus median, over annulus robust sigma),
compared with random sky positions in the footprint, matched in pipeline mag."""
import sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
rng = np.random.default_rng(3)
B = {'187N': (22.5, 25.5), '200W': (23.5, 27.0)}
for b, (mlo, mhi) in B.items():
    z = np.load(f'classify_{b}.npz')
    h = fits.open(f'/orange/adamginsburg/jwst/wd2/wd2_F{b}_AB_i2d.fits', memmap=False)
    img = h['SCI'].data; w = WCS(h['SCI'].header)
    ny, nx = img.shape
    yy, xx = np.mgrid[-11:12, -11:12]
    rr = np.hypot(xx, yy)
    ann = (rr >= 6) & (rr <= 10); core = rr <= 1.5

    def score(ra, dec):
        x, y = w.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
        out = np.full(len(ra), np.nan)
        for i, (xi, yi) in enumerate(zip(np.round(x).astype(int), np.round(y).astype(int))):
            if xi < 12 or yi < 12 or xi >= nx - 12 or yi >= ny - 12:
                continue
            cut = img[yi - 11:yi + 12, xi - 11:xi + 12]
            a = cut[ann]; a = a[np.isfinite(a)]
            c = cut[core]
            if a.size < 50 or not np.isfinite(c).any():
                continue
            sig = 1.4826 * np.median(np.abs(a - np.median(a)))
            if sig <= 0:
                continue
            out[i] = (np.nanmax(c) - np.median(a)) / sig
        return out
    m1 = z['m1']; cls = z['cls']; ff1 = z['ff1']
    res = {}
    for name, sel in (('c', cls == 'c'), ('b', cls == 'b'), ('control', ff1 == 0)):
        s = np.where(sel & (m1 >= mlo) & (m1 < mhi))[0]
        s = rng.choice(s, min(2500, len(s)), replace=False)
        res[name] = score(z['ra'][s], z['dec'][s])
    # random positions inside the catalog footprint: jitter control positions by 1.5" to a random direction
    s = np.where((ff1 == 0))[0]; s = rng.choice(s, 2500, replace=False)
    ang = rng.uniform(0, 2 * np.pi, len(s)); d = rng.uniform(1.0, 2.0, len(s)) / 3600
    res['random sky'] = score(z['ra'][s] + d * np.cos(ang) / np.cos(np.deg2rad(z['dec'][s])), z['dec'][s] + d * np.sin(ang))
    print(f'## F{b} (pipeline mag {mlo}-{mhi}): 3x3 peak S/N percentiles 10/25/50/75/90, frac>3')
    for k, v in res.items():
        v = v[np.isfinite(v)]
        print(f'  {k:11s} N={len(v):5d}: ' + ' '.join(f'{p:6.1f}' for p in np.percentile(v, [10, 25, 50, 75, 90])) + f'  frac>3: {np.mean(v > 3):.2f}')
