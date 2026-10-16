"""gc2211 o050 m7, retie (2026-10-10, no CR rejection) vs ng2 rerun (outlier_detection on).

For F200W and F277W:
  * control sample: every vetted m7 star of each product, PSF-matched
    residual S/N in that product's own m7 residual mosaic; fraction with S/N > 10
  * sharp pixels (1-2 px CR-like spikes, spikes.py definition) in the m7
    residual at the same 2000 random sky positions in both products
  * gallery: residual stamps before | after at the 12 brightest-residual
    before-control stars
usage: python o050_compare.py <out-prefix>
"""
import json
import os
import sys

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy.ndimage import median_filter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(S, '..', '..', 'phaseloss'))
from phase_loss import FWHM_PIX, Mosaic, stamp_metrics, _image_hdu  # noqa: E402

BEFORE = os.path.join(S, '..', '..', 'prod_before', 'gc2211_o050_retie')
LIVE = '/orange/adamginsburg/jwst/gc2211_o050'
RESID = 'jw02211-o050_t001_nircam_clear-{f}-nrcb_resbgsub_m7_daophot_basic_mergedcat_residual_i2d.fits'
VET = '{f}_nrcb_o050_indivexp_merged_resbgsub_m7_dao_basic_vetted.fits'
HALF = 10


def paths(which, filt):
    f = filt.lower()
    if which == 'before':
        return os.path.join(BEFORE, RESID.format(f=f)), os.path.join(BEFORE, VET.format(f=f))
    return (os.path.join(LIVE, filt, 'pipeline', RESID.format(f=f)),
            os.path.join(LIVE, 'catalogs', VET.format(f=f)))


def stamps(data, x, y):
    ny, nx = data.shape
    out = np.full((len(x), 2 * HALF + 1, 2 * HALF + 1), np.nan, np.float32)
    for k, (xi, yi) in enumerate(zip(np.round(x).astype(int), np.round(y).astype(int))):
        y0, y1 = max(yi - HALF, 0), min(yi + HALF + 1, ny)
        x0, x1 = max(xi - HALF, 0), min(xi + HALF + 1, nx)
        if y1 <= y0 or x1 <= x0:
            continue
        out[k, y0 - (yi - HALF):y1 - (yi - HALF), x0 - (xi - HALF):x1 - (xi - HALF)] = data[y0:y1, x0:x1]
    return out


def nsharp(st):
    n = 0
    for s in st:
        if not np.isfinite(s).any():
            continue
        s = np.nan_to_num(s, nan=np.nanmedian(s))
        med = np.median(s)
        sig = 1.4826 * np.median(np.abs(s - med)) + 1e-12
        mf = median_filter(s, size=3)
        c = s[1:-1, 1:-1]
        excess = c - mf[1:-1, 1:-1]
        box = sum(np.roll(np.roll(s, dy, 0), dx, 1)[1:-1, 1:-1]
                  for dy in (-1, 0, 1) for dx in (-1, 0, 1)) - 9 * med
        n += int(np.sum((excess > 8 * sig) & ((c - med) > 0.5 * np.maximum(box, 1e-12))))
    return n


outp = sys.argv[1]
rng = np.random.default_rng(0)
res = {}
fig, axs = plt.subplots(2, 3, figsize=(17, 10))
gal = {}
for i, filt in enumerate(('F200W', 'F277W')):
    fwhm = FWHM_PIX[filt]
    res[filt] = {}
    # common random sky positions from the before mosaic footprint
    with fits.open(paths('before', filt)[0], memmap=True) as h:
        hdu = _image_hdu(h)
        w0 = WCS(hdu.header)
        ny, nx = hdu.data.shape
        good = np.isfinite(hdu.data[::50, ::50])
    gy, gx = np.nonzero(good)
    k = rng.choice(len(gx), size=2000, replace=True)
    rx = gx[k] * 50 + rng.uniform(0, 50, 2000)
    ry = gy[k] * 50 + rng.uniform(0, 50, 2000)
    rsky = w0.pixel_to_world(rx, ry)
    ctrl_sky = None
    for which in ('before', 'after'):
        rpath, vpath = paths(which, filt)
        t = Table.read(vpath)
        sc = SkyCoord(t['skycoord'])
        pick = np.arange(len(t))
        mos = Mosaic(rpath)
        x, y = mos.wcs.world_to_pixel(sc[pick])
        m = stamp_metrics(mos, x, y, fwhm)
        snr = m['snr']
        xr, yr = mos.wcs.world_to_pixel(rsky)
        sh = nsharp(stamps(mos.data, xr, yr))
        if which == 'before':
            order = np.argsort(np.nan_to_num(snr, nan=-1))[::-1][:12]
            ctrl_sky = sc[pick][order]
            ctrl_snr_before = snr[order]
        gx_, gy_ = mos.wcs.world_to_pixel(ctrl_sky)
        gal[(filt, which)] = stamps(mos.data, gx_, gy_)
        mos.close()
        res[filt][which] = dict(
            n_vetted=len(t), n_sampled=len(pick),
            p50=float(np.nanmedian(snr)), p90=float(np.nanpercentile(snr, 90)),
            p99=float(np.nanpercentile(snr, 99)), frac_gt10=float(np.nanmean(snr > 10)),
            sharp_px_2000_stamps=sh, resid=rpath, vetted=vpath,
            vetted_mtime=os.path.getmtime(vpath), resid_mtime=os.path.getmtime(rpath))
        hi = snr > 10
        ax = axs[i, 0 if which == 'before' else 1]
        ax.scatter(x[~hi], y[~hi], s=1, c='0.7', rasterized=True)
        ax.scatter(x[hi], y[hi], s=3, c=np.log10(snr[hi]), cmap='inferno', vmin=1, vmax=3)
        ax.set_title(f'o050 {filt} m7 {which}: {hi.sum()}/{len(pick)} with resid S/N>10\n'
                     f'sharp px in 2000 resid stamps: {sh}', fontsize=10)
        ax.set_aspect('equal')
        bins = np.linspace(0, 60, 61)
        axs[i, 2].hist(np.clip(snr, 0, 60), bins=bins, histtype='step', lw=1.5,
                       label=f'{which} ({100 * np.nanmean(hi):.1f}% > 10)')
    axs[i, 2].set_yscale('log')
    axs[i, 2].axvline(10, color='k', ls=':')
    axs[i, 2].set_xlabel('m7 residual S/N at catalogued star (clipped at 60)')
    axs[i, 2].legend()
    res[filt]['gallery_before_snr'] = [float(v) for v in ctrl_snr_before]
fig.tight_layout()
fig.savefig(outp + '_ctrl.png', dpi=90)

fig, axs = plt.subplots(4, 12, figsize=(18, 6.6))
for i, filt in enumerate(('F200W', 'F277W')):
    for j in range(12):
        b = gal[(filt, 'before')][j]
        a = gal[(filt, 'after')][j]
        lim = np.nanpercentile(np.abs(b), 99) or 1
        for r, (st, lab) in enumerate(((b, 'before'), (a, 'after'))):
            ax = axs[2 * i + r, j]
            ax.imshow(st, origin='lower', cmap='RdBu_r', vmin=-lim, vmax=lim)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f'{filt}\n{lab}', fontsize=9)
            if r == 0:
                ax.set_title(f'S/N {res[filt]["gallery_before_snr"][j]:.0f}', fontsize=8)
fig.suptitle('o050 m7 residual at the 12 worst-residual catalogued stars of the retie product '
             '(before = 2026-10-10 retie, no CR rejection; after = outlier_detection on). '
             'Same colour scale per column.', fontsize=10)
fig.tight_layout()
fig.savefig(outp + '_gallery.png', dpi=90)
json.dump(res, open(outp + '.json', 'w'), indent=1)
print(json.dumps({f: {w: {k: v for k, v in d.items() if k not in ('resid', 'vetted')}
                      for w, d in r.items() if w in ('before', 'after')} for f, r in res.items()}, indent=1))
