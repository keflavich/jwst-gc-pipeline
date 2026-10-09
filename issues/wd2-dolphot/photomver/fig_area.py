"""fig_area.png: unsaturated main2 stars in each band's ZP window.  Rows: dm = ours - dolphot - ZP (median of the 25 nearest
stars), the pixel-area prediction pred = 2.5 log10(PIXAR_SR AREA / proj_plane_pixel_area), and dm - pred, for three bands;
right column: dm against pred in pred bins, for all bands."""
import sys
import glob
import numpy as np
from scipy.spatial import cKDTree
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an

an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2')
P = np.load(f'{an.Q}/photomver/areapred_main2.npz')
sky = A.sky[A.idx]
ra0, dec0 = np.median(sky.ra.deg), np.median(sky.dec.deg)
X = (sky.ra.deg - ra0) * np.cos(np.deg2rad(dec0)) * 3600
Y = (sky.dec.deg - dec0) * 3600
SHOW = ['200W', '212N', '300M']
fig = plt.figure(figsize=(19, 13.5))
gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 1.15])


def sel(band):
    dm = A.dm(band)
    lo, hi = an.ZPWIN.get(band, (0, 19))
    return dm, A.matched & np.isfinite(dm) & np.isfinite(P[band]) & ~A.rep[band] & ~A.sat[band] & (A.ref[band] >= lo) & (A.ref[band] < hi)


def outlines(ax, band):
    for fn in sorted(glob.glob(f'{an.Q}/tree_main2/F{band}/pipeline/jw03523005001_*_00001_*_align_o005_crf.fits')):
        h = fits.getheader(fn, 'SCI')
        c = WCS(h).pixel_to_world([0, h['NAXIS1'], h['NAXIS1'], 0, 0], [0, 0, h['NAXIS2'], h['NAXIS2'], 0])
        ax.plot((c.ra.deg - ra0) * np.cos(np.deg2rad(dec0)) * 3600, (c.dec.deg - dec0) * 3600, 'k-', lw=0.6)


for col, band in enumerate(SHOW):
    dm, ok = sel(band)
    idx = np.nonzero(ok)[0]
    tree = cKDTree(np.c_[X[idx], Y[idx]])
    _, nn = tree.query(np.c_[X[idx], Y[idx]], k=25)
    pred = P[band][idx]
    d = dm[idx]
    for row, (val, lab) in enumerate(((d, 'dm'), (pred, 'pred (pixel area)'), (d - pred, 'dm - pred'))):
        ax = fig.add_subplot(gs[row, col])
        sm = np.median(val[nn], axis=1) if row != 1 else val
        sc = ax.scatter(X[idx], Y[idx], c=sm, s=3, cmap='RdBu_r', vmin=-0.04, vmax=0.04, rasterized=True)
        outlines(ax, band)
        ax.set_aspect('equal')
        ax.invert_xaxis()
        rs = 1.4826 * np.median(np.abs(val - np.median(val)))
        ax.set_title(f'F{band} {lab}' + (f', robust std {rs:.4f}' if row != 1 else ''), fontsize=10)
        if row == 2:
            ax.set_xlabel('dRA (arcsec)')
        if col == 0:
            ax.set_ylabel('dDec (arcsec)')
    plt.colorbar(sc, ax=ax, shrink=0.8, label='mag')
ax = fig.add_subplot(gs[:, 3])
edges = np.linspace(-0.04, 0.04, 11)
for k, band in enumerate(an.BANDS):
    dm, ok = sel(band)
    pred = P[band]
    xm, ym = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = ok & (pred >= lo) & (pred < hi)
        if s.sum() >= 30:
            xm.append(np.median(pred[s]))
            ym.append(np.median(dm[s]))
    lw = band in ('250M', '277W', '300M', '335M', '410M', '323N', '405N', '466N')
    ax.plot(xm, ym, marker='o' if lw else 's', ms=3.5, lw=1, ls='-' if lw else '--', color=plt.cm.tab20(k % 20), label=f'F{band}')
xx = np.array([-0.04, 0.04])
ax.plot(xx, xx, 'k-', lw=1.5, label='slope 1')
ax.plot(xx, 1.6 * xx, 'k:', lw=1.2, label='slope 1.6')
ax.axhline(0, color='0.6', lw=0.6)
ax.axvline(0, color='0.6', lw=0.6)
ax.set_xlabel('pred = 2.5 log10(pixel area / nominal) (mag)')
ax.set_ylabel('median dm = ours - dolphot - ZP (mag)')
ax.set_title('unsaturated stars, ZP window: dm in bins of pred')
ax.legend(fontsize=7.5, ncol=2, loc='upper left')
ax.set_xlim(-0.04, 0.04)
ax.set_ylim(-0.07, 0.07)
fig.tight_layout()
fig.savefig(f'{an.Q}/photomver/fig_area.png', dpi=80)
