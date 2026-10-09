"""Map of dm = ours - dolphot - ZP for unsaturated main2 stars in each band's ZP window, on the sky, with the
exposure-1 detector outlines.  Per-detector cell medians in 4 x 4 detector-pixel cells (exposure 1 frame) are also
printed, to see whether the per-detector term is a step at detector edges or a smooth trend within a detector.
usage: python dmmap.py [BAND ...] > dmmap.txt  (writes dmmap.png)"""
import sys
import glob
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an

an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2')
tree = f'{an.Q}/tree_main2'
sky = A.sky[A.idx]
bands = sys.argv[1:] or ['150W', '200W', '212N', '250M', '335M', '410M']
ra0, dec0 = np.median(sky.ra.deg), np.median(sky.dec.deg)
X = (sky.ra.deg - ra0) * np.cos(np.deg2rad(dec0)) * 3600
Y = (sky.dec.deg - dec0) * 3600
fig, axs = plt.subplots(2, (len(bands) + 1) // 2, figsize=(5.2 * ((len(bands) + 1) // 2), 10))
for ax, band in zip(axs.flat, bands):
    dm = A.dm(band)
    lo, hi = an.ZPWIN.get(band, (0, 19))
    ok = A.matched & np.isfinite(dm) & ~A.rep[band] & ~A.sat[band] & (A.ref[band] >= lo) & (A.ref[band] < hi)
    hb = ax.hexbin(X[ok], Y[ok], C=dm[ok], reduce_C_function=np.median, gridsize=45, mincnt=6, cmap='RdBu_r', vmin=-0.05, vmax=0.05)
    print(f'\n### F{band}: 4 x 4 cell medians of dm per detector (exposure 1 pixel cells; rows = y from low to high), N >= 8')
    for fn in sorted(glob.glob(f'{tree}/F{band}/pipeline/jw03523005001_*_00001_*_align_o005_crf.fits')):
        h0 = fits.getheader(fn)
        h = fits.getheader(fn, 'SCI')
        w = WCS(h)
        ny, nx = h['NAXIS2'], h['NAXIS1']
        c = w.pixel_to_world([0, nx, nx, 0, 0], [0, 0, ny, ny, 0])
        ax.plot((c.ra.deg - ra0) * np.cos(np.deg2rad(dec0)) * 3600, (c.dec.deg - dec0) * 3600, 'k-', lw=0.6)
        det = h0['DETECTOR'].lower()
        x, y = w.world_to_pixel(sky)
        inn = ok & (x > 10) & (x < nx - 10) & (y > 10) & (y < ny - 10)
        cx, cy = np.mean(c.ra.deg[:4] - ra0) * np.cos(np.deg2rad(dec0)) * 3600, np.mean(c.dec.deg[:4] - dec0) * 3600
        ax.text(cx, cy, f'{det[3:]}\n{np.median(dm[inn]):+.3f}', ha='center', va='center', fontsize=7)
        print(f'{det} (N {int(inn.sum())}, median {np.median(dm[inn]):+.4f}):')
        for j in range(4):
            cells = []
            for i in range(4):
                s = inn & (x >= i * nx / 4) & (x < (i + 1) * nx / 4) & (y >= j * ny / 4) & (y < (j + 1) * ny / 4)
                cells.append(f'{np.median(dm[s]):+.3f}' if s.sum() >= 8 else '   nan')
            print('    ' + ' '.join(cells))
    ax.set_title(f'F{band}: ours - dolphot - ZP, unsat {lo:.1f}-{hi:.1f} mag')
    ax.set_aspect('equal')
    ax.invert_xaxis()
    ax.set_xlabel('dRA (arcsec)')
    ax.set_ylabel('dDec (arcsec)')
    plt.colorbar(hb, ax=ax, shrink=0.8, label='median dm (mag)')
fig.tight_layout()
fig.savefig(f'{an.Q}/photomver/dmmap.png', dpi=90)
