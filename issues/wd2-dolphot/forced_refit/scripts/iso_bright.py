"""Cutouts of core rows with no dolphot source within 0.1" (spur.py category
isolated), brighter than MAGCUT: F200W data, band data, band m7 residual.
usage: python iso_bright.py BAND N [MAGCUT]"""
import glob
import sys
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.visualization import simple_norm
import astropy.units as u
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an

b = sys.argv[1]
N = int(sys.argv[2])
magcut = float(sys.argv[3]) if len(sys.argv) > 3 else 18.6
bl = 'f' + b.lower()
T = f'{an.Q}/tree_mainfcbg'
s = Table.read(f'../spur_mainfcbg_{b}.ecsv')
s = s[(s['category'] == 'isolated') & (s['mag'] < magcut)]
s.sort('mag')
print(len(s), 'isolated rows brighter than', magcut)
A = an.Arm('mainfcbg')
dsk = SkyCoord(np.asarray(A.m['RA'], float) * u.deg, np.asarray(A.m['DEC'], float) * u.deg)
cat = Table.read(f'{T}/catalogs/{bl}_merged_indivexp_merged_resbgsub_m7_dao_basic.fits')
csk = cat['skycoord']
csk = csk[np.isfinite(csk.ra.deg)]
P = f'{T}/F{b}/pipeline'
imgs = [('F200W data', glob.glob(f'{T}/F200W/pipeline/jw*-o005_t001_nircam_clear-f200w-merged_i2d.fits')[0]),
        (f'F{b} data', glob.glob(f'{P}/jw*-o005_t001_nircam_clear-{bl}-merged_i2d.fits')[0]),
        (f'F{b} m7 residual', glob.glob(f'{P}/jw*-o005_t001_nircam_clear-{bl}-merged_resbgsub_m7_daophot_basic_mergedcat_residual_i2d.fits')[0])]
data = [(lab, fits.getdata(fn, 'SCI'), WCS(fits.getheader(fn, 'SCI'))) for lab, fn in imgs]
pick = np.linspace(0, len(s) - 1, min(N, len(s))).astype(int)
cmap = plt.get_cmap('gray').copy()
cmap.set_bad('steelblue')
fig, axes = plt.subplots(len(pick), 3, figsize=(8, 2.7 * len(pick)), squeeze=False)
for r, k in enumerate(pick):
    c = SkyCoord(s['ra'][k] * u.deg, s['dec'][k] * u.deg)
    for col, (lab, img, w) in enumerate(data):
        ax = axes[r, col]
        co = Cutout2D(img, c, 3.0 * u.arcsec, wcs=w, mode='partial')
        if col == 2:
            ref = Cutout2D(data[1][1], c, 3.0 * u.arcsec, wcs=data[1][2], mode='partial').data
            vmin, vmax = np.nanpercentile(ref, [1, 99.5])
            norm = simple_norm(co.data, 'asinh', vmin=vmin, vmax=vmax)
        else:
            norm = simple_norm(co.data, 'asinh', percent=99.5)
        ax.imshow(co.data, origin='lower', cmap=cmap, norm=norm)
        for sk, kw in ((dsk, dict(marker='o', s=50, facecolors='none', edgecolors='red', lw=1)),
                       (csk, dict(marker='x', s=30, c='cyan', lw=0.8))):
            near = sk.separation(c) < 1.8 * u.arcsec
            x, y = co.wcs.world_to_pixel(sk[near])
            ax.scatter(x, y, **kw)
        x0, y0 = co.wcs.world_to_pixel(c)
        ax.scatter([x0], [y0], marker='o', s=220, facecolors='none', edgecolors='yellow', lw=1.2)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(lab, fontsize=9)
        if col == 0:
            ax.set_ylabel(f'{s["mag"][k]:.2f}  parent {s["nearest_bright_sep"][k]:.1f}"\n'
                          f'{s["ra"][k]:.5f} {s["dec"][k]:.5f}', fontsize=8)
fig.suptitle(f'F{b} core rows without dolphot source (isolated, < {magcut}); 3" cutouts\n'
             'red o dolphot, cyan x pipeline rows, yellow target; blue = NaN', fontsize=9)
fig.tight_layout()
fig.savefig(f'iso_bright_{b}.png', dpi=80)
