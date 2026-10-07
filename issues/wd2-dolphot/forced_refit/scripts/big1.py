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
ra, dec, size = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
b = sys.argv[4] if len(sys.argv) > 4 else '277W'
bl = 'f' + b.lower()
c = SkyCoord(ra * u.deg, dec * u.deg)
T = f'{an.Q}/tree_mainfcbg'
P = f'{T}/F{b}/pipeline'
A = an.Arm('mainfcbg')
dsk = SkyCoord(np.asarray(A.m['RA'], float) * u.deg, np.asarray(A.m['DEC'], float) * u.deg)
cat = Table.read(f'{T}/catalogs/{bl}_merged_indivexp_merged_resbgsub_m7_dao_basic.fits')
csk = cat['skycoord']
fl = np.asarray(cat['flux'], float)
ok = np.isfinite(csk.ra.deg) & (fl > 0)
csk, fl = csk[ok], fl[ok]
m = 24.129 - 2.5 * np.log10(fl)
imgs = [('F200W', glob.glob(f'{T}/F200W/pipeline/jw*-o005_t001_nircam_clear-f200w-merged_i2d.fits')[0]),
        (f'F{b}', glob.glob(f'{P}/jw*-o005_t001_nircam_clear-{bl}-merged_i2d.fits')[0]),
        (f'F{b} m7 resid', glob.glob(f'{P}/jw*-o005_t001_nircam_clear-{bl}-merged_resbgsub_m7_daophot_basic_mergedcat_residual_i2d.fits')[0]),
        (f'F{b} m7 model', glob.glob(f'{P}/jw*-o005_t001_nircam_clear-{bl}-merged_resbgsub_m7_daophot_basic_mergedcat_model_i2d.fits')[0])]
cmap = plt.get_cmap('gray').copy(); cmap.set_bad('steelblue')
fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
for ax, (lab, fn) in zip(axes, imgs):
    co = Cutout2D(fits.getdata(fn, 'SCI'), c, size * u.arcsec, wcs=WCS(fits.getheader(fn, 'SCI')), mode='partial')
    if 'resid' in lab:
        v = np.nanpercentile(np.abs(co.data), 95)
        ax.imshow(co.data, origin='lower', cmap='RdBu_r', vmin=-v, vmax=v)
    else:
        ax.imshow(co.data, origin='lower', cmap=cmap, norm=simple_norm(co.data, 'asinh', percent=99.5))
    near = dsk.separation(c) < size * 0.75 * u.arcsec
    x, y = co.wcs.world_to_pixel(dsk[near])
    ax.scatter(x, y, marker='o', s=40, facecolors='none', edgecolors='red', lw=0.8)
    near = csk.separation(c) < size * 0.75 * u.arcsec
    x, y = co.wcs.world_to_pixel(csk[near])
    ax.scatter(x, y, marker='x', s=25, c=np.where(m[near] < 18.6, 'yellow', 'cyan'), lw=0.8)
    x0, y0 = co.wcs.world_to_pixel(c)
    ax.scatter([x0], [y0], marker='o', s=300, facecolors='none', edgecolors='lime', lw=1.2)
    ax.set_xlim(-0.5, co.data.shape[1] - 0.5); ax.set_ylim(-0.5, co.data.shape[0] - 0.5)
    ax.set_title(lab)
fig.suptitle(f'{ra:.5f} {dec:.5f}; red o dolphot; x pipeline rows (yellow < 18.6)')
fig.tight_layout()
fig.savefig(f'big_{ra:.5f}_{b}.png', dpi=70)
