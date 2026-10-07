"""Mechanism figure: the largest SATURATED component of F277W nrcblong exp1
(mainfcbg tree), its centre of mass (the current hand-off position), the
no-row dolphot stars inside it, accepted satstars, and the peak hand-off
positions of jwst-gc-pipeline-hpeaksrun (048299af, area >= 50 px).
usage: python figmech.py"""
import glob
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.visualization import simple_norm
import astropy.units as u
from scipy import ndimage
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an  # noqa: E402
REPO = '/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-hpeaksrun'
sys.path.insert(0, REPO)
from jwst_gc_pipeline.photometry import cataloging as C  # noqa: E402

arm, b, FWHM = 'mainfcbg', '277W', 1.48
P = f'{an.Q}/tree_{arm}/F{b}/pipeline'
f = sorted(glob.glob(f'{P}/jw03523005001_*_00001_nrcblong_align_o005_crf.fits'))[0]
sci = fits.getdata(f, 'SCI').astype(float)
dq = fits.getdata(f, 'DQ')
w = WCS(fits.getheader(f, 'SCI'))
acc = Table.read(f.replace('.fits', '_resbgsub_m7_satstar_catalog.fits'))
sat = (dq & 2) != 0
lab, n = ndimage.label(sat)
sizes = np.bincount(lab.ravel())
sizes[0] = 0
big = int(np.argmax(sizes))
com = np.asarray(ndimage.center_of_mass(sat, lab, big))[::-1]
comp = lab == big
ys, xs = np.nonzero(comp)
x0, x1, y0, y1 = xs.min() - 5, xs.max() + 6, ys.min() - 5, ys.max() + 6
pk = C._sat_component_peak_xy(sci, dq, lab, [big], FWHM)
ax_, ay_ = np.asarray(acc['xcentroid'], float), np.asarray(acc['ycentroid'], float)
near = C._L._protect_mask(pk[:, 0], pk[:, 1], np.column_stack([ax_, ay_]), 1.5 * FWHM)
pk = pk[~near]
tr = Table.read(f'trace_{arm}_{b}.ecsv')
A = an.Arm(arm)
rs = SkyCoord(np.asarray(A.m['RA'], float) * u.deg, np.asarray(A.m['DEC'], float) * u.deg)
xn, yn = w.world_to_pixel(rs[np.asarray(tr['dolphot_idx'])])
inc = (xn > x0) & (xn < x1) & (yn > y0) & (yn < y1)
xn, yn = xn[inc], yn[inc]
oncomp = comp[np.rint(yn).astype(int), np.rint(xn).astype(int)]
dpk = np.min(np.hypot(pk[:, 0][None] - xn[:, None], pk[:, 1][None] - yn[:, None]), axis=1)
print(f'{f.split("/")[-1]}: comp {big} {sizes[big]} px, COM {com.round(1)}, '
      f'{len(pk)} peaks (after accepted exclusion); no-row stars in box {inc.sum()}, '
      f'on the component {oncomp.sum()}, with a peak within 1 px {(dpk[oncomp] <= 1).sum()}')
d_com = np.hypot(xn - com[0], yn - com[1])
print(f'  no-row on comp: distance to COM px p10/50/90 {np.percentile(d_com[oncomp], [10, 50, 90]).round(1)}')

fig, axs = plt.subplots(1, 2, figsize=(15, 7.5))
cut = sci[y0:y1, x0:x1]
norm = simple_norm(cut[np.isfinite(cut)], 'asinh', percent=99.5)
for k, ax in enumerate(axs):
    ax.imshow(sci, origin='lower', cmap='gray_r', norm=norm, interpolation='nearest')
    ax.contour(comp.astype(float), levels=[0.5], colors='tab:orange', linewidths=0.8)
    ax.plot(pk[:, 0], pk[:, 1], '+', color='tab:cyan', ms=6, mew=1.2,
            label=f'peak hand-off positions ({len(pk)})')
    ax.plot(xn, yn, 'o', mfc='none', mec='red', ms=9, mew=1.2,
            label=f'dolphot stars with no F277W row ({oncomp.sum()} on this component)')
    ax.plot(ax_, ay_, 'x', color='tab:green', ms=8, mew=1.5, label='accepted satstars')
    ax.plot(*com, '*', color='gold', mec='k', ms=22, label='centre of mass (current hand-off)')
    if k == 0:
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_title(f'F277W nrcblong exp1: largest SATURATED component ({sizes[big]} px, orange)')
        ax.legend(loc='lower left', fontsize=8, framealpha=0.9)
    else:
        cx, cy = np.median(xn[oncomp]), np.median(yn[oncomp])
        ax.set_xlim(cx - 30, cx + 30)
        ax.set_ylim(cy - 30, cy + 30)
        ax.set_title('zoom (60x60 px)')
        axs[0].add_patch(plt.Rectangle((cx - 30, cy - 30), 60, 60, fill=False, ec='blue', lw=1))
    ax.set_xlabel('x (px)')
    ax.set_ylabel('y (px)')
fig.tight_layout()
fig.savefig('mech_mainfcbg_277W.png', dpi=110)
