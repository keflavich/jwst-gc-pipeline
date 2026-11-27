"""Cutouts of bright-window candidates with the short-axis width the width gate measures: data i2d on top, detection image below.

usage: python widthfig.py OUT.png [--root /orange/adamginsburg/jwst/gc-treasury] [--code PATH] [--half 10]
Rows of panel pairs: candidates removed by the width gate (width below 0.60 PSF sigma), removed by the hole guard, kept stars, and kept candidates with a width of 0.60-0.72 PSF sigma.  Per cell the top
panel is the module data i2d minus the m6 smoothed background (gray, -5 MJy/sr .. the 3x3 peak), the bottom panel the detection image (m6 residual i2d minus the same
background; magma, -5 MJy/sr .. 1.2 x its 3x3 peak).  The title gives tile, band, module, position (module pixels) and the width in PSF standard deviations
(``_minor_axis_sigma`` of the detection image / (FWHM / 2.355)).  Reads only cutouts of the gc-treasury images and writes only OUT.
"""
import os
import sys
import warnings

import matplotlib
import numpy as np
from astropy.io import fits

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings('ignore')
sys.dont_write_bytecode = True
a = sys.argv[1:]
OUT = a[0]
ROOT = a[a.index('--root') + 1] if '--root' in a else '/orange/adamginsburg/jwst/gc-treasury'
HALF = int(a[a.index('--half') + 1]) if '--half' in a else 10
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = a[a.index('--code') + 1] if '--code' in a else os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, CODE)
os.environ.setdefault('GC_ALLOW_DEV', '1')
from astropy.table import Table  # noqa: E402
from jwst_gc_pipeline.photometry import cataloging as C  # noqa: E402

_ftab = Table.read(C._L.fwhm_table_path())
FWHM = {b: float(_ftab[_ftab['Filter'] == b]['PSF FWHM (pixel)'][0]) for b in ('F480M', 'F212N')}
# Positions are the module-pixel centroids that ``realfunc.py`` writes to its ``*_widths.csv`` (every candidate that reaches the width gate), the hole-guard
# positions are the ones ``holeguard.py`` lists.  A group longer than the figure is wide continues on the next row of panel pairs.
GROUPS = [
    ('removed by the width gate (< 0.60)', [
        ('130', 'F480M', 'nrca', 2362.40, 1769.89), ('130', 'F480M', 'nrcb', 5225.48, 1696.71), ('118', 'F480M', 'nrca', 201.44, 192.58),
        ('118', 'F480M', 'nrcb', 4777.26, 891.80), ('118', 'F480M', 'nrcb', 3924.52, 1069.57), ('105', 'F480M', 'nrcb', 3916.79, 1314.89),
        ('130', 'F212N', 'nrcb', 7719.28, 3081.49), ('045', 'F212N', 'nrcb', 8004.71, 2835.51), ('045', 'F212N', 'nrca', 543.91, 1981.31),
        ('130', 'F212N', 'nrca', 3545.52, 2147.49), ('130', 'F212N', 'nrcb', 5950.89, 904.76), ('105', 'F212N', 'nrcb', 10349.59, 1137.96),
        ('045', 'F212N', 'nrcb', 9379.23, 4005.04), ('138', 'F212N', 'nrcb', 7783.85, 4091.21), ('118', 'F212N', 'nrca', 823.44, 2065.69),
        ('045', 'F212N', 'nrcb', 7218.38, 1371.27), ('138', 'F212N', 'nrca', 2565.98, 607.79)]),
    ('removed by the hole guard', [
        ('138', 'F480M', 'nrcb', 5614, 412), ('130', 'F480M', 'nrca', 2834, 979), ('105', 'F480M', 'nrcb', 4426, 509)]),
    ('kept stars', [
        ('045', 'F480M', 'nrca', 1049, 755), ('045', 'F480M', 'nrca', 2783, 958), ('045', 'F480M', 'nrcb', 3887, 207),
        ('138', 'F480M', 'nrcb', 4223, 735), ('045', 'F212N', 'nrcb', 7600, 2056), ('045', 'F212N', 'nrcb', 5806, 2585), ('130', 'F212N', 'nrcb', 9922, 4546)]),
    ('kept, width 0.60-0.72', [
        ('130', 'F212N', 'nrca', 1920.59, 1456.82), ('118', 'F212N', 'nrcb', 7351.19, 1388.94), ('045', 'F212N', 'nrca', 2200.28, 3066.40),
        ('130', 'F480M', 'nrcb', 4663.46, 1649.05), ('045', 'F480M', 'nrcb', 5331.79, 2212.85), ('045', 'F212N', 'nrcb', 9627.48, 4191.09)]),
]


def cutout(path, x, y, half):
    """The (2 half + 1)-pixel square of the SCI extension around (x, y), NaN outside the image."""
    with fits.open(path, memmap=True) as h:
        sec = h['SCI'].section
        ny, nx = h['SCI'].shape
        y0, y1, x0, x1 = max(y - half, 0), min(y + half + 1, ny), max(x - half, 0), min(x + half + 1, nx)
        out = np.full((2 * half + 1, 2 * half + 1), np.nan)
        out[y0 - (y - half):y1 - (y - half), x0 - (x - half):x1 - (x - half)] = np.asarray(sec[y0:y1, x0:x1], float)
    return out


ncol = 10
CHUNKS = [(gname if k == 0 else '', cells[k:k + ncol]) for gname, cells in GROUPS for k in range(0, len(cells), ncol)]
fig, ax = plt.subplots(2 * len(CHUNKS), ncol, figsize=(1.6 * ncol, 3.7 * len(CHUNKS)), squeeze=False)
for gi, (gname, cells) in enumerate(CHUNKS):
    for ci in range(ncol):
        if ci >= len(cells):
            ax[2 * gi, ci].axis('off')
            ax[2 * gi + 1, ci].axis('off')
            continue
        tile, band, mod, x, y = cells[ci]
        x, y = int(round(x)), int(round(y))
        base = f'{ROOT}/{band}/pipeline/jw10678-o{tile}_t001_nircam_clear-{band.lower()}-{mod}'
        bgc = cutout(base + '_resbgsub_hybpsf_m6_daophot_basic_mergedcat_residual_smoothed_bg_i2d.fits', x, y, HALF + 7)
        bgc = np.where(np.isfinite(bgc), bgc, 0.0)
        dat = cutout(base + '_data_i2d.fits', x, y, HALF + 7) - bgc
        det = cutout(base + '_resbgsub_hybpsf_m6_daophot_basic_mergedcat_residual_i2d.fits', x, y, HALF + 7) - bgc
        c = HALF + 7
        width = C._minor_axis_sigma(det, [float(c)], [float(c)])[0] / (FWHM[band] / 2.355)
        dpk = np.nanmax(dat[c - 1:c + 2, c - 1:c + 2])
        tpk = np.nanmax(det[c - 1:c + 2, c - 1:c + 2])
        sl = (slice(c - HALF, c + HALF + 1), slice(c - HALF, c + HALF + 1))
        for row, (im, vmax, cm) in enumerate(((dat, dpk, 'gray_r'), (det, 1.2 * tpk, 'magma'))):
            axx = ax[2 * gi + row, ci]
            axx.imshow(im[sl], origin='lower', cmap=cm, vmin=-5, vmax=vmax)
            axx.set_xticks([])
            axx.set_yticks([])
        ax[2 * gi, ci].set_title(f'o{tile} {band}\n{mod[-1]}({x},{y})\nwidth {width:.2f}' if np.isfinite(width) else f'o{tile} {band}\n{mod[-1]}({x},{y})\nwidth n/a', fontsize=6.5)
    ax[2 * gi, 0].set_ylabel(gname, fontsize=7)
fig.tight_layout()
fig.savefig(OUT, dpi=100)
print('wrote', OUT)
