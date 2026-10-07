"""Predicted m7 free-fit drift from the F150W inter-detector DVA shift.

Seeds come from the cross-band catalog (bands without DVACORR) and are placed
on F150W pixels with the DVA-shifted WCS; stars sit where the unshifted
(_cal) WCS puts them.  Predicted drift = pixel_cal(sky) - pixel_crf(sky),
evaluated at the median position of the frame's bright free fits.
usage: python dva_predict.py"""
import glob
import warnings
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

warnings.simplefilter('ignore')
T = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/tree_mainfcbg'
P = '/orange/adamginsburg/jwst/wd2/F150W/pipeline'
print('frame | DVASHRA* (mas) | DVASHDE (mas) | pred dx | pred dy | meas dx | meas dy')
rows = []
for f in sorted(glob.glob(f'{T}/F150W/f150w_nrc*_visit001_vgroup10101_exp0000?_resbgsub_m7_daophot_basic.fits')):
    det = f.split('_')[-7] if False else f.split('/')[-1].split('_')[1]
    exp = f.split('_exp')[1][:5]
    base = f'{P}/jw03523005001_10101_{exp}_{det}'
    hc = fits.getheader(f'{base}_cal.fits', 'SCI')
    ha = fits.getheader(f'{base}_align_o005_crf.fits', 'SCI')
    wc, wa = WCS(hc), WCS(ha)
    t = Table.read(f)
    fr = np.asarray(t['forced_refit'], bool)
    x0 = np.asarray(t['x_init'], float)
    y0 = np.asarray(t['y_init'], float)
    dx = np.asarray(t['x_fit'], float) - x0
    dy = np.asarray(t['y_fit'], float) - y0
    fl = np.asarray(t['flux_fit'], float)
    free = ~fr & np.isfinite(dx) & (fl > 0)
    br = free & (fl > np.nanpercentile(fl[free], 50))
    sky = wa.pixel_to_world(x0[br], y0[br])
    xc, yc = wc.world_to_pixel(sky)
    pdx, pdy = np.median(xc - x0[br]), np.median(yc - y0[br])
    mdx, mdy = np.median(dx[br]), np.median(dy[br])
    cd = np.cos(np.deg2rad(ha['CRVAL2']))
    rows.append((pdx, pdy, mdx, mdy))
    print(f"{det}_{exp} | {ha['DVASHRA'] * cd * 3.6e6:+.1f} | {ha['DVASHDE'] * 3.6e6:+.1f} | "
          f"{pdx:+.3f} | {pdy:+.3f} | {mdx:+.3f} | {mdy:+.3f}")
r = np.array(rows)
print(f'median residual meas - pred: dx {np.median(r[:, 2] - r[:, 0]):+.3f} dy {np.median(r[:, 3] - r[:, 1]):+.3f} px; '
      f'rms {np.std(r[:, 2] - r[:, 0]):.3f} {np.std(r[:, 3] - r[:, 1]):.3f}')
