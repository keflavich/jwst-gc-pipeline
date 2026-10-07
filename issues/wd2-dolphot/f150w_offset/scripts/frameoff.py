"""Per-frame (x_fit - x_init, y_fit - y_init) of free m7 fits, and of the
forced rows' seed offset from the frame's own free-fit solution.
usage: python frameoff.py BAND"""
import glob
import sys
import numpy as np
from astropy.table import Table

b = sys.argv[1].upper().lstrip('F')
T = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/tree_mainfcbg'
fs = sorted(glob.glob(f'{T}/F{b}/f{b.lower()}_nrc*_visit001_vgroup*_exp0000?_resbgsub_m7_daophot_basic.fits'))
print(f'F{b}: {len(fs)} frames')
print('frame | N free | med dx | med dy | N forced | flux>0 forced')
for f in fs:
    t = Table.read(f)
    fr = np.asarray(t['forced_refit'], bool) if 'forced_refit' in t.colnames else np.zeros(len(t), bool)
    dx = np.asarray(t['x_fit'], float) - np.asarray(t['x_init'], float)
    dy = np.asarray(t['y_fit'], float) - np.asarray(t['y_init'], float)
    fl = np.asarray(t['flux_fit'], float)
    free = ~fr & np.isfinite(dx) & (fl > 0)
    bright = free & (fl > np.nanpercentile(fl[free], 50))
    name = f.split('/')[-1].replace(f'f{b.lower()}_', '').replace('_resbgsub_m7_daophot_basic.fits', '')
    print(f'{name} | {bright.sum()} | {np.median(dx[bright]):+.3f} | {np.median(dy[bright]):+.3f} | '
          f'{fr.sum()} | {(fr & (fl > 0)).sum()}')
