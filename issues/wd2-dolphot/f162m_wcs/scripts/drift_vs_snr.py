"""Seed drift (x_fit - x_init, y_fit - y_init) of free m7 fits in SNR bins,
F162M and F212N, exposure 00001, mainfcbg tree.  Tests whether the bright-star
drift (0.16-0.35 px in F162M) differs from the bulk (<= 0.08 px)."""
import glob
import numpy as np
from astropy.table import Table

T = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/tree_mainfcbg'
BINS = [(0, 10), (10, 20), (20, 50), (50, 200), (200, 1e9)]
first = True
for b in ['F162M', 'F212N']:
    for d in ['nrca1', 'nrca4', 'nrcb1', 'nrcb3']:
        f = sorted(glob.glob(f'{T}/{b}/{b.lower()}_{d}_visit001_vgroup*_exp00001_resbgsub_m7_daophot_basic.fits'))[0]
        t = Table.read(f)
        if first:
            print('columns:', ' '.join(t.colnames))
            first = False
        snr = np.asarray(t['flux_fit'], float) / np.asarray(t['flux_err'], float)
        free = ~np.asarray(t['forced_refit'], bool) & np.isfinite(snr)
        dx = np.asarray(t['x_fit'], float) - np.asarray(t['x_init'], float)
        dy = np.asarray(t['y_fit'], float) - np.asarray(t['y_init'], float)
        cells = []
        for lo, hi in BINS:
            k = free & (snr >= lo) & (snr < hi)
            cells.append(f'snr {lo:g}-{hi:g}: ({np.nanmedian(dx[k]):+.3f},{np.nanmedian(dy[k]):+.3f}) n={k.sum()}')
        print(f'{b} {d}: ' + ' | '.join(cells))
