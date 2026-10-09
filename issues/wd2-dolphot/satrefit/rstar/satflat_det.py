"""Satstar rows: PSF^2-weighted flat over fit px and over rim px, by detector; within-detector slope of per-star dm vs flat."""
import sys, re
import numpy as np
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/capbind')
from cb_lib import Band, mad
from an4 import REFV
from score7f import load, caps, BINS

for band in ('150W', '200W'):
    B = Band(band)
    col, G = load(B)
    det_r = np.array([re.search(r'_(nrc\w+?)_align', B.map['files'][k]).group(1) for k in B.frame_of])
    fl = col('flat_rimw')
    D = {}
    for nm in ('H', 'Hf'):
        c = caps(B, G, nm) * B.rcor
        c = np.where(np.isfinite(c), c, np.inf)
        D[nm] = B.dm_of(np.minimum(col('a_' + nm), c)) - REFV[band]
    isb3 = B.med_per_star((det_r == 'nrcb3').astype(float))
    sfl = B.med_per_star(fl)
    print(f'\n### F{band}')
    for det in ('nrcb1', 'nrcb3'):
        q = B.good & (det_r == det) & np.isfinite(fl)
        print(f'{det}: rows {q.sum()}, rim flat p16/50/84 {np.percentile(fl[q], [16, 50, 84]).round(4)}')
    lo0, hi0 = BINS[band][0][0], BINS[band][-1][1]
    for nmq, v in (('nrcb1 only', 0), ('nrcb3 only', 1)):
        s = B.have0 & (isb3 == v) & np.isfinite(sfl) & np.isfinite(D['H']) & np.isfinite(D['Hf']) & (B.ref >= lo0) & (B.ref < 17)
        x = 2.5 * np.log10(sfl[s])
        out = []
        for nm in ('H', 'Hf'):
            # robust slope: Theil-Sen on dm vs 2.5 log10 flat, after removing the per-bin median
            y = D[nm][s].copy()
            for lo, hi in BINS[band]:
                b = (B.ref[s] >= lo) & (B.ref[s] < hi)
                if b.any():
                    y[b] -= np.median(y[b])
            from scipy.stats import theilslopes
            sl, ic, l1, l2 = theilslopes(y, x)
            out.append(f'{nm} slope {sl:+.2f} [{l1:+.2f},{l2:+.2f}]')
        print(f'{nmq} ({s.sum()} stars < 17 mag, star flat p16/50/84 {np.percentile(sfl[s], [16, 50, 84]).round(4)}): ' + '; '.join(out))
