"""Frame-to-frame repeatability of the recentre_ab fluxes: for stars measured on 4
frames, robust std over frames of -2.5 log10(f_X) (per-frame noise) and that over 2
(noise of the 4-frame mean), against the catalogue flux_err of the same rows.

usage: python rc_repeat.py BAND [BAND ...]
"""
import glob
import sys

import numpy as np
from astropy.table import Table, vstack

sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an  # noqa: E402

D = f'{an.Q}/gridfix'


def mad(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x)))


for b in sys.argv[1:]:
    R = vstack([Table.read(f) for f in sorted(glob.glob(f'{D}/recentre/rc_{b}_*.ecsv'))])
    out = []
    for v in ('oi', 'ni', 'nf'):
        m = -2.5 * np.log10(np.asarray(R[f'f_{v}'], float))
        ids, inv, cnt = np.unique(R['i'], return_inverse=True, return_counts=True)
        mean = np.bincount(inv, weights=m) / cnt
        dev = (m - mean[inv])[cnt[inv] == 4] * np.sqrt(4 / 3)
        out.append(f'{v} per-frame {mad(dev):.4f} -> 4-frame mean {mad(dev) / 2:.4f}')
    print(f'F{b}: ' + '; '.join(out) + f'  (N rows {len(R)})')
