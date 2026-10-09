"""Per-detector median of the recentre_ab score (variant ni, nf) and the rstd after
removing those medians, per band.  Stars are assigned to the detector holding most of
their frames.

usage: python rc_detmed.py BAND [BAND ...]
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
    S = Table.read(f'{D}/stars_{b}.ecsv')
    S = S[np.isin(S['i'], R['i'])]
    idx = {k: n for n, k in enumerate(S['i'])}
    row = np.array([idx[k] for k in R['i']])
    dets = np.unique(R['det'])
    cnt_det = np.array([np.bincount(row[R['det'] == d], minlength=len(S)) for d in dets])
    sdet = dets[np.argmax(cnt_det, axis=0)]
    print(f'F{b}')
    for v in ('oi', 'ni', 'of', 'nf'):
        num = np.bincount(row, weights=R[f'f_{v}'] / R['fmain'], minlength=len(S))
        cnt = np.bincount(row, minlength=len(S))
        sc = np.asarray(S['dm']) - 2.5 * np.log10(num / cnt) - np.asarray(S['pred'])
        med = {d: np.nanmedian(sc[sdet == d]) for d in dets}
        res = sc - np.array([med[d] for d in sdet])
        print(f'  {v}: rstd {mad(sc):.4f}, after per-detector medians {mad(res):.4f}; medians '
              + ' '.join(f'{d[3:]} {med[d]:+.4f}' for d in dets))
