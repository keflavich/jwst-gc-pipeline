"""dm vs pixel-area pred slope for satstar-replaced rows, unsaturated rows in the ZP window, and unsaturated rows by
dolphot magnitude bin (main2).  Uses areapred_main2.npz.
usage: python satslope.py > satslope.txt"""
import sys
import numpy as np
from scipy import stats
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an

an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2')
P = np.load(f'{an.Q}/photomver/areapred_main2.npz')


def ts(dm, pred, m):
    m = m & np.isfinite(dm) & np.isfinite(pred)
    if m.sum() < 40:
        return ''
    s = stats.theilslopes(dm[m], pred[m])
    return f'{s[0]:+.2f} [{s[2]:+.2f},{s[3]:+.2f}] ({m.sum()})'


BINS = [(13, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22)]
print('Theil-Sen slope of dm (ours - dolphot - ZP) against pred [95% CI] (N)\n')
print('| band | sat-replaced all | unsat ZP window | ' + ' | '.join(f'unsat {lo}-{hi}' for lo, hi in BINS) + ' |')
print('|---' * (3 + len(BINS)) + '|')
for band in an.BANDS:
    dm, pred = A.dm(band), P[band]
    base = A.matched & np.isfinite(A.ref[band])
    lo, hi = an.ZPWIN.get(band, (0, 19))
    uns = base & ~A.rep[band] & ~A.sat[band]
    cells = [ts(dm, pred, base & A.rep[band]), ts(dm, pred, uns & (A.ref[band] >= lo) & (A.ref[band] < hi))]
    cells += [ts(dm, pred, uns & (A.ref[band] >= a) & (A.ref[band] < b)) for a, b in BINS]
    print(f'| F{band} | ' + ' | '.join(cells) + ' |')
