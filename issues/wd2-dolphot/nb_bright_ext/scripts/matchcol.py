"""Matched-sample NB-broad colour vs dolphot: per dolphot NB-mag bin, median (our colour - dolphot colour), MAD, N,
using only stars where dolphot and the arm both have finite NB and broad magnitudes."""
import numpy as np
from astropy.table import Table
import analyze as an
PAIRS = [('164N', '162M'), ('187N', '182M'), ('212N', '200W'), ('405N', '410M'), ('466N', '410M')]
BINS = [(10, 12), (12, 13), (13, 14), (14, 15), (15, 18), (18, 20)]
arms = ['prod', 'mainfcbg', 'main2', 'main2kf']
ms = {a: Table.read(an.PATH[a][1]) for a in arms}
for nb, bb in PAIRS:
    print(f'\n### F{nb} - F{bb}: (our - dolphot) colour, median / MAD / N, per dolphot F{nb} bin')
    print('| arm | ' + ' | '.join(f'{lo}-{hi}' for lo, hi in BINS) + ' |')
    print('|---|' + '---|' * len(BINS))
    for a, m in ms.items():
        ok = np.asarray(m['matched'], bool)
        r1, r2 = an.fl(m['ref_' + nb]), an.fl(m['ref_' + bb])
        o1, o2 = an.fl(m['our_' + nb]), an.fl(m['our_' + bb])
        g = ok & np.isfinite(r1) & np.isfinite(r2) & np.isfinite(o1) & np.isfinite(o2)
        d = (o1 - o2) - (r1 - r2)
        cells = []
        for lo, hi in BINS:
            s = g & (r1 >= lo) & (r1 < hi)
            cells.append(f'{np.median(d[s]):+.3f} / {an.mad(d[s]):.3f} / {s.sum()}' if s.sum() >= 3 else f'— / — / {s.sum()}')
        print(f'| {a} | ' + ' | '.join(cells) + ' |')
