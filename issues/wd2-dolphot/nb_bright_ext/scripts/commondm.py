"""Per-band dm vs dolphot at the bright end on stars common to all arms, after subtracting each arm's own
15-18 mag median dm (unflagged+replaced together).  Rows of matched_*.fits are dolphot stars in the same order."""
import numpy as np
from astropy.table import Table
import analyze as an
BB = ['162M', '182M', '200W', '164N', '187N', '212N', '410M', '405N', '466N']
BINS = [(10, 12), (12, 13), (13, 14), (14, 15)]
arms = ['prod', 'mainfcbg', 'main2', 'main2kf']
ms = {a: Table.read(an.PATH[a][1]) for a in arms}
n = {len(m) for m in ms.values()}; assert len(n) == 1, n
print('| band | arm | ' + ' | '.join(f'{lo}-{hi}' for lo, hi in BINS) + ' | 15-18 zp |')
print('|---|---|' + '---|' * len(BINS) + '---|')
for b in BB:
    r = an.fl(ms['prod']['ref_' + b])
    o = {a: an.fl(m['our_' + b]) for a, m in ms.items()}
    com = np.isfinite(r) & np.all([np.asarray(m['matched'], bool) & np.isfinite(o[a]) for a, m in ms.items()], axis=0)
    for a in arms:
        d = o[a] - r
        zp = np.median(d[com & (r >= 15) & (r < 18)])
        cells = []
        for lo, hi in BINS:
            s = com & (r >= lo) & (r < hi)
            cells.append(f'{np.median(d[s]) - zp:+.3f} / {s.sum()}' if s.sum() >= 3 else f'— / {s.sum()}')
        print(f'| F{b} | {a} | ' + ' | '.join(cells) + f' | {zp:+.3f} |')
