import numpy as np
from astropy.table import Table
import analyze as an
PAIRS = [('f164n', 'f162m'), ('f187n', 'f182m'), ('f212n', 'f200w'), ('f405n', 'f410m'), ('f466n', 'f410m')]
BINS = [(10, 12), (12, 13), (13, 14), (15, 18)]
arms = ['prod', 'mainfcbg', 'main2', 'main2kf']
cats = {a: Table.read(an.PATH[a][0]) for a in arms}
for nb, bb in PAIRS:
    print(f'\n{nb}-{bb}: raw median colour (N) per {nb} bin')
    for a, t in cats.items():
        m1 = an.fl(t[f'mag_vega_{nb}']); m2 = an.fl(t[f'mag_vega_{bb}'])
        ok = np.isfinite(m1) & np.isfinite(m2); c = m1 - m2
        cells = []
        for lo, hi in BINS:
            s = ok & (m1 >= lo) & (m1 < hi)
            cells.append(f'{lo}-{hi}: {np.median(c[s]):+.3f} ({s.sum()})')
        print(f'  {a:9s} ' + '  '.join(cells))
