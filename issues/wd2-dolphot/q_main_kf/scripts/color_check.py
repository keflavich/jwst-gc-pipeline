"""Narrowband - adjacent-band colour of bright stars, per arm, as a check that does not need dolphot
(dolphot lacks or saturates many of the brightest stars).  Pairs: F164N-F162M, F187N-F182M, F212N-F200W,
F323N-F335M (?), F405N-F410M, F466N-F410M (colour offsets are set from 15-18 mag stars per arm).
For narrowband mag bins, N stars with both magnitudes finite, median colour offset (after subtracting
the 15-18 mag median), and the fraction with |offset| > 0.3 and > 1.
usage: python color_check.py ARM [ARM ...]   (ARM: prod or a Q_integ arm)"""
import sys
import numpy as np
from astropy.table import Table
import analyze as an

PAIRS = [('f164n', 'f162m'), ('f187n', 'f182m'), ('f212n', 'f200w'), ('f405n', 'f410m'), ('f466n', 'f410m')]
BINS = [(6, 10), (10, 12), (12, 13), (13, 14), (14, 15)]
arms = sys.argv[1:]
cats = {a: Table.read(an.PATH[a][0]) for a in arms}
for nb, bb in PAIRS:
    print(f'\n### {nb.upper()} - {bb.upper()}: per {nb.upper()} mag bin, N / median offset / f>0.3 / f>1 (offset = colour - 15-18 mag median)')
    print('| arm | ' + ' | '.join(f'{lo}-{hi}' for lo, hi in BINS) + ' |')
    print('|---|' + '---|' * len(BINS))
    for a, t in cats.items():
        m1 = an.fl(t[f'mag_vega_{nb}']); m2 = an.fl(t[f'mag_vega_{bb}'])
        ok = np.isfinite(m1) & np.isfinite(m2)
        c = m1 - m2
        ref = ok & (m1 >= 15) & (m1 < 18)
        c0 = np.median(c[ref]) if ref.sum() > 20 else np.nan
        cells = []
        for lo, hi in BINS:
            s = ok & (m1 >= lo) & (m1 < hi)
            d = c[s] - c0
            cells.append(f'{s.sum()} / {np.median(d):+.3f} / {np.mean(np.abs(d) > 0.3):.3f} / {np.mean(np.abs(d) > 1):.3f}' if s.sum() else '0')
        print(f'| {a} | ' + ' | '.join(cells) + ' |')
