"""dm (ours - dolphot) median / MAD per ref-mag bin, LW vs SW, matched_Q_<tag>.fits."""
import sys
import numpy as np
from astropy.table import Table

tag = sys.argv[1] if len(sys.argv) > 1 else 'mainfcbg'
t = Table.read(f'/orange/adamginsburg/jwst/wd2/dolphot_benchmark/matched_Q_{tag}.fits')
bands = ['150W', '200W', '212N', '250M', '277W', '300M', '335M', '410M', '323N', '405N', '466N']
edges = np.arange(12, 25, 1.0)
print(f'tag={tag}')
print('band  ' + ' '.join(f'{e:>11.0f}' for e in edges[:-1]))
for b in bands:
    r = np.asarray(t[f'ref_{b}'], float)
    o = np.asarray(t[f'our_{b}'], float)
    sat = np.asarray(t[f'our_is_saturated_{b}']).astype(bool) | np.asarray(t[f'our_replaced_saturated_{b}']).astype(bool)
    ok = np.isfinite(r) & np.isfinite(o) & ~sat & (r < 90) & (o < 90)
    dm = o - r
    cells = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = ok & (r >= lo) & (r < hi)
        if s.sum() < 20:
            cells.append(f'{"":>11}')
            continue
        d = dm[s]
        med = np.median(d)
        mad = 1.4826 * np.median(np.abs(d - med))
        cells.append(f'{med:+.3f}/{mad:.3f}')
    print(f'{b:5s} ' + ' '.join(cells))
