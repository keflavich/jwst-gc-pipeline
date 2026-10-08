"""Eddington check: bin LW dm by (a) ref mag, (b) our mag, (c) mean of both, (d) ref F212N mag (independent band)."""
import sys
import numpy as np
from astropy.table import Table

tag = sys.argv[1] if len(sys.argv) > 1 else 'mainfcbg'
t = Table.read(f'/orange/adamginsburg/jwst/wd2/dolphot_benchmark/matched_Q_{tag}.fits')
edges = np.arange(15, 24, 1.0)
bands = ['200W', '277W', '300M', '335M', '410M', '405N', '466N']
r212 = np.asarray(t['ref_212N'], float)
for b in bands:
    r = np.asarray(t[f'ref_{b}'], float)
    o = np.asarray(t[f'our_{b}'], float)
    sat = np.asarray(t[f'our_is_saturated_{b}']).astype(bool) | np.asarray(t[f'our_replaced_saturated_{b}']).astype(bool)
    ok = np.isfinite(r) & np.isfinite(o) & ~sat & (r < 90) & (o < 90)
    print(f'== {b}  (median dm per bin)')
    print('bin by    ' + ' '.join(f'{e:>7.0f}' for e in edges[:-1]))
    for lab, x in (('ref', r), ('ours', o), ('mean', 0.5 * (o + r)), ('ref212N', r212)):
        cells = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = ok & (x >= lo) & (x < hi) & np.isfinite(x)
            cells.append(f'{np.median(o[s]-r[s]):+7.3f}' if s.sum() >= 20 else f'{"":>7}')
        print(f'{lab:9s} ' + ' '.join(cells))
