"""Narrow - medium colour vs medium-band mag, ours vs dolphot. Photospheric NB-MB colours are ~flat with mag;
an additive bkg error hits the narrow band harder and makes the colour drift at the faint end."""
import sys
import numpy as np
from astropy.table import Table

tag = sys.argv[1] if len(sys.argv) > 1 else 'mainfcbg'
t = Table.read(f'/orange/adamginsburg/jwst/wd2/dolphot_benchmark/matched_Q_{tag}.fits')
edges = np.arange(14, 23, 1.0)
pairs = [('405N', '410M'), ('466N', '410M'), ('323N', '335M'), ('212N', '200W'), ('187N', '182M'), ('164N', '162M')]

def good(b, pre):
    m = np.asarray(t[f'{pre}_{b}'], float)
    ok = np.isfinite(m) & (m < 90)
    if pre == 'our':
        sat = np.asarray(t[f'our_is_saturated_{b}']).astype(bool) | np.asarray(t[f'our_replaced_saturated_{b}']).astype(bool)
        ok &= ~sat
    return m, ok

print(f'tag={tag}; cells = median colour (ours / dolphot), binned by mean medium-band mag')
print('pair        ' + ' '.join(f'{e:>13.0f}' for e in edges[:-1]))
for nb, mb in pairs:
    on, okon = good(nb, 'our')
    om, okom = good(mb, 'our')
    rn, okrn = good(nb, 'ref')
    rm, okrm = good(mb, 'ref')
    ok = okon & okom & okrn & okrm
    x = 0.5 * (om + rm)
    cells = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = ok & (x >= lo) & (x < hi)
        if s.sum() < 20:
            cells.append(f'{"":>13}')
            continue
        cells.append(f'{np.median(on[s]-om[s]):+.3f}/{np.median(rn[s]-rm[s]):+.3f}')
    print(f'{nb}-{mb:5s} ' + ' '.join(cells))
