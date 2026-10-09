import numpy as np
from astropy.table import Table
e = Table.read('/orange/adamginsburg/jwst/wd2/catalogs/wd2_nircam_wf_mf_nf.ecsv')
def m(b):
    x = np.asarray(e['MAG' + b].filled(np.nan), float); x[np.abs(x) > 50] = np.nan; return x
M = {b: m(b) for b in ('115W','150W','200W','250M','277W','300M','335M','410M')}
for n in M: print(n, 'N finite', np.isfinite(M[n]).sum(), 'min', np.nanmin(M[n]), 'p1', np.nanpercentile(M[n],1))
sl = M['335M'] - M['410M']
for lw, nb in (('250M','300M'), ('250M','335M'), ('300M','335M'), ('300M','410M')):
    mm = M[lw]; c = mm - M[nb]
    print(f'\n{lw}-{nb} by {lw} bin in slices of F335M-F410M (N)  [bins 12.5-13 | 13-13.5 | 13.5-14 | 14-15 | 15-16]')
    for lo, hi in [(-1,0.0),(0.0,0.1),(0.1,0.2),(0.2,0.4),(0.4,2)]:
        cells=[]
        for a,b in [(12.5,13),(13,13.5),(13.5,14),(14,15),(15,16)]:
            s=(mm>=a)&(mm<b)&(sl>=lo)&(sl<hi)&np.isfinite(c)
            cells.append(f'{np.median(c[s]):+.3f}({s.sum()})' if s.sum()>=4 else '   -   ')
        print(f' F335M-F410M {lo}..{hi}: '+' | '.join(cells))
    s=(mm>=12.5)&(mm<13)&np.isfinite(c); print('  no slicing 12.5-13:', np.median(c[s]) if s.any() else None, s.sum())
