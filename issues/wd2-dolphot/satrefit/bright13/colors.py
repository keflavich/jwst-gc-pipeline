import numpy as np
from astropy.table import Table
e = Table.read('/orange/adamginsburg/jwst/wd2/catalogs/wd2_nircam_wf_mf_nf.ecsv')
def m(b):
    x = np.asarray(e['MAG' + b].filled(np.nan) if hasattr(e['MAG' + b], 'filled') else e['MAG' + b], float); x[np.abs(x) > 50] = np.nan; return x
def er(b):
    x = np.asarray(e['ERRMAG' + b].filled(np.nan) if hasattr(e['ERRMAG' + b], 'filled') else e['ERRMAG' + b], float); x[np.abs(x) > 50] = np.nan; return x
print(len(e))
for b in ('250M', '300M'):
    mm = m(b); ee = er(b)
    print('---', b)
    for lo, hi in [(10,12),(12,12.5),(12.5,13),(13,13.5),(13.5,14),(14,15),(15,16)]:
        s = (mm>=lo)&(mm<hi)
        if s.sum()==0: print(lo,hi,'N=0'); continue
        print(f'{lo}-{hi}: N={s.sum()} err med {np.nanmedian(ee[s]):.4f} p90 {np.nanpercentile(ee[s],90):.4f} max {np.nanmax(ee[s]):.4f}; frac err==0 {np.mean(ee[s]==0):.2f}')
# colours: LW - neighbours as function of mag, in SW colour slices
b1, b2 = '250M', '300M'
F115, F150, F200, F277, F335, F410 = (m(x) for x in ('115W', '150W', '200W', '277W', '335M', '410M'))
sw = F150 - F200
for lw, nb in (('250M', None), ('300M', None)):
    mm = m(lw)
    # colours vs neighbours: F277W - F250M, F300M - F335M ...
cols = {'250M': [('277W', F277), ('200W', F200), ('300M', m('300M'))], '300M': [('277W', F277), ('335M', F335), ('250M', m('250M'))]}
for lw in ('250M', '300M'):
    mm = m(lw)
    for nm, x in cols[lw]:
        print(f'\n{lw} - {nm}, by LW bin, in SW-colour (F150W-F200W) slices; median colour (N)')
        for slo, shi in [(-1,0.3),(0.3,0.6),(0.6,1.0),(1.0,2)]:
            cells = []
            for lo, hi in [(11,12),(12,12.5),(12.5,13),(13,13.5),(13.5,14),(14,15)]:
                s = (mm>=lo)&(mm<hi)&(sw>=slo)&(sw<shi)&np.isfinite(x)
                cells.append(f'{np.median((mm-x)[s]):+.3f}({s.sum()})' if s.sum()>=5 else '   -   ')
            print(f'  F150W-F200W {slo}..{shi}: ' + ' | '.join(cells))
print('bins: 11-12 | 12-12.5 | 12.5-13 | 13-13.5 | 13.5-14 | 14-15')
