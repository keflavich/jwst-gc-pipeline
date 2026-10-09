import numpy as np
from cb_lib import Band, mad
BINS=[(12.3,13),(13,13.5),(13.5,14),(14,15),(15,16),(16,17)]
TAGS=['g10','g20','g30','g50','g70','f10','f20','f30','f50']
L=[]
for band in ('250M','300M'):
    B=Band(band)
    rr=np.full(B.nrow,np.nan); rr[B.j]=B.ref[B.i]
    a0=B.a_H_h0_bgfree
    L+=[f'### F{band}: median calibrated a_g0 / a_(H+h0+bgfree) by bin (calibrated on 15-17 mag); dm of a_g0/c per star',
        '','| variant | median n px | '+' | '.join(f'{lo}-{hi}' for lo,hi in BINS)+' | dm 12.3-13 | dm 13-14 |','|---|---|'+'---|'*(len(BINS)+2)]
    for t in TAGS:
        ag=getattr(B,'a_'+t); n=getattr(B,'n_'+t)
        ok=B.good&np.isfinite(rr)&np.isfinite(ag)&(ag>0)&np.isfinite(a0)&(a0>0)
        cal=ok&(rr>=15)&(rr<17)
        if cal.sum()<20: L.append(f'| {t} | - |'+' - |'*(len(BINS)+2)); continue
        c=np.median((ag/a0)[cal]); dm=B.dm_of(ag/c)
        cells=[]
        for lo,hi in BINS:
            rs=ok&(rr>=lo)&(rr<hi); cells.append(f'{np.median((ag/a0)[rs])/c:.3f} (n={int(rs.sum())})' if rs.sum()>=5 else '-')
        d=[]
        for lo,hi in ((12.3,13),(13,14)):
            s=B.have0&np.isfinite(dm)&(B.ref>=lo)&(B.ref<hi); d.append(f'{np.median(dm[s]):+.3f}')
        L.append(f'| {t} | {np.median(n[ok]):.0f} | '+' | '.join(cells)+' | '+' | '.join(d)+' |')
    L.append('')
open('an3b_tables.md','w').write('\n'.join(L)+'\n'); print('\n'.join(L))
