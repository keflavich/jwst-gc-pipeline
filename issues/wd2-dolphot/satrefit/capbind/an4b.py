import numpy as np
from cb_lib import Band, mad
from an4 import bind_info, BINS, REFV
L=['| band | tau | '+' | '.join(f'bin{k}' for k in range(6))+' | max dev |','|---|---|'+'---|'*7]
for band in ('250M','300M','150W','200W'):
    B=Band(band); cap0,_=bind_info(B,'cutH0'); a=B.a_H_h0_bgfree
    c=cap0*B.rcor; c=np.where(np.isfinite(c),c,np.inf)
    for tau in (0.0,0.93,0.94,0.95,0.96,0.97,0.98):
        d=B.dm_of(np.maximum(np.minimum(a,c),tau*a)); have=B.have0&np.isfinite(d)
        cells=[];dev=[]
        for lo,hi in BINS[band]:
            s=have&(B.ref>=lo)&(B.ref<hi)
            if s.sum()>=5: m=np.median(d[s]); cells.append(f'{lo}-{hi}: {m:+.3f}'); dev.append(abs(m-REFV[band]))
        L.append(f'| F{band} | {tau} | '+' | '.join(cells)+f' | {max(dev):.3f} |')
open('an4b_tables.md','w').write('\n'.join(L)+'\n'); print('\n'.join(L))
