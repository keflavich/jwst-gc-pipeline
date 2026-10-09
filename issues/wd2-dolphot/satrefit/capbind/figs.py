import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from cb_lib import Band
from an4 import bind_info, BINS, REFV
fig,ax=plt.subplots(1,2,figsize=(11,4))
for a_,band in zip(ax,('250M','300M')):
    B=Band(band); cap0,_=bind_info(B,'cutH0'); a=B.a_H_h0_bgfree
    c=cap0*B.rcor; c=np.where(np.isfinite(c),c,np.inf)
    cen=[0.5*(lo+hi) for lo,hi in BINS[band]]
    for lab,d in (('uncapped',B.dm_of(a)),('round-7 cap',B.dm_of(np.minimum(a,c))),('floor tau=0.96',B.dm_of(np.maximum(np.minimum(a,c),0.96*a))),('final',B.dm_final)):
        h=B.have0&np.isfinite(d)
        a_.plot(cen,[np.median(d[h&(B.ref>=lo)&(B.ref<hi)]) for lo,hi in BINS[band]],'o-',label=lab)
    a_.axhline(REFV[band],c='k',ls=':'); a_.axhspan(REFV[band]-.02,REFV[band]+.02,color='gray',alpha=.15)
    a_.set_title(f'F{band} (H+h0+bgfree)'); a_.set_xlabel('dolphot mag'); a_.set_ylabel('median dm'); a_.legend(fontsize=8)
fig.tight_layout(); fig.savefig('fig1_cap_rules.png',dpi=120)
d=['1','2','3','4-5','6-8','9-12','13-20']; x=np.arange(7)
q={'sat 12.3-13':[1.144,1.107,1.071,1.029,0.995,0.994,0.986],'sat 13-14':[1.090,1.064,1.038,1.011,0.979,0.980,0.993],'sat 14-15':[1.045,1.014,0.995,0.971,0.951,0.954,0.976],'sat 15-17':[1.026,0.984,0.945,0.938,0.945,0.963,0.976],'control g0 1500-4000':[1.026,1.036,0.997,0.968,0.965,0.980,0.983],'control g0 4000-10000':[1.052,1.064,1.027,0.991,0.975,0.960,0.989]}
plt.figure(figsize=(6,4))
for k,v in q.items(): plt.plot(x,v,'o-' if 'sat' in k else 's--',label=k)
plt.xticks(x,d); plt.xlabel('distance to saturated region (px)'); plt.ylabel('q = cal/(R_h g0), F250M+F300M'); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig('fig2_wing_q.png',dpi=120)
