import numpy as np, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from exposure import Exposure
k=int(sys.argv[1]); tag=sys.argv[2]
J1=np.load('tgt_e1.npz')['J']; ex=Exposure(f'tgt_e{k}.npz',J1)
z=np.load(f'{tag}_e{k}.npz'); pred=z['pred']; cov=z['cov']; varQ=z['varQ']
chic=(ex.d-pred)/np.sqrt(ex.var+varQ)
m=ex.good&(cov>=3)&np.isfinite(chic)
dq=ex.dq
yy,xx=np.mgrid[:1024,:1024]; r=np.hypot(xx-ex.xt0,yy-ex.yt0)
def line(lbl,mm):
    c=chic[mm]; return f'{lbl:22s} n={mm.sum():7d} mean={np.mean(c):+.3f} rstd={1.4826*np.median(np.abs(c-np.median(c))):.3f} chi2={np.mean(c**2):7.3f} f|>5|={np.mean(np.abs(c)>5):.4f}'
print(line('ALL',m))
clean=m&((dq&6)==0)
print(line('no SAT/JUMP flag',clean))
print(line('SAT-flagged',m&((dq&2)>0))); print(line('JUMP-flagged',m&((dq&4)>0)&((dq&2)==0)))
for lo,hi in [(0,50),(50,80),(80,120),(120,200),(200,300),(300,500),(500,800)]:
    print(line(f'clean r {lo}-{hi}',clean&(r>=lo)&(r<hi)))
for lo,hi in [(0,50),(50,100),(100,300),(300,1000),(1000,1e5)]:
    print(line(f'clean pred {lo}-{hi}',clean&(pred>=lo)&(pred<hi)))
fig,ax=plt.subplots(1,2,figsize=(24,12))
ax[0].imshow(np.where(m,chic,np.nan),origin='lower',cmap='RdBu_r',vmin=-5,vmax=5); ax[0].set_title(f'{tag} LOO chi e{k}')
ax[1].imshow(np.where(m,chic,np.nan)[362:662,362:662],origin='lower',cmap='RdBu_r',vmin=-5,vmax=5)
plt.tight_layout(); plt.savefig(f'diag_{tag}_{k}.png',dpi=35)
