"""Paired per-star residual comparison of a fullfield_residual.py output: median chi^2/pix
in annuli around ISOLATED catalog stars (no source >10% of the star within 8 px, >30 px from
any saturated core of >=30 masked px), per flux bin, STPSF vs ePSF vs hybrid.

    python fullfield_paired.py <o081_<det>.npz> <out.json>     (also writes <out>.npz)
"""
import numpy as np, sys, json
from scipy import ndimage
from scipy.spatial import cKDTree
fn,out=sys.argv[1:3]
z=np.load(fn); n=int(z['ncat']); sci,err,g=z['sci'],z['err'],z['good']; ny,nx=sci.shape
bad=~g; lab,nl=ndimage.label(bad); sz=ndimage.sum(bad,lab,np.arange(1,nl+1))
big=np.isin(lab,np.nonzero(sz>=30)[0]+1); dsat=ndimage.distance_transform_edt(~big)
X=z['stpsf_x']; Y=z['stpsf_y']; F=z['stpsf_f']
T=cKDTree(np.c_[X,Y])
iso=np.array([F[j]>0 and not any(i!=j and F[i]>0.1*F[j] for i in T.query_ball_point([X[j],Y[j]],8)) for j in range(n)])
ix=np.round(X[:n]).astype(int); iy=np.round(Y[:n]).astype(int)
inb=(ix>15)&(ix<nx-16)&(iy>15)&(iy<ny-16)
sel=np.nonzero(iso&inb&(dsat[np.clip(iy,0,ny-1),np.clip(ix,0,nx-1)]>30))[0]
# also: the star's own core must be unmasked
sel=sel[g[iy[sel],ix[sel]]]
k=np.arange(-10,11); rr=np.hypot(k[:,None],k[None])
A=[(0,1.5),(1.5,3),(3,6),(6,10)]
ms=[m for m in ['stpsf','epsf','hybrid'] if m+'_resid' in z.files]
C={m:np.full((len(sel),len(A)),np.nan) for m in ms}
Fr={m:np.full((len(sel),len(A)),np.nan) for m in ms}
for q,j in enumerate(sel):
    s=np.s_[iy[j]-10:iy[j]+11, ix[j]-10:ix[j]+11]; u=g[s]
    for m in ms:
        r=z[m+'_resid'][s]; c2=(r/err[s])**2
        fj=z[m+'_f'][j]
        for a,(r0,r1) in enumerate(A):
            w=u&(rr>=r0)&(rr<r1)
            if w.sum(): C[m][q,a]=c2[w].mean(); Fr[m][q,a]=np.abs(r[w]).sum()/max(fj,1e-9)
f=F[sel]; qs=np.percentile(f,[0,50,80,95,100])
res=dict(n_sel=int(len(sel)),n_cat=n,flux_edges=qs.tolist(),annuli=A,bins=[])
print(f'{fn}: {len(sel)} isolated catalog stars (no >10% neighbour within 8 px, >30 px from saturated cores)')
for b in range(4):
    mm=(f>=qs[b])&(f<=qs[b+1]); row=dict(N=int(mm.sum()))
    line=f'flux pct {[0,50,80,95][b]:2d}-{[50,80,95,100][b]:3d} N={mm.sum():4d} |'
    for a,(r0,r1) in enumerate(A):
        vals={m:float(np.nanmedian(C[m][mm,a])) for m in ms}
        fr={m:float(np.nanmedian(Fr[m][mm,a])) for m in ms}
        win=float(np.nanmean(C['epsf'][mm,a]<C['stpsf'][mm,a]))
        row[f'{r0}-{r1}']=dict(chi2=vals,absres_frac=fr,epsf_better_frac=win)
        line+=f' r{r0}-{r1}: S {vals["stpsf"]:6.1f} E {vals["epsf"]:6.1f} H {vals.get("hybrid",np.nan):6.1f} ({100*win:3.0f}% E<S) |'
    print(line); res['bins'].append(row)
json.dump(res,open(out,'w'),indent=1)
np.savez(out.replace('.json','.npz'),sel=sel,f=f,**{f'C_{m}':C[m] for m in ms},**{f'F_{m}':Fr[m] for m in ms})
