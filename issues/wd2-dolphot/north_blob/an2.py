import numpy as np
from an import *
from astropy.table import Table
from scipy.ndimage import uniform_filter
mag=fl(ca['mag_vega_f150w'])
out={}
for b in ['150W','162M','182M','200W']:
    lam=int(b[:3])/100
    has,si,x,y,A,B=offs(b)
    out[b]=(has,si,x,y,A,B)
    for R in [0.10,0.15,0.20]:
        c=np.array([0,0.45*lam])
        for _ in range(10):
            w=np.hypot(x-c[0],y-c[1])<R; wm=np.hypot(x+c[0],y+c[1])<R
            wt=w.sum()-wm.sum()
            c=np.array([(x[w].sum()+x[wm].sum())/wt,(y[w].sum()+y[wm].sum())/wt])
        print(f'F{b} R={R}: mirror-sub centroid arcsec=({c[0]:+.3f},{c[1]:+.3f}) "/um=({c[0]/lam:+.3f},{c[1]/lam:+.3f}) net={wt}')
    e=np.arange(-1.2,1.2+1e-9,0.05); h,_,_=np.histogram2d(x,y,bins=[e,e])
    hs=uniform_filter(h,3,mode='constant'); hm=hs-hs[::-1,::-1]
    cc=0.5*(e[1:]+e[:-1]); X,Y=np.meshgrid(cc,cc,indexing='ij')
    w=(np.abs(X)<0.4)&(Y>0.4)&(Y<1.2)
    k=np.argmax(np.where(w,hm,-9)); print(f'   hist peak (0.15" boxcar, mirror-sub) arcsec=({X.flat[k]:+.3f},{Y.flat[k]:+.3f})')
np.savez('offs.npz',**{f'{b}_{n}':v for b,t in out.items() for n,v in zip(['has','si','x','y','A','B'],t)})
has,si,x,y,A,B=out['150W']
c=np.array([0.05,0.69])
sel=(np.hypot(x-c[0],y-c[1])<0.08*1.5*1.5)&~A&~B
print('task1 selection:',sel.sum())
t=Table(); t['dolphot_row']=has[sel]; t['RA']=np.asarray(ma['RA'])[has[sel]]; t['DEC']=np.asarray(ma['DEC'])[has[sel]]
t['sat_row']=si[sel]; t['dx_arcsec']=x[sel]; t['dy_arcsec']=y[sel]; t['sat_mag150']=mag[si[sel]]; t['ref_150W']=fl(ma['ref_150W'])[has[sel]]
t.write('/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/kf_lost/north_blob_f150w_unmatched.ecsv',format='ascii.ecsv',overwrite=True)
print('ref mag median in list',np.nanmedian(t['ref_150W']),'all',np.nanmedian(fl(ma['ref_150W'])[has]))
smag=mag[si]
sat_all=np.ma.filled(ca['replaced_saturated_f150w'],0).astype(bool)
print('sat mag pct',np.nanpercentile(mag[sat_all],[0,10,50,90,100]))
for lo,hi in [(-99,12),(12,14),(14,99)]:
    m=(smag>=lo)&(smag<hi); sm=sat_all&(mag>=lo)&(mag<hi)
    R=0.12
    inb=m&(np.hypot(x-c[0],y-c[1])<R); mir=m&(np.hypot(x+c[0],y+c[1])<R)
    print(f'mag {lo}-{hi}: nsat={sm.sum()} blob {inb.sum()} mirror {mir.sum()} excess/sat {(inb.sum()-mir.sum())/max(sm.sum(),1):.3f}; neither blob {(inb&~A&~B).sum()} mirror {(mir&~A&~B).sum()}')
