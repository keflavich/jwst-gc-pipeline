import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
D='/orange/adamginsburg/jwst/wd2/dolphot_benchmark'; Q=f'{D}/Q_integ'
M8='catalogs/basic_merged_indivexp_photometry_tables_merged_resbgsub_m8_dedup.fits'
ma=Table.read(f'{D}/matched_Q_mainfcbg.fits'); mb=Table.read(f'{D}/matched_Q_mainfcbgkf.fits')
ca=Table.read(f'{Q}/tree_mainfcbg/{M8}')
def fl(x): return np.ma.filled(x,np.nan).astype(float) if np.ma.isMaskedArray(x) else np.asarray(x,float)
ref=SkyCoord(ma['RA'],ma['DEC'],unit='deg')
def offs(b, satmask=None):
    s=np.ma.filled(ca[f'replaced_saturated_f{b.lower()}'],0).astype(bool)
    if satmask is not None: s&=satmask
    sat=SkyCoord(ca['skycoord_ref'][s])
    has=np.where(np.isfinite(fl(ma[f'ref_{b}'])))[0]
    i,sep,_=ref[has].match_to_catalog_sky(sat)
    dra,dd=sat[i].spherical_offsets_to(ref[has])
    inA=np.asarray(ma['matched'])[has]&np.isfinite(fl(ma[f'our_{b}']))[has]
    inB=np.asarray(mb['matched'])[has]&np.isfinite(fl(mb[f'our_{b}']))[has]
    return has,np.where(s)[0][i],dra.arcsec,dd.arcsec,inA,inB
def centroid(x,y,c0,r,n=8):
    c=np.array(c0)
    for _ in range(n):
        m=np.hypot(x-c[0],y-c[1])<r
        c=np.array([x[m].mean(),y[m].mean()])
    return c,m.sum()
if __name__=='__main__':
    for b in ['150W','162M','182M','200W']:
        lam=int(b[:3])/100
        has,si,x,y,A,B=offs(b)
        # 2D hist peak, 0.04 "/um bins smoothed, minus mirror
        r=0.08*lam
        c0=(0,0.45*lam)
        c,n=centroid(x,y,c0,r*1.5)
        c2,n2=centroid(x[~A&~B],y[~A&~B],c0,r*1.5)
        # hist peak
        e=np.arange(-1.2,1.2+1e-9,0.04*lam); h,_,_=np.histogram2d(x,y,bins=[e,e])
        from scipy.ndimage import uniform_filter
        hs=uniform_filter(h,2)
        # mirror-subtract
        hm=hs-hs[::-1,::-1]
        cc=0.5*(e[1:]+e[:-1]); X,Y=np.meshgrid(cc,cc,indexing='ij')
        w=(np.hypot(X,Y)>0.2*lam)&(np.hypot(X,Y)<1.2*lam)&(Y>0)
        k=np.argmax(np.where(w,hm,-9)); 
        inb=np.hypot(x-c[0],y-c[1])<0.08*lam; mir=np.hypot(x+c[0],y+c[1])<0.08*lam
        print(f'F{b} lam={lam}: centroid(all) arcsec=({c[0]:+.3f},{c[1]:+.3f}) n={n}; "/um=({c[0]/lam:+.3f},{c[1]/lam:+.3f}); neither-only centroid arcsec=({c2[0]:+.3f},{c2[1]:+.3f}); histpeak(mirror-sub) arcsec=({X.flat[k]:+.3f},{Y.flat[k]:+.3f}); circle {inb.sum()} mirror {mir.sum()} excess {inb.sum()-mir.sum()}; neither {(inb&~A&~B).sum()} vs {(mir&~A&~B).sum()}')
