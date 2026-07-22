#!/usr/bin/env python3
"""PROTOTYPE: forced photometry of close-pair companions at external (SF/peppar) seeds.

The companions jicama misses are faint secondaries on a brighter primary's wing that never
peak -> daofind can't seed them. But at FIXED positions PSF flux is LINEAR, so a joint
primary+companion fit is a 2-parameter linear least-squares (no degeneracy unless the pair
is unresolved). This measures how many of the ~4k companions are cleanly recoverable vs
deblend-degenerate -- the honest close-pair completeness ceiling.

Seeds: companion position from STARFINDER (∩peppar), primary from jicama. Data: F212N NRCA4
mosaic. Model: stpsf F212N NRCA4 PSF. Gate: companion flux S/N + joint-fit reduced chi2.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord; import astropy.units as u
from astropy.io import fits; from astropy.wcs import WCS
from scipy.spatial import cKDTree
from scipy.ndimage import shift as ndshift
import os
BM='/orange/adamginsburg/jwst/benchmark-team'
RA0,DEC0=266.47699800418224,-28.856142119344067; COSD=np.cos(np.radians(DEC0))
MOS='/orange/adamginsburg/jwst/arches/F212N/pipeline/jw02045-o001_t001_nircam_clear-f212n-nrca_data_i2d.fits'

def off(a,b,c,d):
    xa=(a-RA0)*COSD*3600;ya=(b-DEC0)*3600;xb=(c-RA0)*COSD*3600;yb=(d-DEC0)*3600
    dd,i=cKDTree(np.c_[xb,yb]).query(np.c_[xa,ya],k=1);n=dd<0.5
    dx=(xb[i[n]]-xa[n])*1000;dy=(yb[i[n]]-ya[n])*1000
    H,xe,ye=np.histogram2d(dx,dy,bins=100,range=[[-400,400],[-400,400]]);iy,ix=np.unravel_index(H.argmax(),H.shape)
    return 0.5*(xe[iy]+xe[iy+1]),0.5*(ye[ix]+ye[ix+1])

# ---- seeds: missed companions (SF∩peppar not jicama) + nearest jicama primary ----
sf=Table.read(f'{BM}/starfinder/01_nrca4_stars_calib.txt',format='ascii')
SRA,SDEC=np.asarray(sf['ra']),np.asarray(sf['dec'])
ramin,ramax=SRA.min()-1/3600,SRA.max()+1/3600;demin,demax=SDEC.min()-1/3600,SDEC.max()+1/3600
pe=Table.read(f'{BM}/peppar/L2_20260629/combo_starlist_F212N_NRCA4.fits')
PRA=RA0+np.asarray(pe['x_wcs'],float)/3600/COSD;PDEC=DEC0+np.asarray(pe['y_wcs'],float)/3600
m=(PRA>ramin)&(PRA<ramax)&(PDEC>demin)&(PDEC<demax);PRA,PDEC=PRA[m],PDEC[m]
ji=Table.read(f'{BM}/jicama/nrca4_f212n_catalog.fits');JS=ji['skycoord'];JRA=JS.ra.deg.copy();JDEC=JS.dec.deg.copy()
ox,oy=off(PRA,PDEC,SRA,SDEC);PRA+=ox/1000/3600/COSD;PDEC+=oy/1000/3600
ox,oy=off(JRA,JDEC,SRA,SDEC);JRA+=ox/1000/3600/COSD;JDEC+=oy/1000/3600
SC=SkyCoord(SRA*u.deg,SDEC*u.deg);PC=SkyCoord(PRA*u.deg,PDEC*u.deg);JC=SkyCoord(JRA*u.deg,JDEC*u.deg)
R=0.08*u.arcsec
_,sp,_=SC.match_to_catalog_sky(PC);_,sj,_=SC.match_to_catalog_sky(JC)
miss=(sp<R)&(sj>=R)
# primary = nearest jicama source to the miss; keep field misses (companion, primary <0.4")
ij,dj,_=SC.match_to_catalog_sky(JC)
comp = miss & (dj<0.4*u.arcsec) & (dj>0.03*u.arcsec)   # a jicama primary 30-400 mas away
crd_c=SC[comp]; crd_p=JC[ij[comp]]
print(f"missed companions with a jicama primary 30-400mas away: {comp.sum()}")

# ---- PSF (stpsf F212N NRCA4, oversample handled via fine grid + subpixel shift) ----
os.environ.setdefault('STPSF_PATH','/orange/adamginsburg/repos/webbpsf/data')
import stpsf
nrc=stpsf.NIRCam(); nrc.filter='F212N'; nrc.detector='NRCA4'
psf=nrc.calc_psf(fov_pixels=21, oversample=1)['DET_DIST' if False else 0].data
psf=psf/psf.sum(); ph=psf.shape[0]//2
def template(dx,dy,shape,cx,cy):
    """PSF centered at (cx+dx, cy+dy) in a cutout of given shape (dx,dy subpixel)."""
    t=np.zeros(shape)
    y0=int(round(cy+dy))-ph; x0=int(round(cx+dx))-ph
    sub=ndshift(psf,( (cy+dy)-round(cy+dy), (cx+dx)-round(cx+dx) ),order=1,mode='constant')
    ys,xs=max(0,y0),max(0,x0); ye,xe=min(shape[0],y0+psf.shape[0]),min(shape[1],x0+psf.shape[1])
    if ye<=ys or xe<=xs: return t
    t[ys:ye,xs:xe]=sub[ys-y0:ye-y0,xs-x0:xe-x0]; return t

h=fits.open(MOS);sci=h['SCI'].data.astype(float);err=h['ERR'].data.astype(float);w=WCS(h['SCI'].header)
rng=np.random.default_rng(0); idx=rng.permutation(comp.sum())[:1500]
res=[]
for k in idx:
    rc,dc=crd_c[k].ra.deg,crd_c[k].dec.deg; rp,dp=crd_p[k].ra.deg,crd_p[k].dec.deg
    xc,yc=w.world_to_pixel_values(rc,dc); xp,yp=w.world_to_pixel_values(rp,dp)
    xc,yc,xp,yp=float(xc),float(yc),float(xp),float(yp)
    cx,cy=int(round((xc+xp)/2)),int(round((yc+yp)/2)); rad=8
    if not(rad<=cy<sci.shape[0]-rad and rad<=cx<sci.shape[1]-rad): continue
    cut=sci[cy-rad:cy+rad+1,cx-rad:cx+rad+1]; ecut=err[cy-rad:cy+rad+1,cx-rad:cx+rad+1]
    if not np.all(np.isfinite(cut)) or not np.all(np.isfinite(ecut)) or (ecut<=0).any(): continue
    sh=cut.shape; lc=(cx-rad,cy-rad)
    Tp=template(xp-lc[0]-round(xp-lc[0]) + (round(xp-lc[0])), 0,sh,0,0)  # placeholder
    # build templates directly at cutout-local positions
    def tmpl(xg,yg):
        t=np.zeros(sh); yy=int(round(yg))-ph; xx=int(round(xg))-ph
        sub=ndshift(psf,(yg-round(yg),xg-round(xg)),order=1,mode='constant')
        ys,xs=max(0,yy),max(0,xx);ye,xe=min(sh[0],yy+psf.shape[0]),min(sh[1],xx+psf.shape[1])
        if ye<=ys or xe<=xs: return None
        t[ys:ye,xs:xe]=sub[ys-yy:ye-yy,xs-xx:xe-xx]; return t
    Tp=tmpl(xp-lc[0],yp-lc[1]); Tc=tmpl(xc-lc[0],yc-lc[1])
    if Tp is None or Tc is None: continue
    A=np.c_[Tp.ravel(),Tc.ravel(),np.ones(cut.size)]; wgt=1/ecut.ravel()
    Aw=A*wgt[:,None]; bw=cut.ravel()*wgt
    try: coef,_,_,_=np.linalg.lstsq(Aw,bw,rcond=None); cov=np.linalg.inv(Aw.T@Aw)
    except np.linalg.LinAlgError: continue
    fp,fc,bg=coef; sig_c=np.sqrt(max(cov[1,1],0))
    model=A@coef; chi2=np.sum(((cut.ravel()-model)*wgt)**2)/(cut.size-3)
    sep=crd_c[k].separation(crd_p[k]).to(u.arcsec).value*1000
    res.append((fc/sig_c if sig_c>0 else 0, chi2, sep, fc, fp, fc/fp if fp>0 else np.nan))
res=np.array(res)
snr,chi2,sep,fc,fp,ratio=res.T
rec=(snr>=5)&(chi2<3)&(fc>0)
print(f"\nforced 2-component fits: N={len(res)}")
print(f"  companion S/N>=5 & chi2<3 & flux>0 (RECOVERABLE): {rec.sum()} ({100*rec.mean():.0f}%)")
print(f"  by separation: <70mas {100*rec[sep<70].mean() if (sep<70).sum() else 0:.0f}% (N{(sep<70).sum()}) | 70-100 {100*rec[(sep>=70)&(sep<100)].mean() if ((sep>=70)&(sep<100)).sum() else 0:.0f}% | 100-150 {100*rec[(sep>=100)&(sep<150)].mean() if ((sep>=100)&(sep<150)).sum() else 0:.0f}% | >150 {100*rec[sep>=150].mean() if (sep>=150).sum() else 0:.0f}%")
print(f"  median companion/primary flux ratio (recovered): {np.nanmedian(ratio[rec]):.3f}")
np.save(f'{BM}/analysis/threeway/forced_companion_result.npy',res)
