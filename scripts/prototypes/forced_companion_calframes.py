#!/usr/bin/env python3
"""PROTOTYPE v2: forced companion photometry on CAL FRAMES (matched detector PSF).

Mosaic fit was PSF-mismatch-limited (drizzle broadens the PSF -> chi2~270 -> a fake
'companion' just absorbs the primary's residual). On individual _cal frames the stpsf
DETECTOR PSF matches, so the fit residual is ~noise and a real companion is a genuine
positive. Gate = coadded Delta-chi2 (F-test: does adding the companion at its fixed
position significantly improve the joint fit over primary-only?), scale-invariant to the
per-pixel error normalisation. Coadd across the 12 dithers.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, glob, os
from astropy.table import Table
from astropy.coordinates import SkyCoord; import astropy.units as u
from astropy.io import fits; from astropy.wcs import WCS
from scipy.spatial import cKDTree
from scipy.ndimage import shift as ndshift
BM='/orange/adamginsburg/jwst/benchmark-team'
RA0,DEC0=266.47699800418224,-28.856142119344067; COSD=np.cos(np.radians(DEC0))
def off(a,b,c,d):
    xa=(a-RA0)*COSD*3600;ya=(b-DEC0)*3600;xb=(c-RA0)*COSD*3600;yb=(d-DEC0)*3600
    dd,i=cKDTree(np.c_[xb,yb]).query(np.c_[xa,ya],k=1);n=dd<0.5
    dx=(xb[i[n]]-xa[n])*1000;dy=(yb[i[n]]-ya[n])*1000
    H,xe,ye=np.histogram2d(dx,dy,bins=100,range=[[-400,400],[-400,400]]);iy,ix=np.unravel_index(H.argmax(),H.shape)
    return 0.5*(xe[iy]+xe[iy+1]),0.5*(ye[ix]+ye[ix+1])
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
miss=(sp<R)&(sj>=R); ij,dj,_=SC.match_to_catalog_sky(JC)
comp=miss&(dj<0.4*u.arcsec)&(dj>0.03*u.arcsec)
crd_c=SC[comp];crd_p=JC[ij[comp]]
os.environ.setdefault('STPSF_PATH','/orange/adamginsburg/repos/webbpsf/data')
import stpsf
nrc=stpsf.NIRCam();nrc.filter='F212N';nrc.detector='NRCA4'
psf=nrc.calc_psf(fov_pixels=21,oversample=1)[0].data;psf/=psf.sum();ph=psf.shape[0]//2
cals=[c for c in sorted(glob.glob('/orange/adamginsburg/jwst/arches/F212N/pipeline/jw02045001001_02101_000*_nrca4_cal.fits')) if 'uncal' not in c]
frames=[(fits.open(c)['SCI'].data.astype(float),fits.open(c)['ERR'].data.astype(float),WCS(fits.open(c)['SCI'].header)) for c in cals]
def tmpl(xg,yg,sh):
    t=np.zeros(sh);yy=int(round(yg))-ph;xx=int(round(xg))-ph
    sub=ndshift(psf,(yg-round(yg),xg-round(xg)),order=1,mode='constant')
    ys,xs=max(0,yy),max(0,xx);ye,xe=min(sh[0],yy+psf.shape[0]),min(sh[1],xx+psf.shape[1])
    if ye<=ys or xe<=xs:return None
    t[ys:ye,xs:xe]=sub[ys-yy:ye-yy,xs-xx:xe-xx];return t
rng=np.random.default_rng(0);idx=rng.permutation(comp.sum())[:400];rad=7
out=[]
for k in idx:
    rc,dc=crd_c[k].ra.deg,crd_c[k].dec.deg;rp,dp=crd_p[k].ra.deg,crd_p[k].dec.deg
    dchi=0.0;fc_sum=0.0;fc_var=0.0;nf=0
    for sci,err,w in frames:
        xc,yc=w.world_to_pixel_values(rc,dc);xp,yp=w.world_to_pixel_values(rp,dp)
        xc,yc,xp,yp=float(xc),float(yc),float(xp),float(yp)
        cx,cy=int(round((xc+xp)/2)),int(round((yc+yp)/2))
        if not(rad<=cy<sci.shape[0]-rad and rad<=cx<sci.shape[1]-rad):continue
        cut=sci[cy-rad:cy+rad+1,cx-rad:cx+rad+1];ec=err[cy-rad:cy+rad+1,cx-rad:cx+rad+1]
        if not(np.all(np.isfinite(cut))and np.all(np.isfinite(ec))and (ec>0).all()):continue
        sh=cut.shape;lx,ly=cx-rad,cy-rad
        Tp=tmpl(xp-lx,yp-ly,sh);Tc=tmpl(xc-lx,yc-ly,sh)
        if Tp is None or Tc is None:continue
        wv=1/ec.ravel();b=cut.ravel()*wv
        # primary-only
        A1=np.c_[Tp.ravel(),np.ones(cut.size)]*wv[:,None]
        c1,_,_,_=np.linalg.lstsq(A1,b,rcond=None);chi1=np.sum((b-A1@c1)**2)
        # primary+companion
        A2=np.c_[Tp.ravel(),Tc.ravel(),np.ones(cut.size)]*wv[:,None]
        try:c2,_,_,_=np.linalg.lstsq(A2,b,rcond=None);cov=np.linalg.inv(A2.T@A2)
        except np.linalg.LinAlgError:continue
        chi2=np.sum((b-A2@c2)**2);dof=cut.size-3
        s2=chi2/dof  # per-frame noise scale (matched PSF => ~1 if err ok)
        dchi+=(chi1-chi2)/s2  # scale-invariant improvement
        fc=c2[1];var=cov[1,1]*s2
        if var>0:fc_sum+=fc/var;fc_var+=1/var
        nf+=1
    if nf>=6 and fc_var>0:
        fc_co=fc_sum/fc_var;snr_co=fc_co*np.sqrt(fc_var)
        out.append((snr_co,dchi,crd_c[k].separation(crd_p[k]).arcsec*1000,fc_co,nf))
out=np.array(out)
snr,dchi,sep,fc,nf=out.T
# gate: coadded companion S/N>=5 AND total Delta-chi2 (F) > 25 (5sigma, 1 param) AND flux>0
rec=(snr>=5)&(dchi>25)&(fc>0)
print(f"\ncal-frame forced fits: N={len(out)}  (per-frame matched detector PSF)")
print(f"  companion coadd S/N: med={np.median(snr):.1f}  Delta-chi2: med={np.median(dchi):.0f}")
print(f"  RECOVERABLE (S/N>=5 & Dchi2>25 & flux>0): {rec.sum()} ({100*rec.mean():.0f}%)")
for lo,hi,l in [(0,100,'<100mas'),(100,150,'100-150'),(150,250,'150-250'),(250,999,'>250')]:
    s=(sep>=lo)&(sep<hi)
    if s.sum():print(f"    sep {l:8s}: {100*rec[s].mean():.0f}% recoverable (N={s.sum()})")
np.save(f'{BM}/analysis/threeway/forced_companion_calframe.npy',out)
