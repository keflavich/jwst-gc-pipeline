#!/usr/bin/env python3
"""Why are ~31k Hosek NRCA4 stars still missed (recover+nmatch m6)?
 (a) spatial: are they on saturated-star spikes / clustered / uniform?
 (b) recovery vs Hosek ndet: are the missed ones low-ndet (marginal for Hosek)?
 (c) recovery vs OUR nmatch (from m2-raw detections): do we drop low-nmatch?
"""
import warnings, numpy as np
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

ARCH='/orange/adamginsburg/jwst/arches/catalogs'
RA0,DEC0=266.47699800418224,-28.856142119344067; COSD=np.cos(np.radians(DEC0))
H=Table.read('/orange/adamginsburg/jwst/benchmark-team/arches_L2_20260629/combo_starlist_F212N_NRCA4.fits')
H['ra']=RA0+H['x_wcs'].astype(float)/3600/COSD; H['dec']=DEC0+H['y_wcs'].astype(float)/3600
hc=SkyCoord(H['ra']*u.deg,H['dec']*u.deg); mh=np.asarray(H['m'],float)
hndet=np.asarray(H['ndet'],float)

# our final (recover+nmatch) m6 + m2-raw (our nmatch for all detections) + satstars
m6=Table.read(f'{ARCH}/f212n_nrca_indivexp_merged_resbgsub_m6_dao_basic_vetted.fits')
c6=SkyCoord(m6['skycoord'].ra,m6['skycoord'].dec)
raw=Table.read(f'{ARCH}/f212n_nrca_indivexp_merged_m2_dao_basic.fits')
craw=SkyCoord(raw['skycoord'].ra,raw['skycoord'].dec); raw_nm=np.asarray(raw['nmatch'],float)
sat=np.asarray(m6['is_saturated'],bool) if 'is_saturated' in m6.colnames else np.zeros(len(m6),bool)
satc=c6[sat]

# Hosek -> our m6 (recovered?), -> m2-raw (our nmatch), -> nearest satstar
_,s6,_=hc.match_to_catalog_sky(c6); matched=s6.arcsec<0.08
ir,sr,_=hc.match_to_catalog_sky(craw); det=sr.arcsec<0.08
our_nm=np.where(det, raw_nm[ir], 0.0)          # our nmatch (0 = we never detected it)
_,ssat,_=hc.match_to_catalog_sky(satc) if len(satc) else (None,np.full(len(hc),np.inf)*u.deg,None)
dsat=ssat.arcsec
miss=~matched
print(f"Hosek {len(H)}  matched {matched.sum()}  missed {miss.sum()}")
print(f"missed & near satstar(<1.5\"): {(miss&(dsat<1.5)).sum()} ({(miss&(dsat<1.5)).mean():.2f} of all)")
print(f"missed & we NEVER detected (our_nm=0): {(miss&(our_nm==0)).sum()}")
print(f"missed & we detected but dropped (our_nm>=1): {(miss&(our_nm>=1)).sum()}")
print(f"missed & our_nm in 1-2 (sub-3-frame): {(miss&(our_nm>=1)&(our_nm<3)).sum()}")
print(f"missed & Hosek ndet==3 (his minimum): {(miss&(hndet==3)).sum()}  of all-ndet3 {int((hndet==3).sum())}")

fig,ax=plt.subplots(2,2,figsize=(15,12))
# (a) spatial
a=ax[0,0]
a.scatter(H['ra'][matched],H['dec'][matched],s=1,c='0.75',alpha=0.3,label=f'matched {matched.sum()}')
a.scatter(H['ra'][miss],H['dec'][miss],s=2,c='tab:red',alpha=0.4,label=f'Hosek-only {miss.sum()}')
if len(satc): a.scatter(satc.ra.deg,satc.dec.deg,s=40,marker='x',c='blue',label=f'our satstars in-fp')
# zoom to the Hosek comparison footprint (with a small pad), not the whole NRCA field
_pad=2/3600
a.set_xlim(H['ra'].max()+_pad, H['ra'].min()-_pad)   # RA descending (already inverted)
a.set_ylim(H['dec'].min()-_pad, H['dec'].max()+_pad)
a.set_xlabel('RA'); a.set_ylabel('Dec'); a.set_aspect('equal'); a.legend(fontsize=8)
a.set_title('(a) spatial (Hosek footprint): Hosek-only (red) vs satstars (blue X)')
# (b) recovery vs Hosek ndet
a=ax[0,1]
nd=np.arange(3,12); rec_nd=[matched[hndet==n].mean() if (hndet==n).sum() else np.nan for n in nd]
n_nd=[int((hndet==n).sum()) for n in nd]
a.bar(nd,rec_nd,color='tab:green'); a.set_xlabel("Hosek ndet"); a.set_ylabel('our recovery fraction')
a.set_ylim(0,1); a.set_title('(b) recovery vs Hosek ndet (are misses low-ndet?)')
for x,y,nn in zip(nd,rec_nd,n_nd): a.text(x,(y or 0)+0.02,str(nn),ha='center',fontsize=7,rotation=90)
# (c) recovery vs OUR nmatch (from m2-raw)
a=ax[1,0]
onb=np.arange(0,13); rec_on=[matched[(our_nm==n)].mean() if (our_nm==n).sum() else np.nan for n in onb]
n_on=[int((our_nm==n).sum()) for n in onb]
a.bar(onb,rec_on,color='tab:purple'); a.set_xlabel("OUR nmatch (0 = never detected)"); a.set_ylabel('recovery fraction')
a.set_ylim(0,1); a.set_title('(c) recovery vs OUR nmatch (do we drop low-nmatch?)')
for x,y,nn in zip(onb,rec_on,n_on): a.text(x,(y or 0)+0.02,str(nn),ha='center',fontsize=7,rotation=90)
# (d) missed: spike proximity + our-nmatch breakdown
a=ax[1,1]
a.hist(np.clip(dsat[miss],0,5),bins=40,color='tab:red',alpha=0.7)
a.axvline(1.5,color='k',ls='--',label='1.5" satstar zone')
a.set_xlabel('missed source: dist to nearest satstar (arcsec)'); a.set_ylabel('N')
a.legend(); a.set_title(f'(d) missed vs satstar dist: {(miss&(dsat<1.5)).sum()} of {miss.sum()} within 1.5"')
fig.suptitle('Arches F212N NRCA4: anatomy of the remaining Hosek-only (recover+nmatch m6)')
fig.tight_layout(); fig.savefig('/orange/adamginsburg/jwst/benchmark-team/comparison_keflavich/remaining_misses.png',dpi=110)
print("\nwrote remaining_misses.png")
