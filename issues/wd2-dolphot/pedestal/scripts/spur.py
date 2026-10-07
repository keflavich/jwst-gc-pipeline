import numpy as np, warnings, json
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
warnings.filterwarnings('ignore')
T='/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/tree_mainfcbg/'
ref=Table.read('/orange/adamginsburg/jwst/wd2/catalogs/wd2_nircam_wf_mf_nf.ecsv')
vg={'F115W':'08101','F150W':'10101','F162M':'14101','F182M':'16101','F200W':'12101'}
off={'F150W':(-28,-13)}
out=[]
for b in vg:
    bl=b.lower()
    mag=np.asarray(ref['MAG'+bl[1:].upper()],float)
    rv=np.isfinite(mag)&(mag<50)
    rsc=SkyCoord(np.asarray(ref['RA'])[rv]*u.deg,np.asarray(ref['DEC'])[rv]*u.deg)
    rsc_all=SkyCoord(np.asarray(ref['RA'])*u.deg,np.asarray(ref['DEC'])*u.deg)
    for det in ['nrca1','nrcb3']:
        for ex in ['00001']:
            t=Table.read(f'{T}{b}/{bl}_{det}_visit001_vgroup{vg[b]}_exp{ex}_resbgsub_m7_daophot_basic.fits')
            n=len(t)
            sc=t['skycoord_centroid']
            if not isinstance(sc,SkyCoord): sc=SkyCoord(sc)
            snr=np.asarray(t['flux_fit'],float)/np.asarray(t['flux_err'],float)
            # footprint of frame sources
            ra0,dec0=sc.ra.deg,sc.dec.deg
            box=(np.asarray(ref['RA'])>ra0.min()-1e-3)&(np.asarray(ref['RA'])<ra0.max()+1e-3)&(np.asarray(ref['DEC'])>dec0.min()-1e-3)&(np.asarray(ref['DEC'])<dec0.max()+1e-3)
            nref_in=int((box&rv).sum())
            # offset estimate from bright rows
            o=off.get(b,(-42,5))
            res={}
            # catalogue = ref + offset  (ref positions shifted by (dRA*cos, dDec) in mas), test both signs
            for sign in (+1,-1):
                dra=sign*o[0]/1000/3600/np.cos(np.deg2rad(dec0.mean())); dde=sign*o[1]/1000/3600
                rs=SkyCoord((np.asarray(ref['RA'])[rv]+dra)*u.deg,(np.asarray(ref['DEC'])[rv]+dde)*u.deg)
                idx,sep,_=sc.match_to_catalog_sky(rs)
                hi=snr>10
                res[sign]=(float(np.median(sep.arcsec[hi&(sep.arcsec<0.3)])) if (hi&(sep.arcsec<0.3)).any() else np.nan, float(np.mean(sep.arcsec<0.08)), sep.arcsec)
            sign=min(res,key=lambda s: res[s][0])
            sep=res[sign][2]
            ok=np.isfinite(snr)
            # nearest-in-frame dedup stats: rows within 0.08 of a ref
            has=sep<0.08
            r=dict(band=b,det=det,rows=n,ref_in_box=nref_in,sign=sign,match_frac=float(has.mean()),n_nomatch=int((~has).sum()),
                   snr_lt3=float(np.mean(snr<3)),snr_lt5=float(np.mean(snr<5)),
                   nomatch_and_snr_lt3=float(np.mean(~has&(snr<3))),
                   nomatch_snr_ge5=int((~has&(snr>=5)).sum()),
                   med_snr_match=float(np.nanmedian(snr[has])),med_snr_nomatch=float(np.nanmedian(snr[~has])),
                   n_refs_matched=int(len(np.unique(np.where(has)[0]))), local_bkg_med=float(np.nanmedian(t['local_bkg'])))
            out.append(r); print(json.dumps(r),flush=True)
json.dump(out,open('spur.json','w'),indent=1)
