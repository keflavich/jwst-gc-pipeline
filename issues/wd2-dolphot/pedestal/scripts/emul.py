"""Emulate the m7 per-frame daofind + local-S/N>=3 filter (cataloging.py:1360-1381, 2847-2896)
for nrca1 exp00001, with the m6 smoothed-bg mosaic subtracted from (a) the align crf (what m7 did) and
(b) the destreak crf (the frames the m6 bg was built from = self-consistent case, analogous to F150W)."""
import numpy as np, json, warnings
from astropy.io import fits
from astropy.table import Table
from astropy import wcs as awcs
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans, convolve_fft
from photutils.detection import DAOStarFinder
from scipy import ndimage
from stdatamodels.jwst import datamodels
warnings.filterwarnings('ignore')
T='/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/tree_mainfcbg'
P='/orange/adamginsburg/jwst/wd2/'
ft=Table.read('/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline/jwst_gc_pipeline/reduction/fwhm_table.ecsv')
cfg={'F150W':'10101','F115W':'08101','F162M':'14101','F182M':'16101','F200W':'12101'}
def noise_map(d,s=3.0):
    im=np.nan_to_num(d); sm=ndimage.gaussian_filter(im,s); r=im-sm
    return np.sqrt(np.clip(ndimage.gaussian_filter(r**2,s),0,None))
out=[]
for band,vg in cfg.items():
    fwhm=float(ft[ft['Filter']==band]['PSF FWHM (pixel)'][0])
    bgp=f'{P}{band}/pipeline/jw03523-o005_t001_nircam_clear-{band.lower()}-merged_resbgsub_m6_daophot_basic_mergedcat_residual_smoothed_bg_i2d.fits'
    with fits.open(bgp) as h:
        hd=h['SCI'] if 'SCI' in [x.name for x in h] else h[0]
        bgw=awcs.WCS(hd.header); bgd=hd.data.astype(float)
    for suf in ['align_o005_crf','destreak_o005_crf']:
        crf=f'{P}{band}/pipeline/jw03523005001_{vg}_00001_nrca1_{suf}.fits'
        try: h=fits.open(crf)
        except FileNotFoundError: continue
        d=h['SCI'].data.astype(float); dq=h['DQ'].data
        with datamodels.open(f'{P}{band}/pipeline/jw03523005001_{vg}_00001_nrca1_align_o005_crf.fits') as dm:
            yy,xx=np.mgrid[0:d.shape[0],0:d.shape[1]]
            ra,dec=dm.meta.wcs(xx,yy)
        bx,by=bgw.world_to_pixel_values(ra,dec)
        bg=ndimage.map_coordinates(np.nan_to_num(bgd),[by,bx],order=1,mode='constant',cval=0.0)
        zeros=d==0
        raw_med=float(np.nanmedian(d[~zeros])); d=d-bg; d[zeros]=0
        d=np.where(dq&1,np.nan,d)
        k=Gaussian2DKernel(x_stddev=fwhm/2.355)
        nr=interpolate_replace_nans(d,k,convolve=convolve_fft,allow_huge=True)
        nm=noise_map(nr); fin=np.isfinite(nm)&(nm>0); gmin=nm[fin].min()
        det=DAOStarFinder(threshold=gmin,fwhm=fwhm,roundlo=-1,roundhi=1,sharplo=0.3,sharphi=1.4)(nr)
        x=np.rint(det['xcentroid']).astype(int);y=np.rint(det['ycentroid']).astype(int)
        snr=np.abs(det['peak'])/nm[y,x]
        res=dict(band=band,frame_suffix=suf,raw_frame_med=raw_med,post_sub_med=float(np.nanmedian(nr)),ndet=len(det),nkeep=int((snr>=3).sum()),keep_frac=float((snr>=3).mean()),
                 median_peak=float(np.median(np.abs(det['peak']))),median_noise=float(np.median(nm[y,x])))
        print(json.dumps(res),flush=True); out.append(res)
json.dump(out,open('emul.json','w'),indent=1)
