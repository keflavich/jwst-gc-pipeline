import numpy as np
from astropy.io import fits
P='/orange/adamginsburg/jwst/wd2/psfs/'
S='/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/f150x/samp4/grids/'
for f in ('f150w','f200w'):
    n=f'nircam_nrcb1_{f}_fovp101_samp2_npsf16.fits'
    a=fits.open(P+n)[0]; b=fits.open(S+n)[0]
    d=np.abs(a.data-b.data); print(f,'shape',a.data.shape,b.data.shape,'max|d|',d.max(),'max rel (to plane peak)',(d.max(axis=(1,2))/a.data.max(axis=(1,2))).max(),
      'max rel where |a|>1e-4 peak',(d/np.abs(a.data))[np.abs(a.data)>1e-4*a.data.max()].max())
    for k in ('OPD_FILE','JITRSIGM','NWAVES','DET_SAMP','OVERSAMP','VERSION','FOVPIXEL','CHDFSIGM'):
        print(' ',k,a.header[k],b.header[k])
    c=fits.open(S+n.replace('samp2','samp4'))[0]; print(' samp4',c.data.shape,c.header['OVERSAMP'],c.header['DET_SAMP'],c.header['NWAVES'],'sum plane0',c.data[0].sum(),a.data[0].sum())
