import numpy as np, warnings
from astropy.io import fits
warnings.filterwarnings('ignore')
P='/orange/adamginsburg/jwst/wd2/'
for b,vg in [('F115W','08101'),('F162M','14101'),('F182M','16101'),('F200W','12101'),('F150W','10101')]:
    bl=b.lower()
    base=f'{P}{b}/pipeline/jw03523-o005_t001_nircam_clear-{bl}-nrca1_visit001_vgroup{vg}_exp00001_'
    for ph in ['m1_daophot_basic','resbgsub_m6_daophot_basic','resbgsub_m7_daophot_basic']:
        rm=fits.getdata(base+ph+'_residual.fits','SCI')+fits.getdata(base+ph+'_model.fits','SCI')
        s=[]
        for n in ['align_o005_crf','destreak_o005_crf']:
            try: x=fits.getdata(P+f'{b}/pipeline/jw03523005001_{vg}_00001_nrca1_{n}.fits','SCI')
            except FileNotFoundError: s.append(f'{n}: none'); continue
            y=(x-rm)[np.isfinite(x-rm)]
            s.append(f'{n}: median(frame-(res+model)) {np.median(y):.4f} MAD {np.median(np.abs(y-np.median(y))):.4f}')
        print(b,ph,'|',' | '.join(s))
