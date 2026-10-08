import glob, sys, numpy as np
sys.path.insert(0,'/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an
from astropy.table import Table
from astropy.coordinates import SkyCoord, concatenate
import astropy.units as u
Q='/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
an.ZPWIN.update(an.zp_windows()); A=an.Arm('main2')
for b in ['150W','200W','250M','300M']:
    ok=A.matched & A.rep[b] & np.isfinite(A.ref[b])
    print(b,'replaced matched',ok.sum(), flush=True)
    tgt=A.sky[A.idx[ok]]
    for fn in sorted(glob.glob(f'{Q}/tree_main2/F{b}/pipeline/*_nrc*_align_o005_crf_resbgsub_m7_satstar_catalog.fits')):
        t=Table.read(fn); c=t['skycoord_fit']; c=c if isinstance(c,SkyCoord) else SkyCoord(c)
        i,j,_,_=c.search_around_sky(tgt,0.1*u.arcsec)
        print('  ',fn.split('/')[-1][:40], len(t), len(np.unique(i)), flush=True)
