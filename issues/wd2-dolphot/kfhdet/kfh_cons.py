import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
Q='/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
for b in ['f115w','f150w','f200w','f182m','f250m','f277w','f300m','f410m']:
    A=Table.read(f'{Q}/tree_main2kf/catalogs/{b}_consolidated_satstar_catalog.fits')
    B=Table.read(f'{Q}/tree_main2kfh/catalogs/{b}_consolidated_satstar_catalog.fits')
    ca=A['skycoord_fit'] if 'skycoord_fit' in A.colnames else None
    cb=B['skycoord_fit']
    i,d,_=cb.match_to_catalog_sky(ca)
    m=d<0.02*u.arcsec
    fa=np.asarray(A['flux_fit'])[i[m]]; fb=np.asarray(B['flux_fit'])[m]
    dm=-2.5*np.log10(fb/fa)
    ch=np.abs(dm)>1e-4
    print(b, len(A), len(B), 'matched', m.sum(), 'changed', ch.sum(), 'dm p10/50/90 of changed', np.round(np.nanpercentile(dm[ch],[10,50,90]),4) if ch.sum() else '-', 'min/max', np.round(np.nanmin(dm),3), np.round(np.nanmax(dm),3))
