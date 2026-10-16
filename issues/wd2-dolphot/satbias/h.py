from astroquery.mast import Catalogs
from astropy.coordinates import SkyCoord
from astropy.table import vstack
import astropy.units as u, numpy as np
ts=[]
for dec in np.arange(-57.805,-57.715,0.03):
    for dra in np.arange(-0.047,0.048,0.03):
        c=SkyCoord(155.9575+dra/np.cos(np.radians(dec)),dec,unit='deg')
        for dra in [0]:
            pass
        t=Catalogs.query_region(c,radius=0.025,catalog='HSC',version=3,pagesize=100000)
        print(c.ra.deg,dec,len(t),flush=True)
        ts.append(t)
T=vstack(ts)
_,i=np.unique(np.asarray(T['MatchID']),return_index=True)
T=T[i]; print(len(T))
T.write('hsc_wd2.fits',overwrite=True)
