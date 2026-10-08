"""Query Gaia DR3 around the wd2 NIRCam footprint; write gaia_dr3_wd2.fits."""
import numpy as np
from astroquery.gaia import Gaia
# footprint from F150W m6 per-frame catalogs: RA 155.874-156.041, Dec -57.800..-57.719
ra0, dec0 = 155.958, -57.760
w = 0.100   # half-width deg in RA*cos(dec) ~ 0.085 + margin
q = f"""SELECT source_id, ra, dec, ra_error, dec_error, pmra, pmdec, pmra_error, pmdec_error,
 parallax, parallax_error, ref_epoch, phot_g_mean_mag, ruwe, astrometric_params_solved, visibility_periods_used
FROM gaiadr3.gaia_source
WHERE 1=CONTAINS(POINT('ICRS',ra,dec), BOX('ICRS',{ra0},{dec0},{2*w/np.cos(np.deg2rad(dec0)):.4f},{2*0.060:.4f}))"""
job = Gaia.launch_job_async(q)
t = job.get_results()
print(len(t))
t.write('gaia_dr3_wd2.fits', overwrite=True)
