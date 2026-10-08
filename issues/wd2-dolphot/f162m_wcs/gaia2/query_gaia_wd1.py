"""Query Gaia DR3 around the wd1 NIRCam footprint (F212N m6 catalogs: RA 251.677-251.843, Dec -45.912..-45.804); write gaia_dr3_wd1.fits."""
from astroquery.gaia import Gaia
ra0, dec0 = 251.760, -45.858
q = f"""SELECT source_id, ra, dec, ra_error, dec_error, pmra, pmdec, pmra_error, pmdec_error,
 parallax, parallax_error, ref_epoch, phot_g_mean_mag, ruwe, astrometric_params_solved, visibility_periods_used
FROM gaiadr3.gaia_source
WHERE 1=CONTAINS(POINT('ICRS',ra,dec), BOX('ICRS',{ra0},{dec0},0.22,0.16))"""
t = Gaia.launch_job_async(q).get_results()
print(len(t))
t.write('gaia_dr3_wd1.fits', overwrite=True)
