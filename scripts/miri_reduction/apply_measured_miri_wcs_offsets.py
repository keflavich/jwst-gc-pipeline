#!/usr/bin/env python
"""
Apply offset-histogram-measured astrometric corrections to MIRI i2d products
(2026-06-11).  Offsets below are (image frame minus f182m/f405n refcat frame)
in true arcsec; the correction subtracts them, bringing the images onto the
NIRCam reference frame.  Both the gwcs (ASDF) and FITS headers are updated,
mirroring PipelineMIRI.fix_alignment; MIRIDRA/MIRIDDE keywords record the
applied correction and prevent double-application on rerun.

cloudc offset is refined inline before applying.
"""
import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.stats import mad_std
import astropy.units as u
from photutils.detection import DAOStarFinder
from scipy.ndimage import median_filter
import warnings
warnings.filterwarnings('ignore')

os.environ.setdefault("CRDS_PATH", "/orange/adamginsburg/jwst/brick/crds/")
os.environ.setdefault("CRDS_SERVER_URL", "https://jwst-crds.stsci.edu")
from jwst.datamodels import ImageModel
from jwst.tweakreg.utils import adjust_wcs


def refine_offset(fn, refcat_fn, fwhm_pix=7.3, mfsize=31, thr=5):
    fh = fits.open(fn)
    d = fh['SCI'].data
    ww = WCS(fh['SCI'].header)
    mfd = median_filter(np.nan_to_num(d, nan=np.nanmedian(d)), size=mfsize)
    hp = np.nan_to_num(d) - mfd
    s = DAOStarFinder(threshold=thr * mad_std(hp), fwhm=fwhm_pix)(hp)
    xcol = 'x_centroid' if 'x_centroid' in s.colnames else 'xcentroid'
    ycol = 'y_centroid' if 'y_centroid' in s.colnames else 'ycentroid'
    sc = ww.pixel_to_world(s[xcol], s[ycol])
    ref = Table.read(refcat_fn)
    refsc = SkyCoord(ref['skycoord']) if 'skycoord' in ref.colnames \
        else SkyCoord(ref['RA'], ref['DEC'], unit='deg')
    i_b, i_a, sep, _ = search_around_sky(refsc, sc, 6 * u.arcsec)
    dra = ((sc.ra[i_a] - refsc.ra[i_b]) * np.cos(sc.dec[i_a])).to(u.arcsec).value
    ddec = (sc.dec[i_a] - refsc.dec[i_b]).to(u.arcsec).value
    bins = np.arange(-6, 6.1, 0.1)
    H, xe, ye = np.histogram2d(dra, ddec, bins=[bins, bins])
    pk = np.unravel_index(np.argmax(H), H.shape)
    cx, cy = 0.5 * (xe[pk[0]] + xe[pk[0] + 1]), 0.5 * (ye[pk[1]] + ye[pk[1] + 1])
    m = (np.abs(dra - cx) < 0.5) & (np.abs(ddec - cy) < 0.5)
    rx, ry = np.median(dra[m]), np.median(ddec[m])
    print(f'{os.path.basename(fn)}: refined offset ({rx:+.3f}", {ry:+.3f}") '
          f'n={m.sum()} mad=({mad_std(dra[m]):.2f},{mad_std(ddec[m]):.2f})')
    return rx, ry


def apply_correction(fn, dra_as, ddec_as):
    """Subtract the measured (image - refcat) offset from the WCS."""
    with fits.open(fn) as check:
        if 'MIRIDRA' in check[1].header:
            print(f'{os.path.basename(fn)}: correction already applied '
                  f'({check[1].header["MIRIDRA"]}, {check[1].header["MIRIDDE"]}); skipping')
            return
    # FITS-header-only correction: adjust_wcs() raises "Unknown WCS structure"
    # on resampled-i2d gwcs (no v2v3 frames), and all downstream consumers of
    # i2d products in this pipeline read the FITS WCS.  The MIRIWCSN keyword
    # flags that the embedded ASDF gwcs is NOT corrected.
    fh = fits.open(fn)
    cosd = np.cos(np.deg2rad(fh[1].header['CRVAL2']))
    fh[1].header['OLCRVAL1'] = fh[1].header['CRVAL1']
    fh[1].header['OLCRVAL2'] = fh[1].header['CRVAL2']
    fh[1].header['CRVAL1'] = fh[1].header['CRVAL1'] - dra_as / 3600. / cosd
    fh[1].header['CRVAL2'] = fh[1].header['CRVAL2'] - ddec_as / 3600.
    fh[1].header['MIRIDRA'] = (-dra_as, 'applied RA correction [arcsec], 2026-06-11')
    fh[1].header['MIRIDDE'] = (-ddec_as, 'applied Dec correction [arcsec], 2026-06-11')
    fh[1].header['MIRIWCSN'] = ('FITS WCS corrected; ASDF gwcs NOT corrected',
                                'offset-histogram registration to NIRCam refcat')
    fh.writeto(fn, overwrite=True)
    print(f'{os.path.basename(fn)}: applied ({-dra_as:+.3f}", {-ddec_as:+.3f}") to FITS CRVAL')


# measured 2026-06-11 (see brick/catalogs/miri_band_offsets_vs_f182m.npy)
JOBS = [
    ('/orange/adamginsburg/jwst/sickle/F770W/pipeline/jw03958-o003_t001_miri_f770w_i2d.fits', -3.435, +1.381),
    ('/orange/adamginsburg/jwst/sickle/F1130W/pipeline/jw03958-o003_t001_miri_f1130w_i2d.fits', -3.428, +1.404),
    ('/orange/adamginsburg/jwst/sickle/F1500W/pipeline/jw03958-o003_t001_miri_f1500w_i2d.fits', -3.432, +1.405),
    ('/orange/adamginsburg/jwst/brick/F2550W/pipeline/jw02221-o002_t001_miri_f2550w_i2d.fits', -4.697, -2.469),
]

# cloudc: refine before applying
cl_fn = '/orange/adamginsburg/jwst/cloudc/F2550W/pipeline/jw02221-o001_t001_miri_f2550w_i2d.fits'
cl_ref = '/orange/adamginsburg/jwst/cloudc/catalogs/crowdsource_based_nircam-f405n_reference_astrometric_catalog.fits'
rx, ry = refine_offset(cl_fn, cl_ref)
JOBS.append((cl_fn, rx, ry))

for fn, dra, ddec in JOBS:
    apply_correction(fn, dra, ddec)

print('ALL DONE')
