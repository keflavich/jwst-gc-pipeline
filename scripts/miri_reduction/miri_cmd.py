#!/usr/bin/env python
"""
MIRI color-magnitude diagrams for the sickle program fields (o001/o002 =
Sickle pos1/pos2, o003 = Brick background) using astrometry-corrected
matching (2026-06-11).

Per field: verify the relative F770W<->F1130W<->F1500W registration by
offset-histogram stacking (the bands share a WCS chain and agreed to 0.02"
in o003; this re-checks o001/o002), detect stars per band, do aperture
photometry (r=1.5 FWHM, annulus 2.5-4 FWHM), match across bands at 0.4",
and plot [F770W] vs [F770W]-[F1130W] and [F770W]-[F1500W] Vega CMDs.

Vega zeropoints from SVO FPS.  No aperture correction is applied (a
roughly constant ~0.1-0.2 mag offset per band; colors are mostly immune).

Outputs: catalogs/<field>_miri_cmd_matched.fits per field (in the sickle
tree) and a combined PNG.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.stats import mad_std
from astropy.table import Table
import astropy.units as u
from astroquery.svo_fps import SvoFps
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry, ApertureStats
from scipy.ndimage import median_filter
import warnings
warnings.filterwarnings('ignore')

BANDS = {'F770W': dict(fwhm_as=0.269, fwhm_pix=2.4, mfs=15),
         'F1130W': dict(fwhm_as=0.375, fwhm_pix=3.4, mfs=21),
         'F1500W': dict(fwhm_as=0.488, fwhm_pix=4.4, mfs=27)}
FIELDS = {'o001 (Sickle pos1)': 'o001', 'o002 (Sickle pos2)': 'o002',
          'o003 (Brick bg)': 'o003'}

jfilts = SvoFps.get_filter_list('JWST')
jfilts.add_index('filterID')
ZP = {b: float(jfilts.loc[f'JWST/MIRI.{b}']['ZeroPoint']) for b in BANDS}
print('Vega ZPs (Jy):', ZP)


def band_data(obs, band):
    fn = (f'/orange/adamginsburg/jwst/sickle/{band}/pipeline/'
          f'jw03958-{obs}_t001_miri_{band.lower()}_i2d.fits')
    fh = fits.open(fn)
    return fh['SCI'].data, fh['ERR'].data, WCS(fh['SCI'].header), fn


def detect_and_phot(obs, band):
    d, err, ww, fn = band_data(obs, band)
    info = BANDS[band]
    mfd = median_filter(np.nan_to_num(d, nan=np.nanmedian(d)), size=info['mfs'])
    hp = np.nan_to_num(d) - mfd
    # detect on the ERR-normalized high-pass image: a global mad_std threshold
    # is blown up by bright structured nebulosity (o001 found only 7 F1130W
    # stars), whereas the S/N image keeps the threshold local
    err = np.where(np.isfinite(err) & (err > 0), err, np.inf)
    snr_img = hp / err
    s = DAOStarFinder(threshold=5.0, fwhm=info['fwhm_pix'])(snr_img)
    xc = 'x_centroid' if 'x_centroid' in s.colnames else 'xcentroid'
    x, y = np.asarray(s[xc]), np.asarray(s[xc.replace('x', 'y')])
    pixscale = np.sqrt(np.abs(np.linalg.det(ww.pixel_scale_matrix))) * 3600
    pixar_sr = (pixscale / 206265.)**2
    r = 1.5 * info['fwhm_as'] / pixscale
    ap = CircularAperture(np.transpose([x, y]), r=r)
    ann = CircularAnnulus(np.transpose([x, y]),
                          r_in=2.5 * info['fwhm_as'] / pixscale,
                          r_out=4.0 * info['fwhm_as'] / pixscale)
    bkg = ApertureStats(d, ann).median
    flux_jy = (aperture_photometry(d, ap)['aperture_sum'] - bkg * ap.area) * 1e6 * pixar_sr
    good = np.isfinite(flux_jy) & (flux_jy > 0)
    mag = -2.5 * np.log10(np.asarray(flux_jy)[good] / ZP[band])
    return ww.pixel_to_world(x[good], y[good]), mag


def rel_offset(sc_a, sc_b, label):
    i_b, i_a, sep, _ = search_around_sky(sc_b, sc_a, 3 * u.arcsec)
    dra = ((sc_a.ra[i_a] - sc_b.ra[i_b]) * np.cos(sc_a.dec[i_a])).to(u.arcsec).value
    ddec = (sc_a.dec[i_a] - sc_b.dec[i_b]).to(u.arcsec).value
    bins = np.arange(-3, 3.05, 0.1)
    H, xe, ye = np.histogram2d(dra, ddec, bins=[bins, bins])
    pk = np.unravel_index(np.argmax(H), H.shape)
    cx, cy = 0.5 * (xe[pk[0]] + xe[pk[0] + 1]), 0.5 * (ye[pk[1]] + ye[pk[1] + 1])
    m = (np.abs(dra - cx) < 0.4) & (np.abs(ddec - cy) < 0.4)
    rx, ry = np.median(dra[m]), np.median(ddec[m])
    print(f'  {label}: relative offset ({rx:+.3f}", {ry:+.3f}") n={m.sum()}')
    return rx, ry


fig, axs = pl.subplots(2, 3, figsize=(16, 10), sharey=True)
for col, (title, obs) in enumerate(FIELDS.items()):
    print(f'=== {title}')
    cats = {b: detect_and_phot(obs, b) for b in BANDS}
    for b in BANDS:
        print(f'  {b}: {len(cats[b][1])} stars')
    # relative registration check + correction (shift the redder band onto F770W)
    shifted = {'F770W': cats['F770W'][0]}
    for b in ('F1130W', 'F1500W'):
        rx, ry = rel_offset(cats[b][0], cats['F770W'][0], f'{b}-F770W')
        sc = cats[b][0]
        shifted[b] = SkyCoord(ra=sc.ra - (rx * u.arcsec) / np.cos(sc.dec),
                              dec=sc.dec - ry * u.arcsec)
    # match
    tab = Table()
    sc7, m7 = shifted['F770W'], cats['F770W'][1]
    tab['ra'], tab['dec'], tab['mag_F770W'] = sc7.ra.deg, sc7.dec.deg, m7
    for b in ('F1130W', 'F1500W'):
        idx, sep, _ = sc7.match_to_catalog_sky(shifted[b])
        mb = np.full(len(sc7), np.nan)
        ok = sep < 0.4 * u.arcsec
        mb[ok] = cats[b][1][idx[ok]]
        tab[f'mag_{b}'] = mb
        print(f'  matched F770W-{b}: {ok.sum()}')
    out = f'/orange/adamginsburg/jwst/sickle/catalogs/{obs}_miri_cmd_matched.fits'
    tab.write(out, overwrite=True)
    print(f'  wrote {out}')
    for row_i, b in enumerate(('F1130W', 'F1500W')):
        ax = axs[row_i, col]
        color = tab['mag_F770W'] - tab[f'mag_{b}']
        ok = np.isfinite(color)
        ax.scatter(color[ok], tab['mag_F770W'][ok], s=8, alpha=0.6, color='k')
        ax.set_xlabel(f'[F770W] - [{b}]')
        if col == 0:
            ax.set_ylabel('[F770W] (Vega)')
        ax.set_title(f'{title}  (n={ok.sum()})', fontsize=10)
        ax.set_xlim(-1.5, 5)
for ax in axs.ravel():
    ax.invert_yaxis() if not ax.yaxis_inverted() else None
pl.suptitle('MIRI CMDs, astrometry-corrected matches (no aperture corr.)', fontsize=14)
pl.tight_layout()
outpng = '/blue/adamginsburg/adamginsburg/logs/miri_phot/miri_cmds.png'
pl.savefig(outpng, dpi=130, bbox_inches='tight')
print('wrote', outpng)
