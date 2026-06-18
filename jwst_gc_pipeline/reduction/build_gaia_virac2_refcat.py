#!/usr/bin/env python
"""Build a dense Gaia-frame absolute astrometric reference catalog for the brick.

Gaia DR3 is the absolute frame (= GSC 3.2 Gaia-backed subset, verified 0 mas) but
sparse in the GC.  VIRAC2 (II/387) is on the Gaia frame (~5 mas) and dense in NIR.
Combined catalog: every Gaia DR3 source + every VIRAC2 source with no Gaia match
within 0.3", all propagated to the F115W epoch 2022.70.  Written in the layout the
jwst-gc-pipeline tweakreg consumer expects (RA, DEC, skycoord, VERSION/EPOCH meta).
"""
import numpy as np
import astropy.units as u
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord

BASE = '/orange/adamginsburg/jwst/brick'
EPOCH = 2022.70
GAIA_EPOCH = 2016.0    # Gaia DR3 reference epoch
VIRAC2_EPOCH = 2014.0  # VIRAC2 reference epoch (Smith+2025: "fixed at the reference epoch, 2014.0")
OUT = f'{BASE}/catalogs/gaia_virac2_refcat_epoch2022.70.fits'


def farr(x):
    return np.asarray(np.ma.filled(np.ma.masked_invalid(np.asarray(x, float)), np.nan), float)


def prop(ra, dec, pmra, pmde, dt):
    pmra = np.where(np.isfinite(pmra), pmra, 0.); pmde = np.where(np.isfinite(pmde), pmde, 0.)
    return ra + (pmra * dt / 3.6e6) / np.cos(np.radians(dec)), dec + (pmde * dt / 3.6e6)


# Gaia DR3 (epoch 2016.0)
g = Table.read(f'{BASE}/astrometry_analysis/reference_cache/basic_merged_photometry_tables_merged_gaia.fits')
gra, gdec = prop(farr(g['RA_ICRS']), farr(g['DE_ICRS']), farr(g['pmRA']), farr(g['pmDE']), EPOCH - GAIA_EPOCH)
gfin = np.isfinite(gra) & np.isfinite(gdec)
gaia_sc = SkyCoord(gra[gfin] * u.deg, gdec[gfin] * u.deg)

# VIRAC2 (epoch 2014.0, Gaia frame)
v = Table.read(f'{BASE}/astrometry_diag/refcache/virac2.fits')
vra, vdec = prop(farr(v['RAJ2000']), farr(v['DEJ2000']), farr(v['pmRA']), farr(v['pmDE']), EPOCH - VIRAC2_EPOCH)
vfin = np.isfinite(vra) & np.isfinite(vdec)
virac_sc = SkyCoord(vra[vfin] * u.deg, vdec[vfin] * u.deg)
vJ = farr(v['Jmag'])[vfin]; vH = farr(v['Hmag'])[vfin]; vK = farr(v['Ksmag'])[vfin]

# VIRAC2 sources with NO Gaia counterpart within 0.3" -> the NIR fill
idx, sep, _ = virac_sc.match_to_catalog_sky(gaia_sc)
fill = sep > 0.3 * u.arcsec
print(f"Gaia DR3: {gfin.sum()} sources; VIRAC2 fill (no Gaia <0.3\"): {fill.sum()} of {vfin.sum()}")

rows_gaia = Table()
rows_gaia['RA'] = gaia_sc.ra.deg
rows_gaia['DEC'] = gaia_sc.dec.deg
rows_gaia['source'] = np.full(len(gaia_sc), 'GaiaDR3', dtype='U8')
rows_gaia['refmag'] = farr(g['Gmag'])[gfin]

rows_v = Table()
rows_v['RA'] = virac_sc.ra.deg[fill]
rows_v['DEC'] = virac_sc.dec.deg[fill]
rows_v['source'] = np.full(int(fill.sum()), 'VIRAC2', dtype='U8')
# use J as the reference magnitude for the NIR fill
rows_v['refmag'] = vJ[fill]

ref = vstack([rows_gaia, rows_v])
ref['skycoord'] = SkyCoord(ref['RA'] * u.deg, ref['DEC'] * u.deg)
ref.meta['VERSION'] = 'gaia_dr3+virac2_fill'
ref.meta['FRAME'] = 'Gaia DR3 (ICRS); VIRAC2 (II/387) tied to Gaia DR3 to ~5 mas'
ref.meta['EPOCH'] = EPOCH
ref.meta['NOTE'] = ('Gaia DR3 = absolute frame (= GSC3.2 Gaia subset, 0 mas). VIRAC2 fills '
                    'the dense NIR regime where Gaia is incomplete. Propagated to epoch 2022.70.')
ref.write(OUT, overwrite=True)
ref.write(OUT.replace('.fits', '.ecsv'), overwrite=True)
print(f"Wrote {OUT}: {len(ref)} sources ({len(rows_gaia)} Gaia + {len(rows_v)} VIRAC2 fill)")
