#!/usr/bin/env python
"""Build a dense Gaia+VIRAC2 absolute astrometric reference catalog for ANY Galactic-Center field,
by querying Vizier (VIRAC2 II/387) and Gaia DR3 over the field footprint.

Standing policy (feedback_reference_frame_policy): GC fields -> VIRAC2 positions PM-propagated
per-star from the VIRAC2 reference epoch 2014.0 to the observation epoch; Gaia DR3 (PM-propagated
from 2016.0) provides the absolute frame where it is complete. The combined catalog = every Gaia DR3
source + every VIRAC2 source with no Gaia match within 0.3", all at the observation epoch.

Usage:
    python build_gaia_virac2_refcat_byquery.py --base /orange/adamginsburg/jwst/sgrb2 \
        --epoch 2024.685 --ra 266.835 --dec -28.398 --radius 0.1 --out-epoch-tag 2024.68
"""
import argparse
import numpy as np
import astropy.units as u
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord

GAIA_EPOCH = 2016.0    # Gaia DR3 reference epoch
VIRAC2_EPOCH = 2014.0  # VIRAC2 reference epoch (Smith+2025 II/387: fixed at 2014.0)


def farr(x):
    return np.asarray(np.ma.filled(np.ma.masked_invalid(np.asarray(x, float)), np.nan), float)


def prop(ra, dec, pmra, pmde, dt):
    pmra = np.where(np.isfinite(pmra), pmra, 0.); pmde = np.where(np.isfinite(pmde), pmde, 0.)
    return ra + (pmra * dt / 3.6e6) / np.cos(np.radians(dec)), dec + (pmde * dt / 3.6e6)


def query_virac2(ra, dec, radius):
    from astroquery.vizier import Vizier
    Vizier.ROW_LIMIT = -1
    Vizier.columns = ['RAJ2000', 'DEJ2000', 'pmRA', 'pmDE', 'Jmag', 'Hmag', 'Ksmag']
    res = Vizier.query_region(SkyCoord(ra * u.deg, dec * u.deg), radius=radius * u.deg,
                              catalog='II/387/virac2')
    if not res:
        raise RuntimeError("VIRAC2 query returned nothing")
    return res[0]


def _query_gaia_vizier(ra, dec, radius):
    """VizieR fallback (I/355/gaiadr3) -- works from compute nodes where the Gaia ESA
    TAP service is firewalled.  Returns a table with ESA-TAP-style column names."""
    from astroquery.vizier import Vizier
    Vizier.ROW_LIMIT = -1
    Vizier.columns = ['RA_ICRS', 'DE_ICRS', 'pmRA', 'pmDE', 'Gmag']
    res = Vizier.query_region(SkyCoord(ra * u.deg, dec * u.deg), radius=radius * u.deg,
                              catalog='I/355/gaiadr3')
    if not res:
        raise RuntimeError("VizieR Gaia DR3 query returned nothing")
    t = res[0]
    t.rename_column('RA_ICRS', 'ra'); t.rename_column('DE_ICRS', 'dec')
    t.rename_column('pmRA', 'pmra'); t.rename_column('pmDE', 'pmdec')
    t.rename_column('Gmag', 'phot_g_mean_mag')
    return t


def query_gaia(ra, dec, radius, retries=6):
    import time
    from astroquery.gaia import Gaia
    Gaia.ROW_LIMIT = -1
    q = ("SELECT ra,dec,pmra,pmdec,phot_g_mean_mag,ref_epoch FROM gaiadr3.gaia_source "
         f"WHERE CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{ra},{dec},{radius}))=1")
    last = None
    for i in range(retries):
        # async first: it is NOT capped (sync launch_job silently caps at 2000)
        for launcher in (Gaia.launch_job_async, Gaia.launch_job):
            try:
                res = launcher(q).get_results()
                if launcher is Gaia.launch_job and len(res) == 2000:
                    raise RuntimeError("sync query hit the 2000-row cap; need async")
                return res
            except Exception as e:
                last = e
                print(f"  Gaia query attempt {i} ({launcher.__name__}) failed: {e}")
                time.sleep(5 * (i + 1))
    print(f"  Gaia ESA TAP failed ({last}); falling back to VizieR I/355/gaiadr3")
    return _query_gaia_vizier(ra, dec, radius)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='target basepath (writes <base>/catalogs/...)')
    ap.add_argument('--epoch', type=float, required=True, help='observation epoch (jyear)')
    ap.add_argument('--ra', type=float, required=True)
    ap.add_argument('--dec', type=float, required=True)
    ap.add_argument('--radius', type=float, default=0.1, help='query radius (deg)')
    ap.add_argument('--out-epoch-tag', default=None, help='epoch tag in filename, e.g. 2024.68')
    args = ap.parse_args()
    tag = args.out_epoch_tag or f'{args.epoch:.2f}'
    out = f'{args.base}/catalogs/gaia_virac2_refcat_epoch{tag}.fits'

    g = query_gaia(args.ra, args.dec, args.radius)
    gra, gdec = prop(farr(g['ra']), farr(g['dec']), farr(g['pmra']), farr(g['pmdec']),
                     args.epoch - GAIA_EPOCH)
    gfin = np.isfinite(gra) & np.isfinite(gdec)
    gaia_sc = SkyCoord(gra[gfin] * u.deg, gdec[gfin] * u.deg)

    v = query_virac2(args.ra, args.dec, args.radius)
    vra, vdec = prop(farr(v['RAJ2000']), farr(v['DEJ2000']), farr(v['pmRA']), farr(v['pmDE']),
                     args.epoch - VIRAC2_EPOCH)
    vfin = np.isfinite(vra) & np.isfinite(vdec)
    virac_sc = SkyCoord(vra[vfin] * u.deg, vdec[vfin] * u.deg)
    vJ = farr(v['Jmag'])[vfin]

    idx, sep, _ = virac_sc.match_to_catalog_sky(gaia_sc)
    fill = sep > 0.3 * u.arcsec
    print(f"Gaia DR3: {gfin.sum()} sources; VIRAC2 fill (no Gaia <0.3\"): {fill.sum()} of {vfin.sum()}")

    rows_gaia = Table()
    rows_gaia['RA'] = gaia_sc.ra.deg
    rows_gaia['DEC'] = gaia_sc.dec.deg
    rows_gaia['source'] = np.full(len(gaia_sc), 'GaiaDR3', dtype='U8')
    rows_gaia['refmag'] = farr(g['phot_g_mean_mag'])[gfin]

    rows_v = Table()
    rows_v['RA'] = virac_sc.ra.deg[fill]
    rows_v['DEC'] = virac_sc.dec.deg[fill]
    rows_v['source'] = np.full(int(fill.sum()), 'VIRAC2', dtype='U8')
    rows_v['refmag'] = vJ[fill]

    ref = vstack([rows_gaia, rows_v])
    ref['skycoord'] = SkyCoord(ref['RA'] * u.deg, ref['DEC'] * u.deg)
    ref.meta['VERSION'] = 'gaia_dr3+virac2_fill'
    ref.meta['FRAME'] = 'Gaia DR3 (ICRS); VIRAC2 (II/387) tied to Gaia DR3 ~5 mas'
    ref.meta['EPOCH'] = args.epoch
    ref.meta['V2EPOCH'] = VIRAC2_EPOCH
    ref.meta['GAEPOCH'] = GAIA_EPOCH
    ref.meta['NOTE'] = (f'GC reference-frame policy: Gaia DR3 abs frame + VIRAC2 NIR fill, per-star '
                        f'PM-propagated (VIRAC2 from {VIRAC2_EPOCH}, Gaia from {GAIA_EPOCH}) to {args.epoch}.')
    ref.write(out, overwrite=True)
    ref.write(out.replace('.fits', '.ecsv'), overwrite=True)
    print(f"Wrote {out}: {len(ref)} sources ({len(rows_gaia)} Gaia + {len(rows_v)} VIRAC2 fill)")


if __name__ == '__main__':
    main()
