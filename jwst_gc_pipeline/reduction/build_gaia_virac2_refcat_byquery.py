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

from jwst_gc_pipeline.astrometry_utils import farr, prop

GAIA_EPOCH = 2016.0    # Gaia DR3 reference epoch
VIRAC2_EPOCH = 2014.0  # VIRAC2 reference epoch (Smith+2025 II/387: fixed at 2014.0)


class ThinReferenceCoverageError(RuntimeError):
    """The queried reference coverage is far below anything the sky can explain.

    This is a BROKEN-QUERY guard, not a thin-sky guard, and the distinction is
    the reason it is expressed as a DENSITY rather than a row count.  Measured
    2026-08-23 over all 139 program-10678 tile centres with a VizieR cone at
    0.02 deg against II/387 and I/355:

        thinnest tile   GC_130 (l=0.641, b=+0.006)   2094 VIRAC2 per cone
        median                                       4356
        densest         GC_49  (l=359.874, b=-0.030) 6184
        brick's refcat centre, same cone             2908  (its same-star tie reads ~0.6 mas)

    The whole survey spans a factor of 3.0 and the thinnest tile carries 0.72x
    the density of the field whose tie is known good, so no floor set for thin
    SKY would fire on this program -- VVV's bulge coverage is uniform over it,
    and the gradient is stellar density falling away from Sgr A*.  What a floor
    can still catch is a query that went wrong: a truncated VizieR response, a
    wrong --radius, a cone placed off the field.  Those return orders of
    magnitude less, so the floor sits at roughly HALF the thinnest real tile and
    two orders above a broken query, and as a density it applies unchanged at
    any radius (an absolute row count tuned for the CMZ would trip on legitimate
    non-GC fields -- ngc6334's refcats are ~1.4 MB against sgrb2's 8.4 MB for
    comparable radii).  Issue #415 gap 4.
    """


# Usable reference sources per square degree, below which the query is treated
# as broken.  1000 per 0.02 deg cone = 1000 / (pi * 0.02**2) ~ 8e5 deg^-2.
MIN_REF_DENSITY_PER_SQDEG = 8.0e5


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


def check_reference_coverage(n_usable, radius_deg, context="",
                             min_density=MIN_REF_DENSITY_PER_SQDEG):
    """Refuse to write a refcat whose usable source density is below the floor.

    ``n_usable`` counts the sources that survive the finite-position mask, i.e.
    the rows a tie can actually be measured against, not the rows the query
    returned.  Returns the measured density so a caller can record it.
    """
    area = np.pi * float(radius_deg) ** 2
    if area <= 0:
        raise ValueError(f"refcat coverage check: non-positive radius {radius_deg}")
    density = float(n_usable) / area
    if density < min_density:
        raise ThinReferenceCoverageError(
            f"reference coverage {density:.3g} usable sources/deg^2 "
            f"({n_usable} within {radius_deg} deg) is below the "
            f"{min_density:.3g} deg^-2 floor{' for ' + context if context else ''}.  "
            f"Every program-10678 tile measured 1.7e6-4.9e6 deg^-2 and the brick's "
            f"working refcat 2.3e6, so this is a BROKEN QUERY -- a truncated VizieR "
            f"response, the wrong --radius, or a cone off the field -- rather than "
            f"thin sky.  Check the query before building; --min-ref-density 0 "
            f"records a deliberate override.")
    return density


def refcat_filename(epoch_tag, obs_token=None):
    """``gaia_virac2_refcat_epoch<tag>[_o<obs>].fits``.

    The token is what ``astrometry_utils.pick_refcat`` matches on to give each
    observation its OWN catalog in a shared field directory; without one it
    falls back to the alphabetically last file for every observation alike.
    ``o`` is not doubled if the caller already passed ``o037``, and the number
    is zero-padded to three digits so ``37`` and ``037`` name one file.
    """
    if obs_token in (None, ''):
        return f'gaia_virac2_refcat_epoch{epoch_tag}.fits'
    token = str(obs_token).lstrip('oO')
    if token.isdigit():
        token = f'{int(token):03d}'
    return f'gaia_virac2_refcat_epoch{epoch_tag}_o{token}.fits'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='target basepath (writes <base>/catalogs/...)')
    ap.add_argument('--epoch', type=float, required=True, help='observation epoch (jyear)')
    ap.add_argument('--ra', type=float, required=True)
    ap.add_argument('--dec', type=float, required=True)
    ap.add_argument('--radius', type=float, default=0.1, help='query radius (deg)')
    ap.add_argument('--out-epoch-tag', default=None, help='epoch tag in filename, e.g. 2024.68')
    ap.add_argument('--obs-token', default=None, metavar='NNN',
                    help='observation number to stamp into the filename '
                         '(gaia_virac2_refcat_epoch<tag>_o<NNN>.fits).  REQUIRED '
                         'for a field whose observations share one directory: '
                         'pick_refcat hands an untokened catalog to every '
                         'observation, which for tiles arcminutes apart is the '
                         'wrong sky (gc2211 o023 took a -9.28" correction that '
                         'way).')
    ap.add_argument('--min-ref-density', type=float,
                    default=MIN_REF_DENSITY_PER_SQDEG, metavar='PER_SQDEG',
                    help='refuse to build below this usable-source density '
                         '(deg^-2).  A BROKEN-QUERY guard: every program-10678 '
                         'tile measures 1.7e6-4.9e6 and the brick refcat 2.3e6, '
                         'so the default sits ~2x below the thinnest real sky '
                         'and two orders above a truncated query.  0 disables '
                         'it, and setting it is the record of that decision.')
    args = ap.parse_args()
    tag = args.out_epoch_tag or f'{args.epoch:.2f}'
    out = f'{args.base}/catalogs/{refcat_filename(tag, args.obs_token)}'

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

    # Coverage floor BEFORE anything is written (issue #415 gap 4).  Counts the
    # sources that survive the finite-position masks -- the rows a tie can be
    # measured against -- not the rows the query returned.
    density = check_reference_coverage(
        int(gfin.sum()) + int(vfin.sum()), args.radius,
        context=f"{args.base} epoch {args.epoch} at ({args.ra}, {args.dec})",
        min_density=args.min_ref_density)
    print(f"reference coverage: {density:.3g} usable sources/deg^2 "
          f"(floor {args.min_ref_density:.3g})")

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
    ref.meta['REFDENS'] = density
    ref.meta['NOTE'] = (f'GC reference-frame policy: Gaia DR3 abs frame + VIRAC2 NIR fill, per-star '
                        f'PM-propagated (VIRAC2 from {VIRAC2_EPOCH}, Gaia from {GAIA_EPOCH}) to {args.epoch}.')
    ref.write(out, overwrite=True)
    ref.write(out.replace('.fits', '.ecsv'), overwrite=True)
    print(f"Wrote {out}: {len(ref)} sources ({len(rows_gaia)} Gaia + {len(rows_v)} VIRAC2 fill)")


if __name__ == '__main__':
    main()
