import numpy as np
import time
import datetime
import os
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from astropy.io import fits
import glob
from photutils.background import MMMBackground, MADStdBackgroundRMS
from photutils.aperture import CircularAperture, CircularAnnulus
from photutils.detection import DAOStarFinder, IRAFStarFinder, find_peaks
from photutils.psf import (extract_stars, EPSFBuilder)
from astropy.modeling.fitting import LevMarLSQFitter
from astropy import stats
from astropy.table import Table, Column, MaskedColumn
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import coordinates
from astropy.visualization import simple_norm
from astropy import wcs
from astropy import table
from astropy import units as u
from astroquery.svo_fps import SvoFps
from astropy.stats import sigma_clip, mad_std
import dask
import dask.array
import yaml # DEBUG 2025-12-11
import yaml.representer # DEBUG 2025-12-11

from tqdm.auto import tqdm

import pylab as pl
pl.rcParams['figure.facecolor'] = 'w'
pl.rcParams['image.origin'] = 'lower'
pl.rcParams['figure.figsize'] = (10, 8)
pl.rcParams['figure.dpi'] = 100

# https://en.wikipedia.org/wiki/AB_magnitude
ABMAG_OFFSET = 8.90

# Instrument/filter tokens live in photometry/naming.py (heavy-import-free) so
# this module shares one source of truth without importing webbpsf.  The
# flags-based bgsub token is imported as ``_bgsub_token`` (this module calls it
# with explicit booleans, matching the producer-side names).
from jwst_gc_pipeline.photometry.naming import (
    MIRI_FILTERS, _inst_token, _svo_filter_id,
    _bgsub_token_from_flags as _bgsub_token,
)

filternames = filternames_narrow = ['f410m', 'f212n', 'f466n', 'f405n', 'f187n', 'f182m']
all_filternames = ['f410m', 'f212n', 'f466n', 'f405n', 'f187n', 'f182m', 'f444w', 'f356w', 'f200w', 'f115w']
obs_filters = {'brick': {'2221': filternames,
                         '1182': ['f444w', 'f356w', 'f200w', 'f115w'],
                         },
               'cloudc': {'2221': filternames},
               # sickle NIRCam (obs 007) + MIRI (obs 001/002/003)
               'sickle': {'3958': ['f187n', 'f210m', 'f335m', 'f470n', 'f480m',
                                   'f770w', 'f1130w', 'f1500w']},
               # cloudef NIRCam (obs 002/005) + MIRI (obs 004/006/008)
               'cloudef': {'2092': ['f162m', 'f210m', 'f360m', 'f480m',
                                    'f770w', 'f2100w']},
               'sgrc': {'4147': ['f115w', 'f162m', 'f182m', 'f212n', 'f360m', 'f405n', 'f470n', 'f480m']},
               'sgrb2': {'5365': ['f150w', 'f182m', 'f187n', 'f210m', 'f212n', 'f300m', 'f360m', 'f405n', 'f410m', 'f466n', 'f480m']},
               'arches': {'2045': ['f212n', 'f323n']},
               'quintuplet': {'2045': ['f212n', 'f323n']},
               'sgra': {'1939': ['f115w', 'f212n', 'f405n']},
               'gc2211': {'2211': ['f150w', 'f200w', 'f277w']},
               # Westerlund 1 (Guarcello 1905) + Westerlund 2 (Guarcello 3523)
               'wd1': {'1905': ['f115w', 'f150w', 'f164n', 'f187n', 'f200w', 'f212n',
                                'f277w', 'f323n', 'f405n', 'f444w', 'f466n']},
               'wd2': {'3523': ['f115w', 'f150w', 'f162m', 'f164n', 'f182m', 'f187n',
                                'f200w', 'f212n', 'f250m', 'f277w', 'f300m', 'f323n',
                                'f335m', 'f405n', 'f410m', 'f444w', 'f466n']},
               # W51 (Goddard prop 6151 NIRCam obs 001).  In disk -- use Gaia DR3
               # as astrometric ref.  Filter list per user 2026-06-13: F140M
               # F162M F182M F187N F210M F335M F360M F405N F410M F480M.
               'w51': {'6151': ['f140m', 'f162m', 'f182m', 'f187n', 'f210m',
                                'f335m', 'f360m', 'f405n', 'f410m', 'f480m']},
               }

# Using the 'brick' keyword here makes it work for now, need to figure out how to
# refactor it in cases where there are more filters available for other targets!
filter_to_project = {vv: key for target_filters in obs_filters.values() for key, val in target_filters.items() for vv in val}
# need to refactor this somehow for cloudc
# project_obsnum = {'2221': '001',
#                   '1182': '004',
#                  }

project_obsnum = {'brick': {'2221': '001',
                            '1182': '004',
                            },
                  'cloudc': {'2221': '002',
                             },
                  # sickle NIRCam is obs 007 but the MIRI pointings are obs
                  # 001/002/003; use a wildcard (the per-filter glob already
                  # disambiguates because the filter token is in the name).
                  'sickle': {'3958': '*',
                             },
                  # cloudef (proposal 2092) covers Cloud E (obs 002, t001)
                  # and Cloud F (obs 005, t002) as two adjacent pointings.
                  # The pipeline output retains t001 in both names (the
                  # asn rename hardcodes t001), so the per-filter i2d glob
                  # only needs an obs-id wildcard.
                  'cloudef': {'2092': '*',
                              },
                  'sgrc': {'4147': '012',
                           },
                  'sgrb2': {'5365': '001',
                            },
                  'arches': {'2045': '001',
                             },
                  'quintuplet': {'2045': '003',
                                 },
                  'sgra': {'1939': '001',
                           },
                  # gc2211 (asteroid proposal 2211) has 5 GC pointings
                  # (obs IDs 023, 028, 046, 049, 050) all sharing the
                  # python target name 'gc2211'.  Use a glob wildcard
                  # for the obs ID so per-filter merge picks up i2d
                  # files from any of the 5 pointings.
                  'gc2211': {'2211': '*',
                             },
                  'wd1': {'1905': '001',
                          },
                  'wd2': {'3523': '005',
                          },
                  'w51': {'6151': '001',
                          '1182': '002',
                          },
                  }


def getmtime(x):
    return datetime.datetime.fromtimestamp(os.path.getmtime(x)).strftime('%Y-%m-%d %H:%M:%S')


def tryint(x):
    # int(x) can only raise ValueError/TypeError on non-numeric input.
    try:
        return int(x)
    except (ValueError, TypeError):
        return -1


def sanity_check_individual_table(tbl):
    wl = filtername = tbl.meta['filter']
    print(f"SANITY CHECK {wl}")

    tbl = tbl.copy()
    tbl.sort('flux_jy')
    finite_fluxes = tbl['flux_jy'] > 0

    jfilts = SvoFps.get_filter_list('JWST')
    jfilts.add_index('filterID')
    zeropoint = u.Quantity(jfilts.loc[_svo_filter_id(filtername)]['ZeroPoint'], u.Jy)

    flux_jy = tbl['flux_jy'][finite_fluxes].quantity
    abmag_tbl = tbl['mag_ab'][finite_fluxes].quantity

    vegamag = -2.5 * np.log10(flux_jy / zeropoint) * u.mag
    abmag = (-2.5 * np.log10(flux_jy / u.Jy) + ABMAG_OFFSET) * u.mag

    print(f'Units of abmag columns are: abmag={abmag.unit}, abmag_tbl={abmag_tbl.unit}')
    assert abmag.unit == u.mag
    assert abmag_tbl.unit == u.mag
    assert flux_jy.unit == u.Jy

    # there are negative fluxes -> nan mags
    # print(f"Maximum difference between the two tables: {np.abs(abmag-abmag_tbl).max()}")
    print(f"Nanmax difference between the two tables: {np.nanmax(np.abs(abmag-abmag_tbl))}")
    # print("NaNs: (mag, flux) ", abmag_tbl[np.isnan(abmag_tbl)], flux_jy[np.isnan(abmag_tbl)])

    fluxcolname = 'flux' if 'flux' in tbl.colnames else 'flux_fit'
    print(f"Max flux in tbl for {wl}: {tbl[fluxcolname].max()};"
          f" in jy={flux_jy.max()}; magmin={abmag_tbl.min()}={np.nanmin(abmag_tbl)}, magmax={abmag_tbl.max()}={np.nanmax(abmag_tbl)}")
    print(f"100th brightest flux={flux_jy[-100]} abmag={abmag[-100]} abmag_tbl={abmag_tbl[-100]}")


def nanaverage_numpy(data, weights, **kwargs):
    print(data.shape, weights.shape)
    weights = np.where(np.isnan(data) | np.isnan(weights), 0, weights)
    bad = np.all(weights == 0, axis=1)
    weights[bad, :] = 1
    avg = np.average(np.nan_to_num(data),
                     weights=weights,
                     **kwargs
                     )
    avg[bad] = np.nan
    return avg


def nanaverage_dask(data, weights, **kwargs):
    weights = dask.array.from_array(weights)
    data = dask.array.from_array(data)
    weights = dask.array.where(dask.array.isnan(data) | dask.array.isnan(weights), 0, weights)
    bad = dask.array.all(weights == 0, axis=1)
    weights[bad, :] = 1
    avg = dask.array.average(np.nan_to_num(data), weights=weights, **kwargs)
    avg[bad] = np.nan
    return avg.compute()


def shift_individual_catalog(tbl, offsets_table, verbose=True):
    """
    offsets_table:
        A table to use to re-calculate sky coordinates from the WCS after
        shifting it.  This can be used because the catalogs are all
        intrinsically in pixel space, so changing the shift after the fact is OK.
        Using an offset table enables splitting out the re-alignment task from
        here; I want to be able to measure the alignment and be sure it's right
        before applying it.
    """
    if 'Visit' in tbl.meta:
        visit = int(tbl.meta['Visit'])
    elif 'VISIT' in tbl.meta:
        visit = int(tbl.meta['VISIT'])
    elif 'visit' in tbl.meta:
        visit = int(tbl.meta['visit'])
    else:
        print(tbl.meta)
        raise KeyError("'Visit' not found in meta")
    exposure = int(tbl.meta['EXPOSURE'][-5:])
    thismodule = tbl.meta['MODULE']
    if thismodule.endswith('a') or thismodule.endswith('b'):
        thismodule = thismodule+'long'
    filtername = tbl.meta['FILTER']

    offsets_visit_number = np.array([int(vis[-3:]) for vis in offsets_table['Visit']])

    match = ((offsets_visit_number == visit) &
             (offsets_table['Exposure'] == exposure) &
             ((offsets_table['Module'] == thismodule) | (offsets_table['Module'] == thismodule.strip('1234'))) &
             (offsets_table['Filter'] == filtername)
             )

    assert match.sum() == 1
    row = offsets_table[match]

    if 'RAOFFSET' in tbl.meta:
        raoffset = tbl.meta['RAOFFSET'] * u.arcsec
        decoffset = tbl.meta['DEOFFSET'] * u.arcsec
    else:
        # not measured, so we have to assume zero
        raoffset = 0 * u.arcsec
        decoffset = 0 * u.arcsec

    dra = row['dra'][0]*u.arcsec
    ddec = row['ddec'][0]*u.arcsec

    skycoord_colname = 'skycoord' if 'skycoord' in tbl.colnames else 'skycoord_centroid'

    skycoord = tbl[skycoord_colname]
    skycoord = SkyCoord(ra=skycoord.ra - raoffset + dra, dec=skycoord.dec - decoffset + ddec, frame=skycoord.frame)
    tbl[skycoord_colname] = skycoord

    print(f"Shifted table from {raoffset:0.4f},{decoffset:0.4f} to {dra:0.4f},{ddec:0.4f}, a difference of {dra-raoffset:0.4f},{ddec-decoffset:0.4f}")

    return tbl


def combine_singleframe(tbls, max_offset=0.10 * u.arcsec, realign=False, nanaverage=nanaverage_dask,
                        min_offset=0.10*u.arcsec,
                        offsets_table=None,
                        verbose=True
                        ):
    """

    min_offset :
        The minimum allowed offset to declare a 'new' star.  Anything below this is assumed same star.

    offsets_table:
        A table to use to re-calculate sky coordinates from the WCS after
        shifting it.  This can be used because the catalogs are all
        intrinsically in pixel space, so changing the shift after the fact is OK.
        Using an offset table enables splitting out the re-alignment task from
        here; I want to be able to measure the alignment and be sure it's right
        before applying it.
    """
    if offsets_table is not None:
        tbls = [shift_individual_catalog(tbl, offsets_table, verbose=verbose) for tbl in tbls]

    # set up DAO vs crowd column names
    if 'qf' in tbls[0].colnames:
        qfcn = 'qf'
        ffcn = 'fracflux'
        flux_error_colname = 'dflux'
        flux_colname = 'flux'
        skycoord_colname = 'skycoord'
        column_names = (flux_colname, flux_error_colname, 'qf', 'rchi2', 'fracflux', 'fwhm', 'fluxiso', 'flags', 'spread_model', 'sky', 'ra', 'dec', 'dra', 'ddec', )
        dao = False
    else:
        dao = True
        qfcn = 'qfit'
        ffcn = 'cfit'
        flux_error_colname = 'flux_err'
        flux_colname = 'flux_fit'
        # skycoord comes in as skycoord_centroid but we want it to leave as skycoord
        skycoord_colname = 'skycoord_centroid'
        column_names = (flux_colname, flux_error_colname, 'qfit', 'cfit', 'flux_init', 'flags', 'local_bkg', 'iter_detected', 'group_id', 'group_size', 'ra', 'dec', 'dra', 'ddec', )

    # Loop 1: Add new sources, which are any that don't have a match in the existing catalog closer than min_offset
    # this loop _only_ adds new sources
    basecrds = None   # set by the first NON-EMPTY frame
    for ii, tbl in enumerate(tbls):
        crds = tbl[skycoord_colname]
        # corner case: some fits resulted in flagged x, y that propagate through.  A parallel edit to crowdsource_catalogs_long.py removes these at the source, but I'm adding a catch here too
        bad = np.isnan(crds.ra) | np.isnan(crds.dec)
        if np.any(bad):
            tbl = tbl[~bad]
            crds = crds[~bad]
            tbls[ii] = tbl

        # A frame can legitimately have ZERO sources (sickle F1500W: longest
        # wavelength, mostly extended emission -> some exposures detect nothing).
        # match_to_catalog_sky crashes on a length-0 catalog, so skip empty frames
        # (they contribute no new sources) and let the first non-empty frame seed
        # the base.
        if len(crds) == 0:
            print(f"combine_singleframe: exposure "
                  f"{tbl.meta.get('exposure','?')} has 0 sources; skipping",
                  flush=True)
            continue
        if basecrds is None:
            basecrds = crds
        else:
            matches, sep, _ = crds.match_to_catalog_sky(basecrds, nthneighbor=1)
            reverse_matches, reverse_sep, _ = basecrds.match_to_catalog_sky(crds, nthneighbor=1)

            # add new sources to the cat iff their separation from an existing source in the catalog is >min
            keep = sep > min_offset

            newcrds = crds[keep]
            basecrds = SkyCoord([basecrds, newcrds])
            print(f"Added {len(newcrds)} new sources in exposure {tbl.meta['exposure']} {tbl.meta['MODULE'] if 'MODULE' in tbl.meta else ''} [total={len(basecrds)}]")
            # f" ({mutual_matches.sum()} mutual matches ({(~mutual_matches).sum()} not), {(sep > max_offset).sum()} above {max_offset}, keeping {keep.sum()}), ", flush=True)
        print(f"Iteration {ii}: There are a total of {len(basecrds)} sources in the base coordinate list [method={'daophot' if dao else 'crowdsource'}]")

    # do one loop of re-matching
    # We use only mutual best-matches for the realignment measurement to avoid spurious matches, e.g., if there are three stars in a line, we only want to match two if they are each other's best match
    print("Starting re-matching", flush=True)
    for ii, tbl in enumerate(tbls):
        crds = tbl[skycoord_colname]
        if len(crds) == 0:   # empty exposure -> nothing to re-match
            tbl.meta['ra_offset'] = np.nan
            tbl.meta['dec_offset'] = np.nan
            tbl.meta['dra_offset'] = np.nan
            tbl.meta['ddec_offset'] = np.nan
            continue

        match_inds, sep, _ = crds.match_to_catalog_sky(basecrds, nthneighbor=1)
        reverse_match_inds, reverse_sep, _ = basecrds.match_to_catalog_sky(crds, nthneighbor=1)
        mutual_reverse_matches = (match_inds[reverse_match_inds] == np.arange(len(reverse_match_inds)))
        mutual_matches = (reverse_match_inds[match_inds] == np.arange(len(match_inds)))

        # do one iteration of bulk offset measurement
        radiff = (crds.ra[reverse_match_inds[mutual_reverse_matches]] - basecrds[mutual_reverse_matches].ra).to(u.arcsec)
        decdiff = (crds.dec[reverse_match_inds[mutual_reverse_matches]] - basecrds[mutual_reverse_matches].dec).to(u.arcsec)

        # don't allow sep=0, since that's self-reference.  Use stringent qf, fracflux
        # DEBUG print(f"len(crds) = {len(crds)} len(basecrds) = {len(basecrds)} len(match_inds)={len(match_inds)} match_inds.max={match_inds.max()} len(reverse_match_inds)={len(reverse_match_inds)} reverse_match_inds.max={reverse_match_inds.max()} len(mutual_matches)={len(mutual_matches)}")
        if dao:
            oksep = (reverse_sep[mutual_reverse_matches] < max_offset) & (reverse_sep[mutual_reverse_matches] != 0) & (tbl[reverse_match_inds[mutual_reverse_matches]][qfcn] < 0.40) & (tbl[reverse_match_inds[mutual_reverse_matches]][ffcn] < 0.40)
        else:
            oksep = (reverse_sep[mutual_reverse_matches] < max_offset) & (reverse_sep[mutual_reverse_matches] != 0) & (tbl[reverse_match_inds[mutual_reverse_matches]][qfcn] > 0.95) & (tbl[reverse_match_inds[mutual_reverse_matches]][ffcn] > 0.85)
        medsep_ra, medsep_dec = np.median(radiff[oksep]), np.median(decdiff[oksep])
        dmedsep_ra, dmedsep_dec = mad_std(radiff[oksep]), mad_std(decdiff[oksep])
        tbl.meta['ra_offset'] = medsep_ra
        tbl.meta['dec_offset'] = medsep_dec
        tbl.meta['dra_offset'] = dmedsep_ra
        tbl.meta['ddec_offset'] = dmedsep_dec

        with fits.open(tbl.meta['FILENAME']) as fh:
            if 'RAOFFSET' in fh['SCI'].header:
                dra_header = fh['SCI'].header['RAOFFSET']
                ddec_header = fh['SCI'].header['DEOFFSET']
            else:
                # assume zero
                dra_header = 0.0
                ddec_header = 0.0

        print(f"Exposure {tbl.meta['exposure']} {tbl.meta['MODULE' if 'MODULE' in tbl.meta else '']} was offset by {medsep_ra.to(u.marcsec):10.3f}+/-{dmedsep_ra.to(u.marcsec):7.3f},"
              f" {medsep_dec.to(u.marcsec):10.3f}+/-{dmedsep_dec.to(u.marcsec):7.3f} based on {oksep.sum()} matches.  dra={dra_header:7.5g} ddec={ddec_header:7.5g}")

        # for tbl0, should be nan (all self-match)
        if realign and not np.isnan(medsep_ra) and not np.isnan(medsep_dec):
            newcrds = SkyCoord(crds.ra - medsep_ra, crds.dec - medsep_dec, frame=crds.frame)
            tbl[skycoord_colname] = newcrds

    if realign:
        print("Realigning")
        # remake base coordinates after the rematching
        basecrds = None
        for ii, tbl in enumerate(tbls):
            crds = tbl[skycoord_colname]
            if len(crds) == 0:
                continue
            if basecrds is None:
                basecrds = crds
            else:
                matches, sep, _ = crds.match_to_catalog_sky(basecrds, nthneighbor=1)
                
                # add new sources to the cat iff their separation from an existing source in the catalog is >min
                keep = (sep > min_offset)

                newcrds = crds[keep]
                basecrds = SkyCoord([basecrds, newcrds])
                print(f"Added {len(newcrds)} new sources in exposure {tbl.meta['exposure']} {tbl.meta['MODULE' if 'MODULE' in tbl.meta else '']}")

    print(f"There are a total of {len(basecrds)} sources in the base coordinate list after rematching")

    assert flux_error_colname in tbls[0].colnames
    assert flux_error_colname in column_names

    # -- Memory-streaming refactor (2026-04-23) -------------------------------
    # The previous version allocated all per-column 2-D arrays of shape
    # (n_src, n_tbl) up front -- ~94 GB for the F200W brick merge, which
    # OOM'd even at 512 GB due to squaring temporaries later on.  The
    # refactor below runs in two phases:
    #
    #   Phase 1 -- allocate just ra/dec/flux/flux_err to compute the
    #              mask (sigma-clip on flux and position), weights, the
    #              position averages, and the flux/flux_err reductions.
    #              Saves per-tbl ``match_inds`` and ``keep`` so Phase 2
    #              doesn't repeat the expensive ``match_to_catalog_sky``.
    #              Peak Phase 1 memory for F200W brick ~= 4 x 4.4M x 192
    #              x 8 bytes = 27 GB.
    #   Phase 2 -- stream each remaining column one at a time: allocate
    #              one 2-D array, fill from saved match indices, compute
    #              ``_avg`` + ``std_*_avg``, free.  Peak ~7 GB per column.
    #
    # Output columns preserved: the 1-D ``_avg`` / ``std_*_avg`` columns
    # for every key in ``column_names``, plus ``skycoord_avg``, ``nmatch``,
    # ``nmatch_good``, ``std_ra``, ``std_dec``, and
    # ``f'{flux_error_colname}_prop'``.  The 2-D per-exposure arrays are
    # NOT kept in the returned table (they were the memory culprit and
    # downstream code doesn't consume them; the _allcols variant written
    # by the caller now contains only these per-source columns).
    # ------------------------------------------------------------------------

    n_src = len(basecrds)
    n_tbl = len(tbls)

    # Save per-tbl match results for Phase 2 reuse.
    # match_inds is a length-n_tbl_rows int array (index into basecrds for
    # each row in this tbl).  Kept as int32 to save RAM.
    saved_match_inds = [None] * n_tbl
    saved_keep = [None] * n_tbl

    # Phase 1: ra/dec/flux/flux_err stack.
    # ra/dec stay float64: at ~270 deg, float32 has ~0.1" quantum which
    # destroys astrometric precision (we need ~5 mas).  flux and
    # flux_err are fine in float32 (values span 10-1e6, flux ratios and
    # sigma-clip masks are tolerant to 1e-7 relative precision).
    print(f"Phase 1: stacking ra/dec/flux/flux_err for {n_src} sources x {n_tbl} tables", flush=True)
    arr_ra = np.full((n_src, n_tbl), np.nan, dtype='float64')
    arr_dec = np.full((n_src, n_tbl), np.nan, dtype='float64')
    arr_flux = np.full((n_src, n_tbl), np.nan, dtype='float32')
    arr_fluxerr = np.full((n_src, n_tbl), np.nan, dtype='float32')

    for ii, tbl in enumerate(tqdm(tbls, desc='Phase 1 (ra/dec/flux stack)')):
        crds = tbl[skycoord_colname]
        if len(crds) == 0:
            # empty exposure (e.g. F1500W frame with no detections): contributes
            # no matches; leave its column NaN.
            saved_match_inds[ii] = np.array([], dtype=np.int32)
            saved_keep[ii] = np.array([], dtype=bool)
            print(f"P1 {ii}: exposure {tbl.meta.get('exposure','?')} has 0 "
                  f"sources; skipping", flush=True)
            continue
        match_inds, sep, _ = crds.match_to_catalog_sky(basecrds, nthneighbor=1)
        reverse_match_inds, _, _ = basecrds.match_to_catalog_sky(crds, nthneighbor=1)
        mutual_matches = (reverse_match_inds[match_inds] == np.arange(len(match_inds)))
        keep = (sep < max_offset) & mutual_matches

        # Cast match_inds to int32 to halve memory (n_src < 2^31 in
        # practice) and keep keep as bool.
        saved_match_inds[ii] = match_inds.astype(np.int32, copy=False)
        saved_keep[ii] = keep

        mi_keep = match_inds[keep]
        arr_ra[mi_keep, ii] = crds.ra.deg[keep]
        arr_dec[mi_keep, ii] = crds.dec.deg[keep]
        arr_flux[mi_keep, ii] = tbl[flux_colname][keep]
        arr_fluxerr[mi_keep, ii] = tbl[flux_error_colname][keep]
        print(f"P1 {ii}: Added {keep.sum()} of {len(keep)} sources from exposure "
              f"{tbl.meta['exposure']} {tbl.meta['MODULE'] if 'MODULE' in tbl.meta else ''} [total={n_src}]",
              flush=True)

    print("Phase 1 stack done; computing mask / weights / position averages", flush=True)
    nmatch = np.isfinite(arr_flux).sum(axis=1).astype(np.int32)

    # Sigma-clip flux and positions to identify per-source outliers.
    clip_flux = sigma_clip(arr_flux, stdfunc='mad_std', axis=1)
    clip_ra = sigma_clip(arr_ra, stdfunc='mad_std', axis=1)
    clip_dec = sigma_clip(arr_dec, stdfunc='mad_std', axis=1)
    to_mask = clip_flux.mask | clip_ra.mask | clip_dec.mask
    keepmask = ~to_mask
    nmatch_good = keepmask.sum(axis=1).astype(np.int32)

    # free clip objects (they reference the big arrays)
    del clip_flux, clip_ra, clip_dec

    # weights: inverse-variance flux weighting, zeroed where masked
    weights = (1.0 / (arr_fluxerr**2)) * keepmask

    # 2026-06-04: position averages MUST NOT depend on flux_err.
    # Per-frame fits with NaN flux_err (flag=48/49 near_bound +
    # no_covariance: singular covariance matrix but flux_fit / x_fit /
    # y_fit are valid) would otherwise contribute zero weight to
    # position averaging.  A source whose every matched frame has NaN
    # flux_err, or whose few valid-err matches all get sigma-clipped
    # by clip_ra/dec on tiny position spread, ends up with all-zero
    # weights -> nanaverage returns NaN -> avg_ra/dec NaN -> the
    # source is then dropped by the minimal-step NaN-sky reject in
    # ``combine_singleframe``'s caller.  Diagnosed on sickle F480M
    # iter3 star 2 faint target (basecrds[25338]): 8 valid finite
    # positions across 8 frames, 6 of 8 had NaN flux_err, the 2
    # valid-err entries were clipped by clip_ra -> all-zero weights ->
    # NaN avg -> source dropped.  ~31k of 45k merged-exposure rows
    # were silently lost this way before the fix.  Use uniform
    # (keepmask) weights for position averaging.
    pos_weights = keepmask.astype('float32')

    # For flux / flux_err averages we keep inverse-variance weighting
    # where it is available, but fall back to uniform (keepmask)
    # weights per row when the inverse-variance weights are all NaN /
    # zero -- otherwise the same sources have NaN flux_avg.
    finite_iv_per_row = np.any(np.isfinite(weights) & (weights > 0),
                               axis=1)
    weights_with_fallback = np.where(finite_iv_per_row[:, None],
                                     weights, pos_weights)

    # position averages
    _t0 = time.time()
    print(f"Phase 1: position+flux nanaverages over {n_src}x{n_tbl} "
          f"(nanaverage={getattr(nanaverage, '__name__', nanaverage)})...", flush=True)
    avg_ra = nanaverage(arr_ra, axis=1, weights=pos_weights)
    avg_dec = nanaverage(arr_dec, axis=1, weights=pos_weights)
    std_ra = nanaverage((arr_ra - avg_ra[:, None])**2, weights=pos_weights, axis=1)**0.5
    std_dec = nanaverage((arr_dec - avg_dec[:, None])**2, weights=pos_weights, axis=1)**0.5
    avgpos = SkyCoord(avg_ra, avg_dec, unit=(u.deg, u.deg), frame='icrs')
    print(f"Phase 1: position averages done in {time.time()-_t0:.1f}s", flush=True)

    # free ra/dec arrays -- no longer needed
    del arr_ra, arr_dec

    # flux and flux_err reductions
    _t0 = time.time()
    flux_avg = nanaverage(arr_flux, weights=weights_with_fallback, axis=1)
    std_flux_avg = nanaverage((arr_flux - flux_avg[:, None])**2, weights=weights_with_fallback, axis=1)**0.5
    flux_err_avg = nanaverage(arr_fluxerr, weights=weights_with_fallback, axis=1)
    std_flux_err_avg = nanaverage((arr_fluxerr - flux_err_avg[:, None])**2, weights=weights_with_fallback, axis=1)**0.5
    # flux_err_prop uses original inverse-variance weights only, since it
    # is the formal propagated uncertainty 1/sqrt(sum(1/sigma^2)); rows
    # with no inverse-variance contribution legitimately have NaN here.
    flux_err_prop = (np.nansum(arr_fluxerr**2 * weights, axis=1)
                     / np.nansum(weights, axis=1))**0.5
    print(f"Phase 1: flux averages done in {time.time()-_t0:.1f}s", flush=True)

    # free phase-1 big arrays (keep weights / keepmask -- needed in Phase 2)
    del arr_flux, arr_fluxerr

    # Build newtbl with per-source columns
    newtbl = Table()
    newtbl.meta = dict(tbls[0].meta)
    newtbl.meta['offsets'] = {tbl.meta['exposure']: (tbl.meta['ra_offset'], tbl.meta['dec_offset'])
                              for tbl in tbls}
    newtbl['skycoord_avg'] = avgpos
    newtbl['std_ra'] = std_ra
    newtbl['std_dec'] = std_dec
    newtbl['nmatch'] = nmatch
    newtbl['nmatch_good'] = nmatch_good
    newtbl[f'{flux_colname}_avg'] = flux_avg
    newtbl[f'std_{flux_colname}_avg'] = std_flux_avg
    newtbl[f'{flux_error_colname}_avg'] = flux_err_avg
    newtbl[f'std_{flux_error_colname}_avg'] = std_flux_err_avg
    newtbl[f'{flux_error_colname}_prop'] = flux_err_prop
    newtbl.meta[f'{flux_error_colname}_prop'] = 'propagated uncertainty on flux = 1/sum(weights)'

    # Phase 2: stream the remaining columns one at a time.
    # ra/dec are skipped because their summaries are already in newtbl
    # (skycoord_avg, std_ra, std_dec).  flux and flux_err are already done.
    already_done = {flux_colname, flux_error_colname, 'ra', 'dec',
                    'skycoord', skycoord_colname}
    _p2_cols = [k for k in column_names
                if k not in already_done and k in tbls[0].colnames]
    print(f"Phase 2: streaming {len(_p2_cols)} remaining columns one at a time "
          f"(n_src={n_src}, n_tbl={n_tbl}, ~{n_src*n_tbl*4/1e9:.2f} GB/column "
          f"float32); columns={_p2_cols}", flush=True)
    for _ci, key in enumerate(column_names):
        if key in already_done:
            continue
        if key not in tbls[0].colnames:
            print(f"  Skipping {key} (not in tbls[0])", flush=True)
            continue
        _t0 = time.time()
        print(f"  Phase 2 [{_ci}] column {key!r}: allocating + filling "
              f"({n_src}x{n_tbl})...", flush=True)
        arr = np.full((n_src, n_tbl), np.nan, dtype='float32')
        for ii, tbl in enumerate(tbls):
            if key not in tbl.colnames:
                continue
            keep = saved_keep[ii]
            mi = saved_match_inds[ii]
            arr[mi[keep], ii] = tbl[key][keep]
        _t_fill = time.time()
        print(f"  Phase 2 [{_ci}] column {key!r}: fill done in {_t_fill-_t0:.1f}s; "
              f"running nanaverage (mean) over {n_src}x{n_tbl}...", flush=True)
        # Use weights_with_fallback (same rationale as flux averages
        # in Phase 1): a basecrds source whose every matched frame has
        # NaN flux_err must still produce a non-NaN average of qfit,
        # cfit, flags, etc. or it gets dropped at the minimal step.
        key_avg = nanaverage(arr, weights=weights_with_fallback, axis=1)
        _t_mean = time.time()
        print(f"  Phase 2 [{_ci}] column {key!r}: mean done in {_t_mean-_t_fill:.1f}s; "
              f"running nanaverage (std)...", flush=True)
        std_key = nanaverage((arr - key_avg[:, None])**2, weights=weights_with_fallback, axis=1)**0.5
        newtbl[f'{key}_avg'] = key_avg
        newtbl[f'std_{key}_avg'] = std_key
        del arr
        print(f"  Phase 2 [{_ci}] column {key!r}: DONE (std {time.time()-_t_mean:.1f}s, "
              f"total {time.time()-_t0:.1f}s)", flush=True)

    # weights, keepmask, saved_match_inds, saved_keep kept until function
    # return; Python will free them after caller drops newtbl reference.
    return newtbl


def _skycoord_colname_for(tbl):
    """Match combine_singleframe's column-name convention (crowd vs dao)."""
    return 'skycoord' if 'qf' in tbl.colnames else 'skycoord_centroid'


def _make_spatial_tiles(ras_deg, decs_deg, n_chunks):
    """Partition the sky into ~n_chunks count-balanced rectangular CORE tiles.

    Splits RA into nx quantile bands, then each band into ny quantile rows, so
    each tile holds roughly equal source COUNT (good load balance in fields with
    strong density gradients).  Returns a list of (ra_lo, ra_hi, dec_lo, dec_hi)
    half-open core boxes whose union covers all input points (outer edges padded
    so nothing falls outside).
    """
    n_chunks = max(1, int(n_chunks))
    nx = int(np.ceil(np.sqrt(n_chunks)))
    ny = int(np.ceil(n_chunks / nx))
    eps = 1e-9
    ra_edges = np.quantile(ras_deg, np.linspace(0, 1, nx + 1))
    ra_edges[0] -= eps
    ra_edges[-1] += eps
    tiles = []
    for ix in range(nx):
        rlo, rhi = ra_edges[ix], ra_edges[ix + 1]
        in_band = (ras_deg >= rlo) & (ras_deg < rhi)
        band_decs = decs_deg[in_band]
        if band_decs.size == 0:
            continue
        dec_edges = np.quantile(band_decs, np.linspace(0, 1, ny + 1))
        dec_edges[0] -= eps
        dec_edges[-1] += eps
        for iy in range(ny):
            tiles.append((rlo, rhi, dec_edges[iy], dec_edges[iy + 1]))
    return tiles


def _subset_tables_to_region(tbls, skycoord_colname, bounds, halo_deg):
    """Subset each table to the core box grown by halo_deg (RA halo scaled by
    1/cos(dec)).  Drops tables with no rows in the region.  Returns the list of
    non-empty subset tables (meta preserved)."""
    rlo, rhi, dlo, dhi = bounds
    dec_c = 0.5 * (dlo + dhi)
    halo_ra = halo_deg / max(np.cos(np.radians(dec_c)), 1e-6)
    out = []
    for tbl in tbls:
        sc = tbl[skycoord_colname]
        ra = sc.ra.deg
        dec = sc.dec.deg
        m = ((ra >= rlo - halo_ra) & (ra < rhi + halo_ra) &
             (dec >= dlo - halo_deg) & (dec < dhi + halo_deg) &
             np.isfinite(ra) & np.isfinite(dec))
        if m.any():
            sub = tbl[m]
            sub.meta = dict(tbl.meta)
            out.append(sub)
    return out


def _combine_chunk_worker(args):
    """Pickleable worker: combine one spatial tile, then keep only sources whose
    AVERAGED position falls in the tile CORE (so each star is owned by exactly
    one tile -- the halo copies in neighbouring tiles are dropped)."""
    subset_tables, bounds, kwargs = args
    if len(subset_tables) == 0:
        return None
    newtbl = combine_singleframe(subset_tables, verbose=False, **kwargs)
    if newtbl is None or len(newtbl) == 0:
        return newtbl
    rlo, rhi, dlo, dhi = bounds
    avg = newtbl['skycoord_avg']
    ra = avg.ra.deg
    dec = avg.dec.deg
    # half-open core ownership; NaN-avg rows are kept here (and dropped by the
    # caller's existing NaN-sky reject) so they are not silently lost.
    core = ((ra >= rlo) & (ra < rhi) & (dec >= dlo) & (dec < dhi)) | ~np.isfinite(ra) | ~np.isfinite(dec)
    return newtbl[core]


def combine_singleframe_chunked(tbls, n_chunks=8, halo=2.0 * u.arcsec,
                                workers=None, **kwargs):
    """Parallel, count-balanced spatial-tiling wrapper around combine_singleframe.

    The serial combine_singleframe builds (n_src x n_tbl) arrays and runs
    per-axis sigma-clip/nanaverage over them -- O(n_src) memory and serial CPU
    that dominate dense-field merges (brick/sgrc: ~78k sources/frame, days of
    wall time at ~1 core).  Here we split the sky into ~n_chunks count-balanced
    CORE tiles, grow each by a halo (>= the cross-match radius so every core
    star gathers ALL its per-frame detections), run combine_singleframe on each
    tile IN PARALLEL, and keep only sources whose averaged position lands in the
    tile core.  Core tiles are disjoint and cover the plane, so every star has
    exactly one owner -- no edge cross-match needed, no duplicates, none lost.

    Identical output schema to combine_singleframe.  n_chunks<=1 -> no chunking.
    """
    if n_chunks is None or int(n_chunks) <= 1 or len(tbls) == 0:
        return combine_singleframe(tbls, **kwargs)

    # Each tile runs in its own subprocess; use the plain-numpy reducer so we
    # don't spawn dask worker threads inside every subprocess (oversubscription).
    kwargs.setdefault('nanaverage', nanaverage_numpy)
    skycoord_colname = _skycoord_colname_for(tbls[0])
    # global detection distribution -> count-balanced tiles
    all_ra = np.concatenate([np.asarray(t[skycoord_colname].ra.deg) for t in tbls])
    all_dec = np.concatenate([np.asarray(t[skycoord_colname].dec.deg) for t in tbls])
    finite = np.isfinite(all_ra) & np.isfinite(all_dec)
    tiles = _make_spatial_tiles(all_ra[finite], all_dec[finite], n_chunks)
    halo_deg = halo.to(u.deg).value
    print(f"combine_singleframe_chunked: {len(tbls)} frames, "
          f"{int(finite.sum())} detections -> {len(tiles)} tiles "
          f"(halo={halo.to(u.arcsec):.2f}), workers={workers}", flush=True)

    jobs = []
    for bounds in tiles:
        subset = _subset_tables_to_region(tbls, skycoord_colname, bounds, halo_deg)
        if subset:
            jobs.append((subset, bounds, kwargs))

    results = []
    nworkers = max(1, int(workers or 1))
    if nworkers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(nworkers, len(jobs))) as ex:
            futs = {ex.submit(_combine_chunk_worker, j): j[1] for j in jobs}
            for fut in as_completed(futs):
                r = fut.result()
                if r is not None and len(r) > 0:
                    results.append(r)
    else:
        for j in jobs:
            r = _combine_chunk_worker(j)
            if r is not None and len(r) > 0:
                results.append(r)

    if not results:
        raise ValueError("combine_singleframe_chunked produced no rows from any tile")
    merged = table.vstack(results, metadata_conflicts='silent')
    merged.meta = dict(tbls[0].meta)
    print(f"combine_singleframe_chunked: stitched {len(results)} tiles "
          f"-> {len(merged)} merged sources", flush=True)
    return merged


def merge_catalogs(tbls, catalog_type='crowdsource', module='nrca',
                   ref_filter='f405n',
                   epsf=False, bgsub=False, desat=False, blur=False,
                   max_offset=0.10 * u.arcsec, target='brick',
                   indivexp=False,
                   qfcut=None, fracfluxcut=None,
                   min_nmatch_narrow=4,
                   iteration_label=None,
                   resbgsub=False,
                   basepath='/blue/adamginsburg/adamginsburg/jwst/brick/'):
    print(f'Starting merge catalogs: catalog_type: {catalog_type} module: {module} target: {target}', flush=True)

    if iteration_label in (None, ''):
        iter_token = ''
    elif str(iteration_label).startswith('_'):
        iter_token = str(iteration_label)
    else:
        iter_token = f'_{iteration_label}'

    epsf_ = "_epsf" if epsf else ""
    blur_ = "_blur" if blur else ""

    matching_ref_tables = [tb for tb in tbls if tb.meta['filter'] == ref_filter]
    if len(matching_ref_tables) == 0:
        # Silently switching ref filters changes the astrometric reference
        # of the merge.  Caller should fix the configuration explicitly.
        available = sorted({tb.meta['filter'] for tb in tbls})
        raise ValueError(
            f"Requested ref_filter={ref_filter!r} not present in input "
            f"tables; available filters = {available}"
        )
    basetable = matching_ref_tables[0].copy()
    basetable.meta['astrometric_reference_wavelength'] = ref_filter

    jfilts = SvoFps.get_filter_list('JWST')
    jfilts.add_index('filterID')

    desat = "_unsatstar" if desat else ""
    bgsub = _bgsub_token(bgsub, resbgsub)

    reffiltercol = [ref_filter] * len(basetable)
    print(f"Started with {len(basetable)} in filter {ref_filter}", flush=True)

    # build up a reference coordinate catalog by adding in those with no matches each time
    basecrds = basetable['skycoord']
    for tb in tqdm(tbls, desc='Table Meta Loop'):
        if tb.meta['filter'] == ref_filter:
            continue
        crds = tb['skycoord']

        matches, sep, _ = crds.match_to_catalog_sky(basecrds, nthneighbor=1)
        reverse_matches, reverse_sep, _ = basecrds.match_to_catalog_sky(crds, nthneighbor=1)

        mutual_matches = (reverse_matches[matches] == np.arange(len(matches)))

        newcrds = crds[(sep > max_offset) | (~mutual_matches)]
        basecrds = SkyCoord([basecrds, newcrds])

        reffiltercol += [tb.meta['filter']] * len(newcrds)
        print(f"Added {len(newcrds)} new sources in filter {tb.meta['filter']}", flush=True)
    print(f"Base coordinate length = {len(basecrds)}", flush=True)

    basetable = Table()
    basetable['skycoord_ref'] = basecrds
    basetable['skycoord_ref_filtername'] = reffiltercol

    # flag_near_saturated(basetable, filtername=ref_filter)
    # # replace_saturated adds more rows
    # replace_saturated(basetable, filtername=ref_filter)
    # print(f"filter {basetable.meta['filter']} has {len(basetable)} rows")

    meta = {}

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # for colname in basetable.colnames:
        #     basetable.rename_column(colname, colname+"_"+basetable.meta['filter'])

        for tbl in tqdm(tbls, desc='Table Loop'):
            t0 = time.time()
            wl = tbl.meta['filter']
            flag_near_saturated(tbl, filtername=wl, target=target, basepath=basepath)
            # replace_saturated adds more rows
            replace_saturated(tbl, filtername=wl, target=target, basepath=basepath)
            # DEBUG print(f"DEBUG: tbl['replaced_saturated'].sum(): {tbl['replaced_saturated'].sum()}")

            crds = tbl['skycoord']
            matches, sep, _ = basecrds.match_to_catalog_sky(crds, nthneighbor=1)
            reverse_matches, reverse_sep, _ = crds.match_to_catalog_sky(basecrds, nthneighbor=1)

            mutual_matches = (reverse_matches[matches] == np.arange(len(matches)))
            # limit to one-to-one nearest neighbor matches
            # matches = matches[mutual_matches]
            # sep = sep[mutual_matches]

            print(f"filter {wl} has {len(tbl)} rows.  {mutual_matches.sum()} of {len(tbl)} are mutual.  Matching took {time.time()-t0:0.1f} seconds", flush=True)

            # removed Jan 21, 2023 because this *should* be handled by the pipeline now
            # # do one iteration of bulk offset measurement
            # radiff = (crds.ra[matches]-basecrds.ra).to(u.arcsec)
            # decdiff = (crds.dec[matches]-basecrds.dec).to(u.arcsec)
            # oksep = sep < max_offset
            # medsep_ra, medsep_dec = np.median(radiff[oksep]), np.median(decdiff[oksep])
            # tbl.meta[f'ra_offset_from_{ref_filter}'] = medsep_ra
            # tbl.meta[f'dec_offset_from_{ref_filter}'] = medsep_dec
            # newcrds = SkyCoord(crds.ra - medsep_ra, crds.dec - medsep_dec, frame=crds.frame)
            # tbl['skycoord'] = newcrds
            # matches, sep, _ = basecrds.match_to_catalog_sky(newcrds, nthneighbor=1)

            basetable.add_column(name=f"sep_{wl}", col=sep)
            basetable.add_column(name=f"id_{wl}", col=matches)
            matchtb = tbl[matches]
            badsep = sep > max_offset
            for cn in matchtb.colnames:
                if isinstance(matchtb[cn], SkyCoord):
                    matchtb.rename_column(cn, f"{cn}_{wl}")
                    matchtb[f'mask_{wl}'] = badsep
                else:
                    matchtb[f'{cn}_{wl}'] = MaskedColumn(data=matchtb[cn], name=f'{cn}_{wl}')
                    matchtb[f'{cn}_{wl}'].mask[badsep] = True
                    # mask non-mutual matches
                    matchtb[f'{cn}_{wl}'].mask[~mutual_matches] = True
                    if hasattr(matchtb[cn], 'meta'):
                        matchtb[f'{cn}_{wl}'].meta = matchtb[cn].meta
                    matchtb.remove_column(cn)

            print(f"Max flux in tbl for {wl}: {tbl['flux'].max()}; in jy={np.nanmax(np.array(tbl['flux_jy']))}; mag={np.nanmin(np.array(tbl['mag_ab']))}")
            print(f"merging tables step: max flux for {wl} is {matchtb['flux_'+wl].max()} {matchtb['flux_jy_'+wl].max()} {matchtb['mag_ab_'+wl].min()}")
            print(f"Basetable has length {len(basetable)} and ncols={len(basetable.colnames)} before stack")

            basetable = table.hstack([basetable, matchtb], join_type='exact')
            meta[f'{wl[1:-1]}pxdg'.upper()] = tbl.meta['pixelscale_deg2']
            meta[f'{wl[1:-1]}pxas'.upper()] = tbl.meta['pixelscale_arcsec']
            for key in tbl.meta:
                if isinstance(tbl.meta[key], (str, int, float)):
                    meta[f'{wl[1:-1]}{key[:4]}'.upper()] = tbl.meta[key]
                else:
                    # specifically to handle the case of astropy.io.fits.card objects that are unserializable
                    meta[f'{wl[1:-1]}{key[:4]}'.upper()] = str(tbl.meta[key])

            print(f"Basetable has length {len(basetable)} and ncols={len(basetable.colnames)} after stack")
            print(f"merging tables step: max flux for {wl} in merged table is {basetable['flux_'+wl].max()}"
                  f" {np.nanmax(np.array(basetable['flux_jy_'+wl]))} {np.nanmin(np.array(basetable['mag_ab_'+wl]))}")
            # DEBUG
            # DEBUG if hasattr(basetable[f'{cn}_{wl}'], 'mask'):
            # DEBUG     print(f"Table has mask sum for column {cn} {basetable[cn+'_'+wl].mask.sum()}")
            # DEBUG if 'replaced_saturated_f410m' in basetable.colnames:
            # DEBUG     print(f"'replaced_saturated_f410m' has {basetable['replaced_saturated_f410m'].sum()}")
            # There can be more stars in replaced_saturated_f410m than there were stars replaced because
            # there can be multiple stars in the merged coordinate list whose closest match is a saturated
            # star.  i.e., there could be two coordinates that both see the same F410M flux.

            bad = np.isnan(tbl['mag_ab']) & (tbl['flux'] > 0)
            if any(bad):
                raise ValueError("Bad magnitudes for good fluxes")

            print(f"Flagged {tbl[f'near_saturated_{wl}'].sum()} stars that are near saturated stars "
                  f"in filter {wl} out of {len(tbl)}.  "
                  f"There are then {basetable[f'near_saturated_{wl}_{wl}'].sum()} in the merged table.  "
                  f"There are also {basetable[f'replaced_saturated_{wl}'].sum()} replaced saturated.", flush=True)

        print(f"Stacked all rows into table with len={len(basetable)}", flush=True)
        if 'flux_jy_f410m' in basetable.colnames and 'flux_jy_f405n' in basetable.colnames:
            zeropoint410 = u.Quantity(jfilts.loc['JWST/NIRCam.F410M']['ZeroPoint'], u.Jy)
            zeropoint405 = u.Quantity(jfilts.loc['JWST/NIRCam.F405N']['ZeroPoint'], u.Jy)

            # Line-subtract the F410 continuum band
            # 0.16 is from BrA_separation
            # 0.196 is the 'post-destreak' version, which might (?) be better
            # 0.11 is the theoretical version from RecombFilterDifferencing
            # 0.16 still looks like the best; 0.175ish is the median, but 0.16ish is the mode
            # but we use 0.11, the theoretica one, because we don't necessarily expect a good match!
            f405to410_scale = 0.11
            basetable.add_column(basetable['flux_jy_f410m'] - basetable['flux_jy_f405n'] * f405to410_scale, name='flux_jy_410m405')

            basetable.add_column(-2.5*np.log10(basetable['flux_jy_410m405']) + ABMAG_OFFSET, name='mag_ab_410m405')
            basetable.add_column(-2.5*np.log10(basetable['flux_jy_410m405'] / zeropoint410), name='mag_vega_410m405')
            # Then subtract that remainder back from the F405 band to get the continuum-subtracted F405
            basetable.add_column(basetable['flux_jy_f405n'] - basetable['flux_jy_410m405'], name='flux_jy_405m410')
            basetable.add_column(-2.5*np.log10(basetable['flux_jy_405m410']) + ABMAG_OFFSET, name='mag_ab_405m410')
            basetable.add_column(-2.5*np.log10(basetable['flux_jy_405m410'] / zeropoint405), name='mag_vega_405m410')

        if 'flux_jy_f182m' in basetable.colnames and 'flux_jy_f187n' in basetable.colnames:
            zeropoint182 = u.Quantity(jfilts.loc['JWST/NIRCam.F182M']['ZeroPoint'], u.Jy)
            zeropoint187 = u.Quantity(jfilts.loc['JWST/NIRCam.F187N']['ZeroPoint'], u.Jy)

            # Line-subtract the F182 continuum band
            # 0.11 is the theoretical bandwidth fraction
            # PaA_separation_nrcb gives 0.175ish -> 0.183 with "latest"
            # 0.18 is closer to the histogram mode
            f187to182_scale = 0.11
            basetable.add_column(basetable['flux_jy_f182m'] - basetable['flux_jy_f187n'] * f187to182_scale, name='flux_jy_182m187')
            basetable.add_column(-2.5*np.log10(basetable['flux_jy_182m187']) + ABMAG_OFFSET, name='mag_ab_182m187')
            basetable.add_column(-2.5*np.log10(basetable['flux_jy_182m187'] / zeropoint182), name='mag_vega_182m187')
            # Then subtract that remainder back from the F187 band to get the continuum-subtracted F187
            basetable.add_column(basetable['flux_jy_f187n'] - basetable['flux_jy_182m187'], name='flux_jy_187m182')
            basetable.add_column(-2.5*np.log10(basetable['flux_jy_187m182']) + ABMAG_OFFSET, name='mag_ab_187m182')
            basetable.add_column(-2.5*np.log10(basetable['flux_jy_187m182'] / zeropoint187), name='mag_vega_187m182')
        """ # this adds to the file size too much
        # Add some important colors

        colors=[('f410m', 'f466n'),
                ('f405n', 'f410m'),
                ('f405n', 'f466n'),
                ('f187n', 'f182m', ),
                ('f182m', 'f410m'),
                ('f182m', 'f212n', ),
                ('f187n', 'f405n'),
                ('f187n', 'f212n'),
                #('f212n', '410m405'), no emag defined
                ('f212n', 'f410m'),
                #('182m187', '410m405'), no emag defined
                ('f356w', 'f444w'),
                ('f356w', 'f410m'),
                ('f410m', 'f444w'),
                ('f405n', 'f444w'),
                ('f444w', 'f466n'),
                ('f200w', 'f356w'),
                ('f200w', 'f212n'),
                ('f182m', 'f200w'),
                ('f115w', 'f182m'),
                ('f115w', 'f212n'),
                ('f115w', 'f200w'),
            ]
        for c1, c2 in colors:
            if f'mag_ab_{c1}' in basetable.colnames and f'mag_ab_{c2}' in basetable.colnames:
                basetable.add_column(basetable[f'mag_ab_{c1}']-basetable[f'mag_ab_{c2}'], name=f'color_{c1}-{c2}')
                basetable.add_column((basetable[f'emag_ab_{c1}']**2 + basetable[f'emag_ab_{c2}']**2)**0.5, name=f'ecolor_{c1}-{c2}')
        """

        # DEBUG for colname in basetable.colnames:
        # DEBUG     print(f"colname {colname} has mask: {hasattr(basetable[colname], 'mask')}")
        basetable.meta = meta
        if '212PXDG' not in meta:
            print("WARNING: 212PXDG not present in metadata for this target")

        indivexp = '_indivexp' if indivexp else ''
        tablename = f"{basepath}/catalogs/{catalog_type}_{module}{indivexp}_photometry_tables_merged{desat}{bgsub}{epsf_}{blur_}{iter_token}"
        t0 = time.time()
        print(f"Writing table {tablename} with len={len(basetable)} and ncols={len(basetable.colnames)}", flush=True)
        # use caps b/c FITS will force it to caps anyway
        basetable.meta['VERSION'] = datetime.datetime.now().isoformat()
        # FITS can mishandle masked bool columns; force saturated flags to plain bool.
        for colname in basetable.colnames:
            if 'saturated' in colname:
                col = basetable[colname]
                if hasattr(col, 'mask'):
                    fixed = np.array(col.filled(False), dtype=bool)
                else:
                    arr = np.array(col)
                    if arr.dtype.kind in 'fc':
                        fixed = np.isfinite(arr) & (arr != 0)
                    else:
                        fixed = arr.astype(bool)
                basetable.replace_column(colname, Column(fixed, name=colname))
        # DO NOT USE FITS in production, it drops critical metadata
        # I wish I had noted *what* metadata it drops, though, since I still seem to be using
        # it in production code down the line...
        # OH, I think the FITS file turns "True" into "False"?
        # Yes, specifically: it DROPS masked data types, converting "masked" into "True"?
        basetable.write(f"{tablename}.fits", overwrite=True)
        print(f"Done writing table {tablename}.fits in {time.time()-t0:0.1f} seconds", flush=True)

        # strip out bad metadata that the yaml serializer can't handle
        for k, v in basetable.meta.items():
            # Check for astropy.io.fits.card.Undefined objects explicitly
            try:
                yaml.dump({k: v}, )
            except Exception as ex:
                if isinstance(v, fits.card.Undefined):
                    basetable.meta[k] = str(v)
                    print("BAD META (Undefined):", k, type(v), v)
                    continue
                else:
                    basetable.meta[k] = str(v)
                    print("BAD META:", k, type(v), v, ex)

        t0 = time.time()
        # takes FOR-EV-ER
        try:
            basetable.write(f"{tablename}.ecsv", overwrite=True)
        except yaml.representer.RepresenterError:
            import astropy
            print("astropy version: ", astropy.__version__)
            # https://github.com/astropy/astropy/pull/18677 ?
            # DEBUG
            print("YAML RepresenterError: trying again after removing masks")
            # for colname in basetable.colnames:
            #     print("DEBUG Column: ", colname, type(basetable[colname]), hasattr(basetable[colname], 'mask'))
            # for key in basetable.meta:
            #     print("DEBUG Meta: ", key, type(basetable.meta[key]), basetable.meta[key])
            raise
        print(f"Done writing table {tablename}.ecsv in {time.time()-t0:0.1f} seconds", flush=True)

        # keep any rows with at least two qf cut pass
        if qfcut is not None:
            qfkeep = np.array([basetable[qfkey] > qfcut for qfkey in basetable.colnames if 'qf' in qfkey]).sum(axis=0) > 1
            print(f"Keeping {qfkeep.sum()} sources of {len(basetable)} with qf > {qfcut}")
            basetable = basetable[qfkeep]

        if fracfluxcut is not None:
            fracfluxkeep = np.array([basetable[fracfluxkey] > fracfluxcut for fracfluxkey in basetable.colnames if 'fracflux' in fracfluxkey]).sum(axis=0) > 1
            print(f"Keeping {fracfluxkeep.sum()} sources of {len(basetable)} with fracflux > {fracfluxcut}")
            basetable = basetable[fracfluxkeep]

        if min_nmatch_narrow is not None:
            available_nmatch_cols = [f'nmatch_good_{filn}' for filn in filternames_narrow if f'nmatch_good_{filn}' in basetable.colnames]
            if len(available_nmatch_cols) >= 2:
                match_brick_narrow = np.array([basetable[colname] > min_nmatch_narrow for colname in available_nmatch_cols]).sum(axis=0) > 1
                print(f"Keeping {match_brick_narrow.sum()} sources of {len(basetable)} with at least {min_nmatch_narrow} matches in two or more of the narrower bands")
                basetable = basetable[match_brick_narrow]
            else:
                print("Skipping narrow-band nmatch cut: insufficient narrow-band columns for this target")

        if qfcut is not None or fracfluxcut is not None:
            print(f"Saving merged version with qualcuts: {tablename}_qualcuts.fits with len={len(basetable)}")
            basetable.write(f"{tablename}_qualcuts.fits", overwrite=True)

        sep_cols = [f'sep_{filtername}' for filtername in filternames if 'w' not in filtername and f'sep_{filtername}' in basetable.colnames]
        if len(sep_cols) >= 2:
            oksep = (np.array([basetable[colname] < 0.1*u.arcsec for colname in sep_cols]).sum(axis=0) > 1)
        else:
            oksep = np.ones(len(basetable), dtype=bool)
            print("Skipping oksep cut: insufficient sep_* columns for this target")
        print(f"Writing {tablename}_qualcuts_oksep2221.fits")
        basetable[oksep].write(f"{tablename}_qualcuts_oksep2221.fits", overwrite=True)


def merge_individual_frames(module='merged', suffix="", desat=False, filtername='f410m',
                                        progid='2221',
                                        bgsub=False, epsf=False, fitpsf=False, blur=False, target='brick',
                                        exposure_numbers=np.arange(1, 25),
                                        max_visitid=10,
                                        method='crowdsource',
                                        offsets_table=None,
                                        iteration_label=None,
                                        resbgsub=False,
                                        do_replace_saturated=True,
                                        group=False,
                                        fwhm_basepath=None,
                                        n_spatial_chunks=1,
                                        merge_workers=1,
                                        basepath='/blue/adamginsburg/adamginsburg/jwst/brick/'):

    desat = "_unsatstar" if desat else ""
    bgsub = _bgsub_token(bgsub, resbgsub)
    fitpsf = '_fitpsf' if fitpsf else ''
    blur_ = "_blur" if blur else ""
    # the per-frame writer inserts a ``_group`` token (between {blur_} and the
    # iteration token) when sources were fit with a SourceGrouper
    group_ = "_group" if group else ""
    # iter_token is inserted *between* the {blur_} block and the
    # {method_suffix} so the per-frame filename
    # ``..._{blur_}{iter_token}_{method_suffix}{suffix}.fits``
    # matches what crowdsource_catalogs_long.py writes for iter2/iter3.
    if iteration_label in (None, ''):
        iter_token = ''
    elif str(iteration_label).startswith('_'):
        iter_token = str(iteration_label)
    else:
        iter_token = f'_{iteration_label}'
    if module == 'merged':
        modules = ['nrca', 'nrcb',]
        modules += [f'nrc{ab}{n}' for ab in 'ab' for n in range(1, 5)]
        modules += ['nrcalong', 'nrcblong']  # sgrb2 LW modules
        # iter3 LW per-frame outputs are written with module='merged'
        # in the filename (the script passes --modules=merged); include
        # the literal 'merged' token so the merge glob picks them up.
        modules += ['merged']
    elif module in ('nrca', 'nrcb'):
        # Exposure-level catalogs are saved with detector-qualified module
        # names.  SW exposures use nrc{a,b}1..4; LW exposures use nrc{a,b}long.
        # A single filter is only ever one or the other, so include BOTH
        # families here -- the globs for the absent family simply match nothing.
        # (Omitting the 'long' detector silently dropped EVERY long-wave filter
        # at merge time: module='nrcb' expanded only to nrcb1..4 and the literal
        # 'nrcb', so f335m/f470n/f480m '*_nrcblong_*' files were never found ->
        # "No tables found" crash.  Regression from the SW per-detector fix.)
        modules = ([module] + [f'{module}{n}' for n in range(1, 5)]
                   + [f'{module}long'])
    else:
        modules = (module, )

    if method == 'crowdsource':
        flux_error_colname = 'dflux'
        column_names = ('flux', flux_error_colname, 'skycoord', 'qf', 'rchi2', 'fracflux', 'fwhm', 'fluxiso', 'spread_model')
        method_suffix = 'crowdsource'
        # flux_colname = 'flux_fit'
    elif method in ('dao', 'daophot', 'basic', 'daobasic', 'iterative', 'daoiterative'):
        flux_error_colname = 'flux_err'
        column_names = ('flux_fit', flux_error_colname, 'skycoord', 'qfit', 'cfit', 'flux_init', 'flags', 'local_bkg', 'iter_detected', 'group_size')
        # flux_colname = 'flux'
        method_suffix = 'daophot'
    else:
        raise ValueError(f"Method must be dao or crowdsource but was {method}")

    # Glob both the un-chunked per-frame catalogs and the chunked variants
    # produced by --n-seed-chunks > 1 in crowdsource_catalogs_long.py.  The
    # chunked filename inserts a ``_chunkXXofYY`` token between the iter
    # token and ``_{method_suffix}``; chunks for the same frame share the
    # rest of the path.
    raw_fns = []
    for module_ in modules:
        for progid in obs_filters[target]:
            for visitid in range(1, max_visitid + 1):
                for exposure in exposure_numbers:
                    base_pat = (
                        f"{basepath}/{filtername.upper()}/"
                        f"{filtername.lower()}_{module_}_visit{visitid:03d}_vgroup*_exp{exposure:05d}"
                        f"{desat}{bgsub}{fitpsf}{blur_}{group_}{iter_token}"
                    )
                    raw_fns.extend(glob.glob(
                        f"{base_pat}_{method_suffix}{suffix}.fits"))
                    raw_fns.extend(glob.glob(
                        f"{base_pat}_chunk*of*_{method_suffix}{suffix}.fits"))
    raw_fns = sorted(set(raw_fns))

    # Group chunks of the same frame by their chunk-stripped path, then
    # vstack them so combine_singleframe sees one table per physical frame.
    _chunk_re = re.compile(r'_chunk\d+of\d+')
    frame_groups = {}
    for fn in raw_fns:
        key = _chunk_re.sub('', fn)
        frame_groups.setdefault(key, []).append(fn)

    tblfns = sorted(frame_groups.keys())
    n_chunked = sum(1 for fns in frame_groups.values() if len(fns) > 1)
    print(f"Found {len(raw_fns)} files across {len(tblfns)} frames "
          f"({n_chunked} chunked) for "
          f"{filtername.lower()}_*_visit*_exp*{desat}{bgsub}{fitpsf}{blur_}:")
    for key in tblfns:
        chunks = frame_groups[key]
        if len(chunks) == 1:
            print(chunks[0])
        else:
            print(f"{key}  <- vstack of {len(chunks)} chunks")

    if len(tblfns) == 0:
        raise ValueError(f"No tables found matching {basepath}/{filtername.upper()}/{filtername.lower()}_{module}....{desat}{bgsub}{fitpsf}{blur_}_{method}{suffix}.fits")

    tables = []
    for key in tblfns:
        chunks = frame_groups[key]
        if len(chunks) == 1:
            tb = Table.read(chunks[0])
        else:
            sub_tables = [Table.read(fn) for fn in sorted(chunks)]
            tb = table.vstack(sub_tables, metadata_conflicts='silent')
            # Preserve the first chunk's FILENAME (absolute path to the
            # CRF input) so combine_singleframe can fits.open it to read
            # the SCI header (RAOFFSET/DEOFFSET).
            if 'FILENAME' in sub_tables[0].meta:
                tb.meta['FILENAME'] = sub_tables[0].meta['FILENAME']
        tables.append(tb)
    for tb, fn in zip(tables, tblfns):
        if 'exposure' not in tb.meta:
            tb.meta['exposure'] = fn.split("_exp")[-1][:5]
        if 'FILENAME' not in tb.meta:
            print('tb.meta:', tb.meta)
            raise ValueError(f"Table file {fn} is not correctly formatted; it is missing FILENAME metadata")

    # Note (2026-04-23): an earlier "dedup bug" diagnosis (realign=True +
    # min_offset=0.25") was based on analysing the FIRST 200k rows of the
    # F200W brick catalog which are exposure-adjacent and therefore
    # artificially correlated.  A random 500k-row sample found only
    # ~0.1% definite duplicates (ratio>0.95 AND nmatch_sum>24 pigeonhole);
    # median flux ratio of near-neighbour pairs was 0.38, indicating
    # most "near-neighbours" are genuinely distinct close-pair sources in
    # a dense field.  min_offset=0.25" would actively mis-merge ~36% of
    # real neighbours (separations 0.10-0.15").  Keeping the historical
    # defaults.
    # Spatial chunking parallelizes + bounds the memory of the otherwise serial
    # O(n_src) combine (the dense-field bottleneck).  n_spatial_chunks<=1 keeps
    # the exact serial path.  Auto-scale chunk count to the source volume when
    # the caller asks for chunking with workers but leaves n_spatial_chunks=1.
    _nchunks = int(n_spatial_chunks)
    if _nchunks <= 1 and int(merge_workers) > 1:
        _total_det = sum(len(t) for t in tables)
        if _total_det > 1_000_000:
            # ~one tile per 250k detections, capped at the worker count
            _nchunks = min(int(merge_workers),
                           max(2, int(np.ceil(_total_det / 250_000))))
    if _nchunks > 1:
        print(f"merge_individual_frames: spatial-chunked combine "
              f"({_nchunks} tiles, {merge_workers} workers)", flush=True)
        merged_exposure_table = combine_singleframe_chunked(
            tables, n_chunks=_nchunks, workers=int(merge_workers),
            offsets_table=offsets_table)
    else:
        merged_exposure_table = combine_singleframe(tables, offsets_table=offsets_table)

    outfn = f"{basepath}/catalogs/{filtername.lower()}_{module}_indivexp_merged{desat}{bgsub}{fitpsf}{blur_}{iter_token}_{method}{suffix}_allcols.fits"
    print(f"Writing {outfn} with length {len(merged_exposure_table)}")
    merged_exposure_table.write(outfn, overwrite=True)

    # make a table that is nearly equivalent to standard tables (with no 'x' or 'y' coordinate)
    minimal_version = {colname: merged_exposure_table[f'{colname}_avg']
                       for colname in column_names if f'{colname}_avg' in merged_exposure_table.colnames}
    for key in ('dra_avg', 'ddec_avg', 'std_ra', 'std_dec', 'nmatch', 'nmatch_good', f'{flux_error_colname}_prop'):
        if key in merged_exposure_table.colnames:
            minimal_version[key.split("_avg")[0]] = merged_exposure_table[key]

    minimal_table = Table(minimal_version)
    minimal_table.meta = merged_exposure_table.meta.copy()
    for ii, fn in enumerate(tblfns):
        minimal_table.meta[f'fn{ii}'] = os.path.basename(fn)

    # Ensure saturated stars are represented in indivexp merged products too.
    # Skippable (do_replace_saturated=False) for cutout runs, whose basepath
    # lacks the target-level resources replace_saturated needs (reduction/
    # fwhm_table.ecsv, full satstar catalogs).
    if do_replace_saturated:
        replace_saturated(minimal_table, filtername=filtername, target=target,
                          fwhm_basepath=fwhm_basepath, basepath=basepath)

    reject = np.isnan(minimal_table['skycoord'].ra) | np.isnan(minimal_table['skycoord'].dec)
    if np.any(reject):
        print(f"Rejected {reject.sum()} sources that had nan coordinates.")
        minimal_table = minimal_table[~reject]

    outfn = f"{basepath}/catalogs/{filtername.lower()}_{module}_indivexp_merged{desat}{bgsub}{fitpsf}{blur_}{iter_token}_{method}{suffix}.fits"
    print(f"Final table length is {len(minimal_table)}.  Writing {outfn}")
    minimal_table.write(outfn, overwrite=True)

    for colname in minimal_table.colnames:
        assert minimal_table[colname].ndim == 1

    return minimal_table


def merge_crowdsource(module='nrca', suffix="", desat=False, bgsub=False,
                      epsf=False, fitpsf=False, blur=False, target='brick',
                      min_qf=0.75,
                      indivexp=False,
                      iteration_label=None,
                      resbgsub=False,
                      basepath='/blue/adamginsburg/adamginsburg/jwst/brick/'):
    if epsf:
        raise NotImplementedError
    print()
    print(f'Starting merge crowdsource module: {module} suffix: {suffix} target: {target} iter: {iteration_label}', flush=True)
    imgfns = [x
              for obsid in obs_filters[target]
              for filn in obs_filters[target][obsid]
              for x in glob.glob(f"{basepath}/{filn.upper()}/pipeline/"
                                 f"jw0{obsid}-o{project_obsnum[target][obsid]}_t001_{_inst_token(filn)}*{filn.lower()}*{module}_i2d.fits")
              if f'{module}_' in x or f'{module}1_' in x
             ]

    desat = "_unsatstar" if desat else ""
    bgsub_flag = bool(bgsub)
    bgsub = _bgsub_token(bgsub, resbgsub)
    fitpsf = '_fitpsf' if fitpsf else ''
    blur_ = "_blur" if blur else ""
    if iteration_label in (None, ''):
        iter_token = ''
    elif str(iteration_label).startswith('_'):
        iter_token = str(iteration_label)
    else:
        iter_token = f'_{iteration_label}'

    jfilts = SvoFps.get_filter_list('JWST')
    jfilts.add_index('filterID')

    filternames = [filn for obsid in obs_filters[target] for filn in obs_filters[target][obsid]]
    print(f"Merging filters {filternames}", flush=True)
    if indivexp:
        catfns = [x
                  for filn in filternames
                  for x in glob.glob(f"{basepath}/catalogs/{filn.lower()}*{module}*indivexp_merged{desat}{bgsub}{fitpsf}{blur_}{iter_token}_crowdsource{suffix}.fits")
                  ]
        if len(catfns) == 0:
            filn = 'f405n'
            raise ValueError(f"{basepath}/catalogs/{filn.lower()}*{module}*indivexp_merged{desat}{bgsub}{fitpsf}{blur_}{iter_token}_crowdsource{suffix}.fits had no matches")
        if len(catfns) != len(imgfns):
            print("WARNING: Different length of imgfns & catfns!")
            print("imgfns:", imgfns)
            print("catfns:", catfns)
            print(dict(zip(imgfns, catfns)))
            raise ValueError(f"{basepath}/catalogs/FILTER*{module}*obs*indivexp_merged{desat}{bgsub}{fitpsf}{blur_}{iter_token}_crowdsource{suffix}.fits had different n(imgs) than n(cats)")
    else:
        catfns = [x
                  for filn in filternames
                  for x in glob.glob(f"{basepath}/{filn.upper()}/{filn.lower()}*{module}{desat}{bgsub}{fitpsf}{blur_}_crowdsource{suffix}.fits")
                  ]
        if target == 'brick' and len(catfns) != 10:
            raise ValueError(f"len(catfns) = {len(catfns)}.  catfns: {catfns}")
        elif target == 'cloudc' and len(catfns) != 6:
            raise ValueError(f"len(catfns) = {len(catfns)}.  catfns: {catfns}")

    for catfn in catfns:
        print(catfn, getmtime(catfn))

    # added a fq cut at read time to reduce memory usage during merge
    def read_cat(catfn, min_qf=min_qf):
        tbl = Table.read(catfn)
        if min_qf is not None:
            tbl = tbl[tbl['qf'] > min_qf]
        return tbl
    tbls = [read_cat(catfn) for catfn in tqdm(catfns, desc='Reading Tables')]

    for catfn, tbl in zip(catfns, tbls):
        tbl.meta['filename'] = catfn
        tbl.meta['filter'] = os.path.basename(catfn).split("_")[0]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # wcses = [wcs.WCS(fits.getheader(fn.replace("_crowdsource", "_crowdsource_skymodel"))) for fn in catfns]
        # imgs = [fits.getdata(fn, ext=('SCI', 1)) for fn in imgfns]
        wcses = [wcs.WCS(fits.getheader(fn, ext=('SCI', 1))) for fn in imgfns]

    for tbl, ww in zip(tbls, wcses):
        # Now done in the original catalog making step tbl['y'],tbl['x'] = tbl['x'],tbl['y']
        if 'skycoord' not in tbl.colnames:
            crds = ww.pixel_to_world(tbl['x'], tbl['y'])
            tbl.add_column(crds, name='skycoord')
        else:
            crds = tbl['skycoord']
        tbl.meta['pixelscale_deg2'] = ww.proj_plane_pixel_area()
        tbl.meta['pixelscale_arcsec'] = (ww.proj_plane_pixel_area()**0.5).to(u.arcsec)
        print(f'Calculating Flux [Jy].  fwhm={tbl["fwhm"].mean()}, pixscale={tbl.meta["pixelscale_arcsec"]}')
        # The 'flux' is the sum of pixels whose values are each in MJy/sr.
        # To get to the correct flux, we need to multiply by the pixel area in steradians to get to megaJanskys, which can be summed
        # That's it.  There's no need to account for the FWHM.  We only needed that if tbl['flux'] was the _peak_, but it's not.
        #flux_jy = (tbl['flux'] * u.MJy/u.sr * (2*np.pi / (8*np.log(2))) * tbl['fwhm']**2 * tbl.meta['pixelscale_deg2']).to(u.Jy)
        #eflux_jy = (tbl['dflux'] * u.MJy/u.sr * (2*np.pi / (8*np.log(2))) * tbl['fwhm']**2 * tbl.meta['pixelscale_deg2']).to(u.Jy)
        flux_jy = (tbl['flux'] * u.MJy/u.sr * tbl.meta['pixelscale_deg2']).to(u.Jy)
        eflux_jy = (tbl['dflux'] * u.MJy/u.sr * tbl.meta['pixelscale_deg2']).to(u.Jy)
        with np.errstate(all='ignore'):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                filtername = tbl.meta["filter"]
                zeropoint = u.Quantity(jfilts.loc[_svo_filter_id(filtername)]['ZeroPoint'], u.Jy)
                print(f"Zeropoint for {filtername} is {zeropoint}.  Max flux is {flux_jy.max()}")
                # True AB (3631 Jy zeropoint) for mag_ab; Vega (SVO zeropoint) as a separate column.
                # Previously mag_ab held the Vega magnitude, mislabeled (off by AB-Vega ~1.8 mag);
                # this matched the daophot/satstar paths' convention bug for crowdsource only.
                abmag = (-2.5 * np.log10(flux_jy / u.Jy) + ABMAG_OFFSET) * u.mag
                vegamag = -2.5 * np.log10(flux_jy / zeropoint) * u.mag
                abmag_err = 2.5 / np.log(10) * np.abs(eflux_jy / flux_jy) * u.mag
                tbl.add_column(Column(flux_jy, name='flux_jy', unit=u.Jy))
                tbl.add_column(Column(eflux_jy, name='eflux_jy', unit=u.Jy))
                tbl.add_column(Column(abmag, name='mag_ab', unit=u.mag))
                tbl.add_column(Column(vegamag, name='mag_vega', unit=u.mag))
                tbl.add_column(Column(abmag_err, name='emag_ab', unit=u.mag))
                print(f"Max flux={tbl['flux_jy'].max()}, min mag={np.nanmin(tbl['mag_ab'])}, median={np.nanmedian(tbl['mag_ab'])}")
        if hasattr(tbl['mag_ab'], 'mask'):
            print(f'ab mag tbl col has mask sum = {tbl["mag_ab"].mask.sum()} masked values')
        if hasattr(abmag, 'mask'):
            print(f'ab mag has mask sum = {abmag.mask.sum()} masked values')
        if hasattr(tbl['flux'], 'mask'):
            print(f'flux has mask sum = {tbl["flux"].mask.sum()} masked values')

    for tbl in tbls:
        try:
            sanity_check_individual_table(tbl)
        except Exception as ex:
            print(ex)
            print(tbl.meta)
            raise ex

    merge_catalogs(tbls,
                   catalog_type=f'crowdsource{suffix}{"_desat" if desat else ""}{bgsub}',
                   module=module, bgsub=bgsub_flag, resbgsub=resbgsub, desat=desat, epsf=epsf, target=target,
                   blur=blur,
                   indivexp=indivexp,
                   iteration_label=iteration_label,
                   qfcut=0.9,
                   fracfluxcut=0.75,
                   basepath=basepath)


def merge_daophot(module='nrca', detector='', daophot_type='basic', desat=False, bgsub=False, epsf=False, blur=False, target='brick',
                  indivexp=False,
                  ref_filter=None,
                  iteration_label=None,
                  resbgsub=False,
                  vetted=False,
                  filternames_override=None,
                  basepath='/blue/adamginsburg/adamginsburg/jwst/brick/'):
    """Cross-filter merge of per-filter daophot catalogs.

    ``vetted`` (manual path): read the ``_vetted`` per-filter merged catalogs
    (the quality-cut science catalogs) instead of the raw merged ones.
    ``filternames_override``: restrict to this explicit filter list instead of
    the full ``obs_filters[target]`` set -- REQUIRED for the manual NIRCam path,
    whose target (e.g. sickle) registers MIRI filters in the same obs_filters
    entry that must NOT enter the NIRCam cross-band merge.
    """

    desat = "_unsatstar" if desat else ""
    bgsub_flag = bool(bgsub)
    bgsub = _bgsub_token(bgsub, resbgsub)
    epsf_ = "_epsf" if epsf else ""
    blur_ = "_blur" if blur else ""
    if iteration_label in (None, ''):
        iter_token = ''
    elif str(iteration_label).startswith('_'):
        iter_token = str(iteration_label)
    else:
        iter_token = f'_{iteration_label}'

    jfilts = SvoFps.get_filter_list('JWST')
    jfilts.add_index('filterID')

    if filternames_override is not None:
        filternames = [f.lower() for f in filternames_override]
    else:
        filternames = [filn for obsid in obs_filters[target] for filn in obs_filters[target][obsid]]
    print(f"Merging daophot {daophot_type}, {detector}, {module}, {desat}, {bgsub}, {epsf_}, {blur_}. filters {filternames}")

    # Use _project_for_target_filter rather than the global filter_to_project
    # so a filter shared across targets (e.g. f187n in both brick/2221 and
    # sgrb2/5365) resolves to the project matching this run's ``target``.
    # Map daophot_type -> per-filter merge filename token written by
    # merge_individual_frames(): basic uses ``dao``, iterative uses
    # ``daoiterative`` (matches the suffix-vs-method dict in main()).  The
    # pattern was once hardcoded to ``_dao_{daophot_type}`` which only matched
    # basic; the daoiterative filename is ``_daoiterative_iterative.fits``.
    method_name = 'dao' if daophot_type == 'basic' else 'daoiterative'

    vetted_tok = '_vetted' if vetted else ''

    # Resolve each filter's catalog AND its i2d (for WCS) TOGETHER, keeping
    # filternames / catfns / imgfns / tbls / wcses strictly aligned 1:1.
    # Previously these were INDEPENDENT flattened globs: when a configured filter
    # had no catalog (e.g. a partial-filter run), the catfns glob silently
    # dropped it, then ``zip(catfns, tbls, filternames)`` paired the survivors
    # with the WRONG filter names -- a silent filter mislabel that propagated
    # wrong-band photometry.  imgfns could misalign with catfns the same way.
    # Now we iterate filters once and drop any with no catalog (with a warning),
    # so the remaining parallel lists are guaranteed consistent.
    present_filters, catfns, imgfns = [], [], []
    for filn in filternames:
        if indivexp:
            _cat_matches = sorted(glob.glob(
                f"{basepath}/catalogs/{filn.lower()}*{module}*indivexp_merged"
                f"{desat}{bgsub}{blur_}{iter_token}_{method_name}_{daophot_type}{vetted_tok}.fits"))
            if not _cat_matches:
                print(f"WARNING: no indivexp daophot {daophot_type} catalog for "
                      f"filter {filn} (module={module}, vetted={vetted}); dropping "
                      f"it from the cross-filter merge", flush=True)
                continue
            if len(_cat_matches) > 1:
                print(f"WARNING: multiple indivexp catalogs for {filn}; using "
                      f"{_cat_matches[0]}", flush=True)
            catfn = _cat_matches[0]
        else:
            catfn = (f"{basepath}/{filn.upper()}/{filn.lower()}_{module}{detector}"
                     f"{desat}{bgsub}{epsf_}{blur_}_daophot_{daophot_type}.fits")
            if not os.path.exists(catfn):
                print(f"WARNING: no daophot {daophot_type} catalog {catfn} for "
                      f"filter {filn}; dropping it from the cross-filter merge",
                      flush=True)
                continue
        _proj = _project_for_target_filter(target, filn)
        _img_matches = [x for x in glob.glob(
            f"{basepath}/{filn.upper()}/pipeline/jw0{_proj}-o{project_obsnum[target][_proj]}"
            f"_t001_{_inst_token(filn)}*{filn.lower()}*{module}_i2d.fits")
            if f'{module}_' in x or f'{module}1_' in x]
        if not _img_matches:
            print(f"WARNING: no {module} i2d for filter {filn}; will fall back to "
                  f"the catalog's own skycoord/pixelscale for WCS", flush=True)
        present_filters.append(filn)
        catfns.append(catfn)
        imgfns.append(_img_matches[0] if _img_matches else None)

    filternames = present_filters
    if len(catfns) == 0:
        raise ValueError(
            f"No daophot {daophot_type} catalogs found across filters for "
            f"module={module} in {basepath} (indivexp={indivexp}, "
            f"iter_token={iter_token!r}, desat={desat!r}, bgsub={bgsub!r}, "
            f"vetted={vetted})")

    tbls = [Table.read(catfn) for catfn in catfns]

    for catfn, tbl, filtername in zip(catfns, tbls, filternames):
        tbl.meta['filename'] = catfn
        tbl.meta['filter'] = filtername

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        wcses = [wcs.WCS(fits.getheader(fn, ext=('SCI', 1))) if fn is not None else None
                 for fn in imgfns]

    fwhm_tbl = Table.read(f'{basepath}/reduction/fwhm_table.ecsv')

    for ii, tbl in enumerate(tbls):
        ww = wcses[ii] if ii < len(wcses) else None

        if ww is not None and 'x_fit' in tbl.colnames:
            crds = ww.pixel_to_world(tbl['x_fit'], tbl['y_fit'])
        elif ww is not None and 'x_0' in tbl.colnames:
            crds = ww.pixel_to_world(tbl['x_0'], tbl['y_0'])
        else:
            crds = tbl['skycoord']
        if 'skycoord' not in tbl.colnames:
            tbl.add_column(crds, name='skycoord')

        if ww is not None:
            pixelscale_deg2 = ww.proj_plane_pixel_area()
            pixelscale_arcsec = (ww.proj_plane_pixel_area()**0.5).to(u.arcsec)
        elif 'pixelscale_deg2' in tbl.meta:
            pixelscale_deg2 = tbl.meta['pixelscale_deg2']
            pixelscale_arcsec = (u.Quantity(pixelscale_deg2, u.deg**2)**0.5).to(u.arcsec)
        elif 'PIXSCALE' in tbl.meta:
            pixelscale_arcsec = u.Quantity(float(tbl.meta['PIXSCALE']), u.arcsec)
            pixelscale_deg2 = pixelscale_arcsec.to(u.deg)**2
        else:
            raise ValueError(f"Could not determine pixel scale for {tbl.meta.get('filter', 'unknown')} table {tbl.meta.get('filename', '')}")

        tbl.meta['pixelscale_deg2'] = pixelscale_deg2
        tbl.meta['pixelscale_arcsec'] = pixelscale_arcsec
        if 'flux_fit' in tbl.colnames:
            flux = tbl['flux_fit']
        elif 'flux_0' in tbl.colnames:
            flux = tbl['flux_0']
        elif 'flux' in tbl.colnames:
            flux = tbl['flux']
        else:
            raise KeyError(f"No supported flux column found in table columns={tbl.colnames}")
        filtername = tbl.meta['filter']

        row = fwhm_tbl[fwhm_tbl['Filter'] == filtername.upper()]
        fwhm = u.Quantity(float(row['PSF FWHM (arcsec)'][0]), u.arcsec)
        fwhm_pix = float(row['PSF FWHM (pixel)'][0])
        tbl.meta['fwhm_arcsec'] = fwhm
        tbl.meta['fwhm_pix'] = fwhm_pix

        with np.errstate(all='ignore'):
            flux_jy = (flux * u.MJy/u.sr * tbl.meta['pixelscale_deg2']).to(u.Jy)
            zeropoint = u.Quantity(jfilts.loc[_svo_filter_id(filtername)]['ZeroPoint'], u.Jy)
            vegamag = -2.5 * np.log10(flux_jy / zeropoint) * u.mag
            abmag = (-2.5 * np.log10(flux_jy / u.Jy) + ABMAG_OFFSET) * u.mag
            try:
                eflux_jy = (tbl['flux_unc'] * u.MJy/u.sr * tbl.meta['pixelscale_deg2']).to(u.Jy)
            except KeyError:
                eflux_jy = (tbl['flux_err'] * u.MJy/u.sr * tbl.meta['pixelscale_deg2']).to(u.Jy)
            abmag_err = 2.5 / np.log(10) * eflux_jy / flux_jy * u.mag
        tbl.add_column(Column(flux_jy, name='flux_jy', unit=u.Jy))
        tbl.add_column(Column(abmag, name='mag_ab', unit=u.mag))
        tbl.add_column(Column(vegamag, name='mag_vega', unit=u.mag))
        tbl.add_column(Column(eflux_jy, name='eflux_jy', unit=u.Jy))
        tbl.add_column(Column(abmag_err, name='emag_ab', unit=u.mag))

    for tbl in tbls:
        try:
            sanity_check_individual_table(tbl)
        except Exception as ex:
            print(ex)
            print(tbl.meta)
            raise ex

    _merge_kwargs = dict(catalog_type=daophot_type, module=module, bgsub=bgsub_flag,
                         resbgsub=resbgsub, desat=desat,
                         epsf=epsf, target=target, blur=blur, indivexp=indivexp,
                         iteration_label=iteration_label,
                         basepath=basepath)
    if ref_filter is not None:
        _merge_kwargs['ref_filter'] = ref_filter
    merge_catalogs(tbls, **_merge_kwargs)


def _project_for_target_filter(target, filtername):
    """Return the project_id under which ``target`` observes ``filtername``.

    ``filter_to_project`` is a global dict that collapses filter->project
    across all targets, so for a filter observed by multiple targets (e.g.
    f187n appears under both brick/2221 and sgrb2/5365) it picks whichever
    target was iterated last and breaks lookups for the other targets.
    This helper resolves the correct project for the target in hand.
    """
    target_filters = obs_filters[target]
    filt_l = filtername.lower()
    for proj, filts in target_filters.items():
        if filt_l in filts:
            return proj
    raise KeyError(
        f'filter {filtername!r} not observed by target {target!r}; '
        f'known target/filter map: {target_filters}'
    )


def load_satstar_catalog(filtername, target='brick',
                         basepath='/blue/adamginsburg/adamginsburg/jwst/brick/'):
    proj = _project_for_target_filter(target, filtername)
    # Targets that span multiple observations within one proposal (e.g.
    # gc2211 has obs 023/028/046/049/050) don't have a single primary
    # i2d satstar; fall back to globbing the per-exposure satstar
    # catalogs in the pipeline directory.
    # The single-file "primary" satstar product only exists for NIRCam
    # (the *_nircam_clear-<filt>-merged_i2d naming); MIRI runs always use
    # the per-exposure fallback below.
    if (_inst_token(filtername) == 'nircam'
            and target in project_obsnum and proj in project_obsnum[target]):
        primary = (f'{basepath}/{filtername.upper()}/pipeline/'
                   f'jw0{proj}-o{project_obsnum[target][proj]}'
                   f'_t001_nircam_clear-{filtername}-merged_i2d_satstar_catalog.fits')
        # project_obsnum may hold a glob wildcard for multi-obs targets
        # (sickle/cloudef/gc2211), so resolve via glob rather than exists().
        primary_matches = sorted(glob.glob(primary))
        if len(primary_matches) == 1:
            print(f"Using saturated star catalog {primary_matches[0]}")
            return Table.read(primary_matches[0])

    fallback = sorted(glob.glob(f'{basepath}/{filtername.upper()}/pipeline/*satstar_catalog.fits'))
    if len(fallback) == 0:
        print(f"No saturated star catalog files found for {filtername} in {basepath}/{filtername.upper()}/pipeline")
        return None

    # Consolidated per-filter satstar cache.  Building the deduped catalog from
    # the ~1000-1400 per-exposure satstar FITS (read + vstack + dedup) is the
    # dominant cost of every merge: replace_saturated() / flag_near_saturated()
    # call this once each, in EVERY phase's single-filter merge_individual_frames
    # AND in the cross-band merge -- so the same ~1400-file re-read happens dozens
    # of times per run over the shared filesystem (the cross-band merge alone took
    # ~5 h).  Cache the deduped result in catalogs/ and reuse it while it is at
    # least as new as the newest per-exposure satstar catalog; rebuild (and
    # rewrite) whenever any per-exposure catalog is newer.  The cache lives in
    # catalogs/ (NOT the pipeline dir) so the *satstar_catalog.fits glob above
    # never re-globs it.  Result is identical to rebuilding every time.
    cache = (f'{basepath}/catalogs/'
             f'{filtername.lower()}_consolidated_satstar_catalog.fits')
    try:
        newest_src = max(os.path.getmtime(fn) for fn in fallback)
        if os.path.exists(cache) and os.path.getmtime(cache) >= newest_src:
            print(f"Using consolidated satstar catalog {cache} "
                  f"(cache fresh vs {len(fallback)} per-exposure catalogs)")
            return Table.read(cache)
    except OSError:
        pass  # stat/read failure -> rebuild from per-exposure catalogs below

    print(f"Building consolidated satstar catalog for {filtername} from "
          f"{len(fallback)} per-exposure catalogs")
    sat_tables = [Table.read(fn) for fn in fallback]
    combined = table.vstack(sat_tables, metadata_conflicts='silent')
    # The fallback globs the PER-EXPOSURE satstar catalogs, so the same physical
    # saturated star appears once per frame it was fit in.  Without dedup,
    # replace_saturated() add_row()s each copy -> the merged catalog gets N
    # duplicate rows at one position (sickle cutouts showed a star 3-8x), and
    # the merged-cat residual subtracts it N times.  Collapse to one row per
    # physical star (keep the brightest as representative).
    deduped = _dedup_satstar_catalog(combined)
    try:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        # write to a temp sibling + atomic rename so a concurrent reader never
        # sees a half-written cache
        tmp = f'{cache}.tmp{os.getpid()}'
        deduped.write(tmp, overwrite=True, format='fits')
        os.replace(tmp, cache)
        print(f"Wrote consolidated satstar catalog {cache} (n={len(deduped)})")
    except OSError as ex:
        print(f"Could not write consolidated satstar cache {cache}: {ex}")
    return deduped


def _dedup_satstar_catalog(tbl, radius=0.15 * u.arcsec):
    """Collapse repeated per-frame satstar fits of the same physical star into
    one row (the brightest), so downstream merging doesn't duplicate them.

    Greedy brightest-first spatial dedup: process stars in descending flux and
    keep one unless it falls within ``radius`` of an already-kept (brighter)
    star.  Vectorized with ``search_around_sky`` (one O(N log N) KD-tree query
    for all within-radius pairs) instead of rebuilding a SkyCoord of the kept
    set inside the loop -- the old O(N^2) form took ~hours on the ~12k
    per-exposure sickle F210M satstar pile-up.  Output is identical (same kept
    set, original row order).
    """
    if tbl is None or len(tbl) <= 1 or 'skycoord_fit' not in tbl.colnames:
        return tbl
    from astropy.coordinates import search_around_sky
    coords = tbl['skycoord_fit']
    finite = np.isfinite(coords.ra.deg) & np.isfinite(coords.dec.deg)
    flux = np.asarray(tbl['flux_fit'], dtype=float) if 'flux_fit' in tbl.colnames \
        else np.zeros(len(tbl))
    fin_idx = np.where(finite)[0]            # original-row indices of finite rows
    if len(fin_idx) == 0:
        return tbl[[]]
    sc = coords[fin_idx]
    fl = flux[fin_idx]
    # all within-radius neighbor pairs among the finite sources (both directions,
    # plus self-pairs, which we drop)
    i1, i2, _, _ = search_around_sky(sc, sc, radius)
    nbrs = [[] for _ in range(len(fin_idx))]
    for a, b in zip(i1, i2):
        if a != b:
            nbrs[a].append(b)
    suppressed = np.zeros(len(fin_idx), dtype=bool)
    kept_local = []
    for i in np.argsort(-fl):                # brightest first
        if suppressed[i]:
            continue
        kept_local.append(i)
        for j in nbrs[i]:                    # drop everything within radius of a kept star
            suppressed[j] = True
    kept = fin_idx[kept_local]
    n_drop = int(finite.sum()) - len(kept)
    if n_drop:
        print(f"  satstar dedup: {int(finite.sum())} -> {len(kept)} unique "
              f"(dropped {n_drop} per-frame duplicates within {radius})")
    return tbl[np.sort(kept)]


def flag_near_saturated(cat, filtername, radius=None, target='brick',
                        basepath='/blue/adamginsburg/adamginsburg/jwst/brick/'):
    print(f"Flagging near saturated stars for filter {filtername}")
    satstar_cat = load_satstar_catalog(filtername, target=target, basepath=basepath)
    if satstar_cat is None:
        print(f"No saturated star catalog found for {filtername}")
        cat.add_column(np.zeros(len(cat), dtype='bool'), name=f'near_saturated_{filtername}')
        return
    satstar_coords = satstar_cat['skycoord_fit']

    cat_coords = cat['skycoord']

    if radius is None:
        # 0.55" flagging radius for every NIRCam filter in the project map;
        # keep the filter list in sync with obs_filters so shared code paths
        # (sickle/sgrb2/etc.) don't KeyError on filters that aren't listed.
        radius = {# short-wave (< ~2.5 um)
                  'f115w': 0.55*u.arcsec,
                  'f150w': 0.55*u.arcsec,
                  'f162m': 0.55*u.arcsec,
                  'f182m': 0.55*u.arcsec,
                  'f187n': 0.55*u.arcsec,
                  'f200w': 0.55*u.arcsec,
                  'f210m': 0.55*u.arcsec,
                  'f212n': 0.55*u.arcsec,
                  'f277w': 0.55*u.arcsec,
                  # long-wave (> ~2.5 um)
                  'f300m': 0.55*u.arcsec,
                  'f323n': 0.55*u.arcsec,
                  'f335m': 0.55*u.arcsec,
                  'f356w': 0.55*u.arcsec,
                  'f360m': 0.55*u.arcsec,
                  'f405n': 0.55*u.arcsec,
                  'f410m': 0.55*u.arcsec,
                  'f444w': 0.55*u.arcsec,
                  'f466n': 0.55*u.arcsec,
                  'f470n': 0.55*u.arcsec,
                  'f480m': 0.55*u.arcsec,
                  # MIRI: scale the 0.55" NIRCam value by the PSF FWHM ratio
                  # relative to F480M (0.16" FWHM) -> ~3.4x FWHM
                  'f560w': 0.7*u.arcsec,
                  'f770w': 0.9*u.arcsec,
                  'f1000w': 1.1*u.arcsec,
                  'f1130w': 1.3*u.arcsec,
                  'f1280w': 1.45*u.arcsec,
                  'f1500w': 1.7*u.arcsec,
                  'f1800w': 2.0*u.arcsec,
                  'f2100w': 2.3*u.arcsec,
                  'f2550w': 2.75*u.arcsec,
                  }[filtername]

    satfinite = np.isfinite(satstar_coords.ra.deg) & np.isfinite(satstar_coords.dec.deg)
    catfinite = np.isfinite(cat_coords.ra.deg) & np.isfinite(cat_coords.dec.deg)

    satstar_cat = satstar_cat[satfinite]
    satstar_coords = satstar_coords[satfinite]

    valid_cat_inds = np.where(catfinite)[0]
    if len(valid_cat_inds) > 0 and len(satstar_cat) > 0:
        idx_cat_sub, idx_sat, sep, _ = satstar_coords.search_around_sky(cat_coords[catfinite], radius)
        idx_cat = valid_cat_inds[idx_cat_sub]
    else:
        idx_cat = np.array([], dtype=int)
        idx_sat = np.array([], dtype=int)

    near_sat = np.zeros(len(cat), dtype='bool')
    near_sat[idx_cat] = True

    cat.add_column(near_sat, name=f'near_saturated_{filtername}')


def replace_saturated(cat, filtername, radius=None, target='brick',
                      fwhm_basepath=None,
                      basepath='/blue/adamginsburg/adamginsburg/jwst/brick/'):
    # ``basepath`` locates the satstar catalogs (the cutout's own, for cutout
    # runs); ``fwhm_basepath`` (default = basepath) locates the target-level
    # reduction/fwhm_table.ecsv, which the cutout tree doesn't contain.
    satstar_cat = load_satstar_catalog(filtername, target=target, basepath=basepath)
    if satstar_cat is None:
        print(f"No saturated star catalog found for {filtername}; skipping replacement")
        if 'replaced_saturated' not in cat.colnames:
            cat.add_column(np.zeros(len(cat), dtype='bool'), name='replaced_saturated')
        else:
            print(f"Found existing 'replaced_saturated' column in cat; leaving it unchanged.  "
                  f"max={np.nanmax(cat['replaced_saturated'])}, sum={np.nansum(cat['replaced_saturated'])}")
        if 'is_saturated' not in cat.colnames:
            cat.add_column(np.zeros(len(cat), dtype='bool'), name='is_saturated')
        else:
            print(f"Found existing 'is_saturated' column in cat; leaving it unchanged.  "
                  f"max={np.nanmax(cat['is_saturated'])}, sum={np.nansum(cat['is_saturated'])}")
        if 'flux_fit' in cat.colnames:
            cat.rename_column('flux_fit', 'flux')
        return

    print(f"Loaded saturated star catalog for {filtername} with {len(satstar_cat)} rows")
    satstar_coords = satstar_cat['skycoord_fit']

    cat_coords = cat['skycoord']

    jfilts = SvoFps.get_filter_list('JWST')
    jfilts.add_index('filterID')

    if radius is None:
        radius = {# short-wave (< ~2.5 um)
                  'f115w': 0.05*u.arcsec,
                  'f150w': 0.05*u.arcsec,
                  'f162m': 0.05*u.arcsec,
                  'f182m': 0.05*u.arcsec,
                  'f187n': 0.05*u.arcsec,
                  'f200w': 0.05*u.arcsec,
                  'f210m': 0.05*u.arcsec,
                  'f212n': 0.05*u.arcsec,
                  'f140m': 0.05*u.arcsec,
                  'f164n': 0.05*u.arcsec,
                  'f250m': 0.08*u.arcsec,
                  'f277w': 0.1*u.arcsec,
                  # long-wave (> ~2.5 um)
                  'f300m': 0.1*u.arcsec,
                  'f323n': 0.1*u.arcsec,
                  'f335m': 0.1*u.arcsec,
                  'f356w': 0.1*u.arcsec,
                  'f360m': 0.1*u.arcsec,
                  'f405n': 0.1*u.arcsec,
                  'f410m': 0.1*u.arcsec,
                  'f444w': 0.1*u.arcsec,
                  'f466n': 0.1*u.arcsec,
                  'f470n': 0.1*u.arcsec,
                  'f480m': 0.1*u.arcsec,
                  # MIRI: 0.11"/pix vs NIRCam LW 0.063"/pix; scale the 0.1"
                  # LW match radius accordingly (positions are coarser)
                  'f560w': 0.2*u.arcsec,
                  'f770w': 0.2*u.arcsec,
                  'f1000w': 0.2*u.arcsec,
                  'f1130w': 0.25*u.arcsec,
                  'f1280w': 0.25*u.arcsec,
                  'f1500w': 0.3*u.arcsec,
                  'f1800w': 0.35*u.arcsec,
                  'f2100w': 0.4*u.arcsec,
                  'f2550w': 0.5*u.arcsec,
                  }[filtername]

    fwhm_tbl = Table.read(f'{fwhm_basepath or basepath}/reduction/fwhm_table.ecsv')
    fwhm = u.Quantity(fwhm_tbl[fwhm_tbl['Filter'] == filtername.upper()]['PSF FWHM (arcsec)'], u.arcsec)

    filtername_meta = cat.meta.get('filter', filtername)
    zeropoint = u.Quantity(jfilts.loc[_svo_filter_id(filtername_meta)]['ZeroPoint'], u.Jy)

    pixelscale_deg2 = cat.meta.get('pixelscale_deg2', None)
    if pixelscale_deg2 is not None:
        flux_jy = (satstar_cat['flux_fit'] * u.MJy/u.sr * pixelscale_deg2).to(u.Jy)
        try:
            eflux_jy = (satstar_cat['flux_err'] * u.MJy/u.sr * pixelscale_deg2).to(u.Jy)
        except KeyError:
            eflux_jy = (satstar_cat['flux_unc'] * u.MJy/u.sr * pixelscale_deg2).to(u.Jy)
        abmag = (-2.5*np.log10(flux_jy / u.Jy) + ABMAG_OFFSET) * u.mag
        abvega = -2.5*np.log10(flux_jy / zeropoint) * u.mag
        abmag_err = 2.5 / np.log(10) * np.abs(eflux_jy / flux_jy) * u.mag
    else:
        print(f"Catalog metadata lacks pixelscale_deg2 for {filtername}; skipping satstar mag conversion")
        abmag = np.full(len(satstar_cat), np.nan) * u.mag
        abvega = np.full(len(satstar_cat), np.nan) * u.mag
        abmag_err = np.full(len(satstar_cat), np.nan) * u.mag
    satstar_cat['mag_ab'] = abmag
    satstar_cat['mag_vega'] = abvega
    satstar_cat['emag_ab'] = abmag_err

    satfinite = np.isfinite(satstar_coords.ra.deg) & np.isfinite(satstar_coords.dec.deg)
    catfinite = np.isfinite(cat_coords.ra.deg) & np.isfinite(cat_coords.dec.deg)

    satstar_cat = satstar_cat[satfinite]
    satstar_coords = satstar_coords[satfinite]

    valid_cat_inds = np.where(catfinite)[0]
    if len(valid_cat_inds) > 0 and len(satstar_cat) > 0:
        idx_cat_sub, idx_sat, sep, _ = satstar_coords.search_around_sky(cat_coords[catfinite], radius)
        idx_cat = valid_cat_inds[idx_cat_sub]
    else:
        idx_cat = np.array([], dtype=int)
        idx_sat = np.array([], dtype=int)

    replaced_sat = np.zeros(len(cat), dtype='bool')
    replaced_sat[idx_cat] = True

    if 'flux_err' in satstar_cat.colnames:
        flux_err_colname = 'flux_err'
        xerr_colname = 'x_err'
        yerr_colname = 'y_err'
    elif 'flux_unc' in satstar_cat.colnames:
        flux_err_colname = 'flux_unc'
        xerr_colname = 'x_0_unc'
        yerr_colname = 'y_0_unc'
    else:
        print(satstar_cat.colnames)
        raise KeyError("Missing flux error column")

    if 'flux' in cat.colnames:
        if 'dflux' in cat.colnames:
            cat_fluxerr_col = 'dflux'
        elif 'flux_err' in cat.colnames:
            cat_fluxerr_col = 'flux_err'
        elif 'eflux' in cat.colnames:
            cat_fluxerr_col = 'eflux'
        else:
            cat_fluxerr_col = None

        cat['flux'][idx_cat] = satstar_cat['flux_fit'][idx_sat]
        if cat_fluxerr_col is not None:
            cat[cat_fluxerr_col][idx_cat] = satstar_cat[flux_err_colname][idx_sat]
        cat['skycoord'][idx_cat] = satstar_cat['skycoord_fit'][idx_sat]
        if 'x' in cat.colnames:
            # the merged, individual field catalogs don't have these
            cat['x'][idx_cat] = satstar_cat['x_fit'][idx_sat]
            cat['y'][idx_cat] = satstar_cat['y_fit'][idx_sat]
            cat['dx'][idx_cat] = satstar_cat[xerr_colname][idx_sat]
            cat['dy'][idx_cat] = satstar_cat[yerr_colname][idx_sat]

        if 'mag_ab' in cat.colnames:
            cat['mag_ab'][idx_cat] = abmag[idx_sat]
        if 'mag_vega' in cat.colnames:
            cat['mag_vega'][idx_cat] = abvega[idx_sat]
        if 'emag_ab' in cat.colnames:
            cat['emag_ab'][idx_cat] = abmag_err[idx_sat]

        # ID the stars that are saturated-only (not INCluded in the orig cat)
        satstar_not_inc = np.ones(len(satstar_cat), dtype='bool')
        satstar_not_inc[idx_sat] = False
        satstar_toadd = satstar_cat[satstar_not_inc]

        satstar_toadd.rename_column('flux_fit', 'flux')
        if cat_fluxerr_col is not None:
            satstar_toadd.rename_column(flux_err_colname, cat_fluxerr_col)
        satstar_toadd.rename_column('skycoord_fit', 'skycoord')
        if 'x' in cat.colnames:
            satstar_toadd.rename_column('x_fit', 'x')
            satstar_toadd.rename_column('y_fit', 'y')
            satstar_toadd.rename_column(xerr_colname, 'dx')
            satstar_toadd.rename_column(yerr_colname, 'dy')

        for colname in cat.colnames:
            if colname not in satstar_toadd.colnames:
                satstar_toadd.add_column(np.ones(len(satstar_toadd)) * np.nan, name=colname)
        for colname in satstar_toadd.colnames:
            if colname not in cat.colnames:
                satstar_toadd.remove_column(colname)

        for row in satstar_toadd:
            cat.add_row(dict(row))

    elif 'flux_fit' in cat.colnames:
        # DAOPHOT
        cat['flux_fit'][idx_cat] = satstar_cat['flux_fit'][idx_sat]
        cat['flux_err'][idx_cat] = satstar_cat[flux_err_colname][idx_sat]
        cat['skycoord'][idx_cat] = satstar_cat['skycoord_fit'][idx_sat]
        if 'x_fit' in satstar_cat.colnames and 'x_fit' in cat.colnames:
            cat['x_fit'][idx_cat] = satstar_cat['x_fit'][idx_sat]
            cat['y_fit'][idx_cat] = satstar_cat['y_fit'][idx_sat]
            cat['x_err'][idx_cat] = satstar_cat[xerr_colname][idx_sat]
            cat['y_err'][idx_cat] = satstar_cat[yerr_colname][idx_sat]

        if 'mag_ab' in cat.colnames:
            cat['mag_ab'][idx_cat] = abmag[idx_sat]
        if 'mag_vega' in cat.colnames:
            cat['mag_vega'][idx_cat] = abvega[idx_sat]
        if 'emag_ab' in cat.colnames:
            cat['emag_ab'][idx_cat] = abmag_err[idx_sat]

        # ID the stars that are saturated-only (not INCluded in the orig cat)
        satstar_not_inc = np.ones(len(satstar_cat), dtype='bool')
        satstar_not_inc[idx_sat] = False
        satstar_toadd = satstar_cat[satstar_not_inc]

        satstar_toadd.rename_column('skycoord_fit', 'skycoord')
        satstar_toadd['skycoord_centroid'] = satstar_toadd['skycoord']

        for colname in cat.colnames:
            if colname not in satstar_toadd.colnames:
                satstar_toadd.add_column(np.ones(len(satstar_toadd))*np.nan, name=colname)
        for colname in satstar_toadd.colnames:
            if colname not in cat.colnames:
                satstar_toadd.remove_column(colname)

        #print("cat colnames: ",cat.colnames)
        #print("satstar toadd_colnames: ",satstar_toadd.colnames)
        for row in satstar_toadd:
            cat.add_row(dict(row))

    # we've added on more rows that are all 'replaced_sat'
    replaced_sat_ = np.ones(len(cat), dtype='bool')
    replaced_sat_[:len(replaced_sat)] = replaced_sat

    print(f"Replacing {len(idx_cat)} stars that are saturated of {len(cat)} "
          f"in filter {filtername}.  "
          f"{satstar_not_inc.sum()} are newly added.  The total replaced stars={replaced_sat_.sum()}")

    if 'is_saturated' in cat.colnames:
        cat.remove_column('is_saturated')
    cat.add_column(replaced_sat_.astype(bool), name='is_saturated')

    if 'replaced_saturated' in cat.colnames:
        cat.remove_column('replaced_saturated')
    cat.add_column(replaced_sat_, name='replaced_saturated')
    if 'flux_fit' in cat.colnames:
        cat.rename_column('flux_fit', 'flux')
    else:
        print(f"Catalog did not have flux_fit.  colnames={cat.colnames}.  (this is expected for crowdsource)")
    # DEBUG print(f"DEBUG: cat['replaced_saturated'].sum(): {cat['replaced_saturated'].sum()}")


def main():
    print("Starting main")
    import time
    t0 = time.time()

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-m", "--modules", dest="modules",
                      default='merged,merged-reproject',
                      help="module list", metavar="modules")
    parser.add_option('--merge-singlefields', dest='merge_singlefields',
                      default=False, action='store_true',)
    parser.add_option("--target", dest="target",
                      default='brick',
                      help="target", metavar="target")
    parser.add_option("--skip-crowdsource", dest="skip_crowdsource",
                      default=False,
                      action="store_true",
                      help="skip_crowdsource", metavar="skip_crowdsource")
    parser.add_option("--skip-daophot", dest="skip_daophot",
                      default=False,
                      action="store_true",
                      help="skip_daophot", metavar="skip_daophot")
    parser.add_option("--strict-require-blur", dest="strict_require_blur",
                      default=False,
                      action="store_true",
                      help="Fail if blur files are not found?", metavar="strict_require_blur")
    parser.add_option("--make-refcat", dest='make_refcat', default=False,
                      action='store_true')
    parser.add_option('--max-expnum', dest='max_expnum', default=24, type='int')
    parser.add_option('--indiv-merge-methods', dest='indiv_merge_methods', default='dao,crowdsource,daoiterative')
    parser.add_option('--iteration-label', dest='iteration_label', default=None,
                      help='Filter per-frame inputs to those tagged with this iteration label '
                           '(e.g. "iter2" or "iter3").  Default merges the iter1 catalogs.')
    parser.add_option('--ref-filter', dest='ref_filter', default=None,
                      help='Astrometric reference filter for the cross-filter merge '
                           '(default: f405n, which is correct for brick/cloudc/sgrb2/etc. '
                           'Targets that do not include F405N must override -- e.g. '
                           'sickle uses f470n.)')
    parser.add_option('--use-iter3-residual-bg', dest='resbgsub',
                      default=False, action='store_true',
                      help='Merge the catalogs produced with the merged iter3 '
                           'residual-bg subtraction (filenames carry a "_resbgsub" '
                           'token).  Must match the photometry run\'s flag.')
    parser.add_option('--merge-workers', dest='merge_workers', default=4, type='int',
                      help='Number of parallel workers for the spatial-chunked merge '
                           '(combine_singleframe_chunked).  >1 triggers auto-spatial-chunking '
                           'for fields with >1M detections; safe to leave at default 4 for '
                           'smaller fields (no-op below the 1M threshold).  Requires the '
                           'SLURM job to request --cpus-per-task >= this value.')
    parser.add_option('--n-spatial-chunks', dest='n_spatial_chunks', default=0, type='int',
                      help='Explicit spatial tile count for chunked merge.  0 = auto-derive '
                           'from --merge-workers (recommended).  Only needed to force a '
                           'specific tiling.')
    (options, args) = parser.parse_args()

    modules = options.modules.split(",")
    target = options.target
    indiv_merge_methods = options.indiv_merge_methods.split(",")
    print("Options:", options)

    if target in ('sickle', 'cloudef', 'sgrc', 'sgrb2', 'arches', 'quintuplet', 'sgra', 'gc2211'):
        basepath = f'/orange/adamginsburg/jwst/{target}/'
    else:
        basepath = f'/blue/adamginsburg/adamginsburg/jwst/{target}/'

    offsets_tables = {'1182': Table.read(f'/blue/adamginsburg/adamginsburg/jwst/brick/offsets/Offsets_JWST_Brick1182_F444ref.csv'),
                      '2221': None,
                      '3958': None,
                      '2092': None,
                      '4147': None,
                      '5365': None,
                      '2045': None,
                      '1939': None,
                      '2211': None,
                      '6151': None,  # W51; astrometry handled in imaging, no offsets table
    }

    # need to have incrementing _before_ test
    index = -1

    for module in modules:
        for desat in (False, True):
            for bgsub in (False, True):
                for epsf in (False, True):
                    for fitpsf in (False, True):
                        for blur in (False, True):

                            if options.merge_singlefields:
                                singlefield_done = False
                                for progid in obs_filters[target]:
                                    for filtername in (obs_filters[target][progid]):
                                        if singlefield_done:
                                            # skip ahead to merge-all-indiv step
                                            continue
                                        index += 1
                                        print(index, filtername, progid)
                                        # enable array jobs based only on filters
                                        if os.getenv('SLURM_ARRAY_TASK_ID') is not None and int(os.getenv('SLURM_ARRAY_TASK_ID')) != index:
                                            print(f'Task={os.getenv("SLURM_ARRAY_TASK_ID")} does not match index {index}')
                                            continue

                                        for method in indiv_merge_methods:
                                            print(method)
                                            # could loop & also do _iterative...
                                            suffix = {'crowdsource': '_nsky0',
                                                      'dao': '_basic',
                                                      'daoiterative': '_iterative',
                                                      'iterative': '_iterative',
                                                      }[method]
                                            try:
                                                merge_individual_frames(module=module,
                                                                        desat=desat,
                                                                        filtername=filtername,
                                                                        progid=progid,
                                                                        bgsub=bgsub,
                                                                        epsf=epsf,
                                                                        fitpsf=fitpsf,
                                                                        blur=blur,
                                                                        suffix=suffix,
                                                                        target=target,
                                                                        exposure_numbers=np.arange(1, options.max_expnum + 1),
                                                                        offsets_table=offsets_tables[progid],
                                                                        method=method,
                                                                        iteration_label=options.iteration_label,
                                                                        resbgsub=options.resbgsub,
                                                                        basepath=basepath,
                                                                        n_spatial_chunks=int(options.n_spatial_chunks),
                                                                        merge_workers=int(options.merge_workers))
                                            except ValueError as ex:
                                                if blur and not options.strict_require_blur:
                                                    print("Skipping missing blur files")
                                                else:
                                                    raise ex
                                            print(f"Finished merge_individual_frames {suffix} {progid} {filtername} {method}")
                                            if os.getenv('SLURM_ARRAY_TASK_ID') is not None:
                                                singlefield_done = True
                                            #except Exception as ex:
                                            #    print(f"Exception: {ex}, {type(ex)}, {str(ex)}")
                                            #    exc_type, exc_obj, exc_tb = sys.exc_info()
                                            #    print(f"Exception occurred on line {exc_tb.tb_lineno}")

                            else:
                                index += 1

                            # enable array jobs
                            if os.getenv('SLURM_ARRAY_TASK_ID') is not None and int(os.getenv('SLURM_ARRAY_TASK_ID')) != index:
                                print(f'Task={os.getenv("SLURM_ARRAY_TASK_ID")} does not match index {index}')
                                continue

                            t0 = time.time()
                            print(f"Index {index}")
                            if not options.skip_crowdsource:
                                print(f'crowdsource {module} desat={desat} bgsub={bgsub} epsf={epsf} blur={blur} fitpsf={fitpsf} target={target}. ', flush=True)
                                try:
                                    merge_crowdsource(module=module, desat=desat, bgsub=bgsub, epsf=epsf,
                                                      fitpsf=fitpsf, target=target, basepath=basepath, blur=blur, indivexp=options.merge_singlefields,
                                                      resbgsub=options.resbgsub,
                                                      iteration_label=options.iteration_label)
                                except Exception as ex:
                                    print(f"Living with this error: {ex}, {type(ex)}, {str(ex)}")
                                try:
                                    for suffix in ("_nsky0", ):#"_nsky15"): "_nsky1",
                                        print(f'crowdsource {suffix} {module}')
                                        merge_crowdsource(module=module, suffix=suffix, desat=desat, bgsub=bgsub, epsf=epsf,
                                                          fitpsf=fitpsf, target=target, basepath=basepath, blur=blur, indivexp=options.merge_singlefields,
                                                          resbgsub=options.resbgsub,
                                                          iteration_label=options.iteration_label)
                                except Exception as ex:
                                    print(f"Exception: {ex}, {type(ex)}, {str(ex)}")
                                    exc_type, exc_obj, exc_tb = sys.exc_info()
                                    print(f"Exception occurred on line {exc_tb.tb_lineno}")
                                    raise ex

                                try:
                                    print(f'crowdsource unweighted {module}', flush=True)
                                    merge_crowdsource(module=module, suffix='_unweighted', desat=desat, bgsub=bgsub, epsf=epsf,
                                                      fitpsf=fitpsf, target=target, basepath=basepath, blur=blur, indivexp=options.merge_singlefields,
                                                      resbgsub=options.resbgsub,
                                                      iteration_label=options.iteration_label)
                                except NotImplementedError:
                                    continue
                                except Exception as ex:
                                    print(f"Exception for unweighted crowdsource: {ex}, {type(ex)}, {str(ex)}")
                                    #raise ex

                                print(f'crowdsource phase done.  time elapsed={time.time()-t0}')

                            if not options.skip_daophot:
                                t0 = time.time()
                                print("DAOPHOT")
                                print(f'daophot basic {module} desat={desat} bgsub={bgsub} epsf={epsf} blur={blur} fitpsf={fitpsf} target={target}', flush=True)
                                try:
                                    merge_daophot(daophot_type='basic', module=module, desat=desat,
                                                  bgsub=bgsub, epsf=epsf,
                                                  target=target, basepath=basepath, blur=blur, indivexp=options.merge_singlefields,
                                                  ref_filter=options.ref_filter,
                                                  resbgsub=options.resbgsub,
                                                  iteration_label=options.iteration_label)
                                except Exception as ex:
                                    print(f'daophot basic {module} desat={desat} bgsub={bgsub} epsf={epsf} blur={blur} fitpsf={fitpsf} target={target}', flush=True)
                                    if blur and not options.strict_require_blur:
                                        print("Skipping missing blur files")
                                    elif isinstance(ex, ValueError) and 'had no matches' in str(ex):
                                        print(f"Skipping missing daophot basic catalogs (only daoiterative was run): {ex}", flush=True)
                                    else:
                                        print(f"Exception when running merge_daophot: {ex}, {type(ex)}, {str(ex)}", flush=True)
                                        exc_tb = sys.exc_info()[2]
                                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                                        print(f"Exception {ex} was in {fname} line {exc_tb.tb_lineno}", flush=True)
                                        print(f"Exception {ex} was in {fname} line {exc_tb.tb_next.tb_lineno}", flush=True)
                                        raise ex
                                try:
                                    print(f'daophot iterative {module} desat={desat} bgsub={bgsub} epsf={epsf} blur={blur} fitpsf={fitpsf} target={target}', flush=True)
                                    merge_daophot(daophot_type='iterative', module=module, desat=desat,
                                                  bgsub=bgsub, epsf=epsf,
                                                  target=target, basepath=basepath, blur=blur, indivexp=options.merge_singlefields,
                                                  ref_filter=options.ref_filter,
                                                  resbgsub=options.resbgsub,
                                                  iteration_label=options.iteration_label)
                                except Exception as ex:
                                    if blur and not options.strict_require_blur:
                                        print("Skipping missing blur files")
                                    else:
                                        print(f"Exception running merge daophot iterative: {ex}, {type(ex)}, {str(ex)}", flush=True)
                                        exc_tb = sys.exc_info()[2]
                                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                                        print(f"Exception {ex} was in {fname} line {exc_tb.tb_lineno}", flush=True)
                                print(f'dao phase done.  time elapsed={time.time()-t0}')
                                print()

                            if os.getenv('SLURM_ARRAY_TASK_ID') is None:
                                if options.make_refcat:
                                    raise Exception("make_refcat is not supported with SLURM_ARRAY_TASK_ID")
                                return

    if options.make_refcat:
        import make_reftable
        make_reftable.main()

    print("Done")


if __name__ == "__main__":
    main()
