import numpy as np
import os
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
import regions
from astroquery.vizier import Vizier
from astropy import log
from astropy.table import Table
import warnings

from astropy.wcs import WCS
from astropy.io import fits

import datetime

from jwst_gc_pipeline.catalog_utils import catalog_skycoord


def print(*args, **kwargs):
    now = datetime.datetime.now().isoformat()
    from builtins import print as printfunc
    log.info(f"{now}: {' '.join(map(str, args))}")
    return printfunc(f"{now}:", *args, **kwargs)


def _get_catalog_selection_column(cat):
    preferred_columns = (
        'aper_total_vegamag',
        'aper_total_abmag',
        'aper70_vegamag',
        'aper70_abmag',
        'isophotal_vegamag',
        'isophotal_abmag',
    )

    for column_name in preferred_columns:
        if column_name in cat.colnames:
            return column_name, 'magnitude'

    if 'flux' in cat.colnames:
        return 'flux', 'flux'

    raise KeyError(
        "Could not find a magnitude or flux column in catalog; looked for "
        f"{preferred_columns}. Available columns include: {cat.colnames}"
    )


def _get_catalog_pixel_coordinates(cat, ww):
    if 'xcentroid' in cat.colnames and 'ycentroid' in cat.colnames:
        return cat['xcentroid'], cat['ycentroid']

    skycoord = catalog_skycoord(cat, on_missing='none')
    if skycoord is not None:
        return ww.world_to_pixel(skycoord)

    raise KeyError(
        "Could not find pixel or sky-coordinate columns in catalog; available columns include: "
        f"{cat.colnames}"
    )

# NOTE (2026-07-11): the legacy standalone `main()` driver that produced
# `*_realigned-to-vvv.fits` / `*_realigned-to-refcat.fits` for every filter/module
# was removed with the realign retirement (it called the now-retired
# realign_to_vvv/realign_to_catalog). The reduction no longer produces those files;
# the `_i2d.fits` mosaic is the deliverable. See ASTROMETRY_WCS_CORRECTION_FLOW.md.


def retrieve_vvv(
    basepath = '/orange/adamginsburg/jwst/brick/',
    filtername = 'f212n',
    proposal_id='2221',
    module = 'nrca',
    imfile = None,
    catfile = None,
    fov_regname='regions/nircam_brick_fov.reg',
    fieldnumber='001',
):
    # VVV query region.  Prefer the actual image footprint (``imfile``): the FOV
    # region file holds only ONE box (``fov[0]``), so for a multi-pointing target
    # (e.g. cloudef = Cloud E obs002 + Cloud F obs005, which are disjoint on sky)
    # using fov[0] queries VVV in the wrong region for every observation that
    # isn't fov[0]'s footprint -> 0 crossmatches in realign_to_catalog (the
    # symptom that masqueraded as "no VVV overlap").  Deriving the region from
    # the image being aligned is correct for any pointing.
    if imfile is not None and os.path.exists(imfile):
        with fits.open(imfile) as _hl:
            _names = [h.name for h in _hl]
            _hdr = _hl['SCI'].header if 'SCI' in _names else _hl[0].header
            _w = WCS(_hdr)
            _fp = _w.calc_footprint()   # (N,2) RA,Dec of corners, deg
        _ra_lo, _ra_hi = float(_fp[:, 0].min()), float(_fp[:, 0].max())
        _dec_lo, _dec_hi = float(_fp[:, 1].min()), float(_fp[:, 1].max())
        _dec_c = 0.5 * (_dec_lo + _dec_hi)
        _cosd = np.cos(np.radians(_dec_c))
        coord = SkyCoord(0.5 * (_ra_lo + _ra_hi), _dec_c, unit='deg', frame='icrs')
        # angular box covering the footprint + 15% margin (RA span -> on-sky via cos dec)
        width = (_ra_hi - _ra_lo) * _cosd * 1.15 * u.deg
        height = (_dec_hi - _dec_lo) * 1.15 * u.deg
        log.info(f"retrieve_vvv: VVV query from {os.path.basename(imfile)} "
                 f"footprint center={coord.to_string('hmsdms')} "
                 f"width={width.to(u.arcmin):.2f} height={height.to(u.arcmin):.2f}")
    else:
        # Legacy fallback: single FOV box.  WRONG for multi-pointing targets;
        # only safe when fov[0] actually covers the image being aligned.
        log.warning(f"retrieve_vvv: no usable imfile ({imfile!r}); falling back to "
                    f"fov[0] of {fov_regname} for the VVV query region -- this is "
                    f"incorrect for multi-pointing targets.")
        fov = regions.Regions.read(os.path.join(basepath, fov_regname))
        coord = fov[0].center
        height = fov[0].height
        width = fov[0].width
        height, width = width, height # CARTA wrote it wrong

    vvvdr4filename = f'{basepath}/{filtername.upper()}/pipeline/jw0{proposal_id}-o{fieldnumber}_t001_nircam_clear-{filtername}-{module}_vvvcat.ecsv'

    if os.path.exists(vvvdr4filename):
        vvvdr4 = Table.read(vvvdr4filename)
        vvvdr4['RA'] = vvvdr4['RAJ2000']
        vvvdr4['DEC'] = vvvdr4['DEJ2000']

        # FK5 because it says 'J2000' on the Vizier page (same as twomass)
        vvvdr4_crds = SkyCoord(vvvdr4['RAJ2000'], vvvdr4['DEJ2000'], frame='fk5')
        if 'skycoord' not in vvvdr4.colnames:
            vvvdr4['skycoord'] = vvvdr4_crds
            vvvdr4.write(vvvdr4filename, overwrite=True)
            vvvdr4.write(vvvdr4filename.replace(".ecsv", ".fits"), overwrite=True)
    else:
        Vizier.ROW_LIMIT = 5e4
        vvvdr4 = Vizier.query_region(coordinates=coord, width=width, height=height, catalog=['II/376/vvv4'])[0]
        vvvdr4['RA'] = vvvdr4['RAJ2000']
        vvvdr4['DEC'] = vvvdr4['DEJ2000']

        # FK5 because it says 'J2000' on the Vizier page (same as twomass)
        vvvdr4_crds = SkyCoord(vvvdr4['RAJ2000'], vvvdr4['DEJ2000'], frame='fk5')
        vvvdr4['skycoord'] = vvvdr4_crds

        vvvdr4.write(vvvdr4filename, overwrite=True)
        vvvdr4.write(vvvdr4filename.replace(".ecsv", ".fits"), overwrite=True)

    assert 'skycoord' in vvvdr4.colnames
    return vvvdr4_crds, vvvdr4

def realign_to_vvv(*args, **kwargs):
    raise NotImplementedError(
        "realign_to_vvv was RETIRED 2026-07-11. The post-Image3 VVV/refcat realign is gone; "
        "the mosaic tie comes from per-exposure fix_alignment. Align dense refs with "
        "offset-histogram stacking (jwst_gc_pipeline.photometry.astrometry_offsets.measure_offset).")


def realign_to_catalog(*args, **kwargs):
    raise NotImplementedError(
        "realign_to_catalog was RETIRED 2026-07-11. It was a post-resample rigid CRVAL nudge that, "
        "after the dense-NN-median guard, was a no-op on GC/brick and only wrote a byte-identical "
        "_realigned-to-refcat.fits duplicate of _i2d.fits (not the release deliverable). The mosaic "
        "tie comes from per-exposure fix_alignment; validate with astrometry_offsets.measure_offset.")


def merge_a_plus_b(filtername,
    basepath = '/orange/adamginsburg/jwst/brick/',
    parallel=True,
    fieldnumber='001',
    proposal_id='2221',
    suffix='realigned-to-vvv',
    outsuffix='merged-reproject'
    ):
    """suffix can be realigned-to-vvv, realigned-to-refcat, or i2d"""
    import reproject
    from reproject.mosaicking import find_optimal_celestial_wcs, reproject_and_coadd
    filename_nrca = f'{basepath}/{filtername.upper()}/pipeline/jw0{proposal_id}-o{fieldnumber}_t001_nircam_clear-{filtername.lower()}-nrca{suffix}.fits'
    filename_nrcb = f'{basepath}/{filtername.upper()}/pipeline/jw0{proposal_id}-o{fieldnumber}_t001_nircam_clear-{filtername.lower()}-nrcb{suffix}.fits'
    files = [filename_nrca, filename_nrcb]
    missing_files = [filename for filename in files if not os.path.exists(filename)]
    if missing_files:
        raise FileNotFoundError(
            f"Missing expected module file(s) for {filtername} field={fieldnumber} proposal={proposal_id} suffix={suffix}: "
            + ", ".join(missing_files)
        )

    hdus = [fits.open(fn)[('SCI', 1)] for fn in files]
    ehdus = [fits.open(fn)[('ERR', 1)] for fn in files]
    weights = [fits.open(fn)[('WHT', 1)] for fn in files]

    # headers are only attached to the SCI frame for some reason!?
    for ehdu, hdu in zip(ehdus, hdus):
        ehdu.header.update(WCS(hdu).to_header())

    target_wcs, target_shape = find_optimal_celestial_wcs(hdus)
    merged, weightmap = reproject_and_coadd(hdus,
                                            output_projection=target_wcs,
                                            input_weights=weights,
                                            shape_out=target_shape,
                                            parallel=parallel,
                                            reproject_function=reproject.reproject_exact)
    merged_err, weightmap_ = reproject_and_coadd(ehdus,
                                                 output_projection=target_wcs,
                                                 input_weights=weights,
                                                 shape_out=target_shape,
                                                 parallel=parallel,
                                                 reproject_function=reproject.reproject_exact)
    header = fits.getheader(files[0])
    header.update(target_wcs.to_header())
    hdul = fits.HDUList([fits.PrimaryHDU(header=header),
                         fits.ImageHDU(data=merged, name='SCI', header=header),
                         fits.ImageHDU(data=merged_err, name='ERR', header=header),
                         fits.ImageHDU(data=weightmap, name='WHT', header=header),
                        ])
    outfn = f'{basepath}/{filtername.upper()}/pipeline/jw0{proposal_id}-o{fieldnumber}_t001_nircam_clear-{filtername.lower()}-{outsuffix}_i2d.fits'
    hdul.writeto(outfn, overwrite=True)
    return outfn
