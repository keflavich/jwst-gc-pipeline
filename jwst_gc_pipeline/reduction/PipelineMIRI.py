#!/usr/bin/env python
from glob import glob
from astroquery.mast import Mast, Observations
import copy
import os
import re
import shutil
import numpy as np
import json
import requests
import asdf # requires asdf < 3.0 (there is no replacement for this functionality w/o a major pattern change https://github.com/asdf-format/asdf/issues/1680)
import stdatamodels
try:
    from asdf.fits_embed import AsdfInFits
except ImportError:
    from stdatamodels import asdf_in_fits as AsdfInFits
from astropy import log
from astropy.coordinates import SkyCoord
from astropy.io import ascii, fits
from astropy.table import Table
from astropy.utils.data import download_file
from astropy.wcs import WCS
from astropy.visualization import ImageNormalize, ManualInterval, LogStretch, LinearStretch
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime

# do this before importing webb
os.environ["CRDS_PATH"] = "/orange/adamginsburg/jwst/brick/crds/"
os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"

from jwst.pipeline import calwebb_image3
from jwst.pipeline import Detector1Pipeline, Image2Pipeline

# Individual steps that make up calwebb_image3
from jwst.tweakreg import TweakRegStep
from jwst.skymatch import SkyMatchStep
from jwst.outlier_detection import OutlierDetectionStep
from jwst.resample import ResampleStep
from jwst.source_catalog import SourceCatalogStep
from jwst import datamodels
from jwst.associations import asn_from_list
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base
from jwst.tweakreg.utils import adjust_wcs
from jwst.datamodels import ImageModel

from jwst_gc_pipeline.reduction.align_to_catalogs import merge_a_plus_b, retrieve_vvv

import crds
import jwst

filter_regex = re.compile('f[0-9][0-9][0-9][nmw]')

import warnings
from astropy.utils.exceptions import AstropyWarning, AstropyDeprecationWarning
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.simplefilter('ignore', category=AstropyDeprecationWarning)
warnings.simplefilter('ignore', category=FITSFixedWarning)

def print(*args, **kwargs):
    now = datetime.datetime.now().isoformat()
    from builtins import print as printfunc
    # redundant log.info(f"{now}: {' '.join(map(str, args))}",)
    return printfunc(f"{now}:", *args, **kwargs)


print(jwst.__version__)

fov_regname = {'brick': 'regions_/nircam_brick_fov.reg',
               'cloudc': 'regions_/nircam_cloudc_fov.reg',
               'sickle': 'regions_/nircam_sickle_fov.reg',
               'w51': 'nope',
               'sgrb2': 'nope',
               }

# Reference catalog configuration by proposal and field.
# Paths are relative to basepath.
REFERENCE_ASTROMETRIC_CATALOG_CANDIDATES_BY_FIELD = {
    # Program 3958 MIRI fields.  obs 001/002 are the SICKLE; obs 003 is the
    # BRICK (routed to the brick/ tree -- see field_to_reg_mapping below).
    # Paths are relative to basepath, which is sickle/ for 001/002 and brick/
    # for 003, so 003 must point at the brick's NIRCam refcat (f405n-based),
    # NOT the sickle f210m catalog (which does not exist under brick/).
    '3958': {
        '001': (
            'catalogs/pipeline_based_nircam-f210m_reference_astrometric_catalog.fits',
            'catalogs/nircam_bootstrapped_to_gns_refcat.fits',
            'catalogs/nircam_bootstrapped_to_vvv_refcat.fits',
        ),
        '002': (
            'catalogs/pipeline_based_nircam-f210m_reference_astrometric_catalog.fits',
            'catalogs/nircam_bootstrapped_to_gns_refcat.fits',
            'catalogs/nircam_bootstrapped_to_vvv_refcat.fits',
        ),
        # obs 003 == brick field: align to the brick NIRCam absolute refcat.
        # NOTE 2026-06-16: the previous crowdsource_based_nircam-f405n_reference_
        # astrometric_catalog.fits DOES NOT EXIST under brick/catalogs/, so this
        # silently fell back to twomass (sparse/poor in the crowded GC).  Use the
        # existing F182M absolute reference (per user); twomass kept as last
        # resort only.
        '003': (
            'catalogs/pipeline_based_nircam-f182m_reference_astrometric_catalog.fits',
            'catalogs/twomass.fits',
        ),
    },
    '2221': {
        '001': (
            'catalogs/pipeline_based_nircam-f182m_reference_astrometric_catalog.fits',
            'catalogs/twomass.fits',
        ),
        '002': (
            'catalogs/pipeline_based_nircam-f182m_reference_astrometric_catalog.fits',
            'catalogs/twomass.fits',
        ),
    },
    # 2526 obs 021 == the "G0" CMZ cloud-c filament F770W pointing; routed into
    # the cloudc/ tree.  cloudc/catalogs has NO pipeline_based_nircam-f182m
    # refcat (unlike sickle/brick), so that first candidate silently fell through
    # to sparse twomass.  cloudc's NIRCam absolute frame is the Gaia DR3 + VIRAC2
    # seed (gaia_virac2_refcat_epoch2023.30.fits, the same frame cloudc NIRCam
    # 2221/002 aligns to); use it first, keeping the pipeline_based-* name (in
    # case it is ever built) and twomass as fallbacks.
    '2526': {
        '021': (
            'catalogs/gaia_virac2_refcat_epoch2023.30.fits',
            'catalogs/pipeline_based_nircam-f182m_reference_astrometric_catalog.fits',
            'catalogs/twomass.fits',
        ),
    },
}


def get_reference_astrometric_catalog_path(basepath, proposal_id, field, explicit_refcat=None):
    if explicit_refcat is not None:
        return explicit_refcat
    if proposal_id in REFERENCE_ASTROMETRIC_CATALOG_CANDIDATES_BY_FIELD:
        if field in REFERENCE_ASTROMETRIC_CATALOG_CANDIDATES_BY_FIELD[proposal_id]:
            for relpath in REFERENCE_ASTROMETRIC_CATALOG_CANDIDATES_BY_FIELD[proposal_id][field]:
                candidate = f'{basepath}/{relpath}'
                if os.path.exists(candidate):
                    return candidate
    twomass = f'{basepath}/catalogs/twomass.fits'
    if os.path.exists(twomass):
        return twomass
    return None


# Detector edge-glow trim margins (columns/rows DQ-flagged DO_NOT_USE per frame
# before image3).  MIRI imaging frames have excess flux at the detector edges --
# worst on the high-x ("east") edge (thermal glow reaches +150-1400 MJy/sr at
# 25um) and varying frame-to-frame -- which neither skymatch nor per-tile planes
# can remove.  Left in, it produces tile-boundary SEAMS and bright edge rims that
# the cataloger detects as fake sources (brick F2550W seam at x=1609; w51 edge
# artifacts at all boundaries).  Excluding the margin removes them; in tile
# overlaps the trimmed pixels are covered by neighbouring frames' interiors, so
# the mosaic only shrinks by the trim width at the true outer boundary.
# Generalized from scripts/miri_reduction/miri_f2550w_edgetrim_v4.py (2026-06-11).
# MIRI_TRIM_EAST is now a FLOOR (minimum east columns always trimmed); the
# actual east trim is ADAPTIVE per frame -- see _adaptive_east_trim.  A blind
# fixed east margin discarded clean edge data in frames whose east edge does NOT
# glow (brick F2550W visit-001 _08/10/12101 cover a detector-gap notch at their
# far-east edge with flat, un-glowing data -- a fixed E40 flagged them
# DO_NOT_USE and reopened the notch, since the eastern-neighbour tile carries
# the coronagraph defect there).  The adaptive trim flags only the contiguous
# east-edge run whose column-median is elevated above the frame interior (the
# thermal-glow signature), so glowing frames are trimmed (often deeper than the
# old fixed margin) while clean frames keep their real edge data.
MIRI_TRIM_EAST = int(os.environ.get('MIRI_TRIM_EAST', 0))
MIRI_TRIM_WEST = int(os.environ.get('MIRI_TRIM_WEST', 16))
MIRI_TRIM_ROWS = int(os.environ.get('MIRI_TRIM_ROWS', 12))
# adaptive east-trim controls
MIRI_TRIM_EAST_ADAPT = int(os.environ.get('MIRI_TRIM_EAST_ADAPT', 1))
MIRI_TRIM_EAST_THRESH = float(os.environ.get('MIRI_TRIM_EAST_THRESH', 0.08))
MIRI_TRIM_EAST_MAX = int(os.environ.get('MIRI_TRIM_EAST_MAX', 96))


def _adaptive_east_trim(dq, sci, lo, hi, thresh=MIRI_TRIM_EAST_THRESH,
                        east_max=MIRI_TRIM_EAST_MAX, gap_tol=3,
                        rowlo=200, rowhi=800):
    """Number of east-edge columns to DQ-flag, measured as the contiguous run
    (scanning inward from the science-column maximum ``hi``) whose central
    column-median exceeds the frame's interior baseline by ``thresh`` -- the
    edge-glow signature.  Returns 0 for frames with a flat (un-glowing) east
    edge so genuine edge data is preserved.  ``gap_tol`` bridges short
    column-median dips inside an otherwise-elevated glow run."""
    good = (dq & 1) == 0
    m = np.where(good, sci, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        colmed = np.nanmedian(m[rowlo:rowhi, :], axis=0)
        ref = np.nanmedian(colmed[max(lo, hi - 200):max(lo + 1, hi - 70)])
    if not np.isfinite(ref) or ref <= 0:
        return 0
    thr = ref * (1.0 + thresh)
    run = 0
    flagged = 0
    gap = 0
    for x in range(hi, max(lo, hi - east_max) - 1, -1):
        if np.isfinite(colmed[x]) and colmed[x] > thr:
            run += 1
            gap = 0
            flagged = run
        else:
            gap += 1
            if gap > gap_tol:
                break
            run += 1
    return flagged


def _edge_trim_dq(fn, east=MIRI_TRIM_EAST, west=MIRI_TRIM_WEST, rows=MIRI_TRIM_ROWS):
    """DQ-flag (DO_NOT_USE) a detector-edge margin in-place on a cal/align frame
    so the edge-glow-contaminated pixels are excluded from the image3 resample.
    The science-column span is detected per frame (where not everything is
    already DO_NOT_USE) so the trim is measured from the illuminated region, not
    the raw array edge.  The EAST trim is adaptive (per-frame glow detection)
    unless MIRI_TRIM_EAST_ADAPT=0; ``east`` then acts as a floor (minimum east
    columns).  Idempotent: re-flagging already-flagged pixels is a no-op."""
    with fits.open(fn, mode='update') as fh:
        dq = fh['DQ'].data
        if dq is None:
            return
        colgood = ((dq & 1) == 0).any(axis=0)
        if not colgood.any():
            return
        sci_cols = np.where(colgood)[0]
        lo, hi = int(sci_cols.min()), int(sci_cols.max())
        east_eff = east
        if MIRI_TRIM_EAST_ADAPT:
            adaptive = _adaptive_east_trim(dq, fh['SCI'].data, lo, hi)
            east_eff = max(east, adaptive)
        if east_eff > 0:
            dq[:, hi - east_eff + 1:] |= 1
        if west > 0:
            dq[:, :lo + west] |= 1
        if rows > 0:
            dq[:rows, :] |= 1
            dq[-rows:, :] |= 1
        fh['DQ'].data = dq
        try:
            fh[1].header['EDGETRIM'] = (f'E{east_eff} W{west} R{rows}',
                                        'detector edge-glow margin DQ-flagged')
        except (KeyError, IndexError):
            pass


def _regen_per_exposure_i2d(output_dir, field):
    """Regenerate the per-exposure single-frame ``_i2d`` from the FINAL aligned
    crf so the on-disk ``jw..._mirimage_i2d.fits`` carry the corrected
    (post-tweakreg + fix_alignment) WCS, not the raw-pointing Image2 WCS.  The
    Image2 _i2d are resampled from the raw cal -- leaving them in place puts
    "final" files with wrong astrometry on disk (brick F2550W visit-001 was off
    by 4.28")."""
    import re
    from jwst.resample import ResampleStep
    crfs = sorted(glob(os.path.join(output_dir, f'jw*_mirimage_*o{field}_crf.fits')))
    crfs = [c for c in crfs
            if re.search(r'_\d{5}_\d{5}_mirimage', os.path.basename(c))]
    # dedupe by output stem -> newest crf (a stale non-homogenized `_o{field}_crf`
    # and the canonical `_align_o{field}_crf` both map to the same _i2d; a blind
    # loop would overwrite last-write-wins and could pick the stale one).
    by_stem = {}
    for c in crfs:
        stem = os.path.basename(c).split('_mirimage')[0] + '_mirimage'
        if stem not in by_stem or os.path.getmtime(c) > os.path.getmtime(by_stem[stem]):
            by_stem[stem] = c
    crfs = sorted(by_stem.values())
    n = 0
    for crf in crfs:
        stem = os.path.basename(crf).split('_mirimage')[0] + '_mirimage'
        out = os.path.join(output_dir, f'{stem}_i2d.fits')
        try:
            res = ResampleStep.call(crf, output_dir=output_dir, save_results=False)
            res.save(out, overwrite=True)
            if hasattr(res, 'close'):
                res.close()
            n += 1
        except Exception as ex:
            print(f"  per-exposure _i2d regen FAILED {os.path.basename(crf)}: {ex}",
                  flush=True)
    print(f"regenerated {n}/{len(crfs)} per-exposure _i2d with corrected (crf) WCS",
          flush=True)


def relocate_manifest_products(manifest, output_dir):
    """Flatten MAST download tree into output_dir with idempotent relocation."""
    for row in manifest:
        src = str(row['Local Path'])
        dst = os.path.join(output_dir, os.path.basename(src))

        if os.path.exists(dst):
            # Common when rerunning and MAST points to a file already moved earlier.
            print(f"Relocation skipped: destination already exists ({dst})")
            continue

        try:
            shutil.move(src, dst)
        except FileNotFoundError:
            if os.path.exists(dst):
                print(f"Relocation skipped: source missing but destination exists ({dst})")
            else:
                raise FileNotFoundError(
                    f"MAST manifest source missing and destination not present: src={src} dst={dst}"
                )
        except shutil.Error as ex:
            print(f"Failed to move file with error {ex}")


def main(filtername, Observations=None, regionname='brick',
         field='001', proposal_id='2221', skip_step1and2=False, use_average=True,
         reference_catalog=None, skip_download_for_existing=False,
         marshall_tuning=False):
    """
    skip_step1and2 will not re-fit the ramps to produce the _cal images.  This
    can save time if you just want to redo the tweakreg steps but already have
    the zero-frame stuff done.
    """
    print(f"Processing filter {filtername} with and skip_step1and2={skip_step1and2} for field {field} and proposal id {proposal_id} in region {regionname}")

    wavelength = int(filtername[1:4])

    basepath = f'/orange/adamginsburg/jwst/{regionname}/'
    fwhm_tbl = Table.read(f'{basepath}/reduction/fwhm_table.ecsv')
    row = fwhm_tbl[fwhm_tbl['Filter'] == filtername]
    fwhm = fwhm_arcsec = float(row['PSF FWHM (arcsec)'][0])
    fwhm_pix = float(row['PSF FWHM (pixel)'][0])

    # sanity check
    if regionname == 'brick':
        if proposal_id == '2221':
            # jw02221-o002_t001_miri_f2550w_i2d.fits
            assert field == '002'
        elif proposal_id == '3958':
            # 3958 obs 003 (t003) is the brick MIRI field (shares the 3958
            # program id with the sickle, but is spatially/scientifically the
            # brick); routed here so it lands in brick/, not sickle/.
            assert field == '003'
    elif regionname == 'cloudc':
        if proposal_id == '2221':
            # jw02221-o001_t001_miri_f2550w_i2d.fits
            assert field == '001'
        elif proposal_id == '2526':
            # 2526 obs 021 (t013) is the "G0" CMZ cloud-c filament F770W pointing
            # (~5.4' from the 2221 cloudc field; distinct pointing, same region
            # tree -- reuses cloudc reduction/fwhm_table + crds + catalogs).
            assert field == '021'
    elif regionname == 'sickle':
        # ONLY obs 001/002 are the sickle; obs 003 is the brick (see above).
        assert proposal_id == '3958'
        assert field in ('001', '002')

    # Use the per-target CRDS cache when it is writable; some target caches
    # (e.g. cloudc) are owned by another user and CRDS cannot update them for
    # newer contexts, so fall back to the shared brick cache in that case.
    crds_path = f"{basepath}/crds/"
    crds_mapdir = os.path.join(crds_path, 'mappings', 'jwst')
    if os.path.isdir(crds_mapdir) and not os.access(crds_mapdir, os.W_OK):
        print(f"CRDS cache {crds_path} is not writable; using shared brick cache instead")
        crds_path = "/orange/adamginsburg/jwst/brick/crds/"
    else:
        os.makedirs(crds_mapdir, exist_ok=True)
    os.environ["CRDS_PATH"] = crds_path
    os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"
    mpl.rcParams['savefig.dpi'] = 80
    mpl.rcParams['figure.dpi'] = 80



    # Files created in this notebook will be saved
    # in a subdirectory of the base directory called `Stage3`
    output_dir = f'/orange/adamginsburg/jwst/{regionname}/{filtername}/pipeline/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)

    # the files are one directory up
    for fn in glob("../*cal.fits"):
        try:
            os.link(fn, './'+os.path.basename(fn))
        except FileExistsError as ex:
            print(f'Failed to link {fn} to {os.path.basename(fn)} because of {ex}')

    Observations.cache_location = output_dir
    obs_table = Observations.query_criteria(
                                            proposal_id=proposal_id,
                                            #proposal_pi="Ginsburg*",
                                            #calib_level=3,
                                            )
    print("Obs table length:", len(obs_table))

    if 'filters' in obs_table.colnames and 'obs_id' in obs_table.colnames:
        try:
            filters_col = np.array([str(val).upper() for val in obs_table['filters'].filled('')])
            obs_id_col = np.array([str(val).lower() for val in obs_table['obs_id'].filled('')])
        except AttributeError:
            filters_col = np.array([str(val).upper() for val in obs_table['filters']])
            obs_id_col = np.array([str(val).lower() for val in obs_table['obs_id']])
        msk = ((np.char.find(filters_col, filtername.upper()) >= 0) |
               (np.char.find(obs_id_col, filtername.lower()) >= 0))
    else:
        print("Warning: 'filters' or 'obs_id' column missing in obs_table; selecting all observations for this proposal")
        msk = np.ones(len(obs_table), dtype=bool)
    data_products_by_obs = Observations.get_product_list(obs_table[msk])
    print("data prodcts by obs length: ", len(data_products_by_obs))

    products_asn = Observations.filter_products(data_products_by_obs, extension="json")
    print("products_asn length:", len(products_asn))
    #valid_obsids = products_asn['obs_id'][np.char.find(np.unique(products_asn['obs_id']), 'jw02221-o001', ) == 0]
    #match = [x for x in valid_obsids if filtername.lower() in x][0]

    asn_mast_data = products_asn#[products_asn['obs_id'] == match]
    print("asn_mast_data:", asn_mast_data)

    manifest = Observations.download_products(asn_mast_data, download_dir=output_dir)
    print("manifest:", manifest)

    # MAST creates deep directory structures we don't want
    relocate_manifest_products(manifest, output_dir)

    products_fits = Observations.filter_products(data_products_by_obs, extension="fits")
    print("products_fits length:", len(products_fits))
    uncal_mask = np.array([uri.endswith('_uncal.fits') and f'jw0{proposal_id}{field}' in uri for uri in products_fits['dataURI']])
    uncal_mask &= products_fits['productType'] == 'SCIENCE'
    print("uncal length:", (uncal_mask.sum()))

    if skip_download_for_existing:
        already_downloaded = np.array([os.path.exists(os.path.basename(uri)) for uri in products_fits['dataURI']])
        uncal_mask &= ~already_downloaded
        print(f"uncal to download: {uncal_mask.sum()}; {already_downloaded.sum()} were already downloaded")

    if uncal_mask.any():
        manifest = Observations.download_products(products_fits[uncal_mask], download_dir=output_dir)
        print("manifest:", manifest)

        # MAST creates deep directory structures we don't want
        relocate_manifest_products(manifest, output_dir)

    if True: # just to preserve indendation
        print(f"Working on MIRI: running initial pipeline setup steps (skip_step1and2={skip_step1and2})")
        print(f"Searching for {os.path.join(output_dir, f'jw0{proposal_id}-o{field}*_image3_*0[0-9][0-9]_asn.json')}")
        asn_file_search = glob(os.path.join(output_dir, f'jw0{proposal_id}-o{field}*_image3_*0[0-9][0-9]_asn.json'))
        if len(asn_file_search) == 1:
            asn_file = asn_file_search[0]
        elif len(asn_file_search) > 1:
            asn_file = sorted(asn_file_search)[-1]
            print(f"Found multiple asn files: {asn_file_search}.  Using the more recent one, {asn_file}.")
        else:
            raise ValueError(f"Mismatch: Did not find any asn files for field {field} in {output_dir}")

        mapping = crds.rmap.load_mapping(f'{os.environ["CRDS_PATH"]}/mappings/jwst/jwst_miri_pars-tweakregstep_0003.rmap')
        print(f"Mapping: {mapping.todict()['selections']}")
        print(f"Filtername: {filtername}")
        filter_match = [x for x in mapping.todict()['selections'] if filtername.upper() in x]
        print(f"Filter_match: {filter_match} n={len(filter_match)}")
        tweakreg_asdf_filename = filter_match[0][3]
        tweakreg_asdf = asdf.open(f'https://jwst-crds.stsci.edu/unchecked_get/references/jwst/{tweakreg_asdf_filename}')
        tweakreg_parameters = tweakreg_asdf.tree['parameters']
        """
        # may not be needed for MIRI
        #tweakreg_parameters.update({'fitgeometry': 'general',
        #                            # brightest = 5000 was causing problems- maybe the cross-alignment was getting caught on PSF artifacts?
        #                            'brightest': 5000,
        #                            'snr_threshold': 20, # was 5, but that produced too many stars
        #                            # define later 'abs_refcat': abs_refcat,
        #                            'save_catalogs': True,
        #                            'catalog_format': 'fits',
        #                            'kernel_fwhm': fwhm_pix,
        #                            'nclip': 5,
        #                            # expand_refcat: A boolean indicating whether or not to expand reference catalog with new sources from other input images that have been already aligned to the reference image. (Default=False)
        #                            'expand_refcat': True,
        #                            # based on DebugReproduceTweakregStep
        #                            'sharplo': 0.3,
        #                            'sharphi': 0.9,
        #                            'roundlo': -0.25,
        #                            'roundhi': 0.25,
        #                            'separation': 0.5, # minimum separation; default is 1
        #                            'tolerance': 0.1, # tolerance: Matching tolerance for xyxymatch in arcsec. (Default=0.7)
        #                            'save_results': True,
        #                            # 'clip_accum': True, # https://github.com/spacetelescope/tweakwcs/pull/169/files
        #                            })
        """

        print(f'Filter {filtername} tweakreg parameters: {tweakreg_parameters}')

        with open(asn_file) as f_obj:
            asn_data = json.load(f_obj)

        print(f"In cwd={os.getcwd()}")
        members = asn_data['products'][0]['members']
        if skip_step1and2:
            missing_cal = [member['expname'] for member in members if not os.path.exists(member['expname'])]
            if len(missing_cal) == 0:
                print("Skipped step 1 and step2")
            else:
                print(f"skip_step1and2 requested, but {len(missing_cal)} _cal files are missing; running detector/image2 for missing files")

        if (not skip_step1and2) or (skip_step1and2 and len([member['expname'] for member in members if not os.path.exists(member['expname'])]) > 0):
            # re-calibrate uncal files -> cal files *without* suppressing first group
            for member in members:
                assert f'jw0{proposal_id}{field}' in member['expname']
                cal_name = member['expname']
                if skip_step1and2 and os.path.exists(cal_name):
                    continue

                print(f"DETECTOR PIPELINE on {cal_name}")
                print("Detector1Pipeline step")
                # from Hosek: expand_large_events -> false; turn off "snowball" detection
                detector1_steps = {'ramp_fit': {'suppress_one_group': False},
                                   'refpix': {'use_side_ref_pixels': True}}
                if marshall_tuning:
                    detector1_steps.update({'saturation': {'skip': True, 'n_pix_grow_sat': 0},
                                            'firstframe': {'skip': True},
                                            'rscd': {'skip': True}})
                Detector1Pipeline.call(cal_name.replace("_cal.fits", "_uncal.fits"),
                                       save_results=True, output_dir=output_dir,
                                       save_calibrated_ramp=True,
                                       steps=detector1_steps)

                print(f"IMAGE2 PIPELINE on {cal_name}")
                Image2Pipeline.call(cal_name.replace("_cal.fits", "_rate.fits"),
                                    save_results=True, output_dir=output_dir,
                                    #steps={'background': {'run': False}},
                                   )

    if True:
        print(f"Filter {filtername}: doing tweakreg.  ")

        with open(asn_file) as f_obj:
            asn_data = json.load(f_obj)
        asn_data['products'][0]['name'] = f'jw0{proposal_id}-o{field}_t001_miri_{filtername.lower()}'
        asn_data['products'][0]['members'] = [row for row in asn_data['products'][0]['members']]

        for member in asn_data['products'][0]['members']:
            print(f"Running  maybe alignment on {member}")
            fname = member['expname']
            # Idempotent re-run: this loop rewrites the asn with _align.fits
            # members at the end (json.dump below), so on a RE-RUN the members
            # are already _align.fits and the old `assert _cal.fits` failed.
            # Normalize back to the _cal.fits source, then re-copy + re-align
            # fresh from cal each run.
            cal_name = fname.replace("_align.fits", "_cal.fits")
            assert cal_name.endswith('_cal.fits'), cal_name
            align_name = cal_name.replace("_cal.fits", "_align.fits")
            member['expname'] = align_name
            # copyFILE (data only), NOT copy: shutil.copy also copymode()s the
            # destination to match the source, which raises PermissionError
            # (EPERM) when the cal is owned by another user / the output dir is
            # setgid+ACL (w51 F2100W re-reduction 2026-06-25).  The align copy
            # only needs the cal DATA for fix_alignment; its mode is irrelevant.
            shutil.copyfile(cal_name, align_name)

            # Detector edge-glow trim: DQ-flag a margin around the frame so the
            # contaminated edge pixels are excluded from the image3 resample
            # (removes tile-boundary seams + bright edge rims that the cataloger
            # detects as fake sources).  Operates on DQ only; independent of the
            # WCS correction that fix_alignment applies next.  Disable per-run
            # with MIRI_TRIM_EAST=MIRI_TRIM_WEST=MIRI_TRIM_ROWS=0.
            _edge_trim_dq(align_name)

            fix_alignment(member['expname'], proposal_id=proposal_id,
                          field=field, basepath=basepath,
                          regionname=regionname,
                          filtername=filtername,
                          use_average=use_average,
                          visit=fname[10:13])

        asn_file_each = asn_file
        with open(asn_file_each, 'w') as fh:
            json.dump(asn_data, fh)

        # Force SHIFT-ONLY alignment (no rotation/scale).  MIRI BRIGHTSKY fields
        # have very few matched stars per frame, and allowing rotation lets
        # tweakreg fit a catastrophic spurious rotation -- e.g. sickle F1130W
        # o001 came out rotated ~45deg + flipped.  We have never found a real
        # nonzero MIRI rotation correction, so disallow it for both the relative
        # (frame-to-frame) and absolute (to-refcat) fits.
        tweakreg_parameters['fitgeometry'] = 'shift'
        tweakreg_parameters['abs_fitgeometry'] = 'shift'

        abs_refcat = get_reference_astrometric_catalog_path(basepath, proposal_id, field, explicit_refcat=reference_catalog)
        if abs_refcat is not None:
            reftbl = Table.read(abs_refcat)
            reftbl.meta['name'] = 'Reference Astrometric Catalog'

            tweakreg_parameters['abs_searchrad'] = 0.4
            # try forcing searchrad to be tighter to avoid bad crossmatches
            # (the raw data are very well-aligned to begin with, though CARTA
            # can't display them b/c they are using SIP)
            tweakreg_parameters['searchrad'] = 0.05
            # MIRI BRIGHTSKY fields can have very few matched stars per frame.
            tweakreg_parameters['minobj'] = 2
            # abs_minobj=2 let UNDER-COVERED frames (whose footprint extends
            # beyond the reference catalog) latch onto 2 spurious cross-matches
            # and apply a catastrophic, per-frame-divergent absolute shift --
            # sickle F770W o001 came out shifted 20-62" southward (each frame
            # different) because the f210m/GNS/VVV refcats only cover the
            # NORTHERN strip (Dec >= -28.808) while o001 extends to -28.828, so
            # its southern frames had ~0 real reference stars.  shift-only
            # geometry does NOT help: a 2-point fit to wrong pairs still yields a
            # wrong translation.  Require >=5 absolute matches so an
            # under-covered frame instead FAILS the absolute fit and is left at
            # its (good, ~0.1" guide-star) raw pointing -- a near-zero no-op
            # rather than a tens-of-arcsec blunder.  Well-covered frames (o002,
            # brick o003) have many matches and are unaffected.
            tweakreg_parameters['abs_minobj'] = 5
            print(f"Reference catalog is {abs_refcat}")

            tweakreg_parameters.update({'abs_refcat': abs_refcat,})
        else:
            print(f"No reference catalog found for proposal_id={proposal_id} field={field} in {basepath}; running without abs_refcat")

        # subtract=True is essential: with subtract=False the matched sky
        # levels are only recorded, so outlier_detection's median image sees
        # the raw inter-visit thermal-background jumps and flags entire
        # regions as OUTLIER in every frame -> resample gets zero valid
        # inputs -> large NaN patches (brick F2550W showed 87-99% OUTLIER
        # fractions in its NaN zones; fixed by this + gentler snr,
        # validated 2026-06-10, jw02221-o002 f2550w pipeline_v2 experiment).
        skymatch_params = {'save_results': True,
                           'subtract': True,
                           'skymethod': 'match',
                           'match_down': False}
        outlier_params = {'snr': '30.0 25.0',
                          'good_bits': "SATURATED, JUMP_DET"}
        if marshall_tuning:
            skymatch_params = {'save_results': True,
                               'subtract': True,
                               'skymethod': 'global',
                               'match_down': True}
            outlier_params = {'snr': "30.0 5.0",
                              'good_bits': "SATURATED, JUMP_DET",
                              'save_intermediate_results': True}

        print("Running tweakreg")
        calwebb_image3.Image3Pipeline.call(
            asn_file_each,
            steps={'tweakreg': tweakreg_parameters,
                   'skymatch': skymatch_params,
                   'outlier_detection': outlier_params,
            },
            output_dir=output_dir,
            save_results=True)
        print(f"DONE running {asn_file_each}")

        # CRF NAMING FIX (2026-06-20): because we set the asn product name above
        # (line ~375) to 'jw0{prop}-o{field}_t001_miri_{filt}', outlier_detection
        # names the CR-flagged products after the PRODUCT, not the exposure:
        #   jw03958-o001_t001_miri_f770w_<N>_o{field}_crf.fits
        # The per-frame photometry (crowdsource_catalogs_long.get_filenames) globs
        # PER-EXPOSURE crf:  jw0{prop}{field}{visit}*{module}*o{field}_crf.fits
        # i.e. jw03958001001_*_mirimage_o001_crf.fits .  Those two never matched,
        # so a corrected re-reduction's crf silently never reached cataloging
        # (the o001 37deg-rotation fix was invisible for days for exactly this).
        # Map product-named crf -> per-exposure names by EXPSTART (1:1) and copy
        # them into place so cataloging consumes the freshly-corrected WCS.
        from astropy.io import fits as _fits
        prod_name = asn_data['products'][0]['name']
        prod_crf = sorted(glob(os.path.join(output_dir, f'{prod_name}_*_o{field}_crf.fits')))
        if prod_crf:
            def _expstart(fn):
                h = _fits.getheader(fn)
                es = h.get('EXPSTART')
                if es is None and len(_fits.open(fn)) > 1:
                    es = _fits.getheader(fn, 1).get('EXPSTART')
                return round(float(es), 6)
            # target per-exposure name derived from each cal/align member
            targ_by_es = {}
            for member in asn_data['products'][0]['members']:
                cal_base = os.path.basename(member['expname']).replace('_align.fits', '_cal.fits')
                target = os.path.join(output_dir, cal_base.replace('_cal.fits', f'_o{field}_crf.fits'))
                # cal may live in output_dir or resolve relative to cwd; try both
                cal_path = cal_base if os.path.exists(cal_base) else os.path.join(output_dir, cal_base)
                try:
                    targ_by_es[_expstart(cal_path)] = target
                except (FileNotFoundError, OSError, TypeError):
                    print(f"  WARNING: cannot read EXPSTART of {cal_base}; skipping its crf mapping")
            for pc in prod_crf:
                es = _expstart(pc)
                target = targ_by_es.get(es)
                if target is None:
                    print(f"  WARNING: product crf {os.path.basename(pc)} EXPSTART={es} "
                          f"has no per-exposure cal match; per-exposure crf NOT written")
                    continue
                shutil.copy(pc, target)
                print(f"  crf rename: {os.path.basename(pc)} -> {os.path.basename(target)}")
        else:
            print(f"  (no product-named crf {prod_name}_*_o{field}_crf.fits found; "
                  f"assuming crf already per-exposure named)")

        # Update the per-exposure single-frame _i2d to the FINAL aligned WCS
        # (regenerate from crf) -- the pipeline must not leave "final" _i2d on
        # disk with the raw-pointing Image2 astrometry.
        _regen_per_exposure_i2d(output_dir, field)

        print("After tweakreg step, checking WCS headers:")
        for member in asn_data['products'][0]['members']:
            check_wcs(member['expname'])
            check_wcs(member['expname'].replace('cal', 'i2d').replace('destreak', 'i2d'))
        check_wcs(asn_data['products'][0]['name'] + "_i2d.fits")

    globals().update(locals())
    return locals()


def fix_alignment(fn, proposal_id=None, regionname='brick', field=None, basepath=None, filtername=None,
                  use_average=True, visit='003'):
    if os.path.exists(fn):
        print(f"Running manual align for data ({proposal_id} + {field}): {fn}", flush=True)
    else:
        print(f"Skipping manual align for nonexistent file ({proposal_id} + {field}): {fn}", flush=True)
        return

    mod = ImageModel(fn)
    if proposal_id is None:
        proposal_id = os.path.basename(fn)[3:7]
    if filtername is None:
        try:
            filtername = filter_regex.search(fn).group()
        except AttributeError:
            filters = tuple(map(str.lower, (mod.meta.instrument.filter, mod.meta.instrument.pupil)))
            if 'clear' in filters:
                filtername = [x for x in filters if x != 'clear'][0]
            else:
                # any filter that is not the wideband filter
                filtername = [x for x in filters if 'W' not in x][0]
    if field is None:
        field = mod.meta.observation.observation_number
    if basepath is None:
        basepath = f'/orange/adamginsburg/jwst/{regionname}'

    # 2026-06-11: the historical "Brick/CloudC" shift (-3.895", +1.28") was
    # measured offset-histogram-stacking the final mosaics against the NIRCam
    # reference catalogs to be the dominant astrometric ERROR, not a fix:
    # sickle o003 and cloudc F2550W both come out displaced by ~(-3.4..-3.5",
    # +1.4..+1.5") -- i.e. by this shift -- because modern raw pointing is
    # already good and tweakreg's search radii (0.05"/0.4") cannot recover a
    # 4" imposed offset.  Default is now NO shift; per-target values can be
    # reinstated here if a target is shown (by offset-histogram measurement,
    # not nearest-neighbor matching!) to need one.
    rashift = 0 * u.arcsec
    decshift = 0 * u.arcsec
    # Marshall W51 tuning: small global RA correction for MIRI.
    if regionname == 'w51':
        rashift = 0.2 * u.arcsec
        decshift = 0 * u.arcsec
    # PER-VISIT WCS corrections (2026-06-25, restored).  Some obs have one VISIT
    # mutually misregistered from the other by several arcsec -- a misregistration
    # that standard tweakreg/refcat alignment CANNOT recover (too few F2550W
    # refcat counterparts at 25um; frame-vs-mosaic cross-correlation follows each
    # visit locally and is circular).  Left uncorrected it makes a DOUBLED star
    # and a tile-boundary SEAM where the two visits' tiles abut.  Values measured
    # from per-visit sub-mosaics vs NIRCam-confirmed stars (brick F2550W obs002
    # visit001 = the v14 fix that achieved the seamless mosaic; the standard
    # pipeline applying NO per-visit shift is the seam regression).
    # (regionname, field/obs, visit) -> (dRA arcsec, dDec arcsec).
    _PER_VISIT_SHIFT = {
        ('brick', '002', '001'): (-1.01, 4.12),
    }
    _pvs = _PER_VISIT_SHIFT.get((regionname, str(field), str(visit)))
    if _pvs is not None:
        rashift = _pvs[0] * u.arcsec
        decshift = _pvs[1] * u.arcsec
        print(f"  PER-VISIT WCS correction ({regionname}/{field}/v{visit}): "
              f"dRA={_pvs[0]}\", dDec={_pvs[1]}\"", flush=True)
    print(f"Shift for {fn} is {rashift}, {decshift}")
    align_fits = fits.open(fn)
    if 'RAOFFSET' in align_fits[1].header:
        # don't shift twice if we re-run
        print(f"{fn} is already aligned ({align_fits[1].header['RAOFFSET']}, {align_fits[1].header['DEOFFSET']})")
    else:
        # ASDF header
        fa = ImageModel(fn)
        wcsobj = fa.meta.wcs
        print(f"Before shift, crval={wcsobj.to_fits()[0]['CRVAL1']}, {wcsobj.to_fits()[0]['CRVAL2']}, {wcsobj.forward_transform.param_sets[-1]}")
        fa.meta.oldwcs = copy.copy(wcsobj)
        ww = adjust_wcs(wcsobj, delta_ra=rashift, delta_dec=decshift)
        print(f"After shift, crval={ww.to_fits()[0]['CRVAL1']}, {ww.to_fits()[0]['CRVAL2']}, {wcsobj.forward_transform.param_sets[-1]}")
        fa.meta.wcs = ww
        fa.save(fn, overwrite=True)

        # FITS header
        align_fits = fits.open(fn)
        align_fits[1].header['OLCRVAL1'] = align_fits[1].header['CRVAL1']
        align_fits[1].header['OLCRVAL2'] = align_fits[1].header['CRVAL2']
        align_fits[1].header.update(ww.to_fits()[0])
        align_fits[1].header['RAOFFSET'] = rashift.value
        align_fits[1].header['DEOFFSET'] = decshift.value
        align_fits.writeto(fn, overwrite=True)
        assert 'RAOFFSET' in fits.getheader(fn, ext=1)
    check_wcs(fn)


def check_wcs(fn):
    if os.path.exists(fn):
        print(f"Checking WCS of {fn}")
        fa = ImageModel(fn)
        wcsobj = fa.meta.wcs
        print(f"fa['meta']['wcs'] crval={wcsobj.to_fits()[0]['CRVAL1']}, {wcsobj.to_fits()[0]['CRVAL2']}, {wcsobj.forward_transform.param_sets[-1]}")
        new_1024 = wcsobj.pixel_to_world(1024, 1024)
        print(f"new pixel_to_world(1024,1024) = {new_1024}")
        if 'oldwcs' in fa.meta:
            oldwcsobj = fa.meta.oldwcs
            print(f"fa['meta']['oldwcs'] crval={oldwcsobj.to_fits()[0]['CRVAL1']}, {oldwcsobj.to_fits()[0]['CRVAL2']}, {oldwcsobj.forward_transform.param_sets[-1]}")
            old_1024 = oldwcsobj.pixel_to_world(1024, 1024)
            print(f"old pixel_to_world(1024,1024) = {old_1024}, sep from new GWCS={old_1024.separation(new_1024).to(u.arcsec)}")
        fa.close()

        # FITS header
        fh = fits.open(fn)
        print(f"CRVAL1={fh[1].header['CRVAL1']}, CRVAL2={fh[1].header['CRVAL2']}")
        if 'OLCRVAL1' in fh[1].header:
            print(f"OLCRVAL1={fh[1].header['OLCRVAL1']}, OLCRVAL2={fh[1].header['OLCRVAL2']}")
        if 'RAOFFSET' in fh[1].header:
            print("RA, DE offset: ", fh[1].header['RAOFFSET'], fh[1].header['DEOFFSET'])
        ww = WCS(fh[1].header)
        fits_1024 = ww.pixel_to_world(1024, 1024)
        print(f"FITS pixel_to_world(1024,1024) = {fits_1024}, sep from new GWCS={fits_1024.separation(new_1024).to(u.arcsec)}")
        fh.close()
    else:
        print(f"COULD NOT CHECK WCS FOR {fn}: does not exist")


if __name__ == "__main__":
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-f", "--filternames", dest="filternames",
                      default='F2550W',
                      help="filter name list", metavar="filternames")
    parser.add_option("-d", "--field", dest="field",
                    default='002',
                    help="list of target fields", metavar="field")
    parser.add_option("-s", "--skip_step1and2", dest="skip_step1and2",
                      default=False,
                      action='store_true',
                      help="Skip the image-remaking step?", metavar="skip_Step1and2")
    parser.add_option("-p", "--proposal_id", dest="proposal_id",
                      default='2221',
                      help="proposal id (string)", metavar="proposal_id")
    parser.add_option("--reference_catalog", dest="reference_catalog",
                      default=None,
                      help="Path to explicit astrometric reference catalog for tweakreg (optional)", metavar="reference_catalog")
    parser.add_option("--skip_download_for_existing", dest="skip_download_for_existing",
                      default=False, action='store_true',
                      help="Skip downloading _uncal files already present in output directory", metavar="skip_download_for_existing")
    parser.add_option("--marshall_tuning", dest="marshall_tuning",
                      default=False, action='store_true',
                      help="Enable Marshall W51-inspired MIRI tuning (Detector1/skymatch/outlier settings)", metavar="marshall_tuning")
    (options, args) = parser.parse_args()

    filternames = options.filternames.split(",")
    fields = options.field.split(",")
    proposal_id = options.proposal_id
    skip_step1and2 = options.skip_step1and2
    reference_catalog = options.reference_catalog
    skip_download_for_existing = options.skip_download_for_existing
    marshall_tuning = options.marshall_tuning
    print(options)

    with open(os.path.expanduser('~/.mast_api_token'), 'r') as fh:
        api_token = fh.read().strip()
        os.environ['MAST_API_TOKEN'] = api_token.strip()
    Mast.login(api_token.strip())
    Observations.login(api_token)


    # NOTE: program 3958 obs 003 (t003) is the BRICK MIRI field, NOT the
    # sickle.  Only 3958 obs 001/002 belong to the sickle.  Route o003 to the
    # brick/ tree so its images + catalogs land under /orange/.../jwst/brick/
    # and never clash with sickle/ products (which share the 3958 program id).
    field_to_reg_mapping = {'2221': {'002': 'brick', '001': 'cloudc'},
                            '3958': {'001': 'sickle', '002': 'sickle', '003': 'brick'},
                            # 5365 sgrb2 MIRI: the actual observations are obs
                            # 002 + obs 998 ("..._skipped_redo"); together their
                            # 4 mosaic tiles (002: 0210b/02105, 998: 06101/12101)
                            # tile the full Sgr B2 field -- BOTH must be reduced
                            # and joint-cataloged (--field=002-998) or the mosaic
                            # is missing half the data.
                            '5365': {'001': 'sgrb2', '002': 'sgrb2', '998': 'sgrb2'},
                            '6151': {'001': 'w51_background', '002': 'w51'},
                            # 2526 obs 021 = "G0" CMZ cloud-c filament F770W
                            '2526': {'021': 'cloudc'},
                            }[proposal_id]

    for field in fields:
        for filtername in filternames:
            print(f"Main Loop: {proposal_id} + {filtername} + {field}={field_to_reg_mapping[field]}")
            results = main(filtername=filtername, Observations=Observations, field=field,
                           regionname=field_to_reg_mapping[field],
                           proposal_id=proposal_id,
                           skip_step1and2=skip_step1and2,
                           reference_catalog=reference_catalog,
                           skip_download_for_existing=skip_download_for_existing,
                           marshall_tuning=marshall_tuning,
                          )


