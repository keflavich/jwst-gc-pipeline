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
os.environ["CRDS_PATH"] = "/orange/adamginsburg/jwst/crds/"
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

from jwst_gc_pipeline.reduction.destreak import destreak

from jwst_gc_pipeline.reduction.align_to_catalogs import merge_a_plus_b, retrieve_vvv
from jwst_gc_pipeline.reduction.saturated_star_finding import remove_saturated_stars

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


def _stamp_imaging_product(path):
    """Best-effort provenance sidecar for an imaging product (_i2d / _crf).

    Records the pipeline tag, stage='imaging', code hash, and the output facet
    hashes (data/wcs/meta) next to ``path``, with env (jwst version / CRDS
    context / DVACORR) auto-read from the product header.  FAIL-SOFT: imaging
    provenance must never break a reduction, so all failures are swallowed.
    """
    try:
        from jwst_gc_pipeline.versioning import stamping as _vstamp
    except ImportError:
        return
    _vstamp.try_stamp_product(path, 'imaging')


# see 'destreak410.ipynb' for tests of this
medfilt_size = {'F410M': 15, 'F405N': 256, 'F466N': 55,
                'F182M': 55, 'F187N': 512, 'F212N': 512,
                # added purely from guessing on 2026-06-07
                'F200W': 55, 'F335M': 55, 'F470N': 256, 'F480M': 256,
                'F356W': 55, 'F444W': 55,
                'F277W': 55, 'F300M': 55, 'F360M': 55,
                }

fov_regname = {'brick': 'regions_/nircam_brick_fov.reg',
               'cloudc': 'regions_/nircam_cloudc_fov.reg',
               'sickle': 'regions_/nircam_sickle_fov.reg',
               'sgrb2': 'regions_/nircam_sgrb2_fov.reg',
               'w51': 'regions_/nircam_w51_fov.reg',
               'cloudef': 'regions_/nircam_cloudef_fov.reg',
               'sgrc': 'regions_/nircam_sgrc_fov.reg',
               'arches': 'regions_/nircam_arches_fov.reg',
               'quintuplet': 'regions_/nircam_quintuplet_fov.reg',
               'sgra': 'regions_/nircam_sgra_fov.reg',
               'wd1': 'regions_/nircam_wd1_fov.reg',
               'wd2': 'regions_/nircam_wd2_fov.reg',
               }

refnames = {'2221': 'THIS_IS_A_BUG_IF_YOU_USE_THIS', # <--- we aren't using F405ref, which was the previous one, and I want hard-failures if this gets encountered
            # 1182 obs 004 = brick.  refnames[proposal_id] is used by
            # fix_alignment() to build the per-exposure offsets-table filename
            # offsets/Offsets_JWST_Brick1182_<refname>_average.csv.
            #
            # 2026-06-18: switched 'F200ref' -> 'VIRAC2' (UPSTREAM FOLD).  The old
            # F200ref_average table is F200W-derived (relative frame); applied to
            # F115W it left ~97 mas (bulk + per-detector wrong-filter distortion)
            # between the crf/_i2d/model/residual and the absolute (catalog/
            # realigned) frame.  Offsets_JWST_Brick1182_VIRAC2_average.csv is a
            # copy of F200ref_average with the F115W rows folded to the absolute
            # VIRAC2-2014.0 frame (build_virac2_fixalign_offsets.py); non-F115W
            # rows are still identical to F200ref (fold per-filter as processed).
            # So crf -> _i2d -> model -> residual -> catalog -> realigned now
            # share ONE frame and realign_to_catalog ~= 0.  See
            # ASTROMETRY_WCS_CORRECTION_FLOW.md.
            '1182': 'VIRAC2',
            # sickle. Repointed 2026-07-16 to the Gaia-DR3+VIRAC2 seed
            # (gaia_virac2_refcat_epoch2024.64.fits); token flipped GNS -> VIRAC2 so the
            # offsets-table name + realignment gate match the new frame.
            '3958': 'VIRAC2',
            # sgrb2: re-anchored 2026-06-18 to Gaia DR3 + VIRAC2
            # (gaia_virac2_refcat_epoch2024.68.fits).  Token left as 'VVV' after
            # that switch, so VVV realignment kept firing on a VIRAC2-frame field.
            '5365': 'VIRAC2',
            '6151': 'Gaia',  # w51: switched 2026-06-10 (was UKIDSS)
            # 2026-07-16: GC fields below repointed VVV/GNS -> VIRAC2-GaiaDR3 seed;
            # tokens flipped to 'VIRAC2' so refname (offsets-table name + realign gate) is
            # consistent with the new anchor. No Offsets_JWST_Brick{prop}_VIRAC2_average.csv
            # exists for these tweakreg fields (rashift falls through to 0), same as before.
            '2092': 'VIRAC2',
            # sgrc: re-anchored 2026-07-16 to Gaia DR3 + VIRAC2 (same GC
            # reference-frame policy as sgrb2/brick/cloudc).  Was 'VVV', which
            # (a) pointed tweakreg/realign at the raw VVV frame
            # (nircam_bootstrapped_to_vvv_refcat.fits, II/376/vvv4 -- no PM,
            # ~tens of mas off Gaia) and (b) armed the VVV realign gate on what
            # is now a VIRAC2-frame field.  'VIRAC2' turns that gate off and
            # names the per-exposure consensus offsets table fix_alignment reads.
            '4147': 'VIRAC2',
            '2045': 'VIRAC2',
            '1939': 'VIRAC2',
            '2211': 'VIRAC2',
            # Westerlund 1 (Guarcello prop 1905) + Westerlund 2 (Guarcello prop 3523).
            # Both clusters are outside the deep Galactic Center, so Gaia DR3 is
            # the correct astrometric reference (NOT GNS, NOT VVV).
            '1905': 'Gaia',
            '3523': 'Gaia',
            # M92 (halo GC) + M4/NGC6397 (Bedin): non-GC clusters, pure Gaia DR3
            # frame.  Tied per-(visit,filter) in fix_alignment (added 2026-07-11);
            # 'Gaia' here keeps the VVV-realign gate OFF (it only fires on 'VVV').
            '1334': 'Gaia',
            '1979': 'Gaia',
            }

# Reference catalog configuration by proposal and field.
# Paths are relative to basepath.
REFERENCE_ASTROMETRIC_CATALOG_BY_FIELD = {
    '2221': {
        # obs 001 = Brick. Re-anchored 2026-07-11 to the Gaia-tied seed (Gaia DR3 +
        # VIRAC2 fill, per-star PM-propagated to epoch 2022.70; obs epoch 2022.655,
        # 0.045 yr * 5.6 mas/yr = 0.25 mas -- negligible).  This retires the LAST
        # remaining crowdsource-F405N bootstrap entry (VVV-DR4/2MASS frame, ~90 mas
        # off Gaia) -- the frame that silently propagated into the NIRSpec 6927 MSA
        # plan v11 target list (measured TA(Gaia)-vs-plan mismatch (+47,+73) mas).
        # RELEASED CATALOGS MUST STATE THEIR FRAME + EPOCH (see stage_release.py).
        '001': 'catalogs/gaia_virac2_refcat_epoch2022.70.fits',
        # obs 002 = Cloud C. Re-anchored 2026-06-22 to the Gaia-tied seed (Gaia DR3 +
        # VIRAC2 fill, per-star PM-propagated to obs epoch 2023.30). Audit found the old
        # F405N crowdsource bootstrap (VVV-DR4/2MASS frame) was ~+22 mas RA / ~+90 mas Dec
        # off Gaia.  Built by build_gaia_virac2_refcat_byquery.py (ra=266.588 dec=-28.583).
        '002': 'catalogs/gaia_virac2_refcat_epoch2023.30.fits',
    },
    '1182': {
        # obs 004 = brick. Re-anchored 2026-06 to the Gaia-tied seed
        # (Gaia DR3 + VIRAC2 fill, epoch 2022.70). Replaces the F405N
        # self-bootstrap, which inherited the VVV-DR4 (2MASS) frame ~24 mas
        # off Gaia. VIRAC2 is Gaia-tied (~5 mas) with contiguous footprint.
        '004': 'catalogs/gaia_virac2_refcat_epoch2022.70.fits',
        # obs 002 = w51 short-wave coverage; use Gaia (W51 is in the disk)
        '002': 'catalogs/gaia_refcat.fits',
    },
    '3958': {
        # sickle (GC field). Repointed 2026-07-16 GNS-bootstrap (2MASS-tied) -> Gaia-DR3+
        # VIRAC2 seed, PM-propagated to obs epoch 2024.64. NOTE: sickle is MIRI+NIRCam;
        # VIRAC2 is NIR so the NIRCam tie is clean; verify MIRI-only bands separately.
        '007': 'catalogs/gaia_virac2_refcat_epoch2024.64.fits',
    },
    '5365': {
        # Sgr B2 (GC field). Switched 2026-06-18 from the VVV-tied crowdsource F405N
        # bootstrap (crowdsource deprecated) to the GC reference-frame policy: Gaia DR3
        # + VIRAC2 NIR fill, per-star PM-propagated (VIRAC2 from 2014.0, Gaia from 2016.0)
        # to the obs epoch 2024.685. Built by build_gaia_virac2_refcat_byquery.py.
        '001': 'catalogs/gaia_virac2_refcat_epoch2024.68.fits',
    },
    '6151': {
        # w51 main NIRCam pointing -- Gaia DR3 (disk field, not GC)
        '001': 'catalogs/gaia_refcat.fits',
    },
    '2092': {
        # cloudef 002 (Cloud E) + 005 (Cloud F). Repointed 2026-07-16 VVV-bootstrap
        # (2MASS-tied, measured ~80-160 mas off Gaia) -> Gaia-DR3+VIRAC2 seed, PM-propagated
        # to obs epoch 2023.21. VIRAC2 covers these (outside GNS but inside VIRAC footprint).
        '002': 'catalogs/gaia_virac2_refcat_epoch2023.21.fits',
        '005': 'catalogs/gaia_virac2_refcat_epoch2023.21.fits',
    },
    '4147': {
        # obs 012 = Sgr C.  Re-anchored 2026-07-16 to the Gaia-tied seed (Gaia
        # DR3 + VIRAC2 fill, PM-propagated to obs epoch 2023.725).  Replaces the
        # F212N self-bootstrap tied to raw VVV (II/376/vvv4, no PM, ~tens of mas
        # off Gaia) -- Sgr C is inside the VIRAC2 bulge footprint (same region as
        # sgrb2), so it follows the GC policy.  Built by
        # build_gaia_virac2_refcat_byquery.py (ra=266.171 dec=-29.442).
        '012': 'catalogs/gaia_virac2_refcat_epoch2023.72.fits',
    },
    '2045': {
        # arches (001) + quintuplet (003). Repointed 2026-07-16 GNS-bootstrap (2MASS-tied)
        # -> Gaia-DR3+VIRAC2 seed. VIRAC2 (deep NIR, Gaia-tied) IS usable in the inner GC
        # (the "Gaia unusable" note applied to Gaia-alone; VIRAC2 supplies the density).
        # Different obs epochs: arches 2023.64, quintuplet 2024.62.
        '001': 'catalogs/gaia_virac2_refcat_epoch2023.64.fits',
        '003': 'catalogs/gaia_virac2_refcat_epoch2024.62.fits',
    },
    '1939': {
        # sgra: inner GC. Repointed 2026-07-16 GNS-bootstrap -> Gaia-DR3+VIRAC2 seed
        # (epoch 2022.72). VIRAC2 supplies density where Gaia-alone is too sparse.
        '001': 'catalogs/gaia_virac2_refcat_epoch2022.72.fits',
    },
    '2211': {
        # gc2211. Repointed 2026-07-16 GNS-bootstrap -> Gaia-DR3+VIRAC2 seed (epoch 2023.71).
        # The wide-band saturation that made Gaia-alone unusable is handled by VIRAC2's deep
        # NIR density (Ks, Gaia-tied) -- same depth argument as GNS but on the Gaia frame.
        '023': 'catalogs/gaia_virac2_refcat_epoch2023.71.fits',
        '046': 'catalogs/gaia_virac2_refcat_epoch2023.71.fits',
        '049': 'catalogs/gaia_virac2_refcat_epoch2023.71.fits',
        '050': 'catalogs/gaia_virac2_refcat_epoch2023.71.fits',
        # obs 028 (F150W, different/offset FOV) is SOUTH of the main gc2211 seed footprint
        # (obs028 Dec -29.19..-29.10 vs main seed -29.05..-28.81), so it needs its OWN seed.
        # Dedicated Gaia-DR3+VIRAC2 seed at the obs028 center (266.365,-29.144, r=0.10).
        '028': 'catalogs/gaia_virac2_refcat_epoch2023.71_o028.fits',
    },
    # Westerlund 1 / Westerlund 2 main pointings: Gaia DR3 is the
    # astrometric reference (outside GC, no GNS/VVV needed).
    '1905': {
        '001': 'catalogs/gaia_refcat.fits',
    },
    '3523': {
        '005': 'catalogs/gaia_refcat.fits',
    },
    # --- Globular clusters (Jay Anderson co-I programs; added 2026-06-30) ---
    # Non-GC clusters outside the VIRAC2/VVV footprint -> pure Gaia DR3 frame
    # (PM-propagated to obs epoch). Built by build_gc_gaia_refcat.py.
    '1334': {  # Weisz ERS; M92 (halo GC)
        '001': 'catalogs/gaia_refcat.fits',
    },
    '1979': {  # Bedin; NGC6397 (o001), M-4 (o002), M-4-shift (o003)
        '001': 'catalogs/gaia_refcat.fits',
        '002': 'catalogs/gaia_refcat.fits',
        '003': 'catalogs/gaia_refcat.fits',
    },
    '8322': {  # Haeberle oMEGACat; Omega Cen (NGC5139)
        '001': 'catalogs/gaia_refcat.fits',
    },
    '12587': {  # Haeberle oMEGACat; Omega Cen (NGC5139)
        '001': 'catalogs/gaia_refcat.fits',
    },
    # --- NGC 6334 (Cat's Paw SFR; extended emission; galactic plane l=351) ---
    # Gaia DR3 + VIRAC2 fill (VVV bulge footprint), PM-propagated to obs epoch.
    '7213': {  # Cheng; NGC6334_I_N (proprietary), epoch 2026.30
        '001': 'catalogs/gaia_virac2_refcat_epoch2026.30.fits',
    },
    '6778': {  # Garcia Marin; SF_reg_1, epoch 2024.68
        '001': 'catalogs/gaia_virac2_refcat_epoch2024.68.fits',
    },
}

# Module restrictions per proposal/field/filter for single-module datasets
# Sickle is NRCB-only (SUB640 subarray) but detectors differ by wavelength:
# - Short-wavelength (F187N, F210M): nrcb1, nrcb2, nrcb3, nrcb4
# - Long-wavelength (F335M, F470N, F480M): nrcb only
MODULES_BY_PROPOSAL_FIELD_FILTER = {
    '3958': {
        '007': {
            'F187N': ('nrcb1', 'nrcb2', 'nrcb3', 'nrcb4'),
            'F210M': ('nrcb1', 'nrcb2', 'nrcb3', 'nrcb4'),
            'F335M': ('nrcb',),
            'F470N': ('nrcb',),
            'F480M': ('nrcb',),
        }
    }
}


# Per-filter overrides for the (proposal_id, field) refcat lookup.  Lets
# us hand a different refcat to one specific filter (e.g. brick-1182
# F115W tweakreg, where the F405N-based refcat has poor blue-star
# overlap).  Lookup precedence is filter > field default.
REFERENCE_ASTROMETRIC_CATALOG_BY_FILTER = {
    '1182': {
        '004': {
            # F115W (bluest/sharpest SW) anchored directly to the Gaia-tied seed
            # (Gaia DR3 + VIRAC2). Was the F200W self-bootstrap (VVV-DR4 frame).
            'F115W': 'catalogs/gaia_virac2_refcat_epoch2022.70.fits',
        },
    },
}


def get_reference_astrometric_catalog_path(basepath, proposal_id, field, filtername=None):
    if filtername is not None:
        override = (REFERENCE_ASTROMETRIC_CATALOG_BY_FILTER
                    .get(proposal_id, {}).get(field, {}).get(filtername.upper()))
        if override is not None:
            return f'{basepath}/{override}'
    if proposal_id not in REFERENCE_ASTROMETRIC_CATALOG_BY_FIELD:
        raise KeyError(f"No reference catalog mapping configured for proposal_id={proposal_id}")
    if field not in REFERENCE_ASTROMETRIC_CATALOG_BY_FIELD[proposal_id]:
        raise KeyError(f"No reference catalog mapping configured for proposal_id={proposal_id} field={field}")
    relpath = REFERENCE_ASTROMETRIC_CATALOG_BY_FIELD[proposal_id][field]
    return f'{basepath}/{relpath}'


def get_existing_reference_astrometric_catalog_path(basepath, proposal_id, field, filtername=None):
    if filtername is not None:
        override = (REFERENCE_ASTROMETRIC_CATALOG_BY_FILTER
                    .get(proposal_id, {}).get(field, {}).get(filtername.upper()))
        if override is not None:
            path = f'{basepath}/{override}'
            if os.path.exists(path):
                return path
    if proposal_id not in REFERENCE_ASTROMETRIC_CATALOG_BY_FIELD:
        return None
    if field not in REFERENCE_ASTROMETRIC_CATALOG_BY_FIELD[proposal_id]:
        return None
    path = f"{basepath}/{REFERENCE_ASTROMETRIC_CATALOG_BY_FIELD[proposal_id][field]}"
    if os.path.exists(path):
        return path
    # FAIL LOUD: this field IS wired to a specific reference catalog but the file is
    # missing.  Silently returning None here makes Image3 run first-pass WITHOUT
    # abs_refcat realignment -> the field ships OFF-FRAME with no error (the exact
    # quiet-off-frame-release failure the VIRAC2 repoint is meant to prevent).  The
    # gaia_virac2_refcat_epoch<obs> seeds live outside the repo, so a typo/missing build
    # must abort, not degrade.
    raise FileNotFoundError(
        f"Configured reference astrometric catalog is MISSING: {path} "
        f"(proposal_id={proposal_id} field={field}). Build the seed "
        f"(build_gaia_virac2_refcat_byquery.py) or fix the wiring in "
        f"REFERENCE_ASTROMETRIC_CATALOG_BY_FIELD before reducing -- refusing to run "
        f"first-pass off-frame.")


def _module_group(module):
    if module == 'merged':
        return 'merged'
    if module.startswith('nrca'):
        return 'nrca'
    if module.startswith('nrcb'):
        return 'nrcb'
    return module


def get_allowed_modules(proposal_id, field, requested_modules, filtername=None):
    allowed_modules = None
    
    # Check for filter-specific policy first
    if proposal_id in MODULES_BY_PROPOSAL_FIELD_FILTER:
        if field in MODULES_BY_PROPOSAL_FIELD_FILTER[proposal_id]:
            field_policy = MODULES_BY_PROPOSAL_FIELD_FILTER[proposal_id][field]
            if filtername and filtername in field_policy:
                allowed_modules = field_policy[filtername]
    
    if allowed_modules is None:
        return requested_modules

    requested_groups = {_module_group(module) for module in requested_modules}
    filtered_modules = [module for module in allowed_modules if _module_group(module) in requested_groups]
    if len(filtered_modules) == 0:
        raise ValueError(
            f"No requested modules are allowed for proposal_id={proposal_id} field={field} "
            f"filtername={filtername}. "
            f"Requested modules={requested_modules}, allowed modules={allowed_modules}"
        )
    if tuple(filtered_modules) != tuple(requested_modules):
        print(
            f"Restricting modules for proposal_id={proposal_id} field={field} filtername={filtername} "
            f"to {filtered_modules} because this dataset is explicitly single-module."
        )
    return filtered_modules

# it's very difficult to modify the Webb pipeline in this way
# # replace Image2Pipeline's 'resample' with one that uses our hand-corrected coordinates
# def pre_resample(func):
#   def wrapper(self, input, *args, **kwargs):
#     print("Before resample is called, fixing coordinates")
#     for member in inputs:
#         print(f"Fixing alignment for {member.meta.filename}")
#         fix_alignment(member.meta.filename)
#     result = func(*args, **kwargs)
#     return result
#   return wrapper
#
# Image2Pipeline.step_defs['resample'] = pre_resample(Image2Pipeline.resample)


def main(filtername, module, Observations=None, regionname='brick', do_destreak=True,
         field='001', proposal_id='2221', skip_step1and2=False, use_average=True,
         skymatch_method=None):
    """
    skip_step1and2 will not re-fit the ramps to produce the _cal images.  This
    can save time if you just want to redo the tweakreg steps but already have
    the zero-frame stuff done.
    """
    print(f"Processing filter {filtername} module {module} with do_destreak={do_destreak} and skip_step1and2={skip_step1and2} for field {field} and proposal id {proposal_id} in region {regionname}")

    # ------------------------------------------------------------------
    # Field-dependent destreak policy.
    #
    # The destreak step subtracts a per-row percentile.  With
    # use_background_map=True it does NOT add the smoothed large scales
    # back (add_smoothed = not use_background_map); those large scales are
    # instead supposed to be restored by add_background_map().  If no
    # background map exists for the field (background_mapping in
    # destreak.py currently only has the Brick, '2221'), the large-scale
    # flux is simply removed -- and because each dither places bright
    # extended emission on different detector rows, it is removed
    # *inconsistently between frames*.  outlier_detection then sees the
    # frames disagree at the same sky position and rejects the bright
    # pixels, producing coverage holes + flux jumps in nebulosity-
    # dominated fields (confirmed on W51 F335M: cal frames -> ~4% flagged,
    # destreaked frames -> ~26% flagged, same jwst/version).
    #
    # Policy:
    #  - Nebulosity-dominated fields -> destreak OFF.  There is no
    #    background map to add the emission back, so destreaking corrupts
    #    it.  (W51, Sickle.)
    #  - Star-dominated fields -> destreak is OK, BUT we should still build
    #    a background map (or rely on add_smoothed=True, the streak-removal
    #    mode that adds the smoothed large scales back) so large angular
    #    scales are restored rather than lost.  TODO: audit each remaining
    #    field below and confirm it is star-dominated before trusting
    #    destreak there.
    #
    # TODO: build a proper extended-emission background map for Sickle,
    # WD2, and W51 and register them in background_mapping (destreak.py).
    # Once tested, destreak can be re-enabled for those fields.  Until
    # then they run with do_destreak=False.
    # ------------------------------------------------------------------
    EXTENDED_EMISSION_FIELDS = ('w51', 'sickle', 'wd2', 'ngc6334')
    if do_destreak and regionname in EXTENDED_EMISSION_FIELDS:
        print(f"Region {regionname} is extended-emission-dominated and has no "
              f"background map yet; forcing do_destreak=False to avoid "
              f"outlier_detection coverage holes.", flush=True)
        do_destreak = False

    # Sickle policy (user 2026-06-17, [[sickle...]]): the SHORT-wave filters USE
    # destreak ("the streaks are worse than the destreak artifacts" for SW),
    # while the LONG-wave filters stay nodestreak/align.  This overrides the
    # extended-emission force above PER FILTER for sickle so SW -> *_destreak_crf
    # and LW -> *_align_crf (the suffixes the cataloging --each-suffix consumes).
    if regionname == 'sickle':
        _sw_destreak = filtername.upper() in (
            'F070W', 'F090W', 'F115W', 'F140M', 'F150W', 'F162M', 'F164N',
            'F182M', 'F187N', 'F200W', 'F210M', 'F212N')
        do_destreak = _sw_destreak
        print(f"Sickle per-filter destreak policy: {filtername} -> "
              f"do_destreak={do_destreak} "
              f"({'SW=destreak' if _sw_destreak else 'LW=align/nodestreak'})",
              flush=True)

    wavelength = int(filtername[1:4])

    basepath = f'/orange/adamginsburg/jwst/{regionname}/'
    fwhm_tbl = Table.read(f'{basepath}/reduction/fwhm_table.ecsv')
    row = fwhm_tbl[fwhm_tbl['Filter'] == filtername]
    if module == 'merged':
        expected_modules = ('merged',)
        do_merge = False
    else:
        expected_modules = get_allowed_modules(proposal_id, field, ('nrca', 'nrcb'), filtername=filtername)
        do_merge = 'nrca' in expected_modules and 'nrcb' in expected_modules
    fwhm = fwhm_arcsec = float(row['PSF FWHM (arcsec)'][0])
    fwhm_pix = float(row['PSF FWHM (pixel)'][0])

    destreak_suffix = '' if do_destreak else '_nodestreak'

    # sanity check
    if regionname == 'brick':
        if proposal_id == '2221':
            assert field == '001'
    if regionname == 'sgrb2':
        if proposal_id == '5365':
            assert field == '001'
    if regionname == 'w51':
        if proposal_id == '6151':
            assert field == '001'
        elif proposal_id == '1182':
            assert field == '004'
    elif regionname == 'cloudc':
        assert field == '002'
    elif regionname == 'sickle':
        if proposal_id == '3958':
            assert field == '007'
    elif regionname == 'arches':
        if proposal_id == '2045':
            assert field == '001'
    elif regionname == 'quintuplet':
        if proposal_id == '2045':
            assert field == '003'
    elif regionname == 'sgra':
        if proposal_id == '1939':
            assert field == '001'
    elif regionname == 'wd1':
        if proposal_id == '1905':
            assert field in ('001', '003')
    elif regionname == 'wd2':
        if proposal_id == '3523':
            assert field in ('003', '005')

    if "CRDS_PATH" not in os.environ:
        os.environ["CRDS_PATH"] = f"{basepath}/crds/"
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
        except Exception as ex:
            print(f'Failed to link {fn} to {os.path.basename(fn)} because of {ex}')

    Observations.cache_location = output_dir

    # The MAST block below exists only to fetch the image3 asn jsons (globbed
    # back off disk further down) and, when not skip_step1and2, the uncal files.
    # `Observations.get_product_list` has no effective timeout and was observed
    # to hang forever in ssl.read on flaky compute-node networks (22h hang, job
    # 35773493, 2026-06-26).  Skip the network entirely when the files we need
    # are already on disk; otherwise set a hard TIMEOUT so a stalled MAST
    # connection fails fast instead of hanging the whole pipeline.
    existing_asn = glob(os.path.join(output_dir, f'jw0{proposal_id}-o{field}*_image3_*0[0-9][0-9]_asn.json'))
    existing_uncal = glob(os.path.join(output_dir, f'jw0{proposal_id}{field}*_uncal.fits'))
    mast_needed = (len(existing_asn) == 0) or (not skip_step1and2 and len(existing_uncal) == 0)
    if not mast_needed:
        print(f"Skipping MAST query for {filtername}: {len(existing_asn)} asn json(s) and "
              f"{len(existing_uncal)} uncal already on disk (skip_step1and2={skip_step1and2})")
    else:
        Observations.TIMEOUT = 120  # seconds; avoid indefinite hang on a stalled MAST connection
        obs_table = Observations.query_criteria(
                                                proposal_id=proposal_id,
                                                #proposal_pi="Ginsburg*",
                                                #calib_level=3,
                                                )
        print("Obs table length:", len(obs_table))

        # np.array wrapper needed as of 2026-04-10 to avoid masked array type error that shouldn't happen
        msk = ((np.char.find(np.array(obs_table['filters']), filtername.upper()) >= 0) |
               (np.char.find(np.array(obs_table['obs_id']), filtername.lower()) >= 0))
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
        for row in manifest:
            try:
                shutil.move(row['Local Path'], os.path.join(output_dir, os.path.basename(row['Local Path'])))
            except Exception as ex:
                print(f"Failed to move file with error {ex}")

        products_fits = Observations.filter_products(data_products_by_obs, extension="fits")
        print("products_fits length:", len(products_fits))
        uncal_mask = np.array([
            uri.endswith('_uncal.fits')
            and f'jw0{proposal_id}{field}' in uri
            and ('_nrc' in uri)
            for uri in products_fits['dataURI']
        ])
        uncal_mask &= products_fits['productType'] == 'SCIENCE'
        print("uncal length:", (uncal_mask.sum()))

        already_downloaded = np.array([os.path.exists(os.path.basename(uri)) for uri in products_fits['dataURI']])
        uncal_mask &= ~already_downloaded
        print(f"uncal to download: {uncal_mask.sum()}; {already_downloaded.sum()} were already downloaded")

        if uncal_mask.any():
            manifest = Observations.download_products(products_fits[uncal_mask], download_dir=output_dir)
            print("manifest:", manifest)

            # MAST creates deep directory structures we don't want
            for row in manifest:
                try:
                    shutil.move(row['Local Path'], os.path.join(output_dir, os.path.basename(row['Local Path'])))
                except Exception as ex:
                    print(f"Failed to move file with error {ex}")


    # all cases, except if you're just doing a merger?
    if module in ('nrca', 'nrcb', 'merged'):
        print(f"Working on module {module}: running initial pipeline setup steps (skip_step1and2={skip_step1and2})")
        print(f"Searching for {os.path.join(output_dir, f'jw0{proposal_id}-o{field}*_image3_*0[0-9][0-9]_asn.json')}")
        asn_file_search = glob(os.path.join(output_dir, f'jw0{proposal_id}-o{field}*_image3_*0[0-9][0-9]_asn.json'))
        # Filter out non-NIRCam asn files (e.g. NIRISS asns from same proposal/obs).
        # Members of NIRCam asns have 'nrc' in expname (nrca/nrcb/nrcalong/nrcblong); NIRISS members are '_nis_'.
        nircam_asn_files = []
        for candidate in asn_file_search:
            try:
                with open(candidate) as fh:
                    cand_data = json.load(fh)
                members = cand_data.get('products', [{}])[0].get('members', [])
                if members and any('nrc' in m.get('expname', '') for m in members):
                    nircam_asn_files.append(candidate)
            except (json.JSONDecodeError, KeyError, IndexError) as exc:
                raise ValueError(f"Could not parse asn candidate {candidate}: {exc}")
        if len(asn_file_search) > 0 and len(nircam_asn_files) != len(asn_file_search):
            skipped = set(asn_file_search) - set(nircam_asn_files)
            print(f"Filtered out non-NIRCam asn files: {sorted(skipped)}")
        asn_file_search = nircam_asn_files
        # Disambiguate by effective bandpass.  MAST names two-element products
        # alphabetically as 'nircam_{a}-{b}' (e.g. clear-f444w, f405n-f444w,
        # f444w-f470n), so a substring match on the filter name pulls in the
        # narrowband products that merely use this filter as a blocker.  Keep
        # only the asn whose effective band (pupil narrow/medium band if present,
        # else the wide filter) equals the filter we are reducing.  Additive:
        # fall back to the legacy sorted()[-1] pick if this leaves 0 or >1.
        def _asn_effective_band(asn_path):
            try:
                with open(asn_path) as _fh:
                    prod = json.load(_fh)['products'][0]['name']
            except (json.JSONDecodeError, KeyError, IndexError, OSError):
                return None
            tail = prod.split('nircam_')[-1]
            toks = [t.upper() for t in tail.split('-') if t]
            ftoks = [t for t in toks if t.startswith('F') and t[1:2].isdigit()]
            if len(ftoks) == 1:
                return ftoks[0]
            narrow = [t for t in ftoks if t.endswith('N') or t.endswith('M')]
            if len(narrow) == 1:
                return narrow[0]
            return None
        band_match = [a for a in asn_file_search
                      if _asn_effective_band(a) == filtername.upper()]
        if len(band_match) == 1:
            asn_file_search = band_match
            print(f"Selected asn by effective band {filtername}: {band_match[0]}")
        if len(asn_file_search) == 1:
            asn_file = asn_file_search[0]
        elif len(asn_file_search) > 1:
            asn_file = sorted(asn_file_search)[-1]
            print(f"Found multiple asn files: {asn_file_search}.  Using the more recent one, {asn_file}.")
        else:
            raise ValueError(f"Mismatch: Did not find any NIRCam asn files for module {module} for field {field} in {output_dir}")

        crds_dir = os.getenv("CRDS_PATH") or f'/orange/adamginsburg/jwst/{regionname}/crds'
        mapping = crds.rmap.load_mapping(f'{crds_dir}/mappings/jwst/jwst_nircam_pars-tweakregstep_0003.rmap')
        print(f"Mapping: {mapping.todict()['selections']}")
        print(f"Filtername: {filtername}")
        filter_match = [x for x in mapping.todict()['selections'] if filtername in x]
        print(f"Filter_match: {filter_match} n={len(filter_match)}")
        tweakreg_asdf_filename = filter_match[0][4]
        tweakreg_asdf = asdf.open(f'https://jwst-crds.stsci.edu/unchecked_get/references/jwst/{tweakreg_asdf_filename}')
        tweakreg_parameters = tweakreg_asdf.tree['parameters']
        tweakreg_parameters.update({'skip': True,
                                    'fitgeometry': 'general',
                                    # brightest = 5000 was causing problems- maybe the cross-alignment was getting caught on PSF artifacts?
                                    'brightest': 5000,
                                    'snr_threshold': 20, # was 5, but that produced too many stars
                                    # define later 'abs_refcat': abs_refcat,
                                    'save_catalogs': True,
                                    'catalog_format': 'fits',
                                    'kernel_fwhm': fwhm_pix,
                                    'nclip': 5,
                                    'starfinder': 'dao',
                                    # expand_refcat: A boolean indicating whether or not to expand reference catalog with new sources from other input images that have been already aligned to the reference image. (Default=False)
                                    'expand_refcat': True,
                                    # based on DebugReproduceTweakregStep
                                    'sharplo': 0.3,
                                    'sharphi': 0.9,
                                    'roundlo': -0.25,
                                    'roundhi': 0.25,
                                    'separation': 0.5, # minimum separation; default is 1
                                    'tolerance': 0.1, # tolerance: Matching tolerance for xyxymatch in arcsec. (Default=0.7)
                                    'save_results': True,
                                    # 'clip_accum': True, # https://github.com/spacetelescope/tweakwcs/pull/169/files
                                    })

        print(f'Filter {filtername} tweakreg parameters: {tweakreg_parameters}')

        with open(asn_file) as f_obj:
            asn_data = json.load(f_obj)

        print(f"In cwd={os.getcwd()}")
        if not skip_step1and2:
            # re-calibrate all uncal files -> cal files *without* suppressing first group
            for member in asn_data['products'][0]['members']:
                if '_nrc' not in member['expname']:
                    print(f"Skipping non-NIRCam member {member['expname']}")
                    continue
                # example filename: jw02221002001_02201_00002_nrcalong_cal.fits
                assert f'jw0{proposal_id}{field}' in member['expname']
                print(f"DETECTOR PIPELINE on {member['expname']}")
                print("Detector1Pipeline step")
                # from Hosek: expand_large_events -> false; turn off "snowball" detection
                Detector1Pipeline.call(member['expname'].replace("_cal.fits",
                                                                 "_uncal.fits"),
                                       save_results=True, output_dir=output_dir,
                                       save_calibrated_ramp=True,
                                       steps={'ramp_fit': {'suppress_one_group':False, 'save_results':True},
                                              "refpix": {"use_side_ref_pixels": True},
                                              "jump":{"save_results":True}})

                # apparently "rate" files have no WCS, but this is where it's needed...
                # print("Aligning RATE images before doing IMAGE2 pipeline")
                # for member in asn_data['products'][0]['members']:
                #     align_image = member['expname'].replace("_cal.fits", "_rate.fits")
                #     fix_alignment(align_image, proposal_id=proposal_id, module=module, field=field, basepath=basepath, filtername=filtername)
                #else:
                #    print(f"Field {field} proposal {proposal_id} did not require re-alignment")
                print(f"IMAGE2 PIPELINE on {member['expname']}")
                Image2Pipeline.call(member['expname'].replace("_cal.fits",
                                                              "_rate.fits"),
                                    save_results=True, output_dir=output_dir,
                                   )
        else:
            print("Skipped step 1 and step2")

        # don't need to do this / it affects Savannah's fixing approach
        #print("Doing pre-alignment from offsets tables")
        #for member in asn_data['products'][0]['members']:
        #    if (field == '004' and proposal_id == '1182') or ((field == '001' or field  == '002') and proposal_id == '2221'):
        #        for suffix in ("_cal.fits", "_destreak.fits"):
        #            align_image = member['expname'].replace("_cal.fits", suffix)
        #            fix_alignment(align_image, proposal_id=proposal_id, module=module, field=field, basepath=basepath, filtername=filtername)
        #    else:
        #        print(f"Field {field} proposal {proposal_id} did not require re-alignment")


    else:
        raise ValueError(f"Module is {module} - not allowed!")

    if module in ('nrca', 'nrcb'):
        print(f"Filter {filtername} module {module}: doing tweakreg.  do_destreak={do_destreak}")

        with open(asn_file) as f_obj:
            asn_data = json.load(f_obj)
        asn_data['products'][0]['name'] = f'jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername.lower()}-{module}'
        asn_data['products'][0]['members'] = [row for row in asn_data['products'][0]['members']
                                                if f'{module}' in row['expname']]

        if len(asn_data['products'][0]['members']) == 0:
            raise ValueError(
                f"No {module} members found in {asn_file} for filter {filtername} field {field} proposal {proposal_id}. "
                f"This is not a valid pipeline state because the module output cannot be produced."
            )

        for member in asn_data['products'][0]['members']:
            print(f"Running destreak={do_destreak} and maybe alignment on {member} for module={module}")
            hdr = fits.getheader(member['expname'])
            if do_destreak:
                if filtername in (hdr['PUPIL'], hdr['FILTER']):
                    outname = destreak(member['expname'],
                                    use_background_map=True,
                                    median_filter_size=2048)  # median_filter_size=medfilt_size[filtername])
                    member['expname'] = outname
                    fix_alignment(outname, proposal_id=proposal_id,
                                module=module, field=field,
                                basepath=basepath, filtername=filtername,
                                use_average=use_average)
            else: # make align files
                fname = member['expname']
                assert fname.endswith('_cal.fits')
                member['expname'] = fname.replace("_cal.fits", "_align.fits")
                # copyfile (not copy) skips chmod, avoiding PermissionError when
                # a previous run by another user owns the destination file in a
                # group-writable shared workspace (e.g. W51 with t.yoo files).
                shutil.copyfile(fname, member['expname'])

                fix_alignment(member['expname'], proposal_id=proposal_id,
                              module=module, field=field, basepath=basepath,
                              filtername=filtername, use_average=use_average)

        asn_file_each = asn_file.replace("_asn.json", f"_{module}_asn.json")
        with open(asn_file_each, 'w') as fh:
            json.dump(asn_data, fh)

        # don't use VVV at all; the catalog does not play nicely with JWST pipe catalogs
        if False: #filtername.lower() == 'f405n':
            # for the VVV cat, use the merged version: no need for independent versions
            abs_refcat = vvvdr2fn = (f'{basepath}/{filtername.upper()}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername}-merged_vvvcat.ecsv')
            print(f"Loaded VVV catalog {vvvdr2fn}")
            retrieve_vvv(basepath=basepath, filtername=filtername, proposal_id=proposal_id, fov_regname=fov_regname[regionname], module='merged', fieldnumber=field)
            tweakreg_parameters['abs_refcat'] = vvvdr2fn
            tweakreg_parameters['abs_searchrad'] = 1
            reftbl = Table.read(abs_refcat)
            reftbl.meta['name'] = f'VVV Reference Catalog {filtername}'
            assert 'skycoord' in reftbl.colnames
        else:
            # Use the existence-checked getter: it returns None only when the field is
            # NOT configured for an absolute refcat (legitimate skip), and RAISES
            # FileNotFoundError when the field IS wired to a refcat whose file is missing.
            # That prevents this merged-mosaic path from silently producing an OFF-FRAME
            # _i2d (the release deliverable) on a typo'd / not-yet-built seed.
            abs_refcat = get_existing_reference_astrometric_catalog_path(basepath, proposal_id, field, filtername=filtername)
            if abs_refcat is not None:
                reftbl = Table.read(abs_refcat)
                reftblversion = reftbl.meta['VERSION']
                reftbl.meta['name'] = 'Reference Astrometric Catalog'
                reftbl.meta['filename'] = abs_refcat
            else:
                print(f"No absolute reference catalog configured for proposal_id="
                      f"{proposal_id} field={field}; skipping refcat realignment (mosaic "
                      f"still produced; absolute zero point unset).", flush=True)
                reftbl = None
                reftblversion = None

            # truncate to top 10,000 sources
            # more recent versions are already truncated to only very high quality matches
            # reftbl[:10000].write(f'{basepath}/catalogs/crowdsource_based_nircam-f405n_reference_astrometric_catalog_truncated10000.ecsv', overwrite=True)
            # abs_refcat = f'{basepath}/catalogs/crowdsource_based_nircam-f405n_reference_astrometric_catalog_truncated10000.ecsv'

            tweakreg_parameters['abs_searchrad'] = 0.4
            # try forcing searchrad to be tighter to avoid bad crossmatches
            # (the raw data are very well-aligned to begin with, though CARTA
            # can't display them b/c they are using SIP)
            tweakreg_parameters['searchrad'] = 0.05
            print(f"Reference catalog is {abs_refcat} with version {reftblversion}")

        tweakreg_parameters.update({'abs_refcat': abs_refcat})
        tweakreg_parameters.update({'skip': True})

        if regionname in ('brick', 'cloudc'):
            # for the VVV cat, use the merged version: no need for independent versions
            abs_refcat = vvvdr2fn = (f'{basepath}/{filtername.upper()}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername}-merged_vvvcat.ecsv')
            print(f"Loaded VVV catalog {vvvdr2fn}")
            retrieve_vvv(basepath=basepath, filtername=filtername, proposal_id=proposal_id, fov_regname=fov_regname[regionname], module='merged', fieldnumber=field)
            tweakreg_parameters['abs_refcat'] = vvvdr2fn
            tweakreg_parameters['abs_searchrad'] = 1
            reftbl = Table.read(abs_refcat)
            reftbl.meta['name'] = f'VVV Reference Catalog {filtername}'
            assert 'skycoord' in reftbl.colnames
            # Use the existence-checked getter: it returns None only when the field is
            # NOT configured for an absolute refcat (legitimate skip), and RAISES
            # FileNotFoundError when the field IS wired to a refcat whose file is missing.
            # That prevents this merged-mosaic path from silently producing an OFF-FRAME
            # _i2d (the release deliverable) on a typo'd / not-yet-built seed.
            abs_refcat = get_existing_reference_astrometric_catalog_path(basepath, proposal_id, field, filtername=filtername)
            if abs_refcat is not None:
                reftbl = Table.read(abs_refcat)
                reftblversion = reftbl.meta['VERSION']
                reftbl.meta['name'] = 'Reference Astrometric Catalog'
                reftbl.meta['filename'] = abs_refcat
            else:
                print(f"No absolute reference catalog configured for proposal_id="
                      f"{proposal_id} field={field}; skipping refcat realignment (mosaic "
                      f"still produced; absolute zero point unset).", flush=True)
                reftbl = None
                reftblversion = None

            # truncate to top 10,000 sources
            # more recent versions are already truncated to only very high quality matches
            # reftbl[:10000].write(f'{basepath}/catalogs/crowdsource_based_nircam-f405n_reference_astrometric_catalog_truncated10000.ecsv', overwrite=True)
            # abs_refcat = f'{basepath}/catalogs/crowdsource_based_nircam-f405n_reference_astrometric_catalog_truncated10000.ecsv'

            tweakreg_parameters['abs_searchrad'] = 0.4
            # try forcing searchrad to be tighter to avoid bad crossmatches
            # (the raw data are very well-aligned to begin with, though CARTA
            # can't display them b/c they are using SIP)
            tweakreg_parameters['searchrad'] = 0.05
            print(f"Reference catalog is {abs_refcat} with version {reftblversion}")
            tweakreg_parameters.update({'abs_refcat': abs_refcat})
            tweakreg_parameters.update({'skip': True})

        # skymatch: OFF by default (skymatch_method=None) -- historically left
        # skipped here because a global subtraction can eat real GC diffuse
        # emission.  Opt-in via --skymatch-method=match to remove the per-exposure
        # background pedestals that otherwise leave visible seams/stripes in the
        # mosaic (sickle F470N: per-exposure medians spanned -27..+72, range ~99,
        # never level-matched).  'match' equalizes RELATIVE inter-frame offsets
        # using overlap regions only (it does NOT subtract a global sky), so the
        # common diffuse structure is preserved; match_down=False matches up.
        # subtract=True is essential (else the matched levels are only recorded,
        # not applied -- see PipelineMIRI.py).
        image3_steps = {'tweakreg': tweakreg_parameters}
        if skymatch_method:
            image3_steps['skymatch'] = {'save_results': True,
                                        'subtract': True,
                                        'skymethod': skymatch_method,
                                        'match_down': False}
            print(f"Running skymatch skymethod={skymatch_method} subtract=True ({module})")
        print(f"Running tweakreg ({module})")
        calwebb_image3.Image3Pipeline.call(
            asn_file_each,
            steps=image3_steps,
            output_dir=output_dir,
            save_results=True)
        print(f"DONE running {asn_file_each}")

        # CRF NAMING FIX (port of PipelineMIRI 2026-06-20): outlier_detection in
        # Image3 names the CR-flagged crfs after the asn PRODUCT
        #   jw0{prop}-o{field}_t001_nircam_clear-{filt}-{module}_<N>_o{field}_crf.fits
        # but the manual cataloging globs PER-EXPOSURE crf with the destreak/align
        # suffix
        #   jw0{prop}{field}{visit}_..._{module}_{align|destreak}_o{field}_crf.fits .
        # Those never matched, so a corrected re-reduction's crf (e.g. the skymatch
        # background fix) silently never reached cataloging -- the per-exposure
        # *_{align|destreak}_o{field}_crf.fits stayed at the OLD reduction's mtime.
        # Map product-named crf -> per-exposure names by EXPSTART (1:1) and copy
        # into place.  member['expname'] already carries the _align/_destreak
        # suffix (set in the destreak/align loop above), so the target name matches
        # what cataloging --each-suffix consumes.
        _prod_name = asn_data['products'][0]['name']
        _prod_crf = sorted(glob(os.path.join(
            output_dir, f'{_prod_name}_*_o{field}_crf.fits')))
        if _prod_crf:
            def _crf_key(fn):
                # (EXPSTART, DETECTOR): SW filters read nrcb1-4 SIMULTANEOUSLY, so
                # EXPSTART alone collides across the 4 detectors of one exposure --
                # the detector disambiguates (LW nrcblong is 1:1 on EXPSTART alone).
                h = fits.getheader(fn)
                es = h.get('EXPSTART')
                det = h.get('DETECTOR')
                if (es is None or det is None) and len(fits.open(fn)) > 1:
                    h1 = fits.getheader(fn, 1)
                    es = es if es is not None else h1.get('EXPSTART')
                    det = det if det is not None else h1.get('DETECTOR')
                return (round(float(es), 6), str(det))
            _targ_by_es = {}
            for member in asn_data['products'][0]['members']:
                _mb = os.path.basename(member['expname'])
                _target = os.path.join(
                    output_dir, _mb.replace('.fits', f'_o{field}_crf.fits'))
                _cal_path = (member['expname'] if os.path.exists(member['expname'])
                             else os.path.join(output_dir, _mb))
                try:
                    _targ_by_es[_crf_key(_cal_path)] = _target
                except (FileNotFoundError, OSError, TypeError, ValueError):
                    print(f"  WARNING: cannot read EXPSTART/DETECTOR of {_mb}; "
                          f"skipping its crf mapping", flush=True)
            for _pc in _prod_crf:
                try:
                    _target = _targ_by_es.get(_crf_key(_pc))
                except (FileNotFoundError, OSError, TypeError, ValueError):
                    _target = None
                if _target is None:
                    print(f"  WARNING: product crf {os.path.basename(_pc)} has no "
                          f"per-exposure cal match; per-exposure crf NOT written",
                          flush=True)
                    continue
                shutil.copy(_pc, _target)
                print(f"  crf rename: {os.path.basename(_pc)} -> "
                      f"{os.path.basename(_target)}", flush=True)
        else:
            print(f"  (no product-named crf {_prod_name}_*_o{field}_crf.fits found; "
                  f"assuming crf already per-exposure named)", flush=True)

        print("After tweakreg step, checking WCS headers:")
        for member in asn_data['products'][0]['members']:
            check_wcs(member['expname'])
            check_wcs(member['expname'].replace('destreak', 'i2d'))
        check_wcs(asn_data['products'][0]['name'] + "_i2d.fits")
        _stamp_imaging_product(os.path.join(
            output_dir, asn_data['products'][0]['name'] + "_i2d.fits"))

        # NOTE (2026-07-11): retired the post-Image3 realign_to_vvv / realign_to_catalog
        # step -- the mosaic tie comes from per-exposure fix_alignment; this rigid CRVAL
        # nudge was a no-op on dense-refcat fields and only wrote a ~5GB _realigned-to-refcat
        # duplicate of _i2d.  Not the release deliverable (release uses _i2d).

        # saturated star "removal" should only be done in the cataloging stage
        # print(f"Removing saturated stars.  cwd={os.getcwd()}")
        # try:
        #     remove_saturated_stars(f'jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername.lower()}-{module}_i2d.fits')
        #     if did_vvv_realign:
        #         remove_saturated_stars(f'jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername.lower()}-{module}{destreak_suffix}_realigned-to-vvv.fits')
        # except (TimeoutError, requests.exceptions.ReadTimeout) as ex:
        #     print("Failed to run remove_saturated_stars with failure {ex}")


    if module == 'nrcb':
        # assume nrca is run before nrcb
        if do_merge:
            print("nrca+nrcb merged mosaic comes from Image3 module='merged'; no realign merge")
        else:
            print("NRCB-only subarray mode; merge step is not expected or required.")

        #try:
        #    # this is probably wrong / has wrong path names.
        #    remove_saturated_stars(f'jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername.lower()}-{module}-reproject_i2d.fits')
        #    remove_saturated_stars(f'jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername.lower()}-{module}_realigned-to-vvv.fits')
        #except (TimeoutError, requests.exceptions.ReadTimeout) as ex:
        #    print("Failed to run remove_saturated_stars with failure {ex}")

    if module == 'merged':
        # try merging all frames & modules
        print(f"Working on merged reduction (both modules):  asn_file={asn_file}")

        # Load asn_data for both modules
        with open(asn_file) as f_obj:
            asn_data = json.load(f_obj)

        for member in asn_data['products'][0]['members']:
            print(f"Running destreak={do_destreak} and maybe alignment on {member} for module={module}")
            hdr = fits.getheader(member['expname'])
            if do_destreak:
                if filtername in (hdr['PUPIL'], hdr['FILTER']):
                    outname = destreak(member['expname'],
                                    use_background_map=True,
                                    median_filter_size=2048)  # median_filter_size=medfilt_size[filtername])
                    member['expname'] = outname

                    # re-do alignment if destreak file doesn't exist at the earlier step above
                    fix_alignment(outname, proposal_id=proposal_id, module=module, field=field, basepath=basepath, filtername=filtername, use_average=use_average)
            else: # make align files
                fname = member['expname']
                assert fname.endswith('_cal.fits')
                member['expname'] = fname.replace("_cal.fits", "_align.fits")
                # copyfile (not copy) skips chmod, avoiding PermissionError when
                # a previous run by another user owns the destination file in a
                # group-writable shared workspace (e.g. W51 with t.yoo files).
                shutil.copyfile(fname, member['expname'])

                fix_alignment(member['expname'], proposal_id=proposal_id, module=module, field=field, basepath=basepath, filtername=filtername, use_average=use_average)

        asn_data['products'][0]['name'] = f'jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername.lower()}-merged'
        asn_file_merged = asn_file.replace("_asn.json", f"_merged_asn.json")
        with open(asn_file_merged, 'w') as fh:
            json.dump(asn_data, fh)

        # don't re-fit to VVV - it's not accurate enough with the JWST-derived
        # catalogs.  We needed to use our own much more extensive cataloging to
        # beat down the noise enough to make this approach viable
        if False: # filtername.lower() == 'f405n':
            vvvdr2fn = (f'{basepath}/{filtername.upper()}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername}-{module}_vvvcat.ecsv')
            print(f"Loaded VVV catalog {vvvdr2fn}")
            retrieve_vvv(basepath=basepath, filtername=filtername, proposal_id=proposal_id, fov_regname=fov_regname[regionname], module=module, fieldnumber=field)
            tweakreg_parameters['abs_refcat'] = abs_refcat = vvvdr2fn
            tweakreg_parameters['abs_searchrad'] = 1
            reftbl = Table.read(abs_refcat)
            assert 'skycoord' in reftbl.colnames
        else:
            abs_refcat = get_existing_reference_astrometric_catalog_path(basepath, proposal_id, field, filtername=filtername)
            reftbl = None
            if abs_refcat is not None:
                reftbl = Table.read(abs_refcat)
                assert 'skycoord' in reftbl.colnames
                reftblversion = reftbl.meta.get('VERSION', 'unknown')

                # truncate to top 10,000 sources for speed when this is ECSV
                if abs_refcat.endswith('.ecsv'):
                    abs_refcat_truncated = abs_refcat.replace('.ecsv', '_truncated10000.ecsv')
                    reftbl[:10000].write(abs_refcat_truncated, overwrite=True)
                    abs_refcat = abs_refcat_truncated

                tweakreg_parameters['abs_searchrad'] = 0.4
                tweakreg_parameters['searchrad'] = 0.05
                print(f"Reference catalog is {abs_refcat} with version {reftblversion}")
            else:
                print(f"No configured reference catalog found for proposal_id={proposal_id} field={field} in {basepath}. Running first-pass without abs_refcat realignment.")

        if abs_refcat is not None:
            tweakreg_parameters.update({'abs_refcat': abs_refcat,})

        print("Running Image3Pipeline with tweakreg (merged)")
        calwebb_image3.Image3Pipeline.call(
            asn_file_merged,
            steps={'tweakreg': tweakreg_parameters,},
            #steps={'tweakreg': False,}
            output_dir=output_dir,
            save_results=True)
        print(f"DONE running Image3Pipeline {asn_file_merged}.  This should have produced file {asn_data['products'][0]['name']}_i2d.fits")

        print("After tweakreg step, checking WCS headers:")
        for member in asn_data['products'][0]['members']:
            check_wcs(member['expname'])
        check_wcs(asn_data['products'][0]['name'] + "_i2d.fits")
        _stamp_imaging_product(os.path.join(
            output_dir, asn_data['products'][0]['name'] + "_i2d.fits"))

        vvv_region_file = f"{basepath}/{fov_regname[regionname]}" if regionname in fov_regname else None
        # Only run VVV realignment for targets whose refnames is 'VVV'.  Gaia /
        # GNS / UKIDSS targets (Wd1, Wd2, W51, GC fields) must skip this
        # because retrieve_vvv returns no rows outside VVV coverage.
        # NOTE (2026-07-11): retired the post-Image3 realign_to_vvv / realign_to_catalog
        # step -- the mosaic tie comes from per-exposure fix_alignment; this rigid CRVAL
        # nudge was a no-op on dense-refcat fields and only wrote a ~5GB _realigned-to-refcat
        # duplicate of _i2d.  Not the release deliverable (release uses _i2d).

        # removing saturated stars should only be done in cataloging stage
        # print(f"Removing saturated stars.  cwd={os.getcwd()}")
        # try:
        #     remove_saturated_stars(f'jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername.lower()}-merged_i2d.fits')
        #     remove_saturated_stars(f'jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername.lower()}-{module}{destreak_suffix}_realigned-to-vvv.fits')
        # except (TimeoutError, requests.exceptions.ReadTimeout) as ex:
        #     print("Failed to run remove_saturated_stars with failure {ex}")

    globals().update(locals())
    return locals()


# offsets tables already collapse-checked this process (warn once per file, not per frame)
_VALIDATED_OFFSETS_TABLES = set()


def _apply_consensus_offsets_table(fn, basepath, proposal_id, filtername, field):
    """Return (rashift, decshift) astropy Quantities for THIS exposure from the
    per-exposure consensus offsets table seeded by the m2 astrometry checkpoint
    (``seed_offsets_table_from_consensus``).

    The table lives at ``{basepath}/offsets/Offsets_JWST_Brick{proposal_id}_consensus.csv``
    and is keyed (Visit, Filter, Exposure, Module) with ``dra (arcsec)`` /
    ``ddec (arcsec)`` in the Δα-coordinate convention ``fix_alignment`` applies.

    Returns (0,0) arcsec when the table does not exist yet (the FIRST reduction
    pass, before cataloging has measured the consensus) so the frame stays at the
    tweakreg/assign_wcs frame and the checkpoint can measure the raw scatter, and
    (0,0) when this exposure has no row (it was already within consensus
    tolerance, so no shift is needed)."""
    tblfn = f'{basepath}/offsets/Offsets_JWST_Brick{proposal_id}_consensus.csv'
    if not os.path.exists(tblfn):
        print(f"[consensus] no table {tblfn} yet; leaving "
              f"{os.path.basename(fn)} at frame (0,0)")
        return 0 * u.arcsec, 0 * u.arcsec
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        lookup_consensus_offset)
    tbl = Table.read(tblfn)
    visit = fn.split('_')[0]
    exposure = int(fn.split('_')[-3])
    thismodule = fn.split('_')[-2]
    try:
        dra, ddec = lookup_consensus_offset(
            tbl, visit, exposure, thismodule, filtername)
    except ValueError as ex:
        raise ValueError(f"{ex} in {tblfn} (frame {fn})") from ex
    return dra * u.arcsec, ddec * u.arcsec


def fix_alignment(fn, proposal_id=None, module=None, field=None, basepath=None, filtername=None,
                  use_average=True):
    if os.path.exists(fn):
        print(f"Running manual align for {module} data ({proposal_id} + {field}): {fn}", flush=True)
    else:
        print(f"Skipping manual align for nonexistent file {module} ({proposal_id} + {field}): {fn}", flush=True)
        return

    if os.environ.get('APPLY_DVA_CORRECTION', '1') != '0':
        # Inter-detector differential-velocity-aberration shift (see
        # dva_correction.py).  ON BY DEFAULT until STScI corrects assign_wcs
        # upstream (spacetelescope/jwst#9400) -- set APPLY_DVA_CORRECTION=0 to
        # disable, and DISABLE it if/when the upstream fix lands (the DVACORR
        # marker + the network-selfcal closure test guard double-correction).
        # Idempotent; applied BEFORE the reference tie so the offsets absorb
        # its common-mode part.
        from jwst_gc_pipeline.reduction.dva_correction import apply_dva_correction
        apply_dva_correction(fn)

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
        basepath = f'/orange/adamginsburg/jwst/{field}'
    if module is None:
        module = 'nrc' + mod.meta.instrument.module.lower()

    _prov_tbl = None       # offsets table actually consumed (header provenance)
    _prov_row_stage = ''   # checkpoint stage that last corrected the row, if any
    _frame_gen = None      # this frame's WCS-generation stamp (set in the locked branch)
    if (field == '004' and proposal_id == '1182') or (field in ('001', '002') and proposal_id == '2221'):
        # field 002 (Cloud C) added 2026-06-22: route through the per-exposure VIRAC2-locked
        # table (cloudc/offsets/Offsets_JWST_Brick2221_VIRAC2locked.csv, built by
        # build_virac2_locked_perexp.py --region cloudc) instead of the old hardcoded per-visit
        # shifts below. Replaces the deprecated F405N-crowdsource frame (~90 mas off Gaia).
        refname = refnames[proposal_id]
        exposure = int(fn.split("_")[-3])
        thismodule = fn.split("_")[-2]
        visit = fn.split("_")[0]
        # MODULE-LOCKED per-VISIT offsets (preferred). The NIRCam detectors are SIAF-locked
        # to <0.01 px within an exposure, and the SIAF/assign_wcs solution already carries the
        # correct per-exposure dithers + per-detector geometry; the only uncorrected term is the
        # per-VISIT guide-star pointing error.  So the alignment must be ONE shift per visit applied
        # identically to all exposures AND all 8 detectors -- NOT an independent per-detector (or
        # per-exposure) tweakreg shift, which breaks the lock and produces filter-to-filter
        # 'quiltwork' (~10-15 mas) and injects per-exposure VIRAC2 noise.  The locked table (built by
        # brick2221/analysis/relock_exposures.py: undo the recorded per-detector offset -> SIAF ->
        # one VIRAC2-tied shift jointly solved over all exposures+detectors of the visit) is keyed on
        # (Visit, Filter) only -- NO Module, NO Exposure column.
        locked_tbl = f'{basepath}/offsets/Offsets_JWST_Brick{proposal_id}_VIRAC2locked.csv'
        if os.path.exists(locked_tbl):
            offsets_tbl = Table.read(locked_tbl)
            # GENERATION GUARD (layered, 2026-07-13).  A correction is only
            # valid on the WCS GENERATION it was solved against (the frame
            # drifted ~30-48 mas between Brick runs; ~69 mas VIRAC2 offset
            # root cause).  Verification layers, strongest first:
            #   1. per-row BASE STAMPS (base_calver / base_crds_ctx /
            #      base_dvacorr, written by the tie builders from the crf they
            #      solved on): compared against THIS frame's generation keys.
            #      A mismatch is deterministic evidence of a stale tie ->
            #      hard-fail (override: GENLOCK_ALLOW_MISMATCH=1).
            #   2. mtime fallback (columns absent): WEAK -- the standard chain
            #      regenerates destreak fresh, so crf mtime > table mtime on
            #      EVERY run including correct ones; warn-only, and
            #      GENLOCK_STRICT applies only to this fallback.
            _frame_gen = None
            try:
                from jwst_gc_pipeline.astrometry_utils import generation_stamp
                with fits.open(fn) as _gfh:
                    _hdr0 = dict(_gfh[0].header)
                    _hdr0.update({k: v for k, v in _gfh[1].header.items()
                                  if k in ('DVACORR',)})
                    _frame_gen = generation_stamp(_hdr0)
            except (OSError, KeyError, IndexError) as _gex:
                print(f"[genlock] could not read generation keys from {fn}: {_gex}")
            _has_stamps = all(f'base_{k}' in offsets_tbl.colnames
                              for k in ('calver', 'crds_ctx', 'dvacorr'))
            if not _has_stamps:
                try:
                    _t_tbl = os.path.getmtime(locked_tbl); _t_crf = os.path.getmtime(fn)
                except OSError:
                    _t_tbl = _t_crf = None
                if _t_tbl is not None and _t_tbl < _t_crf - 1.0:
                    import datetime as _dt
                    _gmsg = (f"[genlock] offsets table {os.path.basename(locked_tbl)} has no "
                             f"base_* generation stamps and predates crf "
                             f"{os.path.basename(fn)}; the tie may be a reduction "
                             f"generation behind (mtime is a WEAK proxy -- rebuild the "
                             f"table with the stamping builders for a real check).")
                    if os.environ.get('GENLOCK_STRICT'):
                        raise RuntimeError(_gmsg)
                    print("WARNING: " + _gmsg, flush=True)
            # One-time collapse check: the ad-hoc VIRAC2locked curation once overwrote
            # brick-1182 visit-001's offset with visit-002's (both ~+1.9" for a visit
            # truly ~20" off). Warn if distinct visits share a value here so it can't
            # be applied blind again. Once per file per process (not per frame).
            if locked_tbl not in _VALIDATED_OFFSETS_TABLES:
                _VALIDATED_OFFSETS_TABLES.add(locked_tbl)
                from jwst_gc_pipeline.reduction.validate_offsets_table import assert_offsets_table_sane
                assert_offsets_table_sane(offsets_tbl, context=os.path.basename(locked_tbl))
            match = ((offsets_tbl['Visit'] == visit)
                     & (offsets_tbl['Filter'] == filtername))
            # Support BOTH conventions: per-VISIT tables (1 row/visit, no usable Exposure) and
            # per-EXPOSURE tables (N rows/visit, Exposure int).  Narrow by Exposure only when
            # >1 row matches.  Per-exposure removes real per-exposure jitter (~7-8 mas, measured
            # 2026-06-20; same sign across all filters) WITHOUT injecting per-exposure VIRAC2
            # noise -- the per-exposure term is solved against the dense INTERNAL consensus
            # (build_virac2_locked_perexp.py), only the per-visit bulk touches VIRAC2.
            if match.sum() > 1 and 'Exposure' in offsets_tbl.colnames:
                match = match & (offsets_tbl['Exposure'] == exposure)
            # Per-MODULE narrowing (default OFF: filters lock NRCA==NRCB together, per the
            # <1 mas CRDS inter-module policy). Documented exception: F410M. Our reprocessing
            # (CAL_VER 1.14.1.dev43 + CRDS jwst_1253.pmap) applies FILTER-SPECIFIC distortion
            # (0249 F410M/NRCALONG vs 0300 NRCBLONG); these leave NRCALONG ~40 mas inconsistent
            # with NRCBLONG, which a single per-filter shift cannot correct (2026-07-02 audit vs
            # VIRAC2). Such filters carry per-module rows (a 'Module' column); narrow to this
            # module. Filters with a single row still lock both modules identically.
            if match.sum() > 1 and 'Module' in offsets_tbl.colnames:
                match = match & ((offsets_tbl['Module'] == thismodule)
                                 | (offsets_tbl['Module'] == thismodule.strip('1234')))
            if match.sum() != 1:
                raise ValueError(f"module-locked offset match={match.sum()} for {fn} "
                                 f"(visit={visit}, exposure={exposure}, filter={filtername}); "
                                 f"expected exactly 1 row in {locked_tbl}")
            row = offsets_tbl[match]
            if _has_stamps and _frame_gen is not None:
                _mismatch = {k: (str(row[f'base_{k}'][0]), _frame_gen[k])
                             for k in ('calver', 'crds_ctx', 'dvacorr')
                             if str(row[f'base_{k}'][0]) not in ('', 'nan')
                             and str(row[f'base_{k}'][0]) != _frame_gen[k]}
                if _mismatch:
                    _gmsg = (f"[genlock] GENERATION MISMATCH for {fn}: the tie row was "
                             f"solved on {_mismatch} (base vs frame). Applying it would "
                             f"stack a stale correction on a moved frame. Rebuild the "
                             f"VIRAC2locked table on THIS generation "
                             f"(GENLOCK_ALLOW_MISMATCH=1 to override).")
                    if os.environ.get('GENLOCK_ALLOW_MISMATCH') == '1':
                        print("WARNING (override): " + _gmsg, flush=True)
                    else:
                        raise RuntimeError(_gmsg)
            rashift = float(row['dra (arcsec)'][0]) * u.arcsec
            decshift = float(row['ddec (arcsec)'][0]) * u.arcsec
            print(f"MODULE-LOCKED per-visit offset for {fn}: ({rashift}, {decshift})")
        elif use_average:
            if 'bug' in refname.lower():
                raise ValueError("This is a disallowed reference file")
            tblfn = f'{basepath}/offsets/Offsets_JWST_Brick{proposal_id}_{refname}_average.csv'
            print(f"Using average offset table {tblfn}")
            offsets_tbl = Table.read(tblfn)
            match = (((offsets_tbl['Module'] == thismodule) |
                      (offsets_tbl['Module'] == thismodule.strip('1234'))) &
                     (offsets_tbl['Filter'] == filtername)
                     )
            if 'Visit' in offsets_tbl.colnames:
                match &= (offsets_tbl['Visit'] == visit)
            row = offsets_tbl[match]
            print(f'Running manual align for merged for {filtername} {row["Module"][0]}.')
        else:
            if 'bug' in refname.lower():
                raise ValueError("This is a disallowed reference file")
            tblfn = f'{basepath}/offsets/Offsets_JWST_Brick{proposal_id}_{refname}.csv'
            print(f"Using offset table {tblfn}")
            offsets_tbl = Table.read(tblfn)
            match = ((offsets_tbl['Visit'] == visit) &
                     (offsets_tbl['Exposure'] == exposure) &
                     ((offsets_tbl['Module'] == thismodule) | (offsets_tbl['Module'] == thismodule.strip('1234'))) &
                     (offsets_tbl['Filter'] == filtername)
                     )
            row = offsets_tbl[match]
            print(f'Running manual align for merged for {filtername} {row["Group"][0]} {row["Module"][0]} {row["Exposure"][0]}.')
        if match.sum() != 1:
            raise ValueError(f"too many or too few matches for {fn} (match.sum() = {match.sum()}).  exposure={exposure}, thismodule={thismodule}, filtername={filtername}")
        rashift = float(row['dra (arcsec)'][0])*u.arcsec
        decshift = float(row['ddec (arcsec)'][0])*u.arcsec
        _prov_tbl = locked_tbl if os.path.exists(locked_tbl) else tblfn
        if 'prov_stage' in offsets_tbl.colnames:
            _prov_row_stage = str(row['prov_stage'][0])
    elif (field == '002' and proposal_id == '2221'):
        visit = fn.split('_')[0][-3:]
        thismodule = fn.split("_")[-2].strip('1234')
        if visit == '001':
            decshift = 7.95*u.arcsec
            rashift = 0.6*u.arcsec
        elif visit == '002':
            decshift = 3.85*u.arcsec
            rashift = 1.57*u.arcsec
        else:
            decshift = 0*u.arcsec
            rashift = 0*u.arcsec
        if filtername.upper() in ('F212N', 'F187N', 'F182M'):
            print('Short wavelength offset correction.')
            if 'nrca' in thismodule.lower():
                decshift += 0.1*u.arcsec
                rashift += -0.23*u.arcsec
    elif (field == '002' and str(proposal_id) == '2092'):
        # Cloud E (2092 obs002) is a 2-visit mosaic.  There is no per-frame
        # absolute tweakreg here (TweakRegStep is skipped) and the post-resample
        # realign is a single global median shift, so an internal visit-to-visit
        # pointing difference passes straight through and blurs stars in the
        # visit-001/002 overlap.  Measured offset (277 matched stars, F480M
        # nrcblong overlap, 2026-06-10): visit002 - visit001 = (dRA -98, dDec
        # +171) mas -- a PURE translation (linear-fit gradients <=0.3 mas/arcsec,
        # residual <7 mas; no rotation/scale).  Bring visit 002 onto visit 001
        # (verified in-WCS: this shift takes the overlap offset to (-6, -5) mas);
        # the subsequent realign-to-refcat sets the common absolute zero point.
        # The offset is a guide-star/pointing difference, so it is the same for
        # all detectors/filters of the visit -> keyed on visit only.
        visit = fn.split('_')[0][-3:]
        if visit == '002':
            rashift = 0.098*u.arcsec
            decshift = -0.171*u.arcsec
        else:
            rashift = 0*u.arcsec
            decshift = 0*u.arcsec
    elif (field == '007' and str(proposal_id) == '3958'):
        # Sickle NIRCam: tie each exposure to the GNS frame.  Audit 2026-06-20:
        # sickle had NO per-exposure alignment (fell through to the else ->
        # rashift=0), so its catalogs sat at the raw assign_wcs frame ~90 mas off
        # the GNS-tied mosaics/refcat (mosaic frame != catalog frame).  The
        # sickle->GNS offset is a single field translation, constant across
        # filters/exposures to <3 mas; measured per filter on the m6/m7 merged
        # catalogs vs catalogs/nircam_bootstrapped_to_gns_refcat.fits (see
        # _bench/build_sickle_gns_offsets.py + offsets/Offsets_JWST_Brick3958_GNS.csv).
        # corr below is ON-SKY (GNS - catalog), mas.  adjust_wcs's delta_ra is a
        # COORDINATE (Delta-alpha) rotation -> on-sky RA = delta_ra*cos(dec)
        # (preflight: delta_ra=-90 mas gave -78.9 mas on-sky = -90*cos(28.8));
        # so delta_ra = corr_dRA_onsky / cos(dec).  delta_dec is on-sky 1:1.
        _gns = {'F187N': (-89.7, -34.2), 'F210M': (-88.5, -34.5),
                'F335M': (-89.5, -33.2), 'F470N': (-91.4, -33.9),
                'F480M': (-90.6, -33.1)}
        _cdra, _cddec = _gns.get(filtername.upper(), (-90.0, -34.0))
        _cosd = np.cos(np.radians(-28.805))
        rashift = (_cdra / 1000.0 / _cosd) * u.arcsec
        decshift = (_cddec / 1000.0) * u.arcsec
    elif str(proposal_id) in ('1979', '1334'):
        # M4 (1979 o002 + o003="M-4-shift") and M92 (1334 o001): non-GC halo
        # clusters outside VIRAC2/VVV -> tie each exposure to Gaia DR3
        # (gaia_refcat.fits, PM-propagated).  Audit 2026-07-11: these fell through
        # to the else (rashift=0), sat at the raw assign_wcs frame ~2" off Gaia
        # (RAOFFSET=0, no offsets table, no catalog).  Bulk tie = measure_offset
        # histogram of the untied destreak crf vs gaia_refcat, per (visit,filter)
        # (measure_perobs; c=94-530).  M92 is a PURE per-visit shift (all 4 filters
        # agree to <20 mas); M4 differs SW(F150W2) vs LW(F322W2) ~300-500 mas so is
        # keyed per (visit,filter).  corr is ON-SKY (Gaia - detection) mas; adjust_wcs
        # delta_ra is a COORDINATE rotation -> delta_ra = corr_dRA_onsky / cos(dec),
        # delta_dec 1:1 (same convention as the sickle GNS branch above).
        _gaia_tie = {
            ('jw01979002001', 'F150W2'): (104.7, -180.3),
            ('jw01979002001', 'F322W2'): (-442.9, -87.9),
            ('jw01979003001', 'F150W2'): (-2189.0, 370.7),
            ('jw01979003001', 'F322W2'): (-1914.7, 546.9),
            ('jw01334001001', 'F090W'): (-1832.1, -708.2),
            ('jw01334001001', 'F150W'): (-1853.5, -710.6),
            ('jw01334001001', 'F277W'): (-1852.1, -711.7),
            ('jw01334001001', 'F444W'): (-1852.7, -710.7),
        }
        _visit = fn.split('_')[0]
        _key = (_visit, filtername.upper())
        _cdra, _cddec = _gaia_tie.get(_key, (0.0, 0.0))
        _cosd = np.cos(np.radians(-26.427 if str(proposal_id) == '1979' else 43.139))
        rashift = (_cdra / 1000.0 / _cosd) * u.arcsec
        decshift = (_cddec / 1000.0) * u.arcsec
        if _key not in _gaia_tie:
            print(f"WARNING: no Gaia tie for {_key}; leaving {fn} at raw frame (0,0)")
    elif str(proposal_id) == '4147':
        # sgrc: per-exposure CONSENSUS re-tie (2026-07-16).  tweakreg is skipped
        # in this pipeline, so sgrc had NO per-exposure alignment (fell through to
        # the else -> rashift=0) and its exposures scattered ~2-8 mas around the
        # visit consensus (m2 astrometry checkpoint flagged 39 exposures > 2 mas
        # in F115W/nrcb alone).  The consensus offsets table -- seeded by the m2
        # checkpoint (seed_offsets_table_from_consensus) and refined by
        # update_offsets_table on later iterations -- shifts each exposure ONTO
        # the dense INTERNAL consensus, removing the raw guide-star jitter WITHOUT
        # injecting per-exposure reference noise.  The consensus->VIRAC2 bulk is
        # set by the '012' gaia_virac2 refcat (Fix A) + realign_to_catalog.  On
        # the FIRST reduction pass the table does not exist yet, so the helper
        # returns (0,0) and the checkpoint gets to measure the raw scatter.
        # Keyed (Visit, Filter, Exposure, Module); dra/ddec are arcsec Δα
        # coordinate (same convention as the brick VIRAC2locked table).
        rashift, decshift = _apply_consensus_offsets_table(
            fn, basepath, str(proposal_id), filtername, field)
        _prov_tbl = f'Offsets_JWST_Brick{proposal_id}_consensus.csv'
        _prov_row_stage = 'm2-consensus'
    else:
        rashift = 0*u.arcsec
        decshift = 0*u.arcsec
    print(f"Shift for {fn} is {rashift}, {decshift}")
    align_fits = fits.open(fn)
    if 'RAOFFSET' in align_fits[1].header:
        # don't shift twice if we re-run
        print(f"{fn} is already aligned ({align_fits[1].header['RAOFFSET']}, {align_fits[1].header['DEOFFSET']})")
        # DISAGREEMENT GUARD: the plain skip-if-present check silently KEPT a stale
        # RAOFFSET after the offsets table was corrected -- brick-1182 v001 crf held
        # +1.9" while the table said -17.5", so half the mosaic stayed ~20" off and
        # the idempotent guard blocked the fix. Compare the baked-in RAOFFSET to the
        # value we WOULD apply now; if they disagree, this frame is stale.
        _cur_ra = float(align_fits[1].header['RAOFFSET'])
        _cur_de = float(align_fits[1].header.get('DEOFFSET', 'nan'))
        _dra = abs(_cur_ra - rashift.value)
        _dde = abs(_cur_de - decshift.value)
        _tol = float(os.environ.get('RAOFFSET_DISAGREE_TOL_ARCSEC', 0.05))
        if _dra > _tol or _dde > _tol:
            _msg = (f"STALE ASTROMETRY: {fn} carries RAOFFSET/DEOFFSET "
                    f"({_cur_ra:+.4f},{_cur_de:+.4f})\" but the current table would "
                    f"apply ({rashift.value:+.4f},{decshift.value:+.4f})\" "
                    f"(disagree {_dra:.3f},{_dde:.3f}\" > {_tol}\"). This frame was "
                    f"built from an OLD table and the skip-if-present guard is hiding "
                    f"it. Regenerate the working copy from _cal (destreak overwrite) "
                    f"so RAOFFSET resets and the current table is applied, OR set "
                    f"FORCE_REALIGN_ON_DISAGREE=1 to re-apply now.")
            if os.environ.get('FORCE_REALIGN_ON_DISAGREE') == '1':
                raise RuntimeError(
                    _msg + " [FORCE_REALIGN_ON_DISAGREE=1: refusing to silently keep "
                    "a stale frame; regenerate it from _cal.]")
            warnings.warn(_msg)
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

        # BASE -> TARGET proof (2026-07-13): record the fiducial-pixel sky
        # coordinate BEFORE the correction (the coordinate the offset applies
        # to) and AFTER it (the coordinate it must produce), then VERIFY
        # target == base + coordinate-shift.  With these stamped, any later
        # reader can re-derive and re-check the correction no matter when --
        # and a correction can never be silently applied to the wrong base.
        _base_ra, _base_dec = float(wcsobj.pixel_to_world(1024, 1024).ra.deg), \
            float(wcsobj.pixel_to_world(1024, 1024).dec.deg)
        _tgt_ra, _tgt_dec = float(ww.pixel_to_world(1024, 1024).ra.deg), \
            float(ww.pixel_to_world(1024, 1024).dec.deg)
        _exp_ra = _base_ra + rashift.to(u.deg).value   # COORDINATE convention
        _exp_dec = _base_dec + decshift.to(u.deg).value
        _cosd = np.cos(np.radians(_base_dec))
        _resid_mas = float(np.hypot((_tgt_ra - _exp_ra) * _cosd,
                                    _tgt_dec - _exp_dec) * 3.6e6)
        if _resid_mas > 0.5:
            raise RuntimeError(
                f"astrometric apply verification FAILED for {fn}: fiducial moved to "
                f"({_tgt_ra:.8f},{_tgt_dec:.8f}) but base+shift predicts "
                f"({_exp_ra:.8f},{_exp_dec:.8f}) -- residual {_resid_mas:.2f} mas. "
                f"The offset convention or the WCS apply path is wrong; NOT writing.")

        # FITS header
        align_fits = fits.open(fn)
        align_fits[1].header['OLCRVAL1'] = align_fits[1].header['CRVAL1']
        align_fits[1].header['OLCRVAL2'] = align_fits[1].header['CRVAL2']
        align_fits[1].header.update(ww.to_fits()[0])
        align_fits[1].header['RAOFFSET'] = rashift.value
        align_fits[1].header['DEOFFSET'] = decshift.value
        # correction provenance: base/target fiducials + convention + the
        # generation this frame carried when corrected (audit at any time:
        # recompute pixel_to_world(1024,1024) and compare to ATGTRA/ATGTDE)
        align_fits[1].header['ABASERA'] = (_base_ra, '[deg] fiducial(1024,1024) BEFORE correction')
        align_fits[1].header['ABASEDE'] = (_base_dec, '[deg] fiducial dec BEFORE correction')
        align_fits[1].header['ATGTRA'] = (_tgt_ra, '[deg] fiducial AFTER correction (verify me)')
        align_fits[1].header['ATGTDE'] = (_tgt_dec, '[deg] fiducial dec AFTER correction')
        align_fits[1].header['AOFFCONV'] = ('coordinate', 'RAOFFSET is dra_coordinate (on-sky = *cos(dec))')
        align_fits[1].header['AVERMAS'] = (_resid_mas, '[mas] base+shift vs target residual (proof)')
        if _frame_gen is not None:
            align_fits[1].header['AGENCAL'] = (_frame_gen.get('cal_ver', ''), 'CAL_VER at correction')
            align_fits[1].header['AGENCTX'] = (_frame_gen.get('crds_ctx', ''), 'CRDS_CTX at correction')
            align_fits[1].header['AGENDVA'] = (_frame_gen.get('dvacorr', ''), 'DVACORR at correction')
        # provenance: WHY these RAOFFSET/DEOFFSET (which table, which checkpoint
        # last corrected the row, when) -- see astrometry_checkpoint.py
        from jwst_gc_pipeline.photometry.astrometry_checkpoint import provenance_header_cards
        _cosd_prov = np.cos(np.radians(float(align_fits[1].header['CRVAL2'])))
        for _k, _v, _c in provenance_header_cards(
                stage=_prov_row_stage or 'fix_alignment',
                dra_onsky_mas=rashift.value * _cosd_prov * 1000.0,
                ddec_onsky_mas=decshift.value * 1000.0,
                method='offsets-table (histogram-stacked tie)',
                references=refnames.get(proposal_id, 'n/a'),
                table_name=_prov_tbl or 'hardcoded/none'):
            align_fits[1].header[_k] = (_v, _c)
        align_fits.writeto(fn, overwrite=True)
        assert 'RAOFFSET' in fits.getheader(fn, ext=1)
        # provenance: the per-exposure aligned crf is what cataloging re-fits, so
        # its data/wcs facets gate the cataloging-skip decision.  Stamp only on
        # the apply branch (RAOFFSET freshly baked); fail-soft.
        _stamp_imaging_product(fn)
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
                      default='F466N,F405N,F410M,F212N,F182M,F187N',
                      help="filter name list", metavar="filternames")
    parser.add_option("-m", "--modules", dest="modules",
                    default='nrca,nrcb,merged',
                    help="module list", metavar="modules")
    parser.add_option("-d", "--field", dest="field",
                    default='001,002',
                    help="list of target fields", metavar="field")
    parser.add_option("-s", "--skip_step1and2", dest="skip_step1and2",
                      default=False,
                      action='store_true',
                      help="Skip the image-remaking step?", metavar="skip_Step1and2")
    parser.add_option("--no_destreak", dest="no_destreak",
                      default=False,
                      action='store_true',
                      help="Skip the destreaking step?", metavar="skip_destreak")
    parser.add_option("-p", "--proposal_id", dest="proposal_id",
                      default='2221',
                      help="proposal id (string)", metavar="proposal_id")
    parser.add_option("--skymatch-method", dest="skymatch_method",
                      default='',
                      help="Image3 skymatch skymethod ('match'/'global'/'local'). "
                           "Empty (default) skips skymatch (historical NIRCam "
                           "behavior). Use 'match' to level-match per-exposure "
                           "background pedestals and remove mosaic seams "
                           "(subtract=True, match_down=False).",
                      metavar="skymatch_method")
    (options, args) = parser.parse_args()

    filternames = options.filternames.split(",")
    modules = options.modules.split(",")
    fields = options.field.split(",")
    proposal_id = options.proposal_id
    skip_step1and2 = options.skip_step1and2
    no_destreak = bool(options.no_destreak)
    skymatch_method = (options.skymatch_method or '').strip() or None
    print(options)

    with open(os.path.expanduser('~/.mast_api_token'), 'r') as fh:
        api_token = fh.read().strip()
        os.environ['MAST_API_TOKEN'] = api_token.strip()
    Mast.login(api_token.strip())
    Observations.login(api_token)


    field_to_reg_mapping = {'2221': {'001': 'brick', '002': 'cloudc'},
                            '1182': {'004': 'brick', '002': 'w51'},
                            '5365': {'001': 'sgrb2'},
                            '6151': {'001': 'w51'},
                            '3958': {'007': 'sickle', '001': 'sickle', '002': 'sickle'},
                            '2092': {'002': 'cloudef', '005': 'cloudef'},
                            '4147': {'012': 'sgrc'},
                            '2045': {'001': 'arches', '003': 'quintuplet'},
                            '1939': {'001': 'sgra'},
                            '2211': {'023': 'gc2211', '028': 'gc2211',
                                     '046': 'gc2211', '049': 'gc2211',
                                     '050': 'gc2211'},
                            '1905': {'001': 'wd1', '003': 'wd1'},
                            '3523': {'003': 'wd2', '005': 'wd2'},
                            # Globular clusters (Jay Anderson co-I; added 2026-06-30)
                            '1334': {'001': 'm92'},
                            '1979': {'001': 'ngc6397', '002': 'm4', '003': 'm4'},
                            '8322': {'001': 'omegacen'},
                            '12587': {'001': 'omegacen'},
                            # NGC 6334 (Cat's Paw SFR; extended emission)
                            '7213': {'001': 'ngc6334'},
                            '6778': {'001': 'ngc6334'},
                            }[proposal_id]

    for field in fields:
        for filtername in filternames:
            modules_for_field = get_allowed_modules(proposal_id, field, modules, filtername=filtername)
            for module in modules_for_field:
                module_family = _module_group(module)
                print(f"Main Loop: {proposal_id} + {filtername} + {module} (family={module_family}) + {field}={field_to_reg_mapping[field]}")
                results = main(filtername=filtername, module=module_family, Observations=Observations, field=field,
                               regionname=field_to_reg_mapping[field],
                               proposal_id=proposal_id,
                               skip_step1and2=skip_step1and2,
                               do_destreak=not no_destreak,
                               skymatch_method=skymatch_method,
                              )


    if proposal_id == '2221':
        print("Running notebooks")
        from run_notebook import run_notebook
        basepath = '/orange/adamginsburg/jwst/brick/'
        if 'merge' in modules:
            run_notebook(f'{basepath}/notebooks/BrA_Separation_nrca.ipynb')
            run_notebook(f'{basepath}/notebooks/BrA_Separation_nrcb.ipynb')
            run_notebook(f'{basepath}/notebooks/F466_separation_nrca.ipynb')
            run_notebook(f'{basepath}/notebooks/F466_separation_nrcb.ipynb')
            run_notebook(f'{basepath}/notebooks/StarDestroyer_nrca.ipynb')
            run_notebook(f'{basepath}/notebooks/StarDestroyer_nrcb.ipynb')
            run_notebook(f'{basepath}/notebooks/Stitch_A_to_B.ipynb')
            run_notebook(f'{basepath}/notebooks/PaA_Separation_nrcb.ipynb')
            run_notebook(f'{basepath}/notebooks/StarDestroyer_PaA_nrcb.ipynb')


"""
await app.openFile("/jwst/brick/F410M/pipeline/jw02221-o001_t001_nircam_clear-f410m-merged_i2d.fits")
await app.appendFile("/jwst/brick/F410M/pipeline/jw02221-o001_t001_nircam_clear-f410m-nrca_i2d.fits")
await app.appendFile("/jwst/brick/F410M/pipeline/jw02221-o001_t001_nircam_clear-f410m-nrcb_i2d.fits")
await app.appendFile("/jwst/brick/F182M/pipeline/jw02221-o001_t001_nircam_clear-f182m-merged_i2d.fits")
await app.appendFile("/jwst/brick/F182M/pipeline/jw02221-o001_t001_nircam_clear-f182m-nrca_i2d.fits")
await app.appendFile("/jwst/brick/F182M/pipeline/jw02221-o001_t001_nircam_clear-f182m-nrcb_i2d.fits")
await app.appendFile("/jwst/brick/F212N/pipeline/jw02221-o001_t001_nircam_clear-f212n-merged_i2d.fits")
await app.appendFile("/jwst/brick/F212N/pipeline/jw02221-o001_t001_nircam_clear-f212n-nrca_i2d.fits")
await app.appendFile("/jwst/brick/F212N/pipeline/jw02221-o001_t001_nircam_clear-f212n-nrcb_i2d.fits")
await app.appendFile("/jwst/brick/F466N/pipeline/jw02221-o001_t001_nircam_clear-f466n-merged_i2d.fits")
await app.appendFile("/jwst/brick/F466N/pipeline/jw02221-o001_t001_nircam_clear-f466n-nrca_i2d.fits")
await app.appendFile("/jwst/brick/F466N/pipeline/jw02221-o001_t001_nircam_clear-f466n-nrcb_i2d.fits")
"""
