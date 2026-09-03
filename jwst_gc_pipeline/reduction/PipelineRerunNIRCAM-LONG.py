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

# Before importing jwst: CRDS reads its cache path when jwst loads.  config.yaml
# supplies the default; an exported CRDS_PATH wins.
from jwst_gc_pipeline.config import apply_crds_environment
from jwst_gc_pipeline.mast_names import jw_prefix, proposal_id_from_datamodel
# Printed because the cache decides which reference files -- and so which
# distortion and filter-offset solutions -- this run uses.
print(f"CRDS: {apply_crds_environment()}")

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
from jwst_gc_pipeline.reduction.mast_obs_scope import observation_scope_mask

from jwst_gc_pipeline.reduction.align_to_catalogs import merge_a_plus_b
from jwst_gc_pipeline.reduction.fits_wcs_sync import sync_header_to_gwcs
from jwst_gc_pipeline.reduction.saturated_star_finding import remove_saturated_stars
from jwst_gc_pipeline.reduction.stage12_selection import (member_in_stage12_pass,
                                                          note_stage12_processed,
                                                          stage12_skip_reason)

import crds
import jwst

filter_regex = re.compile('f[0-9][0-9][0-9][nmw]')

# Detector1 keeps the fitted ramp alongside the rate/cal products (the satstar
# path reads it).  Ramp retention policy is #421's concern; the stage-1/2
# resume check is told about this flag so it cannot look for a _ramp.fits that
# the driver never writes.
SAVE_CALIBRATED_RAMP = True

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

# Registered in jwst_gc_pipeline/fields.yaml -- the one place a target is
# declared.  See docs/FIELDS.md.
# Imported as field_registry: `fields` is a local variable in these
# drivers (the --field list), and shadowed the module.
from jwst_gc_pipeline import fields as field_registry
from jwst_gc_pipeline.reduction.crds_cache import open_crds_reference
# The check itself lives in `reduction.wcs_check` -- one implementation for
# all three drivers.  The three copies had diverged: MIRI and NIRISS still
# evaluated the hardcoded NIRCam centre (1024, 1024), which is OFF a MIRI
# (1024, 1032) array, so every separation they printed was `nan`.
from jwst_gc_pipeline.reduction.wcs_check import check_wcs


# Reference catalog configuration by proposal and field.
# Paths are relative to basepath.

# Module restrictions per proposal/field/filter for single-module datasets
# Sickle is NRCB-only (SUB640 subarray) but detectors differ by wavelength:
# - Short-wavelength (F187N, F210M): nrcb1, nrcb2, nrcb3, nrcb4
# - Long-wavelength (F335M, F470N, F480M): nrcb only
#
# gc2211 2211/050 is NRCB-only in both of its filters, counted from the `_cal`
# frames on disk 2026-08-22 (issue #436): F200W 48 frames, nrcb1-4 only; F277W
# 12 frames, nrcblong only.  Its four sibling observations (023/028/046/049)
# carry both modules.  Asking for nrca stops the reduce ~41 s in with `No nrca
# members found in ... jw02211-o050_..._asn.json`; with this entry the request
# is narrowed here instead, and `--modules merged` raises immediately (no nrca
# means no merged mosaic to produce).
#
# The families are spelled once each rather than detector-by-detector: every
# consumer maps the entry through `_module_group` / `module_family`
# (`get_allowed_modules` below, `preflight_reduce_inputs.allowed_modules`), and
# the main loop makes one `main()` pass per surviving list entry -- so the
# four-detector spelling above costs sickle four nrcb-family passes over the
# same members (see `test_a_repeated_module_family_fits_each_ramp_once`, which
# pins that the stage-1/2 memo still fits each ramp once).
MODULES_BY_PROPOSAL_FIELD_FILTER = {
    '3958': {
        '007': {
            'F187N': ('nrcb1', 'nrcb2', 'nrcb3', 'nrcb4'),
            'F210M': ('nrcb1', 'nrcb2', 'nrcb3', 'nrcb4'),
            'F335M': ('nrcb',),
            'F470N': ('nrcb',),
            'F480M': ('nrcb',),
        }
    },
    '2211': {
        '050': {
            'F200W': ('nrcb',),
            'F277W': ('nrcb',),
        }
    },
}


# Per-filter overrides for the (proposal_id, field) refcat lookup.  Lets
# us hand a different refcat to one specific filter (e.g. brick-1182
# F115W tweakreg, where the F405N-based refcat has poor blue-star
# overlap).  Lookup precedence is filter > field default.
# Which reference catalog each observation ties to is registered in
# jwst_gc_pipeline/fields.yaml, alongside everything else about a field.
# See docs/FIELDS.md.


def get_reference_astrometric_catalog_path(basepath, proposal_id, field, filtername=None):
    """Where this observation's absolute reference catalog lives.

    Registered in fields.yaml; the error names the block to add for a field
    that has none yet.
    """
    return field_registry.reference_catalog_path(
        proposal_id, field, filtername=filtername, basepath=basepath)


def get_existing_reference_astrometric_catalog_path(basepath, proposal_id, field, filtername=None):
    """The registered reference catalog, once its file is confirmed on disk.

    NOTE ON WHAT THIS IS FOR.  TweakRegStep is skipped on every NIRCam path
    here (``tweakreg_parameters['skip'] = True``), so this catalog does NOT
    realign anything during Image3.  NIRCam's absolute tie is applied per
    exposure by ``fix_alignment`` from an offsets table, and that TABLE is what
    the catalog produced: ``build_virac2_offsets.py`` measures the per-visit
    consensus against this file to make it, and the m2 astrometry checkpoint
    re-ties against it.  It is read here so the run records which reference it
    belongs to, and so a field wired to a catalog that was never built stops
    before producing products whose provenance names a missing file.

    ``None`` when the observation has no entry in fields.yaml AND its frame
    does not depend on one -- a field whose absolute zero point is set some
    other way (m4, m92, ngc6397, w51, wd1 obs003, wd2 obs003).

    NOT registered but REQUIRED raises instead.  ``alignment_config.
    reference_catalog_required`` is the one rule for that, and the m2
    checkpoint asks it too: a VIRAC2-framed field's tie is MADE against this
    catalog, so with none registered the reduce would run to completion and
    the FIRST refusal would come hours later at m2.  Program 10678 is 139
    such tiles, reduced by an automated trigger
    (``data_qa.pipeline_trigger`` -> ``submit_reduction.sbatch`` -> this
    module) that never passes through ``run_pipeline.build_plan``, where the
    plan-time refusal lives.  The message names the block to add.

    A registered catalog whose FILE is absent raises: the seeds live outside
    the repo, so a typo or a not-yet-run build must abort rather than reduce
    with an unverifiable frame.
    """
    from jwst_gc_pipeline.reduction import alignment_config as _ac
    try:
        path = field_registry.reference_catalog_path(
            proposal_id, field, filtername=filtername, basepath=basepath)
    except (field_registry.FieldRegistryError, KeyError) as exc:
        if not _ac.reference_catalog_required(proposal_id, field):
            return None
        raise field_registry.FieldRegistryError(
            f"No reference astrometric catalog is REGISTERED for "
            f"proposal_id={proposal_id} field={field}, and this field's "
            f"alignment config declares the VIRAC2 frame, so its tie is made "
            f"against that catalog.  Reducing without it would produce "
            f"products at the raw assign_wcs frame and stop only at the m2 "
            f"astrometry checkpoint, after the full reduce.  Register it in "
            f"fields.yaml under the field's observations.{proposal_id!r} "
            f"block:\n"
            f"        reference_catalog:\n"
            f"          {str(field)!r}: catalogs/<your-refcat>.fits\n"
            f"For program 10678 the per-tile catalogs and the block to paste "
            f"come from scripts/reduction/build_treasury_refcats.py.\n"
            f"Registry said: {exc}") from exc
    if os.path.exists(path):
        return path
    raise FileNotFoundError(
        f"Configured reference astrometric catalog is MISSING: {path} "
        f"(proposal_id={proposal_id} field={field}). Build the seed "
        f"(build_gaia_virac2_refcat_byquery.py) or fix the entry in "
        f"fields.yaml before reducing -- refusing to run first-pass off-frame.")


def _drop_excluded_asn_members(candidate, cand_data, members):
    """``(members, asn_path)`` with excluded exposures removed.

    Writes a SIBLING file, never in place.  MAST's association is the record of
    what was observed: rewriting it would make this read path a writer, would
    not be idempotent, and would destroy the evidence that the exposure was ever
    part of the product.  When nothing is excluded -- the overwhelmingly common
    case -- the original path is returned untouched and nothing is written.
    """
    import json as _json
    import os as _os

    from jwst_gc_pipeline.reduction.exposure_exclusions import (
        drop_excluded, is_excluded)

    kept = [m for m in members if not is_excluded(m.get('expname', ''))]
    if len(kept) == len(members):
        return members, candidate

    drop_excluded([m.get('expname', '') for m in members],
                  label=f'asn {_os.path.basename(candidate)}')
    out = candidate.replace('_asn.json', '_exclfiltered_asn.json')
    filtered = _json.loads(_json.dumps(cand_data))   # never mutate the caller's dict
    filtered['products'][0]['members'] = kept
    with open(out, 'w') as fh:
        _json.dump(filtered, fh, indent=4)
    return kept, out


def _module_group(module):
    if module == 'merged':
        return 'merged'
    if module.startswith('nrca'):
        return 'nrca'
    if module.startswith('nrcb'):
        return 'nrcb'
    return module


def registry_obs_key(field):
    """``field`` spelled the way the module registry keys an observation.

    MAST, the association names and ``MODULES_BY_PROPOSAL_FIELD_FILTER`` all
    spell an observation with three digits (``'007'``, ``'050'``), while
    ``--field`` is taken as typed, so ``7`` and ``07`` name the same
    observation and miss the registry.  A miss here is SILENT: `allowed_modules`
    stays None and the request is returned unrestricted with nothing printed,
    so an unpadded field asks for both modules on an observation that has only
    one and the reduce goes looking for members that were never taken (issue
    #438, sickle 3958/007 and gc2211 2211/050).

    Anything that is not a plain number is handed back untouched -- the
    wildcard obsid ``'*'`` a not-yet-executed program registers, and any
    non-numeric field spelling -- so this only ever narrows the gap between two
    spellings of the same number.
    """
    text = str(field).strip()
    return f'{int(text):03d}' if text.isdigit() else text


def get_allowed_modules(proposal_id, field, requested_modules, filtername=None):
    allowed_modules = None
    field_key = registry_obs_key(field)

    # Check for filter-specific policy first
    if proposal_id in MODULES_BY_PROPOSAL_FIELD_FILTER:
        if field_key in MODULES_BY_PROPOSAL_FIELD_FILTER[proposal_id]:
            field_policy = MODULES_BY_PROPOSAL_FIELD_FILTER[proposal_id][field_key]
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
         skymatch_method=None, skip_outlier_detection=True):
    """
    skip_step1and2 will not re-fit the ramps to produce the _cal images.  This
    can save time if you just want to redo the tweakreg steps but already have
    the zero-frame stuff done.
    """
    # `field` is canonicalised ONCE, here, because every glob and every product
    # name below is interpolated from it: the association search
    # (`jw02221-o{field}*_image3_*asn.json`), the uncal download filter
    # (`jw02221{field}*_uncal.fits`), the drizzle product names
    # (`...-o{field}_t001_nircam_clear-...`) and the `_o{field}_crf` frames.
    # MAST spells an observation with three digits, so `--field 1` built
    # `jw02221-o1*`, which matches nothing on disk and at MAST; the run then
    # stopped at "Did not find any NIRCam asn files" -- loud, for the wrong
    # stated reason -- or re-entered the MAST call.  Padding at the entry point
    # keeps the two spellings of one observation from diverging below this line.
    # `registry_obs_key` hands a non-number back untouched, so a joint
    # registration ('002-998') and the wildcard obsid are unchanged; the
    # cataloging side has the stricter `photometry.naming.observation_field_token`,
    # which refuses those instead.  Issue #438.
    field = registry_obs_key(field)
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
    from jwst_gc_pipeline.reduction.destreak_policy import destreaks
    _was = do_destreak
    do_destreak = destreaks(regionname, filtername, do_destreak)
    if _was and not do_destreak:
        print(f"Region {regionname} filter {filtername}: destreak off "
              f"(see reduction/destreak_policy.py); the working copy is a "
              f"plain _cal -> _align.fits copy.", flush=True)

    wavelength = int(filtername[1:4])

    # The field's data directory comes from the registry (fields.yaml `roots:`
    # plus the field's `root:`), so a field on a tree other than /orange reduces
    # where it lives.
    basepath = field_registry.basepath(regionname)
    # Non-destructive experimental reduction: same GC_BASEPATH_OVERRIDE redirect as
    # the cataloging driver (jwst_gc_pipeline.scratch_basepath).  With a scratch
    # tree staged (stage_scratch_basepath.sh, MODE=reduce) with symlinks to the
    # real _cal inputs, a re-reduction (e.g. a consensus retie loop) writes
    # crf/mosaics into scratch and never overwrites released products.  output_dir
    # below derives from basepath, so it follows the redirect too.  Empty -> normal.
    from jwst_gc_pipeline.scratch_basepath import apply_basepath_override
    _bp0 = basepath
    basepath = apply_basepath_override(basepath)
    if basepath != _bp0:
        print(f"GC_BASEPATH_OVERRIDE active (reduction): basepath -> {basepath}", flush=True)
    from jwst_gc_pipeline.reduction.fwhm import fwhm_table_path
    fwhm_tbl = Table.read(fwhm_table_path(basepath))
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
    output_dir = f'{basepath}{filtername}/pipeline/'
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
    existing_asn = glob(os.path.join(output_dir, f'{jw_prefix(proposal_id)}-o{field}*_image3_*0[0-9][0-9]_asn.json'))
    existing_uncal = glob(os.path.join(output_dir, f'{jw_prefix(proposal_id)}{field}*_uncal.fits'))
    mast_needed = (len(existing_asn) == 0) or (not skip_step1and2 and len(existing_uncal) == 0)
    if not mast_needed:
        print(f"Skipping MAST query for {filtername}: {len(existing_asn)} asn json(s) and "
              f"{len(existing_uncal)} uncal already on disk (skip_step1and2={skip_step1and2})")
    else:
        Observations.TIMEOUT = 120  # seconds; avoid indefinite hang on a stalled MAST connection
        obs_table = Observations.query_criteria(
                                                proposal_id=proposal_id,
                                                # JWST only: a proposal NUMBER is
                                                # not unique across missions.  9438
                                                # is both a JWST program (Schlafly,
                                                # NIRCam) and an HST one (West,
                                                # WFPC2/ACS of ABELL1185), and the
                                                # unscoped query returned all 211
                                                # rows of both.  Downstream filter/
                                                # instrument cuts happened to drop
                                                # the HST rows, but nothing
                                                # guaranteed that.
                                                obs_collection='JWST',
                                                #proposal_pi="Ginsburg*",
                                                #calib_level=3,
                                                )
        print("Obs table length:", len(obs_table))

        # np.array wrapper needed as of 2026-04-10 to avoid masked array type error that shouldn't happen
        msk = ((np.char.find(np.array(obs_table['filters']), filtername.upper()) >= 0) |
               (np.char.find(np.array(obs_table['obs_id']), filtername.lower()) >= 0))
        # Restrict to the observation under reduction (issue #416): all 139
        # gc-treasury tiles share FILTERS='F212N;F480M', so the filter mask
        # alone selects every released observation and each fresh tile would
        # download the whole program's asn products.  The table is queried per
        # PROPOSAL, so this narrows the two-field proposals as well (2221 =
        # brick o001 + cloudc o002, 3958 = brick + sickle, 2045 = arches +
        # quintuplet): a brick reduce stops pulling the other field's asn
        # products into brick's output_dir.  The asn glob below is already
        # -o{field}-scoped, so the narrowing removes download volume.
        msk &= observation_scope_mask(np.array(obs_table['obs_id']),
                                      proposal_id, field)
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

    # Ramp files.  Only stage 1+2 read them, so -s / skip_step1and2 skips the
    # download -- otherwise a run that only wanted the association file above
    # pulled the program's whole uncal set (~1.2 GB for one filter) as well.
    if mast_needed and not skip_step1and2:
        products_fits = Observations.filter_products(data_products_by_obs, extension="fits")
        print("products_fits length:", len(products_fits))
        # TODO(#438): `field` is used RAW here while the obs mask above pads it
        # through `observation_number`, so `--field 1` narrows the obs table to
        # jw10678-o001 correctly and then this substring test looks for
        # `jw106781` and keeps nothing.  #438 normalises --field once, at the
        # driver's entry, for this site and the asn glob below.
        uncal_mask = np.array([
            uri.endswith('_uncal.fits')
            and f'{jw_prefix(proposal_id)}{field}' in uri
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
    #
    # TODO(#438): the association glob below uses `field` RAW, while the obs
    # mask above pads it through `observation_number`: `--field 1` downloads
    # jw10678-o001's association and then globs `-o1*`, which matches nothing.
    # #438 normalises --field once, at the driver's entry, for both sites.
    if module in ('nrca', 'nrcb', 'merged'):
        print(f"Working on module {module}: running initial pipeline setup steps (skip_step1and2={skip_step1and2})")
        print(f"Searching for {os.path.join(output_dir, f'{jw_prefix(proposal_id)}-o{field}*_image3_*0[0-9][0-9]_asn.json')}")
        asn_file_search = glob(os.path.join(output_dir, f'{jw_prefix(proposal_id)}-o{field}*_image3_*0[0-9][0-9]_asn.json'))
        # Filter out non-NIRCam asn files (e.g. NIRISS asns from same proposal/obs).
        # Members of NIRCam asns have 'nrc' in expname (nrca/nrcb/nrcalong/nrcblong); NIRISS members are '_nis_'.
        nircam_asn_files = []
        for candidate in asn_file_search:
            try:
                with open(candidate) as fh:
                    cand_data = json.load(fh)
                members = cand_data.get('products', [{}])[0].get('members', [])
                # Excluded exposures must not reach the drizzle either.  The
                # instruction is "all imaging AND analysis"; an exposure dropped
                # from cataloging but still coadded into the mosaic leaves the
                # catalog and the image disagreeing about what was observed.
                members, candidate = _drop_excluded_asn_members(
                    candidate, cand_data, members)
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

        crds_dir = os.getenv("CRDS_PATH") or os.path.join(basepath, 'crds')
        mapping = crds.rmap.load_mapping(f'{crds_dir}/mappings/jwst/jwst_nircam_pars-tweakregstep_0003.rmap')
        print(f"Mapping: {mapping.todict()['selections']}")
        print(f"Filtername: {filtername}")
        filter_match = [x for x in mapping.todict()['selections'] if filtername in x]
        print(f"Filter_match: {filter_match} n={len(filter_match)}")
        tweakreg_asdf_filename = filter_match[0][4]
        tweakreg_asdf = open_crds_reference(crds_dir, 'nircam',
                                            tweakreg_asdf_filename)
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
                # This whole-asn provenance check runs on every member of
                # every pass, above the per-module scoping below, so a foreign
                # member from a mis-globbed asn is caught even on a field
                # whose module policy narrows to a single module.
                assert f'{jw_prefix(proposal_id)}{field}' in member['expname']
                # #417: each module pass claims only its own members, with the
                # same substring semantics as the per-module member trim in
                # the tweakreg block below ('nrca' claims nrca1-4 + nrcalong).
                # The merged pass keeps every NIRCam member so a merged-only
                # or single-module run still produces every _cal it needs; on
                # the default nrca,nrcb,merged sequence the in-process memo
                # below skips the members the module passes already produced.
                if not member_in_stage12_pass(member['expname'], module):
                    print(f"Skipping member {member['expname']}: the {module} pass does not claim it")
                    continue
                uncal_fn = member['expname'].replace("_cal.fits", "_uncal.fits")
                # #417: skip a member THIS process already ran stage 1+2 on in
                # an earlier module pass (and, only under STAGE12_RESUME=1, one
                # whose products on disk are newer than its uncal).  A fresh
                # process with SKIP=0 still re-fits every ramp.
                skip_stage12 = stage12_skip_reason(uncal_fn,
                                                  require_ramp=SAVE_CALIBRATED_RAMP)
                if skip_stage12:
                    print(f"Skipping stage 1+2 for {member['expname']}: {skip_stage12}")
                    continue
                print(f"DETECTOR PIPELINE on {member['expname']}")
                print("Detector1Pipeline step")
                # from Hosek: expand_large_events -> false; turn off "snowball" detection
                Detector1Pipeline.call(uncal_fn,
                                       save_results=True, output_dir=output_dir,
                                       save_calibrated_ramp=SAVE_CALIBRATED_RAMP,
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
                # #417: a later module pass in this same interpreter must not
                # redo the member we just calibrated.
                note_stage12_processed(uncal_fn)
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
        asn_data['products'][0]['name'] = f'{jw_prefix(proposal_id)}-o{field}_t001_nircam_clear-{filtername.lower()}-{module}'
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
                  f"{proposal_id} field={field}.  TweakRegStep is skipped either "
                  f"way; this only means the run records no reference frame.  "
                  f"The applied tie comes from the offsets table.", flush=True)
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
        # TweakRegStep is retired for NIRCam: the tie is applied per exposure by
        # fix_alignment, exactly once, so the _crf frames and the _i2d mosaic
        # inherit the same solution.  abs_refcat is set anyway so the step's
        # parameters record which reference the run belongs to.
        tweakreg_parameters.update({'skip': True})

        if regionname in ('brick', 'cloudc'):
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
                      f"{proposal_id} field={field}.  TweakRegStep is skipped "
                      f"either way; this only means the run records no reference "
                      f"frame.  The applied tie comes from the offsets table.",
                      flush=True)
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
            # Retired for NIRCam -- see the note at the other skip site: the tie
            # is applied per exposure by fix_alignment, not here.
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

        # outlier_detection: SKIPPED by default on these crowded GC fields (#161).
        # Diagnosis (PR #180) established the step over-flags real bright-star PSF
        # signal -- diffraction spikes and the dark inter-spike gaps -- as OUTLIER,
        # punching NaN holes into the _crf/_i2d that cataloging then loses.  The
        # cause is a MIS-SPECIFIED VARIANCE MODEL, not a tunable threshold:
        # outlier_detection compares each exposure to the resampled-stack median
        # with a tolerance built from ERR (photon+read noise only), but an
        # UNDERSAMPLED PSF sampled at different sub-pixel dither phases legitimately
        # disperses 5-9x ERR wherever the PSF is steep (0.98x ERR on flat sky ->
        # 9.4x on the spikes).  Decisive test: two exposures at the SAME dither
        # pointing agree at the ERR level even at the flagged pixels, while
        # exposures at DIFFERENT pointings differ ~9x more -- which excludes every
        # per-frame defect (cosmic rays, persistence, brighter-fatter, ramp
        # nonlinearity are all independent per exposure and would show up
        # within-pointing too; they do not).  Raising snr/scale (closed PR #163)
        # only rescales the wrong tolerance and would suppress genuine CRs equally.
        # Cosmic rays are already rejected per-ramp by JumpStep in Detector1
        # (independent, and this issue's ramp analysis found <1% genuine jumps
        # among the flagged pixels), so dropping the inapplicable image-space
        # comparison costs little.  Re-enable with --run-outlier-detection if a
        # field is sparse enough that residual (post-JumpStep) CRs dominate.
        if skip_outlier_detection:
            image3_steps['outlier_detection'] = {'skip': True}
            print(f"outlier_detection SKIPPED (#161; JumpStep handles CRs) ({module})")
        else:
            print(f"outlier_detection ENABLED at pipeline defaults ({module})")

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
        #   jw{prop:05d}-o{field}_t001_nircam_clear-{filt}-{module}_<N>_o{field}_crf.fits
        # but the manual cataloging globs PER-EXPOSURE crf with the destreak/align
        # suffix
        #   jw{prop:05d}{field}{visit}_..._{module}_{align|destreak}_o{field}_crf.fits .
        # Those never matched, so a corrected re-reduction's crf (e.g. the skymatch
        # background fix) silently never reached cataloging -- the per-exposure
        # *_{align|destreak}_o{field}_crf.fits stayed at the OLD reduction's mtime.
        # Map product-named crf -> per-exposure names by EXPSTART (1:1) and copy
        # into place.  member['expname'] already carries the _align/_destreak
        # suffix (set in the destreak/align loop above), so the target name matches
        # what cataloging --each-suffix consumes.
        #
        # ORDER MATTERS: outlier_detection is the ONLY step that emits product-named
        # crf, so when it is skipped THIS run wrote none and any on disk are
        # leftovers from an older reduction.  Copying those forward overwrites the
        # per-exposure crf with a previous generation's WCS while refreshing their
        # mtime -- invisible to every mtime-based staleness check, and cataloging
        # then photometers the old alignment.  sickle hit exactly this (#270): 96
        # product crf from 2026-06-27 (its last run with outlier_detection on) were
        # copied over the per-exposure names on every iteration of the VIRAC2 retie,
        # so all 96 carried one constant GNS RAOFFSET while their aligned
        # `_destreak.fits` inputs carried the new per-exposure VIRAC2 tie ~200 mas
        # away.  The loop re-measured the same ~110 mas gap every iteration and
        # could not converge.  So test skip_outlier_detection FIRST.
        _prod_name = asn_data['products'][0]['name']
        _prod_crf = sorted(glob(os.path.join(
            output_dir, f'{_prod_name}_*_o{field}_crf.fits')))
        if _prod_crf and not skip_outlier_detection:
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
        elif skip_outlier_detection:
            # outlier_detection is the step that emits the CR-flagged crf; with it
            # skipped (#161) Image3 writes none, so cataloging's per-exposure
            # *_o{field}_crf.fits glob would starve (or silently reuse a stale
            # reduction's crf).  tweakreg is also skip=True here (alignment was done
            # upstream -- members already carry the final WCS), so the correct crf
            # is just the member frame itself: same SCI/ERR/WCS, DQ WITHOUT the
            # spurious OUTLIER flags.  Copy each member -> its per-exposure crf name.
            #
            # Reached whether or not product-named crf happen to sit in output_dir:
            # if they do, they are an older reduction's and the branch above
            # deliberately declines them.  This is the only correct source for the
            # crf on a skip_outlier_detection run.
            if _prod_crf:
                print(f"  {len(_prod_crf)} product-named crf are on disk but "
                      f"outlier_detection is SKIPPED, so THIS run wrote none -- they "
                      f"are an EARLIER reduction's and carry its WCS. NOT copying "
                      f"them forward (#270); writing crf from this run's aligned "
                      f"member frames instead. First stale file: "
                      f"{os.path.basename(_prod_crf[0])}", flush=True)
            if skymatch_method:
                print("  WARNING: --skymatch-method set WITH outlier_detection "
                      "skipped: the per-exposure crf are copied from the PRE-skymatch "
                      "member frames (skymatch's subtraction is applied in-memory and "
                      "only reaches the resampled i2d, not these copies).", flush=True)
            _n_crf = 0
            for member in asn_data['products'][0]['members']:
                _mb = os.path.basename(member['expname'])
                _src = (member['expname'] if os.path.exists(member['expname'])
                        else os.path.join(output_dir, _mb))
                _target = os.path.join(
                    output_dir, _mb.replace('.fits', f'_o{field}_crf.fits'))
                if not os.path.exists(_src):
                    print(f"  WARNING: member frame {_mb} missing; crf NOT written",
                          flush=True)
                    continue
                shutil.copy(_src, _target)
                _n_crf += 1
            print(f"  outlier_detection skipped: wrote {_n_crf} per-exposure crf as "
                  f"copies of the aligned member frames (no OUTLIER flags added)",
                  flush=True)
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
        #     remove_saturated_stars(f'{jw_prefix(proposal_id)}-o{field}_t001_nircam_clear-{filtername.lower()}-{module}_i2d.fits')
        #     if did_vvv_realign:
        #         remove_saturated_stars(f'{jw_prefix(proposal_id)}-o{field}_t001_nircam_clear-{filtername.lower()}-{module}{destreak_suffix}_realigned-to-vvv.fits')
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
        #    remove_saturated_stars(f'{jw_prefix(proposal_id)}-o{field}_t001_nircam_clear-{filtername.lower()}-{module}-reproject_i2d.fits')
        #    remove_saturated_stars(f'{jw_prefix(proposal_id)}-o{field}_t001_nircam_clear-{filtername.lower()}-{module}_realigned-to-vvv.fits')
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

        asn_data['products'][0]['name'] = f'{jw_prefix(proposal_id)}-o{field}_t001_nircam_clear-{filtername.lower()}-merged'
        asn_file_merged = asn_file.replace("_asn.json", f"_merged_asn.json")
        with open(asn_file_merged, 'w') as fh:
            json.dump(asn_data, fh)

        # don't re-fit to VVV - it's not accurate enough with the JWST-derived
        # catalogs.  We needed to use our own much more extensive cataloging to
        # beat down the noise enough to make this approach viable
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
            print(f"No configured reference catalog found for proposal_id={proposal_id} "
                  f"field={field} in {basepath}.  TweakRegStep is skipped either way; "
                  f"this only means the run records no reference frame.  The applied "
                  f"tie comes from the offsets table.")

        if abs_refcat is not None:
            tweakreg_parameters.update({'abs_refcat': abs_refcat,})

        # 'with tweakreg' names the STEP that is configured, not one that runs:
        # tweakreg_parameters carries skip=True on every NIRCam path.
        print("Running Image3Pipeline on the merged association "
              "(tweakreg configured but skipped; the tie is already baked in)")
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

        _fov = field_registry.fov_region(regionname)
        vvv_region_file = f"{basepath}/{_fov}" if _fov else None
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
        #     remove_saturated_stars(f'{jw_prefix(proposal_id)}-o{field}_t001_nircam_clear-{filtername.lower()}-merged_i2d.fits')
        #     remove_saturated_stars(f'{jw_prefix(proposal_id)}-o{field}_t001_nircam_clear-{filtername.lower()}-{module}{destreak_suffix}_realigned-to-vvv.fits')
        # except (TimeoutError, requests.exceptions.ReadTimeout) as ex:
        #     print("Failed to run remove_saturated_stars with failure {ex}")

    globals().update(locals())
    return locals()


# NOTE: _apply_consensus_offsets_table and the module-level
# _VALIDATED_OFFSETS_TABLES set lived here.  Both moved into
# jwst_gc_pipeline/reduction/unified_alignment.py when the per-proposal
# dispatch collapsed; leaving the originals behind would have implied the
# consensus path still routes through this file, which it does not.

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

    if os.environ.get('STATIC_PLACEMENT_CORRECTION', '0') == '1':
        # Static per-detector SIAF placement field measured by the network
        # self-calibration (1-2.5 mas, SW detectors only).  OPT-IN: the field
        # was measured in sky coordinates on GC pointings and is only valid at
        # GC-survey-like position angles (see static_placement_correction.py).
        from jwst_gc_pipeline.reduction.static_placement_correction import (
            apply_placement_correction)
        apply_placement_correction(fn)

    mod = ImageModel(fn)
    if proposal_id is None:
        # Read the proposal out of the frame's own PROGRAM header, which
        # travels inside the file: a renamed or copied frame still reports the
        # proposal it was observed under (issue #440).  `fn` is the fallback
        # for a model with no PROGRAM, parsed five digits wide -- the [3:7]
        # slice that preceded it read '0678' off a jw10678 product (#414).
        proposal_id = proposal_id_from_datamodel(mod, fn)
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
    # The second `if proposal_id is None:` that stood here read the program
    # number off `mod` as a late fallback.  The assignment above now reads the
    # same header first and raises when neither source names a proposal, so
    # the branch could never run.
    if basepath is None:
        # Every in-pipeline caller passes basepath.  Reaching here means an
        # ad-hoc call, so name the field from the registry rather than guessing
        # a path: the guess used to be the observation number, which is a
        # directory that never exists.
        basepath = field_registry.basepath(
            field_registry.target_for_obsid(proposal_id, field))
    if module is None:
        module = 'nrc' + mod.meta.instrument.module.lower()

    # ---------------------------------------------------------------------
    # ONE path for every field.  Which reference frame and which shift source
    # applies to this (proposal, observation) is declared in
    # jwst_gc_pipeline/reduction/alignment_config.py; resolving it -- reading the
    # table, narrowing to this exposure, checking the WCS generation -- happens
    # once, in unified_alignment.resolve_shift, for all of them.
    #
    # This replaces a per-proposal if/elif chain whose `else` returned (0, 0).
    # Any proposal without an explicit branch was silently left unaligned while
    # the m2 checkpoint wrote corrections into a table nothing read, so a re-tie
    # loop on such a field re-measured the same residual forever (arches/2045,
    # quintuplet/2045, sgrb2/5365, cloudef/2092 obs 005 were all in that state).
    # An unconfigured field still gets (0, 0), but now says so loudly and is
    # distinguishable from a genuine zero tie via `_shift.configured`.
    # ---------------------------------------------------------------------
    from jwst_gc_pipeline.reduction.unified_alignment import (
        resolve_shift, warn_or_raise_if_stale, write_alignment_header,
        alignment_apply_plan, APPLY_FULL, APPLY_DELTA, SKIP_STALE,
        PREV_RA_KEY, PREV_DEC_KEY, NREALIGN_KEY)
    _shift = resolve_shift(fn, proposal_id, field, filtername, module, basepath,
                           refname=field_registry.reference_frame(str(proposal_id)),
                           use_average=use_average)
    rashift = _shift.ra_quantity
    decshift = _shift.dec_quantity
    _prov_tbl = _shift.prov_table          # offsets table actually consumed
    _prov_row_stage = _shift.prov_stage    # checkpoint stage that last corrected it
    _frame_gen = _shift.frame_generation   # this frame's WCS-generation stamp
    # Check the frame token before anything is written.  The provenance stamp
    # below refuses a placeholder, and it sits between the GWCS save and the
    # RAOFFSET write -- raising there would leave the shift baked in with no
    # RAOFFSET, which the idempotency guard reads as "not yet aligned".
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        assert_not_placeholder)
    assert_not_placeholder(_shift.reference_frame,
                           f'the astrometric reference frame for {fn}')
    print(f"Shift for {fn} is {_shift}")
    align_fits = fits.open(fn)
    # `_delta` is None on the first apply (the frame carries no correction) and
    # is the SHIFT STILL OWED when the frame is already corrected but its table
    # row has moved underneath it.
    # DISAGREEMENT POLICY: the plain skip-if-present check silently KEPT a stale
    # RAOFFSET after the offsets table was corrected -- brick-1182 v001 crf held
    # +1.9" while the table said -17.5", so half the mosaic stayed ~20" off and
    # the idempotent guard blocked its own fix. The header records what WAS
    # applied and the table says what SHOULD be, so a stale frame does not need
    # regenerating from _cal to move: apply the DIFFERENCE and rewrite the
    # totals (issue #274). Frames carrying the component keywords are compared
    # PER COMPONENT, so a re-measured bulk can no longer be masked by an
    # opposite jitter change that happens to sum back to the same total.
    _verdict, _delta, _why = alignment_apply_plan(align_fits[1].header, _shift, fn)
    if 'RAOFFSET' in align_fits[1].header:
        # don't shift twice if we re-run
        print(f"{fn} is already aligned ({align_fits[1].header['RAOFFSET']}, {align_fits[1].header['DEOFFSET']})")
    if _verdict == APPLY_DELTA:
        print(f"STALE ASTROMETRY -- RE-CORRECTING {fn} by the delta: "
              f"baked ({_delta.baked_ra:+.4f},{_delta.baked_dec:+.4f})\" -> "
              f"table ({_shift.total_ra:+.4f},{_shift.total_dec:+.4f})\" "
              f"= applying ({_delta.dra:+.4f},{_delta.ddec:+.4f})\".", flush=True)
    elif _verdict == SKIP_STALE:
        print(f"NOT re-correcting {fn}: {_why}", flush=True)
        warn_or_raise_if_stale(align_fits[1].header, _shift, fn)
        _delta = None
    if _verdict in (APPLY_FULL, APPLY_DELTA):
        # What goes into the WCS is the delta when re-correcting and the whole
        # shift on a first apply.  What goes into RAOFFSET below is the TOTAL in
        # both cases, so the keyword keeps meaning "the correction this frame
        # carries" and the next run's idempotency check reads it unchanged.
        _apply_ra = (_delta.dra * u.arcsec) if _delta is not None else rashift
        _apply_dec = (_delta.ddec * u.arcsec) if _delta is not None else decshift
        # ASDF header
        fa = ImageModel(fn)
        wcsobj = fa.meta.wcs
        print(f"Before shift, crval={wcsobj.to_fits()[0]['CRVAL1']}, {wcsobj.to_fits()[0]['CRVAL2']}, {wcsobj.forward_transform.param_sets[-1]}")
        fa.meta.oldwcs = copy.copy(wcsobj)
        ww = adjust_wcs(wcsobj, delta_ra=_apply_ra, delta_dec=_apply_dec)
        print(f"After shift, crval={ww.to_fits()[0]['CRVAL1']}, {ww.to_fits()[0]['CRVAL2']}, {wcsobj.forward_transform.param_sets[-1]}")
        fa.meta.wcs = ww
        fa.save(fn, overwrite=True)

        # BASE -> TARGET proof (2026-07-13): record the fiducial-pixel sky
        # coordinate BEFORE the correction (the coordinate the offset applies
        # to) and AFTER it (the coordinate it must produce), then VERIFY
        # target == base + coordinate-shift.  With these stamped, any later
        # reader can re-derive and re-check the correction no matter when --
        # and a correction can never be silently applied to the wrong base.
        # Fiducial = ARRAY CENTER, not a hardcoded (1024,1024): subarray frames
        # (e.g. sickle SUB640, 640x640) have no pixel 1024 -> the GWCS returns
        # NaN outside its bounding box and the header write below rejects it.
        _fy, _fx = fa.data.shape[0] / 2.0, fa.data.shape[1] / 2.0
        _base_ra, _base_dec = float(wcsobj.pixel_to_world(_fx, _fy).ra.deg), \
            float(wcsobj.pixel_to_world(_fx, _fy).dec.deg)
        _tgt_ra, _tgt_dec = float(ww.pixel_to_world(_fx, _fy).ra.deg), \
            float(ww.pixel_to_world(_fx, _fy).dec.deg)
        _exp_ra = _base_ra + _apply_ra.to(u.deg).value   # COORDINATE convention
        _exp_dec = _base_dec + _apply_dec.to(u.deg).value
        if not np.isfinite([_base_ra, _base_dec, _tgt_ra, _tgt_dec]).all():
            raise RuntimeError(
                f"astrometric apply for {fn}: fiducial pixel ({_fx},{_fy}) maps to a "
                f"non-finite sky coordinate (base={_base_ra},{_base_dec}). The WCS "
                f"bounding box likely excludes the array center; NOT writing.")
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
        # OLCRVAL must keep pointing at the UNCORRECTED sky: on a delta
        # re-correction the current CRVAL already carries the old shift, so
        # re-stamping it would lose the only record of where the frame started.
        if 'OLCRVAL1' not in align_fits[1].header:
            align_fits[1].header['OLCRVAL1'] = align_fits[1].header['CRVAL1']
            align_fits[1].header['OLCRVAL2'] = align_fits[1].header['CRVAL2']
        # NOT ``header.update(ww.to_fits()[0])``: gwcs's to_fits defaults to
        # max_pix_error=0.25 px, which fitted a degree-3 SIP disagreeing with
        # the GWCS by up to 5.5 mas (SW) / 6.6 mas (LW) -- on top of the 2 mas
        # m2 and 5 mas m7 astrometric tolerances -- and merged over the
        # delivered degree-4 fit, orphaning its high-order coefficients.
        # sync_header_to_gwcs strips stale SIP, fits at 0.01 px, and VERIFIES.
        _sip_max, _sip_med = sync_header_to_gwcs(
            align_fits[1].header, ww, fa.data.shape, label=os.path.basename(fn))
        print(f"FITS/SIP header synced to GWCS: max {_sip_max:.4f} mas, "
              f"median {_sip_med:.4f} mas")
        align_fits[1].header['SIPGWMAX'] = (
            _sip_max, '[mas] max FITS/SIP vs GWCS disagreement')
        # total (historical keywords, unchanged meaning) + the bulk/jitter split
        write_alignment_header(align_fits[1].header, _shift)
        # correction provenance: base/target fiducials + convention + the
        # generation this frame carried when corrected (audit at any time:
        # recompute pixel_to_world(ABASEPX,ABASEPY) and compare to ATGTRA/ATGTDE)
        align_fits[1].header['ABASEPX'] = (_fx, 'fiducial pixel x (array center) for ABASE/ATGT')
        align_fits[1].header['ABASEPY'] = (_fy, 'fiducial pixel y (array center) for ABASE/ATGT')
        align_fits[1].header['ABASERA'] = (_base_ra, f'[deg] fiducial({_fx},{_fy}) BEFORE correction')
        align_fits[1].header['ABASEDE'] = (_base_dec, '[deg] fiducial dec BEFORE correction')
        align_fits[1].header['ATGTRA'] = (_tgt_ra, '[deg] fiducial AFTER correction (verify me)')
        align_fits[1].header['ATGTDE'] = (_tgt_dec, '[deg] fiducial dec AFTER correction')
        align_fits[1].header['AOFFCONV'] = ('coordinate', 'RAOFFSET is dra_coordinate (on-sky = *cos(dec))')
        align_fits[1].header['AVERMAS'] = (_resid_mas, '[mas] base+shift vs target residual (proof)')
        if _delta is not None:
            # Keep the delta step as reversible as the first apply: record what
            # the frame carried before it and how many times it has moved.
            align_fits[1].header[PREV_RA_KEY] = (
                _delta.baked_ra, 'arcsec, total dRA before this re-correction')
            align_fits[1].header[PREV_DEC_KEY] = (
                _delta.baked_dec, 'arcsec, total dDec before this re-correction')
            align_fits[1].header[NREALIGN_KEY] = (
                int(align_fits[1].header.get(NREALIGN_KEY, 0)) + 1,
                'delta re-corrections applied to this frame')
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
                # The frame this exposure was actually tied to, from the shift
                # that was just resolved.  alignment_config declares it per
                # (proposal, observation); the registry's per-proposal token is
                # for naming a legacy table file, not for provenance.
                # 'NONE' spelt as unified_alignment spells it in ALIGNREF, which
                # records the same value from the same shift.
                references=_shift.reference_frame or 'NONE',
                table_name=_prov_tbl or 'hardcoded/none'):
            align_fits[1].header[_k] = (_v, _c)
        align_fits.writeto(fn, overwrite=True)
        assert 'RAOFFSET' in fits.getheader(fn, ext=1)
        # provenance: the per-exposure aligned crf is what cataloging re-fits, so
        # its data/wcs facets gate the cataloging-skip decision.  Stamp only on
        # the apply branch (RAOFFSET freshly baked); fail-soft.
        _stamp_imaging_product(fn)
    check_wcs(fn)



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
    parser.add_option("--run-outlier-detection", dest="run_outlier_detection",
                      default=False, action='store_true',
                      help="Re-enable Image3 outlier_detection (OFF by default on "
                           "these crowded GC fields; see #161). It over-flags real "
                           "bright-star PSF signal because ERR under-models the "
                           "dither-phase dispersion of the undersampled PSF; CRs are "
                           "already handled per-ramp by JumpStep. Use this only for "
                           "fields sparse enough that residual post-JumpStep CRs "
                           "dominate.",
                      metavar="run_outlier_detection")
    (options, args) = parser.parse_args()

    # Production run guard: refuse to run the imaging stage on an untagged or
    # dirty tree (so every product carries a real release tag) unless GC_ALLOW_DEV
    # is set for a development run.  See jwst_gc_pipeline/versioning/tags.py.
    from jwst_gc_pipeline.versioning.tags import assert_runnable_version
    _run_tag = assert_runnable_version('imaging')
    print(f"imaging: running under pipeline tag {_run_tag}")

    filternames = options.filternames.split(",")
    modules = options.modules.split(",")
    # Padded here as well as in `main`, because the CLI uses the value before
    # `main` sees it: `field_to_reg_mapping[field]` keys on the three-digit
    # spelling, so `--field 1` raised `KeyError: '1'` against a registry that
    # holds the observation it names.  Issue #438.
    fields = [registry_obs_key(part) for part in options.field.split(",")]
    proposal_id = options.proposal_id
    skip_step1and2 = options.skip_step1and2
    no_destreak = bool(options.no_destreak)
    skymatch_method = (options.skymatch_method or '').strip() or None
    skip_outlier_detection = not options.run_outlier_detection
    print(options)

    with open(os.path.expanduser('~/.mast_api_token'), 'r') as fh:
        api_token = fh.read().strip()
        os.environ['MAST_API_TOKEN'] = api_token.strip()
    Mast.login(api_token.strip())
    Observations.login(api_token)


    field_to_reg_mapping = field_registry.field_to_reg_mapping(proposal_id, 'nircam')

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
                               skip_outlier_detection=skip_outlier_detection,
                              )


    if proposal_id == '2221':
        print("Running notebooks")
        from run_notebook import run_notebook
        basepath = field_registry.basepath('brick')
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
