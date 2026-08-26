#!/usr/bin/env python
"""NIRISS imaging Image3 runner (single-detector, GC crowded-field pipeline).

Ported from ``PipelineMIRI.py`` (the single-detector template) rather than
``PipelineRerunNIRCAM-LONG.py`` (whose nrca/nrcb module machinery, quadrant
destreak, and short/long split do not apply to NIRISS's single NIS detector).

Key NIRISS specifics vs the MIRI template:
  * detector token ``_nis_`` (not ``_mirimage_``); product instrument
    ``niriss``.
  * NIRISS shares its obsid with the NIRCam observation of the same field
    (Sgr C 4147 obs 012 is BOTH nircam and niriss), so every product/asn glob
    MUST be disambiguated to ``_nis_`` / a ``niriss`` product name, else the
    NIRCam frames leak in.
  * output lands in ``{region}/niriss/{FILTER}/pipeline/`` so NIRISS F480M /
    F356W do NOT collide with the NIRCam ``{region}/{FILTER}`` trees.
  * NO destreak (NIRCam-quadrant-specific) and NO MIRI edge-glow trim.
  * per-filter PSF FWHM read from ``reduction/fwhm_table_niriss.ecsv`` (NIRISS
    0.0656"/pix scale; the shared fwhm_table.ecsv pixel column is NIRCam-scale).

Astrometry: like the MIRI runner, this ties to the absolute reference catalog
(Sgr C -> VIRAC2/Gaia seed) via tweakreg ``abs_refcat`` during Image3, with
``fix_alignment`` applying a zero baseline shift (modern raw pointing is good to
~0.1").  The cataloging m2 astrometry checkpoint re-verifies/refines later.
"""
from glob import glob
from astroquery.mast import Mast, Observations
import copy
import os

# Imported as field_registry: `fields` is a local variable in these
# drivers (the --field list), and shadowed the module.
from jwst_gc_pipeline import fields as field_registry
# The shared spelling rule for an observation number: three digits, anything
# that is not a plain number handed back untouched.  Issue #438.
from jwst_gc_pipeline.reduction.mast_obs_scope import observation_number
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
# supplies the default; an exported CRDS_PATH wins.  The per-target cache
# selection further down replaces it once the target is known.
from jwst_gc_pipeline.config import apply_crds_environment
from jwst_gc_pipeline.mast_names import jw_prefix, proposal_id_from_datamodel
from jwst_gc_pipeline.reduction.crds_cache import open_crds_reference
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
    return printfunc(f"{now}:", *args, **kwargs)


print(jwst.__version__)

# NIRISS detector token: single detector, named 'nis' in the exposure filenames
# (jw{prop}{field}{visit}_{...}_nis_cal.fits) and 'niriss' in the level-3 product
# name (jw{prop}-o{field}_t001_niriss_{filter}).
DETECTOR_TOKEN = 'nis'
INSTRUMENT_PRODUCT_TOKEN = 'niriss'

# Reference catalog configuration by proposal and field.  Paths relative to basepath.
# Which reference catalogs this instrument's observations may tie to is
# registered in jwst_gc_pipeline/fields.yaml.  See docs/FIELDS.md.


def get_reference_astrometric_catalog_path(basepath, proposal_id, field, explicit_refcat=None):
    if explicit_refcat is not None:
        return explicit_refcat
    try:
        candidates = field_registry.reference_catalog_candidates(
            proposal_id, field, basepath=basepath, instrument='niriss')
    except (field_registry.FieldRegistryError, KeyError):
        candidates = []
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    twomass = f'{basepath}/catalogs/twomass.fits'
    if os.path.exists(twomass):
        return twomass
    return None


def _select_asn_for_filter(asn_candidates, filtername):
    """From all level-3 asn files for this obs (NIRCam + NIRISS share the obsid),
    pick the single NIRISS asn for ``filtername``.  Matches on the product name
    (contains 'niriss' + the filter) and requires the members to be ``_nis_``
    exposures, so a same-obsid NIRCam asn can never be selected."""
    matches = []
    for asn in asn_candidates:
        try:
            with open(asn) as fh:
                d = json.load(fh)
            prod = d['products'][0]
            name = prod['name'].lower()
            members = [m['expname'].lower() for m in prod['members']]
        except (json.JSONDecodeError, KeyError, IndexError, OSError):
            continue
        if INSTRUMENT_PRODUCT_TOKEN not in name:
            continue
        if filtername.lower() not in name:
            continue
        if not all(f'_{DETECTOR_TOKEN}_' in m for m in members):
            continue
        matches.append(asn)
    return matches


def _regen_per_exposure_i2d(output_dir, field):
    """Regenerate the per-exposure single-frame ``_i2d`` from the FINAL aligned
    crf so the on-disk ``jw..._nis_i2d.fits`` carry the corrected (post-tweakreg +
    fix_alignment) WCS, not the raw-pointing Image2 WCS."""
    from jwst.resample import ResampleStep
    crfs = sorted(glob(os.path.join(output_dir, f'jw*_{DETECTOR_TOKEN}_*o{field}_crf.fits')))
    crfs = [c for c in crfs
            if re.search(rf'_\d{{5}}_\d{{5}}_{DETECTOR_TOKEN}', os.path.basename(c))]
    by_stem = {}
    for c in crfs:
        stem = os.path.basename(c).split(f'_{DETECTOR_TOKEN}')[0] + f'_{DETECTOR_TOKEN}'
        if stem not in by_stem or os.path.getmtime(c) > os.path.getmtime(by_stem[stem]):
            by_stem[stem] = c
    crfs = sorted(by_stem.values())
    n = 0
    for crf in crfs:
        stem = os.path.basename(crf).split(f'_{DETECTOR_TOKEN}')[0] + f'_{DETECTOR_TOKEN}'
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


def main(filtername, Observations=None, regionname='sgrc',
         field='012', proposal_id='4147', skip_step1and2=False, use_average=True,
         reference_catalog=None, skip_download_for_existing=False,
         skymatch_method='match'):
    """
    skip_step1and2 will not re-fit the ramps to produce the _cal images.  This
    can save time if you just want to redo the tweakreg steps but already have
    the ramp/cal stuff done.
    """
    # FIRST statement: every name below is interpolated from `field` -- the
    # uncal download filter (`jw{PPPPP}{field}`, whose '_nis_' sibling test is
    # the only other disambiguator), the association search
    # (`jw{PPPPP}-o{field}*_image3_*asn.json`), the drizzle product name
    # (`...-o{field}_t001_niriss_{filt}`), the `_o{field}_crf` frames and the
    # per-exposure i2d regeneration glob -- and `assert field == '012'` guards
    # the one registered NIRISS observation.  MAST spells an observation with
    # three digits, so `--field 12` built `jw04147-o12*` and tripped that
    # assert instead of reducing observation 012.  This is the NIRISS half of
    # what #528 did for the NIRCam driver; a non-number is handed back
    # untouched.  Issue #438.
    field = observation_number(field)
    print(f"Processing filter {filtername} skip_step1and2={skip_step1and2} for field {field} and proposal id {proposal_id} in region {regionname}")

    wavelength = int(filtername[1:4])

    # The field's data directory comes from the registry (fields.yaml `roots:`
    # plus the field's `root:`), so a field on a tree other than /orange reduces
    # where it lives.
    # GC_BASEPATH_OVERRIDE is deliberately NOT applied here.  Besides the CRDS
    # and reference-catalog problem the MIRI driver notes, NIRISS reads
    # {basepath}/niriss/{FILTER}/pipeline/ and stage_scratch_basepath.sh stages
    # only {FILTER}/pipeline, so a staged scratch tree would hold no inputs.
    basepath = field_registry.basepath(regionname)
    fwhm_tbl = Table.read(f'{basepath}/reduction/fwhm_table_niriss.ecsv')
    row = fwhm_tbl[fwhm_tbl['Filter'] == filtername]
    if len(row) == 0:
        raise KeyError(f"Filter {filtername} not in fwhm_table_niriss.ecsv "
                       f"(have {list(fwhm_tbl['Filter'])})")
    fwhm = fwhm_arcsec = float(row['PSF FWHM (arcsec)'][0])
    fwhm_pix = float(row['PSF FWHM (pixel)'][0])

    # sanity check: this runner is currently wired only for Sgr C NIRISS.
    if proposal_id == '4147':
        assert field == '012', f"4147 NIRISS is obs 012, got {field}"
        assert regionname == 'sgrc'

    # Per-target CRDS cache when writable; else fall back to the shared brick cache.
    crds_path = f"{basepath}/crds/"
    crds_mapdir = os.path.join(crds_path, 'mappings', 'jwst')
    if os.path.isdir(crds_mapdir) and not os.access(crds_mapdir, os.W_OK):
        # The configured shared cache, which is what CRDS_PATH already holds
        # (apply_crds_environment set it at import).  Reading it from the same
        # place keeps ONE shared cache in this file.
        crds_path = os.environ.get('CRDS_PATH') or crds_path
        print(f"per-target CRDS cache is not writable; using the shared cache "
              f"{crds_path} instead")
    else:
        os.makedirs(crds_mapdir, exist_ok=True)
    os.environ["CRDS_PATH"] = crds_path
    os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"
    mpl.rcParams['savefig.dpi'] = 80
    mpl.rcParams['figure.dpi'] = 80

    # Instrument-namespaced output dir, under the field's own directory, so a
    # scratch reduction (GC_BASEPATH_OVERRIDE) writes to scratch.  The niriss/
    # level keeps NIRISS F480M/F356W clear of the NIRCam {region}/{FILTER} trees.
    output_dir = f'{basepath}niriss/{filtername}/pipeline/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)

    # Reuse any _cal already staged one directory up (niriss/{FILTER}/).
    for fn in glob("../*cal.fits"):
        try:
            os.link(fn, './'+os.path.basename(fn))
        except FileExistsError as ex:
            print(f'Failed to link {fn} to {os.path.basename(fn)} because of {ex}')

    Observations.cache_location = output_dir
    obs_table = Observations.query_criteria(
                                            proposal_id=proposal_id,
                                            )
    print("Obs table length:", len(obs_table))

    if 'filters' in obs_table.colnames and 'obs_id' in obs_table.colnames:
        try:
            filters_col = np.array([str(val).upper() for val in obs_table['filters'].filled('')])
            obs_id_col = np.array([str(val).lower() for val in obs_table['obs_id'].filled('')])
        except AttributeError:
            filters_col = np.array([str(val).upper() for val in obs_table['filters']])
            obs_id_col = np.array([str(val).lower() for val in obs_table['obs_id']])
        # NIRISS obs_id (jw04147-o012_t001_niriss_clear-f200w) carries the filter
        # AND the instrument -- require 'niriss' so a same-filter NIRCam obs of
        # this field cannot leak in here.
        msk = ((np.char.find(obs_id_col, INSTRUMENT_PRODUCT_TOKEN) >= 0) &
               ((np.char.find(filters_col, filtername.upper()) >= 0) |
                (np.char.find(obs_id_col, filtername.lower()) >= 0)))
        if not msk.any():
            # some obs_id rows omit the instrument; fall back to filter-only but
            # still filter members to _nis_ below.
            msk = ((np.char.find(filters_col, filtername.upper()) >= 0) |
                   (np.char.find(obs_id_col, filtername.lower()) >= 0))
    else:
        print("Warning: 'filters'/'obs_id' missing; selecting all observations for this proposal")
        msk = np.ones(len(obs_table), dtype=bool)
    data_products_by_obs = Observations.get_product_list(obs_table[msk])
    print("data products by obs length: ", len(data_products_by_obs))

    products_asn = Observations.filter_products(data_products_by_obs, extension="json")
    print("products_asn length:", len(products_asn))

    manifest = Observations.download_products(products_asn, download_dir=output_dir)
    print("manifest:", manifest)
    relocate_manifest_products(manifest, output_dir)

    products_fits = Observations.filter_products(data_products_by_obs, extension="fits")
    print("products_fits length:", len(products_fits))
    # uncal for THIS obs + NIRISS detector only (the NIRCam frames of obs 012
    # share the jw{prop:05d}{field} prefix -- '_nis_' is the disambiguator).
    uncal_mask = np.array([uri.endswith('_uncal.fits')
                           and f'{jw_prefix(proposal_id)}{field}' in uri
                           and f'_{DETECTOR_TOKEN}_' in uri
                           for uri in products_fits['dataURI']])
    uncal_mask &= products_fits['productType'] == 'SCIENCE'
    print("uncal length:", (uncal_mask.sum()))

    if skip_download_for_existing:
        already_downloaded = np.array([os.path.exists(os.path.basename(uri)) for uri in products_fits['dataURI']])
        uncal_mask &= ~already_downloaded
        print(f"uncal to download: {uncal_mask.sum()}; {already_downloaded.sum()} already present")

    if uncal_mask.any():
        manifest = Observations.download_products(products_fits[uncal_mask], download_dir=output_dir)
        print("manifest:", manifest)
        relocate_manifest_products(manifest, output_dir)

    print(f"Working on NIRISS: initial pipeline setup (skip_step1and2={skip_step1and2})")
    asn_all = glob(os.path.join(output_dir, f'{jw_prefix(proposal_id)}-o{field}*_image3_*0[0-9][0-9]_asn.json'))
    asn_file_search = _select_asn_for_filter(asn_all, filtername)
    print(f"Searching asn for {filtername}: {len(asn_all)} candidates -> {len(asn_file_search)} niriss match(es)")
    if len(asn_file_search) == 1:
        asn_file = asn_file_search[0]
    elif len(asn_file_search) > 1:
        asn_file = sorted(asn_file_search)[-1]
        print(f"Found multiple niriss asn for {filtername}: {asn_file_search}. Using {asn_file}.")
    else:
        raise ValueError(f"No NIRISS asn for filter {filtername} field {field} in {output_dir} "
                         f"(candidates: {asn_all})")

    # tweakreg default parameters from the NIRISS CRDS pars reference (latest).
    tweakreg_rmaps = sorted(glob(os.path.join(os.environ["CRDS_PATH"],
                                              'mappings', 'jwst',
                                              'jwst_niriss_pars-tweakregstep_*.rmap')))
    if not tweakreg_rmaps:
        # ensure the rmap is cached
        crds.getreferences({'INSTRUME': 'NIRISS'}, reftypes=[])
        tweakreg_rmaps = sorted(glob(os.path.join(os.environ["CRDS_PATH"],
                                                  'mappings', 'jwst',
                                                  'jwst_niriss_pars-tweakregstep_*.rmap')))
    mapping = crds.rmap.load_mapping(tweakreg_rmaps[-1])
    print(f"tweakreg pars rmap: {os.path.basename(tweakreg_rmaps[-1])}")
    # NIRISS pars-tweakregstep selection keys are 5-tuples
    # (EXP_TYPE, filter-wheel, pupil-wheel, USEAFTER-date, asdf-filename); the
    # reference filename is the LAST element (MIRI's were 4-tuples with the file
    # at [3], which is why a blind [3] here grabbed the date string).  There are
    # multiple USEAFTER variants per filter -- take the latest.
    filter_match = [x for x in mapping.todict()['selections'] if filtername.upper() in x]
    filter_match = sorted(filter_match, key=lambda x: x[-2])
    if filter_match:
        tweakreg_asdf_filename = filter_match[-1][-1]
        tweakreg_asdf = open_crds_reference(os.environ['CRDS_PATH'], 'niriss',
                                            tweakreg_asdf_filename)
        tweakreg_parameters = tweakreg_asdf.tree['parameters']
    else:
        print(f"No filter-specific tweakreg pars for {filtername}; using empty parameter set")
        tweakreg_parameters = {}
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
            print(f"skip_step1and2 requested, but {len(missing_cal)} _cal missing; running detector/image2 for those")

    if (not skip_step1and2) or (skip_step1and2 and len([m['expname'] for m in members if not os.path.exists(m['expname'])]) > 0):
        for member in members:
            assert f'{jw_prefix(proposal_id)}{field}' in member['expname']
            assert f'_{DETECTOR_TOKEN}_' in member['expname'], member['expname']
            cal_name = member['expname']
            if skip_step1and2 and os.path.exists(cal_name):
                continue

            print(f"DETECTOR PIPELINE on {cal_name}")
            # Hosek crowded-field: keep one-group ramps (do not suppress).
            detector1_steps = {'ramp_fit': {'suppress_one_group': False},
                               'refpix': {'use_side_ref_pixels': True}}
            Detector1Pipeline.call(cal_name.replace("_cal.fits", "_uncal.fits"),
                                   save_results=True, output_dir=output_dir,
                                   save_calibrated_ramp=True,
                                   steps=detector1_steps)

            print(f"IMAGE2 PIPELINE on {cal_name}")
            Image2Pipeline.call(cal_name.replace("_cal.fits", "_rate.fits"),
                                save_results=True, output_dir=output_dir,
                               )

    print(f"Filter {filtername}: doing alignment + Image3.")

    with open(asn_file) as f_obj:
        asn_data = json.load(f_obj)
    asn_data['products'][0]['name'] = f'{jw_prefix(proposal_id)}-o{field}_t001_{INSTRUMENT_PRODUCT_TOKEN}_{filtername.lower()}'

    for member in asn_data['products'][0]['members']:
        print(f"Preparing alignment copy for {member}")
        fname = member['expname']
        # Idempotent re-run: normalize _align.fits back to _cal.fits source.
        cal_name = fname.replace("_align.fits", "_cal.fits")
        assert cal_name.endswith('_cal.fits'), cal_name
        align_name = cal_name.replace("_cal.fits", "_align.fits")
        member['expname'] = align_name
        # copyfile (data only), NOT copy (avoid EPERM from copymode on foreign-owned cal).
        shutil.copyfile(cal_name, align_name)

        fix_alignment(member['expname'], proposal_id=proposal_id,
                      field=field, basepath=basepath,
                      regionname=regionname,
                      filtername=filtername,
                      use_average=use_average,
                      visit=fname[10:13])

    asn_file_each = asn_file
    with open(asn_file_each, 'w') as fh:
        json.dump(asn_data, fh)

    # Shift-only alignment: NIRISS single-detector frames align well from the
    # guide-star pointing; disallow rotation/scale so a sparse-match frame cannot
    # fit a spurious rotation.
    tweakreg_parameters['fitgeometry'] = 'shift'
    tweakreg_parameters['abs_fitgeometry'] = 'shift'

    abs_refcat = get_reference_astrometric_catalog_path(basepath, proposal_id, field, explicit_refcat=reference_catalog)
    if abs_refcat is not None:
        reftbl = Table.read(abs_refcat)
        reftbl.meta['name'] = 'Reference Astrometric Catalog'
        tweakreg_parameters['abs_searchrad'] = 0.4
        tweakreg_parameters['searchrad'] = 0.05
        tweakreg_parameters['minobj'] = 5
        # require >=5 absolute matches so an under-covered frame FAILS the abs fit
        # and stays at its (good) raw pointing rather than latching onto a few
        # spurious pairs and applying a catastrophic per-frame shift.
        tweakreg_parameters['abs_minobj'] = 5
        tweakreg_parameters.update({'abs_refcat': abs_refcat})
        print(f"Reference catalog is {abs_refcat}")
    else:
        print(f"No reference catalog for proposal_id={proposal_id} field={field} in {basepath}; running without abs_refcat")

    # subtract=True: with subtract=False the matched sky levels are only recorded,
    # so outlier_detection's median sees inter-visit background jumps and flags
    # whole regions -> NaN patches in the resample (the MIRI F2550W failure mode).
    skymatch_params = {'save_results': True,
                       'subtract': True,
                       'skymethod': skymatch_method,
                       'match_down': False}
    if skymatch_method in (None, '', 'none', 'off'):
        skymatch_params = {'save_results': True, 'skymethod': 'match', 'subtract': False}
    outlier_params = {'snr': '30.0 25.0',
                      'good_bits': "SATURATED, JUMP_DET"}

    print(f"Running Image3 (skymatch={skymatch_params.get('skymethod')} subtract={skymatch_params.get('subtract')})")
    calwebb_image3.Image3Pipeline.call(
        asn_file_each,
        steps={'tweakreg': tweakreg_parameters,
               'skymatch': skymatch_params,
               'outlier_detection': outlier_params,
               # Skip the Image3 SourceCatalogStep: on a deep, crowded NIRISS GC
               # mosaic its aperture/deblend catalog took ~10 h (F200W) while the
               # products we actually use (mosaic + crf) are done at resample in
               # minutes.  Real photometry is the separate cataloging pipeline;
               # this _cat.ecsv is throwaway.
               'source_catalog': {'skip': True},
        },
        output_dir=output_dir,
        save_results=True)
    print(f"DONE running {asn_file_each}")

    # CRF NAMING FIX: because the asn product name is set above, outlier_detection
    # names the CR-flagged products after the PRODUCT, not the exposure.  The
    # per-frame photometry globs PER-EXPOSURE crf, so map product-named crf ->
    # per-exposure names by EXPSTART (1:1) and copy them into place.
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
        targ_by_es = {}
        for member in asn_data['products'][0]['members']:
            cal_base = os.path.basename(member['expname']).replace('_align.fits', '_cal.fits')
            target = os.path.join(output_dir, cal_base.replace('_cal.fits', f'_o{field}_crf.fits'))
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

    # Update per-exposure single-frame _i2d to the FINAL aligned WCS.
    _regen_per_exposure_i2d(output_dir, field)

    print("After Image3, checking WCS headers:")
    for member in asn_data['products'][0]['members']:
        check_wcs(member['expname'])
        check_wcs(member['expname'].replace('cal', 'i2d'))
    check_wcs(asn_data['products'][0]['name'] + "_i2d.fits")

    globals().update(locals())
    return locals()


def fix_alignment(fn, proposal_id=None, regionname='sgrc', field=None, basepath=None, filtername=None,
                  use_average=True, visit='012'):
    if os.path.exists(fn):
        print(f"Running manual align for data ({proposal_id} + {field}): {fn}", flush=True)
    else:
        print(f"Skipping manual align for nonexistent file ({proposal_id} + {field}): {fn}", flush=True)
        return

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
            nonclear = [x for x in filters if x not in ('clear', 'clearp')]
            if nonclear:
                filtername = nonclear[0]
            else:
                filtername = [x for x in filters if 'w' not in x][0]
    if field is None:
        field = mod.meta.observation.observation_number
    if basepath is None:
        basepath = field_registry.basepath(regionname)

    # NIRISS raw pointing is good to ~0.1"; the absolute tie is applied by
    # tweakreg's abs_refcat fit in Image3.  Baseline shift is zero (no measured
    # per-target/per-visit NIRISS blunder yet); reinstate here per-target only if
    # an offset-histogram measurement (never NN-median) shows one is needed.
    rashift = 0 * u.arcsec
    decshift = 0 * u.arcsec

    print(f"Shift for {fn} is {rashift}, {decshift}")
    align_fits = fits.open(fn)
    if 'RAOFFSET' in align_fits[1].header:
        print(f"{fn} is already aligned ({align_fits[1].header['RAOFFSET']}, {align_fits[1].header['DEOFFSET']})")
    else:
        fa = ImageModel(fn)
        wcsobj = fa.meta.wcs
        print(f"Before shift, crval={wcsobj.to_fits()[0]['CRVAL1']}, {wcsobj.to_fits()[0]['CRVAL2']}")
        fa.meta.oldwcs = copy.copy(wcsobj)
        ww = adjust_wcs(wcsobj, delta_ra=rashift, delta_dec=decshift)
        print(f"After shift, crval={ww.to_fits()[0]['CRVAL1']}, {ww.to_fits()[0]['CRVAL2']}")
        fa.meta.wcs = ww
        fa.save(fn, overwrite=True)

        align_fits = fits.open(fn)
        align_fits[1].header['OLCRVAL1'] = align_fits[1].header['CRVAL1']
        align_fits[1].header['OLCRVAL2'] = align_fits[1].header['CRVAL2']
        # See fits_wcs_sync: gwcs's to_fits() default (0.25 px) writes a SIP
        # fit several mas off the GWCS and merges over the delivered one.
        from jwst_gc_pipeline.reduction.fits_wcs_sync import sync_header_to_gwcs
        _sip_max, _sip_med = sync_header_to_gwcs(
            align_fits[1].header, ww, fa.data.shape, label=os.path.basename(fn))
        print(f"FITS/SIP header synced to GWCS: max {_sip_max:.4f} mas, "
              f"median {_sip_med:.4f} mas")
        align_fits[1].header['SIPGWMAX'] = (
            _sip_max, '[mas] max FITS/SIP vs GWCS disagreement')
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
        print(f"fa['meta']['wcs'] crval={wcsobj.to_fits()[0]['CRVAL1']}, {wcsobj.to_fits()[0]['CRVAL2']}")
        new_1024 = wcsobj.pixel_to_world(1024, 1024)
        print(f"new pixel_to_world(1024,1024) = {new_1024}")
        if 'oldwcs' in fa.meta:
            oldwcsobj = fa.meta.oldwcs
            old_1024 = oldwcsobj.pixel_to_world(1024, 1024)
            print(f"old pixel_to_world(1024,1024) = {old_1024}, sep from new GWCS={old_1024.separation(new_1024).to(u.arcsec)}")
        fa.close()

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
                      default='F200W',
                      help="filter name list", metavar="filternames")
    parser.add_option("-d", "--field", dest="field",
                    default='012',
                    help="list of target fields", metavar="field")
    parser.add_option("-s", "--skip_step1and2", dest="skip_step1and2",
                      default=False,
                      action='store_true',
                      help="Skip the ramp/cal remaking step?", metavar="skip_Step1and2")
    parser.add_option("-p", "--proposal_id", dest="proposal_id",
                      default='4147',
                      help="proposal id (string)", metavar="proposal_id")
    parser.add_option("--reference_catalog", dest="reference_catalog",
                      default=None,
                      help="Path to explicit astrometric reference catalog for tweakreg (optional)", metavar="reference_catalog")
    parser.add_option("--skip_download_for_existing", dest="skip_download_for_existing",
                      default=False, action='store_true',
                      help="Skip downloading _uncal files already present in output directory", metavar="skip_download_for_existing")
    parser.add_option("--skymatch-method", dest="skymatch_method",
                      default='match',
                      help="skymatch skymethod (match/global/local/off); default match+subtract", metavar="skymatch_method")
    (options, args) = parser.parse_args()

    filternames = options.filternames.split(",")
    # Padded here as well as in `main`, because the CLI uses the value before
    # `main` sees it: `field_to_reg_mapping[field]` keys on the three-digit
    # spelling, so `--field 12` raised `KeyError: '12'` against a registry that
    # holds the observation it names.  Issue #438.
    fields = [observation_number(part) for part in options.field.split(",")]
    proposal_id = options.proposal_id
    skip_step1and2 = options.skip_step1and2
    reference_catalog = options.reference_catalog
    skip_download_for_existing = options.skip_download_for_existing
    skymatch_method = options.skymatch_method
    print(options)

    with open(os.path.expanduser('~/.mast_api_token'), 'r') as fh:
        api_token = fh.read().strip()
        os.environ['MAST_API_TOKEN'] = api_token.strip()
    Mast.login(api_token.strip())
    Observations.login(api_token)

    field_to_reg_mapping = field_registry.field_to_reg_mapping(proposal_id, 'niriss')

    for field in fields:
        for filtername in filternames:
            print(f"Main Loop: {proposal_id} + {filtername} + {field}={field_to_reg_mapping[field]}")
            results = main(filtername=filtername, Observations=Observations, field=field,
                           regionname=field_to_reg_mapping[field],
                           proposal_id=proposal_id,
                           skip_step1and2=skip_step1and2,
                           reference_catalog=reference_catalog,
                           skip_download_for_existing=skip_download_for_existing,
                           skymatch_method=skymatch_method,
                          )
