"""Download one observation's MAST uncal + association files, without reducing.

The reduce owns the download today, and it takes the association files and the
uncal set in the same breath as running Detector1.  That couples them: a fresh
observation whose LEVEL-3 association MAST has not published yet cannot be
downloaded at all, because PipelineRerunNIRCAM-LONG raises

    ValueError: Mismatch: Did not find any NIRCam asn files for module nrca
    for field 135 in /orange/adamginsburg/jwst/gc-treasury/F212N/pipeline/

before it runs a single step.  GC_135 (program 10678) landed on 2026-09-12 with
level 1-2 products only -- 48 F212N + 12 F480M + 6 F770W exposures, 22 image2
associations, no image3 -- so its reduce failed in two minutes while the data
itself was fetchable.

This fetches the same files the reduce would (uncal + every association json,
flattened out of MAST's deep per-exposure tree) and stops there, so an
observation's data is on disk and staged the moment it is released.  Re-running
is cheap: already-downloaded uncal files are skipped, which also makes this the
way to pick up the rest of a delivery that is still arriving.
"""
import argparse
import os
import shutil
import sys

import numpy as np
from astroquery.mast import Observations

from jwst_gc_pipeline.fields import basepath
from jwst_gc_pipeline.mast_names import (
    filtername_from_mast_filters, jw_prefix)
from jwst_gc_pipeline.reduction.mast_obs_scope import observation_scope_mask

DETECTOR_TOKEN = {'nircam': '_nrc', 'miri': '_mirimage', 'niriss': '_nis'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True)
    ap.add_argument('--proposal', required=True)
    ap.add_argument('--obsid', required=True)
    ap.add_argument('--filters', required=True, help='comma separated')
    ap.add_argument('--instrument', default='nircam',
                    choices=sorted(DETECTOR_TOKEN))
    args = ap.parse_args()

    Observations.clear_cache()
    Observations.TIMEOUT = 300
    bp = basepath(args.target)
    obs_table = Observations.query_criteria(proposal_id=args.proposal,
                                            obs_collection='JWST')
    print(f'obs table: {len(obs_table)} rows', flush=True)
    scope = observation_scope_mask(np.array(obs_table['obs_id']),
                                   args.proposal, args.obsid)
    token = DETECTOR_TOKEN[args.instrument]

    rc = 0
    for filtername in args.filters.split(','):
        want = filtername.upper()
        output_dir = os.path.join(bp, want, 'pipeline')
        os.makedirs(output_dir, exist_ok=True)
        Observations.cache_location = output_dir

        band = np.array([filtername_from_mast_filters(_f) == want
                         for _f in np.array(obs_table['filters'])])
        msk = band & scope
        print(f'{want}: {msk.sum()} observation row(s)', flush=True)
        if not msk.any():
            continue

        products = Observations.get_product_list(obs_table[msk])
        print(f'{want}: {len(products)} product(s)', flush=True)

        asn = Observations.filter_products(products, extension='json')
        fits = Observations.filter_products(products, extension='fits')
        uncal = np.array([
            uri.endswith('_uncal.fits')
            and f'{jw_prefix(args.proposal)}{args.obsid}' in uri
            and token in uri
            for uri in fits['dataURI']])
        uncal &= fits['productType'] == 'SCIENCE'
        have = np.array([os.path.exists(os.path.join(output_dir,
                                                     os.path.basename(uri)))
                         for uri in fits['dataURI']])
        uncal &= ~have
        print(f'{want}: {len(asn)} asn, {uncal.sum()} uncal to fetch '
              f'({have.sum()} already on disk)', flush=True)

        for label, table in (('asn', asn), ('uncal', fits[uncal])):
            if len(table) == 0:
                continue
            manifest = Observations.download_products(table,
                                                      download_dir=output_dir)
            for row in manifest:
                dest = os.path.join(output_dir,
                                    os.path.basename(row['Local Path']))
                if os.path.abspath(row['Local Path']) == os.path.abspath(dest):
                    continue
                try:
                    shutil.move(row['Local Path'], dest)
                except (OSError, shutil.Error) as ex:
                    print(f'  move failed for {row["Local Path"]}: {ex}',
                          flush=True)
                    rc = 1
            print(f'{want}: {label} -> {len(manifest)} file(s)', flush=True)

        # Scoped to THIS observation.  Every tile of a program shares one
        # `<FILTER>/pipeline/` directory, so an unscoped count reports the
        # neighbours' files as this tile's: o134's run said "1 image3 asn" when
        # the only one on disk was o135's, which reads as "o134 can reduce" on
        # a tile that has no level-3 product at all.
        names = os.listdir(output_dir)
        exposure_tag = f'{jw_prefix(args.proposal)}{args.obsid}'
        product_tag = f'{jw_prefix(args.proposal)}-o{args.obsid}'
        n_uncal = len([f for f in names
                       if f.endswith('_uncal.fits') and f.startswith(exposure_tag)])
        n_asn3 = len([f for f in names
                      if '_image3_' in f and f.startswith(product_tag)])
        print(f'{want}: on disk now {n_uncal} uncal, {n_asn3} image3 asn '
              f'for o{args.obsid}', flush=True)
        if n_asn3 == 0:
            print(f'{want}: NO image3 association for o{args.obsid} -- MAST has '
                  f'not published level 3 for this observation, so the reduce '
                  f'cannot run yet', flush=True)
    return rc


if __name__ == '__main__':
    sys.exit(main())
