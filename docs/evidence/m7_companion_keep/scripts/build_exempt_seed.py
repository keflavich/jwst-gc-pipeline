"""The real ``_build_m7_band_seed`` on the gc-treasury 10678 F212N m6 vetted catalogs of o045 and o105, far-companion exemption off and on.

usage: python build_exempt_seed.py OUTDIR LEFTOVERS_045.json LEFTOVERS_105.json [SNR ...] [--root /orange/adamginsburg/jwst/gc-treasury]

For each tile and module (nrca, nrcb):
1. With the exemption off (``companion_keep_snr`` 0) the builder must reproduce the production band seed
   (``crossband_seed_manual_o<T>_<module>_f212n_vetted.fits``: row count and NCOMPAN).
2. For each SNR the builder reports the exempted rows (header keyword NCOMPKEP).  The rows added at the first SNR of the
   list go to OUTDIR/o<T>/exempt_<module>.fits (with the control leftover peak under each), and the production
   i2d-augmented seed plus those rows to OUTDIR/o<T>/crossband_seed_manual_o<T>_<module>_f212n_i2dseed.fits (the seed
   a one-frame replay fits).  SNR defaults to 50.
3. The added rows are matched to the leftover list of the control m7 residual (0.072" = one FWHM).  A leftover list is a
   JSON list of records with ``ra``, ``dec`` (degrees) and ``resid_peak`` (MJy/sr): the whole-star leftovers of the
   m7 residual i2d (resid_peak >= 50, resid_peak / data_peak > 0.7).  The scoring script is not part of the repo.
writes OUTDIR/summary.json.  Reads <root>/catalogs; writes only under OUTDIR.
"""
import json
import os
import shutil
import sys

sys.dont_write_bytecode = True
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
sys.path.insert(0, REPO)

import astropy.units as u  # noqa: E402
import numpy as np  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402
from astropy.table import Table, vstack  # noqa: E402

from jwst_gc_pipeline.photometry.cataloging import _build_m7_band_seed  # noqa: E402

args = sys.argv[1:]
ROOT = '/orange/adamginsburg/jwst/gc-treasury'
if '--root' in args:
    i = args.index('--root')
    ROOT = args[i + 1]
    del args[i:i + 2]
OUT, LEFT = args[0], {'045': args[1], '105': args[2]}
SNRS = [float(s) for s in args[3:]] or [50.0]
CAT = f'{ROOT}/catalogs'
summary = {}
for tile in ('045', '105'):
    wd = f'{OUT}/o{tile}'
    os.makedirs(wd, exist_ok=True)
    xb = f'{wd}/crossband_seed_manual_o{tile}.fits'
    shutil.copy(f'{CAT}/crossband_seed_manual_o{tile}.fits', xb)
    left = json.load(open(LEFT[tile]))
    lsc = SkyCoord([r['ra'] for r in left] * u.deg, [r['dec'] for r in left] * u.deg)
    lres = np.array([r['resid_peak'] for r in left])
    for mod in ('nrca', 'nrcb'):
        own = f'{CAT}/f212n_{mod}_o{tile}_indivexp_merged_resbgsub_m6_dao_basic_vetted.fits'
        prod = Table.read(f'{CAT}/crossband_seed_manual_o{tile}_{mod}_f212n_vetted.fits')
        built = {}
        for snr in [0.0] + SNRS:
            p = _build_m7_band_seed(xb, own, 'F212N', mod, companion_keep_snr=snr, label=f'o{tile}:{mod}:snr{snr:g}')
            dst = f'{wd}/band_{mod}_snr{snr:g}.fits'
            shutil.move(p, dst)
            built[snr] = Table.read(dst)
        off = built[0.0]
        same = len(off) == len(prod) and off.meta['NCOMPAN'] == prod.meta['NCOMPAN']
        print(f'o{tile} {mod}: off-run rows {len(off)} vs production {len(prod)}, '
              f'NCOMPAN {off.meta["NCOMPAN"]} vs {prod.meta["NCOMPAN"]}; reproduces: {same}', flush=True)
        for snr in SNRS:
            on = built[snr]
            nx = int(np.sum(np.asarray(on['seed_origin']).astype(str) == 'crossband'))
            onsc = SkyCoord(on['skycoord'])[nx:]
            ofsc = SkyCoord(off['skycoord'])[nx:]
            _, sep, _ = onsc.match_to_catalog_sky(ofsc)
            new = sep.to_value(u.mas) > 1e-3
            nsc = onsc[new]
            idl, dl, _ = nsc.match_to_catalog_sky(lsc)
            hit = dl.arcsec < 0.072
            r = np.where(hit, lres[idl], 0.0)
            key = f'o{tile}_{mod}_snr{snr:g}'
            summary[key] = dict(added=int(new.sum()), ncompkep=int(on.meta['NCOMPKEP']), ncompan=int(on.meta['NCOMPAN']),
                                on_leftover_ge50=int((r >= 50).sum()), ge100=int((r >= 100).sum()), ge200=int((r >= 200).sum()))
            print(key, summary[key], flush=True)
            if snr == SNRS[0]:
                ex = Table()
                ex['skycoord'] = nsc
                ex['flux'] = np.asarray(on['flux'])[nx:][new]
                ex['seed_origin'] = np.array(['own_m6'] * int(new.sum()))
                ex['leftover_resid_peak'] = r
                ex.write(f'{wd}/exempt_{mod}.fits', overwrite=True)
                aug = Table.read(f'{CAT}/crossband_seed_manual_o{tile}_{mod}_f212n_i2dseed.fits')
                add = Table()
                add['skycoord'] = nsc
                add['flux'] = ex['flux']
                add['seed_origin'] = ex['seed_origin']
                if 'seed_round_loose' in aug.colnames:
                    add['seed_round_loose'] = np.zeros(len(add), dtype=aug['seed_round_loose'].dtype)
                both = vstack([aug, add], metadata_conflicts='silent')
                both.meta.update(aug.meta)
                both.write(f'{wd}/crossband_seed_manual_o{tile}_{mod}_f212n_i2dseed.fits', overwrite=True)
                print(f'  wrote augmented seed: {len(aug)} + {len(add)} = {len(both)}', flush=True)
json.dump(summary, open(f'{OUT}/summary.json', 'w'), indent=1)
