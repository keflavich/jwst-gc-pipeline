"""phase_loss.run over every reference-field run (field x variant x seed x band).

usage: python run_ref.py <variant[,variant...]> <outdir>
writes <outdir>/<field>_<variant>_s<seed>_<band>_{lost.fits,summary.json}
"""
import glob
import json
import os
import re
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jwst_gc_pipeline.photometry import reference_fields as RF  # noqa: E402
from phase_loss import run  # noqa: E402

variants, OUT = sys.argv[1].split(','), sys.argv[2]
os.makedirs(OUT, exist_ok=True)
_, fields = RF.load_config()
for name, spec in fields.items():
    half = spec['size_arcsec'] / 2 - spec['inner_margin_arcsec']
    for variant in variants:
        for seed in [0] + list(spec['seeds']):
            d = RF.run_dir(spec, variant, seed)
            for filt in spec['filters']:
                f = filt.lower()
                hits = glob.glob(f'{d}/catalogs/{f}_merged*_indivexp_merged_resbgsub_m7_dao_basic_vetted.fits')
                if len(hits) != 1:
                    print('MISSING', d, filt, hits, flush=True)
                    continue
                obstok = re.match(rf'{f}_merged(.*)_indivexp', os.path.basename(hits[0])).group(1)
                out = f'{OUT}/{name}_{variant}_s{seed}_{f}'
                try:
                    _, summ = run(f'{d}/catalogs', f'{d}/{filt}/pipeline', filt, 'merged', obstok,
                                  1.0, out, f'{name}_{variant}_s{seed}',
                                  inner=(spec['ra'], spec['dec'], half))
                    print(name, variant, seed, filt, summ['n_lost'], summ['n_starlike_left'],
                          json.dumps(summ['m7_seed_reason']), flush=True)
                except (FileNotFoundError, ValueError, KeyError, OSError) as ex:
                    print('FAIL', d, filt, repr(ex), flush=True)
                    traceback.print_exc()
