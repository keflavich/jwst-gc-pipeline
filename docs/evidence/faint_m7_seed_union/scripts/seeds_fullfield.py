"""Full-field m7 seed composition for #1015 (Brick 2221/o001, all six bands).

Builds, OUTSIDE the production tree, exactly what the #1015 m7 code builds per
band from the production m6 products:
  cross-band seed (>=2-band confirmed)  -> 'crossband'
  + own-band m6 vetted not near those   -> 'own_m6'
  + daofind on the m6 residual - m6 bg  -> 'i2d'
and writes one seed table per band (with seed_origin) plus the production m7
vetted catalog's membership for each seed row.

usage: python seeds_fullfield.py <pipeline_worktree> <outdir> [BAND ...]
"""
import glob
import json
import os
import sys
import time
import types

WT, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, WT)
import numpy as np                                       # noqa: E402
import astropy.units as u                                # noqa: E402
from astropy.coordinates import SkyCoord                 # noqa: E402
from astropy.table import Table                          # noqa: E402
from jwst_gc_pipeline.photometry import cataloging as C  # noqa: E402
from jwst_gc_pipeline.photometry.manual_defaults import MANUAL_DEFAULTS as MD  # noqa: E402

assert os.path.realpath(C.__file__).startswith(os.path.realpath(WT)), C.__file__
PROD = '/orange/adamginsburg/jwst/brick'
BANDS = sys.argv[3:] or ['F182M', 'F187N', 'F212N', 'F405N', 'F410M', 'F466N']
ALL = ['F182M', 'F187N', 'F212N', 'F405N', 'F410M', 'F466N']
M6 = '{f}_merged_o001_indivexp_merged_resbgsub_m6_dao_basic_vetted.fits'
M7 = '{f}_merged_o001_indivexp_merged_resbgsub_m7_dao_basic_vetted.fits'

cut_bp = os.path.abspath(OUT)
os.makedirs(f'{cut_bp}/catalogs', exist_ok=True)
assert not os.path.realpath(cut_bp).startswith(os.path.realpath(PROD))
for f in ALL:
    src = f'{PROD}/catalogs/' + M6.format(f=f.lower())
    dst = f'{cut_bp}/catalogs/' + M6.format(f=f.lower())
    if not os.path.lexists(dst):
        os.symlink(src, dst)

opts = types.SimpleNamespace(desaturated=False, bgsub=False, blur=False,
                             proposal_id='2221', field='001', modules='merged')
t0 = time.time()
xb = C._build_crossband_seed(cut_bp, ['merged'], ALL, opts)
print(f'cross-band seed {xb} ({time.time() - t0:.0f}s)', flush=True)


def sky(t):
    sc = t['skycoord']
    return sc if isinstance(sc, SkyCoord) else SkyCoord(sc)


summary = {}
for f in BANDS:
    fl = f.lower()
    t0 = time.time()
    own = f'{cut_bp}/catalogs/' + M6.format(f=fl)
    band = C._build_m7_band_seed(xb, own, f, 'merged',
                                 max_sep_mas=MD['manual_crossband_seed_max_sep_mas'],
                                 label=f'evid:{f}')
    pat = f'{PROD}/{f}/pipeline/jw02221-o001_t001_nircam_*{fl}-merged_resbgsub_m6_daophot_basic_mergedcat_residual_i2d.fits'
    res = sorted(glob.glob(pat))
    assert len(res) == 1, (pat, res)
    bg = res[0].replace('_residual_i2d.fits', '_residual_smoothed_bg_i2d.fits')
    assert os.path.exists(bg), bg
    _sr = MD['manual_seed_round_max']
    seed_path = C._build_i2d_augmented_seed(
        res[0], band, f, local_snr_min=MD['manual_ext_local_snr_min'],
        roundlo=-_sr, roundhi=_sr, sharplo=MD['manual_seed_sharp_lo'],
        sharphi=MD['manual_seed_sharp_hi'], bg_subtract_path=bg,
        coarse_bg_box=MD['coarse_bg_box'], label=f'evid:{f}')
    seed = Table.read(seed_path)
    origin = np.asarray(seed['seed_origin']).astype(str)

    # membership of each seed row in the PRODUCTION m7 vetted catalog
    # (m7 seeded from the cross-band seed alone)
    m7 = Table.read(f'{PROD}/catalogs/' + M7.format(f=fl))
    _, sep, _ = sky(seed).match_to_catalog_sky(sky(m7))
    seed['sep_prod_m7_mas'] = sep.to_value(u.mas)
    # m6 vetted properties of the own_m6 rows
    m6 = Table.read(own)
    idx6, sep6, _ = sky(seed).match_to_catalog_sky(sky(m6))
    for c in ('flux', 'flux_err', 'flux_err_prop', 'qfit', 'prominence', 'peak_sb', 'sky_clean', 'nmatch'):
        if c in m6.colnames:
            seed[f'm6_{c}'] = np.asarray(m6[c])[idx6]
    seed['m6_sep_mas'] = sep6.to_value(u.mas)
    seed.write(f'{OUT}/seed_{fl}.fits', overwrite=True)
    s = {o: int((origin == o).sum()) for o in ('crossband', 'own_m6', 'i2d')}
    for o in ('crossband', 'own_m6', 'i2d'):
        k = origin == o
        s[f'{o}_in_prod_m7_60mas'] = int((seed['sep_prod_m7_mas'][k] < 60).sum())
    s['n_m6_vetted'] = len(m6)
    s['n_prod_m7_vetted'] = len(m7)
    s['dedup_mas'] = float(seed.meta.get('DEDUPMAS', np.nan))
    s['seconds'] = round(time.time() - t0)
    summary[f] = s
    print(f, json.dumps(s), flush=True)
    with open(f'{OUT}/summary_{fl}.json', 'w') as fh:
        json.dump(s, fh, indent=1)
