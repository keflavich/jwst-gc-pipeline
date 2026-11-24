"""The real ``_build_i2d_augmented_seed`` on the m6 residual of one tile and band, bright window off (A) and on (B).

usage: python realfunc.py TILE BAND OUTDIR [--snr S] [--prom P] [--frac F] [--width W] [--rmax 1.5] [--no-a]
                                        [--root /orange/adamginsburg/jwst/gc-treasury] [--code PATH]

--snr, --prom, --frac and --width default to the ``manual_seed_round_bright_*`` values of ``manual_defaults.py``.
Per module (nrca, nrcb): the production m7 i2dseed catalog of the tile is copied to OUTDIR/{A,B}/ as the previous catalog.  The detection image is the m6
mergedcat residual i2d minus its smoothed background, and the loose window is the production one (0.8, prominence 5).
  A: bright window off  -> 0 new rows are expected (every detection is already in the production seed): the unchanged path.
  B: bright window on   -> the new rows are the bright-window seeds.
The stage counts are in the log lines of the function ("bright roundness: ...").  B also writes two tables under OUTDIR/B/:
  o<T>_<B>_<module>_bright.csv  sky position and flux of the bright-window seeds;
  o<T>_<B>_<module>_widths.csv  module pixel position and short-axis width (pixels and PSF standard deviations) of every detection that reaches the width gate,
                                that is, passes prominence, unsubtracted fraction and the hole guard (the width gate does not change the list).
Reads the gc-treasury tree, writes only under OUTDIR.
"""
import os
import shutil
import sys
import warnings

import numpy as np
from astropy.table import Table

warnings.filterwarnings('ignore')
sys.dont_write_bytecode = True
a = sys.argv[1:]
T, B, OUT = a[:3]


def opt(name, default):
    return float(a[a.index(name) + 1]) if name in a else default


HERE = os.path.dirname(os.path.abspath(__file__))
CODE = a[a.index('--code') + 1] if '--code' in a else os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
ROOT = a[a.index('--root') + 1] if '--root' in a else '/orange/adamginsburg/jwst/gc-treasury'
sys.path.insert(0, CODE)
os.environ.setdefault('GC_ALLOW_DEV', '1')
from jwst_gc_pipeline.photometry import cataloging as C  # noqa: E402
from jwst_gc_pipeline.photometry.manual_defaults import MANUAL_DEFAULTS as MD  # noqa: E402

_ftab = Table.read(C._L.fwhm_table_path())
SIGMA_PSF = float(_ftab[_ftab['Filter'] == B]['PSF FWHM (pixel)'][0]) / 2.355
_widths = []
_minor_axis_sigma = C._minor_axis_sigma


def _recording_minor_axis_sigma(img, x, y, *args, **kwargs):
    """The width the gate measures, recorded per detection."""
    w = _minor_axis_sigma(img, x, y, *args, **kwargs)
    _widths.append(np.column_stack([np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float)), np.atleast_1d(np.asarray(w, float))]))
    return w


C._minor_axis_sigma = _recording_minor_axis_sigma

f = B.lower()
for MOD in ('nrca', 'nrcb'):
    base = f'{ROOT}/{B}/pipeline/jw10678-o{T}_t001_nircam_clear-{f}-{MOD}_resbgsub_hybpsf_'
    resid = base + 'm6_daophot_basic_mergedcat_residual_i2d.fits'
    bgp = base + 'm6_daophot_basic_mergedcat_residual_smoothed_bg_i2d.fits'
    dat = f'{ROOT}/{B}/pipeline/jw10678-o{T}_t001_nircam_clear-{f}-{MOD}_data_i2d.fits'
    prod = f'{ROOT}/catalogs/crossband_seed_manual_o{T}_{MOD}_{f}_i2dseed.fits'
    if not all(os.path.exists(p) for p in (resid, bgp, dat, prod)):
        print(f'o{T} {B} {MOD}: an input is missing, skipped', flush=True)
        continue
    for tag in (('B',) if '--no-a' in a else ('A', 'B')):
        d = f'{OUT}/{tag}'
        os.makedirs(d, exist_ok=True)
        prev = f'{d}/o{T}_{MOD}_{f}_vetted.fits'
        shutil.copy(prod, prev)
        kw = dict(local_snr_min=float(MD['manual_ext_local_snr_min']),
                  roundlo=-float(MD['manual_seed_round_max']), roundhi=float(MD['manual_seed_round_max']),
                  round_loose_max=0.8, round_loose_prom_min=float(MD['manual_seed_round_loose_prom_min']),
                  sharplo=float(MD['manual_seed_sharp_lo']), sharphi=float(MD['manual_seed_sharp_hi']),
                  bg_subtract_path=bgp, coarse_bg_box=int(MD['coarse_bg_box']), label=f'{tag}:o{T}{B}{MOD}')
        if tag == 'B':
            kw.update(round_bright_max=opt('--rmax', 1.5),
                      round_bright_snr_min=opt('--snr', float(MD['manual_seed_round_bright_snr_min'])),
                      round_bright_prom_min=opt('--prom', float(MD['manual_seed_round_bright_prom_min'])),
                      round_bright_frac_min=opt('--frac', float(MD['manual_seed_round_bright_frac_min'])),
                      round_bright_width_min=opt('--width', float(MD['manual_seed_round_bright_width_min'])), data_i2d_path=dat)
        _widths.clear()
        out = C._build_i2d_augmented_seed(resid, prev, B, **kw)
        t = Table.read(out)
        n_prod = len(Table.read(prod))
        print(f'RESULT {tag} o{T} {B} {MOD}: production seed {n_prod} rows; output {len(t)} rows; rows new to the production seed: {len(t) - n_prod}'
              + (f"; seed_round_bright {int(np.asarray(t['seed_round_bright'], bool).sum())}" if 'seed_round_bright' in t.colnames else ''), flush=True)
        if tag == 'B' and 'seed_round_bright' in t.colnames:
            br = np.asarray(t['seed_round_bright'], bool)
            sc = t['skycoord'][br]
            tt = Table()
            tt['ra'] = sc.ra.deg
            tt['dec'] = sc.dec.deg
            tt['flux'] = np.asarray(t['flux'])[br]
            tt.write(f'{d}/o{T}_{B}_{MOD}_bright.csv', format='ascii.csv', overwrite=True)
        if tag == 'B':
            wd = np.vstack(_widths) if _widths else np.zeros((0, 3))
            tw = Table()
            tw['x'] = wd[:, 0]
            tw['y'] = wd[:, 1]
            tw['width_px'] = wd[:, 2]
            tw['width_over_sigma_psf'] = wd[:, 2] / SIGMA_PSF
            tw.write(f'{d}/o{T}_{B}_{MOD}_widths.csv', format='ascii.csv', overwrite=True)
