#!/usr/bin/env python
"""Regenerate the per-exposure single-frame _i2d products from the FINAL,
aligned crf (2026-06-26).

The per-exposure ``jw..._mirimage_i2d.fits`` files are an Image2 byproduct
resampled from the RAW-pointing cal -- so they carry the UNCORRECTED WCS (e.g.
the brick F2550W visit-001 frames are off by the 4.2" per-visit misregistration,
and tweakreg/fix_alignment refinements are missing entirely).  "Final" files on
disk with wrong astrometry are confusing + dangerous.  The aligned per-exposure
product is the crf (jw..._mirimage_o{field}_crf.fits): it has the post-tweakreg
+ fix_alignment WCS.  Re-resample each crf into its _i2d so the single-frame i2d
matches the science astrometry.

Usage (env): BASEPATH FILT FIELD, e.g.
    BASEPATH=/orange/adamginsburg/jwst/brick FILT=F2550W FIELD=002 \
        python regenerate_per_exposure_i2d.py
"""
import os
import glob
import warnings
warnings.filterwarnings('ignore')

BASEPATH = os.environ['BASEPATH']
FILT = os.environ['FILT']
FIELD = os.environ['FIELD']
os.environ.setdefault("CRDS_PATH", f"{BASEPATH}/crds/")
os.environ.setdefault("CRDS_SERVER_URL", "https://jwst-crds.stsci.edu")

pipedir = f'{BASEPATH}/{FILT}/pipeline'
from jwst.resample import ResampleStep

crfs = sorted(glob.glob(f'{pipedir}/jw*_mirimage_*o{FIELD}_crf.fits'))
# prefer per-exposure crf (have a 5-digit_5-digit visit_exposure token)
import re
crfs = [c for c in crfs if re.search(r'_\d{5}_\d{5}_mirimage', os.path.basename(c))]
# DEDUPE by output stem: a stem can have multiple crf (e.g. a stale
# non-homogenized `_o002_crf` from the reduce image3 AND the canonical
# tile-homogenized `_align_o002_crf` from the homogenize rebuild).  Both map to
# the same _i2d, so a blind loop overwrites last-write-wins -- which silently
# picked the NON-homogenized one.  Keep only the newest crf per stem.
by_stem = {}
for c in crfs:
    stem = os.path.basename(c).split('_mirimage')[0] + '_mirimage'
    if stem not in by_stem or os.path.getmtime(c) > os.path.getmtime(by_stem[stem]):
        by_stem[stem] = c
crfs = sorted(by_stem.values())
print(f"{len(crfs)} per-exposure crf (deduped by stem -> newest) -> regenerating _i2d")

os.chdir(pipedir)
n_ok = 0
for crf in crfs:
    base = os.path.basename(crf)
    # jw..._mirimage_<...>o{field}_crf.fits -> jw..._mirimage_i2d.fits
    stem = base.split('_mirimage')[0] + '_mirimage'
    out = f'{pipedir}/{stem}_i2d.fits'
    try:
        res = ResampleStep.call(crf, output_dir=pipedir, save_results=False)
        res.save(out, overwrite=True)
        if hasattr(res, 'close'):
            res.close()
        n_ok += 1
    except Exception as ex:
        print(f"  FAILED {base}: {ex}", flush=True)
print(f"regenerated {n_ok}/{len(crfs)} per-exposure _i2d with corrected (crf) WCS")
