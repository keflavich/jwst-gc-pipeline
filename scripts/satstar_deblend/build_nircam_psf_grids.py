#!/usr/bin/env python
"""Build NIRCam fovp101 PSF grids for the injection harness (the sizes
load_psf_grid / the daophot path expect: nircam_<det>_<band>_fovp101_samp2_npsf16.fits).

art.PSF_DIR only ships F200W/F212N grids, so the F115W/F150W/F405N/... injection
tasks failed. This builds the missing (detector, band) grids with STPSF (nominal
OPD -- injection and recovery use the SAME grid, so consistency, not absolute OPD,
is what matters here).

Usage: build_nircam_psf_grids.py <out_dir> <DET> <BAND>
"""
import os
import sys
import stpsf

OUT = sys.argv[1]
DET = sys.argv[2].lower()
BAND = sys.argv[3].upper()
FOV, NPSF, OVER = 101, 16, 2
fn = f"{OUT}/nircam_{DET}_{BAND.lower()}_fovp{FOV}_samp{OVER}_npsf{NPSF}.fits"
if os.path.exists(fn):
    print(f"[skip] {fn} exists"); sys.exit(0)
print(f"[build] NIRCam {DET} {BAND} fovp{FOV} -> {fn}", flush=True)
n = stpsf.NIRCam()
n.filter = BAND
n.detector = DET.upper()
n.psf_grid(num_psfs=NPSF, oversample=OVER, all_detectors=False, fov_pixels=FOV,
           outdir=OUT, save=True, outfile=None, overwrite=True)
print(f"[done] {fn}", flush=True)
