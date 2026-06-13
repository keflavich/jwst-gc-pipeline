#!/usr/bin/env python
"""
Pre-build large fov_pixels MIRI PSF grids for forced (outside-FOV)
saturated-star fits (2026-06-13).

MIRI analogue of sickle/build_large_psf_grids.py (which is NIRCam-only).
Outside-FOV bright stars throw diffraction spikes ~40" into the imager FOV;
at the MIRI 0.11"/px scale that is ~360 px, so the default fovp512 grid
(256 px radius) truncates them and the forced fit produces wrong fluxes.
saturated_star_finding.py now looks for
    {path_prefix}/miri_mirim_<filt>_fovp1024_samp2_npsf16.fits
when a forced source is present; this builds it.

Usage:
    python build_large_psf_grids_miri.py <out_dir> <FILTER> [obsdate]
e.g.
    python build_large_psf_grids_miri.py /orange/adamginsburg/jwst/sickle/psfs F770W 2024-08-23T12:03:43
"""
import os
import sys
import stpsf

OUT_DIR = sys.argv[1]
FILTER = sys.argv[2]
OBSDATE = sys.argv[3] if len(sys.argv) > 3 else None
FOV_PIXELS = 1024
NPSF = 16
OVERSAMPLE = 2

fn = (f"{OUT_DIR}/miri_mirim_{FILTER.lower()}"
      f"_fovp{FOV_PIXELS}_samp{OVERSAMPLE}_npsf{NPSF}.fits")
if os.path.exists(fn):
    print(f"[skip] {fn} already exists")
    sys.exit(0)

print(f"[build] MIRI {FILTER} fov_pixels={FOV_PIXELS} -> {fn}", flush=True)
psfgen = stpsf.MIRI()
psfgen.filter = FILTER
psfgen.detector = 'MIRIM'
if OBSDATE:
    try:
        psfgen.load_wss_opd_by_date(OBSDATE)
    except Exception as exc:
        print(f"  load_wss_opd_by_date warning: {exc}")
psfgen.psf_grid(num_psfs=NPSF, oversample=OVERSAMPLE, all_detectors=False,
                fov_pixels=FOV_PIXELS, outdir=OUT_DIR, save=True,
                outfile=None, overwrite=True)
print(f"[done] wrote {fn}", flush=True)
