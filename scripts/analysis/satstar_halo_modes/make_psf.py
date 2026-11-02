"""STPSF F480M PSF for one exposure of a saturated star (input to halo_photometry.py).

    python make_psf.py <x> <y> <out.npy> [detector=NRCB5] [opd=R2026091802-NRCA1_FP6-1.fits]

Saves the OVERDIST image (fov 401 native px, oversample 4) as float32.
"""
import sys

import numpy as np
import stpsf

from jwst_gc_pipeline.photometry.psf_channel import nircam_channel_safe_psf_kwargs

x, y, out = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
det = sys.argv[4] if len(sys.argv) > 4 else 'NRCB5'
opd = sys.argv[5] if len(sys.argv) > 5 else 'R2026091802-NRCA1_FP6-1.fits'
nc = stpsf.NIRCam()
nc.filter = 'F480M'
nc.detector = det
nc.detector_position = (x, y)
nc.load_wss_opd(opd, plot=False, verbose=False, use_exact_wss_target_phase=False)
p = nc.calc_psf(fov_pixels=401, oversample=4, **nircam_channel_safe_psf_kwargs(nc))
np.save(out, p['OVERDIST'].data.astype(np.float32))
