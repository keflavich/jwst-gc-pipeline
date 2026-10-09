"""Generate an STPSF grid (nircam, one detector) matching the cached wd2 grids.
usage: make_grid.py FILT DET OVERSAMPLE  -> grids/nircam_<det>_<filt>_fovp101_samp<N>_npsf16.fits
Mirrors crowdsource_catalogs_long.get_psf_model: NIRCam(), load OPD, filter, detector,
psf_grid(num_psfs=16, all_detectors=False, fov_pixels=101, oversample=N, **nircam_channel_safe_psf_kwargs).
"""
import os
import sys
os.environ.setdefault('STPSF_PATH', '/orange/adamginsburg/repos/webbpsf/data/')
import stpsf
from jwst_gc_pipeline.photometry.psf_channel import nircam_channel_safe_psf_kwargs

filt, det, ov = sys.argv[1].upper(), sys.argv[2].lower(), int(sys.argv[3])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'grids')
nrc = stpsf.NIRCam()
nrc.load_wss_opd(os.path.join(os.environ['STPSF_PATH'], 'MAST_JWST_WSS_OPDs', 'R2024071302-NRCA3_FP1-1.fits'))
nrc.filter = filt
nrc.detector = det.upper()
kw = nircam_channel_safe_psf_kwargs(nrc)
print('channel-safe kwargs', kw, flush=True)
nrc.psf_grid(num_psfs=16, all_detectors=False, verbose=True, save=True, fov_pixels=101,
             oversample=ov, outdir=out, overwrite=True, **kw)
