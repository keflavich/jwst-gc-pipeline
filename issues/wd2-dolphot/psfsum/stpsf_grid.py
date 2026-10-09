"""Build a small stpsf psf_grid (NRCA1 F200W, 2x2 = (0,0),(0,2047),(2047,0),
(2047,2047), monochromatic) and compare each PSF's sum with the SIAF pixel
area at its labelled (x, y) and at (y, x).  Also evaluate the
GriddedPSFModel returned by psf_grid at the labelled positions."""
import numpy as np
import stpsf
from siafarea import siaf_area

nrc = stpsf.NIRCam()
nrc.filter = 'F200W'
nrc.detector = 'NRCA1'
grid = nrc.psf_grid(num_psfs=4, all_detectors=False, save=False, verbose=False,
                    fov_pixels=51, oversample=2, monochromatic=2.0e-6)
print('grid_xypos', grid.grid_xypos)
for k in sorted(grid.meta):
    if k.lower().startswith('det_yx'):
        print(k, grid.meta[k])
d = np.asarray(grid.data)
for i, (x, y) in enumerate(grid.grid_xypos):
    s = d[i].sum() / 4
    print(f'PSF {i} at grid_xypos (x={x:.0f}, y={y:.0f}): sum {s:.5f}  '
          f'siaf area (x,y) {siaf_area("nrca1", x, y) * 1e6:.1f}  (y,x) {siaf_area("nrca1", y, x) * 1e6:.1f} [1e-6 arcsec2... idl units]')
