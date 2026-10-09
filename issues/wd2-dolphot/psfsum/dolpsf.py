"""Per-position sums of the dolphot NIRCam PSF library (nircammakepsf.c layout:
25 positions, y-major (ct = y*5 + x) at x, y = 0, 512, ..., 2048; 7x7 subpixel
phases; 49x49 pixels) against the SIAF pixel area at (x, y) and (y, x)."""
import numpy as np
from siafarea import siaf_area, corr, slope

NP, NPH, NR = 5, 7, 49
for band, det in [('F200W', 'nrca1'), ('F200W', 'nrca3'), ('F200W', 'nrcb3'),
                  ('F200W', 'nrcb4'), ('F150W', 'nrcb1'),
                  ('F250M', 'nrca5'), ('F300M', 'nrca5'), ('F300M', 'nrcb5'),
                  ('F410M', 'nrca5')]:
    d = np.fromfile(f'dolsrc/nircam/data/{band}.{det}.psf', dtype='>f4')
    d = d.reshape(NP, NP, NPH, NPH, NR, NR)  # [y][x][phy][phx][yy][xx]
    s0 = d[:, :, 3, 3].sum(axis=(-2, -1))
    sm = d[:, :, 1:6, 1:6].sum(axis=(-2, -1)).mean(axis=(-2, -1))
    pos = np.array([0, 512, 1024, 1536, 2047.])
    yy, xx = np.meshgrid(pos, pos, indexing='ij')
    a = siaf_area(det, xx.ravel(), yy.ravel())
    aT = siaf_area(det, yy.ravel(), xx.ravel())
    s = sm.ravel()
    print(f'{band} {det}: sum range {s.min():.4f}-{s.max():.4f} (phase0 {s0.min():.4f}-{s0.max():.4f}); '
          f'vs area(x,y) corr {corr(s, a):+.3f} slope {slope(s, a):+.3f}; '
          f'vs area(y,x) corr {corr(s, aT):+.3f} slope {slope(s, aT):+.3f}')
