"""Compare the unit-PSF grids per detector: daophot grid (fovp101) vs satstar grid (fovp512 SW / fovp1024 LW).
Reports for each detector and band: sum of the grid cutout /oversample^2 (enclosed fraction in the cutout) and the central-pixel peak for
the grid member closest to the detector centre (index npsf//2)."""
import sys
import glob
import numpy as np
from astropy.io import fits
P = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/tree_main2/psfs'
rows = []
for band, dets, big in (('f150w', ['nrca1', 'nrca2', 'nrca3', 'nrca4', 'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4'], 512),
                        ('f200w', ['nrca1', 'nrca2', 'nrca3', 'nrca4', 'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4'], 512),
                        ('f250m', ['nrca5', 'nrcb5'], 1024), ('f300m', ['nrca5', 'nrcb5'], 1024)):
    for d in dets:
        r = [band, d]
        for fov in (101, big):
            fn = f'{P}/nircam_{d}_{band}_fovp{fov}_samp2_npsf16.fits'
            with fits.open(fn) as h:
                a = np.asarray(h[0].data, float)
            if a.ndim == 4:
                a = a.reshape(-1, a.shape[-2], a.shape[-1])
            tot = a.sum(axis=(1, 2)) / 4.0
            ny, nx = a.shape[1:]
            c = a[:, ny // 2 - 1:ny // 2 + 1, nx // 2 - 1:nx // 2 + 1]
            r += [a.shape, float(np.median(tot)), float(np.median(c.max(axis=(1, 2)))), float(np.median(a.max(axis=(1, 2))))]
        rows.append(r)
        print(r, flush=True)
