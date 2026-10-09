"""Unsaturated main2 stars: median dm (ours - dolphot - ZP) for stars that fall only on nrcb1 or only on nrcb3 frames."""
import sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit')
import analyze as an
import run_frames5 as R5

an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2')
BINS = [(14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22)]
for band in sys.argv[1:] or ['150W', '200W']:
    on = {}
    for det in ('nrcb1', 'nrcb3'):
        inside = np.zeros(A.n, bool)
        for e in (1, 2, 3, 4):
            fn = R5.path_of(band, det, e)
            h = fits.getheader(fn, 'SCI')
            w = WCS(h)
            ny, nx = h['NAXIS2'], h['NAXIS1']
            x, y = w.world_to_pixel(A.sky[A.idx])
            inside |= (x > 10) & (x < nx - 10) & (y > 10) & (y < ny - 10)
        on[det] = inside
    dm = A.dm(band)
    ok = A.matched & np.isfinite(dm) & np.isfinite(A.ref[band]) & ~A.rep[band] & ~A.sat[band]
    s1, s3 = ok & on['nrcb1'] & ~on['nrcb3'], ok & on['nrcb3'] & ~on['nrcb1']
    print(f'\n### F{band}: unsaturated main2 stars (not replaced, not saturated), dm = ours - dolphot - ZP (ZP {A.zp[band]:+.3f})')
    print('| dolphot bin | nrcb1 only | nrcb3 only | b3 - b1 |')
    print('|---|---|---|---|')
    for lo, hi in BINS:
        q1, q3 = s1 & (A.ref[band] >= lo) & (A.ref[band] < hi), s3 & (A.ref[band] >= lo) & (A.ref[band] < hi)
        if q1.sum() < 10 or q3.sum() < 10:
            continue
        f = lambda q: f'{np.median(dm[q]):+.3f} / {an.mad(dm[q]):.3f} [{q.sum()}]'
        print(f'| {lo}-{hi} | {f(q1)} | {f(q3)} | {np.median(dm[q3]) - np.median(dm[q1]):+.3f} |')
    for lo, hi in ((16, 21), (18, 21)):
        q1, q3 = s1 & (A.ref[band] >= lo) & (A.ref[band] < hi), s3 & (A.ref[band] >= lo) & (A.ref[band] < hi)
        print(f'| pooled {lo}-{hi} | {np.median(dm[q1]):+.4f} [{q1.sum()}] | {np.median(dm[q3]):+.4f} [{q3.sum()}] | {np.median(dm[q3]) - np.median(dm[q1]):+.4f} |')
