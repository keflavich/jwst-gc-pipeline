"""What are the giant SATURATED components in the F277W nrcblong frames?"""
import glob, sys
import numpy as np
from astropy.io import fits
from scipy import ndimage
from jwst.datamodels import dqflags
P = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/tree_mainfcbg/F277W/pipeline'
f = sorted(glob.glob(f'{P}/jw03523005001_*_nrcblong_align_o005_crf.fits'))[0]
h0 = fits.getheader(f, 0)
print(f.split('/')[-1], {k: h0.get(k) for k in ('READPATT', 'NGROUPS', 'NINTS', 'NFRAMES', 'EFFEXPTM', 'TGROUP', 'SUBARRAY')})
sci = fits.getdata(f, 'SCI'); dq = fits.getdata(f, 'DQ')
sat = (dq & 2) != 0
lab, n = ndimage.label(sat)
sizes = np.bincount(lab.ravel()); sizes[0] = 0
order = np.argsort(sizes)[::-1]
print('n comps', n, 'sat px', sat.sum(), 'px in comps >1000:', sizes[sizes > 1000].sum(), 'n comps >1000:', (sizes > 1000).sum())
names = {v: k for k, v in dqflags.pixel.items() if v and (v & (v - 1)) == 0}
for L in order[:6]:
    m = lab == L
    s = sci[m]
    ys, xs = np.where(m)
    bits = {}
    for v, nm in names.items():
        c = int(((dq[m] & v) != 0).sum())
        if c: bits[nm] = c
    print(f'comp {L}: size {sizes[L]}, bbox x {xs.min()}-{xs.max()} y {ys.min()}-{ys.max()}, '
          f'finite SCI {np.isfinite(s).mean():.3f}, SCI p10/50/90 {np.nanpercentile(s, [10, 50, 90]).round(1)}')
    print('   bits:', dict(sorted(bits.items(), key=lambda kv: -kv[1])[:8]))
# ramp-level check: the giant comp in the _jump file groupdq
