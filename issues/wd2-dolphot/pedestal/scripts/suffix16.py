"""Which frame file each band's m6 per-frame fit read (align_o005_crf or
destreak_o005_crf), from median(frame - (residual + model)) on exposure 00001
of one detector; plus the frame-minus-m6-bg pedestal the m7 fit sees.
usage: python suffix16.py ROOT   (production /orange/adamginsburg/jwst/wd2 or a Q_integ tree)"""
import glob
import os
import sys
import warnings

import numpy as np
from astropy.io import fits

warnings.filterwarnings('ignore')
P = '/orange/adamginsburg/jwst/wd2'
ROOT = sys.argv[1]
ALL16 = ['F115W', 'F150W', 'F162M', 'F164N', 'F182M', 'F187N', 'F200W', 'F212N',
         'F250M', 'F277W', 'F300M', 'F323N', 'F335M', 'F405N', 'F410M', 'F466N']


def med(a):
    a = a[np.isfinite(a)]
    return np.median(a) if a.size else np.nan


print(f'{"band":6s} {"det":9s} {"m6 read":10s} {"align-(r+m)":>12s} {"destrk-(r+m)":>13s} '
      f'{"med align":>10s} {"med m6res":>10s} {"m6res mtime":>12s}')
for b in ALL16:
    bl = b.lower()
    det = 'nrcalong' if int(b[1:4]) >= 250 else 'nrca1'
    res = sorted(glob.glob(f'{ROOT}/{b}/pipeline/jw03523-o005_t001_nircam_clear-{bl}-{det}_visit001_vgroup*_exp00001_resbgsub_m6_daophot_basic_residual.fits'))
    if not res:
        print(b, det, 'no m6 residual')
        continue
    r = res[0]
    vg = r.split('vgroup')[1][:5]
    rm = fits.getdata(r, 'SCI') + fits.getdata(r.replace('_residual.fits', '_model.fits'), 'SCI')
    out = {}
    for n in ('align_o005_crf', 'destreak_o005_crf'):
        p = f'{P}/{b}/pipeline/jw03523005001_{vg}_00001_{det}_{n}.fits'
        if os.path.exists(p):
            x = fits.getdata(p, 'SCI')
            out[n] = (med(x - rm), med(x))
        else:
            out[n] = (np.nan, np.nan)
    a, d = out['align_o005_crf'][0], out['destreak_o005_crf'][0]
    read = 'align' if abs(a) < 1e-3 else ('destreak' if abs(d) < 1e-3 else '?')
    mt = os.path.getmtime(os.path.realpath(r))
    import time
    print(f'{b:6s} {det:9s} {read:10s} {a:12.4f} {d:13.4f} {out["align_o005_crf"][1]:10.3f} '
          f'{med(fits.getdata(r, "SCI")):10.3f} {time.strftime("%Y-%m-%d", time.localtime(mt)):>12s}')
