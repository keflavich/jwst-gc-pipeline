"""Split the per-detector offsets of ours - dolphot with r = 5 px AREA-corrected apertures.

Inputs: apclosure/frames_<band>.ecsv (isolated stars, per-frame PSF flux and
aperture sums on our crf frames).  Per star, averaged over frames:
  e = ours(PSF, old loader) - ap_area ;  e - (pred - predT) estimates ours after the grid fix
  c = dolphot - ap_area                ;  c + pred removes the dolphot area double count
Per detector (from the frame name), medians relative to the band median.
"""
import os
import re
import sys

import numpy as np
from astropy.table import Table

sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an  # noqa: E402

an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2')
P = np.load(f'{an.Q}/photomver/areapred_main2.npz')
PT = np.load(f'{an.Q}/psfsum/predT_main2.npz')
AP = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/apclosure'
BANDS = ['150W', '164N', '182M', '187N', '200W', '212N']
DET = re.compile(r'_(nrc[ab][1-4]|nrc[ab]long)_')
R = 5


def mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


print('| band | det | N | ours_fix - ap | dolphot + pred - ap | ours_fix - (dolphot + pred) | closure.md median |')
print('|---|---|---|---|---|---|---|')
for b in BANDS:
    fn = f'{AP}/frames_{b}.ecsv'
    if not os.path.exists(fn):
        continue
    T = Table.read(fn)
    T = T[(T['src'] == 1) & (T['psf'] > 0) & (T[f'area{R}'] > 0)]
    dets = np.array([DET.search(f).group(1) for f in T['frame']])
    ids, inv = np.unique(T['i'], return_inverse=True)
    n = np.bincount(inv)
    psf = np.bincount(inv, T['psf']) / n
    ar = np.bincount(inv, T[f'area{R}']) / n
    det = np.array([dets[inv == k][0] for k in range(len(ids))])
    keep = n >= 2
    ids, psf, ar, det = ids[keep], psf[keep], ar[keep], det[keep]
    pr, pt, dol = P[b][ids], PT[b][ids], A.ref[b][ids]
    m = lambda f: -2.5 * np.log10(f)
    ours = m(psf) - m(ar) - (pr - pt)
    dolc = dol - m(ar) + pr
    ok = np.isfinite(ours) & np.isfinite(dolc)
    ours, dolc, det = ours[ok] - np.median(ours[ok]), dolc[ok] - np.median(dolc[ok]), det[ok]
    cl = None
    cfn = f'/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/gridfix/stars_{b}.ecsv'
    if os.path.exists(cfn):
        S = Table.read(cfn)
        v = np.asarray(S['dm']) + np.asarray(S['dfix5']) - np.asarray(S['pred'])
        g = np.isfinite(v)
        cl = {d: np.median(v[g & (np.asarray(S['det']) == d)]) - np.median(v[g]) for d in np.unique(S['det'][g])}
    for d in np.unique(det):
        s = det == d
        if s.sum() < 15:
            continue
        o, dc = np.median(ours[s]), np.median(dolc[s])
        c = f'{cl[d]:+.4f}' if cl and d in cl else ''
        print(f'| F{b} | {d} | {s.sum()} | {o:+.4f} | {dc:+.4f} | {o - dc:+.4f} | {c} |')
