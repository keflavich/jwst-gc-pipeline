"""Core width of unsaturated stars vs detector position and exposure (README §7).

Hypothesis: the per-exposure halo change of saturated stars is a field-dependent LW
focus effect.  If so, the SAME unsaturated field stars around a saturated star should
change core width between dithers in step with its halo.

Per _cal frame: isolated local maxima (S/N > NSIG, nothing flagged within 4 px, no
peak above 10% of it within 8 px), each fitted on 9x9 px by an elliptical-free
circular Gaussian + constant; sigma is the width.  Sky positions from the GWCS
(meta.wcs), for matching the same star across dithers.

    python focus_unsat.py <out.npz> <cal> [<cal> ...]
"""
import os, sys
import numpy as np
from astropy.io import fits
from scipy import ndimage
from scipy.optimize import least_squares
import stdatamodels.jwst.datamodels as dm

NSIG = 50
H = 4
YY, XX = np.mgrid[-H:H+1, -H:H+1]


def gfit(c):
    def res(p):
        a, x0, y0, s, b = p
        return (a*np.exp(-((XX-x0)**2+(YY-y0)**2)/(2*s*s))+b-c).ravel()
    p0 = [c[H, H]-np.median(c), 0, 0, 1.0, np.median(c)]
    r = least_squares(res, p0, bounds=([0, -1, -1, 0.4, -np.inf], [np.inf, 1, 1, 3, np.inf]))
    return r.x, r.success


def measure(fn):
    with fits.open(fn) as f:
        sci = f['SCI'].data.astype(float); dq = f['DQ'].data; err = f['ERR'].data
    bad = ((dq & 3) > 0) | ~np.isfinite(sci)
    badg = ndimage.binary_dilation(bad, iterations=4)
    s = np.nan_to_num(sci)
    loc = ndimage.maximum_filter(s, size=17) == s
    snr = s/np.maximum(np.nan_to_num(err, nan=1e9), 1e-9)
    ys, xs = np.nonzero(loc & (snr > NSIG) & ~badg)
    k = (xs > 12) & (xs < 2035) & (ys > 12) & (ys < 2035)
    ys, xs = ys[k], xs[k]
    # isolation: second-highest local max within 8 px below 10%
    loc2 = (ndimage.maximum_filter(s, size=5) == s) & (s > 0)
    out = []
    for x, y in zip(xs, ys):
        box = s[y-8:y+9, x-8:x+9]; lb = loc2[y-8:y+9, x-8:x+9].copy(); lb[8, 8] = False
        if np.any(box[lb] > 0.1*s[y, x]):
            continue
        p, ok = gfit(s[y-H:y+H+1, x-H:x+H+1])
        if not ok:
            continue
        out.append((x+p[1], y+p[2], p[3], p[0], snr[y, x]))
    out = np.array(out)
    # GWCS (meta.wcs), not the SIP header.  frame_wcs() would be the helper, but its
    # datamodels.open(..., memmap=False) raises TypeError on current stdatamodels
    with dm.open(fn) as m:
        ra, dec = m.meta.wcs(out[:, 0], out[:, 1])
    return np.c_[out, ra, dec]


if __name__ == '__main__':
    res = {}
    for fn in sys.argv[2:]:
        a = measure(fn); res[os.path.basename(fn)] = a
        print(os.path.basename(fn), len(a), 'stars, median sigma', np.median(a[:, 2]).round(4), flush=True)
    np.savez_compressed(sys.argv[1], **res)
