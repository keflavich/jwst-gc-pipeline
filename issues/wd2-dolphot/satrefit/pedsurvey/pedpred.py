"""Does a crf/group-0 pedestal B predict the survey's R(2440)/R_header?

B_dn = median(crf / R_header - g0) over far-field pixels (edt >= 25 from SATURATED, not DO_NOT_USE,
crf != 0) with 500 <= g0 < 2000.  If crf = R_header (g0 + B_dn), the measured curve at g0 = 2440
reads R_header (1 + B_dn / 2440).  Also stores far-field crf / (R_header g0) in log-g0 bins for plotting.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit')
from satrefit_core import S, fits, ndimage  # noqa: E402
import pedsurvey as P  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
EDGES = np.geomspace(150, 40000, 25)


def one(field, band, det, fn):
    with fits.open(fn, memmap=False) as fh:
        hdr = fh[0].header
        crf = np.array(fh['SCI'].data, float)
        dq = np.array(fh['DQ'].data)
        photmjsr = fh['SCI'].header.get('PHOTMJSR', hdr.get('PHOTMJSR'))
    Rh = S.zeroframe_header_R(hdr, photmjsr)
    with fits.open(S._find_ramp_for(fn), memmap=True) as r:
        g0 = np.array(r['SCI'].data[0, 0], float)
    sat = (dq & S.dqflags.pixel['SATURATED']) != 0
    dnu = (dq & S.dqflags.pixel['DO_NOT_USE']) != 0
    edt = ndimage.distance_transform_edt(~sat)
    far = np.isfinite(crf) & np.isfinite(g0) & ~sat & ~dnu & (edt >= 25) & (crf != 0)
    s = far & (g0 >= 500) & (g0 < 2000)
    B = float(np.median(crf[s] / Rh - g0[s])) if s.sum() >= 50 else float('nan')
    gc, rc, nc = [], [], []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = far & (g0 >= lo) & (g0 < hi)
        if m.sum() >= 30:
            gc.append(float(np.median(g0[m])))
            rc.append(float(np.median(crf[m] / (Rh * g0[m]))))
            nc.append(int(m.sum()))
    return dict(field=field, band=band, det=det, file=os.path.basename(fn), Rh=float(Rh), B=B, nB=int(s.sum()),
                g=gc, r=rc, n=nc, destrkmd=hdr.get('DESTRKMD'))


if __name__ == '__main__':
    res = []
    for field, band, det, fn in P.frames():
        if not os.path.exists(fn) or S._find_ramp_for(fn) is None:
            continue
        r = one(field, band, det, fn)
        res.append(r)
        print(f"{field:6s} {band} {det:9s} B={r['B']:+.0f} DN (n={r['nB']}) 1+B/2440={1 + r['B'] / 2440:.4f} "
              f"DESTRKMD={r['destrkmd']}", flush=True)
        json.dump(res, open(OUT + '/pedpred.json', 'w'))
    print('ALLDONE', len(res), flush=True)
