"""crf/group-0 pedestal and far-field crf/(R_header g0) on the w51 exposure-1 frames of dryall.py
(pedsurvey.one: cal - crf offset and free-slope fit; pedpred.one: B and binned far-field ratio).
``python w51ped.py i n`` runs frames i, i + n, ... into w51ped_<i>.json (run_w51ped.sh)."""
import json
import os
import sys

from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dryall  # noqa: E402
import pedpred  # noqa: E402
import pedsurvey  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
i, n = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (0, 1)
res = []
for field, band, det, fn in dryall.w51_frames()[i::n]:
    a = pedsurvey.one(field, band, det, fn)
    b = pedpred.one(field, band, det, fn)
    a.update({k: b[k] for k in ('B', 'nB', 'g', 'r', 'n', 'destrkmd')})
    with fits.open(fn) as fh:
        a['hdr'] = {k: fh[0].header.get(k) for k in ('READPATT', 'NFRAMES', 'NGROUPS', 'TFRAME', 'TGROUP', 'CAL_VER')}
        a['hdr']['PHOTMJSR'] = fh['SCI'].header.get('PHOTMJSR')
    res.append(a)
    print(f"ROW {band} {det} off={a['off']:+.3f} off_dn={a['off_dn']:+.0f} slope={a['slope']:.3f} ped={a['ped_dn']:+.0f} "
          f"slope_cal={a['slope_cal']:.3f} B={a['B']:+.0f} {a['hdr']} {a['steps']}", flush=True)
    with open(OUT + f'/w51ped_{i}.json', 'w') as fh:
        json.dump(res, fh, default=str)
print('ALLDONE', len(res), flush=True)
