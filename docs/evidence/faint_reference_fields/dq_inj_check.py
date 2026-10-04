"""How often an injected star's 3x3 core lands on DO_NOT_USE or SATURATED
pixels in the reference-field cutout frames.

DQ is the parent frame's (the cutout is re-cropped from it every phase), so the
frames of one run serve every seed.  usage: python dq_inj_check.py <variant> <out.json>
"""
import glob
import json
import os
import sys

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy import units as u

from jwst_gc_pipeline.frame_wcs import frame_wcs
from jwst_gc_pipeline.photometry import reference_fields as RF

DO_NOT_USE, SATURATED = 1, 2


def main(variant, out):
    _, fields = RF.load_config()
    res = {}
    for name, spec in sorted(fields.items()):
        rdir = RF.run_dir(spec, variant, 1)
        tabs = {s: Table.read(RF.injection_table_path(name, s)) for s in spec['seeds']}
        r = dict(frames=0, star_frames=0, flagged_dnu=0, flagged_sat=0,
                 stars=0, stars_any=0, stars_majority=0, per_seed={})
        hits = {}
        for filt in spec['filters']:
            files = sorted(glob.glob(f'{rdir}/{filt}/pipeline/*_cutout_{os.path.basename(rdir)}.fits'))
            for fn in files:
                r['frames'] += 1
                dq = fits.getdata(fn, 'DQ')
                ny, nx = dq.shape
                ww = frame_wcs(fn)
                for s, t in tabs.items():
                    xs, ys = ww.world_to_pixel(SkyCoord(np.asarray(t['ra'], float) * u.deg,
                                                        np.asarray(t['dec'], float) * u.deg))
                    ix = np.rint(np.asarray(xs, float)).astype(int)
                    iy = np.rint(np.asarray(ys, float)).astype(int)
                    for i in range(len(t)):
                        if not (1 <= ix[i] < nx - 1 and 1 <= iy[i] < ny - 1):
                            continue
                        core = dq[iy[i] - 1:iy[i] + 2, ix[i] - 1:ix[i] + 2]
                        dnu = bool(np.any(core & DO_NOT_USE))
                        sat = bool(np.any(core & SATURATED))
                        r['star_frames'] += 1
                        r['flagged_dnu'] += dnu
                        r['flagged_sat'] += sat
                        h = hits.setdefault((filt, s, i), [0, 0])
                        h[0] += 1
                        h[1] += dnu or sat
        for (filt, s, i), (n, k) in hits.items():
            r['stars'] += 1
            r['stars_any'] += k > 0
            r['stars_majority'] += k > n / 2
            if k:
                ps = r['per_seed'].setdefault(f'{filt}_s{s}', [])
                ps.append([i, k, n])
        res[name] = r
        print(name, {k: v for k, v in r.items() if k != 'per_seed'})
    with open(out, 'w') as fh:
        json.dump(res, fh, indent=1)


if __name__ == '__main__':
    main(*sys.argv[1:])
