"""Stream every _cal frame of a detector from the AWS mirror, select stars, delete the frame.

    OMP_NUM_THREADS=1 python run_extract.py --detector nrcblong --workers 3 \
        --scratch $SCRATCH/epsfmap [--every 1] [--dithers 1,2,3,4,5,6]

Per frame writes <scratch>/stamps/<detector>/<rootname>_stars.npz (see
extract_stars.py).  Frames already measured are skipped, so the run can be resumed.
Frames that happen to be on disk already in --shared (read-only) are used in place and
never deleted.
"""
import argparse
import os
import sys
import time
import traceback
from multiprocessing import Pool

import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_stars  # noqa: E402
import s3io  # noqa: E402

ALLDET = ['nrca1', 'nrca2', 'nrca3', 'nrca4', 'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4', 'nrcalong', 'nrcblong']
TEMPLATES = {'LW': 'stpsf_test_F480M_NRCB5.fits', 'SW': 'stpsf_test_F212N_NRCB1.fits'}


def work(args):
    key, scratch, shared = args
    det = key.split('_')[-2]
    outdir = os.path.join(scratch, 'stamps', det)
    base = os.path.basename(key)
    outfn = os.path.join(outdir, base.replace('_cal.fits', '_stars.npz'))
    if os.path.exists(outfn):
        return base, 'skip', 0, 0, 0.0
    t0 = time.time()
    local = os.path.join(shared, base) if shared else None
    own = not (local and os.path.exists(local))
    try:
        fn = s3io.fetch(key, os.path.join(scratch, 'tmp')) if own else local
        satname = fits.getheader(fn)['R_SATURA']
        satref = s3io.crds_reference(satname, os.path.join(scratch, 'crds'))
        ch = extract_stars.channel_of(det)
        psf = fits.getdata(os.path.join(scratch, TEMPLATES[ch]), 'DET_DIST')
        n, nw, nc, rej = extract_stars.select_frame(fn, satref, psf, outfn)
    except (OSError, ValueError, KeyError, IndexError) as ex:
        return base, 'FAIL ' + repr(ex) + traceback.format_exc()[-400:], 0, 0, time.time() - t0
    finally:
        if own:
            p = os.path.join(scratch, 'tmp', base)
            if os.path.exists(p):
                os.remove(p)
    return base, 'ok', n, nw, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', required=True)
    ap.add_argument('--scratch', required=True)
    ap.add_argument('--shared', default='')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--every', type=int, default=1, help='use every Nth observation')
    ap.add_argument('--dithers', default='1,2,3,4,5,6')
    ap.add_argument('--program', default='10678')
    a = ap.parse_args()
    keys = s3io.program_cal_keys(a.program, ALLDET, os.path.join(a.scratch, 'cal_keys.json'))
    keys = sorted(k for k, _ in keys if f'_{a.detector}_cal' in k)
    dith = {int(d) for d in a.dithers.split(',')}
    obs = sorted({os.path.basename(k)[7:10] for k in keys})
    use_obs = set(obs[::a.every])
    keys = [k for k in keys if os.path.basename(k)[7:10] in use_obs and int(os.path.basename(k).split('_')[2]) in dith]
    os.makedirs(os.path.join(a.scratch, 'stamps', a.detector), exist_ok=True)
    os.makedirs(os.path.join(a.scratch, 'tmp'), exist_ok=True)
    print(f'{a.detector}: {len(keys)} frames', flush=True)
    tot = 0
    with Pool(a.workers) as pool:
        for i, (base, st, n, nw, dt) in enumerate(pool.imap_unordered(work, [(k, a.scratch, a.shared) for k in keys])):
            tot += n
            print(f'{i+1}/{len(keys)} {base} {st} n={n} wing={nw} {dt:.1f}s total={tot}', flush=True)


if __name__ == '__main__':
    np.seterr(all='ignore')
    main()
