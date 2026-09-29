"""STPSF models on a grid of detector positions, for comparison with the empirical ePSFs.

    OMP_NUM_THREADS=1 python stpsf_models.py --detector NRCB5 --filter F480M --fov 65 \
        --ngrid 5 --opd R2026091802-NRCA1_FP6-1.fits --out stpsf_NRCB5_F480M.npz

* OPD: the on-orbit WSS OPD closest in time to the observations (10678 ran
  2026-09-15..19; R2026091802 was sensed 2026-09-18 05:22 UTC).  The file is read
  from the MAST AWS mirror (s3://stpubdata/jwst/public/R2026091802/...) into
  $STPSF_PATH/MAST_JWST_WSS_OPDs, where stpsf finds it without querying MAST.
* ``use_exact_wss_target_phase=False``: the WAS target-phase map for field point FP6
  (wss_target_phase_fp6.fits) is not in the stpsf data files available here, so the SI
  WFE at the sensing field point is backed out with the ISIM CV3 Zernike model.
* Output: OVERDIST (oversampled x4, with distortion, charge diffusion and IPC applied
  -- its 4x4 block sum equals DET_DIST exactly) per grid position.
"""
import argparse
import time

import numpy as np
import stpsf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', required=True)
    ap.add_argument('--filter', required=True)
    ap.add_argument('--fov', type=int, required=True)
    ap.add_argument('--ngrid', type=int, default=5)
    ap.add_argument('--oversample', type=int, default=4)
    ap.add_argument('--opd', default='R2026091802-NRCA1_FP6-1.fits')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    nc = stpsf.NIRCam()
    nc.filter = a.filter
    nc.detector = a.detector
    if a.opd:
        nc.load_wss_opd(a.opd, plot=False, verbose=False, use_exact_wss_target_phase=False)
    c = (np.arange(a.ngrid) + 0.5) * 2048 / a.ngrid
    pos, psfs = [], []
    for y in c:
        for x in c:
            t = time.time()
            nc.detector_position = (x, y)
            p = nc.calc_psf(fov_pixels=a.fov, oversample=a.oversample)
            psfs.append(p['OVERDIST'].data.astype(np.float32))
            pos.append((x, y))
            print(a.detector, x, y, f'{time.time() - t:.1f}s', flush=True)
    np.savez_compressed(a.out, overdist=np.array(psfs), pos=np.array(pos), detector=a.detector,
                        filter=a.filter, oversample=a.oversample, opd=a.opd or 'default',
                        fov=a.fov)


if __name__ == '__main__':
    main()
