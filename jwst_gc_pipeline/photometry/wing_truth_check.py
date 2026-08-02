"""Rule out forward-model over-suppression: does the forward model preserve the
TRUE PSF amplitude in the surviving (unsaturated) wing pixels?

If forward/true ~ 1 in the wings, the injected star carries the true flux and any
recovery shortfall is the RECOVERY's fault (wing-selfcal under-correction), not a
harness artefact. Result (NRCA1 F200W, NGROUPS=7): forward/true = 0.96-1.00 in
all unsaturated annuli up to hard saturation -> wings preserved; the ~1 mag
hard-sat recovery bias is real under-correction (wing-selfcal applies ~median
1.1x while the deep-star truth/masked ratio reaches 2-4x at r>15px).
"""
import numpy as np
from jwst_gc_pipeline.photometry import artificial_stars as art
from jwst_gc_pipeline.photometry import saturation_forward_model as sfm
from jwst.datamodels import dqflags


def check(det='NRCA1', band='F200W', photmjsr=1.964, tgroup=53.68, ngroups=7,
          mags=(12.0, 10.0, 8.5), pixar_sr=2.29e-14):
    grid = art.load_psf_grid(band, det.lower())
    refs = sfm.load_detector_refs(det, box=(1000, 1101, 1000, 1101))
    ny = nx = 101
    yy, xx = np.mgrid[0:ny, 0:nx]
    gain = float(np.median(refs['gain']))
    rad = np.hypot(xx - 50, yy - 50)
    for mag in mags:
        flux = float(art.mag_to_imflux(mag, band, pixar_sr))
        truth = np.clip(grid.evaluate(xx, yy, flux, 50, 50), 0, None)
        sim = sfm.simulate_cal(truth / photmjsr * gain, refs, ngroups=ngroups,
                               tgroup=tgroup, f_bf=0.0)
        rec = sim['sci'] * photmjsr
        sat = (sim['dq'] & dqflags.pixel['SATURATED']) != 0
        print(f'mag={mag} nsat={int(sat.sum())}:')
        for lo, hi in [(3, 5), (5, 8), (8, 12), (12, 20), (20, 35)]:
            ann = (rad >= lo) & (rad < hi) & ~sat & np.isfinite(rec) & (truth > 0)
            if ann.sum() > 3:
                print(f'   r{lo}-{hi}: forward/true={np.median(rec[ann] / truth[ann]):.3f} '
                      f'(n={int(ann.sum())})')


if __name__ == '__main__':
    import os
    os.environ.setdefault('CRDS_PATH', '/orange/adamginsburg/jwst/crds')
    os.environ.setdefault('STPSF_PATH', '/orange/adamginsburg/jwst/stpsf-data/')
    check()
