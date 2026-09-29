"""Program-wide extraction of saturated-star cutouts for the halo-vs-detector-position
calibration (README §8, issue in the #993 series).

Streams every 10678 F480M LW `_cal` from the public S3 mirror (s3data.py), keeps only

  * the list of SATURATED (DQ bit 2) connected regions >= NMIN_LIST px (for masking and
    for linking stars between exposures), with their pixel centroid, size and GWCS
    sky position;
  * a star-centred cutout (SCI float32, VAR_POISSON+VAR_RNOISE float16, DQ low byte)
    around every region >= NMIN_CUT px, half-width 128 px (256 px for >= 1500 px);
  * the exposure's GWCS (ASDF), so that per-pixel V2V3 can be recomputed later
    (CLAUDE.md astrometry rule #2: never the SIP header),

and deletes the full frame.  Output: <out>/<rootname>.npz and <out>/<rootname>_wcs.asdf.

    python halocal_extract.py <outdir> [nworkers]
"""
import os, sys, json, subprocess, warnings, traceback
import concurrent.futures as cf
import numpy as np
from scipy import ndimage
import asdf
import stdatamodels.jwst.datamodels as dm
warnings.simplefilter('ignore')

BASE = 'https://stpubdata.s3.amazonaws.com/'
NMIN_LIST = 20
NMIN_CUT = 250
NBIG = 1500
HW_SMALL, HW_BIG = 128, 256


def cut(a, xc, yc, hw, fill):
    n = 2*hw
    X0, Y0 = xc-hw, yc-hw
    o = np.full((n, n), fill, dtype=a.dtype)
    y0, y1 = max(0, Y0), min(a.shape[0], Y0+n)
    x0, x1 = max(0, X0), min(a.shape[1], X0+n)
    if y1 > y0 and x1 > x0:
        o[y0-Y0:y1-Y0, x0-X0:x1-X0] = a[y0:y1, x0:x1]
    return o, X0, Y0


def process(key, outdir):
    root = os.path.basename(key).replace('.fits', '')
    out = os.path.join(outdir, root+'.npz')
    if os.path.exists(out):
        return root, 'exists'
    tmp = os.path.join(outdir, 'tmp_'+root+'.fits')
    subprocess.run(['curl', '-sS', '--retry', '4', '-o', tmp, BASE+key], check=True)
    try:
        m = dm.open(tmp)
        sci = m.data.astype(np.float32)
        dq = m.dq.astype(np.uint32)
        var = (m.var_poisson+m.var_rnoise).astype(np.float32)
        sat = (dq & 2) > 0
        lab, nl = ndimage.label(sat, structure=np.ones((3, 3)))
        sz = np.bincount(lab.ravel(), minlength=nl+1)
        keep = np.nonzero(sz >= NMIN_LIST)[0]
        keep = keep[keep > 0]
        cy, cx = np.array(ndimage.center_of_mass(sat, lab, keep)).T if len(keep) else (np.zeros(0), np.zeros(0))
        ra, dec = m.meta.wcs(cx, cy)
        dq8 = (dq & 0xff).astype(np.uint8)
        cuts = {}
        for i, (x, y, n) in enumerate(zip(cx, cy, sz[keep])):
            if n < NMIN_CUT:
                continue
            hw = HW_BIG if n >= NBIG else HW_SMALL
            xc, yc = int(round(x)), int(round(y))
            s, X0, Y0 = cut(sci, xc, yc, hw, np.float32(np.nan))
            v, _, _ = cut(var, xc, yc, hw, np.float32(np.nan))
            q, _, _ = cut(dq8, xc, yc, hw, np.uint8(1))
            cuts[f's{i}_sci'] = s
            cuts[f's{i}_var'] = v.astype(np.float16)
            cuts[f's{i}_dq'] = q
            cuts[f's{i}_org'] = np.array([X0, Y0])
        meta = dict(root=root, filter=m.meta.instrument.filter, detector=m.meta.instrument.detector,
                    expstart=m.meta.exposure.start_time, pa_v3=m.meta.pointing.pa_v3,
                    obs=m.meta.observation.observation_number, visit=m.meta.observation.visit_number,
                    exposure=m.meta.observation.exposure_number,
                    bkg=float(np.nanmedian(sci[::8, ::8])))
        np.savez_compressed(out, x=cx, y=cy, n=sz[keep], ra=np.asarray(ra), dec=np.asarray(dec),
                            meta=json.dumps(meta), **cuts)
        asdf.AsdfFile({'wcs': m.meta.wcs}).write_to(os.path.join(outdir, root+'_wcs.asdf'))
        m.close()
        return root, f'{len(keep)} regions, {sum(1 for k in cuts if k.endswith("_sci"))} cutouts'
    finally:
        os.remove(tmp)


if __name__ == '__main__':
    outdir = sys.argv[1]
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    os.makedirs(outdir, exist_ok=True)
    keys = [k for k, _ in json.load(open(os.path.join(os.path.dirname(os.path.abspath(outdir)), 'lw_cal_keys.json')))]
    with cf.ProcessPoolExecutor(nw) as ex:
        futs = {ex.submit(process, k, outdir): k for k in keys}
        for i, f in enumerate(cf.as_completed(futs)):
            try:
                r, msg = f.result()
                print(i, r, msg, flush=True)
            except (OSError, subprocess.CalledProcessError, ValueError, KeyError) as err:
                print(i, 'FAILED', futs[f], repr(err), flush=True)
                traceback.print_exc()
