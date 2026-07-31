"""Does the over-rejection scale with the RESAMPLE KERNEL? (issue #161, test C)

If ``outlier_detection`` flags real PSF signal because the drizzle+blot round
trip smears an undersampled PSF, then the number of flagged pixels must fall
when the resample kernel is made narrower -- the thresholds are untouched, only
the kernel changes.  A genuine cosmic-ray population cannot care what kernel the
median was drizzled with.

Rebuilds the outlier-detection median for several ``kernel``/``pixfrac``
settings (using the pipeline's own ``median_with_resampling``), blots it to the
probe frame, and applies ``stcal``'s production flagging test.  Nothing is
written except a small summary; no 96-frame ``_crf`` products are produced.
"""

import glob
import json
import os

import numpy as np
from astropy.io import fits

BASE = os.environ.get("OUTLIER_BASE_DIR", "/orange/adamginsburg/jwst/brick/F200W/pipeline")
DIST = os.environ.get("OUTLIER_DIST_DIR", "/blue/adamginsburg/adamginsburg/tmp/outlier_dist")
SRC_ASN = os.environ.get(
    "OUTLIER_ASN", f"{BASE}/jw01182-o004_20251215t004215_image3_00002_nrca_asn.json"
)
OUTDIR = os.environ.get("OUTLIER_DIAG_OUT", f"{DIST}/diagnosis")
FRAME = os.environ.get("OUTLIER_FRAME", "jw01182004001_04101_00001_nrca1")
BRIGHT_STARS = [(1362, 84), (1586, 1643), (207, 1115)]
BOX = 110
SNR1, SNR2, SCALE1, SCALE2 = 5.0, 4.0, 1.2, 0.7
MASKPT = 0.7
VARIANTS = [("square", 1.0), ("square", 0.5), ("point", 1.0)]


def build_asn(dst):
    """Point the association at the saved post-tweakreg frames, absolute paths."""
    with open(SRC_ASN) as fh:
        asn = json.load(fh)
    for member in asn["products"][0]["members"]:
        tweaked = member["expname"].replace("_destreak.fits", "_destreak_tweakreg.fits")
        rel = tweaked if os.path.exists(os.path.join(BASE, tweaked)) else member["expname"]
        member["expname"] = os.path.join(BASE, rel)
    asn["products"][0]["name"] = asn["products"][0]["name"] + "_kernelscan"
    with open(dst, "w") as fh:
        json.dump(asn, fh, indent=2)
    return dst


def main():
    import jwst.datamodels as dm
    from jwst.datamodels import ModelLibrary, dqflags
    from jwst.outlier_detection.utils import median_with_resampling
    from jwst.resample import resample
    from stcal.outlier_detection.utils import flag_resampled_crs, gwcs_blot

    os.makedirs(OUTDIR, exist_ok=True)
    asn = build_asn(f"{OUTDIR}/asn_kernelscan.json")

    pre = f"{BASE}/{FRAME}_destreak_tweakreg.fits"
    if not os.path.exists(pre):
        pre = f"{BASE}/{FRAME}_destreak.fits"
    probe = dm.open(pre)
    sci = np.asarray(probe.data, dtype=float)
    err = np.asarray(probe.err, dtype=float)
    wcs_in = probe.meta.wcs

    ods = sorted(
        f for f in glob.glob(f"{DIST}/*outlierdetectionstep.fits") if FRAME in os.path.basename(f)
    )[0]
    dq = fits.getdata(ods, "DQ").astype(np.uint32)
    sat = (dq & dqflags.pixel["SATURATED"]) != 0
    real = ((dq & dqflags.pixel["OUTLIER"]) != 0) & ~sat
    print(f"[kern] production (square/1.0) flagged {int(real.sum())} px", flush=True)

    summary = {}
    for kernel, pixfrac in VARIANTS:
        lib = ModelLibrary(asn, on_disk=True)
        resamp = resample.ResampleImage(
            lib, blendheaders=False, weight_type="ivm", pixfrac=pixfrac, kernel=kernel,
            fillval="NAN", good_bits="~DO_NOT_USE", enable_ctx=False, enable_var=False,
            compute_err=None,
        )
        print(f"[kern] building median: kernel={kernel} pixfrac={pixfrac}", flush=True)
        median_data, median_wcs = median_with_resampling(lib, resamp, MASKPT)
        blot = gwcs_blot(median_data, median_wcs, sci.shape, wcs_in, fillval=np.nan)
        blot = np.asarray(blot, dtype=float)
        cov = np.isfinite(blot)
        mask = flag_resampled_crs(sci, err, np.nan_to_num(blot), SNR1, SNR2,
                                  SCALE1, SCALE2, 0.0) & cov & ~sat
        n = int(mask.sum())
        summary[f"{kernel}_{pixfrac}"] = n
        print(f"[kern] kernel={kernel} pixfrac={pixfrac}: {n} flagged px "
              f"({100 * n / max(1, real.sum()):.1f}% of production)", flush=True)
        for cx, cy in BRIGHT_STARS:
            sl = (slice(max(0, cy - BOX), cy + BOX), slice(max(0, cx - BOX), cx + BOX))
            npix = int((~sat[sl]).sum())
            print(f"       star ({cx},{cy}) box: {100 * mask[sl].sum() / npix:5.2f}% "
                  f"(production {100 * real[sl].sum() / npix:5.2f}%)", flush=True)
        fits.writeto(f"{OUTDIR}/blot_{kernel}_{pixfrac}_{FRAME}.fits",
                     blot.astype(np.float32), overwrite=True)
        del median_data, lib, resamp

    with open(f"{OUTDIR}/kernel_scan.json", "w") as fh:
        json.dump({"production_square_1.0": int(real.sum()), **summary}, fh, indent=2)
    print(f"[kern] wrote {OUTDIR}/kernel_scan.json")


if __name__ == "__main__":
    main()
