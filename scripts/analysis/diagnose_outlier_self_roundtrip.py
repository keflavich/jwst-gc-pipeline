"""Self round-trip control for ``outlier_detection`` over-rejection (issue #161).

The decisive control for "do the frames disagree, or is the comparison invalid?"
is to run the *exact same flagging test* with the other exposures removed:

    blot_self = blot( drizzle( this exposure alone ) )

``blot_self`` contains no information from any other exposure, so it cannot
contain any frame-to-frame inconsistency, misalignment, persistence or
brighter-fatter difference.  It differs from ``sci`` only by the resample/blot
round trip.  Feeding it to ``stcal``'s ``flag_resampled_crs`` with the production
thresholds answers the question directly:

  * many pixels flagged  -> the round trip alone trips the test; the algorithm is
    invalid for an undersampled PSF and no threshold makes it valid,
  * few pixels flagged   -> the flags in the real run really do come from
    exposure-to-exposure differences, and those are the bug to chase.

Everything is derived from the frame itself plus its WCS; no production file is
written.
"""

import glob
import os

import numpy as np
from astropy.io import fits

DIST = os.environ.get("OUTLIER_DIST_DIR", "/blue/adamginsburg/adamginsburg/tmp/outlier_dist")
BASE = os.environ.get("OUTLIER_BASE_DIR", "/orange/adamginsburg/jwst/brick/F200W/pipeline")
FRAME = os.environ.get("OUTLIER_FRAME", "jw01182004001_04101_00001_nrca1")
OUTDIR = os.environ.get("OUTLIER_DIAG_OUT", f"{DIST}/diagnosis")
BRIGHT_STARS = [(1362, 84), (1586, 1643), (207, 1115)]
BOX = 110
SNR1, SNR2, SCALE1, SCALE2 = 5.0, 4.0, 1.2, 0.7


def self_roundtrip(sci, err, wcs_in, wcs_out, out_shape, kernel="square", pixfrac=1.0):
    """Drizzle this one exposure onto the outlier-detection grid, blot it back."""
    from drizzle.resample import Drizzle
    from stcal.outlier_detection.utils import gwcs_blot
    from stcal.resample.utils import calc_pixmap

    pixmap = calc_pixmap(wcs_in, wcs_out, sci.shape)
    good = np.isfinite(sci) & np.isfinite(err) & (err > 0)
    data = np.where(good, sci, 0).astype(np.float32)
    wht = np.where(good, 1.0 / np.maximum(err, 1e-8) ** 2, 0.0).astype(np.float32)
    driz = Drizzle(kernel=kernel, fillval=np.nan, out_shape=tuple(out_shape), disable_ctx=True)
    driz.add_image(data=data, exptime=1.0, pixmap=pixmap, weight_map=wht, pixfrac=pixfrac)
    resampled = driz.out_img
    blot = gwcs_blot(resampled, wcs_out, sci.shape, wcs_in, fillval=np.nan)
    return np.asarray(blot, dtype=float), resampled


def flag(sci, err, blot):
    from stcal.outlier_detection.utils import flag_resampled_crs

    return flag_resampled_crs(sci, err, blot, SNR1, SNR2, SCALE1, SCALE2, 0.0)


def main():
    import jwst.datamodels as dm
    from jwst.datamodels import dqflags

    os.makedirs(OUTDIR, exist_ok=True)
    pre = f"{BASE}/{FRAME}_destreak_tweakreg.fits"
    if not os.path.exists(pre):
        pre = f"{BASE}/{FRAME}_destreak.fits"
    model = dm.open(pre)
    sci = np.asarray(model.data, dtype=float)
    err = np.asarray(model.err, dtype=float)
    wcs_in = model.meta.wcs

    med_file = sorted(glob.glob(f"{DIST}/*median.fits"))[0]
    medm = dm.open(med_file)
    wcs_out = medm.meta.wcs
    out_shape = medm.data.shape

    blot_med = fits.getdata(
        sorted(f for f in glob.glob(f"{DIST}/*blot.fits") if FRAME in os.path.basename(f))[0],
        "SCI",
    ).astype(float)
    ods = sorted(
        f for f in glob.glob(f"{DIST}/*outlierdetectionstep.fits") if FRAME in os.path.basename(f)
    )[0]
    dq = fits.getdata(ods, "DQ").astype(np.uint32)
    sat = (dq & dqflags.pixel["SATURATED"]) != 0
    real = ((dq & dqflags.pixel["OUTLIER"]) != 0) & ~sat

    print(f"[self] frame {os.path.basename(pre)}; grid {out_shape}")
    print(f"[self] production run flagged {int(real.sum())} px (OUTLIER & ~SATURATED)")

    results = {}
    for kernel, pixfrac in [("square", 1.0), ("point", 1.0), ("turbo", 1.0)]:
        blot_self, _ = self_roundtrip(sci, err, wcs_in, wcs_out, out_shape,
                                      kernel=kernel, pixfrac=pixfrac)
        cov = np.isfinite(blot_self)
        mask = flag(sci, err, np.nan_to_num(blot_self, nan=0.0)) & cov & ~sat
        results[(kernel, pixfrac)] = (mask, blot_self)
        both = mask & real
        print(f"\n[self] kernel={kernel} pixfrac={pixfrac}: "
              f"SELF-round-trip flags {int(mask.sum())} px "
              f"({100 * mask.sum() / max(1, real.sum()):.1f}% of the production count)")
        print(f"       overlap with the production OUTLIER mask: {int(both.sum())} px "
              f"= {100 * both.sum() / max(1, real.sum()):.1f}% of production flags "
              f"are ALSO flagged with no other exposure involved")
        rms = np.nanstd((sci - blot_self)[cov & ~sat])
        print(f"       rms(sci - blot_self) = {rms:.3f} MJy/sr; "
              f"rms(sci - blot_median) = {np.nanstd((sci - blot_med)[~sat]):.3f}")
        for cx, cy in BRIGHT_STARS:
            sl = (slice(max(0, cy - BOX), cy + BOX), slice(max(0, cx - BOX), cx + BOX))
            n = (~sat[sl]).sum()
            print(f"       star ({cx},{cy}) box: production {100 * real[sl].sum() / n:5.2f}%  "
                  f"self-round-trip {100 * mask[sl].sum() / n:5.2f}%")

    mask_sq, blot_self_sq = results[("square", 1.0)]
    cov = np.isfinite(blot_self_sq)
    fl = real & cov
    d_self = np.abs(sci - blot_self_sq)
    d_med = np.abs(sci - blot_med)
    print("\n[self] at the pixels the production run flagged (median over "
          f"{int(fl.sum())} px):")
    print(f"       |sci - blot_self  | = {np.nanmedian(d_self[fl]):8.3f}  "
          "(round trip only, NO other exposure)")
    print(f"       |sci - blot_median| = {np.nanmedian(d_med[fl]):8.3f}  (what the step tested)")
    print(f"       ratio self/median   = {np.nanmedian(d_self[fl]) / np.nanmedian(d_med[fl]):8.3f}")

    fits.writeto(f"{OUTDIR}/blot_self_{FRAME}.fits", blot_self_sq.astype(np.float32),
                 overwrite=True)
    fits.writeto(f"{OUTDIR}/mask_self_{FRAME}.fits", mask_sq.astype(np.uint8), overwrite=True)
    print(f"[self] wrote {OUTDIR}/blot_self_{FRAME}.fits")


if __name__ == "__main__":
    main()
