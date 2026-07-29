"""Are the flagged pixels a cosmic-ray population? Multi-frame coincidence (#161).

A cosmic ray hits ONE exposure at one place on the sky: at that sky position the
other exposures are clean.  A PSF/sampling artifact is a property of the sky, so
every exposure that looks at that sky position is liable to be flagged there.

This maps the ``OUTLIER`` mask of all 96 ``_crf``-equivalent frames onto the
common outlier-detection output grid and asks, for each pixel the probe frame
flagged: of the N other exposures that cover this sky position, how many flagged
it too?

  * ratio ~ 1/N  -> single-exposure events, consistent with cosmic rays,
  * ratio >> 1/N -> the same sky positions are rejected over and over, which no
    cosmic-ray population can do.
"""

import glob
import os

import numpy as np
from astropy.io import fits

BASE = os.environ.get("OUTLIER_BASE_DIR", "/orange/adamginsburg/jwst/brick/F200W/pipeline")
DIST = os.environ.get("OUTLIER_DIST_DIR", "/blue/adamginsburg/adamginsburg/tmp/outlier_dist")
OUTDIR = os.environ.get("OUTLIER_DIAG_OUT", f"{DIST}/diagnosis")
FRAME = os.environ.get("OUTLIER_FRAME", "jw01182004001_04101_00001_nrca1")
BRIGHT_STARS = [(1362, 84), (1586, 1643), (207, 1115)]
BOX = 130


def frame_stem(path):
    return os.path.basename(path).split("_destreak")[0]


def main():  # noqa: PLR0915
    import jwst.datamodels as dm
    from jwst.datamodels import dqflags
    from stcal.resample.utils import calc_pixmap

    os.makedirs(OUTDIR, exist_ok=True)
    med_file = sorted(glob.glob(f"{DIST}/*median.fits"))[0]
    medm = dm.open(med_file)
    wcs_out = medm.meta.wcs
    shape = medm.data.shape
    nglob = shape[0] * shape[1]
    print(f"[coin] output grid {shape}")

    ods_files = sorted(glob.glob(f"{DIST}/*outlierdetectionstep.fits"))
    print(f"[coin] {len(ods_files)} frames")

    ncov = np.zeros(nglob, dtype=np.uint8)
    nflag = np.zeros(nglob, dtype=np.uint8)
    probe_flat = None
    probe_flag = None
    probe_yx = None

    for i, ods in enumerate(ods_files):
        stem = frame_stem(ods)
        pre = f"{BASE}/{stem}_destreak_tweakreg.fits"
        if not os.path.exists(pre):
            pre = f"{BASE}/{stem}_destreak.fits"
        model = dm.open(pre)
        dq = fits.getdata(ods, "DQ").astype(np.uint32)
        sat = (dq & dqflags.pixel["SATURATED"]) != 0
        outl = ((dq & dqflags.pixel["OUTLIER"]) != 0) & ~sat

        pixmap = calc_pixmap(model.meta.wcs, wcs_out, dq.shape, stepsize=4)
        gx = pixmap[..., 0]
        gy = pixmap[..., 1]
        ix = np.rint(gx)
        iy = np.rint(gy)
        good = np.isfinite(ix) & np.isfinite(iy)
        ixi = np.where(good, ix, 0).astype(np.int64)
        iyi = np.where(good, iy, 0).astype(np.int64)
        good &= (ixi >= 0) & (ixi < shape[1]) & (iyi >= 0) & (iyi < shape[0])
        flat = (iyi * shape[1] + ixi)

        cov = np.zeros(nglob, dtype=bool)
        cov[flat[good]] = True
        ncov += cov
        fl = np.zeros(nglob, dtype=bool)
        fl[flat[good & outl]] = True
        nflag += fl
        if stem == FRAME:
            probe_flat = flat
            probe_flag = outl
            probe_yx = good
        model.close()
        if i % 8 == 0 or i == len(ods_files) - 1:
            print(f"[coin] {i + 1}/{len(ods_files)} {stem}  flagged {int(outl.sum())}", flush=True)

    if probe_flat is None:
        raise RuntimeError(f"probe frame {FRAME} not among the frames")

    ny, nx = probe_flag.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    sel = probe_flag & probe_yx
    f_at = nflag[probe_flat[sel]].astype(float)
    c_at = ncov[probe_flat[sel]].astype(float)
    ratio = (f_at - 1) / np.maximum(c_at - 1, 1)     # exclude the probe frame itself

    print("\n" + "=" * 96)
    print("MULTI-FRAME COINCIDENCE at the pixels the probe frame flagged")
    print("=" * 96)
    print(f"  N flagged px                          : {int(sel.sum())}")
    print(f"  median exposures covering the position: {np.median(c_at):.0f}")
    print(f"  median exposures flagging it          : {np.median(f_at):.0f}")
    print(f"  median fraction of the OTHER covering exposures that also flag it: "
          f"{np.median(ratio):.3f}")
    print(f"  mean   fraction                       : {np.mean(ratio):.3f}")
    print(f"  expectation for single-exposure events (cosmic rays): "
          f"~{1 / max(np.median(c_at) - 1, 1):.3f}")
    for thr in (0.1, 0.25, 0.5):
        print(f"  fraction of flagged px where >{thr:.0%} of the other covering exposures "
              f"flag the same sky position: {100 * np.mean(ratio > thr):5.1f}%")

    for cx, cy in BRIGHT_STARS:
        m = sel & (np.hypot(yy - cy, xx - cx) < BOX)
        if m.sum() < 10:
            continue
        f2 = nflag[probe_flat[m]].astype(float)
        c2 = ncov[probe_flat[m]].astype(float)
        r2 = (f2 - 1) / np.maximum(c2 - 1, 1)
        print(f"  star ({cx},{cy}) box: N={int(m.sum())}  median coincidence fraction "
              f"{np.median(r2):.3f}  (median coverage {np.median(c2):.0f} exposures)")

    # control: unflagged pixels of the probe frame
    rng = np.random.default_rng(161)
    unf = np.where((~probe_flag & probe_yx).ravel())[0]
    unf = rng.choice(unf, size=min(200000, unf.size), replace=False)
    fu = nflag[probe_flat.ravel()[unf]].astype(float)
    cu = ncov[probe_flat.ravel()[unf]].astype(float)
    ru = fu / np.maximum(cu, 1)
    print(f"  control (unflagged probe px, N={unf.size}): median fraction of covering "
          f"exposures flagging that sky position = {np.median(ru):.3f}, mean {np.mean(ru):.3f}")

    np.savez_compressed(f"{OUTDIR}/coincidence_{FRAME}.npz",
                        ratio=ratio.astype(np.float32), ncov=c_at.astype(np.uint8),
                        nflag=f_at.astype(np.uint8))
    print(f"\n[coin] wrote {OUTDIR}/coincidence_{FRAME}.npz")


if __name__ == "__main__":
    main()
