"""Is the frame-to-frame disagreement a per-frame defect or a dither effect? (#161)

``diagnose_outlier_frames_vs_blot.py`` shows that at the flagged pixels the
exposures really do disagree with each other (the blot round trip contributes
<10% of ``sci - blot``).  That leaves two very different explanations:

  * a *random per-frame* defect (cosmic rays, persistence, brighter-fatter,
    ramp nonlinearity) -- these are independent from exposure to exposure, so
    exposures taken at the SAME dither point would disagree just as much as
    exposures at different dither points;
  * a *deterministic function of the dither point* -- an undersampled PSF
    resampled from a different sub-pixel phase, or a per-pointing registration
    error.  Then exposures sharing a dither point agree with each other and only
    disagree across dither points.

This script measures exactly that: the scatter of the resampled values WITHIN a
dither group versus ACROSS dither groups, at the same sky pixels.  It also
decomposes each exposure's difference from the median into a shift term
(gradient of the median) and a resolution term (laplacian of the median).
"""

import glob
import os

import numpy as np
from astropy.io import fits

from diagnose_outlier_frames_vs_blot import (  # noqa: E402
    BASE,
    BOX,
    BRIGHT_STARS,
    DIST,
    FRAME,
    OUTDIR,
    build_sample,
    map_to_grid,
    nmad,
    sample_stack,
)


POINTING_TOL = float(os.environ.get("OUTLIER_POINTING_TOL", "0.15"))  # arcsec


def dither_key(i2d_file):
    """(pattern point, x/y dither offset) identifying the pointing of a group."""
    hdr = fits.getheader(i2d_file, 0)
    return (hdr.get("PATT_NUM"), float(hdr.get("XOFFSET", np.nan)),
            float(hdr.get("YOFFSET", np.nan)), hdr.get("VISIT_ID", hdr.get("OBSERVTN")))


def cluster_pointings(keys, tol=POINTING_TOL):
    """Greedy single-link clustering of exposures by commanded dither offset.

    The brick dither pattern puts two exposures within a fraction of a pixel of
    each other at each mosaic tile, so exact offsets never repeat; clustering
    within ``tol`` arcsec recovers the "same pointing" pairs.
    """
    centers = []
    labels = []
    for _patt, xoff, yoff, _vis in keys:
        for j, (cxo, cyo) in enumerate(centers):
            if np.hypot(xoff - cxo, yoff - cyo) <= tol:
                labels.append(j)
                break
        else:
            centers.append((xoff, yoff))
            labels.append(len(centers) - 1)
    return labels, centers


def main():  # noqa: PLR0915
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

    ods = sorted(
        f for f in glob.glob(f"{DIST}/*outlierdetectionstep.fits") if FRAME in os.path.basename(f)
    )[0]
    dq = fits.getdata(ods, "DQ").astype(np.uint32)
    sat = (dq & dqflags.pixel["SATURATED"]) != 0
    outl = ((dq & dqflags.pixel["OUTLIER"]) != 0) & ~sat

    med_file = sorted(glob.glob(f"{DIST}/*median.fits"))[0]
    medm = dm.open(med_file)
    med_img = np.asarray(medm.data, dtype=float)
    wcs_out = medm.meta.wcs

    i2ds = sorted(glob.glob(f"{DIST}/*_outlier_i2d.fits"))
    keys = [dither_key(f) for f in i2ds]
    labels, centers = cluster_pointings(keys)
    groups = {}
    for k, lab in enumerate(labels):
        groups.setdefault(lab, []).append(k)
    print(f"[dith] {len(i2ds)} exposures in {len(groups)} pointing clusters "
          f"(tolerance {POINTING_TOL}\"):")
    for lab, members in groups.items():
        print(f"       cluster {lab}: XOFF={centers[lab][0]:+9.4f} YOFF={centers[lab][1]:+9.4f}  "
              f"{[os.path.basename(i2ds[m])[2:26] for m in members]}")

    ys, xs = build_sample(sci, sat, outl)
    gx, gy = map_to_grid(wcs_in, wcs_out, ys, xs)
    vals, inside = sample_stack(i2ds, gx, gy, med_img.shape)
    vals = vals.astype(float)

    flag_s = outl[ys, xs]
    err_s = err[ys, xs]
    cx0, cy0 = BRIGHT_STARS[0]
    halo = np.hypot(ys - cy0, xs - cx0) < BOX
    n_ok = np.isfinite(vals).sum(axis=0)
    ok = (n_ok >= 12) & inside

    # ---- within- vs across-dither scatter --------------------------------
    within = []
    for _key, members in groups.items():
        if len(members) < 2:
            continue
        sub = vals[members]
        good = np.isfinite(sub).sum(axis=0) >= 2
        s = nmad(sub, axis=0)
        s[~good] = np.nan
        within.append(s)
    if not within:
        raise SystemExit("[dith] no pointing cluster has >=2 exposures; "
                         "raise OUTLIER_POINTING_TOL")
    within = np.array(within)
    sig_within = np.nanmedian(within, axis=0)          # typical scatter at one pointing
    group_medians = np.array([np.nanmedian(vals[m], axis=0) for m in groups.values()])
    sig_across = nmad(group_medians, axis=0)           # scatter of the pointing means
    sig_all = nmad(vals, axis=0)

    print("\n" + "=" * 100)
    print("WITHIN a dither pointing vs ACROSS dither pointings (1.4826*MAD, MJy/sr)")
    print("=" * 100)
    for label, sel in [("flagged (OUTLIER)", flag_s & ok),
                       ("flagged, bright-star halo", flag_s & halo & ok),
                       ("unflagged control", ~flag_s & ok),
                       ("unflagged, bright-star halo", ~flag_s & halo & ok)]:
        n = int(sel.sum())
        if n == 0:
            continue
        w = np.nanmedian(sig_within[sel])
        a = np.nanmedian(sig_across[sel])
        t = np.nanmedian(sig_all[sel])
        e = np.nanmedian(err_s[sel])
        print(f"  {label:<28s} N={n:7d}  sig_within={w:7.3f}  sig_across={a:7.3f}  "
              f"sig_all={t:7.3f}  err={e:6.3f}   across/within={a / max(w, 1e-9):5.2f}  "
              f"sig_all/err={t / max(e, 1e-9):5.1f}")

    # ---- per-exposure: shift term vs resolution term ---------------------
    mgy, mgx = np.gradient(med_img)
    from scipy import ndimage

    mlap = ndimage.laplace(med_img)
    iy = np.clip(np.rint(gy).astype(int), 0, med_img.shape[0] - 1)
    ix = np.clip(np.rint(gx).astype(int), 0, med_img.shape[1] - 1)
    mgx_s, mgy_s, mlap_s = mgx[iy, ix], mgy[iy, ix], mlap[iy, ix]
    med_all = np.nanmedian(vals, axis=0)

    def r2(y, preds, m):
        A = np.column_stack([p[m] for p in preds] + [np.ones(int(m.sum()))])
        coef, *_ = np.linalg.lstsq(A, y[m], rcond=None)
        resid = y[m] - A @ coef
        return 1 - np.sum(resid**2) / np.sum((y[m] - y[m].mean()) ** 2), coef

    print("\n" + "=" * 100)
    print("PER-EXPOSURE decomposition of (v_k - median) in the bright-star box:")
    print("  shift model  = a*dI/dx + b*dI/dy      (a,b = apparent shift, output px)")
    print("  resol model  = c*laplacian(I)         (c<0 = this exposure is SHARPER than the median)")
    print("=" * 100)
    steep = halo & ok & np.isfinite(mgx_s) & np.isfinite(mlap_s)
    rows = []
    for k, f in enumerate(i2ds):
        y = vals[k] - med_all
        m = steep & np.isfinite(y)
        if m.sum() < 200:
            continue
        r_sh, c_sh = r2(y, [mgx_s, mgy_s], m)
        r_lp, c_lp = r2(y, [mlap_s], m)
        r_bo, c_bo = r2(y, [mgx_s, mgy_s, mlap_s], m)
        rows.append((os.path.basename(f)[:26], keys[k], c_sh[0], c_sh[1], r_sh,
                     c_lp[0], r_lp, r_bo))
        print(f"  {os.path.basename(f)[13:26]}  patt={labels[k]} obs={keys[k][3]}  "
              f"shift=({c_sh[0]:+6.3f},{c_sh[1]:+6.3f})px R2={r_sh:5.3f} | "
              f"lap c={c_lp[0]:+7.4f} R2={r_lp:5.3f} | both R2={r_bo:5.3f}")
    if rows:
        dxs = np.array([r[2] for r in rows])
        dys = np.array([r[3] for r in rows])
        r_sh = np.array([r[4] for r in rows])
        r_lp = np.array([r[6] for r in rows])
        r_bo = np.array([r[7] for r in rows])
        print(f"\n  shift rms over exposures: dx={dxs.std():.3f} px dy={dys.std():.3f} px "
              f"(max |shift| {np.hypot(dxs, dys).max():.3f} px)")
        print(f"  median R^2: shift-only {np.median(r_sh):.3f}, laplacian-only "
              f"{np.median(r_lp):.3f}, both {np.median(r_bo):.3f}  "
              f"-> unexplained by either: {1 - np.median(r_bo):.3f}")

    np.savez_compressed(
        f"{OUTDIR}/dither_{FRAME}.npz", ys=ys, xs=xs, flag=flag_s, err=err_s,
        sig_within=sig_within, sig_across=sig_across, sig_all=sig_all, n_ok=n_ok,
        vals=vals.astype(np.float32),
    )
    print(f"\n[dith] wrote {OUTDIR}/dither_{FRAME}.npz")


if __name__ == "__main__":
    main()
