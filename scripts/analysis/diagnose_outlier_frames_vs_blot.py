"""Diagnose *why* ``outlier_detection`` flags bright-star PSF pixels (issue #161).

This is a DIAGNOSIS, not a tuning study.  A cosmic-ray rejection algorithm
should not flag PSF-edge pixels at all: those pixels should not vary from one
exposure to the next.  If ``outlier_detection`` finds them discrepant, either

  (A1) the frames genuinely disagree there   -> a real upstream defect, or
  (A2) the frames agree with each other but disagree with the *blot*
       -> the resample/blot round trip makes the comparison invalid for an
          undersampled PSF, and no threshold makes it valid.

The step's decision is ``|sci - blot| > scale*|d blot| + snr*err`` where ``blot``
is the median of the resampled exposures, inverse-drizzled back onto the
detector.  We split that difference into three physically distinct terms by
sampling the *same sky position* in the per-group resampled images the median
was built from (``*_outlier_i2d.fits``, saved by
``save_intermediate_results=True``)::

    sci - blot = (sci - v_self)      # t1: resample round-trip loss, SAME exposure
               + (v_self - med_oth)  # t2: genuine frame-to-frame disagreement
               + (med_oth - blot)    # t3: median/blot interpolation

``t2`` is the only term that means "the exposures disagree".  ``t1`` involves no
comparison between exposures at all -- it is what the drizzle+blot round trip
does to this one exposure's own data.

It also runs the follow-up discriminators from the issue:

* (B) per-exposure alignment residual: regress ``v_k - med`` against the median's
  intensity gradient in the output grid; the fitted coefficients are that
  exposure's shift in output pixels (a misalignment signature).  Also regress
  ``sci - blot`` on ``(dI/dx, dI/dy, laplacian(I))``: a shift gives gradient
  terms, smoothing gives a negative laplacian term.
* (C) sign symmetry about the PSF: a genuine CR population has no reason to be
  sign-symmetric; a smoothing (blot) residual is ``-sigma^2/2 * laplacian``, so
  it flips sign between spike ridges and the dark gaps between them.

Inputs are the intermediate products of an earlier ``OutlierDetectionStep`` run
with ``save_intermediate_results=True`` (see ``outlier_value_distributions.py``);
nothing under ``/orange`` is written.
"""

import glob
import os

import numpy as np
from astropy.io import fits

DIST = os.environ.get("OUTLIER_DIST_DIR", "/blue/adamginsburg/adamginsburg/tmp/outlier_dist")
BASE = os.environ.get("OUTLIER_BASE_DIR", "/orange/adamginsburg/jwst/brick/F200W/pipeline")
FRAME = os.environ.get("OUTLIER_FRAME", "jw01182004001_04101_00001_nrca1")
OUTDIR = os.environ.get("OUTLIER_DIAG_OUT", f"{DIST}/diagnosis")
# bright stars in the probe frame (cx, cy), from the ramp-zone figure
BRIGHT_STARS = [(1362, 84), (1586, 1643), (207, 1115)]
BOX = 130  # half-size of the per-star analysis box
# the parameters the intermediate products were produced with
SNR1, SNR2 = 5.0, 4.0
SCALE1, SCALE2 = 1.2, 0.7


def _abs_deriv(array):
    """Max abs difference to the 4 neighbours -- copy of ``stcal ... _abs_deriv``."""
    out = np.zeros_like(array)
    if np.issubdtype(array.dtype, np.floating):
        out[np.isnan(array)] = np.nan
    row_diff = np.abs(np.diff(array, axis=0))
    np.putmask(out[1:], np.isfinite(row_diff), row_diff)
    row_offset_view = out[:-1]
    np.putmask(row_offset_view, row_diff > row_offset_view, row_diff)
    del row_diff
    col_diff = np.abs(np.diff(array, axis=1))
    col_offset_view = out[:, 1:]
    np.putmask(col_offset_view, col_diff > col_offset_view, col_diff)
    col_offset_view = out[:, :-1]
    np.putmask(col_offset_view, col_diff > col_offset_view, col_diff)
    return out


def nmad(a, axis=0):
    """1.4826 * median absolute deviation, nan-aware."""
    med = np.nanmedian(a, axis=axis)
    return 1.4826 * np.nanmedian(np.abs(a - np.expand_dims(med, axis)), axis=axis)


def load_frame():
    """Return sci/err/dq/blot/wcs for the probe frame plus the step's own masks."""
    import jwst.datamodels as dm
    from jwst.datamodels import dqflags

    pre = f"{BASE}/{FRAME}_destreak_tweakreg.fits"
    if not os.path.exists(pre):
        pre = f"{BASE}/{FRAME}_destreak.fits"
    model = dm.open(pre)
    sci = np.asarray(model.data, dtype=float)
    err = np.asarray(model.err, dtype=float)
    wcs_in = model.meta.wcs

    blot_file = sorted(f for f in glob.glob(f"{DIST}/*blot.fits") if FRAME in os.path.basename(f))[0]
    blot = fits.getdata(blot_file, "SCI").astype(float)

    ods = sorted(
        f for f in glob.glob(f"{DIST}/*outlierdetectionstep.fits") if FRAME in os.path.basename(f)
    )[0]
    dq = fits.getdata(ods, "DQ").astype(np.uint32)
    flags = dqflags.pixel
    sat = (dq & flags["SATURATED"]) != 0
    outl = ((dq & flags["OUTLIER"]) != 0) & ~sat
    print(f"[diag] frame     {os.path.basename(pre)}")
    print(f"[diag] blot      {os.path.basename(blot_file)}")
    print(f"[diag] crf(DQ)   {os.path.basename(ods)}")
    print(f"[diag] OUTLIER & ~SATURATED: {int(outl.sum())} px")
    return sci, err, blot, sat, outl, wcs_in


def reproduce_decision(sci, err, blot):
    """Recompute the step's per-pixel decision quantities (sanity check)."""
    from scipy import ndimage

    blot_deriv = _abs_deriv(blot)
    diff = np.abs(sci - blot)
    errd = np.nan_to_num(err)
    thr1 = SCALE1 * blot_deriv + SNR1 * errd
    thr2 = SCALE2 * blot_deriv + SNR2 * errd
    m1 = diff > thr1
    m1s = ndimage.convolve(m1.astype(int), np.ones((3, 3), dtype=int), mode="nearest") > 0
    mask = m1s & (diff > thr2)
    return diff, blot_deriv, thr1, thr2, mask


def build_sample(sci, sat, outl):
    """Pixel index list: all flagged px, all px in the bright-star boxes, plus a
    random field control."""
    ny, nx = sci.shape
    take = outl.copy()
    for cx, cy in BRIGHT_STARS:
        take[max(0, cy - BOX):cy + BOX, max(0, cx - BOX):cx + BOX] = True
    rng = np.random.default_rng(161)
    field = np.zeros_like(take)
    fy = rng.integers(4, ny - 4, 60000)
    fx = rng.integers(4, nx - 4, 60000)
    field[fy, fx] = True
    take |= field
    take &= ~sat & np.isfinite(sci)
    ys, xs = np.where(take)
    print(f"[diag] sampled {len(ys)} pixels "
          f"({int((outl & take).sum())} flagged, {int((~outl & take).sum())} unflagged)")
    return ys, xs


def map_to_grid(wcs_in, wcs_out, ys, xs):
    """Detector (x, y) -> output-grid (gx, gy), nearest integer."""
    ra, dec = wcs_in(xs.astype(float), ys.astype(float))
    gx, gy = wcs_out.invert(ra, dec)
    return np.asarray(gx), np.asarray(gy)


def sample_stack(i2d_files, gx, gy, shape, maskpt=0.7):
    """Values (and weight validity) of every resampled group at the sample pixels."""
    from stcal.outlier_detection.utils import compute_weight_threshold

    n = len(gx)
    vals = np.full((len(i2d_files), n), np.nan, dtype=np.float32)
    ix = np.rint(gx).astype(int)
    iy = np.rint(gy).astype(int)
    inside = (ix >= 0) & (ix < shape[1]) & (iy >= 0) & (iy < shape[0])
    ixc = np.clip(ix, 0, shape[1] - 1)
    iyc = np.clip(iy, 0, shape[0] - 1)
    for k, f in enumerate(i2d_files):
        with fits.open(f, memmap=False) as hdul:
            sci = hdul["SCI"].data
            wht = hdul["WHT"].data
            thresh = compute_weight_threshold(wht, maskpt)
            v = sci[iyc, ixc].astype(np.float32)
            w = wht[iyc, ixc]
        bad = (~inside) | (w < thresh) | ~np.isfinite(v)
        v[bad] = np.nan
        vals[k] = v
        print(f"[diag]   {k + 1:2d}/{len(i2d_files)} {os.path.basename(f)[:36]} "
              f"wthresh={thresh:.3g}  n_valid={int(np.isfinite(v).sum())}", flush=True)
    return vals, inside


def summarize(name, sel, cols):
    """Print median statistics of a set of named per-pixel columns for a subset."""
    n = int(sel.sum())
    if n == 0:
        print(f"  {name:<28s} (empty)")
        return
    out = [f"  {name:<28s} N={n:7d}"]
    for key, arr in cols.items():
        v = arr[sel]
        v = v[np.isfinite(v)]
        med = np.median(v) if v.size else np.nan
        out.append(f"{key}={med:9.3f}")
    print("  ".join(out))


def main():  # noqa: PLR0915  (linear report script)
    os.makedirs(OUTDIR, exist_ok=True)
    import jwst.datamodels as dm

    sci, err, blot, sat, outl, wcs_in = load_frame()
    diff, blot_deriv, thr1, thr2, remask = reproduce_decision(sci, err, blot)
    agree = (remask == outl)[~sat]
    print(f"[diag] recomputed decision reproduces DQ OUTLIER on "
          f"{100 * agree.mean():.2f}% of non-saturated px "
          f"(recomputed {int(remask[~sat].sum())} vs DQ {int(outl.sum())})")

    med_file = sorted(glob.glob(f"{DIST}/*median.fits"))[0]
    medm = dm.open(med_file)
    med_img = np.asarray(medm.data, dtype=float)
    wcs_out = medm.meta.wcs
    print(f"[diag] median    {os.path.basename(med_file)}  shape={med_img.shape}")

    i2ds = sorted(glob.glob(f"{DIST}/*_outlier_i2d.fits"))
    exp_stem = "_".join(FRAME.split("_")[:2])  # jw01182004001_04101
    self_stem = f"{exp_stem}_{FRAME.split('_')[2]}"  # + exposure number
    self_idx = [k for k, f in enumerate(i2ds) if os.path.basename(f).startswith(self_stem)]
    print(f"[diag] {len(i2ds)} resampled groups; self group index {self_idx}")
    if len(self_idx) != 1:
        raise RuntimeError(f"could not identify the probe frame's own group: {self_idx}")
    self_idx = self_idx[0]

    ys, xs = build_sample(sci, sat, outl)
    gx, gy = map_to_grid(wcs_in, wcs_out, ys, xs)
    vals, inside = sample_stack(i2ds, gx, gy, med_img.shape)

    # ---- per-pixel terms -------------------------------------------------
    v_self = vals[self_idx].astype(float)
    others = np.delete(vals, self_idx, axis=0).astype(float)
    with np.errstate(invalid="ignore"):
        n_oth = np.isfinite(others).sum(axis=0)
        med_oth = np.nanmedian(others, axis=0)
        sig_oth = nmad(others, axis=0)
        std_oth = np.nanstd(others, axis=0)
    sci_s = sci[ys, xs]
    err_s = err[ys, xs]
    blot_s = blot[ys, xs]
    thr1_s = thr1[ys, xs]
    flag_s = outl[ys, xs]

    t1 = sci_s - v_self          # resample round-trip loss (same exposure)
    t2 = v_self - med_oth        # frame-to-frame disagreement
    t3 = med_oth - blot_s        # median -> blot interpolation
    total = sci_s - blot_s
    z_ee = np.divide(t2, sig_oth, out=np.full_like(t2, np.nan), where=sig_oth > 0)
    z_tot = np.divide(total, sig_oth, out=np.full_like(total, np.nan), where=sig_oth > 0)

    ok = np.isfinite(v_self) & (n_oth >= 5) & np.isfinite(med_oth) & inside
    print(f"\n[diag] {int(ok.sum())}/{len(ok)} sampled px have self + >=5 other exposures "
          f"(median n_others = {np.median(n_oth[ok]):.0f})")

    cols = {
        "sci": sci_s, "blot": blot_s, "err": err_s, "thr1": thr1_s,
        "|sci-blot|": np.abs(total), "|t1|": np.abs(t1), "|t2|": np.abs(t2),
        "|t3|": np.abs(t3), "sig_oth": sig_oth, "|z_ee|": np.abs(z_ee),
        "|z_tot|": np.abs(z_tot),
    }

    print("\n" + "=" * 108)
    print("A. EXPOSURE-vs-EXPOSURE  versus  EXPOSURE-vs-BLOT  (medians, MJy/sr)")
    print("   t1 = sci - self_resampled | t2 = self_resampled - median(others) | "
          "t3 = median(others) - blot")
    print("=" * 108)
    cx0, cy0 = BRIGHT_STARS[0]
    r0 = np.hypot(ys - cy0, xs - cx0)
    halo = (r0 < BOX) & ok
    summarize("ALL flagged (OUTLIER)", flag_s & ok, cols)
    summarize("flagged, bright-star halo", flag_s & halo, cols)
    summarize("flagged, elsewhere", flag_s & ok & ~halo, cols)
    summarize("UNflagged (control)", ~flag_s & ok, cols)
    summarize("UNflagged, bright-star halo", ~flag_s & halo, cols)

    fl = flag_s & ok
    print("\n  --- key ratios at flagged pixels ---")
    for label, sel in [("all flagged", fl), ("flagged in halo", flag_s & halo)]:
        if sel.sum() == 0:
            continue
        a = np.abs(t2[sel])
        b = np.abs(total[sel])
        c = np.abs(t1[sel])
        s = sig_oth[sel]
        would = np.abs(t2[sel]) > thr1_s[sel]
        print(f"  [{label}]  N={int(sel.sum())}")
        print(f"    median |sci-blot|                     = {np.nanmedian(b):8.3f}")
        print(f"    median |t2| (exposure-vs-exposure)    = {np.nanmedian(a):8.3f}"
              f"   -> {100 * np.nanmedian(a) / np.nanmedian(b):5.1f}% of |sci-blot|")
        print(f"    median |t1| (round-trip, same expo)   = {np.nanmedian(c):8.3f}"
              f"   -> {100 * np.nanmedian(c) / np.nanmedian(b):5.1f}% of |sci-blot|")
        print(f"    median |t3| (median->blot interp)     = {np.nanmedian(np.abs(t3[sel])):8.3f}")
        print(f"    median exposure-to-exposure scatter   = {np.nanmedian(s):8.3f}"
              f"   (1.4826*MAD of the other exposures)")
        print(f"    median |z_ee| = |t2|/scatter          = {np.nanmedian(np.abs(z_ee[sel])):8.2f}")
        print(f"    median |z_tot| = |sci-blot|/scatter   = {np.nanmedian(np.abs(z_tot[sel])):8.2f}")
        print(f"    would still flag on |t2| > threshold1 : {100 * np.nanmean(would):5.1f}%")
        print(f"    fraction with |z_ee| > 5              : "
              f"{100 * np.nanmean(np.abs(z_ee[sel]) > 5):5.1f}%")

    # ---- C. sign symmetry / smoothing signature --------------------------
    from scipy import ndimage

    lap = ndimage.laplace(blot)
    gxi = ndimage.sobel(blot, axis=1) / 8.0
    gyi = ndimage.sobel(blot, axis=0) / 8.0
    lap_s = lap[ys, xs]
    gx_s = gxi[ys, xs]
    gy_s = gyi[ys, xs]

    print("\n" + "=" * 108)
    print("C. SIGN SYMMETRY ABOUT THE PSF (smoothing signature: resid = -sigma^2/2 * laplacian)")
    print("=" * 108)
    pos = fl & (total > 0)
    neg = fl & (total < 0)
    print(f"  flagged positive (sci>blot): {int(pos.sum()):7d} "
          f"({100 * pos.sum() / max(1, fl.sum()):.1f}%)  median laplacian(blot) = "
          f"{np.nanmedian(lap_s[pos]):9.3f}")
    print(f"  flagged negative (sci<blot): {int(neg.sum()):7d} "
          f"({100 * neg.sum() / max(1, fl.sum()):.1f}%)  median laplacian(blot) = "
          f"{np.nanmedian(lap_s[neg]):9.3f}")
    good = fl & np.isfinite(lap_s) & np.isfinite(total)
    if good.sum() > 10:
        r = np.corrcoef(total[good], lap_s[good])[0, 1]
        print(f"  corr( sci-blot , laplacian(blot) ) at flagged px = {r:+.3f} "
              f"(negative => blot is a smoothed version of the truth)")
        r2 = np.corrcoef(t2[good & np.isfinite(t2)], lap_s[good & np.isfinite(t2)])[0, 1]
        print(f"  corr( t2       , laplacian(blot) ) at flagged px = {r2:+.3f}")

    # ---- B. shift vs smoothing regression --------------------------------
    print("\n" + "=" * 108)
    print("B. WHAT EXPLAINS sci-blot: a sub-pixel SHIFT (gradient terms) or SMOOTHING (laplacian)?")
    print("=" * 108)

    def fit(y, preds, names, sel):
        m = sel & np.isfinite(y)
        for p in preds:
            m &= np.isfinite(p)
        if m.sum() < 50:
            print("    (too few points)")
            return
        A = np.column_stack([p[m] for p in preds] + [np.ones(m.sum())])
        coef, *_ = np.linalg.lstsq(A, y[m], rcond=None)
        pred = A @ coef
        ss = 1 - np.sum((y[m] - pred) ** 2) / np.sum((y[m] - y[m].mean()) ** 2)
        txt = "  ".join(f"{nm}={c:+.4g}" for nm, c in zip(names + ["const"], coef, strict=True))
        print(f"    R^2={ss:6.3f}   {txt}")

    for label, sel in [("all flagged", fl), ("bright-star box (all px)", halo)]:
        print(f"  [{label}]  model: sci-blot ~ ...")
        print("    gradient only (shift):", end="")
        fit(total, [gx_s, gy_s], ["dI/dx", "dI/dy"], sel)
        print("    laplacian only (smoothing):", end="")
        fit(total, [lap_s], ["lap"], sel)
        print("    gradient + laplacian:", end="")
        fit(total, [gx_s, gy_s, lap_s], ["dI/dx", "dI/dy", "lap"], sel)

    # ---- B2. per-exposure alignment residual in the output grid ----------
    print("\n" + "=" * 108)
    print("B2. PER-EXPOSURE ALIGNMENT RESIDUAL (regress v_k - median on the median's gradient;")
    print("    coefficients are that exposure's apparent shift in OUTPUT pixels)")
    print("=" * 108)
    mgy, mgx = np.gradient(med_img)
    mgx_s = mgx[np.clip(np.rint(gy).astype(int), 0, med_img.shape[0] - 1),
                np.clip(np.rint(gx).astype(int), 0, med_img.shape[1] - 1)]
    mgy_s = mgy[np.clip(np.rint(gy).astype(int), 0, med_img.shape[0] - 1),
                np.clip(np.rint(gx).astype(int), 0, med_img.shape[1] - 1)]
    med_all = np.nanmedian(vals.astype(float), axis=0)
    steep = halo & np.isfinite(mgx_s) & np.isfinite(mgy_s)
    shifts = []
    for k, f in enumerate(i2ds):
        y = vals[k].astype(float) - med_all
        m = steep & np.isfinite(y)
        if m.sum() < 200:
            continue
        A = np.column_stack([mgx_s[m], mgy_s[m], np.ones(m.sum())])
        coef, *_ = np.linalg.lstsq(A, y[m], rcond=None)
        pred = A @ coef
        ss = 1 - np.sum((y[m] - pred) ** 2) / np.sum((y[m] - y[m].mean()) ** 2)
        shifts.append((os.path.basename(f)[:29], coef[0], coef[1], ss, int(m.sum())))
    for nm, dx, dy, ss, npx in shifts:
        print(f"    {nm}  dx={dx:+7.3f} px  dy={dy:+7.3f} px  R^2={ss:6.3f}  N={npx}")
    if shifts:
        dxs = np.array([s[1] for s in shifts])
        dys = np.array([s[2] for s in shifts])
        print(f"    -> rms shift over exposures: dx {dxs.std():.3f} px, dy {dys.std():.3f} px; "
              f"max |shift| = {np.hypot(dxs, dys).max():.3f} px")

    np.savez_compressed(
        f"{OUTDIR}/terms_{FRAME}.npz",
        ys=ys, xs=xs, sci=sci_s, blot=blot_s, err=err_s, thr1=thr1_s,
        v_self=v_self, med_oth=med_oth, sig_oth=sig_oth, std_oth=std_oth,
        n_oth=n_oth, flag=flag_s, t1=t1, t2=t2, t3=t3, lap=lap_s, gx=gx_s, gy=gy_s,
    )
    print(f"\n[diag] wrote {OUTDIR}/terms_{FRAME}.npz")


if __name__ == "__main__":
    main()
