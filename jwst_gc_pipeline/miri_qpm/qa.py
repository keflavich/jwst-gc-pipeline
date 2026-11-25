"""QA: pane-vs-reference overlap residuals per pane and brightness bin with the control null, geometry check from
pane stars, pixel statistics, quicklook PNG.

Overlap estimator 'pre_rl' (as the calibration study): 8 x 8-block means of the calibrated, NOT de-smeared pane Q against
MK = the reference sampled at the pane's sky positions, convolved with the pane kernel, plus the exposure's sky
offset (both at the exposure's cal level, where the R1 law and its 30-450 MJy/sr range are defined); median of
Q_b / MK_b - 1 (and of Q_b - MK_b in MJy/sr) per pane and MK bin.  No star mask (the study's clean cut also dropped
star and gradient pixels), so bright-star blocks enter the median.  Control: the identical estimator on imager
pixels of the same exposure at the pane footprint + 591 columns (cal vs the unconvolved reference + offset, the
pane's fluence cut and static mask): what the comparison gives where the answer is known.  'post_rl': the same
block estimator on the final pane pixels (de-smeared, trimmed, + the level that was subtracted) against the
reference without the kernel (+ the sky offset); it tests RL and the geometry end to end.
"""
from __future__ import annotations

import numpy as np

from .geometry import region_shift
from .offsets import ANCHOR_MK, bin_bias
from .register import fit_regions


def _cat(res, k):
    return np.concatenate([r[k] for r in res]) if res else np.zeros(0)


def range_bias(Vb, Mb, gb, p, n_min=8):
    """median(V_b / M_b - 1) over the blocks of pane p with M in 30-450 MJy/sr; NaN with fewer than n_min blocks."""
    s = (gb == p) & (Mb >= ANCHOR_MK[0]) & (Mb < ANCHOR_MK[1])
    return float(np.median(Vb[s] / Mb[s] - 1)) if s.sum() >= n_min else np.nan


def overlap_summary(res, corr, post=None):
    """res: per-exposure analysis results (Qb, MKb, gb, Cb, Mb, cgb); corr: per-exposure dict(gain, offset);
    post: per-exposure (Pb, Rb, pgb) of the FINAL de-smeared, trimmed pane pixels (+ the subtracted level) against
    the unconvolved reference (+ the sky offset): the end-to-end check of what goes into the mosaic."""
    Qb, MKb, gb = _cat(res, "Qb"), _cat(res, "MKb"), _cat(res, "gb").astype(int)
    Cb, Mb, cgb = _cat(res, "Cb"), _cat(res, "Mb"), _cat(res, "cgb").astype(int)
    eb = np.concatenate([np.full(len(r["Qb"]), j) for j, r in enumerate(res)]) if res else np.zeros(0, int)
    out = dict(pre_rl=bin_bias(Qb, MKb, gb), control=bin_bias(Cb, Mb, cgb))
    if post:
        Pb, Rb, pgb = (np.concatenate([q[i] for q in post]) for i in range(3))
        out["post_rl"] = bin_bias(Pb, Rb, pgb.astype(int))
        s = (Rb >= ANCHOR_MK[0]) & (Rb < ANCHOR_MK[1])
        out["post_rl_30_450"] = {p: dict(n_blocks=int((s & (pgb == p)).sum()),
                                         bias=float(np.median(Pb[s & (pgb == p)] / Rb[s & (pgb == p)] - 1))
                                         if (s & (pgb == p)).sum() >= 20 else None) for p in (1, 2, 3)}
    rng = (MKb >= ANCHOR_MK[0]) & (MKb < ANCHOR_MK[1])
    crng = (Mb >= ANCHOR_MK[0]) & (Mb < ANCHOR_MK[1])
    tot = {}
    for p in (1, 2, 3):
        s, c = rng & (gb == p), crng & (cgb == p)
        per_exp = [b for b in (range_bias(r["Qb"], r["MKb"], r["gb"], p) for r in res) if np.isfinite(b)]
        tot[p] = dict(n_blocks=int(s.sum()),
                      bias=float(np.median(Qb[s] / MKb[s] - 1)) if s.sum() >= 20 else None,
                      control=float(np.median(Cb[c] / Mb[c] - 1)) if c.sum() >= 20 else None,
                      n_exposures=len(per_exp),
                      per_exposure_p16_p50_p84=np.percentile(per_exp, [16, 50, 84]).tolist() if per_exp else None)
    out["range_30_450"] = tot
    if any(c["offset"] or any(v != 1.0 for v in c["gain"].values()) for c in corr):
        Qa = Qb.copy()
        for j, c in enumerate(corr):
            for p in (1, 2, 3):
                s = (eb == j) & (gb == p)
                Qa[s] = Qa[s] / c["gain"][p] - c["offset"].get(p, 0.0)
        out["after_local_terms_not_blind"] = bin_bias(Qa, MKb, gb)
    return out


def geometry_check(res, models, region, x0, y0):
    """Dataset pane-star offsets vs the static model at each pane's centre (reported, never applied)."""
    rows = [(j, *c) for j, r in enumerate(res) for c in r["cands"]]
    if not rows:
        return {g: dict(n=0) for g in (1, 2, 3)}
    fitted, stats = fit_regions(np.array(rows, float))
    out = {}
    for g in (1, 2, 3):
        if g not in stats:
            out[g] = dict(n=0)
            continue
        ys, xs = np.nonzero(region == g)
        cx, cy = np.array([xs.mean() + x0]), np.array([ys.mean() + y0])
        w = np.ravel(region_shift(fitted, g, cx, cy))
        m = np.ravel(region_shift(models, g, cx, cy))
        out[g] = dict(stats[g], dataset_minus_static_at_centre_px=[float(w[0] - m[0]), float(w[1] - m[1])])
    return out


def pixel_stats(mm, step=1):
    src = np.asarray(mm["SOURCE"][::step, ::step])
    flg = np.asarray(mm["FLAGS"][::step, ::step])
    n = {k: int((src == v).sum()) for k, v in (("none", 0), ("science", 1), ("pane_only", 2), ("pane_in_science", 3),
                                                ("hole_unfilled_above_450", 4))}
    pane = (src == 2) | (src == 3)
    tot = max(int((src > 0).sum()), 1)
    out = dict(counts=n, step=step,
               fraction_of_data={k: n[k] / tot for k in ("science", "pane_only", "pane_in_science",
                                                         "hole_unfilled_above_450")})
    if pane.any():
        out["pane_flag_fractions"] = {name: float(np.mean((flg[pane] & bit) > 0))
                                      for name, bit in (("above_450", 1), ("edge", 2), ("local_terms", 4),
                                                        ("below_30", 8), ("warn", 16), ("no_sky_offset", 32),
                                                        ("saturated", 64), ("fluence_40k", 128))}
    return out


def quicklook(mm, png, title="", max_px=2500):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ny, nx = mm["SCI"].shape
    s = max(1, int(np.ceil(max(ny, nx) / max_px)))
    sci = np.asarray(mm["SCI"][::s, ::s], dtype=np.float32)
    src = np.asarray(mm["SOURCE"][::s, ::s])
    fin = np.isfinite(sci)
    if not fin.any():
        return
    lo, hi = np.percentile(sci[fin], [1, 99.7])
    img = np.arcsinh((np.clip(sci, lo, hi) - lo) / max((hi - lo) / 30, 1e-6))
    fig, ax = plt.subplots(1, 2, figsize=(16, 8 * ny / nx + 1), dpi=110)
    ax[0].imshow(img, origin="lower", cmap="gray", interpolation="nearest")
    ax[0].contour((src == 2) | (src == 3), levels=[0.5], colors="tab:orange", linewidths=0.6)
    ax[0].set_title(f"{title} SCI (asinh, 1-99.7 %); orange = pane data")
    cmap = matplotlib.colors.ListedColormap(["black", "0.6", "tab:orange", "tab:red", "tab:blue"])
    ax[1].imshow(src, origin="lower", cmap=cmap, vmin=-0.5, vmax=4.5, interpolation="nearest")
    ax[1].set_title("SOURCE: 1 science (grey), 2 pane only (orange), 3 pane in hole (red), 4 hole left NaN (blue)")
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(png)
    plt.close(fig)


def science_identity(mm, ref_img, x0, y0, tile=4096):
    """Check that every finite reference pixel is copied bit-for-bit (SOURCE 1) and nothing else is SOURCE 1."""
    ny, nx = mm["SCI"].shape
    rny, rnx = ref_img.shape
    n_ref = n_bad = n_extra = 0
    for ty in range(0, ny, tile):
        for tx in range(0, nx, tile):
            osl = (slice(ty, min(ty + tile, ny)), slice(tx, min(tx + tile, nx)))
            sci = np.asarray(mm["SCI"][osl], dtype=np.float32)
            src = np.asarray(mm["SOURCE"][osl])
            r = np.full(sci.shape, np.nan, np.float32)
            ry0, ry1 = max(ty + y0, 0), min(min(ty + tile, ny) + y0, rny)
            rx0, rx1 = max(tx + x0, 0), min(min(tx + tile, nx) + x0, rnx)
            if ry1 > ry0 and rx1 > rx0:
                r[ry0 - y0 - ty:ry1 - y0 - ty, rx0 - x0 - tx:rx1 - x0 - tx] = ref_img[ry0:ry1, rx0:rx1]
            fin = np.isfinite(r)
            n_ref += int(fin.sum())
            n_bad += int((fin & ((src != 1) | (sci.view(np.uint32) != r.view(np.uint32)))).sum())
            n_extra += int((~fin & (src == 1)).sum())
    return dict(n_reference_finite=n_ref, n_not_identical=n_bad, n_source1_without_reference=n_extra,
                bit_identical=(n_bad == 0 and n_extra == 0))
