"""Figure for the issue #161 diagnosis (reads the npz written by the two scripts)."""

import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DIST = os.environ.get("OUTLIER_DIST_DIR", "/blue/adamginsburg/adamginsburg/tmp/outlier_dist")
OUTDIR = os.environ.get("OUTLIER_DIAG_OUT", f"{DIST}/diagnosis")
FRAME = os.environ.get("OUTLIER_FRAME", "jw01182004001_04101_00001_nrca1")
OUT = os.environ.get("OUTLIER_DIAG_FIG", f"{OUTDIR}/outlier_161_diagnosis.png")


def main():
    t = np.load(f"{OUTDIR}/terms_{FRAME}.npz")
    fl = t["flag"].astype(bool)
    sig = t["sig_oth"]
    ok = np.isfinite(sig) & (sig > 0) & np.isfinite(t["t2"])
    z_ee = np.abs(t["t2"]) / sig
    total = np.abs(t["sci"] - t["blot"])

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    # (a) where does |sci - blot| come from?
    a = ax[0, 0]
    labels = ["|t1|\nround trip\n(same exposure)", "|t2|\nexposure vs\nother exposures",
              "|t3|\nmedian -> blot"]
    vals = [np.nanmedian(np.abs(t[k])[fl & ok]) for k in ("t1", "t2", "t3")]
    a.bar(labels, vals, color=["0.6", "tab:red", "0.6"])
    a.axhline(np.nanmedian(total[fl & ok]), color="k", ls="--",
              label=f"|sci-blot| = {np.nanmedian(total[fl & ok]):.2f}")
    for i, v in enumerate(vals):
        a.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    a.set_ylabel("median |term| at flagged pixels (MJy/sr)")
    a.set_title("(a) the blot round trip is NOT the driver:\nthe exposures really do differ")
    a.legend()

    # (b) how far out is the flagged exposure, in units of the OTHER exposures' scatter?
    b = ax[0, 1]
    bins = np.linspace(0, 8, 60)
    b.hist(z_ee[fl & ok], bins=bins, density=True, histtype="step", color="tab:red",
           label=f"flagged (median {np.nanmedian(z_ee[fl & ok]):.2f})")
    b.hist(z_ee[~fl & ok], bins=bins, density=True, histtype="step", color="0.4",
           label=f"unflagged (median {np.nanmedian(z_ee[~fl & ok]):.2f})")
    b.axvline(5, color="k", ls=":", label="5 sigma (CR-like)")
    b.set_xlabel("|this exposure - median(others)| / scatter(others)")
    b.set_ylabel("density")
    b.set_title("(b) flagged pixels are TYPICAL members of the\nexposure population, not outliers")
    b.legend(fontsize=8)

    # (c) the noise model vs the actual exposure-to-exposure scatter
    c = ax[1, 0]
    sel = fl & ok & np.isfinite(t["err"])
    c.hexbin(t["err"][sel], sig[sel], gridsize=60, bins="log", cmap="viridis",
             extent=(0, 0.6, 0, 6))
    xx = np.linspace(0, 0.6, 10)
    c.plot(xx, xx, "w--", label="scatter = err")
    c.plot(xx, 5 * xx, "r-", label="scatter = snr1*err (the threshold)")
    c.set_xlabel("ERR (the pipeline noise model), MJy/sr")
    c.set_ylabel("measured exposure-to-exposure scatter, MJy/sr")
    c.set_title("(c) at flagged pixels the real frame-to-frame scatter\n"
                "is far above what ERR predicts")
    c.legend(fontsize=8)

    # (d) within a dither point vs across dither points
    d = ax[1, 1]
    dpath = f"{OUTDIR}/dither_{FRAME}.npz"
    if os.path.exists(dpath):
        dd = np.load(dpath)
        fld = dd["flag"].astype(bool)
        m = fld & np.isfinite(dd["sig_within"]) & np.isfinite(dd["sig_across"])
        bins = np.logspace(-2, 1.2, 60)
        d.hist(dd["sig_within"][m], bins=bins, histtype="step", color="tab:blue",
               label=f"within one dither point (median {np.nanmedian(dd['sig_within'][m]):.2f})")
        d.hist(dd["sig_across"][m], bins=bins, histtype="step", color="tab:red",
               label=f"across dither points (median {np.nanmedian(dd['sig_across'][m]):.2f})")
        d.hist(dd["err"][m], bins=bins, histtype="step", color="0.4",
               label=f"ERR (median {np.nanmedian(dd['err'][m]):.2f})")
        d.set_xscale("log")
        d.set_xlabel("scatter of the resampled values (MJy/sr)")
        d.set_ylabel("flagged pixels")
        d.set_title("(d) the disagreement follows the DITHER POINT,\n"
                    "so it is resampling, not a per-frame defect")
        d.legend(fontsize=8)
    else:
        d.text(0.5, 0.5, "run diagnose_outlier_dither_dependence.py", ha="center")

    fig.suptitle(f"outlier_detection over-rejection diagnosis - {FRAME} (brick o004 F200W)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT, dpi=110)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
