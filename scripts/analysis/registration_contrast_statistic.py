#!/usr/bin/env python
"""Why the registration seam check's confidence number depends on star density.

Background, in full, because the issue this supports (#170) was written in
jargon and could not be reviewed.

Before a field's mosaics are released, `scripts/release/registration_failsafes.py`
checks that each mosaic is correctly registered -- that a star's position in the
mosaic is where it should be.  A field-average check is not enough: the failure
it exists to catch (brick 1182 F356W, 2026-07) was several arcseconds of
misregistration confined to the narrow strip where the two NIRCam module
footprints overlap, with a whole-field average of about zero.  So the check is
spatially resolved: it lays a 20x20 grid over the mosaic and asks the question
separately in each cell.

Within one cell it works like this.  Take every bright source detected on the
mosaic, and a "truth set" of positions that must agree with it -- the same
field's per-module mosaics, or another filter, or the mosaic's own vetted
catalog.  Pair each detection with every truth position within 2.5 arcsec, and
histogram the pair separations into 40 mas bins.  If the mosaic is registered,
the true pairs all pile up at zero separation and everything else -- pairs of a
star with some unrelated neighbour -- spreads thinly over the whole search disk.
So the histogram is one sharp peak on a low floor, and where that peak sits is
the cell's measured misregistration.

The cell is then judged on two numbers:

    off    -- where the peak is, in mas.  Over OFF_MAX = 60 it is a candidate
              misregistration.
    ratio  -- H.max() / median(H[H > 0]), the peak bin's count over the median
              of the occupied bins.  Intended as "how confident are we that this
              peak is a real pile-up rather than a fluctuation".

A cell FAILS -- and one failing cell fails the field, blocking the release --
only when `off > OFF_MAX` AND `ratio >= FAIL_MIN_RATIO`, which is 10 for the
own-catalog check and 5 for the other two.

## The problem this script demonstrates

`ratio` is not a contrast ratio, and it is not a measure of confidence.

The search disk holds about 12,270 bins of 40x40 mas inside a 2.5 arcsec
radius, and a grid cell holds a few hundred pairs.  With far more bins than
pairs, essentially every occupied bin outside the peak holds exactly one pair --
so `median(H[H > 0])` is exactly 1, and `ratio` reduces to the raw number of
pairs in the peak bin.

A raw count scales with how many stars the cell contains.  The same physical
misregistration therefore produces a small `ratio` in a sparse cell and a large
one in a crowded cell, and `FAIL_MIN_RATIO` is in practice a cut on star density
rather than on confidence.  It is hardest to clear in the sparsest cells and
easiest in the most crowded -- and the crowded ones are the Galactic Centre
fields' interiors, where a seam matters most and where wrong-pair confusion is
also worst.  The two populations the threshold is supposed to separate are not
separated by construction; that they were separated on brick F405N is an
empirical fact about that band, not a property of the statistic.

Both panels below are computed by running the real estimator -- the same bins,
the same `H.max() / median(H[H>0])` -- over synthetic pair populations, so what
is shown is a property of the statistic and not of any one field.  The two
measured populations from the issue are overplotted for scale.

Usage:
    python scripts/analysis/registration_contrast_statistic.py \
        [--out docs/reports/figures/registration_contrast_statistic.png]
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The estimator's own geometry, copied from registration_failsafes.py so this
# script measures what the gate measures.
MX_ARCSEC = 2.5          # pair-separation search radius
XBIN_ARCSEC = 0.04       # offset-histogram bin
MIN_PAIRS = 80           # pairs needed in a cell before a peak is attempted
MIN_PEAK_RATIO = 5.0     # below this the cell is UNVERIFIED, not failed
FAIL_MIN_RATIO = 10.0    # the own-catalog fail bar

#: The seven brick F405N cells that were a FALSE own-catalog failure in 2026-07:
#: each read an 80 mas offset at ratio 5-8, while the independent same-star check
#: of the same regions read <= 22 mas.  (npairs, ratio) as recorded on #170.
BRICK_F405N_FALSE_POSITIVES = [(321, 8), (232, 5), (323, 6), (278, 6),
                               (287, 5), (241, 6), (266, 5)]


def _bins():
    edges_mas = np.arange(-MX_ARCSEC * 1000,
                          MX_ARCSEC * 1000 + XBIN_ARCSEC * 1000,
                          XBIN_ARCSEC * 1000)
    return edges_mas


def n_disk_bins():
    """Bins whose centres fall inside the search disk -- the real denominator."""
    e = _bins()
    c = (e[:-1] + e[1:]) / 2
    gx, gy = np.meshgrid(c, c, indexing="ij")
    return int((np.hypot(gx, gy) <= MX_ARCSEC * 1000).sum())


def measure(npairs, matched_fraction, offset_mas, rng):
    """Run the gate's own statistic on one synthetic cell.

    `matched_fraction` of the pairs are the same star seen twice, displaced by
    `offset_mas`; the rest are chance pairs, uniform over the search disk.
    Returns the gate's (off, ratio).
    """
    e = _bins()
    n_true = rng.binomial(npairs, matched_fraction)
    n_chance = npairs - n_true

    # chance pairs: uniform in area over the disk
    r = MX_ARCSEC * 1000 * np.sqrt(rng.random(n_chance))
    th = rng.random(n_chance) * 2 * np.pi
    dra = list(r * np.cos(th))
    dde = list(r * np.sin(th))
    # true pairs: at the displacement, with the per-star measurement scatter
    dra += list(rng.normal(offset_mas, 15.0, n_true))
    dde += list(rng.normal(0.0, 15.0, n_true))

    H, xb, yb = np.histogram2d(np.asarray(dra), np.asarray(dde), bins=[e, e])
    occupied = H[H > 0]
    bg = np.median(occupied) if occupied.size else 0.0
    ratio = H.max() / bg if bg > 0 else np.inf
    pi, pj = np.unravel_index(H.argmax(), H.shape)
    off = np.hypot((xb[pi] + xb[pi + 1]) / 2, (yb[pj] + yb[pj + 1]) / 2)
    return off, ratio, bg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(
        "docs", "reports", "figures", "registration_contrast_statistic.png"))
    ap.add_argument("--trials", type=int, default=40)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(20260810)

    pair_counts = np.array([100, 150, 220, 320, 450, 650, 950, 1400, 2000, 3000])
    offset = 90.0            # a real seam, well over OFF_MAX = 60 mas
    # A misregistered cell is one where a fraction of the pairs are the same star
    # seen twice, displaced.  Two fractions, because the fraction is a property of
    # the seam and not of the density: 25% is a cell wholly inside the misregistered
    # strip, 4% a cell the strip only clips -- which is most of them, since the strip
    # is narrower than a grid cell.
    curves = {0.25: dict(color="#1b6ca8", label="cell inside the seam (25% of pairs matched)"),
              0.04: dict(color="#6a3d9a", label="cell clipped by the seam (4% matched)")}

    bgs = []
    for n in pair_counts:
        bgs.append(np.median([measure(int(n), 0.25, offset, rng)[2]
                              for _ in range(args.trials)]))
    for frac, style in curves.items():
        rs_med, rs_lo, rs_hi = [], [], []
        for n in pair_counts:
            rs = np.array([measure(int(n), frac, offset, rng)[1]
                           for _ in range(args.trials)])
            rs_med.append(np.median(rs))
            rs_lo.append(np.percentile(rs, 16))
            rs_hi.append(np.percentile(rs, 84))
        style.update(med=rs_med, lo=rs_lo, hi=rs_hi)
    ratios = curves[0.25]["med"]

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # --- panel A: the background estimate is pinned at 1 -------------------
    axa.plot(pair_counts, bgs, "o-", color="#1b6ca8")
    axa.axhline(1.0, color="0.4", ls="--", lw=1)
    axa.set_xscale("log")
    axa.set_ylim(0, 2.2)
    axa.set_xlabel("pairs in the grid cell")
    axa.set_ylabel(r"median($H[H>0]$)  —  the 'background'")
    axa.set_title("A. The divisor is 1, at every realistic density\n"
                  f"{n_disk_bins():,} bins in the search disk, "
                  "so almost every\noccupied bin holds exactly one pair",
                  fontsize=9, loc="left")
    axa.text(0.5, 0.5, "so  ratio = H.max() / 1\n= the raw peak count",
             transform=axa.transAxes, ha="center", fontsize=11,
             bbox=dict(boxstyle="round", fc="#fff3cd", ec="#e0a800"))

    # --- panel B: the fail bar is a density cut ---------------------------
    for frac, style in curves.items():
        axb.fill_between(pair_counts, style["lo"], style["hi"],
                         color=style["color"], alpha=0.18)
        axb.plot(pair_counts, style["med"], "o-", color=style["color"],
                 label=style["label"])
    axb.axhline(FAIL_MIN_RATIO, color="#c0392b", lw=1.5,
                label=f"FAIL_MIN_RATIO = {FAIL_MIN_RATIO:.0f} (own-catalog)")
    axb.axhline(MIN_PEAK_RATIO, color="#e0a800", lw=1.5, ls="--",
                label=f"MIN_PEAK_RATIO = {MIN_PEAK_RATIO:.0f} (verify floor)")
    axb.scatter([p for p, _ in BRICK_F405N_FALSE_POSITIVES],
                [r for _, r in BRICK_F405N_FALSE_POSITIVES],
                marker="x", s=55, color="k", zorder=5,
                label="brick F405N, 7 cells:\nFALSE failure at 80 mas\n(same-star truth $\\leq$22 mas)")
    axb.set_xscale("log")
    axb.set_yscale("log")
    axb.set_xlabel("pairs in the grid cell")
    axb.set_ylabel("ratio = peak / background")
    axb.set_title("B. One 90 mas misregistration, scored in cells of\n"
                  "different star density: the bar is a density cut",
                  fontsize=9, loc="left")
    axb.legend(fontsize=7, loc="upper left")

    fig.suptitle("The registration seam check's confidence number is a star count",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")

    print(f"\nbins inside the search disk: {n_disk_bins():,}")
    print(f"{'pairs':>7} {'median(H[H>0])':>16} {'ratio':>8}")
    for n, b, r in zip(pair_counts, bgs, ratios):
        print(f"{n:>7} {b:>16.1f} {r:>8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
