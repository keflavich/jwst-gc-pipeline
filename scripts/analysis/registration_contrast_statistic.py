#!/usr/bin/env python
"""Why the registration seam check's confidence number depends on star density.

Supports issue #170.  Prints two tables; there is no figure, because this
repository keeps figures in the Overleaf astrometry-paper project rather than in
the tree (`.gitignore:38-39`).

GLOSSARY, since the issue this supports was filed in shorthand and could not be
reviewed on those terms:

  mosaic      one combined image of a field in one filter, made by resampling
              ("drizzling") many individual exposures onto a common grid.
  module      NIRCam images through two detector modules, A and B ("nrca",
              "nrcb").  Where their two footprints overlap on the sky, the
              combined mosaic has a seam.
  seam        a strip of the mosaic where the two modules' data are stitched
              together, and where a misregistration between them shows up.
  registration  whether a star's position on the mosaic is where the star
              actually is.
  truth set   a second list of positions that ought to agree with the mosaic's
              detections -- the same field's single-module mosaics, another
              filter's detections, or the mosaic's own source catalog.
  mas         milliarcsecond.  1 mas = 1/3600000 degree.  The pixels here are
              ~30 mas, and the effects in question are 60-90 mas.

WHERE THIS RUNS IN THE PIPELINE

`scripts/release/registration_failsafes.py` runs in the RELEASE GATE -- the last
step before a field's mosaics and catalogs would be published -- and can refuse
the release.  A whole-field average is not enough: the failure it exists to catch
(brick, proposal 1182, filter F356W, 2026-07) was several arcseconds of
misregistration confined to the module seam, with a field average of about zero.
So it lays a 20x20 grid over the mosaic and asks per cell.

Per cell: pair every bright detection with every truth position within 2.5
arcsec, and histogram the pair separations into 40x40 mas bins.  Pairs of a star
with ITSELF (seen once in the mosaic, once in the truth set) all land at the same
separation and pile into one bin; pairs of a star with an unrelated neighbour
spread over the whole search disk.  So the histogram is one peak on a thin floor,
and WHERE THE PEAK SITS is the cell's measured misregistration.

    off   = distance of the peak bin from zero, in mas
    ratio = H.max() / median(H[H > 0])      # "is that peak real?"

where H is the 2-D histogram of pair separations.  A cell FAILS -- and one
failing cell fails the whole field -- when ALL of:

    npair >= MIN_PAIRS (80)  and  ratio >= MIN_PEAK_RATIO (5)   [ = "verified" ]
    off   >  OFF_MAX (60 mas)
    ratio >= FAIL_MIN_RATIO

`FAIL_MIN_RATIO` is 10 for the own-catalog check and 5 for the cross-band one.

WHAT THIS SCRIPT SHOWS

Table 1: `median(H[H > 0])` is exactly 1 at every density that occurs, because
the search disk holds ~12,000 bins and a cell holds tens to hundreds of pairs.
So `ratio` is not a ratio -- it is the raw number of pairs in the peak bin.

Table 2: a raw count scales with how many stars the cell holds.  ONE fully
misregistered cell -- every star in it displaced by the same 90 mas -- scores
7 when the cell holds 15 stars and 236 when it holds 400.  The fail bar is
therefore crossed at a STAR DENSITY, not at a misregistration level.

MODELLING NOTE, because the first version of this script got it wrong.  A cell
is modelled as N detections and their N truth counterparts in a 45-arcsec box;
chance pairs then arise on their own from the other truth stars inside the search
radius, rather than being put in by hand.  The earlier version modelled a cell
that a seam only CLIPS as "a few displaced pairs plus uniform noise", which
silently deleted the correctly-registered stars in the rest of that cell.  With
them present the peak sits at zero and the cell is never even a fail candidate
(`off > OFF_MAX` is false), so that curve described nothing real.  A partially
clipped cell is not the interesting case; a fully misregistered one is.

Usage:
    python scripts/analysis/registration_contrast_statistic.py
"""
import argparse

import numpy as np

# The estimator's own geometry, copied from registration_failsafes.py so this
# script measures what the release gate measures.
MX_ARCSEC = 2.5          # pair-separation search radius
XBIN_ARCSEC = 0.04       # offset-histogram bin
MIN_PAIRS = 80           # pairs needed before a cell is judged at all
MIN_PEAK_RATIO = 5.0     # below this a cell is UNVERIFIED -- neither pass nor fail
FAIL_MIN_RATIO = 10.0    # the own-catalog fail bar
OFF_MAX = 60.0           # a verified cell peaking beyond this is a candidate fail

#: Sky size of one cell of the 20x20 grid on a typical NIRCam mosaic.
CELL_ARCSEC = 45.0

#: Per-star position scatter between a mosaic and its truth set, arcsec.
POS_SCATTER_ARCSEC = 0.015

#: The seven brick F405N cells that were a FALSE own-catalog failure in 2026-07:
#: each read an 80 mas offset at ratio 5-8, while an independent same-star
#: comparison of the same regions read <= 22 mas.  (npairs, ratio), from #170.
BRICK_F405N_FALSE_POSITIVES = [(321, 8), (232, 5), (323, 6), (278, 6),
                               (287, 5), (241, 6), (266, 5)]


def bin_edges():
    return np.arange(-MX_ARCSEC * 1000, MX_ARCSEC * 1000 + XBIN_ARCSEC * 1000,
                     XBIN_ARCSEC * 1000)


def n_disk_bins():
    """Bins whose CENTRES fall inside the search disk."""
    c = (bin_edges()[:-1] + bin_edges()[1:]) / 2
    gx, gy = np.meshgrid(c, c, indexing="ij")
    return int((np.hypot(gx, gy) <= MX_ARCSEC * 1000).sum())


def measure_cell(n_stars, offset_mas, rng, displaced_fraction=1.0):
    """Run the gate's own statistic on one synthetic cell.

    ``n_stars`` detections in a ``CELL_ARCSEC`` box, each with a truth-set
    counterpart; ``displaced_fraction`` of those counterparts are displaced by
    ``offset_mas``.  Chance pairs are not injected -- they arise from the other
    truth stars that fall inside the search radius, which is where they come
    from in the real data.

    Returns ``(npairs, off_mas, ratio, background)``.
    """
    x = rng.random(n_stars) * CELL_ARCSEC
    y = rng.random(n_stars) * CELL_ARCSEC
    displaced = rng.random(n_stars) < displaced_fraction
    tx = x + np.where(displaced, offset_mas / 1000.0, 0.0) \
        + rng.normal(0, POS_SCATTER_ARCSEC, n_stars)
    ty = y + rng.normal(0, POS_SCATTER_ARCSEC, n_stars)

    dx = (tx[None, :] - x[:, None]).ravel() * 1000.0
    dy = (ty[None, :] - y[:, None]).ravel() * 1000.0
    inside = np.hypot(dx, dy) <= MX_ARCSEC * 1000
    e = bin_edges()
    H, xb, yb = np.histogram2d(dx[inside], dy[inside], bins=[e, e])
    occupied = H[H > 0]
    bg = np.median(occupied) if occupied.size else 0.0
    pi, pj = np.unravel_index(H.argmax(), H.shape)
    off = np.hypot((xb[pi] + xb[pi + 1]) / 2, (yb[pj] + yb[pj + 1]) / 2)
    return (int(inside.sum()), float(off),
            float(H.max() / bg) if bg > 0 else np.inf, float(bg))


def sweep(counts, rng, trials, offset_mas=90.0, displaced_fraction=1.0):
    rows = []
    for n in counts:
        got = np.array([measure_cell(n, offset_mas, rng, displaced_fraction)
                        for _ in range(trials)])
        rows.append((n,) + tuple(np.median(got, axis=0)))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trials", type=int, default=25)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(20260810)
    counts = [15, 20, 30, 45, 70, 110, 170, 260, 400]
    rows = sweep(counts, rng, args.trials)

    print(f"bins inside the {MX_ARCSEC}\" search disk: {n_disk_bins():,}\n")

    print("TABLE 1 -- the divisor is 1, so `ratio` is the raw peak-bin count")
    print(f"{'stars/cell':>11}{'npairs':>9}{'median(H[H>0])':>17}")
    for n, npair, _off, _ratio, bg in rows:
        print(f"{n:>11}{npair:>9.0f}{bg:>17.1f}")

    print("\nTABLE 2 -- ONE 90 mas misregistration, every star in the cell displaced")
    print(f"{'stars/cell':>11}{'npairs':>9}{'off (mas)':>11}{'ratio':>8}  verdict")
    for n, npair, off, ratio, _bg in rows:
        if npair < MIN_PAIRS or ratio < MIN_PEAK_RATIO:
            verdict = "UNVERIFIED (not even judged)"
        elif off > OFF_MAX and ratio >= FAIL_MIN_RATIO:
            verdict = "FAIL"
        elif off > OFF_MAX:
            verdict = "high offset, under the fail bar -> reported, not failed"
        else:
            verdict = "pass"
        print(f"{n:>11}{npair:>9.0f}{off:>11.0f}{ratio:>8.0f}  {verdict}")

    print(f"\nSame seam throughout.  The own-catalog fail bar is "
          f"FAIL_MIN_RATIO = {FAIL_MIN_RATIO:.0f}, crossed between "
          f"{rows[0][0]} and {rows[2][0]} stars per cell.")
    print("The seven brick F405N cells that were a FALSE failure in 2026-07 sat "
          "at ratio 5-8\n(npairs 232-323), i.e. squarely in that same band.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
