#!/usr/bin/env python
"""Why the registration seam check's confidence number depends on star density.

Supports issue #170.  Prints four tables and writes the figure to
``docs/reports/figures/registration_contrast_statistic.png``.

The figure is NOT committed: ``.gitignore`` carries a blanket ``*.png`` and the
maintainer's position is that figures belong in the pull request and issue
discussion rather than in the tree.  Run this script to produce it.

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

Table 1: `median(H[H > 0])` is exactly 1 at every density the gate actually
judges -- the 15-star row reads 1.5, and is never judged -- because
the search disk holds ~12,000 bins and a cell holds tens to hundreds of pairs.
So `ratio` is not a ratio -- it is the raw number of pairs in the peak bin.

Table 2: a raw count scales with how many stars the cell holds.  ONE fully
misregistered cell -- every star in it displaced by the same 90 mas -- scores
7 when the cell holds 15 stars and 236 when it holds 400.  The fail bar is
therefore crossed at a STAR DENSITY, not at a misregistration level.

And it is crossed low.  The seven brick cells that were a false failure read
ratio 5-8 at 232-323 pairs, their peak bins holding 1.7-2.5% of their pairs.
That fraction does NOT by itself make them false -- TABLE 4 below shows #179's
REAL injected seams reaching the same level, because a seam displaces each cell's
existing peak.  What establishes these seven as false is the independent
same-star reading of <= 22 mas.  The gate's OWN record is the better anchor for where the
bar sits: the comment on registration_failsafes.py's own `FAIL_MIN_RATIO` notes
that clean brick cells verify at median contrast ~18, so the bar sits UNDER what
a correctly registered cell of that field already scores.

Table 3: what the proposed replacement actually does over the same rows.  It is
printed rather than described because two prose claims about it were wrong: the
current statistic does NOT "effectively divide by lam" (it divides by the median
occupied bin, pinned at 1), and (peak-lam)/sqrt(lam) is not flat -- it climbs
x1.43 across the judged rows, against the raw count's x5.8.  Much flatter, not
flat.  Dividing by lam itself over-corrects: peak/lam FALLS with density.

CALIBRATION CAVEAT: run at those same pair counts (solved for, not read off a
nearby row), this model puts a fully misregistered cell at 66-82 -- about 4x the
recorded ~18 for real clean cells.  The run derives and prints that range, and
sweeps two seeds against five trial counts to report how far it moves, rather
than quoting either.  The model assumes every detection has a truth counterpart and
that all of them lie in one cell, so
its absolute values are UPPER BOUNDS.  The argument rests on the scaling
(ratio proportional to star count, thresholds fixed), which is arithmetic; a
replacement threshold must be calibrated on real cells, not on this table.

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
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

#: What a CORRECTLY registered brick cell actually scores, per the release
#: gate's own record (the comment on registration_failsafes.py::FAIL_MIN_RATIO).
#: This is the anchor the fail bar should be compared against -- not this script's model, whose
#: absolute values are upper bounds.
CLEAN_BRICK_CONTRAST = 18.0

#: What the replacement scores on REAL seams, from #179's trial: three synthetic
#: +90 mas seams injected into real brick F405N data (whole field, half field,
#: narrow declination band), as (min, median, max) of the significance over the
#: high-offset cells of each.
#:
#: This is here because the modelled seam rows below are an UPPER BOUND and must
#: not be compared against the real false alarms on their own.  #179 measured the
#: real thing and reported, under a heading of its own, that the two populations
#: OVERLAP at the low end -- both start at sig ~32.6, because an injected seam
#: displaces each cell's existing peak, so a seam's weakest cells are the field's
#: intrinsically weakest cells, which are the artifact cells.  Its conclusion:
#: "No amplitude statistic can separate them per-cell", and "that overlap is
#: exactly why step 3 [contiguity] matters more than step 2".
#: (label, sig min/med/max, RAW RATIO min/med/max).  The ratio column matters as
#: much as the sig one and was omitted at first: #179's real seams score ratio
#: 5-49, so they START at the false alarms' own 5-8; their medians (12-18)
#: CLEAR FAIL_MIN_RATIO = 10 while their sub-median tail does not, so the bar
#: falls INSIDE the real seam distribution.  Any sentence comparing this
#: model's 66-82 against the real 5-8 is the same upper-bound-versus-real
#: mistake Table 4 was corrected for, one axis over.
REAL_SEAM_SIG_179 = [("whole field", 32.6, 78.1, 140.4, 5, 18, 49),
                     ("half field", 32.6, 75.4, 115.0, 5, 17, 39),
                     ("narrow dec band", 30.1, 60.7, 97.0, 5, 12, 27)]

#: #179's chosen bar, and how it was chosen: the log-midpoint of the artifact
#: ceiling (49.3) and the hardest real seam's median (60.7) -- 12% headroom, not
#: a clear gap.
FAIL_MIN_SIG_179 = 55.0

#: Extra seeds and trial counts used only to report how far the derived range
#: moves with the sampling.  Fixed so the run stays reproducible.  BOTH knobs
#: are swept: at a fixed trial count the seed-only range reads much tighter than
#: the quantity really is, which would understate exactly what this is measuring.
SAMPLING_SEEDS = (20260811, 20260812)
SAMPLING_TRIALS = (5, 10, 25, 50, 100)


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


def model_ratio_at_npairs(targets, rows, rng, trials, tol=0.02, passes=6,
                          quiet=False):
    """What this model scores at a GIVEN pair count.

    The real cells being compared against are quoted by pair count, not by star
    count, and they fall BETWEEN the swept rows.  Rather than read the nearest
    row -- or, worse, quote a number by hand -- solve for the star count that
    produces each target pair count and run the model there.

    Interpolating the sweep gets within ~6%, which is not close enough to say
    "at the same density": pair count grows faster than linearly in star count
    (~n^1.5 here, since each extra star pairs with every other one inside the
    search radius), so a straight-line inverse overshoots.  So the interpolated
    guess is refined by rerunning and correcting along that local power law
    until the realised pair count is within ``tol`` of the target.

    A cell's pair count is a random variable, so the target cannot be hit
    exactly and the refinement can run out of passes without converging -- at
    ``--trials 5`` it lands 2.6% off.  The realised pair count is therefore
    RETURNED, never the target, and a run that did not converge says so rather
    than presenting an off-target result as an on-target one.  Callers should
    quote the realised counts.

    Returns ``[(target, realised npairs, ratio, converged), ...]``.
    """
    lp = np.log([r[1] for r in rows])
    ls = np.log([r[0] for r in rows])
    slope = np.gradient(lp, ls)          # d log(npairs) / d log(stars)
    out = []
    for t in targets:
        n = max(1, int(round(np.exp(np.interp(np.log(t), lp, ls)))))
        converged = False
        for _ in range(passes):
            _n, npair, _off, ratio, _bg = sweep([n], rng, trials)[0]
            if npair > 0 and abs(npair - t) / t <= tol:
                converged = True
                break
            k = float(np.interp(np.log(max(npair, 1)), lp, slope))
            n = max(1, int(round(n * (t / max(npair, 1)) ** (1.0 / max(k, 0.5)))))
        if not converged and not quiet:
            print(f"NOTE: solving for {t} pairs stopped at {npair:.0f} after "
                  f"{passes} passes ({100 * abs(npair - t) / t:.1f}% off, "
                  f"tolerance {100 * tol:.0f}%).  Raise --trials to tighten it.")
        out.append((t, npair, ratio, converged))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--out", default=os.path.join(
        "docs", "reports", "figures", "registration_contrast_statistic.png"))
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

    # TABLE 3 -- the replacement, measured rather than asserted.  Two claims
    # about it were wrong in prose before this table existed: that the current
    # statistic "effectively divides by lam" (it divides by the median occupied
    # bin, pinned at 1), and that (peak-lam)/sqrt(lam) is flat (it climbs, just
    # far less than the raw count does).
    nb = n_disk_bins()
    print(f"\nTABLE 3 -- what the proposed replacement does over the same rows")
    print(f"{'stars/cell':>11}{'peak':>7}{'lam':>10}{'peak/lam':>10}"
          f"{'(peak-lam)/sqrt(lam)':>22}")
    sig = []
    for n, npair, _off, ratio, bg in rows:
        lam = npair / nb
        peak = ratio * bg
        s = (peak - lam) / np.sqrt(lam)
        sig.append((n, npair, peak, lam, peak / lam, s))
        print(f"{n:>11}{peak:>7.0f}{lam:>10.4f}{peak / lam:>10.0f}{s:>22.0f}")
    judged = [r for r in sig if r[1] >= MIN_PAIRS]
    if judged:
        raw_fold = judged[-1][2] / judged[0][2]
        sig_fold = judged[-1][5] / judged[0][5]
        lam_fold = judged[-1][4] / judged[0][4]
        print(f"\nOver the rows the gate JUDGES ({judged[0][0]}-{judged[-1][0]} "
              f"stars):\n"
              f"  raw count (what it uses now)   {judged[0][2]:.0f} -> "
              f"{judged[-1][2]:.0f}   x{raw_fold:.1f}\n"
              f"  peak/lam                       {judged[0][4]:.0f} -> "
              f"{judged[-1][4]:.0f}   x{lam_fold:.2f}  (FALLS with density)\n"
              f"  (peak-lam)/sqrt(lam)           {judged[0][5]:.0f} -> "
              f"{judged[-1][5]:.0f}   x{sig_fold:.2f}\n"
              f"So the replacement is much flatter than the raw count, not flat.")

    crossed = next((r[0] for r in rows if r[3] >= FAIL_MIN_RATIO), None)
    failed = next((r[0] for r in rows if r[1] >= MIN_PAIRS
                   and r[3] >= MIN_PEAK_RATIO and r[2] > OFF_MAX
                   and r[3] >= FAIL_MIN_RATIO), None)
    print(f"\nSame seam throughout.  The own-catalog fail bar is "
          f"FAIL_MIN_RATIO = {FAIL_MIN_RATIO:.0f}.  The ratio first reaches it "
          f"at {crossed} stars per\ncell -- but that cell is NOT judged (its "
          f"pair count is under MIN_PAIRS = {MIN_PAIRS:.0f}), so\nclearing the "
          f"bar there decides nothing.  The first row that actually FAILS is "
          f"{failed}\nstars per cell.")
    frac = [100.0 * r / n for n, r in BRICK_F405N_FALSE_POSITIVES]
    obs_npairs = [n for n, _ in BRICK_F405N_FALSE_POSITIVES]
    at = model_ratio_at_npairs([min(obs_npairs), max(obs_npairs)],
                               rows, rng, args.trials)
    lo, hi = at[0][2], at[1][2]
    # Quote the pair counts the model REACHED, not the ones it aimed at -- and
    # say so when it did not reach them, rather than calling them "matching".
    solved = "matching" if all(r[3] for r in at) else "the best it reached for"
    print(f"\nThe seven brick F405N cells that were a FALSE failure in 2026-07 "
          f"read ratio 5-8\nat npairs {min(obs_npairs)}-{max(obs_npairs)} -- "
          f"peak bins holding "
          f"{min(frac):.1f}-{max(frac):.1f}% of their pairs -- a fraction TABLE 4 "
          f"below shows\nreal seams reach too, so it does not by itself make "
          f"them false; the <= 22 mas\nsame-star reading does.\nThe gate's own record (registration_failsafes.py::FAIL_MIN_RATIO) "
          f"puts CLEAN brick cells at\nmedian contrast "
          f"~{CLEAN_BRICK_CONTRAST:.0f}, so FAIL_MIN_RATIO "
          f"= {FAIL_MIN_RATIO:.0f} is under what a correctly registered cell "
          f"scores.\nRun at {solved} pair counts ({at[0][1]:.0f} and "
          f"{at[1][1]:.0f}, solved for), this model scores {lo:.0f}-{hi:.0f} --"
          f"\n~{(lo + hi) / 2 / CLEAN_BRICK_CONTRAST:.0f}x the recorded "
          f"~{CLEAN_BRICK_CONTRAST:.0f} -- so its absolute values are upper "
          f"bounds; the SCALING\nis the argument.")

    # The spread is derived here rather than quoted: a hardcoded range is the
    # defect this script was pulled up on twice, and the second time it was
    # merely moved into prose.  Sweep BOTH knobs that move it -- the seed and
    # the trial count -- because at a fixed trial count the seed-only range
    # reads far tighter (66-67) than the quantity actually is (the both-knobs range the run prints).
    los, his, fold = [lo], [hi], []
    for seed in SAMPLING_SEEDS:
        for trials in SAMPLING_TRIALS:
            got = model_ratio_at_npairs([min(obs_npairs), max(obs_npairs)],
                                        rows, np.random.default_rng(seed),
                                        trials, quiet=True)
            los.append(got[0][2])
            his.append(got[1][2])
    fold = [(a + b) / 2 / CLEAN_BRICK_CONTRAST for a, b in zip(los, his)]
    print(f"\nSampling, over {len(SAMPLING_SEEDS)} seeds x "
          f"{len(SAMPLING_TRIALS)} trial counts {list(SAMPLING_TRIALS)}: the "
          f"low end moves over\n{min(los):.0f}-{max(los):.0f} and the high end "
          f"over {min(his):.0f}-{max(his):.0f}, while the fold above "
          f"~{CLEAN_BRICK_CONTRAST:.0f} -- midpoint over "
          f"{CLEAN_BRICK_CONTRAST:.0f},\nthe one definition used throughout -- "
          f"stays at {min(fold):.1f}-{max(fold):.1f}x.  So quote the range as "
          f"approximate.\nAnd the ~{CLEAN_BRICK_CONTRAST:.0f} has no pair count "
          f"attached, so some of the gap could be density\nrather than "
          f"normalisation.")

    # TABLE 4 -- the DEMONSTRATION.  Table 3 only shows the replacement on cells
    # that are fully misregistered; what a threshold has to do is separate those
    # from the cells that were a FALSE alarm.  Both populations, one statistic.
    print("\nTABLE 4 -- does the replacement separate the real seam from the "
          "false alarm?")
    print(f"{'':>26}{'npairs':>9}{'raw ratio':>11}{'(peak-lam)/sqrt(lam)':>22}")
    fp_sig = []
    for npair, ratio in sorted(BRICK_F405N_FALSE_POSITIVES):
        lam = npair / nb
        s = (ratio - lam) / np.sqrt(lam)
        fp_sig.append(s)
        print(f"{'brick F405N false alarm':>26}{npair:>9}{ratio:>11}{s:>22.1f}")
    for n, npair, _off, ratio, bg in rows:
        if npair < MIN_PAIRS:
            continue
        lam = npair / nb
        s = (ratio * bg - lam) / np.sqrt(lam)
        print(f"{'modelled 90 mas seam':>26}{npair:>9.0f}{ratio:>11.0f}{s:>22.1f}")
    judged_sig = [((r[3] * r[4]) - r[1] / nb) / np.sqrt(r[1] / nb)
                  for r in rows if r[1] >= MIN_PAIRS]
    for label, lo_s, med_s, hi_s, lo_r, med_r, hi_r in REAL_SEAM_SIG_179:
        print(f"{('#179 real, ' + label):>26}{'':>9}"
              f"{f'{lo_r}-{hi_r} (med {med_r})':>11}"
              f"{f'{lo_s:.1f}-{hi_s:.1f} (med {med_s:.1f})':>22}")
    real_lo = min(r[1] for r in REAL_SEAM_SIG_179)
    print(f"\n  false alarms          {min(fp_sig):.1f} - {max(fp_sig):.1f}   "
          f"(real, measured)\n"
          f"  modelled seams      {min(judged_sig):.0f} - {max(judged_sig):.0f}"
          f"   (THIS model -- upper bounds)\n"
          f"  #179's real seams     {real_lo:.1f} - "
          f"{max(r[3] for r in REAL_SEAM_SIG_179):.1f}   (real, measured)\n")
    print(f"  READ THE THIRD ROW, NOT THE SECOND.  Against the modelled seams "
          f"the gap looks like\n  "
          f"{min(judged_sig) / max(fp_sig):.1f}x with nothing in between -- but "
          f"those values are upper bounds, roughly\n  "
          f"{min(judged_sig) / REAL_SEAM_SIG_179[0][2]:.0f}x #179's real seam "
          f"medians, and an upper bound on one population is exactly\n  what "
          f"destroys a claim about the SEPARATION.  Measured on real data the two"
          f"\n  populations OVERLAP at the low end -- false alarms from "
          f"{min(fp_sig):.1f}, real seams from {real_lo:.1f}; #179 rounds\n  both "
          f"to 'sig ~32.6' -- because an "
          f"injected seam displaces each cell's "
          f"existing peak, so a seam's weakest cells are the field's\n  "
          f"intrinsically weakest -- which are the artifact cells.\n")
    print(f"  So #179 set FAIL_MIN_SIG = {FAIL_MIN_SIG_179:.0f} as the "
          f"log-midpoint of the artifact ceiling\n  ({max(fp_sig):.1f}) and the "
          f"hardest seam's median (60.7): 12% headroom, not a clear gap.  Its own\n"
          f"  words: \"no amplitude statistic can separate them per-cell\", and "
          f"\"that overlap is\n  exactly why step 3 [contiguity] matters more "
          f"than step 2\".\n")
    ratio_los = [r[4] for r in REAL_SEAM_SIG_179]
    ratio_meds = [r[5] for r in REAL_SEAM_SIG_179]
    print(f"  What the amplitude axis DOES buy is a field-level margin: "
          f"{max(fp_sig):.1f} -> {FAIL_MIN_SIG_179:.0f} against\n  the raw "
          f"count's 8 -> {FAIL_MIN_RATIO:.0f}, and it fires on 290 cells where "
          f"the raw count fires on 273.\n")
    print(f"  And the RAW COUNT is worse than the modelled rows suggest, for the "
          f"same reason.\n  This model puts a seam at those pair counts at "
          f"{lo:.0f}-{hi:.0f}; #179's REAL seams score ratio\n  "
          f"{min(ratio_los)}-{max(r[6] for r in REAL_SEAM_SIG_179)} with medians "
          f"of {min(ratio_meds)}-{max(ratio_meds)}.  They START at the false "
          f"alarms' own 5-8; their medians\n  ({min(ratio_meds)}-"
          f"{max(ratio_meds)}) CLEAR FAIL_MIN_RATIO = {FAIL_MIN_RATIO:.0f}, "
          f"while their sub-median tail does not.\n  So the bar does not merely "
          f"sit too low to separate the populations -- it falls INSIDE\n  the "
          f"real seam distribution, failing weak seam cells and artifact cells "
          f"alike.  That is the case against the raw\n  count, "
          f"and it is made on measured data rather than on this model.")


    stars = [r[0] for r in rows]
    ratios = [r[3] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    judged = [r[1] >= MIN_PAIRS and r[3] >= MIN_PEAK_RATIO for r in rows]
    ax.plot(stars, ratios, "o-", color="#1b6ca8", zorder=3,
            label="one 90 mas misregistration,\nevery star in the cell displaced")
    ax.scatter([s for s, j in zip(stars, judged) if not j],
               [r for r, j in zip(ratios, judged) if not j],
               s=140, facecolors="none", edgecolors="#888", linewidths=1.6,
               zorder=4, label=f"not judged at all (npairs < MIN_PAIRS = {MIN_PAIRS:.0f})")
    ax.axhline(FAIL_MIN_RATIO, color="#c0392b", lw=1.5,
               label=f"FAIL_MIN_RATIO = {FAIL_MIN_RATIO:.0f} (own-catalog)")
    # The clean-cell anchor belongs ON the figure: the figure is what gets read,
    # and the point of #170 is that the fail bar sits BELOW what a correctly
    # registered cell of this field already scores.
    ax.axhline(CLEAN_BRICK_CONTRAST, color="#2e7d32", lw=1.5, ls="--",
               label=f"clean brick cells score ~{CLEAN_BRICK_CONTRAST:.0f}\n"
                     f"(registration_failsafes.py::FAIL_MIN_RATIO)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("stars in the grid cell")
    ax.set_ylabel("ratio  =  peak bin count / median occupied bin\n(= the raw peak count, since the divisor is 1)")
    ax.set_title("The same seam scores 7 in a sparse cell and 236 in a crowded one",
                 fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=.25, which="both")
    # Read on its own -- as it will be, pasted into the issue -- the curve would
    # look like a calibrated prediction.  It is not.
    ax.text(0.98, 0.03,
            "model: every detection has a truth counterpart, all in one cell\n"
            "-> absolute values are UPPER BOUNDS; the scaling is the argument",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            color="#555")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
