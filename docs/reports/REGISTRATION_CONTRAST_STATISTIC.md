# The registration seam check's confidence number is a star count

**Summary: the number the release gate uses to decide whether a suspicious grid
cell is trustworthy is, arithmetically, the raw number of star pairs in one
histogram bin. It grows with star density, so the same physical misregistration
scores 5 in a sparse cell and 70 in a crowded one, and the threshold behaves as
a cut on density rather than on confidence.**

Regenerate with:

```bash
python scripts/analysis/registration_contrast_statistic.py
```

![the confidence number is a star count](figures/registration_contrast_statistic.png)

Background for issue #170, which was filed in shorthand ("the FAIL discriminant
is density-coupled") and could not be reviewed on those terms.

## Where in the pipeline this happens

At the very end, in the **release gate** — after a field has been reduced,
drizzled into mosaics and cataloged, and immediately before its images and
catalogs would be published. `scripts/release/registration_failsafes.py` runs
there and can refuse the release.

What it checks is **registration**: that a star's position on the delivered
mosaic is where the star actually is. A whole-field average is not enough, and
we know that from a specific failure — brick 1182 F356W in July 2026 carried
several arcseconds of misregistration confined to the narrow strip where the two
NIRCam module footprints overlap, while the field average was about zero. So the
check is spatially resolved: it lays a 20×20 grid over the mosaic and asks the
question separately in each cell.

## What it does inside one cell

Take every bright source detected on the mosaic being checked, and a **truth
set** — a second list of positions that ought to agree with it. There are three
truth sets, run as three independent checks:

| check | truth set | what it can catch |
|---|---|---|
| per-module | the same band's single-module (nrca / nrcb) mosaics | junk created where the two modules are combined |
| cross-band | detections in another JWST filter | a band that has drifted relative to the others |
| own-catalog | the mosaic's own vetted source catalog | a mosaic that disagrees with the catalog derived from it |

None of them uses an external reference catalogue, deliberately: crowding and
extinction cannot fool a check whose two sides come from the same telescope.

Then, per cell: pair each detection with every truth position within 2.5
arcsec, and histogram those pair separations into 40 × 40 mas bins.

* If the mosaic is registered, every true pair lands near zero separation and
  piles into one bin. Pairs of a star with some unrelated neighbour spread
  thinly over the whole search disk.
* If the cell is misregistered by, say, 90 mas, the pile-up sits 90 mas from
  zero instead.

So the histogram is one peak on a thin floor, and **where the peak sits is the
cell's measured misregistration**. Two numbers come out:

```python
off   = distance of the peak bin from zero, in mas
ratio = H.max() / median(H[H > 0])      # peak count over the median occupied bin
```

`off` is the measurement. `ratio` is meant to answer "is that peak real, or did
a handful of chance pairs happen to land in one bin?" A cell FAILS — and one
failing cell fails the whole field, blocking the release — only when

```
off > OFF_MAX (60 mas)   AND   ratio >= FAIL_MIN_RATIO
```

`FAIL_MIN_RATIO` is 10 for the own-catalog check and 5 for the other two. The
asymmetry was introduced in #166/#172 to stop a specific false alarm, described
below.

## Why `ratio` does not mean what its name says

**Panel A.** The search disk of radius 2.5 arcsec contains **12,281** bins of
40 × 40 mas. A grid cell holds a few hundred pairs. With forty times more bins
than pairs, almost every occupied bin outside the peak holds exactly one pair —
so `median(H[H > 0])` is **exactly 1**, at every density that occurs in this
survey:

| pairs in the cell | 100 | 220 | 450 | 950 | 2000 | 3000 |
|---|---|---|---|---|---|---|
| `median(H[H>0])` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| `ratio` | 13 | 32 | 66 | 136 | 296 | 445 |

`ratio` is therefore `H.max() / 1` — **the raw number of pairs in the peak bin**.
It is not a ratio and it is not normalised by anything. `FAIL_MIN_RATIO = 10`
currently means "the peak bin must contain at least ten pairs".

**Panel B.** A count scales with how many stars the cell contains. Both curves
are the *same* 90 mas misregistration, measured by the *same* estimator; they
differ only in how many stars the cell holds and what fraction of its pairs the
seam displaces:

* a cell lying wholly inside the misregistered strip (25% of pairs displaced)
  scores 13 at 100 pairs and 445 at 3000;
* a cell the strip merely clips (4% displaced — which is most of them, since the
  strip is narrower than a grid cell) crosses `FAIL_MIN_RATIO = 10` at roughly
  450 pairs. Below that density the identical seam is recorded as
  "verified-but-not-confident" and does not fail the field. Above it, it does.

So the bar is crossed at a **star density**, not at a misregistration level.
And it is hardest to clear in the sparsest cells — while the crowded cells,
where it is easiest to clear, are the Galactic Centre field interiors, where a
seam matters most *and* where chance-pair confusion is worst.

## The false alarm that motivated the current threshold

In July 2026 the own-catalog check failed brick F405N on seven cells, each
reading an 80 mas offset at `ratio` 5–8. An independent same-star comparison of
those same regions read ≤ 22 mas, so the cells were not misregistered and the
failure was spurious. #166/#172 responded by raising the own-catalog bar from 5
to 10, which removed those seven and left the other two checks at 5.

Those seven cells are the black crosses in panel B, at 232–323 pairs. They land
**on the purple curve** — i.e. exactly where a real seam that clips a cell of
that density would also land. The threshold separated the two populations on
this band, but not because the statistic distinguishes them; at fixed density it
cannot. That is the whole of #170.

One thing recorded in that issue and still unexplained, which should not be lost
in the statistics: all seven cells peak at the **same vector**, +80 mas in RA,
to the bin. Chance pairs would not agree on a direction. That may be a small
real systematic between the F405N mosaic and its own catalog rather than noise —
in which case the raised bar is suppressing a genuine signal, not a false one.
It has not been chased down.

## What would replace it

Two changes, from the #170 discussion, both still unimplemented:

1. **A density-flat significance.** Not `(H.max() − bg) / sqrt(bg)` with the
   same `bg` — with `bg` pinned at 1 that is identically `ratio − 1` and adds
   nothing. The estimator has to use the *expected* chance-pair background,
   `lam = npairs / n_disk_bins`, which is fractional and keeps scaling with
   density where the median saturates. Peak goes as *n* and `lam` as *n²*, so
   the significance is flat in density where the count is linear in it.

2. **Contiguity as a second, independent axis.** A misregistration is a
   connected patch of cells; chance-pair noise is scattered singletons. This
   needs no new measurement — the offsets and the verified flags are already on
   the grid — and in the #179 trial it was much the stronger discriminant,
   firing on 365 / 179 / 45 cells of three injected seams against the count's
   273 / 127 / 29, and on **zero** cells across all ten real brick bands and all
   five cloudc bands.

   With a caveat that was measured and matters: the minimum patch size has to be
   **3 cells, not 2**. Two of the seven brick false positives are 4-adjacent, so
   a 2-cell bar re-creates the exact false alarm this is meant to remove.

A previous attempt at both, PR #179, was closed without a stated reason and
nothing from it is on `main` — `FAIL_MIN_RATIO = 10.0` is still at
`registration_failsafes.py:51`, and neither `FAIL_MIN_SIG` nor
`MIN_SEAM_CELLS` exists anywhere in the tree.

## Caveats on this document

* Both panels are the real estimator — the same bin geometry, the same
  `H.max() / median(H[H>0])` — run over **synthetic** pair populations. That is
  deliberate: it shows a property of the statistic rather than of any one field,
  and it is reproducible without the archive. It is not a measurement of any
  real seam.
* The two curves' matched fractions (25% and 4%) are illustrative of a cell
  inside and a cell clipped by a strip. The *shape* of panel B — a count linear
  in density, crossing a fixed bar at a density — does not depend on them.
* The per-star scatter is taken as 15 mas, which widens the peak into
  neighbouring bins and if anything makes `ratio` look *better* behaved than it
  is.
