# The registration seam check's confidence number is a star count

**Summary: the number the release gate uses to decide whether a suspicious patch
of a mosaic is trustworthy is, arithmetically, the raw number of star pairs in
one histogram bin. It grows with star density, so one and the same
misregistration is silently ignored in a sparse patch and fails the release in a
crowded one.**

Reproduce with:

```bash
python scripts/analysis/registration_contrast_statistic.py
```

Background for issue #170, which was filed in shorthand ("the FAIL discriminant
is density-coupled") and could not be reviewed on those terms.

![the same seam scores 7 in a sparse cell and 236 in a crowded one](https://raw.githubusercontent.com/keflavich/jwst-gc-pipeline/efb8943/docs/reports/figures/registration_contrast_statistic.png)

**The figure is not committed to this repository.** `.gitignore` carries a
blanket `*.png` under a comment saying figures live in the Overleaf project, and
an earlier revision of this branch force-added it anyway on the precedent of
`docs/reports/apphot/` (13 PNGs) and `docs/evidence/satstar_fit_footprint/` (2).
That was reverted at the maintainer's request — figures belong in the pull
request and issue discussion, not in the tree. Regenerate it locally with the
command above; the image linked here is the one this branch produces, served
from the blob at `efb8943`, which stays reachable through `refs/pull/372/head`.

## Glossary

Everything below is written for someone who does not work on this pipeline.

| term | meaning |
|---|---|
| **mosaic** | one combined image of a field in one filter, made by resampling ("drizzling") many exposures onto a common grid |
| **module** | NIRCam images through two detector modules, A and B (`nrca`, `nrcb`). Their footprints overlap on the sky |
| **seam** | the strip where the two modules' data are stitched together — where a misregistration between them appears |
| **registration** | whether a star's position on the mosaic is where the star actually is |
| **truth set** | a second list of positions that ought to agree with the mosaic's detections |
| **mas** | milliarcsecond, 1/3 600 000 of a degree. Pixels here are ~30 mas; the effects in question are 60–90 mas |
| **H** | the 2-D histogram of star-pair separations, described below |

## Where in the pipeline this happens

At the very end, in the **release gate** — after a field has been reduced,
drizzled into mosaics and cataloged, and immediately before its images and
catalogs would be published. `scripts/release/registration_failsafes.py` runs
there and can refuse the release.

A whole-field average is not enough, and that is known from a specific failure:
the brick field (proposal 1182, filter F356W, July 2026) carried several
arcseconds of misregistration confined to the module seam, while the field
average was about zero. So the check is spatially resolved — a 20×20 grid over
the mosaic, and the question asked separately in each cell.

## What it does inside one cell

Take every bright source detected on the mosaic, and a **truth set** — a second
list of positions that ought to agree with it. Pair each detection with every
truth position within 2.5 arcsec, and histogram those pair separations into
40 × 40 mas bins.

- A star paired with **itself** — seen once in the mosaic, once in the truth set
  — lands at the separation between the two, and every such pair in the cell
  lands at the *same* separation, so they pile into one bin.
- A star paired with an unrelated **neighbour** lands somewhere arbitrary, and
  those spread thinly over the whole search disk.

So the histogram is one peak on a thin floor, and **where the peak sits is the
cell's measured misregistration**. Two numbers come out:

```python
off   = distance of the peak bin from zero, in mas      # the measurement
ratio = H.max() / median(H[H > 0])                       # "is that peak real?"
```

A cell **fails** — and one failing cell fails the whole field, blocking the
release — when *all* of:

```
npair >= MIN_PAIRS (80)  and  ratio >= MIN_PEAK_RATIO (5)     [ "verified" ]
off   >  OFF_MAX (60 mas)
ratio >= FAIL_MIN_RATIO
```

`FAIL_MIN_RATIO` is **10** for the own-catalog check and **5** for the
cross-band one. There are three outcomes, not two: below the `verified` bar a
cell is **unverified** — neither passed nor failed, and not reported as a
problem.

### Which checks actually run

Three truth sets exist in the module, but `--scan`, which is what the release
gate runs, uses **two**:

| check | truth set | runs at the gate? |
|---|---|---|
| cross-band | the same stars detected in another filter | **yes** |
| own-catalog | the mosaic's own vetted source catalog | **yes** |
| per-module | the same band's single-module (`nrca` / `nrcb`) mosaics | only from the single-filter command line, and as a fallback view when a band has no combined mosaic |

None uses an external reference catalogue, deliberately: crowding and extinction
cannot fool a check whose two sides both come from JWST.

**The own-catalog check has a known blind spot**, recorded in `CLAUDE.md` and
worth repeating because it bounds how much the gate is worth: the mosaic and its
own catalog derive from the same calibrated exposures, so a per-visit residual is
self-referential and **cancels** — both are wrong the same way, they agree, and
the cell passes. A green `registration_failsafes` is not sufficient on its own.

## Why `ratio` does not mean what its name says

### Table 1 — the divisor is 1

The 2.5 arcsec search disk contains **12,281** bins of 40 × 40 mas (counting
bins whose centres fall inside it). A grid cell holds tens to hundreds of pairs.
With far more bins than pairs, essentially every occupied bin outside the peak
holds exactly one pair:

| stars in the cell | 15 | 20 | 30 | 45 | 70 | 110 | 170 | 260 | 400 |
|---|---|---|---|---|---|---|---|---|---|
| pairs | 16 | 23 | 37 | 63 | 116 | 226 | 440 | 888 | 1886 |
| `median(H[H>0])` | 1.5 | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** |

(The 15-star cell reads 1.5 because with only 16 pairs the median is taken over
a handful of bins. That cell is far below `MIN_PAIRS` and is never judged; every
density the gate actually looks at reads exactly 1.)

So `ratio` is `H.max() / 1` — **the raw number of pairs in the peak bin**. It is
not a ratio and nothing normalises it. `FAIL_MIN_RATIO = 10` means, literally,
"the peak bin must contain at least ten pairs".

### Table 2 — so the bar is crossed at a star density

One misregistration — 90 mas, **every star in the cell displaced by it** —
measured by the real estimator at different star densities. The seam is
identical in every row; only the number of stars changes. (`off` reads 80 for a
90 mas injection because the 40 mas bin edges fall at …, −20, 20, 60, 100 — the
real gate quantises the same way.)

| stars in the cell | pairs | `off` (mas) | `ratio` | verdict |
|---|---|---|---|---|
| 15 | 16 | 80 | 7 | unverified — not judged at all |
| 20 | 23 | 80 | 10 | unverified — not judged at all |
| 30 | 37 | 80 | 17 | unverified — not judged at all |
| 45 | 63 | 80 | 27 | unverified — not judged at all |
| 70 | 116 | 80 | 41 | **FAIL** |
| 110 | 226 | 80 | 65 | **FAIL** |
| 170 | 440 | 80 | 100 | **FAIL** |
| 260 | 888 | 80 | 153 | **FAIL** |
| 400 | 1886 | 80 | 236 | **FAIL** |

Read down the `ratio` column: the same seam scores **7** in a 15-star cell and
**236** in a 400-star one. Two spans are worth separating, because the sparse end
of the table is outside the regime the rest of this document argues about:

- **41 → 236, a factor of 5.8**, across the rows the release gate actually
  judges (70 stars and up — the sparser rows hold too few pairs to be judged at
  all).
- **7 → 236, a factor of 34**, across the whole table. The 15-star row is the
  only one where the divisor is not 1 (median occupied bin 1.5, peak count 10),
  so its `ratio` of 7 is not a raw count like the others.

Either way the verdict flips from "not judged" to "blocks the release" purely on
how many stars the cell happens to contain.

Two thresholds are doing that, and both are counts:

- `MIN_PAIRS = 80` — below roughly 50 stars per cell the check does not run at
  all, and the cell is silently unverified.
- `FAIL_MIN_RATIO` — above the pair floor, the peak-bin count must reach 10.

The scaling is the expected one and it is worth stating so the numbers are not
mistaken for a fitted curve. Each star contributes exactly one same-star pair, so
the number of them grows in step with the star count. Not all of them land in the
*same* bin: the per-star scatter spills some into the neighbouring ones, so the
peak holds a fixed **fraction** of *n* rather than *n* itself — about 0.59 here
(Table 2: 41/70, 65/110, 236/400), which is what the 15 mas scatter puts inside a
40 mas bin at a 90 mas offset. The fraction is set by the bin geometry and does
not change with density, so the peak still grows in step with *n*.

The divisor does **not** grow with it: it is the median *occupied* bin, and with
a few hundred pairs spread over 12,281 bins the typical occupied bin holds one
pair, so the divisor is pinned at 1 (Table 1) and `ratio` is that fraction of
*n*. Total pairs, meanwhile, grow as *n²* — every star can pair with every other
star inside the search radius. `ratio` ∝ *n* is exact and is all the argument
needs; the tidier-looking `ratio` ∝ √(pairs) is only asymptotic, since
pairs ≈ *n* + 0.0097 *n*² still carries a large same-star share at these
densities — measured over the judged rows `ratio` grows ×5.8 while √pairs grows
×4.03, an exponent of 0.63 rather than 0.5. Density coupling is arithmetic here,
not an empirical finding.

(The *n²* growth is the total pair count, not the ratio's denominator. It becomes
relevant only in the replacement proposed at the end of this document, which
divides by **√lam** — the square root of the expected chance-pair occupancy
`lam = pairs / bins` — instead of by the median occupied bin. That is what makes
the replacement far flatter in density, though not flat; Table 3 measures it.)

And the bar is hardest to clear in the sparsest cells, while the crowded cells
where it is easiest are the Galactic Centre field interiors — where a seam
matters most and where chance-pair confusion is also worst.

## The false alarm that set the current threshold

In July 2026 the own-catalog check failed the brick field's F405N band on seven
cells, each reading an 80 mas offset at `ratio` 5–8. An independent same-star
comparison — matching the individual stars between the two lists and taking the
median displacement, rather than histogramming all pairs — read ≤ 22 mas over
those same regions. So the cells were not misregistered and the failure was
spurious. #166/#172 responded by raising the own-catalog bar from 5 to 10, which
removed those seven, and left the cross-band check at 5.

Those seven cells held 232–323 pairs at `ratio` 5–8 — their peak bins held
**1.7–2.5%** of their pairs (median 2.2%) — no coherent same-star signal.

That is *not*, on its own, what makes them false positives, and an earlier
version of this sentence said it was. #179 measured real injected seam cells
reaching the same level: a seam displaces each cell's existing peak, so a seam's
weakest cells are the field's intrinsically weakest cells, which are these cells.
What establishes that these seven are false is the independent evidence — a
same-star comparison of the same regions reading ≤ 22 mas.

**The threshold sits below what a correctly-registered cell scores.** The best
evidence for that is not this document's model but the gate's own record, at
the comment on `scripts/release/registration_failsafes.py::FAIL_MIN_RATIO`:
*"the clean brick cells verify at median contrast ~18"*. So `FAIL_MIN_RATIO = 10` is under the contrast a **clean** cell
of that field produces — a factor of ~1.8 — which puts the bar in the noise, and
means raising it from 5 to 10 removed those seven cells without ever having
distinguished them from a seam — see Table 4, where #179's *real* injected seams
start at the same `ratio` 5 these cells read. (An earlier version argued this
from Table 2's 15-star row instead, which is both modelled and the one row this
document twice fences off as not a raw count and outside the argued regime.
Issue #170 retracted that comparison; the report had kept it.) The threshold
separated the two
populations on this band by luck of density, not because the statistic
distinguishes them. That is the whole of #170.

**Where the model and the real data disagree, and it is worth saying.** The
seven cells fall *between* Table 2's rows, so the model is re-run at their own
pair counts rather than read off the neighbours: `model_ratio_at_npairs` solves
for the star count that reproduces the observed 232 and 323 pairs and measures
there — realising 231 and 317, which is what is quoted — giving
`ratio` **66–82** — against that recorded ~18 for real clean brick cells, a
factor of about four. Two things to keep in mind about that comparison. It is
approximate in the sampling sense: at 25 trials it lands at 66–82, and across
seeds and trial counts the same computation returns **64–72** at the low end and
**81–86** at the high end, while the fold above ~18 — midpoint over 18, the one
definition used throughout — stays at **4.1–4.3×**. The script sweeps two seeds
against five trial counts and prints exactly those figures on every run; the
numbers quoted here are the ones it prints. Sweeping both knobs matters: at a
fixed trial count the seed-only range reads 66–67, far tighter than the quantity
actually is. And **the ~18
comes with no pair count attached** — the gate's comment does not say how
crowded those clean cells were — so part of a factor of four could be density
rather than normalisation, which is the same conflation this document exists to
warn about. The model assumes every detection has a truth-set counterpart and
that all of them sit in the one cell; real catalogs are incomplete and real
cells are not that clean, so **Table 2's absolute values are upper bounds**.
What the argument rests on is the *scaling* — `ratio` growing linearly with star
count while the thresholds stay fixed — and that is arithmetic, unaffected by
the normalisation. Anyone re-deriving a threshold from this must calibrate on
real cells, not on Table 2.

## What would replace it

Two changes, both still unimplemented:

Both were proposed once before, in PR #179, and that PR was closed with the
objection *"I can't parse any of this. 'density-flat significance' means nothing
to me. I don't know what is contiguous."* So both are named in plain language
first, and the jargon is given afterwards only as a label for what was already
explained:

1. **A confidence number that does not grow just because the cell is crowded.**
   (#179 called this a "density-flat significance": *density* = stars per cell,
   *flat* = the number a correctly-registered cell scores stays put as that
   density rises, instead of climbing with it as the current one does.)
   Not `(H.max() − bg)/√bg` with the same `bg`
   — with `bg` pinned at 1 that is identically `ratio − 1` and adds nothing
   (measured: the seven brick cells go 5,5,5,6,6,6,8 → 4,4,4,5,5,5,7). It has to
   use the *expected* chance-pair background, `lam = npairs / n_disk_bins` —
   the average number of chance pairs a bin would hold if they were spread
   evenly. It is a fraction (a few hundred pairs over 12,281 bins), and it keeps
   scaling with density where the median occupied bin saturates at 1.

   The form is `(peak − lam)/√lam`: subtract what the bin would hold by chance,
   then divide by the **spread** of that chance number rather than by the number
   itself. Counts that arrive independently scatter by about the square root of
   their mean, so √lam is the size of an ordinary fluctuation, and the ratio
   answers "how many ordinary fluctuations is this peak above the background" —
   which is what "how confident are we" should mean here.

   Neither of the two obvious alternatives works. The current statistic divides
   by the median *occupied* bin, which is pinned at 1, so it is the raw peak
   count and climbs in step with density. Dividing by `lam` instead over-corrects
   in the other direction — `lam` grows as *n²* while the peak grows as *n*, so
   `peak/lam` *falls* as the cell gets more crowded (Table 3: 4341 down to 1537
   across the judged rows).

   Dividing by √lam sits between the two. What it does is measured below rather
   than asserted.

### Table 3 — the replacement over the same rows

Printed by the same script. `peak` is the peak-bin count, `lam = npairs / 12,281`
is the *expected* chance occupancy of one bin.

| stars | `peak` | `lam` | `peak/lam` | `(peak − lam)/√lam` |
|---|---|---|---|---|
| 15 | 10 | 0.0013 | 7676 | 277 |
| 20 | 10 | 0.0019 | 5340 | 231 |
| 30 | 17 | 0.0030 | 5643 | 310 |
| 45 | 27 | 0.0051 | 5263 | 377 |
| 70 | 41 | 0.0094 | 4341 | 422 |
| 110 | 65 | 0.0184 | 3532 | 479 |
| 170 | 100 | 0.0358 | 2791 | 528 |
| 260 | 153 | 0.0723 | 2116 | 569 |
| 400 | 236 | 0.1536 | 1537 | 602 |

Over the rows the gate actually judges (70–400 stars):

| | 70 stars → 400 stars | |
|---|---|---|
| raw count (what it uses now) | 41 → 236 | **×5.8** |
| `peak/lam` | 4341 → 1537 | ×0.35 — **falls** with density |
| `(peak − lam)/√lam` | 422 → 602 | **×1.43** |

So the replacement is **much flatter than the raw count, not flat**. That is
enough for the argument — a fixed bar means something comparable at 70 stars and
at 400 — but "a crowded cell and a sparse one score the same", which an earlier
version of this section claimed, is not what the model shows.

Two things this table does *not* establish, stated so it is not read as more than
it is. Every row is a **fully misregistered** cell, so it measures how the
statistic grows with density on a real seam, not the "flat on a *correctly
registered* cell" property the word *flat* names. And ×1.43 rather than ×1.00 is
expected here: `lam = pairs/bins` with `pairs ≈ n + 0.0097 n²` is still
linear-dominated at these densities, which is the same reason the pairs exponent
is 0.63 rather than 0.5.

### Table 4 — does it separate the false alarm from the real seam?

That is the question a threshold has to answer, and it needs both populations.
The seven brick F405N cells that were a **false** failure, against the modelled
seam, under one statistic:

| | npairs | raw `ratio` | `(peak − lam)/√lam` |
|---|---|---|---|
| brick F405N false alarm | 232 | 5 | 36.2 |
| brick F405N false alarm | 241 | 6 | 42.7 |
| brick F405N false alarm | 266 | 5 | 33.8 |
| brick F405N false alarm | 278 | 6 | 39.7 |
| brick F405N false alarm | 287 | 5 | 32.6 |
| brick F405N false alarm | 321 | 8 | **49.3** |
| brick F405N false alarm | 323 | 6 | 36.8 |
| modelled 90 mas seam | 116 | 41 | **421.8** |
| modelled 90 mas seam | 226 | 65 | 479.0 |
| modelled 90 mas seam | 440 | 100 | 528.1 |
| modelled 90 mas seam | 888 | 153 | 568.7 |
| modelled 90 mas seam | 1886 | 236 | 601.8 |

The false alarms are **real measured cells**. The seam rows are **this model**,
and the model's absolute values are upper bounds — the same caveat this document
applies to the ~4× above. Comparing the two and reading off a separation is
exactly the inference an upper bound on one population destroys, so the third
block is the one that settles it:

| | raw `ratio` (min / med / max) | sig (min / med / max) |
|---|---|---|
| #179, real seam injected into the whole field | 5 / 18 / 49 | 32.6 / 78.1 / 140.4 |
| #179, real seam injected into half the field | 5 / 17 / 39 | 32.6 / 75.4 / 115.0 |
| #179, real seam injected into a narrow declination band | 5 / 12 / 27 | 30.1 / 60.7 / 97.0 |

**Measured on real data the two populations OVERLAP at the low end.** Both start
at sig ≈ 32.6, because an injected seam displaces each cell's *existing* peak —
so a seam's weakest cells are the field's intrinsically weakest cells, which are
the artifact cells. #179 put this under a heading of its own and concluded: *"No
amplitude statistic can separate them per-cell… that overlap is exactly why step
3 [contiguity] matters more than step 2."*

So `FAIL_MIN_SIG = 55` is not the midpoint of a clear gap. #179 chose it as the
log-midpoint of the artifact ceiling (49.3) and the hardest real seam's median
(60.7) — **12% headroom**. An earlier version of this section read the modelled
422–602 against the real 32.6–49.3, called it "a factor of 8.6 with no overlap",
and concluded one fixed threshold works at every density. That inverts #179's own
finding, and it inverts the priority between the two proposals above.

**What the amplitude axis does buy** is a *field-level* margin rather than a
per-cell separation: 49.3 → 55 against the raw count's 8 → 10, and it fires on
290 cells where the raw count fires on 273.

**And the raw count is worse than the modelled rows suggest, for exactly the same
reason.** This model puts a seam at those pair counts at 66–82; #179's *real*
seams score `ratio` **5–49, with medians of 12–18** (the first column above).
They start at the false alarms' own 5–8; their medians (12–18) **clear**
`FAIL_MIN_RATIO = 10`, but their sub-median tail does not.

So the bar does not merely sit too low to separate the two populations — it falls
**inside the real seam distribution**, failing weak seam cells and artifact cells
alike. (An earlier version said the medians *straddle* 10. They do not: 18, 17
and 12 all clear it by 1.2–1.8×. What straddles the bar is each trial's full
range, 5 up to 27–49. The distinction matters — as written it implied the
*typical* real seam cell falls under the bar, where in fact only the weak tail
does.) That is the case against the raw count, and it is made on measured data
rather than on this model. (An earlier version of this paragraph compared the
modelled 66–82 against the real 5–8 and said the bar "sits below both", which is
both the upper-bound mistake corrected above and wrong on its face — 10 is
*above* 5–8.)

2. **Requiring the failing cells to touch each other before the field is failed.**
   *Yes, this is proposed as a fix, and it is a second, separate test — a cell
   would have to fail the number test above AND be part of a touching group.*

   Concretely: today one failing cell anywhere on the 20×20 grid fails the whole
   field. Under this change a lone failing cell would not; three or more failing
   cells that are neighbours — sharing an edge, so they form one connected patch
   rather than scattered dots — would. Nothing else changes.

   (#179 called this "contiguity, as an independent second axis". *Contiguity*
   means the cells touch. *Independent second axis* means it is judged separately
   from the confidence number rather than folded into it — two tests, both of
   which must fire. That phrasing is the reason #179 was closed unread, so it is
   given here only as a translation of the plain statement above.)

   A misregistration is a
   connected patch of cells; chance-pair noise is scattered singletons. It needs
   no new measurement — the offsets and the verified flags are already on the
   grid — and in the #179 trial it was much the stronger of the two, firing on
   365 / 179 / 45 cells of three synthetic seams injected into real data (a whole
   field, half a field, and a narrow declination band) against the raw count's
   273 / 127 / 29, and on **zero** cells across the ten brick bands and five cloudc bands
   #179 tested (brick has eleven band directories today and cloudc eight, so
   that sweep did not cover every band even then).

   One measured constraint: the minimum patch has to be **3 cells, not 2**. Two
   of the seven brick false positives are edge-adjacent on the grid, so a 2-cell
   bar re-creates the exact false alarm this is meant to remove.

An attempt at both, PR #179, was closed on 2026-07-29 with a stated reason, and
the reason was that it could not be read:

> I can't parse any of this. "density-flat significance" means nothing to me. I
> don't know what is contiguous. I'm closing this and if there's a real problem
> I'll let an agent resuscitate it.

That is why this document exists, and why both proposals are named in ordinary
words above before their labels are given. An earlier revision of this report
said #179 was "closed without a stated reason", which was wrong — the comment is
on the pull request, timestamped at the close.

Nothing from #179 is on `main` — verified at `2ca3cdc`: `FAIL_MIN_RATIO = 10.0`
is still what `scripts/release/registration_failsafes.py` declares, and neither
`FAIL_MIN_SIG` nor `MIN_SEAM_CELLS` exists anywhere in the tree.

## Loose end

All seven brick F405N cells peak at the **same vector**, +80 mas in RA, to the
bin. Chance pairs would not agree on a direction. That looks more like a small
localized systematic between the F405N mosaic and its own catalog than like
noise — the ≤ 22 mas same-star reading is what makes us call them false
positives, but the two measurements are not obviously measuring the same thing
(mosaic-versus-catalog here, catalog-versus-catalog there). If they *are* real,
both the current threshold and any replacement calibrated against them are
suppressing a genuine 80 mas signal in seven cells. Not chased down.

## Caveats

- Tables 1–3 and the modelled rows of Table 4 run the **real** estimator — the same bin geometry, the same
  `H.max() / median(H[H>0])`, the same peak selection — over **synthetic** star
  fields. That is deliberate: it shows a property of the statistic rather than of
  any one field, and it reproduces without the archive. It is not a measurement
  of any real seam.
- A cell is modelled as *n* detections and their *n* truth counterparts in a
  45-arcsec box, with chance pairs arising on their own from the other truth
  stars inside the search radius. An earlier version of this document instead
  modelled a cell that a seam only *clips* as "a few displaced pairs plus uniform
  noise", which silently deleted the correctly-registered stars in the rest of
  that cell. With those present the peak sits at **zero** and the cell is never
  even a fail candidate, so that curve described nothing real. It has been
  removed; a partially clipped cell is not the interesting case.
- The 45-arcsec cell size and the 15 mas per-star scatter are representative,
  not fitted. The shape of Table 2 — a count linear in star number, crossing
  fixed bars — does not depend on them.
- `median(H[H>0]) = 1` holds up to a few thousand pairs per cell and breaks
  around 20 000, which no cell in this survey approaches. It has been measured on
  real data once, by #179: *"`median(H[H>0])` is exactly 1 in every verified cell
  of every brick band"*, across **372 verified cells over 10 bands**. That is a
  measurement of this document's central claim, from the same trial cited above
  for the contiguity counts. What is missing is a standing record: the scan
  results under `registration_scan_results/` do not store `npairs` or `ratio` for
  passing cells, so the check cannot be repeated from what is on disk today, and
  the only per-cell pair counts kept anywhere are the seven brick cells' 232–323.
