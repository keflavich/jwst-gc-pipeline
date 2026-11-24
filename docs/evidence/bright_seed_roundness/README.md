# i2d residual seed: opt-in window for bright detections of any roundness

The i2d-seed daofind keeps `|roundness| ≤ 0.5`, and the opt-in loose window
([../faint_seed_roundness/README.md](../faint_seed_roundness/README.md); 0.8 tested)
adds detections up to that roundness where the annulus prominence is ≥ 5.  Some
bright stars that the m6 fit left mostly in the residual have a roundness beyond
±0.8 (19 of the 80 leftover targets of the probe below).  No window admits such a
star, so it never enters the m7 seed, gets no m7 fit row and stays a bright
leftover in the m7 residual i2d.

This branch adds an **opt-in** window (`--manual-seed-round-bright-max`, default
0 = off; 1.5 tested).  A detection with roundness in (0.8, 1.5] and sharpness in
0.4–1.2 enters the m7 seed when it passes five tests on the detection image (the
m6 residual i2d minus its smoothed background):

| test | flag | default | definition |
|---|---|---|---|
| annulus prominence | `--manual-seed-round-bright-prom-min` | 50 | (core peak, r < 1.5 px − median of the 4–10 px annulus) / annulus MAD |
| unsubtracted fraction | `--manual-seed-round-bright-frac-min` | 0.9 | 3×3 peak of the detection image / 3×3 peak of the data co-add, both after the same background subtraction |
| hole guard | none | 20 px, 4 px | the detection is dropped when an 8-connected masked blob (SCI not finite or WHT ≤ 0) of at least 20 pixels lies within 4 px |
| width | `--manual-seed-round-bright-width-min` | 0.60 | short-axis width over the PSF standard deviation (FWHM / 2.355); 0 = off |
| ERR S/N | `--manual-seed-round-bright-snr-min` | 100 | peak / i2d ERR at the detection |

Admitted detections carry `seed_round_bright` = True.  The column exists only when
the window is on, and with the window off the seed catalog is unchanged.  A run
without an ERR plane, or without the data co-add on the same grid, gets no bright
seeds.  The ERR S/N floor acts as a flux floor and ERR differs between the bands:
100 is tuned on F480M, and F212N needs `--manual-seed-round-bright-snr-min 30`
([S/N floor per band](#sn-floor-per-band)).

## What the evidence covers

Seed counts, position matches against the leftovers of the control m7 residual,
and one-frame replays of the m7 fit.  No full-tile m7 fit, vetting and finalize ran
with the window on, so the effect on the m7 residual i2d is not measured.

`scripts/realfunc.py` runs the committed `_build_i2d_augmented_seed` on the
production m6 residual i2d of each Brick tile (o045, o105, o118, o130, o138), band
(F212N, F480M) and module (nrca, nrcb), with the production m7 i2dseed catalog as
the previous seed.  Pass A has the window off and pass B has it on.  Every count
below is a log line of that function.  The control leftovers come from the m7
residual of the residual-cleanup work with the DQ_SAT dilation and the edge band on;
those runs are not part of this branch.

## Stars the window can reach

A read-only probe (a one-off script that is not part of this branch) ran daofind
on the detection image of each module with the production bounds (|roundness| ≤
0.5, sharpness 0.4–1.2), with the loose roundness bound (±0.8) and with no bound.
The detection threshold is the minimum of the local-noise map, so every local
maximum is a candidate and the bounds alone decide.  The targets are the
NOT_IN_M7_CATALOG leftovers of the control m7 run with a residual peak ≥ 200
MJy/sr: 53 at F480M and 27 at F212N over the five tiles.  Some of them are
single-exposure events or extended objects and not stars.  Each target gets the
best status of the modules that cover it, from the nearest detection within 2.5 px:

| status | meaning | F480M | F212N |
|---|---|---|---|
| TIGHT_OK | a detection passes the production bounds | 20 | 1 |
| LOOSE_ONLY | a detection passes the loose roundness bound and fails the production one | 2 | 0 |
| ROUND | the nearest unbounded detection has \|roundness\| > 0.8 and sharpness in 0.4–1.2 | 9 | 10 |
| SHARP | the nearest unbounded detection has \|roundness\| ≤ 0.8 and sharpness outside 0.4–1.2 | 17 | 4 |
| NO_DET | no detection within 2.5 px | 5 | 12 |
| all targets | | 53 | 27 |

The 19 ROUND targets (9 at F480M, 10 at F212N) are the ones this window can add:
it relaxes the roundness bound and keeps the sharpness bound.  A TIGHT_OK target
already has a detection inside the production window, and the window of this
branch does not address it.  The SHARP and NO_DET targets (22 at F480M, 16 at
F212N) are out of reach of any roundness window.

## Stage counts

Base run, ten modules per band.  The first column counts the detections with
roundness in (0.8, 1.5] and sharpness in 0.4–1.2 over the detection image; each
further column counts the detections left after that test.  The column ERR S/N ≥ 5
is a fixed floor of the function that no candidate of these runs fails; the next
column is the ERR S/N floor of the run (F480M the default of 100, F212N 30).  The
last column counts rows of the output seed catalog that the production seed lacks:
a detection within 0.5 × FWHM of a production row merges with it in the seed step.

**F480M**

| tile | module | roundness in (0.8, 1.5] | prominence >= 50 | fraction >= 0.9 and hole guard | width >= 0.60 | ERR S/N >= 5 | ERR S/N >= 100 | rows new to the production seed |
|---|---|---|---|---|---|---|---|---|
| o045 | nrca | 11603 | 62 | 2 | 2 | 2 | 2 | 2 |
| o045 | nrcb | 11799 | 66 | 2 | 2 | 2 | 2 | 2 |
| o105 | nrca | 11162 | 61 | 4 | 4 | 4 | 4 | 4 |
| o105 | nrcb | 11593 | 87 | 3 | 2 | 2 | 2 | 2 |
| o118 | nrca | 10721 | 46 | 4 | 3 | 3 | 3 | 3 |
| o118 | nrcb | 11249 | 58 | 9 | 7 | 7 | 7 | 7 |
| o130 | nrca | 10449 | 33 | 4 | 3 | 3 | 2 | 2 |
| o130 | nrcb | 10818 | 40 | 5 | 4 | 4 | 2 | 1 |
| o138 | nrca | 10912 | 32 | 0 | 0 | 0 | 0 | 0 |
| o138 | nrcb | 11363 | 36 | 4 | 4 | 4 | 3 | 2 |
| all | | 111669 | 521 | 37 | 31 | 31 | 27 | 25 |

**F212N**

| tile | module | roundness in (0.8, 1.5] | prominence >= 50 | fraction >= 0.9 and hole guard | width >= 0.60 | ERR S/N >= 5 | ERR S/N >= 30 | rows new to the production seed |
|---|---|---|---|---|---|---|---|---|
| o045 | nrca | 58212 | 148 | 18 | 17 | 17 | 16 | 16 |
| o045 | nrcb | 58393 | 160 | 32 | 29 | 29 | 28 | 28 |
| o105 | nrca | 60179 | 144 | 17 | 17 | 17 | 15 | 15 |
| o105 | nrcb | 59346 | 196 | 25 | 24 | 24 | 24 | 24 |
| o118 | nrca | 70491 | 106 | 10 | 9 | 9 | 8 | 8 |
| o118 | nrcb | 69376 | 83 | 4 | 4 | 4 | 3 | 3 |
| o130 | nrca | 71716 | 93 | 6 | 5 | 5 | 5 | 5 |
| o130 | nrcb | 71547 | 90 | 6 | 4 | 4 | 4 | 4 |
| o138 | nrca | 63979 | 103 | 22 | 21 | 21 | 14 | 14 |
| o138 | nrcb | 63891 | 133 | 20 | 19 | 19 | 15 | 15 |
| all | | 647130 | 1256 | 160 | 149 | 149 | 132 | 132 |

With the window off (pass A) the output seed has the same row count as the
production seed in all twenty modules: 0 new rows.

## What each test removes

Rows new to the production seed with one test relaxed (`--frac 0` and `--width 0`
of `scripts/realfunc.py`; the hole guard stays on):

| band, ERR S/N floor | all tests | `--width 0` | `--frac 0` | `--frac 0 --width 0` |
|---|---|---|---|---|
| F480M, 100 | 25 | 29 | 169 | 366 |
| F212N, 30 | 132 | 139 | 434 | 732 |

The fraction test removes 337 of the 366 candidates (92%) that pass the other
tests at F480M and 593 of 732 (81%) at F212N.  The width test removes 4 of 29
(14%) and 7 of 139 (5%) when the fraction test is on, and 197 of 366 (54%) and 298
of 732 (41%) when it is off.  With the fraction test off, 169 (F480M) and 434
(F212N) candidates remain, so the width test does not replace the fraction test.
With the fraction test off, the hole guard alone takes the 521 (F480M) and 1,256
(F212N) detections that pass the prominence test down to 463 and 1,248.

## Hole guard

The rim of a saturated core is unsubtracted too, and it passes the prominence and
fraction tests.  `scripts/holeguard.py` lists the masked blob near a position:

| position | masked blob within 4 px | hole guard |
|---|---|---|
| o138 F480M nrcb (5614, 412) | 156 px, nearest pixel 1.0 px away | drops |
| o130 F480M nrca (2834, 979) | the masked region outside the footprint (6,590,879 px), 2.0 px away | drops |
| o105 F480M nrcb (4426, 509) | 112 px, 1.4 px away | drops |
| o045 F480M nrcb (5332, 2213) | 1 px, 1.4 px away | keeps |
| o045 F480M nrcb (3887, 207), o105 F480M nrcb (4812, 1939), o118 F480M nrcb (5585, 741) | none | keeps |

The three dropped candidates are in the second block of the figure below.  Their
widths (next section) are 0.57, 0.48 and 1.16 PSF standard deviations, so the width
test alone would remove the first two and not the rim at o105 (4426, 509).  The
20-pixel threshold lets the single masked pixel beside o045 (5332, 2213) through.

## Width gate

Thin streaks and single-pixel spikes pass the other tests.  The width is the square
root of the smaller eigenvalue of the covariance of the pixel positions (weighted
by the pixel values) of the 8-connected pixels at 25% of the 3×3 peak or above, in a
15×15 window; it is 0 when that set has two pixels or fewer.  In units of the PSF
standard deviation (FWHM / 2.355, with FWHM from `reduction/fwhm_table.ecsv`:
2.341 px for F212N and 2.574 px for F480M, so σ = 0.994 px and 1.093 px) the default
0.60 is 0.596 px at F212N and 0.656 px at F480M.

**Simulation** (`scripts/widthsim.py`; Gaussian stars of FWHM 2.34 px for F212N and
2.57 px for F480M at 400 random sub-pixel positions each; width over the PSF
standard deviation):

| star | peak S/N | minimum | 5th percentile | median | maximum |
|---|---|---|---|---|---|
| F212N | noiseless | 0.71 | 0.71 | 0.74 | 0.87 |
| F212N | 100 | 0.71 | 0.71 | 0.74 | 0.87 |
| F212N | 30 | 0.69 | 0.71 | 0.76 | 0.88 |
| F480M | noiseless | 0.69 | 0.71 | 0.74 | 0.82 |
| F480M | 100 | 0.66 | 0.71 | 0.73 | 0.82 |
| F480M | 30 | 0.65 | 0.70 | 0.74 | 0.85 |

Noiseless Gaussians narrower than the PSF measure 0.00–0.58 px (FWHM 1.0 px, median
0.00), 0.00–0.58 px (1.4 px, median 0.46) and 0.53–0.67 px (1.8 px, median 0.57).  A
single pixel and a one-pixel streak, horizontal or diagonal, measure 0.0 px.  Equal
pairs of F212N stars 2, 3 and 4 px apart measure 0.77, 0.74 and 0.77 of the PSF
standard deviation, so a blend of two stars passes.

**Candidates** (`scripts/realfunc.py` writes the widths of every detection that
reaches the width test, that is, passes prominence, fraction and the hole guard; the
five Brick tiles, any ERR S/N):

| band | candidates | below 0.60 | range below | at or above 0.60 | range above (median) |
|---|---|---|---|---|---|
| F480M | 37 | 6 | 0.000–0.571 | 31 | 0.701–1.249 (0.830) |
| F212N | 160 | 11 | 0.000–0.582 | 149 | 0.627–1.874 (0.835) |

All 197 widths are finite.  The values next to the cut are 0.494, 0.571 | 0.701,
0.708 (F480M) and 0.564, 0.582 | 0.627, 0.652 (F212N), so the default sits in a gap
of the measured values in both bands.  The counts below 0.55, 0.60, 0.65 and 0.70
are 5, 6, 6 and 6 (F480M) and 8, 11, 12 and 14 (F212N); 0.55 instead of 0.60
changes one seed per band (F212N 133 → 132, F480M 26 → 25).

![Cutouts of bright-window candidates with the width the width test measures](width_cutouts.png)

Each cell has the module data i2d minus the m6 smoothed background on top (gray,
−5 MJy/sr to the 3×3 peak) and the detection image below it (magma, −5 MJy/sr to
1.2 × its 3×3 peak), 21 px across; the title has the tile, band, module, position
(module pixels) and the width in PSF standard deviations (`scripts/widthfig.py`).
The blocks are the 17 candidates the width test removes, the 3 the hole guard
removes, 7 kept stars and the kept candidates with the lowest widths above the cut
(the two lowest of F480M and the four lowest of F212N).

By eye, most of the 17 removed candidates are single pixels, or lines and bars one
or two pixels wide and three or more pixels long, several of them beside a bright
star or a masked hole.  A few are compact blobs about two pixels across: F480M
(3925, 1070) and (3917, 1315), F212N (823, 2066), (7218, 1371) and (2566, 608).
The lowest-width kept candidates (F480M (4663, 1649) and (5332, 2213); F212N
(1921, 1457)) look the same as those blobs.  The kept stars have widths of
0.80–0.87 and PSF-like cores, several with a close neighbour.  The width test
removes the lines, bars and single pixels, and it removes the compact blobs below
0.60 by the position of the cut alone: by this measure a compact source that looks
like a faint star cannot be told from one.

## Where the seeds land

A seed sits on a leftover when a leftover of the control m7 residual lies within
1.5 mosaic pixels of it (a leftover is a star whose residual peak is ≥ 50 MJy/sr
and more than 0.7 of its data peak).  Each seed is matched to the nearest row of
the candidate table of the control run within 1.5 mosaic pixels; a seed with no
leftover sits where the control m7 residual has none, either because its m6 peak is
below 50 MJy/sr or because the control run subtracts it.

**F480M** (25 seeds):

| tile | seeds | matched to a candidate row | on a leftover >= 50 | on a leftover >= 200 | on a probe target (NOT_IN_M7_CATALOG, ≥ 200) | m6 peak < 50 (no leftover) | no leftover and m6 >= 50 |
|---|---|---|---|---|---|---|---|
| o045 | 4 | 4 | 3 | 3 | 3 | 0 | 1 |
| o105 | 6 | 6 | 6 | 4 | 3 | 0 | 0 |
| o118 | 10 | 10 | 10 | 3 | 2 | 0 | 0 |
| o130 | 3 | 3 | 2 | 0 | 0 | 0 | 1 |
| o138 | 2 | 2 | 1 | 0 | 0 | 1 | 0 |
| all | 25 | 25 | 22 | 10 | 8 | 1 | 2 |

Classes of the leftovers that carry a seed: NOT_IN_M7_CATALOG 20 (8 with a residual
peak ≥ 200 MJy/sr), SATSTAR_NEAR 2 (both ≥ 200), none 3.  Every seed:

| tile | module | x, y (module px) | m6 residual peak (MJy/sr) | roundness | ERR S/N | prominence | fraction | nearest satstar row (") | control leftover (MJy/sr) | class of the leftover |
|---|---|---|---|---|---|---|---|---|---|---|
| o045 | nrca | 1049, 755 | 605 | 1.17 | 604 | 87 | 0.98 | 1.01 | 616 | NOT_IN_M7_CATALOG |
| o045 | nrca | 2783, 958 | 436 | 1.35 | 317 | 124 | 1.00 | 5.68 | 444 | NOT_IN_M7_CATALOG |
| o045 | nrcb | 3887, 207 | 198 | 0.81 | 285 | 58 | 0.94 | 3.03 | 201 | NOT_IN_M7_CATALOG |
| o045 | nrcb | 5332, 2213 | 1807 | 0.83 | 262 | 155 | 0.99 | 0.97 | — | — |
| o105 | nrca | 2353, 647 | 125 | 0.91 | 233 | 59 | 1.00 | 6.66 | 127 | NOT_IN_M7_CATALOG |
| o105 | nrca | 866, 788 | 267 | 1.08 | 388 | 71 | 0.95 | 3.02 | 271 | NOT_IN_M7_CATALOG |
| o105 | nrca | 2524, 987 | 368 | 0.85 | 284 | 90 | 1.00 | 5.25 | 373 | NOT_IN_M7_CATALOG |
| o105 | nrca | 40, 1159 | 85 | 1.02 | 133 | 62 | 0.97 | 6.25 | 87 | NOT_IN_M7_CATALOG |
| o105 | nrcb | 5015, 1437 | 360 | 1.35 | 466 | 96 | 1.00 | 3.17 | 367 | NOT_IN_M7_CATALOG |
| o105 | nrcb | 4812, 1939 | 350 | 1.09 | 554 | 123 | 0.99 | 1.02 | 355 | SATSTAR_NEAR |
| o118 | nrca | 2410, 844 | 61 | 0.87 | 163 | 68 | 1.00 | 6.85 | 63 | NOT_IN_M7_CATALOG |
| o118 | nrca | 199, 1288 | 67 | 1.12 | 119 | 57 | 1.00 | 1.21 | 68 | NOT_IN_M7_CATALOG |
| o118 | nrca | 2105, 1989 | 176 | 1.29 | 276 | 113 | 1.00 | 3.10 | 180 | NOT_IN_M7_CATALOG |
| o118 | nrcb | 5585, 741 | 578 | 1.00 | 356 | 137 | 1.00 | 0.69 | 586 | SATSTAR_NEAR |
| o118 | nrcb | 5275, 782 | 61 | 1.16 | 132 | 66 | 1.00 | 4.01 | 61 | NOT_IN_M7_CATALOG |
| o118 | nrcb | 5545, 936 | 117 | 1.00 | 177 | 64 | 0.96 | 1.65 | 119 | NOT_IN_M7_CATALOG |
| o118 | nrcb | 4448, 1449 | 237 | 1.06 | 459 | 124 | 0.96 | 3.30 | 239 | NOT_IN_M7_CATALOG |
| o118 | nrcb | 4809, 1549 | 382 | 1.20 | 593 | 125 | 1.00 | 5.90 | 386 | NOT_IN_M7_CATALOG |
| o118 | nrcb | 4272, 1976 | 51 | 1.17 | 206 | 56 | 0.99 | 4.69 | 52 | NOT_IN_M7_CATALOG |
| o118 | nrcb | 3582, 2179 | 64 | 0.86 | 160 | 71 | 0.95 | 1.92 | 65 | NOT_IN_M7_CATALOG |
| o130 | nrca | 1201, 305 | 297 | 0.99 | 254 | 453 | 1.00 | 6.22 | — | — |
| o130 | nrca | 791, 885 | 50 | 1.07 | 160 | 67 | 0.99 | 9.63 | 51 | NOT_IN_M7_CATALOG |
| o130 | nrcb | 4220, 1427 | 109 | 1.23 | 315 | 78 | 0.94 | 1.98 | 113 | NOT_IN_M7_CATALOG |
| o138 | nrcb | 4223, 735 | 73 | 1.16 | 242 | 52 | 0.95 | 3.36 | 76 | NOT_IN_M7_CATALOG |
| o138 | nrcb | 3256, 956 | 33 | 0.90 | 137 | 51 | 1.00 | 3.29 | — | — |

The three seeds without a leftover are o045 nrcb (5332, 2213), the compact source
in the figure with one masked pixel beside it (m6 residual peak 1,807 MJy/sr, 0.97"
from a saturated-star row); o130 nrca (1201, 305), with a prominence of 453, nearly
three times the next highest of the 25 seeds (155), which I did not inspect; and o138
nrcb (3256, 956) with an m6 residual peak of 33 MJy/sr, below the 50 MJy/sr of a
leftover.

**F212N** (132 seeds):

| tile | seeds | matched to a candidate row | on a leftover >= 50 | on a leftover >= 200 | on a probe target (NOT_IN_M7_CATALOG, ≥ 200) | m6 peak < 50 (no leftover) | no leftover and m6 >= 50 |
|---|---|---|---|---|---|---|---|
| o045 | 44 | 44 | 44 | 4 | 0 | 0 | 0 |
| o105 | 39 | 39 | 38 | 5 | 2 | 0 | 1 |
| o118 | 11 | 11 | 10 | 1 | 1 | 1 | 0 |
| o130 | 9 | 9 | 7 | 3 | 0 | 0 | 2 |
| o138 | 29 | 29 | 29 | 1 | 1 | 0 | 0 |
| all | 132 | 132 | 128 | 14 | 4 | 1 | 3 |

Classes: NOT_IN_M7_CATALOG 80 (4 with a residual peak ≥ 200 MJy/sr), SATSTAR_NEAR 46
(10 of them ≥ 200), VETTED_OUT 2, none 4 (one with a residual peak below 50 MJy/sr,
three above it).  Most F212N seeds sit on leftovers of 50–200 MJy/sr: 14 of the 132
seeds sit on a leftover ≥ 200 MJy/sr, against 10 of 25 at F480M.

## Replays of the m7 fit

One-frame replays of the production m7 command of a frame, with the seed of the
window-on run (treatment) or the production seed (control).  A target is a seed
position mapped into the frame; the table counts targets with a fitted row within
1 px.

| band | frames | targets | row within 1 px, treatment | row within 1 px, control | rows only in the treatment |
|---|---|---|---|---|---|
| F480M o045 | 3 | 4 | 4 | 0 | 4 |
| F212N o045 | 7 | 40 | 38 | 1 | 37 |

Frames: F480M jw10678045001_02101_00001_nrcblong, jw10678045001_02101_00002_nrcalong
and jw10678045001_02101_00004_nrcblong; F212N jw10678045001_02101_NNNNN_DET with
NNNNN_DET = 00002_nrcb4, 00004_nrca3, 00004_nrcb3, 00005_nrca2, 00005_nrcb3,
00006_nrcb1 and 00006_nrcb4.

The seeds of the F480M replay equal the final seeds of o045 (2 rows per module).
The F212N replay used the seed of an earlier run of the same function with the
width test at 0.55; it differs from the final one by a single row, o045 nrcb
(7218, 1371) with a width of 0.56, which the default 0.60 removes.  The two F212N
targets without a row within 1 px are in the frames 00002_nrcb4 and 00006_nrcb4; the
nearest fitted rows are 5.3 px away.  The single control hit is a production row
0.23 px from the target (the treatment row is 0.04 px away).  The fitted
rows lie within 0.35 px (F480M) and 0.47 px (F212N) of their targets, with qfit
0.02–0.29 (median 0.16) and 0.09–0.44 (median 0.31).  The replays show that the new
seeds survive the per-frame fit as rows; they do not measure the residual i2d.

## S/N floor per band

The ERR S/N floor decides how many candidates that pass the other four tests are
kept:

| band | pass ERR S/N ≥ 5 | pass 100 (new rows) | pass 30 (new rows) |
|---|---|---|---|
| F480M | 31 | 27 (25) | 31 (29) |
| F212N | 149 | 3 (3) | 132 (132) |

At F480M the 4 extra seeds of the floor 30 sit on residual peaks of 11–28 MJy/sr
(o130 nrca (2077, 309), o130 nrcb (3186, 869) and (3611, 2051), o138 nrcb
(3070, 695)), none on a leftover ≥ 50 MJy/sr.  At F212N a floor of 100 leaves 3
seeds and a floor of 30 leaves 132, 128 of them on a leftover ≥ 50 MJy/sr.  The
default stays 100, the value validated at F480M; a run on F212N passes
`--manual-seed-round-bright-snr-min 30`.

## Scripts

- `scripts/realfunc.py`: pass A and pass B of the committed function on one tile
  and band; writes the seed catalogs, `o<T>_<band>_<module>_bright.csv` and
  `o<T>_<band>_<module>_widths.csv` under its output directory.  Takes 1–4 minutes
  per tile and band on a compute node (the logs carry the stage counts).
- `scripts/widthsim.py`: the width test on simulated stars, narrow cores, streaks
  and blended pairs (no data access).
- `scripts/holeguard.py`: the masked blob near given positions.
- `scripts/widthfig.py`: the cutout figure above.

## Caveats

- Seed counts and one-frame replays only.  No full-tile m7 fit, vetting and
  finalize ran with the window on; the effect on the residual i2d, on the vetting
  survival of the new rows and on photometry is not measured.
- The window reaches the 19 ROUND targets of the probe at most (9 at F480M, 10 at
  F212N).  Seeds sit on 8 (F480M) and 4 (F212N) NOT_IN_M7_CATALOG leftovers
  ≥ 200 MJy/sr; which test rejects the ROUND targets without a seed was not traced.
  The SHARP and NO_DET targets need a different remedy.
- The thresholds (prominence 50, fraction 0.9, hole guard 20 px and 4 px, width
  0.60) were set on the candidates of one field (Brick, five tiles); other fields
  are unmeasured.
- The width test removes narrow sources only.  An extended object passes it, and
  the o130 nrca (1201, 305) seed (prominence 453) may be one.
- The width test cannot separate a compact source narrower than the PSF from a
  faint star (see the figure); 0.60 and 0.55 differ by one seed per band.
- Seeds without a leftover (F480M 3, F212N 4) can be extended objects or stars that
  the control run already subtracts through a saturated-star fit.  A second row at
  such a star is possible, and whether the post-fit dedup of the m7 fit resolves it
  was not checked.
- The ERR S/N floor needs a per-band value (above).
