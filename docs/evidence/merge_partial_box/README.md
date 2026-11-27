# Merge: leave a partial-fit-box row out of the average of a star that a full-box row measured

`combine_singleframe` (`merge_catalogs.py::combine_singleframe`) averages the
per-frame rows of a star with the weights 1 / `flux_err`^2.  A per-frame fit
that used few of the 25 pixels of its 5x5 fit box comes back with a smaller
`flux_err` than a full-box row of the same star whenever its flux is smaller,
because the flux error grows with the flux.  The inverse-variance mean then
follows the partial row.  With two rows the MAD sigma clip cannot reject
either of them.  This change adds an opt-in switch that leaves such rows out.

| environment variable | default | effect |
|---|---|---|
| `MERGE_PARTIAL_BOX_FRAC_MIN` | unset (off) | a row whose `n_pixels_fit` is below this fraction of the fit box is left out of the flux and position averages of its star, when the star has a kept row at or above the fraction.  Range (0, 1]; 0 or blank is off; other values raise `ValueError` naming the variable |
| `MERGE_PARTIAL_BOX_ZMAX` | unset | only with the first variable: leave out only the partial rows whose flux differs from the inverse-variance mean of the star's rows at or above the fraction by more than this many standard deviations of the difference |

With the switch on, the merged catalogue carries the column
`n_partial_box_dropped` (rows left out of each star) and the header keyword
`PBOXDROP` (their total).  With it off, the output is unchanged
(`tests/test_combine_singleframe_partial_box.py` pins this).  The code is in
`merge_partial_box.py`; the hook in `combine_singleframe` sits between the
sigma clip and the weighted averages, so the position average, the flux
average, the flux error propagation and every column that shares the flux
weights see the same rows.  A row without a finite positive flux error or
without `n_pixels_fit` is never left out and never serves as the full-box
row, a star whose kept rows are all below the fraction is averaged as before,
and `nmatch_good` still counts every kept row.

## The star that motivated it

gc-treasury 10678 o130 F212N, mosaic pixel (5065, 68).  Two per-frame rows
enter its merge:

| row | `flux_fit` | `flux_err` | `qfit` | `n_pixels_fit` | offset from the full-box row |
|---|---|---|---|---|---|
| full box | 14433.84 | 64.715 | 0.0126 | 25 | 0 |
| partial box | 1295.42 | 31.101 | 0.194 | 10 | 0.093 arcsec (2.96 pixels) |

The partial row has `flags` 1, reduced chi^2 71.8 and a forced refit
(`group_size` 2).  Its weight is 4.3 times that of the full-box row, so the
inverse-variance mean is 3760.5 (the production merged flux is 3761; the
vetted catalogue of the previous merge step lists the full-box row alone,
14335).  The position average is not weighted, so the partial row moves the
merged position by half of its offset, 0.046 arcsec (1.5 pixels).  The
residual of the merged-catalogue render at this star is 1241 MJy/sr, which is
a leftover.  With `MERGE_PARTIAL_BOX_FRAC_MIN=0.5` the 10-pixel row is below
12.5 pixels and the merged row is the full-box row: flux 14433.84, error
64.715, position of the full-box row.

Lowering the weight of the partial row in proportion to its pixels does not
repair this.  Weighting by `n_pixels_fit` / 25 = 0.4 gives a merged flux of
6105; the merged flux is within 10% of the full-box flux only when the weight is
at most 0.029 times its inverse-variance weight.  Scaling every row's error by
the square root of its reduced chi^2 would change the weights of all rows of
all stars, including the rows of full-box fits.

## Fit box accounting

`n_pixels_fit` counts the usable pixels of the 5x5 fit box (`fit_shape=(5, 5)`),
25 for a full box.  Rows of the per-frame m7 tables of four tiles (o105, o118,
o130, o138 of 10678; the 48 F212N and 12 F480M exposures of each):

| band | rows per tile | rows at 25 pixels | rows below 25 pixels | rows with 12 pixels or fewer |
|---|---|---|---|---|
| F212N | 0.84e6 - 1.82e6 | 81.1 - 81.4% | 18.6 - 18.9% | 0.17 - 0.20% |
| F480M | 0.51e6 - 0.57e6 | 60.9 - 61.3% | 38.7 - 39.1% | 0.21 - 0.23% |

A threshold of 0.5 of the box is 12.5 pixels, so it applies to the 0.2% of
rows with 12 pixels or fewer.  Rows with 13 to 24 pixels (18.4 - 18.7% of F212N rows,
38.5 - 38.9% of F480M rows) keep their weight.

## How the partial rows compare with full-box rows of the same star

For every star with at least one full-box row, each partial row is compared
with the inverse-variance mean of the full-box rows (after the sigma clip of
the merge): z = (row - mean) / sqrt(row error^2 + mean error^2).  The same
quantity for a full-box row against the mean of the other full-box rows of its
star (leave-one-out) is the baseline for how well the quoted errors describe the
scatter.  Robust standard deviation of z and fraction of rows with |z| > 3,
o130 F212N:

| rows | S/N | rows compared | robust sd of z | fraction with abs(z) > 3 |
|---|---|---|---|---|
| 12 pixels or fewer | all | 1047 | 1.09 | 7.8% |
| 12 pixels or fewer | 50 or more | 95 | 1.79 | 26.3% |
| 13 - 16 pixels | 50 or more | 156 | 1.51 | 21.2% |
| 17 - 19 pixels | 50 or more | 132 | 1.16 | 7.6% |
| 24 pixels | 50 or more | 13176 | 1.18 | 2.5% |
| full box, leave-one-out | 50 or more | 83132 | 1.17 | 2.4% |

The o105 F212N values are 1.86 and 27.8% (12 pixels or fewer, S/N 50 or more,
291 rows), 1.67 and 20.9% (13 - 16 pixels, 478 rows) and 1.26 and 3.7% for the
leave-one-out baseline (275353 rows).  Below S/N 50 the partial rows of 16
pixels or fewer have a robust sd of z of 0.92 - 1.28, within 0.12 of the
leave-one-out baseline of the same S/N range (1.09 and 1.16 for S/N 10 - 50 in
o130 and o105).  At S/N 50 or more the partial rows of 16 pixels or fewer are
2 - 4% fainter than the full-box mean in the median (relative median -0.018
to -0.037 in o105 and o130), and 21 - 28% of them lie more than 3 sigma away.  The 17 - 19 pixel bin
agrees with the baseline.  The same measurement for rows of 20 pixels in o130
(a column or row lost from the box) gives 1.72 and 13.4% at S/N 50 or more, so
the 0.5 threshold does not separate every discrepant row from the others.

## Stars changed, per tile

Merging the per-frame rows of each tile again with the rule on and comparing
with the merge as it ran (before any rule is applied, the flux recomputed from
the per-frame rows differs from the merged catalogue flux by at most a
relative 2.3e-7 - 2.6e-7 in every tile and band).  Columns: stars with a row
left out; stars whose merged flux changes by more than 1%, 10%, 50%; of
those with more than 10%, how many go up and down; stars whose position moves
by more than 1 pixel.

`MERGE_PARTIAL_BOX_FRAC_MIN=0.5`:

| tile | band | stars | stars with a row left out | > 1% | > 10% (up / down) | > 50% | position > 1 pixel |
|---|---|---|---|---|---|---|---|
| o105 | F212N | 573410 | 2523 | 1330 | 191 (96 / 95) | 18 | 12 |
| o118 | F212N | 321810 | 1386 | 761 | 103 (51 / 52) | 15 | 13 |
| o130 | F212N | 275067 | 1205 | 664 | 113 (58 / 55) | 23 | 20 |
| o138 | F212N | 502636 | 2115 | 1120 | 144 (66 / 78) | 17 | 15 |
| o105 | F480M | 160868 | 726 | 415 | 82 (22 / 60) | 9 | 0 |
| o118 | F480M | 159675 | 775 | 404 | 79 (23 / 56) | 9 | 0 |
| o130 | F480M | 161012 | 747 | 415 | 59 (18 / 41) | 5 | 0 |
| o138 | F480M | 179340 | 892 | 471 | 88 (25 / 63) | 6 | 0 |

`MERGE_PARTIAL_BOX_FRAC_MIN=0.5` with `MERGE_PARTIAL_BOX_ZMAX=3`:

| tile | band | stars with a row left out | > 1% | > 10% (up / down) | > 50% | position > 1 pixel |
|---|---|---|---|---|---|---|
| o105 | F212N | 268 | 250 | 90 (41 / 49) | 14 | 8 |
| o118 | F212N | 107 | 99 | 46 (25 / 21) | 10 | 3 |
| o130 | F212N | 103 | 99 | 48 (30 / 18) | 15 | 8 |
| o138 | F212N | 215 | 203 | 72 (34 / 38) | 10 | 7 |
| o105 | F480M | 433 | 342 | 80 (20 / 60) | 9 | 0 |
| o118 | F480M | 309 | 262 | 64 (17 / 47) | 8 | 0 |
| o130 | F480M | 272 | 227 | 47 (15 / 32) | 5 | 0 |
| o138 | F480M | 379 | 323 | 79 (22 / 57) | 5 | 0 |

The star of the previous section is among the F212N stars of both tables
(the first table needs only the pixel count, the second also needs
z = (1295.42 - 14433.84) / sqrt(31.101^2 + 64.715^2) = -183).

## Whether the changed fluxes and positions follow the data

For isolated stars that the vetted m7 catalogue lists, two checks against the
data mosaic of the unmasked render of the tile:

- flux: the aperture flux of the data is the catalogue flux plus the residual
  summed over a 7x7 box (11x11 for F480M) divided by the enclosed fraction of
  the box (0.71 for F212N); a rule counts when it moves the flux by more than
  2 sigma of the aperture noise, and improves it when the new flux is closer
  to the aperture flux;
- position: the iterative flux-weighted centroid of the data (5x5 window for
  F212N, 9x9 for F480M, three iterations, annulus background removed) of stars
  with peak / background sigma of 30 or more; a rule counts when it moves the
  position by more than 0.3 pixel, and improves it when the new position is
  closer to the data centroid.

Pooled over the four F212N tiles:

| rule | flux changes counted | flux closer to the data | position changes counted | position closer to the data |
|---|---|---|---|---|
| `FRAC_MIN=0.5` | 21 | 17 | 44 | 30 |
| `FRAC_MIN=0.5`, `ZMAX=3` | 20 | 16 | 11 | 9 |

The median distance of the changed positions to the data centroid goes from
0.33 - 0.59 pixel to 0.13 - 0.22 pixel with `FRAC_MIN=0.5` (samples of 5, 12,
16 and 11 stars in o105, o118, o130, o138) and from 0.57 - 0.75 pixel to
0.07 - 0.10 pixel with `ZMAX=3` (samples of 3, 5 and 3 stars in o118, o130,
o138; o105 has none).  The samples are small because the checks need isolated stars with
untouched apertures (63, 56, 85 and 64 stars of o105, o118, o130, o138 pass
the selection).  F480M has 12 - 21 selected stars per tile and 0 - 2 changed
positions per rule, too few to say anything.  The tile test below measures the
effect on the residual images.

## Rule choice

The threshold 0.5 is half of the fit box: a row with 12 pixels or fewer has
fitted its three parameters (flux and two positions) with fewer than half of
its pixels.  The support checks above do not separate the thresholds 0.4, 0.5
and 0.6: in o130 F212N the rules 0.5 and 0.6 change the same sample stars
(flux closer to the data for 9 of 12, position for 12 of 16), and 0.4 changes
2 and 5 of them, all closer.  The threshold 0.6 changes 1392 stars of o130
F212N against 1205 for 0.5, and the agreement table shows that rows of 13 - 16
pixels disagree with the full-box rows at S/N 50 or more almost as much as the
rows of 12 pixels or fewer, so a value between 0.5 and 0.6 may be better.  At
0.7 the support check turns negative: with `ZMAX=3` the three position changes
of the o105 F212N sample all move away from the data centroid (median
distance 1.4 -> 2.2 pixel) and 7 of 13 flux changes of o138 F212N move away
from the aperture flux.  `ZMAX=3` reduces the number of stars changed by a
factor of 9 - 13 in F212N (268 against 2523 in o105, 103 against 1205 in
o130) and keeps the stars whose partial row disagrees with the full-box rows.
Both settings are environment variables, so a different value needs no code
change.
