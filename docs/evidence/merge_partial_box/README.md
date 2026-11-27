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

With the switch on, the table that `combine_singleframe` returns carries the
column `n_partial_box_dropped` (rows left out of each star) and the header
keyword `PBOXDROP` (their total).  `merge_frames_for_observation` writes its
catalogues from a fixed list of columns, so neither reaches the written raw
or vetted catalogues; the log line `Phase 1: MERGE_PARTIAL_BOX_FRAC_MIN=...:
left N row(s) of M source(s) out of the averages` is the record of a run.
With the switch off, the output is unchanged
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

At S/N 50 or more in the four F212N tiles (o105, o118, o130, o138), the rows
with 12 pixels or fewer have a robust sd of z of 1.86, 2.29, 1.79 and 1.88
with 25.0 - 27.8% beyond 3 sigma, the rows with 13 - 16 pixels have 1.67, 1.73,
1.51 and 1.58 with 16.2 - 21.2%, and the leave-one-out baseline is 1.26, 1.19,
1.17 and 1.20 with 2.4 - 3.7%.  The median of the partial rows of 16 pixels or
fewer is 0.9 - 3.7% below the full-box mean (relative median -0.009 to -0.037).
Below S/N 50 the partial rows of 16 pixels or fewer have a robust sd of z of
0.84 - 1.28, which differs from the leave-one-out baseline of the same S/N
range (1.01 - 1.16) by -0.19 to +0.16.  The 17 - 19 pixel bin of o130 has the
robust sd of the baseline (1.16 against 1.17) and a larger tail (7.6% beyond
3 sigma against 2.4%).  The same measurement for rows of 20 pixels in o130
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
2 and 5 of them, all closer.  The threshold 0.6 (rows of 14 pixels or fewer)
changes 1392 stars of o130 F212N against 1205 for 0.5, and 214 for 0.4.  At S/N
50 or more the rows with 13 - 16 pixels disagree with the full-box rows by less
than the rows with 12 pixels or fewer but more than the baseline (robust sd of
z 1.51 - 1.73 against 1.79 - 2.29 and 1.17 - 1.26), so a threshold above 0.5
is also defensible.  At 0.7 with `ZMAX=3` the checks give mixed results: o130
F212N improves (flux closer to the data for 12 of 17 changes, position closer
for 6 of 6), while in o105 F212N the three position changes of the sample all
move away from the data centroid (median distance 1.4 -> 2.2 pixel) and in
o138 F212N 7 of 13 flux changes move away from the aperture flux.  `ZMAX=3`
reduces the number of stars changed by a factor of 9 - 13 in F212N (268
against 2523 in o105, 103 against 1205 in o130) and keeps the stars whose
partial row disagrees with the full-box rows.  Both settings are environment
variables, so a different value needs no code change.

## Tile test

The finished e2e5 tile trees (`$SCR/e2e5/<tile>`; the masked product is the
`optfin` render, the unmasked re-render is `optnm`) were copied, the m7
finalize (modules nrca, nrcb and merged, both bands) was re-run with the
switch set and nothing else changed, and both residual renders were produced
again.  Arms: `a` = `MERGE_PARTIAL_BOX_FRAC_MIN=0.5` on o130 and o105, `z` =
`MERGE_PARTIAL_BOX_FRAC_MIN=0.5` with `MERGE_PARTIAL_BOX_ZMAX=3` on o130.
The counting is the one of the e2e5 four-tile runs: a leftover is a residual group with
residual / data above 0.7 and a residual peak at or above the stated value; a
crater is a negative residual group with depth above 50 MJy/sr and depth /
model peak above 0.3; a feature of one run and a feature of another are the
same when they lie within 3 pixels (0.094 arcsec).

Leftovers with a residual peak of at least 50 / 100 / 200 / 500 / 1000 MJy/sr
and craters at least 50 / 200 / 500 MJy/sr deep, e2e5 against the arm:

| run, band | render | leftovers e2e5 | leftovers arm | craters e2e5 | craters arm |
|---|---|---|---|---|---|
| o130 `a`, F212N | unmasked | 73 / 16 / 13 / 6 / 3 | 71 / 14 / 11 / 4 / 2 | 56 / 14 / 3 | 56 / 14 / 3 |
| o130 `a`, F212N | masked | 52 / 3 / 1 / 1 / 1 | 51 / 2 / 0 / 0 / 0 | 5 / 0 / 0 | 5 / 0 / 0 |
| o130 `z`, F212N | unmasked | 73 / 16 / 13 / 6 / 3 | 71 / 14 / 11 / 4 / 2 | 56 / 14 / 3 | 56 / 14 / 3 |
| o130 `z`, F212N | masked | 52 / 3 / 1 / 1 / 1 | 51 / 2 / 0 / 0 / 0 | 5 / 0 / 0 | 5 / 0 / 0 |
| o130 `a` and `z`, F480M | unmasked | 85 / 28 / 19 / 14 / 11 | 85 / 28 / 19 / 14 / 11 | 1 / 0 / 0 | 1 / 0 / 0 |
| o130 `a` and `z`, F480M | masked | 56 / 7 / 1 / 0 / 0 | 56 / 7 / 1 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| o105 `a`, F212N | unmasked | 610 / 118 / 38 / 13 / 8 | 609 / 119 / 38 / 13 / 8 | 102 / 44 / 18 | 102 / 44 / 18 |
| o105 `a`, F212N | masked | 475 / 52 / 8 / 0 / 0 | 474 / 52 / 8 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |
| o105 `a`, F480M | unmasked | 221 / 155 / 146 / 104 / 78 | 220 / 154 / 145 / 103 / 78 | 20 / 12 / 8 | 20 / 12 / 8 |
| o105 `a`, F480M | masked | 49 / 2 / 0 / 0 / 0 | 49 / 2 / 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

Each e2e5 render was compared with the arm render of the same kind (12 pairs:
o130 F212N and F480M, masked and unmasked, arms `a` and `z`; o105 F212N and
F480M, masked and unmasked, arm `a`).  In all 12 pairs the number of arm
leftovers at or above 100 MJy/sr without an e2e5 leftover within 3 pixels is
0, the number of e2e5 leftovers at or above 100 MJy/sr that grew by more than
1.5 times is 0, and no crater appears or disappears (crater counts at 50
MJy/sr and above are equal in every pair).  The one total that goes up (o105
F212N unmasked, 118 to 119 leftovers at 100 MJy/sr and above) is an arm
leftover that has an e2e5 leftover within 3 pixels.  A scan of 13 x 13 pixel
boxes at the threshold 200 MJy/sr finds, in all 12 renders, no box where the
arm has a leftover or crater extremum above the threshold while the e2e5
extremum is below the threshold and more than 1.5 times lower.

Leftovers that the switch clears, named by mosaic pixel (x, y) with the
residual peak in the e2e5 render and the arm render:

| star | render | e2e5 | arm |
|---|---|---|---|
| o130 F212N (5065, 68) | masked and unmasked, arms `a` and `z` | 1241 | 0 (largest value 9 in the 13 x 13 pixel box around it) |
| o130 F212N (1700, 4566) | unmasked, arms `a` and `z` | 597 | 0 (largest value 31 in the 13 x 13 pixel box) |
| o130 F212N (5578, 380) | masked and unmasked, arms `a` and `z` | 2250 | 23 in the 13 x 13 pixel box (data peak 3779; model peak 1450 in e2e5, 3726 in the arm) |
| o105 F480M (496, 13) | unmasked, arm `a` | 785 | 0 |

The star at (5578, 380) is not counted as a leftover by the scoring, because
its residual / data is 0.60, below the 0.7 threshold.  The finalize residual
mosaics of o130 F212N differ in 217576 pixels (arm `a`) and 29680 pixels (arm
`z`) by more than 0.001 MJy/sr; the largest differences are (5578, 380)
2250 to -64, (5065, 68) 1244 to -5 and (25, 1514) 110 to -5.  The finalize
mosaics of the other runs differ by at most 73.0 MJy/sr (o130 F480M, 69100
pixels in arm `a`, 35106 in `z`), 83.9 MJy/sr (o105 F212N, 416898 pixels) and
27.8 MJy/sr (o105 F480M, 96156 pixels).

### Rows that changed

Vetted catalogue rows matched on the sky (mutual nearest neighbours within 0.2
arcsec), flux or position changed; fractions are arm flux / e2e5 flux - 1:

| run, band | rows | changed | flux change above 1 % / 10 % / 30 % | up / down (above 1 %) | position change, median / 90th percentile / maximum (arcsec) |
|---|---|---|---|---|---|
| o130 `a`, F212N | 252597 | 1137 | 604 / 76 / 17 | 330 / 274 | 0.0004 / 0.0063 / 0.048 |
| o130 `z`, F212N | 252597 | 90 | 86 / 37 / 13 | 61 / 25 | 0.0018 / 0.0226 / 0.048 |
| o130 `a`, F480M | 140067 | 686 | 378 / 46 / 8 | 145 / 233 | 0.0007 / 0.0102 / 0.048 |
| o130 `z`, F480M | 140067 | 260 | 215 / 39 / 6 | 70 / 145 | 0.0005 / 0.0134 / 0.040 |
| o105 `a`, F212N | 523550 | 2286 | 1166 / 143 / 18 | 540 / 626 | 0.0003 / 0.0053 / 0.046 |
| o105 `a`, F480M | 146308 | 683 | 382 / 70 / 16 | 110 / 272 | 0.0007 / 0.0109 / 0.049 |

In the 5 x 5 pixel cutouts of the finalize residual mosaic (the masked
render) around the rows whose flux changed by more than 10 % (o130 `a` F212N 76
rows, o105 `a` F212N 143 rows), the median of the largest |residual| goes from
7.1 to 6.8 MJy/sr (o130) and from 10.4 to 9.8 MJy/sr (o105), e2e5 to arm; the largest |residual|
shrinks by more than a factor of 2 for 7 rows of o130 and 6 rows of o105 and
grows by more than a factor of 2 for none, and the rows with a cutout
|residual| above 200 MJy/sr go from 2 to 0 (o130) and stay 0 (o105).  The
F480M cutouts of the masked render are zero for the median row (0.0 in both
runs), so they carry no information.

### Vetted membership

A few rows enter or leave the vetted catalogue; every one of them stays in the
raw catalogue.  Rows with no vetted partner within 0.05 arcsec, lost / gained
by the arm: o130 `a` F212N 19 / 4 (of 252597 rows), `z` F212N 2 / 1, `a` F480M
4 / 3 (of 140067), `z` F480M 1 / 0, o105 `a` F212N 33 / 6 (of 523550), `a`
F480M 7 / 1 (of 146308).  Every one of the 66 lost and 15 gained rows has a raw
row of the other run within 0.05 arcsec.  The merged `qfit` of the arm is
higher than the e2e5 value for 63 of the 66 lost rows and 14 of the 15 gained
rows (median |change| 0.03 to 0.14 for the six groups of lost rows), and the merged `flags` of the
arm is lower for 45 of the 66 lost rows: the columns that the vetting reads
are averaged with the same weights as the flux, so they move when a row
leaves the average.  The 19 rows lost from the vetted catalogue of o130 `a`
F212N have e2e5 fluxes of 33 to 1185 MJy/sr.

### A case where the full-box row is uninformative

The largest of those flux changes is the row at 266.756173, -28.367515 (o130
F212N mosaic pixel (7787, 131)), a companion at 0.088 arcsec of a star with a
saturated core (21 of the 49 pixels of the 7 x 7 pixel cutout are NaN in the
data).  In e2e5 its two per-frame rows merge to flux 1185 with `flux_err_prop`
36.7.  In arm `a` the row with a partial box is left out and the merged flux
is 221.5 with `flux_err_prop` 492.9: the one row that remains has at least 13
fitted pixels and an error 13 times larger.  The difference of the e2e5
merged flux and the arm `a` flux is 1.95 combined standard deviations, below
the `ZMAX` of 3, so arm `z` keeps both rows and the merged row is unchanged
(1185, 36.7).  The largest unmasked residual in the 7 x 7 pixel box around the
row is 973 MJy/sr in e2e5, 813 in arm `a` and 812 in arm `z`.  The residual
does not separate the two rules, because the bright neighbour at 0.088 arcsec
changes in the same way in both arms (7875 to 9364 MJy/sr).

### What the test does not show

- The residual counts of the F480M runs do not change and the largest
  difference of the finalize residual mosaic is 73 MJy/sr (o130) and 28
  MJy/sr (o105); the F480M flux changes (62 to 71 % down) are not tested by the
  data here.
- In o105 the leftover and crater counts do not change at any threshold,
  although 2286 vetted F212N rows changed; the switch fixes a few stars and
  leaves the rest as it was.
- One tile per band in o130 and one in o105 was run, and each changed row has
  one residual cutout.  The rule leaves out the rows with 12 pixels or fewer.
  Rows with 13 to 16 pixels also disagree with the full-box rows by more than
  the baseline (robust standard deviation of z 1.51 to 1.73 against 1.17 to
  1.26, see the agreement table above) and stay in the averages.

### Setting to use

`MERGE_PARTIAL_BOX_FRAC_MIN=0.5` with `MERGE_PARTIAL_BOX_ZMAX=3`: in o130 it
gives the same leftover and crater counts as `MERGE_PARTIAL_BOX_FRAC_MIN=0.5`
alone, clears the same named stars, changes 12.6 times fewer vetted F212N rows
(90 against 1137) and 2.6 times fewer F480M rows (260 against 686), loses 2
rather than 19 vetted F212N rows and keeps the row of the case above.
