# Near-saturation filter on the unrounded fitted position (`NEARSAT_CONTINUOUS_DIST`)

Opt-in, default off.  With the switch on, the daophot-channel near-saturation filter
measures the distance from the fitted position itself to the nearest saturated pixel,
where the default measures it from the pixel the fitted position rounds to.  Two fits
that differ by 0.001 px then no longer get opposite decisions.

## What the filter does

`crowdsource_catalogs_long.py::_filter_near_saturation` runs at the end of every
daophot-channel pass (`cataloging.py::_manual_phot_pass`).  It drops a fit whose centre is
within `max_sat_dist_pix` of a SATURATED-DQ pixel: a fit placed on or next to a saturated
core is driven by the unsaturated wings, and its flux is unreliable.  For NIRCam the
threshold is 1.0 px (`near_sat_dist_pix` in `cataloging.py`); for MIRI it is 1.5 FWHM.
The protect circles of the satstar hand-off and the seeded-companion exemption
(`crowdsource_catalogs_long.py::_seeded_companion_mask`) act after the distance test.

The default rule reads `scipy.ndimage.distance_transform_edt(~saturated)` at
`(np.rint(x_fit), np.rint(y_fit))`.  `np.rint` rounds half to even, and the lookup is a
step function of the position: a pixel edge-adjacent to a saturated pixel has distance
1.0 (dropped, `<=`), a diagonal one 1.414 (kept), and the decision changes where the
fitted position crosses a half-pixel boundary.

## The two rules around one saturated pixel

For an isolated saturated pixel at the origin (pixel centres on integers):

| rule | dropped region | area | farthest dropped point along an axis |
|---|---|---|---|
| rounded pixel (default) | the pixel and its four edge neighbours, each a 1 x 1 cell | 5 px^2 | 1.5 px |
| continuous, threshold 1.0 | disc of radius 1.0 | 3.14 px^2 | 1.0 px |

So the continuous rule keeps a fit 1.0 to 1.5 px from the saturated centre along a pixel
axis, which the default drops, and drops a fit inside the disc on the diagonal (for
example 0.6 px from the centre in both axes, distance 0.85), which the default keeps
because it rounds to a diagonal cell at 1.414.  Next to a blob of saturated pixels the
two regions are unions of these, and the rules differ only at the boundary of the blob's
neighbourhood.

The threshold stays 1.0.  The equal-area disc has radius 1.2616 (the square root of
5 / pi); at that radius the F4 fits below sit 0.002 px (1.2634) and 0.012 px (1.2731) from
the boundary, so a fit that moves by a few thousandths of a pixel would again change
channel; at 1.0 they are 0.26 px away.  A threshold of 1.5 drops both F4 fits.

## The two stars that motivated it

Both are o105 F480M stars that the end-to-end run of the integration branch (every
opt-in switch off) loses and the production catalog keeps, and both leave a leftover (a
star that stays bright in the residual render).

F4, `nrcalong` exposure 1, saturated pixel (1834, 1632):

| run | x_fit | y_fit | rounds to | rounded-rule distance | continuous distance | kept by default | kept by continuous |
|---|---|---|---|---|---|---|---|
| production | 1834.500295 | 1633.160070 | (1835, 1633) | 1.414 | 1.2634 | yes | yes |
| integration branch | 1834.499118 | 1633.171163 | (1834, 1633) | 1.000 | 1.2731 | no | yes |

The two fits differ by 0.0012 px in x and 0.011 px in y, are 1.26 and 1.27 px from the
same saturated pixel, and fall on opposite sides of `x = 1834.5`.  The production row
(`flux_fit` 13,850, flags 1) is lost in the integration run, and the 904 MJy/sr leftover
at pixel (2614, 1826) of the render appears.

F5, `nrcblong` exposure 6, fit next to the saturated blob at x = 1266 to 1270,
y = 630 to 636:

| run | x_fit | y_fit | rounds to | rounded-rule distance | continuous distance | kept by default | kept by continuous |
|---|---|---|---|---|---|---|---|
| production | 1264.4819 | 634.4405 | (1264, 634) | 2.000 | 1.5807 | yes | yes |
| integration branch | 1264.542203 | 634.433635 | (1265, 634) | 1.000 | 1.5209 | no | yes |
| replay with the gate switch | 1264.510691 | 634.433584 | (1265, 634) | 1.000 | 1.5511 | no | yes |

There the shift of 0.060 px comes from the group fit when the neighbouring star is
handed to the daophot channel by the implied-peak gate (`SATSTAR_GATE_EPSF_CORE_SCALE`,
its own switch), and it carries the fit across the half-pixel boundary at 1264.5.  With the
gate switch the star stays in the satstar channel and the row moves by 0.029 px, which is
still across the boundary.  With the continuous rule the fit is 1.52 or 1.55 px from the
blob and is kept whichever channel the neighbouring star is in.  The leftover at pixel
(4452, 858) of the render is that row.

## The switch

* `NEARSAT_CONTINUOUS_DIST=1` (`cataloging.py::_near_sat_continuous_enabled`; accepts
  1/true/yes/on and 0/false/no/off like the other hand-off switches, raises
  `ValueError` naming the variable for another value) passes
  `continuous_dist=True` to `_filter_near_saturation` through
  `_manual_phot_pass(near_sat_continuous=...)`.
* NIRCam only.  For MIRI the threshold is 1.5 FWHM, a distance at which the half-pixel
  rounding is a small part of the radius, and the switch has no effect (the log line
  of the switch is not printed).
* `_filter_near_saturation(continuous_dist=False)` is the default of the function, so
  every other caller keeps the rounded rule: the callers in `legacy/crowdsource_step.py`
  and the tests of the rounded rule run unchanged.
* The continuous distance is the Euclidean distance to the nearest SATURATED pixel centre
  (a `cKDTree` over the saturated pixels, built once per call).  Rows outside the frame or
  with a non-finite position get an infinite distance and are never dropped, as before.
* The same distance is used for every use of the distance inside the call: the final
  table, the per-iteration snapshots of the fit object (`fit_results`), and the
  seeded-companion exemption, which asks for the distance to the nearest saturated
  pixel that the frame still treats as lost.
* Log line, once per pass (printed when either rule would drop a fit):
  `Saturation-proximity filter (<label>): continuous distance <= 1.0 px of a SATURATED
  pixel centre: N fits (rounded-pixel rule: M; kept only by the continuous rule A,
  dropped only by it B)`.  The usual `dropping N fits ...` line gains the words
  `(continuous distance)` when the rule is on.

## Tests

`jwst_gc_pipeline/photometry/tests/test_near_sat_continuous_dist.py`:

* off (the default), the filter keeps and drops the rows of the rounded-pixel rule, row
  for row, on random positions around four saturated discs, including exact half-pixels,
  NaN positions and positions outside the frame, in the final table and in the snapshots;
* on, the two F4 rows are both kept (`x_fit` 1834.500295 and 1834.499118, 1.2634 and
  1.2731 px from the saturated pixel (1834, 1632); the rounded rule keeps the first and
  drops the second) and the two F5 rows are both kept (1.5807 and 1.5209 px);
* the threshold is the distance itself (`<=`), on axis and on the diagonal, and a
  threshold of 1.5 or 0.5 moves the boundary as the table above says;
* fits 0.0025 px apart across a half-pixel boundary get one decision with the switch on,
  and two with it off;
* the per-iteration snapshots, the seeded-companion exemption (kept next to a restored
  pixel, and by the distance to the nearest lost pixel) and the case where every
  saturated pixel was restored follow the same distance; NaN, off-frame and empty-mask
  inputs are never dropped;
* the switch is off unless set, is ignored for MIRI, raises on an unreadable value, and
  `_manual_phot_pass` hands it to the filter: a synthetic star 1.24 px from a saturated
  pixel is dropped with the switch off and kept, at its true position, with it on.

## One-frame replays

The m7 pass of two o105 F480M frames was replayed with the integration code plus the two
opt-in switches (ePSF core on, every other setting that of the end-to-end control run), with
four arms: no switch (control), `SATSTAR_GATE_EPSF_CORE_SCALE` alone, this switch alone and
both.  The rows are those of the final daophot table, after the near-saturation filter,
within 3 px of the position.

Frame `nrcblong` exposure 6: the neighbour at (1264.5, 634.4) of the F5 star (1268.97, 632.68).

| arm | neighbour, daophot row (x_fit, flux, flags) | star |
|---|---|---|
| production (STPSF only) | 1264.4819, 5232.8, 1 | satstar row, flux 29125.9 |
| control | none | daophot row, flux 23597.7; satstar row rejected by the gate |
| gate switch | none | satstar row, flux 28636.7 |
| this switch | 1264.5422, 5360.7, 0 | daophot row, flux 23597.7; satstar row rejected by the gate |
| both | 1264.5107, 5291.3, 1 | satstar row, flux 28636.7 |

Frame `nrcalong` exposure 1: the F4 row at (1834.5, 1633.2).

| arm | daophot row (x_fit, y_fit, flux, flags) |
|---|---|
| production | 1834.5003, 1633.1601, 13850.3, 1 |
| control | none |
| gate switch | none |
| this switch | 1834.4991, 1633.1712, 13502.2, 1 |
| both | 1834.4991, 1633.1712, 13502.2, 1 |

This switch alone returns both rows, and the gate switch returns neither.  The replay F4
row has `flux_fit` 13502.2 against the production 13850.3 (2.5 % lower), and the F5
neighbour 5360.7 (this switch alone) or 5291.3 (both) against 5232.8.

The census line of the filter, and the size of the change, in the same frames.  "Flagged" is
the number of fits within the distance before the exemptions (protect circles, seeded
companions); "dropped" is the number of the usual log line, after them.

| frame | arm | flagged, continuous rule | flagged, rounded rule | kept only by the continuous rule | dropped only by it | dropped (control) | daophot rows (control) |
|---|---|---|---|---|---|---|---|
| `nrcalong` exposure 1 | this switch | 376 | 414 | 39 | 1 | 142 (178) | 54086 (54060) |
| `nrcalong` exposure 1 | both | 368 | 405 | 38 | 1 | 141 (178) | 54073 (54060) |
| `nrcblong` exposure 6 | this switch | 663 | 755 | 95 | 3 | 276 (366) | 53057 (52982) |
| `nrcblong` exposure 6 | both | 657 | 750 | 96 | 3 | 276 (366) | 53050 (52982) |

The rule therefore keeps about 40 to 100 more fits per frame, of about 54,000 rows, and
drops 1 to 3 that the rounded rule keeps.  The control arm reproduces the daophot table of
the earlier control replay of each frame (54060 and 52982 rows; `x_fit`, `y_fit`,
`flux_fit`, `flags` and `qfit` identical in every row), so a run without the switches is
unchanged.

## Tile test (o105)

The o105 tile was rebuilt end to end (stage, 8-shard m7 fan-out over both bands, finalize,
scoring) with the integration code of the end-to-end tests plus the two switches, in the
environment of the end-to-end control (production recipe, ePSF core on).  One arm set
`NEARSAT_CONTINUOUS_DIST=1`; a second arm also set `SATSTAR_GATE_EPSF_CORE_SCALE=1` ("both
switches").  The control column is the control render of the end-to-end tests, built earlier
in another worktree of the same code; no control arm was rebuilt in the test worktree.  The production column is the production residual image,
scored with the same tools as the other renders.

Counts are by peak or depth in MJy/sr.  A leftover is a residual with residual / data > 0.7
and a peak of at least the stated value; a big crater is a negative residual deeper than
50 MJy/sr whose depth exceeds 0.3 times the model peak.

| band | render | leftovers >= 50 | >= 100 | >= 200 | >= 500 | >= 1000 | craters >= 50 | >= 200 | >= 500 |
|---|---|---|---|---|---|---|---|---|---|
| F480M | production | 303 | 202 | 168 | 111 | 84 | 69 | 28 | 16 |
| F480M | control | 297 | 200 | 168 | 107 | 77 | 92 | 38 | 17 |
| F480M | fit-time switches, unmasked | 221 | 155 | 146 | 104 | 78 | 20 | 12 | 8 |
| F480M | fit-time switches, masked | 49 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| F480M | this switch | 277 | 180 | 150 | 103 | 76 | 94 | 39 | 17 |
| F480M | both switches | 277 | 180 | 150 | 103 | 76 | 95 | 40 | 18 |
| F212N | production | 845 | 317 | 178 | 133 | 116 | 489 | 166 | 71 |
| F212N | control | 830 | 315 | 176 | 134 | 117 | 454 | 152 | 65 |
| F212N | fit-time switches, unmasked | 610 | 118 | 38 | 13 | 8 | 102 | 44 | 18 |
| F212N | fit-time switches, masked | 475 | 52 | 8 | 0 | 0 | 10 | 0 | 0 |
| F212N | this switch | 827 | 312 | 175 | 133 | 116 | 452 | 151 | 64 |
| F212N | both switches | 827 | 312 | 175 | 133 | 116 | 452 | 151 | 64 |

The two further renders are the end-to-end render with the fit-time switches of the other
open pull requests and no footprint mask ("unmasked") and the same with the footprint masks
("masked"); neither contains this switch.  The four features followed through the
end-to-end tests:

| feature | production | control | fit-time switches, unmasked | fit-time switches, masked | this switch | both switches |
|---|---|---|---|---|---|---|
| F480M leftover (2614, 1826), "F4" | none | 904 | 904 | none | none | none |
| F480M leftover (4452, 858), "F5" | none | 403 | 403 | none | none | none |
| F212N crater (8306, 1270), depth | 978 | 966 | 1590 (ratio 0.32, big) | none | 966 | 966 |
| F212N crater (4469, 1280), depth | 463 | 465 | 755 (ratio 0.30, big) | none | 465 | 465 |

"None" is no entry in the leftover or crater list within 3 px.  The box maxima of F4 and F5
are 33 and 59 MJy/sr with this switch and 33 and 26 with both.

How much of the render the switch changes, against the control.  The F212N residual image
differs in 53,059 of 51,825,305 finite pixels, in 8,246 of them by more than 1 MJy/sr and in
4,274 by more than 5 MJy/sr (41 connected regions after a 3 pixel dilation; the largest,
19,914 MJy/sr, is at (6508, 2524)).  The F480M image differs in 208,601 of 12,464,795 pixels,
in 13,480 by more than 1 and in 4,612 by more than 5 MJy/sr (156 regions).  The consolidated
F480M satstar catalog of the arm has the 1686 rows of the control, none lost or gained, with
a median flux ratio of 1.0000 (16th to 84th percentile 1.0000 to 1.0000; no row differs by
more than 20 per cent); the F212N catalog also has the rows of the control (10,399).  The two
F212N craters followed through the end-to-end tests, at (8306, 1270) and (4469, 1280), have
the same depth with the switch as in the control: 966 and 465 MJy/sr.

F480M, against the control:

* Leftovers >= 100: 200 to 180.  21 are cleared (19 of the 168 at >= 200, 4 of the 107 at
  >= 500, 1 of the 77 at >= 1000; 15 of class SATSTAR_NEAR and 6 of class NOT_IN_M7_CATALOG);
  none is worse by more than a factor 1.5; one is new.  The largest cleared are 1000 at pixel
  (2160, 277), 904 at (2614, 1826) (the F4 feature), 741 at (4134, 1273), 531 at (2837, 911),
  461 at (4622, 988), 432 at (1201, 983), 427 at (3230, 588) and 403 at (4452, 858) (the F5
  feature).  The position-independent scan (13 pixel boxes, threshold 200) finds 15 of the
  168 control features >= 200 below 200 and more than 1.5 times lower in the arm; the box
  maximum of F4 goes from 904 to 33 and that of F5 from 403 to 59.
* The new leftover is at (5325, 1803): 327 MJy/sr against 42 in the control and 45 in
  production (class SATSTAR_NEAR).  The control has a row within 0.2 arcsec of it, at
  (5324.6, 1803.3) with flux 4509.4 and `qfit` 0.014; the arm has none.  The continuous rule
  drops 78 F480M fits that the rounded rule keeps (census below); this row was not traced to
  one of them.
* Craters: 92 to 94 at >= 50 MJy/sr, 38 to 39 at >= 200 and 17 to 17 at >= 500 (four new,
  two gone).  New: (1111, 1885) depth 275 (the 13 x 13 pixel minimum is -263 against -99 in
  the control and in production), (2795, 813) depth 129 (production has 131 there; the
  control has none), (1304, 1780) depth 87 and (3698, 1289) depth 56.  Gone: (3940, 1391)
  depth 94 and (1114, 1883) depth 111, the shallower form of the crater now at (1111, 1885).
  The arm has a row at (1111.1, 1885.2) with flux 3587.7 and `qfit` 0.032 within 0.2 arcsec
  of the new crater; the control has none.

F212N, against the control:

* Leftovers >= 100: 315 to 312.  Cleared: 19,898 MJy/sr at (6508, 2525), 137 at (6514, 2531)
  and 119 at (8912, 3995); none is new.  The 13 x 13 pixel box around (6508, 2525) has a
  maximum of 682 MJy/sr in the arm, 19,910 in the control and 19,994 in production, and a
  minimum of -10,006, -9335 and -8457.  The arm has a row there that the control lacks, at
  (6508.0, 2524.4) with flux 227,329 and `qfit` 0.009.
* Craters: 454 to 452 at >= 50 MJy/sr, 152 to 151 at >= 200 and 65 to 64 at >= 500 (one new,
  three gone).  Gone: (3357, 4052) depth 3683, (3002, 3715) depth 3295 and (3039, 3731)
  depth 121; the 13 x 13 pixel minima at the first two go from -3674 and -3289 to -1998 and
  -1926.  New: (1481, 3449), depth 3017 and ratio 0.320 against the limit 0.3 (13 x 13
  pixel minimum -3008 against -1353 in the control).  The arm has a row at (1481.2, 3449.1)
  with flux 34,401.6 and `qfit` 0.278 that the control lacks.  At (3357, 4052) the arm
  drops the row at (3357.1, 4051.9), flux 40,645.5 and `qfit` 0.219, and has a row at
  (3360.3, 4056.8), flux 23,497.3 and `qfit` 0.356, that the control lacks; the 13 x 13
  pixel minimum of the arm, -1998, is at pixel (3360, 4057), 0.4 px from that row.  At
  (3002, 3715) it drops the row at (3002.2, 3714.5), flux 37,404.7 and `qfit` 0.222.  The
  rows of the saturated stars next to them, at (3360.3, 4053.3) and (3005.3, 3716.2), are
  in both arms.

The census line of the filter, summed over the detector exposures of the tile (one line per
exposure; the arm with both switches in brackets where it differs):

| band | exposures | flagged, continuous rule | flagged, rounded rule | kept only by the continuous rule | dropped only by it |
|---|---|---|---|---|---|
| F480M | 24 | 12,555 (12,427) | 13,818 (13,694) | 1341 (1345) | 78 (78) |
| F212N | 96 | 2985 | 3471 | 518 | 32 |

In F480M the continuous rule keeps about 56 fits per exposure that the rounded rule drops
(1341 over 24 exposures) and drops about 3 per exposure that the rounded rule keeps (78).
The new crater at (1111, 1885) has a new row in the arm and the new leftover at (5325, 1803)
has a missing one; neither row was traced to a census category.

On this tile the switch clears 21 F480M leftovers >= 100 MJy/sr (19 at >= 200) and adds one
of 327 MJy/sr; it adds two F480M craters at >= 50 MJy/sr (one at >= 200, none at >= 500) and
removes two F212N craters at >= 50 MJy/sr (one at >= 200 and one at >= 500).  This is one
tile, and the threshold stayed at 1.0 px: the equal-area radius 1.26 px of the section above
was not run on it.

Evidence files are in `docs/evidence/near_saturation_tile_o105/`.  `o105_near_summary.txt`
holds the counts and the paired changes against the production, control and two other
renders; `o105_near_compare_F480M.txt` and `o105_near_compare_F212N.txt` list every cleared,
reduced, worse and new feature; `o105_near_scan_ctlfin_F480M.txt` and
`o105_near_scan_ctlfin_F212N.txt` hold the position-independent scan against the control (13
pixel boxes, threshold 200 MJy/sr); `o105_near_satcons.txt` compares the consolidated satstar
catalogs; `o105_near_features.txt` gives the features F1, F2, F4 and F5;
`o105_boxes_at_features.txt` gives the residual extrema in the boxes around the features
named above and `o105_rows_at_features.txt` the rows of the merged catalog there.  These
last two files also hold the arm with both switches and the feature at (1597, 1877) of
#1393.  The files label the renders `prod5` (production), `ctlfin` (control), `optnm` and
`optfin` (the fit-time switches of the other open pull requests, unmasked and masked),
`n1xnear` (this switch) and `n1xboth` (this switch and `SATSTAR_GATE_EPSF_CORE_SCALE`, see
#1393).  The survive files named in line 16 of the compare files are scratch files of the
end-to-end run and are not part of the evidence.

## Limits of this change

* The rule is a different compromise, not a more accurate one: a fit 1.2 px from a
  saturated pixel is no better measured because it lies on an axis.  What it removes is
  the dependence of the decision on which side of a half-pixel boundary the fit lands.
* Fits kept at 1.0 to 1.5 px on-axis come from the wings of a saturated star and can
  carry biased fluxes; the tile test above counts the leftovers and craters the rule
  adds and removes.
