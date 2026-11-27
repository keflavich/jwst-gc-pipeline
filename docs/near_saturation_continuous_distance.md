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

## Limits of this change

* The rule is a different compromise, not a more accurate one: a fit 1.2 px from a
  saturated pixel is no better measured because it lies on an axis.  What it removes is
  the dependence of the decision on which side of a half-pixel boundary the fit lands.
* Fits kept at 1.0 to 1.5 px on-axis come from the wings of a saturated star and can
  carry biased fluxes; the effect on the catalog is counted in the tile test of the
  pull request description.
