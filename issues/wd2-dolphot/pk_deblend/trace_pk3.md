# Where the 327 F277W no-row stars go in pk3 (deblend) vs pk2

Read-only trace of the m7 products of `tree_dbl2` (pk2: peak hand-off) and `tree_dbl3` (pk3: peak hand-off plus `--deblend-satstars`).
Scripts: `trace_pk3.py` (per-frame classification), `analyze_trace.py`, `merge_stage.py`, `consolidated_check.py`, `component_test.py`, `rej_and_figure.py`.
Raw output: `out_*.txt`, `per_frame.ecsv`, `per_star.ecsv`, `per_star_ext.ecsv`.
Zero point 24.130 mag (median of dolphot - (-2.5 log flux) for 18.6-21 mag dolphot stars with a pk2 merged row within 0.05", not satstar-replaced; score_lw.py logic).
Frames: nrcblong exp1-4 cover 322 stars, nrcalong exp1-4 cover 5 stars. Per-frame positions use `skycoord_fit` (satstar catalog and rejected table) and `skycoord_centroid` (daophot).

## 1. Classification per (star, frame), 8 frames, 1308 star-frame pairs
Priority order: 1 satstar row <0.08", 2 satstar row 0.08-0.5", 3 rejected row <0.5", 4 daophot row <0.08", 5 nothing within 0.5".

| class | pk2 | pk3 |
|---|---|---|
| 1 satstar <0.08" | 4 | 118 |
| 2 satstar 0.08-0.5" | 124 | 179 |
| 3 rejected <0.5" (all `fit_quality_gate`) | 72 | 940 |
| 4 daophot <0.08" | 843 | 0 |
| 5 nothing <0.5" | 265 | 71 |

Best class per star over covering frames:

| best class | pk2 stars | pk2 with merged row <0.08" | pk3 stars | pk3 with merged row <0.08" |
|---|---|---|---|---|
| 1 | 3 | 0 | 41 | 28 |
| 2 | 35 | 11 | 55 | 4 |
| 3 | 23 | 20 | 223 | 24 |
| 4 | 228 | 227 | 0 | 0 |
| 5 | 38 | 0 | 8 | 0 |
| total with merged row <0.08" | | 258 | | 56 (27 satstar-replaced) |

Class 4 is empty in pk3 only because class 3 outranks it. Presence flags independent of priority (any covering frame):

| | pk2 | pk3 |
|---|---|---|
| satstar row <0.08" | 3 | 41 |
| rejected row <0.08" | 4 | 270 |
| daophot row <0.08" | 260 | 37 |
| merged row <0.08" | 258 | 56 |

Stars with a pk3 rejected row <0.08" and neither a satstar nor a daophot row there: 222 (merged hit 0).
Stars with none of satstar <0.5", rejected <0.5", daophot <0.08": pk2 38, pk3 8.

### Distributions
- Class 1 magnitude offset (satstar flux_fit -> mag with the zero point, minus dolphot): pk3, 118 star-frames, percentiles 5/16/50/84/95 = -0.42/-0.00/+0.16/+0.35/+0.41 mag; 80 of 118 within |dm|<0.3. pk2 has 4 such rows, median +0.01.
- Class 2 separation: pk3 median 0.39" (25-75%: 0.32-0.44"); pk2 median 0.34" (0.23-0.43").
- Class 3 (pk3): `reject_reason` is `fit_quality_gate` for all 940; separation median 0.05" (5-95%: 0.02-0.20"); rejected flux_fit / dolphot flux percentiles 5/25/50/75/95 = 0.56/0.75/0.85/0.94/1.18; qfit median 29 (5-95%: 8.7-121) for the 169 lost stars with a rejected row <0.08" in exp1. `satstar_implied_peak` and `satstar_observed_peak` are NaN on these rows, so they do not meet the faint-fit-quality hand-off test (implied peak < 0.5 x floor, positive): 0 of 169.
- Class 3 (pk2): 72 star-frames, rejected flux/dolphot median 0.58 and spread 0.04-3.3, so these are mostly off-star or poorly constrained.

## 2. Stars lost in pk3 but recovered in pk2
Merged row <0.08" in pk2 and not in pk3: 219. Gained (pk3 only): 17. Both: 39. Neither: 52.

pk3 products at the 219 lost stars (any frame): satstar row <0.08" 1, satstar row <0.5" 35, rejected row <0.08" 194, rejected <0.5" 207, daophot row <0.08" 1.
pk2 products at the same stars: daophot row <0.08" 219 (all), satstar rows <0.5" 9. The pk2 merged rows are daophot rows (none replaced); their offset to dolphot has percentiles 5/16/50/84/95 = -0.17/-0.10/0.00/+0.13/+0.34 mag.

Nearest pk3 accepted satstar (any of the 8 frames; position from satstar_catalog `skycoord_fit`) to each lost star:

| separation | N | flux_fit / dolphot flux, 5/50/95% |
|---|---|---|
| <0.08" | 1 | 0.85 |
| 0.08-0.2" | 3 | 3.1 / 24 / 25 |
| 0.2-0.5" | 31 | 1.5 / 17 / 347 |
| 0.5-1.0" | 97 | 4.0 / 78 / 536 |
| >1.0" | 87 | 8.2 / 213 / 3237 |

Separation percentiles 5/25/50/75/95 = 0.33/0.62/0.90/1.22/1.79". Flux ratio percentiles 5/16/50/84/95 = 3.2/9.5/82/482/1597.
The nearest accepted satstar is a neighbour in 218 of 219 cases (median 0.9" away, median 82 times brighter than the lost star). One star has an accepted satstar within 0.08" with flux ratio 0.85, i.e. the star itself. No lost star shows a satstar fit of the star at a wrong position. In pk2 the nearest accepted satstar lies a median 3.9" away (only 9 of 219 within 0.5").
The star lies inside the same ~35000 px SATURATED component as many accepted pk3 satstars (section 4).

## 3. Merge stage: per-frame satstar row at the star, no merged row
pk3: 41 stars have a per-frame satstar row <0.08"; 28 have a merged row <0.08". 13 do not. (pk2: 3 stars, 0 merged.)
Chain for the 41 stars: consolidated satstar catalog (`catalogs/f277w_consolidated_satstar_catalog.fits`, 1850 rows in pk3) row <0.08": 27; consolidated row 0.08-0.5": 14.

The 13 unmerged stars: the merged row is the consolidated row (`replaced_saturated` True) at 0.15-0.50" from the star (median 0.29"), with flux 0.9-33 times the dolphot flux (median 8x) and `satstar_nmeas` 6-11 against `satstar_nframes` 4. Four exposures contribute one row each at the star, so the 6-11 measurements include per-frame fits of neighbours merged into the same group. The merged `flux` and position come from the group's kept row, which is the brightest member (`_dedup_satstar_catalog`: brightest-first greedy merge with a radius scaled to the saturated footprint, clamped to 0.8", and a sat_com anchor merge) with the median per-exposure flux. `sat_area` of those consolidated rows is 75-813 px. Two stars (idx 169, 217) have merged/consolidated flux within 10% of dolphot but sit 0.18-0.20" off; the other 11 are neighbours or neighbour-dominated groups. In pk2 the 3 comparable stars also have a consolidated row 0.17-0.19" off with flux ratio 0.87-1.14.
No merged row of these stars is removed by a flag: the nearest merged rows have `satstar_gate_rejected` False; the loss is a position/flux merge inside the satstar consolidation. The merged catalog `satstar_gate_rejected` column is True for 3190 of 42354 pk3 rows, none of them the nearest row of the 13 stars.
Daophot route: 37 stars have a pk3 per-frame daophot row <0.08"; 34 have a merged row (3 do not, not examined further).

## 4. Why daophot has no row at the lost stars (inferred from code and DQ)
From the m7 log, the F277W nrcblong exp1 hand-off keeps 3929 positions and drops 1136 as within 1.5 FWHM of an accepted satstar (pk2: 4166 kept, 899 dropped). The lost stars are 0.9" (median) from the nearest accepted satstar, so that filter can cover 35 or fewer of them.
`component_test.py` labels DQ SATURATED components on each nrcblong frame (3x3 pixels around the star):

| frame group | lost stars (219) | kept in both arms (39) |
|---|---|---|
| star in a SATURATED component >=50 px | 218-219 per exposure | 37 |
| median component area | 34700-35400 px | 350-490 px |
| component holds >=1 accepted satstar, pk2 | 11-30 | 3-5 |
| component holds >=1 accepted satstar, pk3 | 218 of 218 (219) | 18-30 |
| median accepted satstars in component, pk2 / pk3 | 0 / 34-35 | 0 / 0-2 |

All lost stars sit in the single ~35000 px merged component. In pk2 that component holds no accepted satstar; in pk3 deblend accepts about 34 satstars in it. `_handoff_restore_pixels` skips a component holding an accepted satstar centre, so the saturated pixels of the handed-off peaks are not returned to the fit. Inference: the daophot fit at these peaks then has its fit box masked and returns no row. This step was not run; the DQ and accepted-catalog evidence is measured, the causal link is read from the docstrings and code of `cataloging.py` (`_handoff_restore_pixels`, `_unaccepted_sat_component_xy`).
In the same frames the satstar fit does reach the star (pk3 rejected row <0.08" at 194 of 219) and fails `fit_quality_gate` (median qfit 29), and its implied peak is NaN so the fit-quality hand-off does not release it.

## 5. Figure
`cutouts_pk3.png`: 8 lost stars chosen at evenly spaced dolphot F277W (13.9-18.3), nrcblong exp1, 3x3". Columns: pk2 satstar-stage residual, pk3 satstar-stage residual, pk3 satstar model. Red cross dolphot; blue circles pk2 merged rows; orange circles pk3 merged rows; green squares pk3 accepted satstars in exp1. The panels are satstar-stage images, so daophot subtraction does not appear in them.
