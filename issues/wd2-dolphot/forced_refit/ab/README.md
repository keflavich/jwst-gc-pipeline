# Forced-refit A/B (PR #1127, #932): why fr1 has more forced rows

Inputs: `../tree_fr0`, `../tree_fr1` merged m7 catalogs (unvetted), 0.04" cross-match, ZP as in `../cmp_fr.py`.
Scripts here: `classify.py` (classes a/b/c, writes `classify_<band>.npz`), `perframe.py`, `logblocks.py`, `reality.py`, `peaksnr.py`, `recall.py`, `fig_fr_cutouts.py`.

## 1. Counts (fr1 rows with forced_refit_frac > 0)

| band | fr0 rows | fr1 rows | fr0 forced | fr1 forced | (a) fr0 forced too | (b) in fr0, not forced | (c) absent from fr0 |
|---|---|---|---|---|---|---|---|
| F150W | 45189 | 45180 | 3718 | 3718 | 3652 | 47 | 19 |
| F187N | 461389 | 465916 | 1377 | 9211 | 1290 | 2570 | 5351 |
| F200W | 356462 | 363702 | 1704 | 13472 | 1601 | 3707 | 8164 |
| F277W | 44027 | 44051 | 9478 | 9533 | 9435 | 33 | 65 |

Of the fr1 rows absent from fr0, 5351 of 5455 (F187N) and 8164 of 8256 (F200W) are forced rows. 930 (F187N) and 1017 (F200W) fr0 rows have no fr1 row within 0.04".
Recall of dolphot stars (row within 0.08") is the same in both arms (e.g. F187N 18.6-21 mag: 0.9765 vs 0.9768).

## Mechanism

The set of rows flagged and force-refit is identical in the two arms. Summed over the 104 per-shard logs, fr0 and fr1 both report 50202 "refit N overshooting sources" (cataloging.py:1084-1132, flag counts per band identical in `logblocks.py` output: F187N 10939 vs 11439 fits flagged in matched frames, with F150W and F277W exactly equal).
The arms differ in what happens after the refit:

* fr0 solves on the raw frame (cataloging.py:1100-1112 skipped), so the 5x5 solve absorbs neighbour and pedestal light. The refit flux is too large (for rows forced in both arms the per-frame flux ratio fr0/fr1 has median 2.8 in F187N, 70% above 2).
* The final phantom drop (cataloging.py:1187-1215, drop when model peak > `manual_overshoot_drop_ratio`=5x the local bkg-subtracted data peak) then removes those inflated rows. The refit clears `model_overshoot` (line 1129), but the drop re-renders the model and re-tests.
* Frame-matched log counts of "dropped N phantom" (`logblocks.py`): F187N fr0 9496 (30 of 32 frames matched) vs fr1 615; F200W 14504 vs 483; F150W 2141 vs 2248; F277W 2502 vs 2371. The non-positive-flux drop (lines 1225-1245) is not the cause (F187N 46471 vs 50853, fr1 higher).
* Per frame (F187N, `perframe.py`): fr1 has 10318 forced rows, fr0 2048; all 1860 fr1 forced rows that fr0 retains are also forced in fr0, and 8458 are absent from fr0 in that frame. Seeds are identical (x_init,y_init match in every frame).

Class (c): every frame in which the source was forced was dropped in fr0, so no fr0 row survives in the merge (median 1 contributing frame, forced fraction 1.0).
Class (b): the source was dropped in the frames where it was forced and survives in other frames as an unforced fit. The merged fr0 row then has forced_refit_frac = 0 and fewer contributing frames (median nmatch 1 vs 2 in fr1 for both bands; flux ratio fr0/fr1 near 1.0-1.1). Class (a): forced and kept in both (flux inflated in fr0).
Net effect: the ~1400/1700 forced rows in fr0 are the survivors of the phantom drop (the brightest, most strongly inflated rows are the ones removed or kept depending on the 5x test), while fr1 keeps the full set of 9-13k refits with corrected fluxes. The forced-row count is therefore the number of refits actually kept, not an increase in refits.

## 2. Are the extra rows real

Pipeline mags use the per-arm ZP. "dol" = fraction with a dolphot star within 0.1". Dolphot is complete to about 21-22 mag (F187N, median 19.2, 90th percentile 21.9) and about 23 (F200W, 90th percentile 24.1).

| band | class | N | median mag | dol <0.1" | median nmatch | NN brighter row < 0.1" / < 0.2" |
|---|---|---|---|---|---|---|
| F187N | a | 1290 | 20.4 | 0.30 | 3 | 0.04 / 0.32 |
| F187N | b | 2570 | 23.2 | 0.14 | 2 | 0.10 / 0.48 |
| F187N | c | 5351 | 24.1 | 0.012 | 1 | 0.16 / 0.64 |
| F187N | all fr1 unforced | 456373 | 23.5 | 0.032 | 1 | n/a |
| F200W | a | 1601 | 21.6 | 0.28 | 3 | 0.02 / 0.16 |
| F200W | b | 3707 | 24.9 | 0.06 | 2 | 0.04 / 0.29 |
| F200W | c | 8164 | 26.3 | 0.008 | 1 | 0.10 / 0.49 |
| F200W | all fr1 unforced | 349447 | 26.2 | 0.041 | 1 | n/a |

Dolphot match fraction by pipeline mag (c vs unforced control): F187N 18.6-21 0.10 (N=141) vs 0.52; 21-23 0.02 vs 0.05; 23-25 0.01 vs 0.01. F200W 21-23 0.06 (N=242) vs 0.77; 23-25 0.02 vs 0.14. Of the c rows, about 64 (F187N) and 65 (F200W) have a dolphot star within 0.1".

Mosaic local peak S/N (3x3 max minus annulus median over annulus robust sigma, `peaksnr.py`, median / fraction >3):
F187N (22.5-25.5 mag): c 0.9 / 0.02, b 1.2 / 0.04, unforced 1.0 / 0.02, random sky 0.8 / 0.04.
F200W (23.5-27 mag): c 1.2 / 0.11, b 1.5 / 0.18, unforced 1.0 / 0.03, random sky 0.7 / 0.06.

Judgement:
* The extra rows are not new detections produced by fr1. They come from the same per-frame seeds fr0 fitted, which fr0 then discarded because of its inflated forced flux.
* Class (c) rows are mostly at or below the dolphot limit. In F187N their mosaic peak S/N is indistinguishable from random sky and from the unforced faint tail, so the data do not support them individually; they resemble the single-frame faint population already in the unvetted m7 catalog (median nmatch 1). In F200W they sit slightly above sky (11% above 3 sigma vs 6%). Where dolphot is complete, c rows match dolphot less often than unforced rows at the same mag (F200W 21-23: 6% vs 77%), which indicates a fraction are fits to noise, artifacts (e.g. the F200W stripe example) or neighbour-wing residuals. Roughly 130 F187N and 130 F200W c rows have dolphot matches or are brighter than 21 mag and include recovered real stars.
* Class (b) rows are fainter than class (a), a larger share are real (F187N 14%, F200W 6% dolphot match with the dolphot depth limiting) and show higher peak S/N than the control; they were already in fr0 as unforced rows.
* The dolphot-based scores do not change at the star level (recall identical); the added rows live in the regime where dolphot cannot confirm them. Downstream vetting (nmatch, flux/err cuts) is the place to remove them; a cut on forced_refit_frac with nmatch == 1 would remove most class (c) rows. The extra counts do not by themselves indicate a regression.

## 3. Figure `fig_fr_cutouts.png`

Rows per band (F187N, F200W, F277W). Left block: 4 dolphot stars (17-21 mag) forced in both arms with frac >= 0.5, |dm_fr0| > 0.3 and |dm_fr1| < 0.1, picked evenly in dolphot mag (only 12-15 pass per band). Right block: 3 (F187N, F200W, F277W) random class (c) rows with frac >= 0.5 in a typical faint mag range. Per object: AB mosaic cutout (1.2" square, asinh), then the per-frame m7 residual of fr0 and of fr1 from the same frame (median removed, linear, +-max(0.5*peak, 5 sigma)). Markers: red circle dolphot, cyan x fr0 rows, lime + fr1 rows (only rows brighter than 24.5/26/23 mag drawn). Titles give dolphot, fr0 and fr1 mags (dm in parentheses).
Observations: in the left block the dolphot stars are bright and often close pairs; the fr0 row is at the star position but 0.3-1.2 mag too bright, the fr1 row matches dolphot. The frame residuals are near zero at the stars in both arms at this stretch, so the residual panels show little difference. Class (c) rows are mostly faint features in noise; one F187N case (22.8) and one F200W case (25.6) have visible sources near the dolphot circle, one F200W case sits on a detector stripe. Residuals for class (c) in F187N have no covering frame in one case (first F187N c panel, forced row found in a mosaic-edge frame with no residual file match).
