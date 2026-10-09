# LW core stars still lacking a good value after #1140 (arm main2kfpk)

## Setup
- Stars: the no-row sets from f277w_gap/trace_mainfcbg_{band}.ecsv (327 / 98 / 151 for F277W / F250M / F300M). "Good" = merged m7 row within 0.08" and |dm| < 0.3 (same ZP recipe as score_lw.py; reproduces 235 / 41 / 83). Lost: 92 / 57 / 68 (F277W has 1 star with no covering frame).
- Per star-frame (8 frames per band) diagnostics: `gather.py` (DQ geometry, hand-off positions from the pipeline functions called read-only on crf SCI/DQ, per-frame m7 rows, accepted/rejected satstars), `classify.py` (categories), `gather2.py` + `reach_sum.py` (offline hand-off/restore variants), `figs.py` (cutouts). Outputs: categories.md, reach.md, class_*.ecsv, sf_*.ecsv, reach_*.ecsv.
- Hand-off positions are reconstructed with peak_min_area=50, data_floor 0, plus implied_peak_gate rejects; the frame "bad" mask is approximated from ERR/SCI (the pipeline also subtracts the resbg map first). Hand-off radius for "within" = max(1, 0.5 FWHM) = 1 px.

## Categories (lost stars; priority order c, d, b2, b1, a)
| category | F277W | F250M | F300M |
|---|---|---|---|
| b1: position handed off within 1 px in some frame, no per-frame fit row at the star | 52 | 35 | 39 |
| b2: per-frame fit row at the star, no merged row within 0.08" (nearest merged row 0.1-0.5") | 7 | 16 | 19 |
| c: merged row with abs(dm) >= 0.3 (F277W: 14 faint, 5 more than 1 mag too bright; F250M 3; F300M 3) | 19 | 3 | 3 |
| d1: accepted satstar within 1 px but no merged row | 3 | 1 | 2 |
| d2: rejected satstar (fit_quality_gate) within 1 px | 3 | 1 | 2 |
| a4: on a big SAT component, no peak within 1 px (neighbour wing / not a local max) | 7 | 0 | 2 |
| a3: star pixel NaN/DO_NOT_USE, no peak | 0 | 1 | 0 |
| a1: component < 50 px, COM miss | 0 | 0 | 1 |
| no covering frame | 1 | 0 | 0 |
| total | 92 | 57 | 68 |

Category (a) "no hand-off position" now covers only 7 / 1 / 3 stars. The 1.5 FWHM satstar exclusion removed no peak that reached a lost star (a2 = 0 everywhere).

Dolphot magnitudes (p10 / median / p90) and median distance to the nearest accepted satstar (arcsec, min over frames):
- F277W b1 14.4 / 15.8 / 16.7, 0.63"; b2 15.9 / 16.8 / 17.8, 0.26"; c 15.1 / 16.2 / 17.8, 5.9"; a4 14.3 / 15.4 / 16.4, 0.45".
- F250M b1 13.3 / 15.0 / 16.1, 0.57"; b2 16.4 / 16.9 / 17.2, 0.38".
- F300M b1 13.9 / 14.8 / 16.1 (31 of 39 brighter than 16), 0.51"; b2 14.4 / 16.8 / 17.3, 0.41".
Full tables are in categories.md.

## Main cause: restore step skips components that hold an accepted satstar centre
For b1, 130 of 133 (F277W), 79 of 81 (F250M) and 95 of 96 (F300M) handed-off star-frames lie in a SATURATED component that also contains an accepted satstar centroid. `_handoff_restore_pixels` removes every such component from the restore set (`labels -= _labels_at(accepted...)`), so the handed-off star's core pixels stay masked. The near-saturation filter exemption applies, but the fit has no core pixels and no row appears. Only 2 / 1 / 0 of those star-frames have the pixel restored under the current rule. The same step is what the #1125 merged-component mechanism feeds: a big merged component nearly always contains at least one accepted star.

Figure panel 2 shows this: blue = SAT, green = restored; the red-circled stars sit in the blue (unrestored) region next to a green x (accepted satstar) in the same component.

## Figures
`lwrest_cutouts.png` (9 stars x 3 panels: per-frame crf SCI 3" cutout with dolphot red circle, hand-off magenta diamonds, per-frame m7 rows cyan +, accepted satstars green x, rejected satstars orange x, NaN gold/orange and SAT contour; SAT/restore/NaN map; merged m7 residual i2d cutout with merged rows). Stars (band idx, category): F277W 3708 b1, 5218 b1, 4803 b2, 4832 c faint, 4251 c bright, 3692 a4, 17264 d1; F250M 3867 a3 (listed under a3 in the picker; classified from the star's frames); F300M 3848 b1. Indices are rows of the matched table.

## Testable proposals (offline estimates, no pipeline change)
Metric: star reached when some frame has a hand-off within 1 px and the star pixel is in the restore set. Calibration: under the current rule 214 of 238 reached F277W stars (90%) are good, so reach is an upper bound on gain with a roughly 90% hit rate (not measured for the new rule).

1. Restore rule R1: keep components with accepted centres, but leave masked only the pixels within 1.5 FWHM of an accepted centre (matches the hand-off exclusion). Lost stars reached: F277W 24 -> 78 of 91, F250M 5 -> 40 of 57, F300M 6 -> 49 of 68 (about +54 / +35 / +43, so up to 130 stars). Already-good stars reached: 214 -> 221, 34 -> 38, 71 -> 74 of 235 / 41 / 83 (the rest were rescued by other frames). Risks: fits near accepted cores could pick up satstar wing or fill-model residuals; needs a full-chain A/B (control stars and 18.6-21 mag bins) and a check of the #1143 interaction.
2. Lower PEAK_MIN_AREA (10 or 1 instead of 50): +1 / +2 / +2 lost stars reached over the current setting under R1, with COM-only (area 0) reaching 3 / 2 / 2. Little value.
3. Smaller satstar exclusion radius (1.0, 0.5, 0 FWHM): +0 to +2 stars; the exclusion is not limiting.
4. Not estimated: b2 (merge/dedup, 42 stars) and c (bad photometry, 25 stars) need a look at the merge step and per-frame fluxes.

## Open questions
- b2: per-frame rows exist in 2-4 frames but the merged row is 0.1-0.5" away or absent. Is the cross-match picking a brighter neighbour, or is the nmatch/dedup cut removing them? The 8-frame per-frame row positions were not compared with the merged-cat membership here.
- c (F277W, 19): 14 are too faint by 0.3-1+ mag and 5 too bright by more than 1 mag (blends with neighbours inside a merged SAT component; 4251 and 4832 sit among several handed-off positions). Whether R1 changes them is unknown.
- After R1, stars with the core pixels returned still depend on the daophot fit with resbg clipping; yield may fall below the reach numbers.
- The restore bad-mask here is approximate (no resbg subtraction); the pipeline-reported "returned N px" log lines could be used to verify the count of restored pixels per frame.
