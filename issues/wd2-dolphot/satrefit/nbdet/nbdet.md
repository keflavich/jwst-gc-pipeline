# Detector-split reference tests for the nrcb3 satstar offset (wd2, main2 arm)

Files: `groups.py` (detector assignment), `nbdet_nb.py` (+ `nbstep_base.py`, a verbatim copy of `../nbstep/nbstep.py`), `thirdref_det.py` (copy of the `../thirdref/thirdref.py` matching and selection), `fig.py`, `nbdet.png`, `nbdet_tables.md` (all tables), `*_results.json`, logs. No original file was edited.

## Method
- Groups: nrcb1, nrcb3, other SW (nrca1-4, nrcb2, nrcb4). Each matched star takes the detector whose exposure-1 crf WCS footprint contains it (W band of the pair). The footprints do not overlap; 5% of stars fall in gaps and are dropped. For stars with satstar rows (`Q_integ/nrcb3/rows_<band>.npz`, S_starm + S_det) the row detector agrees with the footprint for 100% of stars (2325 F150W, 1781 F162M, 824 F182M, 722 F200W).
- Narrow-band test: selection, edges, quality and isolation cuts, and the linear trend (window 1.0/1.5/2.0 mag) are those of nbstep, with edges taken from all stars. Trend A is one fit over all groups, applied to each group's saturated stars (the unsaturated window comparison for the step is not needed, the step is the median of colour minus trend). Trend B is a fit per group. Errors: 300 paired bootstraps (identical indices for ours and dolphot; fit set and each group's saturated set resampled). Negative step = W reads bright. Only the three SW pairs are run; F200W-F212N has no usable range.
- Ground test: Ascenso 2007 JHKs, matching and cuts as in thirdref (default). Colour term fitted on all unsaturated stars (A) or per group (B); step = saturated median minus unsaturated median in [edge, edge+w] within the same group. Bootstrap as above.
- Control: median (ours - dolphot) in the narrow band (unsaturated here) and in W, for W-saturated stars versus W-unsaturated stars in the 1.5 mag window, per group.

## Narrow-band results (linear trend, window 1.5, trend A; B in parentheses)
| pair | group | N sat | ours | dolphot | ours - dolphot |
|---|---|---|---|---|---|
| F150W-F164N | nrcb1 | 490 | -0.025 +- 0.021 | -0.025 +- 0.020 | +0.000 +- 0.011 (-0.000 +- 0.027) |
| F150W-F164N | nrcb3 | 456 | -0.060 +- 0.022 | -0.009 +- 0.020 | -0.051 +- 0.011 (-0.060 +- 0.030) |
| F150W-F164N | other | 662 | -0.077 +- 0.042 | -0.078 +- 0.040 | +0.001 +- 0.015 (-0.039 +- 0.021) |
| F162M-F164N | nrcb1 | 353 | +0.005 +- 0.005 | +0.005 +- 0.003 | +0.001 +- 0.004 (+0.000 +- 0.005) |
| F162M-F164N | nrcb3 | 350 | -0.031 +- 0.004 | +0.011 +- 0.003 | -0.042 +- 0.003 (-0.051 +- 0.005) |
| F162M-F164N | other | 406 | -0.011 +- 0.005 | -0.015 +- 0.004 | +0.004 +- 0.004 (+0.008 +- 0.005) |
| F182M-F187N | nrcb1 | 67 | -0.016 +- 0.011 | -0.009 +- 0.007 | -0.007 +- 0.009 (-0.010 +- 0.010) |
| F182M-F187N | nrcb3 | 75 | -0.027 +- 0.009 | +0.009 +- 0.004 | -0.036 +- 0.008 (-0.037 +- 0.009) |
| F182M-F187N | other | 60 | -0.013 +- 0.008 | -0.023 +- 0.009 | +0.010 +- 0.008 (+0.011 +- 0.008) |

Windows 1.0 and 2.0 give the same pattern for nrcb3 (ours - dolphot F150W -0.056 / -0.069, F162M -0.040 / -0.044, F182M -0.031 / -0.034) and values consistent with zero for nrcb1 (-0.012 / -0.017, +0.003 / -0.002, -0.001 / -0.004). See `nbdet_tables.md` for all windows, per-bin steps and trend B.

Control (ours - dolphot, saturated minus unsaturated, 1.5 mag window):
| pair | group | narrow band | W band |
|---|---|---|---|
| F150W-F164N | nrcb1 / nrcb3 / other | +0.003 / +0.003 / +0.002 (+-0.001-0.002) | -0.015 / -0.053 / -0.002 |
| F162M-F164N | nrcb1 / nrcb3 / other | +0.005 / +0.002 / +0.002 | +0.003 / -0.038 / -0.001 |
| F182M-F187N | nrcb1 / nrcb3 / other | +0.002 / +0.002 / -0.003 (+-0.003, 0.002, 0.005) | -0.022 / -0.049 / -0.012 |

The narrow-band ours - dolphot offsets are the same in all groups (the saturated-minus-unsaturated change is at most 0.005) and the raw narrow-band offset is -0.02 to -0.036 mag in every group. The W-band offset change for saturated stars is concentrated on nrcb3.

## Where the nrcb3 step sits
- Ours on nrcb3 steps bright relative to the narrow-band continuity by 0.027-0.060 mag (F150W -0.060, F162M -0.031, F182M -0.027); on nrcb1 the step is -0.025 / +0.005 / -0.016. The nrcb3 minus nrcb1 difference in ours is about -0.035 (F150W), -0.036 (F162M), -0.011 (F182M) in the all-group trend. The F150W and F182M differences are 1.2 sigma and 0.8 sigma; the F162M difference is large (9 sigma) because that pair has the smallest colour scatter.
- Dolphot does not step faint on nrcb3. Dolphot steps are -0.009 (F150W), +0.011 (F162M), +0.009 (F182M) on nrcb3 and -0.025, +0.005, -0.009 on nrcb1: dolphot on nrcb3 is within 0.02 mag of continuous, and slightly fainter than nrcb1 by 0.016, 0.006, 0.018 mag. This accounts for a minority of the ours - dolphot difference (-0.04 to -0.05 on nrcb3, ~0 on nrcb1); most of it comes from ours reading bright.
- Ours - dolphot is consistent with zero on nrcb1 and "other" for F162M and F182M, and -0.051 +- 0.011 (A) on nrcb3 in F150W versus 0.000 +- 0.011 on nrcb1. In F150W the "other" group shows large steps in both catalogs (-0.077, -0.078), driven by its brightest bin (colour distribution of that group; see figure) and cancelling in the difference; with trend B, ours - dolphot on "other" is -0.039 +- 0.021.
- Ground reference (F200W vs Ks, 1.5 mag window, trend A): ours steps -0.036 +- 0.029 (nrcb1), -0.036 +- 0.030 (nrcb3), -0.026 +- 0.012 (other); dolphot steps -0.011 +- 0.027, +0.037 +- 0.033, +0.031 +- 0.014. Ours - dolphot is -0.025 +- 0.014 (nrcb1), -0.073 +- 0.018 (nrcb3), -0.056 +- 0.011 (other). In this test ours is similar across groups and the group difference comes from dolphot stepping faint on nrcb3 and other (+0.03 to +0.04) relative to nrcb1 (-0.01). Each ours step has an error of 0.03 on nrcb1/nrcb3, so the ground test does not discriminate between ours and dolphot at the single-group level. It shows ours - dolphot near -0.07 on nrcb3, consistent with the narrow-band values (-0.04 to -0.06).
- F150W vs H (colour H-Ks): N unsaturated in the window is 17 / 9 / 72 for nrcb1 / nrcb3 / other (2 for nrcb3 with J-H colour, so no value). Steps are positive in both catalogs (+0.07 to +0.15, unsaturated stars at the edge are few and blue/red-biased), ours - dolphot -0.03 +- 0.05, -0.02 +- 0.04, -0.04 +- 0.03. The F150W ground test is uninformative per detector.

## Summary
The narrow-band test places most of the nrcb3 excess on our side: ours on nrcb3 steps bright by 0.03-0.06 mag with the narrow band unchanged between the saturated and unsaturated stars, and dolphot on nrcb3 stays within 0.02 mag of continuity. On nrcb1 ours and dolphot differ by 0.00 to -0.01 mag in all three pairs. For F162M and F182M the ours - dolphot step on nrcb3 (-0.042, -0.036) is of the size of the 0.045-0.075 mag nrcb3 - nrcb1 offset, and the ground F200W test gives -0.073 on nrcb3 versus -0.025 on nrcb1, with that test also moving by 0.04-0.05 through dolphot stepping faint. The reading "ours bright on nrcb3" is supported by the narrow band; the ground test shows the same ours - dolphot sign and a partial dolphot contribution.

## Caveats
- Small N: nrcb3 F182M has 75 saturated stars, F200W-Ks has 48 saturated and 46 window stars on nrcb3. Per-group F150W unsaturated stars with ground colours number 9 (nrcb3), so the F150W ground test is unusable per detector. Trend B on nrcb3 uses 300-400 narrow-band stars but few per bin.
- Colour distributions differ between detectors (extinction varies across the cluster and the groups sample different regions); the global-trend step (A) mixes this with saturation, the per-group trend (B) removes it at the cost of larger errors. The ours - dolphot difference cancels shared colour effects because both use the same stars.
- Nebular lines: F164N [Fe II], F187N Pa-alpha, F182M containing Pa-alpha. Diffuse emission varies by detector and shifts colours of stars in bright nebulosity in both catalogs; the control columns (narrow band ours - dolphot unchanged between saturated and unsaturated stars) test only the photometry offset, not the line contamination of the colour itself.
- F182M-F187N has a 0.7 mag range and one 0.5 mag bin (about 60-75 stars per group).
- Edges are taken from all stars. A different local saturation edge on nrcb3 (the satstar fraction by detector) would shift the sample by a fraction of a bin.
- Ground catalog: seeing 0.4-0.5 arcsec, blending, and a drift of about 0.03 mag in the unsaturated residual with magnitude; errors are statistical.
- Bootstrap errors do not include trend-model systematics (linear vs quadratic, not rerun here).
