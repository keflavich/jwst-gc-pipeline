# Faint SW stars that change photometry between mainfcbg (A) and mainfcbgkf (B)

Scripts (rerun with `python <script> main2 main2kf` from Q_integ): `faint_sw.py A B` (per-star and per-frame tables, slow, about 15 min on the login node), `summarize.py A B`, `bkg_check.py A B`, `ffsplit.py A B`, `fig_movers.py A B [N]`. Helpers in `common.py`. Outputs: `stars_*.ecsv`, `frames_*.ecsv`, `tables_*.md`, `bkg_check_*.md`, `ffsplit_*.md`, `fig/faint_movers_mainfcbg_mainfcbgkf.png`.

Sample: stars matched to dolphot in both arms with finite magnitudes, dolphot mag > 20 in the band; "moved" = |B-A| > 0.1 mag (F150W 127, F182M 64, F187N 51, F162M 113); control = the remaining faint matched stars (3600 to 8300 per band). The counts are smaller than the full moved lists in pair_detail because of the > 20 cut. "Collapsed" frame = B/A per-frame satstar count > 2. Position of a star = its A m8 row position.

## Answers

1. Sign. B is fainter than A for about two thirds of the movers (F150W 81 of 127, F182M 40 of 64, F187N 31 of 51, F162M 74 of 113; median B-A about +0.14 to +0.17 mag). When B is fainter, A is the closer arm in 78 of 81 (F150W), 37 of 40 (F182M), 28 of 31 (F187N), 66 of 74 (F162M); median dm to dolphot goes from about +0.05 to +0.16 (A) to +0.35 to +0.97 (B). When B is brighter (46, 24, 20, 39 stars), B is closer in 43, 24, 18, 38 cases (A median dm +0.5 to +0.7 there, i.e. A was too faint, with a +2.4 outlier population in F182M). The worsening is a net effect of two sub-populations with opposite signs; the fainter-in-B group is larger and its shifts are larger (worst cases +2 to +6 mag).
2. Nearest satstar. Moved stars lie near saturated stars: median distance 0.6 to 1.3 arcsec versus 2.1 to 5.9 arcsec for controls; 71 to 96 percent of movers are within 2 arcsec (controls 12 to 47 percent). The distance is mostly set by B's satstar list; A-only distances are larger (F187N 5.7 vs 1.3 arcsec, F182M 1.2 vs 0.6), because B has satstars that A lacks.
3. Collapsed frames. The share of A per-frame detections falling on collapsed frames is 45 percent (F150W), 56 (F182M), 52 (F187N), 69 (F162M) for movers versus 17, 12, 49, 23 percent for controls. F150W, F182M and F162M therefore show a 3 to 5 fold enrichment (collapsed frames: F150W nrca2 e3/e4, nrca3 e1-e4; F182M nrca2 e1-e4; F162M nrca2 and nrca3 all exposures). F187N is not enriched (half of its frames meet the B/A > 2 criterion, including controls). Not all movers are on collapsed frames: 60 of 127 F150W movers have at least one collapsed-frame detection.
4. Model difference. B model - A model is localized on the satstars that exist only in B. At mover positions the PSF-weighted difference relative to the star's per-frame flux is positive (B subtracts more) in 20 to 50 percent of star-frames with Dpsf/fA > +0.1 and essentially never negative; median Dpsf/fA is +0.01 (F150W), +0.04 (F182M), +0.09 (F187N), +0.10 (F162M). Controls have |Dpsf/fA| < 0.02 in 99 percent of star-frames. The sign matches the dominant outcome (B fainter), but the predicted shift is small for F150W and F182M (median predicted +0.03 mag vs observed +0.19; predicted/observed ratio 0.08 to 0.16) and about half the observed size for F187N and F162M (+0.11 to +0.12 vs +0.13 to +0.16). Spearman correlation between predicted and observed per-star shifts is 0.18 (F150W), 0.35, 0.53, 0.48. The wing subtraction therefore explains the sign and a good part of the F187N/F162M size, and does not by itself explain the large F150W/F182M shifts. The per-frame m7 daophot flux of the same stars does not change on average (median (fB-fA)/fA about 0), so the m8 shift is not a simple per-frame flux change.
5. Other properties. (a) Movers are mostly forced-filled: 48 to 69 percent of movers have forced_filled=1 in A versus 7 to 26 percent for controls; this rises in B for F150W (0.48 to 0.69) and F182M (0.69 to 0.81). (b) The sign and size depend on forced_filled status (ffsplit table): stars that are forced-filled only in B are all fainter in B (median +0.42 F150W, +1.16 F182M, +0.34 F162M); stars forced-filled only in A are brighter in B (-0.48 F150W). Stars with unchanged status move by a median +0.12 to +0.21 with a 51 to 85 percent B-fainter share. (c) 22 to 40 percent of movers have no A per-frame m7 detection at all (F182M 24 of 64). (d) Movers are crowded: median 6 to 16 m8 neighbours within 1 arcsec versus 3 for controls. (e) Higher qfit (0.22 to 0.37 vs 0.08 to 0.21) that drops in B. (f) spike_artifact flag in 2 to 8 percent (controls 0.2 to 0.5 percent); near_saturated 13 percent for F150W movers (3 percent controls). (g) Number of contributing frames (nmatch) is 4 in both arms and rarely changes. (h) mean_modelsub_bkg changes by only 0.5 to 1 percent of the star's flux, so the background term is not the cause.

## Figure
`fig/faint_movers_mainfcbg_mainfcbgkf.png`: the 8 worst movers (two per band, largest increase in |dm|; B dm from +1.7 to +6.0 mag). Reading it: in six rows (F182M x2, F150W x2, F162M x2, F187N nrcb1; seven if counted by the "sat only in B" tag) the nearest satstar is a bright saturated star that A did not fit, flagged "sat only in B". In A the saturated core, rings and spikes remain in the residual and A's daophot places a source on the core (green plus on the star). B subtracts the core and most of the ring pattern; the B residual shows a compact negative/positive hole at the satstar and weaker diffraction structure in the surroundings. The m8 position of the mover sits 0.3 to 1.0 arcsec from the saturated star, inside the ring and spike pattern, so A's flux there is contaminated by unsubtracted structure and B's value is lower. For these rows dolphot agrees with A to within 0.0 to 0.3 mag in four cases, which indicates A's brightness there is accidentally close (the unsubtracted wing is cancelled by or comparable to the unmodelled structure) or that dolphot also leaves such light in its fluxes. The B model - A model panel shows red (B model larger) blobs confined to the satstar core and its immediate rings, with a weak spike pattern. One row (F187N nrcb2 exp1, 12.7 arcsec from the nearest satstar, crowded nebular field) has no model difference at all, so its shift comes from another mechanism (forced-fill or crowded-field fitting). The red circle marks the m8 position, so the circle does not coincide with any visible point source in several rows; in the F182M/F162M rows the circle lies on the ring structure.

## Most likely explanation
In A the R(g0) collapse (#1096) leaves many bright saturated stars unmodelled on a subset of frames, so the diffraction rings and spikes remain in the residual image and enter the faint-star measurements (usually as extra flux, but also as detections that dolphot does not have). B fits those stars, removes the wing flux, and the nearby faint stars become fainter. For stars that A got close to dolphot by accident (the unsubtracted wing is close to the mismatch) B looks worse; for stars that A had too faint B is better. The strongest shifts coincide with a change in forced_filled status, i.e. stars near a newly fitted satstar are no longer independently detected in B and receive forced photometry, which is fainter. The A-side confounder (collapsed frames) is a large part of the enrichment: 45 to 69 percent of detections on collapsed frames for F150W/F182M/F162M movers versus 12 to 23 percent for controls. The comparison therefore reflects A's collapse more than a defect of KEEP_FINITE; main2 vs main2kf (where #1096 may be fixed) is the cleaner test. Limitations: the Gaussian PSF weighting underestimates the model-difference effect on pixels in the wings, the satstar distance uses same-band satstar catalogs only, and the forced_filled interpretation is based on catalog flags, not on a direct inspection of the forced-photometry step.

---
# Faint SW movers mainfcbgkf vs mainfcbg (dolphot mag > 20, moved = |B-A| > 0.1 mag)

## Q1 sign and closeness to dolphot
| band | moved | median dolphot | mainfcbgkf brighter (B-A<0) | mainfcbgkf fainter | median B-A | closer to dolphot: B / A, B brighter | B / A, B fainter | median abs(dm) A -> B (all moved) |
|---|---|---|---|---|---|---|---|---|
| F150W | 127 | 24.3 | 46 | 81 | +0.17 | 43 / 3 | 3 / 78 | 0.206 -> 0.509 |
| F182M | 64 | 22.5 | 24 | 40 | +0.14 | 24 / 0 | 3 / 37 | 0.346 -> 1.110 |
| F187N | 51 | 21.5 | 20 | 31 | +0.14 | 18 / 2 | 3 / 28 | 0.237 -> 0.317 |
| F162M | 113 | 23.6 | 39 | 74 | +0.16 | 38 / 1 | 8 / 66 | 0.319 -> 0.466 |

## Q1b signed dm to dolphot (median), moved stars
| band | median dmA | median dmB | median dmA, B brighter | median dmB, B brighter | median dmA, B fainter | median dmB, B fainter |
|---|---|---|---|---|---|---|
| F150W | +0.17 | +0.50 | +0.70 | +0.20 | +0.05 | +0.55 |
| F182M | +0.35 | +1.11 | +2.44 | +2.07 | +0.16 | +0.97 |
| F187N | +0.23 | +0.32 | +0.50 | +0.13 | +0.10 | +0.35 |
| F162M | +0.32 | +0.47 | +0.69 | +0.37 | +0.09 | +0.57 |

## Q2 distance to nearest satstar (same-band per-frame m7 satstar catalogs, A and B pooled; arcsec)
| band | group | N | median dsat | frac < 2" | frac < 5" | frac < 10" | median dsat A only | median dsat B only |
|---|---|---|---|---|---|---|---|---|
| F150W | moved | 127 | 0.8 | 0.961 | 1.000 | 1.000 | 1.0 | 0.8 |
| F150W | control | 8259 | 2.1 | 0.474 | 0.920 | 1.000 | 2.2 | 2.1 |
| F182M | moved | 64 | 0.6 | 0.875 | 0.969 | 1.000 | 1.2 | 0.6 |
| F182M | control | 7342 | 4.1 | 0.206 | 0.607 | 0.918 | 4.4 | 4.1 |
| F187N | moved | 51 | 1.3 | 0.706 | 0.824 | 0.941 | 5.7 | 1.3 |
| F187N | control | 3625 | 5.9 | 0.122 | 0.428 | 0.793 | 7.1 | 5.9 |
| F162M | moved | 113 | 1.0 | 0.956 | 0.991 | 1.000 | 2.3 | 1.0 |
| F162M | control | 7808 | 2.6 | 0.381 | 0.842 | 0.988 | 2.9 | 2.6 |

## Q3 detections on collapsed frames (B/A per-frame satstar count > 2); A m7 daophot rows within 0.1"
| band | group | stars | A detections | on collapsed frames | frac | stars with any collapsed detection | stars with all detections collapsed |
|---|---|---|---|---|---|---|---|
| F150W | moved | 127 | 347 | 157 | 0.452 | 60 | 32 |
| F150W | control | 8259 | 31236 | 5192 | 0.166 | 1801 | 895 |
| F182M | moved | 64 | 131 | 73 | 0.557 | 21 | 21 |
| F182M | control | 7342 | 26970 | 3211 | 0.119 | 853 | 853 |
| F187N | moved | 51 | 170 | 88 | 0.518 | 37 | 19 |
| F187N | control | 3625 | 14113 | 6973 | 0.494 | 2692 | 1327 |
| F162M | moved | 113 | 263 | 182 | 0.692 | 55 | 55 |
| F162M | control | 7808 | 29842 | 6933 | 0.232 | 1807 | 1807 |

frames per band: (det exp nsatA nsatB collapsed)
- F150W: 32 frames with detections, 6 collapsed: nrca2e3 5/90; nrca2e4 4/87; nrca3e1 6/93; nrca3e2 6/96; nrca3e3 4/98; nrca3e4 5/94
- F182M: 31 frames with detections, 4 collapsed: nrca2e1 3/31; nrca2e2 2/32; nrca2e3 1/33; nrca2e4 3/33
- F187N: 32 frames with detections, 16 collapsed: nrca1e1 7/17; nrca1e2 7/17; nrca1e3 5/16; nrca1e4 6/17; nrca2e3 8/17; nrca2e4 8/17; nrca3e2 7/15; nrca4e1 2/13; nrca4e2 2/14; nrca4e3 2/14; nrca4e4 2/14; nrcb1e2 55/119; nrcb4e1 10/27; nrcb4e2 12/28; nrcb4e3 11/29; nrcb4e4 11/26
- F162M: 32 frames with detections, 8 collapsed: nrca2e1 4/61; nrca2e2 3/62; nrca2e3 3/64; nrca2e4 3/63; nrca3e1 1/70; nrca3e2 2/69; nrca3e3 4/68; nrca3e4 5/69

## Q4 B model - A model at the star position (7x7 PSF-weighted, Gaussian PSF, counts in image units) vs star flux
Dpsf = sum(D*P)/sum(P^2) with P a normalized Gaussian of the nominal band FWHM; predicted fB = fA - Dpsf (B residual = data - Bmodel).
| band | group | star-frames | median Dpsf/fA | frac Dpsf/fA < -0.1 | frac > +0.1 | frac abs < 0.02 | median abs(Dpsf/fA) | median (fB-fA)/fA observed (B row found) | median predicted (-Dpsf/fA) same rows |
|---|---|---|---|---|---|---|---|---|---|
| F150W | moved | 347 | +0.010 | 0.003 | 0.205 | 0.536 | 0.013 | -0.000 | -0.008 |
| F150W | control | 31236 | +0.000 | 0.000 | 0.001 | 0.990 | 0.000 | +0.000 | +0.000 |
| F182M | moved | 131 | +0.037 | 0.000 | 0.420 | 0.405 | 0.041 | -0.001 | -0.007 |
| F182M | control | 26970 | +0.000 | 0.000 | 0.000 | 0.994 | 0.000 | +0.000 | +0.000 |
| F187N | moved | 170 | +0.087 | 0.000 | 0.441 | 0.324 | 0.087 | -0.001 | -0.075 |
| F187N | control | 14113 | +0.000 | 0.000 | 0.004 | 0.985 | 0.000 | +0.000 | +0.000 |
| F162M | moved | 263 | +0.100 | 0.000 | 0.498 | 0.285 | 0.100 | -0.008 | -0.055 |
| F162M | control | 29842 | +0.000 | 0.000 | 0.001 | 0.988 | 0.000 | +0.000 | +0.000 |

## Q4b per-star predicted vs observed B-A (mag); predicted = -2.5 log10(sum(fA - Dpsf) / sum(fA)) over A-detected frames
| band | moved stars with frames | Spearman rho(pred, observed) | sign agreement | median observed | median predicted | median abs(pred) / abs(obs) | fraction |pred| > 0.05 |
|---|---|---|---|---|---|---|---|
| F150W | 98 | 0.18 | 0.66 | +0.19 | +0.031 | 0.08 | 0.41 |
| F182M | 40 | 0.35 | 0.60 | +0.17 | +0.032 | 0.16 | 0.47 |
| F187N | 48 | 0.53 | 0.54 | +0.13 | +0.113 | 0.49 | 0.65 |
| F162M | 80 | 0.48 | 0.66 | +0.16 | +0.119 | 0.48 | 0.61 |

## Q5 other properties (medians; moved vs control)
| band | group | N | m8 nmatch A | nmatch B | qfit A | qfit B | spike_artifact A / B | near_saturated A / B | forced_filled A / B | neighbours within 1" (m8, A) | satstar_nframes A / B | rep.sat A / B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F150W | moved | 127 | 4 | 4 | 0.27 | 0.13 | 0.039 / 0.024 | 0.134 / 0.126 | 0.480 / 0.685 | 6 | 0 / 0 | 0.008 / 0.008 |
| F150W | control | 8259 | 4 | 4 | 0.08 | 0.08 | 0.004 / 0.004 | 0.026 / 0.027 | 0.080 / 0.080 | 3 | 0 / 0 | 0.000 / 0.000 |
| F182M | moved | 64 | 4 | 4 | 0.22 | 0.19 | 0.062 / 0.047 | 0.062 / 0.047 | 0.688 / 0.812 | 10 | 0 / 0 | 0.000 / 0.000 |
| F182M | control | 7342 | 4 | 4 | 0.08 | 0.08 | 0.004 / 0.004 | 0.009 / 0.008 | 0.132 / 0.132 | 3 | 0 / 0 | 0.000 / 0.000 |
| F187N | moved | 51 | 4 | 4 | 0.35 | 0.30 | 0.020 / 0.020 | 0.000 / 0.000 | 0.686 / 0.686 | 16 | 0 / 0 | 0.000 / 0.000 |
| F187N | control | 3625 | 4 | 4 | 0.21 | 0.21 | 0.002 / 0.001 | 0.003 / 0.003 | 0.258 / 0.259 | 3 | 0 / 0 | 0.000 / 0.000 |
| F162M | moved | 113 | 4 | 4 | 0.37 | 0.28 | 0.080 / 0.053 | 0.000 / 0.062 | 0.673 / 0.637 | 9 | 0 / 0 | 0.000 / 0.000 |
| F162M | control | 7808 | 4 | 4 | 0.09 | 0.09 | 0.005 / 0.004 | 0.021 / 0.021 | 0.074 / 0.074 | 3 | 0 / 0 | 0.000 / 0.000 |

## Q5b change in contributing frames (nmatch B - nmatch A), moved vs control
| band | group | median | frac B < A | frac B > A |
|---|---|---|---|---|
| F150W | moved | +0 | 0.024 | 0.000 |
| F150W | control | +0 | 0.001 | 0.000 |
| F182M | moved | +0 | 0.000 | 0.016 |
| F182M | control | +0 | 0.000 | 0.000 |
| F187N | moved | +0 | 0.020 | 0.020 |
| F187N | control | +0 | 0.000 | 0.000 |
| F162M | moved | +0 | 0.027 | 0.009 |
| F162M | control | +0 | 0.000 | 0.000 |
## forced_filled status split (ffsplit.py)
| band | forced_filled (A, B) | N | median B-A | frac B fainter |
|---|---|---|---|---|
| F150W | ffA & ffB | 43 | +0.12 | 0.51 |
| F150W | neither | 22 | +0.14 | 0.68 |
| F150W | only B | 44 | +0.42 | 1.00 |
| F150W | only A | 18 | -0.48 | 0.00 |
| F150W | moved stars with >= 1 A per-frame daophot row within 0.1" | 98 of 127 | | |
| F182M | ffA & ffB | 41 | +0.12 | 0.51 |
| F182M | neither | 9 | +0.16 | 0.78 |
| F182M | only B | 11 | +1.16 | 1.00 |
| F182M | only A | 3 | -0.15 | 0.33 |
| F182M | moved stars with >= 1 A per-frame daophot row within 0.1" | 40 of 64 | | |
| F187N | ffA & ffB | 32 | +0.12 | 0.56 |
| F187N | neither | 13 | +0.21 | 0.85 |
| F187N | only B | 3 | +0.85 | 0.67 |
| F187N | only A | 3 | -0.56 | 0.00 |
| F187N | moved stars with >= 1 A per-frame daophot row within 0.1" | 48 of 51 | | |
| F162M | ffA & ffB | 62 | +0.16 | 0.63 |
| F162M | neither | 27 | +0.17 | 0.78 |
| F162M | only B | 10 | +0.34 | 1.00 |
| F162M | only A | 14 | -0.18 | 0.29 |
| F162M | moved stars with >= 1 A per-frame daophot row within 0.1" | 80 of 113 | | |
## m8 flux and background (bkg_check.py)
## m8 flux and background terms, A -> B (medians over stars)
| band | group | N | flux A | flux B | median (fB-fA)/fA | (fB-fA)/flux_err_A | d mean_modelsub_bkg (B-A) | d modelsub_bkg / fluxA | d local_bkg (B-A) | d forced_refit_frac | frac forced_filled A | frac forced_filled B | flux_err_prop A / B | flux A < 0 frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F150W | moved | 127 | 5.54 | 4.2 | -0.144 | -5.24 | -0.079 | -0.005 | -0.136 | +0.000 | 0.480 | 0.685 | 0.342 / 0.456 | 0.000 |
| F150W | control | 8259 | 29.2 | 29.2 | +0.000 | +0.00 | +0 | +0.000 | +0 | +0.000 | 0.080 | 0.080 | 0.361 / 0.362 | 0.000 |
| F182M | moved | 64 | 9.49 | 5.78 | -0.125 | -8.35 | -0.226 | -0.010 | -0.234 | +0.000 | 0.688 | 0.812 | 0.726 / 0.642 | 0.000 |
| F182M | control | 7342 | 42.6 | 42.6 | +0.000 | +0.00 | +0 | +0.000 | +0 | +0.000 | 0.132 | 0.132 | 0.705 / 0.704 | 0.000 |
| F187N | moved | 51 | 76.8 | 65.2 | -0.119 | -2.84 | -0.521 | -0.006 | -0.848 | +0.000 | 0.686 | 0.686 | 5.84 / 6.41 | 0.000 |
| F187N | control | 3625 | 111 | 111 | +0.000 | +0.00 | +0 | +0.000 | +0 | +0.000 | 0.258 | 0.259 | 4.62 / 4.62 | 0.000 |
| F162M | moved | 113 | 7.57 | 6.07 | -0.137 | -2.89 | -0.072 | -0.008 | -0.0939 | +0.000 | 0.673 | 0.637 | 0.431 / 0.447 | 0.000 |
| F162M | control | 7808 | 36.7 | 36.7 | +0.000 | +0.00 | +0 | +0.000 | +0 | +0.000 | 0.074 | 0.074 | 0.57 / 0.571 | 0.000 |