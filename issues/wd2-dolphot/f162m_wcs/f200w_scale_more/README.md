# F200W per-detector scale term on the Brick and NGC 6334 (issue #1137)

All files are in this directory; nothing outside it was modified. Nothing was posted or committed.

## Inputs

Per-frame catalogs: stage `resbgsub_m6`, `*_resbgsub_m6_daophot_basic.fits` (stale files excluded), exposures 1-4 per detector-visit. Anchors: the vetted merged `resbgsub_m6` catalogs.

| field | F200W frames | anchors | crf |
|---|---|---|---|
| brick | program 1182, visits 001+002, vgroup 04101 (8 det x 2 visits x 4 exp = 64 frames) | `catalogs/f187n_merged_o001_indivexp_merged_resbgsub_m6_dao_basic_vetted.fits`, `f182m_merged_o001_...` (program 2221, other epoch) | `F200W/pipeline/jw01182004*_04101_*_destreak_o004_crf.fits` |
| ngc6334 (6778) | `*_j6778_visit00{1,2,3}_vgroup02105` (12 frames/det) | `f187n_merged_j6778_...` (same program), `f182m_merged_j7213_...` (other program) | `jw06778001*_02105_*_align_o001_crf.fits` |
| ngc6334 (7213) | `*_j7213_visit00{1,2}_vgroup02107` | `f182m_merged_j7213_...` (same program) | `jw07213001*_02107_*_align_o001_crf.fits` |

- Brick has an F212N vetted m6 merged catalog and F212N crfs, but the task asked for F187N/F182M anchors, so F212N was not used. NGC 6334 has no F212N data at all.
- NGC 6334 F200W in the folder also contains untagged `*_visit00N_vgroup02105/02107_*` files that duplicate the `j6778`/`j7213` ones; only j-tagged files were used.
- NGC 6334 F187N exists only for program 6778, F182M only for program 7213 (different pointings). The F182M-vs-F187N controls are therefore cross-program.
- In NGC 6334, every frame of module A fails the match/pair requirements against the cross-program anchors (F200W(6778) vs F182M(7213), and both controls), so those columns have NRCB1-B4 only (nan elsewhere). F200W(7213) vs F182M(7213) and F200W(6778) vs F187N(6778) cover all 8 detectors.

## Method changes versus rc_scale.py (`rc_more.py`)

- Matching, quality cuts (qfit <= 0.1, S/N >= 20), 0.15" radius, local pairing, 3-sigma clipped linear fit and the pixel-frame scale definition (mean diagonal of C^-1 J, median J over frames) are unchanged. The sign +1 test was dropped.
- Anchor catalog, per-frame stage glob, program tag filter (`PROG`), exposure cap (`MAXEXP=4`), detector filter (`DETS`) and crf lookup by `jw{prog}*{visit}_{vgroup}_{exp}_{det}` come from environment variables or file names.
- Rotation: taken from `distortion_rotations.ecsv` (F200W/CLEAR rows, `correction_arcsec` used directly as the SIGN=-1 correction, i.e. `rotated_wcs(w, correction)`); the frame's `R_DISTOR` is required to equal `distortion_ref` (the script raises otherwise; no frame failed). For the A1 detector the correction (8.386") equals the angle that the F212N-crf route gave for wd2. F182M and F187N frames get no correction. No F212N crf comparison was run for the brick.
- Brick F200W vs F182M ran as 8 per-detector processes (`rcm_brickL_*`) because the single-process run was too slow (login node load ~270; SLURM jobs stayed pending and were cancelled). The abandoned partial single-process run is `rcm_brick_f200w_v_f182m.txt` and is not used.
- `analyze.py` builds the tables and `fig_f200w_scale_more.png`; `pervisit.py` refits J per (detector, visit) from the stored pairs (`pervisit.txt`). Per-run tables with frames, pairs and uncorrected values: `tables_per_run.txt`. Raw logs: `rcm_*.txt`, pairs `pairsm_*.pkl`.
- Scale (ppm) is the rotation-corrected pixel-frame scale; the uncorrected scale differs by < 1 ppm, as in wd2.

## F200W scale per detector (ppm)

| det | wd2 F200W-F212N | wd1 F200W-F212N | wd mean F200W-F150W | brick F200W-F187N | brick F200W-F182M | NGC F200W(6778)-F187N | NGC F200W(6778)-F182M | NGC F200W(7213)-F182M |
|---|---|---|---|---|---|---|---|---|
| NRCA1 | +42.1 | +37.5 | +34.0 | +41.1 | +54.4 | +37.6 | nan | +47.6 |
| NRCA2 | +20.2 | +25.4 | +19.9 | -23.3 | -6.5 | +21.9 | nan | +26.5 |
| NRCA3 | +22.2 | +44.4 | +21.4 | +56.5 | +29.2 | +35.4 | nan | +53.2 |
| NRCA4 | +15.5 | +44.7 | +13.5 | +18.1 | -20.7 | +13.0 | nan | +23.0 |
| NRCB1 | +1.9 | +22.2 | -8.2 | -10.8 | +14.3 | -11.2 | +3.2 | +12.3 |
| NRCB2 | -17.0 | -4.4 | -12.2 | +5.5 | +9.8 | -18.8 | -77.9 | +8.9 |
| NRCB3 | -18.0 | -21.8 | -10.4 | -74.0 | -35.2 | -30.0 | -34.9 | -21.2 |
| NRCB4 | -17.3 | -10.5 | -13.5 | -2.9 | +6.0 | -11.6 | +8.1 | -4.7 |

Corrected rotation (arcsec), frames and pairs per detector are in `tables_per_run.txt`. Corrected rotations are -0.2 to -13" for the brick and -0.8 to -5.7" for NGC F200W(6778)-F187N (wd2 values were within about 2"). Pair counts per detector: brick F187N 2300-6600, brick F182M 11600-35400 (crowded anchor, more chance matches), NGC F187N 3000-6800, NGC F200W(7213)-F182M 600-3600.

## rms differences to the wd reference rows (ppm, 8 detectors unless noted)

"Offset" is the mean of (field minus reference), i.e. the common isotropic scale. The removal column subtracts it before the rms.

| series | offset vs wd (F200W-F150W) | rms raw | rms after offset removal | offset vs wd2 F200W-F212N | rms raw | rms after removal |
|---|---|---|---|---|---|---|
| brick F200W-F187N | -4.3 | 30.9 | 30.6 | -4.9 | 29.8 | 29.4 |
| brick F200W-F182M | +0.8 | 23.3 | 23.2 | +0.2 | 22.2 | 22.2 |
| NGC F200W(6778)-F187N | -1.0 | 9.0 | 9.0 | -1.7 | 8.3 | 8.2 |
| NGC F200W(7213)-F182M | +12.6 | 17.3 | 11.8 | +12.0 | 15.9 | 10.5 |
| NGC F200W(6778)-F182M (B only, 4 det) | -14.3 | 37.1 | 34.3 | -12.8 | 34.1 | 31.6 |
| wd1 F200W-F212N (for scale) | +11.6 | 18.3 | 14.1 | +11.0 | 15.9 | 11.5 |

Pattern summary (A1-A4 mean / B1-B4 mean, ppm; correlation r with the wd mean row): wd reference 22.2 / -11.1; wd1 38.0 / -3.6 (r 0.82); NGC F187N 27.0 / -17.9 (r 0.96); NGC F182M(7213) 37.6 / -1.2 (r 0.87); brick F187N 23.1 / -20.6 (r 0.60); brick F182M 14.1 / -1.3 (r 0.50); brick mean of the two anchors 18.6 / -10.9 (r 0.59, rms 24.4 vs the wd mean, offset -1.7).

## Controls (corrected scale, ppm; same pipeline, no rotation applied)

| det | brick F182M-F187N | brick F187N-F182M | NGC F182M(7213)-F187N(6778) | NGC F187N(6778)-F182M(7213) |
|---|---|---|---|---|
| NRCA1 | -13.1 | -0.2 | nan | nan |
| NRCA2 | -12.1 | +2.2 | nan | nan |
| NRCA3 | -9.4 | +2.5 | nan | nan |
| NRCA4 | -5.8 | +3.0 | nan | nan |
| NRCB1 | -0.4 | +1.2 | -33.0 | +10.3 |
| NRCB2 | -23.9 | -5.7 | -22.4 | -69.8 |
| NRCB3 | -23.1 | +5.7 | -52.6 | -38.4 |
| NRCB4 | -8.7 | +11.7 | +25.8 | +49.9 |

The brick controls (same program) are within about +-12 ppm for F187N-F182M (4 frames per detector) and -0.4 to -24 ppm for F182M-F187N; the two directions differ in sign as expected for anchor-defined residuals but not in magnitude, because in one direction the 2221 F187N catalog is the target and in the other the F182M catalog is. The brick controls lack the +40 ppm A-module signal. The NGC controls are cross-program, B module only, with 4-8 frames and 400-1400 pairs per detector, and scatter by 30-70 ppm; they set the noise floor for the cross-program NGC F182M columns and are not an anchor check for the F187N column.

## Per-visit scatter (`pervisit.txt`, scale in ppm)

Single-visit values (4 exposures pooled) scatter by roughly +-10-30 ppm about the pooled value for NGC F200W(6778) (e.g. NRCA1 +39/+29/+59, NRCB3 -33/-40/-4, NRCB4 -40/+11/-38) and by up to 30-100 ppm for the brick (NRCA3 +38/+71 vs F187N; NRCB3 +34 (2 frames) / -81; NRCB1 +160 from 144 pairs in 2 frames). Two frames or few hundred pairs give unreliable values. Visit-to-visit differences in the brick are larger than the A-minus-B pattern for several detectors.

## Reading

- NGC 6334 F200W (6778) against the same-program F187N anchor reproduces the wd2 column closely: rms 8-9 ppm over 8 detectors, r = 0.96 with the wd mean row, common offset about -1 to -2 ppm. The A-positive, B2-B4-negative pattern, with NRCA1 at +38 and NRCB3 at -30 ppm, appears again. F200W(7213) against F182M (same program) has the same sign pattern with r = 0.87, a common offset of +12 ppm and rms 11-12 ppm after offset removal, comparable to the wd1-to-wd2 spread (rms 14 after offset removal).
- The brick reproduces the A-module excess on A1 and A3 (and A4 for F187N), and a negative B3, but the remaining detectors scatter by 20-80 ppm (NRCA2 -23 for F187N, NRCA4 -21 for F182M, NRCB3 -74). rms against the wd reference is 22-31 ppm; r with the wd mean row is 0.5-0.6. With the F187N and F182M anchors differing by 10-50 ppm on several detectors, anchor-dependent terms and the epoch/pointing differences (program 1182 vs 2221) appear to contribute at the 20 ppm level. These data neither confirm nor exclude the same 20-40 ppm structure in the brick; the brick mean over the two anchors (A about +19, B about -11 ppm) has the same sign and about the same size as the wd pattern.
- The common offset between F200W and anchor is small for the same-program NGC F187N comparison (-1 ppm) and for the brick (-4 to +1 ppm), larger for NGC F182M(7213) (+12 ppm, with the same offset appearing in the wd1 comparison at +11). A single isotropic scale from epoch or velocity aberration therefore does not appear to be required to explain the F200W patterns, but offsets of 10 ppm in individual fields are within the observed scatter.
- Not shown: the origin of the residual (distortion reference, detector-level plate scale, or catalog systematics); a colour or magnitude dependence; whether the NGC A-module F200W-F182M cross-program columns would agree (all A frames failed matching); whether the brick scatter is intrinsic to the field (heavy extinction, crowding, pair mismatches) or to the cross-epoch anchors. Exposure cap 4 per detector-visit and the pooled median over visits mean the error bars are of order 10 ppm for NGC and 20-30 ppm for the brick; no formal errors were computed.
