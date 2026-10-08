# F200W pixel-frame scale residual: field dependence and colour dependence (issue #1135)

All code and outputs are in this directory. Nothing outside it was modified.

## Method and inputs

- `rc_scale.py` is a copy of `../rotcorr_proto.py` with these changes: glob for crf files generalised to `*_o00?_crf.fits` (wd2 `align_o005`, wd1 `destreak_o001`); the vgroup regex accepts alphanumeric ids (wd1 has `0210b`); the corrected (SIGN=-1) pairs and the anchor positions are dumped to `pairs_<field>.pkl`. The matching (`measure_offset`, `local_residual_map`, 0.15" radius, qfit<=0.1, S/N>=20), the SIGN=-1 rotation correction and the per-detector Jacobian fit are unchanged. Reported pixel-frame scale = mean of the diagonal of C^-1 J (C = pixel-to-sky Jacobian at detector centre), median J over frames, exposures 1-2 (MAXEXP=2), visit001 only.
- wd2 anchor: `/orange/adamginsburg/jwst/wd2/catalogs/f212n_merged_indivexp_merged_resbgsub_m6_dao_basic_vetted.fits`, per-frame stage `resbgsub_m6`.
- wd1 anchor: `/orange/adamginsburg/jwst/wd1/catalogs/f212n_merged_indivexp_merged_resbgsub_m6_dao_basic_vetted.fits`; per-frame catalogs `F200W/f200w_nrc*_visit001_vgroup*_exp0000[12]_resbgsub_m6_daophot_basic.fits`. The wd1 stage names match wd2 (`resbgsub_m6`), so no stage change was needed. Run on the login node with `nice -19` (SLURM jobs stayed pending, so they were cancelled).
- The wd2 rerun reproduces the quoted F200W numbers (+42/+20/+22/+16/+2/-17/-18/-17 ppm).
- `colour_split.py` (B): pairs of all frames of a detector are pooled, colour = -2.5 log10(F115W flux / F212N flux) of the anchor star (F115W from the wd2 vetted m6 merged catalog, 0.1" match, S/N>5; 6812 of 7608 anchor stars). The same F115W-F212N colour is used for the F200W and F150W splits. Pairs are split into colour terciles per detector, J is refit (SIGN=-1 corrected residuals, same 3-sigma clipped fit), and 200 bootstrap resamples of the pairs give the 1-sigma error. Slope = weighted line through the three (median colour, scale) points. Pairs are resampled individually, although the same star appears in several frames, so the errors are probably underestimates; the pooled fit also differs slightly from the per-frame median (e.g. 'all' column vs table A).
- `make_fig.py` makes the tables (`tables_AC.txt`, `tables_B.txt`) and `fig_f200w_scale.png`.
- Raw logs: `rc_wd1.txt`, `rc_wd2.txt`, `colour_split.txt`.

## A. wd1 replication (corrected pixel-frame scale in ppm / rotation in arcsec)

| det | wd2 F200W | wd1 F200W | wd2 F150W | wd1 F150W |
|---|---|---|---|---|
| NRCA1 | +42.1/-2.29 | +37.5/-4.09 | +7.8/-1.18 | +3.7/-3.41 |
| NRCA2 | +20.2/-1.72 | +25.4/-1.95 | -0.9/-1.60 | +6.7/-0.54 |
| NRCA3 | +22.2/+0.49 | +44.4/-1.69 | +1.9/-1.21 | +21.8/-1.05 |
| NRCA4 | +15.5/-0.03 | +44.7/-3.75 | +3.8/+0.37 | +29.5/-3.18 |
| NRCB1 | +1.9/+0.06 | +22.2/-1.49 | +10.0/+0.65 | +30.5/-1.25 |
| NRCB2 | -17.0/+0.25 | -4.4/-2.18 | -6.9/-0.67 | +10.0/-2.19 |
| NRCB3 | -18.0/+0.77 | -21.8/-1.80 | -8.1/+0.24 | -11.0/-2.53 |
| NRCB4 | -17.3/-0.80 | -10.5/-0.14 | -2.4/-0.80 | +1.6/+0.17 |

Full per-detector lines (uncorrected and corrected) are in `rc_wd1.txt` and `rc_wd2.txt`.

## B. wd2 colour terciles (pixel-frame scale, ppm, bootstrap 1-sigma; c = median colour in mag)

### F200W

| det | all | T1 (blue) | T2 | T3 (red) | slope (ppm/mag) |
|---|---|---|---|---|---|
| NRCA1 | +40.6+-1.7 | +37.4+-3.2 (c=0.33, n=186) | +36.4+-3.1 (c=0.79) | +48.1+-2.4 (c=3.77) | +3.45+-1.02 |
| NRCA2 | +20.2+-1.8 | +19.5+-2.2 (c=0.55, n=242) | +18.4+-3.2 (c=2.10) | +21.8+-3.0 (c=3.70) | +0.64+-1.16 |
| NRCA3 | +24.0+-2.1 | +19.5+-3.7 (c=0.66, n=225) | +26.2+-3.8 (c=2.11) | +25.7+-3.9 (c=4.38) | +1.57+-1.43 |
| NRCA4 | +14.1+-1.1 | +10.7+-2.2 (c=0.73, n=291) | +19.7+-2.2 (c=2.39) | +14.4+-2.0 (c=3.58) | +1.38+-1.05 |
| NRCB1 | +2.0+-0.8 | +2.9+-1.4 (c=1.12, n=586) | +1.6+-1.3 (c=1.34) | +1.8+-1.2 (c=2.45) | -0.49+-1.27 |
| NRCB2 | -17.1+-1.1 | -18.0+-1.7 (c=0.82, n=426) | -22.3+-1.5 (c=1.79) | -15.0+-1.9 (c=2.84) | +1.32+-1.24 |
| NRCB3 | -17.5+-0.9 | -16.4+-1.7 (c=1.20, n=695) | -10.8+-1.8 (c=1.45) | -23.2+-1.6 (c=2.28) | -8.46+-2.06 |
| NRCB4 | -16.6+-1.0 | -16.5+-1.8 (c=0.74, n=339) | -13.3+-1.9 (c=2.27) | -20.2+-1.5 (c=3.40) | -1.41+-0.88 |

inverse-variance mean slope over 8 det: +0.42 +- 0.41; excluding nrcb3: +0.79 +- 0.42; chi2 about mean = 34.3

### F150W (control)

| det | all | T1 (blue) | T2 | T3 (red) | slope (ppm/mag) |
|---|---|---|---|---|---|
| NRCA1 | +8.4+-1.8 | +5.9+-3.3 (c=0.42, n=185) | +5.2+-3.0 (c=0.87) | +14.9+-2.7 (c=3.86) | +2.88+-1.09 |
| NRCA2 | +0.6+-1.5 | +0.5+-2.9 (c=0.63, n=235) | -0.1+-3.0 (c=2.31) | -0.1+-2.5 (c=3.71) | -0.18+-1.25 |
| NRCA3 | +2.1+-2.6 | -2.7+-3.1 (c=0.76, n=231) | +11.0+-3.3 (c=2.68) | -2.6+-5.7 (c=4.52) | +2.04+-1.59 |
| NRCA4 | +3.1+-1.6 | +0.6+-2.7 (c=0.78, n=301) | +9.2+-2.4 (c=2.47) | +2.0+-2.6 (c=3.61) | +0.81+-1.30 |
| NRCB1 | +10.2+-0.8 | +9.6+-1.5 (c=1.13, n=556) | +13.9+-1.4 (c=1.37) | +8.3+-1.5 (c=2.50) | -2.34+-1.43 |
| NRCB2 | -7.3+-1.0 | -6.6+-2.1 (c=0.84, n=411) | -12.9+-1.5 (c=1.86) | -5.2+-2.1 (c=2.87) | +0.74+-1.47 |
| NRCB3 | -8.4+-0.9 | -5.6+-2.0 (c=1.23, n=595) | -3.4+-2.1 (c=1.48) | -15.0+-1.4 (c=2.40) | -9.34+-1.88 |
| NRCB4 | -3.1+-1.1 | -0.1+-1.8 (c=0.77, n=355) | -2.7+-1.9 (c=2.42) | -7.6+-2.2 (c=3.47) | -2.58+-1.02 |

inverse-variance mean slope over 8 det: -0.50 +- 0.46; excluding nrcb3: +0.07 +- 0.48; chi2 about mean = 41.9

## C. Anchor control (wd2, no correction needed; scale ppm / rot arcsec after SIGN=-1, which is ~0 for these bands)

| det | F187N scale/rot | F182M scale/rot |
|---|---|---|
| NRCA1 | +0.9/-0.86 | +5.5/-0.80 |
| NRCA2 | -1.3/-0.79 | -3.4/-1.37 |
| NRCA3 | -3.8/-0.19 | -6.4/-0.09 |
| NRCA4 | +1.8/+0.32 | -4.1/+0.05 |
| NRCB1 | +5.4/-0.60 | +4.7/-0.17 |
| NRCB2 | +0.7/-0.26 | -5.8/-0.50 |
| NRCB3 | -5.7/+0.26 | -13.7/+0.57 |
| NRCB4 | +0.9/-0.91 | +0.9/-1.11 |

## Reading

- Field dependence (A): wd1 F200W shows the same sign pattern as wd2 F200W (positive on the A detectors, negative on B2-B4): +37/+25/+44/+45/+22/-4/-22/-10 ppm versus +42/+20/+22/+16/+2/-17/-18/-17 ppm. Detectors A1, A2, B3 agree within about 5 ppm; A3, A4, B1 are 20-30 ppm larger in wd1 and B2 is 13 ppm less negative. The residual therefore appears in both fields at a similar order of magnitude, with some field-to-field scatter of 10-30 ppm that these data do not explain.
- wd1 F150W is not clean: it shows +30 ppm on A4/B1 and +22 on A3 (wd2 F150W is within +-10). So the wd1 per-frame catalogs carry their own band-dependent scale scatter of about 10-30 ppm, and the F200W wd1-vs-wd2 differences are of that size. This limits how tightly "field-independent" can be stated. The F200W minus F150W difference in wd1 is not constant with wd2's, so the wd1 F200W excess is not purely a repeat of the wd2 numbers.
- Colour (B): the 8-detector inverse-variance mean slope is +0.4 +- 0.4 ppm/mag for F200W and -0.5 +- 0.5 ppm/mag for F150W, i.e. consistent with zero. The F200W scale offsets between detectors (up to +42 and down to -18 ppm) persist in all three terciles, so the bulk of the residual does not depend on colour at this precision. Individual detectors scatter more than the quoted errors (chi2 about the mean 34 and 42 for 7 degrees of freedom): NRCB3 shows -8.5 +- 2.1 ppm/mag for F200W and -9.3 +- 1.9 for F150W, and NRCA1 +3.5 +- 1.0 and +2.9 +- 1.1. The F200W and F150W splits use the same anchor stars and colour, so they are not independent; a colour-correlated spatial pattern or anchor term in B3 would produce the same signature as a chromatic term. The tercile ordering is monotonic in only a few detectors (A1, B3 for F200W), and the middle tercile is often off the line, suggesting that bootstrap errors understate the real uncertainty.
- Anchor (C): F187N and F182M against F212N give |scale| <= 6 ppm on all detectors for F187N and <= 14 ppm for F182M (B3 -13.7 ppm), rotation within about 1.4", so the F212N anchor does not generate a systematic of the F200W size (+42 ppm) in the A module. The scatter among these control bands (several ppm, up to 14) indicates the practical noise floor of the method.
- Not shown: why F200W differs; whether the wd1/wd2 differences come from the epoch, the roll, the anchor catalog, or the per-frame catalogs; any dependence on magnitude or on position in the field. Exposures 1-2 only; wd1 visit001 only.
