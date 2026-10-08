# Wing charge migration test: q = cal / (R(g0) * g0) for unsaturated pixels near saturated stars

Data: wd2 NIRCam F150W (visit group 10101) and F200W (12101), exposures 1-2, nrcb1 and nrcb3 (8 frames).
All ramps are SHALLOW4: NFRAMES=4, NGROUPS=7, NINTS=1, TGROUP=53.7 s. Group 0 is the average of 4 frames.
crf SCI and ramp SCI[0,0] share the 2048x2048 detector grid (shape match; SCI-vs-group0 correlation 0.94-0.98).
Ramp located with the `_find_ramp_for` rule (`<stem>_ramp.fits` in the pipeline dir). Scripts: `wingmig_collect.py`, `wingmig_analyze.py`; raw per-frame pickles `raw_*.pkl`, per-star rows `rows_sat.pkl`, `rows_ctl.pkl`.

## Method
1. R(g0) per frame: median of cal/g0 in 8-12 log bins of g0 (200 DN to the 99.9th percentile of g0) over pixels that are finite, not DQ-SATURATED (bit 2), not DO_NOT_USE, g0 > 200 DN, and >= 25 px from any DQ-SATURATED pixel; bins with < 50 px are dropped. Upper bins where R fell below 0.8 x the median R (F200W nrcb1 exp 1 top bin 0.002, ZEROFRAME top bins) are dropped, so q is evaluated only for g0 between ~230 and ~3000-4000 DN. The field R curve is flat to 2-3% (F150W 0.085-0.090, F200W 0.070-0.075). The same curve was built for ZEROFRAME (R_z ~0.22 F150W, ~0.19 F200W).
2. Satstars: positions from the `_resbgsub_m7_satstar_catalog` `x_0,y_0` columns (full-frame detector coordinates; `x_fit,y_fit` are cutout coordinates). Each catalog star is tied to the DQ-SATURATED connected component nearest its position (within 6 px). Wing pixels are non-DQ-SATURATED, non-DO_NOT_USE pixels within 40 px whose nearest saturated pixel belongs to that component, with Euclidean distance d to that pixel <= 20; g0 > 200 DN. d bins use ceil(d): 1, 2, 3, 4-5, 6-8, 9-12, 13-20. q uses the field R curve (log-g0 interpolation, only inside the measured range). Each (star, frame, d bin) pixel median is one unit; the quoted number is the median over units, with the bootstrap standard error over units (500 resamples). The two exposures and the two bands contain the same sky region with dithers, so units are not all independent stars; the quoted errors are statistical and understate scatter from shared stars and the choice of R(g0).
3. Control stars: local maxima (7x7) of group 0 with peak g0 > 1500 DN, >25 px from any DQ-SATURATED pixel, not DO_NOT_USE. Peaks that are single-pixel spikes (median g0 of the 4 nearest neighbours < 0.3 x peak, or neighbour q outside 0.7-1.3) were rejected (cosmic rays/hot pixels gave q ~ 0.03 otherwise). q computed per star in radius bins 1-2, 2-3, 3-4, 4-6, 6-9 px from the peak (non-saturated pixels, g0 > 200). The control sample (about 1000 star-frames) is mostly the 1500-6000 DN peak range, i.e. fainter than the satstars. Control stars share the field-R sample (the 25-px-from-saturation pixels), so q ~ 1 for them is partly by construction; the check is the radial and g0 behaviour.
4. ZEROFRAME check: the same q with the ZEROFRAME (first single frame, un-averaged) and its own field R curve (pixels with zeroframe > 230 DN).
5. Brightness: flux_fit quartiles per band (flux_fit is the satstar fit amplitude; no dolphot mags used), and sat_area bins.

Frame diagnostics (R lists per frame are in `analyze.log`).


### By band (median q, bootstrap err over star-frames, N star-frames)

| group | 1 | 2 | 3 | 4-5 | 6-8 | 9-12 | 13-20 |
|---|---|---|---|---|---|---|---|
| ('F150W',) | 1.071+-0.001 (3045) | 1.053+-0.002 (2912) | 1.049+-0.002 (1476) | 1.033+-0.002 (1371) | 1.021+-0.003 (1144) | 1.008+-0.003 (1205) | 0.983+-0.003 (1512) |
| ('F200W',) | 1.097+-0.002 (1254) | 1.093+-0.001 (1254) | 1.068+-0.001 (1226) | 1.050+-0.001 (1245) | 1.037+-0.001 (1223) | 1.021+-0.002 (1037) | 1.008+-0.002 (978) |

### By band and flux_fit quartile (median q, bootstrap err over star-frames, N star-frames)

| group | 1 | 2 | 3 | 4-5 | 6-8 | 9-12 | 13-20 |
|---|---|---|---|---|---|---|---|
| ('F150W', 'Q1 faint') | 1.028+-0.002 (1057) | 1.015+-0.002 (1025) | 0.984+-0.007 (193) | 0.991+-0.008 (100) | 0.984+-0.017 (149) | 0.936+-0.091 (225) | 0.920+-0.137 (418) |
| ('F150W', 'Q2') | 1.063+-0.002 (921) | 1.052+-0.003 (826) | 0.994+-0.007 (251) | 0.986+-0.007 (267) | 1.003+-0.004 (204) | 0.981+-0.008 (273) | 0.963+-0.013 (424) |
| ('F150W', 'Q3') | 1.122+-0.003 (604) | 1.074+-0.002 (598) | 1.039+-0.003 (569) | 1.018+-0.003 (541) | 0.996+-0.005 (331) | 0.994+-0.005 (251) | 0.984+-0.007 (275) |
| ('F150W', 'Q4 bright') | 1.121+-0.003 (463) | 1.093+-0.002 (463) | 1.083+-0.003 (463) | 1.064+-0.003 (463) | 1.048+-0.002 (460) | 1.032+-0.002 (456) | 1.018+-0.003 (395) |
| ('F200W', 'Q1 faint') | 1.075+-0.002 (343) | 1.083+-0.002 (343) | 1.050+-0.003 (317) | 1.023+-0.003 (334) | 1.002+-0.005 (312) | 0.981+-0.008 (198) | 0.972+-0.010 (214) |
| ('F200W', 'Q2') | 1.090+-0.002 (316) | 1.105+-0.002 (316) | 1.070+-0.004 (314) | 1.042+-0.002 (316) | 1.029+-0.002 (316) | 0.991+-0.006 (256) | 0.996+-0.005 (217) |
| ('F200W', 'Q3') | 1.125+-0.002 (301) | 1.105+-0.003 (301) | 1.068+-0.002 (301) | 1.052+-0.002 (301) | 1.040+-0.002 (301) | 1.010+-0.005 (289) | 0.994+-0.004 (259) |
| ('F200W', 'Q4 bright') | 1.114+-0.003 (294) | 1.089+-0.002 (294) | 1.076+-0.002 (294) | 1.075+-0.002 (294) | 1.062+-0.003 (294) | 1.052+-0.003 (294) | 1.039+-0.002 (288) |

### By sat_area (px) (median q, bootstrap err over star-frames, N star-frames)

| group | 1 | 2 | 3 | 4-5 | 6-8 | 9-12 | 13-20 |
|---|---|---|---|---|---|---|---|
| ('<50',) | 1.065+-0.001 (2983) | 1.054+-0.002 (2869) | 1.040+-0.002 (1446) | 1.019+-0.002 (1356) | 1.006+-0.002 (1148) | 0.982+-0.003 (1066) | 0.968+-0.004 (1400) |
| ('50-150',) | 1.123+-0.002 (979) | 1.089+-0.002 (960) | 1.068+-0.001 (919) | 1.054+-0.001 (923) | 1.040+-0.002 (882) | 1.017+-0.002 (839) | 1.005+-0.004 (760) |
| ('150-400',) | 1.109+-0.004 (241) | 1.095+-0.004 (241) | 1.091+-0.003 (241) | 1.078+-0.003 (241) | 1.062+-0.003 (241) | 1.048+-0.003 (241) | 1.035+-0.003 (238) |
| ('>400',) | 1.110+-0.003 (96) | 1.104+-0.004 (96) | 1.099+-0.006 (96) | 1.087+-0.004 (96) | 1.083+-0.004 (96) | 1.075+-0.005 (96) | 1.063+-0.004 (92) |

### By band, vs ZEROFRAME expectation (median qz, bootstrap err over star-frames, N star-frames)

| group | 1 | 2 | 3 | 4-5 | 6-8 | 9-12 | 13-20 |
|---|---|---|---|---|---|---|---|
| ('F150W',) | 1.059+-0.002 (2190) | 1.066+-0.003 (734) | 1.053+-0.005 (629) | 1.038+-0.004 (583) | 1.027+-0.004 (527) | 1.012+-0.004 (595) | 0.997+-0.005 (763) |
| ('F200W',) | 1.096+-0.002 (1249) | 1.070+-0.003 (1109) | 1.067+-0.003 (634) | 1.039+-0.003 (700) | 1.030+-0.004 (547) | 1.021+-0.005 (457) | 1.009+-0.006 (446) |

### q vs pixel g0 median (star-bin level), d=4-5 and 6-8

- d=4-5, g0 in (200, 400]: q=1.032+-0.002 (N=1798)
- d=4-5, g0 in (400, 800]: q=1.061+-0.002 (N=666)
- d=4-5, g0 in (800, 1600]: q=1.083+-0.004 (N=145)
- d=6-8, g0 in (200, 400]: q=1.025+-0.001 (N=1756)
- d=6-8, g0 in (400, 800]: q=1.053+-0.002 (N=475)
- d=6-8, g0 in (800, 1600]: q=1.047+-0.012 (N=124)

### Control stars: q vs radius from peak (all control stars, per-star-bin medians)

| band | 1-2 | 2-3 | 3-4 | 4-6 | 6-9 |
|---|---|---|---|---|---|
| F150W | 1.026+-0.002 (740) | 0.998+-0.004 (434) | 0.997+-0.005 (264) | 0.998+-0.008 (69) | 0.994+-0.012 (119) |
| F200W | 1.020+-0.002 (721) | 1.019+-0.002 (703) | 0.988+-0.002 (478) | 0.985+-0.004 (359) | 0.985+-0.008 (138) |

### Control stars by peak g0 (r 4-6 px)

- F150W peak g0 (1500.0, 3000.0]: q=0.993+-0.015 (N=46)
- F150W peak g0 (3000.0, 6000.0]: q=0.999+-0.088 (N=23)
- F200W peak g0 (1500.0, 3000.0]: q=0.979+-0.007 (N=176)
- F200W peak g0 (3000.0, 6000.0]: q=0.988+-0.004 (N=183)

### Matched-g0 comparison: satstar wings (d 4-8) vs control (r 4-9)

- g0 200-400: satstar q=1.028+-0.001 (N=3554); control q=0.986+-0.003 (N=518)
- g0 400-800: satstar q=1.058+-0.002 (N=1141); control q=0.993+-0.007 (N=121)
- g0 800-1600: satstar q=1.076+-0.003 (N=269); control q=0.996+-0.235 (N=40)

### Control stars matched to satstar g0 (r 4-6, g0med in same bins as satstar wings)

- control g0 (200, 400]: q=0.983+-0.004 (N=371)
- control g0 (400, 800]: q=1.005+-0.009 (N=43)
- control g0 (800, 1600]: q=1.002+-0.139 (N=12)

## Results summary
- q exceeds 1 in the non-saturated wings of satstars. Band medians: F150W 1.071 (d=1), 1.049 (d=3), 1.033 (d=4-5), 1.021 (d=6-8), 1.008 (d=9-12), 0.983 (d=13-20). F200W 1.097, 1.068, 1.050, 1.037, 1.021, 1.008. The excess decays with d, reaching ~1 at d~10-15 px (faint stars: by d~3).
- Brightness dependence is strong. Brightest flux_fit quartile: F150W 1.121 (d=1), 1.083 (d=3), 1.064 (d=4-5), 1.048 (d=6-8), 1.032 (d=9-12), 1.018 (d=13-20); F200W 1.114, 1.076, 1.075, 1.062, 1.052, 1.039. Faintest quartile: F150W 1.03 (d=1-2), ~0.98 beyond d=3; F200W 1.075-1.08 at d=1-2, ~1.0 beyond d=6. sat_area > 400 px: 1.110 (d=1) to 1.063 (d=13-20); sat_area < 50 px: 1.065 to 0.968.
- At matched g0 the control stars give q = 0.986 (g0 200-400), 0.993 (400-800), 0.996 (800-1600, large error, N=40) at r 4-9 px, versus satstar wings (d 4-8) 1.028, 1.058, 1.076. So the satstar-wing excess over matched-g0 controls is ~4% (200-400 DN), ~6.5% (400-800), ~8% (800-1600). Controls at r 1-3 px read 1.02-1.03 (F150W 1.026, F200W 1.020), and 0.985-0.999 at r >= 3 px, independent of peak g0 within 1500-6000 DN.
- Against the ZEROFRAME expectation the pattern is the same: F150W 1.059, 1.053, 1.038, 1.027, 1.012, 0.997 and F200W 1.096, 1.067, 1.039, 1.030, 1.021, 1.009 (d=1, 3, 4-5, 6-8, 9-12, 13-20), so the excess is not produced by group 0 averaging over 4 frames. (The ZEROFRAME sample is smaller: it requires zeroframe > ~230 DN.)

## Relation to the fit bias (5-20%)
The measured wing excess is 1-3% at d=6-8 for the typical star and up to 5-6% (F150W) / 6% (F200W) at d=6-12 for the brightest quartile, 8-12% at d=1-3 for bright stars. The fit bias was quantified at 0.14-0.28" (4.5-9 px from the star) as 13-22% of the fitted amplitude; the wing pixels used by the fit lie at 4.5-9 px from the star centre, which for a saturated radius of ~5-12 px (sat_area 100-450) corresponds to d ~ 0-5 px from the saturated edge. In that range the measured q excess is ~3-12%, depending on brightness. This accounts for roughly a third to a half of the 13-22% excess for the brightest stars and less for fainter ones. It is a partial explanation: the effect has the right sign, the right brightness dependence and radial decline, but its size falls below the fit-residual excess.

## Caveats
- q is a ratio of the full-ramp fit to the first-read prediction scaled with a field R(g0). R(g0) varies only ~2% across g0, but the choice of reference (field pixels away from saturation, including sky) sets the zero point to ~1-2%; controls (0.985-1.0 at r > 3) imply a ~1% low zero-point offset, and 1.02-1.03 at r=1-2 px.
- Some of the excess at fixed d correlates with g0 (q rises with g0: 1.03, 1.06, 1.08 for g0 200-400, 400-800, 800-1600 at d=4-5), as expected if the effect scales with the core charge, but g0 and star brightness are correlated and not separated here. Bright stars have larger saturated regions, so d is not equivalent to radius.
- Other effects can raise cal above R*g0 in the wings: nonlinearity/classical non-linearity residuals at high flux, the brighter-fatter effect and IPC acting on the group-0 versus full-ramp weighting, ramp-fit weighting with unflagged cosmic-ray jumps, and the 4-frame averaging in group 0. The test does not separate charge migration from these mechanisms. Brighter-fatter would also affect unsaturated bright controls; controls peak below ~6000 DN and show no excess, but they are fainter than the satstars.
- Pixels in the wings of neighbouring stars or on structured background may add noise; no deblending was done beyond nearest-saturated-component assignment. Cosmic-ray pixels (q ~ 0.03) are suppressed by the per-bin median; this fails for bins with 1-2 pixels (outer g0 > 1600 bins were dropped from the g0 table).
- g0 range of validity ~230-3500 DN (limited by the field R curve), so the innermost, brightest wing pixels (g0 > ~3500) are not tested.
- Only 8 frames (2 exposures x 2 detectors x 2 bands); exposures 3-4 not used. Errors are statistical over star-frame units (dithered duplicates are not independent).
- The 0.8 x median R truncation was applied in the analysis step; the g0 ceiling in the analysis follows the field R measurement, not the pipeline's `SATSTAR_ZF_RCURVE_GUARD` logic.

Figure: `wingmig.png` (left, middle: q vs d per flux_fit quartile for F150W, F200W; right: control q vs radius from peak).
