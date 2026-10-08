# nrcb3 satstar offset: detector or environment

Scripts: build.py (row tables), envtest.py (A fallback, B), refdata.py + g0.py + g0cmp.py (C), score_det.py (D), fig.py. Figure: nrcb3.png. Full tables: nrcb3_tables.md. Satstar rows, matching, ZP and c_b follow ovlscale_det.py (row counts reproduced: 8522 / 6581 / 3022 / 2553 for F150W / F162M / F182M / F200W). Statistics are star-level medians (median over a star's 3-4 exposures); errors are bootstrap errors over stars. dm = m_S + c_b - m_dolphot - ZP; negative = ours bright.

## A. Within-star detector test
No star qualifies. Zero of 2326 / 1782 / 824 / 722 stars (F150W / F162M / F182M / F200W) have S rows on more than one detector, and no star has S rows on different SW detectors in different bands either. The four exposures of each band are effectively one pointing: the per-star sky scatter among exposures has median 0.002" and maximum 0.09", so a star stays on the same detector pixels in every exposure. The paired test (median dm on nrcb3 minus median dm on another detector, for nrcb3 and for nrcb1) has N = 0 for both.

Fallback: nearest-pair test across the nrcb1/nrcb3 boundary. Each nrcb3 star is paired with the nearest nrcb1 star within |delta m_dolphot| < 0.25 mag (optionally also matched in 2" density within max(2, 30%) and annulus background within 0.15 dex). Median dm(b3) - dm(b1):

| band | all stars, mag-matched | pairs < 20", mag-matched (N, median sep) | pairs < 40", mag+density+bkg (N) |
|---|---|---|---|
| F150W | -0.057 +/- 0.002 | -0.044 +/- 0.003 (207, 12.6") | -0.045 +/- 0.002 (387) |
| F162M | -0.047 +/- 0.002 | -0.037 +/- 0.005 (190, 13.6") | -0.041 +/- 0.003 (293) |
| F182M | -0.044 +/- 0.003 | -0.047 +/- 0.005 (124, 14.1") | -0.044 +/- 0.004 (153) |
| F200W | -0.072 +/- 0.003 | -0.088 +/- 0.005 (94, 13.8") | -0.092 +/- 0.004 (123) |

The offset persists between spatial neighbours on the two sides of the boundary. For F150W/F162M the paired value is 0.010 smaller than the all-star value, which leaves a small position-dependent component; F200W goes the other way.

## B. Environment
Environment per S row: dolphot neighbours with mag < own+3 within 1" and 2" (same band, self excluded), nearest brighter neighbour distance, annulus background (sigma-clipped median of the crf image, 1.5-2.5"; the clipping masks stars only through the 2.5-sigma rejection), stored satstar local_bkg, and distance to the detector edge. The stored local_bkg and the annulus give the same result.

nrcb3 - nrcb1 (star medians), raw and after stratifying (weighted mean over cells with >= 3 stars per detector):

| band | nrcb1 / nrcb3 median | raw | 1-mag strata | + 2" density | + annulus bkg | + density + bkg | + density + bkg + edge | OLS: mag, log n2, log bkg, edge, nearest-brighter dist |
|---|---|---|---|---|---|---|---|---|
| F150W | -0.017 / -0.075 | -0.058 | -0.057 | -0.060 | -0.055 | -0.055 | -0.056 | -0.055 +/- 0.004 |
| F162M | -0.004 / -0.055 | -0.050 | -0.047 | -0.048 | -0.046 | -0.043 | -0.039 | -0.045 +/- 0.003 |
| F182M | -0.020 / -0.065 | -0.045 | -0.044 | -0.048 | -0.045 | -0.044 | -0.042 | -0.049 +/- 0.003 |
| F200W | -0.021 / -0.093 | -0.072 | -0.072 | -0.076 | -0.075 | -0.079 | -0.071 | -0.073 +/- 0.004 |

Bootstrap errors are 0.002-0.007. nrcb3 - nrcb1 stays between -0.039 and -0.079 in every density, background, edge-distance and brighter-neighbour bin (tables in nrcb3_tables.md). The OLS coefficients on density (-0.000 +/- 0.001 per sd, F150W) and background (-0.002 +/- 0.003) are consistent with zero. Per-detector medians vary little with density: nrcb3 F150W goes -0.081 (0-3 neighbours) to -0.069 (8-26 neighbours), nrcb1 -0.018 to -0.015. Background: nrcb1 F150W goes from -0.005 (lowest quartile) to -0.042 (highest), nrcb3 stays at -0.068 to -0.077, so the difference narrows from -0.07 to -0.04 in the highest-background bin (N 38 for nrcb1). Edge distance: the difference is -0.046 within 50 px of the edge and -0.064 beyond 400 px (F150W).

Daophot control (unsaturated stars in the ZP window, same binning): nrcb3 - nrcb1 = -0.018 / -0.008 / -0.011 / -0.028 (raw), and -0.014 / -0.007 / -0.013 / -0.031 after mag + density + bkg stratification. The satstar excess over the control (stratified) is -0.041 / -0.036 / -0.031 / -0.048 (F150W / F162M / F182M / F200W).

## C. Detector reference data
References used by the SW frames (CRDS context jwst_1595 for F150W, jwst_1568 for the others): nrcb1 saturation_0106 / linearity_0057 / gain_0090, nrcb2 0108 / 0048 / 0088, nrcb3 0109 / 0054 / 0094, nrcb4 0112 / 0050 / 0095; the same files for every band.

| det | saturation, median (DN) | at star rows | linearity correction at 50/70/90% sat | gain |
|---|---|---|---|---|
| nrca1 | 54519 | 54945 | 1.042 / 1.062 / 1.106 | 2.05 |
| nrca2 | 56104 | 56348 | 1.042 / 1.070 / 1.162 | 2.05 |
| nrca3 | 53029 | 53164 | 1.036 / 1.046 / 1.092 | 2.05 |
| nrca4 | 53929 | 53987 | 1.021 / 1.030 / 1.098 | 2.05 |
| nrcb1 | 57872 | 57948 | 1.027 / 1.045 / 1.155 | 2.05 |
| nrcb2 | 57850 | 57818 | 1.050 / 1.075 / 1.152 | 2.05 |
| nrcb3 | 57446 | 57238 | 1.035 / 1.054 / 1.144 | 2.05 |
| nrcb4 | 57935 | 57991 | 1.044 / 1.065 / 1.151 | 2.05 |

(F150W values; saturation and gain are band independent; the linearity correction is the file's polynomial divided by DN at the stated fraction of the saturation level.) nrcb3 sits inside the nrcb1/b2/b4 spread, which is as large as the nrcb1-nrcb3 difference (0.008 at 50%, 0.010 at 90%); the nrca detectors have lower saturation levels (53-56 kDN) than all of module B.

Group-0 core DN (uncal, 5x5 peak above a 15-25 px ring median), compared at matched dolphot magnitude as 2.5 log10(b3/b1): group 0 +0.021 +/- 0.010 (F150W), +0.039 +/- 0.013 (F162M), +0.060 +/- 0.013 (F182M), +0.044 +/- 0.012 (F200W); ZEROFRAME +0.035 / +0.040 / +0.069 / +0.059 (errors 0.012-0.021). nrcb3 cores carry 2-7% more raw DN than nrcb1 cores of the same dolphot magnitude, in the same direction as the satstar excess but not a derived explanation of it (raw DN has not passed the flat and photom steps). The group-1 minus group-0 step does not differ consistently. The fraction of core peak pixels at the A/D limit (>= 64000) in the last group is 0.04-0.10 and similar on both detectors; no peak pixel is flagged saturated in group 0 on any detector.

## D. Satrefit round 3 by detector (F150W, F200W; refit frames exist only for nrcb1/nrcb3)
Median dm over all magnitudes (star medians; per-bin tables in nrcb3_tables.md):

| variant | F150W nrcb1 | F150W nrcb3 | b3 - b1 | F200W nrcb1 | F200W nrcb3 | b3 - b1 |
|---|---|---|---|---|---|---|
| final | -0.017 | -0.075 | -0.058 | -0.021 | -0.093 | -0.072 |
| uncapped | -0.060 | -0.089 | -0.029 | -0.069 | -0.116 | -0.047 |
| bgfree | -0.049 | -0.073 | -0.024 | -0.055 | -0.097 | -0.042 |
| bgfree+cap | -0.015 | -0.066 | -0.051 | -0.019 | -0.084 | -0.065 |
| rw12 | -0.031 | -0.076 | -0.045 | -0.036 | -0.097 | -0.061 |
| rw12+cap | -0.008 | -0.068 | -0.060 | -0.015 | -0.086 | -0.072 |
| rw12+bgfree | -0.020 | -0.064 | -0.044 | -0.026 | -0.085 | -0.059 |
| rw12+bgfree+cap | -0.004 | -0.058 | -0.055 | -0.011 | -0.078 | -0.066 |
| bgfree+v7b+cap | +0.009 | -0.032 | -0.041 | +0.012 | -0.047 | -0.058 |

Every variant moves both detectors together; the b3 - b1 difference stays between -0.024 and -0.072 and does not close in any variant. Uncapped and bgfree narrow it to -0.02..-0.05, because they move nrcb1 by 0.03-0.04 and nrcb3 by 0.00-0.02 (the cap rarely binds on nrcb3). The best variant (bgfree+v7b+cap) brings nrcb1 to +0.009/+0.012 and nrcb3 to -0.032/-0.047. The difference is present in each magnitude bin (F150W final: -0.071, -0.061, -0.058, -0.054, -0.058 from 14-15 to 18-19).

R(g0) per frame: mean R over g0 = 400-2500 DN is 0.0878 (nrcb1) vs 0.0888 (nrcb3) in F150W (+1.2%, +0.013 mag) and 0.0741 vs 0.0737 in F200W (-0.5%). At g0 = 250-600 DN nrcb3/nrcb1 is 1.035-1.023 (F150W) and 1.026-1.006 (F200W); at 1300-3300 DN it is 1.002-0.994 and 0.986-0.981. The curves differ in shape at the 1-3% level; the intrinsic scatter parameter is 0.05-0.07 on nrcb3 and 0.06-0.10 on nrcb1, and the fitted read-noise parameters differ (sig_low 8.7 vs 11.0 in F150W). An R(g0) normalisation difference of 1% corresponds to 0.01 mag and covers less than a quarter of the 0.045-0.075 offset.

## Assessment
- Environment: density, background, nearest-brighter-neighbour distance and edge distance each leave nrcb3 - nrcb1 at -0.04..-0.08 with errors of 0.002-0.007, and the regression coefficients on density and background are zero within errors. Environment explains none of the offset in this sample (the one structured dependence is a smaller difference in the highest-background bin, driven by nrcb1). This favours a detector-level or per-detector-calibration cause.
- Detector: the pair test across the boundary (spatial neighbours, matched mag/density/bkg) keeps 75-125% of the global offset; the daophot control of the same stars carries only 0.01-0.03 of it, which places the excess in the saturated-core recovery (R(g0), core wing, cap) rather than in the PSF-fit photometry. Saturation, linearity and gain references of nrcb3 are ordinary. Two items differ on nrcb3 and match the sign of the offset: raw core DN at fixed dolphot magnitude is 2-7% higher, and the R(g0) curve is 1-3.5% higher below 1000 DN and 0.2-2% lower above. A per-detector effect in the DN-to-flux step (flat/photom at the core pixels), in charge migration/brighter-fatter, or in R(g0) are the candidates; these data do not separate them.
- Strength: environment excluded at the 0.01 mag level in this single-pointing sample; detector attribution over a within-star test is untested (N = 0). The offset is also not removed by any round-3 variant.

## Caveats
- A single pointing means no star crosses detectors; the within-star test needs a dithered or mosaicked dataset.
- nrcb1 and nrcb3 differ in sky position, so detector and position within the cluster cannot be separated beyond the pair test (separations 8-28"). Dolphot is the yardstick for both and its own detector dependence is bounded only by the control (0.01-0.03).
- The annulus background is not star-masked beyond sigma clipping; dolphot density is incomplete in the cluster core, so the density measure underestimates crowding there.
- Group-0 DN comparisons use raw (uncal) counts and a ring-median bias estimate; first saturated group is uninformative (the 7-group ramps reach the A/D limit at most in the last group).
- D covers F150W and F200W only (the round-3 outputs for the LW detectors were not complete); round-3 variants are applied row by row from the a_* amplitudes following score3.py.
