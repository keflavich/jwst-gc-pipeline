# Satrefit round 3: charge-migration rewrite of the unsaturated wing pixels

Arm main2, commit 5434f5e7 (read-only copy). New files only: run_frames3.py, score3.py, plot3.py, out3/, satrefit3.md, satrefit3.png.
Frames: F150W and F200W exposures 1-4, nrcb1 + nrcb3 (16 frames); F250M and F300M exposures 1-4, nrcblong (8 frames).

## Method

Before the amplitude solve, non-saturated cutout pixels within D px of any DQ-SATURATED pixel are rewritten to R(g0)*g0.

- g0: ramp SCI[0,0] (group 0, average of 4 frames).
- DQ SATURATED: the frame DQ after `correct_dq_first_group_saturation` (as prep_frame); distance from a pixel to the nearest SATURATED pixel by a Euclidean distance transform, D in {3, 6, 12, 20, inf}.
- The rewrite is applied as a delta on the neighbour-subtracted cutout (cut += R*g0 - data), so the brighter-first model subtraction is kept. The background annulus, scatter floor and amplitude solve then run on the rewritten cutout. Pixels already masked, in the ZEROFRAME rim, in the deep core, or non-finite are untouched.
- R(g0): the wingmig construction (median cal/g0 in 12 log bins from 200 DN to the g0 99.9th percentile, over finite, non-SATURATED, non-DO_NOT_USE pixels at least 25 px from any saturated pixel; bins with <50 px dropped; upper bins with R < 0.8 x the median dropped). This is the curve that defined q. The pipeline's own curve in `zeroframe_recover_saturated` starts at R_g0_min = 2000 DN (8 bins, step guard, SATURATED-pixel check). That curve covers only g0 >= 2000 DN, so it would leave most wing pixels at d >= 3 px outside its range. It was not reproduced separately in this round. The curve is built per frame and written to out3/*_rcurve.txt.
- Rewritten pixels satisfy g0 > 5 x sigma0 and 200 DN <= g0 <= the highest bin centre of R (the "inside the measured range" rule). The 5 sigma0 floor (about 45-60 DN) lies below the 200 DN start of the curve, so the lower limit is 200 DN. The upper limit is the 99.9th percentile of g0 away from saturation (2700-4000 DN per frame, listed below). Wing pixels at d = 1-4 px with g0 above that limit keep the crf value. This limit is the main reason the rewritten fraction is small.
- Group-0 noise sigma0 from the field: robust sigma of adjacent-pixel differences / sqrt(2) over unsaturated pixels between the 1st and 20th percentile of g0. Values (DN): F150W nrcb1 10.9-11.0, nrcb3 8.6-8.7; F200W nrcb1 11.3-11.4, nrcb3 10.1-10.2; F250M nrcblong 9.4-9.5; F300M nrcblong 11.9. Read-noise part rn0 = sqrt(sigma0^2 - g_low/gain): 8.3-10.9 DN SW, 8.3 (F250M) and 9.9 (F300M) LW.
- Gain: not in the crf or ramp headers. Used 2.0 e/DN (SW) and 1.8 e/DN (LW) as stated assumptions; the gain enters only the Poisson term.
- Error of a rewritten pixel (MJy/sr): sqrt( (R*sqrt(rn0^2 + g0/gain))^2 + (s*R*g0)^2 ). s is the R-curve scatter: per-bin robust scatter of cal/g0 minus the expected group-0 noise in quadrature, median over bins with g0 >= 1000 DN. Values: F150W 0.05-0.08, F200W 0.047-0.059, F300M 0.082-0.099, F250M 0.13-0.20. The cal-side noise is not removed from s, so s overstates the intrinsic scatter by an unknown amount. The solve then adds the annulus scatter in quadrature as for every variant.
- Variants: rw{D} and rw{D}+bgfree (bgfree = free constant background in the linear solve). +cap = min(variant, catalog cap); the cap reads the rim, which is unchanged, and the PSF is unchanged, so peak ratio is 1.
- Baseline check: a_base and a_bgfree of out3 equal the round-1 values (max relative difference 0.0, 40-row test) and a_base equals a_cat to 3e-5.
- dm = ours - dolphot - ZP as in score2: dm_variant = dm_uncapped + median per-star shift of -2.5 log10(a_v/a_base).

Highest R(g0) bin centre (DN) per frame, F150W nrcb1 exp1-4: 3506, 3668, 2872, 3343; nrcb3: 3924, 4030, 3582, 3458. F200W nrcb1: 3003, 3806, 3473, 3205; nrcb3: 3647, 3794, 3765, 3288. F250M: 2844, 3883, 2690, 3123. F300M: 3385, 3763, 3028, 3761. R is flat to about 3 % between 300 and 3000 DN in F150W (0.087-0.090) and F200W (0.072-0.074).

## Results (measured)

Tables: per-bin median dm (MAD), all-star median / MAD, trend (faintest bin - brightest bin), step (faintest bin - unsaturated reference). Magnitude bins are those of score2. Full tables, including D = 6 and 20, in out3/tables3_<band>.md.

#### F150W: 2311 satstar-replaced matched stars; 1471 with rows in the refit frames; 1471 with every variant finite
Satstar faint edge 18.46; unsaturated reference 18.46-19.46 mag: median dm +0.004 (N=1309).

| variant | 14-15 | 15-16 | 16-17 | 17-18 | 18-19 | all med / MAD | trend (faint-bright bin) | step (faint bin - unsat ref) |
|---|---|---|---|---|---|---|---|---|
| final | -0.095 (0.070) | -0.084 (0.048) | -0.057 (0.045) | -0.040 (0.044) | -0.032 (0.047) | -0.047 / 0.047 | +0.063 | -0.035 |
| uncapped | -0.142 (0.063) | -0.122 (0.055) | -0.101 (0.046) | -0.066 (0.045) | -0.047 (0.039) | -0.078 / 0.053 | +0.095 | -0.051 |
| base+cap | -0.095 (0.070) | -0.084 (0.048) | -0.057 (0.045) | -0.040 (0.044) | -0.032 (0.047) | -0.047 / 0.047 | +0.063 | -0.035 |
| bgfree | -0.117 (0.058) | -0.099 (0.053) | -0.083 (0.046) | -0.054 (0.043) | -0.038 (0.039) | -0.064 / 0.048 | +0.079 | -0.042 |
| bgfree+cap | -0.069 (0.052) | -0.070 (0.042) | -0.050 (0.040) | -0.036 (0.041) | -0.026 (0.042) | -0.042 / 0.043 | +0.043 | -0.030 |
| bgfree+v7b+cap | -0.021 (0.030) | -0.035 (0.038) | -0.024 (0.033) | -0.009 (0.041) | -0.002 (0.041) | -0.015 / 0.041 | +0.018 | -0.006 |
| rw12 | -0.095 (0.048) | -0.092 (0.046) | -0.078 (0.047) | -0.045 (0.051) | -0.030 (0.047) | -0.058 / 0.052 | +0.066 | -0.033 |
| rw12+cap | -0.071 (0.052) | -0.069 (0.044) | -0.048 (0.044) | -0.030 (0.047) | -0.025 (0.051) | -0.040 / 0.050 | +0.046 | -0.029 |
| rw12+bgfree | -0.070 (0.032) | -0.074 (0.044) | -0.063 (0.043) | -0.034 (0.048) | -0.022 (0.045) | -0.047 / 0.050 | +0.048 | -0.026 |
| rw12+bgfree+cap | -0.051 (0.041) | -0.058 (0.044) | -0.043 (0.037) | -0.023 (0.046) | -0.020 (0.048) | -0.034 / 0.047 | +0.031 | -0.023 |
| rwinf | -0.095 (0.050) | -0.091 (0.046) | -0.078 (0.047) | -0.044 (0.050) | -0.029 (0.046) | -0.058 / 0.052 | +0.066 | -0.033 |
| rwinf+cap | -0.071 (0.052) | -0.069 (0.044) | -0.048 (0.044) | -0.029 (0.047) | -0.025 (0.052) | -0.040 / 0.050 | +0.046 | -0.029 |
| rwinf+bgfree | -0.072 (0.036) | -0.075 (0.043) | -0.063 (0.043) | -0.036 (0.048) | -0.024 (0.045) | -0.048 / 0.050 | +0.049 | -0.027 |
| rwinf+bgfree+cap | -0.054 (0.044) | -0.059 (0.044) | -0.044 (0.038) | -0.024 (0.047) | -0.021 (0.049) | -0.034 / 0.047 | +0.033 | -0.025 |

N per bin: 57, 159, 385, 562, 307

Fraction of fit pixels rewritten (median over stars per bin) and amplitude ratio variant/base (median over stars per bin, uncapped):

| variant | quantity | 14-15 | 15-16 | 16-17 | 17-18 | 18-19 |
|---|---|---|---|---|---|---|
| rw12 | rewritten frac | 0.044 | 0.031 | 0.015 | 0.010 | 0.007 |
| rw12 | a_v/a_base | 0.958 | 0.969 | 0.980 | 0.983 | 0.988 |
| rw12+bgfree | a_v/a_base | 0.934 | 0.951 | 0.966 | 0.973 | 0.980 |
| rwinf | rewritten frac | 0.050 | 0.036 | 0.018 | 0.013 | 0.010 |
| rwinf | a_v/a_base | 0.956 | 0.967 | 0.979 | 0.983 | 0.988 |
| rwinf+bgfree | a_v/a_base | 0.935 | 0.952 | 0.967 | 0.974 | 0.980 |

Capped rows in the refit frames: 3666 of 6169.

#### F200W: 721 satstar-replaced matched stars; 515 with rows in the refit frames; 515 with every variant finite
Satstar faint edge 16.25; unsaturated reference 16.25-17.25 mag: median dm -0.017 (N=1130).

| variant | 13-14.5 | 14.5-15 | 15-15.5 | 15.5-16 | 16-18 | all med / MAD | trend (faint-bright bin) | step (faint bin - unsat ref) |
|---|---|---|---|---|---|---|---|---|
| final | -0.048 (0.062) | -0.074 (0.055) | -0.090 (0.054) | -0.063 (0.046) | -0.040 (0.052) | -0.066 / 0.054 | +0.008 | -0.023 |
| uncapped | -0.096 (0.057) | -0.095 (0.054) | -0.113 (0.040) | -0.090 (0.040) | -0.052 (0.048) | -0.097 / 0.050 | +0.044 | -0.035 |
| base+cap | -0.048 (0.062) | -0.074 (0.055) | -0.090 (0.054) | -0.063 (0.046) | -0.040 (0.052) | -0.066 / 0.054 | +0.008 | -0.023 |
| bgfree | -0.079 (0.049) | -0.078 (0.046) | -0.095 (0.045) | -0.074 (0.037) | -0.034 (0.055) | -0.079 / 0.047 | +0.045 | -0.018 |
| bgfree+cap | -0.029 (0.060) | -0.060 (0.051) | -0.070 (0.050) | -0.055 (0.043) | -0.031 (0.055) | -0.056 / 0.051 | -0.002 | -0.014 |
| bgfree+v7b+cap | -0.004 (0.060) | -0.016 (0.047) | -0.034 (0.041) | -0.018 (0.038) | +0.006 (0.051) | -0.020 / 0.045 | +0.009 | +0.022 |
| rw12 | -0.081 (0.060) | -0.073 (0.060) | -0.079 (0.053) | -0.061 (0.044) | -0.051 (0.062) | -0.073 / 0.055 | +0.030 | -0.035 |
| rw12+cap | -0.035 (0.067) | -0.068 (0.051) | -0.070 (0.055) | -0.053 (0.044) | -0.038 (0.055) | -0.055 / 0.052 | -0.003 | -0.021 |
| rw12+bgfree | -0.063 (0.056) | -0.063 (0.056) | -0.068 (0.049) | -0.052 (0.040) | -0.039 (0.059) | -0.058 / 0.048 | +0.024 | -0.022 |
| rw12+bgfree+cap | -0.027 (0.062) | -0.054 (0.049) | -0.059 (0.050) | -0.046 (0.042) | -0.034 (0.060) | -0.048 / 0.052 | -0.007 | -0.017 |
| rwinf | -0.081 (0.059) | -0.073 (0.060) | -0.080 (0.053) | -0.061 (0.044) | -0.050 (0.058) | -0.072 / 0.054 | +0.031 | -0.034 |
| rwinf+cap | -0.036 (0.067) | -0.068 (0.052) | -0.069 (0.056) | -0.053 (0.045) | -0.038 (0.055) | -0.055 / 0.052 | -0.002 | -0.021 |
| rwinf+bgfree | -0.064 (0.056) | -0.064 (0.055) | -0.070 (0.048) | -0.053 (0.041) | -0.040 (0.059) | -0.059 / 0.048 | +0.024 | -0.023 |
| rwinf+bgfree+cap | -0.028 (0.062) | -0.055 (0.050) | -0.059 (0.050) | -0.048 (0.042) | -0.034 (0.062) | -0.049 / 0.051 | -0.006 | -0.017 |

N per bin: 80, 85, 147, 167, 36

Fraction of fit pixels rewritten (median over stars per bin) and amplitude ratio variant/base (median over stars per bin, uncapped):

| variant | quantity | 13-14.5 | 14.5-15 | 15-15.5 | 15.5-16 | 16-18 |
|---|---|---|---|---|---|---|
| rw12 | rewritten frac | 0.131 | 0.079 | 0.062 | 0.055 | 0.038 |
| rw12 | a_v/a_base | 0.980 | 0.981 | 0.970 | 0.978 | 0.993 |
| rw12+bgfree | a_v/a_base | 0.967 | 0.970 | 0.959 | 0.970 | 0.977 |
| rwinf | rewritten frac | 0.157 | 0.087 | 0.070 | 0.067 | 0.054 |
| rwinf | a_v/a_base | 0.980 | 0.980 | 0.969 | 0.978 | 0.992 |
| rwinf+bgfree | a_v/a_base | 0.968 | 0.971 | 0.960 | 0.971 | 0.978 |

Capped rows in the refit frames: 1537 of 2515.

#### F250M: 1129 satstar-replaced matched stars; 905 with rows in the refit frames; 905 with every variant finite
Satstar faint edge 16.46; unsaturated reference 16.46-17.46 mag: median dm +0.003 (N=1414).

| variant | 10-13 | 13-14 | 14-15 | 15-16 | 16-18 | all med / MAD | trend (faint-bright bin) | step (faint bin - unsat ref) |
|---|---|---|---|---|---|---|---|---|
| final | +0.125 (0.059) | +0.039 (0.036) | +0.021 (0.032) | +0.028 (0.034) | +0.031 (0.034) | +0.030 / 0.035 | -0.094 | +0.028 |
| uncapped | -0.032 (0.104) | +0.001 (0.053) | -0.001 (0.036) | +0.020 (0.036) | +0.029 (0.033) | +0.016 / 0.038 | +0.060 | +0.026 |
| base+cap | +0.125 (0.059) | +0.039 (0.036) | +0.021 (0.032) | +0.028 (0.034) | +0.031 (0.034) | +0.030 / 0.035 | -0.094 | +0.028 |
| bgfree | -0.007 (0.086) | +0.024 (0.049) | +0.023 (0.040) | +0.036 (0.032) | +0.037 (0.034) | +0.032 / 0.035 | +0.044 | +0.034 |
| bgfree+cap | +0.140 (0.088) | +0.049 (0.040) | +0.035 (0.033) | +0.041 (0.030) | +0.039 (0.035) | +0.040 / 0.035 | -0.101 | +0.036 |
| bgfree+v7b+cap | +0.156 (0.085) | +0.060 (0.040) | +0.041 (0.030) | +0.045 (0.030) | +0.040 (0.035) | +0.045 / 0.035 | -0.116 | +0.037 |
| rw12 | -0.016 (0.100) | +0.021 (0.047) | +0.013 (0.033) | +0.032 (0.035) | +0.042 (0.036) | +0.029 / 0.037 | +0.058 | +0.039 |
| rw12+cap | +0.125 (0.064) | +0.047 (0.042) | +0.027 (0.030) | +0.038 (0.034) | +0.044 (0.037) | +0.039 / 0.037 | -0.081 | +0.041 |
| rw12+bgfree | +0.008 (0.084) | +0.037 (0.043) | +0.027 (0.031) | +0.042 (0.032) | +0.047 (0.033) | +0.040 / 0.034 | +0.039 | +0.045 |
| rw12+bgfree+cap | +0.143 (0.062) | +0.054 (0.041) | +0.037 (0.034) | +0.048 (0.031) | +0.049 (0.035) | +0.047 / 0.034 | -0.094 | +0.046 |
| rwinf | -0.016 (0.101) | +0.021 (0.046) | +0.014 (0.034) | +0.033 (0.034) | +0.043 (0.036) | +0.029 / 0.036 | +0.058 | +0.040 |
| rwinf+cap | +0.125 (0.065) | +0.047 (0.042) | +0.028 (0.030) | +0.038 (0.034) | +0.044 (0.036) | +0.039 / 0.037 | -0.081 | +0.041 |
| rwinf+bgfree | +0.006 (0.083) | +0.035 (0.044) | +0.026 (0.030) | +0.040 (0.032) | +0.046 (0.034) | +0.038 / 0.034 | +0.040 | +0.044 |
| rwinf+bgfree+cap | +0.143 (0.071) | +0.054 (0.040) | +0.035 (0.033) | +0.046 (0.031) | +0.047 (0.034) | +0.046 / 0.034 | -0.096 | +0.045 |

N per bin: 28, 80, 189, 374, 234

Fraction of fit pixels rewritten (median over stars per bin) and amplitude ratio variant/base (median over stars per bin, uncapped):

| variant | quantity | 10-13 | 13-14 | 14-15 | 15-16 | 16-18 |
|---|---|---|---|---|---|---|
| rw12 | rewritten frac | 0.023 | 0.034 | 0.016 | 0.010 | 0.007 |
| rw12 | a_v/a_base | 0.981 | 0.984 | 0.990 | 0.990 | 0.988 |
| rw12+bgfree | a_v/a_base | 0.958 | 0.970 | 0.973 | 0.979 | 0.982 |
| rwinf | rewritten frac | 0.030 | 0.041 | 0.021 | 0.016 | 0.012 |
| rwinf | a_v/a_base | 0.978 | 0.983 | 0.988 | 0.989 | 0.987 |
| rwinf+bgfree | a_v/a_base | 0.962 | 0.972 | 0.977 | 0.982 | 0.983 |

Capped rows in the refit frames: 1546 of 3652.

#### F300M: 1224 satstar-replaced matched stars; 944 with rows in the refit frames; 944 with every variant finite
Satstar faint edge 16.43; unsaturated reference 16.43-17.43 mag: median dm -0.003 (N=1433).

| variant | 10-13 | 13-14 | 14-15 | 15-16 | 16-18 | all med / MAD | trend (faint-bright bin) | step (faint bin - unsat ref) |
|---|---|---|---|---|---|---|---|---|
| final | +0.116 (0.050) | +0.027 (0.035) | +0.014 (0.030) | +0.019 (0.035) | +0.015 (0.034) | +0.018 / 0.035 | -0.101 | +0.018 |
| uncapped | -0.044 (0.090) | -0.007 (0.049) | -0.003 (0.040) | +0.011 (0.035) | +0.014 (0.035) | +0.006 / 0.039 | +0.058 | +0.017 |
| base+cap | +0.116 (0.050) | +0.027 (0.035) | +0.014 (0.030) | +0.019 (0.035) | +0.015 (0.034) | +0.018 / 0.035 | -0.101 | +0.018 |
| bgfree | -0.008 (0.063) | +0.013 (0.040) | +0.014 (0.032) | +0.024 (0.031) | +0.021 (0.034) | +0.019 / 0.034 | +0.029 | +0.024 |
| bgfree+cap | +0.116 (0.043) | +0.035 (0.037) | +0.023 (0.031) | +0.029 (0.032) | +0.022 (0.033) | +0.026 / 0.032 | -0.094 | +0.025 |
| bgfree+v7b+cap | +0.127 (0.043) | +0.038 (0.037) | +0.024 (0.029) | +0.033 (0.032) | +0.027 (0.034) | +0.031 / 0.034 | -0.100 | +0.030 |
| rw12 | -0.038 (0.069) | +0.001 (0.045) | +0.003 (0.033) | +0.020 (0.032) | +0.020 (0.036) | +0.013 / 0.035 | +0.057 | +0.023 |
| rw12+cap | +0.116 (0.050) | +0.030 (0.037) | +0.016 (0.029) | +0.026 (0.033) | +0.021 (0.035) | +0.022 / 0.034 | -0.095 | +0.024 |
| rw12+bgfree | +0.004 (0.048) | +0.016 (0.044) | +0.015 (0.028) | +0.028 (0.031) | +0.024 (0.036) | +0.022 / 0.032 | +0.020 | +0.027 |
| rw12+bgfree+cap | +0.116 (0.043) | +0.033 (0.036) | +0.023 (0.029) | +0.032 (0.030) | +0.026 (0.036) | +0.029 / 0.033 | -0.090 | +0.029 |
| rwinf | -0.037 (0.068) | +0.003 (0.046) | +0.004 (0.032) | +0.021 (0.032) | +0.020 (0.036) | +0.013 / 0.035 | +0.057 | +0.023 |
| rwinf+cap | +0.116 (0.050) | +0.030 (0.037) | +0.016 (0.030) | +0.026 (0.033) | +0.021 (0.035) | +0.023 / 0.034 | -0.095 | +0.024 |
| rwinf+bgfree | +0.004 (0.050) | +0.013 (0.045) | +0.013 (0.028) | +0.026 (0.030) | +0.022 (0.035) | +0.020 / 0.032 | +0.019 | +0.025 |
| rwinf+bgfree+cap | +0.116 (0.043) | +0.033 (0.036) | +0.021 (0.029) | +0.031 (0.031) | +0.023 (0.035) | +0.027 / 0.033 | -0.093 | +0.026 |

N per bin: 19, 97, 232, 399, 197

Fraction of fit pixels rewritten (median over stars per bin) and amplitude ratio variant/base (median over stars per bin, uncapped):

| variant | quantity | 10-13 | 13-14 | 14-15 | 15-16 | 16-18 |
|---|---|---|---|---|---|---|
| rw12 | rewritten frac | 0.135 | 0.099 | 0.060 | 0.034 | 0.020 |
| rw12 | a_v/a_base | 0.984 | 0.991 | 0.992 | 0.991 | 0.995 |
| rw12+bgfree | a_v/a_base | 0.977 | 0.980 | 0.980 | 0.984 | 0.991 |
| rwinf | rewritten frac | 0.182 | 0.138 | 0.079 | 0.052 | 0.031 |
| rwinf | a_v/a_base | 0.984 | 0.991 | 0.992 | 0.990 | 0.995 |
| rwinf+bgfree | a_v/a_base | 0.978 | 0.980 | 0.982 | 0.985 | 0.992 |

Capped rows in the refit frames: 1640 of 3862.


## Summary of the measurements

- Effect on the amplitude. In F150W and F200W the rewrite lowers the amplitude relative to base by 0.4-4.4 % (median per bin; rw12: 0.956 at 14-15 mag to 0.988 at 18-19 mag in F150W; 0.97-0.99 in F200W); with bgfree the ratio is 0.93-0.98. D = 3 gives about half of the D = 12 effect; D = 12, 20 and inf agree within about 0.003. The rewritten pixels are mostly within 12 px, so D beyond 12 changes little.
- Rewritten fraction of fit pixels (median per bin, rw12 / rwinf): F150W 4.4 % / 5.0 % at 14-15 mag falling to 0.7 % / 1.0 % at 18-19 mag; F200W 13 % / 16 % at 13-14.5 mag, 4-6 % at 16-18 mag; F250M 2-3 % at 10-14 mag; F300M 13-18 % at 10-13 mag, 2-3 % at 16-18 mag.
- F150W uncapped all-star median dm: uncapped -0.078, rw12 -0.058, bgfree -0.064, rw12+bgfree -0.047. With cap: final -0.047, bgfree+cap -0.042, rw12+bgfree+cap -0.034, bgfree+v7b+cap -0.015 (MAD 0.041). The rewrite moves the median by +0.02 mag in each of the base and bgfree families. The trend (faint - bright bin) changes from +0.095 (uncapped) to +0.066 (rw12) and from +0.079 (bgfree) to +0.048 (rw12+bgfree). The rewrite reduces the bright-end offset (14-15 mag: -0.142 uncapped, -0.095 rw12, -0.070 rw12+bgfree) and leaves the faint bins nearly unchanged. In the 14-15 mag bin rw12+bgfree+cap gives -0.053 against -0.021 for bgfree+v7b+cap, so the rewrite closes less of the bright-end offset than the v7b PSF correction does.
- F200W: uncapped -0.097, rw12 -0.073, rw12+bgfree -0.058; rw12+bgfree+cap -0.048 against bgfree+v7b+cap -0.020. The trend of rw12+bgfree+cap is -0.007 against +0.009.
- F250M and F300M: all-star medians sit above zero (+0.03 to +0.05 capped) and the rewrite lowers the amplitude (ratio 0.96-0.99), which moves dm from +0.016 to +0.029 (F250M rw12, uncapped) and +0.006 to +0.013 (F300M). The sign of the shift (amplitude down) matches the SW bands, and in the LW bands it increases the positive offset of the all-star median. The bright bin (10-13 mag) is dominated by the cap (+0.12 capped against -0.03 uncapped in F250M).
- MAD is 0.050-0.055 for the rw variants against 0.047-0.053 for base variants in F150W (rw3+bgfree 0.052, rw12+bgfree 0.050 against bgfree 0.048). The rewrite does not lower the scatter.
- Per-star amplitude ratio against magnitude (plot rows 2): the ratio increases monotonically from the bright to the faint end in every band (F150W rw12: 0.958 to 0.988).

## Caveats and inferences (not measured directly)

- Only pixels with 200 DN <= g0 <= about 3000-4000 DN are rewritten. Pixels closer to the core, where wingmig measured q up to 1.07-1.12, have g0 above the top of the curve for bright stars and keep their crf value. The test therefore covers the wing at moderate g0 only. Extending R above the 99.9th percentile (extrapolating flat, the pipeline's behaviour above its last bin) was not run.
- The rewritten pixels carry a larger error than the crf pixels they replace in many cases (R scatter 5-8 % plus read noise), so their weight in the fit is lower. A fit with the crf error on the rewritten pixels was not run.
- The wingmig hypothesis is that the wing-driven amplitude absorbs migrated charge. A 1-4 % amplitude decrease is consistent with this for the pixels rewritten. It has about the size expected if q-1 is 2-3 % over the rewritten pixels. This is an inference from sizes and does not test the mechanism.
- Gain values are assumptions (2.0 and 1.8 e/DN). The effect enters only the Poisson term of the error, where g0/gain is below rn0^2 for g0 below about 200 DN and comparable to it at 1000 DN.
- The cal-side noise is included in the R scatter s, which overstates the intrinsic scatter at the percent level for F250M (s = 0.13-0.20).
- R(g0) comes from the 200 DN-start curve, not the pipeline's 2000 DN-start curve.
- The mag bins and stars match score2 (F150W 1471 stars, F200W 515, F250M 905 and F300M 944 with rows in the refit frames: see table headers).
