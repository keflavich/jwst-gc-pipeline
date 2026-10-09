# Round 6: header R as the rim-rewrite R (arm main2)

R_header = `S.zeroframe_header_R` = PHOTMJSR / (TFRAME (NFRAMES+1)/2), constant in g0. Read-only on pipeline and earlier files. New files: `run_frames6.py`, `survey_tab6.py`, `score6.py`, `plot6.py`, `out6/`, `satrefit6.png`. Same frames, rim pixel set, rim-error rule and cap recompute as round 5 (rim check 1.0000 on the test frame). Caps carry the per-row a_raw/cap_base factor as in round 5. "measured" marks table values; "inferred" marks interpretation.

## Item 1: refit with R = R_header (measured)

Median dm, all stars; the unsaturated reference is F150W +0.004, F200W -0.017, F250M +0.003, F300M -0.003. Per-bin values with MAD and trend for every variant: `out6/tables6_<band>.md`. Columns: final; R_N25+cap and R_N25+rw12h_e0+bgfree+cap copied from round 5 (the second is called "N25+h0+bgfree+cap" in the tables).

| band, det | final | R_N25+cap | N25+h0+bgfree+cap | H+cap | H+bgfree+cap | H+h0+cap | H+h0+bgfree+cap |
|---|---|---|---|---|---|---|---|
| F150W all (MAD) | -0.047 (0.047) | -0.002 (0.032) | +0.008 (0.036) | -0.028 (0.035) | -0.020 (0.036) | -0.022 (0.038) | -0.013 (0.039) |
| F150W nrcb1 | -0.017 (0.037) | +0.004 (0.036) | +0.011 (0.039) | -0.028 (0.038) | -0.023 (0.039) | -0.020 (0.044) | -0.013 (0.045) |
| F150W nrcb3 | -0.075 (0.032) | -0.006 (0.028) | +0.006 (0.032) | -0.028 (0.033) | -0.018 (0.032) | -0.024 (0.034) | -0.013 (0.033) |
| F200W all | -0.066 (0.054) | -0.018 (0.032) | -0.005 (0.031) | -0.034 (0.028) | -0.027 (0.030) | -0.027 (0.028) | -0.018 (0.030) |
| F200W nrcb1 | -0.021 (0.025) | -0.000 (0.025) | +0.010 (0.027) | -0.021 (0.025) | -0.019 (0.026) | -0.015 (0.028) | -0.007 (0.028) |
| F200W nrcb3 | -0.093 (0.025) | -0.034 (0.027) | -0.020 (0.028) | -0.045 (0.026) | -0.038 (0.029) | -0.038 (0.026) | -0.028 (0.030) |
| F250M nrcblong | +0.030 (0.035) | +0.132 (0.040) | +0.142 (0.040) | -0.009 (0.036) | +0.003 (0.033) | -0.008 (0.039) | +0.006 (0.036) |
| F300M nrcblong | +0.018 (0.035) | +0.070 (0.036) | +0.082 (0.035) | -0.013 (0.035) | -0.003 (0.033) | -0.010 (0.035) | +0.001 (0.033) |

nrcb3 minus nrcb1: F150W final -0.058, R_N25+cap -0.010, H+cap 0.000, H+bgfree+cap +0.005, H+h0+bgfree+cap 0.000; F200W final -0.072, R_N25+cap -0.034, H+cap -0.024, H+h0+bgfree+cap -0.021.
Bright-to-faint trend (faint minus bright dm): F150W nrcb1 final +0.039, H+cap +0.045; F150W nrcb3 final +0.052, H+cap +0.059; F250M final -0.094, H+cap -0.062; F300M final -0.101, H+cap -0.078. In F150W the bins with ref < 16 sit 0.03-0.06 below the unsaturated reference under H on both detectors.
Amplitude ratios to base (all stars): F150W nrcb1 a_H/a_base 1.000 (a_RN25 0.987), nrcb3 0.960 (0.951); F200W nrcb1 0.997 (0.986), nrcb3 0.960 (0.955); F250M 1.032 (0.922); F300M 1.025 (0.955). Cap ratios cap_H/cap_base: 1.014 and 0.957 (F150W nrcb1, nrcb3), 1.002 and 0.956 (F200W), 1.056 (F250M), 1.041 (F300M).

## Item 2: rim-pixel data/model, r < 6 px, cat 0 (measured)

Entries: N=0 pipeline / R_N25 / R_header (N = pixel counts in `tables6_*.md`).

| set | nrcb1 | nrcb3 | nrcb3/nrcb1 |
|---|---|---|---|
| F150W all rim | 1.028 / 1.004 / 1.026 | 1.084 / 1.006 / 1.019 | 1.055 / 1.002 / 0.994 |
| F150W peak pixel | 1.016 / 0.997 / 1.029 | 1.079 / 1.006 / 1.033 | 1.062 / 1.010 / 1.004 |
| F200W all rim | 1.042 / 1.017 / 1.034 | 1.106 / 1.037 / 1.042 | 1.061 / 1.020 / 1.008 |
| F200W peak pixel | 1.022 / 1.002 / 1.024 | 1.096 / 1.036 / 1.047 | 1.072 / 1.034 / 1.023 |

LW nrcblong (N=0 / R_N25 / R_header): F250M all rim 0.979 / 0.895 / 1.007, peak 0.984 / 0.887 / 1.038; F300M all rim 0.999 / 0.946 / 1.013, peak 0.986 / 0.937 / 1.027.
By g0 the F150W nrcb3/nrcb1 ratio under H lies between 0.980 and 1.002; in F200W it is 0.98-1.02 for g0 > 800 DN and 1.04-1.07 below 800 DN.

## Item 3: R_N0 / R_header and R_N25 / R_header at 2440 DN, exposure 1 (measured)

Post-satcheck curves from the round-5 reproduction. Full table, all 38 frames, flags, bin counts: `out6/survey6.md`. 17 of 38 frames move by more than 2 % under the header anchor (|R_header/R_N0 - 1| > 2 %).

| band | nrcb1-4 N0/hdr median (range) | nrcb1-4 N25/hdr median (range) | nrcblong N0 / N25 |
|---|---|---|---|
| F150W | 1.052 (1.015-1.083) | 0.983 (0.974-0.997) | |
| F162M | 1.039 (1.001-1.067) | 0.994 (0.985-1.087) | |
| F182M | 1.047 (1.003-1.082) | 0.986 (0.961-1.005) | |
| F200W | 1.069 (1.021-1.094) | 0.992 (0.980-1.005) | |
| F250M | | | 0.993 / 0.943 |
| F277W | | | 1.022 / 0.947 |
| F300M | | | 1.000 / 0.960 |

Flags (header would move R by > 2 %): nrcb3 in F150W/F162M/F182M/F200W (-6.7, -5.5, -5.8, -6.8 %), nrcb4 in the same four (-7.6, -6.3 with a satcheck rebuild, -7.6, -8.6 %), nrcb2 in F150W/F182M/F200W (-3.2, -3.2, -6.1 %), F200W nrcb1 (-2.1 %), F200W nrca1 (-3.3 %), F277W nrcalong (+3.2 %), F277W nrcblong (-2.2 %), F300M nrcalong (+8.1 %, satcheck rebuild). nrcb1 in F150W/F162M/F182M and F250M/F300M nrcblong stay within 1.5 % of R_header at N=0 (R_header/R_N0 within 2 %). Every nrca SW frame and F250M/F300M nrcalong hits guard truncation to one bin, and most trigger a satcheck rebuild; those curves rest on few pixels. F162M nrcb4 has R_N25/header 1.087 after a rebuild, so R_N25 itself is unreliable there.

## Item 4: far-field (edt >= 25) cal/g0 / R_header (measured)

Median over exposures 1-4 (nrcalong exposure 1 only); counts per frame range from 400 to 88 000 (`survey6.md`).

| band, det | 200-500 | 500-1000 | 1000-2000 | 2000-4000 DN |
|---|---|---|---|---|
| F150W nrcb1 | 0.969 | 0.985 | 0.994 | 0.983 |
| F150W nrcb3 | 1.018 | 1.019 | 1.008 | 0.994 |
| F200W nrcb1 | 0.955 | 0.992 | 1.000 | 0.990 |
| F200W nrcb3 | 1.002 | 1.021 | 1.015 | 1.002 |
| F250M nrcblong | 0.990 | 1.000 | 0.976 | 0.912 |
| F300M nrcblong | 0.935 | 1.009 | 0.995 | 0.949 |
| F250M nrcalong (exp 1) | 0.959 | 0.928 | 0.059 | 0.017 |
| F300M nrcalong (exp 1) | 0.924 | 0.949 | 0.913 | 0.045 |

The SW detectors stay within 1-2 % of R_header above 500 DN with no trend in g0. The F250M nrcblong ratio is flat (0.99-1.00) up to 1000 DN and falls to 0.976 and 0.912 above; across exposures the 2000-4000 DN bin reads 0.925, 0.920, 0.905, 0.876. F300M nrcblong falls from 1.009 (500-1000) to 0.995 and 0.949, and the 200-500 bin (0.935) sits low too. nrcalong LW frames have too few valid far-field pixels above 1000 DN (ratios 0.02-0.06 are junk).

## Interpretation (inferred)

- R_header removes most of the nrcb3 offset in F150W (nrcb3 minus nrcb1 from -0.058 to 0.000) and keeps LW within 0.01-0.02 mag of the unsaturated reference, which R_N25 does not. In F200W it leaves a -0.024 nrcb3 offset (R_N25+cap leaves -0.034). R_N25 reaches +0.004/-0.006 in F150W against H+cap -0.028/-0.028, so the header rule keeps F150W about 0.03 mag brighter than the reference.
- The header anchor leaves nrcb1 unchanged (a_H/a_base 1.000) and removes 4 % on nrcb3, which follows from R_header sitting 1.5 % below N=0 on nrcb1 and 7 % below on nrcb3.
- The LW far-field deficit against header R grows with g0 in nrcblong (F250M 1.00 at 500-1000 DN to 0.91 at 2000-4000 DN; F300M 1.01 to 0.95), which points to a g0-dependent effect such as BFE, and a flat-level or header-formula error is less likely. The low 200-500 DN bins in F300M and F200W nrcb1 could come from background or noise in cal/g0 at low g0; I did not test that.
- In LW the N=0 rim curve already agrees with R_header (0.993, 1.000) while the far field at 2000-4000 DN sits 5-9 % below it. The rim-pixel data/model under R_header (1.007, 1.013) and the dm near the reference show that R_header describes the near-core pixels at least as well as the far-field level does; the cause of the far-field deficit stays open.
- The header rule has no distance parameter, but the SW dm still carries the 0.03 mag undershoot at ref < 16 mag in F150W.

Caveats: R_header is a constant, applied to rim pixels at all g0 (including g0 < 2000 DN, where the pipeline curve extrapolates flat). Only exposure-1 curves enter item 3. dm for variants follows round-5 conventions. The nrca frames and F162M nrcb4 are unreliable for the N0/N25 comparison.
