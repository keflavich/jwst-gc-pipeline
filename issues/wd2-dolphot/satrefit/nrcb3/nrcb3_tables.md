# nrcb3 tables (full)

Star-level medians (median over the rows of a star; each star lies on one detector in all of its exposures). dm = m_row + c_b - m_dolphot - ZP (final flux). Errors: bootstrap over stars.


## F150W

Stars with S rows on more than one detector: 0 of 2326; per-star dither extent median 0.002", max 0.09".
nrcb1 median -0.017 (N 706), nrcb3 -0.075 (N 773); raw nrcb3-nrcb1 -0.058 +/- 0.002; matched in 1-mag bins -0.057 +/- 0.002

Nearest-neighbour pair test across the nrcb1/nrcb3 boundary (each nrcb3 star paired with nearest nrcb1 star of |dm_dolphot| < 0.25 mag): median(dm b3 - dm b1), error, N pairs, median separation (arcsec)

| max sep | mag-matched | mag + density + bkg matched |
|---|---|---|
| 10" | -0.042 +/- 0.003 (N 50, sep 8.2) | -0.038 +/- 0.006 (N 20, sep 8.1) |
| 20" | -0.044 +/- 0.003 (N 207, sep 12.6) | -0.044 +/- 0.004 (N 133, sep 14.1) |
| 40" | -0.042 +/- 0.002 (N 567, sep 23.7) | -0.045 +/- 0.002 (N 387, sep 27.1) |

**F150W: neighbours within 2" with mag < own+3**

| bin | nrcb1 median (N) | nrcb3 median (N) | raw b3-b1 | mag-matched b3-b1 (1-mag strata) |
|---|---|---|---|---|
| -1e-09-3 | -0.018 (284) | -0.081 (212) | -0.062 | -0.062 +/- 0.004 |
| 3-5 | -0.017 (188) | -0.082 (139) | -0.065 | -0.066 +/- 0.004 |
| 5-8 | -0.017 (156) | -0.074 (176) | -0.057 | -0.059 +/- 0.004 |
| 8-26 | -0.015 (78) | -0.069 (246) | -0.054 | -0.049 +/- 0.004 |

**F150W: neighbours within 1" with mag < own+3**

| bin | nrcb1 median (N) | nrcb3 median (N) | raw b3-b1 | mag-matched b3-b1 (1-mag strata) |
|---|---|---|---|---|
| -1e-09-1 | -0.017 (474) | -0.077 (393) | -0.061 | -0.061 +/- 0.003 |
| 1-2 | -0.022 (133) | -0.075 (171) | -0.053 | -0.059 +/- 0.004 |
| 2-8 | -0.015 (99) | -0.070 (209) | -0.055 | -0.053 +/- 0.004 |

**F150W: annulus background 1.5-2.5" (MJy/sr)**

| bin | nrcb1 median (N) | nrcb3 median (N) | raw b3-b1 | mag-matched b3-b1 (1-mag strata) |
|---|---|---|---|---|
| 1.24-2.64 | -0.005 (300) | -0.068 (70) | -0.063 | -0.072 +/- 0.006 |
| 2.64-3.22 | -0.024 (208) | -0.072 (162) | -0.048 | -0.050 +/- 0.006 |
| 3.22-4.01 | -0.025 (160) | -0.077 (209) | -0.052 | -0.056 +/- 0.004 |
| 4.01-41.9 | -0.042 (38) | -0.075 (332) | -0.033 | -0.038 +/- 0.007 |

**F150W: stored satstar local_bkg (MJy/sr)**

| bin | nrcb1 median (N) | nrcb3 median (N) | raw b3-b1 | mag-matched b3-b1 (1-mag strata) |
|---|---|---|---|---|
| 1.24-2.6 | -0.005 (308) | -0.063 (62) | -0.058 | -0.067 +/- 0.006 |
| 2.6-3.13 | -0.021 (182) | -0.070 (188) | -0.049 | -0.058 +/- 0.005 |
| 3.13-3.87 | -0.026 (160) | -0.077 (209) | -0.051 | -0.052 +/- 0.005 |
| 3.87-19.9 | -0.045 (56) | -0.077 (314) | -0.032 | -0.038 +/- 0.006 |

**F150W: distance to detector edge (px)**

| bin | nrcb1 median (N) | nrcb3 median (N) | raw b3-b1 | mag-matched b3-b1 (1-mag strata) |
|---|---|---|---|---|
| -1-50 | -0.030 (57) | -0.077 (73) | -0.047 | -0.046 +/- 0.009 |
| 50-150 | -0.014 (128) | -0.066 (111) | -0.052 | -0.053 +/- 0.006 |
| 150-400 | -0.015 (250) | -0.068 (273) | -0.053 | -0.053 +/- 0.004 |
| 400-3e+03 | -0.020 (271) | -0.082 (316) | -0.062 | -0.064 +/- 0.003 |

**F150W: nearest brighter neighbour (arcsec)**

| bin | nrcb1 median (N) | nrcb3 median (N) | raw b3-b1 | mag-matched b3-b1 (1-mag strata) |
|---|---|---|---|---|
| 0-0.3 | -0.019 (20) | -0.071 (30) | -0.051 | -0.046 +/- 0.011 |
| 0.3-0.6 | -0.005 (43) | -0.069 (84) | -0.063 | -0.064 +/- 0.008 |
| 0.6-1 | -0.014 (123) | -0.067 (141) | -0.053 | -0.052 +/- 0.004 |
| 1-1.5 | -0.014 (147) | -0.071 (159) | -0.057 | -0.057 +/- 0.004 |
| 1.5-3 | -0.017 (123) | -0.075 (104) | -0.058 | -0.057 +/- 0.005 |
| 3-99 | -0.023 (250) | -0.091 (255) | -0.068 | -0.068 +/- 0.004 |

**F150W: stratified nrcb3 - nrcb1 (weighted mean over cells with >= 3 stars per detector)**

| strata | b3 - b1 | N b1 / N b3 used |
|---|---|---|
| mag+n2 | -0.060 +/- 0.002 | 703 / 754 |
| mag+abkg | -0.055 +/- 0.003 | 695 / 771 |
| mag+lbkg | -0.054 +/- 0.003 | 693 / 769 |
| mag+n2+abkg | -0.055 +/- 0.006 | 593 / 543 |
| mag+n2+abkg+edge | -0.056 +/- 0.006 | 319 / 287 |
| mag+n1+dbr | -0.061 +/- 0.003 | 670 / 691 |

**F150W: OLS on star medians, dm ~ b3 + standardised covariates**

| model | b3 coefficient | other coefficients (per 1 sd) | N |
|---|---|---|---|
| mag | -0.058 +/- 0.003 | ref +0.020+/-0.002 | 1479 |
| mag+logn2 | -0.058 +/- 0.003 | ref +0.021+/-0.002, logn2 -0.000+/-0.001 | 1479 |
| mag+logn2+logbkg | -0.057 +/- 0.004 | ref +0.020+/-0.003, logn2 +0.000+/-0.002, logbkg -0.002+/-0.003 | 1479 |
| mag+logn2+logbkg+edge+dbr | -0.055 +/- 0.004 | ref +0.020+/-0.002, logn2 +0.001+/-0.002, logbkg -0.003+/-0.003, edge -0.007+/-0.001, ldbr +0.000+/-0.002 | 1479 |

**F150W: daophot control (unsaturated stars in the ZP window, star medians over frames)**

nrcb1 -0.003 (N 489), nrcb3 -0.021 (N 513); raw b3-b1 -0.018 +/- 0.001; mag-matched -0.017 +/- 0.001; mag+n2 -0.017 +/- 0.002; mag+abkg -0.015 +/- 0.003; mag+n2+abkg -0.014 +/- 0.005

| control n2 bin | nrcb1 median (N) | nrcb3 median (N) | mag-matched b3-b1 |
|---|---|---|---|
| -1e-09-3 | -0.001 (88) | -0.017 (74) | -0.015 +/- 0.003 |
| 3-5 | -0.003 (115) | -0.017 (74) | -0.015 +/- 0.003 |
| 5-8 | -0.002 (144) | -0.022 (123) | -0.019 +/- 0.003 |
| 8-26 | -0.008 (142) | -0.026 (231) | -0.018 +/- 0.004 |

| control abkg bin | nrcb1 median (N) | nrcb3 median (N) | mag-matched b3-b1 |
|---|---|---|---|
| 1.24-2.64 | -0.002 (233) | -0.019 (96) | -0.016 +/- 0.002 |
| 2.64-3.22 | -0.005 (111) | -0.023 (121) | -0.017 +/- 0.003 |
| 3.22-4.01 | -0.010 (106) | -0.019 (120) | -0.008 +/- 0.004 |
| 4.01-41.9 | +0.003 (34) | -0.020 (173) | -0.023 +/- 0.018 |

| control edge bin | nrcb1 median (N) | nrcb3 median (N) | mag-matched b3-b1 |
|---|---|---|---|
| -1-50 | -0.004 (41) | -0.020 (44) | -0.013 +/- 0.008 |
| 50-150 | -0.002 (85) | -0.021 (76) | -0.020 +/- 0.006 |
| 150-400 | -0.003 (180) | -0.018 (215) | -0.017 +/- 0.002 |
| 400-3e+03 | -0.004 (183) | -0.023 (178) | -0.018 +/- 0.002 |


## F162M

Stars with S rows on more than one detector: 0 of 1782; per-star dither extent median 0.002", max 0.12".
nrcb1 median -0.004 (N 533), nrcb3 -0.055 (N 670); raw nrcb3-nrcb1 -0.050 +/- 0.002; matched in 1-mag bins -0.047 +/- 0.002

Nearest-neighbour pair test across the nrcb1/nrcb3 boundary (each nrcb3 star paired with nearest nrcb1 star of |dm_dolphot| < 0.25 mag): median(dm b3 - dm b1), error, N pairs, median separation (arcsec)

| max sep | mag-matched | mag + density + bkg matched |
|---|---|---|
| 10" | -0.029 +/- 0.011 (N 37, sep 8.4) | -0.019 +/- 0.026 (N 12, sep 8.4) |
| 20" | -0.037 +/- 0.005 (N 190, sep 13.6) | -0.036 +/- 0.009 (N 93, sep 14.3) |
| 40" | -0.036 +/- 0.004 (N 509, sep 23.0) | -0.041 +/- 0.003 (N 293, sep 27.7) |

**F162M: neighbours within 2" with mag < own+3**

| bin | nrcb1 median (N) | nrcb3 median (N) | raw b3-b1 | mag-matched b3-b1 (1-mag strata) |
|---|---|---|---|---|
| -1e-09-3 | -0.003 (233) | -0.060 (170) | -0.057 | -0.049 +/- 0.004 |
| 3-5 | -0.012 (132) | -0.059 (113) | -0.047 | -0.051 +/- 0.006 |
| 5-8 | -0.004 (119) | -0.058 (140) | -0.054 | -0.047 +/- 0.004 |
| 8-27 | -0.005 (49) | -0.049 (247) | -0.044 | -0.044 +/- 0.005 |

**F162M: neighbours within 1" with mag < own+3**

| bin | nrcb1 median (N) | nrcb3 median (N) | raw b3-b1 | mag-matched b3-b1 (1-mag strata) |
|---|---|---|---|---|
| -1e-09-1 | -0.003 (381) | -0.059 (322) | -0.055 | -0.051 +/- 0.003 |
| 1-2 | -0.012 (91) | -0.051 (131) | -0.039 | -0.041 +/- 0.005 |
| 2-9 | -0.005 (61) | -0.051 (217) | -0.046 | -0.046 +/- 0.005 |

**F162M: annulus background 1.5-2.5" (MJy/sr)**

| bin | nrcb1 median (N) | nrcb3 median (N) | raw b3-b1 | mag-matched b3-b1 (1-mag strata) |
|---|---|---|---|---|
| 1.39-3.09 | +0.006 (240) | -0.054 (61) | -0.059 | -0.058 +/- 0.007 |
| 3.09-3.86 | -0.009 (162) | -0.050 (139) | -0.041 | -0.048 +/- 0.005 |
| 3.86-5.06 | -0.020 (119) | -0.055 (181) | -0.035 | -0.038 +/- 0.004 |
| 5.06-50 | -0.043 (12) | -0.056 (289) | -0.013 | -0.025 +/- 0.014 |

**F162M: stored satstar local_bkg (MJy/sr)**

| bin | nrcb1 median (N) | nrcb3 median (N) | raw b3-b1 | mag-matched b3-b1 (1-mag strata) |
|---|---|---|---|---|
| 1.43-3.13 | +0.006 (245) | -0.053 (56) | -0.060 | -0.062 +/- 0.007 |
| 3.13-3.78 | -0.008 (147) | -0.049 (154) | -0.041 | -0.048 +/- 0.005 |
| 3.78-4.88 | -0.022 (109) | -0.054 (191) | -0.032 | -0.037 +/- 0.004 |
| 4.88-20.1 | -0.038 (32) | -0.057 (269) | -0.019 | -0.019 +/- 0.009 |

**F162M: distance to detector edge (px)**

| bin | nrcb1 median (N) | nrcb3 median (N) | raw b3-b1 | mag-matched b3-b1 (1-mag strata) |
|---|---|---|---|---|
| -1-50 | -0.015 (41) | -0.053 (62) | -0.037 | -0.026 +/- 0.010 |
| 50-150 | +0.003 (95) | -0.049 (101) | -0.053 | -0.049 +/- 0.006 |
| 150-400 | -0.003 (190) | -0.045 (232) | -0.042 | -0.040 +/- 0.004 |
| 400-3e+03 | -0.009 (207) | -0.064 (275) | -0.055 | -0.054 +/- 0.003 |

**F162M: nearest brighter neighbour (arcsec)**

| bin | nrcb1 median (N) | nrcb3 median (N) | raw b3-b1 | mag-matched b3-b1 (1-mag strata) |
|---|---|---|---|---|
| 0-0.3 | -0.004 (10) | -0.051 (23) | -0.048 | -0.055 +/- 0.026 |
| 0.3-0.6 | +0.006 (26) | -0.049 (70) | -0.055 | -0.048 +/- 0.008 |
| 0.6-1 | -0.006 (72) | -0.047 (116) | -0.041 | -0.035 +/- 0.006 |
| 1-1.5 | -0.003 (99) | -0.049 (135) | -0.046 | -0.045 +/- 0.005 |
| 1.5-3 | +0.000 (91) | -0.050 (88) | -0.050 | -0.047 +/- 0.006 |
| 3-99 | -0.013 (235) | -0.072 (238) | -0.059 | -0.054 +/- 0.004 |

**F162M: stratified nrcb3 - nrcb1 (weighted mean over cells with >= 3 stars per detector)**

| strata | b3 - b1 | N b1 / N b3 used |
|---|---|---|
| mag+n2 | -0.048 +/- 0.002 | 531 / 630 |
| mag+abkg | -0.046 +/- 0.003 | 504 / 542 |
| mag+lbkg | -0.043 +/- 0.003 | 507 / 640 |
| mag+n2+abkg | -0.043 +/- 0.003 | 430 / 357 |
| mag+n2+abkg+edge | -0.039 +/- 0.005 | 289 / 226 |
| mag+n1+dbr | -0.048 +/- 0.003 | 498 / 554 |

**F162M: OLS on star medians, dm ~ b3 + standardised covariates**

| model | b3 coefficient | other coefficients (per 1 sd) | N |
|---|---|---|---|
| mag | -0.047 +/- 0.003 | ref +0.012+/-0.002 | 1203 |
| mag+logn2 | -0.046 +/- 0.003 | ref +0.012+/-0.003, logn2 -0.002+/-0.001 | 1203 |
| mag+logn2+logbkg | -0.047 +/- 0.003 | ref +0.013+/-0.002, logn2 -0.003+/-0.002, logbkg +0.001+/-0.002 | 1203 |
| mag+logn2+logbkg+edge+dbr | -0.045 +/- 0.003 | ref +0.014+/-0.002, logn2 +0.000+/-0.002, logbkg -0.001+/-0.003, edge -0.008+/-0.001, ldbr +0.004+/-0.003 | 1203 |

**F162M: daophot control (unsaturated stars in the ZP window, star medians over frames)**

nrcb1 -0.005 (N 531), nrcb3 -0.012 (N 649); raw b3-b1 -0.008 +/- 0.001; mag-matched -0.009 +/- 0.001; mag+n2 -0.006 +/- 0.001; mag+abkg -0.008 +/- 0.002; mag+n2+abkg -0.007 +/- 0.002

| control n2 bin | nrcb1 median (N) | nrcb3 median (N) | mag-matched b3-b1 |
|---|---|---|---|
| -1e-09-3 | -0.003 (107) | -0.005 (93) | -0.004 +/- 0.002 |
| 3-5 | -0.004 (119) | -0.009 (95) | -0.003 +/- 0.003 |
| 5-8 | -0.006 (164) | -0.011 (133) | -0.006 +/- 0.002 |
| 8-27 | -0.007 (141) | -0.015 (313) | -0.009 +/- 0.002 |

| control abkg bin | nrcb1 median (N) | nrcb3 median (N) | mag-matched b3-b1 |
|---|---|---|---|
| 1.39-3.09 | -0.006 (256) | -0.005 (115) | +0.001 +/- 0.001 |
| 3.09-3.86 | -0.004 (148) | -0.011 (165) | -0.009 +/- 0.003 |
| 3.86-5.06 | -0.002 (98) | -0.015 (133) | -0.014 +/- 0.002 |
| 5.06-50 | -0.003 (25) | -0.017 (233) | -0.019 +/- 0.010 |

| control edge bin | nrcb1 median (N) | nrcb3 median (N) | mag-matched b3-b1 |
|---|---|---|---|
| -1-50 | -0.006 (46) | -0.014 (51) | -0.006 +/- 0.006 |
| 50-150 | -0.003 (90) | -0.009 (97) | -0.006 +/- 0.003 |
| 150-400 | -0.004 (196) | -0.012 (263) | -0.009 +/- 0.002 |
| 400-3e+03 | -0.005 (199) | -0.013 (238) | -0.009 +/- 0.001 |


## F182M

Stars with S rows on more than one detector: 0 of 824; per-star dither extent median 0.001", max 0.09".
nrcb1 median -0.020 (N 248), nrcb3 -0.065 (N 352); raw nrcb3-nrcb1 -0.045 +/- 0.003; matched in 1-mag bins -0.044 +/- 0.003

Nearest-neighbour pair test across the nrcb1/nrcb3 boundary (each nrcb3 star paired with nearest nrcb1 star of |dm_dolphot| < 0.25 mag): median(dm b3 - dm b1), error, N pairs, median separation (arcsec)

| max sep | mag-matched | mag + density + bkg matched |
|---|---|---|
| 10" | -0.032 +/- 0.005 (N 22, sep 8.4) | -0.012 +/- 0.023 (N 5, sep 8.1) |
| 20" | -0.047 +/- 0.005 (N 124, sep 14.1) | -0.032 +/- 0.007 (N 44, sep 13.6) |
| 40" | -0.047 +/- 0.003 (N 287, sep 21.8) | -0.044 +/- 0.004 (N 153, sep 27.9) |

**F182M: neighbours within 2" with mag < own+3**

| bin | nrcb1 median (N) | nrcb3 median (N) | raw b3-b1 | mag-matched b3-b1 (1-mag strata) |
|---|---|---|---|---|
| -1e-09-2.75 | -0.017 (92) | -0.075 (58) | -0.059 | -0.059 +/- 0.007 |
| 2.75-5 | -0.024 (97) | -0.074 (86) | -0.050 | -0.050 +/- 0.004 |
| 5-8 | -0.025 (42) | -0.065 (76) | -0.040 | -0.036 +/- 0.007 |
| 8-26 | -0.018 (17) | -0.059 (132) | -0.041 | -0.037 +/- 0.015 |

**F182M: neighbours within 1" with mag < own+3**

| bin | nrcb1 median (N) | nrcb3 median (N) | raw b3-b1 | mag-matched b3-b1 (1-mag strata) |
|---|---|---|---|---|
| -1e-09-1 | -0.020 (183) | -0.071 (168) | -0.051 | -0.048 +/- 0.004 |
| 1-2 | -0.026 (38) | -0.067 (67) | -0.041 | -0.041 +/- 0.007 |
| 2-8 | -0.018 (27) | -0.061 (117) | -0.044 | -0.039 +/- 0.011 |

**F182M: annulus background 1.5-2.5" (MJy/sr)**

| bin | nrcb1 median (N) | nrcb3 median (N) | raw b3-b1 | mag-matched b3-b1 (1-mag strata) |
|---|---|---|---|---|
| 3.27-8.61 | -0.014 (127) | -0.078 (23) | -0.064 | -0.067 +/- 0.005 |
| 8.61-10.8 | -0.022 (58) | -0.078 (92) | -0.055 | -0.050 +/- 0.006 |
| 10.8-12.5 | -0.037 (47) | -0.069 (103) | -0.032 | -0.031 +/- 0.005 |
| 12.5-55 | -0.023 (16) | -0.058 (134) | -0.036 | -0.036 +/- 0.009 |

**F182M: stored satstar local_bkg (MJy/sr)**

| bin | nrcb1 median (N) | nrcb3 median (N) | raw b3-b1 | mag-matched b3-b1 (1-mag strata) |
|---|---|---|---|---|
| 3.35-8.47 | -0.014 (124) | -0.083 (26) | -0.069 | -0.067 +/- 0.006 |
| 8.47-10.7 | -0.023 (57) | -0.071 (93) | -0.048 | -0.048 +/- 0.006 |
| 10.7-12.2 | -0.031 (42) | -0.068 (108) | -0.037 | -0.035 +/- 0.007 |
| 12.2-27.9 | -0.034 (25) | -0.059 (125) | -0.025 | -0.031 +/- 0.007 |

**F182M: distance to detector edge (px)**

| bin | nrcb1 median (N) | nrcb3 median (N) | raw b3-b1 | mag-matched b3-b1 (1-mag strata) |
|---|---|---|---|---|
| -1-50 | -0.017 (18) | -0.059 (38) | -0.042 | -0.032 +/- 0.009 |
| 50-150 | -0.015 (46) | -0.058 (54) | -0.044 | -0.048 +/- 0.009 |
| 150-400 | -0.012 (85) | -0.056 (131) | -0.044 | -0.044 +/- 0.004 |
| 400-3e+03 | -0.026 (99) | -0.079 (129) | -0.052 | -0.052 +/- 0.003 |

**F182M: nearest brighter neighbour (arcsec)**

| bin | nrcb1 median (N) | nrcb3 median (N) | raw b3-b1 | mag-matched b3-b1 (1-mag strata) |
|---|---|---|---|---|
| 0-0.3 | -0.011 (3) | -0.061 (6) | -0.050 | +nan +/- nan |
| 0.3-0.6 | -0.007 (3) | -0.058 (23) | -0.051 | +nan +/- nan |
| 0.6-1 | -0.026 (19) | -0.061 (48) | -0.035 | -0.035 +/- 0.007 |
| 1-1.5 | -0.018 (32) | -0.062 (57) | -0.044 | -0.044 +/- 0.006 |
| 1.5-3 | -0.013 (35) | -0.060 (54) | -0.047 | -0.038 +/- 0.009 |
| 3-99 | -0.021 (156) | -0.074 (164) | -0.053 | -0.054 +/- 0.004 |

**F182M: stratified nrcb3 - nrcb1 (weighted mean over cells with >= 3 stars per detector)**

| strata | b3 - b1 | N b1 / N b3 used |
|---|---|---|
| mag+n2 | -0.048 +/- 0.004 | 241 / 298 |
| mag+abkg | -0.045 +/- 0.004 | 233 / 301 |
| mag+lbkg | -0.045 +/- 0.004 | 239 / 306 |
| mag+n2+abkg | -0.044 +/- 0.005 | 165 / 158 |
| mag+n2+abkg+edge | -0.042 +/- 0.007 | 59 / 83 |
| mag+n1+dbr | -0.050 +/- 0.003 | 225 / 242 |

**F182M: OLS on star medians, dm ~ b3 + standardised covariates**

| model | b3 coefficient | other coefficients (per 1 sd) | N |
|---|---|---|---|
| mag | -0.047 +/- 0.002 | ref +0.003+/-0.001 | 600 |
| mag+logn2 | -0.049 +/- 0.003 | ref +0.002+/-0.001, logn2 +0.002+/-0.001 | 600 |
| mag+logn2+logbkg | -0.051 +/- 0.003 | ref +0.003+/-0.001, logn2 +0.001+/-0.001, logbkg +0.002+/-0.002 | 600 |
| mag+logn2+logbkg+edge+dbr | -0.049 +/- 0.003 | ref +0.002+/-0.001, logn2 -0.000+/-0.001, logbkg +0.001+/-0.002, edge -0.008+/-0.001, ldbr -0.002+/-0.001 | 600 |

**F182M: daophot control (unsaturated stars in the ZP window, star medians over frames)**

nrcb1 -0.001 (N 556), nrcb3 -0.012 (N 613); raw b3-b1 -0.011 +/- 0.001; mag-matched -0.012 +/- 0.001; mag+n2 -0.012 +/- 0.001; mag+abkg -0.014 +/- 0.002; mag+n2+abkg -0.013 +/- 0.003

| control n2 bin | nrcb1 median (N) | nrcb3 median (N) | mag-matched b3-b1 |
|---|---|---|---|
| -1e-09-2.75 | +0.005 (46) | -0.006 (41) | -0.012 +/- 0.003 |
| 2.75-5 | +0.001 (180) | -0.008 (118) | -0.010 +/- 0.002 |
| 5-8 | -0.003 (164) | -0.015 (137) | -0.013 +/- 0.002 |
| 8-26 | -0.004 (166) | -0.015 (302) | -0.012 +/- 0.002 |

| control abkg bin | nrcb1 median (N) | nrcb3 median (N) | mag-matched b3-b1 |
|---|---|---|---|
| 3.27-8.61 | -0.005 (299) | -0.006 (109) | -0.004 +/- 0.002 |
| 8.61-10.8 | +0.004 (122) | -0.015 (210) | -0.019 +/- 0.002 |
| 10.8-12.5 | +0.003 (82) | -0.013 (153) | -0.017 +/- 0.003 |
| 12.5-55 | +0.004 (47) | -0.012 (138) | -0.021 +/- 0.006 |

| control edge bin | nrcb1 median (N) | nrcb3 median (N) | mag-matched b3-b1 |
|---|---|---|---|
| -1-50 | -0.002 (46) | -0.012 (48) | -0.010 +/- 0.005 |
| 50-150 | +0.000 (98) | -0.014 (91) | -0.015 +/- 0.005 |
| 150-400 | -0.001 (212) | -0.012 (242) | -0.011 +/- 0.002 |
| 400-3e+03 | -0.001 (200) | -0.012 (232) | -0.011 +/- 0.002 |


## F200W

Stars with S rows on more than one detector: 0 of 722; per-star dither extent median 0.002", max 0.09".
nrcb1 median -0.021 (N 233), nrcb3 -0.093 (N 282); raw nrcb3-nrcb1 -0.072 +/- 0.002; matched in 1-mag bins -0.072 +/- 0.003

Nearest-neighbour pair test across the nrcb1/nrcb3 boundary (each nrcb3 star paired with nearest nrcb1 star of |dm_dolphot| < 0.25 mag): median(dm b3 - dm b1), error, N pairs, median separation (arcsec)

| max sep | mag-matched | mag + density + bkg matched |
|---|---|---|
| 10" | -0.068 +/- 0.014 (N 16, sep 8.3) | -0.112 +/- 0.025 (N 4, sep 7.9) |
| 20" | -0.088 +/- 0.005 (N 94, sep 13.8) | -0.083 +/- 0.008 (N 37, sep 14.0) |
| 40" | -0.092 +/- 0.003 (N 221, sep 21.8) | -0.092 +/- 0.004 (N 123, sep 28.3) |

**F200W: neighbours within 2" with mag < own+3**

| bin | nrcb1 median (N) | nrcb3 median (N) | raw b3-b1 | mag-matched b3-b1 (1-mag strata) |
|---|---|---|---|---|
| -1e-09-2.5 | -0.015 (81) | -0.099 (48) | -0.084 | -0.081 +/- 0.005 |
| 2.5-5 | -0.023 (94) | -0.097 (73) | -0.074 | -0.074 +/- 0.005 |
| 5-8 | -0.021 (41) | -0.093 (65) | -0.072 | -0.075 +/- 0.006 |
| 8-26 | -0.016 (17) | -0.088 (96) | -0.072 | -0.075 +/- 0.011 |

**F200W: neighbours within 1" with mag < own+3**

| bin | nrcb1 median (N) | nrcb3 median (N) | raw b3-b1 | mag-matched b3-b1 (1-mag strata) |
|---|---|---|---|---|
| -1e-09-1 | -0.020 (171) | -0.097 (138) | -0.077 | -0.074 +/- 0.004 |
| 1-2 | -0.025 (38) | -0.090 (61) | -0.065 | -0.059 +/- 0.009 |
| 2-8 | -0.012 (24) | -0.091 (83) | -0.079 | -0.076 +/- 0.007 |

**F200W: annulus background 1.5-2.5" (MJy/sr)**

| bin | nrcb1 median (N) | nrcb3 median (N) | raw b3-b1 | mag-matched b3-b1 (1-mag strata) |
|---|---|---|---|---|
| 2.82-6.45 | -0.018 (109) | -0.105 (20) | -0.087 | -0.087 +/- 0.004 |
| 6.45-8.25 | -0.014 (58) | -0.098 (71) | -0.085 | -0.085 +/- 0.006 |
| 8.25-9.82 | -0.032 (51) | -0.097 (77) | -0.065 | -0.065 +/- 0.006 |
| 9.82-48.2 | -0.021 (15) | -0.087 (114) | -0.066 | -0.056 +/- 0.009 |

**F200W: stored satstar local_bkg (MJy/sr)**

| bin | nrcb1 median (N) | nrcb3 median (N) | raw b3-b1 | mag-matched b3-b1 (1-mag strata) |
|---|---|---|---|---|
| 2.9-6.42 | -0.015 (110) | -0.107 (19) | -0.092 | -0.093 +/- 0.007 |
| 6.42-8.14 | -0.019 (53) | -0.097 (76) | -0.079 | -0.079 +/- 0.005 |
| 8.14-9.44 | -0.034 (40) | -0.096 (88) | -0.062 | -0.068 +/- 0.007 |
| 9.44-24.4 | -0.029 (30) | -0.089 (99) | -0.059 | -0.058 +/- 0.007 |

**F200W: distance to detector edge (px)**

| bin | nrcb1 median (N) | nrcb3 median (N) | raw b3-b1 | mag-matched b3-b1 (1-mag strata) |
|---|---|---|---|---|
| -1-50 | -0.018 (17) | -0.082 (38) | -0.064 | -0.059 +/- 0.019 |
| 50-150 | -0.007 (37) | -0.086 (43) | -0.079 | -0.081 +/- 0.006 |
| 150-400 | -0.014 (84) | -0.090 (101) | -0.076 | -0.074 +/- 0.003 |
| 400-3e+03 | -0.031 (95) | -0.104 (100) | -0.073 | -0.076 +/- 0.004 |

**F200W: nearest brighter neighbour (arcsec)**

| bin | nrcb1 median (N) | nrcb3 median (N) | raw b3-b1 | mag-matched b3-b1 (1-mag strata) |
|---|---|---|---|---|
| 0-0.3 | - (1) | -0.093 (5) | +nan | +nan +/- nan |
| 0.3-0.6 | -0.018 (6) | -0.091 (13) | -0.073 | -0.067 +/- 0.017 |
| 0.6-1 | -0.020 (21) | -0.088 (28) | -0.068 | -0.064 +/- 0.018 |
| 1-1.5 | -0.024 (34) | -0.089 (44) | -0.065 | -0.070 +/- 0.007 |
| 1.5-3 | -0.018 (29) | -0.089 (38) | -0.071 | -0.069 +/- 0.010 |
| 3-99 | -0.021 (142) | -0.097 (154) | -0.076 | -0.073 +/- 0.004 |

**F200W: stratified nrcb3 - nrcb1 (weighted mean over cells with >= 3 stars per detector)**

| strata | b3 - b1 | N b1 / N b3 used |
|---|---|---|
| mag+n2 | -0.076 +/- 0.003 | 214 / 267 |
| mag+abkg | -0.075 +/- 0.003 | 218 / 270 |
| mag+lbkg | -0.074 +/- 0.004 | 220 / 274 |
| mag+n2+abkg | -0.079 +/- 0.004 | 163 / 134 |
| mag+n2+abkg+edge | -0.071 +/- 0.007 | 52 / 50 |
| mag+n1+dbr | -0.073 +/- 0.003 | 197 / 220 |

**F200W: OLS on star medians, dm ~ b3 + standardised covariates**

| model | b3 coefficient | other coefficients (per 1 sd) | N |
|---|---|---|---|
| mag | -0.074 +/- 0.003 | ref -0.005+/-0.002 | 515 |
| mag+logn2 | -0.074 +/- 0.003 | ref -0.005+/-0.002, logn2 +0.001+/-0.002 | 515 |
| mag+logn2+logbkg | -0.075 +/- 0.004 | ref -0.005+/-0.002, logn2 +0.000+/-0.002, logbkg +0.001+/-0.002 | 515 |
| mag+logn2+logbkg+edge+dbr | -0.073 +/- 0.004 | ref -0.006+/-0.002, logn2 -0.000+/-0.002, logbkg -0.001+/-0.002, edge -0.008+/-0.001, ldbr -0.002+/-0.002 | 515 |

**F200W: daophot control (unsaturated stars in the ZP window, star medians over frames)**

nrcb1 +0.003 (N 504), nrcb3 -0.025 (N 508); raw b3-b1 -0.028 +/- 0.001; mag-matched -0.028 +/- 0.001; mag+n2 -0.028 +/- 0.001; mag+abkg -0.031 +/- 0.002; mag+n2+abkg -0.031 +/- 0.003

| control n2 bin | nrcb1 median (N) | nrcb3 median (N) | mag-matched b3-b1 |
|---|---|---|---|
| -1e-09-2.5 | +0.011 (32) | -0.017 (37) | -0.024 +/- 0.008 |
| 2.5-5 | +0.005 (149) | -0.021 (88) | -0.025 +/- 0.002 |
| 5-8 | +0.001 (162) | -0.024 (128) | -0.025 +/- 0.002 |
| 8-26 | +0.001 (161) | -0.029 (248) | -0.032 +/- 0.002 |

| control abkg bin | nrcb1 median (N) | nrcb3 median (N) | mag-matched b3-b1 |
|---|---|---|---|
| 2.82-6.45 | -0.002 (243) | -0.020 (88) | -0.018 +/- 0.002 |
| 6.45-8.25 | +0.008 (117) | -0.026 (180) | -0.034 +/- 0.002 |
| 8.25-9.82 | +0.007 (89) | -0.028 (131) | -0.035 +/- 0.003 |
| 9.82-48.2 | +0.018 (32) | -0.026 (107) | -0.043 +/- 0.011 |

| control edge bin | nrcb1 median (N) | nrcb3 median (N) | mag-matched b3-b1 |
|---|---|---|---|
| -1-50 | +0.011 (45) | -0.026 (43) | -0.036 +/- 0.007 |
| 50-150 | +0.008 (93) | -0.024 (69) | -0.034 +/- 0.004 |
| 150-400 | +0.004 (176) | -0.026 (202) | -0.030 +/- 0.002 |
| 400-3e+03 | -0.001 (190) | -0.024 (194) | -0.025 +/- 0.002 |


### R_* references used by the SW crf frames (distinct per band/detector)

| band | det | R_SATURA | R_LINEAR | R_GAIN | R_READNO | R_DARK | R_FLAT | CRDS_CTX |
|---|---|---|---|---|---|---|---|---|
| F150W | nrca1 | jwst_nircam_saturation_0114.fits | jwst_nircam_linearity_0056.fits | jwst_nircam_gain_0092.fits | jwst_nircam_readnoise_0321.fits | jwst_nircam_dark_0469.fits | jwst_nircam_flat_0657.fits | jwst_1595.pmap |
| F150W | nrca2 | jwst_nircam_saturation_0113.fits | jwst_nircam_linearity_0051.fits | jwst_nircam_gain_0093.fits | jwst_nircam_readnoise_0319.fits | jwst_nircam_dark_0470.fits | jwst_nircam_flat_0727.fits | jwst_1595.pmap |
| F150W | nrca3 | jwst_nircam_saturation_0111.fits | jwst_nircam_linearity_0053.fits | jwst_nircam_gain_0091.fits | jwst_nircam_readnoise_0322.fits | jwst_nircam_dark_0474.fits | jwst_nircam_flat_0689.fits | jwst_1595.pmap |
| F150W | nrca4 | jwst_nircam_saturation_0107.fits | jwst_nircam_linearity_0055.fits | jwst_nircam_gain_0096.fits | jwst_nircam_readnoise_0318.fits | jwst_nircam_dark_0468.fits | jwst_nircam_flat_0685.fits | jwst_1595.pmap |
| F150W | nrcb1 | jwst_nircam_saturation_0106.fits | jwst_nircam_linearity_0057.fits | jwst_nircam_gain_0090.fits | jwst_nircam_readnoise_0315.fits | jwst_nircam_dark_0475.fits | jwst_nircam_flat_0738.fits | jwst_1595.pmap |
| F150W | nrcb2 | jwst_nircam_saturation_0108.fits | jwst_nircam_linearity_0048.fits | jwst_nircam_gain_0088.fits | jwst_nircam_readnoise_0316.fits | jwst_nircam_dark_0477.fits | jwst_nircam_flat_0675.fits | jwst_1595.pmap |
| F150W | nrcb3 | jwst_nircam_saturation_0109.fits | jwst_nircam_linearity_0054.fits | jwst_nircam_gain_0094.fits | jwst_nircam_readnoise_0323.fits | jwst_nircam_dark_0476.fits | jwst_nircam_flat_0641.fits | jwst_1595.pmap |
| F150W | nrcb4 | jwst_nircam_saturation_0112.fits | jwst_nircam_linearity_0050.fits | jwst_nircam_gain_0095.fits | jwst_nircam_readnoise_0324.fits | jwst_nircam_dark_0472.fits | jwst_nircam_flat_0645.fits | jwst_1595.pmap |
| F162M | nrca1 | jwst_nircam_saturation_0114.fits | jwst_nircam_linearity_0056.fits | jwst_nircam_gain_0092.fits | jwst_nircam_readnoise_0321.fits | jwst_nircam_dark_0469.fits | jwst_nircam_flat_0707.fits | jwst_1568.pmap |
| F162M | nrca2 | jwst_nircam_saturation_0113.fits | jwst_nircam_linearity_0051.fits | jwst_nircam_gain_0093.fits | jwst_nircam_readnoise_0319.fits | jwst_nircam_dark_0470.fits | jwst_nircam_flat_0665.fits | jwst_1568.pmap |
| F162M | nrca3 | jwst_nircam_saturation_0111.fits | jwst_nircam_linearity_0053.fits | jwst_nircam_gain_0091.fits | jwst_nircam_readnoise_0322.fits | jwst_nircam_dark_0474.fits | jwst_nircam_flat_0651.fits | jwst_1568.pmap |
| F162M | nrca4 | jwst_nircam_saturation_0107.fits | jwst_nircam_linearity_0055.fits | jwst_nircam_gain_0096.fits | jwst_nircam_readnoise_0318.fits | jwst_nircam_dark_0468.fits | jwst_nircam_flat_0716.fits | jwst_1568.pmap |
| F162M | nrcb1 | jwst_nircam_saturation_0106.fits | jwst_nircam_linearity_0057.fits | jwst_nircam_gain_0090.fits | jwst_nircam_readnoise_0315.fits | jwst_nircam_dark_0475.fits | jwst_nircam_flat_0724.fits | jwst_1568.pmap |
| F162M | nrcb2 | jwst_nircam_saturation_0108.fits | jwst_nircam_linearity_0048.fits | jwst_nircam_gain_0088.fits | jwst_nircam_readnoise_0316.fits | jwst_nircam_dark_0477.fits | jwst_nircam_flat_0677.fits | jwst_1568.pmap |
| F162M | nrcb3 | jwst_nircam_saturation_0109.fits | jwst_nircam_linearity_0054.fits | jwst_nircam_gain_0094.fits | jwst_nircam_readnoise_0323.fits | jwst_nircam_dark_0476.fits | jwst_nircam_flat_0701.fits | jwst_1568.pmap |
| F162M | nrcb4 | jwst_nircam_saturation_0112.fits | jwst_nircam_linearity_0050.fits | jwst_nircam_gain_0095.fits | jwst_nircam_readnoise_0324.fits | jwst_nircam_dark_0472.fits | jwst_nircam_flat_0643.fits | jwst_1568.pmap |
| F182M | nrca1 | jwst_nircam_saturation_0114.fits | jwst_nircam_linearity_0056.fits | jwst_nircam_gain_0092.fits | jwst_nircam_readnoise_0321.fits | jwst_nircam_dark_0469.fits | jwst_nircam_flat_0683.fits | jwst_1568.pmap |
| F182M | nrca2 | jwst_nircam_saturation_0113.fits | jwst_nircam_linearity_0051.fits | jwst_nircam_gain_0093.fits | jwst_nircam_readnoise_0319.fits | jwst_nircam_dark_0470.fits | jwst_nircam_flat_0688.fits | jwst_1568.pmap |
| F182M | nrca3 | jwst_nircam_saturation_0111.fits | jwst_nircam_linearity_0053.fits | jwst_nircam_gain_0091.fits | jwst_nircam_readnoise_0322.fits | jwst_nircam_dark_0474.fits | jwst_nircam_flat_0731.fits | jwst_1568.pmap |
| F182M | nrca4 | jwst_nircam_saturation_0107.fits | jwst_nircam_linearity_0055.fits | jwst_nircam_gain_0096.fits | jwst_nircam_readnoise_0318.fits | jwst_nircam_dark_0468.fits | jwst_nircam_flat_0729.fits | jwst_1568.pmap |
| F182M | nrcb1 | jwst_nircam_saturation_0106.fits | jwst_nircam_linearity_0057.fits | jwst_nircam_gain_0090.fits | jwst_nircam_readnoise_0315.fits | jwst_nircam_dark_0475.fits | jwst_nircam_flat_0687.fits | jwst_1568.pmap |
| F182M | nrcb2 | jwst_nircam_saturation_0108.fits | jwst_nircam_linearity_0048.fits | jwst_nircam_gain_0088.fits | jwst_nircam_readnoise_0316.fits | jwst_nircam_dark_0477.fits | jwst_nircam_flat_0649.fits | jwst_1568.pmap |
| F182M | nrcb3 | jwst_nircam_saturation_0109.fits | jwst_nircam_linearity_0054.fits | jwst_nircam_gain_0094.fits | jwst_nircam_readnoise_0323.fits | jwst_nircam_dark_0476.fits | jwst_nircam_flat_0739.fits | jwst_1568.pmap |
| F182M | nrcb4 | jwst_nircam_saturation_0112.fits | jwst_nircam_linearity_0050.fits | jwst_nircam_gain_0095.fits | jwst_nircam_readnoise_0324.fits | jwst_nircam_dark_0472.fits | jwst_nircam_flat_0709.fits | jwst_1568.pmap |
| F200W | nrca1 | jwst_nircam_saturation_0114.fits | jwst_nircam_linearity_0056.fits | jwst_nircam_gain_0092.fits | jwst_nircam_readnoise_0321.fits | jwst_nircam_dark_0469.fits | jwst_nircam_flat_0695.fits | jwst_1568.pmap |
| F200W | nrca2 | jwst_nircam_saturation_0113.fits | jwst_nircam_linearity_0051.fits | jwst_nircam_gain_0093.fits | jwst_nircam_readnoise_0319.fits | jwst_nircam_dark_0470.fits | jwst_nircam_flat_0654.fits | jwst_1568.pmap |
| F200W | nrca3 | jwst_nircam_saturation_0111.fits | jwst_nircam_linearity_0053.fits | jwst_nircam_gain_0091.fits | jwst_nircam_readnoise_0322.fits | jwst_nircam_dark_0474.fits | jwst_nircam_flat_0639.fits | jwst_1568.pmap |
| F200W | nrca4 | jwst_nircam_saturation_0107.fits | jwst_nircam_linearity_0055.fits | jwst_nircam_gain_0096.fits | jwst_nircam_readnoise_0318.fits | jwst_nircam_dark_0468.fits | jwst_nircam_flat_0653.fits | jwst_1568.pmap |
| F200W | nrcb1 | jwst_nircam_saturation_0106.fits | jwst_nircam_linearity_0057.fits | jwst_nircam_gain_0090.fits | jwst_nircam_readnoise_0315.fits | jwst_nircam_dark_0475.fits | jwst_nircam_flat_0740.fits | jwst_1568.pmap |
| F200W | nrcb2 | jwst_nircam_saturation_0108.fits | jwst_nircam_linearity_0048.fits | jwst_nircam_gain_0088.fits | jwst_nircam_readnoise_0316.fits | jwst_nircam_dark_0477.fits | jwst_nircam_flat_0712.fits | jwst_1568.pmap |
| F200W | nrcb3 | jwst_nircam_saturation_0109.fits | jwst_nircam_linearity_0054.fits | jwst_nircam_gain_0094.fits | jwst_nircam_readnoise_0323.fits | jwst_nircam_dark_0476.fits | jwst_nircam_flat_0672.fits | jwst_1568.pmap |
| F200W | nrcb4 | jwst_nircam_saturation_0112.fits | jwst_nircam_linearity_0050.fits | jwst_nircam_gain_0095.fits | jwst_nircam_readnoise_0324.fits | jwst_nircam_dark_0472.fits | jwst_nircam_flat_0720.fits | jwst_1568.pmap |

### Per detector: saturation threshold, linearity correction, gain

| band | det | sat ref | sat med full-det (DN) | sat med at star rows (DN) | lin ref | lin corr at 50/70/90% sat, full-det median | lin corr at star rows | gain ref | gain med full-det | gain at star rows |
|---|---|---|---|---|---|---|---|---|---|---|
| F150W | nrca1 | jwst_nircam_saturation_0114.fits | 54519 | 54945 | jwst_nircam_linearity_0056.fits | 1.0422/1.0624/1.1059 | 1.0397/1.0580/1.1089 | jwst_nircam_gain_0092.fits | 2.050 | 2.050 |
| F150W | nrca2 | jwst_nircam_saturation_0113.fits | 56104 | 56348 | jwst_nircam_linearity_0051.fits | 1.0424/1.0695/1.1618 | 1.0447/1.0713/1.1583 | jwst_nircam_gain_0093.fits | 2.050 | 2.050 |
| F150W | nrca3 | jwst_nircam_saturation_0111.fits | 53029 | 53164 | jwst_nircam_linearity_0053.fits | 1.0356/1.0458/1.0917 | 1.0353/1.0454/1.0980 | jwst_nircam_gain_0091.fits | 2.050 | 2.050 |
| F150W | nrca4 | jwst_nircam_saturation_0107.fits | 53929 | 53987 | jwst_nircam_linearity_0055.fits | 1.0205/1.0300/1.0977 | 1.0227/1.0328/1.0938 | jwst_nircam_gain_0096.fits | 2.050 | 2.050 |
| F150W | nrcb1 | jwst_nircam_saturation_0106.fits | 57872 | 57948 | jwst_nircam_linearity_0057.fits | 1.0268/1.0446/1.1547 | 1.0285/1.0446/1.1475 | jwst_nircam_gain_0090.fits | 2.050 | 2.050 |
| F150W | nrcb2 | jwst_nircam_saturation_0108.fits | 57850 | 57818 | jwst_nircam_linearity_0048.fits | 1.0502/1.0746/1.1518 | 1.0476/1.0716/1.1577 | jwst_nircam_gain_0088.fits | 2.050 | 2.050 |
| F150W | nrcb3 | jwst_nircam_saturation_0109.fits | 57446 | 57238 | jwst_nircam_linearity_0054.fits | 1.0353/1.0535/1.1441 | 1.0371/1.0541/1.1335 | jwst_nircam_gain_0094.fits | 2.050 | 2.050 |
| F150W | nrcb4 | jwst_nircam_saturation_0112.fits | 57935 | 57991 | jwst_nircam_linearity_0050.fits | 1.0441/1.0653/1.1506 | 1.0421/1.0636/1.1556 | jwst_nircam_gain_0095.fits | 2.050 | 2.050 |
| F162M | nrca1 | jwst_nircam_saturation_0114.fits | 54519 | 55136 | jwst_nircam_linearity_0056.fits | 1.0422/1.0624/1.1059 | 1.0391/1.0573/1.1104 | jwst_nircam_gain_0092.fits | 2.050 | 2.050 |
| F162M | nrca2 | jwst_nircam_saturation_0113.fits | 56104 | 56192 | jwst_nircam_linearity_0051.fits | 1.0424/1.0695/1.1618 | 1.0449/1.0710/1.1568 | jwst_nircam_gain_0093.fits | 2.050 | 2.050 |
| F162M | nrca3 | jwst_nircam_saturation_0111.fits | 53029 | 53061 | jwst_nircam_linearity_0053.fits | 1.0356/1.0458/1.0917 | 1.0342/1.0439/1.0968 | jwst_nircam_gain_0091.fits | 2.050 | 2.050 |
| F162M | nrca4 | jwst_nircam_saturation_0107.fits | 53929 | 53837 | jwst_nircam_linearity_0055.fits | 1.0205/1.0300/1.0977 | 1.0226/1.0321/1.0931 | jwst_nircam_gain_0096.fits | 2.050 | 2.050 |
| F162M | nrcb1 | jwst_nircam_saturation_0106.fits | 57872 | 57966 | jwst_nircam_linearity_0057.fits | 1.0268/1.0446/1.1547 | 1.0284/1.0445/1.1477 | jwst_nircam_gain_0090.fits | 2.050 | 2.050 |
| F162M | nrcb2 | jwst_nircam_saturation_0108.fits | 57850 | 57844 | jwst_nircam_linearity_0048.fits | 1.0502/1.0746/1.1518 | 1.0480/1.0719/1.1583 | jwst_nircam_gain_0088.fits | 2.050 | 2.050 |
| F162M | nrcb3 | jwst_nircam_saturation_0109.fits | 57446 | 57215 | jwst_nircam_linearity_0054.fits | 1.0353/1.0535/1.1441 | 1.0371/1.0542/1.1330 | jwst_nircam_gain_0094.fits | 2.050 | 2.050 |
| F162M | nrcb4 | jwst_nircam_saturation_0112.fits | 57935 | 58008 | jwst_nircam_linearity_0050.fits | 1.0441/1.0653/1.1506 | 1.0416/1.0634/1.1578 | jwst_nircam_gain_0095.fits | 2.050 | 2.050 |
| F182M | nrca1 | jwst_nircam_saturation_0114.fits | 54519 | 55256 | jwst_nircam_linearity_0056.fits | 1.0422/1.0624/1.1059 | 1.0394/1.0580/1.1137 | jwst_nircam_gain_0092.fits | 2.050 | 2.050 |
| F182M | nrca2 | jwst_nircam_saturation_0113.fits | 56104 | 56402 | jwst_nircam_linearity_0051.fits | 1.0424/1.0695/1.1618 | 1.0432/1.0696/1.1565 | jwst_nircam_gain_0093.fits | 2.050 | 2.050 |
| F182M | nrca3 | jwst_nircam_saturation_0111.fits | 53029 | 53067 | jwst_nircam_linearity_0053.fits | 1.0356/1.0458/1.0917 | 1.0351/1.0457/1.1020 | jwst_nircam_gain_0091.fits | 2.050 | 2.050 |
| F182M | nrca4 | jwst_nircam_saturation_0107.fits | 53929 | 53977 | jwst_nircam_linearity_0055.fits | 1.0205/1.0300/1.0977 | 1.0234/1.0319/1.0924 | jwst_nircam_gain_0096.fits | 2.050 | 2.050 |
| F182M | nrcb1 | jwst_nircam_saturation_0106.fits | 57872 | 57990 | jwst_nircam_linearity_0057.fits | 1.0268/1.0446/1.1547 | 1.0285/1.0446/1.1480 | jwst_nircam_gain_0090.fits | 2.050 | 2.050 |
| F182M | nrcb2 | jwst_nircam_saturation_0108.fits | 57850 | 58059 | jwst_nircam_linearity_0048.fits | 1.0502/1.0746/1.1518 | 1.0483/1.0719/1.1607 | jwst_nircam_gain_0088.fits | 2.050 | 2.050 |
| F182M | nrcb3 | jwst_nircam_saturation_0109.fits | 57446 | 57166 | jwst_nircam_linearity_0054.fits | 1.0353/1.0535/1.1441 | 1.0372/1.0542/1.1324 | jwst_nircam_gain_0094.fits | 2.050 | 2.050 |
| F182M | nrcb4 | jwst_nircam_saturation_0112.fits | 57935 | 58081 | jwst_nircam_linearity_0050.fits | 1.0441/1.0653/1.1506 | 1.0416/1.0631/1.1587 | jwst_nircam_gain_0095.fits | 2.050 | 2.050 |
| F200W | nrca1 | jwst_nircam_saturation_0114.fits | 54519 | 55362 | jwst_nircam_linearity_0056.fits | 1.0422/1.0624/1.1059 | 1.0394/1.0576/1.1126 | jwst_nircam_gain_0092.fits | 2.050 | 2.050 |
| F200W | nrca2 | jwst_nircam_saturation_0113.fits | 56104 | 55949 | jwst_nircam_linearity_0051.fits | 1.0424/1.0695/1.1618 | 1.0448/1.0704/1.1532 | jwst_nircam_gain_0093.fits | 2.050 | 2.050 |
| F200W | nrca3 | jwst_nircam_saturation_0111.fits | 53029 | 53071 | jwst_nircam_linearity_0053.fits | 1.0356/1.0458/1.0917 | 1.0350/1.0449/1.0990 | jwst_nircam_gain_0091.fits | 2.050 | 2.050 |
| F200W | nrca4 | jwst_nircam_saturation_0107.fits | 53929 | 54051 | jwst_nircam_linearity_0055.fits | 1.0205/1.0300/1.0977 | 1.0230/1.0327/1.0902 | jwst_nircam_gain_0096.fits | 2.050 | 2.050 |
| F200W | nrcb1 | jwst_nircam_saturation_0106.fits | 57872 | 58001 | jwst_nircam_linearity_0057.fits | 1.0268/1.0446/1.1547 | 1.0287/1.0447/1.1467 | jwst_nircam_gain_0090.fits | 2.050 | 2.050 |
| F200W | nrcb2 | jwst_nircam_saturation_0108.fits | 57850 | 57940 | jwst_nircam_linearity_0048.fits | 1.0502/1.0746/1.1518 | 1.0473/1.0714/1.1594 | jwst_nircam_gain_0088.fits | 2.050 | 2.050 |
| F200W | nrcb3 | jwst_nircam_saturation_0109.fits | 57446 | 57177 | jwst_nircam_linearity_0054.fits | 1.0353/1.0535/1.1441 | 1.0370/1.0544/1.1333 | jwst_nircam_gain_0094.fits | 2.050 | 2.050 |
| F200W | nrcb4 | jwst_nircam_saturation_0112.fits | 57935 | 57924 | jwst_nircam_linearity_0050.fits | 1.0441/1.0653/1.1506 | 1.0414/1.0630/1.1544 | jwst_nircam_gain_0095.fits | 2.050 | 2.050 |

### Group-0 / ZEROFRAME DN at satstar cores (5x5 peak minus 15-25 px ring median), per detector

| band | det | N rows | g0 peak DN median (16/84%) | g0 peak / sat ref | ZEROFRAME peak DN median | first saturated group, median (0 = saturated in group 0) | frac saturated in g0 |
|---|---|---|---|---|---|---|---|
| F150W | nrca1 | 273 | 9458 (5196/25437) | 0.172 | 3868 | 7.0 | 0.00 |
| F150W | nrca2 | 329 | 11335 (5907/37188) | 0.195 | 4830 | 7.0 | 0.00 |
| F150W | nrca3 | 343 | 13336 (5375/37269) | 0.254 | 5691 | 7.0 | 0.00 |
| F150W | nrca4 | 383 | 7564 (4605/24552) | 0.137 | 3343 | 7.0 | 0.00 |
| F150W | nrcb1 | 2513 | 11978 (5650/30901) | 0.207 | 4932 | 7.0 | 0.00 |
| F150W | nrcb2 | 959 | 9986 (5659/24906) | 0.174 | 4278 | 7.0 | 0.00 |
| F150W | nrcb3 | 2660 | 12475 (6047/36971) | 0.216 | 5161 | 7.0 | 0.00 |
| F150W | nrcb4 | 664 | 9053 (5155/30329) | 0.156 | 3956 | 7.0 | 0.00 |
| F162M | nrca1 | 191 | 7420 (4944/19735) | 0.133 | 3390 | 7.0 | 0.00 |
| F162M | nrca2 | 230 | 12672 (5189/29915) | 0.215 | 5003 | 7.0 | 0.00 |
| F162M | nrca3 | 266 | 12605 (5469/31407) | 0.240 | 5070 | 7.0 | 0.00 |
| F162M | nrca4 | 238 | 7946 (4412/22461) | 0.151 | 3344 | 7.0 | 0.00 |
| F162M | nrcb1 | 1921 | 9862 (5412/25486) | 0.169 | 4044 | 7.0 | 0.00 |
| F162M | nrcb2 | 686 | 8349 (5331/20274) | 0.143 | 3626 | 7.0 | 0.00 |
| F162M | nrcb3 | 2303 | 10772 (5456/36641) | 0.187 | 4534 | 7.0 | 0.00 |
| F162M | nrcb4 | 432 | 8415 (5214/24350) | 0.145 | 3740 | 7.0 | 0.00 |
| F182M | nrca1 | 68 | 28644 (15197/44373) | 0.529 | 11324 | 7.0 | 0.00 |
| F182M | nrca2 | 116 | 25522 (15555/42178) | 0.451 | 10797 | 7.0 | 0.00 |
| F182M | nrca3 | 133 | 24509 (16579/40430) | 0.463 | 10252 | 7.0 | 0.00 |
| F182M | nrca4 | 75 | 25828 (18258/43617) | 0.492 | 11180 | 7.0 | 0.00 |
| F182M | nrcb1 | 879 | 21715 (15195/41144) | 0.374 | 9029 | 7.0 | 0.00 |
| F182M | nrcb2 | 257 | 21923 (15629/38435) | 0.375 | 9238 | 7.0 | 0.00 |
| F182M | nrcb3 | 1192 | 26224 (15623/47257) | 0.456 | 10704 | 7.0 | 0.00 |
| F182M | nrcb4 | 154 | 25485 (17950/40617) | 0.436 | 10512 | 7.0 | 0.00 |
| F200W | nrca1 | 59 | 34098 (28128/46595) | 0.613 | 14881 | 7.0 | 0.00 |
| F200W | nrca2 | 86 | 38118 (26954/48953) | 0.672 | 18306 | 7.0 | 0.00 |
| F200W | nrca3 | 121 | 36559 (28528/46411) | 0.686 | 16676 | 7.0 | 0.00 |
| F200W | nrca4 | 56 | 39711 (28732/48663) | 0.733 | 18552 | 7.0 | 0.00 |
| F200W | nrcb1 | 793 | 34857 (26589/47625) | 0.602 | 15154 | 7.0 | 0.00 |
| F200W | nrcb2 | 227 | 35748 (26770/44546) | 0.614 | 16039 | 7.0 | 0.00 |
| F200W | nrcb3 | 926 | 37800 (26792/48046) | 0.660 | 17215 | 7.0 | 0.00 |
| F200W | nrcb4 | 144 | 36848 (30274/45879) | 0.635 | 16908 | 7.0 | 0.00 |

### Group-0 and ZEROFRAME core DN, nrcb3 vs nrcb1 at matched dolphot magnitude (1-mag bins; ratio as 2.5 log10(b3/b1), positive = nrcb3 brighter in DN)

| band | quantity | 13-14 | 14-15 | 15-16 | 16-17 | 17-18 | 18-19 | weighted mean +/- boot |
|---|---|---|---|---|---|---|---|---|
| F150W | g0 | - | +0.011 | +0.030 | +0.049 | -0.012 | +0.050 | +0.021 +/- 0.010 |
| F150W | zf | - | +0.076 | +0.074 | +0.026 | +0.017 | +0.052 | +0.035 +/- 0.012 |
| F150W | d01 | - | -0.261 | -0.100 | -0.007 | -0.009 | +0.037 | -0.020 +/- 0.011 |
| F162M | g0 | +0.009 | +0.045 | +0.095 | +0.020 | +0.021 | - | +0.039 +/- 0.013 |
| F162M | zf | +0.048 | +0.109 | +0.076 | -0.003 | +0.060 | - | +0.040 +/- 0.016 |
| F162M | d01 | -0.171 | -0.168 | +0.030 | +0.012 | +0.017 | - | -0.006 +/- 0.014 |
| F182M | g0 | +0.012 | +0.031 | +0.081 | +0.014 | - | - | +0.060 +/- 0.013 |
| F182M | zf | +0.031 | +0.063 | +0.081 | -0.023 | - | - | +0.069 +/- 0.015 |
| F182M | d01 | -0.135 | -0.089 | -0.020 | +0.001 | - | - | -0.045 +/- 0.016 |
| F200W | g0 | +0.011 | +0.018 | +0.057 | +0.096 | - | - | +0.044 +/- 0.012 |
| F200W | zf | +0.011 | +0.041 | +0.067 | +0.216 | - | - | +0.059 +/- 0.021 |
| F200W | d01 | +0.057 | -0.027 | -0.062 | +0.053 | - | - | -0.047 +/- 0.036 |

First saturated group of the peak pixel (A/D or sat-ref criterion; 7 = never within 7 groups), median, and fraction at the A/D limit in the last group:

| band | nrcb1 median gsat / frac last group >= 64000 | nrcb3 |
|---|---|---|
| F150W | 7 / 0.04 | 7 / 0.05 |
| F162M | 7 / 0.04 | 7 / 0.06 |
| F182M | 7 / 0.06 | 7 / 0.06 |
| F200W | 7 / 0.07 | 7 / 0.10 |

# Round-3 satrefit variants split by detector

Star-level medians (median over the exposures of each star), dm = variant magnitude - dolphot - ZP; negative = ours bright. Error = 1.25 MAD/sqrt(N). Only the refit frames (nrcb1, nrcb3; F150W, F200W; 4 exposures each) are included.


## F150W: 5430 rows, 1479 star-detector medians (nrcb1 706, nrcb3 773)

| variant | det | 14-15 | 15-16 | 16-17 | 17-18 | 18-19 | all |
|---|---|---|---|---|---|---|---|
| final | nrcb1 | -0.035 (23) | -0.044 (65) | -0.028 (188) | -0.013 (273) | +0.004 (156) | -0.017 (706) |
| final | nrcb3 | -0.107 (34) | -0.105 (94) | -0.085 (197) | -0.068 (293) | -0.055 (153) | -0.075 (773) |
| final | b3 - b1 | -0.071 +/- 0.014 | -0.061 +/- 0.005 | -0.058 +/- 0.003 | -0.054 +/- 0.003 | -0.058 +/- 0.005 | -0.058 +/- 0.002 |
| uncapped | nrcb1 | -0.146 (23) | -0.104 (65) | -0.095 (188) | -0.048 (273) | -0.028 (156) | -0.060 (706) |
| uncapped | nrcb3 | -0.146 (34) | -0.129 (94) | -0.107 (197) | -0.079 (293) | -0.064 (153) | -0.089 (773) |
| uncapped | b3 - b1 | -0.000 +/- 0.024 | -0.025 +/- 0.011 | -0.012 +/- 0.006 | -0.030 +/- 0.004 | -0.036 +/- 0.005 | -0.029 +/- 0.004 |
| base+cap | nrcb1 | -0.035 (23) | -0.044 (65) | -0.028 (188) | -0.013 (273) | +0.004 (156) | -0.017 (706) |
| base+cap | nrcb3 | -0.107 (34) | -0.105 (94) | -0.085 (197) | -0.068 (293) | -0.055 (153) | -0.075 (773) |
| base+cap | b3 - b1 | -0.071 +/- 0.014 | -0.061 +/- 0.005 | -0.058 +/- 0.003 | -0.054 +/- 0.003 | -0.058 +/- 0.005 | -0.058 +/- 0.002 |
| bgfree | nrcb1 | -0.131 (23) | -0.087 (65) | -0.080 (188) | -0.038 (273) | -0.019 (156) | -0.049 (706) |
| bgfree | nrcb3 | -0.096 (34) | -0.102 (94) | -0.087 (197) | -0.066 (293) | -0.052 (153) | -0.073 (773) |
| bgfree | b3 - b1 | +0.034 +/- 0.016 | -0.015 +/- 0.011 | -0.007 +/- 0.006 | -0.028 +/- 0.004 | -0.033 +/- 0.005 | -0.024 +/- 0.003 |
| bgfree+cap | nrcb1 | -0.035 (23) | -0.042 (65) | -0.026 (188) | -0.008 (273) | +0.008 (156) | -0.015 (706) |
| bgfree+cap | nrcb3 | -0.088 (34) | -0.094 (94) | -0.077 (197) | -0.059 (293) | -0.047 (153) | -0.066 (773) |
| bgfree+cap | b3 - b1 | -0.053 +/- 0.014 | -0.052 +/- 0.005 | -0.051 +/- 0.004 | -0.051 +/- 0.003 | -0.055 +/- 0.005 | -0.051 +/- 0.002 |
| rw12 | nrcb1 | -0.081 (23) | -0.063 (65) | -0.068 (188) | -0.019 (273) | -0.005 (156) | -0.031 (706) |
| rw12 | nrcb3 | -0.099 (34) | -0.102 (94) | -0.091 (197) | -0.069 (293) | -0.056 (153) | -0.076 (773) |
| rw12 | b3 - b1 | -0.017 +/- 0.017 | -0.039 +/- 0.009 | -0.024 +/- 0.006 | -0.050 +/- 0.004 | -0.051 +/- 0.004 | -0.045 +/- 0.003 |
| rw12+cap | nrcb1 | -0.033 (23) | -0.038 (65) | -0.023 (188) | -0.003 (273) | +0.012 (156) | -0.008 (706) |
| rw12+cap | nrcb3 | -0.090 (34) | -0.096 (94) | -0.077 (197) | -0.060 (293) | -0.049 (153) | -0.068 (773) |
| rw12+cap | b3 - b1 | -0.057 +/- 0.011 | -0.058 +/- 0.006 | -0.054 +/- 0.004 | -0.057 +/- 0.003 | -0.061 +/- 0.004 | -0.060 +/- 0.002 |
| rw12+bgfree | nrcb1 | -0.062 (23) | -0.049 (65) | -0.050 (188) | -0.012 (273) | +0.001 (156) | -0.020 (706) |
| rw12+bgfree | nrcb3 | -0.066 (34) | -0.086 (94) | -0.074 (197) | -0.058 (293) | -0.050 (153) | -0.064 (773) |
| rw12+bgfree | b3 - b1 | -0.004 +/- 0.012 | -0.037 +/- 0.010 | -0.024 +/- 0.005 | -0.047 +/- 0.004 | -0.052 +/- 0.004 | -0.044 +/- 0.003 |
| rw12+bgfree+cap | nrcb1 | -0.030 (23) | -0.026 (65) | -0.021 (188) | +0.004 (273) | +0.014 (156) | -0.004 (706) |
| rw12+bgfree+cap | nrcb3 | -0.065 (34) | -0.082 (94) | -0.067 (197) | -0.053 (293) | -0.042 (153) | -0.058 (773) |
| rw12+bgfree+cap | b3 - b1 | -0.035 +/- 0.011 | -0.056 +/- 0.006 | -0.046 +/- 0.004 | -0.058 +/- 0.003 | -0.056 +/- 0.004 | -0.055 +/- 0.002 |
| bgfree+v7b+cap | nrcb1 | -0.016 (23) | -0.013 (65) | -0.008 (188) | +0.014 (273) | +0.023 (156) | +0.009 (706) |
| bgfree+v7b+cap | nrcb3 | -0.031 (34) | -0.050 (94) | -0.040 (197) | -0.029 (293) | -0.021 (153) | -0.032 (773) |
| bgfree+v7b+cap | b3 - b1 | -0.016 +/- 0.008 | -0.036 +/- 0.008 | -0.031 +/- 0.005 | -0.043 +/- 0.004 | -0.044 +/- 0.005 | -0.041 +/- 0.002 |

## F200W: 1819 rows, 515 star-detector medians (nrcb1 233, nrcb3 282)

| variant | det | 13-14.5 | 14.5-15 | 15-15.5 | 15.5-16 | 16-18 | all |
|---|---|---|---|---|---|---|---|
| final | nrcb1 | +0.003 (36) | -0.023 (36) | -0.025 (63) | -0.026 (76) | -0.009 (22) | -0.021 (233) |
| final | nrcb3 | -0.084 (44) | -0.092 (49) | -0.105 (84) | -0.082 (91) | -0.070 (14) | -0.093 (282) |
| final | b3 - b1 | -0.087 +/- 0.004 | -0.069 +/- 0.005 | -0.081 +/- 0.004 | -0.056 +/- 0.004 | -0.062 +/- 0.014 | -0.072 +/- 0.003 |
| uncapped | nrcb1 | -0.074 (36) | -0.062 (36) | -0.080 (63) | -0.069 (76) | -0.026 (22) | -0.069 (233) |
| uncapped | nrcb3 | -0.123 (44) | -0.108 (49) | -0.129 (84) | -0.106 (91) | -0.078 (14) | -0.116 (282) |
| uncapped | b3 - b1 | -0.048 +/- 0.012 | -0.046 +/- 0.012 | -0.049 +/- 0.009 | -0.037 +/- 0.007 | -0.053 +/- 0.015 | -0.047 +/- 0.005 |
| base+cap | nrcb1 | +0.003 (36) | -0.023 (36) | -0.025 (63) | -0.026 (76) | -0.009 (22) | -0.021 (233) |
| base+cap | nrcb3 | -0.084 (44) | -0.092 (49) | -0.105 (84) | -0.082 (91) | -0.070 (14) | -0.093 (282) |
| base+cap | b3 - b1 | -0.087 +/- 0.004 | -0.069 +/- 0.005 | -0.081 +/- 0.004 | -0.056 +/- 0.004 | -0.062 +/- 0.014 | -0.072 +/- 0.003 |
| bgfree | nrcb1 | -0.060 (36) | -0.040 (36) | -0.065 (63) | -0.058 (76) | -0.005 (22) | -0.055 (233) |
| bgfree | nrcb3 | -0.099 (44) | -0.093 (49) | -0.105 (84) | -0.090 (91) | -0.058 (14) | -0.097 (282) |
| bgfree | b3 - b1 | -0.039 +/- 0.010 | -0.053 +/- 0.012 | -0.041 +/- 0.008 | -0.032 +/- 0.007 | -0.053 +/- 0.013 | -0.042 +/- 0.005 |
| bgfree+cap | nrcb1 | +0.003 (36) | -0.021 (36) | -0.025 (63) | -0.024 (76) | -0.001 (22) | -0.019 (233) |
| bgfree+cap | nrcb3 | -0.075 (44) | -0.083 (49) | -0.098 (84) | -0.078 (91) | -0.055 (14) | -0.084 (282) |
| bgfree+cap | b3 - b1 | -0.078 +/- 0.005 | -0.061 +/- 0.006 | -0.073 +/- 0.005 | -0.054 +/- 0.005 | -0.054 +/- 0.012 | -0.065 +/- 0.003 |
| rw12 | nrcb1 | -0.042 (36) | -0.030 (36) | -0.037 (63) | -0.038 (76) | -0.019 (22) | -0.036 (233) |
| rw12 | nrcb3 | -0.109 (44) | -0.099 (49) | -0.101 (84) | -0.089 (91) | -0.075 (14) | -0.097 (282) |
| rw12 | b3 - b1 | -0.067 +/- 0.009 | -0.069 +/- 0.008 | -0.064 +/- 0.007 | -0.051 +/- 0.006 | -0.056 +/- 0.016 | -0.061 +/- 0.004 |
| rw12+cap | nrcb1 | +0.005 (36) | -0.020 (36) | -0.016 (63) | -0.020 (76) | -0.004 (22) | -0.015 (233) |
| rw12+cap | nrcb3 | -0.084 (44) | -0.087 (49) | -0.095 (84) | -0.075 (91) | -0.069 (14) | -0.086 (282) |
| rw12+cap | b3 - b1 | -0.089 +/- 0.004 | -0.067 +/- 0.005 | -0.079 +/- 0.005 | -0.055 +/- 0.004 | -0.065 +/- 0.015 | -0.072 +/- 0.003 |
| rw12+bgfree | nrcb1 | -0.033 (36) | -0.022 (36) | -0.025 (63) | -0.030 (76) | -0.006 (22) | -0.026 (233) |
| rw12+bgfree | nrcb3 | -0.096 (44) | -0.084 (49) | -0.087 (84) | -0.079 (91) | -0.064 (14) | -0.085 (282) |
| rw12+bgfree | b3 - b1 | -0.063 +/- 0.010 | -0.062 +/- 0.009 | -0.062 +/- 0.006 | -0.048 +/- 0.006 | -0.058 +/- 0.015 | -0.059 +/- 0.004 |
| rw12+bgfree+cap | nrcb1 | +0.006 (36) | -0.013 (36) | -0.012 (63) | -0.018 (76) | +0.002 (22) | -0.011 (233) |
| rw12+bgfree+cap | nrcb3 | -0.075 (44) | -0.082 (49) | -0.086 (84) | -0.073 (91) | -0.058 (14) | -0.078 (282) |
| rw12+bgfree+cap | b3 - b1 | -0.081 +/- 0.006 | -0.069 +/- 0.006 | -0.074 +/- 0.005 | -0.055 +/- 0.004 | -0.060 +/- 0.012 | -0.066 +/- 0.003 |
| bgfree+v7b+cap | nrcb1 | +0.021 (36) | +0.009 (36) | +0.005 (63) | +0.004 (76) | +0.025 (22) | +0.012 (233) |
| bgfree+v7b+cap | nrcb3 | -0.044 (44) | -0.048 (49) | -0.054 (84) | -0.041 (91) | -0.025 (14) | -0.047 (282) |
| bgfree+v7b+cap | b3 - b1 | -0.065 +/- 0.007 | -0.057 +/- 0.008 | -0.059 +/- 0.006 | -0.045 +/- 0.005 | -0.051 +/- 0.016 | -0.058 +/- 0.003 |

# R(g0) curves per frame (satrefit/out3/*_rcurve.txt)


**F150W: R at fixed g0 (DN), per frame; mean and nrcb3/nrcb1 ratio**

| frame | det | 250 | 400 | 600 | 900 | 1300 | 1800 | 2500 | 3300 | mean R (g0 400-2500) | intrinsic scatter (400-2500) | sig_low / rn0 / gain / s_flat |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | nrcb1 | 0.0866 | 0.0875 | 0.0878 | 0.0877 | 0.0880 | 0.0879 | 0.0880 | 0.0875 | 0.0878 | 0.090 | 11.04 / 11.04 / 2.00 / 0.063 |
| 1 | nrcb3 | 0.0880 | 0.0897 | 0.0898 | 0.0889 | 0.0890 | 0.0880 | 0.0873 | 0.0856 | 0.0890 | 0.056 | 8.65 / 8.29 / 2.00 / 0.053 |
| 2 | nrcb1 | 0.0849 | 0.0866 | 0.0872 | 0.0880 | 0.0884 | 0.0881 | 0.0873 | 0.0858 | 0.0878 | 0.099 | 10.88 / 10.88 / 2.00 / 0.084 |
| 2 | nrcb3 | 0.0895 | 0.0899 | 0.0898 | 0.0885 | 0.0876 | 0.0877 | 0.0869 | 0.0862 | 0.0885 | 0.074 | 8.65 / 8.35 / 2.00 / 0.080 |
| 3 | nrcb1 | 0.0857 | 0.0869 | 0.0873 | 0.0883 | 0.0882 | 0.0877 | 0.0876 | nan | 0.0877 | 0.088 | 10.95 / 10.95 / 2.00 / 0.066 |
| 3 | nrcb3 | 0.0889 | 0.0898 | 0.0893 | 0.0886 | 0.0883 | 0.0881 | 0.0874 | 0.0863 | 0.0888 | 0.064 | 8.70 / 8.41 / 2.00 / 0.060 |
| 4 | nrcb1 | 0.0859 | 0.0865 | 0.0877 | 0.0874 | 0.0882 | 0.0886 | 0.0876 | 0.0866 | 0.0877 | 0.092 | 10.90 / 10.90 / 2.00 / 0.067 |
| 4 | nrcb3 | 0.0889 | 0.0896 | 0.0892 | 0.0886 | 0.0885 | 0.0878 | 0.0877 | 0.0865 | 0.0885 | 0.069 | 8.75 / 8.44 / 2.00 / 0.061 |
| mean of 4 | nrcb1 | 0.0858 | 0.0869 | 0.0875 | 0.0878 | 0.0882 | 0.0881 | 0.0876 | 0.0866 | | | |
| mean of 4 | nrcb3 | 0.0888 | 0.0897 | 0.0895 | 0.0887 | 0.0884 | 0.0879 | 0.0873 | 0.0862 | | | |
| ratio b3/b1 | | 1.035 | 1.033 | 1.023 | 1.009 | 1.002 | 0.998 | 0.997 | 0.994 | | | |

**F200W: R at fixed g0 (DN), per frame; mean and nrcb3/nrcb1 ratio**

| frame | det | 250 | 400 | 600 | 900 | 1300 | 1800 | 2500 | 3300 | mean R (g0 400-2500) | intrinsic scatter (400-2500) | sig_low / rn0 / gain / s_flat |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | nrcb1 | 0.0702 | 0.0730 | 0.0737 | 0.0743 | 0.0743 | 0.0743 | 0.0741 | nan | 0.0741 | 0.060 | 11.37 / 10.75 / 2.00 / 0.054 |
| 1 | nrcb3 | 0.0721 | 0.0742 | 0.0740 | 0.0738 | 0.0737 | 0.0737 | 0.0730 | 0.0718 | 0.0738 | 0.049 | 10.22 / 8.39 / 2.00 / 0.047 |
| 2 | nrcb1 | 0.0699 | 0.0738 | 0.0735 | 0.0743 | 0.0748 | 0.0743 | 0.0737 | 0.0726 | 0.0741 | 0.061 | 11.28 / 10.64 / 2.00 / 0.055 |
| 2 | nrcb3 | 0.0719 | 0.0739 | 0.0739 | 0.0734 | 0.0734 | 0.0735 | 0.0728 | 0.0718 | 0.0735 | 0.051 | 10.13 / 8.24 / 2.00 / 0.047 |
| 3 | nrcb1 | 0.0704 | 0.0732 | 0.0737 | 0.0744 | 0.0745 | 0.0741 | 0.0739 | 0.0736 | 0.0741 | 0.063 | 11.44 / 10.82 / 2.00 / 0.055 |
| 3 | nrcb3 | 0.0721 | 0.0742 | 0.0739 | 0.0736 | 0.0737 | 0.0736 | 0.0728 | 0.0715 | 0.0737 | 0.054 | 10.20 / 8.35 / 2.00 / 0.048 |
| 4 | nrcb1 | 0.0705 | 0.0737 | 0.0735 | 0.0742 | 0.0746 | 0.0742 | 0.0736 | nan | 0.0740 | 0.066 | 11.33 / 10.70 / 2.00 / 0.059 |
| 4 | nrcb3 | 0.0722 | 0.0742 | 0.0744 | 0.0734 | 0.0731 | 0.0731 | 0.0725 | nan | 0.0736 | 0.054 | 10.21 / 8.35 / 2.00 / 0.049 |
| mean of 4 | nrcb1 | 0.0702 | 0.0734 | 0.0736 | 0.0743 | 0.0746 | 0.0742 | 0.0738 | 0.0731 | | | |
| mean of 4 | nrcb3 | 0.0721 | 0.0741 | 0.0741 | 0.0736 | 0.0735 | 0.0734 | 0.0728 | 0.0717 | | | |
| ratio b3/b1 | | 1.026 | 1.010 | 1.006 | 0.990 | 0.986 | 0.990 | 0.986 | 0.981 | | | |
