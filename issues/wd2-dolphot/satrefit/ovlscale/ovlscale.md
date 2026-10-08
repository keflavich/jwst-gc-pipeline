# Internal flux-scale test: satstar fits versus daophot fits of the same star (wd2, main2)

Files: `ovlscale.py` (code; `python ovlscale.py BAND` per band, `python ovlscale.py --report` for tables and figure), `ovlscale.png`
(dm versus magnitude), `ovlscale_results_<band>.json` (all numbers), `ovlscale_stars_<band>_<sample>.npz` (per-star dm),
`ovlscale_tables.md` (generated tables, reproduced below), `log_<band>.txt`.
No dolphot data enter this test. Run tree: `tree_main2`, phase m7, code commit 5434f5e7.

## Result in brief

- Stars that saturate in some frames and are unsaturated in others sit at the saturation threshold. In SW bands they have 9-12 saturated
  pixels, AB 19-20 (F150W), 18.5-19.5 (F162M); in LW they lie at AB 18-19.5. No overlap stars exist at brighter magnitudes, so the test
  constrains the satstar scale at the threshold only.
- Strict sample (daophot row from a frame with no SATURATED DQ pixel in the 5x5 block, flags 0, not forced): median dm = m_satstar - m_daophot
  (final flux_fit, after the pooled wing calibration) is -0.002 (F150W, N = 231), -0.000 (F162M, N = 187), +0.045 (F250M, N = 106),
  +0.038 (F277W, N = 40), +0.031 (F300M, N = 81). F182M and F200W have no strict overlap stars (N = 0).
- The SW satstar scale agrees with daophot at the threshold to within about 0.005 mag (error of the median 0.004-0.005; scatter per star MAD 0.04-0.05).
  The LW satstar rows read faint by 0.03-0.05 mag relative to daophot, the same sign as the LW offset seen against dolphot (+0.03/+0.02 in F250M/F300M).
- Relaxed sample (daophot row from any frame without a satstar row, including frames whose DQ marks the core saturated): dm is
  -0.015 (F150W), -0.007 (F162M), -0.019 (F182M), -0.030 (F200W), +0.042 (F250M), +0.036 (F277W), +0.033 (F300M). In F150W the relaxed bins at AB 17.5-19
  read -0.043 to -0.051 and F200W at AB 16.5-17.5 reads -0.03 to -0.09, the sign and size of the SW offset against dolphot. The relaxed daophot rows
  may themselves be clipped (see caveats), so these values bound the bias but do not isolate the satstar scale.
- The satstar scale depends on detector. Per detector, the median dm of satstar rows against the same star's daophot flux spans -0.06 to +0.07 mag
  in the SW bands (nrcb3 -0.037 to -0.059, nrca3 -0.044 to +0.067, depending on band) and nrcalong +0.04 to +0.08 versus nrcblong 0.00 to +0.03 in LW.
  Daophot-only control stars show detector offsets below 0.003 mag, so the spread belongs to the satstar fits (or to what their frames contain),
  not to the daophot calibration.
- The merge replaces the daophot flux of these stars with the satstar median (228/231 F150W overlap stars have `replaced_saturated`); the daophot rows from
  the unsaturated frames do not enter the merged magnitude.

## Method

1. Per band, 32 frames (SW: 8 detectors x 4 dithers) or 8 frames (LW: nrcalong/nrcblong x 4). Rows:
   - S rows: `F*/pipeline/*_align_o005_crf_resbgsub_m7_satstar_catalog.fits`. Columns `flux_fit` (final), `flux_fit_precap` (before the saturated-core
     cap and the wing calibration), `flux_fit_raw` (after the cap, before the wing calibration). `final = raw / wingcal_ratio`. The pooled wing
     calibration is applied the way `load_satstar_catalog` applies it (`apply_pooled_wingcal(basepath=tree_main2, phase='m7')`): F182M and F200W have
     pooled tables; the other bands have none and keep `final = raw`.
   - D rows: `F*/*_resbgsub_m7_daophot_basic.fits`, column `flux_fit`, positions `skycoord_centroid`. Selection: `flags == 0`, not `forced_refit`,
     finite positive flux.
   - Units: both tables carry `flux_fit` in MJy/sr summed over the PSF. The merge converts either with `flux x MJy/sr x pixelscale_deg2`
     (`replace_saturated` for satstar rows). The same pixel area (PIXSCALE of the daophot header) converts both here. Magnitudes are AB. Only ratios enter dm.
   - The crf `DQ` extension supplies SATURATED (bit 2). `sat5` = any SATURATED pixel in the 5x5 block around the rounded fit position, `sat1` = the central pixel.
2. Stars: friends-of-friends clusters (0.1") of all S rows of the band (star centre = median position). For each star and frame, daophot rows within 0.1" of the
   centre are candidates (nearest per frame). A frame counts as saturated for the star when an S row lies within 0.3"; daophot rows from such frames are excluded.
3. Samples. `strict`: D rows with `sat5` false. `core`: `sat1` false (identical to strict within one star, so not tabulated further). `relaxed`: no DQ condition.
   Overlap star = at least one S row and at least one D row in the sample. `iso` = no other row (any frame) within 0.5" and beyond 0.15" of the centre brighter than
   0.2 times the star's satstar flux.
4. dm = m_S - m_D from the median flux over the star's S rows and over its D rows (negative = satstar brighter), for final, precap, raw. Bins use the daophot AB magnitude
   (independent of the satstar fit). Error of the median = 1.253 MAD / sqrt(N) (listed in parentheses in the binned table).
5. Pixel phase r = distance of the fitted position from the nearest pixel centre (0 to 0.71 px). Control: stars with no satstar row within 0.3" in any frame and at least three clean
   D rows in different frames, restricted to the 5-95 percentile magnitude range of the overlap stars; per row, m_i minus the median of the other frames (leave-one-out).
6. Merge check: overlap stars matched to `basic_merged_indivexp_photometry_tables_merged_resbgsub_m8_dedup.fits` within 0.1" (`skycoord_<band>`); `replaced_saturated_<band>` and
   `mag_ab_<band>` read.

## Merge logic (merge_catalogs.py, `replace_saturated`)

- `load_satstar_catalog` reads each exposure's m7 satstar catalog, applies the pooled wing calibration to rows with `wingcal_ratio == 1`, deduplicates across exposures
  (friends-of-friends), and by default (`SATSTAR_FLUX_STAT` = median) sets the star's flux to the median over exposures of the per-exposure final fluxes (`flux_median_fit`);
  the brightest exposure's value stays in `flux_brightest_fit`. Only exposures in which the star was fitted as saturated contribute.
- `replace_saturated` matches each consolidated satstar to the merged daophot catalog (0.05" SW, 0.08-0.1" LW; a mutual-nearest second pass at a wider radius) and
  overwrites flux, position and magnitudes of the matched row. A faint-replacement guard keeps the daophot flux when the satstar flux is below 0.8 times the daophot flux
  (3 of 231 F150W overlap stars; 1 of 187 F162M).
- Stars saturated in some frames are therefore merged from the satstar rows only. No blending of satstar and daophot fluxes takes place for them, and no per-frame zero point
  links the two. The merged magnitude of the overlap stars equals m_S(median) plus a constant +0.007 mag (SW) or +0.003 mag (LW), the same constant that appears
  for the 1-3 overlap stars that kept their daophot row (+0.005 to +0.012). The constant probably reflects the pixel-area value in the merge catalog metadata versus the
  PIXSCALE header value used here (not checked) and does not affect dm.

## Tables (generated by ovlscale.py --report)

### Summary, sample `strict`

| band | S rows | stars with D cand. | N overlap (all/iso) | final med (MAD) | precap med (MAD) | raw med (MAD) | iso final med |
|---|---|---|---|---|---|---|---|
| F150W | 9552 | 377 | 231 / 206 | -0.002 (0.044) | -0.025 (0.036) | -0.002 (0.044) | +0.001 |
| F162M | 7323 | 272 | 187 / 151 | -0.000 (0.051) | -0.021 (0.043) | -0.000 (0.051) | -0.000 |
| F182M | 3664 | 132 | 0 / 0 | - | - | - | - |
| F200W | 3428 | 150 | 0 / 0 | - | - | - | - |
| F250M | 4539 | 271 | 106 / 96 | +0.045 (0.043) | +0.037 (0.041) | +0.045 (0.043) | +0.045 |
| F277W | 5920 | 312 | 40 / 36 | +0.038 (0.054) | +0.037 (0.054) | +0.038 (0.054) | +0.038 |
| F300M | 4929 | 276 | 81 / 72 | +0.031 (0.057) | +0.031 (0.057) | +0.031 (0.057) | +0.032 |

### Summary, sample `core`

| band | S rows | stars with D cand. | N overlap (all/iso) | final med (MAD) | precap med (MAD) | raw med (MAD) | iso final med |
|---|---|---|---|---|---|---|---|
| F150W | 9552 | 377 | 232 / 206 | -0.002 (0.044) | -0.025 (0.036) | -0.002 (0.044) | +0.001 |
| F162M | 7323 | 272 | 187 / 151 | -0.000 (0.051) | -0.021 (0.043) | -0.000 (0.051) | -0.000 |
| F182M | 3664 | 132 | 0 / 0 | - | - | - | - |
| F200W | 3428 | 150 | 0 / 0 | - | - | - | - |
| F250M | 4539 | 271 | 106 / 96 | +0.045 (0.043) | +0.037 (0.041) | +0.045 (0.043) | +0.045 |
| F277W | 5920 | 312 | 40 / 36 | +0.038 (0.054) | +0.037 (0.054) | +0.038 (0.054) | +0.038 |
| F300M | 4929 | 276 | 81 / 72 | +0.031 (0.057) | +0.031 (0.057) | +0.031 (0.057) | +0.032 |

### Summary, sample `relaxed`

| band | S rows | stars with D cand. | N overlap (all/iso) | final med (MAD) | precap med (MAD) | raw med (MAD) | iso final med |
|---|---|---|---|---|---|---|---|
| F150W | 9552 | 377 | 337 / 273 | -0.015 (0.056) | -0.037 (0.046) | -0.015 (0.056) | -0.008 |
| F162M | 7323 | 272 | 252 / 184 | -0.007 (0.047) | -0.026 (0.041) | -0.007 (0.047) | -0.005 |
| F182M | 3664 | 132 | 119 / 92 | -0.019 (0.034) | -0.046 (0.040) | -0.023 (0.034) | -0.019 |
| F200W | 3428 | 150 | 138 / 116 | -0.030 (0.050) | -0.069 (0.045) | -0.039 (0.049) | -0.033 |
| F250M | 4539 | 271 | 205 / 186 | +0.042 (0.037) | +0.037 (0.037) | +0.042 (0.037) | +0.044 |
| F277W | 5920 | 312 | 220 / 195 | +0.036 (0.056) | +0.026 (0.049) | +0.036 (0.056) | +0.038 |
| F300M | 4929 | 276 | 217 / 181 | +0.033 (0.043) | +0.024 (0.035) | +0.033 (0.043) | +0.033 |

### Binned dm (isolated stars), bins in daophot AB magnitude; median (error of median) [N]


Sample `strict`

| band | bin (m_D, AB) | N | final | precap | raw |
|---|---|---|---|---|---|
| F150W | 19.0-19.5 | 19 | +0.006 (0.011) | -0.020 | +0.006 |
| F150W | 19.5-20.0 | 176 | +0.002 (0.004) | -0.022 | +0.002 |
| F162M | 18.5-19.0 | 134 | +0.000 (0.005) | -0.019 | +0.000 |
| F162M | 19.0-19.5 | 5 | -0.010 (0.013) | -0.024 | -0.010 |
| F250M | 18.0-18.5 | 22 | +0.095 (0.013) | +0.086 | +0.095 |
| F250M | 18.5-19.0 | 71 | +0.029 (0.005) | +0.029 | +0.029 |
| F277W | 19.5-20.0 | 28 | +0.043 (0.012) | +0.040 | +0.043 |
| F277W | 20.0-20.5 | 7 | -0.004 (0.016) | -0.004 | -0.004 |
| F300M | 18.5-19.0 | 51 | +0.049 (0.009) | +0.038 | +0.049 |
| F300M | 19.0-19.5 | 21 | +0.011 (0.010) | +0.011 | +0.011 |

Sample `relaxed`

| band | bin (m_D, AB) | N | final | precap | raw |
|---|---|---|---|---|---|
| F150W | 17.5-18.0 | 8 | -0.046 (0.011) | -0.066 | -0.046 |
| F150W | 18.0-18.5 | 14 | -0.051 (0.020) | -0.077 | -0.051 |
| F150W | 18.5-19.0 | 18 | -0.043 (0.014) | -0.048 | -0.043 |
| F150W | 19.0-19.5 | 41 | -0.020 (0.013) | -0.041 | -0.020 |
| F150W | 19.5-20.0 | 176 | +0.002 (0.004) | -0.022 | +0.002 |
| F162M | 16.5-17.0 | 6 | -0.033 (0.013) | -0.050 | -0.033 |
| F162M | 17.0-17.5 | 8 | -0.025 (0.018) | -0.043 | -0.025 |
| F162M | 17.5-18.0 | 6 | -0.027 (0.017) | -0.046 | -0.027 |
| F162M | 18.0-18.5 | 11 | -0.004 (0.029) | -0.008 | -0.004 |
| F162M | 18.5-19.0 | 138 | +0.000 (0.005) | -0.019 | +0.000 |
| F162M | 19.0-19.5 | 5 | -0.010 (0.013) | -0.024 | -0.010 |
| F182M | 17.0-17.5 | 25 | -0.005 (0.010) | -0.048 | -0.005 |
| F182M | 17.5-18.0 | 54 | -0.026 (0.007) | -0.052 | -0.030 |
| F182M | 18.0-18.5 | 7 | -0.030 (0.015) | -0.053 | -0.035 |
| F200W | 16.5-17.0 | 7 | -0.087 (0.013) | -0.099 | -0.089 |
| F200W | 17.0-17.5 | 42 | -0.028 (0.008) | -0.069 | -0.035 |
| F200W | 17.5-18.0 | 38 | -0.052 (0.010) | -0.081 | -0.057 |
| F200W | 18.0-18.5 | 16 | -0.011 (0.013) | -0.032 | -0.023 |
| F200W | 18.5-19.0 | 8 | -0.018 (0.023) | -0.041 | -0.025 |
| F200W | 19.0-19.5 | 5 | +0.007 (0.047) | -0.001 | +0.001 |
| F250M | 17.0-17.5 | 19 | +0.052 (0.010) | +0.046 | +0.052 |
| F250M | 17.5-18.0 | 26 | +0.041 (0.006) | +0.039 | +0.041 |
| F250M | 18.0-18.5 | 52 | +0.055 (0.010) | +0.046 | +0.055 |
| F250M | 18.5-19.0 | 76 | +0.029 (0.005) | +0.029 | +0.029 |
| F277W | 16.5-17.0 | 7 | +0.077 (0.020) | +0.033 | +0.077 |
| F277W | 17.0-17.5 | 6 | +0.028 (0.030) | +0.013 | +0.028 |
| F277W | 17.5-18.0 | 24 | +0.052 (0.018) | +0.026 | +0.052 |
| F277W | 18.0-18.5 | 34 | +0.074 (0.015) | +0.063 | +0.074 |
| F277W | 18.5-19.0 | 40 | +0.021 (0.008) | +0.021 | +0.021 |
| F277W | 19.0-19.5 | 26 | +0.019 (0.015) | +0.018 | +0.019 |
| F277W | 19.5-20.0 | 48 | +0.028 (0.011) | +0.028 | +0.028 |
| F277W | 20.0-20.5 | 7 | -0.004 (0.016) | -0.004 | -0.004 |
| F300M | 16.0-16.5 | 6 | +0.047 (0.020) | +0.021 | +0.047 |
| F300M | 17.0-17.5 | 8 | +0.051 (0.019) | +0.038 | +0.051 |
| F300M | 17.5-18.0 | 30 | +0.037 (0.007) | +0.033 | +0.037 |
| F300M | 18.0-18.5 | 22 | +0.038 (0.010) | +0.036 | +0.038 |
| F300M | 18.5-19.0 | 88 | +0.032 (0.006) | +0.031 | +0.032 |
| F300M | 19.0-19.5 | 21 | +0.011 (0.010) | +0.011 | +0.011 |

### Dependence on detector (S-row level dm, median [N]), sample `strict` or `relaxed` if strict is empty

- F150W: nrca1 -0.010 [27], nrca2 +0.030 [25], nrca3 +0.040 [35], nrca4 +0.012 [38], nrcb1 +0.001 [129], nrcb2 +0.004 [60], nrcb3 -0.043 [87], nrcb4 -0.005 [48]
  - isolated stars only: nrca1 -0.008 [24], nrca2 +0.030 [25], nrca3 +0.041 [32], nrca4 +0.012 [38], nrcb1 +0.002 [103], nrcb2 +0.006 [55], nrcb3 -0.037 [73], nrcb4 -0.005 [48]
- F162M: nrca1 +0.008 [8], nrca2 -0.012 [12], nrca3 +0.061 [11], nrca4 -0.000 [27], nrcb1 +0.015 [95], nrcb2 +0.026 [51], nrcb3 -0.037 [123], nrcb4 -0.000 [40]
  - isolated stars only: nrca1 +0.008 [8], nrca2 -0.017 [11], nrca3 +0.067 [10], nrca4 -0.000 [27], nrcb1 +0.015 [75], nrcb2 +0.025 [46], nrcb3 -0.046 [90], nrcb4 -0.007 [34]
- F182M: nrca1 -0.053 [8], nrca2 -0.002 [4], nrca3 +0.015 [15], nrca4 -0.031 [8], nrcb1 -0.004 [66], nrcb2 +0.017 [18], nrcb3 -0.034 [106], nrcb4 -0.034 [8]
  - isolated stars only: nrca1 -0.053 [8], nrca2 -0.002 [4], nrca3 +0.015 [15], nrca4 -0.048 [4], nrcb1 -0.004 [53], nrcb2 +0.018 [17], nrcb3 -0.036 [74], nrcb4 -0.034 [8]
- F200W: nrca1 +0.007 [9], nrca2 +0.026 [4], nrca3 -0.044 [12], nrca4 -0.025 [8], nrcb1 -0.025 [83], nrcb2 +0.025 [16], nrcb3 -0.055 [108], nrcb4 -0.035 [9]
  - isolated stars only: nrca1 +0.011 [6], nrca2 +0.026 [4], nrca3 -0.044 [12], nrca4 -0.025 [8], nrcb1 -0.025 [73], nrcb2 +0.022 [15], nrcb3 -0.059 [82], nrcb4 -0.035 [9]
- F250M: nrcalong +0.081 [55], nrcblong +0.032 [146]
  - isolated stars only: nrcalong +0.081 [55], nrcblong +0.029 [125]
- F277W: nrcalong +0.043 [68], nrcblong +0.000 [12]
  - isolated stars only: nrcalong +0.040 [62], nrcblong -0.004 [11]
- F300M: nrcalong +0.058 [70], nrcblong +0.015 [88]
  - isolated stars only: nrcalong +0.058 [63], nrcblong +0.019 [81]

### Dependence on sat_area and number of frames (median dm, final), sample `strict` or `relaxed`

- F150W sat_area bins: 0-12: +0.000 [223]
  - nS bins: 1-2: -0.012 [90]; 2-3: +0.011 [64]; 3-5: -0.002 [77]
- F162M sat_area bins: 0-12: +0.000 [181]
  - nS bins: 1-2: -0.006 [71]; 2-3: +0.006 [52]; 3-5: -0.009 [64]
- F182M sat_area bins: 12-20: -0.019 [71]; 20-40: -0.023 [30]; 40-80: -0.022 [18]
  - nS bins: 1-2: -0.011 [50]; 2-3: -0.029 [24]; 3-5: -0.021 [45]
- F200W sat_area bins: 12-20: -0.042 [48]; 20-40: -0.030 [37]; 40-80: -0.026 [52]
  - nS bins: 1-2: -0.026 [65]; 2-3: -0.025 [35]; 3-5: -0.047 [38]
- F250M sat_area bins: 0-12: +0.043 [101]
  - nS bins: 1-2: +0.045 [45]; 2-3: +0.039 [27]; 3-5: +0.047 [34]
- F277W sat_area bins: 0-12: +0.038 [36]
  - nS bins: 1-2: +0.022 [14]; 2-3: +0.051 [12]; 3-5: +0.046 [14]
- F300M sat_area bins: 0-12: +0.031 [80]
  - nS bins: 1-2: +0.043 [27]; 2-3: +0.029 [31]; 3-5: +0.030 [23]

### Pixel phase (r = distance of fitted position from the nearest pixel centre, pixels)

Overlap: rows are single D rows, dm_i = m_S(star median) - m_D,i. Control: D-only stars of the same magnitude range, m_i - median(other frames).

- F150W (sample strict): median r of S fits 0.28, of D rows 0.49
  - overlap D rows: r 0-0.2: -4.246 [3]; r 0.2-0.3: -0.035 [13]; r 0.3-0.4: -0.018 [63]; r 0.4-0.5: +0.004 [160]; r 0.5-0.6: +0.004 [152]; r 0.6-0.75: +0.004 [57]
  - control: r 0-0.2: -0.005 [77]; r 0.2-0.3: -0.001 [152]; r 0.3-0.4: -0.002 [270]; r 0.4-0.5: -0.001 [328]; r 0.5-0.6: +0.005 [170]; r 0.6-0.75: +0.005 [61]; mag range 19.4-20.0
  - control by detector: nrca1 +0.000 [40], nrca2 -0.001 [66], nrca3 -0.000 [19], nrca4 +0.000 [8], nrcb1 +0.000 [328], nrcb2 +0.003 [134], nrcb3 -0.001 [437], nrcb4 -0.000 [26]
- F162M (sample strict): median r of S fits 0.29, of D rows 0.49
  - overlap D rows: r 0-0.2: -0.028 [4]; r 0.2-0.3: -0.012 [14]; r 0.3-0.4: +0.006 [41]; r 0.4-0.5: +0.004 [129]; r 0.5-0.6: -0.004 [109]; r 0.6-0.75: +0.003 [50]
  - control: r 0-0.2: -0.003 [584]; r 0.2-0.3: -0.002 [835]; r 0.3-0.4: -0.001 [1319]; r 0.4-0.5: +0.001 [1631]; r 0.5-0.6: +0.002 [955]; r 0.6-0.75: +0.005 [279]; mag range 18.6-20.0
  - control by detector: nrca1 -0.000 [205], nrca2 +0.000 [248], nrca3 +0.001 [217], nrca4 +0.000 [290], nrcb1 +0.000 [1410], nrcb2 -0.000 [653], nrcb3 +0.000 [2051], nrcb4 -0.000 [529]
- F182M (sample relaxed): median r of S fits 0.45, of D rows 0.34
  - overlap D rows: r 0-0.2: -0.020 [59]; r 0.2-0.3: -0.015 [39]; r 0.3-0.4: -0.031 [35]; r 0.4-0.5: -0.006 [63]; r 0.5-0.6: -0.011 [27]; r 0.6-0.75: -0.022 [5]
  - control: r 0-0.2: +0.005 [317]; r 0.2-0.3: +0.000 [402]; r 0.3-0.4: -0.000 [601]; r 0.4-0.5: -0.002 [749]; r 0.5-0.6: -0.000 [406]; r 0.6-0.75: -0.002 [132]; mag range 17.1-18.4
  - control by detector: nrca1 +0.000 [72], nrca2 +0.001 [57], nrca3 +0.000 [75], nrca4 -0.000 [54], nrcb1 +0.000 [782], nrcb2 -0.001 [300], nrcb3 -0.000 [1111], nrcb4 -0.000 [156]
- F200W (sample relaxed): median r of S fits 0.42, of D rows 0.36
  - overlap D rows: r 0-0.2: -0.033 [54]; r 0.2-0.3: -0.012 [47]; r 0.3-0.4: -0.028 [69]; r 0.4-0.5: -0.030 [75]; r 0.5-0.6: -0.031 [29]; r 0.6-0.75: -0.024 [11]
  - control: r 0-0.2: +0.001 [641]; r 0.2-0.3: +0.001 [764]; r 0.3-0.4: -0.000 [1118]; r 0.4-0.5: -0.001 [1421]; r 0.5-0.6: -0.001 [846]; r 0.6-0.75: -0.000 [246]; mag range 16.7-18.9
  - control by detector: nrca1 -0.000 [148], nrca2 +0.001 [136], nrca3 +0.000 [136], nrca4 +0.000 [139], nrcb1 -0.000 [1353], nrcb2 -0.000 [571], nrcb3 -0.000 [2171], nrcb4 +0.000 [382]
- F250M (sample strict): median r of S fits 0.30, of D rows 0.49
  - overlap D rows: r 0-0.2: +0.044 [4]; r 0.2-0.3: +0.041 [3]; r 0.3-0.4: +0.048 [28]; r 0.4-0.5: +0.042 [69]; r 0.5-0.6: +0.050 [59]; r 0.6-0.75: +0.035 [23]
  - control: r 0-0.2: -0.001 [35]; r 0.2-0.3: -0.006 [57]; r 0.3-0.4: -0.003 [124]; r 0.4-0.5: +0.002 [152]; r 0.5-0.6: -0.002 [90]; r 0.6-0.75: +0.011 [25]; mag range 18.3-18.8
  - control by detector: nrcalong -0.002 [94], nrcblong -0.000 [389]
- F277W (sample strict): median r of S fits 0.35, of D rows 0.49
  - overlap D rows: r 0-0.2: +0.077 [2]; r 0.2-0.3: -0.002 [7]; r 0.3-0.4: +0.019 [8]; r 0.4-0.5: +0.028 [23]; r 0.5-0.6: +0.058 [21]; r 0.6-0.75: +0.043 [8]
  - control: r 0-0.2: +0.001 [25]; r 0.2-0.3: -0.002 [42]; r 0.3-0.4: -0.006 [68]; r 0.4-0.5: +0.001 [90]; r 0.5-0.6: -0.000 [66]; r 0.6-0.75: +0.022 [12]; mag range 19.5-20.1
  - control by detector: nrcalong -0.001 [243], nrcblong -0.002 [60]
- F300M (sample strict): median r of S fits 0.30, of D rows 0.49
  - overlap D rows: r 0-0.2: +0.018 [4]; r 0.2-0.3: +0.067 [5]; r 0.3-0.4: +0.028 [22]; r 0.4-0.5: +0.050 [51]; r 0.5-0.6: +0.022 [51]; r 0.6-0.75: +0.036 [13]
  - control: r 0-0.2: -0.007 [42]; r 0.2-0.3: -0.003 [63]; r 0.3-0.4: -0.002 [103]; r 0.4-0.5: +0.002 [134]; r 0.5-0.6: +0.002 [75]; r 0.6-0.75: +0.011 [22]; mag range 18.7-19.2
  - control by detector: nrcalong -0.001 [127], nrcblong -0.000 [312]

### Merge treatment of the overlap stars (sample `strict` or `relaxed`)

- F150W: 231 overlap stars, 231 matched in m8_dedup (0.1 arcsec), 228 with replaced_saturated; median(m_merged - m_S) = +0.007 for replaced; median(m_merged - m_D) = +0.005 replaced, +0.008 unreplaced [3]
- F162M: 187 overlap stars, 187 matched in m8_dedup (0.1 arcsec), 186 with replaced_saturated; median(m_merged - m_S) = +0.007 for replaced; median(m_merged - m_D) = +0.006 replaced, +0.012 unreplaced [1]
- F182M: 119 overlap stars, 119 matched in m8_dedup (0.1 arcsec), 118 with replaced_saturated; median(m_merged - m_S) = +0.007 for replaced; median(m_merged - m_D) = -0.013 replaced, +0.005 unreplaced [1]
- F200W: 138 overlap stars, 138 matched in m8_dedup (0.1 arcsec), 138 with replaced_saturated; median(m_merged - m_S) = +0.007 for replaced; median(m_merged - m_D) = -0.024 replaced, nan unreplaced [0]
- F250M: 106 overlap stars, 106 matched in m8_dedup (0.1 arcsec), 105 with replaced_saturated; median(m_merged - m_S) = +0.003 for replaced; median(m_merged - m_D) = +0.048 replaced, -0.014 unreplaced [1]
- F277W: 40 overlap stars, 40 matched in m8_dedup (0.1 arcsec), 40 with replaced_saturated; median(m_merged - m_S) = +0.003 for replaced; median(m_merged - m_D) = +0.041 replaced, nan unreplaced [0]
- F300M: 81 overlap stars, 81 matched in m8_dedup (0.1 arcsec), 81 with replaced_saturated; median(m_merged - m_S) = +0.003 for replaced; median(m_merged - m_D) = +0.034 replaced, nan unreplaced [0]

## Interpretation

- At the saturation threshold the SW satstar flux scale (final) matches daophot: F150W -0.002 +- 0.004, F162M -0.000 +- 0.005 (strict). The precap values read 0.021-0.025 bright
  (the cap lowers the flux by about 2% for these stars); the final flux sits on the daophot scale. The wing calibration does not act in F150W/F162M (no pooled table). The
  SW offsets against dolphot (-0.05 F150W, -0.07 F200W) therefore do not appear for the faintest saturated stars. The nbstep result (offset growing toward the bright end of the
  saturated range, -0.14 in the brightest F150W bin) agrees with this: the offset would arise at larger saturated areas.
- Brighter SW stars (relaxed sample) show -0.04 to -0.05 (F150W 17.5-19), -0.03 to -0.09 (F200W 16.5-18), -0.03 (F162M 16.5-18), -0.026 (F182M 17.5-18). The sign and size resemble the dolphot
  comparison. Whether the satstar rows read bright or the DQ-saturated daophot rows read faint cannot be separated with this sample (below).
- LW: the strict overlap stars read +0.03 to +0.05 faint, in agreement with the sign of the dolphot LW offset. F250M shows +0.095 at AB 18.0-18.5 and +0.029 at 18.5-19; F300M +0.049 and +0.011 for 18.5-19 and
  19-19.5; F277W +0.043 and -0.004 for 19.5-20 and 20-20.5. The trend toward brighter stars repeats in the three bands (N per bin 7-71) and agrees with an offset that increases with brightness.
  The relaxed sample extends this to AB 16-18 with +0.03 to +0.08 (F277W +0.077 at 16.5-17, +0.074 at 18-18.5).
- Detector dependence: nrcb3 reads 0.034-0.059 bright and nrca3 0.04-0.07 faint (F150W/F162M; F200W reads nrca3 -0.044) against the star's daophot flux; nrcalong reads +0.04-0.08 and nrcblong 0.00-0.03 in LW.
  The daophot-only control shows no detector offsets (|offset| <= 0.003). Satstar fluxes appear to carry a detector-dependent scale of about +-0.04 mag; the band medians combine detectors with
  different weights (nrcb3 and nrcb1 host most overlap stars in SW because they cover the cluster core). This pattern also suggests that the offset against dolphot varies across the field and
  that a global median hides it.
- Dependence on saturated area: overlap stars have `sat_area` in 9-12 px (SW strict) or similar for LW, a single bin. In the relaxed F182M/F200W samples `sat_area` bins 12-80 px show no trend
  (F182M -0.019/-0.023/-0.022; F200W -0.042/-0.030/-0.026). Number of satstar frames (nS) and of daophot frames (nD) show no trend beyond +-0.02.
- Pixel phase: satstar frames have median r 0.28-0.35 (peak near a pixel centre, as expected for saturation), daophot frames of the strict sample 0.49. dm does not depend on the daophot
  row phase within the errors (F150W: +0.004 for r > 0.4, -0.018 for r 0.3-0.4, -0.035 for r 0.2-0.3 with N = 63 and 13; LW rows at r 0.2-0.75 are all +0.02 to +0.07). The daophot-only control shows
  relative flux variation with phase of at most 0.005 mag in the SW and 0.011-0.022 (r > 0.6, N = 12-25) in the LW, with corner rows reading faint. This effect would lower the LW dm
  for corner-only D rows and cannot account for +0.03 to +0.05.

## Caveats

- Magnitude coverage: overlap stars exist only within about 1 mag of the threshold (strict) or within 2-3 mag brighter (relaxed). Stars with more than about 15 saturated pixels in all frames cannot be tested.
  The dolphot-based offsets, which apply to the whole satstar-replaced population, extend to brighter stars.
- F182M and F200W have no strict overlap stars. Every frame of a F182M/F200W star in which the satstar did not fit (D candidate rows 258 and 316) has `SATURATED` pixels in the 5x5 block (253/258 and 311/316). The relaxed
  sample for these bands consists of such rows. In the SW the DQ SATURATED flag marks pixels that saturate in any group, and the ramp-fit value of such a pixel uses the groups before saturation, so the
  daophot fit (`flags == 0`, hence no masked pixel) can be correct or clipped by an amount this test cannot determine. A clipped daophot row would make dm more negative.
- Selection: a star enters the satstar channel in the frames where its peak pixel reaches the threshold (centred peaks, median r 0.28-0.35); the daophot rows come from frames with the peak near a pixel
  corner (r 0.49) and a lower peak. A selection on the peak value correlates weakly with noise in the flux; the expected size is about 0.005 mag, small against the observed effects. Frame-to-frame
  scatter of single stars is large (MAD 0.04-0.06 mag), so per-star dm is noisy and the medians carry the information.
- Daophot rows near the saturation threshold come from pixels close to full well; the linearity correction of the calibration pipeline acts on them and no independent check of it exists here. A
  nonlinearity residual would make the daophot flux faint (positive dm = satstar fainter would be reduced, not enhanced), so it cannot explain the LW sign.
- Blending: the isolated subsample (no row brighter than 0.2 times the star within 0.5") changes the medians by at most 0.01 (F150W +0.001, F182M -0.019, F250M +0.044). The detector spread persists in the isolated subsample
  (nrcb3 -0.037, nrca3 +0.041 in F150W).
- Cluster matching uses a 0.1" radius against satstar position scatter of about 0.08" (stated in the merge code); stars with unmatched frames lose D rows, which lowers N but does not bias dm. Stars whose
  satstar frames split into two clusters (positional scatter > 0.1") count with a subset of the S rows.
- Single-epoch sample: the program has 4 dithers per detector and visit; no repeat observations test long-term stability. Only m7 (the main2 final phase) was examined; the merged catalog used the `m8_dedup` file.
- The per-detector offset has not been traced to a cause (PSF model per detector, local background, wing calibration, or the density of the field on that detector). The field-dependence of the SW
  offset against dolphot could be tested directly with the existing dolphot comparison split by detector.
