# capcore: recovered-core cap and core-fit amplitude vs the wing fit (wd2 main2, commit 5434f5e7)

Question: is the ZEROFRAME-recovered core a better flux estimator than the wing fit?  All numbers below are measured offline on the 24 main2 frames
(F150W nrcb1/nrcb3, F200W nrcb1/nrcb3, F250M/F300M nrcblong; exposures 1-4) and scored against dolphot (wd2_nircam_wf_mf_nf matches in `matched_Q_main2.fits`).
No existing file was modified; everything lives in `satrefit/capcore/`.

Files: `capcore_run.py` (per-row cap, core fits, peak-pixel diagnostics -> `out/<band>_<frame>_capcore.fits`), `capcore_score.py` (scoring, tables, png, row tables `out/rows_<band>.fits`),
`capcore_lwdiag.py` (first-frame diagnosis -> `capcore_lwdiag.md`), `capcore_tables.md`, `capcore_diag.md`, `capcore.png`.

## 1. Method

Frame reconstruction: `satrefit_core.prep_frame` (ZEROFRAME anchor from `*_ramp.fits`, `zf_deep`, rim, ERR with rim errors) and `make_setup` (cutout, weights, local background, bkg scatter), with the sequential brighter-first neighbour subtraction of `run_frame` (models at `flux_fit_raw`).  Each row's PSF is evaluated once at the catalog (x_fit, y_fit) from the same grid (`load_grid`).

(a) Cap, exactly as `get_saturated_stars`, for every row (not only where it binds):
- `cutout` = neighbour-subtracted `data_working` window (NaN -> 0, background not subtracted), `_unrecoverable = isnan(VAR_POISSON)` of the crf.
- `_cap_region = recovered_cap_region(label component, own deep core (zf_deep & component), own deep core dilated by compute_adaptive_mask_buffer(sat_area, 2), own_cell)`; `own_cell` is built from sibling seeds of the same label in the catalog plus the `_rejected` table (no deblended siblings occur in these frames: every row has a unique label).
- `recovered_core_peak(cutout, region, unrecoverable, 0.2)` gives (peak, lost fraction, n_rec); when the peak is finite, `recovered_cap_flux(..., min_psf_frac=0.005)` gives the cap and psf_frac.  The pipeline functions are imported read-only from the wd2main2 copy.

Cap reproduction check (rows with `flux_fit_raw < 0.999 flux_fit_precap`): the reproduced cap is finite for every such row (3666/3666 F150W, 1537/1537 F200W, 1546/1546 F250M, 1640/1640 F300M) and equals `flux_fit_raw` to a maximum relative difference of 5.7e-8 to 5.9e-8 (float32 catalog precision).  No uncapped row has a reproduced cap below 0.999 x precap.  The cap is therefore reproduced exactly.  The cap is finite on 5554/6169 (F150W), 2023/2515, 3364/3652, 3552/3862 rows; it is undefined where the lost fraction is >= 0.2, fewer than 3 recovered pixels remain, or psf_frac < 0.005.

(b) Core fit: weighted linear least squares at the catalog (x_fit, y_fit) using measured/recovered pixels with r <= 2.5 px (SW) / 2.0 px (LW).  A pixel is used when it is not `unrecoverable`, not in the deep core mask (`zf_deep`, or the SATURATED mask when no ZEROFRAME recovery exists), the cutout value is non-zero and finite, and ERR is finite.  ZEROFRAME-rewritten rim pixels are included (with the rim errors).  Weights are 1/(ERR^2 + sigma_bkg^2), the same as `satrefit_core.solve`.  Two variants:
- `core_c`: amplitude only, with the local background subtracted (`local_bkg`; the offline value equals the catalog `local_bkg` to 4e-8);
- `core_f`: two parameters (amplitude, constant).
A core estimator is "defined" when at least 5 pixels are used and the amplitude is positive.  `core_c(cov>=0.5)` additionally requires that the used pixels hold at least half of the PSF flux within the radius.

Scoring follows `score2.py`: dm = ours - dolphot - ZP (`analyze.Arm('main2')`, ZP from unreplaced unsaturated stars), satstar-replaced matched stars, rows matched to stars within 0.1".  An estimator's per-star dm is the final-catalog dm plus the per-star median over its rows of -2.5 log10(a_est / flux_fit_raw), so every estimator carries the pipeline's wingcal factor like `uncapped` in score2 (wingcal_ratio is 1 for all rows in F150W, F250M, F300M; in F200W the median is 1.004, 99th percentile 1.06).  Estimators: `final` (catalog), `precap` (flux_fit_precap), `cap` (cap used as the flux, where defined), `core_c`, `core_f` (where defined), `X|precap` (X where defined, else precap), `cap|precap`.  Stars in a table row are those with at least one defined row for that estimator, so N varies by estimator.  Trend = slope of dm vs dolphot mag (least squares, |dm - median| < 0.5) and "faint-bright" = median of the faintest minus the brightest 1-mag bin with N >= 10.  The sample contains no matched satstar rows brighter than 12 mag in F250M/F300M (the 10-13 mag bin of score2 is entirely 12-13 mag) and no stars brighter than 14 mag in F150W.

## 2. Summary of results (all-star median dm; slope in mag per mag of dolphot magnitude)

| band | final | precap | cap | core_c | core_f | core_c or precap | core_f or precap |
|---|---|---|---|---|---|---|---|
| F150W | -0.047, +0.020 | -0.078, +0.032 | -0.056, +0.018 | -0.048, +0.009 | -0.050, +0.011 | -0.048, +0.009 | -0.050, +0.011 |
| F200W | -0.066, -0.006 | -0.097, +0.010 | -0.072, -0.006 | -0.067, +0.007 | -0.067, +0.002 | -0.067, +0.007 | -0.067, +0.002 |
| F250M | +0.030, -0.007 | +0.016, +0.016 | +0.009, -0.020 | +0.049, -0.006 | +0.044, -0.019 | +0.049, -0.004 | +0.044, -0.016 |
| F300M | +0.018, -0.009 | +0.006, +0.014 | +0.002, -0.019 | +0.029, -0.006 | +0.017, -0.017 | +0.029, -0.003 | +0.017, -0.013 |

Availability (stars with at least one defined row / rows over all frames): cap 0.98-1.00 / 0.80-0.92; core_c 1.00 / 0.80-0.92 (F150W 0.92, F200W 0.80, F250M 0.91, F300M 0.91); core_f 1.00 / 0.80-0.92; core_c(cov>=0.5) 0.99-1.00 / 0.73-0.89.  Per bin the star availability of the core estimators is 1.00 in every bin except the brightest LW bin (0.96 for core_c(cov>=0.5) in F250M, 1.00 elsewhere) and cap 0.96-1.00 (F200W 16-17 mag: 0.81).  The row availability is lower than the star availability because deep-saturated frames of a star often lack 5 measured pixels or a valid cap.

Main points:
- SW (F150W, F200W): the core estimators do not remove the SW excess.  core_c has the same all-star median as the catalog (-0.048 vs -0.047 in F150W, -0.067 vs -0.066 in F200W) and the faint-bright difference of the 1-mag bins is +0.056 (core_c) vs +0.063 (final) in F150W and +0.046 vs +0.037 in F200W.  The core amplitude is as bright relative to dolphot as the wing fit: at 14-15 mag core_c reads -0.098 (final -0.095), and the 18-19 mag stars read -0.041 (final -0.032).  The wing-fit "bright SW" bias is thus present in the core amplitude as well.  precap is 0.03 mag brighter than final, and the cap (-0.056) lowers it towards the core values.
- LW (F250M, F300M): the recovered-core cap as flux (`cap`) reproduces the faint bright end (+0.127 / +0.116 at 12-13 mag) and carries a negative trend (-0.020 / -0.019 per mag); it reads 0.02 mag brighter than dolphot at 16-17 mag, where the cap lies above the fit (cap/precap 1.01-1.03 at 15-18 mag; the cap is a maximum over noisy, background-carrying pixels).  core_c is +0.03 to +0.05 mag faint at all magnitudes with slope -0.006 per mag (12-13 mag bin +0.082 / +0.053 against +0.059 / +0.032 at 13-14 mag, MAD 0.06-0.09 at 12-13).  core_f reproduces the cap-like negative trend (-0.019 / -0.017).  `precap` (the wing fit) has a +0.016 / +0.014 slope and reads -0.032 / -0.044 at 12-13 mag.  In LW the core amplitude has a smaller bright-end deficit (+0.05 to +0.08 vs +0.12) but a global offset of +0.03 to +0.05 mag that the wing fit does not have (precap +0.016 / +0.006).
- Neither estimator is better than the other across the board: the core amplitude flattens the LW trend and fixes part of the 12-13 mag deficit but sits 0.03-0.05 mag faint; in SW it matches the catalog median and has a similar trend.

## 3. LW cap diagnosis (F250M, F300M, 12-13 mag)

The cap binds at the model peak pixel in all rows of the brightest bins (`frac peak = model-peak px` = 1, median pk_psf_rel = 1), reading max(recovered pixel) / max(psf_unit).  Per-row numbers (`capcore_diag.md`, dolphot-implied flux = catalog flux x 10^(0.4 dm)):

| quantity (rows at 12-13 mag) | F250M | F300M |
|---|---|---|
| rows / cap defined | 104 / 98 | 72 / 72 |
| median lost fraction of the cap region (limit 0.2) | 0.010 | 0.016 |
| median recovered pixels in the region | 96 | 108 |
| cap / precap | 0.863 | 0.866 |
| cap / dolphot-implied flux | 0.893 | 0.902 |
| precap / dolphot-implied flux | 1.02 | 1.03 |
| recovered peak pixel / (dolphot-implied flux x PSF peak) | 0.866 | 0.875 |
| peak pixel value (MJy/sr); relative to L, the 99.999 pct of unsaturated data (262 / 154 MJy/sr) | 5.7e3; 21.6 L | 3.4e3; 22.2 L |
| peak pixel in the ZEROFRAME-rewritten rim | all rows | all rows |
| peak pixel's group 0 flagged saturated (value = k x first frame) | all rows | all rows |
| peak-pixel first-frame value (DN); frame max first frame | 3.9e4; 4.75e4 | 4.0e4; 4.76e4 |
| peak-pixel group 0 (DN) / 99th pct of group 0 at SATURATED px (4.65e4 DN) | 0.913 | 0.901 |

The loss fraction is 1-2 percent, so the 0.2 limit is not the cause; the peak pixel is always a rewritten rim pixel and its value is the scaled first frame.  `capcore_lwdiag.md` bins the rows whose peak pixel has a flagged group 0 by first-frame level (first frame / frame maximum):

| first frame / max | F250M peak/(f_dol x ppk) | F300M peak/(f_dol x ppk) |
|---|---|---|
| 0.1-0.5 (1e4-2.4e4 DN) | 0.98-0.99 | 0.99-1.00 |
| 0.5-0.7 | 0.96-0.99 | 0.97-0.98 |
| 0.7-0.8 | 0.942 | 0.967 |
| 0.8-0.9 (4.0e4 DN) | 0.902 | 0.907 |
| 0.9-1.0 (4.4e4 DN) | 0.905 | 0.912 |

The recovered peak is within 1-2 percent of the dolphot-implied value for first-frame readings up to about 2.4e4 DN, falls to 0.94-0.97 at 3.5e4 DN and to 0.90-0.91 above 4e4 DN (0.10 mag), which matches the 0.12-0.16 mag deficit of the 12-13 mag bin (the cap is a maximum over pixels and the star set differs slightly).  The value/first-frame ratio of these pixels is a constant (0.1445 MJy/sr per DN in F250M, 0.0852 in F300M; the header R is 0.0620 / 0.0360, so k x R_used / R_header = 2.33 / 2.37 with k about 2.4-2.5), so the recovered value is linear in the first frame by construction; the deficit therefore appears as a first-frame reading that is about 10 percent lower than linear above roughly 80 percent of the frame's first-frame maximum.  The source of the reading is the ramp `ZEROFRAME` extension (`_find_first_frame_for`), while `_find_zeroframe_for` uses the linearity-corrected `SCI[0,0]` for group 0 (the comment in the code states it is superbias/refpix/linearity corrected); the scaling `k(first frame)` is calibrated on first-frame values up to about 6e3 DN (F250M log: 8 bins to first frame ~6078) and held constant above.  This is consistent with an uncorrected non-linearity of the first read at high fill and with the extrapolated k, but the analysis does not separate the two, and the ZEROFRAME linearity treatment was not verified in the ramp products.  Rows whose peak pixel reads group 0 directly (not flagged, group 0 up to 0.6 of the ceiling) show 0.98-1.01 relative to dolphot, so the effect belongs to the first-frame path.  The core fit over r <= 2.0 px uses the same rewritten pixels but spreads the weight over about a dozen pixels, and it reads 0.95-0.98 of the dolphot-implied flux over the whole range (0.956 / 0.975 in the 0.8-1.0 first-frame bins), i.e. it is less sensitive to the saturated first-frame readings but is 3-5 percent low everywhere.

In SW the same table gives a flat 1.04-1.12 ratio at every first-frame level (F150W 1.07-1.12, F200W 1.04-1.09): the peak pixel is as bright as the wing fit relative to dolphot, and the first frame does not show the high-fill deficit.  In SW the first-frame maximum is 5.2e4 DN and about 10 percent of the rows lie above 0.8 of it.

## 4. Caveats
- The wing fit and the cap use the neighbour-subtracted frame with models at `flux_fit_raw`; neighbours rejected or deblended differently in the pipeline run are handled as in `satrefit_core`.
- The dolphot-implied flux uses the per-star dm of the final catalog and the ZP from unsaturated stars; it inherits the dolphot-vs-satstar flux scale (a +0.03 to +0.05 mag offset and trend in the unsaturated-satstar overlap is part of the reference).
- Per-star dm is a median over frames, and rows in frames where an estimator is undefined are skipped, so the star sets differ slightly between columns (N is given in each cell).
- Core estimators use a fixed PSF grid position; for stars with locked positions (sat_area >= 0.5 arcsec^2) the core fit is sensitive to a sub-pixel offset; this was not tested.
- The ZEROFRAME/first-frame explanation is a measured association (value/first-frame is constant, dolphot-relative deficit rises with first-frame level), not a test of linearity of the ramp reads.
- No stars brighter than 12 mag (LW) or 14 mag (SW) are in the matched replaced set, so the "10-13 mag" LW bin of earlier notes is the 12-13 mag bin here (19 / 28 stars).

## 5. Tables

#### F150W: 2311 satstar-replaced matched stars, 1471 with rows in the frames.  Cells: median dm (MAD) [N]; unsaturated reference 1 mag beyond the satstar faint edge: median dm +0.004

| estimator | 14-15 | 15-16 | 16-17 | 17-18 | 18-19 | all: med / MAD [N] | trend: slope dm per mag | faint-bright (bins N>=10) |
|---|---|---|---|---|---|---|---|---|
| final | -0.095 (0.070) [57] | -0.084 (0.048) [159] | -0.057 (0.045) [385] | -0.040 (0.044) [562] | -0.032 (0.047) [307] | -0.047 / 0.047 [1471] | +0.020 | +0.063 |
| precap | -0.142 (0.063) [57] | -0.122 (0.055) [159] | -0.101 (0.046) [385] | -0.066 (0.045) [562] | -0.047 (0.039) [307] | -0.078 / 0.053 [1471] | +0.032 | +0.095 |
| cap | -0.098 (0.072) [56] | -0.095 (0.054) [159] | -0.061 (0.051) [384] | -0.050 (0.048) [560] | -0.040 (0.054) [305] | -0.056 / 0.053 [1464] | +0.018 | +0.058 |
| core_c | -0.098 (0.058) [57] | -0.060 (0.042) [159] | -0.050 (0.045) [385] | -0.045 (0.043) [562] | -0.041 (0.040) [307] | -0.048 / 0.045 [1471] | +0.009 | +0.056 |
| core_f | -0.087 (0.079) [57] | -0.082 (0.054) [159] | -0.047 (0.051) [385] | -0.049 (0.046) [562] | -0.047 (0.055) [307] | -0.050 / 0.051 [1471] | +0.011 | +0.040 |
| core_c|precap | -0.098 (0.058) [57] | -0.060 (0.042) [159] | -0.050 (0.045) [385] | -0.045 (0.043) [562] | -0.041 (0.040) [307] | -0.048 / 0.045 [1471] | +0.009 | +0.056 |
| core_f|precap | -0.087 (0.079) [57] | -0.082 (0.054) [159] | -0.047 (0.051) [385] | -0.049 (0.046) [562] | -0.047 (0.055) [307] | -0.050 / 0.051 [1471] | +0.011 | +0.040 |
| core_c(cov>=0.5)|precap | -0.102 (0.061) [57] | -0.062 (0.042) [159] | -0.050 (0.045) [385] | -0.045 (0.043) [562] | -0.041 (0.040) [307] | -0.048 / 0.045 [1471] | +0.010 | +0.061 |
| cap|precap | -0.100 (0.071) [57] | -0.095 (0.053) [159] | -0.061 (0.051) [385] | -0.050 (0.049) [562] | -0.040 (0.054) [307] | -0.056 / 0.053 [1471] | +0.017 | +0.059 |

Availability (core estimator defined: >= 5 measured px within r <= 2.5 px and amplitude > 0; stars with >= 1 defined row / rows):

| estimator | 14-15 | 15-16 | 16-17 | 17-18 | 18-19 | all | rows (all frames) |
|---|---|---|---|---|---|---|---|
| cap | 0.98 [57] | 1.00 [159] | 1.00 [385] | 1.00 [562] | 0.99 [307] | 1.00 | 5554/6169 = 0.90 |
| core_c | 1.00 [57] | 1.00 [159] | 1.00 [385] | 1.00 [562] | 1.00 [307] | 1.00 | 5686/6169 = 0.92 |
| core_f | 1.00 [57] | 1.00 [159] | 1.00 [385] | 1.00 [562] | 1.00 [307] | 1.00 | 5678/6169 = 0.92 |
| core_c(cov>=0.5) | 0.98 [57] | 0.99 [159] | 1.00 [385] | 1.00 [562] | 1.00 [307] | 1.00 | 5422/6169 = 0.88 |

Cap reproduction (150W): 6169 rows with a label; 3666 rows have flux_fit_raw < 0.999 flux_fit_precap; for 3666 of them the reproduced cap is finite and its max |cap/flux_fit_raw - 1| is 5.88e-08 (99.9th pct 5.72e-08); 0 capped rows have a NaN reproduced cap; 0 uncapped rows have a reproduced cap below 0.999 precap; cap finite on 5554 rows; rows with wingcal_ratio != 1: 0.

#### F200W: 721 satstar-replaced matched stars, 515 with rows in the frames.  Cells: median dm (MAD) [N]; unsaturated reference 1 mag beyond the satstar faint edge: median dm -0.017

| estimator | 13-14 | 14-15 | 15-16 | 16-17 | 17-18 | all: med / MAD [N] | trend: slope dm per mag | faint-bright (bins N>=10) |
|---|---|---|---|---|---|---|---|---|
| final | +0.007 (0.032) [8] | -0.071 (0.052) [157] | -0.068 (0.053) [314] | -0.034 (0.046) [31] | -0.075 (0.014) [5] | -0.066 / 0.054 [515] | -0.006 | +0.037 |
| precap | -0.117 (0.073) [8] | -0.095 (0.053) [157] | -0.102 (0.048) [314] | -0.042 (0.038) [31] | -0.075 (0.014) [5] | -0.097 / 0.050 [515] | +0.010 | +0.053 |
| cap | +0.001 (0.062) [8] | -0.075 (0.060) [156] | -0.074 (0.058) [313] | -0.040 (0.052) [25] | -0.084 (0.006) [5] | -0.072 / 0.058 [507] | -0.006 | +0.036 |
| core_c | -0.041 (0.015) [8] | -0.074 (0.058) [157] | -0.068 (0.057) [314] | -0.028 (0.040) [31] | -0.076 (0.001) [5] | -0.067 / 0.058 [515] | +0.007 | +0.046 |
| core_f | -0.002 (0.082) [8] | -0.073 (0.060) [157] | -0.072 (0.059) [314] | -0.025 (0.041) [31] | -0.077 (0.012) [5] | -0.067 / 0.058 [515] | +0.002 | +0.049 |
| core_c|precap | -0.041 (0.015) [8] | -0.074 (0.058) [157] | -0.068 (0.057) [314] | -0.028 (0.040) [31] | -0.076 (0.001) [5] | -0.067 / 0.058 [515] | +0.007 | +0.046 |
| core_f|precap | -0.002 (0.082) [8] | -0.073 (0.060) [157] | -0.072 (0.059) [314] | -0.025 (0.041) [31] | -0.077 (0.012) [5] | -0.067 / 0.058 [515] | +0.002 | +0.049 |
| core_c(cov>=0.5)|precap | -0.082 (0.070) [8] | -0.074 (0.058) [157] | -0.068 (0.057) [314] | -0.028 (0.039) [31] | -0.076 (0.001) [5] | -0.068 / 0.057 [515] | +0.009 | +0.046 |
| cap|precap | +0.001 (0.062) [8] | -0.075 (0.060) [157] | -0.074 (0.057) [314] | -0.040 (0.053) [31] | -0.084 (0.006) [5] | -0.072 / 0.057 [515] | -0.007 | +0.036 |

Availability (core estimator defined: >= 5 measured px within r <= 2.5 px and amplitude > 0; stars with >= 1 defined row / rows):

| estimator | 13-14 | 14-15 | 15-16 | 16-17 | 17-18 | all | rows (all frames) |
|---|---|---|---|---|---|---|---|
| cap | 1.00 [8] | 0.99 [157] | 1.00 [314] | 0.81 [31] | 1.00 [5] | 0.98 | 2023/2515 = 0.80 |
| core_c | 1.00 [8] | 1.00 [157] | 1.00 [314] | 1.00 [31] | 1.00 [5] | 1.00 | 2010/2515 = 0.80 |
| core_f | 1.00 [8] | 1.00 [157] | 1.00 [314] | 1.00 [31] | 1.00 [5] | 1.00 | 2008/2515 = 0.80 |
| core_c(cov>=0.5) | 1.00 [8] | 1.00 [157] | 1.00 [314] | 0.97 [31] | 1.00 [5] | 1.00 | 1833/2515 = 0.73 |

Cap reproduction (200W): 2515 rows with a label; 1537 rows have flux_fit_raw < 0.999 flux_fit_precap; for 1537 of them the reproduced cap is finite and its max |cap/flux_fit_raw - 1| is 5.86e-08 (99.9th pct 5.78e-08); 0 capped rows have a NaN reproduced cap; 0 uncapped rows have a reproduced cap below 0.999 precap; cap finite on 2023 rows; rows with wingcal_ratio != 1: 2515.

#### F250M: 1129 satstar-replaced matched stars, 905 with rows in the frames.  Cells: median dm (MAD) [N]; unsaturated reference 1 mag beyond the satstar faint edge: median dm +0.003

| estimator | 12-13 | 13-14 | 14-15 | 15-16 | 16-17 | all: med / MAD [N] | trend: slope dm per mag | faint-bright (bins N>=10) |
|---|---|---|---|---|---|---|---|---|
| final | +0.125 (0.059) [28] | +0.039 (0.036) [80] | +0.021 (0.032) [189] | +0.028 (0.034) [374] | +0.031 (0.034) [233] | +0.030 / 0.035 [905] | -0.007 | -0.094 |
| precap | -0.032 (0.104) [28] | +0.001 (0.053) [80] | -0.001 (0.036) [189] | +0.020 (0.036) [374] | +0.029 (0.033) [233] | +0.016 / 0.038 [905] | +0.016 | +0.060 |
| cap | +0.127 (0.060) [27] | +0.029 (0.034) [79] | +0.014 (0.034) [188] | +0.006 (0.032) [372] | -0.009 (0.037) [232] | +0.009 / 0.038 [899] | -0.020 | -0.136 |
| core_c | +0.082 (0.064) [28] | +0.059 (0.033) [80] | +0.046 (0.022) [189] | +0.050 (0.025) [374] | +0.043 (0.032) [233] | +0.049 / 0.028 [905] | -0.006 | -0.039 |
| core_f | +0.190 (0.066) [28] | +0.061 (0.044) [80] | +0.052 (0.028) [189] | +0.039 (0.031) [374] | +0.032 (0.039) [233] | +0.044 / 0.037 [905] | -0.019 | -0.157 |
| core_c|precap | +0.075 (0.057) [28] | +0.059 (0.033) [80] | +0.046 (0.022) [189] | +0.050 (0.025) [374] | +0.043 (0.032) [233] | +0.049 / 0.028 [905] | -0.004 | -0.031 |
| core_f|precap | +0.161 (0.090) [28] | +0.061 (0.044) [80] | +0.052 (0.028) [189] | +0.039 (0.031) [374] | +0.032 (0.039) [233] | +0.044 / 0.037 [905] | -0.016 | -0.128 |
| core_c(cov>=0.5)|precap | +0.064 (0.067) [28] | +0.059 (0.033) [80] | +0.046 (0.022) [189] | +0.050 (0.026) [374] | +0.043 (0.031) [233] | +0.049 / 0.028 [905] | -0.003 | -0.021 |
| cap|precap | +0.125 (0.059) [28] | +0.029 (0.033) [80] | +0.014 (0.034) [189] | +0.007 (0.033) [374] | -0.009 (0.037) [233] | +0.009 / 0.038 [905] | -0.019 | -0.134 |

Availability (core estimator defined: >= 5 measured px within r <= 2.0 px and amplitude > 0; stars with >= 1 defined row / rows):

| estimator | 12-13 | 13-14 | 14-15 | 15-16 | 16-17 | all | rows (all frames) |
|---|---|---|---|---|---|---|---|
| cap | 0.96 [28] | 0.99 [80] | 0.99 [189] | 0.99 [374] | 1.00 [233] | 0.99 | 3364/3652 = 0.92 |
| core_c | 1.00 [28] | 1.00 [80] | 1.00 [189] | 1.00 [374] | 1.00 [233] | 1.00 | 3320/3652 = 0.91 |
| core_f | 1.00 [28] | 1.00 [80] | 1.00 [189] | 1.00 [374] | 1.00 [233] | 1.00 | 3298/3652 = 0.90 |
| core_c(cov>=0.5) | 0.96 [28] | 1.00 [80] | 1.00 [189] | 1.00 [374] | 1.00 [233] | 1.00 | 3273/3652 = 0.90 |

Cap reproduction (250M): 3652 rows with a label; 1546 rows have flux_fit_raw < 0.999 flux_fit_precap; for 1546 of them the reproduced cap is finite and its max |cap/flux_fit_raw - 1| is 5.75e-08 (99.9th pct 5.63e-08); 0 capped rows have a NaN reproduced cap; 0 uncapped rows have a reproduced cap below 0.999 precap; cap finite on 3364 rows; rows with wingcal_ratio != 1: 0.

#### F300M: 1224 satstar-replaced matched stars, 944 with rows in the frames.  Cells: median dm (MAD) [N]; unsaturated reference 1 mag beyond the satstar faint edge: median dm -0.003

| estimator | 12-13 | 13-14 | 14-15 | 15-16 | 16-17 | all: med / MAD [N] | trend: slope dm per mag | faint-bright (bins N>=10) |
|---|---|---|---|---|---|---|---|---|
| final | +0.116 (0.050) [19] | +0.027 (0.035) [97] | +0.014 (0.030) [232] | +0.019 (0.035) [399] | +0.015 (0.035) [196] | +0.018 / 0.035 [944] | -0.009 | -0.101 |
| precap | -0.044 (0.090) [19] | -0.007 (0.049) [97] | -0.003 (0.040) [232] | +0.011 (0.035) [399] | +0.014 (0.035) [196] | +0.006 / 0.039 [944] | +0.014 | +0.058 |
| cap | +0.116 (0.050) [19] | +0.022 (0.037) [97] | +0.007 (0.036) [229] | +0.001 (0.031) [397] | -0.022 (0.031) [192] | +0.002 / 0.035 [935] | -0.019 | -0.138 |
| core_c | +0.053 (0.086) [19] | +0.032 (0.033) [97] | +0.029 (0.023) [232] | +0.033 (0.031) [399] | +0.020 (0.035) [196] | +0.029 / 0.030 [944] | -0.006 | -0.033 |
| core_f | +0.172 (0.089) [19] | +0.027 (0.042) [97] | +0.023 (0.026) [232] | +0.014 (0.032) [399] | +0.006 (0.040) [196] | +0.017 / 0.033 [944] | -0.017 | -0.166 |
| core_c|precap | +0.025 (0.094) [19] | +0.032 (0.033) [97] | +0.029 (0.023) [232] | +0.033 (0.031) [399] | +0.020 (0.035) [196] | +0.029 / 0.030 [944] | -0.003 | -0.005 |
| core_f|precap | +0.125 (0.142) [19] | +0.027 (0.042) [97] | +0.023 (0.026) [232] | +0.014 (0.032) [399] | +0.006 (0.040) [196] | +0.017 / 0.033 [944] | -0.013 | -0.120 |
| core_c(cov>=0.5)|precap | +0.024 (0.091) [19] | +0.032 (0.033) [97] | +0.029 (0.023) [232] | +0.033 (0.031) [399] | +0.020 (0.035) [196] | +0.029 / 0.030 [944] | -0.001 | -0.004 |
| cap|precap | +0.116 (0.050) [19] | +0.022 (0.036) [97] | +0.007 (0.034) [232] | +0.002 (0.031) [399] | -0.020 (0.033) [196] | +0.002 / 0.035 [944] | -0.019 | -0.136 |

Availability (core estimator defined: >= 5 measured px within r <= 2.0 px and amplitude > 0; stars with >= 1 defined row / rows):

| estimator | 12-13 | 13-14 | 14-15 | 15-16 | 16-17 | all | rows (all frames) |
|---|---|---|---|---|---|---|---|
| cap | 1.00 [19] | 1.00 [97] | 0.99 [232] | 0.99 [399] | 0.98 [196] | 0.99 | 3552/3862 = 0.92 |
| core_c | 1.00 [19] | 1.00 [97] | 1.00 [232] | 1.00 [399] | 1.00 [196] | 1.00 | 3496/3862 = 0.91 |
| core_f | 1.00 [19] | 1.00 [97] | 1.00 [232] | 1.00 [399] | 1.00 [196] | 1.00 | 3473/3862 = 0.90 |
| core_c(cov>=0.5) | 1.00 [19] | 1.00 [97] | 1.00 [232] | 1.00 [399] | 1.00 [196] | 1.00 | 3438/3862 = 0.89 |

Cap reproduction (300M): 3862 rows with a label; 1640 rows have flux_fit_raw < 0.999 flux_fit_precap; for 1640 of them the reproduced cap is finite and its max |cap/flux_fit_raw - 1| is 5.91e-08 (99.9th pct 5.83e-08); 0 capped rows have a NaN reproduced cap; 0 uncapped rows have a reproduced cap below 0.999 precap; cap finite on 3552 rows; rows with wingcal_ratio != 1: 0.


## 6. Per-row diagnostics

#### F150W per-row diagnostics by dolphot mag bin (rows matched to a replaced star)

| bin | rows | cap defined | median lost frac (cap-def rows) | cap/precap | cap/f_dolphot | precap/f_dolphot | core_c/f_dolphot | core_f/f_dolphot | peak px / (f_dol*ppk) | peak val MJy/sr | peak val / L | peak g0 DN | g0 / g0sat99 | peak first-frame DN | frac peak in rim | frac peak g0-sat | frac peak = model-peak px | median pk_psf_rel | median n_rec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 13-14.5 | 4 | 3 (0.75) | 0 | 0.873 | 1.06 | 1.15 | 1.13 | 1.09 | 1.06 | 1.09e+04 | 32.7 | 5.1e+04 | 0.968 | 4.97e+04 | 1 | 1 | 0 | 0.989 | 97 |
| 14.5-15 | 220 | 178 (0.81) | 0 | 0.956 | 1.09 | 1.13 | 1.09 | 1.08 | 1.07 | 9.8e+03 | 28.6 | 5.02e+04 | 0.945 | 4.32e+04 | 1 | 1 | 1 | 1 | 71 |
| 15-16 | 627 | 623 (0.99) | 0 | 0.97 | 1.09 | 1.12 | 1.06 | 1.08 | 1.09 | 5.07e+03 | 14.7 | 4.49e+04 | 0.843 | 2.23e+04 | 1 | 1 | 1 | 1 | 50 |
| 16-17 | 1466 | 1460 (1.00) | 0 | 0.968 | 1.06 | 1.1 | 1.05 | 1.05 | 1.06 | 2.04e+03 | 5.94 | 2.28e+04 | 0.43 | 9.12e+03 | 1 | 0 | 1 | 1 | 28 |
| 17-19 | 3100 | 3076 (0.99) | 0 | 0.989 | 1.04 | 1.06 | 1.04 | 1.05 | 1.04 | 688 | 1.99 | 7.65e+03 | 0.144 | 3.01e+03 | 1 | 0 | 1 | 1 | 16 |

Frame constants (median over frames): L (99.999 pct of finite unsaturated non-rim data, MJy/sr) 353.4; 0.99-pct of ZEROFRAME/g0 at SATURATED px (DN) 53491; max g0 54907; max first-frame 52055; PHOTMJSR 2.351; header R 0.0876

#### F200W per-row diagnostics by dolphot mag bin (rows matched to a replaced star)

| bin | rows | cap defined | median lost frac (cap-def rows) | cap/precap | cap/f_dolphot | precap/f_dolphot | core_c/f_dolphot | core_f/f_dolphot | peak px / (f_dol*ppk) | peak val MJy/sr | peak val / L | peak g0 DN | g0 / g0sat99 | peak first-frame DN | frac peak in rim | frac peak g0-sat | frac peak = model-peak px | median pk_psf_rel | median n_rec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 13-14.5 | 317 | 314 (0.99) | 0 | 0.958 | 1.07 | 1.1 | 1.08 | 1.06 | 1.04 | 7.52e+03 | 26.1 | 5.04e+04 | 0.943 | 3.93e+04 | 1 | 1 | 1 | 1 | 84 |
| 14.5-15 | 328 | 328 (1.00) | 0 | 0.978 | 1.08 | 1.1 | 1.07 | 1.07 | 1.08 | 4.86e+03 | 17.1 | 4.76e+04 | 0.89 | 2.58e+04 | 1 | 1 | 1 | 1 | 61 |
| 15-16 | 1130 | 1119 (0.99) | 0 | 0.972 | 1.07 | 1.11 | 1.07 | 1.07 | 1.07 | 2.53e+03 | 8.85 | 3.25e+04 | 0.609 | 1.34e+04 | 1 | 1 | 1 | 1 | 24 |
| 16-17 | 39 | 31 (0.79) | 0.0222 | 0.996 | 1.04 | 1.05 | 1.03 | 1.03 | 1.04 | 1.12e+03 | 3.94 | 1.51e+04 | 0.283 | 6.15e+03 | 1 | 0 | 1 | 1 | 44 |
| 17-19 | 5 | 5 (1.00) | 0.0444 | 1.01 | 1.08 | 1.07 | 1.08 | 1.08 | 1.08 | 434 | 1.49 | 5.73e+03 | 0.107 | 2.31e+03 | 1 | 0 | 1 | 1 | 43 |

Frame constants (median over frames): L (99.999 pct of finite unsaturated non-rim data, MJy/sr) 290.6; 0.99-pct of ZEROFRAME/g0 at SATURATED px (DN) 53603; max g0 55019; max first-frame 53102; PHOTMJSR 1.944; header R 0.0724

#### F250M per-row diagnostics by dolphot mag bin (rows matched to a replaced star)

| bin | rows | cap defined | median lost frac (cap-def rows) | cap/precap | cap/f_dolphot | precap/f_dolphot | core_c/f_dolphot | core_f/f_dolphot | peak px / (f_dol*ppk) | peak val MJy/sr | peak val / L | peak g0 DN | g0 / g0sat99 | peak first-frame DN | frac peak in rim | frac peak g0-sat | frac peak = model-peak px | median pk_psf_rel | median n_rec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 12-13 | 104 | 98 (0.94) | 0.0104 | 0.863 | 0.893 | 1.02 | 0.931 | 0.843 | 0.866 | 5.7e+03 | 21.6 | 4.25e+04 | 0.913 | 3.93e+04 | 1 | 1 | 1 | 1 | 96.5 |
| 13-14 | 308 | 301 (0.98) | 0 | 0.978 | 0.974 | 1 | 0.946 | 0.947 | 0.974 | 3.45e+03 | 13 | 4.05e+04 | 0.87 | 2.38e+04 | 1 | 1 | 1 | 1 | 71 |
| 14-15 | 723 | 705 (0.98) | 0 | 0.984 | 0.988 | 1 | 0.956 | 0.953 | 0.988 | 1.42e+03 | 5.39 | 2.32e+04 | 0.499 | 9.83e+03 | 1 | 1 | 1 | 1 | 45 |
| 15-16 | 1353 | 1313 (0.97) | 0 | 1.01 | 0.993 | 0.981 | 0.953 | 0.966 | 0.992 | 606 | 2.29 | 1.03e+04 | 0.222 | 4.11e+03 | 1 | 0 | 1 | 1 | 20 |
| 16-18 | 700 | 688 (0.98) | 0 | 1.03 | 1.01 | 0.974 | 0.959 | 0.969 | 1 | 314 | 1.19 | 5.35e+03 | 0.115 | 2.12e+03 | 1 | 0 | 1 | 1 | 9 |

Frame constants (median over frames): L (99.999 pct of finite unsaturated non-rim data, MJy/sr) 262.2; 0.99-pct of ZEROFRAME/g0 at SATURATED px (DN) 46511; max g0 73205; max first-frame 47495; PHOTMJSR 1.665; header R 0.0620

#### F300M per-row diagnostics by dolphot mag bin (rows matched to a replaced star)

| bin | rows | cap defined | median lost frac (cap-def rows) | cap/precap | cap/f_dolphot | precap/f_dolphot | core_c/f_dolphot | core_f/f_dolphot | peak px / (f_dol*ppk) | peak val MJy/sr | peak val / L | peak g0 DN | g0 / g0sat99 | peak first-frame DN | frac peak in rim | frac peak g0-sat | frac peak = model-peak px | median pk_psf_rel | median n_rec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 12-13 | 72 | 72 (1.00) | 0.0156 | 0.866 | 0.902 | 1.03 | 0.955 | 0.859 | 0.875 | 3.41e+03 | 22.2 | 4.2e+04 | 0.901 | 4.01e+04 | 1 | 1 | 1 | 1 | 108 |
| 13-14 | 362 | 352 (0.97) | 0 | 0.975 | 0.98 | 1.01 | 0.97 | 0.976 | 0.979 | 1.93e+03 | 12.6 | 4.06e+04 | 0.872 | 2.26e+04 | 1 | 1 | 1 | 1 | 82 |
| 14-15 | 876 | 854 (0.97) | 0 | 0.987 | 0.995 | 1 | 0.971 | 0.98 | 0.993 | 884 | 5.75 | 2.44e+04 | 0.524 | 1.04e+04 | 1 | 1 | 1 | 1 | 55 |
| 15-16 | 1443 | 1403 (0.97) | 0 | 1.01 | 0.999 | 0.99 | 0.969 | 0.987 | 0.998 | 370 | 2.41 | 1.07e+04 | 0.23 | 4.3e+03 | 1 | 0 | 1 | 1 | 23 |
| 16-18 | 604 | 590 (0.98) | 0 | 1.03 | 1.02 | 0.987 | 0.979 | 0.995 | 1.02 | 189 | 1.23 | 5.47e+03 | 0.118 | 2.16e+03 | 1 | 0 | 1 | 1 | 12 |

Frame constants (median over frames): L (99.999 pct of finite unsaturated non-rim data, MJy/sr) 153.7; 0.99-pct of ZEROFRAME/g0 at SATURATED px (DN) 46582; max g0 73204; max first-frame 47607; PHOTMJSR 0.965; header R 0.0360


## 7. First-frame level diagnosis

#### F250M: recovered peak pixel / (dolphot-implied flux x PSF peak), rows whose peak pixel has a flagged group 0 (scaled first-frame value), lost < 0.2; N=829

| first frame / frame max | N | peak/(f_dol*ppk) median (MAD) | cap/f_dol | core_c/f_dol | precap/f_dol | peak value / f_dol-implied peak vs value/first-frame (MJy/sr per DN) | median ff DN |
|---|---|---|---|---|---|---|---|
| 0.1-0.2 | 75 | 0.983 (0.034) | 0.984 | 0.968 | 1.008 | 0.1445 | 9071 |
| 0.2-0.3 | 305 | 0.992 (0.030) | 0.992 | 0.957 | 1.010 | 0.1445 | 11390 |
| 0.3-0.4 | 141 | 0.988 (0.033) | 0.988 | 0.952 | 1.009 | 0.1445 | 16302 |
| 0.4-0.5 | 88 | 0.978 (0.035) | 0.978 | 0.937 | 0.987 | 0.1445 | 21026 |
| 0.5-0.6 | 62 | 0.986 (0.030) | 0.986 | 0.936 | 0.989 | 0.1445 | 25527 |
| 0.6-0.7 | 42 | 0.960 (0.033) | 0.961 | 0.940 | 0.997 | 0.1445 | 30349 |
| 0.7-0.8 | 38 | 0.942 (0.051) | 0.947 | 0.975 | 1.024 | 0.1445 | 35415 |
| 0.8-0.9 | 52 | 0.902 (0.054) | 0.902 | 0.956 | 1.033 | 0.1445 | 40156 |
| 0.9-1.0 | 23 | 0.905 (0.044) | 0.905 | 0.952 | 1.043 | 0.1445 | 44263 |

median pk_val/pk_ff (MJy/sr per first-frame DN) overall 0.1445; header R 0.0620; implied k*R_used: header R x (4+1)/... see text. Frame max first frame 47495 DN.

Rows whose peak pixel reads group 0 directly (not flagged): N=2276; by g0 / (99th pct of g0 at SATURATED px):

| g0 / g0sat99 | N | peak/(f_dol*ppk) | median g0 DN |
|---|---|---|---|
| 0.0-0.2 | 1184 | 0.999 | 6335 |
| 0.2-0.4 | 964 | 0.989 | 12643 |
| 0.4-0.6 | 128 | 0.983 | 19780 |

#### F300M: recovered peak pixel / (dolphot-implied flux x PSF peak), rows whose peak pixel has a flagged group 0 (scaled first-frame value), lost < 0.2; N=1037

| first frame / frame max | N | peak/(f_dol*ppk) median (MAD) | cap/f_dol | core_c/f_dol | precap/f_dol | peak value / f_dol-implied peak vs value/first-frame (MJy/sr per DN) | median ff DN |
|---|---|---|---|---|---|---|---|
| 0.0-0.1 | 9 | 0.110 (0.064) | 0.933 | 0.978 | 1.014 | 0.0893 | 2857 |
| 0.1-0.2 | 112 | 0.991 (0.027) | 0.993 | 0.977 | 1.011 | 0.0852 | 9116 |
| 0.2-0.3 | 396 | 1.004 (0.026) | 1.004 | 0.974 | 1.008 | 0.0852 | 11428 |
| 0.3-0.4 | 195 | 0.994 (0.026) | 0.994 | 0.969 | 1.010 | 0.0852 | 16290 |
| 0.4-0.5 | 135 | 0.992 (0.027) | 0.992 | 0.972 | 1.003 | 0.0852 | 21478 |
| 0.5-0.6 | 56 | 0.976 (0.022) | 0.976 | 0.954 | 0.989 | 0.0852 | 25914 |
| 0.6-0.7 | 38 | 0.970 (0.028) | 0.970 | 0.963 | 0.993 | 0.0852 | 30912 |
| 0.7-0.8 | 36 | 0.967 (0.040) | 0.967 | 0.972 | 1.033 | 0.0852 | 35700 |
| 0.8-0.9 | 50 | 0.907 (0.054) | 0.907 | 0.971 | 1.041 | 0.0852 | 40651 |
| 0.9-1.0 | 10 | 0.912 (0.022) | 0.912 | 0.975 | 1.088 | 0.0852 | 43674 |

median pk_val/pk_ff (MJy/sr per first-frame DN) overall 0.0852; header R 0.0360; implied k*R_used: header R x (4+1)/... see text. Frame max first frame 47607 DN.

Rows whose peak pixel reads group 0 directly (not flagged): N=2234; by g0 / (99th pct of g0 at SATURATED px):

| g0 / g0sat99 | N | peak/(f_dol*ppk) | median g0 DN |
|---|---|---|---|
| 0.0-0.2 | 1103 | 1.010 | 6514 |
| 0.2-0.4 | 1000 | 0.995 | 13041 |
| 0.4-0.6 | 131 | 0.975 | 19644 |

#### F150W: recovered peak pixel / (dolphot-implied flux x PSF peak), rows whose peak pixel has a flagged group 0 (scaled first-frame value), lost < 0.2; N=1195

| first frame / frame max | N | peak/(f_dol*ppk) median (MAD) | cap/f_dol | core_c/f_dol | precap/f_dol | peak value / f_dol-implied peak vs value/first-frame (MJy/sr per DN) | median ff DN |
|---|---|---|---|---|---|---|---|
| 0.1-0.2 | 7 | 0.359 (0.367) | 1.063 | 1.102 | 1.135 | 0.2341 | 7469 |
| 0.2-0.3 | 413 | 1.085 (0.054) | 1.085 | 1.059 | 1.126 | 0.2337 | 12934 |
| 0.3-0.4 | 250 | 1.099 (0.060) | 1.099 | 1.062 | 1.122 | 0.2340 | 17918 |
| 0.4-0.5 | 151 | 1.089 (0.056) | 1.089 | 1.054 | 1.118 | 0.2337 | 23515 |
| 0.5-0.6 | 109 | 1.085 (0.056) | 1.085 | 1.064 | 1.132 | 0.2340 | 28387 |
| 0.6-0.7 | 81 | 1.079 (0.052) | 1.079 | 1.047 | 1.108 | 0.2339 | 34040 |
| 0.7-0.8 | 67 | 1.066 (0.062) | 1.066 | 1.066 | 1.116 | 0.2339 | 39141 |
| 0.8-0.9 | 62 | 1.074 (0.053) | 1.074 | 1.063 | 1.116 | 0.2339 | 43867 |
| 0.9-1.0 | 54 | 1.116 (0.045) | 1.116 | 1.108 | 1.134 | 0.2340 | 49748 |

median pk_val/pk_ff (MJy/sr per first-frame DN) overall 0.2339; header R 0.0876; implied k*R_used: header R x (4+1)/... see text. Frame max first frame 52055 DN.

Rows whose peak pixel reads group 0 directly (not flagged): N=4145; by g0 / (99th pct of g0 at SATURATED px):

| g0 / g0sat99 | N | peak/(f_dol*ppk) | median g0 DN |
|---|---|---|---|
| 0.0-0.2 | 2370 | 1.039 | 6590 |
| 0.2-0.4 | 1325 | 1.054 | 14640 |
| 0.4-0.6 | 450 | 1.054 | 24515 |

#### F200W: recovered peak pixel / (dolphot-implied flux x PSF peak), rows whose peak pixel has a flagged group 0 (scaled first-frame value), lost < 0.2; N=1498

| first frame / frame max | N | peak/(f_dol*ppk) median (MAD) | cap/f_dol | core_c/f_dol | precap/f_dol | peak value / f_dol-implied peak vs value/first-frame (MJy/sr per DN) | median ff DN |
|---|---|---|---|---|---|---|---|
| 0.1-0.2 | 7 | 0.194 (0.075) | 1.014 | 1.162 | 1.284 | 0.1928 | 7560 |
| 0.2-0.3 | 548 | 1.073 (0.059) | 1.077 | 1.069 | 1.108 | 0.1871 | 13139 |
| 0.3-0.4 | 319 | 1.094 (0.053) | 1.095 | 1.082 | 1.121 | 0.1934 | 17962 |
| 0.4-0.5 | 174 | 1.084 (0.063) | 1.084 | 1.067 | 1.104 | 0.1931 | 23310 |
| 0.5-0.6 | 124 | 1.081 (0.050) | 1.081 | 1.070 | 1.089 | 0.1934 | 29312 |
| 0.6-0.7 | 119 | 1.041 (0.065) | 1.041 | 1.065 | 1.089 | 0.1871 | 34028 |
| 0.7-0.8 | 85 | 1.044 (0.057) | 1.044 | 1.076 | 1.091 | 0.1934 | 39110 |
| 0.8-0.9 | 55 | 1.082 (0.013) | 1.082 | 1.095 | 1.111 | 0.1935 | 44998 |
| 0.9-1.0 | 65 | 1.079 (0.033) | 1.079 | 1.085 | 1.107 | 0.1934 | 50429 |

median pk_val/pk_ff (MJy/sr per first-frame DN) overall 0.1933; header R 0.0724; implied k*R_used: header R x (4+1)/... see text. Frame max first frame 53102 DN.

Rows whose peak pixel reads group 0 directly (not flagged): N=299; by g0 / (99th pct of g0 at SATURATED px):

| g0 / g0sat99 | N | peak/(f_dol*ppk) | median g0 DN |
|---|---|---|---|
| 0.0-0.2 | 11 | 1.032 | 7416 |
| 0.2-0.4 | 30 | 1.042 | 17972 |
| 0.4-0.6 | 258 | 1.060 | 25708 |

