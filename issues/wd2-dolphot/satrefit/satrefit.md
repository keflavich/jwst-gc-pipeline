# satrefit: masked-core PSF refit diagnostic (wd2 main2, commit 5434f5e7)

Read-only offline reproduction of `get_saturated_stars` (saturated_star_finding.py, wd2main2 copy) with fit variants.
Code: `satrefit_core.py` (fit + variants), `run_frames.py` (variant list, per-frame driver), `score.py` (dolphot scoring, follows capoff_tol.py), `plot.py`, `test1.py`/`count_det.py` (validation, counts).
Outputs: `out/<band>_<frame>_satrefit.fits` (amplitudes per catalog row and variant), `out/tables_<band>.md`, `satrefit.png`.
Frames: F150W nrcb1 + nrcb3 (most replaced matched stars), exposures 1-4 of 10101; F250M nrcblong, exposures 1-4 of 04101. F200W, F300M and F150W nrcb2 were not run.

## 1. What the fit does (measured from the code)

- Data: the crf SCI image after (i) NaN VAR_POISSON set to 0, (ii) ZEROFRAME fit anchor (`zeroframe_fit_anchor`) rewriting rim pixels from the group-0 ramp read, (iii) `zf_deep_core` replacing the SATURATED mask. The image is not background-subtracted; the local background is fitted per star.
- Per star: 162 px cutout, PSF fit_shape 81x81 (photutils PSFPhotometry, LevMarLSQFitter), weights 1/err. Errors come from the crf ERR plus `zeroframe_rim_error` for recovered pixels, with `bkg_scatter_fit_error` adding sigma (1.4826 MAD of the annulus) in quadrature.
- Mask: DQ SATURATED pixels plus an adaptive buffer (`compute_adaptive_mask_buffer`); pixels of previously fitted neighbours are handled by sequential brighter-first model subtraction in `data_working`.
- Background: `LocalBackground` sigma-clipped median in an adaptive annulus (`compute_adaptive_bkg_annulus`), subtracted before the fit and held fixed.
- Position: x,y fitted unless sat_area >= 0.5 arcsec^2, where they lock to the initial position.
- PSF: GriddedPSFModel (SW fovp512, LW fovp1024, tree_main2/psfs) wrapped by `psf_in_cutout_coords`.
- Flux chain: `flux_fit_precap` (fit amplitude) -> cap -> `flux_fit_raw` -> wingcal. "uncapped" below is precap plus wingcal; "final" is the capped catalog value plus wingcal. About 71 percent of F150W nrcb1 rows are capped.
- With position fixed or free the amplitude is linear in the PSF at the fitted position, so each variant is solved as a weighted linear least squares at the pipeline's fitted (x,y).

## 2. Validation (measured)

Baseline refit amplitude over catalog flux_fit_precap, all 12 frames: median 1.00000001 to 1.00000002, 16-84 percentile 0.9999992 to 1.0000021, at most 0.12 percent of rows differ by more than 1e-3 per frame (703-928 rows each). Summed per-star models match the pipeline model image to 1e-8. The reproduction is therefore exact to numerical precision for SW and LW, and the variant differences below reflect only the changed fit choice.

ZEROFRAME recovery (measured): the in-memory fit used ZF-recovered pixels. The log reports, for F150W nrcb1 exp 2, "recovered 27192 rim/core pixels from group-0 (R=0.08748); 2821 deep-core pixels remain masked"; other frames: nrcb3 69913 recovered / 5394 masked, F250M 86536 / 5386. The recovery is reproduced from the on-disk ramp files, which is why the amplitude reproduces. `wingcal_rmask` (0.564 px for about 97 percent of stars) is a wing-calibration radius and differs from the fit mask (the DQ SATURATED core has median radius about 2.5 px before recovery).
`fill` variant: masked core pixels filled with a_precap * PSF and Poisson errors. This pseudo-data lies exactly on the baseline model, so the baseline amplitude is a fixed point of the solve and `fill` returns the baseline (dm identical to uncapped within 0.001). It carries no information about the true core flux.

## 3. Variants

Amplitude ratios to baseline are scored through dm = ours - dolphot - ZP, using per-star shifts (median of -2.5 log10(a_variant/a_base) over m7 rows within 0.1") applied to the catalog magnitude of each star; stars as in capoff_tol.py. Only rows in the refit frames enter the shifts.
- V1 g2,g4,g6,g10: exclude pixels within g px of the saturated core.
- V2 w2,w4,w8: downweight by 1-exp(-d^2/2s^2), d = distance to the saturated core.
- V3 bgfar: background from far annulus (58-77 px SW, 30-40 px LW); bgfree: free constant fitted jointly.
- V4 uni: uniform weights (sigma only).
- V5 r10/r20/r40 (SW), r6/r12/r25 (LW): footprint radius limit.
- V6 nbr_all / nbr_none: subtract all other satstar models / none, against the pipeline's sequential subtraction.
- V7a/V7b: P' = P(1+Delta_unsat(r)). Source: Q_integ/radprof/radprof.md, "per-bin Delta(r) (neighbours removed), iso", unsat column, main2, precap tables (F150W lines ~394-420, F250M ~801-823); Delta set to zero beyond 0.35" (SW) / 0.5" (LW); dbar offset-corrected tables not used. V7a renormalised to the original PSF total flux, V7b not.
- Combinations as listed in the tables.

## 4. Results (measured)

Reference: unsaturated matched stars (not replaced) give median dm +0.003 (F150W 19-20 mag) and -0.001 (F250M 18-19 mag).
Baseline uncapped: F150W dm runs -0.142 (14-15 mag) to -0.047 (18-19); F250M runs -0.001 (14-15) to +0.029 (16-18), a 0.03 mag trend. "final" (capped) is 0.03-0.05 mag closer to zero at the bright end in F150W; in F250M it is +0.12 at 10-13 mag, +0.04 at 13-14.

Sensitivity (all-star median dm, F150W / F250M; uncapped baseline -0.078 / +0.016, final -0.047 / +0.030):
- g6, g10, w8, bgfar, r20 (r6/r12), nbr_all, nbr_none: F150W -0.076 to -0.087; F250M +0.013 to +0.023. Differences from baseline are 0.01 mag or less (g10 about 0.01 in both bands, moving fainter in SW and slightly brighter in LW).
- bgfree: -0.064 / +0.032 (F150W 14-15 mag moves from -0.142 to -0.117).
- uni: -0.054 / +0.033 (F150W 14-15 mag -0.091).
- uni+bgfree: -0.051 / +0.037.
- v7a (flux-conserving): -0.071 / +0.018. v7b (flux not conserved): -0.038 / +0.019; the SW shift of +0.040 follows from the added flux in the 1+Delta term.
- fill: identical to baseline (fixed point).

No single choice removes the offsets. In F150W the choices with the largest effect are free background and uniform weights (and V7b as an overall flux gain), each recovering about half of the 0.1 mag bright-to-faint gap. In F250M the trend (0.03 mag) is reduced to about 0.014-0.016 by bgfree or r6; uniform weights reverse its sign.

Full per-bin tables follow (cells are median dm with MAD in parentheses; N per bin at the end).

#### F150W: 2311 satstar-replaced matched stars; 1471 with rows in the refit frames; 1469 with every variant finite

| variant | 13-14 | 14-15 | 15-16 | 16-17 | 17-18 | 18-19 | all (med / MAD) |
|---|---|---|---|---|---|---|---|
| final | - | -0.095 (0.070) | -0.084 (0.048) | -0.057 (0.045) | -0.040 (0.044) | -0.032 (0.047) | -0.047 / 0.047 |
| uncapped | - | -0.142 (0.063) | -0.122 (0.055) | -0.101 (0.046) | -0.066 (0.045) | -0.047 (0.039) | -0.078 / 0.053 |
| g2 | - | -0.149 (0.073) | -0.122 (0.056) | -0.101 (0.046) | -0.066 (0.045) | -0.047 (0.039) | -0.078 / 0.053 |
| g4 | - | -0.144 (0.068) | -0.122 (0.057) | -0.101 (0.046) | -0.067 (0.045) | -0.047 (0.038) | -0.078 / 0.054 |
| g6 | - | -0.129 (0.067) | -0.123 (0.056) | -0.106 (0.045) | -0.073 (0.048) | -0.056 (0.038) | -0.083 / 0.056 |
| g10 | - | -0.149 (0.086) | -0.123 (0.056) | -0.109 (0.050) | -0.075 (0.052) | -0.056 (0.046) | -0.087 / 0.061 |
| w2 | - | -0.147 (0.071) | -0.122 (0.055) | -0.101 (0.046) | -0.066 (0.045) | -0.047 (0.039) | -0.078 / 0.053 |
| w4 | - | -0.149 (0.073) | -0.123 (0.056) | -0.103 (0.046) | -0.067 (0.044) | -0.047 (0.038) | -0.078 / 0.053 |
| w8 | - | -0.148 (0.072) | -0.123 (0.056) | -0.105 (0.045) | -0.068 (0.047) | -0.048 (0.037) | -0.079 / 0.054 |
| bgfar | - | -0.142 (0.064) | -0.129 (0.058) | -0.104 (0.048) | -0.068 (0.046) | -0.046 (0.039) | -0.080 / 0.056 |
| bgfree | - | -0.117 (0.058) | -0.099 (0.053) | -0.083 (0.046) | -0.054 (0.043) | -0.038 (0.039) | -0.064 / 0.048 |
| uni | - | -0.091 (0.071) | -0.092 (0.049) | -0.059 (0.046) | -0.048 (0.049) | -0.041 (0.051) | -0.054 / 0.050 |
| r10 | - | -0.137 (0.056) | -0.116 (0.051) | -0.095 (0.044) | -0.063 (0.042) | -0.046 (0.037) | -0.073 / 0.051 |
| r20 | - | -0.133 (0.052) | -0.117 (0.054) | -0.099 (0.046) | -0.064 (0.044) | -0.045 (0.038) | -0.076 / 0.053 |
| r40 | - | -0.140 (0.060) | -0.121 (0.055) | -0.101 (0.046) | -0.065 (0.044) | -0.047 (0.039) | -0.078 / 0.053 |
| nbr_all | - | -0.149 (0.057) | -0.123 (0.057) | -0.101 (0.046) | -0.065 (0.043) | -0.047 (0.037) | -0.077 / 0.053 |
| nbr_none | - | -0.136 (0.062) | -0.121 (0.048) | -0.102 (0.046) | -0.071 (0.047) | -0.053 (0.040) | -0.080 / 0.053 |
| fill | - | -0.142 (0.063) | -0.122 (0.055) | -0.101 (0.046) | -0.066 (0.045) | -0.047 (0.039) | -0.078 / 0.053 |
| v7a | - | -0.104 (0.054) | -0.101 (0.050) | -0.087 (0.046) | -0.063 (0.047) | -0.049 (0.042) | -0.071 / 0.048 |
| v7b | - | -0.072 (0.054) | -0.068 (0.050) | -0.055 (0.046) | -0.031 (0.047) | -0.016 (0.043) | -0.038 / 0.047 |
| g4+bgfree | - | -0.105 (0.062) | -0.099 (0.054) | -0.083 (0.046) | -0.054 (0.043) | -0.038 (0.038) | -0.064 / 0.049 |
| g4+bgfar | - | -0.153 (0.078) | -0.129 (0.058) | -0.104 (0.047) | -0.070 (0.047) | -0.049 (0.039) | -0.081 / 0.057 |
| g4+uni | - | -0.090 (0.074) | -0.092 (0.050) | -0.059 (0.048) | -0.046 (0.050) | -0.042 (0.051) | -0.055 / 0.050 |
| g4+uni+bgfree | - | -0.067 (0.064) | -0.088 (0.050) | -0.056 (0.046) | -0.043 (0.048) | -0.037 (0.051) | -0.051 / 0.049 |
| g6+bgfree | - | -0.089 (0.052) | -0.098 (0.052) | -0.085 (0.047) | -0.058 (0.044) | -0.047 (0.038) | -0.067 / 0.050 |
| g6+bgfar | - | -0.135 (0.070) | -0.129 (0.060) | -0.108 (0.049) | -0.077 (0.052) | -0.058 (0.042) | -0.086 / 0.059 |
| g6+uni | - | -0.090 (0.077) | -0.092 (0.050) | -0.064 (0.046) | -0.063 (0.044) | -0.061 (0.045) | -0.064 / 0.047 |
| g6+uni+bgfree | - | -0.059 (0.057) | -0.088 (0.050) | -0.059 (0.046) | -0.054 (0.046) | -0.049 (0.050) | -0.057 / 0.048 |
| uni+bgfree | - | -0.073 (0.055) | -0.088 (0.050) | -0.056 (0.045) | -0.044 (0.048) | -0.037 (0.052) | -0.051 / 0.050 |
| uni+bgfar | - | -0.091 (0.068) | -0.092 (0.049) | -0.058 (0.046) | -0.048 (0.049) | -0.041 (0.051) | -0.054 / 0.050 |
| r20+bgfree | - | -0.105 (0.048) | -0.094 (0.048) | -0.084 (0.041) | -0.053 (0.040) | -0.043 (0.035) | -0.063 / 0.045 |
| r20+uni+bgfree | - | -0.079 (0.062) | -0.090 (0.048) | -0.055 (0.046) | -0.045 (0.048) | -0.035 (0.054) | -0.051 / 0.049 |
| g4+r20+bgfree | - | -0.097 (0.041) | -0.095 (0.048) | -0.084 (0.040) | -0.053 (0.041) | -0.043 (0.036) | -0.063 / 0.046 |
| bgfree+v7a | - | -0.073 (0.047) | -0.075 (0.044) | -0.068 (0.045) | -0.050 (0.043) | -0.040 (0.040) | -0.057 / 0.042 |
| g4+bgfree+v7a | - | -0.073 (0.048) | -0.076 (0.044) | -0.069 (0.045) | -0.050 (0.045) | -0.041 (0.039) | -0.058 / 0.043 |

N per bin: 0, 57, 159, 385, 561, 306

Unsaturated matched stars (not replaced), 19-20 mag: median dm +0.003, MAD 0.029, N=1378
Unsaturated matched stars, 12-13 mag: median dm +nan, N=0

#### F250M: 1129 satstar-replaced matched stars; 905 with rows in the refit frames; 904 with every variant finite

| variant | 10-13 | 13-14 | 14-15 | 15-16 | 16-18 | all (med / MAD) |
|---|---|---|---|---|---|---|
| final | +0.125 (0.059) | +0.039 (0.036) | +0.021 (0.032) | +0.028 (0.034) | +0.031 (0.034) | +0.030 / 0.035 |
| uncapped | -0.032 (0.104) | +0.001 (0.053) | -0.001 (0.036) | +0.020 (0.036) | +0.029 (0.033) | +0.016 / 0.038 |
| g2 | -0.033 (0.089) | +0.001 (0.053) | -0.001 (0.036) | +0.019 (0.036) | +0.028 (0.032) | +0.016 / 0.038 |
| g4 | -0.041 (0.104) | -0.000 (0.056) | -0.001 (0.039) | +0.019 (0.036) | +0.029 (0.034) | +0.016 / 0.039 |
| g6 | -0.041 (0.124) | -0.003 (0.060) | -0.001 (0.043) | +0.018 (0.039) | +0.027 (0.035) | +0.015 / 0.042 |
| g10 | -0.062 (0.157) | -0.005 (0.062) | -0.004 (0.054) | +0.017 (0.041) | +0.025 (0.037) | +0.013 / 0.047 |
| w2 | -0.032 (0.099) | +0.001 (0.052) | -0.001 (0.037) | +0.019 (0.036) | +0.028 (0.032) | +0.016 / 0.038 |
| w4 | -0.033 (0.104) | +0.001 (0.052) | -0.001 (0.037) | +0.019 (0.038) | +0.028 (0.032) | +0.016 / 0.039 |
| w8 | -0.031 (0.096) | -0.004 (0.061) | -0.004 (0.042) | +0.019 (0.038) | +0.027 (0.034) | +0.015 / 0.041 |
| bgfar | -0.026 (0.105) | +0.001 (0.050) | +0.000 (0.037) | +0.019 (0.037) | +0.028 (0.034) | +0.016 / 0.039 |
| bgfree | -0.007 (0.086) | +0.024 (0.049) | +0.023 (0.040) | +0.036 (0.032) | +0.037 (0.035) | +0.032 / 0.035 |
| uni | +0.082 (0.070) | +0.047 (0.030) | +0.035 (0.026) | +0.033 (0.028) | +0.022 (0.035) | +0.033 / 0.032 |
| r6 | -0.003 (0.077) | +0.020 (0.041) | +0.014 (0.031) | +0.025 (0.033) | +0.030 (0.032) | +0.023 / 0.034 |
| r12 | -0.029 (0.077) | +0.006 (0.051) | +0.006 (0.034) | +0.023 (0.035) | +0.030 (0.032) | +0.019 / 0.037 |
| r25 | -0.027 (0.093) | +0.003 (0.052) | +0.002 (0.035) | +0.021 (0.035) | +0.030 (0.034) | +0.017 / 0.038 |
| nbr_all | -0.040 (0.106) | +0.001 (0.053) | -0.001 (0.035) | +0.022 (0.036) | +0.029 (0.033) | +0.016 / 0.039 |
| nbr_none | -0.042 (0.088) | +0.002 (0.057) | +0.001 (0.043) | +0.022 (0.035) | +0.028 (0.031) | +0.017 / 0.039 |
| fill | -0.032 (0.104) | +0.001 (0.053) | -0.001 (0.036) | +0.020 (0.036) | +0.029 (0.033) | +0.016 / 0.038 |
| v7a | -0.032 (0.103) | +0.001 (0.051) | +0.002 (0.035) | +0.023 (0.035) | +0.030 (0.033) | +0.018 / 0.039 |
| v7b | -0.030 (0.103) | +0.003 (0.051) | +0.002 (0.036) | +0.024 (0.036) | +0.031 (0.034) | +0.019 / 0.039 |
| g4+bgfree | -0.009 (0.092) | +0.023 (0.051) | +0.023 (0.040) | +0.035 (0.031) | +0.036 (0.036) | +0.032 / 0.035 |
| g4+bgfar | -0.025 (0.105) | -0.000 (0.054) | +0.000 (0.037) | +0.019 (0.037) | +0.028 (0.035) | +0.016 / 0.039 |
| g4+uni | +0.076 (0.090) | +0.046 (0.030) | +0.034 (0.026) | +0.033 (0.029) | +0.022 (0.035) | +0.033 / 0.033 |
| g4+uni+bgfree | +0.107 (0.068) | +0.051 (0.031) | +0.039 (0.028) | +0.037 (0.027) | +0.027 (0.035) | +0.037 / 0.032 |
| g6+bgfree | -0.009 (0.105) | +0.023 (0.050) | +0.022 (0.040) | +0.035 (0.032) | +0.037 (0.037) | +0.032 / 0.037 |
| g6+bgfar | -0.037 (0.123) | -0.003 (0.057) | -0.000 (0.043) | +0.018 (0.040) | +0.027 (0.036) | +0.015 / 0.042 |
| g6+uni | +0.077 (0.083) | +0.046 (0.030) | +0.033 (0.027) | +0.031 (0.029) | +0.019 (0.037) | +0.031 / 0.034 |
| g6+uni+bgfree | +0.105 (0.061) | +0.050 (0.031) | +0.040 (0.030) | +0.037 (0.028) | +0.027 (0.037) | +0.037 / 0.034 |
| uni+bgfree | +0.106 (0.067) | +0.051 (0.030) | +0.038 (0.027) | +0.037 (0.027) | +0.024 (0.033) | +0.037 / 0.031 |
| uni+bgfar | +0.082 (0.071) | +0.047 (0.030) | +0.035 (0.026) | +0.033 (0.028) | +0.022 (0.035) | +0.033 / 0.032 |
| r12+bgfree | +0.008 (0.079) | +0.029 (0.041) | +0.029 (0.031) | +0.034 (0.029) | +0.039 (0.032) | +0.033 / 0.032 |
| r12+uni+bgfree | +0.096 (0.058) | +0.049 (0.031) | +0.037 (0.026) | +0.035 (0.028) | +0.026 (0.037) | +0.036 / 0.032 |
| g4+r12+bgfree | +0.016 (0.078) | +0.030 (0.046) | +0.028 (0.031) | +0.034 (0.031) | +0.039 (0.032) | +0.033 / 0.033 |
| bgfree+v7a | -0.009 (0.086) | +0.025 (0.051) | +0.024 (0.040) | +0.038 (0.032) | +0.038 (0.034) | +0.034 / 0.035 |
| g4+bgfree+v7a | -0.010 (0.091) | +0.024 (0.049) | +0.024 (0.041) | +0.038 (0.032) | +0.037 (0.035) | +0.033 / 0.036 |

N per bin: 28, 80, 189, 374, 233

Unsaturated matched stars (not replaced), 18-19 mag: median dm -0.001, MAD 0.045, N=1522
Unsaturated matched stars, 9-10 mag: median dm +nan, N=0

## 5. Figure

satrefit.png: running median dm per 0.5 mag (bins with at least 8 stars), baseline uncapped, final (capped) and four variants per band.

# Round 2

Code: `delta_unsat.py` (Delta_unsat for F200W/F300M, validation on F150W/F250M), `run_frames2.py` (new variants, peak ratios, wing emulation; writes out2/), `satrefit_core.py` (`wing_emulation`, `pk_` columns; round-1 copy of the module kept in the session scratchpad), `score2.py`, `plot2.py`, `floor_check.py`. Figure: `satrefit2.png`. Full tables: `out2/tables2_<band>.md`.

## R2.1 Delta_unsat for F200W and F300M (V7 input)
radprof did not measure these bands. `delta_unsat.py` imports `radprof_measure.process_frame` unmodified and copies the per-bin `stack()` fit of `radprof_analyze.py`: isolated (iso) dolphot-matched unsaturated daophot stars (flags 0/1) in the 2 mag below the satstar faint edge (95th percentile of dolphot magnitudes of replaced matched stars: F200W 16.25, F300M 16.43), neighbours removed, per radial bin fit of sum(data-bg-nbrs) = (1+S) sum(model), weights 1/npix, 4-sigma clipping. Frames: F200W nrcb1+nrcb3, F300M nrcblong, exp 1-4. Profiles: `out/delta_unsat_F200W.txt`, `out/delta_unsat_F300M.txt`; per-star data `out/prof_main2_F200W.fits`, `out/prof_main2_F300M.fits`. Validation: the same aggregation applied to radprof/data/prof_main2_F250M.fits and F150W reproduces the radprof.md unsat column (F250M: +0.018, +0.015, -0.018, -0.043, +0.062, +0.007, -0.082, -0.079, ...; `out/delta_unsat_F250M_validate.txt`). V7b uses Delta set to zero beyond 0.35" (SW) / 0.5" (LW), not renormalised, as in round 1.

## R2.2 Scoring definitions
dm per bin as in round 1. "+cap": a = min(a_variant, cap * peak(P)/peak(P')) with cap = flux_fit_raw on rows with flux_fit_raw < 0.999 flux_fit_precap and +inf elsewhere; base+cap reproduces "final" exactly (check). The peak ratio is 0.9925 (F150W), 0.9856 (F200W), 0.9836 (F250M), 0.9934 (F300M) median, minimum 0.982; it changes the capped amplitude of 37-81 % of capped rows by at most 1.8 % (0.018 mag). trend = dm(faintest bin) - dm(brightest bin). step = dm(faintest satstar bin) - median dm of unsaturated matched stars in the 1 mag fainter than the satstar faint edge (F150W +0.004, F200W -0.017, F250M +0.003, F300M -0.003).

## R2.3 Wing self-calibration emulation
`wing_emulation` calls the production `_wing_selfcal` (read-only import) on data - satstar model image (read from the on-disk model, which the baseline reproduces to 1e-8) with the in-memory err (ZF recovery included), DQ SATURATED mask, the satstar PSF grid, radii 1,2,3,4,5,6,8 px (the function clips radii to 3-30 px, so buckets are 3,4,5,6,8) and fwhm from `get_fwhm`. Peak window: L = 99.999th percentile of finite, non-saturated, non-rim in-memory data (a saturation level measured from the frame: F150W 323-355, F200W 279-293, F250M 262-269, F300M 152-156 MJy/sr), window 0.12 L to 0.875 L. The crf SCI values at DQ SATURATED pixels (median 20-150 MJy/sr) are not usable as a level. The gate (n >= 8 stars in the best bucket, standard-error gate, C(0) = 1 anchor via `interp_wingcal_ratio`) follows `apply_wing_selfcal`; "wingU" applies all buckets without the gate. r_star: a = wingcal_rmask (median 0.56 px), b = circular-equivalent radius of the star's DQ SATURATED component before ZF recovery (median 2.7-4.0 px), c = radius of the deep-core mask left after ZF recovery (median 0 px; the mask is empty for most stars). Stock floors: F200W has an entry (4000), F150W/F250M/F300M none (`floor_check.py`: F150W returns "no usable peak window"). wd2 saturates near 150-350 MJy/sr in these frames, below the 4000 floor of F200W.

Measured per-frame C(r) (median ratio, n stars, madstd) are in `out2/tables2_<band>.md`. Summary: F150W nrcb1 C(3)=0.97-0.98, C(4-8)=0.59-0.91 (n 18-27); nrcb3 C(3-8)=1.00-1.09 (n 16-20); F200W nrcb1 C(3)=0.96-0.98, C(4)=0.95-0.98 (n 16-22); F250M 0.32-0.86 (n 2-5) in exp 1-3 and 1.4-2.9 in exp 4; F300M 0-12 (n 2-5). The ratios sit at or below 1 in SW (wing-only fits read slightly faint to unbiased), with madstd 0.06-1.2. LW frames reach 2-5 calibrators and fail the n >= 8 gate.

## R2.4 Results (median dm per bin with MAD; selected rows)

### F150W: 2311 satstar-replaced matched stars; 1471 with rows in the refit frames; 1469 with every variant finite
Satstar faint edge (95th percentile of replaced dolphot mags) 18.46; unsaturated reference 18.46-19.46 mag: median dm +0.004 (N=1309).

| variant | 14-15 | 15-16 | 16-17 | 17-18 | 18-19 | all med / MAD | trend (faint-bright bin) | step (faint bin - unsat ref) |
|---|---|---|---|---|---|---|---|---|
| final | -0.095 (0.070) | -0.084 (0.048) | -0.057 (0.045) | -0.040 (0.044) | -0.032 (0.047) | -0.047 / 0.047 | +0.063 | -0.035 |
| uncapped | -0.142 (0.063) | -0.122 (0.055) | -0.101 (0.046) | -0.066 (0.045) | -0.047 (0.039) | -0.078 / 0.053 | +0.095 | -0.051 |
| bgfree+v7b | -0.040 (0.047) | -0.043 (0.044) | -0.036 (0.045) | -0.017 (0.043) | -0.007 (0.040) | -0.024 / 0.043 | +0.033 | -0.011 |
| uni+v7b | -0.059 (0.047) | -0.076 (0.049) | -0.044 (0.047) | -0.032 (0.049) | -0.025 (0.052) | -0.039 / 0.049 | +0.033 | -0.029 |
| uni+bgfree+v7b | -0.023 (0.047) | -0.073 (0.050) | -0.041 (0.045) | -0.029 (0.048) | -0.022 (0.052) | -0.035 / 0.050 | +0.001 | -0.025 |
| r20+bgfree+v7b | -0.033 (0.038) | -0.041 (0.037) | -0.037 (0.039) | -0.018 (0.040) | -0.013 (0.035) | -0.024 / 0.040 | +0.020 | -0.017 |
| g4+bgfree+v7b | -0.040 (0.047) | -0.044 (0.044) | -0.037 (0.045) | -0.018 (0.044) | -0.009 (0.039) | -0.025 / 0.043 | +0.031 | -0.012 |
| bgfree | -0.117 (0.058) | -0.099 (0.053) | -0.083 (0.046) | -0.054 (0.043) | -0.038 (0.039) | -0.064 / 0.048 | +0.079 | -0.041 |
| v7b | -0.072 (0.054) | -0.068 (0.050) | -0.055 (0.046) | -0.031 (0.047) | -0.016 (0.043) | -0.038 / 0.047 | +0.055 | -0.020 |
| uni | -0.091 (0.071) | -0.092 (0.049) | -0.059 (0.046) | -0.048 (0.049) | -0.041 (0.051) | -0.054 / 0.050 | +0.050 | -0.044 |
| base+cap | -0.095 (0.070) | -0.084 (0.048) | -0.057 (0.045) | -0.040 (0.044) | -0.032 (0.047) | -0.047 / 0.047 | +0.063 | -0.035 |
| bgfree+v7b+cap | -0.021 (0.030) | -0.035 (0.038) | -0.024 (0.033) | -0.009 (0.041) | -0.002 (0.041) | -0.015 / 0.041 | +0.018 | -0.006 |
| uni+bgfree+v7b+cap | -0.014 (0.061) | -0.071 (0.050) | -0.036 (0.046) | -0.026 (0.049) | -0.019 (0.054) | -0.031 / 0.051 | -0.005 | -0.023 |
| r20+bgfree+v7b+cap | -0.016 (0.031) | -0.037 (0.035) | -0.024 (0.035) | -0.012 (0.041) | -0.007 (0.039) | -0.017 / 0.039 | +0.010 | -0.010 |
| bgfree+cap | -0.069 (0.052) | -0.070 (0.042) | -0.050 (0.040) | -0.036 (0.042) | -0.026 (0.042) | -0.042 / 0.043 | +0.043 | -0.030 |
| uni+bgfree+cap | -0.065 (0.066) | -0.084 (0.052) | -0.048 (0.048) | -0.040 (0.050) | -0.032 (0.054) | -0.045 / 0.051 | +0.034 | -0.035 |
| wing_a | -0.142 (0.066) | -0.122 (0.054) | -0.104 (0.043) | -0.068 (0.043) | -0.047 (0.037) | -0.079 / 0.053 | +0.094 | -0.051 |
| wing_b | -0.148 (0.061) | -0.128 (0.056) | -0.111 (0.050) | -0.073 (0.040) | -0.052 (0.034) | -0.085 / 0.054 | +0.096 | -0.056 |
| wing_b+cap | -0.094 (0.052) | -0.087 (0.030) | -0.069 (0.029) | -0.049 (0.035) | -0.036 (0.037) | -0.057 / 0.037 | +0.058 | -0.040 |
| bgfree+v7b+wing_b+cap | -0.036 (0.029) | -0.046 (0.040) | -0.037 (0.038) | -0.019 (0.037) | -0.006 (0.035) | -0.025 / 0.040 | +0.030 | -0.010 |
| wingU_b | -0.139 (0.127) | -0.125 (0.108) | -0.091 (0.076) | -0.053 (0.042) | -0.038 (0.034) | -0.061 / 0.054 | +0.101 | -0.042 |
| wingU_b+cap | -0.093 (0.081) | -0.075 (0.073) | -0.053 (0.041) | -0.031 (0.031) | -0.021 (0.033) | -0.039 / 0.038 | +0.072 | -0.025 |
| bgfree+v7b+wingU_b+cap | -0.016 (0.101) | -0.028 (0.081) | -0.016 (0.060) | +0.001 (0.035) | +0.004 (0.032) | -0.002 / 0.043 | +0.021 | +0.001 |
| wing_c+cap | -0.095 (0.071) | -0.084 (0.048) | -0.057 (0.045) | -0.040 (0.044) | -0.032 (0.047) | -0.047 / 0.047 | +0.063 | -0.035 |

N per bin: 57, 159, 385, 561, 306
r_star a: median 0.56 px (16-84%: 0.56-0.56); median gated C 1.000, ungated 1.000
r_star b: median 2.71 px (16-84%: 1.95-4.69); median gated C 1.000, ungated 1.001
r_star c: median 0.00 px (16-84%: 0.00-0.00); median gated C 1.000, ungated 1.000

### F200W: 721 satstar-replaced matched stars; 515 with rows in the refit frames; 515 with every variant finite
Satstar faint edge (95th percentile of replaced dolphot mags) 16.25; unsaturated reference 16.25-17.25 mag: median dm -0.017 (N=1130).

| variant | 13-14.5 | 14.5-15 | 15-15.5 | 15.5-16 | 16-18 | all med / MAD | trend (faint-bright bin) | step (faint bin - unsat ref) |
|---|---|---|---|---|---|---|---|---|
| final | -0.048 (0.062) | -0.074 (0.055) | -0.090 (0.054) | -0.063 (0.046) | -0.040 (0.052) | -0.066 / 0.054 | +0.008 | -0.023 |
| uncapped | -0.096 (0.057) | -0.095 (0.054) | -0.113 (0.040) | -0.090 (0.040) | -0.052 (0.048) | -0.097 / 0.050 | +0.044 | -0.035 |
| uni | -0.067 (0.056) | -0.083 (0.052) | -0.090 (0.055) | -0.066 (0.058) | -0.029 (0.048) | -0.069 / 0.060 | +0.038 | -0.012 |
| bgfree | -0.079 (0.049) | -0.078 (0.046) | -0.095 (0.045) | -0.074 (0.037) | -0.034 (0.055) | -0.079 / 0.047 | +0.045 | -0.018 |
| v7b | -0.046 (0.054) | -0.038 (0.052) | -0.058 (0.044) | -0.042 (0.045) | -0.019 (0.051) | -0.046 / 0.051 | +0.026 | -0.003 |
| bgfree+v7b | -0.022 (0.044) | -0.021 (0.048) | -0.041 (0.041) | -0.025 (0.041) | +0.002 (0.056) | -0.027 / 0.047 | +0.024 | +0.019 |
| uni+bgfree+v7b | -0.034 (0.063) | -0.060 (0.060) | -0.068 (0.059) | -0.039 (0.056) | +0.004 (0.052) | -0.044 / 0.060 | +0.037 | +0.020 |
| base+cap | -0.048 (0.062) | -0.074 (0.055) | -0.090 (0.054) | -0.063 (0.046) | -0.040 (0.052) | -0.066 / 0.054 | +0.008 | -0.023 |
| bgfree+cap | -0.029 (0.060) | -0.060 (0.051) | -0.070 (0.050) | -0.055 (0.043) | -0.031 (0.055) | -0.056 / 0.051 | -0.002 | -0.014 |
| bgfree+v7b+cap | -0.004 (0.060) | -0.016 (0.047) | -0.034 (0.041) | -0.018 (0.038) | +0.006 (0.051) | -0.020 / 0.045 | +0.009 | +0.022 |
| uni+bgfree+v7b+cap | -0.004 (0.072) | -0.057 (0.061) | -0.065 (0.059) | -0.039 (0.054) | +0.004 (0.053) | -0.039 / 0.060 | +0.007 | +0.020 |
| wing_a | -0.107 (0.056) | -0.103 (0.057) | -0.122 (0.040) | -0.097 (0.039) | -0.070 (0.054) | -0.104 / 0.049 | +0.038 | -0.053 |
| wing_b | -0.138 (0.042) | -0.120 (0.046) | -0.134 (0.039) | -0.112 (0.039) | -0.081 (0.045) | -0.121 / 0.047 | +0.057 | -0.064 |
| wing_b+cap | -0.083 (0.039) | -0.094 (0.029) | -0.104 (0.044) | -0.081 (0.034) | -0.068 (0.028) | -0.088 / 0.040 | +0.015 | -0.051 |
| bgfree+v7b+wing_b+cap | -0.050 (0.024) | -0.051 (0.033) | -0.055 (0.034) | -0.043 (0.034) | -0.027 (0.033) | -0.048 / 0.035 | +0.023 | -0.010 |
| wingU_b | -0.336 (0.106) | -0.156 (0.052) | -0.134 (0.039) | -0.113 (0.039) | -0.108 (0.069) | -0.139 / 0.067 | +0.228 | -0.091 |
| wingU_b+cap | -0.254 (0.089) | -0.132 (0.040) | -0.104 (0.044) | -0.082 (0.037) | -0.095 (0.056) | -0.109 / 0.057 | +0.158 | -0.079 |
| bgfree+v7b+wingU_b+cap | -0.223 (0.094) | -0.080 (0.053) | -0.055 (0.034) | -0.044 (0.033) | -0.049 (0.055) | -0.063 / 0.053 | +0.174 | -0.032 |
| wing_c+cap | -0.050 (0.061) | -0.077 (0.053) | -0.092 (0.053) | -0.065 (0.044) | -0.052 (0.049) | -0.069 / 0.052 | -0.002 | -0.036 |

N per bin: 80, 85, 147, 167, 36
r_star a: median 0.56 px (16-84%: 0.56-5.23); median gated C 0.995, ungated 0.996
r_star b: median 4.03 px (16-84%: 2.76-6.63); median gated C 0.984, ungated 0.973
r_star c: median 0.00 px (16-84%: 0.00-2.52); median gated C 1.000, ungated 1.000

### F250M: 1129 satstar-replaced matched stars; 905 with rows in the refit frames; 904 with every variant finite
Satstar faint edge (95th percentile of replaced dolphot mags) 16.46; unsaturated reference 16.46-17.46 mag: median dm +0.003 (N=1414).

| variant | 10-13 | 13-14 | 14-15 | 15-16 | 16-18 | all med / MAD | trend (faint-bright bin) | step (faint bin - unsat ref) |
|---|---|---|---|---|---|---|---|---|
| final | +0.125 (0.059) | +0.039 (0.036) | +0.021 (0.032) | +0.028 (0.034) | +0.031 (0.034) | +0.030 / 0.035 | -0.094 | +0.028 |
| uncapped | -0.032 (0.104) | +0.001 (0.053) | -0.001 (0.036) | +0.020 (0.036) | +0.029 (0.033) | +0.016 / 0.038 | +0.061 | +0.026 |
| bgfree+v7b | -0.008 (0.085) | +0.026 (0.051) | +0.025 (0.040) | +0.039 (0.032) | +0.038 (0.033) | +0.034 / 0.035 | +0.046 | +0.035 |
| uni+v7b | +0.095 (0.070) | +0.057 (0.030) | +0.044 (0.025) | +0.042 (0.030) | +0.032 (0.036) | +0.042 / 0.033 | -0.062 | +0.030 |
| uni+bgfree+v7b | +0.116 (0.067) | +0.060 (0.031) | +0.047 (0.025) | +0.047 (0.027) | +0.034 (0.034) | +0.046 / 0.032 | -0.082 | +0.031 |
| r12+bgfree+v7b | +0.010 (0.075) | +0.030 (0.040) | +0.031 (0.030) | +0.039 (0.030) | +0.040 (0.033) | +0.036 / 0.032 | +0.030 | +0.038 |
| r6+bgfree+v7b | +0.045 (0.071) | +0.047 (0.037) | +0.043 (0.027) | +0.046 (0.028) | +0.042 (0.033) | +0.044 / 0.029 | -0.003 | +0.040 |
| g4+bgfree+v7b | -0.010 (0.092) | +0.024 (0.049) | +0.024 (0.041) | +0.039 (0.032) | +0.037 (0.034) | +0.034 / 0.036 | +0.047 | +0.034 |
| bgfree | -0.007 (0.086) | +0.024 (0.049) | +0.023 (0.040) | +0.036 (0.032) | +0.037 (0.035) | +0.032 / 0.035 | +0.044 | +0.034 |
| v7b | -0.030 (0.103) | +0.003 (0.051) | +0.002 (0.036) | +0.024 (0.036) | +0.031 (0.034) | +0.019 / 0.039 | +0.061 | +0.028 |
| uni | +0.082 (0.070) | +0.047 (0.030) | +0.035 (0.026) | +0.033 (0.028) | +0.022 (0.035) | +0.033 / 0.032 | -0.060 | +0.019 |
| base+cap | +0.125 (0.059) | +0.039 (0.036) | +0.021 (0.032) | +0.028 (0.034) | +0.031 (0.034) | +0.030 / 0.035 | -0.094 | +0.028 |
| bgfree+v7b+cap | +0.156 (0.085) | +0.060 (0.040) | +0.041 (0.030) | +0.045 (0.030) | +0.040 (0.035) | +0.045 / 0.035 | -0.116 | +0.037 |
| uni+bgfree+v7b+cap | +0.158 (0.061) | +0.065 (0.034) | +0.052 (0.030) | +0.048 (0.029) | +0.036 (0.036) | +0.049 / 0.034 | -0.122 | +0.033 |
| bgfree+cap | +0.140 (0.088) | +0.049 (0.040) | +0.035 (0.033) | +0.041 (0.030) | +0.039 (0.035) | +0.040 / 0.035 | -0.101 | +0.036 |
| uni+bgfree+cap | +0.141 (0.060) | +0.053 (0.032) | +0.041 (0.029) | +0.038 (0.028) | +0.027 (0.035) | +0.039 / 0.032 | -0.114 | +0.024 |
| wing_a | -0.032 (0.104) | +0.001 (0.053) | -0.001 (0.036) | +0.020 (0.036) | +0.029 (0.033) | +0.016 / 0.038 | +0.061 | +0.026 |
| wing_b | -0.032 (0.104) | +0.001 (0.053) | -0.001 (0.036) | +0.020 (0.036) | +0.029 (0.033) | +0.016 / 0.038 | +0.061 | +0.026 |
| wing_b+cap | +0.125 (0.059) | +0.039 (0.036) | +0.021 (0.032) | +0.028 (0.034) | +0.031 (0.034) | +0.030 / 0.035 | -0.094 | +0.028 |
| bgfree+v7b+wing_b+cap | +0.140 (0.088) | +0.050 (0.041) | +0.035 (0.033) | +0.043 (0.031) | +0.040 (0.035) | +0.042 / 0.036 | -0.100 | +0.037 |
| wingU_b | -0.762 (0.157) | -0.342 (0.127) | -0.228 (0.059) | -0.132 (0.052) | -0.077 (0.053) | -0.139 / 0.095 | +0.685 | -0.080 |
| wingU_b+cap | -0.567 (0.176) | -0.293 (0.124) | -0.196 (0.048) | -0.123 (0.050) | -0.071 (0.050) | -0.127 / 0.083 | +0.496 | -0.074 |
| bgfree+v7b+wingU_b+cap | -0.543 (0.197) | -0.283 (0.135) | -0.184 (0.044) | -0.108 (0.046) | -0.063 (0.050) | -0.113 / 0.083 | +0.479 | -0.066 |
| wing_c+cap | +0.125 (0.059) | +0.039 (0.036) | +0.021 (0.032) | +0.028 (0.034) | +0.031 (0.034) | +0.030 / 0.035 | -0.094 | +0.028 |

N per bin: 28, 80, 189, 374, 233
r_star a: median 0.56 px (16-84%: 0.56-0.56); median gated C 1.000, ungated 0.966
r_star b: median 2.76 px (16-84%: 1.95-4.92); median gated C 1.000, ungated 0.868
r_star c: median 0.00 px (16-84%: 0.00-0.00); median gated C 1.000, ungated 1.000

### F300M: 1224 satstar-replaced matched stars; 944 with rows in the refit frames; 944 with every variant finite
Satstar faint edge (95th percentile of replaced dolphot mags) 16.43; unsaturated reference 16.43-17.43 mag: median dm -0.003 (N=1433).

| variant | 10-13 | 13-14 | 14-15 | 15-16 | 16-18 | all med / MAD | trend (faint-bright bin) | step (faint bin - unsat ref) |
|---|---|---|---|---|---|---|---|---|
| final | +0.116 (0.050) | +0.027 (0.035) | +0.014 (0.030) | +0.019 (0.035) | +0.015 (0.034) | +0.018 / 0.035 | -0.101 | +0.018 |
| uncapped | -0.044 (0.090) | -0.007 (0.049) | -0.003 (0.040) | +0.011 (0.035) | +0.014 (0.035) | +0.006 / 0.039 | +0.058 | +0.017 |
| uni | +0.031 (0.093) | +0.026 (0.032) | +0.019 (0.023) | +0.022 (0.030) | +0.014 (0.036) | +0.020 / 0.031 | -0.016 | +0.017 |
| bgfree | -0.008 (0.063) | +0.013 (0.040) | +0.014 (0.032) | +0.024 (0.031) | +0.021 (0.034) | +0.019 / 0.034 | +0.029 | +0.024 |
| v7b | -0.047 (0.104) | -0.008 (0.048) | -0.003 (0.041) | +0.015 (0.036) | +0.018 (0.036) | +0.008 / 0.040 | +0.065 | +0.021 |
| bgfree+v7b | -0.011 (0.064) | +0.012 (0.045) | +0.013 (0.034) | +0.027 (0.033) | +0.025 (0.035) | +0.021 / 0.036 | +0.036 | +0.028 |
| uni+bgfree+v7b | +0.079 (0.090) | +0.036 (0.030) | +0.031 (0.025) | +0.035 (0.030) | +0.024 (0.034) | +0.032 / 0.030 | -0.056 | +0.027 |
| base+cap | +0.116 (0.050) | +0.027 (0.035) | +0.014 (0.030) | +0.019 (0.035) | +0.015 (0.034) | +0.018 / 0.035 | -0.101 | +0.018 |
| bgfree+cap | +0.116 (0.043) | +0.035 (0.037) | +0.023 (0.031) | +0.029 (0.032) | +0.022 (0.033) | +0.026 / 0.032 | -0.094 | +0.025 |
| bgfree+v7b+cap | +0.127 (0.043) | +0.038 (0.037) | +0.024 (0.029) | +0.033 (0.032) | +0.027 (0.034) | +0.031 / 0.034 | -0.100 | +0.030 |
| uni+bgfree+v7b+cap | +0.150 (0.080) | +0.043 (0.027) | +0.034 (0.026) | +0.035 (0.031) | +0.024 (0.034) | +0.034 / 0.031 | -0.126 | +0.027 |
| wing_a | -0.044 (0.090) | -0.007 (0.049) | -0.003 (0.040) | +0.011 (0.035) | +0.014 (0.035) | +0.006 / 0.039 | +0.058 | +0.017 |
| wing_b | -0.044 (0.090) | -0.007 (0.049) | -0.003 (0.040) | +0.011 (0.035) | +0.014 (0.035) | +0.006 / 0.039 | +0.058 | +0.017 |
| wing_b+cap | +0.116 (0.050) | +0.027 (0.035) | +0.014 (0.030) | +0.019 (0.035) | +0.015 (0.034) | +0.018 / 0.035 | -0.101 | +0.018 |
| bgfree+v7b+wing_b+cap | +0.116 (0.050) | +0.033 (0.034) | +0.023 (0.031) | +0.032 (0.032) | +0.027 (0.034) | +0.028 / 0.034 | -0.089 | +0.030 |
| wingU_b | +0.967 (0.188) | +1.079 (0.199) | +1.108 (0.089) | +0.671 (0.184) | +0.508 (0.114) | +0.791 / 0.367 | -0.459 | +0.511 |
| wingU_b+cap | +1.160 (0.217) | +1.114 (0.180) | +1.150 (0.066) | +0.687 (0.183) | +0.509 (0.112) | +0.810 / 0.386 | -0.652 | +0.512 |
| bgfree+v7b+wingU_b+cap | +1.166 (0.220) | +1.126 (0.183) | +1.158 (0.072) | +0.696 (0.184) | +0.520 (0.109) | +0.824 / 0.385 | -0.646 | +0.523 |
| wing_c+cap | +0.116 (0.050) | +0.027 (0.035) | +0.014 (0.030) | +0.019 (0.035) | +0.015 (0.034) | +0.018 / 0.035 | -0.101 | +0.018 |

N per bin: 19, 97, 232, 399, 197
r_star a: median 0.56 px (16-84%: 0.56-0.56); median gated C 1.000, ungated 1.301
r_star b: median 3.78 px (16-84%: 2.19-5.32); median gated C 1.000, ungated 2.126
r_star c: median 0.00 px (16-84%: 0.00-0.00); median gated C 1.000, ungated 1.000

## R2.5 Conclusions
Measured:
- F150W: bgfree+v7b+cap gives all-star median -0.015 (MAD 0.041), trend +0.018, step -0.006; uncapped -0.078, final -0.047. r20+bgfree+v7b+cap: -0.017, trend +0.010, MAD 0.039.
- F200W: bgfree+v7b+cap -0.020 (trend +0.009, step +0.022); final -0.066 (trend +0.008); uncapped -0.097.
- F250M, F300M: v7b alone does not move the medians (0.003 mag); bgfree changes the bright bins by 0.02 and the faint bins by 0.01. The cap dominates the bright end (final +0.125 at 10-13 mag in F250M, +0.116 in F300M), and all +cap variants stay within 0.03 of final there. Uncapped bgfree+v7b has medians +0.034 (F250M) and +0.021 (F300M), trend +0.046 and +0.036.
- Wing self-calibration: gated C is 1.000 in LW (calibrator count 2-5) and 0.98-1.00 in SW; wing_b+cap equals final within 0.01 in F250M/F300M and moves F150W/F200W by -0.01 to -0.02 mag (away from dolphot). Ungated LW buckets give C up to 12 and dm shifts of 0.2-1 mag; ungated SW with r_b raises scatter to MAD 0.1.
Inferred: the SW offset behaves like a PSF-wing normalisation plus background-degeneracy effect that bgfree+v7b removes to about 0.02 mag; the LW residual of +0.02 to +0.04 and the bright-end rise come from the cap and the footprint, not from a wing deficit measurable with the existing calibrator code.
