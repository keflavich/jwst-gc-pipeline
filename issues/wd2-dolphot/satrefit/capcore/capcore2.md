# capcore2: recovered-core cap with high-fill ZEROFRAME pixels excluded

Files: capcore2_run.py, capcore2_score.py, capcore2_tables.md (all variants, all bands), capcore2_sff.md (per-frame S_ff), capcore2.png, out2/*.fits.

## Method
- Fit and precap unchanged. flux_new = min(precap, cap_new); where cap_new is NaN, flux_new = precap.
- Flagged pixel = rim pixel rewritten from group 0 where group 0 is saturated. Excluded if first-frame DN > t * S_ff. Excluded pixels are treated as not measured for the cap.
- S_ff(a) = 99.9th percentile of positive first-frame pixels. S_ff(b) = max positive first-frame pixel. Per-frame values in capcore2_sff.md.
- S_ff(a) is 3.7e3-9.0e3 DN (SW) and 4.2e3-5.6e3 DN (LW). S_ff(b) is 4.6e4-5.4e4 DN, close to the group-0 saturation level (g0 99th pct at SATURATED pixels, 4.6e4-5.4e4). The (a) thresholds therefore sit near 10% of full well and exclude nearly every flagged pixel; (b) thresholds at 0.5-0.9 sit at 50-90% of full well.
- Variant A (suffix _gm): excluded pixels are also unrecoverable in recovered_core_peak, so they add to lost_frac and can trip the 0.2 skip. Variant B (suffix _go): excluded only inside recovered_cap_flux; lost_frac unchanged.
- B*_gm: A plus exclusion of unflagged rim pixels with group 0 > t * g0sat99. Results equal the a*_gm rows to the printed precision (no additional pixels excluded).
- Reproduction check: threshold 1.0 (nothing excluded) and the original cap reproduce flux_fit_raw to max |ratio-1| = 5.9e-08 in all four bands.

## Results (median dm, MAD in parentheses; dm = ours - dolphot - ZP; positive = ours fainter)
Bins of interest. SW = F150W 14-15 / 15-16; F200W 14-15 / 15-16; LW = F250M 12-13 / 13-14; F300M 12-13 / 13-14.

| variant | F150W 14-15 | F150W 15-16 | F200W 14-15 | F200W 15-16 | F250M 12-13 | F250M 13-14 | F300M 12-13 | F300M 13-14 |
|---|---|---|---|---|---|---|---|---|
| final | -0.095 | -0.084 | -0.071 | -0.068 | +0.125 | +0.039 | +0.116 | +0.027 |
| precap | -0.142 | -0.122 | -0.095 | -0.102 | -0.032 | +0.001 | -0.044 | -0.007 |
| a0.6 A | -0.103 | -0.034 | -0.075 | -0.088 | -0.021 | +0.121 | +0.020 | +0.066 |
| a0.6 B | -0.103 | -0.024 | -0.058 | -0.051 | -0.016 | +0.124 | +0.057 | +0.066 |
| b0.6 A | -0.066 | -0.084 | -0.065 | -0.068 | +0.117 | +0.049 | +0.137 | +0.032 |
| b0.6 B | -0.066 | -0.084 | -0.065 | -0.068 | +0.117 | +0.049 | +0.137 | +0.032 |
| b0.8 B | -0.077 | -0.084 | -0.069 | -0.068 | +0.113 | +0.038 | +0.095 | +0.025 |

All-star median / MAD / slope per mag:

| variant | F150W | F200W | F250M | F300M |
|---|---|---|---|---|
| final | -0.047 / 0.047 / +0.020 | -0.066 / 0.054 / -0.006 | +0.030 / 0.035 / -0.007 | +0.018 / 0.035 / -0.009 |
| precap | -0.078 / 0.053 / +0.032 | -0.097 / 0.050 / +0.010 | +0.016 / 0.038 / +0.016 | +0.006 / 0.039 / +0.014 |
| a0.6 A | -0.042 / 0.050 / +0.008 | -0.081 / 0.056 / +0.005 | +0.038 / 0.044 / -0.017 | +0.025 / 0.042 / -0.016 |
| a0.6 B | -0.040 / 0.051 / +0.003 | -0.051 / 0.059 / -0.001 | +0.038 / 0.044 / -0.020 | +0.026 / 0.043 / -0.018 |
| b0.6 A or B | -0.047 / 0.047 / +0.017 | -0.065 / 0.054 / -0.004 | +0.030 / 0.036 / -0.008 | +0.018 / 0.035 / -0.010 |

Rows switched from cap-evaluated to cap-skipped under A (rows with a finite cap: original / variant):
- a0.6 A: F150W 262 (5554 -> 5292), F200W 1158 (2023 -> 865), F250M 200 (3364 -> 3164), F300M 193 (3552 -> 3359).
- a0.5 / 0.7 / 0.8 A: F150W 342/207/169; F200W 1230/1025/864; F250M 228/174/147; F300M 224/147/113.
- b0.6 A: 1, 6, 0, 1 rows. B never skips (0 in all cases).

Fraction of rows where the cap binds (flux_new < 0.999 precap), final vs a0.6 A / a0.6 B / b0.6: F150W 0.59 / 0.57 / 0.61 / 0.60; F200W 0.61 / 0.22 / 0.57 / 0.60; F250M 0.42 / 0.40 / 0.43 / 0.43; F300M 0.42 / 0.40 / 0.44 / 0.42.

## Reading
- Excluding pixels lowers the measured maximum, so the cap drops. Where the old cap was already too low (LW 13-14, and 12-13 relative to precap) the (a) thresholds raise dm further (F250M 13-14 +0.039 -> +0.12). Where the old cap was too high or the fill was corrupted (F150W 15-16, F200W 14-16), dm moves toward zero (F150W 15-16 -0.084 -> -0.024/-0.034).
- (a) thresholds act as a switch for nearly every flagged pixel. They help SW and F200W medians and slopes (F150W slope +0.020 -> +0.003 under B) and degrade LW (all-star MAD 0.035 -> 0.043, slope -0.009 -> -0.018).
- (b) thresholds change very few rows. Gains are limited to the brightest SW bin (F150W 14-15: -0.095 -> -0.066 at 0.6, N=57) and F200W 14-15 (+0.006). LW 12-13 is unchanged or slightly worse at 0.6 (F300M +0.116 -> +0.137, N=19); b0.8-0.9 is neutral in LW.
- LW 12-13 offset (+0.12) stays at precap-to-final difference scale under every threshold in (b). The offset only falls under (a), and then 13-14 rises. No tested threshold removes the LW 12-13 offset without worsening 13-14.
- Variant A removes the cap for many rows at (a) thresholds (F200W loses 57% of finite caps at 0.6), which makes F200W fall back toward precap (-0.081). Variant B keeps all caps and gives the better F200W result (-0.051).
- Full grid (0.5-0.8 for a, 0.5-0.9 for b, both A and B) is in capcore2_tables.md.

## Recommendation
- Design B (exclusion only inside recovered_cap_flux). A discards finite caps and changes which rows are skipped.
- If one threshold has to be chosen: b with t = 0.8 of max first frame. It is neutral or slightly positive in all four bands (F150W 14-15 -0.077, F300M 12-13 +0.095, all-star medians/MADs equal to final) and does not skip rows. The gain is small.
- a0.6 under B improves SW and F200W (all-star medians -0.040 and -0.051 vs -0.047 and -0.066; F150W slope +0.003) but costs LW (MAD +0.008, slope doubled in magnitude). It fits only if applied to SW/F200W alone; that band split is a choice outside this test.

## Caveats
- Bins at the bright end are small (F150W 14-15 N=57; LW 12-13 N=19-28); differences of 0.02-0.03 mag are within about 1 sigma of the MAD-based median error.
- S_ff(a) sits at about 10% of full well, so the (a) fractions do not map to a physical fill fraction. The (b) scale matches the group-0 saturation level.
- The dm reference is dolphot, which has its own bright-end systematics.
- The conclusion from capcore.md that the high-fill deficit follows from uncorrected ZEROFRAME nonlinearity is superseded: stcal linearity corrects the ZEROFRAME, so a failing correction near full well or the k extrapolation above about 6e3 DN are the candidates. This test does not separate them.
