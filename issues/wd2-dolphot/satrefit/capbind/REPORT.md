# LW recovered-core cap experiment (F250M, F300M; SW checked with F150W, F200W)

dm = ours - dolphot - ZP; positive dm means ours reads faint. References: F150W +0.004, F200W -0.017, F250M +0.003, F300M -0.003. Labels: (M) measured, (I) inferred. Detailed tables: `an1_tables.md`, `an2_tables.md`, `an3_tables.md`, `an3b_tables.md`, `an4_tables_full.md`, `an4b_tables.md`. Figures: `fig1_cap_rules.png`, `fig2_wing_q.png`.

## Conclusion

Reading A (cap too tight because near-full-well recovered core pixels under-read) carries most of the effect. Reading B (post-saturation charge in the ramp-fit wings) is present at d <= 4 px but is smaller than the cap bias and does not appear in the amplitude fit. A cap with a floor, a_eff = max(min(a, cap), tau * a), tau = 0.96, brings F250M and F300M to within about 0.02 of the references in every bin, with no change at 15 mag and fainter in SW and shifts of 0.001-0.004 in the bright SW bins.

## Task 1: binding pixel of cap_H (M)

Binding rows: 12.3-13 mag, 80% (F250M) and 89% (F300M); 13-13.5, 70% and 74%; 13.5-14, 45% and 53%; 14-15, 45% and 45%; 15-16, 21%; 16-17, 8% and 6%.

| dolphot mag | F250M binding pixel | frac replaced by first frame x k | median source DN / full well | median data/model (a_H) | median data/model (a_H+h0+bgfree) |
|---|---|---|---|---|---|
| 12.3-13 | 61% model-peak pixel, offset 0.53 px | 0.95 | 0.85 | 0.868 | 0.905 |
| 13-13.5 | 97% model-peak | 1.00 | 0.73 | 0.940 | 0.959 |
| 13.5-14 | 93% model-peak | 1.00 | 0.43 | 0.962 | 0.991 |
| 14-15 | 91% model-peak | 0.64 | 0.28 | 0.969 | 0.992 |
| 15-16 | 91% model-peak | 0.01 | 0.24 | 0.973 | 0.990 |
| 16-17 | 96% model-peak | 0.00 | 0.11 | 0.959 | 0.978 |

F300M is the same within 0.01 (data/model 0.872 at 12.3-13). The binding pixel is a first-frame-replaced pixel whose g0 sits at the ceiling (g0/ceiling 1.0) and whose source DN is 0.84-0.85 of full well. The data read 13% below the H model there.

Why the capcore2 "b" exclusion did not raise the 12-13 cap (M): excluding replaced pixels with first frame > 0.6 or 0.8 of the maximum removes the binding pixel in 62-77% of rows, but the cap rises by only 2-5% in those rows (median cap_new/cap_old 1.02-1.05). The new binding pixel is again a replaced pixel (95-100%) with source DN/FW 0.41-0.66 and data/model 0.84-0.89. Neighbouring core pixels under-read by the same 11-16%, so the exclusion hands the cap to the next equally low pixel. 98% (F250M) and 88-89% (F300M) of rows still have cap < a_H after the exclusion.

## Task 2: wing migration (M)

q = cal / (R_h g0), median per unit, F250M + F300M combined (`fig2_wing_q.png`; per-band and q_f in `an2_tables.md`).

| group | d=1 | 2 | 3 | 4-5 | 6-8 | 9-12 | 13-20 |
|---|---|---|---|---|---|---|---|
| sat 12.3-13 | 1.144 | 1.107 | 1.071 | 1.029 | 0.995 | 0.994 | 0.986 |
| sat 13-14 | 1.090 | 1.064 | 1.038 | 1.011 | 0.979 | 0.980 | 0.993 |
| sat 14-15 | 1.045 | 1.014 | 0.995 | 0.971 | 0.951 | 0.954 | 0.976 |
| sat 15-17 | 1.026 | 0.984 | 0.945 | 0.938 | 0.945 | 0.963 | 0.976 |
| control g0 1500-4000 | 1.026 | 1.036 | 0.997 | 0.968 | 0.965 | 0.980 | 0.983 |
| control g0 4000-10000 | 1.052 | 1.064 | 1.027 | 0.991 | 0.975 | 0.960 | 0.989 |

Satstars at 12.3-13 mag exceed the 15-17 mag set by 0.12 (d=1), 0.12 (d=2), 0.13 (d=3), 0.09 (d=4-5) and reach parity by d=6-8. Against the g0 4000-10000 control the excess is 0.09, 0.04, 0.04, 0.04 at d=1-5. Measured migration therefore amounts to +4 to +9% at d <= 5 px for the brightest stars. The R(g0) drift (0.93 at 2800 DN) is common to all groups and cancels in the comparisons above. The "unmatched/other" group (satstar not tied to a saturated component) reads 1.15 at d=1 and decays slowly, consistent with overlapping bright neighbours.

## Task 3: group-0 photometry (M, with I for interpretation)

Calibration on 15-17 mag rows (a_g0 / a_(H+h0+bgfree) = 1.000 to 1.01). Calibrated ratio by g0 cut (g10 = pixels with g0 < 0.1 x ceiling, etc.; f-variants use the field R(g0) curve; `an3b_tables.md`):

| F250M variant | 12.3-13 | 13-13.5 | 13.5-14 | 14-15 | 15-16 | dm 12.3-13 |
|---|---|---|---|---|---|---|
| g10 | 1.012 | 1.019 | 1.041 | 1.021 | 0.991 | -0.037 |
| g20 | 1.067 | 1.053 | 1.023 | 0.998 | 0.989 | -0.100 |
| g30 | 1.105 | 1.078 | 1.016 | 0.993 | 0.997 | -0.147 |
| g50 | 1.136 | 1.087 | 1.014 | 0.999 | 0.998 | -0.175 |
| f10 | 1.023 | 1.023 | 1.051 | 1.040 | 0.993 | -0.052 |
| f50 | 1.142 | 1.087 | 1.062 | 1.045 | 0.981 | -0.173 |

F300M g50: 1.158 / 1.091 / 1.018 / 1.001 at 12.3-13 / 13-13.5 / 13.5-14 / 14-15; g10 0.986 at 12.3-13. For g50, a_g0 / cap_H (capped rows) is 1.19 at 13-13.5 against a_H0bg / cap_H of 1.05.

The field R(g0) curve changes the 12.3-13 ratio by under 0.01, so the excess is not an R(g0) drift artefact. The ratio grows with the g0 cut at 12.3-13 (1.01 at g10, 1.14 at g50). Low-g0 outer-wing pixels agree with the ramp-fit amplitude within 1.4%, so the wing amplitude carries no large migration bias at the 0.05 mag level (I). Pixels nearer the core (g0 up to 0.5 x ceiling) imply an amplitude 10-16% higher. The resulting dm of -0.17 for a_g0 g50 is larger than any plausible star-to-star offset relative to dolphot, so the inner-pixel g0 values probably carry a PSF-shape or nonlinearity term as well (I). The g0 photometry supports the statement that the uncapped amplitude is not too bright by more than a few percent and that the cap (a_H0bg/cap_H = 1.09-1.12 at 12.5-13 mag) trims real flux.

## A versus B

- A (M + I): the binding core pixels read 0.87-0.91 of the model at 12.3-13 mag, at 0.84 of full well, and neighbouring pixels read equally low. The capped dm (+0.082 F250M, +0.079 F300M at 12.3-13) is 0.08 faint versus the reference, while the uncapped amplitude gives -0.030 and -0.025.
- B (M): migration of +4 to +9% in cal at d <= 5 px exists. The H+h0 rewrite replaces these wing pixels with R x g0, and the g10 fit agrees with the H0bg amplitude to 1.4%, so the migration does not carry into a_H+h0+bgfree at a level that explains an 8% cap deficit. Residual uncapped brightness of 0.03 mag at 12.3-13 (and -0.033 at 13-13.5 in F250M) is the share B could still account for.

## Task 4: cap rules (M)

Median dm per dolphot bin for H+h0+bgfree (`an4_tables_full.md`; H-only variants are also listed there). "max dev" is the largest |bin median - reference|.

| F250M variant | 12.3-13 | 13-13.5 | 13.5-14 | 14-15 | 15-16 | 16-17 | max dev |
|---|---|---|---|---|---|---|---|
| final | +0.125 | +0.051 | +0.032 | +0.021 | +0.028 | +0.031 | 0.122 |
| uncapped | -0.030 | -0.033 | -0.011 | -0.013 | +0.002 | +0.011 | 0.036 |
| round-7 cap | +0.082 | +0.004 | -0.005 | -0.008 | +0.006 | +0.013 | 0.079 |
| (i) exclude src frac > 0.5 | +0.069 | +0.022 | -0.004 | -0.008 | +0.006 | +0.013 | 0.066 |
| (i) f = 0.7 | +0.068 | +0.004 | -0.005 | -0.008 | +0.006 | +0.013 | 0.065 |
| (ii) skip if bind > 0.5 FW | -0.015 | -0.033 | -0.005 | -0.008 | +0.006 | +0.013 | 0.036 |
| (ii) f = 0.7 | -0.015 | -0.029 | -0.005 | -0.008 | +0.006 | +0.013 | 0.032 |
| (ii) f = 0.9 | +0.056 | +0.004 | -0.005 | -0.008 | +0.006 | +0.013 | 0.053 |
| (iii-b) floor tau = 0.96 | +0.005 | -0.021 | -0.005 | -0.010 | +0.005 | +0.012 | 0.024 |
| (iii-a) kappa-corrected cap | +0.005 to +0.018 | -0.030 | -0.005 | -0.008 | +0.005 | +0.013 | 0.033 |

F300M: round-7 +0.079 / -0.006 / -0.003 / -0.006 / +0.001 / +0.005 (max dev 0.082); (i) f = 0.5 and 0.7 give +0.08 at 12.3-13 with MAD 0.12 (worse scatter); (ii) f = 0.7 gives -0.009 / -0.043 / -0.003 (max dev 0.040); floor tau = 0.96 gives +0.017 / -0.021 / -0.006 / -0.006 / +0.000 / +0.005 (max dev 0.020); tau = 0.97 gives max dev 0.018.

Rule results (M):
- (i) fails. The 0.5-0.8 source-fraction thresholds leave 12.3-13 at +0.065 to +0.085 and raise the MAD from 0.06 to 0.10-0.12, because switching to the second cap branch yields erratic caps. The gate-off variants behave alike (listed in `an4_tables_full.md`).
- (ii) with f = 0.5-0.7 removes the 12.3-13 offset (-0.015 F250M, -0.009 F300M) and overshoots at 13-13.5 (-0.03 to -0.045, since those rows have binding src fraction 0.7-0.73 and then run uncapped). The f = 0.9 variant leaves the bias in place. No f in 0.5-0.8 meets +-0.02 in all bins. F150W 14-15 worsens from -0.040 to -0.065 at f = 0.5-0.7 (SW regression).
- (iii-a) a cap divided by the measured data/model ratio vs source fraction (kappa drops from 1.07 at low DN to 0.90 above 0.9 FW) recovers 12.3-13 but leaves 13-13.5 at -0.03 to -0.05, and SW regresses (F150W 14-15 -0.065, F200W 14-15 -0.027).
- (iii-b) a floor at tau x a, a_eff = max(min(a, cap), tau a). tau scan (`an4b_tables.md`): F250M max dev 0.019 at tau = 0.95, 0.024 at 0.96; F300M 0.031 at 0.95, 0.020 at 0.96, 0.018 at 0.97. SW at tau = 0.96: F150W 14-15 -0.047 (round 7: -0.040), 15-16 -0.044 (-0.041), 16-17 -0.029 (-0.027), 17-19 shifts under 0.003; F200W 13-14 +0.006 (from +0.060), 14-15 -0.018 (-0.014), 15-16 -0.026 (-0.023), 16-19 unchanged. The SW floor shifts are 0.001-0.007 and the F200W 13-14 bin (N = 8) improves.

Reference states are not met in every SW bin by any variant (F150W 14-16 sits at -0.04 for the round-7 cap too); that offset predates this experiment.

## Recommendation (I)

Replace the hard cap by a floored cap with tau = 0.96 (0.95 for F250M alone, 0.97 for F300M alone). The floor limits how far the cap can pull the amplitude below the H+h0+bgfree fit, which matches the observation that core pixels near full well under-read by 10-15% while the wings are consistent with the g0 data. Applying it to LW only leaves SW untouched. Remaining F250M/F300M deviations (0.02-0.024) sit at 12.3-13 and 13-13.5, where N = 19-28 stars per bin and the MAD is 0.04-0.07.

## Method notes

- Pipeline cap reproduced exactly (`capfun.py`, `cb_lib.py`; F150W cap and dm match score7).
- Group-0 fit: PSF amplitude plus constant on R x g0_eff, pixels with g0 < frac x ceiling, not flagged, inside the fit region and outside other sources' saturated cores; weights 1/(sigma_bg^2 + kg max(sb, 0)).
- Wing analysis keeps 0.5 < q < 2 to reject junk pixels (g0 > 800 with cal << R_h g0). A first unfiltered run is kept in `v0_unfiltered/`.
- (iii-a) uses a_g10 as the reference amplitude when forming data/model at the binding pixel.
- Caveats: small N at 12.3-13 (19-28 stars); the 8 frames per band share stars; 4 binding rows per band are direct g0 pixels and are mixed into the replaced-pixel statistics.

## Files

Scripts: `stage1.py`, `stage2.py`, `mapping.py`, `capfun.py`, `cb_lib.py`, `an1.py`, `an2.py`, `an3.py`, `an3b.py`, `an4.py`, `an4b.py`, `figs.py`, `run1.sh`, `run2.sh`. Logs: `log1_*.txt`, `log2_*.txt`, `map_*.log`, `an4.log`. Tables: listed at top. Large intermediates: `s1_*.pkl`, `s2_*.pkl`, `map_*.pkl`, `an1_recs.pkl`, `an2_units.pkl`, `an3_cal.pkl`, `an4_*.pkl`, `v0_unfiltered/`.
