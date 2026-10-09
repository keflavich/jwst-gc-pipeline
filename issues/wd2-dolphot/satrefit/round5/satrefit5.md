# Round 5: R-curve exclusion distance as a fix for the satstar R level offset (arm main2)

All work is read-only on the pipeline and on earlier outputs. New files: `run_frames5.py`, `curve_tab5.py`, `score5.py`, `plot5.py`, `out5/` (curves5.md/pkl, per-frame satrefit5.fits, pix5_*.npz, tables5_<band>.md, score5_<band>.pkl, logs), `satrefit5.png`, this file.

## Method

Pipeline R curve (`zeroframe_recover_saturated`): good = ~sat & finite(data, g0) & data > 0 & g0 > 2000 & g0 < ceiling (0.9 x p99 of g0 at SATURATED), 8 log bins, >= 20 px per bin, `rcurve_maxstep` 1.3 guard, satcheck rebuild, <50 px fallback. The reproduction at N=0 follows that code path. N > 0 adds `edt >= N`, with edt the distance in px to the nearest DQ SATURATED pixel. N in {0, 2, 3, 5, 8, 12, 25} plus N=40 as an extra check. Frames: exposure 1 of all 8 SW detectors in F150W and F200W, exposures 1-4 of nrcb1/nrcb3 (the round-3 refit frames), and F250M/F300M nrcalong and nrcblong exposure 1 (nrcblong exposures 1-4 for the refit). Full per-frame table (pixel count, bins raw -> kept, R(2440), ratio to wingmig R and header R, flags): `out5/curves5.md`.

Refit: rim pixels are rewritten with R_N (same g0, same pixel set, new curve; the rim-error rule is unchanged), the recovered-core cap is recomputed from the rewritten cutout with the pipeline cap functions, and the fit is redone for base, bgfree, +cap and bgfree+cap. Validation (measured): rim rewrite with the pipeline curve reproduces the stored rim to 1.0000; the error rule check gives 1.0; the cap computed from the base cutout reproduces the catalogue cap within 1 % for 3666 of 3666 capped rows in F150W (1546/1546 in F250M, 1640/1640 in F300M), and no uncapped row would be capped by the recomputed cap. Caps of the variants carry the per-row factor a_raw/cap_base from the capped rows. dm and the per-pixel ratio follow the round-4 definitions.

## Item 1: R(2440 DN) versus exclusion distance (measured)

R(2440)/wingmig R, median over frames (exposures 1-4 for nrcb1/nrcb3/nrcblong):

| band, det | N=0 | 2 | 3 | 5 | 8 | 12 | 25 | 40 | R/header at N=0 -> 25 |
|---|---|---|---|---|---|---|---|---|---|
| F150W nrcb1 | 1.027 | 1.018 | 1.013 | 1.008 | 1.003 | 1.001 | 1.001 | 1.002 | 1.015 -> 0.990 |
| F150W nrcb3 | 1.076 | 1.065 | 1.057 | 1.044 | 1.022 | 1.012 | 0.999 | 0.992 | 1.075 -> 0.999 |
| F200W nrcb1 | 1.027 | 1.017 | 1.012 | 1.006 | 1.004 | 1.003 | 1.000 | 1.004 | 1.013 -> 0.987 |
| F200W nrcb3 | 1.067 | 1.058 | 1.051 | 1.041 | 1.020 | 1.010 | 1.000 | 0.993 | 1.070 -> 1.003 |
| F250M nrcblong | 1.067 | 1.058 | 1.053 | 1.047 | 1.036 | 1.028 | 1.003 | 0.981 | 0.993 -> 0.943 (exp 1) |
| F300M nrcblong | 1.050 | 1.039 | 1.032 | 1.026 | 1.019 | 1.012 | 1.003 | 0.997 | 1.000 -> 0.960 (exp 1) |
| F150W nrcb2 / nrcb4 (exp 1) | 1.057 / 1.112 | 1.033 / 1.068 | 1.019 / 1.050 | 1.004 / 1.033 | 0.999 / 1.016 | 0.995 / 1.005 | 0.998 / 1.003 | 0.999 / 0.995 | |

Share of the N=0 good set within 3 / 5 / 12 px of SATURATED: nrcb3 0.59-0.63 / 0.75-0.76 / 0.88-0.89, nrcb1 0.41-0.50 / 0.49-0.58 / 0.60-0.67, nrcb2 0.29-0.37 / 0.34-0.47, nrcb4 0.38-0.40 / 0.46-0.50 (per-detector values in the last column of the `curves5.md` summary table). The cal/g0 of N=0 good pixels at g0 2000-3000 DN falls with distance on nrcb3 (F150W: 0.0955 at d 1-2 px, 0.0938 at 3-5, 0.0915 at 8-12, 0.0873 beyond 25) and on nrcb1 (0.0919, 0.0902, 0.0886, 0.0878). The hypothesis that near-core pixels inflate the pipeline R is supported; the effect is larger on nrcb3 because more of its good set lies close to saturated cores.

N*: nrcb1 stops changing (successive steps < 0.3 %) from N=8 (F150W) and N=8-12 (F200W). nrcb3 keeps falling: 1.012 (N=12), 0.999 (25), 0.992 (40) in F150W, and 1.010, 1.000, 0.993 in F200W. No plateau appears on nrcb3 or nrcblong within N <= 40. I adopt N* = 25 as a pooled choice: it is the smallest tested N at which nrcb3 and nrcb1 both sit within 0.3 % of wingmig R. The wingmig curve uses edt >= 25, so agreement at N=25 follows partly from the same selection. N=5 is reported for comparison.

Pixel-count margins at N=25 (refit frames): F150W nrcb1 447-529, nrcb3 463-493; F200W nrcb1 527-566, nrcb3 515-596; F250M nrcblong 898-1062; F300M nrcblong 897-1048. Minimum over all surveyed frames 317 (F150W nrca2), median 527. No refit frame falls below 50 px at any N <= 25. At N=40 the refit-frame counts are 257-318 (SW) and 351-550 (LW), and three nrcblong frames (F250M exp 1 and 4, F300M exp 4) truncate to one bin.

Flags: all nrca frames (F150W, F200W, F250M, F300M) hit guard truncation to one bin and several trigger a satcheck rebuild at N >= 5; the R/wingmig ratios there (20-120) show the wingmig curve is not usable on those sparse frames. nrcb2/nrcb4 exposure 1 truncate to one bin at N=8-25 in F150W (`curves5.md` flags list). F250M exposure 4 at N=40 gives R/wingmig 0.61, an outlier I did not investigate.

## Item 2: refit with R_N (measured)

dm = ours - reference - zp, per star, median in magnitude bins. Unsaturated-reference dm: F150W +0.004, F200W -0.017, F250M +0.003, F300M -0.003. Full tables per band and detector (bins, MAD, trend, amplitude and cap ratios): `out5/tables5_<band>.md`.

Overall median dm (all stars in the refit frames). "final" is the delivered catalogue; rw12h_e0 variants are from round 4.

| band, det | final | rw12h_e0+cap | rw12h_e0+bgfree+cap | R_N25+cap | R_N25+bgfree+cap | R_N5+cap | R_N25+h0+cap | R_N25+h0+bgfree+cap |
|---|---|---|---|---|---|---|---|---|
| F150W all | -0.047 | -0.044 | -0.037 | -0.002 | +0.004 | -0.025 | +0.001 | +0.008 |
| F150W nrcb1 | -0.017 | -0.012 | -0.008 | +0.004 | +0.006 | -0.002 | +0.008 | +0.011 |
| F150W nrcb3 | -0.075 | -0.071 | -0.059 | -0.006 | +0.002 | -0.044 | -0.004 | +0.006 |
| F200W all | -0.066 | -0.056 | -0.045 | -0.018 | -0.013 | -0.041 | -0.013 | -0.005 |
| F200W nrcb1 | -0.021 | -0.016 | -0.009 | -0.000 | +0.001 | -0.006 | +0.005 | +0.010 |
| F200W nrcb3 | -0.093 | -0.086 | -0.074 | -0.034 | -0.028 | -0.066 | -0.029 | -0.020 |
| F250M nrcblong | +0.030 | +0.032 | +0.044 | +0.132 | +0.139 | +0.047 | +0.133 | +0.142 |
| F300M nrcblong | +0.018 | +0.019 | +0.030 | +0.070 | +0.079 | +0.036 | +0.072 | +0.082 |

Uncapped R_N25 (no cap) gives F150W all -0.035, nrcb1 -0.044, nrcb3 -0.028 (final uncapped: -0.078, -0.060, -0.087); the cap contributes 0.02-0.05 mag of the correction on top of the rim change. Per magnitude bin (F150W nrcb3, final -> R_N25+cap): 14-15 -0.107 -> -0.040, 15-16 -0.105 -> -0.033, 16-17 -0.085 -> -0.016, 17-18 -0.067 -> +0.003, 18-19 -0.055 -> +0.010. nrcb1: -0.035 -> -0.012, -0.044 -> -0.024, -0.028 -> -0.007, -0.013 -> +0.008, +0.004 -> +0.025. The nrcb3 minus nrcb1 offset in F150W falls from -0.058 (final) to -0.010 (R_N25+cap), -0.004 (R_N25+bgfree+cap), -0.012 (+h0+cap) and -0.005 (+h0+bgfree+cap). In F200W the offset falls from -0.072 to -0.034 (R_N25+cap), -0.029 (+bgfree), -0.034 (+h0), -0.030 (+h0+bgfree). With R_N5+cap the F150W offset is -0.042 and the F200W offset -0.060. The bright-to-faint trend (F150W nrcb1) remains +0.037 to +0.043 under all R_N variants; the R change moves the level and leaves the trend. At the faint end (18-19 mag) R_N25+cap on nrcb1 sits +0.025 above 0 and +0.021 above the unsaturated reference, an overshoot of order 0.02 mag.

Amplitude and cap ratios to base (all stars): F150W a_RN25/a_base 0.987 (nrcb1), 0.951 (nrcb3); cap_RN25/cap_base 0.979 and 0.932; F200W 0.986 / 0.955 and 0.980 / 0.945. In the LW the ratios are 0.922 (F250M) and 0.955 (F300M) with caps 0.901 and 0.952.

LW (F250M, F300M, nrcblong): R_N25 moves dm away from the unsaturated reference. R_N25+cap gives +0.132 (F250M) and +0.070 (F300M), against +0.030 and +0.018 for final; R_N5+cap gives +0.047 and +0.036. On nrcblong the N=0 pipeline R already agrees with header R (0.993 F250M, 1.000 F300M); R_N25 is 4-6 % below header R (0.943, 0.960). The wingmig curve itself lies below header R on these frames (0.0574 vs 0.0620 F250M; 0.0344 vs 0.0360 F300M).

## Item 3: per-pixel rim data/model ratio (measured)

Rim pixels (catalogue cat 0, in the fit), median of u/F_dol, r < 6 px of the fitted position:

| set | pipeline N=0 (nrcb1 / nrcb3 / ratio) | R_N25 | R_N5 |
|---|---|---|---|
| F150W all rim | 1.028 / 1.084 / 1.055 | 1.004 / 1.006 / 1.002 | 1.009 / 1.049 / 1.040 |
| F150W peak pixel | 1.016 / 1.079 / 1.062 | 0.997 / 1.006 / 1.010 | 1.002 / 1.044 / 1.041 |
| F200W all rim | 1.042 / 1.106 / 1.061 | 1.017 / 1.037 / 1.020 | 1.023 / 1.074 / 1.050 |
| F200W peak pixel | 1.022 / 1.096 / 1.072 | 1.002 / 1.036 / 1.034 | 1.008 / 1.066 / 1.058 |

In F150W the rim ratio moves to 1.00 on both detectors and the nrcb3/nrcb1 ratio to 1.002. By g0 the F150W nrcb3/nrcb1 ratio falls from 1.044-1.063 to 0.991-1.008 in every bin. In F200W nrcb3 keeps a 2-4 % excess under R_N25 (1.037 vs 1.017), larger at g0 < 800 DN (1.06-1.09) and > 6400 DN (1.03). The per-pixel ratio therefore moves to about 1 in F150W and to within 2-4 % in F200W. Pixels never rewritten (crf, r < 6 px, outside the rim set) show data/model 1.176 (F150W) and 1.252-1.271 (F200W) on both detectors, unchanged by the rim rewrite and equal between detectors within 0.2-1.5 %.

## Caveats

- Measured: R at 2440 DN, the dm tables and per-pixel ratios above. Inferred: that the cal/g0 gradient with distance comes from saturated-neighbour inflation; the 10 % gradient in nrcblong (F250M 0.0662 at d 1-2 px to 0.0576 beyond 25) could come from other effects, and the lack of a plateau at N <= 40 in nrcb3 and nrcblong leaves the right N open.
- N=25 agrees with wingmig partly by construction (same edt cut). Header R is 1-2 % above R_N25 on nrcb1 and within 0.3 % on nrcb3.
- The cap shift (0.02-0.05 mag) arises from lowering the recovered rim by 2-7 %; the cap-only variants (R_N5, R_N25) change dm more than the round-4 wing rewrites did. Because dm for variants adds the median per-star amplitude shift to dm_unc, the variants inherit round-4 scoring conventions, including a factor a_raw/cap_base applied to recomputed caps.
- LW result contradicts the SW result: applying the SW-derived N=25 to nrcblong worsens dm by 0.04-0.10 mag. The fix as tested applies to SW nrcb1/nrcb3 only; LW needs its own criterion. I did not test N between 5 and 25 in the refit.
- Residual: F200W nrcb3 keeps -0.028 to -0.034 under R_N25 variants; nrcb1 faint end overshoots by about 0.02 mag.
- Sparse nrca frames and F250M exposure 4 at N=40 behave erratically; they are outside the refit.

Figure: `satrefit5.png`: (a) R/wingmig versus N, (b-d) dm by magnitude for F150W nrcb1/nrcb3 and F200W nrcb3, (e-f) LW, (g) rim data/model nrcb3 over nrcb1 versus g0, (h) rim r < 6 px medians.
