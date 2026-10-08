# satrefit round 4: high-g0 extension of the charge-migration rewrite, norim variant, per-pixel data/model ratio

Arm main2, masked-core PSF fits of `get_saturated_stars` (commit 5434f5e7), read-only analysis. Frames: F150W and F200W nrcb1 + nrcb3, F250M and F300M nrcblong, exposures 1-4. Scripts: `run_frames4.py`, `run_frames4b.py`, `score4.py`, `score4b.py`, `q4_collect.py`, `q4_analyze.py`, `curve_cmp4.py`, `plot4.py`. Outputs: `out4/` (per-frame fits `*_satrefit4.fits`, `*_satrefit4b.fits`, pixel records `pix_*.npz`, curves `*_pcurve.txt`, tables `tables4_<band>.md`, `tables4b_<band>.md`, `q_lw.md`, `pcurve_compare.md`, pickles). Figure: `satrefit4.png`. Nothing in out3/ or in round 1-3 files was changed.

dm = our - dolphot - ZP (negative = ours brighter). A variant's dm = dm_unc + per-star median shift of -2.5 log10(a_v/a_base); "+cap" applies min(a_v, cap), the pipeline cap. "final" is the pipeline's shipped value (with cap). MAD values in parentheses.

## 1. Summary of measured results

### 1.1 High-g0 extension (variants rw12h, rw12p, e0 versus e1)

- rw12h_e1 and rw12p_e1 reproduce round-3 rw12 to 0.001 mag or better in every band and detector (all-star medians identical to three digits in F150W, F250M, F300M; F200W within 0.001 in single bins). The extension to the pipeline ceiling and the pipeline-curve version above 2000 DN change no measurable amplitude.
- Pixels newly rewritten by the extension (g0 above the top wingmig bin): 1655 of 39.6 million fit pixels in F150W, 2047 of 16.0 million in F200W, 3090 of 23.5 million in F250M, 2266 of 24.7 million in F300M (fractions 0.00004-0.00013). Their q (cal / (R g0), R held flat from the top bin): pixel-weighted mean 0.946 (F200W), 0.683 (F250M), 0.525 (F300M); per-star median of star medians 1.017 (F150W), 1.026 (F200W), 0.79 (F250M), 0.52 (F300M). Round-3 rw12 pixels of the same stars have q 1.055 (F150W), 1.066 (F200W), 1.03 (F250M), 1.04 (F300M). In the SW bands the flat extension sits 3-4 % high relative to these pixels; in the LW bands the flat extension overshoots by a factor of order 2.
- rw12h_e0 (crf ERR kept) moves the amplitude about half as far as e1 (all-star a/a_base 0.990 versus 0.981 in F150W, 0.998 versus 0.988 in F250M, 0.997 versus 0.992 in F300M) and gives a more negative dm in the SW bands (F150W -0.069 versus -0.058) and a smaller positive dm in the LW bands (F250M +0.018 versus +0.029).

### 1.2 All-star median dm (MAD) and per detector

| band / group | final | rw12 (round 3) | rw12h_e1 | rw12h_e0 | rw12p_e1 | rw12h_e1+cap | rw12h_e0+cap | norim+cap |
|---|---|---|---|---|---|---|---|---|
| F150W all | -0.047 (0.047) | -0.058 (0.052) | -0.058 (0.052) | -0.069 (0.053) | -0.058 (0.052) | -0.040 (0.050) | -0.044 (0.048) | -0.048 (0.058) |
| F150W nrcb1 | -0.017 (0.037) | -0.030 (0.053) | -0.030 (0.053) | -0.042 (0.063) | -0.030 (0.053) | -0.007 (0.038) | -0.012 (0.040) | -0.023 (0.038) |
| F150W nrcb3 | -0.075 (0.032) | -0.076 (0.040) | -0.076 (0.040) | -0.081 (0.042) | -0.076 (0.040) | -0.067 (0.033) | -0.071 (0.032) | -0.082 (0.054) |
| F200W all | -0.066 (0.054) | -0.073 (0.055) | -0.073 (0.055) | -0.075 (0.051) | -0.073 (0.055) | -0.055 (0.053) | -0.056 (0.053) | -0.076 (0.067) |
| F200W nrcb1 | -0.021 (0.025) | -0.035 (0.030) | -0.035 (0.029) | -0.043 (0.035) | -0.035 (0.029) | -0.014 (0.025) | -0.016 (0.028) | -0.024 (0.028) |
| F200W nrcb3 | -0.093 (0.025) | -0.096 (0.036) | -0.096 (0.036) | -0.095 (0.039) | -0.097 (0.036) | -0.085 (0.023) | -0.086 (0.026) | -0.104 (0.033) |
| F250M all (nrcblong) | +0.030 (0.035) | +0.029 (0.037) | +0.029 (0.037) | +0.018 (0.041) | +0.029 (0.037) | +0.039 (0.037) | +0.032 (0.036) | see tables |
| F300M all (nrcblong) | +0.018 (0.035) | +0.013 (0.035) | +0.013 (0.035) | +0.009 (0.039) | +0.013 (0.035) | +0.022 (0.034) | +0.019 (0.035) | see tables |

The LW arm has one detector (nrcblong), so no detector split exists there. The per-1-mag-bin tables, trends, rewritten fractions and amplitude ratios for every variant (base, bgfree, +cap, bgfree+cap) are in section 6 (all stars and, for SW, per detector).

- Detector offset (measured): in F150W the final catalogue differs by 0.058 mag between nrcb1 (-0.017) and nrcb3 (-0.075); in F200W by 0.072 (-0.021 versus -0.093). The rewrite variants move nrcb1 toward brighter values (rw12 -0.030 / -0.035) and nrcb3 by 0.001-0.003 mag, so the rewrite widens the gap to 0.046 (F150W) and 0.061 (F200W) only through nrcb1's shift. Amplitude shift of the rewrite (a_rw12h/a_base, all-star median, uncapped): nrcb1 0.973 (F150W), 0.967 (F200W); nrcb3 0.987 (F150W), 0.985 (F200W). The rewritten fraction of fit pixels is larger on nrcb3 (F150W 0.019 versus 0.009; F200W 0.113 versus 0.041).
- The +cap versions of rw12h_e1 reach -0.007 (F150W nrcb1) and -0.067 (F150W nrcb3), -0.014 and -0.085 in F200W.

## 2. Pipeline ZEROFRAME R curve versus the wingmig curve

Table in section 7 (`out4/pcurve_compare.md`). The pipeline measured path (R_g0_min 2000, ceiling 0.9 x p99 of g0 at SATURATED, guard SATSTAR_ZF_RCURVE_MAXSTEP 1.3) measures 4 bins (6 in F300M) per frame and keeps 2 after the guard, so R is defined at g0 ~2440 and ~3630 DN and held flat beyond. The reproduction matches the in-memory rim pixels (rim check 1.0000 on every frame).

- At the first kept bin (g0 ~2440 DN), pipeline R divided by wingmig R: F150W nrcb1 1.027 (range 1.026-1.032), nrcb3 1.075 (1.074-1.082); F200W nrcb1 1.027, nrcb3 1.067; F250M 1.066 (1.057-1.091); F300M 1.049. At the second kept bin (g0 ~3630 DN) the ratio is 1.00-1.03 on nrcb1 and 1.04-1.08 on nrcb3.
- R values: F150W nrcb1 0.0902 versus 0.0880; nrcb3 0.0939 versus 0.0874 (exposure 1). The wingmig curve is nearly flat between 400 and 2500 DN (0.087-0.090), so the pipeline's higher R is a level offset of 2.7 % (nrcb1) and 7.5 % (nrcb3) at g0 ~2400 DN.
- The satcheck ratio (curve versus ramp-fit rate at SATURATED pixels) is 0.99-1.00 on nrcb1, 1.011-1.017 on nrcb3 (SW) and 0.95 in F300M.
- Effect on the fit (measured): rw12p (pipeline R above 2000 DN, wingmig below) gives the same dm as rw12h to 0.001 mag. Inference: pixels with g0 > 2000 DN carry large propagated errors in e1 (s_flat R g0 term), so the amplitude weights them little. rw12p was run with e1 errors only.

## 3. norim variant (rim pixels excluded from the amplitude fit)

Measured (tables in section 6 and `tables4b_<band>.md`):

- Rim (ZEROFRAME-rewritten) pixels are a small share of the fit region: median 1.0 % of fit pixels in F150W (nrcb1 0.8 %, nrcb3 1.2 %), 2.1 % in F200W (1.6 %, 2.9 %). norim fits about 6400-6500 pixels per star.
- norim raises the amplitude: median a_norim/a_base = 1.067 (F150W nrcb1), 1.050 (F150W nrcb3), 1.078 (F200W nrcb1), 1.071 (F200W nrcb3); with the bgfree variant 1.046, 1.009, 1.052, 1.024. The detector difference of the amplitude response to rim exclusion is 0.016 (F150W, base) and 0.037 (F150W, bgfree), 0.007 (F200W, base) and 0.027 (F200W, bgfree).
- norim all-star dm: F150W -0.142 (MAD 0.123) uncapped, -0.048 (0.058) with cap; F200W -0.187 (0.095) uncapped, -0.076 (0.067) with cap. The scatter grows by a factor 2.4 (F150W) and 1.8 (F200W) relative to base, so norim is a noisier amplitude estimate. By detector with cap: F150W nrcb1 -0.023 (0.038), nrcb3 -0.082 (0.054); F200W nrcb1 -0.024 (0.028), nrcb3 -0.104 (0.033). The nrcb1 to nrcb3 difference with norim+cap is 0.059 (F150W) and 0.080 (F200W), the same size as in final (0.058, 0.072).
- Interpretation (inferred): excluding the rim removes the pixels carrying the ZEROFRAME R, and the nrcb1 to nrcb3 difference stays at 0.06-0.08 mag. The detector offset then sits in the never-rewritten crf pixels, in the PSF and aperture-level comparison with dolphot, or in the stars' own cap and masking behaviour, and the R(g0) level of the rewrite is a smaller contributor. The norim scatter is too large to separate a 0.01 mag effect.
- By magnitude (norim+cap, F150W): nrcb1 -0.036 / -0.048 / -0.033 / -0.016 / +0.010 at 14-15 ... 18-19; nrcb3 -0.145 / -0.129 / -0.104 / -0.070 / -0.031.

## 4. Per-pixel data/model ratio by detector

Model = F_dolphot x PSF (satstar PSF grid at the pipeline fitted position); F_dolphot per star = median over its rows of flux_fit x 10^(0.4 dm_final). Data = neighbour-subtracted cutout minus the annulus background. Pixels within 30 px of the star with PSF >= 1e-3 of peak. Full g0 and r tables, with N, in section 6 (`tables4b_150W.md`, `tables4b_200W.md`). Matched stars: 1471 (F150W), 515 (F200W).

Measured, rim pixels in the fit (after rewrite), r < 6 px region dominated by the core:

| band | g0 bin (DN) | nrcb1 | nrcb3 | nrcb3/nrcb1 |
|---|---|---|---|---|
| F150W | 400-800 | 1.012 | 1.066 | 1.053 |
| F150W | 800-1600 | 1.056 | 1.102 | 1.044 |
| F150W | 1600-3200 | 1.057 | 1.109 | 1.049 |
| F150W | 3200-6400 | 1.027 | 1.086 | 1.058 |
| F150W | >12800 | 1.034 | 1.097 | 1.062 |
| F200W | 800-1600 | 1.062 | 1.117 | 1.051 |
| F200W | 1600-3200 | 1.094 | 1.136 | 1.039 |
| F200W | 3200-6400 | 1.095 | 1.141 | 1.042 |
| F200W | >12800 | 1.027 | 1.100 | 1.071 |

- Rim pixels (all g0 bins pooled by r): nrcb3/nrcb1 = 1.059 (r 0-2), 1.046 (2-3), 1.021 (3-4), 1.027 (4-6) in F150W; 1.070, 1.042, 1.067, 1.023 in F200W. Brighter and fainter halves of the star sample give 1.051 and 1.061 (F150W), 1.060 and 1.068 (F200W).
- Model peak pixel, rim pixel in the fit: F150W nrcb1 1.016 (N 2584), nrcb3 1.079 (N 2755), ratio 1.062; F200W nrcb1 1.022 (N 807), nrcb3 1.096 (N 976), ratio 1.072. Peak pixels that are rim pixels masked from the fit (cat 2) or never rewritten number 0-8 per detector, too few for a median.
- Never-rewritten crf pixels in the fit: nrcb3/nrcb1 = 1.015, 0.991, 0.970 in the first three g0 bins in F150W (200-1600 DN), 1.014, 1.033, 1.031 in F200W; by r: 0.99-1.04 in F150W, 1.02-1.05 in F200W (F200W r 3-14). The brighter and fainter halves give 1.021 and 1.036 (F150W), 1.035 and 1.029 (F200W). The absolute level of this ratio is 1.0-1.3 (r-dependent: 1.03-1.08 at r < 3, 1.15-1.29 at r 3-9, 0.99-1.04 at r 9-14 in F150W), which includes any flux-scale mismatch between the satstar PSF normalization and dolphot's flux, so the detector ratio carries the information.
- Rim pixels at r > 6 px give ratios of 5-12 with large scatter and no consistent detector ratio (0.6-1.7). They are core-adjacent pixels with PSF at 1e-3 of peak where an additive offset dominates the ratio.

Reading (inferred): after the ZEROFRAME rewrite, rim-pixel data exceed the dolphot-based model by 3-10 % on nrcb1 and by 8-14 % on nrcb3, with nrcb3/nrcb1 = 1.04-1.07 in nearly every g0 and r bin. Never-rewritten pixels show a detector ratio of 1.00-1.03. A ZEROFRAME R that sits about 2.7 % (nrcb1) to 7.5 % (nrcb3) above the wingmig curve at g0 ~2400 DN (section 2) would produce a rim-pixel offset of the observed sign and a nrcb3/nrcb1 ratio of about 1.045 at that g0; the measured 1.04-1.07 over g0 200-25000 DN (where the pipeline curve is held flat at the 2400-3600 DN value) is consistent in size with that. The data do not isolate other contributions to the nrcb3 rim excess, for example a detector-dependent group-0 gain or the dolphot flux scale.

## 5. LW q = cal / (R(g0) g0) versus distance d from the DQ-SATURATED edge (nrcblong)

Full tables (all, flux quartiles, control stars at r = 3-9 px) in `out4/q_lw.md` (section 8). Method as wingmig: R(g0) from pixels at least 25 px from saturation (200 DN to the top bin), q in distance bins d = 1, 2, 3, 4-5, 6-8, 9-12, 13-20 px.

| group | d=1 | d=2 | d=3 | d=4-5 | d=6-8 | d=9-12 | d=13-20 |
|---|---|---|---|---|---|---|---|
| F250M all | 1.044 | 1.032 | 1.035 | 1.020 | 1.015 | 1.007 | 1.000 |
| F250M brightest quartile | 1.152 | 1.124 | 1.102 | 1.070 | 1.043 | 1.038 | 1.028 |
| F250M faintest quartile | 1.026 | 0.976 | 0.995 | 0.985 | 0.984 | 0.975 | 0.975 |
| F300M all | 1.059 | 1.046 | 1.034 | 1.017 | 1.007 | 0.998 | 0.993 |
| F300M brightest quartile | 1.133 | 1.111 | 1.098 | 1.087 | 1.059 | 1.047 | 1.027 |
| F300M faintest quartile | 1.033 | 1.019 | 0.982 | 0.954 | 0.965 | 0.970 | 0.978 |

Control (unsaturated, peak g0 > 1500 DN) q at r 1-2 / 2-3 / 3-4 / 4-6 / 6-9 px: F250M 1.048 / 1.009 / 0.992 / 1.001 / 0.981. The brightest-quartile excess (q up to 1.13-1.15 at d = 1) is the same sign and a similar size to the SW rewritten pixels (q ~1.05-1.07 for the typical star). The LW q dependence on stellar flux indicates that R depends on the pixel's neighbourhood flux, and a single R(g0) curve cannot remove it (measured pattern; the mechanism is inferred).

## 6. Per-band tables (all stars, per detector, per 1-mag bin)

