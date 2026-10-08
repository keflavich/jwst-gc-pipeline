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

- Detector offset (measured): in F150W the final catalogue differs by 0.058 mag between nrcb1 (-0.017) and nrcb3 (-0.075); in F200W by 0.072 (-0.021 versus -0.093). The rewrite variants move nrcb1 toward brighter values (rw12 -0.030 / -0.035) and nrcb3 by 0.001-0.003 mag, so the rewrite narrows the gap to 0.046 (F150W) and 0.061 (F200W) through nrcb1's shift alone. Amplitude shift of the rewrite (a_rw12h/a_base, all-star median, uncapped): nrcb1 0.973 (F150W), 0.967 (F200W); nrcb3 0.987 (F150W), 0.985 (F200W). The rewritten fraction of fit pixels is larger on nrcb3 (F150W 0.019 versus 0.009; F200W 0.113 versus 0.041).
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
- norim raises the amplitude: median a_norim/a_base = 1.067 (F150W nrcb1), 1.050 (F150W nrcb3), 1.078 (F200W nrcb1), 1.071 (F200W nrcb3); with the bgfree variant 1.046, 1.009, 1.052, 1.024. The detector difference of the amplitude response to rim exclusion is 0.016 (F150W, base) and 0.037 (F150W, bgfree), 0.006 (F200W, base) and 0.027 (F200W, bgfree).
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


### F150W (score4: final, round-3 rw12, rw12h_e1, rw12h_e0, rw12p_e1; each base, +cap, +bgfree, +bgfree+cap; rewritten fraction; amplitude ratio; newly rewritten pixels)

#### F150W all: 1471 stars (unsaturated reference 18.46-19.46 mag: dm +0.004)

| variant | 14-15 | 15-16 | 16-17 | 17-18 | 18-19 | all med / MAD | trend |
|---|---|---|---|---|---|---|---|
| final | -0.095 (0.070) | -0.084 (0.048) | -0.057 (0.045) | -0.040 (0.044) | -0.032 (0.047) | -0.047 / 0.047 | +0.063 |
| uncapped | -0.142 (0.063) | -0.122 (0.055) | -0.101 (0.046) | -0.066 (0.045) | -0.047 (0.039) | -0.078 / 0.053 | +0.095 |
| bgfree | -0.117 (0.058) | -0.099 (0.053) | -0.083 (0.046) | -0.054 (0.043) | -0.038 (0.039) | -0.064 / 0.048 | +0.079 |
| bgfree+v7b+cap | -0.021 (0.030) | -0.035 (0.038) | -0.024 (0.033) | -0.009 (0.041) | -0.002 (0.041) | -0.015 / 0.041 | +0.018 |
| rw12 | -0.095 (0.048) | -0.092 (0.046) | -0.078 (0.047) | -0.045 (0.051) | -0.030 (0.047) | -0.058 / 0.052 | +0.066 |
| rw12+cap | -0.071 (0.052) | -0.069 (0.044) | -0.048 (0.044) | -0.030 (0.047) | -0.025 (0.051) | -0.040 / 0.050 | +0.046 |
| rw12+bgfree | -0.070 (0.032) | -0.074 (0.044) | -0.063 (0.043) | -0.034 (0.048) | -0.022 (0.045) | -0.047 / 0.050 | +0.048 |
| rw12+bgfree+cap | -0.051 (0.041) | -0.058 (0.044) | -0.043 (0.037) | -0.023 (0.046) | -0.020 (0.048) | -0.034 / 0.047 | +0.031 |
| rw12h_e1 | -0.095 (0.048) | -0.092 (0.046) | -0.078 (0.047) | -0.045 (0.050) | -0.030 (0.047) | -0.058 / 0.052 | +0.066 |
| rw12h_e1+cap | -0.071 (0.052) | -0.069 (0.044) | -0.048 (0.044) | -0.030 (0.047) | -0.025 (0.051) | -0.040 / 0.050 | +0.046 |
| rw12h_e1+bgfree | -0.070 (0.032) | -0.074 (0.044) | -0.063 (0.043) | -0.034 (0.048) | -0.022 (0.045) | -0.047 / 0.050 | +0.048 |
| rw12h_e1+bgfree+cap | -0.051 (0.041) | -0.058 (0.043) | -0.043 (0.037) | -0.023 (0.046) | -0.020 (0.048) | -0.034 / 0.047 | +0.031 |
| rw12h_e0 | -0.114 (0.054) | -0.106 (0.055) | -0.084 (0.049) | -0.057 (0.048) | -0.043 (0.046) | -0.069 / 0.053 | +0.071 |
| rw12h_e0+cap | -0.073 (0.056) | -0.080 (0.049) | -0.051 (0.044) | -0.036 (0.047) | -0.029 (0.049) | -0.044 / 0.048 | +0.043 |
| rw12h_e0+bgfree | -0.087 (0.043) | -0.080 (0.051) | -0.068 (0.049) | -0.044 (0.045) | -0.034 (0.043) | -0.054 / 0.049 | +0.053 |
| rw12h_e0+bgfree+cap | -0.056 (0.048) | -0.062 (0.047) | -0.045 (0.040) | -0.029 (0.044) | -0.024 (0.048) | -0.037 / 0.045 | +0.032 |
| rw12p_e1 | -0.095 (0.048) | -0.092 (0.046) | -0.078 (0.047) | -0.045 (0.050) | -0.030 (0.047) | -0.058 / 0.052 | +0.066 |
| rw12p_e1+cap | -0.071 (0.052) | -0.069 (0.044) | -0.048 (0.044) | -0.030 (0.047) | -0.025 (0.051) | -0.040 / 0.050 | +0.046 |
| rw12p_e1+bgfree | -0.070 (0.032) | -0.074 (0.044) | -0.063 (0.043) | -0.034 (0.048) | -0.022 (0.045) | -0.047 / 0.050 | +0.048 |
| rw12p_e1+bgfree+cap | -0.051 (0.041) | -0.058 (0.043) | -0.043 (0.037) | -0.023 (0.046) | -0.020 (0.048) | -0.034 / 0.047 | +0.031 |

N per bin: 57, 159, 385, 562, 307

Rewritten fraction of fit pixels (median over stars) and amplitude ratio a_v/a_base (uncapped), F150W all:

| variant | quantity | 14-15 | 15-16 | 16-17 | 17-18 | 18-19 | all |
|---|---|---|---|---|---|---|---|
| rw12 | rewritten frac | 0.0443 | 0.0314 | 0.0151 | 0.0102 | 0.0068 | 0.0127 |
| rw12 | a_v/a_base | 0.9578 | 0.9688 | 0.9795 | 0.9829 | 0.9878 | 0.9815 |
| rw12 | a_v/a_base bgfree | 0.9339 | 0.9510 | 0.9658 | 0.9733 | 0.9798 | 0.9704 |
| rw12h_e1 | rewritten frac | 0.0443 | 0.0314 | 0.0151 | 0.0102 | 0.0068 | 0.0128 |
| rw12h_e1 | a_v/a_base | 0.9578 | 0.9688 | 0.9795 | 0.9829 | 0.9878 | 0.9814 |
| rw12h_e1 | a_v/a_base bgfree | 0.9341 | 0.9510 | 0.9658 | 0.9733 | 0.9798 | 0.9704 |
| rw12h_e0 | rewritten frac | 0.0443 | 0.0314 | 0.0151 | 0.0102 | 0.0068 | 0.0128 |
| rw12h_e0 | a_v/a_base | 0.9742 | 0.9848 | 0.9853 | 0.9905 | 0.9965 | 0.9899 |
| rw12h_e0 | a_v/a_base bgfree | 0.9503 | 0.9616 | 0.9689 | 0.9796 | 0.9896 | 0.9769 |
| rw12p_e1 | rewritten frac | 0.0443 | 0.0314 | 0.0151 | 0.0102 | 0.0068 | 0.0128 |
| rw12p_e1 | a_v/a_base | 0.9578 | 0.9688 | 0.9795 | 0.9829 | 0.9878 | 0.9814 |
| rw12p_e1 | a_v/a_base bgfree | 0.9341 | 0.9510 | 0.9658 | 0.9733 | 0.9798 | 0.9704 |

Pixels newly rewritten by rw12h (g0 above the top wingmig R bin), F150W all: per-star median of (new pixels / fit pixels), median per-star count, q of the new pixels, q of the round-3 rw12 pixels of the same stars:

| quantity | 14-15 | 15-16 | 16-17 | 17-18 | 18-19 | all |
|---|---|---|---|---|---|---|
| new/fit | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| count | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| q new px (median of star medians; stars with new px) | 1.051 | 1.017 | 1.018 | 1.017 | 0.975 | 1.017 |
| q rw12 px (same stars) | 1.057 | 1.061 | 1.056 | 1.052 | 1.051 | 1.055 |

#### F150W nrcb1: 701 stars (unsaturated reference 18.46-19.46 mag: dm +0.004)

| variant | 14-15 | 15-16 | 16-17 | 17-18 | 18-19 | all med / MAD | trend |
|---|---|---|---|---|---|---|---|
| final | -0.035 (0.042) | -0.044 (0.021) | -0.028 (0.028) | -0.013 (0.033) | +0.004 (0.041) | -0.017 / 0.037 | +0.039 |
| uncapped | -0.149 (0.062) | -0.106 (0.065) | -0.096 (0.056) | -0.047 (0.048) | -0.027 (0.040) | -0.060 / 0.065 | +0.122 |
| bgfree | -0.126 (0.048) | -0.087 (0.059) | -0.082 (0.058) | -0.037 (0.046) | -0.020 (0.039) | -0.048 / 0.061 | +0.107 |
| bgfree+v7b+cap | -0.014 (0.020) | -0.013 (0.040) | -0.008 (0.039) | +0.015 (0.038) | +0.024 (0.039) | +0.009 / 0.042 | +0.038 |
| rw12 | -0.090 (0.055) | -0.065 (0.051) | -0.066 (0.049) | -0.018 (0.038) | -0.004 (0.028) | -0.030 / 0.053 | +0.086 |
| rw12+cap | -0.030 (0.031) | -0.038 (0.028) | -0.023 (0.034) | -0.003 (0.033) | +0.011 (0.032) | -0.007 / 0.038 | +0.041 |
| rw12+bgfree | -0.070 (0.032) | -0.051 (0.050) | -0.051 (0.050) | -0.011 (0.038) | -0.000 (0.028) | -0.018 / 0.050 | +0.070 |
| rw12+bgfree+cap | -0.030 (0.031) | -0.026 (0.034) | -0.021 (0.036) | +0.004 (0.033) | +0.014 (0.029) | -0.002 / 0.039 | +0.043 |
| rw12h_e1 | -0.090 (0.055) | -0.065 (0.051) | -0.066 (0.048) | -0.018 (0.038) | -0.004 (0.028) | -0.030 / 0.053 | +0.086 |
| rw12h_e1+cap | -0.030 (0.031) | -0.038 (0.028) | -0.023 (0.034) | -0.003 (0.033) | +0.011 (0.032) | -0.007 / 0.038 | +0.041 |
| rw12h_e1+bgfree | -0.070 (0.032) | -0.051 (0.050) | -0.051 (0.050) | -0.011 (0.038) | -0.000 (0.028) | -0.018 / 0.050 | +0.070 |
| rw12h_e1+bgfree+cap | -0.030 (0.031) | -0.026 (0.034) | -0.021 (0.036) | +0.004 (0.033) | +0.014 (0.029) | -0.002 / 0.039 | +0.043 |
| rw12h_e0 | -0.114 (0.054) | -0.078 (0.048) | -0.072 (0.059) | -0.028 (0.050) | -0.018 (0.040) | -0.042 / 0.063 | +0.096 |
| rw12h_e0+cap | -0.035 (0.035) | -0.041 (0.027) | -0.023 (0.035) | -0.005 (0.038) | +0.008 (0.039) | -0.012 / 0.040 | +0.044 |
| rw12h_e0+bgfree | -0.082 (0.039) | -0.057 (0.050) | -0.058 (0.058) | -0.018 (0.047) | -0.009 (0.039) | -0.030 / 0.056 | +0.072 |
| rw12h_e0+bgfree+cap | -0.031 (0.029) | -0.031 (0.028) | -0.021 (0.037) | -0.001 (0.037) | +0.009 (0.037) | -0.008 / 0.041 | +0.041 |
| rw12p_e1 | -0.090 (0.055) | -0.065 (0.051) | -0.066 (0.048) | -0.018 (0.038) | -0.004 (0.028) | -0.030 / 0.053 | +0.086 |
| rw12p_e1+cap | -0.030 (0.031) | -0.038 (0.028) | -0.023 (0.034) | -0.003 (0.033) | +0.011 (0.032) | -0.007 / 0.038 | +0.041 |
| rw12p_e1+bgfree | -0.070 (0.032) | -0.051 (0.050) | -0.051 (0.050) | -0.011 (0.038) | -0.000 (0.028) | -0.018 / 0.050 | +0.070 |
| rw12p_e1+bgfree+cap | -0.030 (0.031) | -0.026 (0.034) | -0.021 (0.036) | +0.004 (0.033) | +0.014 (0.029) | -0.002 / 0.039 | +0.043 |

N per bin: 23, 65, 188, 270, 155

Rewritten fraction of fit pixels (median over stars) and amplitude ratio a_v/a_base (uncapped), F150W nrcb1:

| variant | quantity | 14-15 | 15-16 | 16-17 | 17-18 | 18-19 | all |
|---|---|---|---|---|---|---|---|
| rw12 | rewritten frac | 0.0266 | 0.0196 | 0.0112 | 0.0074 | 0.0058 | 0.0089 |
| rw12 | a_v/a_base | 0.9546 | 0.9583 | 0.9718 | 0.9745 | 0.9792 | 0.9731 |
| rw12 | a_v/a_base bgfree | 0.9314 | 0.9420 | 0.9600 | 0.9649 | 0.9749 | 0.9628 |
| rw12h_e1 | rewritten frac | 0.0266 | 0.0196 | 0.0112 | 0.0074 | 0.0058 | 0.0089 |
| rw12h_e1 | a_v/a_base | 0.9546 | 0.9583 | 0.9718 | 0.9745 | 0.9792 | 0.9731 |
| rw12h_e1 | a_v/a_base bgfree | 0.9314 | 0.9420 | 0.9600 | 0.9649 | 0.9749 | 0.9628 |
| rw12h_e0 | rewritten frac | 0.0266 | 0.0196 | 0.0112 | 0.0074 | 0.0058 | 0.0089 |
| rw12h_e0 | a_v/a_base | 0.9628 | 0.9725 | 0.9788 | 0.9829 | 0.9932 | 0.9829 |
| rw12h_e0 | a_v/a_base bgfree | 0.9503 | 0.9529 | 0.9629 | 0.9744 | 0.9877 | 0.9719 |
| rw12p_e1 | rewritten frac | 0.0266 | 0.0196 | 0.0112 | 0.0074 | 0.0058 | 0.0089 |
| rw12p_e1 | a_v/a_base | 0.9546 | 0.9583 | 0.9718 | 0.9745 | 0.9792 | 0.9731 |
| rw12p_e1 | a_v/a_base bgfree | 0.9314 | 0.9420 | 0.9600 | 0.9649 | 0.9749 | 0.9628 |

Pixels newly rewritten by rw12h (g0 above the top wingmig R bin), F150W nrcb1: per-star median of (new pixels / fit pixels), median per-star count, q of the new pixels, q of the round-3 rw12 pixels of the same stars:

| quantity | 14-15 | 15-16 | 16-17 | 17-18 | 18-19 | all |
|---|---|---|---|---|---|---|
| new/fit | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| count | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| q new px (median of star medians; stars with new px) | - | 1.009 | 1.018 | 0.983 | 0.975 | 1.010 |
| q rw12 px (same stars) | - | 1.055 | 1.058 | 1.056 | 1.054 | 1.057 |

#### F150W nrcb3: 770 stars (unsaturated reference 18.46-19.46 mag: dm +0.004)

| variant | 14-15 | 15-16 | 16-17 | 17-18 | 18-19 | all med / MAD | trend |
|---|---|---|---|---|---|---|---|
| final | -0.107 (0.043) | -0.105 (0.029) | -0.085 (0.026) | -0.067 (0.030) | -0.055 (0.024) | -0.075 / 0.032 | +0.052 |
| uncapped | -0.138 (0.051) | -0.128 (0.049) | -0.105 (0.039) | -0.078 (0.033) | -0.063 (0.027) | -0.087 / 0.043 | +0.075 |
| bgfree | -0.095 (0.047) | -0.100 (0.044) | -0.083 (0.040) | -0.064 (0.033) | -0.052 (0.028) | -0.072 / 0.039 | +0.043 |
| bgfree+v7b+cap | -0.022 (0.038) | -0.050 (0.035) | -0.038 (0.034) | -0.026 (0.030) | -0.021 (0.029) | -0.031 / 0.033 | +0.000 |
| rw12 | -0.096 (0.041) | -0.105 (0.038) | -0.086 (0.044) | -0.067 (0.037) | -0.056 (0.032) | -0.076 / 0.040 | +0.040 |
| rw12+cap | -0.090 (0.033) | -0.095 (0.031) | -0.076 (0.032) | -0.059 (0.030) | -0.049 (0.027) | -0.067 / 0.033 | +0.041 |
| rw12+bgfree | -0.068 (0.034) | -0.086 (0.038) | -0.069 (0.039) | -0.055 (0.037) | -0.050 (0.034) | -0.062 / 0.037 | +0.018 |
| rw12+bgfree+cap | -0.063 (0.038) | -0.081 (0.033) | -0.064 (0.032) | -0.052 (0.032) | -0.043 (0.030) | -0.056 / 0.033 | +0.020 |
| rw12h_e1 | -0.096 (0.041) | -0.105 (0.038) | -0.086 (0.044) | -0.067 (0.037) | -0.056 (0.032) | -0.076 / 0.040 | +0.040 |
| rw12h_e1+cap | -0.090 (0.033) | -0.095 (0.031) | -0.076 (0.032) | -0.059 (0.030) | -0.049 (0.027) | -0.067 / 0.033 | +0.041 |
| rw12h_e1+bgfree | -0.068 (0.034) | -0.086 (0.037) | -0.069 (0.039) | -0.055 (0.037) | -0.050 (0.034) | -0.062 / 0.037 | +0.018 |
| rw12h_e1+bgfree+cap | -0.063 (0.038) | -0.081 (0.033) | -0.064 (0.031) | -0.052 (0.031) | -0.043 (0.030) | -0.056 / 0.033 | +0.020 |
| rw12h_e0 | -0.114 (0.054) | -0.116 (0.042) | -0.092 (0.042) | -0.075 (0.034) | -0.063 (0.029) | -0.081 / 0.042 | +0.051 |
| rw12h_e0+cap | -0.095 (0.039) | -0.101 (0.028) | -0.080 (0.031) | -0.063 (0.029) | -0.054 (0.026) | -0.071 / 0.032 | +0.041 |
| rw12h_e0+bgfree | -0.087 (0.043) | -0.092 (0.044) | -0.073 (0.043) | -0.060 (0.035) | -0.055 (0.030) | -0.065 / 0.038 | +0.032 |
| rw12h_e0+bgfree+cap | -0.068 (0.040) | -0.088 (0.035) | -0.066 (0.034) | -0.053 (0.030) | -0.048 (0.027) | -0.059 / 0.032 | +0.020 |
| rw12p_e1 | -0.096 (0.041) | -0.105 (0.038) | -0.086 (0.044) | -0.067 (0.037) | -0.056 (0.032) | -0.076 / 0.040 | +0.040 |
| rw12p_e1+cap | -0.090 (0.033) | -0.095 (0.031) | -0.076 (0.032) | -0.059 (0.030) | -0.049 (0.027) | -0.067 / 0.033 | +0.041 |
| rw12p_e1+bgfree | -0.068 (0.034) | -0.086 (0.037) | -0.069 (0.039) | -0.055 (0.037) | -0.050 (0.034) | -0.062 / 0.037 | +0.018 |
| rw12p_e1+bgfree+cap | -0.063 (0.038) | -0.081 (0.033) | -0.064 (0.031) | -0.052 (0.031) | -0.043 (0.030) | -0.056 / 0.033 | +0.020 |

N per bin: 34, 94, 197, 292, 152

Rewritten fraction of fit pixels (median over stars) and amplitude ratio a_v/a_base (uncapped), F150W nrcb3:

| variant | quantity | 14-15 | 15-16 | 16-17 | 17-18 | 18-19 | all |
|---|---|---|---|---|---|---|---|
| rw12 | rewritten frac | 0.0711 | 0.0411 | 0.0251 | 0.0162 | 0.0081 | 0.0191 |
| rw12 | a_v/a_base | 0.9632 | 0.9758 | 0.9845 | 0.9885 | 0.9945 | 0.9872 |
| rw12 | a_v/a_base bgfree | 0.9366 | 0.9585 | 0.9702 | 0.9790 | 0.9851 | 0.9748 |
| rw12h_e1 | rewritten frac | 0.0712 | 0.0411 | 0.0252 | 0.0162 | 0.0081 | 0.0191 |
| rw12h_e1 | a_v/a_base | 0.9632 | 0.9758 | 0.9846 | 0.9885 | 0.9945 | 0.9872 |
| rw12h_e1 | a_v/a_base bgfree | 0.9366 | 0.9585 | 0.9702 | 0.9790 | 0.9851 | 0.9748 |
| rw12h_e0 | rewritten frac | 0.0712 | 0.0411 | 0.0252 | 0.0162 | 0.0081 | 0.0191 |
| rw12h_e0 | a_v/a_base | 0.9846 | 0.9889 | 0.9906 | 0.9955 | 1.0012 | 0.9943 |
| rw12h_e0 | a_v/a_base bgfree | 0.9578 | 0.9673 | 0.9731 | 0.9832 | 0.9920 | 0.9806 |
| rw12p_e1 | rewritten frac | 0.0712 | 0.0411 | 0.0252 | 0.0162 | 0.0081 | 0.0191 |
| rw12p_e1 | a_v/a_base | 0.9632 | 0.9758 | 0.9846 | 0.9885 | 0.9945 | 0.9872 |
| rw12p_e1 | a_v/a_base bgfree | 0.9366 | 0.9585 | 0.9702 | 0.9790 | 0.9851 | 0.9748 |

Pixels newly rewritten by rw12h (g0 above the top wingmig R bin), F150W nrcb3: per-star median of (new pixels / fit pixels), median per-star count, q of the new pixels, q of the round-3 rw12 pixels of the same stars:

| quantity | 14-15 | 15-16 | 16-17 | 17-18 | 18-19 | all |
|---|---|---|---|---|---|---|
| new/fit | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| count | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| q new px (median of star medians; stars with new px) | 1.051 | 1.037 | 1.010 | 1.018 | 0.515 | 1.021 |
| q rw12 px (same stars) | 1.057 | 1.070 | 1.050 | 1.045 | 1.040 | 1.052 |

Pooled over all rows of the refit frames (all all): newly rewritten pixels 1655 of 39632524 fit pixels (0.00004); pixel-weighted mean q of the new pixels 0.821.

### F200W (score4: final, round-3 rw12, rw12h_e1, rw12h_e0, rw12p_e1; each base, +cap, +bgfree, +bgfree+cap; rewritten fraction; amplitude ratio; newly rewritten pixels)

#### F200W all: 515 stars (unsaturated reference 16.25-17.25 mag: dm -0.017)

| variant | 13-14 | 14-15 | 15-16 | 16-17 | 17-18 | all med / MAD | trend |
|---|---|---|---|---|---|---|---|
| final | +0.007 (0.032) | -0.071 (0.052) | -0.068 (0.053) | -0.034 (0.046) | -0.075 (0.014) | -0.066 / 0.054 | -0.082 |
| uncapped | -0.117 (0.073) | -0.095 (0.053) | -0.102 (0.048) | -0.042 (0.038) | -0.075 (0.014) | -0.097 / 0.050 | +0.042 |
| bgfree | -0.065 (0.095) | -0.079 (0.046) | -0.086 (0.045) | -0.019 (0.047) | -0.056 (0.028) | -0.079 / 0.047 | +0.009 |
| bgfree+v7b+cap | +0.087 (0.025) | -0.016 (0.050) | -0.023 (0.042) | +0.021 (0.055) | -0.024 (0.033) | -0.020 / 0.045 | -0.111 |
| rw12 | -0.064 (0.042) | -0.080 (0.058) | -0.071 (0.050) | -0.029 (0.045) | -0.075 (0.013) | -0.073 / 0.055 | -0.012 |
| rw12+cap | +0.012 (0.049) | -0.066 (0.053) | -0.058 (0.050) | -0.029 (0.042) | -0.075 (0.013) | -0.055 / 0.052 | -0.087 |
| rw12+bgfree | -0.026 (0.061) | -0.064 (0.056) | -0.059 (0.044) | -0.020 (0.057) | -0.066 (0.018) | -0.058 / 0.048 | -0.040 |
| rw12+bgfree+cap | +0.043 (0.036) | -0.054 (0.051) | -0.050 (0.046) | -0.020 (0.052) | -0.066 (0.018) | -0.048 / 0.052 | -0.109 |
| rw12h_e1 | -0.064 (0.042) | -0.080 (0.058) | -0.071 (0.050) | -0.029 (0.045) | -0.075 (0.013) | -0.073 / 0.055 | -0.012 |
| rw12h_e1+cap | +0.012 (0.049) | -0.066 (0.053) | -0.058 (0.050) | -0.029 (0.042) | -0.075 (0.013) | -0.055 / 0.053 | -0.087 |
| rw12h_e1+bgfree | -0.026 (0.061) | -0.064 (0.056) | -0.059 (0.045) | -0.020 (0.057) | -0.066 (0.018) | -0.058 / 0.049 | -0.040 |
| rw12h_e1+bgfree+cap | +0.043 (0.036) | -0.054 (0.051) | -0.050 (0.046) | -0.020 (0.052) | -0.066 (0.018) | -0.048 / 0.052 | -0.109 |
| rw12h_e0 | -0.088 (0.062) | -0.080 (0.059) | -0.075 (0.046) | -0.030 (0.050) | -0.074 (0.010) | -0.075 / 0.051 | +0.014 |
| rw12h_e0+cap | +0.012 (0.032) | -0.063 (0.055) | -0.058 (0.051) | -0.030 (0.050) | -0.074 (0.009) | -0.056 / 0.053 | -0.087 |
| rw12h_e0+bgfree | -0.035 (0.081) | -0.061 (0.055) | -0.057 (0.045) | -0.018 (0.053) | -0.057 (0.023) | -0.058 / 0.051 | -0.022 |
| rw12h_e0+bgfree+cap | +0.036 (0.032) | -0.049 (0.054) | -0.047 (0.045) | -0.004 (0.053) | -0.057 (0.023) | -0.045 / 0.051 | -0.092 |
| rw12p_e1 | -0.064 (0.042) | -0.080 (0.058) | -0.071 (0.050) | -0.029 (0.045) | -0.075 (0.013) | -0.073 / 0.055 | -0.012 |
| rw12p_e1+cap | +0.012 (0.049) | -0.066 (0.053) | -0.058 (0.050) | -0.029 (0.042) | -0.075 (0.013) | -0.055 / 0.053 | -0.087 |
| rw12p_e1+bgfree | -0.026 (0.061) | -0.064 (0.056) | -0.059 (0.045) | -0.020 (0.057) | -0.066 (0.018) | -0.059 / 0.049 | -0.040 |
| rw12p_e1+bgfree+cap | +0.043 (0.036) | -0.054 (0.051) | -0.050 (0.046) | -0.020 (0.052) | -0.066 (0.018) | -0.048 / 0.052 | -0.109 |

N per bin: 8, 157, 314, 31, 5

Rewritten fraction of fit pixels (median over stars) and amplitude ratio a_v/a_base (uncapped), F200W all:

| variant | quantity | 13-14 | 14-15 | 15-16 | 16-17 | 17-18 | all |
|---|---|---|---|---|---|---|---|
| rw12 | rewritten frac | 0.0789 | 0.0974 | 0.0580 | 0.0450 | 0.0106 | 0.0663 |
| rw12 | a_v/a_base | 0.9682 | 0.9807 | 0.9745 | 0.9913 | 0.9929 | 0.9777 |
| rw12 | a_v/a_base bgfree | 0.9283 | 0.9707 | 0.9650 | 0.9740 | 0.9858 | 0.9671 |
| rw12h_e1 | rewritten frac | 0.0789 | 0.0974 | 0.0580 | 0.0450 | 0.0106 | 0.0664 |
| rw12h_e1 | a_v/a_base | 0.9682 | 0.9807 | 0.9745 | 0.9913 | 0.9929 | 0.9773 |
| rw12h_e1 | a_v/a_base bgfree | 0.9283 | 0.9707 | 0.9650 | 0.9740 | 0.9858 | 0.9668 |
| rw12h_e0 | rewritten frac | 0.0789 | 0.0974 | 0.0580 | 0.0450 | 0.0106 | 0.0664 |
| rw12h_e0 | a_v/a_base | 0.9759 | 0.9855 | 0.9784 | 0.9962 | 0.9997 | 0.9815 |
| rw12h_e0 | a_v/a_base bgfree | 0.9286 | 0.9655 | 0.9636 | 0.9772 | 0.9919 | 0.9647 |
| rw12p_e1 | rewritten frac | 0.0789 | 0.0974 | 0.0580 | 0.0450 | 0.0106 | 0.0664 |
| rw12p_e1 | a_v/a_base | 0.9682 | 0.9807 | 0.9745 | 0.9912 | 0.9929 | 0.9774 |
| rw12p_e1 | a_v/a_base bgfree | 0.9282 | 0.9707 | 0.9650 | 0.9740 | 0.9858 | 0.9669 |

Pixels newly rewritten by rw12h (g0 above the top wingmig R bin), F200W all: per-star median of (new pixels / fit pixels), median per-star count, q of the new pixels, q of the round-3 rw12 pixels of the same stars:

| quantity | 13-14 | 14-15 | 15-16 | 16-17 | 17-18 | all |
|---|---|---|---|---|---|---|
| new/fit | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| count | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| q new px (median of star medians; stars with new px) | - | 1.038 | 1.018 | 1.023 | - | 1.026 |
| q rw12 px (same stars) | - | 1.072 | 1.062 | 1.023 | - | 1.066 |

#### F200W nrcb1: 233 stars (unsaturated reference 16.25-17.25 mag: dm -0.017)

| variant | 13-14 | 14-15 | 15-16 | 16-17 | 17-18 | all med / MAD | trend |
|---|---|---|---|---|---|---|---|
| final | - | -0.010 (0.022) | -0.026 (0.022) | -0.011 (0.034) | - | -0.021 / 0.025 | -0.001 |
| uncapped | - | -0.065 (0.044) | -0.074 (0.044) | -0.028 (0.034) | - | -0.069 / 0.047 | +0.038 |
| bgfree | - | -0.049 (0.046) | -0.063 (0.041) | -0.008 (0.030) | - | -0.054 / 0.044 | +0.041 |
| bgfree+v7b+cap | - | +0.019 (0.025) | +0.005 (0.028) | +0.025 (0.045) | - | +0.010 / 0.029 | +0.006 |
| rw12 | - | -0.035 (0.029) | -0.038 (0.028) | -0.022 (0.040) | - | -0.035 / 0.030 | +0.013 |
| rw12+cap | - | -0.009 (0.024) | -0.018 (0.023) | -0.005 (0.042) | - | -0.014 / 0.025 | +0.003 |
| rw12+bgfree | - | -0.026 (0.035) | -0.029 (0.027) | -0.010 (0.047) | - | -0.027 / 0.033 | +0.016 |
| rw12+bgfree+cap | - | -0.002 (0.025) | -0.014 (0.023) | -0.000 (0.038) | - | -0.010 / 0.026 | +0.002 |
| rw12h_e1 | - | -0.035 (0.029) | -0.037 (0.027) | -0.022 (0.040) | - | -0.035 / 0.029 | +0.013 |
| rw12h_e1+cap | - | -0.009 (0.024) | -0.018 (0.022) | -0.005 (0.042) | - | -0.014 / 0.025 | +0.003 |
| rw12h_e1+bgfree | - | -0.026 (0.035) | -0.029 (0.027) | -0.010 (0.047) | - | -0.026 / 0.033 | +0.016 |
| rw12h_e1+bgfree+cap | - | -0.002 (0.026) | -0.013 (0.022) | -0.000 (0.038) | - | -0.010 / 0.026 | +0.002 |
| rw12h_e0 | - | -0.037 (0.032) | -0.046 (0.032) | -0.019 (0.037) | - | -0.043 / 0.035 | +0.018 |
| rw12h_e0+cap | - | -0.006 (0.025) | -0.020 (0.024) | -0.006 (0.041) | - | -0.016 / 0.028 | +0.001 |
| rw12h_e0+bgfree | - | -0.023 (0.039) | -0.033 (0.032) | -0.001 (0.040) | - | -0.027 / 0.037 | +0.022 |
| rw12h_e0+bgfree+cap | - | +0.000 (0.024) | -0.013 (0.029) | +0.001 (0.037) | - | -0.009 / 0.027 | +0.001 |
| rw12p_e1 | - | -0.035 (0.029) | -0.037 (0.027) | -0.022 (0.040) | - | -0.035 / 0.029 | +0.013 |
| rw12p_e1+cap | - | -0.009 (0.024) | -0.018 (0.022) | -0.005 (0.042) | - | -0.014 / 0.025 | +0.003 |
| rw12p_e1+bgfree | - | -0.026 (0.035) | -0.029 (0.027) | -0.010 (0.047) | - | -0.026 / 0.033 | +0.016 |
| rw12p_e1+bgfree+cap | - | -0.002 (0.026) | -0.013 (0.022) | -0.000 (0.038) | - | -0.010 / 0.026 | +0.002 |

N per bin: 4, 68, 139, 21, 1

Rewritten fraction of fit pixels (median over stars) and amplitude ratio a_v/a_base (uncapped), F200W nrcb1:

| variant | quantity | 13-14 | 14-15 | 15-16 | 16-17 | 17-18 | all |
|---|---|---|---|---|---|---|---|
| rw12 | rewritten frac | - | 0.0530 | 0.0358 | 0.0305 | - | 0.0414 |
| rw12 | a_v/a_base | - | 0.9695 | 0.9660 | 0.9898 | - | 0.9672 |
| rw12 | a_v/a_base bgfree | - | 0.9610 | 0.9569 | 0.9740 | - | 0.9587 |
| rw12h_e1 | rewritten frac | - | 0.0531 | 0.0358 | 0.0305 | - | 0.0414 |
| rw12h_e1 | a_v/a_base | - | 0.9695 | 0.9660 | 0.9898 | - | 0.9672 |
| rw12h_e1 | a_v/a_base bgfree | - | 0.9596 | 0.9569 | 0.9740 | - | 0.9587 |
| rw12h_e0 | rewritten frac | - | 0.0531 | 0.0358 | 0.0305 | - | 0.0414 |
| rw12h_e0 | a_v/a_base | - | 0.9712 | 0.9698 | 0.9947 | - | 0.9724 |
| rw12h_e0 | a_v/a_base bgfree | - | 0.9600 | 0.9576 | 0.9772 | - | 0.9601 |
| rw12p_e1 | rewritten frac | - | 0.0531 | 0.0358 | 0.0305 | - | 0.0414 |
| rw12p_e1 | a_v/a_base | - | 0.9695 | 0.9660 | 0.9898 | - | 0.9672 |
| rw12p_e1 | a_v/a_base bgfree | - | 0.9596 | 0.9569 | 0.9740 | - | 0.9587 |

Pixels newly rewritten by rw12h (g0 above the top wingmig R bin), F200W nrcb1: per-star median of (new pixels / fit pixels), median per-star count, q of the new pixels, q of the round-3 rw12 pixels of the same stars:

| quantity | 13-14 | 14-15 | 15-16 | 16-17 | 17-18 | all |
|---|---|---|---|---|---|---|
| new/fit | 0.00000 | 0.00000 | 0.00000 | 0.00000 | - | 0.00000 |
| count | 0.00 | 0.00 | 0.00 | 0.00 | - | 0.00 |
| q new px (median of star medians; stars with new px) | - | 1.019 | 1.026 | 1.028 | - | 1.026 |
| q rw12 px (same stars) | - | 1.067 | 1.060 | 1.023 | - | 1.058 |

#### F200W nrcb3: 282 stars (unsaturated reference 16.25-17.25 mag: dm -0.017)

| variant | 13-14 | 14-15 | 15-16 | 16-17 | 17-18 | all med / MAD | trend |
|---|---|---|---|---|---|---|---|
| final | - | -0.088 (0.016) | -0.097 (0.026) | -0.063 (0.035) | - | -0.093 / 0.025 | +0.025 |
| uncapped | - | -0.115 (0.042) | -0.117 (0.039) | -0.077 (0.048) | - | -0.115 / 0.043 | +0.038 |
| bgfree | - | -0.101 (0.048) | -0.098 (0.044) | -0.057 (0.056) | - | -0.096 / 0.046 | +0.044 |
| bgfree+v7b+cap | - | -0.046 (0.031) | -0.048 (0.030) | -0.022 (0.039) | - | -0.046 / 0.032 | +0.024 |
| rw12 | - | -0.101 (0.031) | -0.094 (0.035) | -0.073 (0.063) | - | -0.096 / 0.036 | +0.028 |
| rw12+cap | - | -0.086 (0.018) | -0.086 (0.027) | -0.056 (0.039) | - | -0.085 / 0.023 | +0.029 |
| rw12+bgfree | - | -0.093 (0.039) | -0.081 (0.034) | -0.057 (0.070) | - | -0.083 / 0.036 | +0.036 |
| rw12+bgfree+cap | - | -0.081 (0.024) | -0.078 (0.029) | -0.050 (0.032) | - | -0.078 / 0.027 | +0.032 |
| rw12h_e1 | - | -0.101 (0.031) | -0.094 (0.035) | -0.073 (0.063) | - | -0.096 / 0.036 | +0.028 |
| rw12h_e1+cap | - | -0.086 (0.018) | -0.086 (0.027) | -0.056 (0.039) | - | -0.085 / 0.023 | +0.029 |
| rw12h_e1+bgfree | - | -0.093 (0.039) | -0.081 (0.033) | -0.057 (0.070) | - | -0.083 / 0.036 | +0.036 |
| rw12h_e1+bgfree+cap | - | -0.082 (0.024) | -0.078 (0.029) | -0.050 (0.032) | - | -0.078 / 0.027 | +0.032 |
| rw12h_e0 | - | -0.104 (0.035) | -0.094 (0.038) | -0.074 (0.050) | - | -0.095 / 0.039 | +0.030 |
| rw12h_e0+cap | - | -0.086 (0.016) | -0.088 (0.029) | -0.063 (0.038) | - | -0.086 / 0.026 | +0.023 |
| rw12h_e0+bgfree | - | -0.085 (0.039) | -0.077 (0.034) | -0.056 (0.055) | - | -0.078 / 0.035 | +0.030 |
| rw12h_e0+bgfree+cap | - | -0.077 (0.023) | -0.073 (0.031) | -0.051 (0.028) | - | -0.074 / 0.028 | +0.027 |
| rw12p_e1 | - | -0.101 (0.031) | -0.095 (0.035) | -0.073 (0.063) | - | -0.097 / 0.036 | +0.028 |
| rw12p_e1+cap | - | -0.086 (0.018) | -0.087 (0.027) | -0.056 (0.039) | - | -0.085 / 0.023 | +0.029 |
| rw12p_e1+bgfree | - | -0.093 (0.039) | -0.081 (0.033) | -0.057 (0.070) | - | -0.083 / 0.036 | +0.036 |
| rw12p_e1+bgfree+cap | - | -0.082 (0.024) | -0.078 (0.028) | -0.050 (0.032) | - | -0.078 / 0.027 | +0.032 |

N per bin: 4, 89, 175, 10, 4

Rewritten fraction of fit pixels (median over stars) and amplitude ratio a_v/a_base (uncapped), F200W nrcb3:

| variant | quantity | 13-14 | 14-15 | 15-16 | 16-17 | 17-18 | all |
|---|---|---|---|---|---|---|---|
| rw12 | rewritten frac | - | 0.1861 | 0.1014 | 0.0860 | - | 0.1130 |
| rw12 | a_v/a_base | - | 0.9879 | 0.9831 | 0.9923 | - | 0.9850 |
| rw12 | a_v/a_base bgfree | - | 0.9748 | 0.9697 | 0.9749 | - | 0.9721 |
| rw12h_e1 | rewritten frac | - | 0.1861 | 0.1014 | 0.0860 | - | 0.1130 |
| rw12h_e1 | a_v/a_base | - | 0.9879 | 0.9831 | 0.9923 | - | 0.9850 |
| rw12h_e1 | a_v/a_base bgfree | - | 0.9748 | 0.9697 | 0.9749 | - | 0.9721 |
| rw12h_e0 | rewritten frac | - | 0.1861 | 0.1014 | 0.0860 | - | 0.1130 |
| rw12h_e0 | a_v/a_base | - | 0.9896 | 0.9839 | 0.9985 | - | 0.9869 |
| rw12h_e0 | a_v/a_base bgfree | - | 0.9704 | 0.9669 | 0.9768 | - | 0.9686 |
| rw12p_e1 | rewritten frac | - | 0.1861 | 0.1014 | 0.0860 | - | 0.1130 |
| rw12p_e1 | a_v/a_base | - | 0.9879 | 0.9831 | 0.9922 | - | 0.9850 |
| rw12p_e1 | a_v/a_base bgfree | - | 0.9748 | 0.9697 | 0.9749 | - | 0.9721 |

Pixels newly rewritten by rw12h (g0 above the top wingmig R bin), F200W nrcb3: per-star median of (new pixels / fit pixels), median per-star count, q of the new pixels, q of the round-3 rw12 pixels of the same stars:

| quantity | 13-14 | 14-15 | 15-16 | 16-17 | 17-18 | all |
|---|---|---|---|---|---|---|
| new/fit | 0.00004 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| count | 0.25 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| q new px (median of star medians; stars with new px) | - | 1.056 | 1.006 | - | - | 1.030 |
| q rw12 px (same stars) | - | 1.074 | 1.066 | - | - | 1.070 |

Pooled over all rows of the refit frames (all all): newly rewritten pixels 2047 of 16045385 fit pixels (0.00013); pixel-weighted mean q of the new pixels 0.946.

### F250M (score4: final, round-3 rw12, rw12h_e1, rw12h_e0, rw12p_e1; each base, +cap, +bgfree, +bgfree+cap; rewritten fraction; amplitude ratio; newly rewritten pixels)

#### F250M all: 905 stars (unsaturated reference 16.46-17.46 mag: dm +0.003)

| variant | 10-13 | 13-14 | 14-15 | 15-16 | 16-17 | all med / MAD | trend |
|---|---|---|---|---|---|---|---|
| final | +0.125 (0.059) | +0.039 (0.036) | +0.021 (0.032) | +0.028 (0.034) | +0.031 (0.034) | +0.030 / 0.035 | -0.094 |
| uncapped | -0.032 (0.104) | +0.001 (0.053) | -0.001 (0.036) | +0.020 (0.036) | +0.029 (0.033) | +0.016 / 0.038 | +0.060 |
| bgfree | -0.007 (0.086) | +0.024 (0.049) | +0.023 (0.040) | +0.036 (0.032) | +0.036 (0.034) | +0.032 / 0.035 | +0.044 |
| bgfree+v7b+cap | +0.156 (0.085) | +0.060 (0.040) | +0.041 (0.030) | +0.045 (0.030) | +0.040 (0.035) | +0.045 / 0.035 | -0.117 |
| rw12 | -0.016 (0.100) | +0.021 (0.047) | +0.013 (0.033) | +0.032 (0.035) | +0.042 (0.035) | +0.029 / 0.037 | +0.058 |
| rw12+cap | +0.125 (0.064) | +0.047 (0.042) | +0.027 (0.030) | +0.038 (0.034) | +0.044 (0.036) | +0.039 / 0.037 | -0.081 |
| rw12+bgfree | +0.008 (0.084) | +0.037 (0.043) | +0.027 (0.031) | +0.042 (0.032) | +0.047 (0.033) | +0.040 / 0.034 | +0.039 |
| rw12+bgfree+cap | +0.143 (0.062) | +0.054 (0.041) | +0.037 (0.034) | +0.048 (0.031) | +0.049 (0.034) | +0.047 / 0.034 | -0.094 |
| rw12h_e1 | -0.016 (0.100) | +0.021 (0.047) | +0.014 (0.035) | +0.032 (0.035) | +0.042 (0.035) | +0.029 / 0.037 | +0.058 |
| rw12h_e1+cap | +0.125 (0.064) | +0.047 (0.042) | +0.027 (0.030) | +0.038 (0.034) | +0.044 (0.036) | +0.039 / 0.037 | -0.081 |
| rw12h_e1+bgfree | +0.006 (0.087) | +0.036 (0.044) | +0.027 (0.031) | +0.042 (0.033) | +0.047 (0.033) | +0.040 / 0.034 | +0.041 |
| rw12h_e1+bgfree+cap | +0.143 (0.062) | +0.054 (0.041) | +0.037 (0.034) | +0.048 (0.031) | +0.049 (0.034) | +0.047 / 0.034 | -0.094 |
| rw12h_e0 | -0.022 (0.102) | +0.005 (0.051) | +0.003 (0.037) | +0.023 (0.036) | +0.032 (0.035) | +0.018 / 0.041 | +0.053 |
| rw12h_e0+cap | +0.125 (0.070) | +0.041 (0.038) | +0.022 (0.031) | +0.030 (0.035) | +0.034 (0.036) | +0.032 / 0.036 | -0.091 |
| rw12h_e0+bgfree | +0.004 (0.081) | +0.030 (0.052) | +0.023 (0.040) | +0.039 (0.032) | +0.039 (0.036) | +0.035 / 0.036 | +0.035 |
| rw12h_e0+bgfree+cap | +0.140 (0.060) | +0.050 (0.040) | +0.035 (0.033) | +0.044 (0.032) | +0.043 (0.035) | +0.044 / 0.036 | -0.097 |
| rw12p_e1 | -0.016 (0.100) | +0.021 (0.047) | +0.014 (0.035) | +0.032 (0.035) | +0.042 (0.035) | +0.029 / 0.037 | +0.058 |
| rw12p_e1+cap | +0.125 (0.064) | +0.047 (0.042) | +0.027 (0.030) | +0.038 (0.034) | +0.044 (0.036) | +0.039 / 0.037 | -0.081 |
| rw12p_e1+bgfree | +0.006 (0.087) | +0.036 (0.044) | +0.027 (0.031) | +0.042 (0.033) | +0.047 (0.033) | +0.040 / 0.034 | +0.041 |
| rw12p_e1+bgfree+cap | +0.143 (0.062) | +0.054 (0.041) | +0.037 (0.034) | +0.048 (0.031) | +0.049 (0.034) | +0.047 / 0.034 | -0.094 |

N per bin: 28, 80, 189, 374, 233

Rewritten fraction of fit pixels (median over stars) and amplitude ratio a_v/a_base (uncapped), F250M all:

| variant | quantity | 10-13 | 13-14 | 14-15 | 15-16 | 16-17 | all |
|---|---|---|---|---|---|---|---|
| rw12 | rewritten frac | 0.0233 | 0.0336 | 0.0161 | 0.0096 | 0.0072 | 0.0105 |
| rw12 | a_v/a_base | 0.9805 | 0.9842 | 0.9896 | 0.9898 | 0.9876 | 0.9884 |
| rw12 | a_v/a_base bgfree | 0.9584 | 0.9704 | 0.9731 | 0.9791 | 0.9817 | 0.9780 |
| rw12h_e1 | rewritten frac | 0.0234 | 0.0340 | 0.0162 | 0.0097 | 0.0072 | 0.0106 |
| rw12h_e1 | a_v/a_base | 0.9805 | 0.9840 | 0.9896 | 0.9898 | 0.9876 | 0.9883 |
| rw12h_e1 | a_v/a_base bgfree | 0.9585 | 0.9697 | 0.9731 | 0.9791 | 0.9818 | 0.9779 |
| rw12h_e0 | rewritten frac | 0.0234 | 0.0340 | 0.0162 | 0.0097 | 0.0072 | 0.0106 |
| rw12h_e0 | a_v/a_base | 0.9865 | 0.9955 | 0.9994 | 0.9983 | 0.9969 | 0.9978 |
| rw12h_e0 | a_v/a_base bgfree | 0.9638 | 0.9722 | 0.9784 | 0.9843 | 0.9887 | 0.9835 |
| rw12p_e1 | rewritten frac | 0.0234 | 0.0340 | 0.0162 | 0.0097 | 0.0072 | 0.0106 |
| rw12p_e1 | a_v/a_base | 0.9805 | 0.9840 | 0.9896 | 0.9898 | 0.9876 | 0.9883 |
| rw12p_e1 | a_v/a_base bgfree | 0.9585 | 0.9697 | 0.9731 | 0.9791 | 0.9818 | 0.9779 |

Pixels newly rewritten by rw12h (g0 above the top wingmig R bin), F250M all: per-star median of (new pixels / fit pixels), median per-star count, q of the new pixels, q of the round-3 rw12 pixels of the same stars:

| quantity | 10-13 | 13-14 | 14-15 | 15-16 | 16-17 | all |
|---|---|---|---|---|---|---|
| new/fit | 0.00000 | 0.00008 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| count | 0.00 | 0.50 | 0.00 | 0.00 | 0.00 | 0.00 |
| q new px (median of star medians; stars with new px) | 1.055 | 0.876 | 0.958 | 0.793 | 0.040 | 0.787 |
| q rw12 px (same stars) | 1.064 | 1.056 | 1.032 | 1.024 | 1.016 | 1.028 |

Pooled over all rows of the refit frames (all all): newly rewritten pixels 3090 of 23484160 fit pixels (0.00013); pixel-weighted mean q of the new pixels 0.683.

### F300M (score4: final, round-3 rw12, rw12h_e1, rw12h_e0, rw12p_e1; each base, +cap, +bgfree, +bgfree+cap; rewritten fraction; amplitude ratio; newly rewritten pixels)

#### F300M all: 944 stars (unsaturated reference 16.43-17.43 mag: dm -0.003)

| variant | 10-13 | 13-14 | 14-15 | 15-16 | 16-17 | all med / MAD | trend |
|---|---|---|---|---|---|---|---|
| final | +0.116 (0.050) | +0.027 (0.035) | +0.014 (0.030) | +0.019 (0.035) | +0.015 (0.035) | +0.018 / 0.035 | -0.101 |
| uncapped | -0.044 (0.090) | -0.007 (0.049) | -0.003 (0.040) | +0.011 (0.035) | +0.014 (0.035) | +0.006 / 0.039 | +0.058 |
| bgfree | -0.008 (0.063) | +0.013 (0.040) | +0.014 (0.032) | +0.024 (0.031) | +0.021 (0.035) | +0.019 / 0.034 | +0.029 |
| bgfree+v7b+cap | +0.127 (0.043) | +0.038 (0.037) | +0.024 (0.029) | +0.033 (0.032) | +0.026 (0.034) | +0.031 / 0.034 | -0.100 |
| rw12 | -0.038 (0.069) | +0.001 (0.045) | +0.003 (0.033) | +0.020 (0.032) | +0.019 (0.036) | +0.013 / 0.035 | +0.057 |
| rw12+cap | +0.116 (0.050) | +0.030 (0.037) | +0.016 (0.029) | +0.026 (0.033) | +0.020 (0.035) | +0.022 / 0.034 | -0.095 |
| rw12+bgfree | +0.004 (0.048) | +0.016 (0.044) | +0.015 (0.028) | +0.028 (0.031) | +0.024 (0.036) | +0.022 / 0.032 | +0.020 |
| rw12+bgfree+cap | +0.116 (0.043) | +0.033 (0.036) | +0.023 (0.029) | +0.032 (0.030) | +0.025 (0.036) | +0.029 / 0.033 | -0.091 |
| rw12h_e1 | -0.038 (0.069) | +0.001 (0.045) | +0.003 (0.032) | +0.020 (0.032) | +0.019 (0.036) | +0.013 / 0.035 | +0.057 |
| rw12h_e1+cap | +0.116 (0.050) | +0.030 (0.037) | +0.016 (0.029) | +0.026 (0.033) | +0.020 (0.035) | +0.022 / 0.034 | -0.095 |
| rw12h_e1+bgfree | +0.004 (0.048) | +0.016 (0.044) | +0.015 (0.028) | +0.028 (0.031) | +0.024 (0.036) | +0.022 / 0.032 | +0.020 |
| rw12h_e1+bgfree+cap | +0.116 (0.043) | +0.033 (0.035) | +0.023 (0.029) | +0.032 (0.030) | +0.025 (0.036) | +0.029 / 0.033 | -0.091 |
| rw12h_e0 | -0.039 (0.077) | -0.003 (0.048) | -0.001 (0.038) | +0.015 (0.037) | +0.017 (0.035) | +0.009 / 0.039 | +0.056 |
| rw12h_e0+cap | +0.116 (0.050) | +0.029 (0.039) | +0.015 (0.030) | +0.021 (0.036) | +0.017 (0.034) | +0.019 / 0.035 | -0.098 |
| rw12h_e0+bgfree | -0.000 (0.046) | +0.017 (0.045) | +0.017 (0.030) | +0.027 (0.032) | +0.024 (0.033) | +0.023 / 0.034 | +0.024 |
| rw12h_e0+bgfree+cap | +0.117 (0.040) | +0.036 (0.033) | +0.025 (0.031) | +0.032 (0.032) | +0.025 (0.033) | +0.030 / 0.032 | -0.092 |
| rw12p_e1 | -0.038 (0.069) | +0.001 (0.045) | +0.003 (0.032) | +0.020 (0.032) | +0.019 (0.036) | +0.013 / 0.035 | +0.057 |
| rw12p_e1+cap | +0.116 (0.050) | +0.030 (0.037) | +0.016 (0.029) | +0.026 (0.033) | +0.020 (0.035) | +0.022 / 0.034 | -0.095 |
| rw12p_e1+bgfree | +0.004 (0.048) | +0.016 (0.044) | +0.015 (0.028) | +0.028 (0.031) | +0.024 (0.036) | +0.022 / 0.032 | +0.020 |
| rw12p_e1+bgfree+cap | +0.116 (0.043) | +0.033 (0.035) | +0.023 (0.029) | +0.032 (0.030) | +0.025 (0.036) | +0.029 / 0.033 | -0.091 |

N per bin: 19, 97, 232, 399, 196

Rewritten fraction of fit pixels (median over stars) and amplitude ratio a_v/a_base (uncapped), F300M all:

| variant | quantity | 10-13 | 13-14 | 14-15 | 15-16 | 16-17 | all |
|---|---|---|---|---|---|---|---|
| rw12 | rewritten frac | 0.1355 | 0.0993 | 0.0596 | 0.0338 | 0.0199 | 0.0376 |
| rw12 | a_v/a_base | 0.9844 | 0.9912 | 0.9923 | 0.9912 | 0.9951 | 0.9922 |
| rw12 | a_v/a_base bgfree | 0.9773 | 0.9799 | 0.9805 | 0.9835 | 0.9908 | 0.9842 |
| rw12h_e1 | rewritten frac | 0.1355 | 0.0994 | 0.0597 | 0.0338 | 0.0199 | 0.0377 |
| rw12h_e1 | a_v/a_base | 0.9844 | 0.9912 | 0.9923 | 0.9912 | 0.9951 | 0.9921 |
| rw12h_e1 | a_v/a_base bgfree | 0.9773 | 0.9798 | 0.9805 | 0.9835 | 0.9908 | 0.9842 |
| rw12h_e0 | rewritten frac | 0.1355 | 0.0994 | 0.0597 | 0.0338 | 0.0199 | 0.0377 |
| rw12h_e0 | a_v/a_base | 0.9911 | 0.9966 | 0.9971 | 0.9977 | 0.9959 | 0.9971 |
| rw12h_e0 | a_v/a_base bgfree | 0.9745 | 0.9798 | 0.9805 | 0.9866 | 0.9884 | 0.9851 |
| rw12p_e1 | rewritten frac | 0.1355 | 0.0994 | 0.0597 | 0.0338 | 0.0199 | 0.0377 |
| rw12p_e1 | a_v/a_base | 0.9844 | 0.9912 | 0.9923 | 0.9912 | 0.9951 | 0.9921 |
| rw12p_e1 | a_v/a_base bgfree | 0.9773 | 0.9798 | 0.9805 | 0.9835 | 0.9908 | 0.9842 |

Pixels newly rewritten by rw12h (g0 above the top wingmig R bin), F300M all: per-star median of (new pixels / fit pixels), median per-star count, q of the new pixels, q of the round-3 rw12 pixels of the same stars:

| quantity | 10-13 | 13-14 | 14-15 | 15-16 | 16-17 | all |
|---|---|---|---|---|---|---|
| new/fit | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| count | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| q new px (median of star medians; stars with new px) | 0.532 | 0.550 | 0.534 | 0.501 | 0.490 | 0.521 |
| q rw12 px (same stars) | 1.076 | 1.070 | 1.051 | 1.030 | 1.020 | 1.044 |

Pooled over all rows of the refit frames (all all): newly rewritten pixels 2266 of 24747675 fit pixels (0.00009); pixel-weighted mean q of the new pixels 0.525.

### F150W norim and per-pixel ratio (score4b)

#### F150W all: 1471 stars

| variant | 14-15 | 15-16 | 16-17 | 17-18 | 18-19 | all med / MAD | trend |
|---|---|---|---|---|---|---|---|
| final | -0.095 (0.070) | -0.084 (0.048) | -0.057 (0.045) | -0.040 (0.044) | -0.032 (0.047) | -0.047 / 0.047 | +0.063 |
| uncapped | -0.142 (0.063) | -0.122 (0.055) | -0.101 (0.046) | -0.066 (0.045) | -0.047 (0.039) | -0.078 / 0.053 | +0.095 |
| bgfree | -0.117 (0.058) | -0.099 (0.053) | -0.083 (0.046) | -0.054 (0.043) | -0.038 (0.039) | -0.064 / 0.048 | +0.079 |
| bgfree+v7b+cap | -0.021 (0.030) | -0.035 (0.038) | -0.024 (0.033) | -0.009 (0.041) | -0.002 (0.041) | -0.015 / 0.041 | +0.018 |
| rw12 | -0.095 (0.048) | -0.092 (0.046) | -0.078 (0.047) | -0.045 (0.051) | -0.030 (0.047) | -0.058 / 0.052 | +0.066 |
| rw12+bgfree | -0.070 (0.032) | -0.074 (0.044) | -0.063 (0.043) | -0.034 (0.048) | -0.022 (0.045) | -0.047 / 0.050 | +0.048 |
| rw12+bgfree+cap | -0.051 (0.041) | -0.058 (0.044) | -0.043 (0.037) | -0.023 (0.046) | -0.020 (0.048) | -0.034 / 0.047 | +0.031 |
| norim | -0.233 (0.086) | -0.260 (0.112) | -0.211 (0.097) | -0.111 (0.084) | -0.044 (0.080) | -0.142 / 0.123 | +0.189 |
| norim+cap | -0.126 (0.099) | -0.100 (0.067) | -0.064 (0.054) | -0.043 (0.048) | -0.016 (0.047) | -0.048 / 0.058 | +0.110 |
| norim+bgfree | -0.179 (0.057) | -0.190 (0.089) | -0.162 (0.088) | -0.078 (0.077) | -0.018 (0.075) | -0.108 / 0.106 | +0.161 |
| norim+bgfree+cap | -0.095 (0.077) | -0.082 (0.057) | -0.055 (0.048) | -0.033 (0.046) | +0.007 (0.049) | -0.037 / 0.057 | +0.102 |

norim bookkeeping, F150W all: median over stars per bin

| quantity | 14-15 | 15-16 | 16-17 | 17-18 | 18-19 | all |
|---|---|---|---|---|---|---|
| rim px / fit px (base fit) | 0.019 | 0.016 | 0.014 | 0.009 | 0.005 | 0.010 |
| fit px left after removing rim | 6378 | 6436 | 6463 | 6498 | 6525 | 6485 |
| a_norim / a_base | 1.0711 | 1.1205 | 1.0983 | 1.0439 | 1.0055 | 1.0601 |
| a_norim+bgfree / a_base | 1.0432 | 1.0550 | 1.0620 | 1.0207 | 0.9864 | 1.0292 |

#### F150W nrcb1: 701 stars

| variant | 14-15 | 15-16 | 16-17 | 17-18 | 18-19 | all med / MAD | trend |
|---|---|---|---|---|---|---|---|
| final | -0.035 (0.042) | -0.044 (0.021) | -0.028 (0.028) | -0.013 (0.033) | +0.004 (0.041) | -0.017 / 0.037 | +0.039 |
| uncapped | -0.149 (0.062) | -0.106 (0.065) | -0.096 (0.056) | -0.047 (0.048) | -0.027 (0.040) | -0.060 / 0.065 | +0.122 |
| bgfree | -0.126 (0.048) | -0.087 (0.059) | -0.082 (0.058) | -0.037 (0.046) | -0.020 (0.039) | -0.048 / 0.061 | +0.107 |
| bgfree+v7b+cap | -0.014 (0.020) | -0.013 (0.040) | -0.008 (0.039) | +0.015 (0.038) | +0.024 (0.039) | +0.009 / 0.042 | +0.038 |
| rw12 | -0.090 (0.055) | -0.065 (0.051) | -0.066 (0.049) | -0.018 (0.038) | -0.004 (0.028) | -0.030 / 0.053 | +0.086 |
| rw12+bgfree | -0.070 (0.032) | -0.051 (0.050) | -0.051 (0.050) | -0.011 (0.038) | -0.000 (0.028) | -0.018 / 0.050 | +0.070 |
| rw12+bgfree+cap | -0.030 (0.031) | -0.026 (0.034) | -0.021 (0.036) | +0.004 (0.033) | +0.014 (0.029) | -0.002 / 0.039 | +0.043 |
| norim | -0.216 (0.060) | -0.223 (0.085) | -0.204 (0.101) | -0.113 (0.092) | -0.061 (0.093) | -0.137 / 0.117 | +0.155 |
| norim+cap | -0.036 (0.042) | -0.048 (0.026) | -0.033 (0.027) | -0.016 (0.035) | +0.010 (0.046) | -0.023 / 0.038 | +0.046 |
| norim+bgfree | -0.183 (0.028) | -0.192 (0.072) | -0.170 (0.083) | -0.090 (0.083) | -0.037 (0.085) | -0.115 / 0.107 | +0.146 |
| norim+bgfree+cap | -0.036 (0.037) | -0.046 (0.029) | -0.030 (0.027) | -0.012 (0.037) | +0.011 (0.044) | -0.019 / 0.041 | +0.047 |

norim bookkeeping, F150W nrcb1: median over stars per bin

| quantity | 14-15 | 15-16 | 16-17 | 17-18 | 18-19 | all |
|---|---|---|---|---|---|---|
| rim px / fit px (base fit) | 0.013 | 0.013 | 0.012 | 0.007 | 0.004 | 0.008 |
| fit px left after removing rim | 6455 | 6464 | 6474 | 6510 | 6528 | 6495 |
| a_norim / a_base | 1.0549 | 1.1148 | 1.0965 | 1.0545 | 1.0310 | 1.0666 |
| a_norim+bgfree / a_base | 1.0482 | 1.0714 | 1.0689 | 1.0411 | 1.0122 | 1.0462 |

#### F150W nrcb3: 770 stars

| variant | 14-15 | 15-16 | 16-17 | 17-18 | 18-19 | all med / MAD | trend |
|---|---|---|---|---|---|---|---|
| final | -0.107 (0.043) | -0.105 (0.029) | -0.085 (0.026) | -0.067 (0.030) | -0.055 (0.024) | -0.075 / 0.032 | +0.052 |
| uncapped | -0.138 (0.051) | -0.128 (0.049) | -0.105 (0.039) | -0.078 (0.033) | -0.063 (0.027) | -0.087 / 0.043 | +0.075 |
| bgfree | -0.095 (0.047) | -0.100 (0.044) | -0.083 (0.040) | -0.064 (0.033) | -0.052 (0.028) | -0.072 / 0.039 | +0.043 |
| bgfree+v7b+cap | -0.022 (0.038) | -0.050 (0.035) | -0.038 (0.034) | -0.026 (0.030) | -0.021 (0.029) | -0.031 / 0.033 | +0.000 |
| rw12 | -0.096 (0.041) | -0.105 (0.038) | -0.086 (0.044) | -0.067 (0.037) | -0.056 (0.032) | -0.076 / 0.040 | +0.040 |
| rw12+bgfree | -0.068 (0.034) | -0.086 (0.038) | -0.069 (0.039) | -0.055 (0.037) | -0.050 (0.034) | -0.062 / 0.037 | +0.018 |
| rw12+bgfree+cap | -0.063 (0.038) | -0.081 (0.033) | -0.064 (0.032) | -0.052 (0.032) | -0.043 (0.030) | -0.056 / 0.033 | +0.020 |
| norim | -0.239 (0.090) | -0.280 (0.113) | -0.216 (0.093) | -0.110 (0.076) | -0.038 (0.056) | -0.150 / 0.127 | +0.201 |
| norim+cap | -0.145 (0.070) | -0.129 (0.050) | -0.104 (0.040) | -0.070 (0.036) | -0.031 (0.038) | -0.082 / 0.054 | +0.114 |
| norim+bgfree | -0.171 (0.071) | -0.190 (0.102) | -0.156 (0.079) | -0.069 (0.073) | -0.006 (0.059) | -0.100 / 0.105 | +0.165 |
| norim+bgfree+cap | -0.109 (0.047) | -0.114 (0.044) | -0.090 (0.038) | -0.053 (0.041) | -0.005 (0.054) | -0.065 / 0.057 | +0.104 |

norim bookkeeping, F150W nrcb3: median over stars per bin

| quantity | 14-15 | 15-16 | 16-17 | 17-18 | 18-19 | all |
|---|---|---|---|---|---|---|
| rim px / fit px (base fit) | 0.026 | 0.023 | 0.016 | 0.011 | 0.006 | 0.012 |
| fit px left after removing rim | 6325 | 6395 | 6432 | 6485 | 6516 | 6473 |
| a_norim / a_base | 1.1112 | 1.1327 | 1.0993 | 1.0292 | 0.9786 | 1.0503 |
| a_norim+bgfree / a_base | 1.0271 | 1.0466 | 1.0392 | 0.9987 | 0.9528 | 1.0094 |

### F150W per-pixel data/model, model = F_dolphot x PSF at the fitted position (data = neighbour-subtracted, minus pipeline annulus bkg); pixels within 30 px with PSF >= 1e-3 of peak; se = 1.2533 MAD/sqrt(N)

Pixel counts (cat, det): 0/nrcb1: 71868, 0/nrcb3: 85883, 1/nrcb1: 652267, 1/nrcb3: 707476, 2/nrcb1: 778, 2/nrcb3: 1408 (cat 0: rim in fit, values after rewrite; 1: never-rewritten crf pixel in fit; 2: rim pixel masked from the fit)

**(a) rim pixels in the fit: by g0 (DN)**

| bin | nrcb1 median +- se (N) | nrcb3 median +- se (N) | nrcb3/nrcb1 |
|---|---|---|---|
| 200-400 | 0.951+-0.004 (6887) | 1.010+-0.004 (8080) | 1.063 |
| 400-800 | 1.012+-0.002 (15372) | 1.066+-0.002 (18244) | 1.053 |
| 800-1600 | 1.056+-0.002 (16539) | 1.102+-0.002 (19553) | 1.044 |
| 1600-3200 | 1.057+-0.002 (12127) | 1.109+-0.002 (15298) | 1.049 |
| 3200-6400 | 1.027+-0.002 (8683) | 1.086+-0.001 (10259) | 1.058 |
| 6400-12800 | 1.029+-0.002 (5091) | 1.088+-0.001 (5896) | 1.058 |
| >12800 | 1.034+-0.001 (4493) | 1.097+-0.001 (5722) | 1.062 |

**(a) rim pixels in the fit: by r from the fitted position (px)**

| bin | nrcb1 median +- se (N) | nrcb3 median +- se (N) | nrcb3/nrcb1 |
|---|---|---|---|
| 0-2 | 1.014+-0.001 (30958) | 1.073+-0.001 (33411) | 1.059 |
| 2-3 | 1.029+-0.002 (21152) | 1.077+-0.002 (24285) | 1.046 |
| 3-4 | 1.106+-0.004 (11796) | 1.129+-0.003 (15786) | 1.021 |
| 4-6 | 1.193+-0.007 (5565) | 1.224+-0.006 (8047) | 1.027 |
| 6-9 | 5.163+-0.517 (1131) | 6.317+-0.312 (2082) | 1.223 |
| 9-14 | 9.548+-0.856 (1066) | 5.814+-0.529 (1790) | 0.609 |
| 14-31 | 5.757+-1.430 (200) | 8.528+-1.414 (482) | 1.481 |

**(b) never-rewritten crf pixels in the fit: by g0 (DN)**

| bin | nrcb1 median +- se (N) | nrcb3 median +- se (N) | nrcb3/nrcb1 |
|---|---|---|---|
| 200-400 | 1.274+-0.002 (62291) | 1.294+-0.002 (96986) | 1.015 |
| 400-800 | 1.330+-0.003 (24121) | 1.318+-0.002 (36690) | 0.991 |
| 800-1600 | 1.390+-0.006 (4992) | 1.348+-0.006 (6992) | 0.970 |
| 1600-3200 | 1.447+-0.049 (573) | 2.548+-0.260 (798) | 1.761 |
| 3200-6400 | 14.619+-4.413 (36) | 20.920+-3.989 (95) | 1.431 |
| 6400-12800 | - (13) | - (7) | nan |
| >12800 | - (2) | - (1) | nan |

**(b) never-rewritten crf pixels in the fit: by r from the fitted position (px)**

| bin | nrcb1 median +- se (N) | nrcb3 median +- se (N) | nrcb3/nrcb1 |
|---|---|---|---|
| 0-2 | 1.031+-0.003 (1479) | 1.020+-0.004 (1215) | 0.989 |
| 2-3 | 1.066+-0.003 (19514) | 1.082+-0.003 (19039) | 1.015 |
| 3-4 | 1.150+-0.002 (45130) | 1.138+-0.002 (45338) | 0.989 |
| 4-6 | 1.203+-0.001 (145782) | 1.209+-0.001 (155806) | 1.006 |
| 6-9 | 1.256+-0.001 (220113) | 1.291+-0.001 (241992) | 1.028 |
| 9-14 | 0.996+-0.001 (187167) | 1.032+-0.001 (205289) | 1.037 |
| 14-31 | 1.230+-0.002 (33082) | 1.239+-0.003 (38797) | 1.008 |

- rim in the fit, brighter half of stars (dolphot mag < 17.40): nrcb1 1.044 (N=55029), nrcb3 1.098 (N=66912), nrcb3/nrcb1 1.051
- rim in the fit, fainter half of stars (dolphot mag >= 17.40): nrcb1 1.007 (N=16839), nrcb3 1.069 (N=18971), nrcb3/nrcb1 1.061
- never-rewritten, brighter half of stars (dolphot mag < 17.40): nrcb1 1.190 (N=357977), nrcb3 1.214 (N=388796), nrcb3/nrcb1 1.021
- never-rewritten, fainter half of stars (dolphot mag >= 17.40): nrcb1 1.093 (N=294290), nrcb3 1.133 (N=318680), nrcb3/nrcb1 1.036

**(a) model peak pixel (psf maximum of the cutout)**

| category | nrcb1 median +- se (N) | nrcb3 median +- se (N) | nrcb3/nrcb1 |
|---|---|---|---|
| peak pixel is a rim pixel in the fit | 1.016+-0.001 (2584) | 1.079+-0.001 (2755) | 1.062 |
| peak pixel is a rim pixel masked from the fit | - (4) | - (8) | nan |
| peak pixel is a never-rewritten pixel in the fit | - (2) | - (1) | nan |


### F200W norim and per-pixel ratio (score4b)

#### F200W all: 515 stars

| variant | 13-14 | 14-15 | 15-16 | 16-17 | 17-18 | all med / MAD | trend |
|---|---|---|---|---|---|---|---|
| final | +0.007 (0.032) | -0.071 (0.052) | -0.068 (0.053) | -0.034 (0.046) | -0.075 (0.014) | -0.066 / 0.054 | -0.082 |
| uncapped | -0.117 (0.073) | -0.095 (0.053) | -0.102 (0.048) | -0.042 (0.038) | -0.075 (0.014) | -0.097 / 0.050 | +0.042 |
| bgfree | -0.065 (0.095) | -0.079 (0.046) | -0.086 (0.045) | -0.019 (0.047) | -0.056 (0.028) | -0.079 / 0.047 | +0.009 |
| bgfree+v7b+cap | +0.087 (0.025) | -0.016 (0.050) | -0.023 (0.042) | +0.021 (0.055) | -0.024 (0.033) | -0.020 / 0.045 | -0.111 |
| rw12 | -0.064 (0.042) | -0.080 (0.058) | -0.071 (0.050) | -0.029 (0.045) | -0.075 (0.013) | -0.073 / 0.055 | -0.012 |
| rw12+bgfree | -0.026 (0.061) | -0.064 (0.056) | -0.059 (0.044) | -0.020 (0.057) | -0.066 (0.018) | -0.058 / 0.048 | -0.040 |
| rw12+bgfree+cap | +0.043 (0.036) | -0.054 (0.051) | -0.050 (0.046) | -0.020 (0.052) | -0.066 (0.018) | -0.048 / 0.052 | -0.109 |
| norim | -0.163 (0.075) | -0.197 (0.109) | -0.185 (0.083) | -0.131 (0.157) | -0.254 (0.215) | -0.187 / 0.095 | -0.091 |
| norim+cap | +0.001 (0.040) | -0.081 (0.071) | -0.076 (0.061) | -0.048 (0.085) | -0.088 (0.031) | -0.076 / 0.067 | -0.089 |
| norim+bgfree | -0.138 (0.075) | -0.135 (0.085) | -0.147 (0.071) | -0.026 (0.127) | -0.141 (0.141) | -0.141 / 0.079 | -0.004 |
| norim+bgfree+cap | +0.023 (0.056) | -0.038 (0.067) | -0.068 (0.057) | -0.002 (0.091) | -0.079 (0.050) | -0.057 / 0.061 | -0.102 |

norim bookkeeping, F200W all: median over stars per bin

| quantity | 13-14 | 14-15 | 15-16 | 16-17 | 17-18 | all |
|---|---|---|---|---|---|---|
| rim px / fit px (base fit) | 0.021 | 0.025 | 0.019 | 0.021 | 0.012 | 0.021 |
| fit px left after removing rim | 6394 | 6390 | 6422 | 6398 | 6482 | 6413 |
| a_norim / a_base | 1.0430 | 1.0829 | 1.0724 | 1.0897 | 1.1693 | 1.0734 |
| a_norim+bgfree / a_base | 1.0123 | 1.0383 | 1.0400 | 1.0093 | 1.0630 | 1.0389 |

#### F200W nrcb1: 233 stars

| variant | 13-14 | 14-15 | 15-16 | 16-17 | 17-18 | all med / MAD | trend |
|---|---|---|---|---|---|---|---|
| final | - | -0.010 (0.022) | -0.026 (0.022) | -0.011 (0.034) | - | -0.021 / 0.025 | -0.001 |
| uncapped | - | -0.065 (0.044) | -0.074 (0.044) | -0.028 (0.034) | - | -0.069 / 0.047 | +0.038 |
| bgfree | - | -0.049 (0.046) | -0.063 (0.041) | -0.008 (0.030) | - | -0.054 / 0.044 | +0.041 |
| bgfree+v7b+cap | - | +0.019 (0.025) | +0.005 (0.028) | +0.025 (0.045) | - | +0.010 / 0.029 | +0.006 |
| rw12 | - | -0.035 (0.029) | -0.038 (0.028) | -0.022 (0.040) | - | -0.035 / 0.030 | +0.013 |
| rw12+bgfree | - | -0.026 (0.035) | -0.029 (0.027) | -0.010 (0.047) | - | -0.027 / 0.033 | +0.016 |
| rw12+bgfree+cap | - | -0.002 (0.025) | -0.014 (0.023) | -0.000 (0.038) | - | -0.010 / 0.026 | +0.002 |
| norim | - | -0.162 (0.072) | -0.170 (0.064) | -0.123 (0.155) | - | -0.167 / 0.077 | +0.039 |
| norim+cap | - | -0.012 (0.025) | -0.027 (0.025) | -0.030 (0.058) | - | -0.024 / 0.028 | -0.017 |
| norim+bgfree | - | -0.133 (0.073) | -0.141 (0.061) | -0.026 (0.089) | - | -0.132 / 0.069 | +0.106 |
| norim+bgfree+cap | - | -0.011 (0.024) | -0.026 (0.026) | -0.002 (0.047) | - | -0.020 / 0.027 | +0.009 |

norim bookkeeping, F200W nrcb1: median over stars per bin

| quantity | 13-14 | 14-15 | 15-16 | 16-17 | 17-18 | all |
|---|---|---|---|---|---|---|
| rim px / fit px (base fit) | - | 0.018 | 0.015 | 0.017 | - | 0.016 |
| fit px left after removing rim | - | 6434 | 6455 | 6427 | - | 6443 |
| a_norim / a_base | - | 1.0784 | 1.0760 | 1.0897 | - | 1.0776 |
| a_norim+bgfree / a_base | - | 1.0518 | 1.0551 | 1.0293 | - | 1.0517 |

#### F200W nrcb3: 282 stars

| variant | 13-14 | 14-15 | 15-16 | 16-17 | 17-18 | all med / MAD | trend |
|---|---|---|---|---|---|---|---|
| final | - | -0.088 (0.016) | -0.097 (0.026) | -0.063 (0.035) | - | -0.093 / 0.025 | +0.025 |
| uncapped | - | -0.115 (0.042) | -0.117 (0.039) | -0.077 (0.048) | - | -0.115 / 0.043 | +0.038 |
| bgfree | - | -0.101 (0.048) | -0.098 (0.044) | -0.057 (0.056) | - | -0.096 / 0.046 | +0.044 |
| bgfree+v7b+cap | - | -0.046 (0.031) | -0.048 (0.030) | -0.022 (0.039) | - | -0.046 / 0.032 | +0.024 |
| rw12 | - | -0.101 (0.031) | -0.094 (0.035) | -0.073 (0.063) | - | -0.096 / 0.036 | +0.028 |
| rw12+bgfree | - | -0.093 (0.039) | -0.081 (0.034) | -0.057 (0.070) | - | -0.083 / 0.036 | +0.036 |
| rw12+bgfree+cap | - | -0.081 (0.024) | -0.078 (0.029) | -0.050 (0.032) | - | -0.078 / 0.027 | +0.032 |
| norim | - | -0.222 (0.106) | -0.197 (0.094) | -0.213 (0.225) | - | -0.202 / 0.104 | +0.009 |
| norim+cap | - | -0.099 (0.027) | -0.111 (0.031) | -0.089 (0.134) | - | -0.104 / 0.033 | +0.010 |
| norim+bgfree | - | -0.136 (0.096) | -0.153 (0.084) | -0.032 (0.226) | - | -0.147 / 0.088 | +0.104 |
| norim+bgfree+cap | - | -0.085 (0.028) | -0.098 (0.035) | -0.020 (0.137) | - | -0.092 / 0.035 | +0.064 |

norim bookkeeping, F200W nrcb3: median over stars per bin

| quantity | 13-14 | 14-15 | 15-16 | 16-17 | 17-18 | all |
|---|---|---|---|---|---|---|
| rim px / fit px (base fit) | - | 0.040 | 0.025 | 0.026 | - | 0.029 |
| fit px left after removing rim | - | 6270 | 6360 | 6342 | - | 6342 |
| a_norim / a_base | - | 1.0888 | 1.0642 | 1.0836 | - | 1.0712 |
| a_norim+bgfree / a_base | - | 1.0219 | 1.0252 | 0.9518 | - | 1.0244 |

### F200W per-pixel data/model, model = F_dolphot x PSF at the fitted position (data = neighbour-subtracted, minus pipeline annulus bkg); pixels within 30 px with PSF >= 1e-3 of peak; se = 1.2533 MAD/sqrt(N)

Pixel counts (cat, det): 0/nrcb1: 46217, 0/nrcb3: 61505, 1/nrcb1: 240110, 1/nrcb3: 292899, 2/nrcb1: 549, 2/nrcb3: 881 (cat 0: rim in fit, values after rewrite; 1: never-rewritten crf pixel in fit; 2: rim pixel masked from the fit)

**(a) rim pixels in the fit: by g0 (DN)**

| bin | nrcb1 median +- se (N) | nrcb3 median +- se (N) | nrcb3/nrcb1 |
|---|---|---|---|
| 200-400 | 0.927+-0.017 (674) | 1.083+-0.021 (699) | 1.167 |
| 400-800 | 0.955+-0.004 (5956) | 1.068+-0.005 (6814) | 1.118 |
| 800-1600 | 1.062+-0.002 (13287) | 1.117+-0.002 (17218) | 1.051 |
| 1600-3200 | 1.094+-0.003 (10735) | 1.136+-0.002 (15656) | 1.039 |
| 3200-6400 | 1.095+-0.003 (5944) | 1.141+-0.002 (8452) | 1.042 |
| 6400-12800 | 1.043+-0.002 (3263) | 1.118+-0.002 (4393) | 1.072 |
| >12800 | 1.027+-0.001 (6314) | 1.100+-0.001 (8225) | 1.071 |

**(a) rim pixels in the fit: by r from the fitted position (px)**

| bin | nrcb1 median +- se (N) | nrcb3 median +- se (N) | nrcb3/nrcb1 |
|---|---|---|---|
| 0-2 | 1.023+-0.001 (10101) | 1.094+-0.001 (12306) | 1.070 |
| 2-3 | 1.046+-0.002 (12667) | 1.090+-0.001 (15322) | 1.042 |
| 3-4 | 1.046+-0.003 (14032) | 1.116+-0.002 (17648) | 1.067 |
| 4-6 | 1.144+-0.004 (7438) | 1.171+-0.003 (11628) | 1.023 |
| 6-9 | 7.322+-0.329 (930) | 7.844+-0.247 (2064) | 1.071 |
| 9-14 | 12.768+-0.667 (836) | 8.897+-0.337 (1969) | 0.697 |
| 14-31 | 5.776+-1.641 (213) | 9.848+-0.769 (568) | 1.705 |

**(b) never-rewritten crf pixels in the fit: by g0 (DN)**

| bin | nrcb1 median +- se (N) | nrcb3 median +- se (N) | nrcb3/nrcb1 |
|---|---|---|---|
| 200-400 | 1.243+-0.001 (89656) | 1.260+-0.001 (141365) | 1.014 |
| 400-800 | 1.272+-0.002 (31685) | 1.314+-0.002 (57555) | 1.033 |
| 800-1600 | 1.244+-0.004 (9231) | 1.282+-0.003 (15076) | 1.031 |
| 1600-3200 | 1.263+-0.007 (1927) | 1.245+-0.006 (2954) | 0.986 |
| 3200-6400 | 1.298+-0.061 (43) | 1.342+-0.059 (83) | 1.034 |
| 6400-12800 | - (5) | - (4) | nan |
| >12800 | - (1) | - (1) | nan |

**(b) never-rewritten crf pixels in the fit: by r from the fitted position (px)**

| bin | nrcb1 median +- se (N) | nrcb3 median +- se (N) | nrcb3/nrcb1 |
|---|---|---|---|
| 0-2 | - (10) | - (4) | nan |
| 2-3 | -0.023+-0.212 (31) | -0.013+-0.201 (25) | 0.558 |
| 3-4 | 1.147+-0.006 (3702) | 1.168+-0.005 (3891) | 1.018 |
| 4-6 | 1.261+-0.002 (43032) | 1.281+-0.002 (49577) | 1.016 |
| 6-9 | 1.206+-0.001 (87055) | 1.244+-0.001 (106375) | 1.032 |
| 9-14 | 1.165+-0.001 (85324) | 1.220+-0.001 (104627) | 1.047 |
| 14-31 | 1.046+-0.002 (20956) | 1.069+-0.003 (28400) | 1.022 |

- rim in the fit, brighter half of stars (dolphot mag < 15.35): nrcb1 1.053 (N=27033), nrcb3 1.116 (N=39207), nrcb3/nrcb1 1.060
- rim in the fit, fainter half of stars (dolphot mag >= 15.35): nrcb1 1.041 (N=19184), nrcb3 1.112 (N=22298), nrcb3/nrcb1 1.068
- never-rewritten, brighter half of stars (dolphot mag < 15.35): nrcb1 1.191 (N=120155), nrcb3 1.232 (N=162013), nrcb3/nrcb1 1.035
- never-rewritten, fainter half of stars (dolphot mag >= 15.35): nrcb1 1.187 (N=119955), nrcb3 1.222 (N=130886), nrcb3/nrcb1 1.029

**(a) model peak pixel (psf maximum of the cutout)**

| category | nrcb1 median +- se (N) | nrcb3 median +- se (N) | nrcb3/nrcb1 |
|---|---|---|---|
| peak pixel is a rim pixel in the fit | 1.022+-0.001 (807) | 1.096+-0.001 (976) | 1.072 |
| peak pixel is a rim pixel masked from the fit | - (3) | - (6) | nan |
| peak pixel is a never-rewritten pixel in the fit | - (0) | - (0) | nan |


## 7. Pipeline versus wingmig R curves per frame

| band | det | exp | pipeline R (g0) kept bins | wingmig R at the same g0 | pipeline / wingmig | bins measured -> kept | wingmig R range (g0 max) | satcheck curve/sat |
|---|---|---|---|---|---|---|---|---|
| 150W | nrcb1 | 1 | 0.0902 (2438), 0.0882 (3622) | 0.0880, 0.0874 | 1.026, 1.009 | 4 -> 2 | 0.0865-0.0880 (3506) | 0.998 |
| 150W | nrcb3 | 1 | 0.0939 (2440), 0.0914 (3631) | 0.0874, 0.0847 | 1.074, 1.079 | 4 -> 2 | 0.0838-0.0899 (3924) | 1.011 |
| 150W | nrcb1 | 2 | 0.0901 (2438), 0.0875 (3622) | 0.0874, 0.0852 | 1.032, 1.026 | 4 -> 2 | 0.0847-0.0884 (3668) | 0.990 |
| 150W | nrcb3 | 2 | 0.0939 (2440), 0.0914 (3632) | 0.0868, 0.0847 | 1.082, 1.079 | 4 -> 2 | 0.0830-0.0901 (4030) | 1.011 |
| 150W | nrcb1 | 3 | 0.0898 (2438), 0.0876 (3621) | 0.0876, 0.0876 | 1.026, 0.999 | 4 -> 2 | 0.0856-0.0886 (2872) | 0.991 |
| 150W | nrcb3 | 3 | 0.0940 (2440), 0.0916 (3632) | 0.0875, 0.0859 | 1.075, 1.066 | 4 -> 2 | 0.0859-0.0899 (3582) | 1.013 |
| 150W | nrcb1 | 4 | 0.0902 (2438), 0.0877 (3621) | 0.0876, 0.0865 | 1.029, 1.013 | 4 -> 2 | 0.0857-0.0888 (3343) | 0.992 |
| 150W | nrcb3 | 4 | 0.0944 (2440), 0.0916 (3631) | 0.0878, 0.0862 | 1.076, 1.062 | 4 -> 2 | 0.0862-0.0898 (3458) | 1.013 |
| 200W | nrcb1 | 1 | 0.0760 (2437), 0.0743 (3619) | 0.0742, 0.0738 | 1.025, 1.006 | 4 -> 2 | 0.0698-0.0743 (3003) | 1.002 |
| 200W | nrcb3 | 1 | 0.0777 (2440), 0.0757 (3631) | 0.0731, 0.0714 | 1.064, 1.060 | 4 -> 2 | 0.0713-0.0742 (3647) | 1.014 |
| 200W | nrcb1 | 2 | 0.0759 (2437), 0.0741 (3618) | 0.0738, 0.0720 | 1.028, 1.029 | 3 -> 2 | 0.0692-0.0749 (3806) | 0.999 |
| 200W | nrcb3 | 2 | 0.0777 (2440), 0.0758 (3631) | 0.0729, 0.0711 | 1.067, 1.066 | 4 -> 2 | 0.0707-0.0739 (3794) | 1.016 |
| 200W | nrcb1 | 3 | 0.0758 (2437), 0.0742 (3618) | 0.0739, 0.0735 | 1.025, 1.010 | 4 -> 2 | 0.0700-0.0745 (3473) | 1.002 |
| 200W | nrcb3 | 3 | 0.0778 (2440), 0.0759 (3631) | 0.0729, 0.0707 | 1.067, 1.073 | 4 -> 2 | 0.0704-0.0743 (3765) | 1.017 |
| 200W | nrcb1 | 4 | 0.0760 (2437), 0.0746 (3618) | 0.0736, 0.0735 | 1.033, 1.016 | 4 -> 2 | 0.0699-0.0747 (3205) | 1.007 |
| 200W | nrcb3 | 4 | 0.0776 (2440), 0.0757 (3631) | 0.0725, 0.0727 | 1.070, 1.042 | 4 -> 2 | 0.0717-0.0744 (3288) | 1.015 |
| 250M | nrcblong | 1 | 0.0616 (2419), 0.0590 (3537) | 0.0583, 0.0575 | 1.057, 1.026 | 6 -> 2 | 0.0575-0.0625 (2844) | 0.943 |
| 250M | nrcblong | 2 | 0.0615 (2419), 0.0587 (3537) | 0.0580, 0.0522 | 1.061, 1.124 | 6 -> 2 | 0.0500-0.0623 (3883) | 0.938 |
| 250M | nrcblong | 3 | 0.0610 (2419), 0.0587 (3537) | 0.0570, 0.0562 | 1.070, 1.045 | 6 -> 2 | 0.0562-0.0625 (2690) | 0.939 |
| 250M | nrcblong | 4 | 0.0613 (2419), 0.0588 (3537) | 0.0562, 0.0526 | 1.091, 1.118 | 6 -> 2 | 0.0526-0.0624 (3123) | 0.940 |
| 300M | nrcblong | 1 | 0.0360 (2419), 0.0345 (3538) | 0.0345, 0.0337 | 1.044, 1.024 | 6 -> 2 | 0.0331-0.0364 (3385) | 0.951 |
| 300M | nrcblong | 2 | 0.0362 (2419), 0.0345 (3538) | 0.0346, 0.0335 | 1.047, 1.030 | 6 -> 2 | 0.0333-0.0363 (3763) | 0.951 |
| 300M | nrcblong | 3 | 0.0361 (2419), 0.0345 (3538) | 0.0344, 0.0336 | 1.050, 1.028 | 6 -> 2 | 0.0332-0.0365 (3028) | 0.951 |
| 300M | nrcblong | 4 | 0.0361 (2419), 0.0345 (3539) | 0.0342, 0.0320 | 1.054, 1.078 | 6 -> 2 | 0.0315-0.0365 (3761) | 0.950 |

Median pipeline/wingmig ratio at the first kept bin (g0 ~ 2400 DN) by band and detector:

- 150W nrcb1: 1.027 (range 1.026-1.032, 4 frames)
- 150W nrcb3: 1.075 (range 1.074-1.082, 4 frames)
- 200W nrcb1: 1.027 (range 1.025-1.033, 4 frames)
- 200W nrcb3: 1.067 (range 1.064-1.070, 4 frames)
- 250M nrcblong: 1.066 (range 1.057-1.091, 4 frames)
- 300M nrcblong: 1.049 (range 1.044-1.054, 4 frames)

## 8. LW q tables

### LW q = cal / (R(g0) g0) vs distance d from the DQ-SATURATED edge (wingmig method, g0 in the R curve range 200 DN - top bin)

- F250M_1_nrcblong: 8 R bins, g0 239-2844 DN, R [0.0609, 0.0623, 0.0625, 0.0623, 0.0617, 0.0607, 0.0592, 0.0575]
- F250M_2_nrcblong: 9 R bins, g0 238-3883 DN, R [0.0607, 0.062, 0.0623, 0.062, 0.0617, 0.0603, 0.0593, 0.0571, 0.05]
- F250M_3_nrcblong: 8 R bins, g0 238-2690 DN, R [0.0606, 0.0616, 0.0625, 0.0619, 0.0617, 0.0608, 0.0586, 0.0562]
- F250M_4_nrcblong: 8 R bins, g0 240-3123 DN, R [0.0608, 0.0622, 0.0624, 0.062, 0.0611, 0.0597, 0.0575, 0.0526]
- F300M_1_nrcblong: 11 R bins, g0 229-3385 DN, R [0.0331, 0.0349, 0.0359, 0.0364, 0.0362, 0.0363, 0.0361, 0.0354, 0.035, 0.0343, 0.0337]
- F300M_2_nrcblong: 11 R bins, g0 230-3763 DN, R [0.0333, 0.0353, 0.0362, 0.0362, 0.0362, 0.0363, 0.036, 0.0354, 0.035, 0.0339, 0.0334]
- F300M_3_nrcblong: 10 R bins, g0 231-3028 DN, R [0.0332, 0.0351, 0.0361, 0.0364, 0.0362, 0.0362, 0.0359, 0.0354, 0.0346, 0.0336]
- F300M_4_nrcblong: 11 R bins, g0 230-3761 DN, R [0.0332, 0.0351, 0.0362, 0.0365, 0.0363, 0.036, 0.0357, 0.0353, 0.0347, 0.0335, 0.0315]

#### F250M: median q (bootstrap err; N star-frames)

| group | 1 | 2 | 3 | 4-5 | 6-8 | 9-12 | 13-20 |
|---|---|---|---|---|---|---|---|
| all | 1.044+-0.001 (3483) | 1.032+-0.003 (1998) | 1.035+-0.003 (1519) | 1.020+-0.003 (1381) | 1.015+-0.002 (1541) | 1.007+-0.002 (1681) | 1.000+-0.002 (2333) |
| Q1 faint | 1.026+-0.002 (1184) | 0.976+-0.006 (460) | 0.995+-0.006 (143) | 0.985+-0.006 (213) | 0.984+-0.005 (340) | 0.975+-0.006 (436) | 0.975+-0.004 (708) |
| Q2 | 1.035+-0.002 (1046) | 0.985+-0.008 (328) | 0.984+-0.009 (301) | 1.004+-0.007 (285) | 1.003+-0.006 (357) | 1.003+-0.004 (461) | 0.996+-0.003 (709) |
| Q3 | 1.053+-0.002 (723) | 1.022+-0.003 (680) | 1.008+-0.005 (547) | 1.007+-0.005 (362) | 1.004+-0.006 (327) | 1.009+-0.005 (345) | 1.007+-0.003 (500) |
| Q4 bright | 1.152+-0.005 (530) | 1.124+-0.004 (530) | 1.102+-0.005 (528) | 1.070+-0.005 (521) | 1.043+-0.004 (517) | 1.038+-0.005 (439) | 1.028+-0.004 (416) |

F250M control stars (unsaturated, peak g0 > 1500 DN), median q by radius from peak (px):

| 1-2 | 2-3 | 3-4 | 4-6 | 6-9 | 3-9 |
|---|---|---|---|---|---|
| 1.048+-0.002 (1160) | 1.009+-0.003 (759) | 0.992+-0.007 (191) | 1.001+-0.005 (185) | 0.981+-0.009 (284) | 0.982+-0.006 (422) |

#### F300M: median q (bootstrap err; N star-frames)

| group | 1 | 2 | 3 | 4-5 | 6-8 | 9-12 | 13-20 |
|---|---|---|---|---|---|---|---|
| all | 1.059+-0.001 (3810) | 1.046+-0.001 (3679) | 1.034+-0.002 (2904) | 1.017+-0.002 (2786) | 1.007+-0.002 (2471) | 0.998+-0.002 (2517) | 0.993+-0.002 (2863) |
| Q1 faint | 1.033+-0.002 (1171) | 1.019+-0.002 (1134) | 0.982+-0.004 (561) | 0.954+-0.005 (477) | 0.965+-0.005 (529) | 0.970+-0.007 (597) | 0.978+-0.005 (793) |
| Q2 | 1.039+-0.002 (989) | 1.024+-0.003 (901) | 0.987+-0.004 (706) | 0.979+-0.003 (713) | 0.978+-0.004 (595) | 0.979+-0.003 (618) | 0.979+-0.003 (733) |
| Q3 | 1.077+-0.002 (870) | 1.054+-0.002 (864) | 1.042+-0.002 (857) | 1.016+-0.002 (818) | 0.994+-0.004 (618) | 0.985+-0.004 (579) | 0.994+-0.005 (653) |
| Q4 bright | 1.133+-0.003 (780) | 1.111+-0.003 (780) | 1.098+-0.003 (780) | 1.087+-0.003 (778) | 1.059+-0.003 (729) | 1.047+-0.003 (723) | 1.027+-0.004 (684) |

F300M control stars (unsaturated, peak g0 > 1500 DN), median q by radius from peak (px):

| 1-2 | 2-3 | 3-4 | 4-6 | 6-9 | 3-9 |
|---|---|---|---|---|---|
| 1.045+-0.002 (1169) | 1.037+-0.002 (1086) | 1.014+-0.003 (851) | 0.985+-0.006 (369) | 0.984+-0.007 (480) | 1.003+-0.003 (933) |

## 9. Caveats

- Matching and fits: dm is relative to dolphot, with the ZP and the catalogue-level cap logic of earlier rounds. The star lists differ between bands (1471, 515, 905, 944 stars), and per-bin counts below 5 stars are blank.
- Variants are computed by linear re-solves at the pipeline's fitted (x, y) with neighbour subtraction; no re-fit of position. Per-star shifts are medians over the star's rows.
- The MAD of norim variants is 1.8-2.4 times the base value, so detector differences of order 0.01 mag in norim are within noise.
- rw12p was run with e1 errors only. The pipeline curve is reproduced from the pipeline's measured path (rim check 1.0000); the guard truncates it to two bins on every frame.
- Gain 2.0 e/DN (SW) and 1.8 (LW) in the e1 errors, and a group-0 noise of 8.6-11.9 DN estimated from the field, are assumptions.
- The per-pixel ratio uses F_dolphot from flux_fit x 10^(0.4 dm_final); this includes the PSF-normalization and aperture-level relation between our flux scale and dolphot, so the absolute ratio level (1.0-1.3) is not interpretable alone. The detector ratio (nrcb3/nrcb1) cancels a common scale factor but not detector-dependent PSF-grid differences.
- Pixel records cover r <= 30 px with PSF >= 1e-3 of peak. At r > 6 px rim-pixel ratios reflect additive offsets and neighbours and carry no weight in the conclusions.
- The "newly rewritten" pixel q uses the pixels remaining in the fit region (not masked, not rim); the number of such pixels is small (0.00004-0.00013 of fit pixels), so per-bin medians for them rest on few stars.
- The round-4 F250M jobs hit the 3000 s timeout once under heavy machine load and were rerun with the same command; results come from the complete rerun.
- LW has no detector split (nrcblong only). The mechanism statements in sections 4 and 5 are inferences from the measured tables.
