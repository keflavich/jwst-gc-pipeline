# Why do the brightest LW satstar bins read faint against dolphot? (wd2, issue #1142)

Read-only diagnostic. All files are in `/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/bright13/`. Nothing outside this directory was modified.
Convention: dm = ours - dolphot - ZP (positive = ours fainter). Stars are the `have & 10 <= ref < 13` rows of `out7/score7_250M.pkl` and `score7_300M.pkl`, mapped to per-frame rows exactly as `score7.py` does (0.1 arcsec match). "Measured" means computed from the files; "Inferred" means interpretation.

## Summary

1. (Measured) The "10-13 mag" bin is 12.40-13.00 mag in F250M (28 stars) and 12.71-13.00 mag in F300M (19 stars). The dolphot catalogue has no F250M star brighter than 12.396 and no F300M star brighter than 12.709, so 10-12 mag is empty. The bin is a 0.6 mag wide strip at dolphot's bright limit.
2. (Measured) Median dm in the bin: F250M final +0.125, H+cap +0.065 (SE ~0.014), H+bgfree+cap +0.081, H+h0+bgfree+cap +0.082; F300M final +0.116, H+cap +0.072, H+bgfree+cap +0.074, H+h0+bgfree+cap +0.079. The same stars with the recovered-core cap removed ("uncapped", the pre-cap catalogue amplitude a_cat) read -0.032 (F250M, SE ~0.025) and -0.044 (F300M, SE ~0.026). The cap accounts for essentially the whole offset: the cap shift (final minus uncapped) is +0.25/+0.22/+0.15 mag in F250M bins 12.3-12.7/12.7-12.85/12.85-13.0.
3. (Measured) The cap binds in 82 of 98 F250M star-frames (64 of 72 F300M) with median a_raw/a_cat = 0.87. Across all LW satstar rows the median cap_H/a_H (>1 means not binding) is 1.06 / 1.03 / 1.00 / 1.00 / 0.96 / 0.87 for dolphot bins 16-17 / 15-16 / 14-15 / 13.5-14 / 13-13.5 / 12-13 (F250M; F300M 1.055 / 1.027 / 1.004 / 0.997 / 0.956 / 0.874). The 13-17 bins read within +-0.02 because the cap rarely binds there.
4. (Measured) Where the cap binds, the recovered (group-0 based) core pixels sit below the PSF scaled to the wings. The brightest recovered pixel lies ~0.5 px from the fit position; its data / model(a_H) is 0.86 (F250M) and 0.86 (F300M) at 12-13 mag versus 0.955 / 0.965 at 14-15 mag. The group-0 value at that pixel is 1.02 x the R-curve ceiling at 12-13 mag versus 0.56-0.58 x at 14-15 mag. The ratio data/model in r < 2 px falls monotonically with brightness (F250M: 1.00, 0.96, 0.93, 0.90, 0.90, 0.88 for 16-17 ... 12-13) and with group-0 peak / ceiling (0.97, 0.93, 0.91, 0.90, 0.87 for the g0 bins in the second table below). Spearman rho(cap_H/a_H, g0 peak / ceiling) = -0.50 (F250M), -0.48 (F300M) over all rows.
5. (Inferred) The faint reading is produced by the recovered-core cap, which is tightest for stars whose group-0 core pixels reach the ceiling (the group-0 plateau). Either (A) R x g0 under-recovers the peak pixels when group 0 sits at the ceiling, so the cap is too low and dolphot (and our uncapped fit) is closer to the truth, or (B) the true peak is below the wing-scaled PSF for bright stars (detector nonlinearity, brighter-fatter, PSF-core mismatch), so the wing-based amplitude and dolphot are both biased bright and the cap is right. The data here do not separate A from B (see "Dolphot's side"). Part of the effect (data/model of ~0.93-0.96 at r<2 already at 14-16 mag, where the cap does not bind) is common to all stars and independent of the ceiling.
6. (Measured) No catalogue property correlates with dm (H+cap) at p < 0.1 in F250M: sat_area, own SATURATED pixel count, fraction of core pixels with group 0 SATURATED/DO_NOT_USE, fit pixel count, rim pixel count in the fit, qfit, dolphot-fit position offset (all |rho| <= 0.28). In F300M dm correlates with cap_H/a_H (rho -0.62) and a_raw/a_cat (-0.53), which follows from how the cap sets the amplitude, and with the amplitude (-0.49, p = 0.03). The median own saturated area is 118 px (F250M) and 139 px (F300M); the median fraction with group 0 SATURATED/DO_NOT_USE is 0.12.


## Task 1/2: tables

### dm by dolphot magnitude (median, MAD in parentheses; H+cap | final | uncapped)

| band | bin | N | H+cap | final | uncapped |
|---|---|---|---|---|---|
| F250M | 12.3-12.7 | 3 | +0.099 | +0.160 | -0.019 |
| F250M | 12.7-12.85 | 11 | +0.086 (0.106) | +0.138 (0.096) | -0.056 (0.139) |
| F250M | 12.85-13 | 14 | +0.050 (0.064) | +0.108 (0.065) | -0.026 (0.082) |
| F250M | 13-13.5 | 27 | -0.003 (0.038) | +0.051 (0.037) | -0.019 (0.043) |
| F250M | 13.5-14 | 53 | -0.023 (0.034) | +0.032 (0.031) | +0.007 (0.044) |
| F250M | 14-15 | 189 | -0.026 (0.033) | +0.021 (0.032) | -0.001 (0.036) |
| F300M | 12.7-12.85 | 8 | +0.102 (0.124) | +0.144 (0.117) | -0.104 (0.170) |
| F300M | 12.85-13 | 11 | +0.072 (0.039) | +0.116 (0.040) | -0.035 (0.077) |
| F300M | 13-13.5 | 27 | -0.007 (0.029) | +0.034 (0.031) | -0.027 (0.053) |
| F300M | 13.5-14 | 70 | -0.018 (0.033) | +0.020 (0.036) | -0.004 (0.041) |
| F300M | 14-15 | 232 | -0.021 (0.033) | +0.014 (0.030) | -0.003 (0.040) |

Within 12.4-13 the offset grows toward the bright end (F250M +0.050, +0.086, +0.099 for the three bins, 3 + 11 + 14 stars; F300M +0.072, +0.102); with N = 3-14 per bin the trend is of order the bin SE (~0.02-0.03), so the monotonic trend is suggestive, not established. The 10-11 and 11-12 mag bins contain no stars. Bin 12.3-13.0 medians: F250M H+cap +0.065 (SE 0.014, MAD 0.059), F300M +0.072 (SE 0.015, MAD 0.050).

### F250M versus F300M and the SW bands

* The two LW bands agree: the same shift at fixed dolphot magnitude (+0.065 vs +0.072 H+cap, +0.125 vs +0.116 final), and the same ~0.87 cap ratio. For the 8 stars with ref(F250M) in 12.3-13 that are refit in both bands, dm(F250M) - dm(F300M) is -0.065 (H+cap) and +0.051 (uncapped); at 13-13.5 (N = 25) -0.031 and +0.016; at 14-17 within +-0.015.
* SW bands (score7, same variants): F150W has no star brighter than dolphot 14.45 and F200W none brighter than 13.90, so SW bins do not overlap 12.4-13. The brightest SW bins read F150W 14-15 H+cap -0.056 (final -0.095, N = 57), F200W 13.5-14 +0.008 (final +0.007, N = 8), F200W 14-15 -0.031 (final -0.071, N = 157). The SW brightest bins read bright or neutral; they do not show the +0.07 shift, and they sit 1.5-2 mag fainter than the LW stars in this report.

### Per-star tables (task 1) and correlations
Median over the frame rows of each star. dolphot mag = ref (the ecsv magnitude, identical); dolphot err 0.000 and no flag columns exist (see below). a_raw/a_cat is the cap binding (1 = not binding); sat_area is the catalogue value from `*_resbgsub_m7_satstar_catalog.fits`; nsat = own-source DQ SATURATED pixels in the fit window; f_g0sat = fraction of those with group 0 SATURATED or DO_NOT_USE; n_rim = recovered rim pixels within 12 px of the saturated region; frac rim g0>ceil = share of those with g0 above the R-curve ceiling.

### F250M: 28 stars, 104 frame rows

Per-star table (median over the frame rows): ref, dm(final), dm(H+cap), sat_area, nsat (own SATURATED px), g0-sat fraction, rim px near, a_raw/a_cat (cap binding), cap_H/a_H, nfit, nrim_fit, rim px with g0>ceiling

| istar | ref | dm final | dm H+cap | sat_area | nsat | f_g0sat | n_rim | a_raw/a_cat | cap_H/a_H | nfit | nrim_fit | frac rim g0>ceil | qfit | nrows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1019 | 12.825 | -0.994 | -1.019 | 121 | 121 | 0.14 | 287 | 1.000 | nan | 6299 | 1652 | 0.00 | 3.370 | 1 |
| 1014 | 12.701 | +0.621 | +0.563 | 154 | 154 | 0.10 | 161 | 0.308 | 0.320 | 6442 | 432 | 0.00 | 1.033 | 4 |
| 1016 | 12.738 | +0.268 | +0.209 | 136 | 136 | 0.12 | 132 | 0.776 | 0.792 | 6366 | 115 | 0.01 | 0.130 | 4 |
| 519 | 12.640 | +0.234 | +0.174 | 138 | 138 | 0.11 | 131 | 0.792 | 0.807 | 6448 | 83 | 0.00 | 0.045 | 3 |
| 348 | 12.973 | +0.225 | +0.167 | 114 | 114 | 0.12 | 124 | 0.743 | 0.751 | 6520 | 146 | 0.01 | 0.485 | 4 |
| 198 | 12.782 | +0.212 | +0.166 | 138 | 138 | 0.11 | 136 | 0.930 | 0.945 | 6488 | 147 | 0.01 | 0.643 | 4 |
| 1021 | 12.818 | +0.163 | +0.106 | 123 | 123 | 0.10 | 122 | 0.818 | 0.834 | 6426 | 149 | 0.01 | 0.316 | 4 |
| 1024 | 12.923 | +0.162 | +0.104 | 70 | 70 | 0.16 | 232 | 0.900 | 0.904 | 6516 | 530 | 0.00 | 2.308 | 1 |
| 299 | 12.918 | -0.046 | -0.103 | 118 | 118 | 0.12 | 120 | 0.873 | 0.815 | 6484 | 89 | 0.00 | 0.221 | 4 |
| 1013 | 12.591 | +0.160 | +0.099 | 84 | 84 | 0.19 | 80 | 0.898 | 0.909 | 6542 | 140 | 0.02 | 0.426 | 4 |
| 1022 | 12.913 | +0.157 | +0.099 | 81 | 81 | 0.14 | 80 | 0.868 | 0.880 | 6492 | 80 | 0.00 | 0.228 | 4 |
| 1023 | 12.927 | +0.157 | +0.099 | 114 | 114 | 0.14 | 113 | 0.853 | 0.862 | 6460 | 165 | 0.01 | 0.589 | 4 |
| 1017 | 12.777 | +0.138 | +0.087 | 78 | 78 | 0.14 | 78 | 0.917 | 0.877 | 6546 | 256 | 0.01 | 1.548 | 4 |
| 249 | 12.708 | +0.144 | +0.086 | 96 | 96 | 0.16 | 100 | 0.854 | 0.874 | 6444 | 150 | 0.02 | 0.353 | 4 |
| 1026 | 12.959 | +0.136 | +0.076 | 74 | 74 | 0.13 | 74 | 0.892 | 0.901 | 6560 | 142 | 0.00 | 0.899 | 4 |
| 1028 | 12.981 | -0.022 | -0.071 | 112 | 112 | 0.12 | 110 | 0.944 | 0.861 | 6474 | 154 | 0.01 | 0.529 | 4 |
| 1015 | 12.720 | +0.127 | +0.069 | 154 | 154 | 0.13 | 150 | 0.776 | 0.800 | 6504 | 197 | 0.00 | 0.453 | 4 |
| 201 | 12.944 | +0.122 | +0.062 | 109 | 109 | 0.12 | 109 | 0.866 | 0.872 | 6530 | 318 | 0.01 | 1.096 | 4 |
| 234 | 12.901 | +0.108 | +0.052 | 126 | 126 | 0.10 | 129 | 0.960 | 0.966 | 6544 | 648 | 0.01 | 2.884 | 4 |
| 1027 | 12.953 | +0.108 | +0.047 | 124 | 124 | 0.11 | 156 | 0.943 | 0.944 | 6558 | 691 | 0.01 | 3.770 | 3 |
| 999 | 12.936 | +0.102 | +0.043 | 136 | 136 | 0.09 | 135 | 0.871 | 0.882 | 6530 | 135 | 0.01 | 0.442 | 4 |
| 1020 | 12.864 | +0.020 | -0.039 | 118 | 118 | 0.11 | 122 | 0.854 | 0.863 | 6510 | 363 | 0.00 | 1.032 | 4 |
| 194 | 12.801 | +0.095 | +0.038 | 132 | 132 | 0.12 | 136 | 0.824 | 0.848 | 6525 | 566 | 0.01 | 1.800 | 4 |
| 209 | 12.995 | +0.084 | +0.027 | 112 | 112 | 0.11 | 166 | 0.913 | 0.927 | 6515 | 170 | 0.01 | 0.447 | 4 |
| 1012 | 12.396 | +0.078 | +0.020 | 102 | 102 | 0.17 | 122 | 0.759 | 0.784 | 6494 | 250 | 0.00 | 0.377 | 4 |
| 1018 | 12.821 | +0.071 | +0.015 | 121 | 121 | 0.11 | 132 | 0.814 | 0.834 | 6532 | 207 | 0.01 | 0.652 | 4 |
| 195 | 12.803 | +0.073 | +0.014 | 70 | 70 | 0.18 | 67 | 0.910 | 0.908 | 6556 | 156 | 0.01 | 0.446 | 4 |
| 298 | 12.889 | +0.069 | +0.013 | 125 | 125 | 0.10 | 125 | 0.876 | 0.884 | 6465 | 126 | 0.01 | 0.335 | 4 |

Spearman rho (p) of dm vs property, excluding stars with |dm_H+cap| > 0.5 as outliers noted separately:
(n = 26 of 28; outliers removed: [(1014, 0.56), (1019, -1.02)])

| property | rho vs dm_H+cap (p) | rho vs dm_final (p) | rho vs dm_uncapped (p) |
|---|---|---|---|
| ref | -0.25 (0.21) | -0.25 (0.22) | +0.07 (0.73) |
| sat_area | +0.09 (0.68) | +0.08 (0.71) | -0.18 (0.38) |
| nsat | +0.09 (0.68) | +0.08 (0.71) | -0.18 (0.38) |
| sat_frac_g0 | +0.10 (0.64) | +0.11 (0.59) | +0.04 (0.86) |
| n_zfdeep | -0.19 (0.35) | -0.21 (0.31) | -0.51 (0.01) |
| n_rim_near | +0.07 (0.73) | +0.07 (0.75) | +0.03 (0.88) |
| rimabove_frac | +0.11 (0.60) | +0.10 (0.62) | +0.19 (0.36) |
| core_above_frac | -0.09 (0.68) | -0.08 (0.68) | -0.22 (0.28) |
| g0max_core | -0.01 (0.97) | -0.01 (0.97) | -0.40 (0.04) |
| capbind | -0.25 (0.22) | -0.25 (0.22) | +0.61 (0.00) |
| capH_ratio | -0.07 (0.73) | -0.06 (0.76) | +0.72 (0.00) |
| nfit | -0.23 (0.26) | -0.24 (0.25) | +0.25 (0.22) |
| nrim_fit | -0.28 (0.17) | -0.28 (0.16) | +0.04 (0.85) |
| qfit | -0.15 (0.45) | -0.17 (0.41) | +0.28 (0.16) |
| our_qfit | -0.25 (0.22) | -0.25 (0.22) | -0.17 (0.40) |
| ecsv_sep_mas | +0.25 (0.21) | +0.25 (0.22) | +0.06 (0.78) |
| amp | -0.27 (0.18) | -0.27 (0.19) | -0.42 (0.03) |

Medians: dm_final=0.125, dm_H+cap=0.065, dm_H+bgfree+cap=0.072, dm_uncapped=-0.029
Medians of frame-level diagnostics: sat_area=118, nsat=118, sat_frac_g0=0.12, n_zfdeep=0, n_rim_near=124, capbind=0.87, rimabove_frac=0.008, core_above_frac=0.00939, g0max_core=4.34e+04, ceiling=4.19e+04, curve_top=3.54e+03

### F300M: 19 stars, 72 frame rows

Per-star table (median over the frame rows): ref, dm(final), dm(H+cap), sat_area, nsat (own SATURATED px), g0-sat fraction, rim px near, a_raw/a_cat (cap binding), cap_H/a_H, nfit, nrim_fit, rim px with g0>ceiling

| istar | ref | dm final | dm H+cap | sat_area | nsat | f_g0sat | n_rim | a_raw/a_cat | cap_H/a_H | nfit | nrim_fit | frac rim g0>ceil | qfit | nrows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1126 | 12.920 | +0.438 | +0.395 | 170 | 170 | 0.12 | 224 | 0.643 | 0.677 | 6518 | 892 | 0.01 | 3.323 | 2 |
| 1058 | 12.732 | +0.286 | +0.243 | 155 | 155 | 0.12 | 146 | 0.781 | 0.816 | 6373 | 110 | 0.00 | 0.272 | 4 |
| 258 | 12.846 | +0.249 | +0.204 | 138 | 138 | 0.12 | 210 | 0.734 | 0.751 | 6512 | 186 | 0.00 | 0.453 | 4 |
| 1120 | 12.709 | +0.191 | +0.167 | 102 | 102 | 0.17 | 111 | 0.723 | 0.754 | 6494 | 360 | 0.00 | 1.055 | 4 |
| 1123 | 12.800 | +0.194 | +0.150 | 102 | 102 | 0.13 | 102 | 0.851 | 0.856 | 6494 | 102 | 0.00 | 0.240 | 4 |
| 263 | 12.960 | +0.150 | +0.106 | 146 | 146 | 0.09 | 182 | 0.885 | 0.885 | 6546 | 942 | 0.01 | 3.163 | 4 |
| 1109 | 12.989 | +0.145 | +0.100 | 144 | 144 | 0.12 | 227 | 0.801 | 0.807 | 6559 | 486 | 0.00 | 1.429 | 4 |
| 368 | 12.956 | +0.143 | +0.099 | 93 | 93 | 0.12 | 93 | 0.853 | 0.857 | 6524 | 231 | 0.00 | 1.181 | 4 |
| 788 | 12.949 | +0.117 | +0.074 | 88 | 88 | 0.15 | 85 | 0.886 | 0.892 | 6525 | 134 | 0.01 | 0.417 | 4 |
| 1107 | 12.976 | +0.116 | +0.072 | 88 | 88 | 0.15 | 122 | 0.947 | 0.950 | 6561 | 184 | 0.00 | 0.992 | 3 |
| 1110 | 12.996 | +0.108 | +0.065 | 142 | 142 | 0.08 | 142 | 0.803 | 0.815 | 6524 | 142 | 0.01 | 0.238 | 4 |
| 1127 | 12.870 | +0.103 | +0.060 | 138 | 138 | 0.13 | 135 | 0.882 | 0.891 | 6482 | 226 | 0.01 | 0.699 | 4 |
| 259 | 12.997 | +0.103 | +0.058 | 120 | 120 | 0.12 | 144 | 0.935 | 0.937 | 6484 | 162 | 0.01 | 0.451 | 3 |
| 1124 | 12.840 | +0.098 | +0.054 | 150 | 150 | 0.13 | 149 | 0.918 | 0.922 | 6462 | 251 | 0.01 | 0.839 | 4 |
| 1122 | 12.724 | -0.028 | -0.048 | 150 | 150 | 0.11 | 184 | 0.883 | 0.929 | 6480 | 498 | 0.00 | 1.024 | 4 |
| 367 | 12.829 | +0.091 | +0.047 | 146 | 146 | 0.10 | 137 | 0.830 | 0.849 | 6443 | 214 | 0.00 | 0.696 | 4 |
| 1121 | 12.778 | +0.005 | -0.038 | 146 | 146 | 0.11 | 160 | 0.798 | 0.817 | 6504 | 234 | 0.00 | 0.583 | 4 |
| 1129 | 12.983 | +0.082 | +0.038 | 113 | 113 | 0.13 | 113 | 0.879 | 0.885 | 6526 | 427 | 0.01 | 1.388 | 4 |
| 1130 | 12.992 | +0.071 | +0.027 | 122 | 122 | 0.12 | 122 | 0.900 | 0.905 | 6547 | 238 | 0.01 | 0.669 | 4 |

Spearman rho (p) of dm vs property, excluding stars with |dm_H+cap| > 0.5 as outliers noted separately:
(n = 19 of 19; outliers removed: [])

| property | rho vs dm_H+cap (p) | rho vs dm_final (p) | rho vs dm_uncapped (p) |
|---|---|---|---|
| ref | -0.13 (0.60) | -0.12 (0.62) | +0.31 (0.19) |
| sat_area | +0.00 (0.99) | +0.00 (0.99) | -0.34 (0.16) |
| nsat | +0.00 (0.99) | +0.00 (0.99) | -0.34 (0.16) |
| sat_frac_g0 | +0.21 (0.38) | +0.21 (0.40) | +0.19 (0.45) |
| n_zfdeep | +0.08 (0.76) | +0.05 (0.84) | -0.61 (0.01) |
| n_rim_near | +0.08 (0.74) | +0.08 (0.74) | -0.31 (0.20) |
| rimabove_frac | -0.15 (0.54) | -0.14 (0.57) | +0.15 (0.54) |
| core_above_frac | -0.05 (0.82) | -0.08 (0.74) | -0.55 (0.02) |
| g0max_core | -0.20 (0.41) | -0.22 (0.37) | -0.68 (0.00) |
| capbind | -0.53 (0.02) | -0.51 (0.02) | +0.68 (0.00) |
| capH_ratio | -0.62 (0.01) | -0.61 (0.01) | +0.60 (0.01) |
| nfit | +0.06 (0.80) | +0.06 (0.80) | +0.08 (0.76) |
| nrim_fit | -0.14 (0.56) | -0.16 (0.50) | -0.46 (0.05) |
| qfit | +0.07 (0.79) | +0.05 (0.85) | -0.25 (0.30) |
| our_qfit | -0.28 (0.24) | -0.27 (0.27) | -0.31 (0.20) |
| ecsv_sep_mas | +0.15 (0.53) | +0.15 (0.54) | -0.24 (0.31) |
| amp | -0.49 (0.03) | -0.50 (0.03) | -0.21 (0.38) |

Medians: dm_final=0.116, dm_H+cap=0.072, dm_H+bgfree+cap=0.074, dm_uncapped=-0.044
Medians of frame-level diagnostics: sat_area=139, nsat=139, sat_frac_g0=0.118, n_zfdeep=0, n_rim_near=140, capbind=0.866, rimabove_frac=0.00551, core_above_frac=0.0106, g0max_core=4.35e+04, ceiling=4.19e+04, curve_top=3.54e+03

## Task 1/3: radial and group-0 evidence (all LW satstar rows, `stage5.py`)

### F250M: per-frame-row medians by dolphot mag (all 3188 mapped satstar rows)

| ref bin | N rows | cap_H/a_H | a_raw/a_cat | data/model(a_H) r<2 | 2-3 | 3-4 | 4-5 | g0 peak (r<3) / ceiling | frac sat px (r<=8) with g0 sat | n sat px r<=8 |
|---|---|---|---|---|---|---|---|---|---|---|
| 12-13 | 104 | 0.874 | 0.870 | 0.880 | 1.087 | 1.025 | 1.185 | 1.04 | 0.13 | 118 |
| 13-13.5 | 105 | 0.963 | 0.949 | 0.900 | 1.038 | 1.084 | 1.308 | 1.01 | 0.14 | 91 |
| 13.5-14 | 203 | 1.002 | 0.989 | 0.902 | 1.060 | 1.187 | 1.611 | 0.92 | 0.26 | 67 |
| 14-15 | 723 | 1.005 | 0.985 | 0.931 | 1.085 | 1.441 | 1.952 | 0.56 | 0.20 | 46 |
| 15-16 | 1353 | 1.033 | 1.000 | 0.959 | 1.238 | 2.056 | 2.875 | 0.25 | 0.06 | 20 |
| 16-17 | 699 | 1.063 | 1.000 | 1.004 | 1.413 | 2.889 | 4.305 | 0.13 | 0.11 | 9 |

F250M: same medians binned by group-0 peak / ceiling (all rows)

| g0pk/ceiling | N | cap_H/a_H | data/model(a_H) r<2 | 3-4 | median ref mag |
|---|---|---|---|---|---|
| 0-0.5 | 2293 | 1.039 | 0.968 | 2.219 | 15.65 |
| 0.5-0.8 | 470 | 1.002 | 0.927 | 1.357 | 14.42 |
| 0.8-0.95 | 141 | 1.003 | 0.906 | 1.206 | 13.88 |
| 0.95-1.05 | 223 | 0.954 | 0.898 | 1.088 | 13.13 |
| 1.05-1.2 | 61 | 0.989 | 0.871 | 1.048 | 12.94 |
F250M 12-13 mag rows: with any group-0-saturated pixel within r<=3 of the fit position: n/a
F250M 12-13 mag rows: g0 at brightest recovered pixel / ceiling median 1.02; its data/model(a_H) median 0.860; r_dmax median 0.54 px
F250M 14-15 mag rows: g0 at brightest recovered pixel / ceiling median 0.56; its data/model(a_H) median 0.955
F250M Spearman cap_H/a_H vs g0pk/ceiling over all rows: -0.50; vs ref mag +0.54

### F300M: per-frame-row medians by dolphot mag (all 3357 mapped satstar rows)

| ref bin | N rows | cap_H/a_H | a_raw/a_cat | data/model(a_H) r<2 | 2-3 | 3-4 | 4-5 | g0 peak (r<3) / ceiling | frac sat px (r<=8) with g0 sat | n sat px r<=8 |
|---|---|---|---|---|---|---|---|---|---|---|
| 12-13 | 72 | 0.874 | 0.866 | 0.904 | 1.170 | 1.013 | 1.314 | 1.04 | 0.12 | 139 |
| 13-13.5 | 98 | 0.956 | 0.950 | 0.928 | 1.108 | 1.092 | 1.503 | 1.02 | 0.13 | 108 |
| 13.5-14 | 264 | 0.997 | 0.987 | 0.936 | 1.095 | 1.174 | 1.818 | 0.95 | 0.25 | 80 |
| 14-15 | 876 | 1.004 | 0.989 | 0.963 | 1.153 | 1.329 | 2.585 | 0.58 | 0.21 | 57 |
| 15-16 | 1443 | 1.027 | 1.000 | 0.996 | 1.328 | 1.820 | 4.579 | 0.26 | 0.05 | 25 |
| 16-17 | 603 | 1.055 | 1.000 | 1.050 | 1.583 | 2.408 | 7.422 | 0.13 | 0.11 | 12 |

F300M: same medians binned by group-0 peak / ceiling (all rows)

| g0pk/ceiling | N | cap_H/a_H | data/model(a_H) r<2 | 3-4 | median ref mag |
|---|---|---|---|---|---|
| 0-0.5 | 2282 | 1.032 | 1.004 | 1.888 | 15.62 |
| 0.5-0.8 | 587 | 1.007 | 0.959 | 1.304 | 14.50 |
| 0.8-0.95 | 195 | 0.997 | 0.938 | 1.179 | 13.93 |
| 0.95-1.05 | 243 | 0.972 | 0.931 | 1.124 | 13.43 |
| 1.05-1.2 | 50 | 0.977 | 0.918 | 1.078 | 13.31 |
F300M 12-13 mag rows: with any group-0-saturated pixel within r<=3 of the fit position: n/a
F300M 12-13 mag rows: g0 at brightest recovered pixel / ceiling median 1.01; its data/model(a_H) median 0.863; r_dmax median 0.50 px
F300M 14-15 mag rows: g0 at brightest recovered pixel / ceiling median 0.58; its data/model(a_H) median 0.965
F300M Spearman cap_H/a_H vs g0pk/ceiling over all rows: -0.48; vs ref mag +0.51


Radial profile of data / model for the 12-13 mag stars (`radial.py`, median over star-frame rows of the median in each annulus about the fit position; capped model amp = min(a_H, cap_H); the single-star model has no sky, so annuli beyond ~5 px are dominated by background and are listed for completeness):

| band | model | r 0-2 | 2-3 | 3-4 | 4-5 | 5-6 | 6-8 | 8-10 | 10-14 | 14-20 |
|---|---|---|---|---|---|---|---|---|---|---|
| F250M | capped | 1.026 | 1.267 | 1.250 | 1.402 | 1.500 | 2.427 | 2.548 | 3.604 | 7.486 |
| F250M | uncapped (a_H) | 0.881 | 1.091 | 1.032 | 1.198 | 1.266 | 2.080 | 2.105 | 2.937 | 6.449 |
| F300M | capped | 1.027 | 1.380 | 1.224 | 1.611 | 1.667 | 2.221 | 4.677 | 4.856 | 12.526 |
| F300M | uncapped (a_H) | 0.904 | 1.170 | 1.013 | 1.314 | 1.410 | 1.812 | 3.818 | 4.015 | 10.415 |

The uncapped model matches the recovered ring at 3-4 px (1.03, 1.01) and over-predicts the innermost recovered pixels by 10-12 %; the capped model under-predicts the ring by 22-25 %.

## Dolphot's side (task 3)

* (Measured) `wd2_nircam_wf_mf_nf.ecsv` has only RA, DEC, MAG<band>, ERRMAG<band> (34 columns). There are no flag, crowding, sharpness or chi columns. ERRMAG250M/300M is 0.000 (3-decimal rounding) for at least 99 % of stars at F250M < 15, so the tabulated errors carry no information at these magnitudes. Dolphot quality columns that would answer the question are not in the catalogue available here, and the matched files only add our own columns. The ecsv magnitudes equal `ref` exactly for all 47 stars (match separation 30-57 mas).
* (Measured) Dolphot has no measurement of these stars in the other bands: the brightest dolphot magnitudes are F115W 14.43, F150W 14.45, F200W 13.90, F277W 13.73, F335M 14.69 (F410M 11.82). A LW-SW colour comparison against 13-15 mag stars of the same SW colour is therefore impossible for these stars. Only 3 of the F250M stars (ids 299, 348, 519) and 4 of the F300M stars have a dolphot F150W/F200W value; their production (final) dm in those SW bands (F250M stars: -0.05 to -0.00; F300M stars: -0.08 to +0.07) shows no pattern with the LW production dm (+0.23 for 348 and 519, -0.05 for 299).
* (Measured) The only available dolphot colour is F250M-F300M. Median over all dolphot stars with both bands, by F250M bin (MAD): 15-17 mag +0.19 (0.10-0.12); 14.5-15 +0.175; 14-14.5 +0.160; 13.5-14 +0.139 (0.082); 13-13.5 +0.100 (0.073); 12.3-13 +0.067 (0.048, N = 20). The colour becomes ~0.12 mag bluer toward the bright end. Our pipeline shows the same direction for the same stars (stars refit in both bands, N = 818; F250M-F300M with the uncapped amplitude: +0.145 at 12.3-13, +0.131, +0.156, +0.208, +0.214, +0.235 at 16-17; with H+cap: +0.025, +0.088, +0.146, +0.200, +0.208, +0.228; dolphot for the same stars: +0.099, +0.085, +0.156, +0.187, +0.221, +0.214). Ours minus dolphot colour at 12.3-13 (N = 8) is +0.046 uncapped and -0.074 with H+cap, i.e. the capped F250M reads fainter than the F300M cap relative to dolphot, the uncapped colour is 0.05 redder.
* (Inferred) A colour trend that appears in both catalogues is consistent with a real population change (or with a shared saturation systematic in both), so the dolphot colour is not anomalous in a way that identifies dolphot as the unreliable side. The result is not a test of dolphot: its bright-end values come from PSF fits to the wings with saturated pixels masked, as ours do in the uncapped case, so agreement between dolphot and our uncapped amplitude (-0.03 to -0.04) shows that the two wing-based measurements agree. It does not show which is correct, since the recovered core pixels argue for ~13 % less flux than the wings (point 4 above). An independent flux anchor is needed: e.g. an unsaturated epoch, or a different filter where the same stars are unsaturated, none of which dolphot has for these stars.
* (Inferred) Counts per 0.5 mag in dolphot F250M: 1 (12-12.5), 38 (12.5-13), 68 (13-13.5), 89 (13.5-14), 365 per 1 mag at 14-15. The luminosity function falls off toward the bright limit; whether that is intrinsic or saturation truncation cannot be told. If dolphot clips its brightest stars (a saturated star measured too faint is moved into the 12.5-13 bin), dm would be biased toward negative values there, the opposite sign of the observed offset. Selecting on the dolphot magnitude with a steep luminosity function biases dm positive by about sigma^2 x 0.9 per mag (~0.01 for sigma = 0.1 mag), too small to explain +0.065.

## Task 4: cutouts

Figures (F250M, first frame row of each star; crf data on a log stretch, SATURATED outline of the star's own source in red, recovered rim pixels as blue squares, dolphot position green x, our fitted position orange +; middle column the PSF model with amplitude min(a_H, cap_H) via `satrefit_core.render` and the catalogue x_fit, y_fit; right column data minus that single-star model in linear stretch, white where the crf data are zero, i.e. the unrecovered deep core):

* `cutouts_F250M_1.png`: the largest |dm|: stars 1014 (+0.56, a_raw/a_cat 0.31), 1019 (-1.02), 1016 (+0.21), 519 (+0.17).
* `cutouts_F250M_2.png`: typical stars 1018, 195, 298, 234 (dm H+cap +0.01 to +0.05).

Observations (measured from the figures and tables): the dolphot and fitted positions agree to a median 0.12 px (p90 0.45 px, max 1.08 px; F300M median 0.15 px), so position error does not drive dm. Star 1014 has its saturated outline extended toward the lower right and a second recovered-rim source at about (87, 65), and 1019 sits in a bright extended background with several rim-recovered neighbours; both have qfit above 1 (1.03 and 3.4) and 1019 has a single row. They look like blends (inferred from the images), and the single-star model leaves large positive residuals. The model is single-star: neighbours are not subtracted in the right column, and the sky is not modelled, hence the positive pedestal. In the typical stars the recovered rim ring (r ~ 2-5 px) lies above the capped model while the central recovered pixels lie below the uncapped model; the radial table below quantifies this. data - model for the full PSF x amplitude model was built cheaply from the existing code, so it is shown.

## Conclusions

* The 7-8 % faint reading of the 10-13 mag LW stars is reproduced by the recovered-core cap and disappears (to -0.03/-0.04, 0.03-0.04 mag bright) when the cap is removed. The 12-13 mag stars are the ones for which group 0 at the peak pixels reaches the group-0 plateau ceiling (g0 / ceiling = 1.02-1.04 versus 0.56-0.58 at 14-15 mag), and for them the recovered peak pixel falls to 0.86 of the wing-scaled PSF.
* A 12-14 % deficit of the peak pixel relative to the wing-scaled PSF at g0 near the ceiling matches the cap amplitude ratio of 0.87 (cap_H/a_H, a_raw/a_cat).
* No saturation-geometry, crowding or fit-size property tested (sat_area, group-0-saturated fraction, nfit, nrim_fit, qfit, rim-above-ceiling fraction) shows a significant correlation with dm; the offset is a median shift of the whole bin (28 stars, MAD 0.06).
* Dolphot's quality cannot be assessed from the catalogue (no flag/crowd/sharp/chi columns, rounded errors, no other-band magnitudes for these stars). The data do not show the dolphot values at 12.4-13 mag to be the less reliable side, and they do not show them to be the more reliable side.
* Suggested next measurements (not run; outside this read-only scope): (1) refit with the cap peak read from recovered pixels with g0 < 0.9 x ceiling only (or the cap skipped when the peak pixel has g0 >= ceiling) and check that 13-17 mag bins stay within +-0.02; (2) measure the same stars' fluxes in an independent band or epoch where they are unsaturated (F277W/F335M/F410M ours vs dolphot, both saturated here, so use F410M or an MIRI/other catalogue if available) to decide between A and B.

## Files

* Scripts: `stage1.py` (star-to-row mapping, 10-13), `stage1all.py` (all `have` stars), `stage2.py` (per-frame cutouts and own-star stats, SLURM `run2.sh`), `stage5.py` (all rows: r<2/2-3/3-4/4-5 data/model, group-0 peak; `run5.sh`), `analyze2.py`, `analyze5.py`, `colors.py`, `colors2.py`, `color3.py`, `trend.py`, `radial.py`, `fig.py`.
* Tables: `star_250M.fits`, `star_300M.fits` (the 47 stars with dolphot values, dm of all variants, own catalogue columns), `rows_*.fits`, `s2_*.fits` (per frame row stats), `s5_*.fits`, `analysis_out.txt`, `analysis5_out.txt`, `joined.pkl`.
* Figures: `cutouts_F250M_1.png`, `cutouts_F250M_2.png`. No F300M cutouts were made (same procedure, `fig.py` with band set).

Limitations: only 28 and 19 stars, 4 frame rows each; the two bands share most stars, so the two columns are not independent; "uncapped" is the pre-cap catalogue amplitude a_cat with no rim rewrite (variant `uncapped` in score7); the data/model ratios use the pipeline-rewritten crf data (rim = R_curve x g0) rather than the H-variant rewrite, and the model is a single star with no sky or neighbours, so ratios beyond ~5 px are dominated by background.
