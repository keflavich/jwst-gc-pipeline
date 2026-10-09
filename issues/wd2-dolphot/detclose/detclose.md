# Per-detector aperture closure of the pipeline PSF fluxes (detclose)

## Question

Does the per-detector term in ours - dolphot (photdet.txt) originate in our PSF-fit fluxes, for example through the PSF model normalisation of each detector? The test compares the PSF-fit flux (flux_fit) with an aperture flux corrected by the model encircled energy, using only our own frames, catalogues and PSF grids.

## Method

- Frames: tree_main2/F<band>/pipeline/jw03523005001_*_<det>_align_o005_crf.fits (SCI in MJy/sr, DQ). Bands F150W, F200W, F182M, F212N (8 SW detectors x 4 exposures) and F250M, F410M (nrcalong, nrcblong x 4 exposures).
- PSF fluxes: per-frame catalogues tree_main2/F<band>/<band>_<det>_visit001_vgroup*_exp0000N_resbgsub_m7_daophot_basic.fits (the per-frame inputs of the m7/resbgsub merge). Columns x_fit, y_fit, flux_fit, flux_err, flags.
- PSF model: tree_main2/psfs/nircam_<det>_<band>_fovp101_samp2_npsf16.fits (nrca5/nrcb5 for LW), loaded with stpsf.utils.to_griddedpsfmodel, the same call and file family the photometry code uses for the PSFPhotometry fit (crowdsource_catalogs_long.py, fovp101 cache). The grid is used as stored. A flux = 1 render sums to 0.975 over the 101 px stamp, and flux_fit is the model amplitude. Closure compares amplitudes, so the grid normalisation enters through EE_model.
- Star selection (per frame): flags == 0, flux_fit/flux_err > 30, no other catalogue source with flux_fit > 1% of the star within 1.5 arcsec, position at least 19 px (SW) or 16 px (LW) from the frame edge (this exceeds the requested 10 px so that the annulus fits in the cutout), no NaN or DQ DO_NOT_USE pixel within 7 px of the star, no DQ SATURATED pixel within the outer annulus radius + 1. NaN and DO_NOT_USE pixels inside the annulus are masked out of the background estimate, and at least half of the annulus pixels must remain. No further magnitude cut is applied. The selected stars span a broad magnitude range (about 17 to 22 mag AB).
- Aperture photometry: photutils CircularAperture (exact), r = 2, 3, 4, 6 px (SW 0.031 arcsec/px, LW 0.063 arcsec/px). Background: sigma-clipped median (3 sigma, 10 iterations) in an annulus of 10-15 px (SW) or 8-12 px (LW).
- Model: the grid is evaluated at (x_fit, y_fit) with flux = flux_fit on the same pixel grid. The same annulus (same masks) gives a model background bm, which is subtracted from the model aperture sums. EE_model(r) = (model aperture sum - area*bm) / flux_fit.
- Closure: c(r) = 2.5 log10( flux_fit / (ap(r)/EE_model(r)) ). Positive c means the PSF fit reads faint relative to the aperture flux.
- Statistics: median over stars, error = 1.2533 * 1.4826 * MAD / sqrt(N).
- Comparison: for each band, c(r) per detector and the measured dm (ours - dolphot - ZP) from photomver/photdet.txt are both reduced by their detector mean. Slope is the OLS slope of measured on c (intercept free), with Pearson and Spearman coefficients. The figure shows c at r = 3 px (SW) or 2 px (LW).
- Selected star counts after cuts (all exposures): F150W 1585, F200W 850, F182M 980, F212N 664, F250M 383, F410M 357. The 1% neighbour criterion removes most stars in the crowded nrcb1 and nrcb3 SW fields (F200W: nrcb1 N=18, nrcb3 N=27).

## Results

### Per-detector closure and the photdet term

Columns c(r) in mag, median +/- error. The last column is the photdet measured dm (ours - dolphot - band ZP) for reference.

#### F150W

| detector | N | c(2px) | c(3px) | c(4px) | c(6px) | ours - dolphot (photdet) |
|---|---|---|---|---|---|---|
| nrca1 | 262 | -0.0091 ± 0.0007 | -0.0143 ± 0.0009 | -0.0212 ± 0.0012 | -0.0261 ± 0.0017 | +0.0189 ± 0.0027 |
| nrca2 | 271 | -0.0095 ± 0.0015 | -0.0133 ± 0.0010 | -0.0201 ± 0.0010 | -0.0264 ± 0.0012 | -0.0018 ± 0.0010 |
| nrca3 | 276 | -0.0030 ± 0.0009 | -0.0119 ± 0.0011 | -0.0196 ± 0.0014 | -0.0246 ± 0.0018 | +0.0061 ± 0.0022 |
| nrca4 | 225 | -0.0098 ± 0.0013 | -0.0121 ± 0.0011 | -0.0203 ± 0.0010 | -0.0251 ± 0.0013 | +0.0023 ± 0.0021 |
| nrcb1 | 87 | -0.0178 ± 0.0022 | -0.0160 ± 0.0025 | -0.0177 ± 0.0032 | -0.0199 ± 0.0045 | -0.0018 ± 0.0011 |
| nrcb2 | 187 | -0.0076 ± 0.0013 | -0.0095 ± 0.0013 | -0.0151 ± 0.0016 | -0.0182 ± 0.0022 | -0.0002 ± 0.0011 |
| nrcb3 | 79 | -0.0123 ± 0.0024 | -0.0117 ± 0.0021 | -0.0180 ± 0.0032 | -0.0176 ± 0.0042 | -0.0189 ± 0.0015 |
| nrcb4 | 194 | -0.0125 ± 0.0009 | -0.0113 ± 0.0013 | -0.0161 ± 0.0016 | -0.0165 ± 0.0020 | +0.0145 ± 0.0013 |

#### F200W

| detector | N | c(2px) | c(3px) | c(4px) | c(6px) | ours - dolphot (photdet) |
|---|---|---|---|---|---|---|
| nrca1 | 107 | -0.0070 ± 0.0008 | -0.0096 ± 0.0009 | -0.0113 ± 0.0012 | -0.0157 ± 0.0013 | +0.0263 ± 0.0024 |
| nrca2 | 161 | -0.0077 ± 0.0005 | -0.0085 ± 0.0005 | -0.0093 ± 0.0007 | -0.0139 ± 0.0009 | +0.0023 ± 0.0015 |
| nrca3 | 188 | -0.0032 ± 0.0007 | -0.0090 ± 0.0005 | -0.0109 ± 0.0009 | -0.0193 ± 0.0010 | +0.0039 ± 0.0030 |
| nrca4 | 121 | -0.0070 ± 0.0005 | -0.0072 ± 0.0006 | -0.0081 ± 0.0007 | -0.0125 ± 0.0008 | +0.0078 ± 0.0018 |
| nrcb1 | 18 | -0.0090 ± 0.0018 | -0.0090 ± 0.0027 | -0.0107 ± 0.0036 | -0.0179 ± 0.0088 | +0.0032 ± 0.0010 |
| nrcb2 | 95 | -0.0064 ± 0.0008 | -0.0105 ± 0.0008 | -0.0143 ± 0.0012 | -0.0198 ± 0.0022 | -0.0041 ± 0.0015 |
| nrcb3 | 27 | -0.0095 ± 0.0022 | -0.0092 ± 0.0019 | -0.0122 ± 0.0016 | -0.0154 ± 0.0039 | -0.0249 ± 0.0010 |
| nrcb4 | 133 | -0.0091 ± 0.0008 | -0.0086 ± 0.0008 | -0.0108 ± 0.0010 | -0.0134 ± 0.0016 | +0.0088 ± 0.0017 |

#### F182M

| detector | N | c(2px) | c(3px) | c(4px) | c(6px) | ours - dolphot (photdet) |
|---|---|---|---|---|---|---|
| nrca1 | 134 | -0.0081 ± 0.0007 | -0.0114 ± 0.0008 | -0.0161 ± 0.0009 | -0.0167 ± 0.0013 | +0.0158 ± 0.0029 |
| nrca2 | 184 | -0.0114 ± 0.0004 | -0.0127 ± 0.0005 | -0.0162 ± 0.0007 | -0.0190 ± 0.0010 | +0.0071 ± 0.0018 |
| nrca3 | 221 | -0.0054 ± 0.0007 | -0.0114 ± 0.0005 | -0.0162 ± 0.0007 | -0.0197 ± 0.0010 | +0.0393 ± 0.0031 |
| nrca4 | 131 | -0.0102 ± 0.0005 | -0.0112 ± 0.0007 | -0.0161 ± 0.0008 | -0.0183 ± 0.0014 | +0.0087 ± 0.0018 |
| nrcb1 | 31 | -0.0137 ± 0.0030 | -0.0164 ± 0.0020 | -0.0191 ± 0.0015 | -0.0220 ± 0.0019 | -0.0005 ± 0.0008 |
| nrcb2 | 123 | -0.0085 ± 0.0008 | -0.0121 ± 0.0010 | -0.0174 ± 0.0012 | -0.0163 ± 0.0018 | +0.0022 ± 0.0018 |
| nrcb3 | 34 | -0.0144 ± 0.0018 | -0.0117 ± 0.0016 | -0.0161 ± 0.0021 | -0.0152 ± 0.0034 | -0.0120 ± 0.0007 |
| nrcb4 | 122 | -0.0130 ± 0.0010 | -0.0140 ± 0.0011 | -0.0170 ± 0.0014 | -0.0138 ± 0.0018 | -0.0007 ± 0.0017 |

#### F212N

| detector | N | c(2px) | c(3px) | c(4px) | c(6px) | ours - dolphot (photdet) |
|---|---|---|---|---|---|---|
| nrca1 | 68 | -0.0090 ± 0.0008 | -0.0083 ± 0.0005 | -0.0099 ± 0.0010 | -0.0096 ± 0.0014 | +0.0036 ± 0.0025 |
| nrca2 | 83 | -0.0099 ± 0.0005 | -0.0074 ± 0.0004 | -0.0093 ± 0.0007 | -0.0089 ± 0.0010 | -0.0012 ± 0.0013 |
| nrca3 | 134 | -0.0073 ± 0.0005 | -0.0113 ± 0.0006 | -0.0165 ± 0.0012 | -0.0206 ± 0.0015 | +0.0149 ± 0.0034 |
| nrca4 | 77 | -0.0110 ± 0.0006 | -0.0078 ± 0.0006 | -0.0104 ± 0.0011 | -0.0114 ± 0.0010 | -0.0041 ± 0.0018 |
| nrcb1 | 84 | -0.0117 ± 0.0009 | -0.0091 ± 0.0006 | -0.0112 ± 0.0007 | -0.0124 ± 0.0009 | +0.0058 ± 0.0009 |
| nrcb2 | 104 | -0.0100 ± 0.0007 | -0.0088 ± 0.0005 | -0.0129 ± 0.0006 | -0.0154 ± 0.0009 | -0.0015 ± 0.0018 |
| nrcb3 | 49 | -0.0108 ± 0.0011 | -0.0094 ± 0.0006 | -0.0136 ± 0.0009 | -0.0162 ± 0.0010 | -0.0045 ± 0.0009 |
| nrcb4 | 65 | -0.0106 ± 0.0009 | -0.0085 ± 0.0008 | -0.0119 ± 0.0014 | -0.0115 ± 0.0017 | -0.0020 ± 0.0019 |

#### F250M

| detector | N | c(2px) | c(3px) | c(4px) | c(6px) | ours - dolphot (photdet) |
|---|---|---|---|---|---|---|
| nrcalong | 242 | -0.0011 ± 0.0008 | -0.0076 ± 0.0012 | -0.0071 ± 0.0017 | -0.0114 ± 0.0032 | -0.0227 ± 0.0022 |
| nrcblong | 137 | -0.0016 ± 0.0010 | -0.0026 ± 0.0015 | +0.0000 ± 0.0023 | -0.0042 ± 0.0029 | +0.0063 ± 0.0010 |

#### F410M

| detector | N | c(2px) | c(3px) | c(4px) | c(6px) | ours - dolphot (photdet) |
|---|---|---|---|---|---|---|
| nrcalong | 260 | -0.0049 ± 0.0014 | +0.0036 ± 0.0027 | +0.0020 ± 0.0045 | +0.0025 ± 0.0085 | -0.0185 ± 0.0018 |
| nrcblong | 93 | -0.0025 ± 0.0022 | +0.0020 ± 0.0035 | +0.0020 ± 0.0055 | +0.0149 ± 0.0085 | +0.0086 ± 0.0014 |

### Band-wide closure versus aperture radius

| band | c(2px) | c(3px) | c(4px) | c(6px) |
|---|---|---|---|---|
| F150W | -0.0089 ± 0.0004 | -0.0124 ± 0.0004 | -0.0192 ± 0.0005 | -0.0231 ± 0.0007 |
| F200W | -0.0067 ± 0.0003 | -0.0087 ± 0.0003 | -0.0104 ± 0.0004 | -0.0154 ± 0.0005 |
| F182M | -0.0095 ± 0.0003 | -0.0120 ± 0.0003 | -0.0164 ± 0.0004 | -0.0181 ± 0.0005 |
| F212N | -0.0098 ± 0.0003 | -0.0089 ± 0.0002 | -0.0118 ± 0.0004 | -0.0131 ± 0.0004 |
| F250M | -0.0014 ± 0.0006 | -0.0059 ± 0.0009 | -0.0052 ± 0.0014 | -0.0094 ± 0.0023 |
| F410M | -0.0038 ± 0.0011 | +0.0036 ± 0.0022 | +0.0020 ± 0.0037 | +0.0044 ± 0.0067 |

### Slope and correlation of measured (ours - dolphot) against closure, detector means removed

| band | r (px) | N det | slope (measured vs c) | slope err | Pearson r | Spearman rho | rms c-mean | rms measured-mean |
|---|---|---|---|---|---|---|---|---|
| F150W | 2 | 8 | +0.66 | 1.06 | +0.25 | +0.34 | 0.0040 | 0.0108 |
| F150W | 3 | 8 | -0.71 | 2.33 | -0.12 | +0.02 | 0.0019 | 0.0108 |
| F150W | 4 | 8 | -1.09 | 2.13 | -0.20 | -0.28 | 0.0020 | 0.0108 |
| F150W | 6 | 8 | -0.86 | 1.07 | -0.31 | -0.12 | 0.0039 | 0.0108 |
| F200W | 2 | 8 | +1.98 | 2.77 | +0.28 | +0.21 | 0.0019 | 0.0134 |
| F200W | 3 | 8 | +1.85 | 6.09 | +0.12 | +0.26 | 0.0009 | 0.0134 |
| F200W | 4 | 8 | +2.75 | 2.98 | +0.35 | +0.33 | 0.0017 | 0.0134 |
| F200W | 6 | 8 | +0.82 | 2.09 | +0.16 | +0.33 | 0.0026 | 0.0134 |
| F182M | 2 | 8 | +4.31 | 0.92 | +0.89 | +0.90 | 0.0029 | 0.0142 |
| F182M | 3 | 8 | +3.24 | 3.23 | +0.38 | +0.64 | 0.0017 | 0.0142 |
| F182M | 4 | 8 | +4.51 | 5.50 | +0.32 | +0.36 | 0.0010 | 0.0142 |
| F182M | 6 | 8 | -2.18 | 2.18 | -0.38 | -0.52 | 0.0025 | 0.0142 |
| F212N | 2 | 8 | +3.30 | 1.43 | +0.69 | +0.48 | 0.0013 | 0.0061 |
| F212N | 3 | 8 | -4.12 | 1.52 | -0.74 | -0.24 | 0.0011 | 0.0061 |
| F212N | 4 | 8 | -1.46 | 0.96 | -0.53 | +0.02 | 0.0022 | 0.0061 |
| F212N | 6 | 8 | -0.85 | 0.59 | -0.51 | -0.07 | 0.0037 | 0.0061 |
| F250M | 2 | 2 | -58.69 | n/a | n/a | n/a | 0.0002 | 0.0145 |
| F250M | 3 | 2 | +5.79 | n/a | n/a | n/a | 0.0025 | 0.0145 |
| F250M | 4 | 2 | +4.05 | n/a | n/a | n/a | 0.0036 | 0.0145 |
| F250M | 6 | 2 | +3.98 | n/a | n/a | n/a | 0.0036 | 0.0145 |
| F410M | 2 | 2 | +11.22 | n/a | n/a | n/a | 0.0012 | 0.0135 |
| F410M | 3 | 2 | -16.76 | n/a | n/a | n/a | 0.0008 | 0.0135 |
| F410M | 4 | 2 | -1227.88 | n/a | n/a | n/a | 0.0000 | 0.0135 |
| F410M | 6 | 2 | +2.18 | n/a | n/a | n/a | 0.0062 | 0.0135 |

### Stability per exposure (c at 3 px)


#### F200W c(3 px) per exposure

| detector | 00001 | 00002 | 00003 | 00004 |
|---|---|---|---|---|
| nrca1 | -0.0093 ± 0.0019 (N=30) | -0.0112 ± 0.0015 (N=27) | -0.0103 ± 0.0016 (N=26) | -0.0063 ± 0.0024 (N=24) |
| nrca2 | -0.0083 ± 0.0011 (N=40) | -0.0090 ± 0.0010 (N=40) | -0.0077 ± 0.0009 (N=40) | -0.0085 ± 0.0012 (N=41) |
| nrca3 | -0.0094 ± 0.0012 (N=47) | -0.0087 ± 0.0011 (N=45) | -0.0087 ± 0.0008 (N=52) | -0.0093 ± 0.0009 (N=44) |
| nrca4 | -0.0070 ± 0.0012 (N=30) | -0.0073 ± 0.0007 (N=30) | -0.0076 ± 0.0012 (N=31) | -0.0073 ± 0.0013 (N=30) |
| nrcb1 | -0.0136 ± 0.0008 (N=3) | -0.0069 ± 0.0044 (N=5) | -0.0116 ± 0.0031 (N=6) | -0.0102 ± 0.0048 (N=4) |
| nrcb2 | -0.0105 ± 0.0021 (N=23) | -0.0108 ± 0.0014 (N=21) | -0.0100 ± 0.0015 (N=26) | -0.0099 ± 0.0013 (N=25) |
| nrcb3 | -0.0092 ± 0.0020 (N=7) | -0.0106 ± 0.0030 (N=7) | -0.0110 ± 0.0057 (N=6) | -0.0082 ± 0.0035 (N=7) |
| nrcb4 | -0.0085 ± 0.0013 (N=33) | -0.0080 ± 0.0016 (N=33) | -0.0087 ± 0.0013 (N=33) | -0.0101 ± 0.0013 (N=34) |

#### F150W c(3 px) per exposure

| detector | 00001 | 00002 | 00003 | 00004 |
|---|---|---|---|---|
| nrca1 | -0.0154 ± 0.0017 (N=71) | -0.0138 ± 0.0017 (N=53) | -0.0155 ± 0.0018 (N=83) | -0.0132 ± 0.0019 (N=55) |
| nrca2 | -0.0134 ± 0.0017 (N=64) | -0.0140 ± 0.0023 (N=71) | -0.0140 ± 0.0022 (N=66) | -0.0121 ± 0.0019 (N=70) |
| nrca3 | -0.0119 ± 0.0023 (N=74) | -0.0113 ± 0.0022 (N=64) | -0.0120 ± 0.0024 (N=76) | -0.0121 ± 0.0020 (N=62) |
| nrca4 | -0.0104 ± 0.0017 (N=56) | -0.0108 ± 0.0022 (N=47) | -0.0119 ± 0.0022 (N=57) | -0.0154 ± 0.0027 (N=65) |
| nrcb1 | -0.0116 ± 0.0048 (N=21) | -0.0160 ± 0.0053 (N=21) | -0.0166 ± 0.0045 (N=24) | -0.0186 ± 0.0045 (N=21) |
| nrcb2 | -0.0085 ± 0.0022 (N=42) | -0.0095 ± 0.0027 (N=44) | -0.0106 ± 0.0027 (N=51) | -0.0094 ± 0.0032 (N=50) |
| nrcb3 | -0.0174 ± 0.0068 (N=15) | -0.0128 ± 0.0035 (N=17) | -0.0117 ± 0.0063 (N=23) | -0.0095 ± 0.0031 (N=24) |
| nrcb4 | -0.0090 ± 0.0020 (N=46) | -0.0110 ± 0.0028 (N=55) | -0.0102 ± 0.0026 (N=50) | -0.0147 ± 0.0027 (N=43) |

## Reading

1. Detector-to-detector spread of the closure is small. With the detector mean removed, the rms of c(3 px) over the 8 SW detectors is 0.0019 mag (F150W), 0.0009 (F200W), 0.0017 (F182M), 0.0011 (F212N), compared with rms 0.011, 0.013, 0.014, 0.006 mag in the photdet term. The closure scatter is comparable to its statistical errors (0.0005-0.0027 mag).
2. The detectors that stand out in photdet do not stand out in c. For F200W nrcb3 has c(3 px) = -0.0092 +/- 0.0019 against a 8-detector median near -0.009, while its photdet term is -0.025. nrca1 has c(3 px) = -0.0096 +/- 0.0009 and a photdet term of +0.026. For F150W nrcb3 (c = -0.0117) and nrca1 (c = -0.0143) lie within 0.003 mag of the other detectors, while photdet gives -0.019 and +0.019. For F182M nrca3 has c(3 px) = -0.0114 +/- 0.0005, equal to nrca1, nrca2 and nrca4, while photdet gives +0.039. In F250M, nrcalong minus nrcblong gives c(2 px) = -0.0011 - (-0.0016) = +0.0005 mag, and c(3 px) gives -0.005 mag, against the photdet difference of -0.029 mag. In F410M the c(2 px) difference is -0.0024 mag against -0.027 mag in photdet.
3. The slope and correlation against the photdet term have no consistent sign. Slopes at the reference radius are -0.71 (F150W, 3 px), +1.85 (F200W), +3.24 (F182M), +3.3 (F212N, 2 px) and -4.1 (F212N, 3 px), each with an error of 1-6 and Pearson r between -0.74 and +0.89 changing sign with radius. A model in which the photdet term was carried by the PSF fit would give a slope of order 1 with r near 1 at every radius. Because the closure range is only 0.001-0.004 mag rms, the slope is poorly constrained. The closure shows an rms of at most 0.004 mag between detectors, a factor 3-10 smaller than the photdet term.
4. The closure depends on radius, with a band-wide trend. In F150W the median c runs from -0.009 (2 px) to -0.023 (6 px), in F200W from -0.007 to -0.015, in F182M from -0.010 to -0.018, in F212N from -0.010 to -0.013, and in F250M from -0.001 to -0.009. The negative sign means the PSF fit reads brighter than the aperture total at larger radius, equivalent to model wings that are weaker than the data wings or a flux_fit amplitude set mostly by the core. This is a common-mode term of about 0.01-0.02 mag in absolute scale, shared by all detectors in a band. It is a candidate contributor to the band-level offsets against dolphot but does not separate detectors. For F410M c is within +/- 0.005 mag with errors of 0.002-0.009 mag, and the radius dependence is not resolved.
5. Per-exposure values (F200W and F150W) agree with the all-exposure medians within the errors for every detector with N >= 20 stars, so the closure is stable between dithers.

Conclusion: the per-detector term in ours - dolphot is not present in the PSF-fit-to-aperture closure on our own frames at the 0.002-0.004 mag level. The PSF grid normalisation per detector matches the data to that level. The source of the per-detector term is more likely a feature of the dolphot side or of the comparison (dolphot detector-dependent aperture corrections or zero points, per-chip photometric calibration applied to dolphot, or ours vs dolphot treatment of the photom reference), or a calibration term common to the aperture and PSF fluxes on our frames, since closure removes anything that multiplies both equally, for example a per-detector flux scaling in the crf SCI image. This test is blind to that last case: a per-detector scale error in the calibrated images changes ap(r) and flux_fit identically.

## Caveats

- Crowding: the 1% neighbour criterion within 1.5 arcsec leaves 18-27 stars on nrcb1 and nrcb3 in F200W, and 31-34 on those detectors in F182M. Errors on these detectors are 0.002-0.003 mag. Faint unresolved neighbours below the catalogue limit raise the annulus median and the aperture sums at the 0.001-0.01 mag level for faint stars.
- Background: the closure uses the crf SCI image with an annulus median. The fit used the local_bkg column of a different annulus (6-10 px) on the resbgsub data. The aperture background is correlated with the PSF wings, and this contributes to the radius dependence (the 6 px column is the most sensitive).
- PSF grid interpolation: the grid is a 4x4 (npsf16) interpolation over the detector, evaluated here with the photutils routine used by the fit. Interpolation errors between grid nodes appear as star-to-star scatter and cancel in the detector medians only partly.
- Selection: the same stars are not used in photdet (which uses stars on one detector only that are matched to dolphot, in the ZP window). The closure sample extends over a different magnitude range, and c may depend on flux through the nonlinearity or charge migration (brightness-dependent effect, not tested here).
- The test is blind to errors that scale both the aperture and PSF fluxes equally on a detector (calibration scaling, PHOTMJSR applied to the image).
- Aperture radii of 2 px in the LW (0.126 arcsec) are close to the PSF core size and are sensitive to the sub-pixel position.
- LW statistics are limited (nrcblong F410M N=93).

## Files

- /orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/detclose/detclose.py : per-frame measurement (run as `detclose.py BAND 4 100000`)
- /orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/detclose/analyze.py : tables and figure
- /orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/detclose/detclose_<band>.ecsv : per-star measurements (F150W, F200W, F182M, F212N, F250M, F410M)
- /orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/detclose/log_<band>.txt : run logs
- /orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/detclose/tables_closure.md, tables_compare.md, tables_radius.md, tables_perexp.md : table fragments included above
- /orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/detclose/detclose.png : measured ours - dolphot vs closure per band
