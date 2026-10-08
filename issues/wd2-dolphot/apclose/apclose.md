# Aperture closure (apclose)

R = annulus_sum(data - bg) / flux_fit, normalised per annulus by the median R of unsaturated daophot stars in the reference window (so the unsaturated plateau is 1.0). Satstar R uses flux_fit_precap ("precap") and flux_fit ("final"). Cells show median +- bootstrap error (N). Magnitudes are dolphot mags (match within 0.1 arcsec). Plot shows the two innermost annuli only; tables cover all. Stars with NaN, DQ SATURATED or DO_NOT_USE pixels in the annulus, or with a neighbour (flux > 5 % of the target) within max(isolation radius, r_out + 2 px), are excluded.

Method notes:
- Image = SCI of the arm tree crf file (SCI equals satstar residual + model wherever SCI is finite; max abs differences are a few tens to ~2000 MJy/sr in a handful of pixels, 107 or less in main2kf).
- Saturated cores hold truncated DQ SATURATED values (SCI ~ 200 where the model is ~1e5) in both arms and are NaN in part of the core, so a circular aperture sum over the star is invalid for every satstar (0 satstars have a clean circle). Closure is therefore tested on annuli outside the saturated region, for satstars and unsaturated stars alike.
- Satstar data = SCI; unsaturated daophot data = SCI minus satstar model. Background = sigma-clipped median of the satstar residual image in a 1.8-2.4 arcsec annulus (the spec annulus 2.5-3.5 arcsec was tested first; a closer one gave a flatter unsaturated plateau). Annulus sums of faint stars (> 18 mag LW) are limited by background systematics (R goes to 0 or negative), so the unsaturated reference window is 0.5 mag brighter than the satstar faint edge up to +1 mag, and the table shows the specified window value too.
- Model-annulus column = sum of the satstar model image in the same annulus / flux_fit (same normalisation); data/model = annulus data sum / annulus model sum (independent of flux_fit and of the normalisation).

## F250M main2kf

satstar saturation faint edge (95th pct of dolphot mag of replaced satstars) = 16.54. Primary reference window for unsaturated stars: 16.04-17.54; specified window (onset+1..+3): 17.54-19.54.
Satstar rows (all frames, pooled): 3536; with dolphot match: 3165.
Satstars with any NaN in the largest circle (r=8 px): 2828; with any NaN/SAT/DNU pixel in it: 3536; median equivalent saturated radius: 3.0 px.

### F250M main2kf annulus 5-8 px (0.32-0.50 arcsec)
normalisation: median unsaturated R = 0.0357 +- 0.0023 (N=437); specified-window value 0.0249 (N=549)

| mag bin | unsat daophot R | satstar R precap | satstar R final | satstar model-annulus / flux_fit (norm.) | satstar data/model annulus (raw) |
|---|---|---|---|---|---|
| 14-15 | n/a | 1.384+-0.017 (70) | 1.423+-0.023 (70) | 1.149+-0.011 (70) | 1.212+-0.020 (70) |
| 15-16 | 1.114+-0.062 (23) | 1.343+-0.011 (188) | 1.376+-0.016 (188) | 1.185+-0.009 (188) | 1.142+-0.019 (188) |
| 16-17 | 1.032+-0.066 (248) | 1.547+-0.102 (115) | 1.588+-0.093 (115) | 1.202+-0.014 (115) | 1.133+-0.040 (115) |
| 17-18 | 0.970+-0.068 (385) | n/a | n/a | n/a | n/a |
| 18-19 | 0.361+-0.146 (248) | n/a | n/a | n/a | n/a |
| 19-20 | 0.146+-0.404 (189) | n/a | n/a | n/a | n/a |
| 20-21 | -2.128+-1.387 (94) | n/a | n/a | n/a | n/a |

R vs wingcal_rmask (satstars, this annulus):

| wingcal_rmask range | N | median R precap | median R final | median data/model | median dolphot mag |
|---|---|---|---|---|---|
| 0 to 0.5642 | 368 | 1.367 | 1.402 | 1.164 | 15.66 |
| 2 to 5 | 8 | 1.145 | 1.212 | 1.012 | 15.30 |

R vs cap_psf_frac (satstars, this annulus):

| cap_psf_frac range | N | median R precap | median R final | median data/model | median dolphot mag |
|---|---|---|---|---|---|
| 0.7 to 0.9999 | 4 | 1.568 | 1.568 | 1.159 | 15.08 |
| 0.9999 to 1e+09 | 361 | 1.365 | 1.396 | 1.166 | 15.66 |

### F250M main2kf annulus 8-12 px (0.50-0.76 arcsec)
normalisation: median unsaturated R = 0.0303 +- 0.0042 (N=162); specified-window value 0.0044 (N=221)

| mag bin | unsat daophot R | satstar R precap | satstar R final | satstar model-annulus / flux_fit (norm.) | satstar data/model annulus (raw) |
|---|---|---|---|---|---|
| 12-13 | n/a | 1.133+-0.089 (7) | 1.365+-0.082 (7) | 0.962+-0.033 (7) | 1.319+-0.119 (7) |
| 13-14 | n/a | 1.333+-0.156 (30) | 1.398+-0.123 (30) | 0.982+-0.006 (30) | 1.415+-0.106 (30) |
| 14-15 | n/a | 1.145+-0.112 (62) | 1.161+-0.115 (62) | 1.028+-0.028 (62) | 1.106+-0.019 (62) |
| 15-16 | 0.591+-0.443 (17) | 1.230+-0.061 (92) | 1.234+-0.071 (92) | 1.179+-0.039 (92) | 1.107+-0.048 (92) |
| 16-17 | 0.989+-0.152 (95) | 2.062+-0.123 (45) | 2.145+-0.164 (45) | 1.081+-0.046 (45) | 1.789+-0.281 (45) |
| 17-18 | 0.889+-0.184 (149) | n/a | n/a | n/a | n/a |
| 18-19 | -0.898+-0.296 (104) | n/a | n/a | n/a | n/a |
| 19-20 | 1.267+-1.674 (64) | n/a | n/a | n/a | n/a |
| 20-21 | 0.186+-7.495 (30) | n/a | n/a | n/a | n/a |

R vs wingcal_rmask (satstars, this annulus):

| wingcal_rmask range | N | median R precap | median R final | median data/model | median dolphot mag |
|---|---|---|---|---|---|
| 0 to 0.5642 | 231 | 1.313 | 1.355 | 1.144 | 15.12 |
| 2 to 5 | 5 | 0.977 | 1.251 | 1.149 | 12.94 |

R vs cap_psf_frac (satstars, this annulus):

| cap_psf_frac range | N | median R precap | median R final | median data/model | median dolphot mag |
|---|---|---|---|---|---|
| 0.7 to 0.9999 | 4 | 1.629 | 1.777 | 1.317 | 15.12 |
| 0.9999 to 1e+09 | 226 | 1.312 | 1.348 | 1.144 | 15.12 |

### F250M main2kf annulus 12-18 px (0.76-1.13 arcsec)
normalisation: median unsaturated R = 0.0370 +- 0.0559 (N=8); specified-window value -0.0697 (N=14)

| mag bin | unsat daophot R | satstar R precap | satstar R final | satstar model-annulus / flux_fit (norm.) | satstar data/model annulus (raw) |
|---|---|---|---|---|---|
| 14-15 | n/a | 0.935+-0.188 (7) | 0.953+-0.162 (7) | 0.630+-0.035 (7) | 1.609+-0.332 (7) |
| 15-16 | n/a | 0.879+-0.088 (7) | 0.956+-0.137 (7) | 1.118+-0.139 (7) | 0.839+-0.114 (7) |
| 16-17 | -2.076+-1.286 (6) | n/a | n/a | n/a | n/a |
| 17-18 | 1.620+-1.634 (8) | n/a | n/a | n/a | n/a |
| 18-19 | -7.529+-4.624 (5) | n/a | n/a | n/a | n/a |
| 19-20 | 19.012+-19.829 (5) | n/a | n/a | n/a | n/a |
| 20-21 | -19.386+-36.845 (5) | n/a | n/a | n/a | n/a |

### annulus 18-26 px: no valid unsaturated reference (N=0)

### Sanity check, unsaturated daophot stars, F250M main2kf: aperture_sum / flux_fit vs dolphot mag (median, N), raw units (not normalised)

| aperture | 14-15 | 15-16 | 16-17 | 17-18 | 18-19 | 19-20 |
|---|---|---|---|---|---|---|
| annulus 5-8 px | n/a | 0.040+-0.002 (23) | 0.037+-0.002 (248) | 0.035+-0.002 (385) | 0.013+-0.004 (248) | 0.005+-0.015 (189) |
| annulus 8-12 px | n/a | 0.018+-0.014 (17) | 0.030+-0.005 (95) | 0.027+-0.005 (149) | -0.027+-0.009 (104) | 0.038+-0.055 (64) |
| annulus 12-18 px | n/a | n/a | -0.077+-0.045 (6) | 0.060+-0.075 (8) | -0.278+-0.163 (5) | 0.703+-0.824 (5) |
| annulus 18-26 px | n/a | n/a | n/a | n/a | n/a | n/a |
| circle r=3 px | n/a | n/a | 0.803+-0.001 (372) | 0.810+-0.001 (569) | 0.803+-0.002 (461) | 0.791+-0.004 (429) |
| circle r=5 px | n/a | n/a | 0.856+-0.002 (250) | 0.864+-0.002 (410) | 0.845+-0.004 (327) | 0.825+-0.013 (299) |
| circle r=8 px | n/a | n/a | 0.881+-0.003 (120) | 0.899+-0.004 (191) | 0.866+-0.011 (142) | 0.838+-0.032 (114) |

Satstar circular apertures with no NaN/SAT/DNU pixel and isolated: r=3 px: 0, r=5 px: 0, r=8 px: 0.

## F250M main2

satstar saturation faint edge (95th pct of dolphot mag of replaced satstars) = 16.54. Primary reference window for unsaturated stars: 16.04-17.54; specified window (onset+1..+3): 17.54-19.54.
Satstar rows (all frames, pooled): 3464; with dolphot match: 3098.
Satstars with any NaN in the largest circle (r=8 px): 2765; with any NaN/SAT/DNU pixel in it: 3464; median equivalent saturated radius: 3.0 px.

### F250M main2 annulus 5-8 px (0.32-0.50 arcsec)
normalisation: median unsaturated R = 0.0346 +- 0.0022 (N=435); specified-window value 0.0244 (N=546)

| mag bin | unsat daophot R | satstar R precap | satstar R final | satstar model-annulus / flux_fit (norm.) | satstar data/model annulus (raw) |
|---|---|---|---|---|---|
| 14-15 | n/a | 1.482+-0.027 (67) | 1.558+-0.029 (67) | 1.185+-0.011 (67) | 1.262+-0.027 (67) |
| 15-16 | 1.137+-0.044 (27) | 1.430+-0.015 (181) | 1.449+-0.017 (181) | 1.223+-0.010 (181) | 1.142+-0.020 (181) |
| 16-17 | 1.028+-0.073 (249) | 1.634+-0.104 (116) | 1.662+-0.106 (116) | 1.238+-0.014 (116) | 1.148+-0.050 (116) |
| 17-18 | 0.964+-0.071 (382) | n/a | n/a | n/a | n/a |
| 18-19 | 0.368+-0.128 (245) | n/a | n/a | n/a | n/a |
| 19-20 | 0.139+-0.391 (188) | n/a | n/a | n/a | n/a |
| 20-21 | -2.136+-1.387 (92) | n/a | n/a | n/a | n/a |

R vs wingcal_rmask (satstars, this annulus):

| wingcal_rmask range | N | median R precap | median R final | median data/model | median dolphot mag |
|---|---|---|---|---|---|
| 0 to 0.5642 | 359 | 1.454 | 1.480 | 1.186 | 15.67 |
| 2 to 5 | 8 | 1.201 | 1.201 | 1.027 | 15.30 |

R vs cap_psf_frac (satstars, this annulus):

| cap_psf_frac range | N | median R precap | median R final | median data/model | median dolphot mag |
|---|---|---|---|---|---|
| 0.7 to 0.9999 | 4 | 1.702 | 1.702 | 1.223 | 15.08 |
| 0.9999 to 1e+09 | 351 | 1.452 | 1.475 | 1.186 | 15.67 |

### F250M main2 annulus 8-12 px (0.50-0.76 arcsec)
normalisation: median unsaturated R = 0.0305 +- 0.0046 (N=162); specified-window value 0.0080 (N=220)

| mag bin | unsat daophot R | satstar R precap | satstar R final | satstar model-annulus / flux_fit (norm.) | satstar data/model annulus (raw) |
|---|---|---|---|---|---|
| 12-13 | n/a | 1.151+-0.088 (7) | 1.361+-0.110 (7) | 0.955+-0.031 (7) | 1.408+-0.143 (7) |
| 13-14 | n/a | 1.337+-0.170 (29) | 1.455+-0.162 (29) | 0.975+-0.006 (29) | 1.482+-0.109 (29) |
| 14-15 | n/a | 1.158+-0.087 (59) | 1.162+-0.123 (59) | 1.014+-0.023 (59) | 1.146+-0.023 (59) |
| 15-16 | 0.519+-0.387 (18) | 1.289+-0.090 (90) | 1.293+-0.072 (90) | 1.168+-0.040 (90) | 1.150+-0.042 (90) |
| 16-17 | 1.005+-0.164 (95) | 2.107+-0.169 (43) | 2.111+-0.134 (43) | 1.071+-0.050 (43) | 1.823+-0.334 (43) |
| 17-18 | 0.850+-0.217 (149) | n/a | n/a | n/a | n/a |
| 18-19 | -0.962+-0.405 (103) | n/a | n/a | n/a | n/a |
| 19-20 | 1.313+-1.818 (64) | n/a | n/a | n/a | n/a |
| 20-21 | 0.729+-6.796 (30) | n/a | n/a | n/a | n/a |

R vs wingcal_rmask (satstars, this annulus):

| wingcal_rmask range | N | median R precap | median R final | median data/model | median dolphot mag |
|---|---|---|---|---|---|
| 0 to 0.5642 | 223 | 1.325 | 1.391 | 1.187 | 15.14 |
| 2 to 5 | 5 | 1.037 | 1.210 | 1.183 | 12.94 |

R vs cap_psf_frac (satstars, this annulus):

| cap_psf_frac range | N | median R precap | median R final | median data/model | median dolphot mag |
|---|---|---|---|---|---|
| 0.7 to 0.9999 | 4 | 1.667 | 1.838 | 1.364 | 15.12 |
| 0.9999 to 1e+09 | 218 | 1.321 | 1.381 | 1.187 | 15.13 |

### F250M main2 annulus 12-18 px (0.76-1.13 arcsec)
normalisation: median unsaturated R = 0.0206 +- 0.0547 (N=8); specified-window value -0.0694 (N=14)

| mag bin | unsat daophot R | satstar R precap | satstar R final | satstar model-annulus / flux_fit (norm.) | satstar data/model annulus (raw) |
|---|---|---|---|---|---|
| 14-15 | n/a | 1.799+-0.381 (7) | 1.799+-0.357 (7) | 1.132+-0.053 (7) | 1.690+-0.337 (7) |
| 15-16 | n/a | 1.639+-0.098 (7) | 1.639+-0.170 (7) | 1.992+-0.226 (7) | 0.815+-0.130 (7) |
| 16-17 | -3.703+-2.069 (6) | n/a | n/a | n/a | n/a |
| 17-18 | 2.842+-3.396 (8) | n/a | n/a | n/a | n/a |
| 18-19 | -13.413+-7.653 (5) | n/a | n/a | n/a | n/a |
| 19-20 | 34.002+-35.888 (5) | n/a | n/a | n/a | n/a |
| 20-21 | -35.111+-70.271 (5) | n/a | n/a | n/a | n/a |

### annulus 18-26 px: no valid unsaturated reference (N=0)

### Sanity check, unsaturated daophot stars, F250M main2: aperture_sum / flux_fit vs dolphot mag (median, N), raw units (not normalised)

| aperture | 14-15 | 15-16 | 16-17 | 17-18 | 18-19 | 19-20 |
|---|---|---|---|---|---|---|
| annulus 5-8 px | n/a | 0.039+-0.002 (27) | 0.036+-0.002 (249) | 0.033+-0.003 (382) | 0.013+-0.004 (245) | 0.005+-0.014 (188) |
| annulus 8-12 px | n/a | 0.016+-0.011 (18) | 0.031+-0.004 (95) | 0.026+-0.006 (149) | -0.029+-0.013 (103) | 0.040+-0.050 (64) |
| annulus 12-18 px | n/a | n/a | -0.076+-0.041 (6) | 0.059+-0.073 (8) | -0.276+-0.160 (5) | 0.701+-0.697 (5) |
| annulus 18-26 px | n/a | n/a | n/a | n/a | n/a | n/a |
| circle r=3 px | n/a | n/a | 0.803+-0.001 (371) | 0.810+-0.001 (567) | 0.803+-0.002 (455) | 0.792+-0.004 (420) |
| circle r=5 px | n/a | n/a | 0.855+-0.002 (249) | 0.864+-0.002 (408) | 0.844+-0.004 (322) | 0.827+-0.012 (294) |
| circle r=8 px | n/a | n/a | 0.880+-0.003 (120) | 0.897+-0.006 (190) | 0.866+-0.009 (139) | 0.839+-0.029 (114) |

Satstar circular apertures with no NaN/SAT/DNU pixel and isolated: r=3 px: 0, r=5 px: 0, r=8 px: 0.

## F150W main2kf

satstar saturation faint edge (95th pct of dolphot mag of replaced satstars) = 18.43. Primary reference window for unsaturated stars: 17.93-19.43; specified window (onset+1..+3): 19.43-21.43.
Satstar rows (all frames, pooled): 6249; with dolphot match: 5623.
Satstars with any NaN in the largest circle (r=10 px): 3185; with any NaN/SAT/DNU pixel in it: 6249; median equivalent saturated radius: 2.8 px.

### F150W main2kf annulus 8-14 px (0.25-0.43 arcsec)
normalisation: median unsaturated R = 0.0546 +- 0.0015 (N=1488); specified-window value 0.0436 (N=2018)

| mag bin | unsat daophot R | satstar R precap | satstar R final | satstar model-annulus / flux_fit (norm.) | satstar data/model annulus (raw) |
|---|---|---|---|---|---|
| 14-15 | n/a | 0.929+-0.009 (79) | 1.029+-0.020 (79) | 0.877+-0.005 (79) | 1.160+-0.023 (79) |
| 15-16 | 3.351+-0.813 (17) | 1.039+-0.015 (277) | 1.108+-0.017 (277) | 0.911+-0.006 (277) | 1.178+-0.008 (277) |
| 16-17 | 1.521+-0.352 (32) | 1.053+-0.015 (733) | 1.102+-0.011 (733) | 0.958+-0.009 (733) | 1.119+-0.007 (733) |
| 17-18 | 2.072+-0.418 (114) | 1.153+-0.028 (1106) | 1.231+-0.029 (1106) | 1.007+-0.010 (1106) | 1.089+-0.007 (1106) |
| 18-19 | 1.007+-0.032 (943) | 1.311+-0.043 (518) | 1.323+-0.057 (518) | 1.137+-0.028 (518) | 1.070+-0.027 (518) |
| 19-20 | 0.981+-0.053 (1237) | n/a | n/a | n/a | n/a |
| 20-21 | 0.636+-0.124 (984) | n/a | n/a | n/a | n/a |

R vs wingcal_rmask (satstars, this annulus):

| wingcal_rmask range | N | median R precap | median R final | median data/model | median dolphot mag |
|---|---|---|---|---|---|
| 0 to 0.5642 | 2685 | 1.107 | 1.168 | 1.120 | 17.31 |
| 2 to 5 | 28 | 0.889 | 0.914 | 1.027 | 14.64 |

R vs cap_psf_frac (satstars, this annulus):

| cap_psf_frac range | N | median R precap | median R final | median data/model | median dolphot mag |
|---|---|---|---|---|---|
| -1 to 0.7 | 27 | 1.071 | 1.179 | 1.203 | 18.30 |
| 0.9999 to 1e+09 | 2662 | 1.106 | 1.167 | 1.120 | 17.30 |

### F150W main2kf annulus 14-22 px (0.43-0.68 arcsec)
normalisation: median unsaturated R = 0.0460 +- 0.0035 (N=676); specified-window value 0.0087 (N=968)

| mag bin | unsat daophot R | satstar R precap | satstar R final | satstar model-annulus / flux_fit (norm.) | satstar data/model annulus (raw) |
|---|---|---|---|---|---|
| 14-15 | n/a | 0.849+-0.024 (58) | 0.927+-0.011 (58) | 0.736+-0.003 (58) | 1.267+-0.019 (58) |
| 15-16 | n/a | 1.009+-0.024 (178) | 1.112+-0.031 (178) | 0.807+-0.017 (178) | 1.299+-0.018 (178) |
| 16-17 | n/a | 1.084+-0.022 (364) | 1.137+-0.030 (364) | 0.855+-0.020 (364) | 1.189+-0.021 (364) |
| 17-18 | 4.119+-1.691 (19) | 1.263+-0.061 (547) | 1.295+-0.057 (547) | 0.977+-0.030 (547) | 1.125+-0.024 (547) |
| 18-19 | 0.948+-0.077 (429) | 1.747+-0.128 (261) | 1.758+-0.141 (261) | 1.206+-0.056 (261) | 1.162+-0.049 (261) |
| 19-20 | 0.892+-0.149 (594) | n/a | n/a | n/a | n/a |
| 20-21 | -0.119+-0.392 (462) | n/a | n/a | n/a | n/a |

R vs wingcal_rmask (satstars, this annulus):

| wingcal_rmask range | N | median R precap | median R final | median data/model | median dolphot mag |
|---|---|---|---|---|---|
| 0 to 0.5642 | 1387 | 1.127 | 1.188 | 1.197 | 17.29 |
| 2 to 5 | 21 | 0.915 | 0.939 | 1.128 | 14.60 |

R vs cap_psf_frac (satstars, this annulus):

| cap_psf_frac range | N | median R precap | median R final | median data/model | median dolphot mag |
|---|---|---|---|---|---|
| -1 to 0.7 | 15 | 0.887 | 1.467 | 1.128 | 16.49 |
| 0.9999 to 1e+09 | 1376 | 1.127 | 1.185 | 1.196 | 17.27 |

### F150W main2kf annulus 22-34 px (0.68-1.05 arcsec)
normalisation: median unsaturated R = 0.0851 +- 0.0161 (N=163); specified-window value -0.0915 (N=214)

| mag bin | unsat daophot R | satstar R precap | satstar R final | satstar model-annulus / flux_fit (norm.) | satstar data/model annulus (raw) |
|---|---|---|---|---|---|
| 14-15 | n/a | 0.268+-0.015 (20) | 0.331+-0.020 (20) | 0.272+-0.006 (20) | 1.200+-0.079 (20) |
| 15-16 | n/a | 0.531+-0.076 (37) | 0.557+-0.096 (37) | 0.318+-0.022 (37) | 1.545+-0.098 (37) |
| 16-17 | n/a | 0.762+-0.052 (87) | 0.777+-0.061 (87) | 0.416+-0.049 (87) | 1.260+-0.071 (87) |
| 17-18 | n/a | 0.623+-0.083 (138) | 0.623+-0.089 (138) | 0.370+-0.053 (138) | 1.257+-0.188 (138) |
| 18-19 | 1.215+-0.241 (92) | 1.389+-0.389 (60) | 1.498+-0.420 (60) | 0.791+-0.140 (60) | 1.047+-0.181 (60) |
| 19-20 | 0.519+-0.226 (151) | n/a | n/a | n/a | n/a |
| 20-21 | -2.771+-0.769 (109) | n/a | n/a | n/a | n/a |

### F150W main2kf annulus 34-50 px (1.05-1.55 arcsec)
normalisation: median unsaturated R = 0.0757 +- 0.0614 (N=10); specified-window value -0.0159 (N=26)

| mag bin | unsat daophot R | satstar R precap | satstar R final | satstar model-annulus / flux_fit (norm.) | satstar data/model annulus (raw) |
|---|---|---|---|---|---|
| 14-15 | n/a | 0.216+-0.009 (5) | 0.252+-0.019 (5) | 0.211+-0.007 (5) | 1.161+-0.092 (5) |
| 16-17 | n/a | 2.014+-0.597 (8) | 2.070+-0.653 (8) | 1.475+-0.502 (8) | 1.498+-0.094 (8) |
| 17-18 | n/a | 0.628+-0.999 (13) | 0.642+-1.031 (13) | 0.626+-0.493 (13) | 1.307+-0.850 (13) |
| 18-19 | 0.469+-2.755 (5) | 4.768+-1.231 (7) | 5.124+-0.914 (7) | 2.811+-0.854 (7) | 1.805+-2.947 (7) |
| 19-20 | 1.160+-15.518 (10) | n/a | n/a | n/a | n/a |
| 20-21 | -1.114+-3.346 (19) | n/a | n/a | n/a | n/a |

### Sanity check, unsaturated daophot stars, F150W main2kf: aperture_sum / flux_fit vs dolphot mag (median, N), raw units (not normalised)

| aperture | 14-15 | 15-16 | 16-17 | 17-18 | 18-19 | 19-20 |
|---|---|---|---|---|---|---|
| annulus 8-14 px | n/a | 0.183+-0.043 (17) | 0.083+-0.019 (32) | 0.113+-0.022 (114) | 0.055+-0.002 (943) | 0.054+-0.003 (1237) |
| annulus 14-22 px | n/a | n/a | n/a | 0.190+-0.073 (19) | 0.044+-0.003 (429) | 0.041+-0.007 (594) |
| annulus 22-34 px | n/a | n/a | n/a | n/a | 0.103+-0.021 (92) | 0.044+-0.019 (151) |
| annulus 34-50 px | n/a | n/a | n/a | n/a | 0.035+-0.210 (5) | 0.088+-1.398 (10) |
| circle r=4 px | n/a | n/a | n/a | n/a | 0.813+-0.001 (1064) | 0.811+-0.001 (1563) |
| circle r=6 px | n/a | n/a | n/a | n/a | 0.853+-0.001 (1012) | 0.851+-0.001 (1493) |
| circle r=10 px | n/a | n/a | n/a | n/a | 0.910+-0.002 (884) | 0.904+-0.002 (1308) |

Satstar circular apertures with no NaN/SAT/DNU pixel and isolated: r=4 px: 0, r=6 px: 0, r=10 px: 0.

## F150W main2

satstar saturation faint edge (95th pct of dolphot mag of replaced satstars) = 18.43. Primary reference window for unsaturated stars: 17.93-19.43; specified window (onset+1..+3): 19.43-21.43.
Satstar rows (all frames, pooled): 6256; with dolphot match: 5627.
Satstars with any NaN in the largest circle (r=10 px): 3189; with any NaN/SAT/DNU pixel in it: 6256; median equivalent saturated radius: 2.8 px.

### F150W main2 annulus 8-14 px (0.25-0.43 arcsec)
normalisation: median unsaturated R = 0.0550 +- 0.0017 (N=1488); specified-window value 0.0441 (N=2015)

| mag bin | unsat daophot R | satstar R precap | satstar R final | satstar model-annulus / flux_fit (norm.) | satstar data/model annulus (raw) |
|---|---|---|---|---|---|
| 14-15 | n/a | 0.923+-0.008 (79) | 1.033+-0.017 (79) | 0.872+-0.004 (79) | 1.163+-0.019 (79) |
| 15-16 | 3.336+-0.799 (17) | 1.005+-0.011 (278) | 1.078+-0.013 (278) | 0.905+-0.005 (278) | 1.158+-0.008 (278) |
| 16-17 | 1.412+-0.354 (32) | 1.041+-0.016 (733) | 1.098+-0.012 (733) | 0.950+-0.009 (733) | 1.126+-0.007 (733) |
| 17-18 | 2.049+-0.448 (109) | 1.158+-0.028 (1103) | 1.220+-0.027 (1103) | 1.002+-0.010 (1103) | 1.094+-0.010 (1103) |
| 18-19 | 1.004+-0.032 (945) | 1.305+-0.043 (518) | 1.332+-0.056 (518) | 1.126+-0.028 (518) | 1.085+-0.021 (518) |
| 19-20 | 0.983+-0.056 (1237) | n/a | n/a | n/a | n/a |
| 20-21 | 0.632+-0.114 (981) | n/a | n/a | n/a | n/a |

R vs wingcal_rmask (satstars, this annulus):

| wingcal_rmask range | N | median R precap | median R final | median data/model | median dolphot mag |
|---|---|---|---|---|---|
| 0 to 0.5642 | 2680 | 1.104 | 1.160 | 1.121 | 17.31 |
| 2 to 5 | 31 | 0.888 | 0.909 | 1.032 | 14.65 |

R vs cap_psf_frac (satstars, this annulus):

| cap_psf_frac range | N | median R precap | median R final | median data/model | median dolphot mag |
|---|---|---|---|---|---|
| -1 to 0.7 | 30 | 0.944 | 1.125 | 1.122 | 17.79 |
| 0.9999 to 1e+09 | 2657 | 1.102 | 1.159 | 1.121 | 17.30 |

### F150W main2 annulus 14-22 px (0.43-0.68 arcsec)
normalisation: median unsaturated R = 0.0452 +- 0.0033 (N=677); specified-window value 0.0082 (N=967)

| mag bin | unsat daophot R | satstar R precap | satstar R final | satstar model-annulus / flux_fit (norm.) | satstar data/model annulus (raw) |
|---|---|---|---|---|---|
| 14-15 | n/a | 0.869+-0.020 (58) | 0.947+-0.017 (58) | 0.751+-0.003 (58) | 1.272+-0.022 (58) |
| 15-16 | n/a | 1.019+-0.024 (178) | 1.139+-0.029 (178) | 0.818+-0.014 (178) | 1.283+-0.016 (178) |
| 16-17 | n/a | 1.099+-0.022 (366) | 1.158+-0.032 (366) | 0.872+-0.022 (366) | 1.181+-0.020 (366) |
| 17-18 | 4.153+-1.617 (19) | 1.297+-0.046 (544) | 1.331+-0.047 (544) | 1.000+-0.028 (544) | 1.137+-0.029 (544) |
| 18-19 | 0.959+-0.068 (430) | 1.753+-0.141 (259) | 1.840+-0.144 (259) | 1.227+-0.061 (259) | 1.169+-0.045 (259) |
| 19-20 | 0.918+-0.149 (596) | n/a | n/a | n/a | n/a |
| 20-21 | -0.125+-0.354 (459) | n/a | n/a | n/a | n/a |

R vs wingcal_rmask (satstars, this annulus):

| wingcal_rmask range | N | median R precap | median R final | median data/model | median dolphot mag |
|---|---|---|---|---|---|
| 0 to 0.5642 | 1381 | 1.154 | 1.220 | 1.200 | 17.28 |
| 2 to 5 | 24 | 0.917 | 0.951 | 1.122 | 14.64 |

R vs cap_psf_frac (satstars, this annulus):

| cap_psf_frac range | N | median R precap | median R final | median data/model | median dolphot mag |
|---|---|---|---|---|---|
| -1 to 0.7 | 18 | 0.906 | 1.345 | 1.119 | 16.71 |
| 0.9999 to 1e+09 | 1370 | 1.154 | 1.220 | 1.199 | 17.27 |

### F150W main2 annulus 22-34 px (0.68-1.05 arcsec)
normalisation: median unsaturated R = 0.0780 +- 0.0166 (N=163); specified-window value -0.0915 (N=214)

| mag bin | unsat daophot R | satstar R precap | satstar R final | satstar model-annulus / flux_fit (norm.) | satstar data/model annulus (raw) |
|---|---|---|---|---|---|
| 14-15 | n/a | 0.292+-0.021 (20) | 0.360+-0.017 (20) | 0.297+-0.009 (20) | 1.207+-0.057 (20) |
| 15-16 | n/a | 0.571+-0.078 (37) | 0.611+-0.105 (37) | 0.345+-0.020 (37) | 1.487+-0.111 (37) |
| 16-17 | n/a | 0.820+-0.056 (87) | 0.836+-0.056 (87) | 0.452+-0.055 (87) | 1.233+-0.080 (87) |
| 17-18 | n/a | 0.672+-0.079 (136) | 0.691+-0.085 (136) | 0.406+-0.054 (136) | 1.237+-0.208 (136) |
| 18-19 | 1.309+-0.242 (92) | 1.528+-0.429 (62) | 1.623+-0.472 (62) | 0.851+-0.163 (62) | 1.059+-0.194 (62) |
| 19-20 | 0.690+-0.231 (151) | n/a | n/a | n/a | n/a |
| 20-21 | -3.114+-0.767 (109) | n/a | n/a | n/a | n/a |

### F150W main2 annulus 34-50 px (1.05-1.55 arcsec)
normalisation: median unsaturated R = 0.0758 +- 0.0716 (N=10); specified-window value -0.0170 (N=26)

| mag bin | unsat daophot R | satstar R precap | satstar R final | satstar model-annulus / flux_fit (norm.) | satstar data/model annulus (raw) |
|---|---|---|---|---|---|
| 14-15 | n/a | 0.218+-0.009 (5) | 0.250+-0.019 (5) | 0.211+-0.007 (5) | 1.157+-0.092 (5) |
| 16-17 | n/a | 1.992+-0.574 (8) | 2.038+-0.632 (8) | 1.484+-0.524 (8) | 1.477+-0.110 (8) |
| 17-18 | n/a | 0.632+-0.633 (13) | 0.660+-0.766 (13) | 0.621+-0.538 (13) | 1.303+-0.854 (13) |
| 18-19 | 0.466+-2.746 (5) | 4.788+-1.212 (7) | 5.103+-0.983 (7) | 2.734+-0.927 (7) | 1.806+-3.024 (7) |
| 19-20 | 1.157+-14.239 (10) | n/a | n/a | n/a | n/a |
| 20-21 | -1.190+-3.291 (19) | n/a | n/a | n/a | n/a |

### Sanity check, unsaturated daophot stars, F150W main2: aperture_sum / flux_fit vs dolphot mag (median, N), raw units (not normalised)

| aperture | 14-15 | 15-16 | 16-17 | 17-18 | 18-19 | 19-20 |
|---|---|---|---|---|---|---|
| annulus 8-14 px | n/a | 0.183+-0.044 (17) | 0.078+-0.019 (32) | 0.113+-0.023 (109) | 0.055+-0.002 (945) | 0.054+-0.003 (1237) |
| annulus 14-22 px | n/a | n/a | n/a | 0.188+-0.078 (19) | 0.043+-0.003 (430) | 0.041+-0.007 (596) |
| annulus 22-34 px | n/a | n/a | n/a | n/a | 0.102+-0.020 (92) | 0.054+-0.017 (151) |
| annulus 34-50 px | n/a | n/a | n/a | n/a | 0.035+-0.210 (5) | 0.088+-1.110 (10) |
| circle r=4 px | n/a | n/a | n/a | n/a | 0.813+-0.001 (1063) | 0.811+-0.001 (1565) |
| circle r=6 px | n/a | n/a | n/a | n/a | 0.853+-0.001 (1011) | 0.851+-0.001 (1495) |
| circle r=10 px | n/a | n/a | n/a | n/a | 0.910+-0.002 (883) | 0.904+-0.002 (1309) |

Satstar circular apertures with no NaN/SAT/DNU pixel and isolated: r=4 px: 0, r=6 px: 0, r=10 px: 0.

## Summary: normalised R of satstars at 13-15 and 16-18 mag

| band | arm | annulus (px) | mag | satstar precap | satstar final | unsat daophot | satstar data/model |
|---|---|---|---|---|---|---|---|
| F250M | main2kf | 5-8 | 13-15 | 1.380+-0.015 (72) | 1.417+-0.025 (72) | n/a | 1.212+-0.016 (72) |
| F250M | main2kf | 5-8 | 16-18 | 1.526+-0.105 (116) | 1.579+-0.096 (116) | 0.995+-0.046 (633) | 1.133+-0.039 (116) |
| F250M | main2kf | 8-12 | 13-15 | 1.243+-0.098 (92) | 1.334+-0.087 (92) | n/a | 1.122+-0.020 (92) |
| F250M | main2kf | 8-12 | 16-18 | 2.062+-0.153 (45) | 2.145+-0.158 (45) | 0.923+-0.112 (244) | 1.789+-0.303 (45) |
| F250M | main2kf | 12-18 | 13-15 | 0.935+-0.156 (9) | 0.953+-0.161 (9) | n/a | 1.609+-0.280 (9) |
| F250M | main2kf | 12-18 | 16-18 | n/a | n/a | -0.165+-1.333 (14) | n/a |
| F250M | main2 | 5-8 | 13-15 | 1.482+-0.026 (69) | 1.551+-0.029 (69) | 1.492+-0.049 (4) | 1.286+-0.025 (69) |
| F250M | main2 | 5-8 | 16-18 | 1.628+-0.110 (117) | 1.640+-0.105 (117) | 0.996+-0.046 (631) | 1.145+-0.046 (117) |
| F250M | main2 | 8-12 | 13-15 | 1.204+-0.083 (88) | 1.345+-0.094 (88) | 1.039+-0.578 (5) | 1.177+-0.017 (88) |
| F250M | main2 | 8-12 | 16-18 | 2.107+-0.145 (43) | 2.111+-0.169 (43) | 0.953+-0.135 (244) | 1.823+-0.341 (43) |
| F250M | main2 | 12-18 | 13-15 | 1.799+-0.321 (9) | 1.799+-0.286 (9) | n/a | 1.663+-0.278 (9) |
| F250M | main2 | 12-18 | 16-18 | n/a | n/a | -1.030+-2.195 (14) | n/a |
| F150W | main2kf | 8-14 | 13-15 | 0.929+-0.009 (79) | 1.029+-0.018 (79) | n/a | 1.160+-0.023 (79) |
| F150W | main2kf | 8-14 | 16-18 | 1.101+-0.010 (1839) | 1.155+-0.014 (1839) | 2.029+-0.211 (146) | 1.105+-0.005 (1839) |
| F150W | main2kf | 14-22 | 13-15 | 0.849+-0.023 (58) | 0.927+-0.014 (58) | n/a | 1.267+-0.021 (58) |
| F150W | main2kf | 14-22 | 16-18 | 1.140+-0.026 (911) | 1.195+-0.026 (911) | 4.113+-1.117 (22) | 1.162+-0.012 (911) |
| F150W | main2kf | 22-34 | 13-15 | 0.268+-0.019 (20) | 0.331+-0.020 (20) | n/a | 1.200+-0.072 (20) |
| F150W | main2kf | 22-34 | 16-18 | 0.688+-0.065 (225) | 0.700+-0.070 (225) | n/a | 1.260+-0.079 (225) |
| F150W | main2kf | 34-50 | 13-15 | 0.216+-0.009 (5) | 0.252+-0.019 (5) | n/a | 1.161+-0.097 (5) |
| F150W | main2kf | 34-50 | 16-18 | 0.721+-0.556 (21) | 0.763+-0.589 (21) | n/a | 1.397+-0.160 (21) |
| F150W | main2 | 8-14 | 13-15 | 0.923+-0.009 (79) | 1.033+-0.017 (79) | n/a | 1.163+-0.021 (79) |
| F150W | main2 | 8-14 | 16-18 | 1.100+-0.010 (1836) | 1.149+-0.012 (1836) | 2.041+-0.209 (141) | 1.110+-0.006 (1836) |
| F150W | main2 | 14-22 | 13-15 | 0.869+-0.019 (58) | 0.947+-0.015 (58) | n/a | 1.272+-0.021 (58) |
| F150W | main2 | 14-22 | 16-18 | 1.182+-0.033 (910) | 1.239+-0.028 (910) | 4.134+-1.145 (22) | 1.171+-0.012 (910) |
| F150W | main2 | 22-34 | 13-15 | 0.292+-0.016 (20) | 0.360+-0.016 (20) | n/a | 1.207+-0.061 (20) |
| F150W | main2 | 22-34 | 16-18 | 0.724+-0.054 (223) | 0.764+-0.064 (223) | n/a | 1.233+-0.090 (223) |
| F150W | main2 | 34-50 | 13-15 | 0.218+-0.010 (5) | 0.250+-0.019 (5) | n/a | 1.157+-0.085 (5) |
| F150W | main2 | 34-50 | 16-18 | 0.926+-0.594 (21) | 0.926+-0.566 (21) | n/a | 1.367+-0.145 (21) |

## Bright/faint ratios of satstar R (13-15 mag over 16-18 mag; independent of the unsaturated normalisation)

| band | arm | annulus (px) | precap | final | data/model |
|---|---|---|---|---|---|
| F250M | main2kf | 5-8 | 0.905+-0.063 | 0.897+-0.057 | 1.070+-0.039 |
| F250M | main2kf | 8-12 | 0.603+-0.065 | 0.622+-0.061 | 0.627+-0.107 |
| F250M | main2 | 5-8 | 0.911+-0.064 | 0.946+-0.063 | 1.123+-0.050 |
| F250M | main2 | 8-12 | 0.571+-0.056 | 0.637+-0.068 | 0.645+-0.121 |
| F150W | main2kf | 8-14 | 0.844+-0.011 | 0.891+-0.019 | 1.050+-0.022 |
| F150W | main2kf | 14-22 | 0.745+-0.027 | 0.776+-0.021 | 1.091+-0.021 |
| F150W | main2kf | 22-34 | 0.389+-0.046 | 0.473+-0.055 | 0.952+-0.083 |
| F150W | main2kf | 34-50 | 0.300+-0.231 | 0.331+-0.257 | 0.831+-0.118 |
| F150W | main2 | 8-14 | 0.839+-0.011 | 0.899+-0.018 | 1.048+-0.019 |
| F150W | main2 | 14-22 | 0.735+-0.026 | 0.765+-0.021 | 1.087+-0.021 |
| F150W | main2 | 22-34 | 0.403+-0.037 | 0.471+-0.045 | 0.979+-0.087 |
| F150W | main2 | 34-50 | 0.235+-0.151 | 0.270+-0.166 | 0.846+-0.109 |
