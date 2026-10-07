# Faint SW movers mainfcbgkf vs mainfcbg (dolphot mag > 20, moved = |B-A| > 0.1 mag)

## Q1 sign and closeness to dolphot
| band | moved | median dolphot | mainfcbgkf brighter (B-A<0) | mainfcbgkf fainter | median B-A | closer to dolphot: B / A, B brighter | B / A, B fainter | median abs(dm) A -> B (all moved) |
|---|---|---|---|---|---|---|---|---|
| F150W | 127 | 24.3 | 46 | 81 | +0.17 | 43 / 3 | 3 / 78 | 0.206 -> 0.509 |
| F182M | 64 | 22.5 | 24 | 40 | +0.14 | 24 / 0 | 3 / 37 | 0.346 -> 1.110 |
| F187N | 51 | 21.5 | 20 | 31 | +0.14 | 18 / 2 | 3 / 28 | 0.237 -> 0.317 |
| F162M | 113 | 23.6 | 39 | 74 | +0.16 | 38 / 1 | 8 / 66 | 0.319 -> 0.466 |

## Q1b signed dm to dolphot (median), moved stars
| band | median dmA | median dmB | median dmA, B brighter | median dmB, B brighter | median dmA, B fainter | median dmB, B fainter |
|---|---|---|---|---|---|---|
| F150W | +0.17 | +0.50 | +0.70 | +0.20 | +0.05 | +0.55 |
| F182M | +0.35 | +1.11 | +2.44 | +2.07 | +0.16 | +0.97 |
| F187N | +0.23 | +0.32 | +0.50 | +0.13 | +0.10 | +0.35 |
| F162M | +0.32 | +0.47 | +0.69 | +0.37 | +0.09 | +0.57 |

## Q2 distance to nearest satstar (same-band per-frame m7 satstar catalogs, A and B pooled; arcsec)
| band | group | N | median dsat | frac < 2" | frac < 5" | frac < 10" | median dsat A only | median dsat B only |
|---|---|---|---|---|---|---|---|---|
| F150W | moved | 127 | 0.8 | 0.961 | 1.000 | 1.000 | 1.0 | 0.8 |
| F150W | control | 8259 | 2.1 | 0.474 | 0.920 | 1.000 | 2.2 | 2.1 |
| F182M | moved | 64 | 0.6 | 0.875 | 0.969 | 1.000 | 1.2 | 0.6 |
| F182M | control | 7342 | 4.1 | 0.206 | 0.607 | 0.918 | 4.4 | 4.1 |
| F187N | moved | 51 | 1.3 | 0.706 | 0.824 | 0.941 | 5.7 | 1.3 |
| F187N | control | 3625 | 5.9 | 0.122 | 0.428 | 0.793 | 7.1 | 5.9 |
| F162M | moved | 113 | 1.0 | 0.956 | 0.991 | 1.000 | 2.3 | 1.0 |
| F162M | control | 7808 | 2.6 | 0.381 | 0.842 | 0.988 | 2.9 | 2.6 |

## Q3 detections on collapsed frames (B/A per-frame satstar count > 2); A m7 daophot rows within 0.1"
| band | group | stars | A detections | on collapsed frames | frac | stars with any collapsed detection | stars with all detections collapsed |
|---|---|---|---|---|---|---|---|
| F150W | moved | 127 | 347 | 157 | 0.452 | 60 | 32 |
| F150W | control | 8259 | 31236 | 5192 | 0.166 | 1801 | 895 |
| F182M | moved | 64 | 131 | 73 | 0.557 | 21 | 21 |
| F182M | control | 7342 | 26970 | 3211 | 0.119 | 853 | 853 |
| F187N | moved | 51 | 170 | 88 | 0.518 | 37 | 19 |
| F187N | control | 3625 | 14113 | 6973 | 0.494 | 2692 | 1327 |
| F162M | moved | 113 | 263 | 182 | 0.692 | 55 | 55 |
| F162M | control | 7808 | 29842 | 6933 | 0.232 | 1807 | 1807 |

frames per band: (det exp nsatA nsatB collapsed)
- F150W: 32 frames with detections, 6 collapsed: nrca2e3 5/90; nrca2e4 4/87; nrca3e1 6/93; nrca3e2 6/96; nrca3e3 4/98; nrca3e4 5/94
- F182M: 31 frames with detections, 4 collapsed: nrca2e1 3/31; nrca2e2 2/32; nrca2e3 1/33; nrca2e4 3/33
- F187N: 32 frames with detections, 16 collapsed: nrca1e1 7/17; nrca1e2 7/17; nrca1e3 5/16; nrca1e4 6/17; nrca2e3 8/17; nrca2e4 8/17; nrca3e2 7/15; nrca4e1 2/13; nrca4e2 2/14; nrca4e3 2/14; nrca4e4 2/14; nrcb1e2 55/119; nrcb4e1 10/27; nrcb4e2 12/28; nrcb4e3 11/29; nrcb4e4 11/26
- F162M: 32 frames with detections, 8 collapsed: nrca2e1 4/61; nrca2e2 3/62; nrca2e3 3/64; nrca2e4 3/63; nrca3e1 1/70; nrca3e2 2/69; nrca3e3 4/68; nrca3e4 5/69

## Q4 B model - A model at the star position (7x7 PSF-weighted, Gaussian PSF, counts in image units) vs star flux
Dpsf = sum(D*P)/sum(P^2) with P a normalized Gaussian of the nominal band FWHM; predicted fB = fA - Dpsf (B residual = data - Bmodel).
| band | group | star-frames | median Dpsf/fA | frac Dpsf/fA < -0.1 | frac > +0.1 | frac abs < 0.02 | median abs(Dpsf/fA) | median (fB-fA)/fA observed (B row found) | median predicted (-Dpsf/fA) same rows |
|---|---|---|---|---|---|---|---|---|---|
| F150W | moved | 347 | +0.010 | 0.003 | 0.205 | 0.536 | 0.013 | -0.000 | -0.008 |
| F150W | control | 31236 | +0.000 | 0.000 | 0.001 | 0.990 | 0.000 | +0.000 | +0.000 |
| F182M | moved | 131 | +0.037 | 0.000 | 0.420 | 0.405 | 0.041 | -0.001 | -0.007 |
| F182M | control | 26970 | +0.000 | 0.000 | 0.000 | 0.994 | 0.000 | +0.000 | +0.000 |
| F187N | moved | 170 | +0.087 | 0.000 | 0.441 | 0.324 | 0.087 | -0.001 | -0.075 |
| F187N | control | 14113 | +0.000 | 0.000 | 0.004 | 0.985 | 0.000 | +0.000 | +0.000 |
| F162M | moved | 263 | +0.100 | 0.000 | 0.498 | 0.285 | 0.100 | -0.008 | -0.055 |
| F162M | control | 29842 | +0.000 | 0.000 | 0.001 | 0.988 | 0.000 | +0.000 | +0.000 |

## Q4b per-star predicted vs observed B-A (mag); predicted = -2.5 log10(sum(fA - Dpsf) / sum(fA)) over A-detected frames
| band | moved stars with frames | Spearman rho(pred, observed) | sign agreement | median observed | median predicted | median abs(pred) / abs(obs) | fraction |pred| > 0.05 |
|---|---|---|---|---|---|---|---|
| F150W | 98 | 0.18 | 0.66 | +0.19 | +0.031 | 0.08 | 0.41 |
| F182M | 40 | 0.35 | 0.60 | +0.17 | +0.032 | 0.16 | 0.47 |
| F187N | 48 | 0.53 | 0.54 | +0.13 | +0.113 | 0.49 | 0.65 |
| F162M | 80 | 0.48 | 0.66 | +0.16 | +0.119 | 0.48 | 0.61 |

## Q5 other properties (medians; moved vs control)
| band | group | N | m8 nmatch A | nmatch B | qfit A | qfit B | spike_artifact A / B | near_saturated A / B | forced_filled A / B | neighbours within 1" (m8, A) | satstar_nframes A / B | rep.sat A / B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F150W | moved | 127 | 4 | 4 | 0.27 | 0.13 | 0.039 / 0.024 | 0.134 / 0.126 | 0.480 / 0.685 | 6 | 0 / 0 | 0.008 / 0.008 |
| F150W | control | 8259 | 4 | 4 | 0.08 | 0.08 | 0.004 / 0.004 | 0.026 / 0.027 | 0.080 / 0.080 | 3 | 0 / 0 | 0.000 / 0.000 |
| F182M | moved | 64 | 4 | 4 | 0.22 | 0.19 | 0.062 / 0.047 | 0.062 / 0.047 | 0.688 / 0.812 | 10 | 0 / 0 | 0.000 / 0.000 |
| F182M | control | 7342 | 4 | 4 | 0.08 | 0.08 | 0.004 / 0.004 | 0.009 / 0.008 | 0.132 / 0.132 | 3 | 0 / 0 | 0.000 / 0.000 |
| F187N | moved | 51 | 4 | 4 | 0.35 | 0.30 | 0.020 / 0.020 | 0.000 / 0.000 | 0.686 / 0.686 | 16 | 0 / 0 | 0.000 / 0.000 |
| F187N | control | 3625 | 4 | 4 | 0.21 | 0.21 | 0.002 / 0.001 | 0.003 / 0.003 | 0.258 / 0.259 | 3 | 0 / 0 | 0.000 / 0.000 |
| F162M | moved | 113 | 4 | 4 | 0.37 | 0.28 | 0.080 / 0.053 | 0.000 / 0.062 | 0.673 / 0.637 | 9 | 0 / 0 | 0.000 / 0.000 |
| F162M | control | 7808 | 4 | 4 | 0.09 | 0.09 | 0.005 / 0.004 | 0.021 / 0.021 | 0.074 / 0.074 | 3 | 0 / 0 | 0.000 / 0.000 |

## Q5b change in contributing frames (nmatch B - nmatch A), moved vs control
| band | group | median | frac B < A | frac B > A |
|---|---|---|---|---|
| F150W | moved | +0 | 0.024 | 0.000 |
| F150W | control | +0 | 0.001 | 0.000 |
| F182M | moved | +0 | 0.000 | 0.016 |
| F182M | control | +0 | 0.000 | 0.000 |
| F187N | moved | +0 | 0.020 | 0.020 |
| F187N | control | +0 | 0.000 | 0.000 |
| F162M | moved | +0 | 0.027 | 0.009 |
| F162M | control | +0 | 0.000 | 0.000 |