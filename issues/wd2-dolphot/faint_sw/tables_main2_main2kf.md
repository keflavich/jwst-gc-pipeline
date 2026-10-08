# Faint SW movers main2kf vs main2 (dolphot mag > 20, moved = |B-A| > 0.1 mag)

## Q1 sign and closeness to dolphot
| band | moved | median dolphot | main2kf brighter (B-A<0) | main2kf fainter | median B-A | closer to dolphot: B / A, B brighter | B / A, B fainter | median abs(dm) A -> B (all moved) |
|---|---|---|---|---|---|---|---|---|
| F150W | 60 | 23.5 | 33 | 27 | -0.12 | 32 / 1 | 1 / 26 | 0.666 -> 0.711 |
| F182M | 29 | 22.3 | 15 | 14 | -0.11 | 15 / 0 | 0 / 14 | 3.086 -> 3.279 |
| F187N | 7 | 21.2 | 3 | 4 | +0.11 | 3 / 0 | 0 / 4 | 0.249 -> 0.962 |
| F162M | 48 | 23.3 | 27 | 21 | -0.11 | 27 / 0 | 1 / 20 | 1.291 -> 1.242 |

## Q1b signed dm to dolphot (median), moved stars
| band | median dmA | median dmB | median dmA, B brighter | median dmB, B brighter | median dmA, B fainter | median dmB, B fainter |
|---|---|---|---|---|---|---|
| F150W | +0.61 | +0.65 | +1.11 | +0.55 | +0.13 | +0.73 |
| F182M | +3.09 | +3.28 | +3.62 | +3.29 | +0.91 | +2.44 |
| F187N | +0.25 | +0.96 | +0.23 | +0.01 | +0.53 | +1.76 |
| F162M | +1.29 | +1.24 | +1.26 | +0.93 | +1.32 | +1.58 |

## Q2 distance to nearest satstar (same-band per-frame m7 satstar catalogs, A and B pooled; arcsec)
| band | group | N | median dsat | frac < 2" | frac < 5" | frac < 10" | median dsat A only | median dsat B only |
|---|---|---|---|---|---|---|---|---|
| F150W | moved | 60 | 0.6 | 0.983 | 1.000 | 1.000 | 0.6 | 0.6 |
| F150W | control | 8327 | 2.1 | 0.479 | 0.921 | 1.000 | 2.1 | 2.1 |
| F182M | moved | 29 | 0.5 | 0.931 | 1.000 | 1.000 | 0.5 | 0.5 |
| F182M | control | 7382 | 4.1 | 0.210 | 0.607 | 0.917 | 4.1 | 4.1 |
| F187N | moved | 7 | 0.9 | 0.857 | 0.857 | 0.857 | 0.9 | 0.9 |
| F187N | control | 3672 | 5.8 | 0.128 | 0.433 | 0.794 | 5.8 | 5.8 |
| F162M | moved | 48 | 0.9 | 1.000 | 1.000 | 1.000 | 0.9 | 0.9 |
| F162M | control | 7869 | 2.6 | 0.384 | 0.841 | 0.988 | 2.6 | 2.6 |

## Q3 detections on collapsed frames (B/A per-frame satstar count > 2); A m7 daophot rows within 0.1"
| band | group | stars | A detections | on collapsed frames | frac | stars with any collapsed detection | stars with all detections collapsed |
|---|---|---|---|---|---|---|---|
| F150W | moved | 60 | 147 | 0 | 0.000 | 0 | 0 |
| F150W | control | 8327 | 31421 | 0 | 0.000 | 0 | 0 |
| F182M | moved | 29 | 35 | 0 | 0.000 | 0 | 0 |
| F182M | control | 7382 | 27936 | 0 | 0.000 | 0 | 0 |
| F187N | moved | 7 | 18 | 0 | 0.000 | 0 | 0 |
| F187N | control | 3672 | 14264 | 0 | 0.000 | 0 | 0 |
| F162M | moved | 48 | 79 | 0 | 0.000 | 0 | 0 |
| F162M | control | 7869 | 30016 | 0 | 0.000 | 0 | 0 |

frames per band: (det exp nsatA nsatB collapsed)
- F150W: 32 frames with detections, 0 collapsed: 
- F182M: 32 frames with detections, 0 collapsed: 
- F187N: 32 frames with detections, 0 collapsed: 
- F162M: 32 frames with detections, 0 collapsed: 

## Q4 B model - A model at the star position (7x7 PSF-weighted, Gaussian PSF, counts in image units) vs star flux
Dpsf = sum(D*P)/sum(P^2) with P a normalized Gaussian of the nominal band FWHM; predicted fB = fA - Dpsf (B residual = data - Bmodel).
| band | group | star-frames | median Dpsf/fA | frac Dpsf/fA < -0.1 | frac > +0.1 | frac abs < 0.02 | median abs(Dpsf/fA) | median (fB-fA)/fA observed (B row found) | median predicted (-Dpsf/fA) same rows |
|---|---|---|---|---|---|---|---|---|---|
| F150W | moved | 147 | -0.000 | 0.007 | 0.007 | 0.959 | 0.001 | -0.000 | +0.000 |
| F150W | control | 31421 | +0.000 | 0.000 | 0.000 | 0.998 | 0.000 | +0.000 | +0.000 |
| F182M | moved | 35 | -0.001 | 0.000 | 0.057 | 0.857 | 0.002 | -0.000 | +0.001 |
| F182M | control | 27936 | +0.000 | 0.000 | 0.000 | 0.998 | 0.000 | +0.000 | +0.000 |
| F187N | moved | 18 | -0.001 | 0.000 | 0.056 | 0.889 | 0.002 | +0.000 | +0.001 |
| F187N | control | 14264 | +0.000 | 0.000 | 0.000 | 0.999 | 0.000 | +0.000 | +0.000 |
| F162M | moved | 79 | +0.000 | 0.000 | 0.013 | 0.823 | 0.003 | -0.000 | -0.000 |
| F162M | control | 30016 | +0.000 | 0.000 | 0.000 | 0.997 | 0.000 | +0.000 | +0.000 |

## Q4b per-star predicted vs observed B-A (mag); predicted = -2.5 log10(sum(fA - Dpsf) / sum(fA)) over A-detected frames
| band | moved stars with frames | Spearman rho(pred, observed) | sign agreement | median observed | median predicted | median abs(pred) / abs(obs) | fraction |pred| > 0.05 |
|---|---|---|---|---|---|---|---|
| F150W | 40 | -0.24 | 0.40 | +0.15 | -0.000 | 0.00 | 0.05 |
| F182M | 10 | 0.16 | 0.60 | +0.18 | -0.002 | 0.01 | 0.20 |
| F187N | 6 | 0.09 | 0.50 | +0.00 | -0.000 | 0.01 | 0.17 |
| F162M | 24 | 0.02 | 0.46 | -0.11 | +0.000 | 0.03 | 0.12 |

## Q5 other properties (medians; moved vs control)
| band | group | N | m8 nmatch A | nmatch B | qfit A | qfit B | spike_artifact A / B | near_saturated A / B | forced_filled A / B | neighbours within 1" (m8, A) | satstar_nframes A / B | rep.sat A / B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F150W | moved | 60 | 4 | 4 | 0.11 | 0.10 | 0.033 / 0.033 | 0.167 / 0.200 | 0.667 / 0.700 | 6 | 0 / 0 | 0.017 / 0.017 |
| F150W | control | 8327 | 4 | 4 | 0.08 | 0.08 | 0.004 / 0.004 | 0.027 / 0.027 | 0.085 / 0.085 | 2 | 0 / 0 | 0.000 / 0.000 |
| F182M | moved | 29 | 4 | 3 | 0.32 | 0.57 | 0.103 / 0.103 | 0.034 / 0.000 | 0.862 / 0.931 | 9 | 0 / 0 | 0.000 / 0.000 |
| F182M | control | 7382 | 4 | 4 | 0.08 | 0.08 | 0.004 / 0.004 | 0.008 / 0.008 | 0.135 / 0.135 | 2 | 0 / 0 | 0.000 / 0.000 |
| F187N | moved | 7 | 3 | 2 | 0.44 | 0.53 | 0.000 / 0.000 | 0.000 / 0.000 | 0.714 / 0.714 | 9 | 0 / 0 | 0.000 / 0.000 |
| F187N | control | 3672 | 4 | 4 | 0.21 | 0.21 | 0.002 / 0.002 | 0.003 / 0.003 | 0.264 / 0.264 | 3 | 0 / 0 | 0.000 / 0.000 |
| F162M | moved | 48 | 3 | 4 | 0.38 | 0.37 | 0.042 / 0.042 | 0.021 / 0.104 | 0.812 / 0.729 | 8 | 0 / 0 | 0.000 / 0.000 |
| F162M | control | 7869 | 4 | 4 | 0.09 | 0.09 | 0.005 / 0.004 | 0.021 / 0.021 | 0.078 / 0.078 | 2 | 0 / 0 | 0.000 / 0.000 |

## Q5b change in contributing frames (nmatch B - nmatch A), moved vs control
| band | group | median | frac B < A | frac B > A |
|---|---|---|---|---|
| F150W | moved | +0 | 0.000 | 0.000 |
| F150W | control | +0 | 0.000 | 0.000 |
| F182M | moved | +0 | 0.000 | 0.000 |
| F182M | control | +0 | 0.000 | 0.000 |
| F187N | moved | -0 | 0.143 | 0.000 |
| F187N | control | +0 | 0.000 | 0.000 |
| F162M | moved | -0 | 0.042 | 0.000 |
| F162M | control | +0 | 0.000 | 0.000 |