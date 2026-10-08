# Third-reference test of saturated-star photometry (A = ours main2, B = dolphot)

## Reference catalog and its limits
- Zeidler et al. 2015 (HST WFC3/IR F125W/F160W) is not in VizieR (J/AJ/150/78 and the Zeidler papers II/III do not exist there; searched by ID, keyword, and the TAP tap_schema). No MAST HLSP found. The arXiv source of Zeidler 2015 has no catalog file. Zeidler 2015 (Sect. 5) also states that WFC3/IR is not saturated for these data; saturation affects only the ACS optical bands.
- Fallback used: Ascenso et al. 2007, A&A 466, 137 (VLT/ISAAC J, H, Ks; VizieR J/A+A/466/137/w2phot, 5727 stars), saved as `ascenso2007_w2phot.fits`. Ground-based, seeing about 0.4-0.5 arcsec, so blending is the main systematic. Faintest H about 20.6 (errors reach 0.1 mag near H~18-19); brightest H = 9.6, so the reference is not saturated at 14-19 mag. No HST F160W is used; H (F150W) and Ks (F200W) replace it, with colours J-H or H-Ks.

## Method (script `thirdref.py`, log `thirdref_log.txt`)
1. Rows of `matched_Q_main2.fits` (B positions, A magnitudes from our_*) matched to the Ascenso catalog. Global offset (ground minus JWST, iterated, 7068 pairs within 1 arcsec): dRA*cos(dec) = +0.042", dDec = -0.067"; applied. Mutual nearest neighbours within 0.3" (median separation 0.08"), 2172 pairs.
2. Isolation: no other Ascenso source within 1.2", and the summed flux of all other B-catalog sources within 0.8" of the star below 5% of the star's flux in the band (neighbour blending seen at ground resolution). Ground error < 0.1 mag in the band (colour error < 0.15).
3. JWST-saturated = our_replaced_saturated or our_is_saturated in that band. Colour term fit (JWST - ground = a + b*colour, 3-sigma clipped, separate for A and B) on JWST-unsaturated stars only; residual = JWST - ground - fitted term (the intercept absorbs the Vega/AB-like zero point and offset to the ground system, so no analyze.py ZP is needed).
4. Saturation edge = 95th percentile of the ground magnitude of JWST-saturated stars (F150W: H = 18.0; F200W: Ks = 15.8). Step = median residual of saturated stars minus median residual of unsaturated stars with ground magnitude in [edge, edge + w]. Uncertainty = quadrature sum of 1.253*MAD/sqrt(N) terms (statistical only).

## Key numbers (default selection)
| test | cat | unsat window w=1 mag: N, med | sat: N, med | step (sat - unsat) |
|---|---|---|---|---|
| F150W-H, colour J-H | A | 39, -0.006 | 436, -0.060 | -0.054 +- 0.047 |
| F150W-H, colour J-H | B | 39, -0.038 | 436, -0.043 | -0.005 +- 0.045 |
| F150W-H, colour H-Ks | A | 96, -0.033 | 525, +0.001 | +0.034 +- 0.028 |
| F150W-H, colour H-Ks | B | 96, -0.031 | 525, +0.049 | +0.080 +- 0.022 |
| F200W-Ks, colour H-Ks | A | 180, +0.037 | 184, -0.012 | -0.049 +- 0.013 |
| F200W-Ks, colour H-Ks | B | 180, +0.042 | 184, +0.023 | -0.019 +- 0.015 |

For F200W the step depends on the unsaturated window width because the unsaturated residual drifts with magnitude (+0.04 at Ks 16-17 to -0.03 at Ks 17-18, a ground-catalog/colour-term systematic of about 0.03): with w = 1.5 mag A -0.028 +- 0.012, B +0.007 +- 0.013; with w = 2 mag A -0.015 +- 0.011, B +0.023 +- 0.013. Looser cuts (`thirdref_log_loose.txt`: error < 0.2, blend < 20%, 0.8" isolation, 0.4" match; N sat/unsat F200W 307/763) give A -0.029 +- 0.011 and B +0.006 +- 0.012 (w=1), A -0.019, B +0.022 (w=1.5). Strict cuts (`thirdref_log_strict.txt`) leave no unsaturated F150W stars.

A-B direct (same stars, default selection): F150W saturated median -0.051 (N=436, J-H sample) / -0.048 (N=525, H-Ks sample), unsaturated just fainter than the edge -0.025 (N=39) / -0.020 (N=98): difference about -0.026 / -0.028. F200W saturated -0.066 (N=184) versus unsaturated -0.025 (N=283, edge..edge+1.5): difference -0.040. A-B by B magnitude, saturated F150W stars: 14-15 -0.098 (N=25), 15-16 -0.092, 16-17 -0.073, 17-18 -0.037, 18-19 -0.014 (N=81); unsaturated 18-19 -0.029 (N=37), so A-B for saturated stars trends from -0.10 to the unsaturated value of about -0.02 to -0.03 by 18-19 mag.

## Residual tables (default selection; residual = JWST - ground - colour term)
```
iter 0 offset ground-minus-JWST (dRA*cos, dDec) arcsec (np.float64(0.03235553669857422), np.float64(-0.0559175458960226)) N 7067
iter 1 offset ground-minus-JWST (dRA*cos, dDec) arcsec (np.float64(0.039685413846643536), np.float64(-0.06575099474162016)) N 7069
iter 2 offset ground-minus-JWST (dRA*cos, dDec) arcsec (np.float64(0.04212230217003129), np.float64(-0.06704663225889362)) N 7068
mutual matches within 0.3 arcsec: 2172  median sep arcsec 0.07779020429734287

=== F150W-H vs J-H: base sample N=479  (sat 436, unsat 43)
A: color term a=+0.253 b=+0.147 (N fit 41, rms 0.230); saturation edge (p95 ground mag of sat stars) = 17.99
A residual (JWST - ground - colorterm) vs A JWST mag, 1-mag bins: bin | unsat N med MAD | sat N med MAD
  13-14:     0 +nan nan |     0 +nan nan
  14-15:     0 +nan nan |    28 -0.202 0.108
  15-16:     0 +nan nan |    59 -0.102 0.097
  16-17:     0 +nan nan |   115 -0.044 0.128
  17-18:     0 +nan nan |   158 -0.034 0.157
  18-19:    34 -0.018 0.210 |    76 -0.042 0.172
  19-20:     9 +0.321 0.251 |     0 +nan nan
  20-21:     0 +nan nan |     0 +nan nan
  21-22:     0 +nan nan |     0 +nan nan
  22-23:     0 +nan nan |     0 +nan nan
  step[A] unsat window ground in [17.99,18.99] N=39 med=-0.006; sat all N=436 med=-0.060 -> step -0.054 +- 0.047; sat within 1.5 mag of edge N=245 med=-0.045 step -0.038 +- 0.048; sat brighter N=191 med=-0.075
  step[A] unsat window ground in [17.99,19.49] N=39 med=-0.006; sat all N=436 med=-0.060 -> step -0.054 +- 0.047; sat within 1.5 mag of edge N=245 med=-0.045 step -0.038 +- 0.048; sat brighter N=191 med=-0.075
  step[A] unsat window ground in [17.99,19.99] N=39 med=-0.006; sat all N=436 med=-0.060 -> step -0.054 +- 0.047; sat within 1.5 mag of edge N=245 med=-0.045 step -0.038 +- 0.048; sat brighter N=191 med=-0.075
B: color term a=+0.155 b=+0.261 (N fit 43, rms 0.235); saturation edge (p95 ground mag of sat stars) = 17.99
B residual (JWST - ground - colorterm) vs B JWST mag, 1-mag bins: bin | unsat N med MAD | sat N med MAD
  13-14:     0 +nan nan |     0 +nan nan
  14-15:     0 +nan nan |    25 -0.088 0.040
  15-16:     0 +nan nan |    55 -0.045 0.089
  16-17:     0 +nan nan |   117 -0.027 0.083
  17-18:     0 +nan nan |   158 -0.043 0.127
  18-19:    37 -0.038 0.209 |    81 -0.091 0.134
  19-20:     6 +0.246 0.029 |     0 +nan nan
  20-21:     0 +nan nan |     0 +nan nan
  21-22:     0 +nan nan |     0 +nan nan
  22-23:     0 +nan nan |     0 +nan nan
  step[B] unsat window ground in [17.99,18.99] N=39 med=-0.038; sat all N=436 med=-0.043 -> step -0.005 +- 0.045; sat within 1.5 mag of edge N=245 med=-0.061 step -0.023 +- 0.046; sat brighter N=191 med=-0.031
  step[B] unsat window ground in [17.99,19.49] N=39 med=-0.038; sat all N=436 med=-0.043 -> step -0.005 +- 0.045; sat within 1.5 mag of edge N=245 med=-0.061 step -0.023 +- 0.046; sat brighter N=191 med=-0.031
  step[B] unsat window ground in [17.99,19.99] N=39 med=-0.038; sat all N=436 med=-0.043 -> step -0.005 +- 0.045; sat within 1.5 mag of edge N=245 med=-0.061 step -0.023 +- 0.046; sat brighter N=191 med=-0.031
A-B direct: sat N=436 med=-0.051  unsat(edge..edge+1.5) N=39 med=-0.025  diff -0.026
   A-B ref-mag 13-14: sat N=0 med=+nan | unsat N=0 med=+nan
   A-B ref-mag 14-15: sat N=25 med=-0.098 | unsat N=0 med=+nan
   A-B ref-mag 15-16: sat N=55 med=-0.092 | unsat N=0 med=+nan
   A-B ref-mag 16-17: sat N=117 med=-0.073 | unsat N=0 med=+nan
   A-B ref-mag 17-18: sat N=158 med=-0.037 | unsat N=0 med=+nan
   A-B ref-mag 18-19: sat N=81 med=-0.014 | unsat N=37 med=-0.029
   A-B ref-mag 19-20: sat N=0 med=+nan | unsat N=6 med=-0.000
   A-B ref-mag 20-21: sat N=0 med=+nan | unsat N=0 med=+nan
   A-B ref-mag 21-22: sat N=0 med=+nan | unsat N=0 med=+nan

=== F150W-H vs H-Ks: base sample N=636  (sat 525, unsat 111)
A: color term a=+0.216 b=+0.300 (N fit 105, rms 0.217); saturation edge (p95 ground mag of sat stars) = 18.08
A residual (JWST - ground - colorterm) vs A JWST mag, 1-mag bins: bin | unsat N med MAD | sat N med MAD
  13-14:     0 +nan nan |     0 +nan nan
  14-15:     0 +nan nan |    33 -0.119 0.128
  15-16:     0 +nan nan |    69 -0.035 0.094
  16-17:     0 +nan nan |   126 +0.012 0.125
  17-18:     0 +nan nan |   185 +0.033 0.166
  18-19:    70 -0.052 0.191 |   112 +0.011 0.220
  19-20:    40 +0.095 0.300 |     0 +nan nan
  20-21:     1 +1.214 0.000 |     0 +nan nan
  21-22:     0 +nan nan |     0 +nan nan
  22-23:     0 +nan nan |     0 +nan nan
  step[A] unsat window ground in [18.08,19.08] N=96 med=-0.033; sat all N=525 med=+0.001 -> step +0.034 +- 0.028; sat within 1.5 mag of edge N=295 med=+0.013 step +0.045 +- 0.029; sat brighter N=230 med=-0.010
  step[A] unsat window ground in [18.08,19.58] N=98 med=-0.036; sat all N=525 med=+0.001 -> step +0.038 +- 0.027; sat within 1.5 mag of edge N=295 med=+0.013 step +0.049 +- 0.029; sat brighter N=230 med=-0.010
  step[A] unsat window ground in [18.08,20.08] N=98 med=-0.036; sat all N=525 med=+0.001 -> step +0.038 +- 0.027; sat within 1.5 mag of edge N=295 med=+0.013 step +0.049 +- 0.029; sat brighter N=230 med=-0.010
B: color term a=+0.217 b=+0.318 (N fit 107, rms 0.185); saturation edge (p95 ground mag of sat stars) = 18.08
B residual (JWST - ground - colorterm) vs B JWST mag, 1-mag bins: bin | unsat N med MAD | sat N med MAD
  13-14:     0 +nan nan |     0 +nan nan
  14-15:     0 +nan nan |    30 -0.043 0.106
  15-16:     0 +nan nan |    62 +0.054 0.072
  16-17:     0 +nan nan |   130 +0.065 0.112
  17-18:     0 +nan nan |   186 +0.057 0.141
  18-19:    73 -0.039 0.191 |   117 +0.010 0.192
  19-20:    37 +0.053 0.248 |     0 +nan nan
  20-21:     0 +nan nan |     0 +nan nan
  21-22:     1 +1.220 0.000 |     0 +nan nan
  22-23:     0 +nan nan |     0 +nan nan
  step[B] unsat window ground in [18.08,19.08] N=96 med=-0.031; sat all N=525 med=+0.049 -> step +0.080 +- 0.022; sat within 1.5 mag of edge N=295 med=+0.043 step +0.074 +- 0.024; sat brighter N=230 med=+0.059
  step[B] unsat window ground in [18.08,19.58] N=98 med=-0.038; sat all N=525 med=+0.049 -> step +0.087 +- 0.023; sat within 1.5 mag of edge N=295 med=+0.043 step +0.081 +- 0.025; sat brighter N=230 med=+0.059
  step[B] unsat window ground in [18.08,20.08] N=98 med=-0.038; sat all N=525 med=+0.049 -> step +0.087 +- 0.023; sat within 1.5 mag of edge N=295 med=+0.043 step +0.081 +- 0.025; sat brighter N=230 med=+0.059
A-B direct: sat N=525 med=-0.048  unsat(edge..edge+1.5) N=98 med=-0.020  diff -0.028
   A-B ref-mag 13-14: sat N=0 med=+nan | unsat N=0 med=+nan
   A-B ref-mag 14-15: sat N=30 med=-0.098 | unsat N=0 med=+nan
   A-B ref-mag 15-16: sat N=62 med=-0.087 | unsat N=0 med=+nan
   A-B ref-mag 16-17: sat N=130 med=-0.076 | unsat N=0 med=+nan
   A-B ref-mag 17-18: sat N=186 med=-0.036 | unsat N=0 med=+nan
   A-B ref-mag 18-19: sat N=117 med=-0.014 | unsat N=73 med=-0.024
   A-B ref-mag 19-20: sat N=0 med=+nan | unsat N=37 med=-0.013
   A-B ref-mag 20-21: sat N=0 med=+nan | unsat N=0 med=+nan
   A-B ref-mag 21-22: sat N=0 med=+nan | unsat N=1 med=-0.039

=== F200W-Ks vs H-Ks: base sample N=622  (sat 184, unsat 438)
A: color term a=-0.022 b=+0.304 (N fit 426, rms 0.138); saturation edge (p95 ground mag of sat stars) = 15.81
A residual (JWST - ground - colorterm) vs A JWST mag, 1-mag bins: bin | unsat N med MAD | sat N med MAD
  13-14:     0 +nan nan |     5 +0.005 0.030
  14-15:     0 +nan nan |    70 -0.030 0.063
  15-16:    21 +0.027 0.098 |   102 -0.005 0.078
  16-17:   195 +0.025 0.127 |     4 +0.037 0.102
  17-18:   181 -0.039 0.147 |     3 +0.037 0.077
  18-19:    40 +0.045 0.192 |     0 +nan nan
  19-20:     0 +nan nan |     0 +nan nan
  20-21:     0 +nan nan |     0 +nan nan
  21-22:     0 +nan nan |     0 +nan nan
  22-23:     1 +5.726 0.000 |     0 +nan nan
  step[A] unsat window ground in [15.81,16.81] N=180 med=+0.037; sat all N=184 med=-0.012 -> step -0.049 +- 0.013; sat within 1.5 mag of edge N=154 med=-0.013 step -0.050 +- 0.013; sat brighter N=30 med=+0.001
  step[A] unsat window ground in [15.81,17.31] N=283 med=+0.016; sat all N=184 med=-0.012 -> step -0.028 +- 0.012; sat within 1.5 mag of edge N=154 med=-0.013 step -0.029 +- 0.012; sat brighter N=30 med=+0.001
  step[A] unsat window ground in [15.81,17.81] N=361 med=+0.003; sat all N=184 med=-0.012 -> step -0.015 +- 0.011; sat within 1.5 mag of edge N=154 med=-0.013 step -0.016 +- 0.012; sat brighter N=30 med=+0.001
B: color term a=-0.005 b=+0.314 (N fit 425, rms 0.143); saturation edge (p95 ground mag of sat stars) = 15.81
B residual (JWST - ground - colorterm) vs B JWST mag, 1-mag bins: bin | unsat N med MAD | sat N med MAD
  13-14:     0 +nan nan |     3 +0.015 0.019
  14-15:     0 +nan nan |    64 +0.003 0.080
  15-16:    16 +0.027 0.113 |   110 +0.044 0.098
  16-17:   194 +0.034 0.132 |     4 +0.023 0.059
  17-18:   184 -0.037 0.138 |     3 +0.010 0.025
  18-19:    43 +0.022 0.199 |     0 +nan nan
  19-20:     0 +nan nan |     0 +nan nan
  20-21:     0 +nan nan |     0 +nan nan
  21-22:     0 +nan nan |     0 +nan nan
  22-23:     1 +5.780 0.000 |     0 +nan nan
  step[B] unsat window ground in [15.81,16.81] N=180 med=+0.042; sat all N=184 med=+0.023 -> step -0.019 +- 0.015; sat within 1.5 mag of edge N=154 med=+0.032 step -0.010 +- 0.015; sat brighter N=30 med=+0.000
  step[B] unsat window ground in [15.81,17.31] N=283 med=+0.015; sat all N=184 med=+0.023 -> step +0.007 +- 0.013; sat within 1.5 mag of edge N=154 med=+0.032 step +0.016 +- 0.014; sat brighter N=30 med=+0.000
  step[B] unsat window ground in [15.81,17.81] N=361 med=-0.000; sat all N=184 med=+0.023 -> step +0.023 +- 0.013; sat within 1.5 mag of edge N=154 med=+0.032 step +0.032 +- 0.013; sat brighter N=30 med=+0.000
A-B direct: sat N=184 med=-0.066  unsat(edge..edge+1.5) N=283 med=-0.025  diff -0.040
   A-B ref-mag 13-14: sat N=3 med=-0.007 | unsat N=0 med=+nan
   A-B ref-mag 14-15: sat N=64 med=-0.050 | unsat N=0 med=+nan
   A-B ref-mag 15-16: sat N=110 med=-0.078 | unsat N=16 med=-0.017
   A-B ref-mag 16-17: sat N=4 med=-0.024 | unsat N=194 med=-0.026
   A-B ref-mag 17-18: sat N=3 med=-0.018 | unsat N=184 med=-0.020
   A-B ref-mag 18-19: sat N=0 med=+nan | unsat N=43 med=-0.027
   A-B ref-mag 19-20: sat N=0 med=+nan | unsat N=0 med=+nan
   A-B ref-mag 20-21: sat N=0 med=+nan | unsat N=0 med=+nan
   A-B ref-mag 21-22: sat N=0 med=+nan | unsat N=0 med=+nan

```

Figure: `thirdref.png` (rows: F150W/J-H, F150W/H-Ks, F200W/H-Ks; columns A, B; blue unsaturated, red saturated, dark lines are running 1-mag medians stepping 0.5 mag with N>=5). Matched samples: `thirdref_matched*.fits`.

## Interpretation and caveats
- F200W: A's saturated stars sit 0.05 mag below the unsaturated trend for the narrowest window (the stars just fainter than the edge) and 0.015-0.03 below for wider windows; B's saturated stars sit 0.02 below to 0.02-0.03 above. B is closer to continuous across the edge (step consistent with 0 to +0.03), A shows a negative (brighter) step of -0.015 to -0.05 (about 1-4 sigma statistical, depending on window). The A-B difference of about -0.04 for saturated stars is reproduced, and the ground photometry places roughly half to all of it on the A side.
- F150W: not informative. Only 39-96 unsaturated stars overlap the ground-reference range (the F150W saturation edge, H ~ 18, is at the ground faint limit), the fitted colour term is poorly constrained (N fit 41-107, rms 0.22-0.23 mag; the unsaturated sample is dominated by faint, noisy stars and the residual rises sharply at H>19), and the sign of the step changes with the colour used (J-H gives A -0.054, B -0.005; H-Ks gives A +0.034, B +0.080, both +- 0.02-0.05). A-B at saturated magnitudes of 14-17 is the same as before, but this test cannot attribute it.
- Within the saturated sample (F150W, J-H colour) B's residual is nearly flat, -0.09 (14-15, N=25), -0.05, -0.03, -0.04, -0.09 (18-19), whereas A rises from -0.20 (14-15, N=28) and -0.10 (15-16) to -0.04 at 16-18. With H-Ks colour A goes -0.12, -0.04, +0.01, +0.03 (14-18) and B -0.04, +0.05, +0.07, +0.06. In both colour choices A is the more magnitude-dependent one at 14-16 (A minus B about -0.10 at 14-15), while the unsaturated overlap needed to fix the absolute level is small. Per-bin scatter (MAD) is 0.04-0.2 mag, so a bin median has an error of about 0.01-0.03.
- Systematics: ground seeing 0.4-0.5" versus JWST 0.06" (blending is only screened using the B catalog flux within 0.8" and an Ascenso 1.2" neighbour cut); Ascenso saturates at H ~ 9.6 and has no flags, so brighter-end ground photometry nonlinearity is not tested; J/H/Ks to F150W/F200W transformation is a linear colour term only (F200W-Ks has colour slope ~0.33, F150W-H ~0.2-0.4) and reddening varies across the field (residual vs colour not tested for curvature); uncertainties are statistical only; the 0.03 mag drift of unsaturated residuals with magnitude is an unquantified systematic. The Zeidler 2015 HST catalog (WFC3/IR 0.13" pixels) would be a cleaner test and needs a separate request to the authors or reduction of the MAST WFC3/IR images.
