# Narrow-band colour-continuity test of the satstar offset (wd2, main2 arm)

Files: `nbstep.py` (code), `nbstep.png` (figure), `nbstep_results.json` (all numbers), `nbstep.log`. Tables below are `nbstep_tables.md`.

## Method
- Stars: `matched_Q_main2.fits` rows with `matched` and `sep_mas <= 150`, finite magnitudes in W and N in both catalogs (ours: `our_<band>`; dolphot: `ref_<band>`). The ecsv supplies dolphot ERRMAG and the neighbour list.
- Colour c = m_W - m_N in each catalog separately; abscissa x = dolphot m_N for both catalogs (same stars, same bins, no selection on the band under test).
- W-saturated = our `replaced_saturated_<W>` flag (also used for dolphot, which has none). W-unsaturated = not replaced and not `is_saturated`.
- Edges (in x): W edge = faintest 0.25 mag bin of x where the W replaced fraction >= 0.1; N edge likewise for N. Saturated sample = replaced-in-W stars with N edge + 0.3 <= x < W edge. Pairs with a range narrower than 0.5 mag are skipped.
- Quality: error < 0.1 mag in N (ours from flux_err/flux, dolphot ERRMAG) and in W for unsaturated stars; isolation = no neighbour within 0.5" brighter than own mag + 2 in either band (dolphot ecsv positions); all four magnitudes finite.
- Trend: polynomial (order 1 and 2, iterative 3-sigma clip) fitted to colours of W-unsaturated stars with W edge <= x < W edge + 1, 1.5, 2 mag; extrapolated across the edge. Step = median(c - trend) of saturated stars, overall and in 0.5 mag bins of x. Errors: 300 paired bootstrap resamples (fit and saturated sets resampled, identical indices for ours and dolphot). Negative step = W reads bright relative to the unsaturated trend.
- Band offsets: median(ours - dolphot) of each band for saturated stars versus the unsaturated 1.5 mag window.

## Pair selection
| pair | W edge | N edge | outcome |
|---|---|---|---|
| F200W-F212N | 15.75 | 15.75 | skipped. F212N reaches the replaced fraction >= 0.1 at the same m_N as F200W (replaced fraction in 15-16: W 0.55, N 0.44). No m_N interval exists where N is unsaturated and W saturated. F200W is untestable with this pair; the premise that the narrow bands saturate 2+ mag brighter does not hold for F212N. |
| F150W-F164N | 18.25 | 15.00 | used (N sat 15.3-18.25, 1609 stars) |
| F162M-F164N | 17.50 | 15.00 | used (added; same N band, medium W) |
| F182M-F187N | 16.25 | 15.25 | used, only 0.7 mag range (202 stars) |
| F300M-F323N | 16.50 | 13.75 | used |
| F250M-F323N | 16.25 | 13.75 | used, large colour baseline |
| F277W-F323N | 17.25 | 13.75 | used (added) |
| F335M-F323N | - | - | skipped; F335M replaced fraction < 0.1 everywhere (1% at 15-16) |
| F410M-F466N, F410M-F405N | 13.75 / 13.5 | 13.0 / 13.5 | skipped; range < 0.5 mag |

The F323N edge (13.75) rests on 127 replaced rows. F182M-F187N: F182M contains Pa-alpha and F187N is the line band; the unsaturated trend shows no visible line-driven jump, but the 0.7 mag range leaves only one bin.

## Results (linear trend, window 1.5 mag; full window/order grid in the tables)
| pair | step ours | step dolphot | ours - dolphot | direct band check: delta dW (ours - dol, sat minus unsat) | delta dN |
|---|---|---|---|---|---|
| F150W-F164N | -0.047 +- 0.022 | -0.023 +- 0.020 | -0.024 +- 0.011 | -0.023 | +0.002 |
| F162M-F164N | -0.013 +- 0.004 | +0.003 +- 0.003 | -0.016 +- 0.003 | -0.012 | +0.003 |
| F182M-F187N | -0.019 +- 0.006 | +0.001 +- 0.005 | -0.020 +- 0.005 | -0.028 | +0.001 |
| F300M-F323N | +0.013 +- 0.010 | -0.006 +- 0.010 | +0.019 +- 0.006 | +0.023 | +0.009 |
| F250M-F323N | -0.113 +- 0.026 | -0.122 +- 0.021 | +0.009 +- 0.010 | +0.027 | +0.007 |
| F277W-F323N | -0.036 +- 0.020 | -0.078 +- 0.017 | +0.042 +- 0.010 | +0.034 | +0.004 |

Linear steps for windows 1.0 / 2.0 agree with 1.5 within the errors for F150W, F162M, F182M, F300M (ours - dolphot: F150W -0.030 / -0.040; F182M -0.015 / -0.017; F300M +0.021 / +0.019; F277W +0.032 / +0.035). The quadratic fits extrapolate over 2-3 mag and diverge (F250M, F277W, F300M quadratic steps of -0.3 to -0.5 mag in both catalogs); they serve as an instability check only and carry no weight. For F162M and F182M the quadratic ours - dolphot difference is consistent with zero.

Binned (F150W-F164N, linear 1.5): ours - dolphot = -0.078, -0.031, -0.035, -0.014, -0.004, -0.012 from the brightest to the faintest bin; ours alone -0.138 in the brightest bin (15.3-15.8). F162M-F164N: -0.044, -0.031, -0.006, -0.009. The difference grows toward the N saturation edge, as expected for a flux offset that increases with brightness. F277W-F323N shows a flat +0.03 to +0.05.

Unmatched-allowed version for ours (all catalog rows): step F150W -0.049 (w1.5), F162M -0.011, F300M +0.014, F250M -0.102, F277W -0.024; these agree with the matched values within 0.01-0.02.

## Interpretation
- Ours - dolphot in the colour step reproduces the direct W-band offset (dW) for saturated versus unsaturated stars to within about 0.01 (F150W -0.024 vs -0.023; F162M -0.016 vs -0.012; F182M -0.020 vs -0.028; F300M +0.019 vs +0.023; F277W +0.042 vs +0.034; F250M +0.009 vs +0.027, the weakest agreement). The N-band offsets change by <= 0.009 between saturated-W and unsaturated-W stars (F323N +0.004 to +0.009; F164N, F187N <= 0.003), so the N bands carry no nonlinearity at these m_N comparable to the effect.
- The sign pattern matches the known dolphot comparison: our satstar W magnitudes read bright for F150W/F162M/F182M and faint for F250M/F277W/F300M relative to dolphot. The amplitudes here (0.02-0.04 averaged over the saturated range, 0.08 in the brightest F150W bin) are smaller than the quoted 0.05 (F150W) and 0.12 (F250M/F300M at 12-13 mag) because the test range ends at the N saturation edge and averages over the full replaced sample.
- Which catalog is continuous across the edge (linear, 1.5-2 mag windows):
  - F150W: dolphot continuous (+0.00 to -0.02 +- 0.02); ours steps -0.04 to -0.05 (2-3 sigma) and -0.14 in the brightest bin. Favours dolphot.
  - F162M: dolphot continuous (+0.003); ours -0.013 to -0.015 (3-5 sigma, small). Favours dolphot.
  - F182M: dolphot continuous (+0.001); ours -0.019 +- 0.006. Favours dolphot, with the Pa-alpha and 0.7 mag range caveats.
  - F300M: both within 2 sigma of zero (ours +0.013, dolphot -0.006 to -0.02); ours - dolphot +0.02. Not discriminating.
  - F250M: both steps -0.11 to -0.12 and identical, driven by the intrinsic colour trend (see below); the catalogs do not differ (+0.01). Not discriminating.
  - F277W: ours -0.02 to -0.04, dolphot -0.06 to -0.08; the linear trend is steeper for dolphot, so here ours is closer to continuous. Mildly favours ours; the quadratic trend is unstable and the F277W-F323N baseline is wide.
  - F200W: not testable (F212N saturates together with F200W). The F200W question needs an external reference (the ground-based test already available), or a reference band such as F187N/F164N with a larger colour baseline, which would carry the intrinsic colour caveats below.

## Caveats
- The colour-magnitude distributions are bimodal (F150W-F164N, F250M-F323N, F277W-F323N; see figure): a blue sequence and a redder branch (extinction spread, circumstellar excess, nebular line contamination of the N band). The median step depends on the mix of the two branches changing across the edge, which affects both catalogs equally but limits the accuracy of the absolute step.
- Intrinsic colour change along the sequence: F250M-F323N and F277W-F323N slope strongly with m_N (F250M contains the 3.3 micron PAH/H2O/ice region). The extrapolated trend is not constrained by data over 2-3 mag; the same-star comparison of ours and dolphot (difference column) cancels this effect, the individual steps do not. The F250M-F323N step is dominated by it.
- Line contributions: F212N and F323N carry H2 and PAH (nebular, background-subtraction sensitive); F187N Pa-alpha; F164N [Fe II]. Stars in bright nebulosity shift in c independent of saturation. The isolation and error cuts do not remove this.
- N saturation edges are taken from our replaced flag. Stars near the N edge + 0.3 may carry residual N saturation effects; the F323N edge is set by few stars. F323N dm between catalogs differs by 0.004-0.009 mag between saturated-W and unsaturated-W stars, which bounds the N-side effect.
- Binning uses dolphot m_N. A bright-end N nonlinearity in dolphot would shift x but not the same-star difference.
- F150W/F162M share F164N, so these two results are not independent; F250M, F277W, F300M share F323N.
- The fit window sits at W-unsaturated stars at the faint edge, where the unsaturated sample is also near the W 0.1 replaced fraction boundary; the 1.0-2.0 mag windows give consistent linear results, indicating limited sensitivity to this choice.
- Errors are bootstrap statistical only; the systematic from the trend model (linear versus quadratic) exceeds them for the wide-baseline pairs.

## Tables
| pair | W edge (m_N) | N edge (m_N) | sat range | N sat (matched, quality) | status |
|---|---|---|---|---|---|
| F200W-F212N | 15.75 | 15.75 | - | - | skipped: sat range [16.05,15.75] narrower than 0.5 mag (N edge 15.75 + 0.3 versus W edge 15.75) |
| F150W-F164N | 18.25 | 15.00 | 15.30-18.25 | 1609 | used |
| F182M-F187N | 16.25 | 15.25 | 15.55-16.25 | 202 | used |
| F300M-F323N | 16.50 | 13.75 | 14.05-16.50 | 908 | used |
| F250M-F323N | 16.25 | 13.75 | 14.05-16.25 | 778 | used |
| F335M-F323N | - | nan | - | - | skipped: no W-saturation edge found (W replaced fraction < 0.1 everywhere; N edge nan) |
| F162M-F164N | 17.50 | 15.00 | 15.30-17.50 | 1110 | used |
| F277W-F323N | 17.25 | 13.75 | 14.05-17.25 | 1155 | used |
| F410M-F466N | 13.75 | 13.00 | - | - | skipped: sat range [13.30,13.75] narrower than 0.5 mag (N edge 13.00 + 0.3 versus W edge 13.75) |
| F410M-F405N | 13.50 | 13.50 | - | - | skipped: sat range [13.80,13.50] narrower than 0.5 mag (N edge 13.50 + 0.3 versus W edge 13.50) |

Overall step (median of c - trend over W-saturated stars), mag; negative = W reads bright. Entries: value +- bootstrap sigma.
| pair | window | order | N fit | ours | dolphot | ours - dolphot |
|---|---|---|---|---|---|---|
| F150W-F164N | 1.0 | 1 | 998 | +0.002 +- 0.031 | +0.031 +- 0.029 | -0.030 +- 0.014 |
| F150W-F164N | 1.0 | 2 | 998 | -0.157 +- 0.215 | -0.197 +- 0.203 | +0.040 +- 0.093 |
| F150W-F164N | 1.5 | 1 | 1472 | -0.047 +- 0.022 | -0.023 +- 0.020 | -0.024 +- 0.011 |
| F150W-F164N | 1.5 | 2 | 1472 | +0.128 +- 0.109 | +0.178 +- 0.094 | -0.049 +- 0.048 |
| F150W-F164N | 2.0 | 1 | 1938 | -0.038 +- 0.016 | +0.001 +- 0.014 | -0.040 +- 0.008 |
| F150W-F164N | 2.0 | 2 | 1938 | -0.035 +- 0.077 | -0.006 +- 0.076 | -0.028 +- 0.033 |
| F182M-F187N | 1.0 | 1 | 767 | -0.018 +- 0.009 | -0.003 +- 0.007 | -0.015 +- 0.006 |
| F182M-F187N | 1.0 | 2 | 767 | -0.040 +- 0.026 | -0.050 +- 0.023 | +0.009 +- 0.018 |
| F182M-F187N | 1.5 | 1 | 1260 | -0.019 +- 0.006 | +0.001 +- 0.005 | -0.020 +- 0.005 |
| F182M-F187N | 1.5 | 2 | 1260 | -0.030 +- 0.015 | -0.025 +- 0.012 | -0.005 +- 0.010 |
| F182M-F187N | 2.0 | 1 | 1808 | -0.015 +- 0.006 | +0.002 +- 0.004 | -0.017 +- 0.005 |
| F182M-F187N | 2.0 | 2 | 1808 | -0.028 +- 0.011 | -0.008 +- 0.010 | -0.019 +- 0.007 |
| F300M-F323N | 1.0 | 1 | 1167 | +0.041 +- 0.015 | +0.020 +- 0.014 | +0.021 +- 0.008 |
| F300M-F323N | 1.0 | 2 | 1167 | -0.167 +- 0.086 | -0.202 +- 0.082 | +0.035 +- 0.049 |
| F300M-F323N | 1.5 | 1 | 1749 | +0.013 +- 0.010 | -0.006 +- 0.010 | +0.019 +- 0.006 |
| F300M-F323N | 1.5 | 2 | 1749 | +0.112 +- 0.043 | +0.088 +- 0.047 | +0.025 +- 0.026 |
| F300M-F323N | 2.0 | 1 | 2295 | -0.001 +- 0.008 | -0.020 +- 0.007 | +0.019 +- 0.006 |
| F300M-F323N | 2.0 | 2 | 2295 | +0.087 +- 0.031 | +0.066 +- 0.031 | +0.021 +- 0.015 |
| F250M-F323N | 1.0 | 1 | 1111 | -0.184 +- 0.038 | -0.188 +- 0.032 | +0.004 +- 0.014 |
| F250M-F323N | 1.0 | 2 | 1111 | -0.248 +- 0.213 | -0.247 +- 0.201 | -0.001 +- 0.078 |
| F250M-F323N | 1.5 | 1 | 1713 | -0.113 +- 0.026 | -0.122 +- 0.021 | +0.009 +- 0.010 |
| F250M-F323N | 1.5 | 2 | 1713 | -0.295 +- 0.097 | -0.337 +- 0.096 | +0.042 +- 0.035 |
| F250M-F323N | 2.0 | 1 | 2271 | -0.116 +- 0.020 | -0.123 +- 0.018 | +0.006 +- 0.009 |
| F250M-F323N | 2.0 | 2 | 2271 | -0.179 +- 0.062 | -0.176 +- 0.052 | -0.003 +- 0.021 |
| F162M-F164N | 1.0 | 1 | 984 | -0.014 +- 0.005 | +0.000 +- 0.005 | -0.014 +- 0.004 |
| F162M-F164N | 1.0 | 2 | 984 | -0.020 +- 0.029 | -0.011 +- 0.026 | -0.009 +- 0.021 |
| F162M-F164N | 1.5 | 1 | 1486 | -0.013 +- 0.004 | +0.003 +- 0.003 | -0.016 +- 0.003 |
| F162M-F164N | 1.5 | 2 | 1486 | -0.018 +- 0.014 | -0.018 +- 0.014 | -0.000 +- 0.010 |
| F162M-F164N | 2.0 | 1 | 1999 | -0.015 +- 0.003 | +0.003 +- 0.002 | -0.018 +- 0.002 |
| F162M-F164N | 2.0 | 2 | 1999 | -0.009 +- 0.010 | -0.006 +- 0.009 | -0.002 +- 0.006 |
| F277W-F323N | 1.0 | 1 | 1121 | -0.111 +- 0.031 | -0.143 +- 0.028 | +0.032 +- 0.014 |
| F277W-F323N | 1.0 | 2 | 1121 | -0.329 +- 0.231 | -0.447 +- 0.224 | +0.118 +- 0.102 |
| F277W-F323N | 1.5 | 1 | 1677 | -0.036 +- 0.020 | -0.078 +- 0.017 | +0.042 +- 0.010 |
| F277W-F323N | 1.5 | 2 | 1677 | -0.427 +- 0.104 | -0.536 +- 0.111 | +0.109 +- 0.044 |
| F277W-F323N | 2.0 | 1 | 2289 | -0.023 +- 0.014 | -0.058 +- 0.013 | +0.035 +- 0.009 |
| F277W-F323N | 2.0 | 2 | 2289 | -0.182 +- 0.061 | -0.239 +- 0.057 | +0.057 +- 0.031 |

Per-bin steps (window 1.5 mag, linear and quadratic): m_N bin, N sat, ours, dolphot, ours - dolphot.
| pair | m_N bin | N | ours lin | dol lin | diff lin | ours quad | dol quad | diff quad |
|---|---|---|---|---|---|---|---|---|
| F150W-F164N | 15.3-15.8 | 157 | -0.138+-0.036 | -0.060+-0.032 | -0.078+-0.020 | +0.328+-0.302 | +0.507+-0.275 | -0.179+-0.134 |
| F150W-F164N | 15.8-16.3 | 228 | -0.070+-0.030 | -0.039+-0.027 | -0.031+-0.016 | +0.265+-0.222 | +0.381+-0.205 | -0.117+-0.097 |
| F150W-F164N | 16.3-16.8 | 295 | -0.058+-0.026 | -0.023+-0.022 | -0.035+-0.015 | +0.185+-0.151 | +0.271+-0.141 | -0.086+-0.066 |
| F150W-F164N | 16.8-17.3 | 356 | -0.025+-0.020 | -0.011+-0.018 | -0.014+-0.011 | +0.123+-0.095 | +0.171+-0.087 | -0.048+-0.042 |
| F150W-F164N | 17.3-17.8 | 398 | -0.009+-0.015 | -0.005+-0.014 | -0.004+-0.008 | +0.069+-0.050 | +0.096+-0.048 | -0.028+-0.022 |
| F150W-F164N | 17.8-18.2 | 175 | -0.076+-0.030 | -0.064+-0.041 | -0.012+-0.022 | -0.030+-0.039 | -0.012+-0.046 | -0.019+-0.023 |
| F182M-F187N | 15.6-16.2 | 202 | -0.019+-0.006 | +0.001+-0.005 | -0.020+-0.005 | -0.030+-0.015 | -0.025+-0.012 | -0.005+-0.010 |
| F300M-F323N | 14.1-14.6 | 138 | +0.019+-0.018 | +0.028+-0.021 | -0.009+-0.015 | +0.242+-0.097 | +0.230+-0.102 | +0.012+-0.060 |
| F300M-F323N | 14.6-15.1 | 165 | +0.038+-0.014 | +0.003+-0.018 | +0.035+-0.014 | +0.191+-0.065 | +0.150+-0.071 | +0.041+-0.042 |
| F300M-F323N | 15.1-15.6 | 236 | +0.008+-0.012 | -0.017+-0.012 | +0.025+-0.008 | +0.103+-0.042 | +0.072+-0.046 | +0.031+-0.026 |
| F300M-F323N | 15.6-16.1 | 220 | +0.020+-0.012 | +0.005+-0.012 | +0.016+-0.009 | +0.073+-0.027 | +0.052+-0.026 | +0.021+-0.015 |
| F300M-F323N | 16.1-16.5 | 149 | -0.008+-0.010 | -0.017+-0.011 | +0.009+-0.008 | +0.019+-0.015 | +0.003+-0.016 | +0.015+-0.011 |
| F250M-F323N | 14.1-14.6 | 134 | -0.051+-0.061 | +0.015+-0.083 | -0.066+-0.047 | -0.552+-0.240 | -0.612+-0.236 | +0.060+-0.090 |
| F250M-F323N | 14.6-15.1 | 169 | -0.072+-0.048 | -0.089+-0.059 | +0.016+-0.026 | -0.413+-0.159 | -0.461+-0.148 | +0.047+-0.058 |
| F250M-F323N | 15.1-15.6 | 240 | -0.119+-0.033 | -0.127+-0.031 | +0.008+-0.015 | -0.311+-0.093 | -0.342+-0.086 | +0.031+-0.034 |
| F250M-F323N | 15.6-16.2 | 235 | -0.133+-0.018 | -0.127+-0.017 | -0.007+-0.008 | -0.202+-0.041 | -0.224+-0.041 | +0.022+-0.016 |
| F162M-F164N | 15.3-15.8 | 162 | -0.037+-0.007 | +0.007+-0.005 | -0.044+-0.006 | -0.047+-0.034 | -0.038+-0.033 | -0.009+-0.024 |
| F162M-F164N | 15.8-16.3 | 231 | -0.017+-0.007 | +0.013+-0.004 | -0.031+-0.006 | -0.024+-0.024 | -0.017+-0.021 | -0.007+-0.016 |
| F162M-F164N | 16.3-16.8 | 299 | -0.006+-0.005 | +0.000+-0.003 | -0.006+-0.004 | -0.010+-0.014 | -0.017+-0.012 | +0.007+-0.010 |
| F162M-F164N | 16.8-17.5 | 418 | -0.011+-0.004 | -0.002+-0.003 | -0.009+-0.003 | -0.012+-0.006 | -0.011+-0.006 | -0.001+-0.005 |
| F277W-F323N | 14.1-14.6 | 128 | -0.028+-0.049 | -0.077+-0.067 | +0.049+-0.042 | -1.134+-0.281 | -1.375+-0.295 | +0.240+-0.119 |
| F277W-F323N | 14.6-15.1 | 156 | -0.040+-0.030 | -0.095+-0.041 | +0.054+-0.026 | -0.848+-0.203 | -1.030+-0.215 | +0.182+-0.088 |
| F277W-F323N | 15.1-15.6 | 227 | -0.080+-0.028 | -0.112+-0.025 | +0.032+-0.018 | -0.643+-0.144 | -0.780+-0.152 | +0.137+-0.062 |
| F277W-F323N | 15.6-16.1 | 204 | -0.028+-0.030 | -0.084+-0.024 | +0.056+-0.017 | -0.407+-0.097 | -0.526+-0.105 | +0.119+-0.042 |
| F277W-F323N | 16.1-16.6 | 224 | -0.003+-0.023 | -0.039+-0.020 | +0.036+-0.013 | -0.223+-0.062 | -0.307+-0.066 | +0.084+-0.026 |
| F277W-F323N | 16.6-17.2 | 216 | -0.035+-0.015 | -0.076+-0.014 | +0.041+-0.011 | -0.121+-0.026 | -0.186+-0.028 | +0.065+-0.013 |

Band offsets (ours - dolphot, raw, no zero point): median for W-saturated stars in the sat range, median for W-unsat stars in the 1.5 mag fit window, and the difference.
| pair | band | sat | N | unsat | N | delta |
|---|---|---|---|---|---|---|
| F150W-F164N | F150W | -0.053 | 1609 | -0.030 | 1482 | -0.023 +- 0.002 |
| F150W-F164N | F164N | -0.028 | 1609 | -0.031 | 1482 | +0.002 +- 0.001 |
| F182M-F187N | F182M | -0.054 | 202 | -0.026 | 1268 | -0.028 +- 0.002 |
| F182M-F187N | F187N | -0.020 | 202 | -0.021 | 1268 | +0.001 +- 0.001 |
| F300M-F323N | F300M | +0.048 | 908 | +0.026 | 1760 | +0.023 +- 0.002 |
| F300M-F323N | F323N | +0.027 | 908 | +0.018 | 1760 | +0.009 +- 0.002 |
| F250M-F323N | F250M | +0.055 | 778 | +0.029 | 1730 | +0.027 +- 0.002 |
| F250M-F323N | F323N | +0.025 | 778 | +0.018 | 1730 | +0.007 +- 0.002 |
| F162M-F164N | F162M | -0.040 | 1110 | -0.028 | 1490 | -0.012 +- 0.002 |
| F162M-F164N | F164N | -0.028 | 1110 | -0.031 | 1490 | +0.003 +- 0.001 |
| F277W-F323N | F277W | +0.072 | 1155 | +0.038 | 1695 | +0.034 +- 0.002 |
| F277W-F323N | F323N | +0.026 | 1155 | +0.022 | 1695 | +0.004 +- 0.002 |

Ours, unmatched allowed (all catalog rows, m_N from our catalog), steps +- sigma [N fit stars]:
| pair | edges W/N | N sat | w1.0 lin | w1.5 lin | w2.0 lin | w1.5 quad |
|---|---|---|---|---|---|---|
| F150W-F164N | 18.25/15.00 | 1631 | -0.012+-0.036 [994] | -0.049+-0.022 [1464] | -0.043+-0.018 [1934] | +0.076+-0.132 [1464] |
| F300M-F323N | 16.50/13.75 | 824 | +0.037+-0.017 [1133] | +0.014+-0.011 [1682] | -0.000+-0.009 [2234] | +0.143+-0.052 [1682] |
| F250M-F323N | 16.25/13.75 | 681 | -0.178+-0.036 [1055] | -0.102+-0.031 [1635] | -0.090+-0.025 [2187] | -0.250+-0.097 [1635] |
| F162M-F164N | 17.50/15.00 | 1139 | -0.011+-0.006 [987] | -0.011+-0.003 [1494] | -0.014+-0.003 [2001] | -0.007+-0.017 [1494] |
| F277W-F323N | 17.25/13.75 | 1084 | -0.105+-0.027 [1093] | -0.024+-0.021 [1659] | -0.001+-0.016 [2345] | -0.421+-0.101 [1659] |