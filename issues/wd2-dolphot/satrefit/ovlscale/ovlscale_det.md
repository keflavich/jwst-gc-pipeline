# Per-detector satstar vs dolphot (per satstar row)

Method: ovlscale_det.py. Every satstar S row (all frames) is clustered at 0.1" to star centres, matched within 0.1" to matched_Q_main2.fits dolphot positions (finite ref and our). dm_row = m_S_row + c_b - m_dolphot - ZP. ZP = analyze.py A.zp[band] (reused read-only). c_b = median over replaced stars of (our_merged - m_S_star_median); it carries the Vega offset and the +0.007/+0.003 merge constant. Tables with all 1-mag bins: ovlscale_det_tables.md. Figure: ovlscale_det.png. Results: ovlscale_det_results.json.

## Overall (final, per row; precap in parentheses)
| band | S rows / stars | all dets | equalised to nrcb1/nrcblong |
|---|---|---|---|
| F150W | 8522 / 2326 | -0.030 (-0.075) | -0.016 |
| F162M | 6581 / 1782 | -0.021 (-0.055) | -0.004 |
| F182M | 3022 / 824 | -0.041 (-0.083) | -0.020 |
| F200W | 2553 / 722 | -0.054 (-0.109) | -0.020 |
| F250M | 3998 / 1134 | +0.031 (+0.013) | +0.031 |
| F277W | 5063 / 1418 | +0.038 (+0.026) | +0.029 |
| F300M | 4355 / 1228 | +0.027 (+0.012) | +0.020 |

## Per-detector median dm_row (final; MAD in table file)
| det | F150W | F200W | F250M/F277W/F300M |
|---|---|---|---|
| nrca1 | -0.034 | -0.021 | |
| nrca2 | -0.002 | -0.017 | |
| nrca3 | +0.019 | -0.010 | |
| nrca4 | -0.003 | -0.019 | |
| nrcb1 | -0.016 | -0.020 | |
| nrcb2 | +0.007 | -0.042 | |
| nrcb3 | -0.074 | -0.093 | |
| nrcb4 | -0.001 | -0.073 | |
| nrcalong | | | +0.032 / +0.060 / +0.080 |
| nrcblong | | | +0.031 / +0.029 / +0.020 |

Rows (stars) on nrcb1 + nrcb3: F150W 5430 of 8522 (1479 of 2326); F200W 1819 of 2553 (515 of 722). The bright cluster core falls on nrcb1 and nrcb3 (about 65-70% of rows). Median dolphot mag per detector is similar (17.1-17.9 F150W, 15.1-15.4 F200W), so brightness differences do not drive the split.

## Findings
- nrcb3 is the outlier in SW: F150W -0.074, F162M -0.054, F182M -0.068, F200W -0.093. Other SW detectors lie within about -0.04..+0.03 (F200W: nrcb2 -0.042, nrcb4 -0.073). Detector-offset sd 0.025-0.028 mag.
- Offsets persist for bright stars. nrcb3 final, F150W: -0.101 (14-15), -0.104 (15-16), -0.085, -0.066, -0.054 (18-19). nrcb1: -0.036, -0.041, -0.026, -0.009, +0.004. Offset grows toward bright mags on both; nrcb3 minus nrcb1 about -0.05 to -0.06 at 14-17 mag. F200W: nrcb3 -0.089/-0.097 vs nrcb1 -0.009/-0.027 at 14-16.
- Explained fraction: equalising all SW detectors to nrcb1 moves the band median F150W -0.030 to -0.016, F162M -0.021 to -0.004, F182M -0.041 to -0.020, F200W -0.054 to -0.020. Detector offsets account for roughly 40-85% of the overall SW offset (F150W ~47%, F162M ~80%, F182M ~50%, F200W ~63%). A residual -0.02 remains in F150W/F182M/F200W (nrcb1 itself), which falls inside the daophot ZP uncertainty but is bright-end dependent (nrcb1 F150W -0.04 at 15-16).
- LW: nrcalong is brighter than nrcblong by +0.03 (F250M ~0), +0.03 (F277W), +0.06 (F300M), with larger MAD (0.08-0.09 vs 0.04). Equalising to nrcblong changes F277W +0.038 to +0.029, F300M +0.027 to +0.020, F250M none. Brightest bin (12-14 mag) nrcblong shows +0.12 to +0.16 in F250M/F277W/F300M (N 46-104; likely partial saturation or non-linearity at the extreme bright end). Final is fainter than precap by 0.01-0.02 in LW (cap does not bind in LW at the median: cap dmag +0.000).
- Control (daophot vs dolphot, unsaturated ZP window): SW detector medians within -0.025..+0.039: nrcb3 -0.021/-0.012/-0.012/-0.025 (F150W/F162M/F182M/F200W), nrca3 up to +0.039 (F182M), nrca1 +0.017..+0.026. Overall ~0. So the daophot ZP is detector-independent to about +/-0.02-0.04, with nrcb3 daophot also 0.01-0.025 faint. Subtracting the control from the satstar offset, nrcb3 satstar excess is about -0.05 to -0.07 (F150W -0.074 vs -0.021, F200W -0.093 vs -0.025). LW control: nrcalong -0.024/-0.017/-0.006, nrcblong +0.005/+0.012/+0.003; the nrcalong satstar excess (+0.06, +0.08) exceeds that.

## Cause checks
- Cap: raw < 0.999 precap in 0.45-0.95 of SW rows. Cap shift is smallest on nrcb3 (fraction 0.52 F150W, 0.45 F162M, 0.58 F182M, 0.66 F200W; mean cap dmag +0.004..+0.019) versus +0.04..+0.14 elsewhere (nrca3 up to +0.143). Precap is already faint on nrcb3 (F150W -0.091 vs nrcb1 -0.062; F200W -0.122 vs -0.076), but precap of other detectors is brighter than dolphot by 0.04-0.16 and the cap pulls them to ~0. nrcb3 therefore stays low because the cap rarely binds there and the raw fit is already faint by 0.07-0.09. Cap trims the bright side on the other detectors and hides the same bias at ~0.05 level in nrcb1/b2/b4.
- Wing calibration (F182M/F200W): wingcal ratio median 1.001-1.011, final-raw 0.001-0.011 mag; smallest on nrcb3 (1.001-1.002) and largest on nrcb2/nrcb4 (1.008-1.010). Cannot explain nrcb3 (it applies the least correction there). Not applicable in F150W/F162M/F250M/F277W/F300M (ratio 1.000, no pooled table).
- PSF grid (psfgrid.py): no detector dependence. Unit-PSF enclosed sums and central peaks are the same across detectors for both daophot (fovp101) and satstar (fovp512/1024) grids, F150W enclosed 0.980 (fovp101) vs 0.9955 (fovp512).
- sat_area median is 16-30 px on all detectors (F250M-F277W nrcblong 23-56); no nrcb3 anomaly.
- R(g0) per frame from the zeroframe log lines: not parsed.

## Caveats
- Dolphot is the yardstick only to the extent its own photometry is detector-independent; the dolphot-vs-daophot control (+/-0.02-0.04) bounds this.
- Rows are not independent (3-5 frames per star share the dolphot mag); stars N is the independent count. Medians use the star-median c_b calibration so the absolute zero rests on replaced stars.
- Clustering at 0.1" can merge blends in the core; MAD 0.03-0.09 is dominated by frame-to-frame scatter.
