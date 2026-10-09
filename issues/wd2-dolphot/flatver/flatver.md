# Flat-field / photom version test for the extra ~0.6*pred term

## Provenance of the dolphot catalogue
The ecsv meta (`catalogs/wd2_nircam_wf_mf_nf.ecsv`) only holds TOPCAT join keys. The input mosaics dolphot ran on
(`/orange/adamginsburg/jwst/wd2/wd2_F*_AB_i2d.fits`, written 2024-11-30 to 2024-12-16) carry the processing record:
CRDS_CTX `jwst_1298.pmap`, CAL_VER 1.16.0, SDP_VER 2024_2b, R_PHOTOM photom_0150/0152/0153/0157 (per band), R_FLAT/R_AREA per band.
Exception: F187N was rebuilt 2025-04-15 with `jwst_1322.pmap` (cal 1.17.1). Our crf frames use `jwst_1568.pmap` (cal 1.21.0.dev) for every
band except F150W (`jwst_1595.pmap`). No bracketing by date was needed: context 1298 is the old context for the whole catalogue.
Contexts compared: 1298 (dolphot), 1322, 1568 (ours), 1595 (ours, F150W). Full per-(context, band, detector) selections are in `refs.json` (script `refs.py`, header built from the crf primary header with DATE-OBS 2024-07-14).

## Reference selection (flat / photom / area, last 4 digits)
| band, detector | flat 1298 = 1322 | flat 1568 = 1595 | photom 1298 = 1322 | photom 1568 | photom 1595 | area (all four) |
|---|---|---|---|---|---|---|
| SW bands F150W, F200W, F212N, all 8 detectors | same | same (identical files) | 0150-0159 | 0166-0175 | 0166-0181 | same |
| F250M NRCALONG / NRCBLONG | 0630 / 0631 | 0750 / 0757 | 0157 / 0153 | 0168 / 0171 | same as 1568 | 0215 / 0281 |
| F300M NRCALONG / NRCBLONG | 0616 / 0618 | 0766 / 0752 | 0157 / 0153 | 0168 / 0171 | same | 0294 / 0249 |
| F410M NRCALONG / NRCBLONG | 0636 / 0613 | 0747 / 0765 | 0157 / 0153 | 0168 / 0171 | same | 0334 / 0287 |

Result: the SW flats and every AREA reference are identical between 1298 and ours. Only the LW flats (new files in 1568) and the PHOTOM tables differ.
The task hypothesis ("older FLAT carries a pattern correlated with the area map") has no SW test case: for F115W-F212N fl = 0 exactly, yet F182M/F200W/F212N show Theil-Sen slopes of 1.6-1.8.

## Method
`flatver_calc.py`: for every matched star and every exposure, sample 5x5 median-filtered flats (DO_NOT_USE pixels masked) of our reference and of the
context-1298 reference (downloaded to `crds_cache/` where not in the shared cache) at the star pixel, average in flux over exposures.
Since crf = rate * PHOTMJSR / flat, the star flux in ours relative to old is (PH_o/flat_o)/(PH_old/flat_old). Magnitude change (ours - old, positive = ours fainter):
`fl = 2.5 log10(flat_o/flat_old)` (larger flat_o makes the star fainter), `ph = -2.5 log10(PH_o/PH_old)` (larger PHOTMJSR_o makes it brighter).
PHOTMJSR_old is read from the 1298 photom table (FULL subarray, filter+pupil); PHOTMJSR_o from the crf header. `flatver_fit.py` regresses unsaturated ZP-window dm
(same selection as areatest.py) on pred (areapred_main2.npz), pred + (fl+ph), pred + fl + ph. `flatver_flstats.py` gives fl statistics. Outputs: `flatver_fit.txt`, `flatver_jwst_1298.npz`, `flatver_jwst_1298_dets.json`, `map_jwst_1298_*.npy`, `flatver.png`.

## Results
Per-band regressions (full table in `flatver_fit.txt`; Theil-Sen slope of dm on pred, OLS coefficient on pred without and with fl+ph, coefficient on fl+ph, robust residual std):

| band | N | TS slope | OLS pred alone | OLS pred with fl+ph | coef(fl+ph) | std resid (mag) pred only -> with fl+ph |
|---|---|---|---|---|---|---|
| F150W | 2585 | 0.95 | 0.83 +- 0.50 | 0.81 +- 0.50 | 0.24 | 0.0232 -> 0.0249 |
| F182M | 2836 | 1.57 | 1.74 +- 0.13 | 1.73 +- 0.13 | 0.10 | 0.0192 -> 0.0193 |
| F200W | 2952 | 1.80 | 1.77 +- 0.13 | 1.77 +- 0.13 | 0.10 | 0.0220 -> 0.0217 |
| F212N | 2982 | 1.62 | 1.55 +- 0.13 | 1.55 +- 0.13 | -0.01 | 0.0170 -> 0.0171 |
| F250M | 3049 | 1.64 | 1.66 +- 0.11 | 1.75 +- 0.11 | 1.27 | 0.0354 -> 0.0283 |
| F277W | 3266 | 1.74 | 1.53 +- 0.08 | 1.53 +- 0.08 | 0.01 | 0.0392 -> 0.0392 |
| F300M | 3076 | 1.66 | 1.56 +- 0.08 | 1.55 +- 0.08 | 0.37 | 0.0308 -> 0.0302 |
| F335M | 3037 | 1.43 | 1.46 +- 0.09 | 1.59 +- 0.08 | 1.79 | 0.0574 -> 0.0402 |
| F410M | 3076 | 1.52 | 1.39 +- 0.10 | 1.40 +- 0.10 | -0.91 | 0.0408 -> 0.0398 |

Flat term: fl is zero for SW; for LW its spatial std is 0.003-0.008 mag (1-99% range 0.016-0.038 mag), median per detector |fl| < 0.0015 mag, and corr(fl, pred) is between -0.06 and +0.05 (`flatver_flstats.py`). The coefficient on pred does not move toward 1.0 in any band; the changes (+-0.1, up to +0.13 for F335M) are within 1-2 sigma and have no consistent sign. Separate fits with fl and ph (LW) give coefficients on pred of 1.5-1.8 (F250M 1.76, F300M 1.58, F410M 1.56), also unchanged.

Photom term: ph differs by detector (median -0.015 to +0.043 mag; e.g. F200W NRCB1 +0.031, NRCA2 +0.039, NRCB2 -0.010; F335M NRCALONG -0.037). Per-detector medians of dm - pred stay within +-0.04 mag and do not track ph. Only the LW broad dichotomy gains: ph removes part of the A/B module offset in F250M (residual std 0.035 -> 0.028) and F335M (0.057 -> 0.040), with coefficients 1.3-1.8 (not 1.0). This is a detector-offset improvement and leaves the pred slope unchanged.

## Conclusion
The flat/photom version change does not explain the extra ~0.6*pred term. (1) The AREA references and all SW flats are identical between context 1298 and ours, while SW bands F182M, F200W, F212N still show slopes 1.6-1.8. (2) The LW flat ratio is small (rms 0.003-0.008 mag) and uncorrelated with the area map. (3) Including fl+ph leaves the coefficient on pred at 1.4-1.8 in all bands. The PHOTMJSR change contributes per-detector offsets of up to ~0.04 mag and accounts for a part of the module-to-module scatter in F250M and F335M. The pred-slope excess has another origin. Candidates not tested here: the dolphot catalogue was measured on resampled i2d mosaics (cal 1.16.0 resample), so how that resample step treats the pixel-area map and the dolphot aperture/PSF-based calibration of mosaic pixels may carry the extra area-like term; this requires a test on the i2d products.
