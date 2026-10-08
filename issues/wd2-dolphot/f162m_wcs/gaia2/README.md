# Gaia DR3 test of the NIRCam in-detector rotation, round 2: unsaturated stars only, wd2 and wd1

Context: keflavich/jwst-gc-pipeline#1135. The first Gaia test is in `../gaia/` (README there). This directory repeats it with the J fit restricted to unsaturated m6 rows, adds narrow bands whose Gaia stars are unsaturated (F187N, F164N), diagnoses the module-B floor, and repeats the test on Westerlund 1.

## Method changes relative to `../gaia/`
- `datascale_gaia2.py` (copy of `datascale_gaia.py`). Env knobs: `QFIT`, `SNRMIN`, `GMIN`, `FITSAT`, `MINSTAR`.
  - Bulk tie per (module, exposure): `measure_offset` on the union of unsaturated rows and satstar-catalog rows (qfit <= 0.6), as before.
  - J fit: only unsaturated m6 rows (`FITSAT=0`). `local_residual_map` is called on the unsaturated subset with the tie from the union. `FITSAT=1` reproduces the old pool (run `wd2_q01s20_sat`).
  - `GMIN` drops Gaia stars brighter than the limit from the J fit (tie unaffected).
  - Detectors with fewer than 8 Gaia stars are not fitted (`MINSTAR`). Pickles keep all per-star pairs.
  - Gaia cuts, propagation (pm to each frame's MJD-BEG, parallax neglected), 3-sigma clipping, per-frame c0 with shared J, and the cluster bootstrap over Gaia sources are as before (300 draws instead of 500; draws are shared across bands and detectors).
- `analyze_gaia2.py` (copy of `analyze_gaia.py`): same m, sigma and beta definitions; takes any band list; E (the F150W-group minus F212N-group vector) comes from the existing pair fits, as before.
  - wd2: `../datascale_other.txt`, `../datascale_f162m.txt`. wd1: `../datascale_wd1_*.txt` (these have the (band - F212N) pair-fit J for F150W, F115W, F164N, F187N, F200W on wd1).
  - E for F212N-group bands (F212N, F187N) is the F150W-minus-F212N pair-fit vector. For F150W-group bands it is that band's own pair-fit vector.
  - beta for F212N-group bands: 0 if the F212N-group references are right, -1 if the F150W-group references are right. For F150W-group bands: +1 if F212N-group right, 0 if F150W-group right. Detectors with fewer than 100 valid bootstrap draws are dropped.
- wd1 E: the pair-fit E vectors (direction and size) were used rather than only |rot| from `../refscale.txt`, because the direction of (tr-, curl+) in sky components depends on the roll angle (wd1 PA_V3 = 284.9 deg, wd2 140.8 deg) and |rot| alone gives only the length. Pair-fit m of E on wd1 (A2/A3): F150W 24.4/22.8, F164N 20.9/25.1, F115W 25.9/23.6, F200W 22.7/31.4. These agree with the `refscale.txt` rotations (F164N 21.1/24.9, F115W 23.7/24.1, F200W 19.1/30.9, F150W 22.8/24.2).
- wd1 Gaia: `query_gaia_wd1.py` -> `gaia_dr3_wd1.fits` (6999 rows, 5298 after the same cuts). Footprint taken from the F212N m6 catalogs (RA 251.677-251.843, Dec -45.912 to -45.804). wd1 catalogs: `{BAND}/{band}_nrc{ab}{1-4}_visit001_vgroup*_exp0000{1-4}_resbgsub_m6_daophot_basic.fits` (same naming as wd2; the default PROGPAT works). Run with `FIELD_ROOT=/orange/adamginsburg/jwst/wd1 GAIAFILE=gaia_dr3_wd1.fits`.
- Runs: `run_<tag>.log`, `results_<tag>.txt`, `summary_<tag>.pkl`, `gaia_fit_<tag>.pkl`. Tags: `wd2_q01s20` (qfit<=0.1, SNR>=20), `wd2_q03s10` (qfit<=0.3, SNR>=10; the reference configuration), `wd2_q03s05` (SNR>=5), `wd2_q03s10_g17` (G >= 17), `wd2_q01s20_sat` (satstar rows in J fit, reproduces `../gaia/`), `wd1_q03s10`, `wd1_q01s20`.

## 1. Unsaturated-only J on wd2

### Pair counts under different selections
Gaia pairs barely change with the relaxed cuts. N = unique Gaia stars / pairs used (after clipping). F212N, module A:

| selection | A1 | A2 | A3 | A4 |
|---|---|---|---|---|
| qfit<=0.1, SNR>=20 | 26/99 | 23/92 | 25/86 | 20/75 |
| qfit<=0.3, SNR>=10 | 26/98 | 23/86 | 26/91 | 20/75 |
| qfit<=0.3, SNR>=5 | 26/98 | 23/86 | 28/94 | 20/75 |
| qfit<=0.3, SNR>=10, G>=17 | 24/93 | 18/69 | 25/96 | 18/69 |

Pair counts per band and detector for all selections are in `selections_wd2.txt`. The F212N and F187N pools had no satstar rows in the fit already in `../gaia/` (95 pairs there against 98 here), so the earlier F212N numbers were mostly unsaturated stars. The number of Gaia stars is limited by the Gaia catalogue (a few percent of it is bright enough, faint enough and clean enough), not by the qfit and SNR cuts. Results are the same across all four selections within the bootstrap errors (beta values below).

### Bands that work and bands that do not
- F212N, F187N, F164N: 20-50 Gaia stars per detector, rms 1-6 mas, bootstrap-limited. F164N (F150W group) is the useful addition: it is a narrow band, its Gaia stars are unsaturated, and it belongs to the F150W reference group.
- F150W: unsaturated-only fit is not possible. Gaia stars match 1-4 unsaturated F150W rows per frame at random offsets (tested with `MINLOC=1`, qfit<=0.3, SNR>=5), because the Gaia stars are saturated in F150W (unsaturated F150W peak flux_fit is about 5.5e3 against 1.4e5 in F212N). F150W is only usable with satstar rows (`wd2_q01s20_sat`, identical to `../gaia/`).
- F182M, F162M, F200W: 3-24 Gaia stars per detector, rms 4-50 mas and m of 40-500 arcsec. Unusable; not interpreted (values in `results_wd2_q03s10.txt`).

### Reference run `wd2_q03s10` (m in arcsec; N/pairs, rms in mas, m with bootstrap 16-84 interval)

| det | F212N N/pairs | rms | F212N m | F187N N/pairs | rms | F187N m | F164N N/pairs | rms | F164N m | E (F150W - F212N pair fit) |
|---|---|---|---|---|---|---|---|---|---|---|
| A1 | 26/98 | 3.3 | 7.6 (5.1-16.3) | 23/89 | 2.8 | 11.3 (6.6-21.3) | 24/88 | 3.6 | 5.0 (4.6-15.7) | 6.0 |
| A2 | 23/86 | 1.0 | 6.7 (4.7-9.5) | 20/72 | 0.9 | 9.2 (7.3-11.2) | 24/86 | 1.3 | 28.3 (24.5-31.3) | 24.1 |
| A3 | 26/91 | 1.1 | 0.8 (1.4-4.5) | 26/96 | 1.3 | 5.0 (3.2-9.5) | 29/102 | 1.6 | 30.8 (28.4-33.7) | 25.4 |
| A4 | 20/75 | 2.6 | 3.0 (4.8-13.0) | 13/51 | 1.4 | 13.0 (9.6-18.3) | 20/64 | 2.3 | 2.3 (3.2-12.2) | 6.8 |
| B1 | 46/163 | 4.1 | 8.9 (5.0-17.7) | 44/144 | 4.1 | 8.4 (4.5-18.5) | 50/155 | 5.8 | 14.2 (8.9-26.0) | 2.5 |
| B2 | 28/101 | 2.2 | 12.2 (9.9-16.4) | 25/93 | 2.2 | 12.9 (10.9-17.2) | 27/88 | 2.9 | 11.3 (8.2-17.5) | 1.8 |
| B3 | 40/130 | 3.6 | 22.1 (18.2-28.1) | 40/130 | 4.4 | 16.7 (11.7-24.9) | 49/143 | 5.1 | 20.2 (16.1-28.7) | 2.0 |
| B4 | 24/95 | 1.8 | 18.8 (14.7-23.4) | 26/97 | 2.0 | 15.2 (12.8-20.2) | 26/95 | 2.2 | 17.4 (14.5-22.3) | 2.6 |

m is positive-definite and biased high at low S/N (the lower edge of the interval can sit above the nominal value). `results_wd2_q03s10.txt` has the (tr-, curl+) components and full J matrices.

With satstar rows in the fit (`wd2_q01s20_sat`, first-test pool) F150W gives A2/A3 m = 22 (16-41) / 20 (14-39) with rms 9-10 mas, beta(A2+A3) = +0.79 +- 0.46 (not changed from `../gaia/`).

### Pooled projection beta (module A; bootstrap sigma; same definitions as before)

| band | detectors | q01s20 | q03s10 | q03s05 | q03s10 G>=17 | wd2 sat-rows pool |
|---|---|---|---|---|---|---|
| F212N | A2+A3 | +0.06 +- 0.06 | +0.13 +- 0.07 | +0.14 +- 0.07 | +0.24 +- 0.10 | +0.13 +- 0.07 |
| F212N | A1-A4 | +0.06 +- 0.06 | +0.12 +- 0.07 | +0.13 +- 0.07 | +0.22 +- 0.10 | +0.12 +- 0.07 |
| F187N | A2+A3 | +0.28 +- 0.08 | +0.28 +- 0.08 | +0.28 +- 0.08 | +0.31 +- 0.10 | +0.29 +- 0.08 |
| F187N | A1-A4 | +0.25 +- 0.09 | +0.26 +- 0.08 | +0.26 +- 0.08 | +0.26 +- 0.10 | +0.26 +- 0.09 |
| F164N | A2+A3 | +1.29 +- 0.10 | +1.26 +- 0.10 | +1.26 +- 0.10 | +1.29 +- 0.11 | +1.35 +- 0.11 |
| F164N | A1-A4 | +1.26 +- 0.10 | +1.23 +- 0.09 | +1.23 +- 0.09 | +1.26 +- 0.11 | +1.31 +- 0.12 |
| F150W | A2+A3 | - | - | - | - | +0.79 +- 0.46 |

Single detectors, `q03s10`: F212N A2 +0.28 +- 0.10, A3 0.00 +- 0.09; F164N A2 +1.42 +- 0.17, A3 +1.20 +- 0.11. F182M: beta not meaningful (too few stars).

### Module-B floor with unsaturated stars
It persists. F212N on B1-B4: 9 / 12 / 22 / 19 arcsec (expected from E: 2-3), and F187N and F164N give the same values within errors (8-14 / 11-13 / 17-20 / 15-17). The rms is 2-5 mas, so the floor does not come from wing-fit centroids. The earlier F212N module-B values (12-19 on B2-B4) are reproduced with a purely unsaturated pool.

## 2. Diagnosis of the module-B floor (wd2)
Files: `fig_gaia2_quiver.png` (F212N and F164N on NRCB3, NRCB4, NRCA2), `diag_floor.py` / `diag_floor_wd2.txt`, `weighted_fit.py` / `weighted_wd2.txt`, `field_compare.py` / `field_compare.txt`.

- Quiver figure: on NRCB3 the residuals are dominated by a few stars with large Gaia pm errors (green and yellow arrows, 5-10 mas). NRCA2 residuals are 1-2 mas and small. NRCB4 has smaller arrows and a visible gradient in the fitted term (red).
- Star influence (leave-one-star-out): m on B3 stays within 16-25 (base 22), B2 10.7-13.9 (12.2), B4 8.7-21 (18.8), B1 1-14 (8.9). No single star produces the floor, although individual stars move m by several arcsec on B1 and B4, as expected for 25-50 stars.
- Subsets (`diag_floor_wd2.txt`): restricting to pm_err < 0.15 or 0.3 mas/yr, ruwe < 1.2, G > 18, G < 18, or stars more than about 170 pix from the edge changes B-detector m by factors of 0.5-1.5 with N falling to 5-15 stars; the floor does not vanish in any subset with N >= 12 on B2-B3.
- Error-weighted fit (per-star sigma^2 = 1 mas^2 + (pm_err * 8.55 yr)^2 + Gaia position error^2, 3-sigma clip on normalised residuals): F212N B1-B4 m = 11.5 / 13.6 / 18.7 / 10.7, F187N 10.4 / 13.9 / 18.5 / 10.0, F164N 13.8 / 17.3 / 20.2 / 11.3 (bootstrap intervals 2-4 arcsec wide). The floor remains on B2 and B3 and drops by about 8 arcsec on B4.
- Proper motion: all matched stars have |pm| of 4-13 mas/yr (median 6-8) and median pm_err 0.25-0.7 mas/yr, so propagation errors are 2-6 mas per star over 8.55 yr, comparable to the 2-5 mas rms. Not propagating pm changes m by up to 30 arcsec on A2, A3, B2 (e.g. F212N A3 0.8 -> 36), so pm propagation is required; the pm errors add noise but the systematic stays after weighting. Parallax: median |plx| of matched stars 0.2-0.5 mas (max 2-5 mas), which over a 2048 pix baseline changes J by about 1e-4 mas/pix (below 1 arcsec of m) and is not the cause.
- Field comparison (`field_compare.txt`): the (tr-, curl+) vector averaged over F212N, F187N, F164N on wd2 and wd1 is rotated by the roll difference (PA_V3(wd1) - PA_V3(wd2) = 144.0 deg; one-sided rotation of the sky components rotates (tr-, curl+) by the same angle). Rotating the wd2 vector by -144 deg reproduces the wd1 vector to 4.3 / 7.0 / 6.3 / 6.8 / 8.0 arcsec on A2, A3, B1, B2, B3 (A1 9.2, B4 12.1). Without rotation the differences are 12-31 arcsec. The sign (-dPA) was chosen from the data between two options. The agreement of six detectors with a fixed pattern in the detector frame makes a term common to all three bands and both fields, tied to the detector, the likeliest origin of the floor, for example a linear term shared by these distortion references. Gaia pm errors, pm propagation and field-specific effects would not rotate with the roll.
- Consequence: the floor affects all bands the same way (F212N, F187N, F164N agree on B2-B4 to a few arcsec). On module A it cancels in band differences but shifts absolute m and beta by a similar amount. The sizes on B (5-20 arcsec) set the scale for a common-mode term on A.

## 3. wd1 (unsaturated-only, qfit<=0.3, SNR>=10, `wd1_q03s10`)
wd1 has far more Gaia matches (80-125 stars per detector for narrow bands, 35-55 for broad bands). Bands run: F212N, F187N (F212N group), F164N, F115W, F200W (F150W group). F150W: the bulk tie offset (80.6 mas) exceeds the 67 mas acceptance limit (MATCH*1000/3), so every F150W frame was skipped, so there is no F150W result on wd1.

| det | F212N N/pairs | rms | F212N m | F187N m | F164N m | F115W m | F200W m | E(F150W) | E(F164N) | E(F115W) | E(F200W) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | 91/191 | 2.8 | 15.1 (12.3-18.1) | 16.7 (13.9-19.8) | 5.8 (3.9-8.6) | 16.4 (9.4-29.4) | 18.9 (13.2-28.1) | 4.2 | 4.0 | 3.1 | 11.0 |
| A2 | 79/168 | 1.8 | 4.4 (3.5-6.6) | 4.1 (3.1-6.4) | 24.9 (22.6-27.3) | 24.9 (21.5-29.2) | 20.8 (17.2-25.5) | 24.4 | 20.9 | 25.9 | 22.7 |
| A3 | 104/210 | 3.2 | 5.1 (2.6-8.1) | 4.5 (2.4-8.3) | 26.2 (23.5-29.2) | 20.5 (11.6-35.0) | 36.9 (25.2-53.7) | 22.8 | 25.1 | 23.6 | 31.4 |
| A4 | 85/182 | 2.3 | 3.3 (2.3-5.6) | 3.1 (2.3-5.5) | 3.6 (2.6-6.0) | 3.3 (3.5-11.5) | 9.3 (6.9-15.2) | 7.0 | 6.5 | 6.8 | 7.0 |
| B1 | 91/186 | 1.9 | 5.4 (4.4-7.2) | 5.8 (4.7-7.7) | 7.8 (6.4-9.8) | 16.2 (12.3-21.9) | 12.3 (7.5-17.7) | 5.4 | 4.2 | 4.3 | 3.7 |
| B2 | 122/262 | 2.7 | 16.5 (14.7-18.6) | 18.5 (16.5-20.9) | 20.4 (18.0-22.7) | 5.7 (5.6-17.0) | 25.5 (22.0-29.5) | 2.1 | 3.0 | 4.0 | 3.1 |
| B3 | 105/213 | 2.3 | 11.7 (9.9-13.6) | 11.5 (9.7-13.3) | 13.8 (11.9-15.7) | 7.8 (6.1-13.3) | 12.3 (10.6-18.3) | 4.1 | 3.1 | 3.7 | 2.9 |
| B4 | 122/264 | 3.1 | 5.3 (3.5-8.2) | 7.3 (5.1-10.0) | 7.9 (5.9-11.0) | 3.1 (6.1-19.9) | 3.2 (5.2-16.3) | 3.3 | 3.0 | 4.3 | 1.2 |

Entries are m in arcsec with the bootstrap 16-84 interval; E columns are the pair-fit |E| for each band against F212N on wd1 (F150W: the F212N-group projection direction; no F150W Gaia fit on wd1).

Pooled beta, wd1:

| band | A2+A3 | A1-A4 |
|---|---|---|
| F212N | +0.10 +- 0.06 | +0.06 +- 0.06 |
| F187N | +0.13 +- 0.07 | +0.08 +- 0.06 |
| F164N | +1.10 +- 0.08 | +1.02 +- 0.08 |
| F115W | +0.88 +- 0.15 | +0.85 +- 0.14 |
| F200W | +0.94 +- 0.17 | +0.96 +- 0.17 |

wd1 selection `q01s20` gives the same betas within 0.04 (`selections_wd1.txt`).

## Reading
- The unsaturated-only wd2 fit gives the same module-A numbers as the first test (A2/A3 F212N m = 6.7 and 0.8 arcsec, rms about 1 mas) with the relaxed cuts adding few pairs. F164N, a band in the F150W reference group whose Gaia stars are unsaturated, gives m = 28 and 31 arcsec on A2 and A3 with rms 1.3-1.6 mas. That value matches the expected F150W-group offset (20-25 arcsec) and is much better constrained (+-3 arcsec) than the F150W saturated-star fit (+-12 arcsec).
- wd1 reproduces the wd2 pattern with more stars: F212N and F187N sit near zero on A2/A3 (m = 4-5 arcsec), while F164N, F115W and F200W sit at the full offset (21-37 arcsec). Pooled beta is +0.10 +- 0.06 (F212N), +0.13 (F187N) against 0 for "F212N-group refs right" and -1 for the opposite, and +1.10 (F164N), +0.88 (F115W), +0.94 (F200W) against +1 and 0.
- Together, wd2 (F212N, F187N, F164N) and wd1 (F212N, F187N, F164N, F115W, F200W) give 8 band-field combinations that put the F182M/F187N/F212N references within about 0.1-0.3 of E (about 3-7 arcsec) of the Gaia frame on A2+A3, and the F115W/F150W/F164N/F200W references about 25 arcsec from it. The F212N-group references agree with Gaia on this evidence; the F150W-group references do not.
- Module-B floor: it is not produced by wing-fit centroids (F212N/F187N/F164N use unsaturated stars with 2-5 mas rms and show the same 9-22 arcsec on B2-B4 on wd2, and 12-17 on B2-B3 on wd1). It is the same in all bands, and its (tr-, curl+) vector tracks the detector frame between wd2 and wd1 (rotation test above). A shared, detector-fixed linear term of 5-20 arcsec on B (and, by implication, a similar common-mode amplitude on A) is the simplest description. It cancels in band-to-band differences and it limits absolute m values at the 5-10 arcsec level.
- Beta values above 1 for F164N (1.26 wd2, 1.10 wd1) and 0.28 for F187N on wd2 are consistent with a common-mode shift of 0.1-0.3 E (3-7 arcsec) on A2/A3; this is smaller than the B floor but of the same type.

## Caveats
- Bootstrap errors resample Gaia stars only; exposures are not independent and the common-mode (detector-fixed) term is not included. F212N, F187N and F164N share the same Gaia stars on a given field, so their betas are correlated.
- A common-mode term that happened to align with E at about 25 arcsec on A2 and A3 in both fields would mimic the opposite hypothesis. The measured common-mode amplitude on B (5-20 arcsec, directions unrelated to E) does not suggest that, but the Gaia test alone cannot exclude it.
- The sign of the roll rotation in the field comparison was chosen from the data (two options, eight detectors); treat the detector-fixed reading as a strong hint that needs a test with a third roll or the reference files.
- F150W on its own remains the weakest band (saturated Gaia stars; unsaturated fit not possible on wd2, tie fails on wd1). F182M and F162M were not usable on wd2; F162M is absent on wd1.
- Gaia epoch gap is 8.5 yr; the matched stars are mostly moving at 4-13 mas/yr, so pm errors (0.25-0.7 mas/yr) contribute 2-6 mas per star.

## Files
- Scripts: `datascale_gaia2.py`, `analyze_gaia2.py`, `compare_sel.py`, `diag_floor.py`, `weighted_fit.py`, `field_compare.py`, `fig_quiver_gaia2.py`, `fig_gaia2_rot.py`, `query_gaia.py`, `query_gaia_wd1.py`.
- Gaia tables: `gaia_dr3_wd2.fits` (copy of `../gaia/`), `gaia_dr3_wd1.fits`.
- Results: `results_<tag>.txt`, `selections_wd2.txt`, `selections_wd1.txt`, `diag_floor_wd2.txt`, `weighted_wd2.txt`, `field_compare.txt`; pickles `gaia_fit_<tag>.pkl`, `summary_<tag>.pkl`.
- Logs: `run_<tag>.log`.
- Figures: `fig_gaia2_rot.png` (rows: wd2 unsaturated, wd2 with satstar rows, wd1), `fig_gaia2_quiver.png`.
