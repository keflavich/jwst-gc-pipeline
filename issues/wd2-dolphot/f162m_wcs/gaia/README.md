# Gaia DR3 as external reference for the module-A in-detector rotation (wd2, program 3523 obs 5)

Context: keflavich/jwst-gc-pipeline#1135. Question: which distortion-reference group (F182M/F187N/F212N or F115W/F150W/F162M/F164N/F200W) carries the correct in-detector rotation on module A?

## Method
- `query_gaia.py`: Gaia DR3 box around the footprint (astroquery.gaia; archive reachable from this node). Raw table: `gaia_dr3_wd2.fits` (4000 rows; ra, dec, pm, parallax, errors, G, ruwe, astrometric_params_solved).
- `datascale_gaia.py`: same pair machinery as `../datascale_field.py` (`measure_offset` then `local_residual_map(return_pairs=True)`, fit r = c0 + J (p - p_centre), 3-sigma clipping), anchor replaced by Gaia.
  - Gaia cuts: ruwe < 1.4, astrometric_params_solved 31/95 (5/6-parameter), position error < 5 mas. Positions propagated linearly (pm, cos dec) from 2016.0 to each frame's MJD-BEG (about 2024-07); parallax neglected.
  - JWST catalog: m6 per-frame `*_resbgsub_m6_daophot_basic.fits`, exposures 00001-00004, qfit <= 0.1, flux SNR >= 20.
  - Deviation from the brief: the Gaia stars are bright and almost none are in the unsaturated m6 catalog. With unsaturated stars only, the bulk tie to Gaia failed in every module-A frame. The `resbgsub_m6_satstar_catalog.fits` entries (wing-fit positions, qfit <= 0.6, detector pixel from `x_0/y_0`) are therefore added. Their centroids are noisier than unsaturated stars (see rms below).
  - Bulk tie: `measure_offset` on the union of all detectors of one module and exposure (sky positions only), then `local_residual_map` per detector with that tie (the tie is about (-25..-33, +33..+38) mas, 40-52 mas total, the same on all detectors). `MATCH` = 0.20 arcsec because the tie is above 0.15/3 arcsec in some frames. An initial 45 mas cut about the per-frame median removes false matches before the linear fit.
  - Exposures are combined in one fit per detector (per-frame c0, shared J).
- Uncertainties: 500-draw cluster bootstrap over Gaia sources (a star is resampled in all four exposures and, with the same random draw, in all bands and detectors, so the J_F212N - J_F150W errors include the common-star correlation). Sigmas quoted are robust 16-84 half-widths. Exposures are not treated as independent.
- `analyze_gaia.py` makes `results_wd2.txt` (full tables) and `fig_gaia_rot.png`. `run_wd2.log` is the raw run log.
- Invariant: (tr-, curl+) = (J00 - J11, J10 + J01); m [arcsec] = |(tr-, curl+)| / (2*31) * 206265; 1 arcsec of m = 3.0e-4 mas/pix. Angle = atan2(curl+, tr-).
- wd1 was not run; wd2 gave enough F212N matches (20-67 Gaia stars per detector).

## Results (m in arcsec; lo-hi = bootstrap 16-84 interval; m is positive-definite so it is biased high at low S/N)

N = unique Gaia stars / pairs used (after clipping). rms = clipped per-star scatter about the linear fit.

| det | F212N N/pairs | rms mas | F212N m (lo-hi) | F150W N/pairs | rms mas | F150W m (lo-hi) | m(J_F212N - J_F150W) | E = m of (F150W - F212N) pair fit |
|---|---|---|---|---|---|---|---|---|
| A1 | 25/95 | 2.8 | 8.1 (4.6-16.7) | 10/31 | 10.3 | 44.5 (33.5-95.9) | 44.1 (33-94) | 6.0 |
| A2 | 23/92 | 1.0 | 6.4 (4.4-9.2) | 14/47 | 10.0 | 22.3 (16.0-39.7) | 17.0 (12-36) | 24.1 |
| A3 | 26/94 | 1.0 | 0.9 (1.1-4.2) | 17/50 | 9.1 | 19.8 (14.8-38.2) | 19.4 (15-38) | 25.4 |
| A4 | 20/75 | 2.6 | 3.0 (4.6-13.0) | 9/32 | 13.8 | 28.6 (22.4-94.5) | 25.7 (23-94) | 6.8 |
| B1 | 49/172 | 3.7 | 5.6 (4.8-14.0) | 63/221 | 13.1 | 41.7 (33.1-52.6) | 45.4 (36-59) | 2.5 |
| B2 | 27/102 | 2.2 | 12.1 (9.9-16.8) | 20/69 | 10.3 | 13.5 (11.4-36.2) | 13.1 (12-36) | 1.8 |
| B3 | 67/242 | 6.8 | 19.5 (14.1-28.0) | 89/309 | 14.5 | 27.3 (18.1-42.0) | 29.0 (19-47) | 2.0 |
| B4 | 24/95 | 1.8 | 18.8 (14.9-23.3) | 15/51 | 13.1 | 19.6 (16.5-49.1) | 38.3 (28-64) | 2.6 |

(tr-, curl+) components with robust sigma (mas/pix), module A:

| det | F212N tr- | F212N curl+ | F150W tr- | F150W curl+ | Delta = F150W - F212N | E (pair fit) |
|---|---|---|---|---|---|---|
| A1 | +0.0004 +-0.0015 | +0.0024 +-0.0019 | +0.0134 +-0.0123 | -0.0002 +-0.0098 | (+0.0130, -0.0026) +-(0.012, 0.010) | (+0.0015, -0.0010) |
| A2 | -0.0012 +-0.0006 | +0.0015 +-0.0007 | -0.0011 +-0.0044 | +0.0066 +-0.0033 | (+0.0001, +0.0051) +-(0.004, 0.004) | (-0.0046, +0.0056) |
| A3 | +0.0001 +-0.0005 | +0.0003 +-0.0007 | -0.0044 +-0.0040 | +0.0041 +-0.0049 | (-0.0044, +0.0038) +-(0.004, 0.005) | (-0.0047, +0.0060) |
| A4 | -0.0002 +-0.0022 | +0.0009 +-0.0017 | +0.0003 +-0.0103 | +0.0086 +-0.0107 | (+0.0004, +0.0077) +-(0.011, 0.011) | (+0.0015, -0.0014) |

Full J matrices (J00, J01, J10, J11 with sigma) and module B components are in `results_wd2.txt`.

Pooled projection onto E (the F150W-minus-F212N vector from the existing pair fits, per detector; inverse-variance weighted over detectors). beta(F150W) = +1 and beta(F212N) = 0 if the F212N-group reference is right; beta(F150W) = 0 and beta(F212N) = -1 if the F150W-group reference is right.

| band | detectors | beta +- sigma (boot 16-84) |
|---|---|---|
| F212N | A2+A3 | +0.13 +- 0.07 (+0.05 to +0.19) |
| F212N | A2 / A3 | +0.27 +- 0.10 / +0.02 +- 0.09 |
| F212N | A1-A4 | +0.12 +- 0.07 |
| F150W | A2+A3 | +0.79 +- 0.46 (+0.20 to +1.12) |
| F150W | A1-A4 | +0.82 +- 0.46 (+0.24 to +1.15) |

F162M: the Gaia fit is unusable (rms 10-35 mas per star, m = 80-340 arcsec on every detector; most of its Gaia stars are saturated with poor wing-fit centroids). Values are in `results_wd2.txt` and are not interpreted.

## Reading the result
- F212N (the F182M/F187N/F212N group) against Gaia: m = 6.4 (4-9) on A2 and 0.9 (1-4) on A3, with 1-3 mas per-star scatter. A 25 arcsec in-detector rotation error is excluded for F212N on A2 and A3 (beta = +0.13 +- 0.07 against -1 for the opposite hypothesis).
- F150W against Gaia: m = 22 (16-40) on A2 and 20 (15-38) on A3, per-star scatter 9-10 mas (saturated-star centroids). The values are compatible with the expected 24-25 arcsec and also lie about 1.7 sigma (beta 0.79 +- 0.46) from the +1 case; beta = 0 is about 1.7 sigma away. Single-detector F150W values do not separate 0 from 25 arcsec.
- J_F212N - J_F150W on A2/A3: 17 (12-36) and 19 (15-38) arcsec against the 24-25 expected; components agree with E within about 1 sigma on A2 and A3.
- Caveat on the error budget: on module B, where E is 2-3 arcsec, F150W gives m = 42 (B1), 27 (B3) and F212N gives 12-19 on B2-B4. These exceed the bootstrap errors and indicate a systematic floor of order 10-20 arcsec for F212N and 30-40 arcsec for F150W in this saturated-star setup. The bootstrap errors therefore understate the F150W uncertainty. The F212N A2/A3 values (1-6 arcsec) are below that floor, which is unexplained, and are small partly because few stars constrain each detector.
