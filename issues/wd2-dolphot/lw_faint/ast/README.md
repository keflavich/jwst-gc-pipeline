# wd2 injection-recovery test of the LW faint-end offset (ours - dolphot)

## Result in one paragraph
Stars injected into real wd2 LW frames and recovered with the pipeline's own
fitting path (artificial_stars.detect_and_fit: m1 + m2-style daofind, PSFPhotometry,
LocalBackground, fit_shape 5x5) come back with median (mag_out - mag_in) between
-0.01 and +0.05 mag at 19-21 mag in F277W, F410M and F405N, and +0.005 in the F200W control.
The table of ours - dolphot reaches +0.05 (F277W 21), +0.10 to +0.20 (F410M 20-21) and
+0.14 to +0.27 (F405N 19-20). The fitting path therefore shows no faint bias of that size
for stars that are recovered, in either background variant. Where the recovered sample is
biased, the sign is bright (negative dm), the opposite of the table, and it appears
only where the recovered fraction falls below about 40 percent (selection, see caveats).
On this evidence the +0.1 to +0.2 mag LW faint-end offset does not arise in the PSF fit
or its LocalBackground annulus, and the test places it on the dolphot side or in a part of the
pipeline outside this test (see "What this does not test").

## What was run
- Wrapper: `ast_wd2.py` (imports `jwst_gc_pipeline.photometry.artificial_stars` read-only from
  /blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline; reuses `detect_and_fit`, `draw_positions`,
  `inject_stars`, `estimate_effective_gain`, `mag_to_imflux`/`imflux_to_mag`). Analysis: `analyze.py`.
  SLURM template: `run_frame.sbatch`. Logs: `logs/`. Truth tables: `truth/<band>_<det>_exp1_<variant>_truth.fits`
  (columns: x, y, ra, dec, mag_in, mag_out, flux, qfit, matched, recovered, local_sb, d_nearest_base_pix, ...).
  Full bias tables with splits: `bias_tables.md`. Figure: `fig_ast_bias.png`.
- Frames (exposure 00001 only, crf, read-only, in-memory copies):
  F410M nrcalong/nrcblong (jw03523005001_16101), F405N (06101), F277W (10101), F200W nrca4/nrcb3 (12101).
  nrcblong exp1 contains the dense cluster core (pixel ~724,1938); nrcb3 is the SW detector over the core
  (core at ~1544,1800). The other module is nrcalong / nrca4.
- 1000 stars requested per frame (draw_positions min separation 12 px; 800-1000 placed), uniform in
  mag 16-23 (F405N 15-22). Seed = 20261008 + 100*band_index(F410M,F405N,F277W,F200W = 0..3) + 10*exp + (1 if module B):
  F410M 20261009/20261019 (A/B), F405N 20261109/20261119, F277W 20261209/20261219, F200W 20261309/20261319.
  The same injected stars (and Poisson noise) feed both variants.
- PSF: /orange/adamginsburg/jwst/wd2/psfs/nircam_{nrca5|nrcb5}_{band}_fovp101_samp2_npsf16.fits for LW
  (the grid get_psf_model loads for nrcalong/nrcblong), nircam_nrcb3/nrca4_f200w_fovp101_samp2_npsf16.fits for SW. Same grid for injection and fit.
- FWHM (pix, reduction/fwhm_table.ecsv): F410M 2.179, F405N 2.165, F277W 1.444, F200W 2.141.
- Vega zero points (Jy, SVO as in merge_catalogs): F410M 208.7505, F405N 206.9694, F277W 430.1399, F200W 757.6538.
  Flux to mag uses the frame WCS pixel area (as merge_catalogs does), 9.329e-14 sr for LW (PIXAR_SR agrees to 0.02 percent).
- Variants: (a) `raw`: injected crf as in the module. (b) `resbg`: injected crf minus the m6
  `..._resbgsub_m6_..._residual_smoothed_bg_i2d.fits` mosaic from
  Q_integ/tree_mainfcbg/<BAND>/pipeline (F200W/F277W/F405N: the main wd2/<BAND>/pipeline copy if absent there),
  cropped and `reproject_interp`-ed onto the frame exactly as cataloging.py does before the m7 fit
  (NaN -> 0, zero pixels preserved). Detection and fitting then run again on that image.
- Compute: SLURM jobs were submitted (astronomy-dept / astronomy-dept-b, 45142201-45142208) and had not started after
  30 minutes. All 8 frames were then run on the login node with `nice -19`, two at a time. The pending SLURM jobs
  are redundant (same seeds, same outputs) and were left in the queue (no cancellation was permitted).
- Statistics use all positional matches within 1.5 px (the 0.75 mag recovery gate is reported as a separate
  "recovered fraction" and is not applied to the bias).

## Bias, both modules pooled (median dm of all matches; 1.4826*MAD in parentheses)
Per 0.5-mag bin tables, quartile and neighbour splits: `bias_tables.md`. Selected bins:

| band | variant | 18-19 | 19-20 | 20-21 | 21-22 |
|---|---|---|---|---|---|
| F200W | raw | +0.007 | +0.005 | +0.006 (0.014) | +0.003 |
| F200W | resbg | +0.006 | +0.006 | +0.006 (0.013) | +0.004 |
| F277W | raw | +0.002 | +0.003 | +0.005 (0.046) | +0.000 |
| F277W | resbg | +0.002 | +0.003 | +0.002 (0.030) | -0.002 |
| F410M | raw | -0.000 | +0.011 | +0.031 (0.17) | -0.19 |
| F410M | resbg | -0.001 | +0.007 | -0.011 (0.15) | -0.14 |
| F405N | raw | -0.004 | -0.09 (0.36) | -0.29 (0.51) | n/a |
| F405N | resbg | -0.003 | -0.09 (0.23) | -0.47 (0.40) | n/a |

(bins here are the averages of the two 0.5-mag rows in `bias_tables.md`; the 19-21 per-frame medians in the
logs are F410M +0.008 (A) / +0.050 (B) raw, +0.000 / +0.012 resbg; F405N -0.090 / -0.143 raw, -0.078 / -0.212 resbg;
F277W +0.003 / +0.003; F200W +0.006 / +0.005.)
Recovered fraction at 19-21 mag (matched / injected): F410M raw 27-41 percent, resbg 59-78 percent (nrcblong core frame lowest);
F405N raw 23-35 percent, resbg 16-51 percent; F277W 84-96 percent; F200W 92-97 percent.

## Comparison with ours - dolphot (matched_Q_mainfcbg.fits, binned by mean mag)
| band | mag bin | ours - dolphot | injection dm (raw / resbg) |
|---|---|---|---|
| F277W | 21 | +0.047 | +0.004 / -0.011 |
| F277W | 22 | +0.077 | +0.00 / -0.01 |
| F410M | 19 | +0.058 | +0.003 / +0.005 |
| F410M | 20 | +0.103 | +0.02 to +0.04 / +0.01 |
| F410M | 21 | +0.203 | -0.08 / -0.12 (recovered 24-44 percent) |
| F405N | 19 | +0.140 | -0.05 / 0.00 |
| F405N | 20 | +0.268 | -0.14 to -0.20 / -0.19 to -0.47 |
| F200W | 20 | -0.020 | +0.006 / +0.006 |

The injection bias is flat near zero (+0.002 to +0.011) from the bright end through 20 mag in all four bands, while the
ours - dolphot curve rises steadily in the LW bands starting at 17-18 mag. F200W shows the same flat injection bias and a
curve of -0.02 to -0.03, which is a constant zero-point-level difference.

## Which side the result points to
For the part of the pipeline tested here (single-frame PSF fit with the production LocalBackground, with or without the
smoothed-residual background), faint LW photometry has no positive (faint) bias: at 19-21 mag the median injection
bias is within +/-0.05 mag (F410M raw frame B, the densest frame, is +0.050; all other frame/variant cases are within 0.02
except F405N, which is negative). That is smaller than the +0.06 to +0.27 mag offset in the table for F410M and F405N
at 19-21 mag. The offset in the table is therefore not produced by this fitting path. The remaining candidates are
dolphot (for example a bright-biased faint-end flux, sky/background treatment or a different detection-selected
sample) and pipeline steps after the per-frame fit (merging, cross-band fills, or the m7 initialisation).
This experiment does not measure dolphot bias directly.

## Caveats and what this does not test
- Injected stars are the fitting PSF grid itself plus Poisson noise. A mismatch between the real LW PSF and the grid
  (the grid is a model; real LW stars are under-sampled and have wing/halo structure) is not tested. A PSF-grid mismatch
  that loses flux for real stars would be invisible here. The 5x5 fit_shape with the grid gives a constant
  +0.005 mag at the bright end (flux truncation), identical in all bands.
- Selection: at faint magnitudes only stars whose noise boosts the flux are recovered (matched fraction falls to
  20-40 percent for F410M/F405N at 20-21 mag), so negative dm at mag_in >= 21 (F410M) and >= 19.5 (F405N) reflects
  that selection (F405N has a larger background, 29-60 MJy/sr, and a lower S/N per star). The test cannot say whether a
  fainter real star is biased faint after selection by a different detector (the production catalog is seeded from
  all passes and cross-band forced fills, which this module does not reproduce).
- F410M/F405N matched fractions differ strongly between variants (e.g. F410M 19-20: 41 vs 78 percent). The resbg image gives a
  higher detection rate because the daofind local-noise map changes after bg subtraction; the production m7 uses the
  catalog from earlier passes, not a fresh detection, so this recovery fraction is not a production completeness.
- Only exposure 1 per module was injected (8 frames, 1 seed each). The F410M nrcblong frame differs from nrcalong at the
  +0.05 vs +0.01 level in the raw variant (76 matches in 19-21 mag), so the statistical error of the 19-21 bins is a few hundredths of a mag
  for LW, about +/-0.01 for F277W and F200W.
- No saturated-star model, no m3-m6 iterations, no multi-frame vetting, which are all present in production.
- The matched-catalog curve is binned by mean of both mags, injection by true mag.
