# NGROUPS=2 frames lost all cosmic-ray rejection under the #161 skip

## Finding

The 2026-10-09 retie-loop re-reduction regenerated the per-exposure
`*_destreak_o<obs>_crf.fits` of cloudc (2221/o002), sgrc (4147/o012),
arches (2045/o001) and gc2211 o050 with the current reduction code, which skips
Image3 `outlier_detection` on every pass (#161, #848). The #161 rationale is
that JumpStep already rejects cosmic rays per ramp. JumpStep needs >= 3 groups:

| field | READPATT | NGROUPS | S_JUMP | S_OUTLIR (regenerated crf) |
|---|---|---|---|---|
| cloudc F212N | BRIGHT2 | 2 | SKIPPED | absent |
| gc2211 o050 F200W/F277W | | 2 | SKIPPED | absent |
| sgrc F212N | SHALLOW2 | 6 | COMPLETE | absent |
| arches F212N | BRIGHT1 | 8 | COMPLETE | absent |
| gc2211 o028 / o046 | | 3 | COMPLETE | absent |

cloudc and o050 therefore reached cataloging with no cosmic-ray rejection at
all. Their previous crf (`*_align_o002_crf.fits` on disk) carry
`S_OUTLIR = COMPLETE`.

brick 2221 F182M/F187N/F212N/F405N/F410M/F466N are also NGROUPS=2. Their crf
on disk date from 2026-08-25 and carry `S_OUTLIR = COMPLETE`; a re-reduction
on the current code would strip their cosmic-ray rejection the same way.

## Measurements (F212N, before = pre-rerun products, after = rerun m7)

`results/spikes.txt` -- pixels in a fixed set of 400 data-mosaic stamps whose
excess over the 3x3 median exceeds 8 robust sigma and holds > 50% of the 3x3
flux (1-2 pixel spikes):

| field | before | after |
|---|---|---|
| cloudc | 68 | 413 |
| sgrc | 0 | 1 |
| arches | 2 | 0 |

`results/*_before_after.json` -- the phase-loss audit's starlike residuals
(`n_left_total_before` is the pre-rerun count; the after counts come from
`phase_loss.py` on the rerun products):

| field | starlike residuals before | after |
|---|---|---|
| cloudc | 41,027 | 77,532 |
| sgrc | 106,802 | 21,587 |
| arches | 101,240 | 23,203 |

`results/*_ctrl_map.json` -- fraction of m7 vetted stars (20,000 sampled) that
leave PSF-matched S/N > 10 in the m7 residual: cloudc 12.6%, sgrc 2.0%. The
cloudc failures are spread uniformly over the field (`figures/cloudc_f212n_ctrl_map.png`).

## Re-run with this fix (2026-10-15/16)

cloudc and gc2211 o050 were re-reduced on 98151f91 + this commit (auto
policy). Every pass of both fields logged `RUNNING outlier_detection`. The
regenerated crf all carry `S_OUTLIR = COMPLETE`.

**cloudc F212N merged mosaic.** In the 400 fixed stamps it has 68 sharp
pixels: 68 pre-rerun, 413 retie. Its pixels match the pre-rerun
(2026-08-26) mosaic exactly (maximum |difference| = 0 over all 400 stamps),
so the re-reduction reproduces the August cosmic-ray rejection.

**o050 m7** (`scripts/o050_compare.py`, `results/o050_m7_compare.json`).
"Before" is the 2026-10-10 retie m7, with no cosmic-ray rejection. "After" is
the re-catalogued m7. The S/N figures are the PSF-matched residual S/N at every
vetted star in that product's own m7 residual. The sharp-pixel counts use the
same 2000 random sky positions in both products.

| o050 m7 | vetted stars | resid S/N > 10 | p99 S/N | sharp px |
|---|---|---|---|---|
| F200W before | 187,637 | 2.40% | 15.8 | 399 |
| F200W after | 186,332 | 2.00% | 14.0 | 341 |
| F277W before | 54,551 | 0.79% | 8.8 | 289 |
| F277W after | 54,956 | 0.69% | 8.1 | 301 |

The o050 change is smaller than the cloudc F212N regression. In the
wide bands each cosmic ray stands against a higher sky level than in a narrow
band; that is the likely reason, but it has not been measured here. The
largest o050 residuals are bright-star cores (`figures/o050_m7_gallery.png`),
which are the same before and after and are a separate PSF-model issue.

## Figures

* `figures/cloudc_f212n_bad_50_150.png`, `figures/cloudc_f212n_bad_500_5000.png`
  -- data | m6 residual | m7 residual for randomly drawn m7 vetted stars that
  leave S/N > 10, in two flux ranges. The sources are 1-2 pixel spikes; m7
  catalogues them (orange x) and the PSF model leaves them in the residual.
* `figures/{cloudc,sgrc,arches}_f212n_before_after.png` -- residual S/N at 300
  pre-rerun starlike-residual positions and 100 random positions, before and
  after, with stamp galleries.
* `figures/{cloudc,sgrc}_f212n_ctrl_map.png` -- position maps of the S/N > 10
  stars and that fraction against catalogue flux.
* `figures/o050_m7_ctrl.png` -- o050 F200W/F277W m7 before | after maps of
  vetted stars with residual S/N > 10, and the S/N histograms.
* `figures/o050_m7_gallery.png` -- m7 residual before | after at the 12
  worst-residual vetted stars of the retie product.

## Scripts

Run from a scratch layout holding the pre-rerun snapshot
(`prod_before/<field>/<band>/{snapshot.json,cutouts.npz}`) and `phase_loss.py`;
the `sys.path` lines at the top of each script point there.

* `compare.py <field> <FILT> <out>` -- before/after figure and JSON.
* `ctrl_map.py <field> <FILT> <out>` -- control-sample residual S/N map.
* `cutouts.py <field> <FILT> <fmin> <fmax>` -- the bad-star galleries.
* `spikes.py <fields...>` -- the sharp-pixel count.
* `o050_compare.py <out>` -- o050 m7 before/after (reads the retie m7 snapshot
  from `prod_before/gc2211_o050_retie/` and the live products).

## Fix

`resolve_skip_outlier_detection` (PipelineRerunNIRCAM-LONG.py): the default
policy becomes `auto` -- skip as #161 does when every member has
`S_JUMP = COMPLETE`, run the step when any member does not. `--outlier-detection
on|off` (sbatch `OUTLIER_DETECTION=`) forces either way.
