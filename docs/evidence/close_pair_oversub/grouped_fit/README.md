# Grouped fitting on the reference fields (grp0 vs grp1)

Test of candidate fix 1 (`--group`) from `../README.md` on the four
reference fields.

## Setup

Both variants run `crowdsource_catalogs_long` (m2-m7, production options) at
commit d40ccf49 (clean tree) on every reference-field cutout, seeds 0-10
(seed 0 is the clean run; seeds 1-10 each add 24 injected stars). grp1 adds
`--group --max-group-size=10`, which groups sources within
`manual_group_min_sep_fwhm` = 2.0 FWHM and fits each group jointly. Example
argv (`ref_dark_grp1_s0/reffield_provenance.json`):

    python -m jwst_gc_pipeline.photometry.crowdsource_catalogs_long --proposal_id=2221 --field=001
      --target=brick --filternames=F182M,F212N --modules=merged --each-exposure
      --each-suffix=destreak_o001_crf --cutout-region=266.5434000,-28.7125000,5
      --cutout-label=ref_dark_grp1_s0 --group --max-group-size=10 --parallel-workers=8

grp0 omits `--group --max-group-size=10`.
Runs live in `<data_root>/<target>/cutouts/ref_<field>_grp{0,1}_s<seed>`.

Median wall time per run, grp0 / grp1: bright_modest 3.3 / 3.4 min, dark
8.3 / 8.2, dense_bright 18.1 / 17.9, superdense 21.2 / 20.1.

`--group` inserts a `_group` token into the mosaic names
(`...-f212n-merged_m2_daophot_...` becomes
`...-f212n-merged_group_m2_daophot_...`); catalog names are unchanged.
`scripts/phase_loss_group_token.patch` is the change to the
`m7_companion_ratio` `phase_loss.py` that lets `ab_gallery_m7.py` find them.

## Reference-field evaluation

`results/eval_grp0.txt`, `results/eval_grp1_vs_grp0.txt`
(`reference_fields.evaluate`):

| field | variant | completeness per injected-S/N bin | flux bias (mag) | residual excess /as2 | over-subtracted /as2 | verdict |
|---|---|---|---|---|---|---|
| superdense | grp0 | 4/51 19/64 48/74 38/51 | -0.042 +- 0.025 | 0.55 | 1.39 | PASS |
| superdense | grp1 | 3/51 16/64 50/74 40/51 | +0.020 +- 0.019 | 0.48 | 1.32 | PASS |
| dense_bright | grp0 | 0/49 1/69 21/58 41/64 | +0.028 +- 0.019 | 1.52 | 0.97 | PASS |
| dense_bright | grp1 | 0/49 3/69 25/58 42/64 | +0.066 +- 0.016 | 1.45 | 0.69 | PASS |
| bright_modest | grp0 | 0/53 0/56 3/68 35/62 | -0.121 | 0.21 | 0.00 | PASS |
| bright_modest | grp1 | 0/53 0/56 3/68 36/62 | -0.115 | 0.21 | 0.00 | PASS |
| dark | grp0 | 1/62 30/55 49/61 54/62 | +0.038 +- 0.007 | 0.69 | 0.83 | PASS |
| dark | grp1 | 1/62 33/55 54/61 56/62 | +0.051 +- 0.007 | 0.69 | 0.69 | PASS |

Star-by-star completeness, grp1 against grp0 on the same injected stars
(gained / lost, all bins): superdense +7 / -7, dense_bright +8 / -1,
bright_modest +1 / -0, dark +10 / -0. Flux bias is the median
-2.5 log10(F_out / F_in) of recovered injected stars, so positive means
recovered fainter; grouping moves it 0.006-0.06 mag fainter. The dark field's
labelled-star recovery goes from 0.70 to 0.73.

## Over-subtraction by neighbour separation

`scripts/ref_nn_oversub.py` scores every m7 vetted source in the inner box
with the reference-field over-subtraction test (3x3 px minimum of the
residual matched-filter S/N below -7) and bins it by the separation to its
nearest catalogued neighbour. Seeds 0-10 pooled
(`results/nn_grp0_grp1.md`; cells = over-subtracted / sources):

| field band | variant | sources | over-subtracted | NN < 2 FWHM: over-subtracted / sources | 0.75-1.25 | 1.25-1.5 | 1.5-2 | 2-3 | 3-5 |
|---|---|---|---|---|---|---|---|---|---|
| dense_bright F187N | grp0 | 1415 | 158 | 35 / 196 | 4/4 | 4/42 | 27/150 | 44/499 | 79/617 |
| dense_bright F187N | grp1 | 1500 | 129 | 7 / 369 | 4/99 | 0/109 | 3/157 | 43/455 | 79/591 |
| dense_bright F212N | grp0 | 1124 | 103 | 58 / 262 | 3/26 | 15/65 | 40/171 | 23/324 | 22/398 |
| dense_bright F212N | grp1 | 1132 | 55 | 9 / 272 | 5/111 | 0/19 | 1/138 | 23/344 | 23/405 |
| dark F182M | grp0 | 1013 | 115 | 54 / 214 | 4/4 | 4/9 | 46/201 | 15/217 | 33/397 |
| dark F182M | grp1 | 1084 | 96 | 26 / 302 | 1/69 | 1/81 | 22/150 | 26/232 | 33/407 |
| dark F212N | grp0 | 328 | 40 | 7 / 10 | 2/2 | 3/4 | 2/4 | 18/33 | 11/90 |
| dark F212N | grp1 | 332 | 34 | 0 / 24 | 0/24 | 0/0 | 0/0 | 18/33 | 11/89 |
| superdense F212N | grp0 | 1450 | 218 | 98 / 416 | 24/30 | 35/101 | 39/285 | 56/557 | 64/434 |
| superdense F212N | grp1 | 1420 | 206 | 89 / 373 | 28/71 | 34/71 | 27/229 | 54/536 | 63/459 |
| superdense F405N | grp0 | 270 | 183 | 67 / 78 | 33/33 | 3/3 | 24/35 | 39/60 | 53/90 |
| superdense F405N | grp1 | 250 | 165 | 37 / 38 | 21/21 | 4/4 | 9/10 | 41/58 | 57/105 |

bright_modest has fewer than 7 sources with NN < 2 FWHM in either variant and
no change. The 2-3 and 3-5 FWHM bins do not change: they lie outside the
2 FWHM grouping radius.

Galleries (`figures/nn_grp0_grp1_gallery_<field>_<band>.png`): rows are data
| grp0 m7 residual | grp1 m7 residual at random grp0 sources that are
over-subtracted and have a neighbour within 2 FWHM; orange x = that variant's
m7 vetted sources. In dense_bright and dark most grp0 holes are gone in grp1.
In superdense F212N the remaining holes sit on bright-star cores, and in
superdense F405N they are saturated-star over-subtractions (S/N -500 to
-4000); grouping does not change either.

## Does grouping split single stars?

grp1 has more close pairs: sources with a neighbour at 0.75-1.25 FWHM go
from 4 to 99 (dense_bright F187N), 26 to 111 (dense_bright F212N), 4 to 69
(dark F182M) and 2 to 24 (dark F212N). Two tests of whether these are single
stars carried by two entries.

**Injected stars** (`scripts/split_test.py`, `results/split_grp0_grp1.txt`).
Injected stars are single by construction. For injected stars with no
clean-run source within 2.5 FWHM, the number carried by two or more m7
entries within 1.5 FWHM:

| field band | grp0 | grp1 |
|---|---|---|
| superdense F212N | 1 / 84 | 1 / 85 |
| dense_bright F187N | 1 / 110 | 2 / 106 |
| bright_modest F187N | 0 / 230 | 0 / 230 |
| dark F182M | 0 / 153 | 1 / 152 |

**Elongation of the data** (`scripts/pair_elong.py`,
`results/pair_elong_grp0_grp1.json`, `figures/pair_elong_grp0_grp1.png`).
For each grp1 pair closer than 1.25 FWHM, the variance of the data stamp
along / across the pair axis (Gaussian window, sigma = 1.5 PSF sigma, about
the flux-weighted pair centroid). The null is the same ratio along a random
axis at isolated grp1 sources (NN > 2.5 FWHM) at least as bright as the pairs.
Pairs with one member that grp0 does not have within 0.4 FWHM:

| field band | pairs (unique positions) | median ratio | fraction above the isolated 84th percentile | isolated median [16, 84] | median flux ratio | median sep (FWHM) |
|---|---|---|---|---|---|---|
| superdense F212N | 34 (19) | 2.28 | 0.91 | 1.00 [0.88, 1.12] | 0.27 | 1.08 |
| superdense F405N | 4 (3) | 1.85 | 1.00 | 0.94 [0.55, 1.42] | 0.02 | 1.00 |
| dense_bright F187N | 49 (13) | 1.41 | 0.82 | 1.00 [0.91, 1.12] | 0.26 | 1.15 |
| dense_bright F212N | 44 (8) | 1.35 | 0.98 | 1.00 [0.92, 1.06] | 0.27 | 1.02 |
| dark F182M | 36 (13) | 1.28 | 0.92 | 0.99 [0.95, 1.05] | 0.33 | 1.08 |
| dark F212N | 12 (2) | 1.60 | 1.00 | 0.99 [0.93, 1.13] | 0.33 | 1.11 |

The same field stars appear in every seed, so the unique-position count is
the independent sample size. The windowed ratios fall below the window-free
two-Gaussian prediction for these separations and flux ratios (median
1.8-2.2), as a window that down-weights the outer light should make them.
The added pairs are elongated along their axis in the data, as unequal
(flux ratio ~0.3) pairs at ~1 FWHM are; a single round star split in two
would sit at the isolated-star ratio.

## Stars one variant catalogues and the other leaves in its residual

`ab_gallery_m7.py` (from `docs/evidence/vetting_hysteresis/scripts`, with
the `_group` token patch) in both directions, seeds 0-10 pooled
(`results/ab_counts_table.md`). "B-only" = an m7 vetted source of B with no
A source within 1 px; "left" = PSF-matched S/N >= 5 at that position in the
other variant's m7 residual.

| field band | grp1-only, left in grp0 residual | grp0-only, left in grp1 residual | of those, near a saturated star (grp1-only / grp0-only) |
|---|---|---|---|
| superdense F212N | 60 | 98 | 46 / 70 |
| superdense F405N | 7 | 13 | 3 / 6 |
| dense_bright F212N | 51 | 40 | 0 / 0 |
| dense_bright F187N | 110 | 25 | 0 / 0 |
| dark F182M | 112 | 29 | 0 / 0 |
| dark F212N | 12 | 4 | 0 / 0 |
| bright_modest F187N + F210M | 2 | 0 | 0 / 0 |

Galleries (seed 0; data | A residual | B residual; red circle = the star,
magenta diamond = brighter neighbour, yellow star = saturated-star fit):

* `figures/ab_grp0_grp1_dense.png`: stars grp1 catalogues and grp0 leaves.
  These are faint companions (flux ratio 0.14-0.46) of a brighter neighbour
  that grp0 leaves at S/N 12-96 and grp1 subtracts.
* `figures/ab_grp1_grp0_dense.png`: sources grp0 catalogues and grp1 does
  not. Rows 1-4 are faint stars (S/N 8-22) that grp1 leaves in its residual.
  The two F187N rows with grp0 S/N -54 and -45 are grp0 sources on blank
  data that grp0 over-subtracts and grp1 does not catalogue.
* `figures/ab_grp1_grp0_superdense.png`: sources grp0 catalogues and grp1
  leaves, near saturated stars in F212N (S/N 16-39 in grp1); in F405N the
  grp0 entries over-subtract (S/N -8, -45) where grp1 leaves S/N 5-10.

## Summary

* Grouping removes most close-pair over-subtraction in dense_bright and
  dark (NN < 2 FWHM over-subtracted 35 -> 7, 58 -> 9, 54 -> 26, 7 -> 0) and
  lowers the field over-subtraction metric 0.97 -> 0.69 (dense_bright) and
  0.83 -> 0.69 (dark). All four fields pass in both variants; run time is
  unchanged.
* The close pairs it adds are elongated in the data, and injected single
  stars split no more often than without grouping (2 or fewer out of
  84-230 per field).
* Costs: in superdense F212N grp1 leaves more S/N >= 5 residuals than grp0
  (98 against 60 pooled positions, 70 and 46 of them near saturated stars);
  the injected-star flux bias moves 0.006-0.06 mag fainter; the 2-5 FWHM
  bins and the saturated-star residuals in superdense are unchanged.
* Regime 1 of `../README.md` (duplicate seeds closer than 0.75 FWHM from the
  m7 seed union) is barely sampled here: the grp0 0-0.75 FWHM bin holds 0-7
  sources per field and band. Candidate fix 2 (seed deduplication) targets
  it separately.

## Scripts

Run with `PYTHONPATH` at a d40ccf49 checkout; `eval_grp.sh` and `ab_grp.sh`
are the SLURM drivers (`<scratch>` = the scratch directory holding the
outputs).

* `ref_nn_oversub.py <A> <B> <out-prefix>` -- over-subtraction by NN bin and
  the `nn_*` galleries.
* `split_test.py <A> <B> <out-prefix>` -- injected-star split counts.
* `pair_elong.py <A> <B> <out-prefix>` -- pair elongation against isolated
  stars.
* `ab_table.py <ab_counts_A_B.json> <ab_counts_B_A.json>` -- the A/B table.
