# m7 own-band seed: opt-in far-companion exemption of the companion cut

A star that m6 fitted, vetted and subtracted can be missing from the m7 seed.
m7 then never fits it, and it stays whole in the m7 residual i2d (the final
residual).  One cause is the own-band companion cut of `_build_m7_band_seed`:
an own-band m6 vetted source within `companion_fwhm` (2.5) PSF FWHM of a pool
source that is more than `1 / companion_max_ratio` (10) times brighter is left
out of the seed (#1015, #1107).  On gc-treasury 10678 F212N (tiles o045 and
o105) 11 of 22 bright leftover stars that a trace followed to their loss stage
(next section) are stars of this kind.  This PR adds an opt-in exemption for
cut sources that are bright and far from every dominating source.

| option | default | documented F212N value |
|---|---|---|
| `--manual-m7-seed-own-band-companion-keep-snr` (`manual_m7_seed_own_band_companion_keep_snr`) | 0 (off) | 50 |
| `--manual-m7-seed-own-band-companion-keep-fwhm` (`manual_m7_seed_own_band_companion_keep_fwhm`) | 1.75 | 1.75 |

With `keep_snr` 0 the builder returns exactly what it returned before (rows
and `NCOMPAN` match the production seed in all four tile × module cases, see
"Builder").  With `keep_snr` > 0 a source the cut would remove is kept when

1. its own-band m6 `flux / flux_err` is at least `keep_snr`, and
2. every dominating pool source is at least `keep_fwhm` PSF FWHM away and
   unsaturated.

A dominating pool source is a cross-band seed position (it inherits the
saturation flag of the own-band row within `match_mas`; with no such row it
counts as saturated) or another own-band candidate (its `is_saturated` or
`replaced_saturated` flag).  Seed-table keywords: `COMPKSNR`, `COMPKFWH`,
`NCOMPKEP` (rows exempted); `NCOMPAN` stays the number still cut.  The S/N
scale depends on the band (it is the ratio of two catalog columns), so the
documented value belongs to F212N and a band needs its own.

All counts below are gc-treasury 10678 F212N, tiles o045 and o105, both
modules (nrca, nrcb) unless a section says otherwise.  A *leftover* is a
whole-star residue of the control m7 residual i2d: residual peak
≥ 50 MJy/sr and ≥ 0.7 of the data peak (the control run used the DQ_SAT
mask dilation of 4 px and the edge band of 8 px; o045 also the spike mask).
The control has 887 leftovers ≥ 50 MJy/sr on o045 (185 ≥ 100, 32 ≥ 200,
3 ≥ 500) and 709 on o105 (168 ≥ 100, 35 ≥ 200, 1 ≥ 500).  The leftover lists
come from a scoring script that is not part of this repo.  A row *hits* a
leftover when it lies within 0.072″ (one F212N FWHM) of it.

## Which leftovers the cut causes

The 22 stars are the SATSTAR_NEAR leftovers ≥ 200 MJy/sr with no m7 catalog
row (or a row that does not subtract the star) that the relaxed bright seeds
(#1376) do not reach.  One-frame replays of the production m7 fit followed
each through seed, raw fit, dedup, saturated-star stages and final catalog
(the replay script is not part of this branch):

| cause | stars |
|---|---|
| own-band companion cut | 11 (o045 5, o105 6) |
| post-fit dedup keeps the non-positive row (#1366, fixed by `DAOPHOT_DEDUP_NONPOSITIVE`) | 2 |
| lost in the merge, a single-coverage fit collapses (candidate fix #1365) | 4 |
| close pair or displaced row | 3 |
| never catalogued | 2 |

`results/target_table.txt` is the table for the 11 companion-cut stars (from
`scripts/target_table.py`): the own-band m6 row under each star and the
sources that dominate it.

| star (tile, mosaic x y) | m6 flux | S/N | nearest dominating source: separation (FWHM), flux ratio | saturated | exempt at S/N 50, 1.75 FWHM |
|---|---|---|---|---|---|
| o045 (4979, 3391) | 3189 | 83 | 2.08, 0.070 | no | yes |
| o045 (6277, 2642) | 3747 | 99 | 2.50, 0.071 | no | yes |
| o045 (6355, 3168) | 2977 | 92 | 2.44, 0.062 | no | yes |
| o045 (7311, 2154) | 3506 | 100 | 2.14, 0.087 | no | yes |
| o045 (6092, 3774) | 2748 | 85 | 2.35, 0.068 | yes | no |
| o105 (5188, 2780) | 2784 | 90 | 2.44, 0.065 | no | yes |
| o105 (6137, 3868) | 2672 | 81 | 2.45, 0.058 | no | yes |
| o105 (6588, 1271) | 1996 | 71 | 2.01, 0.090 | no | yes |
| o105 (7097, 961) | 2567 | 85 | 1.78, 0.089 | no | yes |
| o105 (11328, 2324) | 2859 | 86 | 2.29, 0.064 | no | yes |
| o105 (9794, 833) | 2950 | 76 | 2.48, 0.031 | yes | no |

The 11 stars have an own-band S/N of 71–100 and lie 1.78–2.50 FWHM from a
source 11–32 times brighter (flux ratio 0.031–0.090).  Nine pass the rule.
The other two have a saturated dominating source, which the rule keeps cut
(see "Things to decide or know").  (7097, 961) sits 0.03 FWHM beyond the 1.75
FWHM limit.

## Choosing the rule

`scripts/companion_keep_table.py` counts, for every own-band row the cut
removes, the rows an exemption would return and how many hit leftovers
(`results/companion_keep_table.txt`; the pool is the own-band catalog alone, so
it counts 24,658 (o045) and 23,686 (o105) cut rows against the builder's 24,669
and 23,651, whose pool also holds the cross-band positions).  The table applies separation ≥ 1.75 FWHM and flux
ratio ≥ 0.03 to the brightest dominating source (the rule itself has no ratio
floor; without it one more row appears on o045, 57 against 56 at S/N 50, and
none on o105, with no extra hit).  Hits are at residual peak ≥ 100 MJy/sr:

| S/N floor | o045 unsaturated: rows (hits) | o105 unsaturated: rows (hits) | o045 saturated: rows (hits) | o105 saturated: rows (hits) |
|---|---|---|---|---|
| 20 | 1441 (39) | 1167 (29) | 583 (8) | 494 (7) |
| 30 | 523 (39) | 420 (29) | 547 (8) | 458 (7) |
| 40 | 172 (39) | 143 (29) | 436 (8) | 347 (7) |
| 50 | 56 (27) | 51 (23) | 279 (7) | 229 (7) |
| 60 | 23 (15) | 15 (9) | 180 (5) | 129 (3) |
| 80 | 8 (6) | 6 (5) | 68 (4) | 52 (0) |

- **S/N 50.**  Between S/N 40 and 50 the unsaturated class adds 116 rows (o045)
  and 92 rows (o105) with 12 and 6 hits (10% and 6.5%); between 30 and 40,
  351 and 277 rows with no hit.  Above S/N 50 the hit rate is 27 of 56 (48%)
  and 23 of 51 (45%).  The floor sits where the hit rate drops from about a half
  to about a tenth or less.
- **1.75 FWHM.**  Rows at 1.5–1.75 FWHM (S/N ≥ 50, unsaturated): 9 rows on each
  tile with 1 hit each at ≥ 100 MJy/sr (2 of 18, 11%).  Rows at 1.75–2.0 FWHM:
  5 rows (2 hits) on o045 and 12 rows (6 hits) on o105 (8 of 17, 47%).  The hit
  rate changes between 1.5 and 1.75; the sweep has a step of 0.25 FWHM, so the
  position of the change is known to that resolution.
- **Saturated neighbours.**  At S/N 50 this class has 279 and 229 rows (4.5–5 times
  the unsaturated class) with 7 and 7 hits at ≥ 100 MJy/sr (2.5% and 3.1%),
  against 27 of 56 and 23 of 51.  The rule leaves it cut.

## Builder

`scripts/build_exempt_seed.py` runs the real `_build_m7_band_seed` on the
m6 vetted catalogs of both modules and both tiles (`results/build_exempt_seed.log`,
`results/build_summary.json`).

- With the exemption off the builder reproduces the production seeds: row
  counts and `NCOMPAN` agree in all four tile × module cases.  The companion cut
  removes 24,669 rows on o045 and 23,651 rows on o105 (both modules).
- Rows the exemption returns, by S/N floor (30 / 40 / 50 / 60 / 80): o045
  540 / 177 / 53 / 21 / 7, o105 393 / 127 / 42 / 11 / 4.  At S/N 50: 25 + 28 rows
  on o045 (nrca + nrcb), 15 + 27 on o105; 53 and 41 distinct positions
  (`results/unique_cover.json`).  The band seeds hold 297,145–337,718 rows per
  module, so the exemption adds under 0.01% of a seed.
- Leftovers hit by those rows at S/N 50 (≥ 50 / ≥ 100 / ≥ 200 MJy/sr): o045
  33 of 887, 29 of 185, 6 of 32; o105 28 of 709, 20 of 168, 6 of 35; none of the
  4 leftovers ≥ 500.  26 of 32 and 29 of 35 leftovers ≥ 200 MJy/sr are not
  reached; of the 22 traced stars 9 are (`results/cover_targets.txt`).

## One-frame replays of the production m7 fit

Seven frames that contain the nine reached stars (`results/replay_frames.txt`;
columns: replay index, tile, module, exposure frame, frame x,y of the traced
stars in that frame).  Control: the production m7 fit command for the frame
with the production seed.  Treatment: the same command with the S/N 50
exemption rows added to the seed (the builder output).  A *new row* is a
treatment fit row more than 1 pixel from every control row.

- **Target stars** (`results/replay_rows.txt`, `results/replay_resid.txt`).  Each
  of the nine reached stars gets a fitted row within 0.6 px of its position in the
  frame, and the control has no row within 1 px of it.  The window is the
  7 × 7 pixels around the star in the frame residual image (MJy/sr).

  | star | m6 flux → fitted flux (distance from the star, px) | residual peak, control → treatment | residual minimum, control → treatment |
  |---|---|---|---|
  | o045 (6277, 2642) | 3747 → 3746 (0.5) | 399 → 42 | 8.8 → −24.4 |
  | o045 (6355, 3168) | 2977 → 2986 (0.5) | 279 → 27 | 5.6 → −49.4 |
  | o045 (4979, 3391) | 3189 → 3046 (0.5) | 329 → 34 | −1.9 → −9.6 |
  | o045 (7311, 2154) | 3506 → 3476 (0.6) | 368 → 94 | 0.6 → −67.0 |
  | o105 (7097, 961) | 2567 → 2553 (0.5) | 300 → 54 | −174.2 → −178.9 |
  | o105 (6588, 1271) | 1996 → 2005 (0.3) | 242 → 28 | 4.9 → 2.5 |
  | o105 (6137, 3868) | 2672 → 2667 (0.5) | 246 → 44 | 8.2 → −33.5 |
  | o105 (11328, 2324) | 2859 → 2859 (0.2) | 289 → 52 | 5.6 → −1.9 |
  | o105 (5188, 2780) | 2784 → 2791 (0.4) | 293 → 36 | 8.5 → 0.5 |

  The window of (7097, 961) also reaches the bright neighbour 4.5 px away; the
  control already has −174 there.  The eight traced stars in these frames that
  the rule does not reach (the two with a saturated dominating source, (6092, 3774)
  and (9794, 833), and six of other causes: (7485, 2672), (4952, 3836),
  (6471, 2248), (6760, 471), (2970, 3509), (7030, 3275)) get the same rows and
  the same residual as the control.
- **Added rows** (`results/replay_added_rows.txt`).  The exemption adds 48
  rows to the seeds of the seven frames (+11, +5, +6, +7, +8, +4, +7); the frame
  catalogs gain 47 rows and lose 1 (a control row of flux 127 is replaced by a
  row of flux 75 and S/N 5, 1.1 px away), net +46.  Of the 47 new rows, 45 have
  a frame S/N of 39–100 (median 57) and a fitted flux of 870–3771; the other two
  have S/N 5 and 14.  Statistics over the 45, in the 5 × 5 pixel window at each
  row of the frame residual image:
  - control residual peak 88–399 MJy/sr (median 157; 44 of 45 at ≥ 100, 17 at ≥ 200);
  - treatment residual peak 14–94 MJy/sr (median 28; 4 at ≥ 50, none at ≥ 100),
    at the median 0.18 of the control peak and at most 0.46;
  - treatment residual minimum −67 to +3 MJy/sr (median −9; 2 below −50, 6 below −30);
  - qfit median 0.22 (13 of 45 above 0.3); 6 rows have fit flag 1.

  Six of the 345,822 matched rows change their fitted flux by more than 10%.
  In frame o105 exposure 1 nrcb2 the 4 added seed rows give 2 new catalog rows;
  where the other two end up was not traced.
- **Frame level.**  No pixel changes by more than +50 MJy/sr; 21–94 pixels per
  frame change by less than −50 MJy/sr (the removed leftovers).  The number of
  pixels with a residual below −50 MJy/sr is 8967 → 8968, 8801 → 8801,
  10624 → 10625, 7037 → 7037, 10342 → 10342, 7257 → 7257, 6442 → 6442 in the
  seven frames (59,470 → 59,472 in total).  New craters (a pixel below
  −50 MJy/sr in the treatment and not in the control, with a depth above 0.3 of
  the treatment model at that pixel): 0 in all seven frames.

## Independent-visit test

`scripts/confirm_keep.py` (`results/confirm_keep_f182m.txt`) applies the rule
inside the Brick 2221/o001 F182M m6 vetted catalog and matches each group to
the m6 vetted catalog of 1182/o004 F200W (another visit of the field) within 60
mas.  Purity = (matched − chance) / (1 − chance), with `chance` the matched
fraction of the group's positions shifted by a few arcsec.  The dominance rule
is applied inside the F182M catalog only (the builder pool also holds
cross-band positions).

| group | n | matched | chance | purity | purity of control rows at the same flux |
|---|---|---|---|---|---|
| control (not dominated) | 372,882 | 0.955 | 0.139 | 0.947 | |
| cut, all | 18,283 | 0.578 | 0.138 | 0.511 | 0.940 |
| cut, blocked (dominating row closer than 1.75 FWHM or saturated) | 11,092 | 0.504 | 0.139 | 0.425 | 0.941 |
| cut, far and unsaturated, any S/N | 7,191 | 0.692 | 0.137 | 0.643 | 0.940 |
| exempt, S/N ≥ 20 | 747 | 0.846 | 0.146 | 0.820 | 0.962 |
| exempt, S/N ≥ 30 | 278 | 0.860 | 0.149 | 0.835 | 0.967 |
| exempt, S/N ≥ 40 | 104 | 0.933 | 0.142 | 0.922 | 0.969 |
| exempt, S/N ≥ 50 | 31 | 0.903 | 0.097 | 0.893 | 0.969 |
| exempt, S/N ≥ 60 | 14 | 0.857 | 0.089 | 0.843 | 0.971 |

The binomial standard error of the purity at S/N 50 is
sqrt(0.903 × 0.097 / 31) / (1 − 0.097) = 0.059, and at S/N 20 (n = 747) it is
sqrt(0.846 × 0.154 / 747) / (1 − 0.146) = 0.015.  The purity of the exempt rows
is 0.82–0.92 for S/N 20–50, which is 0.05–0.14 below the 0.96–0.97 of ordinary
rows at the same flux.  The deficit is 9 standard errors at S/N 20 (0.142) and
1.3 at S/N 50 (0.076), so the S/N 50 group alone cannot exclude that its purity
equals that of ordinary rows.  Blocked cut rows (0.43) and far unsaturated rows
at any S/N (0.64) match much less often.

Position angles around the dominating star (`|<exp(i m PA)>|`, m = 2 and 6;
isotropic positions give sqrt(π / 4n)): cut groups 0.065–0.107 (m = 2) against
0.007–0.010 isotropic, i.e. 8–12 times; exempt rows at S/N 20 / 30 / 40:
0.047 / 0.092 / 0.049 against 0.032 / 0.053 / 0.087, i.e. 0.6–1.7 times.  The
test supports the selection (S/N floor, far, unsaturated); it does not set the
F212N S/N value, because the F182M S/N scale is another one.

## Things to decide or know

- **Default off.**  `keep_snr` 0 leaves the cut unchanged.  50 is the F212N
  documented value; the S/N of a source depends on the band, so a band needs
  its own test (the F182M test above does not fix the F212N scale).  A per-band
  default is a small change.
- **Saturated neighbours stay cut.**  Two of the 11 traced companion-cut stars,
  (6092, 3774) and (9794, 833), sit 2.35 and 2.48 FWHM from a saturated
  dominating source (their own S/N is 85 and 76), and stay in the leftover list.
  Including the saturated class at S/N 50 would add 279 and 229 rows for 7 hits
  each at ≥ 100 MJy/sr.
- **A residual peak that disappears does not show the source was a star.**  A
  PSF-shaped fit also removes a PSF-shaped excess.  The independent-visit purity
  of the exempt rows is 0.82–0.92 (S/N 20–50) against 0.96–0.97 for ordinary
  rows at the same flux; the excess mismatch of 0.05–0.14 bounds the fraction of
  rows that are not stars from above, because the other visit also lacks some
  real stars.  The leftover rows have worse fits than ordinary rows (qfit median
  0.21 against 0.07 in the 22-star trace; 0.22 for the 45 replay rows).
- **Rows or a mask.**  The exemption restores a catalog entry that m6 already
  had.  A mask at the star (the ring tier-2 mask style of #1375) would remove
  the residual without adding a catalog row.
- **Reach.**  Six leftovers ≥ 200 MJy/sr per tile (of 32 and 35).  The remaining
  causes need other fixes (#1365, #1366, close-pair handling).

## Not measured

- No full-tile m7 fit, vetting and finalize ran with the exemption on.  The
  replays stop at the per-frame fit catalog: whether the 53 and 42 added rows
  survive the merge and the vetting, and the effect on the m7 residual i2d, are
  unmeasured.
- One field and band: the S/N and separation values come from gc-treasury
  10678 F212N (two tiles) and the F182M independent-visit test.
- The leftover scoring script is not part of this repo; the leftover lists
  are inputs of the scripts.

## Reproduce

All scripts write only under their output argument and read the catalogs
under `/orange/adamginsburg/jwst/gc-treasury` (and `/orange/adamginsburg/jwst/brick/catalogs`
for the independent-visit test).  Run them on a compute node.

```
python scripts/build_exempt_seed.py OUTDIR LEFT_045.json LEFT_105.json 50 30 40 60 80
python scripts/companion_keep_table.py LEFT_045.json LEFT_105.json
python scripts/cover_targets.py OUTDIR LEFT_045.json LEFT_105.json cover.json
python scripts/unique_cover.py OUTDIR LEFT_045.json LEFT_105.json unique.json
python scripts/target_table.py LEFT_045.json LEFT_105.json
python scripts/confirm_keep.py confirm.json 1.75
python scripts/replay_resid_compare.py CONTROL_PIPELINE_DIR TREATMENT_PIPELINE_DIR 'X,Y;X,Y' --json out.json
python scripts/replay_added_rows.py CONTROL_BAND_DIR TREATMENT_BAND_DIR --json out.json
```

Unit tests: `jwst_gc_pipeline/photometry/tests/test_m7_band_seed.py` (the
exemption, the saturated-donor rule, the missing-column error, the flag reader,
the option wiring); 36 pass.
