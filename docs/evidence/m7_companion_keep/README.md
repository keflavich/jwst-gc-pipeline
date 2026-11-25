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

## Tile-level A/B on o105 F212N

The 42 exemption rows of this PR and the 39 rows of the relaxed bright window
(#1376) were fitted together in all 48 frames of o105 F212N (8 detectors, 6
exposures), merged and rendered into the m7 residual i2d.  The aim was to see
whether the rows survive the frame fit and the merge, and what they do to the
whole-star leftovers and the big craters of that image.  The scripts of this
section (the replay of the m7 fit, the combination of the frame catalogs, the
render and the scoring) are not part of this repo;
`results/tile_ab_o105_F212N_*.txt` hold their outputs.

**Arms.**  Each arm replays the production m7 fit command of every frame with
the production options and combines the 48 frame catalogs into one merged
catalog.  The replays used two states of the saturated-star PSF grid (see
[below](#two-psf-grid-states-among-the-replays)); the arms with the same state
are the ones to compare.

| arm | m7 seed | saturated-star PSF grid | `DAOPHOT_DEDUP_NONPOSITIVE` (#1366) | merge-pair rescue (#1365; ratio 3, floor 500) |
|---|---|---|---|---|
| `dnc` | production | rebuilt | off | off |
| `dnp3` | production | rebuilt | on | off |
| `dnp3p` | production | rebuilt | on | on |
| `dncg` | production | cached | on | off |
| `dncgp` | production | cached | on | on |
| `sd1` | production + 81 rows | cached | on | off |
| `sd1p` | production + 81 rows | cached | on | on |

The 81 added rows are the 42 exemption rows of this PR (own-band S/N 50) and 39
rows of the relaxed window (#1376; ERR S/N 30; a relaxed row within 30 mas of
an exemption row is dropped).  Per module the added rows are nrca 15 + 15 and
nrcb 27 + 24 (exemption + relaxed).  `sd1` against `dncg` is the effect of the
81 added rows alone, and `sd1p` against `dncgp` the same with the rescue on in
both arms.  `dnp3` against `dnc` is the effect of #1366 alone, `dncgp` against
`dncg` the effect of the rescue alone, and `dnp3` against `dncg` the effect of
the PSF grid state alone.  Two further arms, `sdloc` and `sdlocp`, were built
before `dncg` existed: they take the `dnp3` and `dnp3p` rows farther than 0.3″
(about 4 FWHM of F212N, 0.072″) from every added row and the `sd1` and `sd1p`
rows inside those discs.  They give the leftover counts of `sd1` and `sd1p` and
the crater lists of `dnp3`; their numbers are in the results files.

An arm enters the render as the difference between its merged catalog and that
of `dnc`: rows of the production vetted catalog within 0.03″ of a row that
disappeared are removed, and new rows with flux ≥ 0 are added.  No added row is
vetted again, so a row that the vetting would drop stays in the catalog, and
the counts below are the best case for the added rows.  Every arm is rendered
with the production options plus three render flags of #1360 (a draft PR that
this branch does not contain): `SINGLE_EXPOSURE_EVENT_MASK=1` sets DO_NOT_USE
on the pixels of a compact saturated core that one frame carries and the other
covering frames do not, `NEGATIVE_PIXEL_MASK_NSIGMA=50` sets DO_NOT_USE on
pixels with SCI below −50 × ERR, and `MERGE_ROW_RESID_SUPPORT_MIN=0.3`
re-renders a row at a lower flux when the PSF fit to the merged-catalog
residual of its frame has an amplitude below −0.7 × the flux of the row.  The
residual image is scored with the same mask settings in every arm: the
footprint mask (a disk of 1 × the radius of the saturated area plus 2 px around
each row of the production consolidated saturated-star catalog, and a disk of
at least 25 px radius around the rows with `flux_fit` ≥ 10⁶; #1361, #1364), the
DQ_SAT dilation of 4 px (#1372) and, in the second table, the edge band of 8 px
(#1373).  The masks were applied to the rendered residual images by emulation
scripts; the PRs that add them state where the emulation was compared with
their code.  None of these PRs is merged.  The counts of this section are those
of the flagged render and are not comparable to the counts of the production
render of o105 F212N.

**Counts.**  Whole-star leftovers (residual peak ≥ N MJy/sr, at least 50, and
more than 0.7 of the data peak at the star) and big craters (depth > 50 MJy/sr
and depth more than 0.3 of the model peak, counted at depth ≥ 50, 200 and 500
MJy/sr) of the m7 residual i2d of o105 F212N.

DQ_SAT dilation of 4 px:

| arm | leftovers ≥ 50 | ≥ 100 | ≥ 200 | ≥ 500 | ≥ 1000 | big craters ≥ 50 | ≥ 200 | ≥ 500 |
|---|---|---|---|---|---|---|---|---|
| `dnc` | 752 | 201 | 48 | 6 | 3 | 35 | 9 | 1 |
| `dnp3` | 741 | 194 | 45 | 5 | 2 | 35 | 9 | 1 |
| `dnp3p` | 729 | 185 | 41 | 5 | 2 | 35 | 9 | 1 |
| `dncg` | 742 | 194 | 45 | 5 | 2 | 35 | 9 | 1 |
| `dncgp` | 730 | 185 | 41 | 5 | 2 | 35 | 9 | 1 |
| `sd1` | 681 | 149 | 34 | 5 | 2 | 35 | 9 | 1 |
| `sd1p` | 669 | 140 | 30 | 5 | 2 | 35 | 9 | 1 |

DQ_SAT dilation of 4 px and edge band of 8 px:

| arm | leftovers ≥ 50 | ≥ 100 | ≥ 200 | ≥ 500 | ≥ 1000 | big craters ≥ 50 | ≥ 200 | ≥ 500 |
|---|---|---|---|---|---|---|---|---|
| `dnc` | 735 | 186 | 38 | 1 | 1 | 35 | 9 | 1 |
| `dnp3` | 724 | 179 | 35 | 0 | 0 | 35 | 9 | 1 |
| `dnp3p` | 712 | 170 | 31 | 0 | 0 | 35 | 9 | 1 |
| `dncg` | 725 | 179 | 35 | 0 | 0 | 35 | 9 | 1 |
| `dncgp` | 713 | 170 | 31 | 0 | 0 | 35 | 9 | 1 |
| `sd1` | 664 | 134 | 24 | 0 | 0 | 35 | 9 | 1 |
| `sd1p` | 652 | 125 | 20 | 0 | 0 | 35 | 9 | 1 |

**Leftovers.**  Against `dncg`, `sd1` clears 45 of the 194 whole-star leftovers
≥ 100 MJy/sr (a leftover is cleared when the treatment has none within 3 px of
its position).  No leftover ≥ 100 MJy/sr gets worse by more than a factor of
1.5, and none appears that `dncg` does not have.  The count at ≥ 200 MJy/sr
falls from 45 to 34 (35 to 24 with the edge band), and the ≥ 500 count stays at
5 (0 with the edge band).  By leftover class, the 45 leftovers ≥ 200 MJy/sr of
`dncg` are 29 SATSTAR_NEAR, 11 NOT_IN_M7_CATALOG, 4 VETTED_BUT_UNSUBTRACTED and
1 VETTED_OUT; the 34 of `sd1` are 21, 8, 4 and 1.  With the rescue on in both
arms, `sd1p` against `dncgp` clears 45 as well and adds none (≥ 200 MJy/sr: 41
to 30, and 31 to 20 with the edge band).  The rescue alone clears 9 leftovers
(`dncgp` against `dncg`), and the same 9 positions (within 3 px) with the added
rows (`sd1p` against `sd1`); these 9 have no added row nearby.  Leftovers of
212 to 358 MJy/sr at (434, 3684), (3245, 3878), (7030, 3275) and (9794, 833)
remain in every arm (the counts file lists the residual peak at 13 stars of
this tile for each arm).

**Where the clears occur.**  Of the 194 leftovers of `dncg`, 48 have an added
row within 0.1″, and `sd1` clears 45 of them.  It clears none of the other 146.
The attribution is by position (`results/tile_ab_o105_F212N_attribution.txt`).
The nearest added row of each of the 45 cleared leftovers lies within 0.035″
(median 0.011″): an exemption row of this PR for 20 leftovers and a relaxed row
of #1376 for 25.  45 of the 81 added rows lie under a cleared leftover.  The
three leftovers with an added row nearby that persist have peaks of 181, 178
and 150 MJy/sr, and the rows under them are relaxed rows (0.008″, 0.025″ and
0.025″ away).  No arm contains one kind of row alone, so the effect of the
exemption alone and of the relaxed window alone is not measured.

**Craters.**  The big craters of `sd1` are those of `dncg`: 35 in the
DQ_SAT-dilated render (9 ≥ 200 MJy/sr, 1 ≥ 500), with no crater new or gone
(positions matched within 3 px), depth changes below 0.1 MJy/sr, and 245
craters deeper than 50 MJy/sr at any depth/model ratio in both.  The same holds
with the edge band (35 in both; 244 deeper than 50 MJy/sr), with the footprint
mask and its ring zone only (46 in both; 271) and in the render without any
mask (134 in both, none new or gone; 6107 craters deeper than 50 MJy/sr in
`dncg` and 6108 in `sd1`).  With the rescue on in both arms, `sd1p` against
`dncgp` gives the same lists at the same four render levels (35, 35, 46 and 135
big craters in both, none new or gone).  The rescue adds one big crater to the
unmasked render, (4658, 4352) of depth 201 MJy/sr, to `dncgp` and `sd1p` alike.
The 81 added rows therefore add no big crater to this tile and remove none at
any of these four render levels (`results/tile_ab_o105_F212N_craters.txt`).
#1366 alone (`dnp3` against `dnc`) leaves the masked crater lists unchanged and
adds five big craters of depth 82 to 161 MJy/sr to the unmasked render (132 to
137).

**Survival of the added rows.**  Of the 42 exemption rows, 38 appear as new
rows of the merged catalog (no row of `dnp3` within 0.1″) and 4 coincide with a
row that `dnp3` already had.  Of the 39 relaxed rows, 33 are new, 5 coincide
with a `dnp3` row and 1 has no merged row within 0.1″ (a seed of flux 3.2 × 10⁵
whose nearest merged row lies 0.22″ away).  The median `flux_fit_avg` of the
new rows (in the unit of the catalog column) is 1411 for the exemption rows
against a median seed flux of 1392, and 1693 for the relaxed rows against 1341
(`results/tile_ab_o105_F212N_survival.txt`).  The frame catalogs of `dnp3` and
`dncg` differ by 4 lost rows and no added row within 0.5″ of an added row; the
reference for these counts is `dnp3`.

### Two PSF grid states among the replays

The replays of the arms `dnc`, `dnp3` and `dnp3p` and those of `sd1` and `sd1p`
used two versions of the replay script, which stage the saturated-star PSF grid
differently.  The first rebuilt it from MAST (the OPD file closest in time to
2026-09-14, O2026091501), and the second loaded the grid that the production
run had cached.  The `EPSFSHFT` keyword of the frame catalogs is 0.34–0.50 mpix
larger with the cached grid in each of the 8 detectors (45.208 against 44.706
mpix in nrca1; `results/tile_ab_o105_F212N_replay_state.txt`).  `dncg` and
`dncgp` repeat `dnp3` and `dnp3p` with the cached grid; their `EPSFSHFT` equals
that of `sd1` in all 8 detectors.

The grid state changes the frame catalogs throughout the tile.  Between `dncg`
and `dnp3`, 3265 frame rows are lost, 3178 are added, and 145,316 of the 2.24
million matched rows change their fitted flux by more than 1%.  Within 3″ of an
added row lie 3.7% of the lost rows, 3.1% of the added rows and 4.0% of the
rows with a changed flux, against 4.1% of all rows; within 0.5″ of an added row
lie 4 lost rows and no added row.  Between `dncg` and `sd1`, which share the
grid state, 11 rows are lost, 279 are added and 126 matched rows change their
flux by more than 1%, and every one of them lies within 0.5″ of one of the 81
added rows (277 of the 279 added rows within 0.15″).  The added rows therefore
change the per-frame fit only locally.  #1366 alone (`dnc` against `dnp3`, same
grid) removes 121 rows, adds 1739 and changes 376 fluxes, with 4.1% of the
removed rows and 4.1% of the added rows within 3″ of an added row.

In the render the grid state changes no leftover count: `dnp3` against `dncg`
clears none and adds none at ≥ 100 MJy/sr (741 and 742 leftovers ≥ 50 MJy/sr).
It changes the crater list by one in the masked renders: (2513, 951; depth 210
MJy/sr, model peak 690) is a big crater of `dnp3` and not of `dncg`, and (6771,
3357; depth 371, model peak 854) is a big crater of `dncg` and `sd1` and not of
`dnp3`.  The unmasked render of `dnp3` has three big craters that `dncg` lacks:
(2513, 951), (2571, 1972) and (9011, 2617), of depth 159 to 210 MJy/sr.
Against the `dnp3` control, `sd1` shows one new and one lost big crater, both
more than 11″ from every added row (11.7″ and 12.0″); against `dncg` it shows
none, so these two come from the grid state.

### What this test does not show

- The added rows are not vetted again.  The leftover and crater counts above
  are the best case for the added rows; a row that the vetting would drop stays
  in the catalog.
- The masks of the counts are emulations applied to the rendered residual
  image.  No frame, merge or vetting stage ran with a mask, and a mask hides a
  leftover or a crater in the image without changing the catalog or its
  photometry.
- Two tiles and one band: o105 F212N (this section) and o045 F212N (the next
  section).  The other tiles of 10678 (o118, o130, o138) and F480M were not run
  with the added rows, and the S/N values (50 for the exemption, 30 for the
  relaxed window) were tuned on F212N.
- No arm contains only one kind of added row, so the share of the exemption and
  of the relaxed window in the cleared leftovers rests on the position of the
  nearest added row.
- A leftover that disappears does not show that its source was a star: a
  PSF-shaped row also subtracts a PSF-shaped excess.  The crater test is the
  check against over-subtraction (no new crater at any mask level); it does not
  separate a faint star from a knot of extended emission.
- An added row does not always clear the leftover under it: 3 of the 48
  leftovers with an added row within 0.1″ persist.
- The added rows leave the 35 big craters of the masked render untouched, so
  they address leftovers only.  With the edge band `sd1p` still has 652, 125,
  20 and 0 leftovers at ≥ 50, 100, 200 and 500 MJy/sr.

## Tile-level A/B on o045 F212N

The 53 exemption rows of this PR and the 45 rows of the relaxed bright window
(#1376) were fitted together in all 48 frames of o045 F212N (8 detectors, 6
exposures), merged and rendered into the m7 residual i2d.  The test repeats
the o105 test above on a second tile of the same band, with a smaller arm set.
The aim is again to see whether the rows survive the frame fit and the merge,
and what they do to the whole-star leftovers and the big craters of that
image.  The scripts (the replay of the m7 fit, the combination of the frame
catalogs, the render and the scoring, adapted from those of the o105 test) are
not part of this repo; `results/tile_ab_o045_F212N_*.txt` hold their outputs.

**Arms.**  Each arm replays the production m7 fit command of every frame with
the production options and combines the 48 frame catalogs into one merged
catalog.  All four arms use the saturated-star PSF grid that the production
run cached and `DAOPHOT_DEDUP_NONPOSITIVE` (#1366).

| arm | m7 seed | saturated-star PSF grid | `DAOPHOT_DEDUP_NONPOSITIVE` (#1366) | merge-pair rescue (#1365; ratio 3, floor 500) |
|---|---|---|---|---|
| `dncg` | production | cached | on | off |
| `dncgp` | production | cached | on | on |
| `sd1` | production + 98 rows | cached | on | off |
| `sd1p` | production + 98 rows | cached | on | on |

The 98 added rows are the 53 exemption rows of this PR (own-band S/N 50) and
the 45 rows of the relaxed window (#1376; ERR S/N 30; a relaxed row within 30
mas of an exemption row is dropped, which removed none on this tile).  Per
module the added rows are nrca 25 + 16 and nrcb 28 + 29 (exemption + relaxed).
`sd1` against `dncg` is the effect of the 98 added rows alone, `sd1p` against
`dncgp` the same with the rescue on in both arms, and `dncgp` against `dncg`
the effect of the rescue alone.

An arm enters the render as the difference between its merged catalog and that
of `dncg`: rows of the production vetted catalog within 0.03″ of a row that
disappeared are removed, and new rows with flux ≥ 0 are added.  The render of
`dncg` is therefore the production vetted catalog.  #1366 is on in all four
arms, so it cancels in every difference and no render of this section contains
its effect.  No added row is vetted again, so a row that the vetting would
drop stays in the catalog, and the counts below are the best case for the
added rows.  The render flags of #1360, the footprint mask with its ring zone,
the DQ_SAT dilation of 4 px and, in the second table, the edge band of 8 px
are those of the o105 test above, applied to the rendered residual images by
the same emulation scripts.  The counts of this section are those of the
flagged render and are not comparable to the counts of the production render
of o045 F212N.

**Counts.**  Whole-star leftovers (residual peak ≥ N MJy/sr, at least 50, and
more than 0.7 of the data peak at the star) and big craters (depth > 50 MJy/sr
and depth more than 0.3 of the model peak, counted at depth ≥ 50, 200 and 500
MJy/sr) of the m7 residual i2d of o045 F212N.

DQ_SAT dilation of 4 px:

| arm | leftovers ≥ 50 | ≥ 100 | ≥ 200 | ≥ 500 | ≥ 1000 | big craters ≥ 50 | ≥ 200 | ≥ 500 |
|---|---|---|---|---|---|---|---|---|
| `dncg` | 970 | 239 | 65 | 15 | 4 | 17 | 0 | 0 |
| `dncgp` | 952 | 230 | 60 | 14 | 4 | 18 | 0 | 0 |
| `sd1` | 904 | 187 | 56 | 15 | 4 | 18 | 0 | 0 |
| `sd1p` | 886 | 178 | 51 | 14 | 4 | 19 | 0 | 0 |

DQ_SAT dilation of 4 px and edge band of 8 px:

| arm | leftovers ≥ 50 | ≥ 100 | ≥ 200 | ≥ 500 | ≥ 1000 | big craters ≥ 50 | ≥ 200 | ≥ 500 |
|---|---|---|---|---|---|---|---|---|
| `dncg` | 948 | 228 | 60 | 10 | 3 | 17 | 0 | 0 |
| `dncgp` | 931 | 220 | 56 | 10 | 3 | 18 | 0 | 0 |
| `sd1` | 882 | 176 | 51 | 10 | 3 | 18 | 0 | 0 |
| `sd1p` | 865 | 168 | 47 | 10 | 3 | 19 | 0 | 0 |

**Leftovers.**  Against `dncg`, `sd1` clears 51 of the 239 whole-star
leftovers ≥ 100 MJy/sr (a leftover is cleared when the treatment has none
within 3 px of its position).  No leftover ≥ 100 MJy/sr of `dncg` gets worse
by more than a factor of 1.5, and 1 leftover ≥ 100 MJy/sr appears that `dncg`
does not have: (5805, 2582; SATSTAR_NEAR, 269 MJy/sr).  The new leftover at
(5805, 2582) lies 3.2 px from the cleared leftover of `dncg` at (5806, 2585;
414 MJy/sr), so the two are probably one leftover that went down to 0.65 of
its peak and moved just beyond the 3 px matching radius.  Counted that way,
`sd1` clears 50 of the 239 and lowers the peak of 1 more (414 to 269 MJy/sr).
The count at ≥ 200 MJy/sr goes from 65 to 56 (60 to 51 with the edge band),
and the count at ≥ 500 MJy/sr stays at 15 (10 with the edge band).  By
leftover class, the 65 leftovers ≥ 200 MJy/sr of `dncg` are 41 SATSTAR_NEAR,
16 NOT_IN_M7_CATALOG, 6 VETTED_BUT_UNSUBTRACTED and 2 VETTED_OUT; the 56 of
`sd1` are 34, 14, 6 and 2.

With the rescue on in both arms, `sd1p` against `dncgp` clears 51 leftovers ≥
100 MJy/sr and has 1 leftover ≥ 100 MJy/sr that `dncgp` does not have, the
same one as in `sd1` against `dncg`.  The count at ≥ 200 MJy/sr goes from 60
to 51 (56 to 47 with the edge band).  The rescue alone clears 9 (`dncgp`
against `dncg`) and adds no new leftover; with the added rows (`sd1p` against
`sd1`) it clears 9.  All 9 of the positions that the rescue clears with the
added rows lie within 3 px of a position that it clears without them, and all
9 of the positions that it clears without them lie within 3 px of one that it
clears with them.  Of the 239 leftovers ≥ 100 MJy/sr of `dncg`, 56 have an
added row within 0.035″, and the rescue clears none of them; it clears 9 of
the 183 without an added row within 0.035″.  The counts file lists the
residual peak at the 9 traced stars of o045 (the o045 members of the 22 stars
traced above; `results/cover_targets.txt`) for each arm: 9 have a whole-star
leftover in `dncg`, 5 in `sd1`, 9 in `dncgp` and 5 in `sd1p`.  The 4 stars
that lose the leftover, (4979, 3391), (6277, 2642), (6355, 3168) and (7311,
2154), are the 4 with an exemption row within 0.1″ at S/N 50 (the stars that
the exemption reaches); the other 5 keep a residual peak of 212 to 1039 MJy/sr
in every arm: (1253, 131), (2970, 3509), (5070, 4424), (6092, 3774) and (6471,
2248).

**Where the clears occur.**  Of the 239 leftovers ≥ 100 MJy/sr of `dncg`, 57
have an added row within 0.1″, and `sd1` clears 51 of them.  It clears none of
the other 182.  The attribution is by position
(`results/tile_ab_o045_F212N_attribution.txt`).  The nearest added row of all
51 cleared leftovers lies within 0.031″ (median 0.014″): 27 under an exemption
row and 24 under a relaxed row.  Of the 98 added rows, 51 lie under a cleared
leftover.  The 5 leftovers with an added row within 0.035″ that persist have
peaks of 155, 125, 121, 108 and 107 MJy/sr, and the rows under them are
exemption and relaxed rows (0.015″, 0.008″, 0.017″, 0.004″ and 0.003″ away).
No arm contains one kind of row alone, so the effect of the exemption alone
and of the relaxed window alone is not measured.

**Craters.**  Big craters of the four arms at four render levels (a crater is
new or gone when no crater of the other tag lies within 3 px of its position;
`results/tile_ab_o045_F212N_craters.txt`):

| render | `dncg` | `dncgp` (against `dncg`) | `sd1` (against `dncg`) | `sd1p` (against `dncgp`) |
|---|---|---|---|---|
| DQ_SAT dilation of 4 px | 17 | 18 (new 2, gone 1) | 18 (new 1, gone 0) | 19 (new 1, gone 0) |
| DQ_SAT dilation and edge band of 8 px | 17 | 18 (new 2, gone 1) | 18 (new 1, gone 0) | 19 (new 1, gone 0) |
| ring zone only | 26 | 27 (new 2, gone 1) | 27 (new 1, gone 0) | 28 (new 1, gone 0) |
| no mask | 147 | 148 (new 2, gone 1) | 148 (new 1, gone 0) | 149 (new 1, gone 0) |

In the render without any mask, 7372 craters are deeper than 50 MJy/sr at any
depth/model ratio in `dncg`, 7372 in `dncgp`, 7376 in `sd1` and 7376 in
`sd1p`.  The number of big craters rises by 1 at each render level (from 17 to
18 with the DQ_SAT dilation of 4 px): with the rescue off (`sd1` against
`dncg`) and on (`sd1p` against `dncgp`), the added rows give a new crater at
(2201, 3065) (depth 60 MJy/sr, model peak 198).  That crater is close to both
criteria (depth 60 against more than 50 MJy/sr, depth over model peak 0.300
against more than 0.3); its nearest vetted-catalog row lies 0.040″ away and
its nearest added row, a relaxed row in nrca, 0.049″ away.  The rescue alone
(`dncgp` against `dncg`) changes the crater list in the same way at each of
the four render levels: a new crater at (4674, 4345) (depth 104 MJy/sr, model
peak 335), a new crater at (8993, 535) (depth 87 MJy/sr, model peak 259) and a
gone crater at (7060, 4678) (depth 85 to 86 MJy/sr, model peak 173).

**Survival of the added rows.**  Of the 53 exemption rows, 45 appear as new
rows of the merged catalog (no row of `dncg` within 0.1″) and 8 coincide with
a row that `dncg` already had.  Of the 45 relaxed rows, 38 are new and 7
coincide with a `dncg` row.  The median `flux_fit_avg` of the new rows (in the
unit of the catalog column) is 1536 for the exemption rows against a median
seed flux of 1522, and 1497 for the relaxed rows against 1190
(`results/tile_ab_o045_F212N_survival.txt`).

**PSF grid state and per-frame differences.**  All four arms use the
saturated-star PSF grid that the production run cached, so the `EPSFSHFT`
keyword of the frame catalogs of `dncg` and `sd1` is equal in all 8 detectors
(26.570 to 69.574 mpix; `results/tile_ab_o045_F212N_replay_state.txt`) and the
two replay sets share one grid state.  Between `dncg` and `sd1`, 16 of the
2,543,465 frame rows are lost, 355 are added and 150 matched rows change their
fitted flux by more than 1%.  Every one of them lies within 0.5″ of one of the
98 added rows (12 of the lost rows, 350 of the added rows and 60 of the
changed rows within 0.15″), and 120 of the 2,543,465 control rows lie within
0.15″ of an added row.  The added rows therefore change the per-frame fit only
locally.

**Remaining leftovers.**  With the edge band, `sd1p` still has 865, 168, 47
and 10 leftovers at ≥ 50, 100, 200 and 500 MJy/sr, and `sd1` has 882, 176, 51
and 10.  By class, the 51 leftovers ≥ 200 MJy/sr of `sd1p` (DQ_SAT dilation
without the edge band) are 32 SATSTAR_NEAR, 13 NOT_IN_M7_CATALOG, 4
VETTED_BUT_UNSUBTRACTED and 2 VETTED_OUT.

**Limits.**  The list under "What this test does not show" above applies to
this tile as well (the added rows are not vetted again, no arm contains one
kind of row alone, and a leftover that disappears does not show that its
source was a star).  The tile-level test covers o105 and o045 in F212N; o118,
o130, o138 and F480M were not run, and the S/N values were tuned on F212N.

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
- **Reach.**  Six leftovers ≥ 200 MJy/sr per tile (of 32 and 35).  In the tile
  tests of o105 F212N and o045 F212N the exemption rows together with the
  relaxed rows of #1376 clear 45 of the 194 (o105) and 51 of the 239 (o045)
  leftovers ≥ 100 MJy/sr, 20 and 27 of them under an exemption row, and the
  count at ≥ 200 MJy/sr falls from 45 to 34 (o105) and from 65 to 56 (o045)
  ([o105](#tile-level-ab-on-o105-f212n), [o045](#tile-level-ab-on-o045-f212n)).
  The remaining causes need other fixes (#1365, #1366, close-pair handling).

## Not measured

- The tile-level A/Bs cover o105 F212N and o045 F212N only, with the exemption
  rows and the relaxed rows of #1376 together.  No vetting or finalize ran with
  the added rows; the production vetted catalog plus the difference of the
  merged catalogs stands in for them, so the counts are the best case for the
  added rows.  The other tiles (o118, o130, o138) and F480M were not run at
  tile level.
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
