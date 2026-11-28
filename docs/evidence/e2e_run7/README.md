# End-to-end four-tile test of #1392, #1394, #1395 and a keep radius of 1.5 FWHM

One scratch-basepath run per arm and tile on the gc-treasury 10678 tiles o105, o118, o130 and o138, F212N and F480M. The run repeats the end-to-end run of #1387 (the four-tile run, `docs/evidence/e2e_four_tiles/README.md`) with the code of that run plus three further opt-in switches and one new value of an existing option: the spike finder that tolerates one unusable neighbour (#1392), the near-saturation filter that measures the distance from the unrounded fitted position (#1394), the exclusion of partial-fit-box rows from the per-frame averages (#1395), and a keep radius of 1.5 FWHM in the far-companion exemption of the m7 seed (#1377; the code tree of this run has the default 1.75, the 1.5 is passed on the command line, and the default of #1377 became 1.5 after this run). #1393 (a scale of the satstar implied-peak gate for the ePSF core) stays a draft and is not in the code tree of this run.

Every tile ran twice. The opt arm has every opt-in switch of the four-tile recipe on and the four additions above. The control arm has every opt-in switch off and no footprint mask. Each arm ran the m7 per-frame fits as an 8-task array, a finalize job (merge, vetting and the residual renders; the opt arm renders with the footprint masks of the recipe) and a post job (unmasked render of the opt arm, scoring and comparison of the renders). The four-tile run had a control arm on o105 only (render `ctlfin`); this run has one on every tile (render `c7fin`), so the difference between the integration code and production can be read off tile by tile (section "Control arms").

The run asks four questions. Do the three new switches clear the features that motivated them: the leftovers at one-pixel spikes next to a bad pixel (#1392), the two F480M leftovers of o105 that the four-tile run found with every switch off (#1394), and the leftover of o130 that the merge lost (#1395)? Does the keep radius of 1.5 FWHM add seed rows that clear leftovers without adding craters? Does any of the four additions add leftovers or craters anywhere (3 px matching and position-independent scans against the control arm, the production render and the render of the four-tile run)? How large is the code drift of the integration tree against production on each tile?

Every number in this file belongs to this run unless the text says otherwise. The tables are generated from the products of the run (mosaics, catalogs, per-frame fit tables and job logs), and `results/` holds the text and JSON outputs of the comparison, scan, seed and trace steps (list at the end).

**Result.**

- **Counts with the real masks** (`e7fin`; sums over the four tiles; the thresholds are explained under "How the counts are made"). The numbers read production `prod5` → control arm `c7fin` → opt arm `e7fin`, with `optfin` of the four-tile run in parentheses. F212N: leftovers ≥100 MJy/sr 702 → 695 → 79 (85) and ≥500 MJy/sr 384 → 385 → 1 (2); craters ≥50 MJy/sr 1382 → 1299 → 28 (28) and ≥200 MJy/sr 443 → 400 → 0 (0). F480M: leftovers ≥100 MJy/sr 518 → 510 → 17 (23) and ≥500 MJy/sr 221 → 212 → 0 (0); craters ≥50 MJy/sr 163 → 193 → 0 (0). The masks set 4.54 to 13.78 % (F212N) and 8.36 to 23.33 % (F480M) of the finite pixels of the data mosaic of a tile to NaN (section "Masked area and grid check"); a leftover or crater inside a masked region has no counterpart in the masked render, so the masked counts are low where the mask hides the pixels.
- **Counts without the masks** (`e7nm`: the catalogs of the opt finalize job rendered with every `MERGEDCAT_FOOTPRINT_MASK_*` variable unset; in parentheses `optnm` of the four-tile run). F212N: leftovers ≥100 MJy/sr 702 → 695 → 201 (209) and ≥500 MJy/sr 384 → 385 → 39 (42); craters ≥200 MJy/sr 443 → 400 → 112 (112). F480M: leftovers ≥100 MJy/sr 518 → 510 → 265 (299) and ≥500 MJy/sr 221 → 212 → 193 (199); craters ≥200 MJy/sr 52 → 60 → 16 (16). The unmasked render shows what the fit-time switches and the seed options do. The masks hide pixels and leave the catalogs unchanged: at every pixel that is finite in both renders the masked and the unmasked render are equal (grid check, section "Masked area and grid check").
- **The four additions against the four-tile run** (3 px matching of leftovers ≥100 MJy/sr; same tiles and same environment, the code trees differ by the three PRs and the keep radius). Masked renders `optfin` → `e7fin`: F212N 0 worse and 0 listed as new, 6 of 85 cleared; F480M 0 worse and 0 listed as new, 6 of 23 cleared. Unmasked renders `optnm` → `e7nm`: F212N 0 worse and 0 new, 9 of 209 cleared; F480M 0 worse and 1 new, 35 of 299 cleared. The position-independent scans list the leftovers of the first render that are absent in the second render, F212N / F480M: 0 / 1 for `e7nm` against `optnm` and 0 / 0 for `e7fin` against `optfin` (section "Position-independent scans"). The lists show three differences between `e7nm` and `optnm`, which the section "Differences against the four-tile run that the lists show" traces to the per-frame fits: the o105 F480M leftover at (5325, 1803) is a cost of #1394 (box maximum 42 → 338 MJy/sr, hidden by both masks); the o138 F212N crater at (8695, 623) is a listing flip at the cut of the crater scan (value 129.12 against a cut of 129.53 MJy/sr in `optnm`, 129.25 against 129.01 in `e7nm`) that follows from one extra o138 F212N row of the opt catalog, which replays of the fit stage trace to #1394; and the o118 F212N crater at (11389, 1475) is cleared by #1395 (box minimum -65 → -5 MJy/sr).
- **The spike finder (#1392).** Six leftovers ≥100 MJy/sr of the masked render of the four-tile run coincide with a one-pixel spike that the finder returns at tolerance 1 and lie in the tiles of this run: four of o118 F480M and two of o130 F480M (the seventh, on o045 F480M, is not in this run). The residual box maximum (13 × 13 px) at the strongest of them, o130 F480M (1932, 146), is 363 MJy/sr in the production render, 363 in `optnm` and 11 in `e7nm`; no leftover ≥100 MJy/sr remains at any of the six in `e7nm` or `e7fin` (table in the section "Features that motivated the switches").
- **The near-saturation distance (#1394).** The two F480M leftovers of o105 at (2614, 1826) and (4452, 858) are present with every switch off in the four-tile run (`ctlfin`: box maxima 917 and 418 MJy/sr, production 29 and 30). In `e7nm` the box maxima are 33 and 59 MJy/sr. The continuous distance also drops one fit that the default rule keeps, at 0.975 px from a SATURATED pixel centre, and the o105 F480M row that stands on it disappears (box maximum 42 → 338 MJy/sr at the pixel 5325, 1803; both masks hide the box).
- **The partial-fit-box rule (#1395).** The o130 F212N leftover at (5065, 68), lost in the merge in the four-tile run, has the box maximum 1244 MJy/sr in `optnm` and 9 MJy/sr in `e7nm`. The o105 F212N leftover at (9362, 1291), the other member of the group, stays at 209 MJy/sr (`optnm` 209). The rule also clears the o118 F212N crater at (11389, 1475): the merged row of the star loses the five-pixel fit of exposure 2, which differs from exposure 1 by 4.16 sigma of the difference, from its averages.
- **The keep radius of 1.5 FWHM (#1377).** The m7 seed of F212N holds 15 merged-module rows that the seed of the four-tile run lacks and drops 0; the seed of F480M is unchanged (keep S/N 0 there). All added rows have the seed origin `own_m6`. Followed to the residual mosaics, 5 of the 15 rows clear a leftover of 50 MJy/sr or more in the unmasked render, no row adds a crater that the four-tile render lacked (verdict "false seed": 0 rows), no row adds a leftover (verdict "new left": 0 rows), and 10 rows show neither (section "Seed rows added by the keep radius"). The two o105 F212N leftovers of group 3 of the four-tile run (cut by the companion cut; section "Cause of each remaining leftover" of its README), at (11052, 3798) and (1660, 341), go from 343 to 111 and from 213 to 73 MJy/sr in `e7nm`.
- **Control arms** (every switch off, render `c7fin`; section "Control arms"). Against production the matching of leftovers ≥100 MJy/sr clears 28 of 702 (F212N) and 42 of 518 (F480M) and lists 20 (F212N) and 33 (F480M) as new, so the code drift moves the F212N leftovers ≥100 MJy/sr from 702 to 695 and the F480M ones from 518 to 510, while the unmasked opt render reaches 201 and 265. The control render of o105 against the control render `ctlfin` of the four-tile run (same tile; code trees 42655da6 and ac6393dc, every switch off in both): F212N leftovers ≥100 MJy/sr 315 against 315, with 0 cleared and 0 listed as new; F480M 200 against 200, with 0 cleared and 0 listed as new.
- **Remaining leftovers of the masked render.** `e7fin` has 79 leftovers ≥100 MJy/sr in F212N and 17 in F480M. Of the F212N leftovers 32 are SATSTAR_NEAR and 45 are NOT_IN_M7_CATALOG; of the F480M leftovers 6 are VETTED_BUT_UNSUBTRACTED and 10 NOT_IN_M7_CATALOG (section "Remaining leftovers of the masked render"). The masked F212N render has 28 craters ≥50 MJy/sr and 0 ≥200 MJy/sr, and the masked F480M render has 0 craters ≥50 MJy/sr.
- **What the run does not separate.** The three switches and the keep radius are on together in the opt arm, so the effect of one switch rests on the feature that it was built for (the targeted features above) and on the seed comparison of the keep radius. No arm has one of the four additions alone, and the one o138 F212N row (flux 8975) that exists in the opt catalog of this run only was traced by one-frame replays to #1394 alone (the other switches were separated for that row only). The control arm has every switch off, so it measures the drift of the code that runs without switches; the code paths that only a switch reaches are not measured by it. The default of the keep radius was decided after this run, 1.5 FWHM (section "Decisions and open items").

## How the counts are made

The whole-star leftover, the class of a leftover, the big crater, the 3 px matching and the position-independent scan are those of the section "How the counts are made" of `docs/evidence/e2e_four_tiles/README.md`, with the same thresholds: leftovers with a residual peak of at least 50, 100, 200, 500 and 1000 MJy/sr, craters of depth at least 50, 200 and 500 MJy/sr, a matching radius of 3 px (0.094 arcsec), a scan box of 13 × 13 px, a scan threshold of 200 MJy/sr and a factor of 1.5. The scoring scripts are the ones of that run, so the production render and every render of this run are scored alike.

Renders that appear in the tables. The tags `prod5`, `optfin` and `optnm` are the renders of the four-tile run (#1387); the tags with `7` are renders of this run.

| render | what it is |
| :--- | :--- |
| `prod5` | production residual i2d of the tile (production tree, read only; no mask) |
| `c7fin` | control arm of this run: residual i2d that its finalize job writes (every switch off, no footprint mask) |
| `e7nm` | opt arm of this run: residual i2d rendered from the catalogs of its finalize job with every `MERGEDCAT_FOOTPRINT_MASK_*` variable unset |
| `e7fin` | opt arm of this run: residual i2d that its finalize job writes, with the footprint masks of the recipe |
| `optnm`, `optfin` | the same two renders of the opt arm of the four-tile run (code tree 42655da6, keep radius 1.75, none of the three new switches) |
| `ctlfin` | o105 only: control render of the four-tile run's control arm on o105 (code tree 42655da6, every switch off, no mask) |

Pairs of renders in the matching and scan tables are written A → B (matching: the leftovers of A are looked up in B) or source against reference (scan). The scan pairs are `c7fin` against `prod5` (code and environment drift of the control arm), `e7nm` against `c7fin` (effect of the fit-time switches, both renders unmasked), `e7fin` against `c7fin` (switches and masks), `e7nm` against `optnm` and `e7fin` against `optfin` (what this recipe changes against the recipe of the four-tile run).

The masked area of `e7fin` is the share of the finite pixels of the data i2d that are NaN in the residual i2d. The grid check compares `e7fin` with `e7nm` pixel by pixel (section "Masked area and grid check").

## Recipe

The opt arm uses the environment of the four-tile run (section "Recipe" of `docs/evidence/e2e_four_tiles/README.md`: 13 fit-time lines, the last of which is the second-pass neighbour guard `SATSTAR_SECOND_PASS_NEIGHBOUR_FRAC_MIN=0.2`, 9 render-time lines and the base options of the production m7 run) and adds four environment lines and one command-line value.

```bash
# fit time, added to the environment of the four-tile run
export NEARSAT_CONTINUOUS_DIST=1                # #1394
export POSITIVE_SPIKE_MASK_MAX_UNUSABLE=1       # #1392
export MERGE_PARTIAL_BOX_FRAC_MIN=0.5           # #1395
export MERGE_PARTIAL_BOX_ZMAX=3                 # #1395
```

| option | PR | default in the code tree of this run | value in the four-tile run | value of this run |
| :--- | :--- | ---: | ---: | ---: |
| `--manual-m7-seed-own-band-companion-keep-fwhm` | #1377 | 1.75 | 1.75 (default) | 1.5 |

The value 1.5 is passed on the command line of the opt arm only. The code tree of this run has the default 1.75. The default of #1377 is 1.5 since its commit 39cfac28 of 2026-11-28 (section "Decisions and open items"), which this tree does not contain.

What each addition does:

- `NEARSAT_CONTINUOUS_DIST=1` (#1394). The near-saturation filter of the daophot channel measures the distance from the unrounded fitted position to the nearest SATURATED pixel centre and drops a fit at 1.0 px or less. The default reads the distance transform at the pixel that the position rounds to. NIRCam only.
- `POSITIVE_SPIKE_MASK_MAX_UNUSABLE=1` (#1392). The one-pixel positive-spike finder of #1374 (`POSITIVE_SPIKE_MASK_MIN_EXCESS=300` in the four-tile run) also judges a pixel that has one unusable neighbour (NaN, DO_NOT_USE or outside the array). A neighbour with the SATURATED flag is never tolerated.
- `MERGE_PARTIAL_BOX_FRAC_MIN=0.5` and `MERGE_PARTIAL_BOX_ZMAX=3` (#1395). A per-frame row that was fitted to fewer than 0.5 of the 25 pixels of its fit box leaves the flux and position averages of its star when the star has a kept row at or above 0.5 and the flux of the partial row differs from the inverse-variance mean of the other rows by more than 3 sigma of the difference.
- `--manual-m7-seed-own-band-companion-keep-fwhm=1.5` (#1377). With a keep S/N above 0 (F212N=50 in the recipe of the four-tile run), an own-band m6 source that the companion cut would remove (a source within 2.5 FWHM of a source more than 10 times brighter) stays in the m7 seed when every dominating source lies at least this many FWHM away and is unsaturated. The four-tile run used 1.75. F480M has the keep S/N 0, so the option does not act in F480M.

The control arm uses the common environment file alone. It holds the ePSF core directory, the satstar settings of production (`SATSTAR_OPTIN_FILTERS=F480M`, `SATSTAR_ZF_KEEP_FINITE=1`, `SATSTAR_OBS_PK_FROM_CRF=1`, `SATSTAR_QFIT_LOCAL_GATE=1`), the two F480M hand-off settings of production (`DAOPHOT_HANDOFF_FILTERS=F480M`, `DAOPHOT_HANDOFF_PEAK_MIN_AREA=50`, `DAOPHOT_HANDOFF_RESTORE_ACCEPTED_EXCL_FWHM=1.5`) and `CRDS_READONLY_CACHE=1`. No opt-in switch is set in it. Its command line carries the four base options of the production m7 run and no seed option, so the seed options keep their defaults (keep S/N 0, keep radius 1.75 as the default of the tree of this run, bright roundness window off). The environment of every submission is saved (`results/run7_envsnap_<arm>_<tile>.txt`), and the saved environment of every control arm holds none of the variables `NEARSAT_*`, `POSITIVE_SPIKE_*`, `MERGE_*`, `MERGEDCAT_*`, `SINGLE_EXPOSURE_*`, `NEGATIVE_PIXEL_*`, `DAOPHOT_DEDUP_*` and the opt-in `SATSTAR_*` variables of the four-tile run.

Command-line options of the opt arm: the base options (`--manual-ext-star-prom-robust-min=-1`, `--residual-bg-median-size=0`, `--residual-bg-satstar-mask-fwhm=3.75`, `--manual-seed-round-loose-max=-1`) and the five seed options `--manual-m7-seed-own-band-companion-keep-snr=F212N=50,F480M=0`, `--manual-seed-round-bright-max=1.5`, `--manual-seed-round-bright-snr-min=F212N=30,F480M=100` and `--manual-m7-seed-own-band-companion-keep-fwhm=1.5`, the last one new in this run.

Pull requests that the integration tree takes in addition to the 17 of the four-tile run (the table of its README lists those). Titles, base branches and heads as of 2026-11-28 (`gh pr view`).

| PR | title | base branch | head | commits taken into the tree |
| :--- | :--- | :--- | :--- | :--- |
| #1392 | Spike finder: tolerate one unusable neighbour (opt-in POSITIVE_SPIKE_MASK_MAX_UNUSABLE) | `positive-spike-pixel-mask` | bc04a89a | 811983a7 and f44a3794 (code and tests) |
| #1394 | near-saturation filter: opt-in distance from the unrounded fitted position (NEARSAT_CONTINUOUS_DIST) | `main` | 616a72a7 | 6a9f042f (code, tests and its own doc page) |
| #1395 | combine_singleframe: opt-in exclusion of partial-fit-box rows from the averages (MERGE_PARTIAL_BOX_FRAC_MIN) | `main` | a230c994 | 945bdc52 and the code part of a230c994 |

#1393 (an opt-in scale of the satstar implied-peak gate for the ePSF core, head c2ca67d5) stays a draft and is not in the tree. #1377, which carries the keep-radius option, is in the tree of the four-tile run (head 1553ed42) and so in the tree of this run, which builds on it. The head of #1377 is 39cfac28 since the default of the keep radius changed to 1.5 FWHM.

## Setup of the run

**Inputs.** The production tree `/orange/adamginsburg/jwst/gc-treasury` is read only. Each arm of each tile has a scratch base path `<scratch>/e2e7/<arm>/<tile>/gc-treasury`, built the way the base paths of the four-tile run were built: symlinks to the production per-frame inputs (destreaked crf, cal and ramp files), the data mosaics and the m6 residual mosaics, and copies of the m6 smoothed-background mosaics, the m6 per-module catalogs with their vetted tables, the m2 consensus tables, the VIRAC2 reference catalog of the tile, the m2 to m6 astrometry records, the offsets table and the PSF grids. No m7 product, cross-band seed table, consolidated satstar catalog or per-frame satstar product is staged, so every arm builds them itself. A tile has 60 frames, 48 in F212N and 12 in F480M. The 8 tasks of the fan-out array share them round-robin, and every shard rebuilds the cross-band seed of the tile from the m6 catalogs of both bands.

The staging plan of each of the eight arm-tiles (destination and source of every symlink and copy) equals, entry by entry, the plan of the same tile in the four-tile run. A plan has 378 entries for o105 (198 symlinks and 180 copies), 346 for o118 (198 and 148), 381 for o130 (198 and 183) and 376 for o138 (198 and 178). After staging, a check followed every link to its planned source, compared every copy byte for byte with its source and listed every file of the tree. It found 0 wrong entries and 0 unplanned files in all eight trees. One staging job (2 min 17 s, 2026-11-27 20:53:19 to 20:55:36) staged the eight trees.

**Jobs.** Per arm and tile, one fan-out array of 8 tasks (2 CPUs and 32 GB per task), one finalize job (4 CPUs, 128 GB; starts after the fan-out ended without error) and one post job (5 CPUs, 96 GB; starts after the finalize job ended without error). The post job scores the finalize render, makes the grid check, re-renders the opt arm without the masks and scores that render. All jobs ran on partition `hpg-dev` with account `astronomy-dept` and QOS `astronomy-dept-b`, excluding the node `c0709a-s30`. The table is generated from `sacct` (`results/run7_jobtable.txt`).

| job name | completed jobs | all COMPLETED 0:0 | CPUs per job | requested memory | elapsed per job (h) | peak RSS (GB) | first start | last end |
| :--- | ---: | :--- | ---: | :--- | ---: | ---: | :--- | :--- |
| `e7-o105-opt-fan` | 8 | yes | 2 | 32G | 2.15 to 2.88 | 17.6 | 2026-11-27T20:56:42 | 2026-11-27T23:49:33 |
| `e7-o105-opt-fin` | 1 | yes | 4 | 128G | 2.08 | 12.7 | 2026-11-27T23:49:35 | 2026-11-28T01:54:13 |
| `e7-o105-opt-post` | 1 | yes | 5 | 96G | 0.20 | 12.8 | 2026-11-28T01:54:15 | 2026-11-28T02:06:01 |
| `e7-o118-opt-fan` | 8 | yes | 2 | 32G | 0.76 to 2.24 | 17.4 | 2026-11-27T20:56:42 | 2026-11-27T23:11:18 |
| `e7-o118-opt-fin` | 1 | yes | 4 | 128G | 1.54 | 12.5 | 2026-11-27T23:27:22 | 2026-11-28T00:59:31 |
| `e7-o118-opt-post` | 1 | yes | 5 | 96G | 0.19 | 12.8 | 2026-11-28T00:59:32 | 2026-11-28T01:10:53 |
| `e7-o130-opt-fan` | 8 | yes | 2 | 32G | 0.70 to 1.29 | 17.3 | 2026-11-27T20:56:42 | 2026-11-27T22:14:18 |
| `e7-o130-opt-fin` | 1 | yes | 4 | 128G | 1.41 | 12.6 | 2026-11-27T22:41:55 | 2026-11-28T00:06:31 |
| `e7-o130-opt-post` | 1 | yes | 5 | 96G | 0.22 | 11.0 | 2026-11-28T00:06:34 | 2026-11-28T00:19:38 |
| `e7-o138-opt-fan` | 8 | yes | 2 | 32G | 1.20 to 1.81 | 17.5 | 2026-11-27T20:56:42 | 2026-11-27T22:45:35 |
| `e7-o138-opt-fin` | 1 | yes | 4 | 128G | 1.93 | 12.9 | 2026-11-27T22:58:47 | 2026-11-28T00:54:27 |
| `e7-o138-opt-post` | 1 | yes | 5 | 96G | 0.19 | 12.7 | 2026-11-28T00:54:28 | 2026-11-28T01:05:43 |
| `e7-o105-ctl-fan` | 8 | yes | 2 | 32G | 1.53 to 2.25 | 17.7 | 2026-11-27T22:10:20 | 2026-11-28T00:28:37 |
| `e7-o105-ctl-fin` | 1 | yes | 4 | 128G | 2.05 | 12.3 | 2026-11-28T00:28:40 | 2026-11-28T02:31:36 |
| `e7-o105-ctl-post` | 1 | yes | 5 | 96G | 0.01 | 0.0 | 2026-11-28T02:31:39 | 2026-11-28T02:32:29 |
| `e7-o118-ctl-fan` | 8 | yes | 2 | 32G | 0.74 to 1.40 | 17.2 | 2026-11-27T20:56:45 | 2026-11-27T22:20:35 |
| `e7-o118-ctl-fin` | 1 | yes | 4 | 128G | 1.39 | 12.1 | 2026-11-27T22:46:56 | 2026-11-28T00:10:15 |
| `e7-o118-ctl-post` | 1 | yes | 5 | 96G | 0.01 | 0.0 | 2026-11-28T00:10:18 | 2026-11-28T00:10:53 |
| `e7-o130-ctl-fan` | 8 | yes | 2 | 32G | 0.62 to 1.28 | 17.3 | 2026-11-27T21:00:24 | 2026-11-27T22:55:42 |
| `e7-o130-ctl-fin` | 1 | yes | 4 | 128G | 1.28 | 12.2 | 2026-11-27T23:06:34 | 2026-11-28T00:23:25 |
| `e7-o130-ctl-post` | 1 | yes | 5 | 96G | 0.01 | 0.0 | 2026-11-28T00:23:29 | 2026-11-28T00:24:01 |
| `e7-o138-ctl-fan` | 8 | yes | 2 | 32G | 1.14 to 2.09 | 17.6 | 2026-11-27T21:54:40 | 2026-11-28T00:12:07 |
| `e7-o138-ctl-fin` | 1 | yes | 4 | 128G | 1.67 | 12.4 | 2026-11-28T00:12:08 | 2026-11-28T01:52:13 |
| `e7-o138-ctl-post` | 1 | yes | 5 | 96G | 0.01 | 0.0 | 2026-11-28T01:52:14 | 2026-11-28T01:52:57 |

80 jobs in total, all COMPLETED with exit code 0:0.

Slurm ids (fan-out array, finalize, post). Opt arm: o105 53122923, 53122924, 53122925; o118 53122926, 53122927, 53122928; o130 53122929, 53122930, 53122931; o138 53122932, 53122933, 53122934. Control arm: o105 53122984, 53122985, 53122986; o118 53122964, 53122965, 53122966; o130 53122974, 53122975, 53122976; o138 53122977, 53122978, 53122980. The chains were submitted between 2026-11-27 20:56:38 and 20:56:50, after the stage job and after the unit tests of the integration tree had passed.

**Code.** The jobs run the integration tree of this run: the integration tree of the four-tile run (HEAD 42655da6) with the code commits of #1392, #1394 and #1395 cherry-picked on top (HEAD ac6393dc; 10 files changed, 1571 insertions and 23 deletions against 42655da6; `results/run7_integration_tree.txt`). The PR heads are not ancestors of the tree because each PR also carries documentation commits that the tree lacks; the record lists them, and every commit of the heads that the tree lacks changes documentation only, apart from the comment lines of `merge_catalogs.py` in a230c994 that the tree took as its last commit. Two cherry-picks needed conflict resolutions that keep both sides: `single_exposure_event_masks` in `crowdsource_catalogs_long.py` (the ramp terms of #1367 stay, and the one-neighbour tolerance is added to the early return, the cache key and the warning) and two hunks in `cataloging.py` (the keyword argument `dedup_recovery_min_sep_pix` of #1366 and the new `near_sat_continuous` of #1394 stay side by side). A third cherry-pick conflicted in a documentation file of #1395 that the tree does not take. The tree lacks the code of #1393. Unit tests of the test files that the cherry-picks touch (the spike finder and its mask hooks, the near-saturation filter, the partial-box rule, the pair rescue, the seeded companion, the merge regressions and the dedup rule) ran before the submission, one file per job: 41, 116, 23, 34, 51, 15, 42 and 77 tests, 399 in total, all passed.

The production reference is the production render of each tile, as in the four-tile run. The latest production m7 fan-out of each tile ran a pinned production tree at commit 921d1ef9, 74 commits behind 803d0794, the base of the four-tile integration tree. A difference between a render of this run and the production render therefore combines the switches with the code drift of the integration tree. The control arm of this run (the tree of the opt arm with every opt-in switch off) measures the code drift on all four tiles, and the opt arm minus the control arm is the effect of the recipe on the same code.

**Checks of the products.** The precheck step of each post job confirmed that every product that the later steps read exists (the F480M precheck file of each arm-tile is empty). The jobs write their products under the scratch base path and set `CRDS_READONLY_CACHE=1`.

## Counts per tile

Whole-star leftovers and big craters in the residual i2d mosaics of the merged module (data minus merged-catalog model), per tile and render, with the definitions of the section "How the counts are made". The columns "leftovers ≥N" count whole-star leftovers with a residual peak of at least N MJy/sr, the columns "craters ≥N" count big craters of depth at least N MJy/sr, and the column "masked % of finite px" is the share of the finite pixels of the data mosaic that are NaN in the residual (section "Masked area and grid check"; it is blank for `optnm` and `optfin`, the renders of the four-tile run, and `optfin` has as many NaN pixels as `e7fin` in all eight tile-bands). The last six rows of each table are the sums over the four tiles.

**F212N**

| tile | render | what it is | leftovers ≥50 | ≥100 | ≥200 | ≥500 | ≥1000 | craters ≥50 | ≥200 | ≥500 | masked % of finite px |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| o105 | `prod5` | production render of the tile (no mask) | 845 | 317 | 178 | 133 | 116 | 489 | 166 | 71 | 0 |
| o105 | `ctlfin` | control render of the four-tile run (code tree 42655da6, every switch off, no mask); o105 only | 830 | 315 | 176 | 134 | 117 | 454 | 152 | 65 | 0 |
| o105 | `c7fin` | control arm, finalize render (every switch off) | 830 | 315 | 176 | 134 | 117 | 454 | 152 | 65 | 0 |
| o105 | `e7nm` | opt arm, unmasked re-render of the finalize catalogs | 604 | 114 | 35 | 12 | 7 | 102 | 44 | 18 | 0.00 |
| o105 | `e7fin` | opt arm, finalize render with the real masks | 471 | 49 | 6 | 0 | 0 | 10 | 0 | 0 | 13.78 |
| o105 | `optnm` | opt arm of the earlier run (#1387), unmasked | 610 | 118 | 38 | 13 | 8 | 102 | 44 | 18 |  |
| o105 | `optfin` | opt arm of the earlier run (#1387), real masks | 475 | 52 | 8 | 0 | 0 | 10 | 0 | 0 |  |
| o118 | `prod5` | production render of the tile (no mask) | 199 | 105 | 82 | 72 | 65 | 308 | 93 | 33 | 0 |
| o118 | `c7fin` | control arm, finalize render (every switch off) | 197 | 103 | 82 | 71 | 64 | 295 | 91 | 33 | 0 |
| o118 | `e7nm` | opt arm, unmasked re-render of the finalize catalogs | 92 | 18 | 7 | 3 | 1 | 73 | 26 | 2 | 0.00 |
| o118 | `e7fin` | opt arm, finalize render with the real masks | 79 | 16 | 4 | 1 | 0 | 5 | 0 | 0 | 5.46 |
| o118 | `optnm` | opt arm of the earlier run (#1387), unmasked | 92 | 18 | 7 | 3 | 1 | 74 | 26 | 2 |  |
| o118 | `optfin` | opt arm of the earlier run (#1387), real masks | 79 | 16 | 4 | 1 | 0 | 5 | 0 | 0 |  |
| o130 | `prod5` | production render of the tile (no mask) | 181 | 115 | 106 | 90 | 84 | 206 | 58 | 28 | 0 |
| o130 | `c7fin` | control arm, finalize render (every switch off) | 183 | 116 | 107 | 91 | 84 | 207 | 55 | 25 | 0 |
| o130 | `e7nm` | opt arm, unmasked re-render of the finalize catalogs | 70 | 13 | 11 | 4 | 2 | 56 | 14 | 3 | 0.00 |
| o130 | `e7fin` | opt arm, finalize render with the real masks | 50 | 1 | 0 | 0 | 0 | 5 | 0 | 0 | 4.54 |
| o130 | `optnm` | opt arm of the earlier run (#1387), unmasked | 73 | 16 | 13 | 6 | 3 | 56 | 14 | 3 |  |
| o130 | `optfin` | opt arm of the earlier run (#1387), real masks | 52 | 3 | 1 | 1 | 1 | 5 | 0 | 0 |  |
| o138 | `prod5` | production render of the tile (no mask) | 472 | 165 | 101 | 89 | 77 | 379 | 126 | 52 | 0 |
| o138 | `c7fin` | control arm, finalize render (every switch off) | 464 | 161 | 99 | 89 | 77 | 343 | 102 | 38 | 0 |
| o138 | `e7nm` | opt arm, unmasked re-render of the finalize catalogs | 329 | 56 | 28 | 20 | 13 | 91 | 28 | 11 | 0.00 |
| o138 | `e7fin` | opt arm, finalize render with the real masks | 241 | 13 | 1 | 0 | 0 | 8 | 0 | 0 | 7.81 |
| o138 | `optnm` | opt arm of the earlier run (#1387), unmasked | 330 | 57 | 28 | 20 | 13 | 90 | 28 | 11 |  |
| o138 | `optfin` | opt arm of the earlier run (#1387), real masks | 242 | 14 | 1 | 0 | 0 | 8 | 0 | 0 |  |
| all 4 | `prod5` | sum over the tiles | 1697 | 702 | 467 | 384 | 342 | 1382 | 443 | 184 | |
| all 4 | `c7fin` | sum over the tiles | 1674 | 695 | 464 | 385 | 342 | 1299 | 400 | 161 | |
| all 4 | `e7nm` | sum over the tiles | 1095 | 201 | 81 | 39 | 23 | 322 | 112 | 34 | |
| all 4 | `e7fin` | sum over the tiles | 841 | 79 | 11 | 1 | 0 | 28 | 0 | 0 | |
| all 4 | `optnm` | sum over the tiles | 1105 | 209 | 86 | 42 | 25 | 322 | 112 | 34 | |
| all 4 | `optfin` | sum over the tiles | 848 | 85 | 14 | 2 | 1 | 28 | 0 | 0 | |

**F480M**

| tile | render | what it is | leftovers ≥50 | ≥100 | ≥200 | ≥500 | ≥1000 | craters ≥50 | ≥200 | ≥500 | masked % of finite px |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| o105 | `prod5` | production render of the tile (no mask) | 303 | 202 | 168 | 111 | 84 | 69 | 28 | 16 | 0 |
| o105 | `ctlfin` | control render of the four-tile run (code tree 42655da6, every switch off, no mask); o105 only | 297 | 200 | 168 | 107 | 77 | 92 | 38 | 17 | 0 |
| o105 | `c7fin` | control arm, finalize render (every switch off) | 297 | 200 | 168 | 107 | 77 | 92 | 38 | 17 | 0 |
| o105 | `e7nm` | opt arm, unmasked re-render of the finalize catalogs | 199 | 133 | 126 | 99 | 77 | 20 | 12 | 8 | 0.00 |
| o105 | `e7fin` | opt arm, finalize render with the real masks | 49 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 23.33 |
| o105 | `optnm` | opt arm of the earlier run (#1387), unmasked | 221 | 155 | 146 | 104 | 78 | 20 | 12 | 8 |  |
| o105 | `optfin` | opt arm of the earlier run (#1387), real masks | 49 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| o118 | `prod5` | production render of the tile (no mask) | 219 | 121 | 76 | 47 | 38 | 26 | 6 | 5 | 0 |
| o118 | `c7fin` | control arm, finalize render (every switch off) | 220 | 122 | 78 | 48 | 36 | 33 | 7 | 4 | 0 |
| o118 | `e7nm` | opt arm, unmasked re-render of the finalize catalogs | 111 | 63 | 51 | 45 | 36 | 5 | 4 | 3 | 0.00 |
| o118 | `e7fin` | opt arm, finalize render with the real masks | 44 | 6 | 1 | 0 | 0 | 0 | 0 | 0 | 11.43 |
| o118 | `optnm` | opt arm of the earlier run (#1387), unmasked | 118 | 71 | 54 | 45 | 36 | 5 | 4 | 3 |  |
| o118 | `optfin` | opt arm of the earlier run (#1387), real masks | 48 | 10 | 1 | 0 | 0 | 0 | 0 | 0 |  |
| o130 | `prod5` | production render of the tile (no mask) | 191 | 90 | 50 | 19 | 14 | 39 | 12 | 1 | 0 |
| o130 | `c7fin` | control arm, finalize render (every switch off) | 186 | 88 | 49 | 18 | 12 | 34 | 8 | 0 | 0 |
| o130 | `e7nm` | opt arm, unmasked re-render of the finalize catalogs | 82 | 26 | 18 | 14 | 11 | 1 | 0 | 0 | 0.00 |
| o130 | `e7fin` | opt arm, finalize render with the real masks | 53 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 8.36 |
| o130 | `optnm` | opt arm of the earlier run (#1387), unmasked | 85 | 28 | 19 | 14 | 11 | 1 | 0 | 0 |  |
| o130 | `optfin` | opt arm of the earlier run (#1387), real masks | 56 | 7 | 1 | 0 | 0 | 0 | 0 | 0 |  |
| o138 | `prod5` | production render of the tile (no mask) | 185 | 105 | 64 | 44 | 37 | 29 | 6 | 2 | 0 |
| o138 | `c7fin` | control arm, finalize render (every switch off) | 181 | 100 | 59 | 39 | 33 | 34 | 7 | 0 | 0 |
| o138 | `e7nm` | opt arm, unmasked re-render of the finalize catalogs | 89 | 43 | 36 | 35 | 32 | 0 | 0 | 0 | 0.00 |
| o138 | `e7fin` | opt arm, finalize render with the real masks | 40 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 10.19 |
| o138 | `optnm` | opt arm of the earlier run (#1387), unmasked | 91 | 45 | 38 | 36 | 33 | 0 | 0 | 0 |  |
| o138 | `optfin` | opt arm of the earlier run (#1387), real masks | 40 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| all 4 | `prod5` | sum over the tiles | 898 | 518 | 358 | 221 | 173 | 163 | 52 | 24 | |
| all 4 | `c7fin` | sum over the tiles | 884 | 510 | 354 | 212 | 158 | 193 | 60 | 21 | |
| all 4 | `e7nm` | sum over the tiles | 481 | 265 | 231 | 193 | 156 | 26 | 16 | 11 | |
| all 4 | `e7fin` | sum over the tiles | 186 | 17 | 1 | 0 | 0 | 0 | 0 | 0 | |
| all 4 | `optnm` | sum over the tiles | 515 | 299 | 257 | 199 | 158 | 26 | 16 | 11 | |
| all 4 | `optfin` | sum over the tiles | 193 | 23 | 2 | 0 | 0 | 0 | 0 | 0 | |

The renders `optnm` and `optfin` are the files of the four-tile run, scored again with the scoring script of this run. Their counts equal those of the tables of the four-tile README for the four tiles (the F212N leftovers ≥100 MJy/sr of the four-tile run, for example, sum to 85 in `optfin` and 209 in `optnm`).

## Matching of the leftovers and craters between renders

For each pair of renders A → B, each leftover ≥100 MJy/sr of A is matched to a leftover of B within 3 px (0.094 arcsec) and counted as cleared (no match; this includes a leftover inside a region that the mask of B sets to NaN), reduced (B peak below 0.5 times the A peak), unchanged, or worse (B peak above 1.5 times the A peak). "New in B" counts the leftovers ≥100 MJy/sr of B without a counterpart in A. The three columns "cleared of A" give the number of leftovers of A with a residual peak of at least 200, 500 and 1000 MJy/sr that are cleared, out of the number of such leftovers in A. In the crater table, "gone" counts the big craters of A (depth ≥50 MJy/sr) without a crater of B within the radius and "listed as new" the craters of B without a counterpart in A; the last two columns restrict both to depth ≥200 MJy/sr. A feature whose extremum moves by more than 3 px between the renders counts once as cleared (or gone) and once as new. The last rows of each table are the sums over the four tiles.

What each pair measures:

- `prod5` → `c7fin`: the code drift of the integration tree against production, with every switch off.
- `c7fin` → `e7nm` and `c7fin` → `e7fin`: the opt arm against the control arm of the same tile, unmasked and masked. The two arms ran the same code tree and differ in the opt-in switches and the seed options.
- `optnm` → `e7nm` and `optfin` → `e7fin`: this run against the four-tile run. The environment is the same and the code trees differ by the three PRs of this run; the command line differs by the keep radius.
- `prod5` → `e7nm` and `prod5` → `e7fin`: this run against production.
- `ctlfin` → `c7fin` (o105 only): the control render of the four-tile run against the control render of this run.

**Leftovers ≥100 MJy/sr**

**F212N**

| tile | pair A → B | leftovers ≥100 in A | in B | cleared | reduced below half | unchanged | worse than 1.5× | new in B | cleared of A ≥200 | of A ≥500 | of A ≥1000 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- | :--- |
| o105 | `prod5` → `c7fin` | 317 | 315 | 13 | 0 | 304 | 0 | 10 | 6 of 178 | 1 of 133 | 0 of 116 |
| o105 | `c7fin` → `e7nm` | 315 | 114 | 202 | 0 | 113 | 0 | 0 | 141 of 176 | 122 of 134 | 110 of 117 |
| o105 | `c7fin` → `e7fin` | 315 | 49 | 268 | 0 | 47 | 0 | 2 | 170 of 176 | 134 of 134 | 117 of 117 |
| o105 | `optnm` → `e7nm` | 118 | 114 | 5 | 0 | 113 | 0 | 0 | 3 of 38 | 1 of 13 | 1 of 8 |
| o105 | `optfin` → `e7fin` | 52 | 49 | 3 | 0 | 49 | 0 | 0 | 2 of 8 | 0 of 0 | 0 of 0 |
| o105 | `prod5` → `e7fin` | 317 | 49 | 270 | 0 | 47 | 0 | 2 | 172 of 178 | 133 of 133 | 116 of 116 |
| o105 | `prod5` → `e7nm` | 317 | 114 | 210 | 0 | 107 | 0 | 5 | 145 of 178 | 122 of 133 | 109 of 116 |
| o105 | `ctlfin` → `c7fin` | 315 | 315 | 0 | 0 | 315 | 0 | 0 | 0 of 176 | 0 of 134 | 0 of 117 |
| o118 | `prod5` → `c7fin` | 105 | 103 | 4 | 0 | 101 | 0 | 2 | 2 of 82 | 2 of 72 | 2 of 65 |
| o118 | `c7fin` → `e7nm` | 103 | 18 | 85 | 0 | 18 | 0 | 0 | 75 of 82 | 68 of 71 | 63 of 64 |
| o118 | `c7fin` → `e7fin` | 103 | 16 | 93 | 0 | 10 | 0 | 6 | 80 of 82 | 70 of 71 | 64 of 64 |
| o118 | `optnm` → `e7nm` | 18 | 18 | 0 | 0 | 18 | 0 | 0 | 0 of 7 | 0 of 3 | 0 of 1 |
| o118 | `optfin` → `e7fin` | 16 | 16 | 0 | 0 | 16 | 0 | 0 | 0 of 4 | 0 of 1 | 0 of 0 |
| o118 | `prod5` → `e7fin` | 105 | 16 | 95 | 0 | 10 | 0 | 6 | 80 of 82 | 71 of 72 | 65 of 65 |
| o118 | `prod5` → `e7nm` | 105 | 18 | 87 | 0 | 18 | 0 | 0 | 75 of 82 | 69 of 72 | 64 of 65 |
| o130 | `prod5` → `c7fin` | 115 | 116 | 1 | 0 | 114 | 0 | 2 | 0 of 106 | 0 of 90 | 0 of 84 |
| o130 | `c7fin` → `e7nm` | 116 | 13 | 103 | 0 | 13 | 0 | 0 | 96 of 107 | 87 of 91 | 82 of 84 |
| o130 | `c7fin` → `e7fin` | 116 | 1 | 115 | 0 | 1 | 0 | 0 | 107 of 107 | 91 of 91 | 84 of 84 |
| o130 | `optnm` → `e7nm` | 16 | 13 | 3 | 0 | 13 | 0 | 0 | 2 of 13 | 2 of 6 | 1 of 3 |
| o130 | `optfin` → `e7fin` | 3 | 1 | 2 | 0 | 1 | 0 | 0 | 1 of 1 | 1 of 1 | 1 of 1 |
| o130 | `prod5` → `e7fin` | 115 | 1 | 114 | 0 | 1 | 0 | 0 | 106 of 106 | 90 of 90 | 84 of 84 |
| o130 | `prod5` → `e7nm` | 115 | 13 | 103 | 0 | 12 | 0 | 1 | 95 of 106 | 86 of 90 | 82 of 84 |
| o138 | `prod5` → `c7fin` | 165 | 161 | 10 | 0 | 155 | 0 | 6 | 5 of 101 | 3 of 89 | 3 of 77 |
| o138 | `c7fin` → `e7nm` | 161 | 56 | 105 | 0 | 56 | 0 | 2 | 73 of 99 | 70 of 89 | 64 of 77 |
| o138 | `c7fin` → `e7fin` | 161 | 13 | 146 | 0 | 15 | 0 | 0 | 98 of 99 | 89 of 89 | 77 of 77 |
| o138 | `optnm` → `e7nm` | 57 | 56 | 1 | 0 | 56 | 0 | 0 | 0 of 28 | 0 of 20 | 0 of 13 |
| o138 | `optfin` → `e7fin` | 14 | 13 | 1 | 0 | 13 | 0 | 0 | 0 of 1 | 0 of 0 | 0 of 0 |
| o138 | `prod5` → `e7fin` | 165 | 13 | 150 | 0 | 15 | 0 | 0 | 100 of 101 | 89 of 89 | 77 of 77 |
| o138 | `prod5` → `e7nm` | 165 | 56 | 115 | 0 | 50 | 0 | 8 | 78 of 101 | 73 of 89 | 67 of 77 |
| all 4 | `prod5` → `c7fin` | 702 | 695 | 28 | 0 | 674 | 0 | 20 | 13 of 467 | 6 of 384 | 5 of 342 |
| all 4 | `c7fin` → `e7nm` | 695 | 201 | 495 | 0 | 200 | 0 | 2 | 385 of 464 | 347 of 385 | 319 of 342 |
| all 4 | `c7fin` → `e7fin` | 695 | 79 | 622 | 0 | 73 | 0 | 8 | 455 of 464 | 384 of 385 | 342 of 342 |
| all 4 | `optnm` → `e7nm` | 209 | 201 | 9 | 0 | 200 | 0 | 0 | 5 of 86 | 3 of 42 | 2 of 25 |
| all 4 | `optfin` → `e7fin` | 85 | 79 | 6 | 0 | 79 | 0 | 0 | 3 of 14 | 1 of 2 | 1 of 1 |
| all 4 | `prod5` → `e7fin` | 702 | 79 | 629 | 0 | 73 | 0 | 8 | 458 of 467 | 383 of 384 | 342 of 342 |
| all 4 | `prod5` → `e7nm` | 702 | 201 | 515 | 0 | 187 | 0 | 14 | 393 of 467 | 350 of 384 | 322 of 342 |

**F480M**

| tile | pair A → B | leftovers ≥100 in A | in B | cleared | reduced below half | unchanged | worse than 1.5× | new in B | cleared of A ≥200 | of A ≥500 | of A ≥1000 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- | :--- |
| o105 | `prod5` → `c7fin` | 202 | 200 | 17 | 0 | 185 | 0 | 14 | 14 of 168 | 13 of 111 | 10 of 84 |
| o105 | `c7fin` → `e7nm` | 200 | 133 | 70 | 0 | 130 | 0 | 3 | 45 of 168 | 10 of 107 | 1 of 77 |
| o105 | `c7fin` → `e7fin` | 200 | 2 | 198 | 0 | 2 | 0 | 0 | 168 of 168 | 107 of 107 | 77 of 77 |
| o105 | `optnm` → `e7nm` | 155 | 133 | 23 | 0 | 132 | 0 | 1 | 21 of 146 | 5 of 104 | 1 of 78 |
| o105 | `optfin` → `e7fin` | 2 | 2 | 0 | 0 | 2 | 0 | 0 | 0 of 0 | 0 of 0 | 0 of 0 |
| o105 | `prod5` → `e7fin` | 202 | 2 | 200 | 0 | 2 | 0 | 0 | 168 of 168 | 111 of 111 | 84 of 84 |
| o105 | `prod5` → `e7nm` | 202 | 133 | 81 | 0 | 121 | 0 | 12 | 53 of 168 | 20 of 111 | 10 of 84 |
| o105 | `ctlfin` → `c7fin` | 200 | 200 | 0 | 0 | 200 | 0 | 0 | 0 of 168 | 0 of 107 | 0 of 77 |
| o118 | `prod5` → `c7fin` | 121 | 122 | 6 | 0 | 115 | 0 | 7 | 6 of 76 | 6 of 47 | 6 of 38 |
| o118 | `c7fin` → `e7nm` | 122 | 63 | 61 | 0 | 61 | 0 | 2 | 29 of 78 | 4 of 48 | 1 of 36 |
| o118 | `c7fin` → `e7fin` | 122 | 6 | 116 | 0 | 6 | 0 | 0 | 77 of 78 | 48 of 48 | 36 of 36 |
| o118 | `optnm` → `e7nm` | 71 | 63 | 8 | 0 | 63 | 0 | 0 | 3 of 54 | 0 of 45 | 0 of 36 |
| o118 | `optfin` → `e7fin` | 10 | 6 | 4 | 0 | 6 | 0 | 0 | 0 of 1 | 0 of 0 | 0 of 0 |
| o118 | `prod5` → `e7fin` | 121 | 6 | 115 | 0 | 6 | 0 | 0 | 75 of 76 | 47 of 47 | 38 of 38 |
| o118 | `prod5` → `e7nm` | 121 | 63 | 66 | 0 | 55 | 0 | 8 | 33 of 76 | 9 of 47 | 6 of 38 |
| o130 | `prod5` → `c7fin` | 90 | 88 | 8 | 0 | 82 | 0 | 6 | 7 of 50 | 7 of 19 | 6 of 14 |
| o130 | `c7fin` → `e7nm` | 88 | 26 | 62 | 0 | 26 | 0 | 0 | 31 of 49 | 4 of 18 | 1 of 12 |
| o130 | `c7fin` → `e7fin` | 88 | 5 | 83 | 0 | 5 | 0 | 0 | 49 of 49 | 18 of 18 | 12 of 12 |
| o130 | `optnm` → `e7nm` | 28 | 26 | 2 | 0 | 26 | 0 | 0 | 1 of 19 | 0 of 14 | 0 of 11 |
| o130 | `optfin` → `e7fin` | 7 | 5 | 2 | 0 | 5 | 0 | 0 | 1 of 1 | 0 of 0 | 0 of 0 |
| o130 | `prod5` → `e7fin` | 90 | 5 | 85 | 0 | 5 | 0 | 0 | 50 of 50 | 19 of 19 | 14 of 14 |
| o130 | `prod5` → `e7nm` | 90 | 26 | 70 | 0 | 20 | 0 | 6 | 38 of 50 | 11 of 19 | 7 of 14 |
| o138 | `prod5` → `c7fin` | 105 | 100 | 11 | 0 | 94 | 0 | 6 | 11 of 64 | 11 of 44 | 10 of 37 |
| o138 | `c7fin` → `e7nm` | 100 | 43 | 57 | 0 | 43 | 0 | 0 | 23 of 59 | 4 of 39 | 1 of 33 |
| o138 | `c7fin` → `e7fin` | 100 | 4 | 96 | 0 | 4 | 0 | 0 | 59 of 59 | 39 of 39 | 33 of 33 |
| o138 | `optnm` → `e7nm` | 45 | 43 | 2 | 0 | 43 | 0 | 0 | 2 of 38 | 1 of 36 | 1 of 33 |
| o138 | `optfin` → `e7fin` | 4 | 4 | 0 | 0 | 4 | 0 | 0 | 0 of 0 | 0 of 0 | 0 of 0 |
| o138 | `prod5` → `e7fin` | 105 | 4 | 101 | 0 | 4 | 0 | 0 | 64 of 64 | 44 of 44 | 37 of 37 |
| o138 | `prod5` → `e7nm` | 105 | 43 | 68 | 0 | 37 | 0 | 6 | 34 of 64 | 15 of 44 | 11 of 37 |
| all 4 | `prod5` → `c7fin` | 518 | 510 | 42 | 0 | 476 | 0 | 33 | 38 of 358 | 37 of 221 | 32 of 173 |
| all 4 | `c7fin` → `e7nm` | 510 | 265 | 250 | 0 | 260 | 0 | 5 | 128 of 354 | 22 of 212 | 4 of 158 |
| all 4 | `c7fin` → `e7fin` | 510 | 17 | 493 | 0 | 17 | 0 | 0 | 353 of 354 | 212 of 212 | 158 of 158 |
| all 4 | `optnm` → `e7nm` | 299 | 265 | 35 | 0 | 264 | 0 | 1 | 27 of 257 | 6 of 199 | 2 of 158 |
| all 4 | `optfin` → `e7fin` | 23 | 17 | 6 | 0 | 17 | 0 | 0 | 1 of 2 | 0 of 0 | 0 of 0 |
| all 4 | `prod5` → `e7fin` | 518 | 17 | 501 | 0 | 17 | 0 | 0 | 357 of 358 | 221 of 221 | 173 of 173 |
| all 4 | `prod5` → `e7nm` | 518 | 265 | 285 | 0 | 233 | 0 | 32 | 158 of 358 | 55 of 221 | 34 of 173 |

**Big craters of depth ≥50 MJy/sr**

**F212N**

| tile | pair A → B | craters ≥50 in A | in B | gone | listed as new | of A ≥200: gone | of B ≥200: listed as new |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| o105 | `prod5` → `c7fin` | 489 | 454 | 82 | 47 | 30 | 17 |
| o105 | `c7fin` → `e7nm` | 454 | 102 | 354 | 2 | 108 | 2 |
| o105 | `c7fin` → `e7fin` | 454 | 10 | 450 | 6 | 152 | 0 |
| o105 | `optnm` → `e7nm` | 102 | 102 | 0 | 0 | 0 | 0 |
| o105 | `optfin` → `e7fin` | 10 | 10 | 0 | 0 | 0 | 0 |
| o105 | `prod5` → `e7fin` | 489 | 10 | 485 | 6 | 166 | 0 |
| o105 | `prod5` → `e7nm` | 489 | 102 | 396 | 9 | 123 | 3 |
| o105 | `ctlfin` → `c7fin` | 454 | 454 | 0 | 0 | 0 | 0 |
| o118 | `prod5` → `c7fin` | 308 | 295 | 41 | 29 | 13 | 9 |
| o118 | `c7fin` → `e7nm` | 295 | 73 | 224 | 2 | 65 | 2 |
| o118 | `c7fin` → `e7fin` | 295 | 5 | 292 | 2 | 91 | 0 |
| o118 | `optnm` → `e7nm` | 74 | 73 | 1 | 0 | 0 | 0 |
| o118 | `optfin` → `e7fin` | 5 | 5 | 0 | 0 | 0 | 0 |
| o118 | `prod5` → `e7fin` | 308 | 5 | 305 | 2 | 93 | 0 |
| o118 | `prod5` → `e7nm` | 308 | 73 | 243 | 8 | 67 | 3 |
| o130 | `prod5` → `c7fin` | 206 | 207 | 30 | 31 | 9 | 3 |
| o130 | `c7fin` → `e7nm` | 207 | 56 | 151 | 0 | 39 | 0 |
| o130 | `c7fin` → `e7fin` | 207 | 5 | 204 | 2 | 55 | 0 |
| o130 | `optnm` → `e7nm` | 56 | 56 | 0 | 0 | 0 | 0 |
| o130 | `optfin` → `e7fin` | 5 | 5 | 0 | 0 | 0 | 0 |
| o130 | `prod5` → `e7fin` | 206 | 5 | 203 | 2 | 58 | 0 |
| o130 | `prod5` → `e7nm` | 206 | 56 | 154 | 4 | 43 | 0 |
| o138 | `prod5` → `c7fin` | 379 | 343 | 76 | 41 | 32 | 11 |
| o138 | `c7fin` → `e7nm` | 343 | 91 | 256 | 4 | 74 | 1 |
| o138 | `c7fin` → `e7fin` | 343 | 8 | 339 | 4 | 102 | 0 |
| o138 | `optnm` → `e7nm` | 90 | 91 | 0 | 1 | 0 | 0 |
| o138 | `optfin` → `e7fin` | 8 | 8 | 0 | 0 | 0 | 0 |
| o138 | `prod5` → `e7fin` | 379 | 8 | 375 | 4 | 126 | 0 |
| o138 | `prod5` → `e7nm` | 379 | 91 | 300 | 12 | 100 | 3 |
| all 4 | `prod5` → `c7fin` | 1382 | 1299 | 229 | 148 | 84 | 40 |
| all 4 | `c7fin` → `e7nm` | 1299 | 322 | 985 | 8 | 286 | 5 |
| all 4 | `c7fin` → `e7fin` | 1299 | 28 | 1285 | 14 | 400 | 0 |
| all 4 | `optnm` → `e7nm` | 322 | 322 | 1 | 1 | 0 | 0 |
| all 4 | `optfin` → `e7fin` | 28 | 28 | 0 | 0 | 0 | 0 |
| all 4 | `prod5` → `e7fin` | 1382 | 28 | 1368 | 14 | 443 | 0 |
| all 4 | `prod5` → `e7nm` | 1382 | 322 | 1093 | 33 | 333 | 9 |

**F480M**

| tile | pair A → B | craters ≥50 in A | in B | gone | listed as new | of A ≥200: gone | of B ≥200: listed as new |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| o105 | `prod5` → `c7fin` | 69 | 92 | 27 | 50 | 12 | 20 |
| o105 | `c7fin` → `e7nm` | 92 | 20 | 72 | 0 | 26 | 0 |
| o105 | `c7fin` → `e7fin` | 92 | 0 | 92 | 0 | 38 | 0 |
| o105 | `optnm` → `e7nm` | 20 | 20 | 0 | 0 | 0 | 0 |
| o105 | `optfin` → `e7fin` | 0 | 0 | 0 | 0 | 0 | 0 |
| o105 | `prod5` → `e7fin` | 69 | 0 | 69 | 0 | 28 | 0 |
| o105 | `prod5` → `e7nm` | 69 | 20 | 59 | 10 | 21 | 5 |
| o105 | `ctlfin` → `c7fin` | 92 | 92 | 0 | 0 | 0 | 0 |
| o118 | `prod5` → `c7fin` | 26 | 33 | 6 | 13 | 1 | 2 |
| o118 | `c7fin` → `e7nm` | 33 | 5 | 29 | 1 | 4 | 1 |
| o118 | `c7fin` → `e7fin` | 33 | 0 | 33 | 0 | 7 | 0 |
| o118 | `optnm` → `e7nm` | 5 | 5 | 0 | 0 | 0 | 0 |
| o118 | `optfin` → `e7fin` | 0 | 0 | 0 | 0 | 0 | 0 |
| o118 | `prod5` → `e7fin` | 26 | 0 | 26 | 0 | 6 | 0 |
| o118 | `prod5` → `e7nm` | 26 | 5 | 21 | 0 | 2 | 0 |
| o130 | `prod5` → `c7fin` | 39 | 34 | 14 | 9 | 5 | 1 |
| o130 | `c7fin` → `e7nm` | 34 | 1 | 33 | 0 | 8 | 0 |
| o130 | `c7fin` → `e7fin` | 34 | 0 | 34 | 0 | 8 | 0 |
| o130 | `optnm` → `e7nm` | 1 | 1 | 0 | 0 | 0 | 0 |
| o130 | `optfin` → `e7fin` | 0 | 0 | 0 | 0 | 0 | 0 |
| o130 | `prod5` → `e7fin` | 39 | 0 | 39 | 0 | 12 | 0 |
| o130 | `prod5` → `e7nm` | 39 | 1 | 38 | 0 | 12 | 0 |
| o138 | `prod5` → `c7fin` | 29 | 34 | 9 | 14 | 2 | 1 |
| o138 | `c7fin` → `e7nm` | 34 | 0 | 34 | 0 | 7 | 0 |
| o138 | `c7fin` → `e7fin` | 34 | 0 | 34 | 0 | 7 | 0 |
| o138 | `optnm` → `e7nm` | 0 | 0 | 0 | 0 | 0 | 0 |
| o138 | `optfin` → `e7fin` | 0 | 0 | 0 | 0 | 0 | 0 |
| o138 | `prod5` → `e7fin` | 29 | 0 | 29 | 0 | 6 | 0 |
| o138 | `prod5` → `e7nm` | 29 | 0 | 29 | 0 | 6 | 0 |
| all 4 | `prod5` → `c7fin` | 163 | 193 | 56 | 86 | 20 | 24 |
| all 4 | `c7fin` → `e7nm` | 193 | 26 | 168 | 1 | 45 | 1 |
| all 4 | `c7fin` → `e7fin` | 193 | 0 | 193 | 0 | 60 | 0 |
| all 4 | `optnm` → `e7nm` | 26 | 26 | 0 | 0 | 0 | 0 |
| all 4 | `optfin` → `e7fin` | 0 | 0 | 0 | 0 | 0 | 0 |
| all 4 | `prod5` → `e7fin` | 163 | 0 | 163 | 0 | 52 | 0 |
| all 4 | `prod5` → `e7nm` | 163 | 26 | 147 | 10 | 41 | 5 |

**Classes of the cleared and of the new leftovers** (sums over the four tiles; the class of a leftover is that of the render in which it appears: A for cleared, B for new)

**F212N**

| pair A → B | tiles | cleared: SATSTAR_NEAR | cleared: VETTED_BUT_UNSUBTRACTED | cleared: VETTED_OUT | cleared: NOT_IN_M7_CATALOG | listed as new: SATSTAR_NEAR | listed as new: VETTED_BUT_UNSUBTRACTED | listed as new: VETTED_OUT | listed as new: NOT_IN_M7_CATALOG |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `prod5` → `c7fin` | 4 | 25 | 1 | 0 | 2 | 17 | 2 | 0 | 1 |
| `c7fin` → `e7nm` | 4 | 211 | 78 | 71 | 135 | 2 | 0 | 0 | 0 |
| `c7fin` → `e7fin` | 4 | 315 | 84 | 73 | 150 | 2 | 0 | 0 | 6 |
| `optnm` → `e7nm` | 4 | 2 | 2 | 0 | 5 | 0 | 0 | 0 | 0 |
| `optfin` → `e7fin` | 4 | 0 | 1 | 0 | 5 | 0 | 0 | 0 | 0 |
| `prod5` → `e7fin` | 4 | 321 | 78 | 75 | 155 | 2 | 0 | 0 | 6 |
| `prod5` → `e7nm` | 4 | 230 | 72 | 73 | 140 | 14 | 0 | 0 | 0 |
| `ctlfin` → `c7fin` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**F480M**

| pair A → B | tiles | cleared: SATSTAR_NEAR | cleared: VETTED_BUT_UNSUBTRACTED | cleared: VETTED_OUT | cleared: NOT_IN_M7_CATALOG | listed as new: SATSTAR_NEAR | listed as new: VETTED_BUT_UNSUBTRACTED | listed as new: VETTED_OUT | listed as new: NOT_IN_M7_CATALOG |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `prod5` → `c7fin` | 4 | 26 | 1 | 0 | 15 | 31 | 1 | 0 | 1 |
| `c7fin` → `e7nm` | 4 | 42 | 101 | 11 | 96 | 5 | 0 | 0 | 0 |
| `c7fin` → `e7fin` | 4 | 262 | 107 | 12 | 112 | 0 | 0 | 0 | 0 |
| `optnm` → `e7nm` | 4 | 19 | 4 | 0 | 12 | 1 | 0 | 0 | 0 |
| `optfin` → `e7fin` | 4 | 0 | 3 | 0 | 3 | 0 | 0 | 0 | 0 |
| `prod5` → `e7fin` | 4 | 250 | 107 | 12 | 132 | 0 | 0 | 0 | 0 |
| `prod5` → `e7nm` | 4 | 62 | 102 | 11 | 110 | 31 | 1 | 0 | 0 |
| `ctlfin` → `c7fin` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

The pairs against the four-tile run (`optnm` → `e7nm` and `optfin` → `e7fin`) have 0 worse and 0 new leftovers in F212N and 0 worse and 1 new in F480M in the unmasked pair, and 0 worse and 0 new in F212N and 0 worse and 0 new in F480M in the masked pair. The unmasked pair has 9 cleared leftovers in F212N and 35 in F480M; the craters ≥50 MJy/sr that are gone in that pair number 1 (F212N) and 0 (F480M), and the craters listed as new number 1 and 0. The section "Differences against the four-tile run that the lists show" traces the crater differences of this pair (the o138 F212N crater at the pixel 8695, 623 and the o118 F212N crater at the pixel 11389, 1475) and the leftover that the scan lists in o105 F480M.

## Position-independent scans

Every leftover of the scanned render X with a residual peak ≥200 MJy/sr, and every big crater of X with depth ≥200 MJy/sr, is compared with the extremum of the reference render Y in the 13 × 13 pixel box centred on it (maximum of the residual for a leftover, depth of the minimum for a crater), so that a shift of the extremum by a few pixels does not change the verdict. With S the extremum of the feature in X and R the extremum in Y: *absent in Y* = R < 200 MJy/sr and S > 1.5 R; *larger than in Y* = R ≥ 200 MJy/sr and S > 1.5 R; *within 1.5×* = S and R differ by at most a factor 1.5; *smaller than in Y* = R > 1.5 S. Each scan file holds both directions, so every scan appears as two rows per band and kind: the features of the source against the reference (the regression direction) and the features of the reference against the source (the improvement direction). The table sums over the four tiles.

| features of X | against Y | band | kind | tiles | X features | absent in Y | larger than in Y | within 1.5× | smaller than in Y |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `c7fin` | `prod5` | F212N | leftovers | 4 | 464 | 5 | 1 | 437 | 21 |
| `c7fin` | `prod5` | F212N | craters | 4 | 400 | 14 | 7 | 334 | 45 |
| `prod5` | `c7fin` | F212N | leftovers | 4 | 467 | 6 | 3 | 436 | 22 |
| `prod5` | `c7fin` | F212N | craters | 4 | 443 | 12 | 9 | 373 | 49 |
| `c7fin` | `prod5` | F480M | leftovers | 4 | 354 | 2 | 0 | 330 | 22 |
| `c7fin` | `prod5` | F480M | craters | 4 | 60 | 13 | 2 | 40 | 5 |
| `prod5` | `c7fin` | F480M | leftovers | 4 | 358 | 1 | 9 | 331 | 17 |
| `prod5` | `c7fin` | F480M | craters | 4 | 52 | 2 | 4 | 46 | 0 |
| `e7nm` | `c7fin` | F212N | leftovers | 4 | 81 | 0 | 0 | 59 | 22 |
| `e7nm` | `c7fin` | F212N | craters | 4 | 112 | 0 | 2 | 94 | 16 |
| `c7fin` | `e7nm` | F212N | leftovers | 4 | 464 | 372 | 13 | 60 | 19 |
| `c7fin` | `e7nm` | F212N | craters | 4 | 400 | 202 | 54 | 120 | 24 |
| `e7nm` | `c7fin` | F480M | leftovers | 4 | 231 | 1 | 0 | 219 | 11 |
| `e7nm` | `c7fin` | F480M | craters | 4 | 16 | 0 | 0 | 14 | 2 |
| `c7fin` | `e7nm` | F480M | leftovers | 4 | 354 | 122 | 1 | 218 | 13 |
| `c7fin` | `e7nm` | F480M | craters | 4 | 60 | 28 | 8 | 21 | 3 |
| `e7fin` | `c7fin` | F212N | leftovers | 4 | 11 | 0 | 0 | 9 | 2 |
| `e7fin` | `c7fin` | F212N | craters | 4 | 0 | 0 | 0 | 0 | 0 |
| `c7fin` | `e7fin` | F212N | leftovers | 4 | 464 | 452 | 1 | 11 | 0 |
| `c7fin` | `e7fin` | F212N | craters | 4 | 400 | 400 | 0 | 0 | 0 |
| `e7fin` | `c7fin` | F480M | leftovers | 4 | 1 | 0 | 0 | 1 | 0 |
| `e7fin` | `c7fin` | F480M | craters | 4 | 0 | 0 | 0 | 0 | 0 |
| `c7fin` | `e7fin` | F480M | leftovers | 4 | 354 | 351 | 1 | 2 | 0 |
| `c7fin` | `e7fin` | F480M | craters | 4 | 60 | 60 | 0 | 0 | 0 |
| `e7nm` | `optnm` | F212N | leftovers | 4 | 81 | 0 | 0 | 61 | 20 |
| `e7nm` | `optnm` | F212N | craters | 4 | 112 | 0 | 0 | 104 | 8 |
| `optnm` | `e7nm` | F212N | leftovers | 4 | 86 | 4 | 1 | 61 | 20 |
| `optnm` | `e7nm` | F212N | craters | 4 | 112 | 0 | 0 | 104 | 8 |
| `e7nm` | `optnm` | F480M | leftovers | 4 | 231 | 1 | 0 | 219 | 11 |
| `e7nm` | `optnm` | F480M | craters | 4 | 16 | 0 | 0 | 15 | 1 |
| `optnm` | `e7nm` | F480M | leftovers | 4 | 257 | 23 | 0 | 220 | 14 |
| `optnm` | `e7nm` | F480M | craters | 4 | 16 | 0 | 0 | 15 | 1 |
| `e7fin` | `optfin` | F212N | leftovers | 4 | 11 | 0 | 0 | 10 | 1 |
| `e7fin` | `optfin` | F212N | craters | 4 | 0 | 0 | 0 | 0 | 0 |
| `optfin` | `e7fin` | F212N | leftovers | 4 | 14 | 3 | 0 | 10 | 1 |
| `optfin` | `e7fin` | F212N | craters | 4 | 0 | 0 | 0 | 0 | 0 |
| `e7fin` | `optfin` | F480M | leftovers | 4 | 1 | 0 | 0 | 1 | 0 |
| `e7fin` | `optfin` | F480M | craters | 4 | 0 | 0 | 0 | 0 | 0 |
| `optfin` | `e7fin` | F480M | leftovers | 4 | 2 | 1 | 0 | 1 | 0 |
| `optfin` | `e7fin` | F480M | craters | 4 | 0 | 0 | 0 | 0 | 0 |

Reading the table (every pair of numbers reads F212N / F480M):

- **The opt arm against its control arm** (`e7fin` and `e7nm` against `c7fin`). `e7fin` has 11 / 1 leftovers and 0 / 0 craters ≥200 MJy/sr. Of those leftovers 0 / 0 are absent in `c7fin` and 0 / 0 are larger than in `c7fin`. In the reverse direction `c7fin` has 464 / 354 leftovers ≥200 MJy/sr, of which 452 / 351 are absent in `e7fin`. For the unmasked render, `e7nm` has 81 / 231 leftovers ≥200 MJy/sr with 0 / 1 absent in `c7fin`, and 112 / 16 craters ≥200 MJy/sr with 0 / 0 absent in `c7fin`.
- **This run against the four-tile run** (`e7nm` against `optnm`, `e7fin` against `optfin`; F212N / F480M). `e7nm` has 81 / 231 leftovers ≥200 MJy/sr; 0 / 1 are absent in `optnm` and 0 / 0 are larger than in `optnm`. The reverse direction lists the improvements of this run: `optnm` has 86 / 257 leftovers ≥200 MJy/sr, of which 4 / 23 are absent in `e7nm`. For the masked renders, 0 / 0 leftovers of `e7fin` are absent in `optfin`, and of the 14 / 2 leftovers of `optfin`, 3 / 1 are absent in `e7fin`. The scans find craters ≥200 MJy/sr in `e7fin` and `optfin` as follows: `e7fin` 0 / 0 and `optfin` 0 / 0. The one leftover of `e7nm` that is absent in `optnm` is the F480M leftover of o105 at (5325, 1803) (section "Differences against the four-tile run that the lists show").
- **The control arm against production** (`c7fin` against `prod5`), the code drift on feature level (F212N / F480M). `c7fin` has 464 / 354 leftovers ≥200 MJy/sr, of which 5 / 2 are absent in production and 1 / 0 are larger than in production. Its craters ≥200 MJy/sr number 400 / 60, with 14 / 13 absent in production and 7 / 2 larger than in production.

## Features that motivated the switches

The tables give the residual extrema of the 13 × 13 px box centred on the features that each addition was built for, read from the residual i2d of every render: the box maximum and the box minimum in MJy/sr, written as maximum / minimum. NaN marks a box inside a region that the footprint mask of that render sets to NaN, and n/a marks a render that does not exist for the tile (`ctlfin` exists for o105 only; the controls of the other tiles are `c7fin`). The features are the five features of the unmasked render of the four-tile run that exceed production (features 1 to 5; section "Cause of the five features that exceed production" of its README) and the leftovers of groups 3, 5 and 7 of its section "Cause of each remaining leftover". The column "switch" names the addition that the feature belongs to: "#1360 guard" is the second-pass neighbour guard that the recipe of the four-tile run already contains, and "#1393" is the gate scale that this run does not contain.

| tile | band | switch | feature | pixel (x, y) | `prod5` | `ctlfin` | `c7fin` | `optnm` | `e7nm` | `optfin` | `e7fin` |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| o105 | F480M | #1394 | feature 4, leftover | (2614, 1826) | 29 / -1064 | 917 / -821 | 917 / -821 | 917 / -821 | 33 / -827 | NaN | NaN |
| o105 | F480M | #1394 | feature 5, leftover | (4452, 858) | 30 / -425 | 418 / 6 | 418 / 6 | 418 / 6 | 59 / 3 | 40 / 11 | 28 / 5 |
| o105 | F212N | #1360 guard | feature 1, crater | (8306, 1270) | 2741 / -971 | 2750 / -958 | 2750 / -958 | 190 / -1583 | 190 / -1583 | NaN | NaN |
| o105 | F212N | #1360 guard | feature 2, crater | (4469, 1280) | 1417 / -456 | 1414 / -459 | 1414 / -459 | 92 / -748 | 92 / -748 | NaN | NaN |
| o138 | F212N | #1393 | feature 3, leftover | (8633, 2587) | 1297 / 8 | n/a | 2216 / -475 | 2216 / -475 | 2216 / -475 | NaN | NaN |
| o105 | F212N | keep radius | group 3, leftover | (11052, 3798) | 342 / -12 | 343 / 2 | 343 / 2 | 343 / 2 | 111 / 2 | 343 / 2 | 111 / 2 |
| o105 | F212N | keep radius | group 3, leftover | (1660, 341) | 214 / -5 | 213 / -20 | 213 / -20 | 213 / -20 | 73 / -17 | 213 / -20 | 73 / -17 |
| o105 | F212N | #1395 | group 5, leftover | (9362, 1291) | 209 / 6 | 209 / 6 | 209 / 6 | 209 / 6 | 209 / 6 | 209 / 6 | 209 / 6 |
| o130 | F212N | #1395 | group 5, leftover | (5065, 68) | 1248 / -131 | n/a | 1244 / -138 | 1244 / -138 | 9 / -22 | 1244 / -138 | 9 / -22 |
| o130 | F480M | #1392 | group 7, leftover | (1932, 146) | 363 / 3 | n/a | 363 / 3 | 363 / 3 | 11 / 3 | 363 / 3 | 11 / 3 |

Reading the table:

- **Features 4 and 5 (#1394).** With every switch off the two F480M leftovers of o105 are present at 917 and 418 MJy/sr (`ctlfin`) and in the unmasked opt render of the four-tile run (`optnm`, 917 and 418); production has 29 and 30. `e7nm` has 33 and 59. In the masked renders feature 4 lies inside a masked region and feature 5 has 40 (`optfin`) and 28 (`e7fin`).
- **Features 1 and 2 (the guard of #1360).** The two o105 F212N craters have the box minima -1583 and -748 MJy/sr in `optnm` and -1583 and -748 in `e7nm` (production -971 and -456). The box minima of `e7nm` equal those of `optnm` at both craters. Both craters lie inside masked regions of the masked renders.
- **Feature 3.** The o138 F212N leftover at (8633, 2587) has the box maximum 2216 MJy/sr in the control render `c7fin`, 2216 in `optnm` and 2216 in `e7nm` (production 1297). The value that the control arm has with every opt-in switch off equals the value of both opt renders, so none of the opt-in switches of either run moves this leftover. The leftover lies inside a masked region of the masked renders.
- **Group 3 (the keep radius).** The two o105 F212N leftovers have the box maxima 343 and 213 MJy/sr in `optnm` and 111 and 73 in `e7nm` and the same values in the masked render of the same run. The seed rows behind this change are in the section "Seed rows added by the keep radius".
- **Group 5 (#1395).** The o130 F212N leftover at (5065, 68) has the box maximum 1248 MJy/sr in production and 1244 in `c7fin`, 1244 in `optnm` and 9 in `e7nm`. The o105 F212N leftover at (9362, 1291) has the box maximum 209 MJy/sr in `e7nm` and 209 in production, so the partial-box rule leaves it unchanged. The four-tile README traces this leftover to two components 0.7 FWHM apart inside the radius of the split-row rule of #1365, which is not a partial-box effect.
- **Group 7 (#1392).** The o130 F480M leftover at (1932, 146) has the box maximum 363 MJy/sr in production and 11 in `e7nm`. The table below lists all six leftovers of this group that lie in the tiles of this run.

The leftovers of the masked render of the four-tile run (`optfin`) with a residual peak of at least 100 MJy/sr at a pixel that the spike finder returns at tolerance 1 (the measurement of the documentation of #1392), with the box maximum of the residual (13 × 13 px) and the largest leftover within 3 px in each render, written as box maximum / leftover peak; a leftover peak of 0 means that the render has no leftover of 50 MJy/sr or more within 3 px.

| tile | band | (x, y) | optfin peak | prod5 box max / leftover | optfin box max / leftover | optnm box max / leftover | c7fin box max / leftover | e7nm box max / leftover | e7fin box max / leftover |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| o118 | F480M | (1545, 315) | 159 | 165 / 159 | 165 / 159 | 165 / 159 | 165 / 159 | 10 / 0 | 10 / 0 |
| o118 | F480M | (1155, 485) | 128 | 132 / 128 | 132 / 128 | 132 / 128 | 132 / 128 | 6 / 0 | 6 / 0 |
| o118 | F480M | (3949, 1355) | 149 | 156 / 149 | 156 / 149 | 156 / 149 | 156 / 149 | 8 / 0 | 8 / 0 |
| o118 | F480M | (3500, 1716) | 189 | 195 / 189 | 195 / 189 | 195 / 189 | 195 / 189 | 44 / 0 | 44 / 0 |
| o130 | F480M | (1932, 146) | 359 | 363 / 359 | 363 / 359 | 363 / 359 | 363 / 359 | 11 / 0 | 11 / 0 |
| o130 | F480M | (774, 312) | 160 | 166 / 160 | 166 / 160 | 166 / 160 | 166 / 160 | 13 / 0 | 13 / 0 |

Not covered by this run (tile without arms): o045 F480M (893, 1812), residual peak 116 MJy/sr in `optfin`.

## Differences against the four-tile run that the lists show

The unmasked render of this run (`e7nm`) and the unmasked render of the four-tile run (`optnm`) come from the same environment and differ by the three PRs of this run and the keep radius. Two lists show three differences between them: the scan of `e7nm` against `optnm` (section "Position-independent scans") lists one leftover of o105 F480M that `optnm` lacks, and the crater lists of the pair `optnm` → `e7nm` (section "Matching of the leftovers and craters between renders") hold one o138 F212N crater that `e7nm` lists only and one o118 F212N crater that `optnm` lists only. Each was followed from the residual extrema to the rows of the merged vetted catalogs within 0.6 arcsec of the pixel, with the catalogs of the opt arm of the four-tile run, the opt arm of this run and the control arm of this run side by side, and from the rows that differ to the per-frame fit tables of the m7 fan-out (a fit belongs to a row when it lies within 3 px of the position of the row in the frame). The extra o138 F212N row was followed one step further, to replays of the fit stage of two frames with each fit-time switch alone (the paragraphs from "Cause of the extra row" on).

Residual extrema of the 13 × 13 px box centred on the pixel of each feature, written maximum / minimum in MJy/sr. NaN marks a box inside a region that the footprint mask of that render sets to NaN.

| tile | band | kind | pixel (x, y) | `prod5` | `c7fin` | `optnm` | `e7nm` | `optfin` | `e7fin` |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| o105 | F480M | leftover | (5325, 1803) | 45 / -207 | 42 / -186 | 42 / -186 | 338 / -172 | NaN | NaN |
| o138 | F212N | crater | (8695, 623) | 236 / -247 | 290 / -220 | 290 / -219 | 272 / -220 | NaN | NaN |
| o118 | F212N | crater | (11389, 1475) | 17 / -64 | 17 / -65 | 17 / -65 | 7 / -5 | 6 / 0 | 6 / 0 |

Rows of the merged catalogs within 0.6 arcsec of the pixel that exist in some of the three catalogs only, together with the rows that all three share, nearest first. A dash marks a row that the catalog lacks. "Group size" is the group size of the row, "frames matched" the number of per-frame rows that the merge assigned to it, "saturated" the `is_saturated` flag of the band and "near-saturated flag" the `near_saturated` flag of the band.

| tile | band | pixel (x, y) | row | distance to the pixel (arcsec) | flux in the four-tile opt catalog | flux in the opt catalog of this run | flux in the control catalog of this run | qfit | flags | group size | frames matched | saturated | near-saturated flag |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| o105 | F480M | (5325, 1803) | 1 | 0.035 | 4509 | - | 4509 | 0.014 | 1.00 | 2 | 1 | no | yes |
| o105 | F480M | (5325, 1803) | 2 | 0.310 | 22707 | 22707 | 22707 | 2.842 | 0.00 | 1 | 2 | yes | yes |
| o138 | F212N | (8695, 623) | 1 | 0.323 | - | 8975 | - | 0.484 | 1.00 | 1 | 2 | no | yes |
| o138 | F212N | (8695, 623) | 2 | 0.351 | 3567 | 3567 | 3567 | 0.053 | 0.00 | 5 | 2 | no | no |
| o138 | F212N | (8695, 623) | 3 | 0.462 | 823 | 823 | - | 0.084 | 0.00 | 5 | 1 | no | no |
| o138 | F212N | (8695, 623) | 4 | 0.533 | 6674656 | 6674656 | 6674656 | 0.034 | 1.00 | 1 | 6 | yes | yes |
| o138 | F212N | (8695, 623) | 5 | 0.534 | 3451 | 3451 | 3451 | 0.087 | 0.46 | 5 | 2 | no | no |
| o118 | F212N | (11389, 1475) | 1 | 0.018 | 1426 | - | 1426 | 0.022 | 1.00 | 1 | 2 | no | no |
| o118 | F212N | (11389, 1475) | 2 | 0.044 | - | 1475 | - | 0.022 | 1.00 | 1 | 2 | no | no |
| o118 | F212N | (11389, 1475) | 3 | 0.116 | 447 | 447 | 447 | 0.124 | 0.00 | 1 | 2 | no | no |
| o118 | F212N | (11389, 1475) | 4 | 0.365 | 119 | 119 | 119 | 0.366 | 0.00 | 1 | 2 | no | no |
| o118 | F212N | (11389, 1475) | 5 | 0.475 | 46 | 46 | 46 | 1.045 | 0.00 | 1 | 2 | no | no |
| o118 | F212N | (11389, 1475) | 6 | 0.482 | 57 | 57 | 57 | 0.628 | 0.00 | 1 | 2 | no | no |
| o118 | F212N | (11389, 1475) | 7 | 0.587 | 183 | 183 | 183 | 0.263 | 0.00 | 1 | 2 | no | no |

**The o105 F480M leftover at (5325, 1803) is a cost of #1394.** The box maximum is 45 MJy/sr in production, 42 in the control render `c7fin`, 42 in `optnm` and 338 in `e7nm`. The scan lists a leftover of 327 MJy/sr in `e7nm`, class SATSTAR_NEAR at 0.310 arcsec from a satstar fit, against 42 MJy/sr in the box of `optnm`. The row that the catalog of `optnm` has and the opt catalog of this run lacks (row 1 of the table above) has a flux of 4509 and lies 0.035 arcsec from the pixel and 0.28 arcsec (1.7 FWHM) from the saturated row 2. The control catalog of this run has it as well. It stands on one frame: exposure 2 of the nrcblong detector has a fit of flux 4508.0 with error 9.9, 19 of 25 pixels fitted, qfit 0.014 and reduced chi^2 38.9 in the catalogs of the four-tile run and of the control arm, and the opt catalog of this run has 0 fits there. The distances of the position of the fit in the frame to the nearest SATURATED pixel centre, by the rounded-pixel rule of the default filter and by the continuous rule of #1394 (a fit at 1.0 px or less is dropped):

| tile | band | pixel (x, y) | row that differs | present in | frame | position in the frame (x, y) | kind of position | distance by the rounded-pixel rule (px) | distance by the continuous rule (px) | rounded-pixel rule | continuous rule |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | ---: | ---: | :--- | :--- |
| o105 | F480M | (5325, 1803) | flux 4509, 0.035 arcsec from the pixel | four-tile opt | 105001_02101_00001_nrcblong | (1757.77, 1572.82) | projected position of the row, no fit | 1.000 | 0.794 | drops | drops |
| o105 | F480M | (5325, 1803) | flux 4509, 0.035 arcsec from the pixel | four-tile opt | 105001_02101_00002_nrcblong | (1755.68, 1745.30) | fit position | 1.414 | 0.975 | keeps | drops |

Exposure 1 has no fit in any of the catalogs, so its line of the table is the distance of a hypothetical fit at the projected position of the row: 1.000 px by the rounded-pixel rule and 0.794 px by the continuous rule, which would drop it under either rule. Exposure 2 is at 1.414 px by the rounded-pixel rule, which keeps it, and at 0.975 px by the continuous rule, which drops it. With the one fit gone the row disappears from the catalog, and the model no longer holds the flux of 4509 at the pixel. The two masked renders hide this box. The continuous rule removes this row, and the same rule clears features 4 and 5 (section "Features that motivated the switches"). The measurement does not decide whether the row is a star or a structure in the wing of the saturated neighbour.

**The o138 F212N crater at (8695, 623) is a listing flip at the cut.** The box minimum is -247 MJy/sr in production, -220 in `c7fin`, -219 in `optnm` and -220 in `e7nm`. The crater scan keeps a candidate pixel when the smoothed map of −(residual − background) has a local maximum there (box of 5 px for F212N) and the unsmoothed −(residual − background) at the pixel exceeds the larger of 25 sigma and 20 MJy/sr, with the background and the sigma of the 64 px block that holds the pixel. The residual at the pixel is -124.70 MJy/sr in `optnm` and -124.85 in `e7nm`, and the block sigma is 5.1812 and 5.1605 (a difference of 0.40 %). The cut moves from 129.53 to 129.01 MJy/sr while the value of −(residual − background) at the pixel is 129.12 and 129.25. The margin of the value over the cut is -0.41 MJy/sr in `optnm`, where the crater is not listed, and 0.24 MJy/sr in `e7nm`, where it is listed; both margins are below 0.5 MJy/sr, which is 0.4 % of the cut. The pixel is a local maximum of the smoothed map in all four renders.

| tile | band | pixel (x, y) | listed in | render | residual at the pixel (MJy/sr) | block background | block sigma | cut = max(25 sigma, 20) | -(residual - background) | margin over the cut | local maximum of the smoothed map | above the cut | listed by the scan |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- | :--- |
| o138 | F212N | (8695, 623) | e7nm only | `prod5` | -129.43 | 4.305 | 5.1294 | 128.23 | 133.74 | +5.50 | yes | yes | yes |
| o138 | F212N | (8695, 623) | e7nm only | `c7fin` | -124.72 | 4.402 | 5.1340 | 128.35 | 129.13 | +0.78 | yes | yes | yes |
| o138 | F212N | (8695, 623) | e7nm only | `optnm` | -124.70 | 4.420 | 5.1812 | 129.53 | 129.12 | -0.41 | yes | no | no |
| o138 | F212N | (8695, 623) | e7nm only | `e7nm` | -124.85 | 4.405 | 5.1605 | 129.01 | 129.25 | +0.24 | yes | yes | yes |

The value at the pixel differs by 0.13 MJy/sr between the two renders, so the crater has the same depth in both and the list difference comes from the 0.4 % change of the block sigma. A crater this close to the cut can change its listing with any change of the residual in its 64 px block that moves the robust sigma by 0.4 %. The change of the block sigma follows the model of the render. The 64 px block that holds the pixel when the blocks are aligned at multiples of 64 px spans x 8640 to 8703 and y 576 to 639 of the mosaic; its median and robust sigma are 4.420 and 5.1812 MJy/sr in `optnm` and 4.405 and 5.1605 in `e7nm`. The model of `e7nm` differs from that of `optnm` by more than 0.5 MJy/sr in 258 of the 4096 pixels of the block; the largest difference is 548.3 MJy/sr at (8694, 633), the pixel of the extra row that the next paragraph describes, and the difference at the crater pixel is 0.147 MJy/sr. Adding the model difference of the block to the residual of `e7nm` returns the robust sigma 5.1812 of `optnm`. The data mosaics of the two renders differ by up to 8.715 MJy/sr over the window of 130 × 140 px around the star, and that difference leaves the block sigma unchanged at the five digits shown. The listing flip is a consequence of the extra row.

The opt catalog of this run has one more row than the catalog of the four-tile run within 0.6 arcsec of this pixel (row 1 of the o138 rows: flux 8975 at 0.323 arcsec from the pixel, 0.21 arcsec or 2.9 FWHM from the saturated row 4, group size 1, qfit 0.484, flags 1.00) and the control catalog and the catalog of the four-tile run lack it. It stands on two fits of the nrcb2 detector (per-frame tables at the end of this section): exposure 3 with flux 9469.9, 15 of 25 pixels, qfit 0.364 and reduced chi^2 215.4, and exposure 6 with flux 8464.8, 19 of 25 pixels, qfit 0.564 and reduced chi^2 162.0. The two fluxes differ by 6.33 sigma of the difference. The catalogs of the four-tile run and of the control have no fit within 3 px of these positions in any frame. The box extrema are 290 / -219 MJy/sr in `optnm` and 272 / -220 in `e7nm`, and the pair `optnm` → `e7nm` lists 0 leftovers as new in F212N.

Distances to the nearest SATURATED pixel centre for each frame that covers the extra row. Exposures 3 and 6 hold the two fits of the row and show the distances at the fit positions; the other frames hold no fit of the row, and their lines show the distance of a hypothetical fit at the projected position of the row.

| tile | band | pixel (x, y) | row that differs | present in | frame | position in the frame (x, y) | kind of position | distance by the rounded-pixel rule (px) | distance by the continuous rule (px) | rounded-pixel rule | continuous rule |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | ---: | ---: | :--- | :--- |
| o138 | F212N | (8695, 623) | flux 8975, 0.323 arcsec from the pixel | this-run opt | 138001_02101_00001_nrcb4 | (1522.83, 146.58) | projected position of the row, no fit | 0.000 | 0.451 | drops | drops |
| o138 | F212N | (8695, 623) | flux 8975, 0.323 arcsec from the pixel | this-run opt | 138001_02101_00002_nrcb4 | (1516.24, 488.26) | projected position of the row, no fit | 1.000 | 0.774 | drops | drops |
| o138 | F212N | (8695, 623) | flux 8975, 0.323 arcsec from the pixel | this-run opt | 138001_02101_00003_nrcb2 | (97.54, 493.51) | fit position | 1.000 | 1.560 | drops | keeps |
| o138 | F212N | (8695, 623) | flux 8975, 0.323 arcsec from the pixel | this-run opt | 138001_02101_00004_nrcb2 | (876.15, 485.92) | projected position of the row, no fit | 2.000 | 2.090 | keeps | keeps |
| o138 | F212N | (8695, 623) | flux 8975, 0.323 arcsec from the pixel | this-run opt | 138001_02101_00005_nrcb2 | (876.68, 143.10) | projected position of the row, no fit | 2.000 | 1.929 | keeps | keeps |
| o138 | F212N | (8695, 623) | flux 8975, 0.323 arcsec from the pixel | this-run opt | 138001_02101_00006_nrcb2 | (100.65, 150.69) | fit position | 1.000 | 1.360 | drops | keeps |

**Cause of the extra row: #1394 alone.** The replays ran the m7 fit stage of one frame at a time, for exposures 3 and 6 of the nrcb2 detector (the two frames that hold the fits of the row), with the environment of the four-tile run and the code tree of this run. The fit-time switches were on in five combinations, the arms a0 to a4 of the table below; the ten replays ended with exit code 0:0. Each replay wrote the per-frame table of its frame, which was compared row by row (0.05 px) with the per-frame table of the opt arm of the four-tile run and with that of the opt arm of this run. "Fits at the target positions" asks for a row of the replay table within 0.3 px of the position of each fit in the opt table of this run. The three columns that follow the arm give the rows only in the replay, the rows only in the table and the matched rows whose flux differs by more than 1e-3 (relative).

| arm | switches on | exposure | rows in the table | fits at the target positions | against the four-tile opt table: rows only in the replay / only in the table / matched rows with a flux change above 1e-3 (relative) | largest relative flux change against the four-tile opt table | against the opt table of this run: the same three counts | largest relative flux change against the opt table of this run |
| :--- | :--- | :--- | ---: | :--- | :--- | ---: | :--- | ---: |
| a0 | none | 3 | 40861 | (97.54, 493.51) absent | 0 / 0 / 0 | 0 | 0 / 4 / 1 | 0.0344 |
| a0 | none | 6 | 41136 | (100.65, 150.69) absent; (100.71, 165.26) absent | 0 / 0 / 0 | 0 | 0 / 3 / 0 | 0 |
| a1 | #1392 | 3 | 40861 | (97.54, 493.51) absent | 0 / 0 / 0 | 0 | 0 / 4 / 1 | 0.0344 |
| a1 | #1392 | 6 | 41136 | (100.65, 150.69) absent; (100.71, 165.26) absent | 0 / 0 / 0 | 0 | 0 / 3 / 0 | 0 |
| a2 | #1394 | 3 | 40865 | (97.54, 493.51) present | 4 / 0 / 1 | 0.0333 | 0 / 0 / 0 | 0 |
| a2 | #1394 | 6 | 41139 | (100.65, 150.69) present; (100.71, 165.26) present | 3 / 0 / 0 | 0 | 0 / 0 / 0 | 0 |
| a3 | #1392 + #1394 | 3 | 40865 | (97.54, 493.51) present | 4 / 0 / 1 | 0.0333 | 0 / 0 / 0 | 0 |
| a3 | #1392 + #1394 | 6 | 41139 | (100.65, 150.69) present; (100.71, 165.26) present | 3 / 0 / 0 | 0 | 0 / 0 / 0 | 0 |
| a4 | #1392 + #1394 + #1395 | 3 | 40865 | (97.54, 493.51) present | 4 / 0 / 1 | 0.0333 | 0 / 0 / 0 | 0 |
| a4 | #1392 + #1394 + #1395 | 6 | 41139 | (100.65, 150.69) present; (100.71, 165.26) present | 3 / 0 / 0 | 0 | 0 / 0 / 0 | 0 |

- Arms a0 and a1, which lack #1394, reproduce the per-frame table of the four-tile opt arm row for row (40861 rows in exposure 3, 41136 in exposure 6) and have none of the fits at the target positions. #1392 changes nothing in them.
- Arms a2, a3 and a4, which have #1394, reproduce the per-frame table of the opt arm of this run (40865 and 41139 rows, largest relative flux difference 0) and hold the fits of the next table; the values of each fit are the same in the three arms.
- #1394 adds 4 rows to the table of exposure 3 and 3 rows to the table of exposure 6, and it changes the flux of one matched row of exposure 3 by 3.3 % (largest relative flux change 0.0333). Two of the added rows of exposure 6 are the fits at the target positions of the next table; the second of them lies on the other side of the saturated star (7.7 px above it, against 6.9 px below it for the fit of the extra row).

| exposure | fit at | arms that have the fit | x_fit | y_fit | flux | flux error | pixels fitted (of 25) | qfit | reduced chi^2 | flags |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | (97.54, 493.51) | a2, a3, a4 | 97.540 | 493.510 | 9469.88 | 123.207 | 15 | 0.36379 | 215.37 | 1 |
| 6 | (100.65, 150.69) | a2, a3, a4 | 100.647 | 150.687 | 8464.84 | 99.982 | 19 | 0.56391 | 161.98 | 1 |
| 6 | (100.71, 165.26) | a2, a3, a4 | 100.710 | 165.258 | 6718.73 | 75.348 | 11 | 0.35976 | 290.75 | 1 |

The spike finder returns 0 spikes in both frames at the tolerance 0 and at the tolerance 1 of #1392 (minimum excess 300 MJy/sr), so the spike mask of #1392 is empty in them, and the arms with and without #1392 give the same tables (a1 equals a0, a3 equals a2, a4 equals a3). By reading of the code, the function that builds the spike mask has two call sites, the resampling of the data mosaic and the DO_NOT_USE bits of the per-frame residual render, and the per-frame fit does not call it. #1395 acts where the per-frame rows are merged, which the one-frame replays do not run. The two fits cover 15 and 19 of the 25 pixels of the fit box, fractions of 0.60 and 0.76, above the floor of 0.5 of its partial-box rule, so that rule excludes neither fit; no finalize rerun tested this argument.

**Mechanism.** The pixel directly above the rounded pixel of each fit is SATURATED. The fit of exposure 3 at (97.54, 493.51) rounds to the pixel (98, 494) and the SATURATED pixel is (98, 495); the fit of exposure 6 at (100.65, 150.69) rounds to (101, 151) and the SATURATED pixel is (101, 152). The rounded-pixel rule of the default filter reads 1.000 px in both frames and drops a fit at 1.0 px or less. The continuous rule of #1394 reads the fitted position and the SATURATED pixel centre, gives 1.560 px (exposure 3) and 1.360 px (exposure 6) and keeps both fits (table above, lines of exposures 3 and 6). Over the whole frame the continuous rule keeps 4 fits that the rounded-pixel rule drops and drops none that the rounded-pixel rule keeps: the filter drops 16 fits against 20 in exposure 3 and 19 against 23 in exposure 6.

**Merged row and seed.** The two fits have fluxes 9469.88 (error 123.207) and 8464.84 (error 99.982). The difference of 1005.03 is 6.33 times the quadrature sum 158.67 of the errors. The merged row of the m7 catalog of this run sits at (8694.14, 633.31) with flux 8975.1, error 110.57, qfit 0.4845, flags 1 and 2 matched frames, 0.323 arcsec from the crater pixel. Its flux is 1.0125 times the inverse-variance mean 8863.9 of the two fits; the origin of that factor is not identified here. The m7 seed of every arm (opt arm of this run, opt arm of the four-tile run, control arm of this run) holds the same own-band m6 row (flux 8828.75) within 0.014 arcsec of each of the two fits, so the arms start the fit from the same seed, and the nearest seed row that the keep radius adds lies 21.0 arcsec away.

**Production has a row at this position.** The m6 catalog of production and the m6 catalog of the opt arm of this run hold the row (8694.32, 633.54) with flux 8828.7 and 2 matched frames. The m7 catalog of production holds (8694.32, 633.55) with flux 8828.3, which is 9.4 mas or 0.30 px from the extra row; it stands on the production fits of exposure 5 at (877.10, 143.39) with flux 9274.7 (20 of 25 pixels, qfit 0.465) and of exposure 6 at (100.35, 151.08) with flux 8300.8 (19 of 25 pixels, qfit 0.522), and its flux equals the inverse-variance mean of the two (8828.2 from the rounded inputs). The residual, the model and the data at the pixel (8694, 633) in MJy/sr are −58.8, 871.6 and 812.8 in production, 608.3, 204.5 and 812.8 in the control render `c7fin` and in `optnm`, and 59.9, 752.8 and 812.8 in `e7nm`. The control run and the four-tile run lose a row that production has, and #1394 returns a row at the same place. The opt table of this run has no fit near the position in exposure 5; the replays do not cover that exposure, and a replay of exposure 5 with and without #1394 would show whether the fit exists before the filter.

**Star or structure.** The extra row lies 6.74 px (0.210 arcsec, 2.88 FWHM of 2.34 px) from the saturated star row of the merged catalog, which has flux 6.6747e6 at (8694.24, 640.04). In the frames the fits lie 6.45 px (2.80 FWHM) from the saturated star of exposure 3 (flux 7.298e6) and 6.86 px (2.98 FWHM) from that of exposure 6 (flux 7.514e6); their fluxes are 0.130 % and 0.113 % of the star flux. Eleven fits with positive flux at 1.5 to 4.5 FWHM from the 12 saturated stars of flux 1e6 or more in exposures 3 and 6 of the opt tables lie within 9° of a multiple of 60° of the angle measured from the detector +y axis (this row at 185° in exposure 3 and 180° in exposure 6); for uniform angles a single row has the probability 0.30 to lie within 9° of a multiple of 60°. The pattern was found after the rows were listed, rows on opposite sides of one star are not independent, and the angles were not compared with a model of the PSF, so the count describes these 11 rows and carries no significance. The data minus the satstar model show a positive pedestal at 6 to 8 px from the star at every azimuth: the medians of the 30° azimuth bins are 341 to 759 MJy/sr (exposure 3) and 226 to 784 MJy/sr (exposure 6), and the 3 × 3 peak at the fit is 1184 and 828 MJy/sr, above every bin median at that radius. The measurements are consistent with a structure of the PSF at this radius and with a faint star on it; no test of this study separates the two.

**The o118 F212N crater at (11389, 1475) is cleared by #1395.** The box minimum is -64 MJy/sr in production, -65 in `c7fin`, -65 in `optnm` and -5 in `e7nm`; the residual at the pixel is -65.45 MJy/sr in `optnm` and 3.09 in `e7nm`, a difference of 68.54 MJy/sr. The value of −(residual − background) at the pixel is 69.91 MJy/sr against a cut of 39.72 in `optnm` and 1.38 against 39.46 in `e7nm`, so the crater is far from the cut in both renders and the cut does not decide it.

| tile | band | pixel (x, y) | listed in | render | residual at the pixel (MJy/sr) | block background | block sigma | cut = max(25 sigma, 20) | -(residual - background) | margin over the cut | local maximum of the smoothed map | above the cut | listed by the scan |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- | :--- |
| o118 | F212N | (11389, 1475) | optnm only | `prod5` | -63.96 | 4.485 | 1.6075 | 40.19 | 68.45 | +28.26 | yes | yes | yes |
| o118 | F212N | (11389, 1475) | optnm only | `c7fin` | -65.45 | 4.465 | 1.5890 | 39.72 | 69.91 | +30.19 | yes | yes | yes |
| o118 | F212N | (11389, 1475) | optnm only | `optnm` | -65.45 | 4.465 | 1.5890 | 39.72 | 69.91 | +30.19 | yes | yes | yes |
| o118 | F212N | (11389, 1475) | optnm only | `e7nm` | 3.09 | 4.470 | 1.5784 | 39.46 | 1.38 | -38.08 | no | no | no |

The merged row of the star changes from flux 1426 at 0.018 arcsec from the pixel (four-tile opt and control) to flux 1475 at 0.044 arcsec (this-run opt), a shift of 0.03 arcsec, about one mosaic pixel (0.031 arcsec), with a flux that is 3.4 % higher. The model of this star changes in position and in flux. The per-frame fits of the star are the same in all three catalogs:

- exposure 1: flux 1457.1 with error 55.9, 15 of 25 pixels fitted, qfit 0.022;
- exposure 2: flux 340.2 with error 262.3, 5 of 25 pixels fitted, qfit 0.026.

The two fluxes differ by 4.16 sigma of the difference, above the ZMAX of 3, and the five-pixel fit of exposure 2 is below the fraction of 0.5 of the 25 pixels, while exposure 1 with 15 pixels is above it. The rule therefore removes exposure 2 from the averages of the star. The merged table shows the signature of the rule (table below): the scatter of the frame positions is NaN, the flux error equals the propagated error and `nmatch_good` stays at 2. The merged flux of the four-tile catalog equals 1.01255 times the inverse-variance mean of the two fits (1408.5), and the merged flux of this run equals 1.01255 times the flux of exposure 1 alone (1457.1); the propagated errors carry the same factor (1.01255 and 1.01255). The merged table applies this constant to the per-frame fluxes of the row; its origin is not identified here.

Merged-table columns of the rows that exist in one of the two opt catalogs only (a single frame, or a rule of #1395 at work, leaves the scatter columns NaN):

| tile | band | pixel (x, y) | catalog | row (flux, distance to the pixel) | frames matched | frames kept (`nmatch_good`) | scatter of the frame positions in RA (deg) | in Dec (deg) | flux error | propagated flux error |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| o105 | F480M | (5325, 1803) | four-tile opt | 4509, 0.035 arcsec | 1 | 1 | nan | nan | 9.93 | 9.93 |
| o105 | F480M | (5325, 1803) | this-run control | 4509, 0.035 arcsec | 1 | 1 | nan | nan | 9.93 | 9.93 |
| o138 | F212N | (8695, 623) | this-run opt | 8975, 0.323 arcsec | 2 | 2 | 2.19e-06 | 2.30e-06 | 110.57 | 78.61 |
| o118 | F212N | (11389, 1475) | four-tile opt | 1426, 0.018 arcsec | 2 | 2 | 6.08e-07 | 7.42e-06 | 65.73 | 55.39 |
| o118 | F212N | (11389, 1475) | this-run control | 1426, 0.018 arcsec | 2 | 2 | 6.08e-07 | 7.42e-06 | 65.73 | 55.39 |
| o118 | F212N | (11389, 1475) | this-run opt | 1475, 0.044 arcsec | 2 | 2 | nan | nan | 56.64 | 56.64 |

**Per-frame fits of the rows that differ.** Fits of the per-frame tables within 3 px of the position of each row in each frame that covers it; "fits within 3 px" is the number of such fits, and a dash marks a frame without one.

| tile | band | merged row (flux, distance to the pixel) | frame | catalog | fits within 3 px | fitted position (x, y) | flux | flux error | pixels fitted of 25 | qfit | reduced chi^2 | flags |
| :--- | :--- | :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| o105 | F480M | 4509, 0.035 arcsec | 105001_02101_00001_nrcblong | four-tile opt | 0 | - | - | - | - | - | - | - |
| o105 | F480M | 4509, 0.035 arcsec | 105001_02101_00001_nrcblong | this-run opt | 0 | - | - | - | - | - | - | - |
| o105 | F480M | 4509, 0.035 arcsec | 105001_02101_00001_nrcblong | this-run control | 0 | - | - | - | - | - | - | - |
| o105 | F480M | 4509, 0.035 arcsec | 105001_02101_00002_nrcblong | four-tile opt | 1 | (1755.68, 1745.30) | 4508.0 | 9.93 | 19 | 0.014 | 38.88 | 1 |
| o105 | F480M | 4509, 0.035 arcsec | 105001_02101_00002_nrcblong | this-run opt | 0 | - | - | - | - | - | - | - |
| o105 | F480M | 4509, 0.035 arcsec | 105001_02101_00002_nrcblong | this-run control | 1 | (1755.68, 1745.30) | 4508.0 | 9.93 | 19 | 0.014 | 38.88 | 1 |

| tile | band | merged row (flux, distance to the pixel) | frame | catalog | fits within 3 px | fitted position (x, y) | flux | flux error | pixels fitted of 25 | qfit | reduced chi^2 | flags |
| :--- | :--- | :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| o118 | F212N | 1475, 0.044 arcsec | 118001_02101_00001_nrcb2 | four-tile opt | 1 | (2043.36, 931.19) | 1457.1 | 55.94 | 15 | 0.022 | 0.81 | 1 |
| o118 | F212N | 1475, 0.044 arcsec | 118001_02101_00001_nrcb2 | this-run opt | 1 | (2043.36, 931.19) | 1457.1 | 55.94 | 15 | 0.022 | 0.81 | 1 |
| o118 | F212N | 1475, 0.044 arcsec | 118001_02101_00001_nrcb2 | this-run control | 1 | (2043.36, 931.19) | 1457.1 | 55.94 | 15 | 0.022 | 0.81 | 1 |
| o118 | F212N | 1475, 0.044 arcsec | 118001_02101_00002_nrcb2 | four-tile opt | 1 | (2042.94, 1277.11) | 340.2 | 262.31 | 5 | 0.026 | 1.35 | 1 |
| o118 | F212N | 1475, 0.044 arcsec | 118001_02101_00002_nrcb2 | this-run opt | 1 | (2042.94, 1277.11) | 340.2 | 262.31 | 5 | 0.026 | 1.35 | 1 |
| o118 | F212N | 1475, 0.044 arcsec | 118001_02101_00002_nrcb2 | this-run control | 1 | (2042.94, 1277.11) | 340.2 | 262.31 | 5 | 0.026 | 1.35 | 1 |

| tile | band | merged row (flux, distance to the pixel) | frame | catalog | fits within 3 px | fitted position (x, y) | flux | flux error | pixels fitted of 25 | qfit | reduced chi^2 | flags |
| :--- | :--- | :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| o138 | F212N | 8975, 0.323 arcsec | 138001_02101_00001_nrcb4 | four-tile opt | 0 | - | - | - | - | - | - | - |
| o138 | F212N | 8975, 0.323 arcsec | 138001_02101_00001_nrcb4 | this-run opt | 0 | - | - | - | - | - | - | - |
| o138 | F212N | 8975, 0.323 arcsec | 138001_02101_00001_nrcb4 | this-run control | 0 | - | - | - | - | - | - | - |
| o138 | F212N | 8975, 0.323 arcsec | 138001_02101_00002_nrcb4 | four-tile opt | 0 | - | - | - | - | - | - | - |
| o138 | F212N | 8975, 0.323 arcsec | 138001_02101_00002_nrcb4 | this-run opt | 0 | - | - | - | - | - | - | - |
| o138 | F212N | 8975, 0.323 arcsec | 138001_02101_00002_nrcb4 | this-run control | 0 | - | - | - | - | - | - | - |
| o138 | F212N | 8975, 0.323 arcsec | 138001_02101_00003_nrcb2 | four-tile opt | 0 | - | - | - | - | - | - | - |
| o138 | F212N | 8975, 0.323 arcsec | 138001_02101_00003_nrcb2 | this-run opt | 1 | (97.54, 493.51) | 9469.9 | 123.21 | 15 | 0.364 | 215.37 | 1 |
| o138 | F212N | 8975, 0.323 arcsec | 138001_02101_00003_nrcb2 | this-run control | 0 | - | - | - | - | - | - | - |
| o138 | F212N | 8975, 0.323 arcsec | 138001_02101_00004_nrcb2 | four-tile opt | 0 | - | - | - | - | - | - | - |
| o138 | F212N | 8975, 0.323 arcsec | 138001_02101_00004_nrcb2 | this-run opt | 0 | - | - | - | - | - | - | - |
| o138 | F212N | 8975, 0.323 arcsec | 138001_02101_00004_nrcb2 | this-run control | 0 | - | - | - | - | - | - | - |
| o138 | F212N | 8975, 0.323 arcsec | 138001_02101_00005_nrcb2 | four-tile opt | 0 | - | - | - | - | - | - | - |
| o138 | F212N | 8975, 0.323 arcsec | 138001_02101_00005_nrcb2 | this-run opt | 0 | - | - | - | - | - | - | - |
| o138 | F212N | 8975, 0.323 arcsec | 138001_02101_00005_nrcb2 | this-run control | 0 | - | - | - | - | - | - | - |
| o138 | F212N | 8975, 0.323 arcsec | 138001_02101_00006_nrcb2 | four-tile opt | 0 | - | - | - | - | - | - | - |
| o138 | F212N | 8975, 0.323 arcsec | 138001_02101_00006_nrcb2 | this-run opt | 1 | (100.65, 150.69) | 8464.8 | 99.98 | 19 | 0.564 | 161.98 | 1 |
| o138 | F212N | 8975, 0.323 arcsec | 138001_02101_00006_nrcb2 | this-run control | 0 | - | - | - | - | - | - | - |

## Seed rows added by the keep radius

The far-companion exemption of #1377 keeps an own-band m6 source in the m7 seed when the companion cut would remove it, every dominating source lies at least the keep radius away and is unsaturated. The recipe applies the exemption in F212N only (keep S/N 50 in F212N and 0 in F480M). The opt arm of this run uses a keep radius of 1.5 FWHM and the opt arm of the four-tile run used 1.75 FWHM. A smaller radius lets more sources pass, so the seed of this run is expected to contain the seed of the four-tile run plus some rows. Both runs build the seed with the same code (no hunk of the difference between the two integration trees lies in the function that builds the seed of a band), from the same m6 catalogs and with the same other seed options, so a difference between the two seeds comes from the keep radius.

The table compares the seed tables of the opt arm of this run with those of the four-tile run, per tile, band and module. A row is added when the seed of this run has a row with no row of the other seed within 1 mas, and dropped in the reverse case.

| tile | band | module | rows in the seed of the four-tile run | rows in the seed of this run | added | dropped |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| o105 | F212N | nrca | 325110 | 325113 | 3 | 0 |
| o105 | F212N | nrcb | 344056 | 344064 | 8 | 0 |
| o105 | F212N | merged | 579338 | 579348 | 10 | 0 |
| o105 | F480M | nrca | 121480 | 121480 | 0 | 0 |
| o105 | F480M | nrcb | 119841 | 119841 | 0 | 0 |
| o105 | F480M | merged | 163849 | 163849 | 0 | 0 |
| o118 | F212N | nrca | 187609 | 187609 | 0 | 0 |
| o118 | F212N | nrcb | 185631 | 185631 | 0 | 0 |
| o118 | F212N | merged | 309294 | 309294 | 0 | 0 |
| o118 | F480M | nrca | 109544 | 109544 | 0 | 0 |
| o118 | F480M | nrcb | 111313 | 111313 | 0 | 0 |
| o118 | F480M | merged | 161027 | 161027 | 0 | 0 |
| o130 | F212N | nrca | 159263 | 159263 | 0 | 0 |
| o130 | F212N | nrcb | 160219 | 160220 | 1 | 0 |
| o130 | F212N | merged | 259706 | 259707 | 1 | 0 |
| o130 | F480M | nrca | 109605 | 109605 | 0 | 0 |
| o130 | F480M | nrcb | 108812 | 108812 | 0 | 0 |
| o130 | F480M | merged | 161863 | 161863 | 0 | 0 |
| o138 | F212N | nrca | 302242 | 302245 | 3 | 0 |
| o138 | F212N | nrcb | 298458 | 298459 | 1 | 0 |
| o138 | F212N | merged | 503660 | 503664 | 4 | 0 |
| o138 | F480M | nrca | 135752 | 135752 | 0 | 0 |
| o138 | F480M | nrcb | 134364 | 134364 | 0 | 0 |
| o138 | F480M | merged | 181652 | 181652 | 0 | 0 |
| all 4 | F212N | nrca | | | 6 | 0 |
| all 4 | F212N | nrcb | | | 10 | 0 |
| all 4 | F212N | merged | | | 15 | 0 |
| all 4 | F480M | nrca | | | 0 | 0 |
| all 4 | F480M | nrcb | | | 0 | 0 |
| all 4 | F480M | merged | | | 0 | 0 |

The seeds differ in F212N only: 15 merged-module rows are added in F212N (10 in o105, 0 in o118, 1 in o130 and 4 in o138), 0 are dropped, and the F480M seeds are identical (0 added, 0 dropped), as expected for a band with keep S/N 0. The next table follows each added merged-module row of F212N to the residual mosaics of the unmasked renders of the two runs (`optnm` and `e7nm`) and of the masked renders (`optfin` and `e7fin`). The columns give the flux and the origin of the row in the seed, the distance to the nearest seed row of the four-tile run, the verdict for the unmasked pair, the maximum and the minimum of the residual within 3 px of the row in MJy/sr (four-tile run → this run), the residual peak of the whole-star leftover within 3 px (none = no leftover of 50 MJy/sr or more) and the verdict for the masked pair. The verdicts are:

- *cleared*: the earlier render has a leftover of 50 MJy/sr or more within 3 px of the row, and the leftover of this render is below 50 MJy/sr or below half the earlier peak.
- *false seed*: the earlier render has no leftover of 50 MJy/sr or more and no big crater within 3 px of the row, and this render has a big crater there.
- *new left*: the earlier render has no leftover of 50 MJy/sr or more within 3 px, and this render has one.
- *masked*: the row lies in pixels that the footprint mask sets to NaN in the new render.
- *outside*: one of the two renders has no pixel at the row.
- *other*: none of these.

| tile | pixel (x, y) | seed flux | origin | nearest seed row of the four-tile run (arcsec) | verdict, unmasked | residual max, four-tile run → this run | residual min | leftover peak | verdict, masked |
| :--- | :--- | ---: | :--- | ---: | :--- | :--- | :--- | :--- | :--- |
| o105 | (10953, 1214) | 1315 | own_m6 | 0.110 | cleared | 130 → 21 | -8 → 5 | 123 → none | cleared |
| o105 | (10892, 1557) | 1375 | own_m6 | 0.110 | other | 111 → 46 | -32 → -12 | none → none | other |
| o105 | (9850, 1051) | 1965 | own_m6 | 0.113 | other | 12 → 11 | -13 → -12 | none → none | other |
| o105 | (11052, 3798) | 3527 | own_m6 | 0.117 | cleared | 343 → 73 | 18 → 8 | 336 → none | cleared |
| o105 | (10996, 2943) | 3283 | own_m6 | 0.113 | other | 73 → 84 | 1 → 2 | none → none | other |
| o105 | (6092, 4248) | 2454 | own_m6 | 0.124 | other | 245 → 34 | 8 → 8 | none → none | other |
| o105 | (2829, 2221) | 1621 | own_m6 | 0.113 | other | 181 → 45 | -64 → -26 | none → none | other |
| o105 | (1660, 341) | 2197 | own_m6 | 0.124 | cleared | 213 → 25 | -1 → 2 | 205 → none | cleared |
| o105 | (3194, 2974) | 1149 | own_m6 | 0.123 | other | 120 → 19 | 6 → 5 | none → none | other |
| o105 | (5858, 4460) | 1947 | own_m6 | 0.119 | other | 66 → 40 | 6 → 4 | none → none | other |
| o130 | (10635, 3094) | 1557 | own_m6 | 0.108 | cleared | 159 → 41 | 2 → 2 | 155 → none | cleared |
| o138 | (804, 1554) | 2308 | own_m6 | 0.116 | other | 7 → 7 | -66 → -76 | none → none | masked |
| o138 | (306, 1399) | 1798 | own_m6 | 0.111 | other | 211 → 37 | -58 → -31 | none → none | other |
| o138 | (5338, 1653) | 1234 | own_m6 | 0.124 | cleared | 110 → 13 | -10 → -1 | 106 → none | cleared |
| o138 | (8093, 333) | 1455 | own_m6 | 0.109 | other | 144 → 6 | -39 → -11 | none → none | other |

- F212N added merged-module rows: 15. Verdicts of the unmasked pair: cleared 5, false seed 0, new left 0, other 10, masked 0, outside 0. Verdicts of the masked pair: cleared 5, false seed 0, new left 0, other 9, masked 1, outside 0.
- Origin of the added rows: own_m6 15. Seed flux of the added rows: 1149 to 3527. Distance to the nearest seed row of the four-tile run: 0.108 to 0.124 arcsec.
- Unmasked comparison of the 15 added rows: the residual maximum within 3 px falls below half of its value in the four-tile render for 11 rows and rises for 1 row (largest rise 11 MJy/sr). The residual minimum within 3 px is deeper for 5 rows (largest deepening 10 MJy/sr). The deepest minimum within 3 px of an added row in the new render is -76 MJy/sr.

Every added row is an own-band m6 row that the exemption keeps, and each lies near a seed row of the four-tile run (the distance column). The verdict is cleared for 5 of the 15 rows in the unmasked pair and for 5 in the masked pair, and each of these rows removes a whole-star leftover of the four-tile render. The verdicts false seed and new left occur 0 and 0 times in the unmasked pair and 0 and 0 times in the masked pair. The table follows the added rows to the residual and does not check that they are real stars; a row can lower the residual by absorbing the light of an unresolved companion.

## Control arms

The control arm of each tile ran the integration tree with every opt-in switch off, the four base options of the production m7 run, no seed option and no footprint mask (section "Recipe"). Its render is `c7fin`. Three comparisons use it.

**Against production** (`prod5` → `c7fin`; sums over the four tiles, the rows of each tile are in the section "Matching of the leftovers and craters between renders"). The matching of leftovers ≥100 MJy/sr clears 28 of 702 in F212N and 42 of 518 in F480M, has 0 (F212N) and 0 (F480M) worse and lists 20 (F212N) and 33 (F480M) as new. The leftovers ≥100 MJy/sr number 702 → 695 (F212N) and 518 → 510 (F480M). The craters ≥50 MJy/sr number 1382 → 1299 (F212N) and 163 → 193 (F480M), with 56 gone and 86 listed as new in F480M. These numbers are the drift of the integration tree against production with every switch off, which includes the commits between the production tree and the integration base (74 in the four-tile run, whose tree this one extends) and the differences of the PRs that the tree holds.

**Against the opt arm** (`c7fin` → `e7nm` for the fit-time switches and the seed options, `c7fin` → `e7fin` with the masks). The unmasked pair clears 495 of 695 leftovers ≥100 MJy/sr in F212N and 250 of 510 in F480M, with 0 (F212N) and 0 (F480M) worse and 2 (F212N) and 5 (F480M) listed as new. The masked pair clears 622 of 695 in F212N and 493 of 510 in F480M, with 0 (F212N) and 0 (F480M) worse and 8 (F212N) and 0 (F480M) listed as new. The scans of the section "Position-independent scans" give the larger features.

**Against the control arm of the four-tile run** (o105 only, `ctlfin` → `c7fin`; code trees 42655da6 and ac6393dc, every switch off in both). The leftovers ≥100 MJy/sr number 315 against 315 in F212N and 200 against 200 in F480M. The craters ≥50 MJy/sr number 454 against 454 in F212N and 92 against 92 in F480M. The matching of leftovers clears 0 (F212N) and 0 (F480M) and lists 0 (F212N) and 0 (F480M) as new. With their switches off, the three PRs of this run leave these counts of the o105 control render unchanged. The saved environment of every control arm holds none of the opt-in variables (section "Recipe").

The control arm has every switch off, so it measures the drift of the code that runs without switches. A render with one switch on and the others off does not exist, and the code paths that only a switch reaches are outside the control arm.

## Remaining leftovers of the masked render

The masked render `e7fin` has 11 whole-star leftovers with a residual peak of at least 200 MJy/sr in F212N and 1 in F480M. The masked render of the four-tile run (`optfin`) has 14 in F212N and 2 in F480M. The table lists the leftovers of `e7fin`. The residual, data and model peaks are the quantities defined under "How the counts are made". The two distances are those from the leftover to the nearest vetted m7 row and to the nearest consolidated satstar fit of the run. The column "group" gives the group of the section "Cause of each remaining leftover" of the four-tile README. The last three columns give the residual peak of the leftover within 3 px in `optfin`, `c7fin` and `prod5` (- = no leftover of 50 MJy/sr or more there, n/a = the tile has no render of that tag).

| tile | band | pixel (x, y) | residual peak | data peak | model peak | class | nearest vetted row (arcsec) | nearest satstar fit (arcsec) | group in the four-tile run | `optfin` | `c7fin` | `prod5` |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- | ---: | ---: | :--- | ---: | ---: | ---: |
| o105 | F212N | (434, 3684) | 358 | 495 | 439 | SATSTAR_NEAR | 0.077 | 0.797 | 1 | 358 | 358 | 358 |
| o105 | F212N | (5368, 4587) | 345 | 345 | 0 | NOT_IN_M7_CATALOG | 0.444 | 1.208 | 2 | 345 | 344 | 345 |
| o105 | F212N | (7030, 3275) | 254 | 259 | 8 | SATSTAR_NEAR | 0.349 | 0.808 | 4 | 254 | 255 | 255 |
| o105 | F212N | (6529, 3558) | 245 | 246 | 1 | NOT_IN_M7_CATALOG | 0.206 | 1.930 | 4 | 245 | 245 | 245 |
| o105 | F212N | (3245, 3878) | 209 | 285 | 135 | SATSTAR_NEAR | 0.091 | 0.783 | 1 | 209 | 209 | 212 |
| o105 | F212N | (9362, 1291) | 201 | 251 | 105 | VETTED_BUT_UNSUBTRACTED | 0.044 | 1.693 | 5 | 201 | 201 | 202 |
| o118 | F212N | (330, 111) | 939 | 939 | 2 | SATSTAR_NEAR | 0.245 | 0.657 | 2 | 939 | 939 | 939 |
| o118 | F212N | (220, 4407) | 423 | 471 | 142 | SATSTAR_NEAR | 0.097 | 0.864 | 1 | 423 | 423 | 423 |
| o118 | F212N | (637, 3732) | 308 | 309 | 1 | NOT_IN_M7_CATALOG | 0.342 | 1.329 | 6 | 308 | - | - |
| o118 | F212N | (641, 3740) | 203 | 206 | 6 | NOT_IN_M7_CATALOG | 0.293 | 1.132 | 6 | 203 | - | - |
| o118 | F480M | (368, 334) | 253 | 357 | 164 | NOT_IN_M7_CATALOG | 0.192 | 9.235 | 1 | 253 | 253 | 253 |
| o138 | F212N | (7366, 4240) | 287 | 357 | 243 | NOT_IN_M7_CATALOG | 0.082 | 1.401 | 1 | 287 | 287 | 287 |

Four leftovers of the four-tile table are absent from this table. They are the two o105 F212N leftovers of its group 3 at (11052, 3798) and (1660, 341) (section "Seed rows added by the keep radius"), the o130 F212N leftover of its group 5 at (5065, 68) and the o130 F480M leftover of its group 7 at (1932, 146) (section "Features that motivated the switches"). Each of the leftovers in the table has the same residual peak in `optfin` and in `e7fin` (the column `optfin`), so the traces of the four-tile README describe them. This includes the o105 F212N leftover at (9362, 1291) of its group 5, whose residual box maximum in the unmasked renders is listed in the section "Features that motivated the switches".

Leftovers of 100 MJy/sr and more, by class, summed over the four tiles:

| band | class | `optfin` of the four-tile run | `e7fin` |
| :--- | :--- | ---: | ---: |
| F212N | SATSTAR_NEAR | 32 | 32 |
| F212N | VETTED_BUT_UNSUBTRACTED | 3 | 2 |
| F212N | VETTED_OUT | 0 | 0 |
| F212N | NOT_IN_M7_CATALOG | 50 | 45 |
| F480M | SATSTAR_NEAR | 0 | 0 |
| F480M | VETTED_BUT_UNSUBTRACTED | 9 | 6 |
| F480M | VETTED_OUT | 1 | 1 |
| F480M | NOT_IN_M7_CATALOG | 13 | 10 |

In F212N the NOT_IN_M7_CATALOG leftovers go from 50 to 45 and the VETTED_BUT_UNSUBTRACTED leftovers from 3 to 2, and the SATSTAR_NEAR leftovers go from 32 to 32. In F480M the NOT_IN_M7_CATALOG leftovers go from 13 to 10 and the VETTED_BUT_UNSUBTRACTED leftovers from 9 to 6.

The masked render has 28 big craters of depth 50 MJy/sr and more in F212N and 0 in F480M (`optfin`: 28 and 0), and 0 in F212N and 0 in F480M at 200 MJy/sr and more (`optfin`: 0 and 0). The table lists the craters of `e7fin` with the depth of the crater within 3 px in the other renders (- = no crater there, n/a = no render of that tag).

| tile | band | pixel (x, y) | depth | model peak | depth / model peak | nearest vetted row (arcsec) | flux of that row | `optfin` | `c7fin` | `prod5` |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| o105 | F212N | (1318, 1735) | 97 | 323 | 0.30 | 0.310 | 1127 | 97 | 97 | 101 |
| o105 | F212N | (9701, 1434) | 91 | 274 | 0.33 | 0.007 | 1745 | 91 | 91 | 89 |
| o105 | F212N | (1467, 1650) | 89 | 169 | 0.53 | 0.113 | 1585 | 89 | - | - |
| o105 | F212N | (7311, 564) | 85 | 275 | 0.31 | 0.196 | 270 | 85 | - | - |
| o105 | F212N | (1022, 714) | 84 | 265 | 0.32 | 0.350 | 587 | 84 | - | - |
| o105 | F212N | (6608, 1963) | 75 | 173 | 0.43 | 0.118 | 782 | 75 | 75 | 77 |
| o105 | F212N | (1466, 1739) | 70 | 144 | 0.49 | 0.146 | 556 | 70 | - | - |
| o105 | F212N | (1313, 1738) | 67 | 153 | 0.44 | 0.262 | 800 | 67 | - | - |
| o105 | F212N | (8775, 2122) | 59 | 170 | 0.35 | 0.193 | 802 | 59 | - | - |
| o105 | F212N | (1146, 1232) | 51 | 162 | 0.32 | 0.046 | 1973 | 51 | 109 | 106 |
| o118 | F212N | (1581, 65) | 176 | 557 | 0.32 | 0.024 | 4479 | 176 | - | - |
| o118 | F212N | (6441, 3999) | 104 | 323 | 0.32 | 0.022 | 2460 | 104 | - | - |
| o118 | F212N | (9053, 1087) | 84 | 272 | 0.31 | 0.148 | 2199 | 84 | 84 | 87 |
| o118 | F212N | (3721, 1152) | 78 | 160 | 0.49 | 0.299 | 1725 | 78 | 78 | 79 |
| o118 | F212N | (5505, 724) | 66 | 139 | 0.47 | 0.279 | 739 | 66 | 66 | 68 |
| o130 | F212N | (1329, 1692) | 94 | 177 | 0.53 | 0.128 | 1438 | 94 | - | - |
| o130 | F212N | (7192, 4655) | 76 | 248 | 0.31 | 0.430 | 325 | 76 | - | - |
| o130 | F212N | (1212, 1690) | 70 | 173 | 0.41 | 0.299 | 603 | 70 | 70 | 74 |
| o130 | F212N | (4260, 2031) | 57 | 124 | 0.46 | 0.528 | 449 | 57 | 57 | 56 |
| o130 | F212N | (4377, 2033) | 53 | 117 | 0.45 | 0.276 | 568 | 53 | 53 | 56 |
| o138 | F212N | (9364, 3422) | 154 | 476 | 0.32 | 0.005 | 2608 | 154 | 154 | 162 |
| o138 | F212N | (3998, 2987) | 79 | 223 | 0.35 | 0.011 | 1477 | 79 | - | - |
| o138 | F212N | (3531, 2144) | 73 | 237 | 0.31 | 0.029 | 1801 | 73 | - | - |
| o138 | F212N | (10878, 2237) | 73 | 207 | 0.35 | 0.453 | 599 | 73 | 73 | 72 |
| o138 | F212N | (9536, 1747) | 65 | 169 | 0.39 | 0.215 | 6547 | 65 | - | - |
| o138 | F212N | (9366, 3028) | 59 | 137 | 0.43 | 0.005 | 1069 | 59 | 59 | 62 |
| o138 | F212N | (9667, 1672) | 58 | 186 | 0.31 | 0.278 | 714 | 58 | 58 | 61 |
| o138 | F212N | (9542, 1737) | 56 | 128 | 0.43 | 0.074 | 239 | 56 | - | - |

## Masked area and grid check

The finalize job renders the residual with the footprint masks (the `MERGEDCAT_FOOTPRINT_MASK_*` variables of the recipe of the four-tile run, unchanged in this run). The table gives, per tile and band, the finite pixels of the data mosaic and the number of those that are NaN in the residual of the control render `c7fin`, of the unmasked opt render `e7nm` and of the masked opt render `e7fin`, and the NaN pixels of `optfin` of the four-tile run for comparison. The counts of `c7fin` and `e7nm` are taken from the pixel comparisons of the post jobs: a pixel counts when it is finite in the production residual and NaN in the render. The production residual has no NaN pixel inside the finite data footprint (table of the section "Masked area" of the four-tile README, which lists none for `prod5`), so the finite pixels of the production residual are the finite pixels of the data mosaic.

| tile | band | finite pixels of the data mosaic | NaN pixels in `c7fin` | NaN pixels in `e7nm` | NaN pixels in `e7fin` | share of the data footprint (`e7fin`, %) | NaN pixels in `optfin` (four-tile run) | share of the data footprint (`optfin`, %) | difference in NaN pixels between `e7fin` and `optfin` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| o105 | F212N | 51825305 | 0 | 29 | 7143322 | 13.78 | 7143322 | 13.78 | 0 |
| o105 | F480M | 12464795 | 0 | 5 | 2907644 | 23.33 | 2907644 | 23.33 | 0 |
| o118 | F212N | 51826881 | 0 | 26 | 2828777 | 5.46 | 2828777 | 5.46 | 0 |
| o118 | F480M | 12470721 | 0 | 4 | 1424840 | 11.43 | 1424840 | 11.43 | 0 |
| o130 | F212N | 51826720 | 0 | 29 | 2354582 | 4.54 | 2354582 | 4.54 | 0 |
| o130 | F480M | 12472748 | 0 | 7 | 1042721 | 8.36 | 1042721 | 8.36 | 0 |
| o138 | F212N | 51827646 | 0 | 87 | 4045369 | 7.81 | 4045369 | 7.81 | 0 |
| o138 | F480M | 12472006 | 0 | 0 | 1270606 | 10.19 | 1270606 | 10.19 | 0 |

The masked share of `e7fin` is 4.54 to 13.78 % of the data footprint in F212N and 8.36 to 23.33 % in F480M, and o105 has the largest share in both bands. The control render has no NaN pixel where the production residual is finite, in any tile and band, so it has no mask. The unmasked opt render has a few NaN pixels per mosaic (column `e7nm`), the same number that `optnm` of the four-tile run has. The last column is 0 for every tile and band: the masked render of this run hides as many pixels as the masked render of the four-tile run. The check compares the numbers of NaN pixels and leaves the two pixel sets uncompared. A leftover or crater inside a masked region has no counterpart in the masked render, and the pair tables count it as cleared or gone. The masked counts of the section "Counts per tile" therefore combine the effect of the catalogs of the run with the pixels that the mask hides, and the unmasked render `e7nm` has the catalogs of the run and no hidden pixels.

**Grid check of `e7fin` against `e7nm`.** The post job compares the masked and the unmasked render of the opt arm pixel by pixel. The columns give the shape of the mosaic, whether the world-coordinate keywords of the two files agree (grid keywords), the finite pixels of both renders, the pixels that are NaN in only one of them, the largest absolute difference over the pixels that are finite in both, and the number of those pixels that differ by more than 1e-3 MJy/sr.

| tile | band | shape | same grid keywords | finite in both | NaN only in unmasked | NaN only in masked | max abs difference (MJy/sr) | finite pixels that differ by more than 1e-3 |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| o105 | F212N | (4737, 11440) | True | 44681983 | 0 | 7143293 | 0 | 0 |
| o105 | F480M | (2280, 5640) | True | 9557151 | 0 | 2907639 | 0 | 0 |
| o118 | F212N | (4737, 11440) | True | 48998104 | 0 | 2828751 | 0 | 0 |
| o118 | F480M | (2280, 5640) | True | 11045881 | 0 | 1424836 | 0 | 0 |
| o130 | F212N | (4737, 11440) | True | 49472138 | 0 | 2354553 | 0 | 0 |
| o130 | F480M | (2280, 5640) | True | 11430027 | 0 | 1042714 | 0 | 0 |
| o138 | F212N | (4737, 11440) | True | 47782277 | 0 | 4045282 | 0 | 0 |
| o138 | F480M | (2280, 5640) | True | 11201400 | 0 | 1270606 | 0 | 0 |

In all eight tile-bands the two renders have the same grid, no pixel is NaN in the unmasked render only, the largest difference is 0 and no finite pixel differs. The masks therefore hide pixels and leave every other pixel of the residual unchanged.

## Log checks

**Code and keep radius.** Each fan-out and finalize log names the commit that the package `jwst_gc_pipeline` resolves to (the line `[pipe-root] jwst_gc_pipeline resolves to <path> @ ac6393dc`), and the fan-out logs print the option dictionary of the m7 run. The table gives, per arm and tile, the number of fan-out and finalize logs, how many of them name the commit ac6393dc of the integration tree, and the value of `manual_m7_seed_own_band_companion_keep_fwhm` in the option dictionaries.

| tile | arm | fan-out logs | fan-out logs with the package at ac6393dc | finalize logs | finalize logs with the package at ac6393dc | keep radius printed by the fan-out logs |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| o105 | opt | 8 | 8 | 1 | 1 | 1.5 |
| o105 | ctl | 8 | 8 | 1 | 1 | 1.75 |
| o118 | opt | 8 | 8 | 1 | 1 | 1.5 |
| o118 | ctl | 8 | 8 | 1 | 1 | 1.75 |
| o130 | opt | 8 | 8 | 1 | 1 | 1.5 |
| o130 | ctl | 8 | 8 | 1 | 1 | 1.75 |
| o138 | opt | 8 | 8 | 1 | 1 | 1.5 |
| o138 | ctl | 8 | 8 | 1 | 1 | 1.75 |

All fan-out and finalize logs of both arms name the integration tree, the fan-out logs of the opt arm print the keep radius 1.5 and those of the control arm print the default of the tree of this run, 1.75.

**Switch lines.** Each switch writes a log line when it is active, and the one-neighbour tolerance of the spike finder writes a warning when it is set while the spike mask is off. The table counts these lines in the logs of the jobs, per tile and arm. The environment snapshots of the two arms (`results/run7_envsnap_<arm>_<tile>.txt`) show which variables were set, and the log lines show that the code read them.

- **fan-out logs**: the logs of the 8 array tasks.
- **lines of NEARSAT_CONTINUOUS_DIST**: the lines "NEARSAT_CONTINUOUS_DIST: near-saturation filter uses the unrounded fitted position" in the fan-out logs (#1394).
- **spike lines in finalize**: the lines "positive spikes: N isolated one-pixel spikes of >= 300 MJy/sr in n of m frames ..." in the finalize log (#1374 mask with the tolerance of #1392). The finalize job runs the spike mask in six passes, for nrca, nrcb and the merged module, each for F212N and F480M. A spike is counted by one frame, and the two module passes add up to the merged pass. The script that makes the table checks this sum and stops when it fails, so the columns "F212N spikes" and "F480M spikes" give the merged pass.
- **spikes with a missing neighbour**: the number M in the clause "POSITIVE_SPIKE_MASK_MAX_UNUSABLE=1: M of the spikes have a missing neighbour" of the same line. These are the spikes with one unusable neighbour (NaN, DO_NOT_USE or outside the array), which the finder returns only with the tolerance of #1392. Each spike sets its 3 × 3 pixel box to DO_NOT_USE.
- **partial-box lines**: the lines "MERGE_PARTIAL_BOX_FRAC_MIN=0.5 MERGE_PARTIAL_BOX_ZMAX=3: left R row(s) of S source(s) out of the averages (fit box 25 pixels)" of the finalize log (#1395). The merge prints one line per call of the combine function and the lines do not add up to a total over the tile, so the table gives the number of lines and the line with the most rows left out.
- **"has no effect while" warnings**: the warning that the tolerance is set while the spike mask is off.

| tile | arm | fan-out logs | lines of NEARSAT_CONTINUOUS_DIST | spike lines in finalize | F212N spikes | F212N spikes with a missing neighbour | F480M spikes | F480M spikes with a missing neighbour | partial-box lines | rows left out in the largest line | sources of that line | "has no effect while" warnings |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| o105 | opt | 8 | 120 | 6 | 16 | 1 | 57 | 3 | 15 | 448 | 435 | 0 |
| o105 | ctl | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| o118 | opt | 8 | 120 | 6 | 12 | 0 | 66 | 6 | 9 | 321 | 310 | 0 |
| o118 | ctl | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| o130 | opt | 8 | 120 | 6 | 10 | 1 | 72 | 3 | 9 | 287 | 274 | 0 |
| o130 | ctl | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| o138 | opt | 8 | 120 | 6 | 5 | 0 | 68 | 1 | 9 | 389 | 378 | 0 |
| o138 | ctl | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Every opt arm shows the NEARSAT line in the fan-out logs, six spike lines with the clause of the tolerance and partial-box lines in the finalize log. Every control arm shows none of these lines and no warning. The tolerance admits spikes with a missing neighbour in F480M in every opt tile (1 to 6 spikes per tile) and in F212N in two of the four opt tiles (1 spike in each). The number of partial-box lines per tile (9 or 15) depends on how many calls of the combine function the finalize job made.

## CRDS cache files

The jobs set `CRDS_READONLY_CACHE=1`. The CRDS cache of the production tree (`/orange/adamginsburg/jwst/crds`) holds `config/jwst/server_config` (1457934 bytes) and `config/jwst/bad_files.txt` (5034 bytes). A listing written at 2026-11-27 20:52:51 EST, before the staging job and the first submission, gave the modification time 2026-11-27 19:29:11 for both files (`results/run7_crds_before.txt`). The four-tile README lists modification times of the two files up to 2026-11-27 15:53:45, so the files changed at least once more before this run started, at a time when no job of this run existed.

Each post job takes a listing at the end of its run and compares it with the listing before (`results/<arm>_<tile>_crds_after.txt` and `results/<arm>_<tile>_crds_diff.txt`). The seven post jobs of the opt arm of the four tiles and of the control arms of o118, o130 and o138 give the modification time 2026-11-27 22:52:37 for both files. The post job of the control arm of o105 gives 2026-11-28 02:26:29. The sizes are equal to the sizes before the run in all eight listings. The check compares sizes and modification times and does not compare contents.

`results/run7_crds_attribution.txt` lists the CRDS client lines and the PARS-RESAMPLESTEP lines of the 81 logs of the run (the staging log and the fan-out, finalize and post logs of the eight chains): 16 CRDS client lines and 72 PARS-RESAMPLESTEP lines.

- At 22:52:37 no line of the run lies within 100 s. The nearest line is 124 s later, in the log of the finalize job of the o130 opt arm. At that time 31 jobs of this run and 207 other jobs of the same account were running.
- At 02:26:29 no line of the run lies within 100 s. The nearest line is 1290 s earlier, in the log of the post job of the o105 opt arm. At that time 1 job of this run (the finalize job of the o105 control arm) and 193 other jobs of the same account were running.

The writer of the modifications at 22:52:37 and 02:26:29 is not identified. The jobs of this run set `CRDS_READONLY_CACHE=1`, and the probe of the installed CRDS client in the four-tile README (`results/run5_crds_probe.txt` of that run) shows that with this setting the client leaves `server_config` unchanged. This test does not establish whether the contents changed. This part is a note and asks for no decision.

## What this run does not show

- **The split between the four additions.** The opt arm has the three switches and the keep radius on together. The control arm has every switch off, so no render has one addition alone. The effect of #1392 rests on the six leftovers at one-pixel spikes, the effect of #1394 on features 4 and 5 and on the o105 F480M row of the section "Differences against the four-tile run that the lists show", the effect of #1395 on the o130 F212N leftover at (5065, 68) and on the o118 F212N crater, and the effect of the keep radius on the comparison of the seed tables. A sum over the four tiles, such as the F480M clearing of 35 of 299 leftovers in the unmasked pair against the four-tile run, is not split by switch.
- **The o138 F212N row of flux 8975.** The opt catalog of this run has the row and the opt catalog of the four-tile run and the control catalog lack it. Replays of exposures 3 and 6 of the nrcb2 detector show that #1394 makes its two fits (section "Differences against the four-tile run that the lists show"). The replays cover those two frames; production has a fit of the row in exposure 5 and the opt table of this run has none, and no replay covers that exposure. The effect of #1395 at the merge is argued from the fitted fractions of the box and no finalize rerun tested it. Whether the row is a star or a structure of the PSF of the saturated neighbour, which lies 2.88 FWHM away, is undecided.
- **Craters near the cut of the crater scan.** The crater of o138 F212N at (8695, 623) changes its listing with a 0.4 % change of the block sigma, which the model of the extra o138 F212N row causes. The lists of the three differences were examined; the number of other craters within a few percent of their cut was not counted, so the crater counts have a granularity of a few craters from this effect.
- **Whether the row of o105 F480M at (5325, 1803) is a star.** The model holds a row at the pixel in the four-tile run and lacks it in this run. The row stands on one fit (qfit 0.014) next to a saturated star at 1.7 FWHM, and no test of this study decides between a star and a structure of the saturated neighbour.
- **The masks.** The masked renders hide 4.54 to 13.78 % (F212N) and 8.36 to 23.33 % (F480M) of the data footprint, so their counts apply to the pixels outside the mask. The masks of `e7fin` and `optfin` have the same number of NaN pixels in every tile-band; the pixel sets were not compared.
- **Feature 3 and the satstar gate.** #1393 is not in the tree, so the o138 F212N leftover at (8633, 2587) is unchanged from the four-tile run (section "Features that motivated the switches").
- **Tiles and bands.** The four tiles o105, o118, o130 and o138 in F212N and F480M. The keep radius acts only where the keep S/N is above 0, which is F212N in the recipe, so the F480M seeds are the same as in the four-tile run. o045 is not in this run.
- **Equality of the environment with production.** The log-template comparison of the four-tile run supports the equality of the environment of the shard jobs with production and cannot prove it. The control arm of this run uses the same environment file as the control arm of the four-tile run.

## Decisions and open items

Items 1 and 3 are decisions of 2026-11-28, taken after the results of this run were read; item 2 reports the trace that was requested on the same day; items 4 to 6 have no decision.

**Decided and done**

1. **The default of the keep radius** of the far-companion exemption of the m7 seed (`--manual-m7-seed-own-band-companion-keep-fwhm`) is 1.5 FWHM, set by commit 39cfac28 of #1377 (it was 1.75). The opt arm of this run passes 1.5 on the command line, so this run measures the new default, and the integration tree of this run still has 1.75. The seed comparison against the four-tile run adds 15 merged-module rows in F212N across the four tiles, drops 0, and the followed rows give 5 cleared leftovers, 0 false seeds and 0 new leftovers (section "Seed rows added by the keep radius"). The default matters where the keep S/N is above 0, which is F212N=50 in the recipe. The two o105 F212N leftovers of group 3 of the four-tile run go from 343 to 111 and from 213 to 73 MJy/sr in the unmasked render. The evidence has a limit: of the 15 rows that the radius 1.5 adds in this run, 5 sit on a leftover of 50 MJy/sr or more, which is 33 %, while the earlier sweep of the radius on the o045 and o105 F212N tiles (documentation of #1377) found 2 hits in the 18 rows added at 1.5 to 1.75 FWHM (11 %) and 8 hits in the 17 rows added at 1.75 to 2.0 FWHM (47 %). The tiles and the definition of a hit differ between the two measurements.
2. **The o138 F212N row of flux 8975 is traced** (section "Differences against the four-tile run that the lists show"): #1394 alone makes its two fits; #1392 and #1395 have no part in them. The listing flip of the o138 F212N crater at (8695, 623) follows from this row.
3. **#1392, #1394 and #1395 stay opt-in** and off by default; the recipes switch them on. The measurements of this run for each: #1392 leaves no leftover ≥100 MJy/sr at the six spike positions (section "Features that motivated the switches"); #1394 clears features 4 and 5 and, in the unmasked pair against the four-tile run, 35 of 299 F480M leftovers with 1 listed as new, costs the o105 F480M row at (5325, 1803) (box maximum 42 → 338 MJy/sr, hidden by both masks) and returns the o138 F212N row that production has; #1395 clears the o130 F212N leftover at (5065, 68) (box maximum 1244 → 9 MJy/sr) and the o118 F212N crater at (11389, 1475). The run has no arm with one switch on, so each default rests on the feature checks above. The defaults are revisited when the PRs are merged.

**Open**

4. **Whether the o138 F212N row is a star.** The measurements of the section "Star or structure" fit a structure of the PSF at 2.9 FWHM from a saturated star and fit a faint star on it. #1394 returns this row, which production has and the control and four-tile runs lose, so the switch trades model flux next to saturated stars (residual cleanup) against catalog purity if such rows are structure. Which of the two holds is not decided by this run.
5. **Exposure 5 of the o138 F212N row.** Production has a fit of the row in exposure 5 and the opt table of this run has none. A replay of exposure 5 with and without #1394 (about 7 minutes per replay) would show whether the fit exists before the filter.
6. **The origin of the flux factors of merged rows.** The merged flux of the o118 F212N star (section "Differences against the four-tile run that the lists show") and that of the extra o138 F212N row (1.0125 times the inverse-variance mean of its two fits) differ from the inverse-variance mean of their fits, and the merged flux of the production row at the o138 position equals that mean. The origin of the factor is not identified.

## Evidence files

The files are in `results/`. `<tile>` stands for o105, o118, o130 and o138, `<band>` for F212N and F480M and `<arm>` for opt and ctl. The residual and data mosaics, the merged catalogs, the per-frame fit tables and the job logs that the tables are made from are on scratch storage of the run and are not copied; the text and JSON files below hold what the comparison, scan, seed and trace steps print from them.

| file | content |
| :--- | :--- |
| `run7_integration_tree.txt` | HEAD, base, cherry-picks and diff stat of the integration tree |
| `run7_jobtable.txt` | the job table of the section "Setup of the run" |
| `run7_sacct.txt` and `<arm>_<tile>_sacct.txt` | Slurm accounting of the jobs (all; one arm of one tile) |
| `run7_envsnap_<arm>_<tile>.txt` | environment of the submission of each arm and tile |
| `<tile>_compare_<band>.txt` and `<tile>_compare_<band>.json` | counts, classes and 3 px matching tables per tile and band, as the comparison step prints them |
| `<tile>_scan_<source>_<reference>_<band>.txt` | position-independent scans: `c7fin` against `prod5`, `e7nm` against `c7fin`, `e7fin` against `c7fin`, `e7nm` against `optnm` and `e7fin` against `optfin` |
| `o105_ctlfin_vs_c7fin_<band>.txt` | counts and matching of the control render of the four-tile run against the control render of this run on o105 |
| `<tile>_seeddiff.txt` and `<tile>_seeddiff.json` | seed tables of this run against those of the four-tile run per module and band, with the added rows followed to the residual renders |
| `run7_features.txt` | residual extrema of the 13 × 13 px boxes at the features of the section "Features that motivated the switches" in every render |
| `run7_seven.txt` | the leftovers at one-pixel spikes with the box maximum and the largest leftover in each render |
| `run7_mask_area.txt` and `run7_grid_check.txt` | NaN pixels of the masked renders and the grid check of `e7fin` against `e7nm` |
| `run7_remaining.txt` | the remaining leftovers of the masked render with class, and the remaining craters |
| `run7_log_checks.txt` | checks of the logs of the jobs (tree, options, switch lines, errors) |
| `run7_newfeatures.json` and `run7_newfeatures.txt` | the three features of the section "Differences against the four-tile run that the lists show": box extrema, merged rows within 0.6 arcsec in the three catalogs, merged-table columns and the distances of the frames |
| `run7_perframe_rows.json` and `run7_perframe_rows.txt` | per-frame fits at the rows that differ, in the three catalogs |
| `run7_craterflips.json` and `run7_craterflips.txt` | residual, block background, block sigma, cut and listing of the crater scan at the two craters that differ |
| `run7_o138trace_tables.txt` and `o138trace/replay_summary.json` | the two replay tables of the section "Differences against the four-tile run that the lists show" and the summary of the ten one-frame replays that they are made from (rows of each per-frame table, the fits at the target positions, the comparison with the two reference tables) |
| `o138trace/replay_compare.txt`, `o138trace/spec.txt`, `o138trace/sacct_done.txt` and `o138trace/filterlines.txt` | the text comparison of each replay, the specification of the ten replays, their Slurm accounting and the lines of the near-saturation filter of their logs |
| `o138trace/rulecheck.txt` and `o138trace/geom.txt` | the DQ map and the two distance rules at the fit positions and at the starting positions, the spike finder at the tolerances 0 and 1, and the separation of each fit from the saturated stars |
| `o138trace/blockdiff.txt`, `o138trace/resid_row_check.txt` and `o138trace/prod_render_check.txt` | the model difference in the 64 px block of the crater scan, the residual extrema around the extra row and the residual, model and data at the pixel of the row in each render |
| `o138trace/m6row.txt`, `o138trace/prod_rows.txt`, `o138trace/prod_contrib.txt`, `o138trace/contrib.txt`, `o138trace/seed_check.txt`, `o138trace/tablediff.txt`, `o138trace/final_numbers.txt` and `o138trace/derived.txt` | the rows of the m6 and m7 catalogs and of the per-frame tables at the position of the row in production and in the opt arm, the seed rows within 1.0 arcsec, the rows that #1394 adds to the per-frame tables, the numbers of the section and the inverse-variance means that it quotes |
| `o138trace/spikeangle.txt`, `o138trace/starcheck.txt` and `o138trace/wingrows.txt` | the angles of the 11 fits around saturated stars, the data minus the satstar model around the fits and the fits with positive flux at 1.5 to 4.5 FWHM from the saturated stars |
| `<arm>_<tile>_precheck_F480M.txt` | output of the precheck step of the post job (empty when every product exists) |
| `run7_crds_before.txt`, `<arm>_<tile>_crds_after.txt` and `<arm>_<tile>_crds_diff.txt` | listing of the two CRDS cache files before the run, after each arm and tile, and the difference |
| `run7_crds_attribution.txt` | log attribution of the modification times of the two CRDS cache files |
