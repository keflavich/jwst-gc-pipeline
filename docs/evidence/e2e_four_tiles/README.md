# End-to-end four-tile test of the stacked opt-in PRs

One scratch-basepath run per tile of the stack of 17 open draft code PRs (#1360 to #1377, listed under Recipe; #1380 and #1383 are documentation only and were not merged) on the gc-treasury 10678 tiles o105, o118, o130 and o138, F212N and F480M. Every opt-in switch of the stack is on, and the second-pass neighbour guard of #1360 (`SATSTAR_SECOND_PASS_NEIGHBOUR_FRAC_MIN=0.2`) is part of the recipe. The jobs carry the two hand-off settings of the production F480M m7 jobs (`DAOPHOT_HANDOFF_PEAK_MIN_AREA=50` and `DAOPHOT_HANDOFF_RESTORE_ACCEPTED_EXCL_FWHM=1.5`; `DAOPHOT_HANDOFF_FILTERS=F480M` limits them to F480M). The run follows the end-to-end test of o045 (#1383). Each tile ran the m7 per-frame fits as an 8-task array and a finalize job (merge, vetting and the residual renders with the footprint masks). A post job rendered the residual without the masks, scored the residual i2d mosaics (data minus merged-catalog model) for whole-star leftovers and big craters, and compared them with the production render of the tile. The 40 jobs ran between 2026-11-26 23:28:54 and 2026-11-27 06:12:19 (Slurm accounting), and all ended COMPLETED with exit code 0:0. The integration tree is a local branch that was never pushed.

An earlier run of the same four tiles lacked the two hand-off settings. Its F212N renders equal those of this run, and its F480M renders differ (section "The two F480M hand-off settings"). Every number in this file belongs to this run unless the text says otherwise, and every table is generated from the files in `results/` (list at the end).

There is no control arm (every switch off) on these four tiles. A difference between the production render and a render of this run comes from the switches and from the code of the integration tree, which is ahead of the code of the production run (section "Setup of the run"). On o045 the control arm of the integration code differs from the production render as well: in F212N the leftovers ≥100 MJy/sr number 376 in the production render and 369 in the control arm (#1383).

**Result.**

- **Counts with the real masks** (`optfin` against the production render `prod5`; sums over the four tiles; the thresholds are explained under "How the counts are made"). F212N: leftovers ≥100 MJy/sr 702 → 85 and ≥500 MJy/sr 384 → 2; craters ≥50 MJy/sr 1382 → 28 and ≥200 MJy/sr 443 → 0. F480M: leftovers ≥100 MJy/sr 518 → 23 and ≥500 MJy/sr 221 → 0; craters ≥50 MJy/sr 163 → 0. The masks set 4.54 to 13.78 % (F212N) and 8.36 to 23.33 % (F480M) of the finite pixels of the data mosaic of a tile to NaN (section "Masked area"), and a leftover or crater inside a masked region has no counterpart in the masked render, so the masked counts are low where the mask hides the pixels.
- **Counts without the masks** (`optnm`: the catalogs of the finalize job, rendered with the mask variables unset). F212N: leftovers ≥100 MJy/sr 702 → 209 and ≥500 MJy/sr 384 → 42; craters ≥50 MJy/sr 1382 → 322, ≥200 MJy/sr 443 → 112 and ≥500 MJy/sr 184 → 34. F480M: leftovers ≥100 MJy/sr 518 → 299, ≥500 MJy/sr 221 → 199 and ≥1000 MJy/sr 173 → 158; craters ≥50 MJy/sr 163 → 26, ≥200 MJy/sr 52 → 16 and ≥500 MJy/sr 24 → 11. The unmasked render shows what the fit-time switches do. The masks hide pixels and leave the catalogs unchanged: at every pixel that is finite in both renders the masked and the unmasked render are equal (grid check, 8 pairs of renders).
- **F480M without the masks, by class.** Leftovers ≥200 MJy/sr near a consolidated satstar fit (SATSTAR_NEAR) go 234 → 236 (o105 140 → 134, o118 44 → 48, o130 13 → 16, o138 37 → 38), and the other three classes shrink: VETTED_BUT_UNSUBTRACTED 36 → 3, VETTED_OUT 4 → 0, NOT_IN_M7_CATALOG 84 → 18. In the 3 px matching of the F480M leftovers ≥100 MJy/sr (518 in `prod5`), 254 are cleared in `optnm`, 264 are unchanged, and 35 leftovers of `optnm` are listed as new (33 of them SATSTAR_NEAR). In F212N every class shrinks (SATSTAR_NEAR 228 → 64).
- **Features of the unmasked render that exceed production** (position-independent scan of leftovers and craters ≥200 MJy/sr in a 13 × 13 pixel box; "How the counts are made"). F212N: none is absent in production, and 3 exceed the production extremum in the box by more than a factor 1.5: the craters at (8306, 1270) and (4469, 1280) of o105 (1590 against 971 MJy/sr and 755 against 456 MJy/sr; the nearest vetted row lies 0.031 and 0.032 arcsec away, is flagged as saturated and has a flux of 49769 and 28265) and the leftover at (8633, 2587) of o138 (2209 against 1297 MJy/sr, SATSTAR_NEAR at 0.095 arcsec). F480M: 2 are absent in production, both leftovers of o105: (2614, 1826) with 904 against 29 MJy/sr (SATSTAR_NEAR at 0.384 arcsec) and (4452, 858) with 403 against 30 MJy/sr (NOT_IN_M7_CATALOG, 3.9 arcsec from the nearest satstar fit); none exceeds a production extremum ≥200 MJy/sr by more than a factor 1.5. The scans of the two masked renders find no such feature in either band. One-frame replays (section "Cause of the five features that exceed production") trace all five to commit 018eb243, which applies the ePSF core to the PSF grid of the saturated-star fit. The two craters also need the second-pass neighbour guard of #1360, and the three leftovers need no switch of the stack.
- **Rows that production has and the run lacks.** Rows of the production merged vetted m7 catalog with a flux ≥1e4 and no row of the run within 0.1 arcsec number 10 in F212N and 8 in F480M (table in the section "Rows that production has and the run lacks"). One-frame replays of o105 F480M show that the two hand-off settings matter for such rows: the production code without them loses 12 rows with `flux_fit` above 1e4 (3 above 3e4) of the production per-frame table of one frame and with them 0, and the integration code with them loses 2 (every switch off) and 0 (every switch on) rows above 1e4 of that frame. In the first run of this tile set, which lacked the settings, 41 production rows with a flux ≥1e4 had no row in the F480M catalogs (`results/run4_catdiff.txt`).
- **Added seed rows.** The opt seed of F212N holds 154 rows that the production seed lacks (66 from the far-companion exemption of #1377, 88 from the relaxed bright window of #1376). 153 of them have a row within 0.1 arcsec in the merged raw catalog of the run and 152 in the merged vetted catalog; 13 of these 152 also have a row in the production merged vetted catalog. The 2 seed rows without a vetted row within 0.1 arcsec are one of o105 with a seed flux of 3.15e5 (its nearest raw row lies 0.22 arcsec away) and one of o130 with a seed flux of 2.48e4 (a raw row lies 0.07 arcsec away with a fitted flux of 42.8, and the nearest vetted row lies 0.19 arcsec away). F480M: 21 rows, all from #1376 (the exemption is off in F480M), all with a vetted row, none with a production row.
- **The two hand-off settings.** Without them the unmasked F480M render has more leftovers: ≥100 MJy/sr 333 against 299 with them, ≥500 MJy/sr 229 against 199, and SATSTAR_NEAR ≥200 MJy/sr 267 against 236. The masked F480M renders have the same counts with and without them, and the F212N renders equal each other (section "The two F480M hand-off settings").
- **What the run does not separate.** The switches and the code drift of the integration tree enter every difference with production. In frame A of the one-frame replays the integration code with every switch off loses 2 per-frame rows above 1e4 that the production code keeps, so the code drift contributes to the production rows that the run lacks. Whether the masks should use the F212N-validated parameters in F480M (8 to 23 % of the footprint is masked) is an open decision (section "Open decisions").

## How the counts are made

- **Whole-star leftover.** The residual mosaic gets a local background and sigma: the median and 1.4826 × the median absolute deviation in blocks of 64 px for F212N and 32 px for F480M. Peaks are the local maxima of the background-subtracted residual after smoothing with a Gaussian whose standard deviation is the larger of FWHM/2.355 and 0.7 px. A peak is the maximum of a box with a side of 2 FWHM rounded to a whole number of pixels, at least 5 px and made odd, and exceeds the larger of 25 sigma and 20 MJy/sr, where sigma is the local sigma of the first sentence. A peak value is the 3 × 3 maximum. FWHM = 0.072 arcsec for F212N and 0.164 arcsec for F480M. A peak is a whole-star leftover when its residual peak exceeds 0.7 times the data peak (3 × 3 maximum of the data i2d minus the block background) and exceeds 50 MJy/sr. The tables count leftovers with a residual peak ≥N MJy/sr for N = 50, 100, 200, 500, 1000.
- **Class of a leftover** (first match wins; the catalogs are those of the tree that made the render). SATSTAR_NEAR = a consolidated satstar fit within 1.0 arcsec; VETTED_BUT_UNSUBTRACTED = a vetted m7 row within the larger of 1 FWHM and 0.06 arcsec; VETTED_OUT = a row of the raw m7 catalog within that radius; NOT_IN_M7_CATALOG = none of these.
- **Big crater.** The same smoothed, background-subtracted residual gives local minima below the negative of the larger of 25 sigma and 20 MJy/sr. The depth is the 3 × 3 maximum of −(residual − background) and the model peak is the 3 × 3 maximum of the merged-catalog model i2d. A big crater has depth above 50 MJy/sr and depth above 0.3 times the model peak; the tables count craters with depth ≥50, 200 and 500 MJy/sr.
- **3 px matching** (pair tables). For each leftover ≥100 MJy/sr of render A: *cleared* = no leftover of render B within 3 px (0.094 arcsec; a leftover inside a region that a mask sets to NaN has no counterpart in the masked render and counts as cleared); *reduced* = the B peak at the matched leftover is below 0.5 times the A peak; *worse* = above 1.5 times the A peak; *unchanged* = otherwise. A leftover ≥100 MJy/sr of B with no counterpart in A within the same radius is *listed as new*. Big craters of A without a crater of B within the radius are *gone*; craters of B without a counterpart in A are *listed as new*. A feature whose extremum moves by more than 3 px between the renders is counted once as cleared (or gone) and once as listed as new.
- **Position-independent scan.** Every leftover of the scanned render with a residual peak ≥200 MJy/sr, and every big crater of depth ≥200 MJy/sr, is compared with the extremum of the production render in the 13 × 13 pixel box centred on it (maximum of the residual for a leftover, depth of the minimum for a crater), so a shift of the extremum by a few pixels does not change the verdict. With S the extremum of the scanned feature and R the production extremum in the box: *absent in production* = R < 200 MJy/sr and S > 1.5 R; *larger than in production* = R ≥ 200 MJy/sr and S > 1.5 R. The scan runs for the unmasked render (`optnm`) and for the masked render (`optfin`) against `prod5`. The saved scan files list the features that are absent in production; the features that are larger than in production are counted and listed by a second run of the scan with the same box. The box extremum comes from the whole residual image, so a reference extremum can belong to a source that fails the whole-star criteria.
- **Renders.** `prod5` is the production residual i2d of the tile in the production tree (read only), scored with the scripts of this run so that production and run are scored alike. `optfin` is the residual i2d that the finalize job of the run writes, with the footprint masks of the recipe. `optnm` is the residual i2d rendered from the same catalogs with every `MERGEDCAT_FOOTPRINT_MASK_*` variable unset. The grid check of each pair of renders reads the shapes and the grid keywords and counts the pixels that are NaN in one render only and the finite pixels that differ.
- **Masked area.** The share of the finite pixels of the merged data i2d that are NaN in the residual i2d of a render. The finite pixels of the data i2d number about 5.18e7 per tile in F212N and 1.25e7 in F480M (table in the section "Masked area").

## Recipe

Environment of the run. The lines are the environment file of the o045 runs (#1383) with its comments replaced by the number of the PR that introduces the switch, plus the guard line at the end. Every tile used the same file.

```bash
# fit time
export SINGLE_EXPOSURE_EVENT_MASK=1                     # #1360
export NEGATIVE_PIXEL_MASK_NSIGMA=50                    # #1360
export MERGE_ROW_RESID_SUPPORT_MIN=0.3                  # #1360
export SINGLE_EXPOSURE_EVENT_RAMP=1                     # #1367
export POSITIVE_SPIKE_MASK_MIN_EXCESS=300               # #1374
export DAOPHOT_DEDUP_NONPOSITIVE=1                      # #1366
export MERGE_PAIR_RESCUE_RATIO=3                        # #1365
export MERGE_PAIR_RESCUE_FLUX_MIN=500                   # #1365
export SATSTAR_FITTER_MAXITERS=1000                     # #1371
export SATSTAR_EDGE_POS_BOUND=12                        # #1371
export SATSTAR_SIDELOBE_MASK_FRAC_MIN=0.5               # #1369
export SATSTAR_DAO_RESTORE_FRAC_MIN=0.8                 # #1362
export SATSTAR_SECOND_PASS_NEIGHBOUR_FRAC_MIN=0.2       # #1360 second-pass neighbour guard
# render time
export MERGEDCAT_FOOTPRINT_MASK_KFAC=4                  # #1361 (K=4 evidence: #1380)
export MERGEDCAT_FOOTPRINT_MASK_MARGIN_PX=2             # #1361
export MERGEDCAT_FOOTPRINT_MASK_RING_FLUX_MIN=1e6       # #1364
export MERGEDCAT_FOOTPRINT_MASK_RING_RADIUS_PX=25       # #1364
export MERGEDCAT_FOOTPRINT_MASK_RING2_FLUX_MIN=3e4      # #1375
export MERGEDCAT_FOOTPRINT_MASK_RING2_RADIUS_PX=8       # #1375
export MERGEDCAT_FOOTPRINT_MASK_REJECTED=1              # #1370
export MERGEDCAT_FOOTPRINT_MASK_DQSAT_PX=4              # #1372
export MERGEDCAT_FOOTPRINT_MASK_EDGE_PX=8               # #1373
```

`SATSTAR_SECOND_PASS_NEIGHBOUR_FRAC_MAX` stays unset (default 0.8). `SATSTAR_SIDELOBE_FRAC_MIN` (#1368, a fractional waiver of the sidelobe cut in the satstar accept gate) is not set: #1369 marks the sidelobe-only rejects that a mask can cover, and `SATSTAR_SIDELOBE_MASK_FRAC_MIN` in the block above hands them to the footprint mask. The guard line is the one that the decision of 2026-11-26 added to the opt-in recipe after the finalize-only test on o045 (#1383); the default of the code stays off.

Command-line options of the m7 stage. A plain number applies to every band; comma-separated `BAND=VALUE` pairs apply per band, and a band that no pair names keeps the default of the option (#1376, #1377):

| option | PR | default | value of this run |
| :--- | :--- | ---: | ---: |
| `--manual-m7-seed-own-band-companion-keep-snr` | #1377 | 0 | F212N=50,F480M=0 |
| `--manual-seed-round-bright-max` | #1376 | 0 | 1.5 |
| `--manual-seed-round-bright-snr-min` | #1376 | 100 | F212N=30,F480M=100 |

The base options of the production m7 run are passed as well: `--manual-ext-star-prom-robust-min=-1` `--residual-bg-median-size=0` `--residual-bg-satstar-mask-fwhm=3.75` `--manual-seed-round-loose-max=-1`.

Further environment of the jobs: `GC_ALLOW_DEV=1`, `SATSTAR_OPTIN_FILTERS=F480M`, `SATSTAR_ZF_KEEP_FINITE=1`, `SATSTAR_OBS_PK_FROM_CRF=1`, `SATSTAR_QFIT_LOCAL_GATE=1`, `DAOPHOT_HANDOFF_FILTERS=F480M`, `DAOPHOT_HANDOFF_PEAK_MIN_AREA=50`, `DAOPHOT_HANDOFF_RESTORE_ACCEPTED_EXCL_FWHM=1.5`, `PSF_EPSF_CORE_DIR` (the directory of the core PSFs), `CRDS_PATH` (the shared CRDS cache in the production tree) and `CRDS_READONLY_CACHE=1`.

The two hand-off settings act in the m7 stage on the bands that `DAOPHOT_HANDOFF_FILTERS` lists, which is F480M here.

- `DAOPHOT_HANDOFF_PEAK_MIN_AREA` (`jwst_gc_pipeline/photometry/cataloging.py::_daophot_handoff_peak_min_area`) is the component area in pixels from which the hand-off of unaccepted SATURATED components to daophot uses the local maxima of the `sci` image inside the component. Unset gives 0, and every component is then handed off at its centre of mass. The run uses 50.
- `DAOPHOT_HANDOFF_RESTORE_ACCEPTED_EXCL_FWHM` (`jwst_gc_pipeline/photometry/cataloging.py::_daophot_handoff_restore_accepted_excl_fwhm`) is an exclusion radius in FWHM. Unset skips every SATURATED component that holds an accepted satstar centre in the pixel restore of the hand-off. A value ≥0 keeps such a component and leaves masked only the pixels within that many FWHM of an accepted centre. The run uses 1.5.

No script of the repository sets the two variables, and no document of the repository names them. The production F480M m7 shard logs of o105 (job 50037748) and o045 (job 50365036) print the hand-off lines that need them. Those lines read `[manual] unaccepted-SATURATED hand-off: ... handed off at P peak(s)` and `[manual] hand-off restore: ...`. Every fan-out shard log of this run (32 of 32, 96 hand-off lines in total) and every finalize log (4 of 4, 24 lines each) prints both lines. None prints the line of the setting-free hand-off, which reads `N of M SATURATED component(s) handed to daophot` without `handed off at`.

What the environment check covers. The message templates of the production shard log of o105 (job 50037748, task 5) and of the shard log of task 5 of this run (digits replaced by a placeholder, table rows left out) differ as follows. The production log has 5 message lines that the log of the run lacks: the pipeline-root line, the shell path line, the pipeline tag line, and two lines that announce the rebuild of a per-frame satstar catalog and print the fit-switch signature of the default fit (`zfg1.3bc_sq_dfr0.5`). The log of the run has 47 templates with 209 lines that the production log lacks: messages of the switches of the stack (post-fit dedup, bright roundness seed), messages of the satstar stages (position QA, zeroframe level) and 62 lines of one saturated-pixel warning whose text both code versions contain. No production-only line reports a non-default setting, so the check finds no sign of a further environment variable in the production jobs. It cannot show that none exists, because a variable that changes numbers without changing a message leaves no trace in a log. The one-frame replays in the section "Rows that production has and the run lacks" test the environment on numbers: the production code with the two settings reproduces six production rows of one frame (`flux_fit`, `qfit` and `local_bkg` equal).

Merge order of the integration tree: #1360, #1361, #1362, #1364, #1370, #1372, #1373, #1374, #1375, #1367, #1365, #1366, #1371, #1368, #1369, #1376, #1377. #1380 (documentation of the K=4 mask) and #1383 (documentation of the o045 test) were not merged. Base branches, titles and heads as of 2026-11-27 (`gh pr view`); all 17 heads are ancestors of the tree (`results/integration_tree.txt`).

| PR | title | base branch | head |
| :--- | :--- | :--- | :--- |
| #1360 | Opt-in guards that remove bright-star leftovers and craters from the merged-catalog residual | `main` | 675ed9a6 |
| #1361 | Opt-in footprint mask for the merged-catalog residual i2d (stacked on #1360) | `mergedcat-residual-guards` | 1424a31e |
| #1362 | Opt-in restore of the daophot row that a dropped satstar row overwrote (SATSTAR_DAO_RESTORE_FRAC_MIN) | `mergedcat-residual-footprint-mask` | 2ebe41c1 |
| #1364 | Add a ring-zone extension to the residual footprint mask (opt-in, stacked on #1362) | `satstar-dao-restore` | f095649f |
| #1370 | Mask the footprints of maskable gate-rejected satstar fits in the residual i2d | `footprint-mask-ring-zone` | aaeceeac |
| #1372 | Add a DQ_SATURATED dilation to the residual footprint mask | `footprint-mask-rejected-satstars` | cae05e94 |
| #1373 | Add an opt-in guard band along the data footprint edge to the residual footprint mask | `footprint-mask-dqsat` | 9434ba67 |
| #1374 | Add an opt-in positive one-pixel spike mask for the per-frame residuals | `footprint-mask-edge-band` | 56173877 |
| #1375 | Add an opt-in second flux/radius tier to the footprint mask ring-zone extension | `footprint-mask-edge-band` | 43d61ec7 |
| #1367 | Opt-in ramp test for single-exposure events (SINGLE_EXPOSURE_EVENT_RAMP) | `mergedcat-residual-guards` | 9733dfc0 |
| #1365 | combine_singleframe: opt-in flux-aware base-row seeding (MERGE_PAIR_RESCUE_RATIO) | `main` | 622fe467 |
| #1366 | m7 post-fit dedup: keep the positive-flux row against a non-positive fit (DAOPHOT_DEDUP_NONPOSITIVE, default off) | `main` | 459ad0aa |
| #1371 | Satstar: opt-in LevMar iteration cap and edge position bound | `main` | 60e6f3a0 |
| #1368 | Satstar accept gate: opt-in fractional waiver of the sidelobe cut | `main` | 3daf043f |
| #1369 | Satstar accept gate: mark sidelobe-only rejects that a mask can cover (opt-in) | `satstar-sidelobe-frac` | 383a1403 |
| #1376 | Opt-in bright roundness window for the m7 i2d seed (relaxed bright seeds) | `main` | 770a1a5c |
| #1377 | Add an opt-in far-companion exemption to the m7 own-band companion cut | `main` | 1553ed42 |
| #1380 | Document the footprint mask at KFAC 2-4 on five tiles (counts, masked area, profile-correction test) | `footprint-mask-edge-band` | not merged |
| #1383 | End-to-end o045 test of the stacked opt-in PRs (documentation) | `main` | not merged |

Two merge conflicts arise in the integration tree and are resolved as follows. #1367 and #1374 extend the same builder of the per-frame single-exposure masks: both sides are kept, and each switch keeps its own variable and its default of off. #1368 and #1371 both append cache-signature parts and rows of the environment table in `SATURATED_PIXEL_HANDLING.md`: both sides are kept, with the parts in the order `mi`, `eb`, `sf`. The resolutions exist in the local integration tree only (`results/integration_tree.txt` lists its merge commits).

## Setup of the run

**Inputs.** The production tree `/orange/adamginsburg/jwst/gc-treasury` is read only. Each tile has a scratch base path `<scratch>/e2e5/<tile>/gc-treasury` (the base-path override of the pipeline; the path holds no underscore). For each band it holds symlinks to the production per-frame inputs (destreaked crf, cal and ramp files), to the data mosaics and to the m6 residual mosaics. It holds copies of the m6 smoothed-background mosaics, of the m6 per-module catalogs with their vetted tables, of the m2 consensus tables, of the VIRAC2 reference catalog of the tile, of the m2 to m6 astrometry records, of the offsets table and of the PSF grids. Not staged are every m7 product, the cross-band seed tables, the consolidated satstar catalogs and every production per-frame satstar product, so the run builds them itself. A tile has 60 frames, 48 in F212N and 12 in F480M. The 8 tasks of the fan-out array share them round-robin, and every shard rebuilds the cross-band seed of the tile from the m6 catalogs of both bands.

**Jobs.** Per tile, one fan-out array of 8 tasks (2 CPUs and 32 GB per task), one finalize job (4 CPUs, 128 GB; starts after the fan-out ended without error) and one post job (5 CPUs, 96 GB; starts after the finalize job ended without error). All jobs ran on partition `hpg-dev` with account `astronomy-dept` and QOS `astronomy-dept-b`. The table is generated from `sacct` (`results/run5_jobtable.txt`).

| job name | completed jobs | all COMPLETED 0:0 | CPUs per job | requested memory | elapsed per job (h) | peak RSS (GB) | first start | last end |
| :--- | ---: | :--- | ---: | :--- | ---: | ---: | :--- | :--- |
| `e5f-o105-fin` | 1 | yes | 4 | 128G | 3.14 | 17.3 | 2026-11-27T02:29:41 | 2026-11-27T05:37:57 |
| `e5f-o118-fin` | 1 | yes | 4 | 128G | 1.50 | 12.3 | 2026-11-27T01:23:01 | 2026-11-27T02:52:50 |
| `e5f-o105-fan` | 8 | yes | 2 | 32G | 1.64 to 3.01 | 17.7 | 2026-11-26T23:28:54 | 2026-11-27T02:29:39 |
| `e5f-o118-fan` | 8 | yes | 2 | 32G | 0.91 to 1.90 | 17.4 | 2026-11-26T23:28:54 | 2026-11-27T01:22:58 |
| `e5f-o130-fin` | 1 | yes | 4 | 128G | 4.69 | 17.8 | 2026-11-27T01:18:52 | 2026-11-27T06:00:14 |
| `e5f-o138-fin` | 1 | yes | 4 | 128G | 1.85 | 12.6 | 2026-11-27T02:04:30 | 2026-11-27T03:55:44 |
| `e5f-o130-fan` | 8 | yes | 2 | 32G | 0.90 to 1.83 | 17.2 | 2026-11-26T23:28:54 | 2026-11-27T01:18:49 |
| `e5f-o138-fan` | 8 | yes | 2 | 32G | 1.23 to 2.59 | 17.5 | 2026-11-26T23:28:54 | 2026-11-27T02:04:27 |
| `e5p-o105` | 1 | yes | 5 | 96G | 0.53 | 13.6 | 2026-11-27T05:37:59 | 2026-11-27T06:09:30 |
| `e5p-o118` | 1 | yes | 5 | 96G | 0.20 | 10.8 | 2026-11-27T02:52:55 | 2026-11-27T03:04:53 |
| `e5p-o130` | 1 | yes | 5 | 96G | 0.20 | 12.9 | 2026-11-27T06:00:17 | 2026-11-27T06:12:19 |
| `e5p-o138` | 1 | yes | 5 | 96G | 0.24 | 13.7 | 2026-11-27T03:55:48 | 2026-11-27T04:09:57 |

jobs in total: 40 ; all COMPLETED 0:0: True

Slurm ids. o105: fan-out 52980462, finalize 52980463, post 52980507. o118: 52980465, 52980466, 52980509. o130: 52980467, 52980469, 52980510. o138: 52980470, 52980472, 52980512. The fan-out arrays and finalize jobs were submitted between 2026-11-26 23:28:52 and 23:28:53, the post jobs at 23:28:59 and 23:29:00. All 40 jobs ended COMPLETED with exit code 0:0.

**Code.** The jobs run the integration tree: `origin/main` at 803d0794 with the 17 code PRs of the Recipe merged in the stated order (173 files changed, 25741 insertions and 82 deletions against 803d0794; `results/integration_tree.txt`). The latest production m7 fan-out of each tile (jobs 50037748 for o105, 50048240 for o118, 50244406 for o130 and 50639978 for o138) prints in its first task the pipeline root of a pinned production tree at commit 921d1ef9, which is 74 commits behind 803d0794 and an ancestor of it. The production shard log of o105 also prints the pipeline tag `2026-11-03_PR1266`. A difference between a render of this run and the production render therefore combines the effect of the switches with the effect of these 74 commits; no control arm separates the two on these four tiles.

**Checks of the products.** The precheck step of each post job confirmed that every product of the run and of production that the later steps read exists (the F480M precheck file of each tile is empty). The jobs write their products under the scratch base path and set `CRDS_READONLY_CACHE=1`. After the run, a read-only check of the production tree (`results/run5_orange_check.txt`, run from 2026-11-27 08:13 to 08:38 EST) used `lfs find` with `--maxdepth 1 --newermt` and the marker 2026-11-26 23:28:52, the submission time of the first job. It lists 0 entries newer than the marker in each of six directories: `offsets`, `psfs`, `astrometry_checkpoints`, `catalogs`, `F480M/pipeline` and `F212N/pipeline`. The 792 symlinks of the staged inputs of the four tiles (153 per tile in F212N and 45 per tile in F480M) point to targets with a newest modification time of 2026-11-10 10:57:23, and none of the targets was modified after the marker. The check covers the entries directly inside the six directories, where a subdirectory is listed when its own modification time is later, and the targets of the symlinks. It does not open subdirectories.

## Counts per tile

Whole-star leftovers and big craters in the residual i2d mosaics of the merged module (data minus merged-catalog model), per tile and render. The columns "leftovers ≥N" count whole-star leftovers with a residual peak of at least N MJy/sr, and the columns "craters ≥N" count big craters of depth at least N MJy/sr (definitions under "How the counts are made"). The column "masked %" is the share of the finite pixels of the data mosaic that are NaN in the residual (section "Masked area"). The renders are `prod5` (production), `optnm` (catalogs of this run, no mask) and `optfin` (catalogs of this run, real footprint masks). The sums over the four tiles are the last three rows of each table.

**F212N**

| tile | render | what it is | leftovers ≥50 | ≥100 | ≥200 | ≥500 | ≥1000 | craters ≥50 | ≥200 | ≥500 | masked % of finite px |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| o105 | `prod5` | production render of the tile (no mask) | 845 | 317 | 178 | 133 | 116 | 489 | 166 | 71 | 0 |
| o105 | `optnm` | opt arm, unmasked re-render of the finalize catalogs | 610 | 118 | 38 | 13 | 8 | 102 | 44 | 18 | 0.00 |
| o105 | `optfin` | opt arm, finalize render with the real masks | 475 | 52 | 8 | 0 | 0 | 10 | 0 | 0 | 13.78 |
| o118 | `prod5` | production render of the tile (no mask) | 199 | 105 | 82 | 72 | 65 | 308 | 93 | 33 | 0 |
| o118 | `optnm` | opt arm, unmasked re-render of the finalize catalogs | 92 | 18 | 7 | 3 | 1 | 74 | 26 | 2 | 0.00 |
| o118 | `optfin` | opt arm, finalize render with the real masks | 79 | 16 | 4 | 1 | 0 | 5 | 0 | 0 | 5.46 |
| o130 | `prod5` | production render of the tile (no mask) | 181 | 115 | 106 | 90 | 84 | 206 | 58 | 28 | 0 |
| o130 | `optnm` | opt arm, unmasked re-render of the finalize catalogs | 73 | 16 | 13 | 6 | 3 | 56 | 14 | 3 | 0.00 |
| o130 | `optfin` | opt arm, finalize render with the real masks | 52 | 3 | 1 | 1 | 1 | 5 | 0 | 0 | 4.54 |
| o138 | `prod5` | production render of the tile (no mask) | 472 | 165 | 101 | 89 | 77 | 379 | 126 | 52 | 0 |
| o138 | `optnm` | opt arm, unmasked re-render of the finalize catalogs | 330 | 57 | 28 | 20 | 13 | 90 | 28 | 11 | 0.00 |
| o138 | `optfin` | opt arm, finalize render with the real masks | 242 | 14 | 1 | 0 | 0 | 8 | 0 | 0 | 7.81 |
| all four | `prod5` | sum over the four tiles | 1697 | 702 | 467 | 384 | 342 | 1382 | 443 | 184 | |
| all four | `optnm` | sum over the four tiles | 1105 | 209 | 86 | 42 | 25 | 322 | 112 | 34 | |
| all four | `optfin` | sum over the four tiles | 848 | 85 | 14 | 2 | 1 | 28 | 0 | 0 | |

**F480M**

| tile | render | what it is | leftovers ≥50 | ≥100 | ≥200 | ≥500 | ≥1000 | craters ≥50 | ≥200 | ≥500 | masked % of finite px |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| o105 | `prod5` | production render of the tile (no mask) | 303 | 202 | 168 | 111 | 84 | 69 | 28 | 16 | 0 |
| o105 | `optnm` | opt arm, unmasked re-render of the finalize catalogs | 221 | 155 | 146 | 104 | 78 | 20 | 12 | 8 | 0.00 |
| o105 | `optfin` | opt arm, finalize render with the real masks | 49 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 23.33 |
| o118 | `prod5` | production render of the tile (no mask) | 219 | 121 | 76 | 47 | 38 | 26 | 6 | 5 | 0 |
| o118 | `optnm` | opt arm, unmasked re-render of the finalize catalogs | 118 | 71 | 54 | 45 | 36 | 5 | 4 | 3 | 0.00 |
| o118 | `optfin` | opt arm, finalize render with the real masks | 48 | 10 | 1 | 0 | 0 | 0 | 0 | 0 | 11.43 |
| o130 | `prod5` | production render of the tile (no mask) | 191 | 90 | 50 | 19 | 14 | 39 | 12 | 1 | 0 |
| o130 | `optnm` | opt arm, unmasked re-render of the finalize catalogs | 85 | 28 | 19 | 14 | 11 | 1 | 0 | 0 | 0.00 |
| o130 | `optfin` | opt arm, finalize render with the real masks | 56 | 7 | 1 | 0 | 0 | 0 | 0 | 0 | 8.36 |
| o138 | `prod5` | production render of the tile (no mask) | 185 | 105 | 64 | 44 | 37 | 29 | 6 | 2 | 0 |
| o138 | `optnm` | opt arm, unmasked re-render of the finalize catalogs | 91 | 45 | 38 | 36 | 33 | 0 | 0 | 0 | 0.00 |
| o138 | `optfin` | opt arm, finalize render with the real masks | 40 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 10.19 |
| all four | `prod5` | sum over the four tiles | 898 | 518 | 358 | 221 | 173 | 163 | 52 | 24 | |
| all four | `optnm` | sum over the four tiles | 515 | 299 | 257 | 199 | 158 | 26 | 16 | 11 | |
| all four | `optfin` | sum over the four tiles | 193 | 23 | 2 | 0 | 0 | 0 | 0 | 0 | |

## Classes of the leftovers of 200 MJy/sr and more

Class of each whole-star leftover with a residual peak ≥200 MJy/sr (first match wins; SATSTAR_NEAR = a consolidated satstar fit within 1.0 arcsec, VETTED_BUT_UNSUBTRACTED = a vetted m7 row within the larger of 1 FWHM and 0.06 arcsec, VETTED_OUT = a row of the raw m7 catalog within that radius, NOT_IN_M7_CATALOG = none of these). The classes of a render use the catalogs of the tree that made it, so the classes of `prod5` use the production catalogs.

**F212N**

| tile | render | SATSTAR_NEAR | VETTED_BUT_UNSUBTRACTED | VETTED_OUT | NOT_IN_M7_CATALOG | total |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| o105 | `prod5` | 122 | 22 | 10 | 24 | 178 |
| o105 | `optnm` | 27 | 3 | 1 | 7 | 38 |
| o105 | `optfin` | 3 | 1 | 0 | 4 | 8 |
| o118 | `prod5` | 22 | 14 | 24 | 22 | 82 |
| o118 | `optnm` | 4 | 2 | 0 | 1 | 7 |
| o118 | `optfin` | 2 | 0 | 0 | 2 | 4 |
| o130 | `prod5` | 30 | 21 | 24 | 31 | 106 |
| o130 | `optnm` | 8 | 3 | 0 | 2 | 13 |
| o130 | `optfin` | 0 | 1 | 0 | 0 | 1 |
| o138 | `prod5` | 54 | 15 | 13 | 19 | 101 |
| o138 | `optnm` | 25 | 0 | 0 | 3 | 28 |
| o138 | `optfin` | 0 | 0 | 0 | 1 | 1 |
| all four | `prod5` | 228 | 72 | 71 | 96 | 467 |
| all four | `optnm` | 64 | 8 | 1 | 13 | 86 |
| all four | `optfin` | 5 | 2 | 0 | 7 | 14 |

**F480M**

| tile | render | SATSTAR_NEAR | VETTED_BUT_UNSUBTRACTED | VETTED_OUT | NOT_IN_M7_CATALOG | total |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| o105 | `prod5` | 140 | 7 | 0 | 21 | 168 |
| o105 | `optnm` | 134 | 1 | 0 | 11 | 146 |
| o105 | `optfin` | 0 | 0 | 0 | 0 | 0 |
| o118 | `prod5` | 44 | 7 | 2 | 23 | 76 |
| o118 | `optnm` | 48 | 0 | 0 | 6 | 54 |
| o118 | `optfin` | 0 | 0 | 0 | 1 | 1 |
| o130 | `prod5` | 13 | 14 | 1 | 22 | 50 |
| o130 | `optnm` | 16 | 2 | 0 | 1 | 19 |
| o130 | `optfin` | 0 | 1 | 0 | 0 | 1 |
| o138 | `prod5` | 37 | 8 | 1 | 18 | 64 |
| o138 | `optnm` | 38 | 0 | 0 | 0 | 38 |
| o138 | `optfin` | 0 | 0 | 0 | 0 | 0 |
| all four | `prod5` | 234 | 36 | 4 | 84 | 358 |
| all four | `optnm` | 236 | 3 | 0 | 18 | 257 |
| all four | `optfin` | 0 | 1 | 0 | 1 | 2 |

## Matching of the leftovers and craters between renders

For each pair of renders A → B, each leftover ≥100 MJy/sr of A is matched to a leftover of B within 3 px (0.094 arcsec) and counted as cleared (no match; this includes a leftover inside a region that the mask of B sets to NaN), reduced (B peak below 0.5 times the A peak), unchanged, or worse (B peak above 1.5 times the A peak). "Listed as new" counts the leftovers ≥100 MJy/sr of B without a counterpart in A, and "of those new, SATSTAR_NEAR" the number of them that are SATSTAR_NEAR. "Cleared of ≥500" is the number of leftovers ≥500 MJy/sr of A that are cleared. The crater columns give the big craters of depth ≥50 MJy/sr in A and in B, those of A without a counterpart in B (gone), and those of B without a counterpart in A (listed as new). A feature whose extremum moves by more than 3 px between the renders counts once as cleared (or gone) and once as listed as new.

**F212N**

| A → B | tile | leftovers ≥100 in A | in B | cleared | reduced | unchanged | worse | listed as new | of those new, SATSTAR_NEAR | cleared of ≥500 | craters ≥50 in A → B | gone | listed as new |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- | ---: | ---: |
| `prod5` → `optfin` | o105 | 317 | 52 | 267 | 0 | 50 | 0 | 2 | 0 | 133 of 133 | 489 → 10 | 485 | 6 |
| `prod5` → `optfin` | o118 | 105 | 16 | 95 | 0 | 10 | 0 | 6 | 2 | 71 of 72 | 308 → 5 | 305 | 2 |
| `prod5` → `optfin` | o130 | 115 | 3 | 112 | 0 | 3 | 0 | 0 | 0 | 89 of 90 | 206 → 5 | 203 | 2 |
| `prod5` → `optfin` | o138 | 165 | 14 | 149 | 0 | 16 | 0 | 0 | 0 | 89 of 89 | 379 → 8 | 375 | 4 |
| `prod5` → `optfin` | all four | 702 | 85 | 623 | 0 | 79 | 0 | 8 | 2 | 382 of 384 | 1382 → 28 | 1368 | 14 |
| `prod5` → `optnm` | o105 | 317 | 118 | 206 | 0 | 111 | 0 | 6 | 6 | 121 of 133 | 489 → 102 | 396 | 9 |
| `prod5` → `optnm` | o118 | 105 | 18 | 87 | 0 | 18 | 0 | 0 | 0 | 69 of 72 | 308 → 74 | 242 | 8 |
| `prod5` → `optnm` | o130 | 115 | 16 | 100 | 0 | 15 | 0 | 1 | 1 | 84 of 90 | 206 → 56 | 154 | 4 |
| `prod5` → `optnm` | o138 | 165 | 57 | 114 | 0 | 51 | 0 | 8 | 8 | 73 of 89 | 379 → 90 | 301 | 12 |
| `prod5` → `optnm` | all four | 702 | 209 | 507 | 0 | 195 | 0 | 15 | 15 | 347 of 384 | 1382 → 322 | 1093 | 33 |
| `optnm` → `optfin` | o105 | 118 | 52 | 68 | 0 | 50 | 0 | 2 | 0 | 13 of 13 | 102 → 10 | 98 | 6 |
| `optnm` → `optfin` | o118 | 18 | 16 | 8 | 0 | 10 | 0 | 6 | 2 | 2 of 3 | 74 → 5 | 71 | 2 |
| `optnm` → `optfin` | o130 | 16 | 3 | 13 | 0 | 3 | 0 | 0 | 0 | 5 of 6 | 56 → 5 | 53 | 2 |
| `optnm` → `optfin` | o138 | 57 | 14 | 43 | 0 | 14 | 0 | 0 | 0 | 20 of 20 | 90 → 8 | 85 | 3 |
| `optnm` → `optfin` | all four | 209 | 85 | 132 | 0 | 77 | 0 | 8 | 2 | 40 of 42 | 322 → 28 | 307 | 13 |

**F480M**

| A → B | tile | leftovers ≥100 in A | in B | cleared | reduced | unchanged | worse | listed as new | of those new, SATSTAR_NEAR | cleared of ≥500 | craters ≥50 in A → B | gone | listed as new |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- | ---: | ---: |
| `prod5` → `optfin` | o105 | 202 | 2 | 200 | 0 | 2 | 0 | 0 | 0 | 111 of 111 | 69 → 0 | 69 | 0 |
| `prod5` → `optfin` | o118 | 121 | 10 | 111 | 0 | 10 | 0 | 0 | 0 | 47 of 47 | 26 → 0 | 26 | 0 |
| `prod5` → `optfin` | o130 | 90 | 7 | 83 | 0 | 7 | 0 | 0 | 0 | 19 of 19 | 39 → 0 | 39 | 0 |
| `prod5` → `optfin` | o138 | 105 | 4 | 101 | 0 | 4 | 0 | 0 | 0 | 44 of 44 | 29 → 0 | 29 | 0 |
| `prod5` → `optfin` | all four | 518 | 23 | 495 | 0 | 23 | 0 | 0 | 0 | 221 of 221 | 163 → 0 | 163 | 0 |
| `prod5` → `optnm` | o105 | 202 | 155 | 62 | 0 | 140 | 0 | 15 | 13 | 17 of 111 | 69 → 20 | 59 | 10 |
| `prod5` → `optnm` | o118 | 121 | 71 | 58 | 0 | 63 | 0 | 8 | 8 | 9 of 47 | 26 → 5 | 21 | 0 |
| `prod5` → `optnm` | o130 | 90 | 28 | 68 | 0 | 22 | 0 | 6 | 6 | 11 of 19 | 39 → 1 | 38 | 0 |
| `prod5` → `optnm` | o138 | 105 | 45 | 66 | 0 | 39 | 0 | 6 | 6 | 14 of 44 | 29 → 0 | 29 | 0 |
| `prod5` → `optnm` | all four | 518 | 299 | 254 | 0 | 264 | 0 | 35 | 33 | 51 of 221 | 163 → 26 | 147 | 10 |
| `optnm` → `optfin` | o105 | 155 | 2 | 153 | 0 | 2 | 0 | 0 | 0 | 104 of 104 | 20 → 0 | 20 | 0 |
| `optnm` → `optfin` | o118 | 71 | 10 | 61 | 0 | 10 | 0 | 0 | 0 | 45 of 45 | 5 → 0 | 5 | 0 |
| `optnm` → `optfin` | o130 | 28 | 7 | 21 | 0 | 7 | 0 | 0 | 0 | 14 of 14 | 1 → 0 | 1 | 0 |
| `optnm` → `optfin` | o138 | 45 | 4 | 41 | 0 | 4 | 0 | 0 | 0 | 36 of 36 | 0 → 0 | 0 | 0 |
| `optnm` → `optfin` | all four | 299 | 23 | 276 | 0 | 23 | 0 | 0 | 0 | 199 of 199 | 26 → 0 | 26 | 0 |

## Features of the unmasked render that exceed production

The position-independent scan compares every leftover and every crater of 200 MJy/sr and more in a render with the extremum of the production render in the 13 × 13 pixel box centred on it (definition under "How the counts are made"). The unmasked render `optnm` has 257 such leftovers and 16 such craters in F480M, and 86 leftovers and 112 craters in F212N. The scan finds 5 features that are absent in production or larger than in production; the table lists them. Production is `prod5`; the scan files are `results/<tile>_regress_scan_nm_<band>.txt` and `results/run5_larger_than_prod.txt`.

| tile | band | kind | pixel (x, y) | extremum in the run (MJy/sr) | production extremum in the box (MJy/sr) | verdict | nearest model rows |
| :--- | :--- | :--- | :--- | ---: | ---: | :--- | :--- |
| o105 | F212N | crater | (8306, 1270) | 1590 | 971 | larger than in production (ratio 1.64) | vetted row 0.031 arcsec away, flagged saturated, flux 49769 |
| o105 | F212N | crater | (4469, 1280) | 755 | 456 | larger than in production (ratio 1.65) | vetted row 0.032 arcsec away, flagged saturated, flux 28265 |
| o138 | F212N | leftover | (8633, 2587) | 2209 | 1297 | larger than in production (ratio 1.70) | SATSTAR_NEAR, satstar fit 0.095 arcsec away |
| o105 | F480M | leftover | (2614, 1826) | 904 | 29 | absent in production | SATSTAR_NEAR, satstar fit 0.384 arcsec away |
| o105 | F480M | leftover | (4452, 858) | 403 | 30 | absent in production | NOT_IN_M7_CATALOG, nearest satstar fit 3.905 arcsec away |

The scans of the masked renders `optfin` find no leftover or crater that is absent in production or larger than in production, in either band.

The scan also runs in the other direction, from the features of production to the render of the run. In F212N, the unmasked render `optnm` is below the threshold and more than a factor 1.5 lower than production at 130 of the 178 production leftovers ≥200 MJy/sr of o105, 74 of 82 (o118), 91 of 106 (o130) and 73 of 101 (o138). In F480M the unmasked render is that much lower at 22 of 168 (o105), 24 of 76 (o118), 29 of 50 (o130) and 22 of 64 (o138) production leftovers. The masked render `optfin` lowers the production leftovers ≥200 MJy/sr by more than a factor 1.5 and below the threshold in 168 of 178 (o105), 79 of 82 (o118), 105 of 106 (o130) and 100 of 101 (o138) cases in F212N, and in 167 of 168 (o105), 74 of 76 (o118), 49 of 50 (o130) and 64 of 64 (o138) cases in F480M.

The next section follows the five features to their cause.

## Cause of the five features that exceed production

The five features of the table above are numbered in its order. Features 1 and 2 are the o105 F212N craters at (8306, 1270) and (4469, 1280). Feature 3 is the o138 F212N leftover at (8633, 2587). Features 4 and 5 are the o105 F480M leftovers at (2614, 1826) and (4452, 858).

The trace used one-frame replays. A replay runs the m7 per-frame step on one exposure, or for feature 3 and for the star gate of feature 5 the saturated-star fit alone, and writes the per-frame table of that exposure. The tables of different arms are compared row by row. The exposures are nrcb4 exposure 1 (feature 1), nrca3 exposure 6 (feature 2), nrcb3 exposure 2 (feature 3), nrcalong exposure 1 (feature 4), nrcblong exposure 6 (the neighbour row of feature 5) and nrcblong exposure 2 (the star gate of feature 5). The replays stop at the per-frame tables. The merge, the vetting and the renders are not repeated, so every statement about merged rows uses the catalogs of the run.

| arm | code and settings |
| :--- | :--- |
| P | production code (commit 921d1ef9), production options and the F480M hand-off settings of the Recipe |
| B | integration base 803d0794, 74 commits after the production code, no code of the opt-in PRs, no switch |
| C | integration tree with the 17 PR merges, every opt-in switch off |
| T | C with the switches of the Recipe and the per-band seed options of this run; the guard of #1360 acts at the merge stage, so a per-frame table does not show it |
| S1360 to S1374 | C with the switches of one PR of the Recipe, production options |
| Cps | C with the production saturated-star products taken from the cache, so that only the DAO stage differs from P |
| Cnse | C with the ePSF core removed from the PSF grid of the saturated-star fit alone |
| Bne, Cne, Tne | the saturated-star fit alone with the ePSF core removed everywhere |
| mi, eb, mieb | the saturated-star fit alone with a maximum of 1000 iterations (#1371), with the edge position bound of 12, and with both |

**Common cause.** One commit between the production code and the integration base sits behind all five features. Commit 018eb243 (#1208 and #1331, 2026-11-17) applies the ePSF core to the PSF grid of the saturated-star fit through `jwst_gc_pipeline/photometry/epsf_hybrid.py::maybe_apply_epsf_core`. The commit is active whenever `PSF_EPSF_CORE_DIR` is set, as in the Recipe, and the production code ran without it. Three tests support the attribution.

- Cnse for the per-frame tables and Bne, Cne and Tne for the saturated-star fits reproduce the production values at all five targets.
- B has no code of the opt-in PRs and no switch. It gives the values of the run at features 3, 4 and 5 and the per-frame values at features 1 and 2.
- Cps equals P bit for bit in x_fit, y_fit, flux_fit, qfit, flags and forced_refit on nrcalong exposure 1 (53983 rows) and nrcblong exposure 6 (52993 rows), and B equals C bit for bit on the same two frames (54060 and 52982 rows). The DAO stage of the integration tree therefore does not change a row when it receives the production saturated-star products.

| arm | feature 1 companion | feature 2 companion | feature 3 saturated-star fit | feature 4 row | feature 5 star gate | feature 5 neighbour row |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| production table | forced, flux 23309 | forced, flux 14083 | flux 34676, flags 24 | kept, x_fit 1834.500295, flux 13850 | accepted, flux 29033, implied peak 2547 | kept, x_fit 1264.4819, flux 5233 |
| P | forced, flux 23309 | forced, flux 14083 | flux 34708.4, flags 0 | kept, x_fit 1834.5003 | accepted, flux 28987.3, implied peak 2543.4 | kept, x_fit 1264.4819, flux 5233 |
| Cps | forced, flux 23309 | forced, flux 14083 | not run | kept, x_fit 1834.5003 | not run | kept, x_fit 1264.4819, flux 5233 |
| Cnse | forced, flux 23309 | forced, flux 14083 | not run | kept, x_fit 1834.5013 | not run | kept, x_fit 1264.4842, flux 5252 |
| Bne, Cne | not run | not run | flux 34708.4, flags 0 | not run | accepted, flux 28987.3, implied peak 2543.4 | not run |
| B, C | free, flux 38983 | free, flux 18835 | flux 33392.3, flags 0 | lost | rejected, flux 28504.9, implied peak 2410.9 | lost |
| T | free, flux 38983 | free, flux 18835 | flux 33392.3, flags 0 | lost | not finished when this was written | lost |

The production table row of feature 3 carries flags 24, which are possible non-convergence and missing parameter covariance. The replay P with the production code reproduces its position and its flux to 0.09 % and gives flags 0. The single-switch arms equal C at the five targets wherever they finished. Some of them change row counts elsewhere in the frame. In nrcb4 exposure 1 the row counts are 53111 (C), 53134 (S1366), 53008 (S1371) and 53031 (T), and the rows of features 1 and 2 stay at the free fluxes 38983 and 18835. `results/run5_five_arms.txt` lists the per-row values of all arms and `results/run5_five_frames.txt` lists the per-frame rows of features 1 and 2.

Cnse reproduces the production rows at the five targets. Over the whole of nrcalong exposure 1 it still differs from P in 86 lost and 103 gained rows of 53957, against 252 lost and 330 gained for C. P differs from the production table in 65 lost and 92 gained rows in the same frame, and the origin of those rows was not traced.

**Features 1 and 2: a companion beside a saturated-star row.** The second pass of the saturated-star replacement pairs a saturated star with a daophot row within 1.5 FWHM (0.108 arcsec in F212N) and overwrites that row. The guard of #1360 keeps the daophot row and appends the saturated star as a separate row when the daophot flux lies from 0.2 up to 0.8 of the saturated-star flux. In production the star of feature 1 has one vetted row with flux 49478 and group_size 2. The run has the saturated-star row with flux 49769 and a companion row with flux 37676 at 0.0728 arcsec (flux ratio 0.757), with nmatch 6 and nmatch_good 4. The star of feature 2 has one vetted row with flux 27551 in production, and in the run a saturated-star row with flux 28265 and a companion row with flux 19314 at 0.0802 arcsec (flux ratio 0.683), with nmatch 4 and nmatch_good 2.

The 13 × 13 pixel sums of the residual images (MJy/sr times pixels) are +18872 in production and −12404 in the run for feature 1 against a data sum of 60508. The residual of production is 31.2 % of the data sum and that of the run is −20.5 % of it. Feature 2 changes from +11467 to −4990 against a data sum of 35241, which is 32.5 % and −14.2 % of the data sum.

The per-frame fits of the companion differ between production and run for feature 1. A fit is called forced when the overshoot refit replaced it by photometry at the seed position, and free otherwise. Six of the 48 frames of the tile have a row of flux 1000 or more within 0.12 arcsec of the crater position. Their fluxes are, in the order nrcb2 exposure 4, nrcb2 exposure 5 and nrcb4 exposures 1, 2, 3 and 6:

| | exp 4 (nrcb2) | exp 5 (nrcb2) | exp 1 (nrcb4) | exp 2 (nrcb4) | exp 3 (nrcb4) | exp 6 (nrcb4) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| production | 37536 free | 21023 forced | 23309 forced | 23719 forced | 23197 forced | 36196 free |
| run | 37523 free | 21354 forced | 38983 free | 37515 free | 23709 forced | 35574 free |

In nrcb4 exposure 1 the arms P, Cps and Cnse give the forced refit and B, C and T give the free fit. Exposure 2 of nrcb4 changes from forced to free between the production table and the table of the run in the same way, and it was not replayed. A replica of the merge, which applies a median-absolute-deviation sigma clip on flux, RA and Dec and takes the inverse-variance mean of the kept frames, keeps three frames of the production pattern, all forced, and gives 23.4k. It keeps four free frames of the run pattern and gives 37.0k against 37.7k in the catalog, with the logged nmatch_good of 4. A box-sum estimate puts the companion flux that the data support at about 22.5k to 22.6k, which is 60 % of the catalog value. The estimate subtracts the model sum of the saturated-star row from the data sum of the 13 × 13 box and divides by the model sum of the companion per unit flux, and it ignores background and other sources in the box. The flips move the merged companion flux from about 4 % above the estimate to about 64 % above it.

Feature 2 has the same flip in nrca3 exposure 6, with the forced flux 14083 in P, Cps and Cnse and the free flux 18835 in B, C and T. The merged companion flux changes little, because the other three frames are free at 18.78k to 19.24k in production and at 18.71k to 18.87k in the run. The replica gives 19.0k for the production pattern and 18.9k for the run pattern, against 19314 in the catalog. The estimate of the data-supported companion flux is about 13k, which is 69 % of the catalog value. In the replica of the merge, feature 2 does not depend on the flip. It follows from the guard keeping a companion whose catalog flux exceeds the data-supported flux. Feature 1 needs the guard and the flip, because the merged companion flux is 37.7k with the flip and 23.4k in the replica of the production pattern.

The render of the integration tree with the guard off and the code change on is not part of this section. The control arm of o105, with every switch off, is that render.

**Guard population.** The vetted F212N catalogs of the four tiles hold 72 pairs that the guard keeps, which are 50 in o105, 7 in o118, 7 in o130 and 8 in o138. A kept pair is an appended saturated-star row and a daophot row within 0.108 arcsec that the replacement did not overwrite, with a flux of 0.2 to 0.8 of the saturated-star flux. Two of the windows of 7 × 7 pixels around a pair hold two pairs, so the pairs fall into 70 windows. Of the 70 windows, the maximum residual is 500 MJy/sr or more in 65 windows of the production render and 7 of the run, and the minimum residual is −500 MJy/sr or less in 33 and 42. Over all windows, the maximum residual drops by 97960 MJy/sr in sum, and the crater deepens by 12986 MJy/sr in sum, taking the positive part of the production minimum minus the run minimum in each window. The ratio of the two sums is 7.5 to 1. A window counts as a deepened crater when the run minimum is 200 MJy/sr or deeper and more than 1.25 times as deep as the production minimum. There are 28 of them.

| windows with | count | drop of the maximum, MJy/sr | deepening of the minimum, MJy/sr | deepened craters |
| :--- | ---: | ---: | ---: | ---: |
| flux ratio 0.2 to 0.5 | 48 | 70080 | 6684 | 13 |
| flux ratio 0.5 to 0.8 | 22 | 27880 | 6302 | 15 |
| separation below 0.07 arcsec | 4 | | | 2 |
| separation 0.07 to 0.08 arcsec | 19 | | | 12 |
| separation 0.08 to 0.09 arcsec | 23 | | | 8 |
| separation 0.09 to 0.10 arcsec | 13 | | | 2 |
| separation 0.10 to 0.108 arcsec | 11 | | | 4 |

The two features sit in the corner of high flux ratio and small separation. Their crater deepening of 612 MJy/sr (feature 1) and 292 MJy/sr (feature 2) ranks 4th and 12th of the 70 windows. The largest deepening is 2306 MJy/sr at o138 (3213, 4576), where the production minimum is −11587 MJy/sr and the run minimum is −13893 MJy/sr. An upper bound of 0.5 on the flux ratio would avoid the 15 deepened craters of the second row, with 3785 of the 12986 MJy/sr of deepening, and would give up 27880 of the 97960 MJy/sr of removed maximum residual. `results/run5_guard_pairs.txt` lists the 72 pairs.

**Feature 3: the position of the saturated star moved by the core.** Saturated-star replays of nrcb3 exposure 2 give a flux of 34708.4 and a qfit of 2.867 in P, Bne and Cne, and a flux of 33392.3 and a qfit of 2.721 in B, C, T, mi, eb and mieb. The fitted sky position differs by +0.0842 arcsec in RA times cos(Dec) and −0.0007 arcsec in Dec between the two groups, which is 2.7 pixels. B has no code of the opt-in PRs, and the other arms with the core give the same fit as C, so no switch is involved. The distances of the three frames of the star from the leftover position are 0.055, 0.028 and 0.027 arcsec in production, and 0.030, 0.028 and 0.027 arcsec in the run. With the core the three frames agree. The merged rows differ little in flux: production has the vetted row 171273 with flux 32640 and group_size 3, and the run has row 171449 with flux 32497. The 13 × 13 pixel sums of the model are 27280 in production and 27292 in the run. The data have two columns of 1352 to 2317 MJy/sr at x 8632 and 8633 from y 2586 up to the edge of the 9 × 9 pixel window at y 2591. The model of production reaches 1020 MJy/sr at (8633, 2587) and the model of the run reaches 101 MJy/sr, so the leftover of 2209 MJy/sr is a mismatch between the core of the model and the saturated streak. The flux sums of the model agree, and the difference lies in the position of the model core.

**Feature 4: an integer-pixel threshold.** The production row is a detection in one frame, nrcalong exposure 1, with flux 13850 and flags 1. P and Cps keep it at x_fit 1834.500295. Cnse keeps it at x_fit 1834.5013. B, C, T and every single-switch arm that ran lose it in `jwst_gc_pipeline/photometry/crowdsource_catalogs_long.py::_filter_near_saturation`. That filter rounds the fit position to a pixel and drops the row when the distance from that pixel to the nearest saturated pixel is 1.0 pixel or less. In the saturation flags of the frame, pixel (1834, 1633) has distance 1.000 pixel to the saturated pixel (1834, 1632), and pixel (1835, 1633) has 1.414 pixels. The rounding of 1834.500295 gives 1835 and the row stays. Any x_fit of 1834.5 or below rounds to 1834 and the row goes, because numpy rounds 1834.5 to the even integer 1834. P, Cps and Cnse lie 0.0003 to 0.0013 pixel above the edge. The arms with the core print the position as (1834.50, 1633.17) to two decimals and lose the row, so their x_fit is 1834.5 or below. The exact value was not recorded. The accepted saturated star 6 pixels away has flux 113163 in production and 111806 in the run and passes the gate in both, so the gate is not involved.

The exposure of the filter to such shifts was counted on the P tables. In nrcalong exposure 1, 92 kept rows have a distance above 1.0 and up to 2.5 pixels at their rounded pixel, and 1, 1, 1, 2 and 6 of them would drop for a shift of at most 0.001, 0.005, 0.02, 0.05 and 0.1 pixel in x and y. In nrcblong exposure 6 the counts are 241 rows and 0, 1, 4, 9 and 18.

**Feature 5: a gate and the same threshold.** The leftover lies at a neighbour row of a bright star 0.32 arcsec away. In production the star is an accepted saturated-star fit with the consolidated flux 28728 in four frames, and the neighbour is a vetted row with flux 5232.8 and flags 1 from nrcblong exposure 6 alone, with group_size 3, x_fit 1264.4819 and y_fit 634.4405.

The saturated-star fit is rejected when its implied peak is below half the saturation floor, which is 2500 in F480M for a floor of 5000, and the observed peak is also below 2500. Replays of nrcblong exposure 2 give an implied peak of 2543.4 and a flux of 28987.3 without the core, and the star is accepted. With the core they give 2410.9 and 28504.9, and the star is rejected. The observed peak is 1959.1. The core lowers the implied peak of this star by 5.2 %. The fitted flux falls by 1.7 % and the implied peak per unit flux falls by 3.6 %, from 0.08774 to 0.08458. A rejected star goes to the daophot channel, which fits it with flux 23598 at (1268.9689, 632.6811) in B, C, T and the single-switch arms. P, Cps and Cnse have no row there. The merged catalog of the run has the star as daophot row 100654 with flux 23268, flagged as a rejected saturated star, against the production flux of 28728.

With the star in the daophot group, the group fit moves the neighbour to an x_fit of 1264.5 or more. It rounds to pixel (1265, 634), which has distance 1.0 pixel to the saturated blob whose edge is at x = 1266, and the filter of feature 4 drops it. Pixel (1264, 634) has distance 2.0 pixels. Arm C prints the rounded pixel 1265 for the neighbour, with flux 5361, qfit 0.049 and flags 0, and the exact x_fit was not recorded. The residual of 403 MJy/sr is at the neighbour row, 0.028 arcsec from the leftover position.

The same gate acts on the whole table. In the satstar tables of nrcblong exposure 2 of o105, 734 rows of production and run match within 0.05 arcsec as mutual nearest neighbours. The ratio of the implied peaks of the run to production has the median 0.9550 and the 16th to 84th percentile range 0.9359 to 0.9674. The ratio of the flux has the median 0.9844 with range 0.9625 to 0.9959, and the ratio of the PSF peak per unit flux has the median 0.9707 with range 0.9676 to 0.9742. Of the 24 production rows with an implied peak from 2500 to 2700, all are accepted in production. 23 of them have a match in the run and 13 of those are accepted in the run. Over the four tiles, 61 consolidated F480M saturated-star rows of production have no row of the run within 0.15 arcsec, which are 19 in o105, 11 in o118, 16 in o130 and 15 in o138. Their median flux is 28.4k to 28.7k in each tile, at the gate boundary. The run has 4, 2, 6 and 3 rows with no production row within 0.15 arcsec. In the frames the gate hands more stars to daophot, 105 to 111 in nrcalong exposure 1 and 183 to 193 in nrcblong exposure 6. `results/run5_five_gate.txt` lists the numbers of this paragraph.

The T arm of the gate replay had not finished when this was written. The arms mi and eb act inside the saturated-star fit and equal C.

**Fixes that these results suggest.** None of them is tested.

- Features 1 and 2: cap the kept companion flux at the data budget of the box, restore the forced refit when a companion lies within 0.1 arcsec of an accepted saturated star, or lower the upper bound of the guard window from 0.8 to 0.5. An upper bound of 0.5 removes both features from the window and would bring back the leftovers of the production render, with peaks of 2741 and 1417 MJy/sr in the 9 × 9 pixel windows. That outcome was not rendered.
- Feature 3: flag a star whose frames disagree in position by more than 0.5 pixel, or use a core that follows a saturated streak.
- Features 4 and 5: replace the integer rounding in the near-saturation filter by the distance from the sub-pixel position to the nearest saturated pixel centre with a threshold raised by 0.5 pixel, or add a hysteresis band of 0.1 pixel around the half-pixel boundary, or exempt rows whose qfit and local support pass floors.
- Feature 5 gate: make the gate fraction an environment switch with a lower default such as 0.45, or scale the gate by the ratio of the PSF peaks per unit flux with and without the core, 0.971. The alternative is to remove the core from the F480M saturated-star PSF.

The core is the intended behaviour of #1208 and #1331, and with it the three frames of feature 3 agree. The choice between recalibrating the gate and removing the core for F480M is therefore a decision about the gate and the intended core.

The masked renders hide all five features. The 9 × 9 pixel windows of features 1 to 4 are entirely NaN in the masked render, and the window of feature 5 has 68 of 81 pixels NaN with a maximum of 40 MJy/sr in the remaining pixels.

## Rows that production has and the run lacks

**Catalog rows.** The production merged vetted m7 catalog of a tile and band (`<production>/catalogs`) and the merged vetted m7 catalog of the run (`<scratch>/e2e5/<tile>/gc-treasury/catalogs`) are matched by nearest neighbour on the sky within 0.1 arcsec in both directions. A production row without a row of the run within 0.1 arcsec counts as lost, and a row of the run without a production row within 0.1 arcsec counts as extra. The flux is the `flux` column of the table of the row. The script writes `results/run5_catdiff.txt`.

| tile | band | production rows | rows of the run | production rows without a row of the run | rows of the run without a production row | production rows without a run row, flux 1e3 to 1e4 | 1e4 to 3e4 | 3e4 to 1e5 | ≥1e5 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| o105 | F212N | 522372 | 523550 | 1440 | 2392 | 74 | 4 | 0 | 0 |
| o118 | F212N | 298329 | 298583 | 621 | 809 | 13 | 2 | 0 | 0 |
| o130 | F212N | 252457 | 252597 | 522 | 658 | 10 | 1 | 0 | 0 |
| o138 | F212N | 474369 | 475021 | 917 | 1498 | 61 | 2 | 1 | 0 |
| all four | F212N | 1547527 | 1549751 | 3500 | 5357 | 158 | 9 | 1 | 0 |
| o105 | F480M | 145740 | 146308 | 733 | 1292 | 14 | 4 | 0 | 0 |
| o118 | F480M | 139908 | 140272 | 481 | 842 | 12 | 0 | 0 | 1 |
| o130 | F480M | 139868 | 140067 | 552 | 729 | 11 | 0 | 2 | 1 |
| o138 | F480M | 160045 | 160071 | 763 | 784 | 21 | 0 | 0 | 0 |
| all four | F480M | 585561 | 586718 | 2529 | 3647 | 58 | 4 | 2 | 2 |

| tile | band | run rows without a production row, flux 1e3 to 1e4 | 1e4 to 3e4 | 3e4 to 1e5 | ≥1e5 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| o105 | F212N | 214 | 22 | 16 | 0 |
| o118 | F212N | 61 | 8 | 4 | 0 |
| o130 | F212N | 42 | 5 | 1 | 0 |
| o138 | F212N | 151 | 10 | 6 | 0 |
| all four | F212N | 468 | 45 | 27 | 0 |
| o105 | F480M | 171 | 12 | 2 | 0 |
| o118 | F480M | 43 | 3 | 0 | 1 |
| o130 | F480M | 23 | 0 | 3 | 3 |
| o138 | F480M | 19 | 0 | 0 | 2 |
| all four | F480M | 256 | 15 | 5 | 6 |

In F212N, 10 production rows with a flux ≥1e4 have no row in the run (9 between 1e4 and 3e4, one between 3e4 and 1e5). In F480M the number is 8 (4 between 1e4 and 3e4, 2 between 3e4 and 1e5 and 2 at ≥1e5). The run has more rows than production in every tile and band. The second table counts the rows of the run without a production row; 72 of them in F212N (45 between 1e4 and 3e4 and 27 between 3e4 and 1e5) and 26 in F480M (15, 5 and 6 in the three bins from 1e4 up) have a flux ≥1e4.

**One-frame replays.** To separate the roles of the code, the switches and the two hand-off settings, the m7 per-frame DAO pass of two frames of the o105 F480M production run was replayed on scratch copies with `<script>` (it reads the production tree and writes only to scratch; the frame, its production satstar products, the smoothed-background mosaic and the band seed are read, and the options of the frame are parsed from the production shard log of job 50037748, task 5). Frame A is exposure 1 of module nrcblong (`jw10678105001_02101_00001_nrcblong`) and frame B is exposure 2 of module nrcalong (`jw10678105001_02101_00002_nrcalong`). Each replay is compared with the production per-frame table of the frame (`f480m_nrcblong_o105_visit001_vgroup02101_exp00001_resbgsub_hybpsf_m7_daophot_basic.fits` and the nrcalong exp00002 table) by `x_fit` and `y_fit` within 0.5 px. A production row with `flux_fit` above the threshold and no replay row counts as lost, and a replay row with `flux_fit` above the threshold and no production row counts as extra. The code is the pinned production tree (921d1ef9), the integration tree with every switch off (control), or the integration tree with the opt-in recipe (treatment), each with and without the two hand-off settings.

| frame | rows of the production table | code | two hand-off settings | rows of the replay | lost (flux > 0) | extra | lost, flux > 1000 | extra | lost, flux > 1e4 | extra | lost, flux > 3e4 | extra |
| :--- | ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A (nrcblong exposure 1) | 52465 | production code (921d1ef9) | unset | 52446 | 64 | 46 | 21 | 1 | 12 | 1 | 3 | 1 |
| A (nrcblong exposure 1) | 52465 | production code (921d1ef9) | set | 52466 | 6 | 7 | 0 | 0 | 0 | 0 | 0 | 0 |
| A (nrcblong exposure 1) | 52465 | integration code, every switch off | unset | 52545 | 411 | 492 | 31 | 42 | 13 | 8 | 3 | 1 |
| A (nrcblong exposure 1) | 52465 | integration code, every switch off | set | 52528 | 390 | 453 | 14 | 39 | 2 | 7 | 0 | 0 |
| A (nrcblong exposure 1) | 52465 | integration code, every switch on | unset | 52629 | 404 | 569 | 30 | 53 | 12 | 10 | 3 | 1 |
| A (nrcblong exposure 1) | 52465 | integration code, every switch on | set | 52614 | 380 | 529 | 11 | 50 | 0 | 9 | 0 | 0 |
| B (nrcalong exposure 2) | 53915 | production code (921d1ef9) | unset | 53954 | 83 | 122 | 5 | 3 | 3 | 1 | 3 | 1 |
| B (nrcalong exposure 2) | 53915 | production code (921d1ef9) | set | 53953 | 74 | 112 | 1 | 0 | 0 | 0 | 0 | 0 |
| B (nrcalong exposure 2) | 53915 | integration code, every switch off | unset | 54093 | 334 | 511 | 9 | 67 | 4 | 10 | 3 | 1 |
| B (nrcalong exposure 2) | 53915 | integration code, every switch off | set | 54093 | 327 | 504 | 6 | 65 | 1 | 12 | 0 | 1 |
| B (nrcalong exposure 2) | 53915 | integration code, every switch on | unset | 54184 | 323 | 591 | 8 | 70 | 4 | 12 | 3 | 1 |
| B (nrcalong exposure 2) | 53915 | integration code, every switch on | set | 54183 | 317 | 584 | 5 | 68 | 1 | 14 | 0 | 1 |

The production code with the two settings reproduces production at the six target rows of the two frames, five rows in frame A and one in frame B. For each of the six rows the production table and the replay have a difference of 0 in `flux_fit`, `qfit`, `local_bkg`, `flux_init`, `x_fit` and `y_fit`, and equal `flags` and `group_size` (`results/run5_replay_target_rows.txt`). That replay loses no row with `flux_fit` above 1000 in frame A. Without them it loses 12 rows with `flux_fit` above 1e4 in frame A, three of them above 3e4. The integration code loses rows that the pinned code keeps, because it differs from the pinned code by 74 commits and by the PRs. With the two settings the integration code with every switch off loses 2 rows above 1e4 in frame A and 1 in frame B, and the integration code with every switch on loses 0 and 1. Over all fluxes (`flux_fit` above 0), the integration code lacks 317 to 411 production rows per frame and adds 453 to 591 rows, while the pinned code with the two settings lacks 6 rows in frame A and 74 in frame B and adds 7 and 112.

## Added seed rows

The options of #1376 and #1377 add rows to the seed of the m7 fits of a tile, the star list that the per-frame fits start from. The added rows of a band are the rows of the seed of the run that have no row of the production seed within 1 mas. Every added row of the run has an origin recorded in the seed. Group A rows are own-band m6 rows that the far-companion exemption of #1377 keeps, and group B rows are i2d detections that the relaxed bright roundness window of #1376 keeps (66 + 88 = 154 rows in F212N and 0 + 21 = 21 rows in F480M). For each added row, the table counts the rows of the merged raw and of the merged vetted m7 catalog of the run within 0.1 arcsec, and how many of the added rows with a vetted row of the run also have a row in the production merged vetted catalog within 0.1 arcsec. The script is `<script>`; it reads `results/survive_<tile>_<band>.json` and writes `results/run5_added_summary.txt`.

| tile | band | rows added | from the exemption of #1377 | from the relaxed window of #1376 | with a row in the merged raw catalog of the run | with a row in the merged vetted catalog of the run | of those, also with a row in the production merged vetted catalog | with no row in the merged vetted catalog of the run |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| o105 | F212N | 81 | 42 | 39 | 80 | 80 | 7 | 1 |
| o118 | F212N | 16 | 5 | 11 | 16 | 16 | 1 | 0 |
| o130 | F212N | 12 | 3 | 9 | 12 | 11 | 1 | 1 |
| o138 | F212N | 45 | 16 | 29 | 45 | 45 | 4 | 0 |
| all four | F212N | 154 | 66 | 88 | 153 | 152 | 13 | 2 |
| o105 | F480M | 6 | 0 | 6 | 6 | 6 | 0 | 0 |
| o118 | F480M | 10 | 0 | 10 | 10 | 10 | 0 | 0 |
| o130 | F480M | 3 | 0 | 3 | 3 | 3 | 0 | 0 |
| o138 | F480M | 2 | 0 | 2 | 2 | 2 | 0 | 0 |
| all four | F480M | 21 | 0 | 21 | 21 | 21 | 0 | 0 |

In F212N, 152 of the 154 added rows have a row in the merged vetted catalog of the run, and 13 of those 152 have a production merged vetted row as well; the other 139 mark positions at which the production vetted catalog has no row within 0.1 arcsec. The two added rows without a vetted row are described in the Result section. In F480M, all 21 added rows are group B rows, all 21 have a row in the merged vetted catalog of the run, and none has a production row. The test follows the added rows to the vetted catalog and does not check that they are real stars.

## Masked area

The finalize job renders the residual with the footprint masks (the `MERGEDCAT_FOOTPRINT_MASK_*` variables of the Recipe). The table gives, per tile and band, the finite pixels of the merged data mosaic and the number of those that are NaN in the residual of the masked render `optfin`, of the unmasked render `optnm` and of the production render `prod5`.

| tile | band | finite pixels of the data mosaic | NaN pixels in `optfin` | share of the data footprint (`optfin`, %) | NaN pixels in `optnm` | NaN pixels in `prod5` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| o105 | F212N | 51825305 | 7143322 | 13.78 | 29 | 0 |
| o118 | F212N | 51826881 | 2828777 | 5.46 | 26 | 0 |
| o130 | F212N | 51826720 | 2354582 | 4.54 | 29 | 0 |
| o138 | F212N | 51827646 | 4045369 | 7.81 | 87 | 0 |
| o105 | F480M | 12464795 | 2907644 | 23.33 | 5 | 0 |
| o118 | F480M | 12470721 | 1424840 | 11.43 | 4 | 0 |
| o130 | F480M | 12472748 | 1042721 | 8.36 | 7 | 0 |
| o138 | F480M | 12472006 | 1270606 | 10.19 | 0 | 0 |

The masked share is 4.54 to 13.78 % of the data footprint in F212N and 8.36 to 23.33 % in F480M, and o105 has the largest share in both bands. The unmasked render has 0 to 87 NaN pixels per mosaic and the production render has none. A leftover or crater inside a masked region has no counterpart in the masked render, and the pair tables count it as cleared or gone. The masked counts of the section "Counts per tile" therefore combine the effect of the catalogs of the run with the pixels that the mask hides. The unmasked render `optnm` has the catalogs of the run and no hidden pixels.

## F480M with the F480M mask values

The masked F480M renders of the previous sections use the mask values validated on F212N. The study of #1380 recommends smaller F480M values: ring radius 20 px, DQ_SAT dilation 3 px and edge band 6 px, with KFAC 4 or, for less masked area, KFAC 3. To test them, the F480M residual of each tile was rendered twice more from the finalized catalogs of the run, with the render function of the finalize job and no refit. The render `mk4f` uses KFAC 4 and the render `mk3f` uses KFAC 3, both with the three F480M values, and every other mask variable keeps the value of the Recipe. The catalogs and the model are those of the run, so the three renders differ in the mask only.

`results/run5_f480m_mask_grid.txt` holds the 16 pixel checks (4 tiles, 2 variants, each against the unmasked render and against the masked render of the run). In all 16, the two images have the same grid and every pixel that is finite in both has the same value (largest difference 0 MJy/sr). In all 8 comparisons with the masked render of the run, no pixel is NaN in the variant and finite in `optfin`, so the mask of each variant is a subset of the mask of the run. `results/run5_f480m_mask_area.txt` has the masked areas and `results/run5_f480m_mask_variants.txt` has the counts and the leftovers that the variants add.

| tile | render | mask values | leftovers ≥50 | ≥100 | ≥200 | ≥500 | ≥1000 | craters ≥50 | ≥200 | ≥500 | masked % of finite px |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| o105 | `optfin` | F212N mask values (KFAC 4, ring radius 25 px, DQ_SAT dilation 4 px, edge band 8 px) | 49 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 23.33 |
| o105 | `mk4f` | F480M mask values with KFAC 4 (ring radius 20 px, DQ_SAT dilation 3 px, edge band 6 px) | 51 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 22.78 |
| o105 | `mk3f` | F480M mask values with KFAC 3 (ring radius 20 px, DQ_SAT dilation 3 px, edge band 6 px) | 55 | 2 | 0 | 0 | 0 | 1 | 0 | 0 | 14.99 |
| o118 | `optfin` | F212N mask values (KFAC 4, ring radius 25 px, DQ_SAT dilation 4 px, edge band 8 px) | 48 | 10 | 1 | 0 | 0 | 0 | 0 | 0 | 11.43 |
| o118 | `mk4f` | F480M mask values with KFAC 4 (ring radius 20 px, DQ_SAT dilation 3 px, edge band 6 px) | 52 | 13 | 2 | 0 | 0 | 0 | 0 | 0 | 10.90 |
| o118 | `mk3f` | F480M mask values with KFAC 3 (ring radius 20 px, DQ_SAT dilation 3 px, edge band 6 px) | 53 | 13 | 2 | 0 | 0 | 0 | 0 | 0 | 7.26 |
| o130 | `optfin` | F212N mask values (KFAC 4, ring radius 25 px, DQ_SAT dilation 4 px, edge band 8 px) | 56 | 7 | 1 | 0 | 0 | 0 | 0 | 0 | 8.36 |
| o130 | `mk4f` | F480M mask values with KFAC 4 (ring radius 20 px, DQ_SAT dilation 3 px, edge band 6 px) | 57 | 8 | 1 | 0 | 0 | 0 | 0 | 0 | 7.84 |
| o130 | `mk3f` | F480M mask values with KFAC 3 (ring radius 20 px, DQ_SAT dilation 3 px, edge band 6 px) | 57 | 8 | 1 | 0 | 0 | 0 | 0 | 0 | 5.36 |
| o138 | `optfin` | F212N mask values (KFAC 4, ring radius 25 px, DQ_SAT dilation 4 px, edge band 8 px) | 40 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 10.19 |
| o138 | `mk4f` | F480M mask values with KFAC 4 (ring radius 20 px, DQ_SAT dilation 3 px, edge band 6 px) | 40 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 9.68 |
| o138 | `mk3f` | F480M mask values with KFAC 3 (ring radius 20 px, DQ_SAT dilation 3 px, edge band 6 px) | 43 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 6.46 |
| all four | `optfin` | sum over the four tiles | 193 | 23 | 2 | 0 | 0 | 0 | 0 | 0 | 13.32 |
| all four | `mk4f` | sum over the four tiles | 200 | 27 | 3 | 0 | 0 | 0 | 0 | 0 | 12.80 |
| all four | `mk3f` | sum over the four tiles | 208 | 27 | 3 | 0 | 0 | 1 | 0 | 0 | 8.52 |

The variants hide less area and change the counts little. The masked share of the data footprint (sum over the four tiles) is 13.32 % in `optfin`, 12.80 % in `mk4f` and 8.52 % in `mk3f`; per tile, `mk3f` masks 5.36 to 14.99 %. The leftovers ≥100 MJy/sr go from 23 to 27 in both variants and the leftovers ≥200 MJy/sr from 2 to 3. The leftovers ≥500 and ≥1000 and the craters ≥200 and ≥500 stay at 0. The render `mk3f` has one crater ≥50 MJy/sr that `optfin` and `mk4f` do not have, at (667, 775) of o105 with a depth of 52 MJy/sr, a model peak of 166 MJy/sr and a depth over model peak of 0.312, just above both limits (50 MJy/sr and 0.3).

The four leftovers ≥100 MJy/sr that the variants add are the same in both. In o118 they are (65, 2223) with 472 MJy/sr (NOT_IN_M7_CATALOG), (1542, 2166) with 175 MJy/sr (VETTED_BUT_UNSUBTRACTED) and (2112, 1715) with 131 MJy/sr (NOT_IN_M7_CATALOG), and in o130 the leftover is (834, 2121) with 113 MJy/sr (NOT_IN_M7_CATALOG). The 472 MJy/sr leftover is the one added leftover ≥200 MJy/sr. `mk4f` has the KFAC of `optfin` and differs from it in the ring radius, the DQ_SAT dilation and the edge band, which change together here, so this test does not say which of the three uncovers the four leftovers. The step from KFAC 4 to KFAC 3 adds no leftover ≥100 MJy/sr.

## The two F480M hand-off settings

The production F480M m7 jobs ran with `DAOPHOT_HANDOFF_PEAK_MIN_AREA=50` and `DAOPHOT_HANDOFF_RESTORE_ACCEPTED_EXCL_FWHM=1.5` in the environment (`DAOPHOT_HANDOFF_FILTERS=F480M` limits both to F480M; the Recipe describes them). The earlier run of these four tiles and the first o045 runs of #1383 did not set them, and this run sets them. The tables sum the counts over the four tiles and compare the production render with the unmasked and the masked render of the earlier run (settings unset) and of this run (settings set). The production row is the same render in both runs, scored by the same scripts, and its counts are equal in both.

**F480M**

| render | settings | leftovers ≥100 | ≥200 | ≥500 | ≥1000 | craters ≥50 | ≥200 | ≥500 | leftovers ≥200 SATSTAR_NEAR | VETTED_BUT_UNSUBTRACTED | VETTED_OUT | NOT_IN_M7_CATALOG |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| production render | none (same render in both runs) | 518 | 358 | 221 | 173 | 163 | 52 | 24 | 234 | 36 | 4 | 84 |
| opt unmasked (`optnm`) | without the two hand-off settings | 333 | 289 | 229 | 185 | 27 | 17 | 11 | 267 | 3 | 0 | 19 |
| opt unmasked (`optnm`) | with the two hand-off settings | 299 | 257 | 199 | 158 | 26 | 16 | 11 | 236 | 3 | 0 | 18 |
| opt masked (`optfin`) | without the two hand-off settings | 23 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 |
| opt masked (`optfin`) | with the two hand-off settings | 23 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 |

**F212N**

| render | settings | leftovers ≥100 | ≥200 | ≥500 | ≥1000 | craters ≥50 | ≥200 | ≥500 | leftovers ≥200 SATSTAR_NEAR | VETTED_BUT_UNSUBTRACTED | VETTED_OUT | NOT_IN_M7_CATALOG |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| production render | none (same render in both runs) | 702 | 467 | 384 | 342 | 1382 | 443 | 184 | 228 | 72 | 71 | 96 |
| opt unmasked (`optnm`) | without the two hand-off settings | 209 | 86 | 42 | 25 | 322 | 112 | 34 | 64 | 8 | 1 | 13 |
| opt unmasked (`optnm`) | with the two hand-off settings | 209 | 86 | 42 | 25 | 322 | 112 | 34 | 64 | 8 | 1 | 13 |
| opt masked (`optfin`) | without the two hand-off settings | 85 | 14 | 2 | 1 | 28 | 0 | 0 | 5 | 2 | 0 | 7 |
| opt masked (`optfin`) | with the two hand-off settings | 85 | 14 | 2 | 1 | 28 | 0 | 0 | 5 | 2 | 0 | 7 |

In F480M, the unmasked render has 333 leftovers ≥100 MJy/sr and 267 SATSTAR_NEAR leftovers ≥200 MJy/sr without the settings, and 299 and 236 with them (production: 518 and 234). The masked F480M renders have the same counts with and without the settings. In F212N, every count is equal with and without the settings.

The grid check compares the masked renders of the earlier run and of this run pixel by pixel. In every tile and band, no pixel is NaN in one of the two renders only.

| tile | band | pixels finite in both renders | pixels with a difference above 1e-3 MJy/sr | share of the pixels finite in both (%) | largest difference (MJy/sr) |
| :--- | :--- | ---: | ---: | ---: | ---: |
| o105 | F212N | 44681983 | 0 | 0.00 | 0 |
| o118 | F212N | 48998104 | 0 | 0.00 | 5.72e-06 |
| o130 | F212N | 49472138 | 0 | 0.00 | 0 |
| o138 | F212N | 47782277 | 0 | 0.00 | 0 |
| o105 | F480M | 9557151 | 91378 | 0.96 | 7.19 |
| o118 | F480M | 11045881 | 42428 | 0.38 | 4.56 |
| o130 | F480M | 11430027 | 30945 | 0.27 | 7.71 |
| o138 | F480M | 11201400 | 42757 | 0.38 | 7.39 |

In F212N no pixel differs by more than 1e-3 MJy/sr, and the largest difference is 0 (o105, o130, o138) or 5.72e-6 MJy/sr (o118). In F480M, 0.27 to 0.96 % of the pixels that are finite in both renders differ by more than 1e-3 MJy/sr, and the largest difference is 4.56 to 7.71 MJy/sr, although the counts of the masked renders are equal.

## Remaining leftovers of the masked render

The masked renders have 16 whole-star leftovers with a residual peak ≥200 MJy/sr: 14 in F212N (o105 8, o118 4, o130 1, o138 1) and 2 in F480M (o118 1, o130 1). The table lists them. The residual, data and model peaks are the quantities defined under "How the counts are made". The two distances are those from the leftover to the nearest vetted m7 row and to the nearest consolidated satstar fit of the run.

| tile | band | pixel (x, y) | residual peak (MJy/sr) | data peak (MJy/sr) | model peak (MJy/sr) | class | distance to the nearest vetted row (arcsec) | distance to the nearest satstar fit (arcsec) |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- | ---: | ---: |
| o105 | F212N | (434, 3684) | 358 | 495 | 439 | SATSTAR_NEAR | 0.077 | 0.797 |
| o105 | F212N | (5368, 4587) | 345 | 345 | 0 | NOT_IN_M7_CATALOG | 0.444 | 1.208 |
| o105 | F212N | (11052, 3798) | 336 | 423 | 181 | NOT_IN_M7_CATALOG | 0.126 | 1.577 |
| o105 | F212N | (7030, 3275) | 254 | 259 | 8 | SATSTAR_NEAR | 0.349 | 0.808 |
| o105 | F212N | (6529, 3558) | 245 | 246 | 1 | NOT_IN_M7_CATALOG | 0.206 | 1.930 |
| o105 | F212N | (3245, 3878) | 209 | 285 | 135 | SATSTAR_NEAR | 0.091 | 0.783 |
| o105 | F212N | (1660, 341) | 205 | 291 | 92 | NOT_IN_M7_CATALOG | 0.139 | 1.228 |
| o105 | F212N | (9362, 1291) | 201 | 251 | 105 | VETTED_BUT_UNSUBTRACTED | 0.044 | 1.693 |
| o118 | F212N | (330, 111) | 939 | 939 | 2 | SATSTAR_NEAR | 0.245 | 0.657 |
| o118 | F212N | (220, 4407) | 423 | 471 | 142 | SATSTAR_NEAR | 0.097 | 0.864 |
| o118 | F212N | (637, 3732) | 308 | 309 | 1 | NOT_IN_M7_CATALOG | 0.342 | 1.329 |
| o118 | F212N | (641, 3740) | 203 | 206 | 6 | NOT_IN_M7_CATALOG | 0.293 | 1.132 |
| o130 | F212N | (5065, 68) | 1241 | 1391 | 316 | VETTED_BUT_UNSUBTRACTED | 0.046 | 2.975 |
| o138 | F212N | (7366, 4240) | 287 | 357 | 243 | NOT_IN_M7_CATALOG | 0.082 | 1.401 |
| o118 | F480M | (368, 334) | 253 | 357 | 164 | NOT_IN_M7_CATALOG | 0.192 | 9.235 |
| o130 | F480M | (1932, 146) | 359 | 361 | 5 | VETTED_BUT_UNSUBTRACTED | 0.017 | 8.029 |

By class, 5 leftovers are SATSTAR_NEAR, 3 are VETTED_BUT_UNSUBTRACTED and 8 are NOT_IN_M7_CATALOG. The largest is the VETTED_BUT_UNSUBTRACTED leftover of o130 F212N at (5065, 68) with a residual peak of 1241 MJy/sr (data peak 1391, model peak 316, vetted row 0.046 arcsec away). The second largest is the SATSTAR_NEAR leftover of o118 F212N at (330, 111) with 939 MJy/sr (data peak 939, model peak 2, nearest satstar fit 0.657 arcsec away, nearest vetted row 0.245 arcsec away), at which the model removes essentially nothing. The next section traces each of the 16 leftovers to the stage of the pipeline at which the star leaves the model.

## Cause of each remaining leftover

Each of the 16 leftovers of the previous table was traced through the catalogs of the run. For the leftovers without a row at the star, the trace went on through the seed of the m7 fits and the per-frame finder. `results/run5_remaining_trace.txt` has one block per leftover with the values quoted below, `results/run5_remaining_persist.txt` lists each leftover in the production render, the unmasked render and the masked render, and `results/run5_remaining_seedreplay.txt` and `results/run5_remaining_spike_variant.txt` hold the replays of the groups 3 and 7.

**The leftovers exist in the production render.** 14 of the 16 leftovers have a leftover of the production render within 4 pixels with a residual peak within 10 % of the masked-render value (the ratio production over masked is 0.995 to 1.014), and the unmasked render of the run has the same 14 within 1 %. The residual peaks of these 14 leftovers are the same in the production render, the unmasked render and the masked render. The two others are the o118 F212N leftovers at (637, 3732) and (641, 3740), which belong to group 6.

The leftovers fall into seven groups. Each group has one mechanism that keeps the star out of the model.

| group | leftovers | what the trace shows |
| :--- | :--- | :--- |
| 1. Fainter member of a close pair has no row | 5: o105 F212N (434, 3684) and (3245, 3878), o118 F212N (220, 4407), o138 F212N (7366, 4240), o118 F480M (368, 334) | The brighter member has a row 0.077 to 0.097 arcsec away in F212N (2.5 to 3.1 pixels) and 0.192 arcsec away in F480M (1.2 FWHM). The fainter member has none. The emulation of the per-frame finder finds one maximum, at the brighter member, in the frames that cover the star. The detection image of the i2d mosaic shows the fainter member with roundness1 0.85 to 1.08, S/N 41 to 239 and prominence 93 to 157, so it passes the S/N floor (30 in F212N, 100 in F480M) and the prominence floor (50) of the bright-roundness seed. Its unsubtracted fraction (3x3 peak of the detection image over the 3x3 peak of the data co-add) is 0.71 to 0.90 and fails the floor of 0.9 in all five (`--manual-seed-round-bright-frac-min`). |
| 2. One covering frame with a NaN core | 2: o105 F212N (5368, 4587), o118 F212N (330, 111) | One frame covers the star and its core pixel is DO_NOT_USE. The m7 fill of the pixel puts 228 (o105) and 649 (o118) MJy/sr in the core against a surrounding ring of 305 to 362 and 900 to 1040 MJy/sr, so the sharpness in the per-frame finder is 0.07 against the lower bound 0.50. The i2d route reaches the o105 star with sharpness 0.33, prominence 42 and S/N 38 (floors 0.4 and 50) and the o118 star with sharpness 0.39, prominence 174 and S/N 68; the sharpness floor of 0.4 fails by 0.01 there. The o118 leftover has a residual of 939 MJy/sr, the second largest of the 16. |
| 3. Cut by the companion cut | 2: o105 F212N (11052, 3798), (1660, 341) | The own-band m6 vetted rows of the two stars (flux 3527 at S/N 87 and flux 2197 at S/N 67, both unsaturated) lie 1.76 and 1.86 FWHM from a star of flux 45252 and 24645. They have 0.078 and 0.089 of its flux, below the cut at 0.1. The exemption of #1377 keeps such a row only when every seed source that triggers the cut lies 1.75 FWHM or more from the bright star. The cross-band seed position of the same star lies 1.628 and 1.728 FWHM from the bright star and triggers the cut. With a keep radius of 1.5 FWHM (the run uses 1.75 FWHM) the replay of the seed builder adds 3 rows in the nrca module and 8 rows in the nrcb module of o105 F212N, among them rows within 0.012 and 0.009 arcsec of the two stars. |
| 4. Outside the roundness window | 2: o105 F212N (7030, 3275), (6529, 3558) | No m6 row exists. The per-frame finder sees each star in 4 of 4 frames with sharpness 0.52 to 0.59 and roundness1 between -1.02 and -1.30, outside the window of plus or minus 1.0. The i2d route has S/N 70 and 68 and an unsubtracted fraction of 0.98 and 1.00 and fails on prominence (30 and 40 against the floor of 50). |
| 5. Lost in the merge | 2: o105 F212N (9362, 1291), o130 F212N (5065, 68) | At (9362, 1291) the m6 catalog has two vetted rows 1.6 pixels apart (flux 2108 and 958). The per-frame fits resolve a strong component (flux 1556 to 2145) and a weak one (flux 1142 to 1704), and the merge matches them to one row (flux 1206, nmatch 5, nmatch_good 3). The two components are 0.7 FWHM apart, inside the split-row radius of #1365, so that PR starts no row. At (5065, 68) two frames cover the star, and in one of them it lies 1.6 pixels from the detector edge with 53 of the 169 pixels of the 13x13 fit box NaN or DO_NOT_USE. The fit of the first frame has flux 14434 with an error of 64.7 (qfit 0.0126). The fit of the edge frame has flux 1295 with an error of 31.1 (qfit 0.19, flags 1). The sigma clip of the merge never clips one of two values. The inverse-variance weights differ by (64.7 / 31.1)^2 = 4.3 in favour of the edge fit, so the inverse-variance mean of the two fluxes is 3762, close to the vetted rows of production (3761) and of the run (3849) and far below the 14335 of the m6 row. The residual is 1241 MJy/sr, the largest of the 16. |
| 6. Wing of a star under the mask | 2: o118 F212N (637, 3732), (641, 3740) | The star at (645, 3733) has a data peak of 30156 MJy/sr and a model peak of 5 MJy/sr in the production render and in the unmasked render of the run (residual 30152 MJy/sr, class NOT_IN_M7_CATALOG in both). The two covering frames carry rejected satstar fits (qfit 17.9 and 29.7). The masked render sets the pixels of the star to NaN (footprint mask of the rejected satstar fits), and the wing at the edge of the NaN disc leaves the two leftovers of 308 and 203 MJy/sr. These two exist only in the masked render. |
| 7. Isolated spike next to a bad pixel | 1: o130 F480M (1932, 146) | Pixel (1525, 141) of the nrcalong frame of exposure 3 has 1572 MJy/sr against 6 to 15 MJy/sr in its neighbours and 73 MJy/sr in the other five exposures. The excess over the annulus background (6.4 MJy/sr) is 1566 MJy/sr, 738 times the MAD-sigma of the annulus (about 2.1 MJy/sr). Both integrations of the ramp carry it (1829 and 1748 DN per group), so the ramp test of #1367 does not fire (ratio of the weakest to the strongest 3x3 sum 0.58 against the limit 0.5). The spike finder of #1374 skips it because one pixel of its 3x3 box (the lower left) is DO_NOT_USE, and the finder judges only complete boxes. |

**Replay of group 7.** A variant of the spike finder tolerates one unusable pixel in the 3x3 box and judges the isolation on the usable neighbours (`results/run5_remaining_spike_variant.txt`). In the 12 F480M frames of o130 it finds 73 spikes where the shipped finder finds 69. The four added spikes have excesses of 1275 to 3332 MJy/sr, and the brightest usable neighbour carries 0.1 to 1.7 % of the excess. One of the four is the pixel of the leftover, and a hot pixel at (760, 303) appears in two frames. In the 48 F212N frames of o130 the variant adds one spike (9 shipped, 10 with the variant).

**What no test here shows.** The false-seed rate of every loosened gate is unmeasured: the unsubtracted-fraction floor of group 1 (0.9), the prominence floor of group 4 (50), the roundness window and the sharpness floor of groups 2 and 4, the sharpness floor of the i2d route (0.4), and the keep radius of group 3 (1.75 FWHM). A row added to the seed gives the per-frame fit a start position and does not by itself subtract the star, and the replay of group 3 stops at the seed table. The merge weighting of group 5 needs new code (zero weight or a lower weight for a per-frame row with the partial-fit-box flag when a full-box fit of the same star exists, or a rule for two frames that disagree), and the production render shows the same leftover. None of these changes is part of the stack of this run.

## CRDS cache files

The jobs set `CRDS_READONLY_CACHE=1`. The CRDS cache of the production tree (`/orange/adamginsburg/jwst/crds`) holds `config/jwst/server_config` (1457934 bytes) and `config/jwst/bad_files.txt` (5034 bytes). A listing of the two files before the first submission (2026-11-26, 23:2x EST) gave the modification time 2026-11-26 23:22:52 for both. The listings that the four post jobs took after the run (`results/<tile>_crds_after.txt`) give the same sizes and the modification time 2026-11-27 03:01:18 (post jobs of o118 and o138) or 2026-11-27 06:04:36 (post jobs of o105 and o130). The check compares sizes and modification times and does not compare contents.

Two observations bear on who modified the files (`results/run5_crds_attribution.txt` and `results/run5_crds_probe.txt`):

- The 40 logs of the run hold 12 CRDS client lines and 40 PARS-RESAMPLESTEP lines. At 03:01:18 no line of the run lies within 100 s (the nearest one, 135 s later, is in the log of the o118 post job), and 4 jobs of the run and 207 other jobs of the same user account were running. At 06:04:36 two lines of the run lie within 100 s: the CRDS client line at 06:04:34 and the PARS-RESAMPLESTEP line at 06:04:36, both in the log of the o130 post job.
- A probe of the installed CRDS client on scratch copies of the two files (2026-11-27, about 07:32 EST) shows that `update_config_info` rewrites `server_config` with the same size and a new modification time when `CRDS_READONLY_CACHE` is 0, and leaves it unchanged when it is 1. In an empty cache directory, the value 0 creates both files and the value 1 creates nothing.

The writer of the modifications at 03:01:18 and 06:04:36 is not identified, and this test does not establish whether the contents changed. The coincidence at 06:04:36 with two log lines of the o130 post job does not show that this job wrote the files: the job ran with the setting 1, and the probe shows that the client leaves the files unchanged with that setting.

## What this run does not show

- **The split between the switches and the code drift.** The four tiles have no control arm, so every difference between a render of this run and the production render combines the opt-in switches with the 74 commits between the production tree and the integration base (section "Setup of the run") and with the differences of the PRs. The one-frame replays (section "Rows that production has and the run lacks") separate the code from the switches for two frames of o105 F480M only.
- **Model quality inside the masks.** The masked renders hide 4.54 to 13.78 % (F212N) and 8.36 to 23.33 % (F480M) of the data footprint, so their counts apply to the pixels outside the mask.
- **Whether the added seed rows are real stars.** The test follows them to the vetted catalog (section "Added seed rows").
- **Which F480M mask value matters.** The section "F480M with the F480M mask values" renders the F480M residual with the values of #1380. The ring radius, the DQ_SAT dilation and the edge band change together in that test, so it does not say which of the three uncovers the four leftovers ≥100 MJy/sr that the variants add. It changes the mask of the render only and says nothing about the fits.
- **Equality of the environment with production.** The log-template comparison of the Recipe supports the equality of the environment of the shard jobs with production and cannot prove it.
- **Fixes for the five features that exceed production.** The section "Cause of the five features that exceed production" traces each of them with one-frame replays that stop at the per-frame tables. The render of the integration tree with the guard off and the code change on is not part of it, and the replay of the pinned production code differs from the production table in 65 lost and 92 gained rows of nrcalong exposure 1 whose origin was not traced. None of the fixes that the section lists was run. The masked renders hide all five features.

## Open decisions

1. **Mask values for F480M.** Whether the F480M render uses the F480M values of #1380 with KFAC 4 (12.80 % of the footprint masked, summed over the four tiles) or KFAC 3 (8.52 %), or keeps the values of the Recipe (13.32 %). The leftovers ≥100 MJy/sr are 23 with the Recipe values and 27 with either F480M variant, and the KFAC 3 render has one more crater ≥50 MJy/sr (section "F480M with the F480M mask values").
2. **Changes that follow from the 16 remaining leftovers of 200 MJy/sr and more** (section "Cause of each remaining leftover"). Each of the seven groups points at one gate or one merge rule. Every loosened gate has an unmeasured false-seed rate and needs a tile test, and the merge weighting of group 5 needs new code.
3. **Changes that follow from the five features of the unmasked render that exceed production** (section "Cause of the five features that exceed production"). Whether to keep the ePSF core in the saturated-star fit and recalibrate the satstar gate for it, or to remove the core for the F480M saturated-star PSF. Whether to replace the integer-pixel distance test of the near-saturation filter by a continuous test or a hysteresis band. Whether to lower the upper bound of the second-pass neighbour guard window from 0.8 to 0.5, or to cap the kept companion flux at the data budget of its box. Each of these needs a tile test before it can become a default.

## Evidence files

The files are in `results/`. `<tile>` stands for o105, o118, o130 and o138, and `<band>` for F212N and F480M. Scratch paths inside the files are written as `<scratch>`. The names say run5 for this run and run4 for the earlier run of the same tiles.

| file | content |
| :--- | :--- |
| `integration_tree.txt` | HEAD, base, merge commits and diff stat of the integration tree |
| `run5_jobtable.txt` | the job table of the section "Setup of the run" |
| `run5_sacct.txt` and `<tile>_sacct.txt` | Slurm accounting of the 40 jobs (all tiles; one tile) |
| `run5_summary.md` | counts of whole-star leftovers and big craters per tile and render, every threshold |
| `<tile>_compare_<band>.txt` and `<tile>_compare_<band>.json` | counts, classes and 3 px matching tables per tile and band, as the comparison step prints them |
| `<tile>_regress_scan_nm_<band>.txt` and `<tile>_regress_scan_fin_<band>.txt` | position-independent scan of `optnm` and of `optfin` against `prod5` |
| `run5_larger_than_prod.txt` | the features that are larger than in production (second scan with the same box) |
| `run5_catdiff.txt` | rows of the merged vetted catalogs without a counterpart, by flux bin |
| `run4_catdiff.txt` | the same for the earlier run of the same tiles, which lacked the two hand-off settings |
| `run5_replay_unmatched.txt` | one-frame replays of o105 F480M |
| `run5_replay_target_rows.txt` | the six target rows of the replays, production table against the replay of the production code with the two hand-off settings, field by field |
| `survive_<tile>_<band>.json` and `run5_added_summary.txt` | added seed rows and how far they get |
| `run5_mask_area.txt` | NaN pixels of the three renders |
| `run5_vs_run4_grid.txt` | grid check of the masked renders of the earlier run against those of this run |
| `run5_remaining_optfin_ge200.txt` | leftovers of 200 MJy/sr and more in the masked render |
| `run5_remaining_trace.txt` | one block per remaining leftover with the catalog rows, per-frame fits, seed gates and detection-image values quoted in the section "Cause of each remaining leftover" |
| `run5_remaining_persist.txt` | each remaining leftover in the production render, the unmasked render and the masked render |
| `run5_remaining_seedreplay.txt` | replay of the seed builder with a keep radius of 1.5 FWHM for the two leftovers of group 3 |
| `run5_remaining_spike_variant.txt` | replay of the spike finder that tolerates one unusable pixel in the 3x3 box (group 7) |
| `run5_f480m_mask_variants.txt` | counts, masked share and the leftovers that the F480M mask variants add, per tile and summed |
| `run5_f480m_mask_area.txt` | masked area of the F480M renders `optfin`, `mk4f`, `mk3f` and `optnm` |
| `run5_f480m_mask_grid.txt` | the 16 pixel checks of `mk4f` and `mk3f` against `optnm` and `optfin` |
| `run5_five_arms.txt` | outcome of every replay arm at the five targets of the section "Cause of the five features that exceed production" |
| `run5_five_frames.txt` | per-frame rows of the companion of features 1 and 2 in the production table and in the table of the run |
| `run5_guard_pairs.txt` | the 72 pairs that the guard of #1360 keeps, with the residual extremes of their windows in the production render and in the unmasked render of the run |
| `run5_five_gate.txt` | population numbers of the satstar gate of feature 5: matched rows of one frame and lost consolidated rows of the four tiles |
| `<tile>_precheck_F480M.txt` | output of the precheck step of the post job (empty when every product exists) |
| `<tile>_crds_after.txt` and `<tile>_crds_diff.txt` | listing of the two CRDS cache files after the run and its difference from the listing before |
| `run5_crds_attribution.txt` and `run5_crds_probe.txt` | log attribution of the modification times and probe of the CRDS client |
| `run5_orange_check.txt` | modification times of the production directories and of the targets of the staged input symlinks after the run |
