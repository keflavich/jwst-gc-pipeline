# End-to-end o045 test of the stacked opt-in PRs

Two scratch-basepath runs of the stack of 17 open draft code PRs (#1360 to #1377, listed under Recipe; #1380 is documentation only and was not merged) on gc-treasury 10678 o045, F212N and F480M. Run 1 (2026-11-24 22:00 to 2026-11-25 04:53, Slurm accounting) integrates all 17 code PRs and runs two arms: ctl (every opt-in switch off) and opt (every switch on). Run 2 re-runs the opt arm with the updated heads of #1366, #1376 and #1377. Each arm runs the m7 per-frame fits (an 8-task array) and a finalize job (merge, vetting and the residual renders), and the residual i2d mosaics (data minus merged-catalog model) are scored for whole-star leftovers and big craters. Both integration trees are local branches that were never pushed. Every number in this file is generated from the files in `results/` (list at the end).

**Result.**

- **Run 1, residual counts** (production, ctl, opt unmasked, opt with the real masks). F212N: leftovers ≥100 MJy/sr 376, 369, 195, 78; craters ≥50 MJy/sr 554, 522, 128, 10. F480M: leftovers ≥100 MJy/sr 224, 257, 213, 7; craters ≥50 MJy/sr 90, 97, 30, 1. The real masks set 15.67 % (F212N) and 24.35 % (F480M) of the finite pixels to NaN; the unmasked opt render is the one that shows what the fit-time switches do.
- **Run 1, regressions.** A position-independent scan (box extremum, see "How the counts are made") finds 2 features of the unmasked opt render that exceed the ctl extremum in the same box by more than a factor 1.5: F212N leftover (8653, 1078) 4803 MJy/sr against 257 MJy/sr in ctl; F480M leftover (410, 1680) 214 MJy/sr against 19 MJy/sr in ctl. Both come from #1366 (replays in the section on the regressions). No crater of either band is worse than in ctl by the same rule. The 3 px matching lists more items as new; in the table of listed items below, every other one has a box extremum below 200 MJy/sr or at most 1.5 times the ctl extremum.
- **Fix.** #1366 was updated (head 38873cf8, used in run 2). Replays of the three frames of the regressions return the ctl rows (F212N) or drop the spurious row (F480M); the table is in the section on the regressions.
- **Added seed rows (run 1).** The opt seed of F212N holds 97 rows that the ctl seed lacks (53 from the far-companion exemption of #1377, 44 from the relaxed bright window of #1376). 97 of them have a row within 0.1 arcsec in the cross-band m7 catalog and 49 of those pass `oksep_pass`; 48 fail it, 47 of them rows of the exemption (open decision below). The process-global seed options of run 1 add 2410 rows in F480M.
- **Run 2, the two regressions of run 1** (opt arm with the updated #1366, #1376, #1377; ctl and production reused from run 1). F212N leftover (8653, 1078): box extremum 257 MJy/sr in the run 2 unmasked render, 257 in ctl, 4811 in the run 1 unmasked render (within a factor 1.5 of ctl); F480M leftover (410, 1680): box extremum 19 MJy/sr in the run 2 unmasked render, 19 in ctl, 226 in the run 1 unmasked render (within a factor 1.5 of ctl). Features ≥200 MJy/sr of the run 2 unmasked render that exceed the ctl extremum in the same box by more than a factor 1.5 (leftovers and craters, both bands): 0.
- **Run 2, counts.** F212N: leftovers ≥100 MJy/sr 195 unmasked and 78 with the real masks (run 1: 195 and 78); craters ≥50 MJy/sr 129 and 10 (run 1: 128 and 10). F480M: leftovers ≥100 MJy/sr 212 unmasked and 6 with the real masks (run 1: 213 and 7); craters ≥50 MJy/sr 29 and 0 (run 1: 30 and 1).
- **Run 2 against run 1.** F212N: 2 features of the run 2 unmasked render exceed the run 1 unmasked render by more than a factor 1.5, a leftover at (8385, 2328) and a crater at (8388, 2328), both at one star; F480M: 0. Box extremes at (8385, 2328), maximum / minimum in MJy/sr: ctl 2430 / -2904, run 1 unmasked 54 / -139, run 2 unmasked 2663 / -529 (the maximum of run 2 is within a factor 1.5 of that of ctl); the masked renders have no finite pixel in the box. The cause is not isolated (section on run 2). The F480M crater (3701, 498) of run 1 has a box depth of 5 MJy/sr in run 2 (ctl 5, run 1 60). The code (the heads of #1366, #1376 and #1377) and the F480M values of the seed options both differ between the runs; no cause is assigned.

## How the counts are made

- **Whole-star leftover.** The residual mosaic gets a local background and sigma (median and 1.4826 × MAD in blocks of 64 px for F212N and 32 px for F480M). Peaks are the local maxima of the background-subtracted residual after smoothing with a Gaussian of sigma = max(FWHM/2.355, 0.7) px, found in a box of max(5, round(2 FWHM)) px (made odd) and above max(25 sigma, 20 MJy/sr); a peak value is the 3 × 3 maximum; FWHM = 0.072 arcsec (F212N) and 0.164 arcsec (F480M). A peak is a whole-star leftover when its residual peak exceeds 0.7 times the data peak (3 × 3 maximum of the data i2d minus the block background) and exceeds 50 MJy/sr. The tables count leftovers with a residual peak ≥N MJy/sr for N = 50, 100, 200, 500, 1000.
- **Class of a leftover** (first match wins; the catalogs are those of the arm that made the render): SATSTAR_NEAR = a consolidated satstar fit within 1.0 arcsec; VETTED_BUT_UNSUBTRACTED = a vetted m7 row within max(1 FWHM, 0.06 arcsec); VETTED_OUT = a row of the raw m7 catalog within that radius; NOT_IN_M7_CATALOG = none of these.
- **Big crater.** The same smoothed, background-subtracted residual gives local minima below −max(25 sigma, 20 MJy/sr). The depth is the 3 × 3 maximum of −(residual − background) and the model peak is the 3 × 3 maximum of the merged-catalog model i2d. A big crater has depth above 50 MJy/sr and depth above 0.3 times the model peak; the tables count craters with depth ≥50, 200, 500 MJy/sr.
- **3 px matching** (pair tables). For each leftover ≥100 MJy/sr of render A: *cleared* = no leftover of render B within 3 px (0.094 arcsec; a leftover inside a region that a mask sets to NaN has no counterpart in the masked render and counts as cleared); *reduced* = the B peak at the matched leftover is below 0.5 times the A peak; *worse* = above 1.5 times the A peak; *unchanged* = otherwise. A leftover ≥100 MJy/sr of B with no counterpart in A within the same radius is *listed as new*. Big craters (depth ≥50 MJy/sr) of A without a crater of B within the radius are *gone*; craters of B without a counterpart in A are *listed as new*. A feature whose extremum moves by more than 3 px between the renders is counted once as cleared (or gone) and once as listed as new.
- **Position-independent scan.** Every leftover of the scanned render with a residual peak ≥200 MJy/sr (and every big crater of depth ≥200 MJy/sr) is compared with the extremum of a reference render in the 13 × 13 pixel box centred on it (maximum of the residual for a leftover, depth of the minimum for a crater), so a shift of the extremum by a few pixels does not change the verdict. With S the extremum of the scanned feature and R the reference extremum in the box: *absent in the reference* = R < 200 MJy/sr and S > 1.5 R; *larger than in the reference* = R ≥ 200 MJy/sr and S > 1.5 R; *within a factor 1.5* = S ≤ 1.5 R and R ≤ 1.5 S; *smaller than in the reference* = R > 1.5 S. The scan runs in both directions: the first direction scans the features of the opt render against ctl (regressions), the second scans the features of ctl against the opt render (the same four classes read as cleared, reduced, within a factor 1.5, larger). The box extremum comes from the whole residual image, so a reference extremum can belong to a source that fails the whole-star criteria.
- **Real masks and library masks.** The finalize step of the opt arm applies the footprint-mask steps of the opt environment (table below) to the residual i2d. A *library mask* applies a named step list to the stored unmasked residual of an arm; the preset `k4full` on the unmasked opt render reproduces the finalize render (validation lines below). *Emulation* presets (`k4emu`, `k1emu`) omit ring tier 2 and the gate-rejected fits, as the emulations of the earlier tile A/Bs did. The ctl tables lack the `sidelobe_maskable` column that the gate-rejected step reads, so the full presets run on the opt arm only.

| preset | K | margin (px) | ring zone | ring tier 2 | gate-rejected fits | DQ_SAT dilation | edge band |
| :--- | ---: | ---: | :--- | :--- | :--- | :--- | :--- |
| `k4full` | 4 | 2 | flux ≥ 1e6 within 25 px | flux ≥ 3e4 within 8 px | yes | 4 px (components ≥ 4 px) | 8 px |
| `k4emu` | 4 | 2 | flux ≥ 1e6 within 25 px | none | no | 4 px (components ≥ 4 px) | 8 px |
| `k1emu` | 1 | 2 | flux ≥ 1e6 within 25 px | none | no | 4 px (components ≥ 4 px) | 8 px |
| `k1full` | 1 | 2 | flux ≥ 1e6 within 25 px | flux ≥ 3e4 within 8 px | yes | 4 px (components ≥ 4 px) | 8 px |
| `v480k4full` | 4 | 2 | flux ≥ 1e6 within 20 px | flux ≥ 3e4 within 8 px | yes | 3 px (components ≥ 4 px) | 6 px |
| `v480k4emu` | 4 | 2 | flux ≥ 1e6 within 20 px | none | no | 3 px (components ≥ 4 px) | 6 px |
| `v480k1emu` | 1 | 2 | flux ≥ 1e6 within 20 px | none | no | 3 px (components ≥ 4 px) | 6 px |

Validation of the library mask against the run 1 finalize render (lines of the validation step; A is the first and B the second render named in a line; `optnm` = unmasked opt render, `optfin` = finalize render with the real masks, `libmask-k4full` = preset `k4full` applied to the unmasked render):

- `optnm-vs-optfin-F212N: NaN only in A 0, NaN only in B 8119113, NaN in both 2363545, finite in both 43708622; max |A-B| 0, px with |A-B| > 1e-3: 0`
- `optnm-vs-optfin-F480M: NaN only in A 0, NaN only in B 3034817, NaN in both 393499, finite in both 9430884; max |A-B| 0, px with |A-B| > 1e-3: 0`
- `optfin-vs-libmask-k4full-F212N: NaN only in A 0, NaN only in B 0, NaN in both 10482658, finite in both 43708622; max |A-B| 0, px with |A-B| > 1e-3: 0`
- `optfin-vs-libmask-k4full-F480M: NaN only in A 0, NaN only in B 0, NaN in both 3428316, finite in both 9430884; max |A-B| 0, px with |A-B| > 1e-3: 0`

## Recipe

Environment of the opt arm (the ctl arm exports none of these variables). The lines are the file of the run with its comments replaced by the number of the PR that introduces the switch; run 2 uses the same file (the environment diff of the two runs shows it unchanged).

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

`SATSTAR_SIDELOBE_FRAC_MIN` (#1368, a fractional waiver of the sidelobe cut in the satstar accept gate) is set in neither arm; #1369 marks the sidelobe-only rejects that a mask can cover, and `SATSTAR_SIDELOBE_MASK_FRAC_MIN` in the block above hands them to the footprint mask. `SATSTAR_DAO_RESTORE_FRAC_MIN` is 0.8 in the opt arm of both runs.

Command-line options of the m7 stage that differ between the arms. A plain number applies to every band; comma-separated `BAND=VALUE` pairs apply per band, and a band that no pair names keeps the default of the option (#1376, #1377). Run 1 used plain numbers, run 2 the per-band form (F480M exemption off, F480M bright-seed S/N bound at its default):

| option | PR | ctl (default) | run 1 opt | run 2 opt |
| :--- | :--- | ---: | ---: | ---: |
| `--manual-m7-seed-own-band-companion-keep-snr` | #1377 | 0 | 50 | F212N=50,F480M=0 |
| `--manual-seed-round-bright-max` | #1376 | 0 | 1.5 | 1.5 |
| `--manual-seed-round-bright-snr-min` | #1376 | 100 | 30 | F212N=30,F480M=100 |

Both arms of both runs pass the same base options: `--manual-ext-star-prom-robust-min=-1` `--residual-bg-median-size=0` `--residual-bg-satstar-mask-fwhm=3.75` `--manual-seed-round-loose-max=-1`.

Merge order of the integration trees (both runs): #1360, #1361, #1362, #1364, #1370, #1372, #1373, #1374, #1375, #1367, #1365, #1366, #1371, #1368, #1369, #1376, #1377. #1380 (documentation of the K=4 mask) was not merged. Base branches, titles and heads as of 2026-11-25 (`gh pr view`); the heads used by the runs come from the merge commits of the integration trees (`results/integration_trees.txt`).

| PR | title | base branch | head in run 1 | head in run 2 | head on 2026-11-25 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| #1360 | Opt-in guards that remove bright-star leftovers and craters from the merged-catalog residual | `main` | d504c167 | as run 1 | d504c167 |
| #1361 | Opt-in footprint mask for the merged-catalog residual i2d (stacked on #1360) | `mergedcat-residual-guards` | 1424a31e | as run 1 | 1424a31e |
| #1362 | Opt-in restore of the daophot row that a dropped satstar row overwrote (SATSTAR_DAO_RESTORE_FRAC_MIN) | `mergedcat-residual-footprint-mask` | 2ebe41c1 | as run 1 | 2ebe41c1 |
| #1364 | Add a ring-zone extension to the residual footprint mask (opt-in, stacked on #1362) | `satstar-dao-restore` | f095649f | as run 1 | f095649f |
| #1370 | Mask the footprints of maskable gate-rejected satstar fits in the residual i2d | `footprint-mask-ring-zone` | aaeceeac | as run 1 | aaeceeac |
| #1372 | Add a DQ_SATURATED dilation to the residual footprint mask | `footprint-mask-rejected-satstars` | cae05e94 | as run 1 | cae05e94 |
| #1373 | Add an opt-in guard band along the data footprint edge to the residual footprint mask | `footprint-mask-dqsat` | 9434ba67 | as run 1 | 9434ba67 |
| #1374 | Add an opt-in positive one-pixel spike mask for the per-frame residuals | `footprint-mask-edge-band` | 56173877 | as run 1 | 56173877 |
| #1375 | Add an opt-in second flux/radius tier to the footprint mask ring-zone extension | `footprint-mask-edge-band` | 43d61ec7 | as run 1 | 43d61ec7 |
| #1367 | Opt-in ramp test for single-exposure events (SINGLE_EXPOSURE_EVENT_RAMP) | `mergedcat-residual-guards` | 9733dfc0 | as run 1 | 9733dfc0 |
| #1365 | combine_singleframe: opt-in flux-aware base-row seeding (MERGE_PAIR_RESCUE_RATIO) | `main` | 622fe467 | as run 1 | 622fe467 |
| #1366 | m7 post-fit dedup: keep the positive-flux row against a non-positive fit (DAOPHOT_DEDUP_NONPOSITIVE, default off) | `main` | cf5a29d2 | 38873cf8 | 38873cf8 |
| #1371 | Satstar: opt-in LevMar iteration cap and edge position bound | `main` | 60e6f3a0 | as run 1 | 60e6f3a0 |
| #1368 | Satstar accept gate: opt-in fractional waiver of the sidelobe cut | `main` | 3daf043f | as run 1 | 3daf043f |
| #1369 | Satstar accept gate: mark sidelobe-only rejects that a mask can cover (opt-in) | `satstar-sidelobe-frac` | 383a1403 | as run 1 | 383a1403 |
| #1376 | Opt-in bright roundness window for the m7 i2d seed (relaxed bright seeds) | `main` | 52549326 | 14b9308c | 770a1a5c |
| #1377 | Add an opt-in far-companion exemption to the m7 own-band companion cut | `main` | 8a15821b | a5926b42 | 1553ed42 |
| #1380 | Document the footprint mask at KFAC 2-4 on five tiles (counts, masked area, profile-correction test) | `footprint-mask-edge-band` | not merged | - | b018b059 |

Two PR pairs conflict when merged in this order; both were resolved by keeping both sides (the resolutions exist in the local integration trees only):

1. **#1367 (ramp events) and #1374 (positive spikes)** extend the same builder of the per-frame single-exposure masks in `crowdsource_catalogs_long.py` and the same helper module of the event tools (six conflicting hunks). Both sides are kept: the constants and functions of the two PRs sit side by side in the helper module, and in the builder the early return and the cache key read `events_on or ramp_on or nsigma or spike_min` and `(files, events_on, ramp_on, nsigma, spike_min)`, while the body holds the ramp, negative-pixel and spike blocks, each OR-ing into the mask of the frame. Every switch keeps its own environment variable and its default of off.
2. **#1368 and #1371** both append cache-signature parts after `'es'` in `saturated_star_finding.py` and rows to the environment table of `SATURATED_PIXEL_HANDLING.md`. The merged signature appends the parts in the order `mi`, `eb`, `sf`; the table keeps the rows of `SATSTAR_FITTER_MAXITERS`, `SATSTAR_EDGE_POS_BOUND` / `SATSTAR_EDGE_MIN_AREA` and `SATSTAR_SIDELOBE_FRAC_MIN`. A frame catalog stamped with another combination of these options is refit, as intended.

## Setup of the two runs

- **Code.** Run 1: origin/main 803d0794 plus the 17 merges above (local branch, tip 3d7b94cd; 81 files, +17,525 / -81 lines against origin/main, evidence files of the PRs included). Run 2: the run 1 tree plus the heads of #1366 (38873cf8), #1377 (a5926b42) and #1376 (14b9308c) (tip dda52d96; 123 files, +21,305 / -82 lines; no conflicts); they replace the heads of run 1 for these three PRs (cf5a29d2, 8a15821b, 52549326). On 2026-11-25 the heads used by the runs equal the current heads of their PRs with 2 exceptions: #1376 moved from 14b9308c to 770a1a5c (1 commit; changed files with the suffixes `.md` and `.txt`; no Python file changed); #1377 moved from a5926b42 to 1553ed42 (1 commit; changed files with the suffixes `.md` and `.txt`; no Python file changed).
- **Renders compared.** `prod` is the production render of the o045 residual of the release tag (read only). `ctl` and `opt` run the integration code with the switches off and on. Production and ctl differ because the integration code is ahead of the code of the production render; the production → ctl rows of the pair tables show the size of that difference. The opt arm is rendered twice: with the real masks of the finalize step (`optfin`) and without (`optnm`), so that masked and unmasked renders can be compared.
- **Inputs and outputs.** Each arm runs on its own scratch base path (`GC_BASEPATH_OVERRIDE`). The inputs of the production tree are staged into it as symlinks (their targets were checked after run 1, see below) and as copies of the files that a job writes.
- **Jobs.** Each arm runs the m7 per-frame fits as an 8-task array, then the finalize job (merge, vetting, finalize render); scoring, masks and comparisons run in later jobs. QOS astronomy-dept-b, partition hpg-dev.

| job name | what it did | completed jobs | elapsed per job (h) | peak RSS (GB) |
| :--- | :--- | ---: | ---: | ---: |
| `e2e045-ctl-m7-fan` | run 1 ctl: m7 per-frame fits (array) | 8 | 1.67 to 3.12 | 18.9 |
| `e2e045-opt-m7-fan` | run 1 opt: m7 per-frame fits (array) | 8 | 1.96 to 3.56 | 17.6 |
| `e2e045-ctl-m7-fin` | run 1 ctl: merge, vetting, finalize render | 1 | 2.16 | 15.4 |
| `e2e045-opt-m7-fin` | run 1 opt: merge, vetting, finalize render with the real masks | 1 | 2.19 | 16.3 |
| `e2f-m7-fan` | run 2 opt: m7 per-frame fits (array) | 8 | 1.72 to 2.77 | 17.6 |
| `e2f-m7-fin` | run 2 opt: merge, vetting, finalize render with the real masks | 1 | 2.19 | 12.9 |
| `e2fpost` | run 2: scoring, library masks, comparisons, CRDS check | 1 | 0.42 | 13.1 |

Run 2 (Slurm accounting, `results/run2_sacct.txt` and `results/run2_sacct_post.txt`): the fan-out array 52781393 started its 8 tasks at 12:39:25 and they ended between 14:22:21 and 15:25:50; the finalize job 52781394 ran from 15:25:51 to 17:37:06 (02:11:15); the post job 52782455 ran from 17:37:08 to 18:02:32 (00:25:24), all on 2026-11-25. All 10 jobs ended COMPLETED with exit code 0:0. The log of the post job marks the step `crds_check` as `FAILED rc=0`. Its two output files exist (`results/run2_crds_after.txt`, `results/run2_crds_diff.txt`).

**Check of /orange after run 1.** The marker time is 2026-11-24T22:00:23-05:00, 5 s before the first job of run 1 started (Slurm accounting). (1) Entries with `o045` in the name that are newer than the marker were listed for 6 directories of the production tree (`catalogs`, `F212N/pipeline`, `F480M/pipeline`, `astrometry_checkpoints`, `offsets`, `psfs`). Only `catalogs` has any: 6 entries (4 FITS catalogs of the m8 stage and 2 `.prov.json` files), with times from 2026-11-24 22:08 to 2026-11-24 22:13. Slurm accounting lists job 52646573 (`gctreasury10678-o045-m8-nbrsub`, partition hpg-milan, 2026-11-24T19:17:17 to 2026-11-24T22:13:37), which is not a job of run 1 (the jobs of run 1 are named `e2e045-...`), started before the marker and ended in the minute of the newest of these files. (2) The 198 symlink targets that the staging links to have the same modification time and size as in a snapshot taken before the submission. (3) A scan of the CRDS tree (bounded to 600 s; it finished with exit code 0) finds two files newer than the marker, `config/jwst/bad_files.txt` and `config/jwst/server_config` of the shared CRDS cache, both with modification time 2026-11-25 03:04:53. The ctl finalize log has a CRDS lookup at 2026-11-25 03:04:53.718, 0.7 s after the file time; the nearest lookup of the opt finalize log is at 2026-11-25 03:02:54.933, 118.1 s before the file time. The CRDS client of the ctl finalize job therefore probably refreshed the two files; another process of the same user querying CRDS in that second is not excluded. Run 2 sets `CRDS_READONLY_CACHE=1` (check of run 2 below).

**Check of the CRDS cache files after run 2.** The listing taken before the first job of run 2 (`results/run2_crds_before.txt`) and the listing taken after its last job (`results/run2_crds_after.txt`; the `crds_check` step of the post job started at 18:02:31) give equal sizes for the two files (1457934 bytes for `server_config`, 5034 bytes for `bad_files.txt`) and the modification times 2026-11-25 03:04:53 before and 2026-11-25 17:50:37 after (`results/run2_crds_diff.txt`). Run 2 sets `CRDS_READONLY_CACHE=1`; the environment files of run 1 do not set it. The finalize log of run 2 has 1 CRDS client line (15:49:13.916) and 8 `PARS-RESAMPLESTEP` lines (15:49:17 to 16:54:10). The post log has CRDS client lines at 17:44:01.218 (step `render_nm_F212N`) and at 17:52:22.605 (step `render_nm_F480M`) and 2 `PARS-RESAMPLESTEP` lines (17:44:02 to 17:52:25); the 8 fan-out logs have none. No line lies within 100 s of 17:50:37; the nearest is 17:52:22.605, 105.6 s after it (`results/run2_crds_attribution.txt`). The CRDS client line at 17:52:22 (step `render_nm_F480M`) is later than the change at 17:50:37; the listing of the `crds_check` step (start 18:02:31) shows the modification time 17:50:37 for both files. The writer of the change at 17:50:37 is not identified. Slurm accounting lists 7 other jobs of the same user with `finalize` in the name that were running at 17:50:37: `wd11905-o001-m4-finalize` (start 16:32:53); `brick2221-o001-m6-finalize-F182M` (start 17:02:38); `brick2221-o001-m6-finalize-F187N` (start 17:25:56); `brick2221-o001-m6-finalize-F212N` (start 17:37:38); `cloudc2221-o002-m5-finalize-F182M` (start 17:41:46); `cloudc2221-o002-m5-finalize-F187N` (start 17:45:37); `cloudc2221-o002-m5-finalize-F212N` (start 17:49:32). They are candidates by name and start time; jobs with other names and the logs of these jobs were not examined. The finalize logs of both runs name the same reference for the parameters of the resample step (`jwst_nircam_pars-resamplestep_0001.asdf`); whether the whole CRDS context is the same in both runs is not verified.

## Run 1 results

Counts of the o045 residual mosaics (leftovers ≥N MJy/sr and craters ≥N MJy/sr as defined above); the last column is the share of finite pixels that the mask steps set to NaN.

**F212N**

| render | what it is | leftovers ≥50 | ≥100 | ≥200 | ≥500 | ≥1000 | craters ≥50 | ≥200 | ≥500 | masked % of finite px |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `prod` | production render of the release tag (no mask) | 1085 | 376 | 190 | 140 | 123 | 554 | 220 | 84 | 0 |
| `ctlfin` | ctl: integration code, every switch off (no mask) | 1076 | 369 | 191 | 141 | 123 | 522 | 190 | 72 | 0 |
| `optnm` | run 1 opt, unmasked re-render of the finalize | 857 | 195 | 77 | 46 | 34 | 128 | 56 | 14 | 0 |
| `optfin` | run 1 opt, finalize render with the real masks | 661 | 78 | 9 | 0 | 0 | 10 | 0 | 0 | 15.67 |

**F480M**

| render | what it is | leftovers ≥50 | ≥100 | ≥200 | ≥500 | ≥1000 | craters ≥50 | ≥200 | ≥500 | masked % of finite px |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `prod` | production render of the release tag (no mask) | 318 | 224 | 180 | 121 | 101 | 90 | 34 | 9 | 0 |
| `ctlfin` | ctl: integration code, every switch off (no mask) | 349 | 257 | 212 | 149 | 123 | 97 | 42 | 10 | 0 |
| `optnm` | run 1 opt, unmasked re-render of the finalize | 262 | 213 | 198 | 146 | 123 | 30 | 24 | 7 | 0 |
| `optfin` | run 1 opt, finalize render with the real masks | 47 | 7 | 3 | 0 | 0 | 1 | 0 | 0 | 24.35 |

Pair tables of the 3 px matching (leftovers ≥100 MJy/sr and craters ≥50 MJy/sr; *cleared* and *gone* are items of A without a counterpart in B, *listed as new* are items of B without a counterpart in A):

**F212N**

| A → B | what differs | leftovers ≥100 in A | in B | cleared | reduced | unchanged | worse | listed as new | craters ≥50 in A → B | gone | listed as new |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ctlfin` → `optnm` | all switches, no mask | 369 | 195 | 176 | 0 | 193 | 0 | 4 | 522 → 128 | 397 | 3 |
| `ctlfin` → `optfin` | all switches and the real masks | 369 | 78 | 294 | 0 | 75 | 0 | 6 | 522 → 10 | 514 | 2 |
| `prod` → `ctlfin` | code drift, production → ctl | 376 | 369 | 21 | 0 | 355 | 0 | 12 | 554 → 522 | 85 | 53 |
| `prod` → `optfin` | production → opt with masks | 376 | 78 | 302 | 0 | 74 | 0 | 7 | 554 → 10 | 547 | 3 |

**F480M**

| A → B | what differs | leftovers ≥100 in A | in B | cleared | reduced | unchanged | worse | listed as new | craters ≥50 in A → B | gone | listed as new |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ctlfin` → `optnm` | all switches, no mask | 257 | 213 | 48 | 0 | 209 | 0 | 3 | 97 → 30 | 69 | 2 |
| `ctlfin` → `optfin` | all switches and the real masks | 257 | 7 | 251 | 0 | 6 | 0 | 1 | 97 → 1 | 97 | 1 |
| `prod` → `ctlfin` | code drift, production → ctl | 224 | 257 | 26 | 0 | 197 | 1 | 59 | 90 → 97 | 23 | 30 |
| `prod` → `optfin` | production → opt with masks | 224 | 7 | 218 | 0 | 6 | 0 | 1 | 90 → 1 | 90 | 1 |

Leftovers of ctl that the 3 px matching clears at higher thresholds, and craters ≥200 MJy/sr:

| band | pair | cleared of A ≥200 | cleared of A ≥500 | cleared of A ≥1000 | craters ≥200 |
| :--- | :--- | ---: | ---: | ---: | :--- |
| F212N | `ctlfin:optnm` | 116 of 191 | 96 of 141 | 90 of 123 | 136 gone, 2 listed as new |
| F212N | `ctlfin:optfin` | 185 of 191 | 141 of 141 | 123 of 123 | 190 gone, 0 listed as new |
| F480M | `ctlfin:optnm` | 17 of 212 | 5 of 149 | 2 of 123 | 18 gone, 1 listed as new |
| F480M | `ctlfin:optfin` | 210 of 212 | 149 of 149 | 123 of 123 | 42 gone, 0 listed as new |

Leftovers ≥200 MJy/sr by class of the nearest catalog entry (SATSTAR_NEAR = consolidated satstar fit within 1 arcsec; VETTED_BUT_UNSUBTRACTED = vetted row within one FWHM; VETTED_OUT = raw row only; NOT_IN_M7_CATALOG = none of these):

| band | render | SATSTAR_NEAR | NOT_IN_M7_CATALOG | VETTED_BUT_UNSUBTRACTED | VETTED_OUT |
| :--- | :--- | ---: | ---: | ---: | ---: |
| F212N | `ctlfin` | 142 | 31 | 15 | 3 |
| F212N | `optnm` | 65 | 8 | 2 | 2 |
| F212N | `optfin` | 1 | 7 | 1 | 0 |
| F480M | `ctlfin` | 181 | 24 | 7 | 0 |
| F480M | `optnm` | 179 | 16 | 3 | 0 |
| F480M | `optfin` | 0 | 1 | 2 | 0 |

**Position-independent scan of run 1.** Features ≥200 MJy/sr of the opt renders against the box extremum of ctl (regressions), then the features of ctl against the opt renders (improvements); classes as defined above.

| band | scanned render | reference render | feature | features ≥200 MJy/sr in the scanned render | absent in the reference | larger than in the reference | within a factor 1.5 | smaller than in the reference |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| F212N | `optnm` | `ctlfin` | leftovers | 77 | 0 | 1 | 66 | 10 |
| F212N | `optnm` | `ctlfin` | craters | 56 | 0 | 0 | 54 | 2 |
| F212N | `optfin` | `ctlfin` | leftovers | 9 | 0 | 0 | 4 | 5 |
| F212N | `optfin` | `ctlfin` | craters | 0 | 0 | 0 | 0 | 0 |
| F480M | `optnm` | `ctlfin` | leftovers | 198 | 1 | 0 | 183 | 14 |
| F480M | `optnm` | `ctlfin` | craters | 24 | 0 | 0 | 22 | 2 |
| F480M | `optfin` | `ctlfin` | leftovers | 3 | 1 | 0 | 2 | 0 |
| F480M | `optfin` | `ctlfin` | craters | 0 | 0 | 0 | 0 | 0 |

| band | render with the features | compared render | feature | features ≥200 MJy/sr | cleared in the compared render | reduced, still ≥200 | within a factor 1.5 | larger in the compared render |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| F212N | `ctlfin` | `optnm` | leftovers | 191 | 114 | 1 | 67 | 9 |
| F212N | `ctlfin` | `optnm` | craters | 190 | 98 | 12 | 73 | 7 |
| F212N | `ctlfin` | `optfin` | leftovers | 191 | 183 | 1 | 6 | 1 |
| F212N | `ctlfin` | `optfin` | craters | 190 | 190 | 0 | 0 | 0 |
| F480M | `ctlfin` | `optnm` | leftovers | 212 | 14 | 3 | 181 | 14 |
| F480M | `ctlfin` | `optnm` | craters | 42 | 13 | 0 | 25 | 4 |
| F480M | `ctlfin` | `optfin` | leftovers | 212 | 209 | 1 | 2 | 0 |
| F480M | `ctlfin` | `optfin` | craters | 42 | 42 | 0 | 0 | 0 |

The scan and the 3 px matching test different things, so their counts for the same pair of renders differ (compare the ≥200 MJy/sr crater counts of the two tables). The matching asks whether the other render has an item within 3 px that meets the leftover criteria or the big-crater criteria; the scan asks for the extremum of the other residual image in the 13 × 13 pixel box, whether or not that extremum meets those criteria.

Features of the unmasked opt render that are worse than ctl by the scan rule (2):

- F212N leftover (8653, 1078): 4803 MJy/sr in `optnm`, 257 MJy/sr in `ctlfin` (weak; class SATSTAR_NEAR)
- F480M leftover (410, 1680): 214 MJy/sr in `optnm`, 19 MJy/sr in `ctlfin` (regression; class VETTED_BUT_UNSUBTRACTED)

In the masked render `optfin` the same scan finds 1 such feature: F480M leftover (410, 1680) 214 against 19 MJy/sr.

**The items that the 3 px matching lists as new.** Each item listed as new in the pair tables was looked up in the scan (box extremum of the scanned render and of ctl at its position). *Worse* uses the scan rule (extremum ≥200 MJy/sr and more than 1.5 times the ctl extremum); every other item has a box extremum below 200 MJy/sr in the scanned render or at most 1.5 times the ctl extremum, so it is no regression by this rule.

| band | pair | items listed as new | count | ≥200 MJy/sr and at most 1.5 × ctl | below 200 MJy/sr | worse than ctl | worse items: scanned against ctl (MJy/sr) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| F212N | `ctlfin:optnm` | leftovers | 4 | 2 | 1 | 1 | (8653, 1078) 4811 against 257 |
| F212N | `ctlfin:optnm` | craters | 3 | 2 | 1 | 0 | - |
| F212N | `ctlfin:optfin` | leftovers | 6 | 5 | 1 | 0 | - |
| F212N | `ctlfin:optfin` | craters | 2 | 0 | 2 | 0 | - |
| F480M | `ctlfin:optnm` | leftovers | 3 | 2 | 0 | 1 | (410, 1680) 226 against 19 |
| F480M | `ctlfin:optnm` | craters | 2 | 1 | 1 | 0 | - |
| F480M | `ctlfin:optfin` | leftovers | 1 | 0 | 0 | 1 | (410, 1680) 226 against 19 |
| F480M | `ctlfin:optfin` | craters | 1 | 0 | 1 | 0 | - |

The F480M crater at (3701, 498) is listed as new with a box depth of 60 MJy/sr against 5 MJy/sr in ctl, below the 200 MJy/sr scan threshold. Its cause is open (see the limits below).

## The two regressions of run 1 and the update of #1366

Both features trace to #1366 (`DAOPHOT_DEDUP_NONPOSITIVE`), through two branches of it. Replays of the per-frame m7 pass with all switches on reproduce the opt rows at both stars, and with #1366 off they reproduce the ctl rows; with #1377 off, both F212N frames keep the two rows (`results/regress_mechanism_and_fix.txt`, section 3 of the analysis).

**F212N (8653, 1078).** In nrcb4 exposure 2 (group 702) the grouped fit of the star holds a +1.07e5 / -1.06e5 pair of rows 0.2 px apart (net flux about +1e3), 1.27 px (0.54 FWHM) from the seed of the star (seed flux 2.77e4). Without #1366 the 1.0 px dedup keeps the negative row of the pair, the ban on non-positive fluxes removes it, and the star keeps one row (fitted flux 3.78e4 in nrcb4 exposure 2). With the form of #1366 used in run 1 the positive row of the pair survives and is refit at its own seed, which takes light from the star: the two rows have fluxes 3.51e4 and 1.72e4 (nrcb2 exposure 4: 3.39e4 and 1.89e4 against one row of 4.77e4 in ctl). The merged DAO flux falls from 6.12e4 to 4.33e4. The faint-replacement guard of the merge keeps the DAO row while the satstar flux is below 0.8 of the DAO flux: in ctl 0.8 × 6.12e4 = 4.90e4 exceeds the satstar flux of 4.81e4 (the guard holds with a margin of 1.7 %; the fluxes printed here are rounded to three digits), in opt 0.8 × 4.33e4 = 3.46e4 is below it, so the satstar row (qfit 2.97, 2.4 px from the DAO row) replaces the DAO row. The residual of the star goes from +257 / -1598 MJy/sr (maximum / minimum) in ctl to +4811 / -954 MJy/sr in opt.

**F480M (410, 1680).** The merged vetted row has flux 1070 (qfit 0.78, 4 matches) in opt and 3976 (qfit 0.01, 2 matches) in ctl. In the frame nrcalong exposure 6 two seeds (fluxes 3936 and 1253) lie 1.32 px (0.51 FWHM) apart. With #1366 the refit recovers the first row (flux 3943.8, qfit 0.006) and keeps the second as a rescued row (flux 107.7, qfit 1.11), because the rescue test asks for 1.0 px between the seeds. The drop of the merged flux to 1070 was not replayed through the merge.

**Update (#1366, head 38873cf8).** A rescued row must lie at least 0.75 FWHM from the seed of the survivor (1.76 px in F212N, 1.94 px in F480M); a survivor whose net flux (its own flux plus the non-positive rows it absorbed) does not exceed the fitted flux of a kept row within the dedup radius of its seed is dropped as a duplicate of that row, judged in order of decreasing strength; a row that the fit must keep is never dropped by that rule. The value 0.75 FWHM lies above the separations of the two regressions (0.54 and 0.51 FWHM) and was chosen by judgement; the seed dedup removes pairs within 1.17 px in F212N (about 0.5 FWHM). Replay of the per-frame tables with the earlier and the updated rule (rows within 2 px of the star: flux, qfit, distance):

| frame | switch off: rows within 2 px of the star (flux, qfit, distance) | earlier rule | this update |
|---|---|---|---|
| F212N nrcb4 exp 2 (mosaic star (8653,1078)) | 37825.3 (qfit 0.0157, 0.08 px) | 35092.7 (qfit 0.0157, 0.05 px); 17249.2 (qfit 0.0106, 1.27 px) | 37825.3 (qfit 0.0157, 0.08 px) |
| F212N nrcb2 exp 4 (mosaic star (8653,1078)) | 47710.9 (qfit 0.0118, 0.06 px) | 33936.2 (qfit 0.0118, 0.19 px); 18947.5 (qfit 0.00791, 1.20 px) | 47710.9 (qfit 0.0118, 0.06 px) |
| F480M nrcalong exp 6 (mosaic star (410,1680)) | none | 3943.8 (qfit 0.00573, 0.05 px); 107.7 (qfit 1.11, 1.36 px) | 3943.8 (qfit 0.00573, 0.05 px) |

Besides these three frames, the update was replayed on 11 frames (six o130 F212N frames, one o105 frame, two o045 frames and two o138 F480M frames) with the control, the earlier form and the update; the stop rule of the update (a star of 1e4 or more that the earlier form recovers must not be lost) was checked on all 11 and not violated. The description of PR #1366 and the `DAOPHOT_DEDUP_NONPOSITIVE` row of the environment table in `SATURATED_PIXEL_HANDLING.md` on that branch hold the rules and the replay numbers. The tile-level A/B of #1366 (o130 and o105 F212N) was measured with the earlier form and has not been repeated.

## Run 2 results

Counts of the o045 residual mosaics of run 2 (`opt2nm`, `opt2fin`) next to the run 1 renders:

**F212N**

| render | what it is | leftovers ≥50 | ≥100 | ≥200 | ≥500 | ≥1000 | craters ≥50 | ≥200 | ≥500 | masked % of finite px |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `prod` | production render of the release tag (no mask) | 1085 | 376 | 190 | 140 | 123 | 554 | 220 | 84 | 0 |
| `ctlfin` | ctl: integration code, every switch off (no mask) | 1076 | 369 | 191 | 141 | 123 | 522 | 190 | 72 | 0 |
| `optnm` | run 1 opt, unmasked re-render of the finalize | 857 | 195 | 77 | 46 | 34 | 128 | 56 | 14 | 0 |
| `optfin` | run 1 opt, finalize render with the real masks | 661 | 78 | 9 | 0 | 0 | 10 | 0 | 0 | 15.67 |
| `opt2nm` | run 2 opt, unmasked re-render of the finalize | 857 | 195 | 77 | 46 | 34 | 129 | 57 | 15 | 0 |
| `opt2fin` | run 2 opt, finalize render with the real masks | 661 | 78 | 9 | 0 | 0 | 10 | 0 | 0 | 15.67 |

**F480M**

| render | what it is | leftovers ≥50 | ≥100 | ≥200 | ≥500 | ≥1000 | craters ≥50 | ≥200 | ≥500 | masked % of finite px |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `prod` | production render of the release tag (no mask) | 318 | 224 | 180 | 121 | 101 | 90 | 34 | 9 | 0 |
| `ctlfin` | ctl: integration code, every switch off (no mask) | 349 | 257 | 212 | 149 | 123 | 97 | 42 | 10 | 0 |
| `optnm` | run 1 opt, unmasked re-render of the finalize | 262 | 213 | 198 | 146 | 123 | 30 | 24 | 7 | 0 |
| `optfin` | run 1 opt, finalize render with the real masks | 47 | 7 | 3 | 0 | 0 | 1 | 0 | 0 | 24.35 |
| `opt2nm` | run 2 opt, unmasked re-render of the finalize | 268 | 212 | 197 | 146 | 123 | 29 | 24 | 7 | 0 |
| `opt2fin` | run 2 opt, finalize render with the real masks | 54 | 6 | 2 | 0 | 0 | 0 | 0 | 0 | 24.35 |

- F212N, run 1 → run 2: the real-mask renders have equal counts in every threshold column; the unmasked renders differ in craters ≥50: 128 → 129; craters ≥200: 56 → 57; craters ≥500: 14 → 15.
- F480M, run 1 → run 2: the real-mask renders differ in leftovers ≥50: 47 → 54; leftovers ≥100: 7 → 6; leftovers ≥200: 3 → 2; craters ≥50: 1 → 0; the unmasked renders differ in leftovers ≥50: 262 → 268; leftovers ≥100: 213 → 212; leftovers ≥200: 198 → 197; craters ≥50: 30 → 29.

**F212N** (pair tables of the 3 px matching)

| A → B | what differs | leftovers ≥100 in A | in B | cleared | reduced | unchanged | worse | listed as new | craters ≥50 in A → B | gone | listed as new |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ctlfin` → `opt2nm` | run 2 all switches, no mask | 369 | 195 | 175 | 0 | 194 | 0 | 3 | 522 → 129 | 396 | 3 |
| `ctlfin` → `opt2fin` | run 2 all switches and the real masks | 369 | 78 | 294 | 0 | 75 | 0 | 6 | 522 → 10 | 514 | 2 |
| `optnm` → `opt2nm` | run 1 → run 2, no mask | 195 | 195 | 1 | 0 | 194 | 0 | 1 | 128 → 129 | 0 | 1 |
| `optfin` → `opt2fin` | run 1 → run 2, real masks | 78 | 78 | 0 | 0 | 78 | 0 | 0 | 10 → 10 | 0 | 0 |

**F480M** (pair tables of the 3 px matching)

| A → B | what differs | leftovers ≥100 in A | in B | cleared | reduced | unchanged | worse | listed as new | craters ≥50 in A → B | gone | listed as new |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ctlfin` → `opt2nm` | run 2 all switches, no mask | 257 | 212 | 48 | 0 | 209 | 0 | 2 | 97 → 29 | 69 | 1 |
| `ctlfin` → `opt2fin` | run 2 all switches and the real masks | 257 | 6 | 251 | 0 | 6 | 0 | 0 | 97 → 0 | 97 | 0 |
| `optnm` → `opt2nm` | run 1 → run 2, no mask | 213 | 212 | 1 | 0 | 212 | 0 | 0 | 30 → 29 | 1 | 0 |
| `optfin` → `opt2fin` | run 1 → run 2, real masks | 7 | 6 | 1 | 0 | 6 | 0 | 0 | 1 → 0 | 1 | 0 |

Pixel comparisons of the run 2 renders (lines of the validation step of the post job; A is the first and B the second render named in a line; `opt2nm` = run 2 unmasked render, `opt2fin` = run 2 finalize render with the real masks, `libmask-k4full` and `libmask-v480k4full` = the presets of the table above applied to the unmasked render):

- `ctlfin-vs-opt2fin-F212N: NaN only in A 0, NaN only in B 8119129, NaN in both 2363529, finite in both 43708622; max |A-B| 2.59e+03, px with |A-B| > 1e-3: 977018`
- `optfin-vs-opt2fin-F212N: NaN only in A 0, NaN only in B 0, NaN in both 10482658, finite in both 43708622; max |A-B| 46.6, px with |A-B| > 1e-3: 17912`
- `prod-vs-opt2fin-F212N: NaN only in A 0, NaN only in B 8119129, NaN in both 2363529, finite in both 43708622; max |A-B| 2.59e+03, px with |A-B| > 1e-3: 41952008`
- `ctlfin-vs-opt2fin-F480M: NaN only in A 0, NaN only in B 3034817, NaN in both 393499, finite in both 9430884; max |A-B| 518, px with |A-B| > 1e-3: 276470`
- `optfin-vs-opt2fin-F480M: NaN only in A 0, NaN only in B 0, NaN in both 3428316, finite in both 9430884; max |A-B| 215, px with |A-B| > 1e-3: 966394`
- `prod-vs-opt2fin-F480M: NaN only in A 0, NaN only in B 3034817, NaN in both 393499, finite in both 9430884; max |A-B| 518, px with |A-B| > 1e-3: 8362590`
- `opt2nm-vs-opt2fin-F212N: NaN only in A 0, NaN only in B 8119113, NaN in both 2363545, finite in both 43708622; max |A-B| 0, px with |A-B| > 1e-3: 0`
- `optnm-vs-opt2nm-F212N: NaN only in A 0, NaN only in B 0, NaN in both 2363545, finite in both 51827735; max |A-B| 5.43e+03, px with |A-B| > 1e-3: 25871`
- `opt2nm-vs-opt2fin-F480M: NaN only in A 0, NaN only in B 3034817, NaN in both 393499, finite in both 9430884; max |A-B| 0, px with |A-B| > 1e-3: 0`
- `optnm-vs-opt2nm-F480M: NaN only in A 0, NaN only in B 0, NaN in both 393499, finite in both 12465701; max |A-B| 215, px with |A-B| > 1e-3: 1182091`
- `opt2fin-vs-libmask-k4full-F212N: NaN only in A 0, NaN only in B 0, NaN in both 10482658, finite in both 43708622; max |A-B| 0, px with |A-B| > 1e-3: 0`
- `opt2fin-vs-libmask-k4full-F480M: NaN only in A 0, NaN only in B 0, NaN in both 3428316, finite in both 9430884; max |A-B| 0, px with |A-B| > 1e-3: 0`
- `opt2fin-vs-libmask-v480k4full-F480M: NaN only in A 65727, NaN only in B 0, NaN in both 3362589, finite in both 9430884; max |A-B| 0, px with |A-B| > 1e-3: 0`

The last line compares the F480M finalize render with the preset `v480k4full` (F480M-validated values): 65727 pixels are NaN in the finalize render and finite in the preset, and no pixel is NaN in the preset only.

**Position-independent scan of run 2** (unmasked render `opt2nm` against ctl and against the run 1 unmasked render `optnm`; the masked render `opt2fin` where scanned):

| band | scanned render | reference render | feature | features ≥200 MJy/sr in the scanned render | absent in the reference | larger than in the reference | within a factor 1.5 | smaller than in the reference |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| F212N | `opt2nm` | `ctlfin` | leftovers | 77 | 0 | 0 | 67 | 10 |
| F212N | `opt2nm` | `ctlfin` | craters | 57 | 0 | 0 | 54 | 3 |
| F212N | `opt2nm` | `optnm` | leftovers | 77 | 1 | 0 | 67 | 9 |
| F212N | `opt2nm` | `optnm` | craters | 57 | 1 | 0 | 56 | 0 |
| F212N | `opt2fin` | `ctlfin` | leftovers | 9 | 0 | 0 | 4 | 5 |
| F212N | `opt2fin` | `ctlfin` | craters | 0 | 0 | 0 | 0 | 0 |
| F212N | `opt2fin` | `optfin` | leftovers | 9 | 0 | 0 | 8 | 1 |
| F212N | `opt2fin` | `optfin` | craters | 0 | 0 | 0 | 0 | 0 |
| F480M | `opt2nm` | `ctlfin` | leftovers | 197 | 0 | 0 | 183 | 14 |
| F480M | `opt2nm` | `ctlfin` | craters | 24 | 0 | 0 | 22 | 2 |
| F480M | `opt2nm` | `optnm` | leftovers | 197 | 0 | 0 | 183 | 14 |
| F480M | `opt2nm` | `optnm` | craters | 24 | 0 | 0 | 22 | 2 |
| F480M | `opt2fin` | `ctlfin` | leftovers | 2 | 0 | 0 | 2 | 0 |
| F480M | `opt2fin` | `ctlfin` | craters | 0 | 0 | 0 | 0 | 0 |
| F480M | `opt2fin` | `optfin` | leftovers | 2 | 0 | 0 | 2 | 0 |
| F480M | `opt2fin` | `optfin` | craters | 0 | 0 | 0 | 0 | 0 |

| band | render with the features | compared render | feature | features ≥200 MJy/sr | cleared in the compared render | reduced, still ≥200 | within a factor 1.5 | larger in the compared render |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| F212N | `ctlfin` | `opt2nm` | leftovers | 191 | 113 | 1 | 68 | 9 |
| F212N | `ctlfin` | `opt2nm` | craters | 190 | 97 | 13 | 73 | 7 |
| F212N | `optnm` | `opt2nm` | leftovers | 77 | 0 | 1 | 67 | 9 |
| F212N | `optnm` | `opt2nm` | craters | 56 | 0 | 0 | 56 | 0 |
| F212N | `ctlfin` | `opt2fin` | leftovers | 191 | 183 | 1 | 6 | 1 |
| F212N | `ctlfin` | `opt2fin` | craters | 190 | 190 | 0 | 0 | 0 |
| F212N | `optfin` | `opt2fin` | leftovers | 9 | 0 | 0 | 8 | 1 |
| F212N | `optfin` | `opt2fin` | craters | 0 | 0 | 0 | 0 | 0 |
| F480M | `ctlfin` | `opt2nm` | leftovers | 212 | 14 | 3 | 181 | 14 |
| F480M | `ctlfin` | `opt2nm` | craters | 42 | 13 | 0 | 25 | 4 |
| F480M | `optnm` | `opt2nm` | leftovers | 198 | 1 | 0 | 183 | 14 |
| F480M | `optnm` | `opt2nm` | craters | 24 | 0 | 0 | 22 | 2 |
| F480M | `ctlfin` | `opt2fin` | leftovers | 212 | 209 | 1 | 2 | 0 |
| F480M | `ctlfin` | `opt2fin` | craters | 42 | 42 | 0 | 0 | 0 |
| F480M | `optfin` | `opt2fin` | leftovers | 3 | 1 | 0 | 2 | 0 |
| F480M | `optfin` | `opt2fin` | craters | 0 | 0 | 0 | 0 | 0 |

Box extremum at the two positions of the run 1 regressions (MJy/sr; the scan accepts a value up to 1.5 times the ctl extremum):

| band | position | ctl | run 1 unmasked | run 2 unmasked | 1.5 × ctl | run 2 within 1.5 × ctl |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| F212N | leftover (8653, 1078) | 257 | 4811 | 257 | 386 | yes |
| F480M | leftover (410, 1680) | 19 | 226 | 19 | 29 | yes |

- F212N: 0 features of `opt2nm` worse than ctl.
- F480M: 0 features of `opt2nm` worse than ctl.

**Run 2 against run 1.** Features ≥200 MJy/sr of the run 2 unmasked render (`opt2nm`) and of the run 1 unmasked render (`optnm`), each against the box extremum of the other render (box 13 × 13 px, factor 1.5):

| band | feature | features of `opt2nm` | within a factor 1.5 of `optnm` | larger in `opt2nm` | smaller in `opt2nm` | features of `optnm` | within a factor 1.5 of `opt2nm` | larger in `optnm` | smaller in `optnm` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F212N | leftovers | 77 | 67 | 1 | 9 | 77 | 67 | 1 | 9 |
| F212N | craters | 57 | 56 | 1 | 0 | 56 | 56 | 0 | 0 |
| F480M | leftovers | 197 | 183 | 0 | 14 | 198 | 183 | 1 | 14 |
| F480M | craters | 24 | 22 | 0 | 2 | 24 | 22 | 0 | 2 |

**One F212N star differs between the runs.** The scan lists the leftover (8385, 2328) and the crater (8388, 2328) of `opt2nm` as larger than in `optnm` by more than a factor 1.5 (leftover 2652 MJy/sr against a box maximum of 54; crater 540 against a box depth of 139). The leftover is of class `SATSTAR_NEAR` (nearest satstar fit 0.072 arcsec away); the vetted row nearest to the crater lies 0.022 arcsec away, is saturated and has flux 37578 (`results/run2_regress_scan_F212N.txt`). Extremes of the residual in the 13 × 13 pixel box centred on (8385, 2328), in MJy/sr (`results/run2_box_F212N_8385_2328.txt`):

| render | what it is | NaN pixels in the box | maximum | minimum |
| :--- | :--- | ---: | ---: | ---: |
| `ctlfin` | ctl, every switch off | 0 of 169 | 2430 | -2904 |
| `optnm` | run 1 opt, no mask | 0 of 169 | 54 | -139 |
| `opt2nm` | run 2 opt, no mask | 0 of 169 | 2663 | -529 |
| `optfin` | run 1 opt, real masks | 169 of 169 | all NaN | all NaN |
| `opt2fin` | run 2 opt, real masks | 169 of 169 | all NaN | all NaN |
| `ctlk4emu` | ctl, K=4 emulation (F212N values) | 169 of 169 | all NaN | all NaN |

The values of the scan (2652 and 540 MJy/sr) are peak values of the finder, which subtracts a local background (definitions above); the table lists extremes of the residual image itself.

The box maximum of `opt2nm` (2663) is within a factor 1.5 of that of ctl (2430); run 1 had 54. The box minimum is -529 in run 2, -139 in run 1 and -2904 in ctl. All 169 pixels of the box are NaN in `optfin`, `opt2fin` and `ctlk4emu`, so the masked renders do not show the star.

**The cause is not isolated.** The code differs between the runs in the heads of three PRs (git, `results/integration_trees.txt` lists the merges):

| PR | head in run 1 | head in run 2 | commits | package modules that exist on main | new package modules | test files | documentation and evidence files |
| :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: |
| #1366 | cf5a29d2 | 38873cf8 | 1 | `cataloging.py` | 0 | 1 | 1 |
| #1377 | 8a15821b | a5926b42 | 2 | `cataloging.py`, `crowdsource_catalogs_long.py` | 1 | 2 | 22 |
| #1376 | 52549326 | 14b9308c | 2 | `cataloging.py`, `crowdsource_catalogs_long.py` | 1 | 3 | 22 |

The head change of #1366 is the dedup update described in the section on the regressions; the head changes of #1377 and #1376 add the per-band form of the seed options (a new package module that parses the per-band values, changes in the modules listed, tests, and evidence files). The F212N values of the three seed options are the same in both runs (plain values in run 1, the per-band form with the same F212N values in run 2; `results/run2_options.txt`). Which of the three head changes acts on this star was not tested, and no run was repeated with identical code, so the variation of the fits between identical runs is not measured.

Rows that the run 2 opt seed adds to the ctl seed, and how many of them have a row (within the matching radius of the survival tables) in the run 2 catalogs:

| band | added rows | module raw | module vetted | merged raw | merged vetted |
| :--- | ---: | ---: | ---: | ---: | ---: |
| F212N | 97 | 97 | 97 | 97 | 97 |
| F480M | 4 | 4 | 4 | 4 | 4 |

F212N: 97 added rows (53 from the far-companion exemption, 44 from the relaxed bright window; run 1: 97, 53 and 44). F480M: 4 added rows (0 from the exemption, 4 from the relaxed window; run 1: 2410, 2406 and 4); the F480M exemption is off in run 2. The `lost` list of the survival file (merged raw rows within the radius of an added row that are absent from the vetted catalog) has 0 entries in F212N and 0 in F480M.

The survival through the cross-band catalogs (and with it the `oksep_pass` selection) and the position-matched attribution were computed for run 1 only.

## Added seed rows (run 1)

Added rows are the rows of the opt image-augmented seed table with no row of the ctl seed table within 1 mas. Group A (#1377, far-companion exemption) holds own-band m6 rows that the exemption keeps; group B (#1376, relaxed bright window) holds image-detection rows kept for bright stars. Each column counts the added rows for which the named opt catalog has a row whose own-band position lies within 0.1 arcsec of the added row and whose flux in the band is finite (flux that forced photometry fills into the m8 tables does not count): *module raw* / *module vetted* = per-module merged catalog before / after the m7 vetting; *merged raw* / *merged vetted* = the two modules merged; *xb m7* = the cross-band m7 catalog; *xb oksep* = the same table cut by the quality and `oksep_pass` selection (`oksep_pass` is true when more than one non-wide band has a separation below 0.1 arcsec; `jwst_gc_pipeline/photometry/merge_catalogs.py::compute_oksep`); *xb m8* = the m8 table (m7 plus upper-limit columns); *xb m8_dedup* = the m8 table after the dedup (an added row counts when the deduplicated table carries its m8 row). The columns *no ctl row* count the added rows for which the ctl catalog of the same level has no row within the same radius.

**F212N**

| group | added rows | module raw | module vetted | merged raw | merged vetted | xb m7 | xb oksep | xb m8 | xb m8_dedup | xb m7, no ctl row | xb oksep, no ctl row |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A: far-companion exemption | 53 | 53 | 53 | 53 | 53 | 53 | 6 | 53 | 53 | 45 | 4 |
| B: relaxed bright seeds | 44 | 44 | 44 | 44 | 44 | 44 | 43 | 44 | 43 | 39 | 38 |
| A + B | 97 | 97 | 97 | 97 | 97 | 97 | 49 | 97 | 96 | 84 | 42 |

**F480M**

| group | added rows | module raw | module vetted | merged raw | merged vetted | xb m7 | xb oksep | xb m8 | xb m8_dedup | xb m7, no ctl row | xb oksep, no ctl row |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A: far-companion exemption | 2406 | 2399 | 2383 | 2398 | 2382 | 2382 | 2345 | 2382 | 2382 | 2235 | 2203 |
| B: relaxed bright seeds | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| A + B | 2410 | 2403 | 2387 | 2402 | 2386 | 2386 | 2349 | 2386 | 2386 | 2239 | 2207 |

**The oksep selection removes most F212N rows of the exemption.** Of the 97 F212N added rows that have a cross-band m7 row within 0.1 arcsec, 49 have `oksep_pass` true and 48 do not (group A: 47 of 53; group B: 1 of 44). Counting only the rows for which ctl has no row at that position, 84 added rows are in the cross-band m7 catalog and 42 in the oksep catalog. In F480M 2349 of the 2386 added rows with a cross-band m7 row pass. `oksep_pass` is true when more than one non-wide band has a separation below 0.1 arcsec between its position and the cross-band position, so a row without a detection in the other band within that radius fails it. The selection acts on the cross-band catalogs after the model is built: the model and the residual render, and with them the residual counts above, include these rows. Whether the far-companion rows that fail it belong in the science catalog is open (decision list).

**Attribution of the cleared leftovers by position.** Every leftover ≥100 MJy/sr of ctl is assigned to the first matching set among the row-level differences of the two arms (match radius 0.094 arcsec; vetted rows within 0.036 arcsec in F212N and 0.081 arcsec in F480M): *added* = an added seed row within 0.1 arcsec; *restrow* = a vetted opt-only row flagged `satstar_dao_restored`; *newvet* = another vetted opt-only row; *lostvet* = a vetted ctl-only row; *fluxchg* = a vetted row in both arms whose flux differs by more than 10 %; *restore* = a restored raw row within 1 arcsec; *satchg* = a consolidated satstar row that differs between the arms; *none* = no row-level difference. A position match shows co-location and does not establish cause: the arms differ by all switches at once, and the *none* set also holds the effect of the render-time masks.

**F212N** (leftovers ≥100 MJy/sr of ctl by set). Row-level census: 97 added seed rows; 537 vetted rows only in opt (95 at an added row, 37 restored by the DAO-restore option, 405 other); 105 only in ctl; 288 of 583828 rows in both with a flux change above 10 %.

| set | `ctlfin:optnm` cleared | persisting | `ctlfin:optfin` cleared | persisting |
| :--- | ---: | ---: | ---: | ---: |
| added | 52 | 3 | 52 | 3 |
| restrow | 22 | 0 | 22 | 0 |
| newvet | 20 | 0 | 20 | 0 |
| lostvet | 0 | 0 | 0 | 0 |
| fluxchg | 4 | 0 | 4 | 0 |
| restore | 0 | 2 | 2 | 0 |
| satchg | 0 | 5 | 5 | 0 |
| none | 78 | 183 | 189 | 72 |
| all | 176 | 193 | 294 | 75 |

**F480M** (leftovers ≥100 MJy/sr of ctl by set). Row-level census: 2410 added seed rows; 2476 vetted rows only in opt (2246 at an added row, 1 restored by the DAO-restore option, 229 other); 106 only in ctl; 702 of 145380 rows in both with a flux change above 10 %.

| set | `ctlfin:optnm` cleared | persisting | `ctlfin:optfin` cleared | persisting |
| :--- | ---: | ---: | ---: | ---: |
| added | 4 | 0 | 4 | 0 |
| restrow | 0 | 0 | 0 | 0 |
| newvet | 5 | 0 | 5 | 0 |
| lostvet | 0 | 0 | 0 | 0 |
| fluxchg | 1 | 1 | 1 | 1 |
| restore | 0 | 1 | 1 | 0 |
| satchg | 0 | 0 | 0 | 0 |
| none | 38 | 207 | 240 | 5 |
| all | 48 | 209 | 251 | 6 |

## F480M under the F212N values of the process-global switches

The mask variables and the fit-time switches take one value per process, so the F480M render of both runs uses the values of the F212N tests (ring zone 25 px, DQ_SAT dilation 4 px, edge band 8 px). The values validated on F480M in the earlier replays are a ring zone of 20 px, DQ_SAT dilation of 3 px and an edge band of 6 px (presets `v480k*`). The table lists the real-mask render of run 1 with each set of values, and the emulations (library masks without ring tier 2 and the gate-rejected fits) of ctl and of the unmasked opt render with each set. Run 1 used plain seed-option values, which apply to F212N and F480M alike; run 2 sets the F480M seed options per band.

| render | what it is | leftovers ≥100 | ≥200 | ≥500 | craters ≥50 | ≥200 | masked % of finite px |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `optfin` | run 1 opt, real masks (F212N values) | 7 | 3 | 0 | 1 | 0 | 24.35 |
| `optv480k4full` | run 1 opt, F480M-validated values | 9 | 4 | 0 | 1 | 0 | 23.82 |
| `ctlk4emu` | ctl, K=4 emulation (F212N values) | 28 | 8 | 0 | 22 | 3 | 24.33 |
| `ctlv480k4emu` | ctl, K=4 emulation (F480M-validated values) | 30 | 9 | 0 | 22 | 3 | 23.80 |
| `optk4emu` | run 1 opt, K=4 emulation (F212N values) | 7 | 3 | 0 | 1 | 0 | 24.34 |
| `optv480k4emu` | run 1 opt, K=4 emulation (F480M-validated values) | 9 | 4 | 0 | 1 | 0 | 23.81 |
| `ctlv480k1emu` | ctl, K=1 emulation (F480M-validated values) | 51 | 15 | 1 | 43 | 7 | 5.28 |
| `optv480k1emu` | run 1 opt, K=1 emulation (F480M-validated values) | 16 | 5 | 0 | 1 | 0 | 5.28 |
| `opt2fin` | run 2 opt, real masks (F212N values) | 6 | 2 | 0 | 0 | 0 | 24.35 |
| `opt2v480k4full` | run 2 opt, F480M-validated values | 8 | 3 | 0 | 0 | 0 | 23.82 |

Between the two real-mask variants of run 1 (`optfin` → `optv480k4full`) 2 leftovers ≥100 MJy/sr appear and 0 are cleared; the masked share changes from 24.35 % to 23.82 % of the finite pixels. The F480M seed options add 2406 far-companion-exemption rows and 4 relaxed rows in run 1 (F212N: 53 and 44); run 2: 0 and 4 in F480M, 53 and 44 in F212N.

## What this does not show, and open decisions

- **ctl is not production.** ctl runs the integration code with every switch off; production → ctl already moves the F212N crater count ≥50 MJy/sr from 554 to 522 and the F480M leftover count ≥100 MJy/sr from 224 to 257. Differences between ctl and opt are the effect of the switches; differences to production include the code drift.
- **All switches at once.** The arms differ by all switches. The position-matched sets locate where rows differ; the effect of one PR alone needs a run with that switch alone. The integration trees contain two local conflict resolutions that a merge into main has to repeat.
- **Unmasked renders keep bright leftovers.** The unmasked opt render still has 46 (F212N) and 146 (F480M) leftovers ≥500 MJy/sr; with the real masks 0 and 0. The masks remove them from the image and leave the catalog unchanged.
- **Default cross-band selection (open decision).** `oksep_pass` fails for 48 of the 97 F212N added rows that have a cross-band m7 row (47 of the 53 rows of the exemption). The residual counts include these rows. Whether they belong in the science catalog is for the owner of that selection; run 1 does not decide it.
- **Mask values per band (open decision).** The render-time mask values are process-global. F480M with the F212N values masks 24.35 % of the finite pixels, against 23.82 % with the F480M-validated values; F212N masks 15.67 %. The seed options accept per-band values (run 2); the mask variables do not.
- **CRDS cache.** Run 1 probably rewrote two files of the shared CRDS cache (check above). In run 2, which sets the read-only flag, the modification time of the same two files changed from 2026-11-25 03:04:53 to 2026-11-25 17:50:37 with unchanged sizes; the writer is not identified (7 other finalize jobs of the same user were running at 17:50:37; no CRDS line of the run 2 jobs lies within 100 s of it). Whether both runs used the same CRDS context is not verified.
- **F480M crater at (3701, 498).** The 3 px matching lists it as new in the unmasked opt render with depth 70.6 MJy/sr (0.47 of the model peak); the scan finds a box depth of 60 MJy/sr in opt against 5 MJy/sr in ctl, below the 200 MJy/sr threshold. In replays of its frame (nrcblong exposure 4) an extra row of the opt arm (flux 1468.3, seed flux 319.8, qfit 0.146, 3.44 px from a star of flux 8881 whose row is identical in every replay) is present with all switches on, with each of #1360, #1362, #1365, #1366, #1367, #1369, #1371 and #1374 off singly, with the #1376 options off, with the #1377 exemption off, and with the staged inputs of the opt tree combined with the ctl options and no environment switch. It is absent with the staged inputs of the ctl tree combined with the opt options and every switch. The seed table read by the replay has 835 more sources on this frame with the opt inputs than with the ctl inputs (1228 more in the tile). Which staged input carries the seed was not isolated. In run 2 the box depth at this position is 5 MJy/sr in the unmasked render (ctl 5, run 1 60); the code (the heads of #1366, #1376 and #1377) and the F480M values of the seed options both differ between the runs, and no cause is assigned.
- **F212N star at (8385, 2328) (run 2).** The run 2 unmasked render shows a leftover of 2652 MJy/sr and a crater of 540 MJy/sr (scan values) at a star where the run 1 unmasked render has a box maximum of 54 and a box depth of 139 MJy/sr and ctl has a box maximum of 2430 and a box depth of 2904 MJy/sr. The cause is not isolated (code differences between the runs: the heads of #1366, #1377 and #1376; equal F212N option values; no run repeated with identical code). The real masks cover the box, so the masked render does not show it.
- **Not repeated with the updated #1366.** The o130 and o105 F212N tile A/Bs of #1366, and the replay of the F480M merged flux of (410, 1680) through the merge.
- **Run 2 scope.** Run 2 re-runs the opt arm only. It has no cross-band survival and so no `oksep_pass` counts, and no position-matched attribution; run 1 has both.
- **Settings of the runs.** `SATSTAR_DAO_RESTORE_FRAC_MIN` is 0.8 in the opt arm of both runs; it was chosen when the runs were set up, and other values were not run. `SATSTAR_SIDELOBE_FRAC_MIN` (#1368) is unset in both arms. This record changes no PR.

## Evidence files

All files are in `docs/evidence/e2e_o045_stack/results/`. Run 1 files come from the results directory of the run; scan files are the position-independent scans described above. Paths of the scratch area in the copied outputs are replaced by `<scratch>` and the names of scratch scripts by `<script>`.

- `results/env_opt_run1_and_run2.txt`: environment block of the opt arm (identical in both runs).
- `results/integration_trees.txt`: merge commits and heads of the two integration trees.
- `results/pr_state.txt`: base branch, title and head of the PRs on 2026-11-25 (gh pr view).
- `results/regress_mechanism_and_fix.txt`: mechanism of the two regressions, and the update of #1366.
- `results/regress_replay_frames.txt`: replay table of the three frames of the regressions (earlier and updated #1366 rule).
- `results/run1_attrib_F212N.txt`: position-matched sets of the cleared leftovers, F212N.
- `results/run1_attrib_F480M.txt`: position-matched sets of the cleared leftovers, F480M.
- `results/run1_compare_F212N.txt`: counts per render and pair tables of the 3 px matching, run 1, F212N.
- `results/run1_compare_F480M.txt`: counts per render and pair tables of the 3 px matching, run 1, F480M.
- `results/run1_crds_attribution.txt`: CRDS cache file times against the CRDS lookups of the finalize logs.
- `results/run1_mask_validation.txt`: validation lines of the library mask against the run 1 finalize render.
- `results/run1_newcheck_F212N.txt`: items listed as new by the 3 px matching, run 1, F212N.
- `results/run1_newcheck_F480M.txt`: items listed as new by the 3 px matching, run 1, F480M.
- `results/run1_orange_check.txt`: check of /orange after run 1.
- `results/run1_orange_writer_sacct.txt`: Slurm accounting of the job that wrote the o045 m8 catalogs of the production tree.
- `results/run1_sacct.txt`: Slurm accounting of the jobs of run 1.
- `results/run1_scan_optfin_F212N.txt`: position-independent scan of the masked opt render against ctl, F212N.
- `results/run1_scan_optfin_F480M.txt`: position-independent scan of the masked opt render against ctl, F480M.
- `results/run1_scan_optnm_F212N.txt`: position-independent scan of the unmasked opt render against ctl, F212N, with the probe positions.
- `results/run1_scan_optnm_F480M.txt`: position-independent scan of the unmasked opt render against ctl, F480M, with the probe positions.
- `results/run1_survive_final_F212N.txt`: survival of the added rows through the module, merged and cross-band catalogs, F212N.
- `results/run1_survive_final_F480M.txt`: survival of the added rows through the module, merged and cross-band catalogs, F480M.
- `results/run2_box_F212N_8385_2328.txt`: box extremes of six renders at (8385, 2328), F212N.
- `results/run2_compare_F212N.txt`: counts and pair tables of run 2, F212N.
- `results/run2_compare_F480M.txt`: counts and pair tables of run 2, F480M.
- `results/run2_crds_after.txt`: CRDS cache files after run 2.
- `results/run2_crds_attribution.txt`: CRDS cache file times against the CRDS lines of the run 2 logs, reference names, and the other finalize jobs of the same user that were running at the time of the change.
- `results/run2_crds_before.txt`: CRDS cache files before run 2.
- `results/run2_crds_diff.txt`: difference of the CRDS cache file lists before and after run 2.
- `results/run2_mask_validation.txt`: validation lines of the library mask against the run 2 finalize render.
- `results/run2_options.txt`: seed options of the two runs.
- `results/run2_regress_scan_F212N.txt`: the scan of the run 2 unmasked render as written by the post job (class of each leftover, nearest vetted row of each crater), F212N.
- `results/run2_sacct.txt`: Slurm accounting of the jobs of run 2.
- `results/run2_sacct_post.txt`: Slurm accounting of the post job of run 2.
- `results/run2_scan_opt2fin_F212N.txt`: position-independent scan of the masked run 2 render against ctl and run 1, F212N.
- `results/run2_scan_opt2fin_F480M.txt`: position-independent scan of the masked run 2 render against ctl and run 1, F480M.
- `results/run2_scan_opt2nm_F212N.txt`: position-independent scan of the run 2 unmasked render against ctl and run 1, F212N.
- `results/run2_scan_opt2nm_F480M.txt`: position-independent scan of the run 2 unmasked render against ctl and run 1, F480M.

