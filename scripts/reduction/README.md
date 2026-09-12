# Running the full reduction → cataloging on SLURM (refactored pipeline)

Two stages, each a SLURM **array over filters** (one array task per filter).

**QOS (required):** all submitters set `#SBATCH --account=astronomy-dept` and
`#SBATCH --qos=astronomy-dept-b`. This is mandatory — the default `adamginsburg`
QOS caps total CPUs at 10, so any task requesting >10 cpus (these request 16/32)
sits forever in `QOSGrpCpuLimit` pending. If you write a new submitter, copy
these two lines.

## 1. Reduction + alignment

### First: check the field spec has inputs

```
python scripts/reduction/preflight_reduce_inputs.py \
    --target sgra --proposal 1939 --obsid 001 --filters "F115W F212N F405N"
```

Reads only; exits nonzero if a filter has no usable input. It cross-checks the
(target, proposal, observation) against `fields.yaml` — which needs no
filesystem access — then confirms on disk that the `image3` association the
reduce would use exists, that `_cal` frames (the stage-2 calibrated single
exposures) exist, and that the association's members cover the requested
modules.

Worth the ten seconds because the alternative is ~20 h of queue: a wrong
proposal fails every task in the array, and the re-tie loop then declines to
catalog the rest. It found `run_sgra_4147_o001.sh`, which had driven Sgr A* as
Sgr C's proposal (4147) for the whole campaign; every check already in the loop
passed on that spec.

`--modules` must match the `MODULES` the runner will pass — a module the
observation genuinely lacks is a real failure, and one the runner was never
going to ask for is not. Where the reduce itself declares an observation
module-restricted (sickle 3958/007 is module B only), the check reads that
declaration and does not report the other module as missing.

Add `--instrument miri` / `--instrument niriss` (and for NIRISS,
`--target <target>/niriss`, which is where its data lives) for those.

It also reports free space on the filesystem the field tree lives on, and
refuses below `--min-free-tb` (default 2 TB, `0` disables) — #421. Nothing else
in this repo looks at free space, and data-qa's `--min-free-tb` floor watches
its download staging, which is a different filesystem whenever a field's
`root:` differs from it: the treasury is `root: blue` while the staging is on
/orange, so the monitor can keep downloading into a healthy /orange and keep
triggering reductions into a full /blue. The check reads `{root}/{target}`
rather than `--root`, because `/orange/adamginsburg/jwst/brick` is a symlink to
`/blue/.../jwst/brick` — several targets under an /orange root write to /blue,
and it is the destination filesystem that runs out. Sizing: one filter of one
field is 1.42 TB all-in on `brick/F212N`, of which 1.23 GB per detector-exposure
is the per-frame product chain.

### A NEW field: seed its PSF grid cache first

```
python scripts/reduction/seed_psf_cache.py --field gc-treasury \
    --filters F212N F480M F770W --apply
```

Cataloging always passes `psf_cache_dir={field_basepath}/psfs`, so a new field
starts with an empty cache and the first run per filter re-does a MAST login
plus a Poppy build the code prices at ~17-20 min and ~300 GB peak per detector
(~7-8 h for the merged/all-detectors path). The grids are set by the physics —
instrument, detector, filter, oversampling — so other fields already have them;
this links them in. Dry run unless `--apply`. Reads only the donors, writes only
into the destination `psfs/`, and never overwrites (#420).

```
sbatch --array=0-7 scripts/reduction/submit_reduction.sbatch          # Sgr C, all 8 filters (default)
```

Runs `PipelineRerunNIRCAM-LONG.py` → Image3 imaging + per-exposure
`fix_alignment` + post-resample `realign_to_catalog`
(see `jwst_gc_pipeline/reduction/ASTROMETRY_WCS_CORRECTION_FLOW.md`).
Defaults: proposal 4147, field 012, modules `nrca,nrcb,merged`, `-s`
(reuse the `*_cal.fits` already on disk; without it,
Detector1/Image2 re-run from `_uncal`, fetching from MAST if needed).

## 2. Cataloging

Runs the active per-exposure manual pipeline (`crowdsource_catalogs_long.py
--each-exposure`; phases m12→m3..m6, then m7 cross-band when >1 filter, then m8).
**m8** is the forced cross-band fill: right after the m7 cross-band merge it
force-fits every band at the merged position of sources that are non-saturated
non-detections there, writing a sibling `..._resbgsub_m8` table (full-frame
only; on by default). The combined m8 is then de-duplicated into
`..._resbgsub_m8_dedup.fits` (the science-final; crowded-field merges split a
star into two reference rows, dedup collapses complementary-coverage pairs but
keeps resolved binaries). m8 runs automatically in whichever job completes the
final phase's finalize — the monolith, the Stream-2 m7 job, or the Stream-3 m7
finalize.

**m8 fan-out (when the inline fill overruns the wall).** The monolithic m8
sweeps every frame serially; on dense fields (sickle: 192 frames in F187N+F210M
alone) it can time out before the long-wave bands run. To split it, run the m7
job with `--no-forced-fill-m8` (skip the inline fill), then:

```
TARGET=sickle PROPOSAL=3958 FIELD=007 MODULES=nrcb \
FILTERS="F187N F210M F335M F470N F480M" \
SUFFIXES="destreak_o007_crf destreak_o007_crf align_o007_crf align_o007_crf align_o007_crf" \
scripts/reduction/submit_cataloging_m8.sh
```

This submits one `submit_cataloging_m8_partial.sbatch` per band (each fills only
its band → `..._resbgsub_m8_partial_<FILT>.fits` via `--manual-m8-partial`) plus
a merge (`submit_cataloging_m8_merge.sbatch` → `m8_merge_partials.py`), chained
`afterok`. The merge **requires every band's partial and fails loudly if one is
missing**, then dedups. `SUFFIXES` must align 1:1 with `FILTERS` (SW vs LW bands
usually differ). Two streams below for the main
m12..m7 work, pick by what the queue will give you:

### Stream 1 — fast / high-resource (per-filter array)

```
sbatch --array=0-7 scripts/reduction/submit_cataloging.sbatch         # after stage 1
```

One filter per task. Each task passes `--parallel-workers=$SLURM_CPUS_PER_TASK`
so the frame fits use every requested core. A single-filter task runs m12..m6.
To also build the cross-band catalog, either run the monolithic multifilter
job (one task, with **every filter listed explicitly**, e.g.
`FILTERS="F182M,F187N,F212N,F405N,F410M,F466N"`) or use stream 2. `FILTERS` must
name each filter: `multifilter` is `len(filternames) > 1`, so `FILTERS="all"`
yields a one-element list, skips m7, and then fails the FWHM lookup on a filter
named `all`.

### Stream 2 — low-resource / dependency-chained (optional)

```
scripts/reduction/submit_cataloging_chain.sh            # Sgr C defaults
PERFILTER_CPUS=4 scripts/reduction/submit_cataloging_chain.sh
```

Trades the one fat 32-core/48 h job for **N small per-filter jobs + one m7
finalize**, chained with `--dependency=afterok`:

- **stage 1:** per-filter array (single filter per task → m12..m6), small slice
  (`PERFILTER_CPUS`, default 4) that fits small queue holes;
- **stage 2:** `submit_cataloging_m7.sbatch` — m7 cross-band over all filters,
  reusing on-disk m6 products via `--manual-start-phase m7`, run only after the
  array finishes OK (4 cpu — m7 is I/O/table-stack bound).

Same science as the monolith. m12 persists its reconciled overrides/drops to
disk (`*_satstar_reconciled_m12.fits`) and every `--manual-start-phase` run
reconstructs them, so the standalone m7 finalize re-applies the same out-of-FOV
satstar flux pins as a monolithic run. Off-FOV satstars additionally get the
Spitzer-prior `_spitzer.reg` + model<=data clamp, loaded fresh each phase in both
paths. See `cataloging.py` `run_manual_pipeline` start_phase note.
Pass `DEBLEND_SATSTARS=1` to plumb the ZEROFRAME satstar deblend through **both**
stages (the m7 finalize honors it too) so a re-made satstar catalog at m7 matches
m12 — required for crowded GC fields (gc2211/arches/quintuplet/sgra).

### Stream 3 — per-frame fan-out (finest split)

```
PIPE_ROOT=/path/to/checkout NSHARDS=16 \
    scripts/reduction/submit_cataloging_perframe.sh        # Sgr C defaults
```

Splits BELOW the filter boundary: for each phase (`m12→m3→m4→m5→m6[→m7]`) it
submits a per-frame **fan-out array** (`NSHARDS` tiny `FANOUT_CPUS`-core tasks,
each fitting a frame shard) then one **finalize** barrier job, chained
`afterok`, phase after phase. The fan-out tasks are the smallest possible ask,
so they backfill into queue holes too small for a per-filter job. (m8 is the one phase with no fan-out array: it runs inside
the m7 finalize barrier, same as the monolith.)

`NSHARDS` is only a granularity knob — the shard predicate (`frame_index % N`)
covers every exposure exactly once for any `N`; the finalize verifies a
completion marker for every frame and **hard-crashes on any miss**. This required
persisting the cross-phase state that used to be in-memory only:
`resid_i2d_for_next` (deterministic path),
`satstar_overrides`/`satstar_drops` (new `_satstar_reconciled_m12.fits`),
`prev_merged_for` (from the prior merged catalog's `iter_found` column); the
smoothed bg already reconstructed (`_reconstruct_smoothed_bg_path`). All gated by
new opts (`--manual-stop-after-phase`, `--manual-frame-shard`,
`--manual-skip-finalize`, `--manual-finalize-only`) that default off, so a
monolithic run is byte-for-byte unchanged.

Verify equivalence on a small cutout before trusting it at scale:
`scripts/reduction/validate_perframe_equivalence.sh` runs mono vs per-frame
locally and diffs the final catalogs. Unit tests:
`jwst_gc_pipeline/photometry/tests/test_perframe_helpers.py`.

Off-FOV satstars: same handling as Stream 2 (persisted m12 overrides
reconstructed from disk, plus the Spitzer-prior path);
pass `DEBLEND_SATSTARS=1` to carry the ZEROFRAME deblend through every
per-frame stage.

#### Per-filter finalize (`PER_FILTER_FINALIZE=1`, opt-in)

The fan-out is split per FRAME; the finalize is not split at all. It walks every
(filter x module) combo serially, so on a big field the barrier is the sum of
its filters. Measured on sgrb2 5365/001's m3 finalize (job 41424864,
2026-09-09/11): **43.2 h over 33 combos**, of which F187N alone was 13.1 h and
the five LW filters together were under 9 h. The time inside it is 65% merge +
astrometry checkpoint + vetting and 35% the per-frame render loop.

```
PER_FILTER_FINALIZE=1 scripts/reduction/submit_cataloging_perframe.sh
```

submits **one finalize per filter** for `m3 m4 m5 m6` instead of one per phase,
all `afterok` on the same fan-out array, named
`<target><program>-o<obsid>-<phase>-finalize-<FILTER>`. The phase barrier is
unchanged in meaning: the next phase's fan-out waits on `afterok` of **every**
one of them. `PER_FILTER_FINALIZE_PHASES` narrows the set (e.g. `"m4"`).

##### Each split job is sized for its own filter

A split finalize does one filter's work and asks for one filter's slice. Both
`--mem` and `--time` scale by the filter's own crf count over the largest
filter's, because that is what each was measured to follow:

* **memory does not pool.** The loop is `for module: for filt:` and what it
  keeps between filters is paths plus a SkyCoord per combo, so the peak is the
  biggest COMBO. sgrb2's whole 11-filter m3 finalize recorded MaxRSS 148 GiB;
  its single-filter F187N m12 finalize (39933196) recorded 151 GiB. F187N *is*
  the peak, so the largest filter keeps the field's memory and the other ten
  stop asking for it.
* **time is near-linear in frames.** F187N has 384 crf to F182M's 192, and took
  13.09 h to its 6.79 h inside that m3 finalize (1.93x on 2.00x the frames).

On sgrb2 5365/001's real counts (F187N 384, four SW 192, six LW 48) an m4
phase therefore asks:

| filters | crf | `--mem` | `--time` | was |
|---|---:|---|---|---|
| F187N | 384 | 256gb | 2-00:00:00 | unchanged |
| F150W F182M F210M F212N | 192 | 128gb | 1-00:00:00 | 256gb / 2-00:00:00 |
| F300M F360M F405N F410M F466N F480M | 48 | 64gb | 12:00:00 | 256gb / 2-00:00:00 |

1152 GiB for the phase instead of 11 x 256 = 2816 GiB. The floor is the phase
sbatch's own hand-launch slice (`--mem=64gb --time=12:00:00`); the whole-field
value is the ceiling. **An explicit `FINALIZE_MEM` / `FINALIZE_TIME` /
`FINALIZE_TIME_<PHASE>` is never scaled** -- it reaches every split job
verbatim. With no crf counts on disk (CI, a fresh checkout) nothing is scaled
and every split job asks the whole-field slice, which is what it asks today.

##### A phase is submitted all-or-nothing

The driver runs under `set -euo pipefail`, so a failed `sbatch` aborts it where
it stands. Whole, that leaves at worst a fan-out with no finalize. Split it
would leave a state that did not exist before -- phase p with some of its
filters finalizing, the rest never submitted, nothing queued behind them, and
every job that *was* submitted reaching COMPLETED on a field missing seven
barriers, which nothing downstream can see.

So the per-filter finalizes of a phase are submitted `--hold` and released in
one `scontrol release` once all of them exist. An abort before that release
cancels the held finalizes **and** the phase's own fan-out (neither has run)
and prints the command that resumes the chain from the last complete phase,
including the barrier dependency:

```
SUBMIT ABORTED (rc=1).  Jobs submitted by this run: 1001 1002 1003 1004 1005 1006 1007
  Phase m4 was left PART-SUBMITTED: ...
    scancel 1006 1007 1005 -- done
  Everything before phase m4 is queued and intact.
  Resume the rest of the chain with:
    DEP='afterok:1004' PHASES='m4 m5 m6 m7' \
      <the same environment as this run> .../submit_cataloging_perframe.sh
```

Being killed outright (SIGKILL) skips that trap and leaves the phase's
finalizes HELD. Held jobs do not run, so that is a stalled chain the queue
shows as `JobHeldUser` -- `scontrol release` recovers it -- rather than a
half-finalized phase.

##### m7 and m12 are never split

Asking for either is refused at submit time (exit 4), not quietly ignored:

* `m7` **is** the cross-band merge (`cataloging._do_crossband`) and the
  cross-filter astrometry anchor gate, both of which read every filter's m6
  vetted catalog. A one-filter run does not even build an m7 phase -- `phases`
  appends m7 only when `len(filternames) > 1` -- so a split m7 dies on
  `--manual-start-phase='m7' not in phases ['m12','m3','m4','m5','m6']`.
* `m12` runs the **correcting** astrometry checkpoint, and a correcting
  checkpoint's verdict on a real misalignment is to STOP THE FIELD: correct the
  offsets table, rename the first-pass mosaics to `*_i2d_im0_badastrom.fits`,
  and raise `AstrometryCorrectionRequiredError` saying the current crf frames
  and catalogs are stale. **One process is what that raise ends.** Whole, the
  m12 finalize stops the field the instant any filter measures a misalignment;
  split, the stop reaches only the filter that raised it.

  sgrb2's eleven per-filter m12 finalizes of 2026-08-22 (39933194-39933204) are
  the measurement. Seven corrected `Offsets_JWST_Brick5365_VIRAC2locked.csv`
  and stale-tagged im0 between 12:04:08 and 12:36:41 UTC; the other four ran on
  and COMPLETED -- F210M 7:27, F150W 9:35, F182M 9:46 and F187N 23:11, i.e.
  ~15 h past the first stop signal -- writing m12 products into a field the
  checkpoint had already quarantined.

  The offsets CSV itself is **not** the reason. `update_offsets_table` has
  wrapped its whole read-modify-write in `with locked(offsets_path)` since
  9f73c05 (2026-08-02), three weeks before that run, so those seven writes were
  serialised. The one shared write with no lock on it is the ledger those
  renames append to, `mark_i2d_stale`'s `stale_i2d_renames.json` -- the file a
  bad run is undone from. It survived that seven-way append (462 lines, all
  parseable), which is luck rather than a guarantee.

m3-m6 have neither property: every path the per-phase body touches is keyed
`(module, filter)`, the strict marker verify and the module-coverage check are
both scoped to the run's own filters, and the astrometry checkpoint at a FROZEN
stage records to a per-filter file and corrects nothing. Pinned by
`jwst_gc_pipeline/tests/test_per_filter_finalize_barrier.py`.

A filter the observation never took is dropped by the preflight
(`requested_filters`, case 4) and no longer fails the job that holds it:
`run_manual_pipeline` returns cleanly when every dropped filter carries that
verdict, so a per-filter finalize for a band the field does not have satisfies
its barrier instead of stranding the other ten. A **waived**
`declared-but-absent` band still refuses -- that waiver exists so the run's
other bands produce, and there are none left.

##### Using it on a field whose chain is already queued

The split phases must replace the queued ones, so cancel those first and
resubmit while the *running* phase is still running, or the new chain has
nothing to hang off. State the limits, because the driver's table is not what
a hand-set `FINALIZE_TIME` gave the queued jobs:

```
scancel <the pending m4..m7 fan-out and finalize ids>
squeue -u $USER -o "%.10i %.34j %.2t %R" | grep <target>   # expect the running one only

PIPE_ROOT=$PWD PER_FILTER_FINALIZE=1 \
  DEP=<the running finalize's id> PHASES="m4 m5 m6 m7" \
  FINALIZE_TIME_M7=<the limit the queued m7 finalize carried> \
  TARGET=... PROPOSAL=... FIELD=... MODULES=... EACH_SUFFIX=... FILTERS="..." \
      scripts/reduction/submit_cataloging_perframe.sh
```

`FINALIZE_TIME_M7` is the one that has to be carried over by hand: **m7 is not
split**, so its finalize does exactly the work the job it replaces was given,
and the field-tier table (`1-12:00:00` for a large field) is shorter than the
`4-00:00:00` a runner's blanket `FINALIZE_TIME` had put on it. A TIMEOUT there
takes the rest of the `afterok` chain with it. The split phases' finalizes do
ask less than the pooled jobs they replace -- that is the point -- and the
driver prints each one's `--mem`/`--time` beside the whole-field value it
replaced, so the reduction is in the submit log rather than implied.

## Overriding the target

```
sbatch --array=0-3 --export=ALL,PROPOSAL=2221,FIELD=001,TARGET=brick,\
       FILTERS="F405N F410M F466N F212N" scripts/reduction/submit_reduction.sbatch
```

`--array=0-N` must match the number of filters (N = count − 1).

## Refactor-compatibility notes (why the old templates fail)

- **Launch the reduction entry point by full path.** The hyphen in
  `PipelineRerunNIRCAM-LONG.py` makes it an illegal module name, so `python -m`
  fails on it. The sbatch resolves that path from the pip-installed
  `jwst_gc_pipeline` (or from `PIPE_ROOT` if pinning a worktree/branch).
- **Cataloging entry point uses `python -m`** (it is a proper module).
- The pre-refactor `_bench/cloudef_o002_rereduce.sbatch` invokes
  `brick-jwst-2221/brick2221/reduction/...` — the OLD package
  (imports `brick2221.*`). **Stale; do not use.** These scripts replace it.
- Pin a non-installed checkout (e.g. a worktree) with
  `--export=ALL,PIPE_ROOT=/path/to/checkout`.
- **`DEBLEND_SATSTARS=1`** (cataloging) passes `--deblend-satstars`: ZEROFRAME-deblend
  merged saturated cores for crowded GC fields (gc2211). Loads the matching
  `_ramp.fits` ZEROFRAME and splits each merged saturated component into one seed per
  star; auto-degrades to legacy where a frame lacks a sibling `_ramp.fits`.

## Resources

Reduction (per-filter task): 16 cpu / 128 gb / 24 h is comfortable for a
single filter's `nrca,nrcb,merged` Image3 (resample is the memory peak).
Cataloging (fast/Stream-1, fat per-filter): 32 cpu / 128 gb / 48 h — dense GC
fields are source-count bound and use multiprocessing.

**Scheduling note:** queue delay here is dominated by *large-cpu node scarcity*.
The light/low-resource stages therefore ask few cpus so they backfill into small
queue holes instead of waiting tens of hours for a big node:
- Stream-2 per-filter (stage 1): 4 cpu (tune with `PERFILTER_CPUS`);
- m7 cross-band finalize (stage 2): 4 cpu — it is I/O + table-stack bound, so
  extra cpus buy only a longer wait.

Only the **cpu** ask is shrunk. `--mem` and `--time` are kept generous on purpose:
trimming those risks an OOM/timeout kill mid-run (losing the whole job's work),
whereas a smaller cpu ask only changes *when* the job starts. Prefer "schedule
later, finish once" over "start sooner, risk a re-run".

Generous, and specific to what the job does: `submit_cataloging_perframe.sh`
counts the field's `_crf` frames once and sizes both `--mem` (#611) and `--time`
(#737) from it, wall time additionally per stage (`_stage_time`). A flat 12 h
was below what the big fields run — sgrb2's m12-finalize has taken 55.3 h — and
a stage killed on its limit takes its whole `afterok` chain with it.

Wall time is not a free parameter in the other direction either: a 4-cpu job
asking a 3-day wall only backfills into a 3-day-wide gap, and sgrb2's
m6-finalize waited exactly its own 3 d on an otherwise finished field. So the
small tier (arches, m92, ngc6397, m4 — nothing measured past 6.5 h over 452
runs, and no small field has ever hit a time limit)
keeps the 12 h it had; only the mid and large tiers get more.

`FINALIZE_MEM`, `FANOUT_TIME` and `FINALIZE_TIME` in the environment still
override. `FANOUT_TIME`/`FINALIZE_TIME` are one knob for SIX phases each, which
is how sgrb2's m3–m7 finalizes (12.5–17.3 h) came to carry the 72 h its runner
sized for m12; prefer the per-phase spelling `FANOUT_TIME_<PHASE>` /
`FINALIZE_TIME_<PHASE>` (e.g. `FINALIZE_TIME_M12=72:00:00`). Every submit line
prints the limit it asked for and where it came from, so a stale blanket
override is visible in the log.
