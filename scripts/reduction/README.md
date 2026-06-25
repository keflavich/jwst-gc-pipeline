# Running the full reduction → cataloging on SLURM (refactored pipeline)

Two stages, each a SLURM **array over filters** (one array task per filter).

**QOS (required):** all submitters set `#SBATCH --account=astronomy-dept` and
`#SBATCH --qos=astronomy-dept-b`. This is mandatory — the default `adamginsburg`
QOS caps total CPUs at 10, so any task requesting >10 cpus (these request 16/32)
sits forever in `QOSGrpCpuLimit` pending. If you write a new submitter, copy
these two lines.

## 1. Reduction + alignment

```
sbatch --array=0-7 scripts/reduction/submit_reduction.sbatch          # Sgr C, all 8 filters (default)
```

Runs `PipelineRerunNIRCAM-LONG.py` → Image3 imaging + per-exposure
`fix_alignment` + post-resample `realign_to_catalog`
(see `jwst_gc_pipeline/reduction/ASTROMETRY_WCS_CORRECTION_FLOW.md`).
Defaults: proposal 4147, field 012, modules `nrca,nrcb,merged`, `-s`
(reuse existing `*_cal.fits`; does **not** re-download from MAST).

## 2. Cataloging

Runs the active per-exposure manual pipeline (`crowdsource_catalogs_long.py
--each-exposure`; phases m12→m3..m6, then m7 cross-band when >1 filter). Two
streams, pick by what the queue will give you:

### Stream 1 — fast / high-resource (per-filter array)

```
sbatch --array=0-7 scripts/reduction/submit_cataloging.sbatch         # after stage 1
```

One filter per task. Each task now passes `--parallel-workers=$SLURM_CPUS_PER_TASK`
so the frame fits actually use every requested core (previously the fat job left
all but one core idle). A single-filter task runs m12..m6 only — no cross-band
m7. To also build the cross-band catalog, either run the monolithic multifilter
job (one task, `FILTERS="all"`) or use stream 2.

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
  array finishes OK (4 cpu — m7 is I/O/table-stack bound, not cpu-parallel).

Same science as the monolith **except** the standalone m7 job does not re-apply
the out-of-FOV saturated-star flux pins computed at m12 (those live in memory in
a monolithic run); this affects only off-FOV bright-star photometry in the final
cross-band catalog. For full off-FOV-satstar fidelity, run the monolithic job.

### Stream 3 — per-frame fan-out (finest split)

```
PIPE_ROOT=/path/to/checkout NSHARDS=16 \
    scripts/reduction/submit_cataloging_perframe.sh        # Sgr C defaults
```

Splits BELOW the filter boundary: for each phase (`m12→m3→m4→m5→m6[→m7]`) it
submits a per-frame **fan-out array** (`NSHARDS` tiny `FANOUT_CPUS`-core tasks,
each fitting a frame shard) then one **finalize** barrier job, chained
`afterok`, phase after phase. The fan-out tasks are the smallest possible ask,
so they backfill into queue holes a per-filter job never sees.

`NSHARDS` is only a granularity knob — the shard predicate (`frame_index % N`)
covers every exposure for any `N` (no double-fit, no gaps); the finalize verifies
a completion marker for every frame and **hard-crashes on any miss** (no silent
exposure drop). This required persisting the cross-phase state that used to be
in-memory only: `resid_i2d_for_next` (deterministic path),
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

Same off-FOV-satstar fidelity caveat as Stream 2's standalone finalize.

## Overriding the target

```
sbatch --array=0-3 --export=ALL,PROPOSAL=2221,FIELD=001,TARGET=brick,\
       FILTERS="F405N F410M F466N F212N" scripts/reduction/submit_reduction.sbatch
```

`--array=0-N` must match the number of filters (N = count − 1).

## Refactor-compatibility notes (why the old templates fail)

- **Reduction entry point cannot use `python -m`.** The file name
  `PipelineRerunNIRCAM-LONG.py` has a hyphen → not an importable module.
  It must be launched by full path. The sbatch resolves that path from the
  pip-installed `jwst_gc_pipeline` (or from `PIPE_ROOT` if pinning a
  worktree/branch).
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

**Scheduling note:** queue delay here is dominated by *large-cpu node scarcity*,
not memory. The light/low-resource stages therefore ask few cpus so they backfill
into small queue holes instead of waiting tens of hours for a big node:
- Stream-2 per-filter (stage 1): 4 cpu (tune with `PERFILTER_CPUS`);
- m7 cross-band finalize (stage 2): 4 cpu — it is I/O + table-stack bound, not
  cpu-parallel, so extra cpus buy nothing but a longer wait.

Only the **cpu** ask is shrunk. `--mem` and `--time` are kept generous on purpose:
trimming those risks an OOM/timeout kill mid-run (losing the whole job's work),
whereas a smaller cpu ask only changes *when* the job starts. Prefer "schedule
later, finish once" over "start sooner, risk a re-run".
