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
  (`PERFILTER_CPUS`, default 8) that fits small queue holes;
- **stage 2:** `submit_cataloging_m7.sbatch` — m7 cross-band over all filters,
  reusing on-disk m6 products via `--manual-start-phase m7`, run only after the
  array finishes OK.

Same science as the monolith **except** the standalone m7 job does not re-apply
the out-of-FOV saturated-star flux pins computed at m12 (those live in memory in
a monolithic run); this affects only off-FOV bright-star photometry in the final
cross-band catalog. For full off-FOV-satstar fidelity, run the monolithic job.

### Finer splitting (per-phase / per-frame) — not yet implemented

Splitting below the filter boundary (e.g. 48× 1-core/16 GB array tasks) requires
persisting three pieces of cross-phase state that are currently in-memory only:
the residual-i2d detection image (`resid_i2d_for_next`), the off-FOV satstar
pins (`satstar_overrides`/`satstar_drops`), and the `iter_found` provenance
(`prev_merged_for`). Only the smoothed background map is reconstructed on resume
today (`_reconstruct_smoothed_bg_path`). Implementing per-phase/per-frame jobs
means adding reconstructors for the other three plus a single-frame + standalone-
join entry point, then validating bit-identical against a monolithic run.

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

## Resources

Reduction (per-filter task): 16 cpu / 128 gb / 24 h is comfortable for a
single filter's `nrca,nrcb,merged` Image3 (resample is the memory peak).
Cataloging: 32 cpu / 128 gb / 48 h — dense GC fields are source-count
bound and use multiprocessing.
