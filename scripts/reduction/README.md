# Running the full reduction → cataloging on SLURM (refactored pipeline)

Two stages, each a SLURM **array over filters** (one array task per filter).

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

```
sbatch --array=0-7 scripts/reduction/submit_cataloging.sbatch         # after stage 1
```

Runs the active per-exposure manual pipeline
(`crowdsource_catalogs_long.py --each-exposure`).

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
