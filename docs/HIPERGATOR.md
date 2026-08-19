# Running on HiPerGator

Everything here is specific to UF's HiPerGator: the account, the queue, and how
to split a run across it. [GETTING_STARTED.md](../GETTING_STARTED.md) covers the
pipeline itself, and
[`scripts/reduction/README.md`](../scripts/reduction/README.md) is the reference
for the submit scripts — what each one does, and every variable it takes. This
page is about getting through the scheduler, and which of those scripts to reach
for.

Terms used below: the cataloging stage fits each exposure in numbered passes,
`m12` then `m3`…`m6`, each one detecting and subtracting on the residual of the
last; `m7` merges the filters against each other and `m8` fits every band at
each source's merged position. `m7` and `m8` are the *cross-band* passes and
need every filter present in one job.
[`PHOTOMETRY_PIPELINE_BRIEF.md`](../PHOTOMETRY_PIPELINE_BRIEF.md) describes them.

## Account and queue

```bash
--account=astronomy-dept --qos=astronomy-dept-b
```

Every job. `config.yaml` sets both, and every script under `scripts/` bakes them
into its `#SBATCH` lines; if you write a new submitter, copy those two lines.

Being in the QOS is not the same as being on the account. Without membership,
`sbatch` answers `Invalid account or account/partition combination specified`
— ask the group's admin to add you. To see what you have:

```bash
sacctmgr show assoc user=$USER format=account,qos
```

| QOS | limit across everything you have running | wall |
|---|---|---|
| `astronomy-dept-b` | cpu 6336, mem 49500 G | 4 days |
| `adamginsburg` | **cpu 10, mem 80000 M** | no limit set |

Both carry `DenyOnLimit`, so a job that would exceed the limit is **rejected
when you submit it** — `sbatch` prints
`Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)`
and exits 1. Under the `adamginsburg` QOS that is what a 16- or 32-CPU job does,
every time. The cap is on the total across your running jobs, not per job, so
two concurrent 6-CPU jobs hit it too, as does `--mem=128gb`.

Read the current limits with:

```bash
sacctmgr show qos where name=astronomy-dept-b format=name,grptres%40,maxwall
```

## Name every job at submit time

```bash
sbatch --job-name=brick2221-o001-reduce-F182M ...
```

`<target><program>-o<obsid>-<stage>[-FILTER]`. Several reduce and catalog jobs
are normally in flight at once, and a pending job shows the name it was
submitted with — the scripts rename themselves once they *start*, which is
exactly the hours you are watching the queue. The convention is in
[CLAUDE.md](../CLAUDE.md). `run_pipeline` names jobs
`brick2221-o001-catalog`, and its merge jobs `brick-merge` and `brick-mergeall`,
because the merge covers every proposal of the target.

## What the queue is short of

Delay comes from large-CPU node scarcity. A 32-CPU ask waits for a node with 32
free cores; a 2-CPU ask backfills into the small holes that open constantly. So
the way to get a run moving is to shrink `--cpus-per-task`, even at the cost of
more jobs and more wall-clock each. Memory is not what you are queueing for, so
leave `--mem` alone: the scripts already ask 128 gb, which is generous.

That is why cataloging has several submitters that do the same science at
different granularities.

## Stage 1 — reduce

`submit_reduction.sbatch` (NIRCam) and `submit_reduction_niriss.sbatch`, plus
`submit_reduction_miri.sbatch` once #236 lands. One array task per filter;
16 cpu, 128 gb, 24 h.

The scripts default to Sgr C, so name the observation you mean — a copied
example that sets only `--job-name` reduces Sgr C under a brick job name:

```bash
export PROPOSAL=2221 FIELD=001 TARGET=brick
export FILTERS="F182M F187N F212N F405N"
sbatch --array=0-3 --export=ALL --job-name=brick2221-o001-reduce \
       scripts/reduction/submit_reduction.sbatch
```

`--array` runs 0 to N-1 for N filters.

`SKIP=1` (the config's `skip_step1and2`, `-s` on the driver) reuses the
`*_cal.fits` on disk. `SKIP=0` re-fits the ramps from `*_uncal`, downloading them
from MAST first — much slower, and needed only when the detector-level products
are wrong or missing. `SKIP=0` re-fits every exposure it is asked for, whatever
is already on disk: products newer than their `_uncal` can still be wrong (a
CRDS repin, a `jwst` version bump, a Detector1 parameter change), and forcing
past exactly that is what `SKIP=0` is for.

A NIRCam run loops the modules (`nrca,nrcb,merged`) inside one interpreter, and
each exposure gets stage 1+2 once per run: the `nrca`/`nrcb` passes take their
own module's exposures, and the `merged` pass processes only what the earlier
passes in that same run did not (#417 — it used to ramp-fit everything three
times). The driver keeps that list of already-calibrated exposures in the
running interpreter, so it covers the module passes of one run and nothing
else. Set `STAGE12_RESUME=1` to additionally skip exposures
whose `_cal`/`_ramp` on disk are newer than their `_uncal`, which resumes a
killed `SKIP=0` job without redoing what it finished. Leave it unset for a
forcing re-reduce. The resume check compares mtimes, so a `_cal.fits` truncated
by the kill that stopped the job reads as current and is kept — delete the
products of the exposure the job died on before resuming, or leave the flag
unset.

Image3 resamples a filter's exposures together, so per-filter is the finest
fan-out any script offers. The NIRCam driver does take `-m nrca` / `-m nrcb`
separately, if you want to halve a task by hand — each of those runs writes only
its own module's `_cal` files, so run both before anything reads the other
module. A later `-m merged` run produces every `_cal`, re-fitting the module you
already did: that already-calibrated list is per-process, and only
`STAGE12_RESUME=1` limits the run to the exposures still missing (measured on brick F212N, 192 exposures with
96 already done: `-m merged` with `SKIP=0` in a fresh process gives 192
Detector1 calls, 96 with `STAGE12_RESUME=1`).

## Stage 2 — catalog

`scripts/reduction/README.md` calls these **streams**, and this page uses the
same names. The monolith and the three streams fit the same frames the same way;
they differ in how the work is cut up, and in whether the cross-band passes
happen. Each takes the same five variables — `PROPOSAL`, `FIELD`, `TARGET`,
`EACH_SUFFIX`, `MODULES` — which travel together or the script exits 64.

### The monolith — one task, every filter

```bash
export PROPOSAL=2221 FIELD=001 TARGET=brick MODULES=merged
export EACH_SUFFIX=destreak_o001_crf
export FILTERS="F182M,F187N,F212N,F405N,F410M,F466N"   # commas: one task, all filters
sbatch --export=ALL --job-name=brick2221-o001-catalog \
       scripts/reduction/submit_cataloging.sbatch
```

32 cpu, 128 gb, 48 h. The only path that runs `m12`…`m8` in one go, so it is the
one that produces the cross-band catalog by itself. Longest queue wait of any of
these, and a wall-clock overrun costs the whole run.

The commas matter. The script indexes `FILTERS` as a bash array by array-task
id, so **space**-separated filters with no `--array` run only the first one — a
single-filter run, silently, with no cross-band pass. A comma-separated value
arrives as one array element and reaches the driver whole. It must be exported
rather than listed inside `--export=ALL,FILTERS=...`, which splits on commas.
`FILTERS="all"` is a one-element list, so it skips `m7` and then fails looking up
the FWHM of a filter called `all`.

### Stream 1 — per-filter array

```bash
export PROPOSAL=2221 FIELD=001 TARGET=brick MODULES=merged
export EACH_SUFFIX=destreak_o001_crf
export FILTERS="F182M F187N F212N F405N F410M F466N"   # spaces: one per task
sbatch --array=0-5 --export=ALL --job-name=brick2221-o001-catalog \
       scripts/reduction/submit_cataloging.sbatch
```

One filter per task, 32 cpu each. `PARALLEL_WORKERS` defaults to the
allocation and forks one worker per frame, so cores past the frame count sit
idle — pass `--cpus-per-task=8` for a field with few exposures per filter.

Each task runs `m12`…`m6`. **No task sees more than one filter, so
no task runs `m7` or `m8`** — the cross-band catalog needs
`submit_cataloging_m7.sbatch` afterwards, or stream 2, which submits it for you.

This is what `run_pipeline` submits (`config.yaml`, `stages.catalog.fan_out:
filter`), so the one-command path also needs that follow-up.

NIRISS has its own cataloging submitter,
`scripts/reduction/submit_cataloging_niriss.sbatch`, taking the same variables.

### Stream 2 — per-filter chain, then the cross-band job

```bash
scripts/reduction/submit_cataloging_chain.sh
```

Per-filter tasks at `PERFILTER_CPUS` (default 4), then one `m7` job (4 cpu,
64 gb) with `afterok` on the array — it starts only if every task succeeded.
Trades the monolith's single fat job for N small ones that backfill, and still
ends with the cross-band catalog. The best default when the queue is busy.

One trap: the script builds its `--export` list inline, including
`MODULES=$MODULES`. A multi-module value (`nrca,nrcb`) truncates at the comma to
`nrca`, and half the field is cataloged while the run reports success. Export
`MODULES` in your shell for a multi-module field, or use `merged`.

### Stream 3 — per-frame fan-out, one phase at a time

```bash
scripts/reduction/submit_cataloging_perframe.sh
```

For each phase `m12`→`m3`→`m4`→`m5`→`m6`[→`m7`]: an array of `NSHARDS` (default
16) tasks at `FANOUT_CPUS` (default 2, 32 gb), each fitting a slice of the frames
and writing a completion marker, then one **finalize** job (`FINALIZE_CPUS`,
default 4) that checks every marker and runs the phase's barrier — the step that
needs all frames at once. The next phase's array waits on that finalize.

The finest split, and the one that gets through a jammed queue: 16 × 2 cpu asks
for the same 32 cores as the monolith but takes them a hole at a time.
`NSHARDS` only sets granularity — the shard predicate (`index % N == I`) covers
every frame for any `N`, and a missing marker crashes the finalize, so a lost
shard shows up as a failure rather than as a quietly smaller catalog.

Its costs are operational: six dependency hops, each waiting on its slowest
shard; a 180-second settle in every finalize (Lustre marker-metadata coherence);
and a shard that exhausts its retries **strands** its finalize, which then needs
`scontrol update jobid=<finalize> dependency=...` by hand after you rerun the
shard. An all-MIRI run has no `m7`, so set `PHASES` explicitly.

### The m8 fill

`m8` runs inline at the end of `m7` by default. On a field with many frames it
can overrun the wall — on sickle, F187N and F210M consumed an 18 h budget and
F470N/F480M never ran. To split it:

```bash
scripts/reduction/submit_cataloging_m8.sh
```

One per-band partial job (6 cpu, 128 gb) each writing
`..._m8_partial_<FILT>.fits`, then a column-merge job (1 cpu, 32 gb, 1 h) with
`afterok` on all of them. Pass `EXTRA_ARGS=--no-forced-fill-m8` to the `m7` job
so the inline fill is skipped and this path owns it.

## Stage 3 — merge

Two submissions, in order. `config.yaml` calls this `fan_out: program-filter`,
and `run_pipeline` does both:

1. an array, one task per `(program, filter)`;
2. one job, after the array, that reads every filter's part-1 output.

4 cpu, 64 gb, 8 h. Submitting only the array leaves the all-filter catalogs
unwritten. `--merge-workers` (default 4) above `--cpus-per-task` oversubscribes
the allocation rather than failing, so it runs slower rather than crashing. To
see the index → `(program, filter)` map and the array bounds — this runs nothing
and prints:

```bash
python -m jwst_gc_pipeline.photometry.merge_catalogs \
       --target=brick --merge-singlefields --list-jobs
```

## Choosing

- **Queue quiet, and you want the cross-band catalog in one submission** → the
  monolith.
- **Filters in parallel, cross-band later** → stream 1, plus
  `submit_cataloging_m7.sbatch`. This is `run_pipeline`'s default.
- **Queue busy, or jobs pending for hours** → stream 2. Same result as the
  monolith, in small asks.
- **Queue jammed, or many frames per filter** → stream 3, plus the split `m8`.
- **A stage overran its wall** → split it one level finer. `astronomy-dept-b`
  allows up to 4 days, so more time is available, but a longer job competes for
  the same scarce large nodes.

Changing the default for every run means editing `config.yaml`
(`stages.<stage>.cpus`, `.fan_out`); changing it for one run means calling the
submit script directly.

## Traps

**`--export` splits on commas.** `--export=ALL,MODULES=nrca,nrcb` gives the job
`MODULES=nrca`, and a cutout's `EXTRA_ARGS=--cutout-region=RA,DEC,SIZE` corrupts
the same way. Export the variable in your shell and pass a bare `--export=ALL`.
Spaces are fine; only commas break. `run_pipeline` exports everything and passes
`--export=ALL` for this reason.

**The five cataloging variables travel together.** `PROPOSAL`, `FIELD`,
`TARGET`, `EACH_SUFFIX`, `MODULES` — set one and you must set all five, or
`submit_cataloging.sbatch` exits 64.

**`EACH_SUFFIX` names the stage-1 products to photometer.** A field that
destreaks writes `*_destreak_o<obs>_crf.fits`, and one that does not writes
`*_align_o<obs>_crf.fits`; sickle differs between its short- and long-wavelength
filters. `reduction/destreak_policy.py` decides, and both stages read it.

**Exit status through a pipe.** `python ... | tee log` reports `tee`'s status, so
a failed run looks clean. `submit_cataloging_perframe_phase.sbatch` pipes to
`tee` and reads `${PIPESTATUS[0]}`; the others run python directly and read `$?`.
A job that shows `COMPLETED` while its work failed is usually a status read
through a pipe.

**`TMPDIR`.** The reduce, catalog, merge, m7 and m8 scripts point it at
node-local `$SLURM_TMPDIR`, falling back to a scratch path on `/blue`.
`submit_cataloging_perframe_phase.sbatch` sets it for its own run log only, so a
stream-3 run inherits whatever the submitting shell had — set it there. If that
is `/tmp`, the run fills the node's root filesystem and wedges it for everyone
else on that node.

## Watching a run

```bash
squeue -u $USER -o '%.10i %.30j %.8T %.10M %.6D %R'
sacct  -j <jobid> --format=JobID,JobName%30,State,Elapsed,MaxRSS,ExitCode
scontrol show job <jobid> | grep -i reason
```

What the pending reasons mean here:

| | |
|---|---|
| `Dependency` | waiting on an earlier stage, as designed |
| `JobArrayTaskLimit` | your own array is at its concurrent-task cap |
| `Resources` | the nodes that fit this ask are all busy — a smaller ask backfills sooner |
| `QOSGrpCpuLimit` | the department's aggregate is saturated. Under `astronomy-dept-b` this is other people's jobs, not the 10-CPU cap, which rejects at submit rather than queueing |

Logs land in `/orange/adamginsburg/jwst/logs`, hard-coded in every
`#SBATCH --output` line — that is the line to change on another machine, since
SLURM reads it before any shell runs and fails the job at start if the path is
not writable. Array scripts name them `<stage>_<jobname>_<jobid>_<task>.out`;
the m7 and m8 scripts drop the task suffix and keep the rest. A non-array job
submitted to an array script writes `4294967294` where the task number goes —
that is SLURM's "no value".
