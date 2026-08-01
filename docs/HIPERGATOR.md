# Running on HiPerGator

Everything here is specific to UF's HiPerGator: the account, the queue, and how
to split a run across it. The pipeline itself is described in
[GETTING_STARTED.md](../GETTING_STARTED.md); this page is about how to get it
through the scheduler quickly.

## Account and queue

```bash
--account=astronomy-dept --qos=astronomy-dept-b
```

Every job. `config.yaml` sets both, so anything submitted by
`run_pipeline` already has them; pass them by hand when you call `sbatch`
directly.

The `adamginsburg` QOS caps a job at **10 CPUs**. A 16- or 32-CPU job submitted
under it is accepted and then sits `QOSGrpCpuLimit` forever — it is not slow, it
will never start. `astronomy-dept-b` is the burst QOS and is what the survey
runs on.

## Name every job at submit time

```bash
sbatch --job-name=brick2221-o001-reduce-F182M ...
```

`<target><program>-o<obsid>-<stage>[-FILTER]`. Several reduce and catalog jobs
are normally in flight at once, and a pending job shows only the name it was
submitted with — the submit scripts also rename themselves, but that fires when
the job *starts*, which is exactly the hours you are watching the queue. The
convention is in [CLAUDE.md](../CLAUDE.md); `run_pipeline` follows it.

## What the queue is actually short of

Delay is **large-CPU node scarcity**, not memory. A 32-CPU ask waits for a node
with 32 free cores; a 2-CPU ask backfills into holes that open constantly. So
the way to get a run moving is to shrink `--cpus-per-task`, even when that means
more jobs and more wall-clock per job. Memory can stay generous — asking for
128 GB costs little scheduling-wise and protects against an OOM kill.

This is why the pipeline has several ways to split the same work.

## The pathways

`python -m jwst_gc_pipeline.run_pipeline --proposal <p> --obsid <o>` submits the
default of each stage, chained with `afterok`. The rows below are what that
default is, and what to reach for instead.

### Stage 1 — reduce

| | fan-out | per task | when |
|---|---|---|---|
| `submit_reduction.sbatch` (default) | one array task per filter | 16 cpu, 128 gb, 24 h | the only path; Image3 is per-filter by construction |

Set `SKIP=1` (the config's `skip_step1and2: true`, `-s` on the driver) to reuse
the `*_cal.fits` already on disk. `SKIP=0` re-fits the ramps from `*_uncal`,
downloading them from MAST first — much slower, and needed only when the
detector-level products are wrong or absent.

Splitting finer than per-filter is not available: Image3 resamples a filter's
exposures together.

### Stage 2 — catalog

Four pathways, coarsest to finest. All produce the same catalogs.

**A. One job, all filters** — `submit_cataloging.sbatch` with `FILTERS="..."` on
a single task.

- 32 cpu, 128 gb, 48 h.
- Fastest once it starts, and the cross-band m7 phase happens inline.
- Worst queue wait of the four, and one wall-clock overrun loses the whole run.
- Reach for it when the queue is quiet and the field is small.

**B. Per-filter array** — `submit_cataloging.sbatch --array=0-N`. This is the
default (`config.yaml`, `stages.catalog.fan_out: filter`).

- 32 cpu per task by default; drop it for a field that does not need it.
- Each task runs m12→m6 for one filter. A single-filter task has no cross-band
  phase, so **m7 must follow separately**.
- Good general choice: filters proceed in parallel, and one failed filter is one
  rerun.

**C. Per-filter chain + m7 finalize** — `submit_cataloging_chain.sh`.

- Per-filter tasks at `PERFILTER_CPUS` (default **4**), then one m7 job (4 cpu,
  64 gb) with `afterok` on the array.
- Trades A's single fat job for N small ones, each of which backfills.
- Best default when the queue is busy. Slower per filter than A; usually much
  faster to *finish*, because it starts.

**D. Per-frame fan-out, per phase** — `submit_cataloging_perframe.sh`.

- For each phase m12→m3→m4→m5→m6[→m7]: an array of `NSHARDS` (default **16**)
  tasks at `FANOUT_CPUS` (default **2**, 32 gb) that fit a shard of the frames
  and write completion markers, then **one** finalize job (`FINALIZE_CPUS`,
  default 4) that verifies every marker and runs the phase barrier. Phase *p+1*
  waits on phase *p*'s finalize.
- The finest split, and the one that gets through a jammed queue. `NSHARDS` only
  sets granularity — the shard predicate (`index % N == I`) covers every frame
  for any `N`.
- Costs a barrier per phase: five or six dependency hops, each waiting for its
  slowest shard. On an empty queue this is the slowest of the four.
- A missing shard makes the finalize crash on the absent marker rather than
  quietly cataloging fewer exposures.

**The m8 fill** — the forced cross-band fill runs inline at the end of m7 by
default. On a field with many frames it can overrun the wall (on sickle, F187N +
F210M consumed an 18 h budget and F470N/F480M never ran). To split it:
`submit_cataloging_m8.sh` submits one `..._m8_partial_<FILT>.fits` job per band
(6 cpu, 128 gb) and one column-merge job (1 cpu, 32 gb, 1 h) with `afterok` on
all of them. Pair it with `--no-forced-fill-m8` on the m7 job so the inline fill
is skipped.

### Stage 3 — merge

Two submissions, in order — `config.yaml` calls this
`fan_out: program-filter`, and `run_pipeline` does both:

1. an array, one task per `(program, filter)`;
2. **one** job, after the array, that reads every filter's part-1 output.

4 cpu, 64 gb, 8 h. Submitting only the array leaves the all-filter catalogs
unwritten. To see the index → `(program, filter)` map and the array bounds
before submitting:

```bash
python -m jwst_gc_pipeline.photometry.merge_catalogs \
       --target=brick --merge-singlefields --list-jobs
```

The merge is I/O-bound, so a large CPU ask buys nothing and costs queue time.

## Choosing

- **Queue quiet, one field, want it today** → A (or B with a large `FILTERS`).
- **Normal** → B, the default.
- **Queue busy, or jobs pending for hours** → C.
- **Queue jammed, or a field with many frames per filter** → D, plus the split
  m8.
- **A stage overran its wall** → split it one level finer rather than asking for
  more time; a longer `--time` competes for the same scarce large nodes.

Changing the default for every run means editing `config.yaml`
(`stages.<stage>.cpus`, `.fan_out`); changing it for one run means calling the
submit script directly.

## Traps

**`--export` eats commas and spaces.** `--export=ALL,FILTERS="F405N F410M"`
truncates at the space, and any comma-valued variable truncates at the comma.
Export the variables in your shell and pass a bare `--export=ALL`:

```bash
export FILTERS="F405N F410M F466N"
sbatch --export=ALL ... scripts/reduction/submit_cataloging.sbatch
```

`run_pipeline` does this for you.

**The five cataloging variables travel together.** `PROPOSAL`, `FIELD`,
`TARGET`, `EACH_SUFFIX`, `MODULES` — set one and you must set all five, or
`submit_cataloging.sbatch` exits 64.

**`EACH_SUFFIX` names the stage-1 products.** A field that destreaks writes
`*_destreak_o<obs>_crf.fits`; one that does not writes `*_align_o<obs>_crf.fits`,
and sickle differs between its short- and long-wavelength filters.
`reduction/destreak_policy.py` decides, and both stages read it.

**Exit codes through a pipe.** `python ... | tee log` reports `tee`'s status, so
a failed run looks like a clean one. `submit_cataloging_perframe_phase.sbatch`
pipes to `tee` and reads `${PIPESTATUS[0]}` for exactly this reason; the others
run python directly and read `$?`. A job that shows `COMPLETED` while its work
failed is usually a status read through a pipe.

**`TMPDIR`.** The scripts point it at node-local `$SLURM_TMPDIR` when there is
one. Leaving it at `/tmp` fills the node's root filesystem and wedges it for
everyone on that node.

## Watching a run

```bash
squeue -u $USER -o '%.10i %.30j %.8T %.10M %.6D %R'   # what is queued, and why it is not running
sacct  -j <jobid> --format=JobID,JobName%30,State,Elapsed,MaxRSS,ExitCode
scontrol show job <jobid> | grep -i reason
```

`Reason=QOSGrpCpuLimit` means the QOS trap above. `Reason=Resources` means the
node it wants does not exist yet — split finer. Logs land in
`/orange/adamginsburg/jwst/logs`, named
`<stage>_<jobname>_<jobid>_<arraytask>.out`.
