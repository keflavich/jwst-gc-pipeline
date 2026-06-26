# Per-frame fan-out for the manual `--each-exposure` pipeline (option C)

Goal: let each exposure frame be cataloged as an independent small SLURM job so
cataloging backfills into small queue holes (the queue bottleneck is large-cpu
node scarcity). Splits BELOW the filter boundary that `submit_cataloging_chain.sh`
already exploits.

## Key finding from the code read

`run_manual_pipeline` (cataloging.py:1738) runs phases `m12→m3→m4→m5→m6[→m7]`.
Each phase, for each `(module, filt)`:

1. build the seed (`prev_seed`) from the prior phase's vetted catalog + an
   i2d-augmented daofind on the prior residual-i2d;
2. **fit every frame independently** (`_run_one_frame_manual` in a
   `ProcessPoolExecutor`) — THIS is the fan-out unit, already pickleable;
3. **barrier**: (m12 only) reconcile out-of-FOV satstars; then merge per-frame
   catalogs → vet → tag `iter_found` provenance → build the mergedcat residual
   i2d → smooth it into the next phase's background map.

So per-frame splitting is naturally **per-phase `{fan-out array → finalize}`**,
chained `afterok`, looped over phases — NOT "one job runs all phases" (phase p
needs phase p-1's *global* merge/residual/bg).

## The four cross-phase state objects (in-memory dicts keyed by `(module,filt)`)

| state | produced | consumed | persistence for split |
|---|---|---|---|
| `bg_for_next` | end of each phase | m5/m6/m7 fit + seed | **already** reconstructs via `_reconstruct_smoothed_bg_path` |
| `resid_i2d_for_next` | end of each phase | m4/m5/m6 seed | on disk already; add deterministic-path reconstructor |
| `prev_merged_for` (skycoord, iter_found) | end of each phase | finalize provenance | reconstruct from prior phase's merged catalog (`iter_found` col is written to it) |
| `satstar_overrides` / `satstar_drops` | m12 finalize | m3..m7 fit | **new**: persist reconciled result to a small FITS, reconstruct on later phases |

Only `satstar_*` needs a genuinely new on-disk artifact; the other three are
already on disk or trivially derived.

## Design — additive control-flow gates (science code untouched)

New `optparse` options on `crowdsource_catalogs_long.py`:

- `--manual-stop-after-phase PH` — run only phases `[start..PH]` (with
  `--manual-start-phase` → exactly one phase per job).
- `--manual-frame-shard I/N` — in the per-frame fit, process only frames with
  `index % N == I`. Default '' = all (monolithic).
- `--manual-skip-finalize` — fan-out worker: fit the (sharded) frames, write a
  per-frame completion marker, then STOP before the barrier.
- `--manual-finalize-only` — barrier job: do NOT fit; verify every candidate
  frame has its phase marker (hard-crash on any miss → no silent drop), then
  reconcile/merge/vet/provenance/residual/bg as usual.

Extend the existing `--manual-start-phase` reconstruction block (cataloging.py
:1841) to rebuild ALL prior-phase state needed by the starting phase from disk:
`bg_for_next` (exists) + `resid_i2d_for_next` (deterministic path) +
`prev_merged_for` (prior merged catalog) + `satstar_overrides/drops` (persisted
m12 file). Then the phase-loop body runs byte-for-byte as today.

### Gated regions in the phase loop

- **fit block (2052-2080)**: if `finalize_only` → skip; else apply frame-shard
  to `frame_args`; fan-out workers write `{markers}/{frame}.{phase}.ok`.
- **frame-completeness**: shard/skip-finalize disables the in-memory
  `_expected vs _got` equality guard (each worker is independent). `finalize_only`
  REPLACES it with a stronger on-disk check: every candidate frame must have a
  marker, else abort.
- **data-i2d build (2189)**: gate to actual `phase=='m12'` (was `phase==phases[0]`
  — identical monolithically, but avoids needless rebuilds when each SLURM job's
  sliced `phases[0]` is a later phase).
- **m12 reconcile (2133-2170)**: additionally PERSIST `_ovr/_drp` to
  `{cut_bp}/catalogs/{filt}_{module}_satstar_reconciled_m12.fits`.
- everything 2172→ (merge/vet/provenance/residual/bg): runs only when not
  `skip_finalize`.

## SLURM orchestration

`submit_cataloging_perframe.sh`: for each phase, per `(module,filt)`:
  stage A = frame array `--array=0-(N-1) ... --manual-skip-finalize` (1-2 cpu),
  stage B = `--manual-finalize-only` afterok stage A (small),
then next phase's stage A `afterok` stage B. m7 finalize also does cross-band.

## Validation

`tests/test_perframe_equivalence.py` (or a script): run a small cutout BOTH
monolithically and via the per-frame path; assert the final per-filter vetted
catalogs + cross-band table are **bit-identical** (same rows, same flux columns).

## Status
- [x] options + reconstructors + gates in cataloging.py / crowdsource_catalogs_long.py
- [x] satstar persist + reconstruct (`_satstar_reconciled_m12.fits`)
- [x] completion markers + finalize completeness check (hard-crash on miss)
- [x] `submit_cataloging_perframe_phase.sbatch` + `submit_cataloging_perframe.sh`
- [x] helper unit tests (`tests/test_perframe_helpers.py`, 10 pass)
- [x] equivalence VALIDATOR script (`validate_perframe_equivalence.sh`)
- [ ] **run** the equivalence validator on a real cutout (needs a chosen target;
      heavy/long -> left for the user to launch) — the one remaining check before
      trusting the SLURM fan-out at scale.
