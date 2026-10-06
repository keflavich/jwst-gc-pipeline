# Product retention

`/orange/adamginsburg/jwst` was 93% full (62 TB free of 834 TB) when this was
written. Most of what the pipeline leaves behind is per-frame image
intermediates from merge phases that have since been superseded, plus
hand-made backup directories and renamed quarantines that nothing ever expires.

This document states which products are dead and why, so the argument is made
once instead of re-derived by hand every time someone needs space.
[`jwst_gc_pipeline/retention.py`](../jwst_gc_pipeline/retention.py) is the
executable form of it; `scripts/maintenance/prune_products.py` is the operator
front end.

## What crosses a phase boundary

Very little. `cataloging._reconstruct_smoothed_bg_path` calls the smoothed
background mosaic **"the only cross-phase state"**, and
`_reconstruct_resid_i2d_path` exists so a restarted per-frame worker can rebuild
the next phase's detection image from disk. Everything else a phase writes is
consumed inside that phase:

| product | who reads it | dead when |
|---|---|---|
| `*_{label}_daophot_{kind}_{residual,model}.fits` | `build_mergedcat_residuals` for the SAME label; the iter2 qfilt-bg path | the next label's mosaic exists |
| `*_{label}_daophot_{kind}_mergedcat_{residual,model}.fits` | the resample that writes `*_mergedcat_residual_i2d.fits` | that i2d exists |
| `*_mergedcat_residual_i2d.fits`, `*_..._smoothed_bg_i2d.fits` | the NEXT phase | never — this is the cross-phase state |
| merged + per-frame catalogs, every stage | the science | never |

The pipeline already makes this argument one product class further up: the
intermediate model i2d is skipped by default because it is *"display-only …
read by no pipeline logic"* (`--manual-keep-intermediate-model-i2d`). Retention
extends the same reasoning to the per-frame images.

## Catalogs are not in scope

Deliberately. Every per-stage merged catalog for brick totals ~41 GB, against
4+ TB of that field's stage images; cloudc's whole m2–m8 set is 13 GB. They are
the scientific record, they are what a re-analysis asks for, and they are the
only artifact showing how a source's photometry moved as the background model
improved. No default rule selects one.

Two derivative-table rules exist and are **off by default**:

* `allcols` — the `_allcols` superset, from which the minimal table beside it is
  derived in memory. `merge_catalogs` says downstream "doesn't consume" the
  extra columns and `diagnostics/inventory.py` lists it in `_DERIVATIVE_RE`, the
  products that are never canonical. Every reader in `scripts/` explicitly
  excludes it. 922 GB on disk, 798 GB of that brick's.
* `duplicate_table_format` — an ECSV that has a FITS twin. Which format is
  canonical is a project decision (the write site prefers ECSV for mixin and
  mask fidelity; the release ships FITS), so this rule will not make it for you.

## What protects a product

Selection never deletes on its own. `retention.Guard` holds four vetoes and
`plan()` applies all of them:

1. **A published release points at it.** `releases/v1.3-*/brick/exposures` is
   2,400 *symlinks* into the live tree, so deleting a live exposure silently
   breaks a public download. Targets are resolved with `realpath`, not assumed.
2. **The field has a queued or running SLURM chain.** A phase that restarts into
   missing inputs either trips the mergedcat guard or resumes from a partial
   marker set. If `squeue` cannot be reached the planner refuses to run rather
   than reading silence as "idle"; `--assume-idle` is the explicit override.
3. **It is younger than the age floor** (global `--min-age-days`, or the rule's
   own floor, whichever is longer).
4. **It matches a `--protect` glob.**

On top of that, `PROTECTED_SUFFIXES` / `PROTECTED_SUBSTRINGS` put every mosaic,
exposure-level science product, astrometry sidecar and PSF cache out of scope
for every rule, so a widened pattern cannot reach them.

`brick`, `cloudc` and `wd1` are symlinks, and `<field>/F<X>` is usually the same
directory as `<field>/mastDownload/JWST/F<X>`, so the same bytes are reachable
up to four ways. The walker deduplicates by resolved path; a naive sum
over-reports by ~4× on those fields.

## Using it

```bash
# what the safe rules would take, and what the guard is protecting
python scripts/maintenance/prune_products.py --target arches

# add the two opt-in derivative rules, and list every file
python scripts/maintenance/prune_products.py --target arches \
    --rule allcols --rule duplicate_table_format --verbose

# whole directories whose NAME says they are superseded, sized as trees
python scripts/maintenance/prune_products.py --target arches --directories

# see what the guard refused and why
python scripts/maintenance/prune_products.py --target brick --show-vetoed

# act on a plan you have read
python scripts/maintenance/prune_products.py --target arches \
    --manifest /orange/adamginsburg/jwst/logs/prune_arches_2026-09-02.json --apply
```

`--apply` is refused without `--manifest`, and the manifest is written and
fsynced *before* the first unlink, so an interrupted run still says exactly what
it was about to remove.

## In-run cleanup

`--manual-gc-superseded-perframe` (default **on**) cleans up at each phase
barrier, at the point where the phase's residual i2d and smoothed bg are on
disk. It removes this phase's mergedcat renders and the previous phase's raw
pair, keeps this phase's own raw pair — a retry of the mosaic still needs it —
and therefore never touches the final phase's.
`--no-manual-gc-superseded-perframe` restores the old leave-everything
behavior.

The selection is exact. Each mergedcat build writes a ledger beside its i2d,
`..._mergedcat_residual_perframe_inputs.json`, naming every per-frame raw pair
it read and every render it wrote. The barrier deletes from this phase's
ledger (renders) and from the ledger of the mosaic the phase was seeded from
(raw pairs), and from nothing else. A mosaic with no ledger, written before
ledgers existed, offers nothing. An earlier version globbed
`{prefix}*-{filt}-*_{label}_daophot_*`; the `*` spanned detector, module and
variant tokens (`_resbgsub`, `_epsf`, `_hybpsf`, `_group`), so a second chain
or variant sharing the pipeline directory lost its pairs, and cloudef's obs005
frames, which carry the `-o002_` prefix in obs002's directory, were offered to
obs002's cleanup.

The previous mosaic is the one this phase was seeded from, rebuilt by name on
a per-phase job (`_reconstruct_resid_i2d_path`). Its ledger is used only when
its name matches this phase's mosaic up to the phase label, ignoring
`_resbgsub` (`retention.same_run_mosaics`): same field, filter, module and
variant tokens (`_epsf`, `_hybpsf`, `_blur`, `_group`, `_bgsub`,
`_unsatstar`). A variant run sharing `{filt}/pipeline/` with production, whose
name rebuild lands on production's mosaic, is refused rather than deleting
production's raw pairs and markers.

Two chains on one field and filter with different `MODULES` still share
per-frame files: the raw pair is keyed by detector, not by merge label (#840),
so an `nrca` chain's m4 barrier retires m3 pairs a concurrently running
`merged` chain's m3 build may still need. The production submit scripts run
all modules of a phase in one job, with the next phase behind a dependency, so
they do not hit this. Concurrent per-module chains on one field need
`--no-manual-gc-superseded-perframe`.

Retiring a raw pair first removes that phase's per-frame completion markers
for the same frames (every merge label and detector spelling), and then marks
the ledger `raw_retired`. A `--skip-if-done` restart of a retired phase
therefore refits those frames rather than resuming from a marker whose
product is gone. `--finalize-only` of a retired phase fails at the
completeness check; rerun the phase from its fan-out instead.

It landed off first so the selection could be reviewed against the offline
tool's output on an idle field before any chain behaved differently. It is on
by default now, following a consumer audit of every path that reads a
per-frame image across a phase boundary: restart/`--manual-start-phase`, the
per-frame SLURM fan-out and its completion markers, the m7 cross-band seed,
the m8 forced fill, release staging, the registration/QA scripts, and
`--finalize-only --iteration-labels` (`mosaic_each_exposure_residuals`,
which re-mosaics per-frame raw residuals for the labels it is given). Every
one of them reads a mosaic (`_i2d.fits`, protected and untouched by this
selector), a catalog (also untouched — see "Catalogs are not in scope" above),
or the current phase's own raw per-frame pair (kept). None reads a per-frame
raw pair from a phase further back than the one just completed, so a
completed run now keeps only the final phase's raw pair plus every mosaic.
The one exception is `mosaic_each_exposure_residuals` asked for an
intermediate label after the run: those residuals are gone, and it raises
`ValueError: No per-exposure residuals found`. Run with
`--no-manual-gc-superseded-perframe` when that re-mosaic is needed.
Turning it on still changes what a completed run leaves behind for
inspection, which is why the offline tool
(`scripts/maintenance/prune_products.py`) and
`--no-manual-gc-superseded-perframe` both remain available.

### Smoothed-background mosaics (on by default)

Each phase writes `*_{label}_daophot_basic_mergedcat_residual_smoothed_bg_i2d.fits`
for the next phase, and each phase reads only its predecessor's map: m5
subtracts m4's, m6 m5's, m7 m6's, and the m8 forced fill reads m7's. m3 and m4
fit raw frames and read none. So at the barrier where phase N's map lands, the
map from two phases back has no reader: phase N+1 reads N's, and a retry of N
reads N-1's. `cataloging._gc_superseded_smoothed_bg` removes that one file,
counting back in the fixed chain order m12, m3, m4, m5, m6, m7: a per-phase
finalize job runs with start and stop phase equal, so its own phase list holds
one entry and cannot be counted back in. A
completed run keeps the last two maps (m6 + m7 multi-band, m5 + m6 single-band)
and every phase's residual and model mosaics, which are the diagnostics.

`--manual-keep-intermediate-smoothed-bg` keeps every map. Measured on
gc-treasury F212N, 2026-10-05: 1,224 maps (68 tiles x 3 modules x m2-m7), 303 GB
including the m2 checkpoint's `_im0_badastrom` quarantine copies; m2-m5 are two
thirds of that.

A restart with `--manual-start-phase=m3` or `m4` no longer requires the
previous map on disk, since neither phase reads it. m5-m7 still require it, and
the two-phase window keeps it for a retry of the phase that just completed.
Restarting further back than that means rerunning from an earlier phase.

Tools that read an intermediate phase's map after a run has finished find it
gone. `reference_fields/evaluate.find_products(phase='m4')` raises
`FileNotFoundError` in that case; `allow_missing_bg=True` returns
`smoothed_bg=None` with a warning. Run with
`--manual-keep-intermediate-smoothed-bg` when a benchmark needs m2-m5 maps.

This selector names `_i2d.fits`, a protected suffix. `retention.superseded_smoothed_bg`
accepts one exact path, rebuilt by `_reconstruct_smoothed_bg_path` (the
function restarts use), and checks the whole basename: the smoothed-bg suffix,
the `_mergedcat_residual_` infix that no residual or model mosaic carries, and
the label. A quarantine rename, a residual or model i2d, or another phase's map
is refused.

### The in-run path is protected by the name filter alone

`cataloging._gc_perframe_images` calls `spent_mergedcat_frames` and
`superseded_perframe_products` on a build's ledger and unlinks what they
return. Those two apply `_is_protected_name` — `PROTECTED_SUFFIXES` and
`PROTECTED_SUBSTRINGS` — and
**not** `Guard`. No release-target check, no busy-field check, no age floor, no
`--protect`. That is deliberate, and each missing veto is missing for a reason:

* **busy-field**: the in-run pruner runs *inside* a live chain, so this veto
  would refuse everything by construction. It is the offline tool's guard
  against a chain it is not part of.
* **release targets**: all 2,400 are `_crf.fits`, which `PROTECTED_SUFFIXES`
  already excludes — measured 2026-10-06, 0 of 2,400 reach the Guard unprotected. The
  in-run selectors also return only `*_daophot_*_{residual,model}.fits` names
  listed in a ledger, a shape no release has ever published.
* **age floor**: meaningless for a file this same run wrote minutes ago.
* **`--protect`**: an operator flag, and there is no operator in a SLURM job.

**If you add a rule to the in-run path, it does not get the Guard.** The two
paths have different safety properties on purpose; anything new on the in-run
side has to earn its safety from `_is_protected_name` and from an exact list
of what one build used, or go through `plan()` instead.

## Measured, 2026-08-31

Floors, from a direct `find -printf` inventory. Pipeline-directory rows cover
brick and cloudc at depth 1; catalog rows cover eight fields; the quarantine
directory and debris sweeps are complete.

| | |
|---|---|
| per-frame residual + model, brick+cloudc | 4.23 TB (4.3 TB below the final phase) |
| per-frame mergedcat renders, brick+cloudc | 2.42 TB |
| quarantine directories, 226 in jwst | 8.19 TB |
| — of which `pre_skycoord_fix_backup_20260602`, 76 dirs | 6.06 TB |
| `brick/catalogs/obsolete/` | 843 GB |
| `_allcols` supersets | 922 GB |
| ECSV/FITS twins | 283 GB |
| renamed quarantines (`*.fits_stale`, `*.STALE_*`) | 687 GB |
| core dumps, 842 files | 350 GB |

`ulimit -c 0` is already in every sbatch template, so the core dumps are from
before that landed (May 2023) plus runners outside this repo; the `core_dump`
rule is what actually clears them.
