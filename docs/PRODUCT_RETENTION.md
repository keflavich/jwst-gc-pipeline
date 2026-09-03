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
   1,200 *symlinks* into the live tree, so deleting a live exposure silently
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

`--manual-gc-superseded-perframe` (default **off**) runs the same selection at
each phase barrier, at the point where the phase's residual i2d and smoothed bg
are on disk. It removes this phase's mergedcat renders and the previous phase's
raw pair, keeps this phase's own raw pair — a retry of the mosaic still needs it
— and therefore never touches the final phase's.

It is off by default because turning it on changes what a completed run leaves
behind for inspection. Landing it off means this PR can be reviewed on the
offline tool's output first, on a field that is idle, before any chain behaves
differently.

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
