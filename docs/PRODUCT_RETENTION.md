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

Two derivative-table rules exist:

* `allcols` (**on by default**) — the `_allcols` superset, from which the
  minimal table beside it is derived in memory. `merge_catalogs` says
  downstream "doesn't consume" the extra columns and `diagnostics/inventory.py`
  lists it in `_DERIVATIVE_RE`, the products that are never canonical. Every
  reader in `scripts/` explicitly excludes it, and that was verified again
  before turning this rule on (grep of `scripts/release`, `jwst_gc_pipeline/cmz`
  and `merge_catalogs.py` turned up no reader anywhere). 922 GB on disk, 798 GB
  of that brick's.
* `duplicate_table_format` (**off by default — do not flip without narrowing
  it first**) — an ECSV that has a same-stem FITS twin. The obvious read is
  "one is derived from the other", but for the field's combined merged table
  (`basic_merged_indivexp_photometry_tables_merged*`) that is wrong:
  `scripts/release/stage_release.py::_emit_table_group` ships BOTH formats as
  separate deliverables whenever both exist (`full_fits` + `full_ecsv` items),
  and real manifests do carry both — checked across the current release tree,
  v1.0 cloudc/gc2211/sgrb2/sgrc each staged one `.ecsv` catalog alongside its
  `.fits` twin. `jwst_gc_pipeline/cmz/catalog_assembly.write_outputs` makes the
  same call about its own output ("FITS/ECSV are the familiar deliverables").
  So "has a twin" does not mean "is dead" for that one stem family, and this
  rule cannot tell that stem apart from a genuinely dead intermediate ECSV
  today. Enabling it safely needs that distinction added first.

## What protects a product

Selection never deletes on its own. `retention.Guard` holds the vetoes and
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
5. **(`stale_badastrom_mosaic` only) a release MANIFEST.json recorded its
   pre-quarantine name as a staging `src`.** See below.

On top of that, `PROTECTED_SUFFIXES` / `PROTECTED_SUBSTRINGS` put every mosaic,
exposure-level science product, astrometry sidecar and PSF cache out of scope
for every rule, so a widened pattern cannot reach them.

`brick`, `cloudc` and `wd1` are symlinks, and `<field>/F<X>` is usually the same
directory as `<field>/mastDownload/JWST/F<X>`, so the same bytes are reachable
up to four ways. The walker deduplicates by resolved path; a naive sum
over-reports by ~4× on those fields.

## Checkpoint-quarantined mosaics (`stale_badastrom_mosaic`, on by default)

`astrometry_checkpoint.mark_i2d_stale` never deletes a mosaic built on
offsets the m2 checkpoint later corrected — it renames it, `..._i2d.fits` ->
`..._i2d_im0_badastrom.fits` (or `..._im0_badastrom.<N>.fits` if that name is
taken), and drops a `.why.json` sidecar. `PROTECTED_SUFFIXES` does **not**
catch this name — it matches `_i2d.fits`, and a quarantined twin ends in
`_badastrom.fits` — which is why this needed its own rule rather than a gap
in an existing one.

The rule fires only when the ORIGINAL (pre-quarantine) name exists on disk
again AND is newer than the stale twin — i.e. the field was actually
regenerated after the quarantine, not merely some unrelated file sharing the
stem. No replacement, or one no newer than the twin (a second checkpoint run
against the same still-bad mosaic), leaves it alone. This matters because
`scripts/release/stage_release.py::_badastrom_sibling` relies on exactly the
*absence* of the original name, with the twin beside it, to refuse staging a
field whose mosaic was quarantined and never regenerated — once the original
is back and newer, that refusal path is already dead, and the twin is pure
debris. `min_age_days` is 14.

**Guard fact #5 exists only for this rule.** `scripts/release/release_freshness.py`
re-stats an *already-staged* release's manifest `src`, and when it finds a
`*_im0_badastrom*.fits` twin NEWER than the staging time it reports that
staged copy `quarantined` rather than `live` — its whole reason for existing
("Cloud C's published images predate the 2026-07-12 astrometry fix, so the
page was showing ~4″ errors as evidence that the astrometry is sound").
Without a twin it falls back to comparing recorded vs. current file size,
which is blind for a re-drizzled `i2d`: the output grid is fixed, so a
re-drizzle after a mas-level correction can write an *identical* byte count.
So for any `src` a release manifest ever recorded, that quarantine twin may be
the only evidence an already-published, still-served release page (older
releases stay servable — `make_webpage.py` renders every version) has that it
needs to stop presenting those bytes as current. `guard_for()` reads every
`releases/*/*/MANIFEST.json` (`retention.release_manifest_srcs`) and the Guard
refuses to let `stale_badastrom_mosaic` touch any twin of a recorded `src`,
regardless of age or replacement. Verified empirically: real release manifests
do ship catalogs in both formats (see `duplicate_table_format` above) — the
same "identical size looks the same as live" trap is generic to this archive,
not a hypothetical.

Only the renamed `.fits` is removed; the `.why.json` sidecar is left in place
(a few KB, and harmless — `stage_release._quarantine_note` only reads it
beside a twin that still exists, so an orphaned sidecar is never read again).

Measured 2026-10-06: 6,696 `*_i2d_im0_badastrom*.fits` files across
`/orange/adamginsburg/jwst` (mostly crowded_l3's 9438 campaign); 27
`releases/*/*/MANIFEST.json` files recording staged `src` paths.

## Using it

```bash
# what the safe (default-on) rules would take, and what the guard is protecting
python scripts/maintenance/prune_products.py --target arches

# add the one opt-in derivative rule not on by default, and list every file
python scripts/maintenance/prune_products.py --target arches \
    --rule duplicate_table_format --verbose

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
`superseded_perframe_products` and unlinks what they return. Those two apply
`_is_protected_name` — `PROTECTED_SUFFIXES` and `PROTECTED_SUBSTRINGS` — and
**not** `Guard`. No release-target check, no busy-field check, no age floor, no
`--protect`. That is deliberate, and each missing veto is missing for a reason:

* **busy-field**: the in-run pruner runs *inside* a live chain, so this veto
  would refuse everything by construction. It is the offline tool's guard
  against a chain it is not part of.
* **release targets**: all 1,200 are `_crf.fits`, which `PROTECTED_SUFFIXES`
  already excludes — measured, 0 of 1,200 reach the Guard unprotected. The
  in-run selectors also glob only `*_{label}_daophot_*_{residual,model}.fits`,
  a shape no release has ever published.
* **age floor**: meaningless for a file this same run wrote minutes ago.
* **`--protect`**: an operator flag, and there is no operator in a SLURM job.

**If you add a rule to the in-run path, it does not get the Guard.** The two
paths have different safety properties on purpose; anything new on the in-run
side has to earn its safety from `_is_protected_name` and from the glob it uses,
or go through `plan()` instead.

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
