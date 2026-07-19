# Versioning, Provenance & Rerun-Skip — design + operating instructions

**Goal:** strong provenance for every pipeline product *and* efficient runs that
skip the computationally expensive steps that provably don't need to re-run.

This document is the authority for the tag scheme, the per-product provenance
record, and the rules that decide whether a stage must be re-run. The tooling
lives in `jwst_gc_pipeline/versioning/` (`tags.py`, `fingerprint.py`,
`prov_sidecar.py`, `rerun.py`).

> **PR1 scope.** This PR ships the tag scheme + run guard, the fingerprint/
> sidecar/decision tooling, the tag-on-merge CI, and this doc. **Wiring the
> stamping into the imaging and cataloging write paths is a follow-up PR** (see
> §7). Until then the decision engine runs on explicitly-supplied provenance
> records; once wiring lands it reads live `.prov.json` sidecars.

---

## 1. The pipeline is a DAG

```
imaging ─► m12 ─► m3 ─► m4 ─► m5 ─► m6 ─► m7 ─► m8
```

* **imaging** — the parent `jwst` pipeline (`Detector1`/`Image2`/`Image3`) plus
  the repo's destreak/align/DVA + bulk-shift steps → per-exposure `_crf` frames
  and `_i2d` mosaics.
* **m12 … m8** — the manual-iteration PSF/crowdsource cataloging phases
  (`cataloging.run_manual_pipeline`). Each phase N **seeds from N−1**: the
  previous phase's merged catalog, its residual detection image, and its
  smoothed-background map. m7/m8 additionally read the previous vetted catalogs
  for the cross-band seed.

### The seeding-cascade invariant (load-bearing)

> **Any change that alters stage N's *output* invalidates N+1…end.** Only a
> change that leaves N's output byte-identical can be absorbed locally.

This is why a bulk astrometric shift introduced at **m12** (where the astrometry
checkpoint may re-tie) forces **all** later stages to re-run: the corrected WCS
moves where sky-seeds land in pixels, so every downstream seed changes. It is
also why the checkpoint ladder **freezes** the solution at m3+ and *raises* on
any shift there (`astrometry_checkpoint.CORRECTION_STAGES = m1/m2/m12`).

---

## 2. Per-facet fingerprints (why we don't hash "the file")

The re-run decision is not "did the product change" but "**which facet** changed",
because different facets permit different, cheaper actions. Every product carries
three independent, header-decoupled output hashes (`fingerprint.facet_hashes`):

| Facet       | Hash over…                                            | A change here means… |
|-------------|-------------------------------------------------------|----------------------|
| `data_hash` | SCI (+ERR/DQ) pixel arrays only, header-excluded      | pixels differ → any downstream PSF fit must re-run |
| `wcs_hash`  | canonicalized WCS cards incl. `RAOFFSET`/`DEOFFSET`   | astrometric solution moved → maybe only refresh RA/Dec |
| `meta_hash` | remaining header cards (WCS + volatile stamps excluded)| header-only → re-stamp, recompute nothing |

Byte-identity of `data_hash` is the mechanism the repo lacked: staleness used to
be judged from mtime + generation stamps, never content. Now, after a jwst/CRDS
re-reduction, if the re-drizzled SCI is **byte-identical**, cataloging is proven
unnecessary — decided, not guessed.

The **input** side of each stage is fingerprinted too:

* `env` — `jwst_version`, `crds_context` (imaging only; = the parent-written
  `CAL_VER`/`CRDS_CTX`).
* `code` — git blob hashes of the curated source files implementing the stage
  (`fingerprint.STAGE_CODE_FILES`). Deliberately coarse: cataloging stages share
  one file-set, so a cataloging-code change forces m12 to *recompute* and the
  resulting `data_hash` comparison prunes whether m3+ actually change. Coarse
  input hashes only ever cost the owning stage's compute, never a spurious
  full-cascade.
* `params` — the non-volatile CLI/options (`fingerprint.params_hash`; job name /
  worker count / paths excluded).
* `upstream` — the upstream stage's output facet hashes (this is how the cascade
  is detected: m4's record carries m3's `data`/`wcs`/`meta`).

All of this is stored in a sidecar next to the product:
`<product>.prov.json` (`prov_sidecar`), and the compact
`{tag, stage, data/wcs/meta}` subset is mirrored into FITS keywords
`GCTAG`/`GCSTAGE`/`GCDATAH`/`GCWCSH`/`GCMETAH` so a product stays self-describing
even if the sidecar is lost.

---

## 3. Tag scheme

* **Release tag** (auto, per merged PR): `YYYY-MM-DD_PR<n>` — merge date + PR
  number, annotated tag on the merge commit. Created by
  `.github/workflows/tag-on-merge.yml`.
* **Dev tag** (untagged or dirty tree): `YYYY-MM-DD_PR<n>_<shortcommit>[-dirty]`
  — the nearest release tag's lineage + this commit. A dev product can never be
  mistaken for a release product.

`tags.get_pipeline_tag()` resolves the current tag; every stage stamps it as
`GCTAG` (extending the existing `GCPIPEV` raw-commit stamp in
`jwst_gc_pipeline/provenance.py`).

### The run guard (enforced)

Every **production** stage entry calls `tags.assert_runnable_version(stage)`,
which **hard-blocks** (raises `UntaggedPipelineError`) an untagged or dirty tree.
A development run must opt in explicitly:

```bash
GC_ALLOW_DEV=1 python -m jwst_gc_pipeline...      # or pass allow_dev=True
```

A dev run is permitted but warns and stamps the dev tag. **The pipeline runs
only on tagged versions** — this is the mechanism that enforces it.

The guard is called at exactly two points — the `__main__`/`main()` CLI entries
of `PipelineRerunNIRCAM-LONG.py` (imaging) and `catalog_long.py`
(cataloging). It is **not** called inside `run_manual_pipeline`, the cutout
sub-runs, or any importable helper, so importing these modules (tests, notebooks,
programmatic callers) never trips it.

#### ⚠️ Rollout (do this in the same change that merges the guard)

Once the guard is live, **every SLURM reduce/catalog job hard-blocks at entry**
unless it runs on a tagged, clean checkout **or** permits a dev run. The
submission scripts are wired for this: `scripts/reduction/submit_reduction.sbatch`
and `submit_cataloging*.sbatch` now `export GC_ALLOW_DEV="${GC_ALLOW_DEV:-1}"`, so
the live working-tree queue keeps running (stamping dev tags) while a **release**
run — a checkout at a release tag, clean — passes as production regardless of the
flag. Set `GC_ALLOW_DEV=0` to force the block (e.g. to guarantee a run is on a
tagged checkout). A hand-launched command must do the same.

---

## 4. The decision matrix (the instruction set)

`rerun.decide_stage` + `rerun.propagate_cascade` implement exactly this:

| Change | imaging | m12 | m3–m6 | m7 | m8 |
|---|---|---|---|---|---|
| jwst/CRDS bump, SCI **byte-identical** after re-reduce | RE_REDUCE → re-stamp | SKIP | SKIP | SKIP | SKIP |
| jwst/CRDS bump, SCI **differs** | RE_REDUCE | REFIT | REFIT | REFIT | REFIT |
| WCS-only, **post-hoc** (data identical) | SKIP | REPROJECT | REPROJECT | REPROJECT | REPROJECT |
| bulk shift **re-tied at m12** (`--wcs-change-mode reseed`) | — | REFIT | REFIT | REFIT | REFIT |
| bulk shift at m3+ (reseed intent) | — | — | **BLOCKED** | BLOCKED | BLOCKED |
| cataloging code/params change in stage S | — | REFIT S… | REFIT S… | REFIT S… | REFIT S… |
| header meta only (non-WCS) | RESTAMP | RESTAMP | RESTAMP | RESTAMP | RESTAMP |

**Verdict meanings**

* `SKIP` — reuse the recorded product.
* `RESTAMP` — only re-write the header stamp.
* `REPROJECT` — refresh `x,y → ra,dec` on the existing catalog
  (`astrometry_utils.reproject_xy_to_world`); **no PSF fit**. Valid because the
  cataloging fits are done in detector (x,y) space with `group=False`, so
  `x_fit`/`y_fit` are generation-invariant — only the sky projection is stale.
* `REFIT` — re-run this cataloging stage's fit (and, by the cascade, all later
  stages).
* `RE_REDUCE` — re-run the parent `jwst` imaging pipeline. Downstream is
  **conditional**: re-reduce, then re-plan — the new product's `data_hash`
  decides whether cataloging skips or refits.
* `BLOCKED` — a bulk shift with reseed intent at a frozen stage; forbidden by the
  checkpoint ladder.

### `posthoc` vs `reseed` — the one operator choice

A WCS-only change (data identical) has two legitimate, distinct intents:

* **`posthoc`** (default): the offsets table was corrected *after* a finished
  run and you only want the final catalog's RA/Dec refreshed. → `REPROJECT`
  everywhere. Cheap.
* **`reseed`**: you are *re-tying* the astrometry at m12 and want the fits redone
  from the corrected seeds. → `REFIT` from m12, full cascade.

`rerun plan --wcs-change-mode {posthoc,reseed}` selects. The engine cannot infer
intent, so it defaults to the cheap, common case and makes the expensive one
explicit.

---

## 5. Using the tool

```bash
# Plan a REAL field from disk: scan its stamped products, recompute the current
# facets (code from the repo, live jwst/CRDS env, parents' facets from disk),
# and print which stages to re-reduce / refit / reproject / skip:
python -m jwst_gc_pipeline.versioning.rerun plan --field /path/to/field/catalogs \
    [--wcs-change-mode posthoc] [--no-live-env] [--json]

# Compare two provenance-record maps (what-if):
python -m jwst_gc_pipeline.versioning.rerun plan \
    --records recorded.json --current current.json [--wcs-change-mode posthoc] [--json]

# Report the recorded provenance state under a field's product tree:
python -m jwst_gc_pipeline.versioning.rerun plan --scan /path/to/field/catalogs
```

`--field` is the operator entry point: point it at a per-module/-filter subtree
(one product per stage) and it emits the ready-to-run plan, each stage annotated
with its action hint; a BLOCKED stage short-circuits the plan with a banner.

Output is one line per stage: `stage  VERDICT  reasons`. `--json` for tooling.

Programmatic:

```python
from jwst_gc_pipeline.versioning import rerun, fingerprint, prov_sidecar
d = rerun.decide_stage('m5', recorded_record, current_record)
plan = rerun.plan_from_records(recorded_map, current_map)   # cascaded
```

---

## 6. How this maps onto what already exists

| Concern | Existing mechanism | This layer adds |
|---|---|---|
| code identity | `GCPIPEV` commit stamp (`provenance.py`) | `GCTAG` release/dev tag + per-stage `code_hash` |
| imaging generation | `CAL_VER`/`CRDS_CTX`/`DVACORR` (`GENERATION_KEYS`) | rolled into the `env` input facet + byte-identity gate |
| x,y→ra,dec refresh | `reproject_xy_to_world` (already the cheap path) | the `REPROJECT` verdict that *dispatches* to it |
| bulk-shift stage lock | checkpoint ladder (m12 correct / m3+ frozen) | the `reseed` cascade + `BLOCKED` verdict that mirror it |
| content hashing | only release `MANIFEST.json` sha256 | per-facet `data/wcs/meta` hashes for rerun decisions |
| staleness | mtime + generation stamp + `_im0_badastrom` tag | content-addressed, facet-decomposed, cascade-aware |

---

## 7. Rollout

* **PR1 — tooling + doc.** (merged, `2026-07-16_PR109`.) tags / fingerprint /
  prov_sidecar / rerun engine + `plan` CLI + tag-on-merge CI.
* **PR2 — stamping.** (this PR.) `provenance.py` now also stamps `GCTAG` on
  every FITS write. `versioning/stamping.py` writes a `.prov.json` sidecar at
  each stage's final write and mirrors `GCSTAGE`/`GCDATAH`/`GCWCSH`/`GCMETAH`
  (16-hex prefixes) into the FITS. Wired **fail-soft** at:
  - imaging `_i2d` (per-module + merged) and per-exposure `_crf`
    (`PipelineRerunNIRCAM-LONG.py`, `_stamp_imaging_product`);
  - each merged per-band catalog (m12/m3–m6) and the m7/m8 combined catalogs
    (`cataloging.py`, `_stamp_catalog_provenance`).
  `env` (jwst/CRDS/DVA) is auto-read from the product header; `code`/`params`
  are recorded. **Upstream-facet threading is deferred** (the recorded
  `inputs.upstream` is empty for now), so these sidecars already drive the
  byte-identity gate (compare a stage's `data`/`wcs` facet across runs) but the
  full cross-stage cascade wiring lands with PR3/PR4.
* **PR3 — guard wiring + upstream threading.** (this PR, WIP.)
  `assert_runnable_version` is called at the imaging and cataloging CLI entry
  points (right after option parsing), so a production run on an untagged/dirty
  tree hard-blocks unless `GC_ALLOW_DEV=1`. `versioning/upstream.py`
  (`STAGE_PARENTS`, `pool_facets`, `upstream_from_sidecars`) reads a stage's
  parent-product sidecars and records their output facets as `inputs.upstream`,
  so the rerun cascade is now driven entirely by on-disk sidecars: m3←m12,
  m4←m3, …, m7←pool(all filters' m6), m8←m7. A CI grep-guard
  (`test_stage_entries_guarded.py`) fails if an entry stops calling the guard.
  The `imaging`-frame upstream facet is **now threaded** at every cataloging
  stamp site: `_stamp_catalog_provenance` records `inputs.upstream['imaging']` =
  `pool_facets` over the per-exposure `_crf` sidecars for that (module, filter)
  — the per-band catalogs pool their own frames, m7/m8 pool across all filters.
  So a re-reduced frame (changed crf `data` facet) now forces a downstream
  `REFIT` through the **pure sidecar** path too, not only through `plan --field`.
  (Fail-soft: if a crf sidecar is absent, that parent is simply omitted.)
* **PR4 — `plan` field locator.** (this PR, WIP.) `rerun plan --field/--proposal`
  locates each stage's product(s) by the naming conventions, reads the recorded
  sidecar, recomputes the current facets + code/params/env from disk, runs the
  decision engine, and prints a ready-to-run plan (which stages to submit, which
  to reproject-only, which to skip), with a BLOCKED stage short-circuiting the
  printed plan.
