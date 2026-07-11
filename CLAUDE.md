# CLAUDE.md — jwst-gc-pipeline

Instructions for any agent (or human) working in this repo. **Read this before
touching astrometry, alignment, reference catalogs, or reduction/cataloging.**

When editing jwst-gc-pipeline, use git worktrees.

---

## ⛔ ASTROMETRY RULE #1 — never nearest-neighbour-median against a dense catalog

**Do NOT compute or validate an astrometric offset as the MEDIAN (or mean) of
nearest-neighbour matches (`match_to_catalog_sky` / `search_around_sky` +
`np.median`/`np.mean`) against a DENSE reference (VIRAC2 / VVV / GNS; median NN
spacing ≲ 3").**

Why: when the true shift exceeds the reference's NN spacing (~0.3"), NN pairs the
WRONG star and the median **collapses toward ~0** (or a spurious value). The method
**fabricates false agreement** and has repeatedly fooled *validation* of the GC
fields (a NN-median check "confirms 0.00 fine" on a frame that is really off) — a
recurring failure mode behind the 2221/1182 astrometry errors. (The specific
brick-1182 v001 ~20" error was an offsets-table CURATION collapse — v001 overwritten
with v002's value — not a NN-median measurement; but it is the same class of silent
false-agreement failure, so NN-median against a dense catalog is banned outright.)

This is now **enforced in code** — `measure_offsets.assert_sparse_reference_for_nn_median`
raises `DenseNNMedianAstrometryError` — and by a **grep-guard test**
(`tests/test_no_adhoc_nn_median_astrometry.py`) that fails CI if a NEW file pairs a
NN match with a median/mean. Do not disable either.

### The ONLY sanctioned ways to measure an astrometric offset

1. **Offset-HISTOGRAM stacking** — histogram ALL pairwise offsets within a window,
   take the peak. Density-immune; correct no matter how large the shift.
   Use `jwst_gc_pipeline.photometry.astrometry_offsets.measure_offset(a, b)` (the
   public, guarded helper) or `scripts/reduction/astrometry_audit.py::xcorr`.
2. **A SPARSE reference** — the Gaia-only subset (`source == b'GaiaDR3'`, medNN
   ~5.7"), never the full dense catalog.

Do **not** hand-roll `match_to_catalog_sky(...).median()` in an ad-hoc script.
Import `measure_offset` instead.

### A bulk offset ≈ 0 does NOT mean "clean"

A field-average / whole-mosaic offset can read ~0 while HALF the mosaic is offset
(brick-1182: visit-001 exposures tile the top half, shifted ~20" from visit-002;
bulk peak washed it out). **Always map the offset PER TILE** (`measure_offset_grid`,
`registration_failsafes.py`) and report per-tile peak-contrast: ≳5 = real tie, ~1 =
broken. A single global number is never sufficient sign-off.

### A LOW contrast can mean "offset ≫ window", NOT "no tie"

A large rigid offset has ZERO true pairs inside a narrow search window, so the peak
is noise (low contrast, and different against a dense vs sparse reference). That is
exactly how brick-1182 v001's ~20" offset first read as ~2"/incoherent. So:
- `measure_offset` now **sweeps the window** (3→10→30→60") by default and takes the
  highest-contrast peak — do not disable `sweep`. A returned `swept=True` /
  `window_arcsec ≫` your expected offset means the frame is grossly shifted.
- On a weak tie, cross-check TWO references (`agree_across_references`, VIRAC vs
  Gaia-only): a real tie agrees; a spurious peak moves.

### Correcting an already-aligned frame after the offsets table changes

`fix_alignment` skips a frame that already has a `RAOFFSET` header (idempotent). If
you CORRECT the offsets table, the stale frame is silently kept (this is how v001
stayed ~20" off). The disagreement guard warns when a frame's baked `RAOFFSET`
differs from the current table value; set `FORCE_REALIGN_ON_DISAGREE=1` to hard-stop.
The fix is to REGENERATE the working copy from `_cal` (destreak overwrite → RAOFFSET
resets → current table applied), never to re-apply on top of the stale shift.

### Reading list before any astrometry change
- `jwst_gc_pipeline/reduction/ASTROMETRY_WCS_CORRECTION_FLOW.md` — the full flow,
  the two authoring points, no-double-correction rule, epochs, module-lock policy.
- The `brick-1182-*` and `dense-nn-median-guard-enforced` memory notes.

---

## Release gate

`scripts/release/stage_release.py` runs `registration_failsafes.py --scan` (per-tile,
cross-band + own-catalog) and **REFUSES to stage** any field with a locally
misregistered band. Do not stage around it. The `--allow-registration-fail`
override additionally requires `ALLOW_REGISTRATION_FAIL=1` in the environment — it
exists only for deliberate, justified overrides, not for making a red gate go green.

---

## Workflow

All pipeline work goes on **worktree branches** (`../jwst-gc-pipeline-<slug>`) pushed
as **pull requests**, one concern per PR. Never accumulate uncommitted changes on the
active `main` working tree (it is the live reduction environment).

## Other conventions
- SLURM: use `--account=astronomy-dept --qos=astronomy-dept-b`. The default
  `adamginsburg` QOS caps cpu=10 and will hang a 16-cpu task.
- New photometry code goes in new modules, not the `crowdsource_catalogs_long.py`
  monolith.
- No bare `try/except`; catch specific exceptions only.
