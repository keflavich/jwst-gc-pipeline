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
WRONG star and the median **collapses toward ~0** (or a spurious value). This has
**silently corrupted brick-1182 astrometry more than once** — both when *building*
the offsets table (it zeroed a real 1.9" shift) and when *validating* it (it
"confirmed 0.00 fine" on a mosaic that was 1.9" off). The method fabricates
agreement. It is the single recurring cause of the 4" errors in programs 2221/1182.

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

A field-average / whole-mosaic offset can read ~0 while HALF the mosaic is untied
(brick-1182: visit-001 exposures tile the top half with a non-rigid/broken WCS;
bulk peak washed it out). **Always map the offset PER TILE** and report per-tile
peak-contrast (peak/median): contrast ≳5 = real tie, ~1 = broken.
`measure_offset(..., per_tile=True)` and `scripts/release/registration_failsafes.py`
do this. A single global number is never sufficient sign-off.

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
