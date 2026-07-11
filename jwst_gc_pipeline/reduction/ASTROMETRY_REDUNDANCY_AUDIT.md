# Astrometry / offsets code — redundancy audit (2026-07-11)

Audit of the astrometric-offset code across `jwst_gc_pipeline` and
`brick-jwst-2221/brick2221/analysis`, focused on **redundancies that cause
RECURRENT errors** (the same computation implemented in several places that
drift, so a fix in one is silently undone by another). Written after the
brick-1182 visit-001 corruption (a builder collapsed distinct visits onto one
visit's offset). All findings below were verified by inspection; false alarms
are called out.

Companion docs: `ASTROMETRY_WCS_CORRECTION_FLOW.md` (the correct flow + the ⛔
dense-NN-median rule). New safeguard shipped with this audit:
`validate_offsets_table.py` (+ tests) — flags a collapsed offsets table.

## Ranked redundancies

### R1 (CRITICAL) — THREE builders write the same `Offsets_JWST_Brick<prop>_VIRAC2locked.csv`
`build_virac2_locked_perexp.py` (per-visit), `relock_exposures.py` (per-exposure),
and `build_calframe_locked_offsets.py` all write the **same production table**
that `fix_alignment` consumes. Whichever ran last wins; a fix in one builder is
silently overwritten by another on the next run. This IS the recurrence engine:
the brick-1182 collapse persisted across rebuilds because the fixed and unfixed
builders competed for one file.
- **Fix:** one authoritative builder (per-exposure preferred — the 2026-06-20
  jitter measurement favors it). Deprecate/delete the others, or give each a
  distinct output path and have `fix_alignment` select by config. Add a
  `validate_offsets_table()` call at the end of the surviving builder.

### R2 (CRITICAL) — 3× duplicate `robust_shift` / `robust_offset` with parameter drift
`relock_exposures.py:robust_shift`, `build_calframe_locked_offsets.py:robust_shift`,
`lock_exposures.py:robust_offset` are near-identical clipped-median solvers with
**different constants** (search 0.3" vs 0.5"; clip 60 vs 80). Tuning one does not
fix the others; two can produce conflicting offsets for the same data.
- **Fix:** one shared solver. Prefer routing all of them through
  `jwst_gc_pipeline.photometry.astrometry_offsets.measure_offset` (histogram
  stacking + window sweep + `NoCoherentTieError`), which is crowding-proof and
  bridges the ~22" per-visit errors a 0.3" NN median cannot. (relock_exposures
  already gained a histogram bridge in PR #34; the other two have NOT.)

### R3 (HIGH) — two near-identical realign blocks in `PipelineRerunNIRCAM-LONG.py`
Two `realign_to_catalog(reftbl['skycoord'], ...)` call blocks (~L962 and ~L1123),
each preceded by a VVV block, differing only slightly. A fix applied to one (e.g.
the dense-NN skip) must be duplicated in the other or one path regresses.
- **Fix:** factor the post-Image3 VVV+refcat realign into one helper called from
  both code paths.

### R4 (MEDIUM) — convention chaos across solvers (cosδ vs no-cosδ, mas vs arcsec)
`measure_offset` = on-sky (×cosδ), mas. `coarse_xcorr`/`coord_shift` = coordinate
(no-cosδ), arcsec. `robust_shift` = on-sky, mas (converted to arcsec at write via
`/1000`). `coord_shift` clips on on-sky distance but returns no-cosδ. Each port
risks a dropped/doubled cosδ (~12% at the GC) or a 1000× unit slip.
- **Fix:** standardize on one convention at the API boundary (astrometry_offsets
  is on-sky mas throughout); annotate table columns with units; centralize the
  coordinate↔on-sky conversion in one place.
- **NOTE — false alarm:** an earlier pass flagged relock_exposures returning mas
  into an arcsec column as a live 1000× bug. It is **not** — line 114 divides by
  1000 (`dra=sr/1000.0  # arcsec`). Verified. Left as documentation only.

### R5 (MEDIUM) — orphaned tables + a reader without a source
`offsets/` accumulates deprecated tables (`*_F200ref*`, `*_F405ref*`, `*_VVV*`)
no current builder writes; a script that names one by mistake reads a stale
frame silently. Separately, `fix_alignment` has a merged-mode path that reads
`*_VIRAC2_average.csv`, but `build_virac2_fixalign_offsets.py` appears
incomplete (no write) — a **reader whose builder is missing**.
- **Fix:** archive/delete deprecated tables (keep VIRAC2locked); either complete
  the average builder or remove the average-table path from `fix_alignment`.

## What is already well-covered (do NOT re-add)
PRs #65/#66/#68 shipped: the dense-NN-median guard (`assert_sparse_reference_for_nn_median`,
raises on dense refs), `astrometry_offsets.measure_offset` (window sweep +
`min_contrast`), `measure_offset_grid` (per-tile — catches half-mosaic untying),
`agree_across_references` (VIRAC vs Gaia), a RAOFFSET-vs-table disagreement guard
in `fix_alignment`, `realign_to_catalog` skip-on-dense, and tests
`test_dense_nn_median_guard`, `test_astrometry_offsets_sweep`,
`test_no_adhoc_nn_median_astrometry`, `test_registration_gate`.

## New in this audit
- `reduction/validate_offsets_table.py` + `reduction/tests/test_validate_offsets_table.py`
  — flag a COLLAPSED offsets table (distinct visits sharing an offset to <5 mas)
  and insane magnitudes (unit slips). Run it in every builder and (as a warning)
  in `fix_alignment` before consuming the table. Verified: it flags the original
  brick-1182 collapse and passes the corrected production table.

## Recommended follow-ups (not done here, to avoid colliding with in-flight work)
1. Collapse the three VIRAC2locked builders to one; route their solvers through
   `measure_offset`. (touches brick-jwst-2221 — coordinate with PR #34.)
2. Factor the duplicated realign block in PipelineRerunNIRCAM-LONG into one helper.
3. Call `validate_offsets_table()` at the end of the surviving builder and as a
   warning in `fix_alignment`.
4. Archive deprecated `offsets/*` tables; resolve the `_VIRAC2_average` reader.
