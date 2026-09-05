# Astrometry / offsets code — redundancy audit (2026-07-11)

Audit of the astrometric-offset code across `jwst_gc_pipeline` and
`brick-jwst-2221/brick2221/analysis`, focused on **redundancies that cause
RECURRENT errors** (the same computation implemented in several places that
drift, so a fix in one is silently undone by another). Written after the
brick-1182 visit-001 corruption (a builder collapsed distinct visits onto one
visit's offset). All findings below were verified by inspection.

Companion docs: `ASTROMETRY_WCS_CORRECTION_FLOW.md` (the correct flow + the ⛔
dense-NN-median rule). The collapse-detection safeguard now lives in
`reduction/validate_offsets_table.py` (`flag_collapsed_visits` /
`assert_offsets_table_sane`, called from the locked-table reader
`unified_alignment._shift_from_locked`, i.e. on every `fix_alignment` that reads a
locked table); this doc records the *structural* redundancies
that let the collapse happen and recur.

## Ranked redundancies

### R1 (CRITICAL) — THREE builders write the same `Offsets_JWST_Brick<prop>_VIRAC2locked.csv`
`brick2221/analysis/build_virac2_locked_perexp.py` (per-visit),
`brick2221/analysis/relock_exposures.py` (per-exposure), and
`brick2221/analysis/build_calframe_locked_offsets.py` — all three live in
**brick-jwst-2221** — all write the **same production table**
that `fix_alignment` consumes. Whichever ran last wins; a fix in one builder is
silently overwritten by another on the next run. This IS the recurrence engine:
the brick-1182 collapse persisted across rebuilds because the fixed and unfixed
builders competed for one file.
- **Fix:** one authoritative builder (per-exposure preferred — the 2026-06-20
  jitter measurement favors it). Deprecate/delete the others, or give each a
  distinct output path and have `fix_alignment` select by config. Add a
  `validate_offsets_table()` call at the end of the surviving builder.

### R2 (CRITICAL) — 3× duplicate `robust_shift` / `robust_offset` with parameter drift
`brick2221/analysis/relock_exposures.py::robust_shift`,
`brick2221/analysis/build_calframe_locked_offsets.py::robust_shift` and
`brick2221/analysis/lock_exposures.py::robust_offset` are near-identical clipped-median solvers with
**different constants** (search 0.3" vs 0.5"; clip 60 vs 80). Tuning one leaves
the others as they were; two can produce conflicting offsets for the same data.
- **Fix:** one shared solver. Prefer routing all of them through
  `jwst_gc_pipeline.photometry.astrometry_offsets.measure_offset` (histogram
  stacking + window sweep + `NoCoherentTieError`), which is crowding-proof and
  bridges the ~22" per-visit errors that dwarf a 0.3" NN median's reach.
  (relock_exposures gained a histogram bridge in PR #34; the other two still
  need one.)

### R3 (HIGH) — two near-identical realign blocks in `PipelineRerunNIRCAM-LONG.py` — ✅ RESOLVED 2026-07-11
Two `realign_to_catalog(reftbl['skycoord'], ...)` call blocks (~L962 and ~L1123),
each preceded by a VVV block, differing only slightly. A fix applied to one (e.g.
the dense-NN skip) must be duplicated in the other or one path regresses.
- **Resolution:** the entire post-Image3 mosaic realign was **retired**. On the
  dense-refcat GC fields it was a guarded no-op (a byte-identical copy of `_i2d`),
  and `_i2d` itself is the release deliverable. Both call blocks are gone;
  `realign_to_catalog` / `realign_to_vvv` are
  now `NotImplementedError` stubs. The astrometric tie has one authoring point
  (per-exposure `fix_alignment`). See `ASTROMETRY_WCS_CORRECTION_FLOW.md`.

### R4 (MEDIUM) — convention chaos across solvers (cosδ vs no-cosδ, mas vs arcsec)
`measure_offset` = on-sky (×cosδ), mas. `coarse_xcorr`/`coord_shift` = coordinate
(no-cosδ), arcsec. `robust_shift` = on-sky, mas (converted to arcsec at write via
`/1000`). `coord_shift` clips on on-sky distance but returns no-cosδ. Each port
risks a dropped/doubled cosδ (~12% at the GC) or a 1000× unit slip.
- **Fix:** standardize on one convention at the API boundary (astrometry_offsets
  is on-sky mas throughout); annotate table columns with units; centralize the
  coordinate↔on-sky conversion in one place.
- **NOTE:** relock_exposures converts at write — line 114 divides by 1000
  (`dra=sr/1000.0  # arcsec`), so its arcsec column is correct.

### R5 (MEDIUM) — orphaned tables + a reader without a source — still open 2026-07-30
`offsets/` accumulates deprecated tables (`*_F200ref*`, `*_F405ref*`, `*_VVV*`)
no current builder writes; a script that names one by mistake reads a stale
frame silently. Separately, the locked-table reader falls back to
`*_VIRAC2_average.csv` when no `VIRAC2locked` table exists
(`unified_alignment._shift_from_locked`, `use_average=True` by default), and that
average table has no builder in this repo — a **reader whose builder is
missing**.
- **Fix:** archive/delete deprecated tables (keep VIRAC2locked); either complete
  the average builder or remove the average-table path from `fix_alignment`.

## What is already well-covered (do NOT re-add)
PRs #65/#66/#68 shipped: the dense-NN-median guard (`assert_sparse_reference_for_nn_median`,
raises on dense refs), `astrometry_offsets.measure_offset` (window sweep +
`min_contrast`), `measure_offset_grid` (per-tile — catches half-mosaic untying),
`agree_across_references` (VIRAC vs Gaia), a RAOFFSET-vs-table disagreement guard
in `fix_alignment` (the `realign_to_catalog` skip-on-dense guard is moot — realign
retired 2026-07-11), and tests
`test_dense_nn_median_guard`, `test_astrometry_offsets_sweep`,
`test_no_adhoc_nn_median_astrometry`, `test_registration_gate`.

## Collapse safeguard (already shipped, on main)
`reduction/validate_offsets_table.py` (`flag_collapsed_visits` /
`assert_offsets_table_sane` + `test_validate_offsets_table.py`) flags a COLLAPSED
offsets table (distinct visits of a filter sharing an offset to within ~20 mas) and
runs inside `unified_alignment._shift_from_locked`, on the `fix_alignment` path, where
it RAISES `CollapsedOffsetsTableError` (PR #770 — that call site passes
`raise_on_issue=True`; it warned by default until then, and the warning is what let
brick-1182 v001 be baked in). `raise_on_issue=True` also promotes
`flag_broadcast_provenance` to `BroadcastProvenanceError`; the as-built/as-corrected
divergence keeps its own switch (`raise_on_diverged` / `OFFSETS_TABLE_DIVERGENCE_RAISE=1`)
and still warns, because it costs the audit trail rather than the applied shift.
`OFFSETS_TABLE_COLLAPSE_RAISE=1` no longer changes anything on this path or on the
`astrometry_checkpoint.update_offsets_table` write path — both already pass
`raise_on_issue=True` — and remains only for a caller that does not.
This audit documents the redundancies below that make the collapse possible in
the first place.

## Recommended follow-ups (deferred, to avoid colliding with in-flight work)
1. Collapse the three VIRAC2locked builders to one; route their solvers through
   `measure_offset`. (touches brick-jwst-2221 — coordinate with PR #34.) Call
   `assert_offsets_table_sane(..., raise_on_issue=True)` at the end of the surviving
   builder so any collapse is caught before it reaches disk.
2. ~~Factor the duplicated realign block in PipelineRerunNIRCAM-LONG into one helper.~~
   ✅ DONE 2026-07-11 — realign retired entirely (see R3); both blocks removed.
3. Add an `|offset| > 60"` insane-magnitude check to `validate_offsets_table`
   (catches mas/arcsec unit slips) — **still open**: `validate_offsets_table.py`
   carries only the collapse check. Note the *write* path already bounds
   corrections (`MAX_CORRECTION_ARCSEC` 0.5" per-exposure /
   `MAX_BULK_CORRECTION_ARCSEC` 60" bulk, plus a cumulative-drift bound — see
   `../photometry/ASTROMETRY_CHECKPOINTS.md`); the gap is a *read*-side
   magnitude check on a table the pipeline is about to apply.
4. Archive deprecated `offsets/*` tables; resolve the `_VIRAC2_average` reader.
