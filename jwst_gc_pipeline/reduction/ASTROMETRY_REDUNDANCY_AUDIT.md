# Astrometry / offsets code — redundancy audit (2026-07-11)

Audit of the astrometric-offset code across `jwst_gc_pipeline` and
`brick-jwst-2221/brick2221/analysis`, focused on **redundancies that cause
RECURRENT errors** (the same computation implemented in several places that
drift, so a fix in one is silently undone by another). Written after the
brick-1182 visit-001 corruption (a builder collapsed distinct visits onto one
visit's offset). All findings below were verified by inspection; false alarms
are called out.

Companion docs: `ASTROMETRY_WCS_CORRECTION_FLOW.md` (the correct flow + the ⛔
dense-NN-median rule). The collapse-detection safeguard now lives in
`reduction/validate_offsets_table.py` (`flag_collapsed_visits` /
`assert_offsets_table_sane`, called from the locked-table reader
`unified_alignment._shift_from_locked`, i.e. on every `fix_alignment` that reads a
locked table); this doc records the *structural* redundancies
that let the collapse happen and recur.

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

### R3 (HIGH) — two near-identical realign blocks in `PipelineRerunNIRCAM-LONG.py` — ✅ RESOLVED 2026-07-11
Two `realign_to_catalog(reftbl['skycoord'], ...)` call blocks (~L962 and ~L1123),
each preceded by a VVV block, differing only slightly. A fix applied to one (e.g.
the dense-NN skip) must be duplicated in the other or one path regresses.
- **Resolution:** rather than factor the duplicate into a helper, the entire
  post-Image3 mosaic realign was **retired**. On the dense-refcat GC fields it was a
  guarded no-op (a byte-identical copy of `_i2d`) and was never the release
  deliverable. Both call blocks are gone; `realign_to_catalog` / `realign_to_vvv` are
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
- **NOTE — false alarm:** an earlier pass flagged relock_exposures returning mas
  into an arcsec column as a live 1000× bug. It is **not** — line 114 divides by
  1000 (`dra=sr/1000.0  # arcsec`). Verified. Left as documentation only.

### R5 (MEDIUM) — orphaned tables + a reader without a source — still open 2026-07-30
`offsets/` accumulates deprecated tables (`*_F200ref*`, `*_F405ref*`, `*_VVV*`)
no current builder writes; a script that names one by mistake reads a stale
frame silently. Separately, the locked-table reader falls back to
`*_VIRAC2_average.csv` when no `VIRAC2locked` table exists
(`unified_alignment._shift_from_locked`, `use_average=True` by default), but no
builder in this repo writes that file — a **reader whose builder is missing**
(verified still true 2026-07-30).
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
runs inside `unified_alignment._shift_from_locked`, on the `fix_alignment` path. It fires a warning
by default, or raises `CollapsedOffsetsTableError` with `OFFSETS_TABLE_COLLAPSE_RAISE=1`.
This audit does NOT re-add it; it documents the redundancies below that make the
collapse possible in the first place.

## Recommended follow-ups (not done here, to avoid colliding with in-flight work)
1. Collapse the three VIRAC2locked builders to one; route their solvers through
   `measure_offset`. (touches brick-jwst-2221 — coordinate with PR #34.) Call
   `assert_offsets_table_sane(..., raise_on_issue=True)` at the end of the surviving
   builder so a collapse can never be written to disk.
2. ~~Factor the duplicated realign block in PipelineRerunNIRCAM-LONG into one helper.~~
   ✅ DONE 2026-07-11 — realign retired entirely (see R3); both blocks removed.
3. Add an `|offset| > 60"` insane-magnitude check to `validate_offsets_table`
   (catches mas/arcsec unit slips) — **still open**: `validate_offsets_table.py`
   carries only the collapse check. Note the *write* path already bounds
   corrections (`MAX_CORRECTION_ARCSEC` 0.5" per-exposure /
   `MAX_BULK_CORRECTION_ARCSEC` 60" bulk, plus a cumulative-drift bound — see
   `../photometry/ASTROMETRY_CHECKPOINTS.md`); what is missing is a *read*-side
   magnitude check on a table the pipeline is about to apply.
4. Archive deprecated `offsets/*` tables; resolve the `_VIRAC2_average` reader.
