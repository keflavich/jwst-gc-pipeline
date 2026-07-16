# Stage astrometry checkpoints — visit-consensus failsafe ladder

**Audience:** anyone touching cataloging, alignment, or the release gate.
**Modules:** `visit_consensus.py`, `astrometry_checkpoint.py`,
`astrometry_offsets.py` (`local_residual_map`), hooks in `cataloging.py`
(`_run_astrometry_stage_checkpoint`, `_run_crossfilter_astrom_checkpoint`),
CLI `scripts/reduction/run_astrometry_checkpoint.py`.

## Why

Astrometry errors at the 17″, 4″, 2″, 150 mas, 100 mas, and 50 mas level have
repeatedly propagated through the full pipeline because the alignment was
measured ONCE (coarsely, against a merged first-pass image — "im0") and never
re-verified against the much better information cataloging itself produces.
The target accuracy is ~1 mas (limited by the VIRAC2 reference density).  A
single measurement is never sufficient; every stage must re-verify, and any
disagreement must either be corrected at its single authoring point (the
offsets table + `fix_alignment`) or stop the pipeline.

## The ladder

| stage (merge token) | what runs | shift found ⇒ |
|---|---|---|
| **m2** (after the m12 merge — first per-frame catalogs) | per-(visit, filter) consensus; every exposure re-measured vs the consensus (tol **2 mas**); consensus tied to VIRAC2/Gaia with the multi-check ladder | **CORRECT**: offsets table updated (provenance columns, validated, backed up), im0 `_i2d` mosaics stale-tagged `*_im0_badastrom.fits`, run STOPS (`AstrometryCorrectionRequiredError`) — the crf frames must be regenerated before any further cataloging |
| **m3, m4, m5, m6** | same measurement | **RED FLAG**: the solution is frozen after m2; positions come from the same crf GWCS, so a shift here is a real defect (centroiding systematics, seed drag, stale frame). `AstrometryRegressionError`, blocking |
| **m7 cross-band merge** | cross-filter agreement: anchor = filter nearest VIRAC2 Ks (2.149 µm); every filter vs anchor < **5 mas** bulk; matched-pair local residual map, no significant **2″** cell > **15 mas** (error bars mandatory — one star is not a measurement) | `CrossFilterAstrometryError`, blocking, before the merge pools positions |

Stage-name mapping: the user-facing plan's "m1 pass" = the repo's m12 phase
(iter1+iter2); its merge is labeled **m2** — that is the correcting
checkpoint.  "m2..m5" of the plan = merge tokens m3..m6 here.  "m6
cross-filter" = the m7 cross-band merge.

## The consensus measurement (`visit_consensus.py`)

1. Reliable-star cut per exposure catalog (qfit ≤ 0.1, S/N ≥ 10, not a
   replaced-saturated fit).
2. Anchor = exposure with the most reliable stars; every exposure's offset to
   the anchor measured with `measure_offset` (histogram + **sweep** — a 20″
   shifted exposure is found, never absorbed).
3. Exposures shifted into the anchor frame, stars associated
   (`search_around_sky` nearest pair — unambiguous only BECAUSE the relative
   offsets were removed first), consensus = per-star median over
   ≥ 2 exposures.
4. Consensus frame re-centred by the **median** of the per-exposure offsets
   (one bad exposure cannot drag the frame toward itself).
5. Every exposure re-measured against the consensus; misaligned = off > 2 mas
   AND significant vs the peak error bars.

## The reference tie (`measure_reference_tie`) — no single number signs off

* A: `measure_offset` vs the full (dense **VIRAC2**) refcat, sweep on — **this is
  the reference tie**;
* B: vs the sparse Gaia-only subset — a **diagnostic** cross-check, not the catalog;
* C: `agree_across_references` (A vs B).  In the GC **Gaia is the frame, never the
  reference catalog, and is too sparse to BLOCK** (memory: `gc-gaia-frame-not-catalog`;
  Gaia↔VIRAC2 agree ~2.3 mas over the Brick, so a fine ~5–10 mas split is a JWST-side
  population effect, not a catalog conflict).  Two tolerances:
  * **fine** (`REFERENCE_AGREE_TOL_MAS`, 5 mas) — recorded as `cross_reference.agree`
    for diagnostics; **does NOT gate**;
  * **gross** (`REFERENCE_CROSSCHECK_GROSS_MAS`, ~100 mas) — `cross_reference_gross_ok`;
    the only cross-check that gates `apply_ok`, catching a spurious/window-limited
    VIRAC peak (brick-1182 v001 ~700 mas tell);
* D: per-tile map (`measure_offset_grid`) must be clean;
* E (bands overlapping VIRAC2, 1.0–2.5 µm): flux-cut source-by-source residual
  (both catalogs bright-cut until the estimated spacing ≥ 3× the match radius,
  then `local_residual_map` — which itself REFUSES to run without a verified
  small global tie).

A correction is applied **only** when A is coherent AND the gross cross-check
passes AND D is clean (`apply_ok`).  The fine sparse-Gaia agreement is reported but
does NOT block.  Anything else is recorded as *could-not-verify* — loud, audited,
never silently green.

## Corrections & provenance

* The offsets table is the ONLY authoring channel (see
  `../reduction/ASTROMETRY_WCS_CORRECTION_FLOW.md`).
  `update_offsets_table` converts on-sky mas → the table's Δα-coordinate
  convention, refuses corrections that match no row, refuses a per-exposure
  correction against a per-visit (module-locked) table, validates the result
  with the collapsed-visit guard (raising — the brick-1182 signature must
  never be written), keeps a timestamped backup, and stamps provenance
  columns (`prov_stage`, `prov_date`, `prov_dra_added_mas`,
  `prov_ddec_added_mas`, `prov_source`).
* Stale im0 mosaics are RENAMED `*_i2d_im0_badastrom.fits` (+ a `.why.json`
  sidecar and a ledger in `astrometry_checkpoints/`), never deleted, never
  edited.
* `fix_alignment` stamps `APROVST/APROVMT/APROVDR/APROVDD/APROVRF/APROVTB/
  APROVDT` header cards when it (re-)applies a table, so every aligned frame
  carries the provenance of its astrometric fix next to `RAOFFSET/DEOFFSET`.
  Changing a baked `RAOFFSET` is done ONLY by regenerating the working copy
  from `_cal` — never by header-poking (no-double-correction rule).

## Records

Every checkpoint writes JSON under `{basepath}/astrometry_checkpoints/`
(timestamped + `_latest`), including all per-exposure offsets, error bars,
contrasts, windows, `swept` flags, per-check reference results, corrections,
failures, and could-not-verify items.  The release gate can (and should)
audit the full ladder from these records.

## Environment switches

| var | effect |
|---|---|
| `ASTROM_CHECKPOINT=0` | disable all checkpoints (emergencies only) |
| `ASTROM_CHECKPOINT_WARN_ONLY=1` | demote blocking failures to loud warnings |
| `ASTROM_CHECKPOINT_APPLY=1` | at m2, auto-apply corrections to the offsets table + stale-tag im0 |
| `ASTROM_REFCAT=<path>` | reference catalog override (default: `{basepath}/catalogs/gaia_virac2_refcat*.fits`) |
| `ALLOW_LATE_STAGE_ASTROM_SHIFT=1` | override the m3+ frozen-solution gate |
| `ALLOW_CROSSFILTER_ASTROM_FAIL=1` | override the cross-filter gate |

Overrides exist for deliberate, justified use — never to make a red gate
green (same policy as `ALLOW_REGISTRATION_FAIL`).

## Relationship to the other astrometry shields

* `measure_offset` / `measure_offset_grid` — the sanctioned measurement
  (CLAUDE.md rule #1); everything here builds on it.  `measure_offset` now
  also returns `dra_err/ddec_err/n_peak` (MAD-based peak standard errors) and
  subsamples internally above `MAX_PAIRS_PER_WINDOW` so dense-catalog sweeps
  are memory-safe.
* `local_residual_map` — fine-scale (2″) matched-pair residual mapping,
  precondition-gated on a verified small global tie (it raises
  `GlobalTieNotVerifiedError` otherwise, so it can never become an ad-hoc
  dense-NN shortcut).
* `registration_failsafes.py` / the inter-frame overlap gate (PR #85) —
  release-time product checks.  The checkpoints complement them: they run
  DURING cataloging, at mas-level tolerances, and can stop the error at its
  source instead of at the release gate.
