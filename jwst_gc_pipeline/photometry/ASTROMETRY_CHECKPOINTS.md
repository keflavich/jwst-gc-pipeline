# Stage astrometry checkpoints — visit-consensus failsafe ladder

**Audience:** anyone touching cataloging, alignment, or the release gate.
**Modules:** `visit_consensus.py`, `astrometry_checkpoint.py`,
`astrometry_offsets.py` (`local_residual_map`), hooks in `cataloging.py`
(`_run_astrometry_stage_checkpoint`, `_run_crossfilter_astrom_checkpoint`),
CLI `scripts/reduction/run_astrometry_checkpoint.py`.

## Why

Astrometry errors at the 17″, 4″, 2″, 150 mas, 100 mas, and 50 mas level have
repeatedly propagated through the full pipeline because the alignment was
measured ONCE (coarsely, against a merged first-pass image — "im0") and carried
forward unchecked. Cataloging itself produces far better information.
The target accuracy is ~1 mas (limited by the VIRAC2 reference density).  Every
stage must therefore re-verify, and any disagreement must either be corrected at
its single authoring point (the offsets table + `fix_alignment`) or stop the
pipeline.

## The ladder

| stage (merge token) | what runs | shift found ⇒ |
|---|---|---|
| **m2** (after the m12 merge — first per-frame catalogs) | per-(visit, filter) consensus; every exposure re-measured vs the consensus (tol **2 mas**); consensus tied to VIRAC2/Gaia with the multi-check ladder | **CORRECT**: offsets table updated (provenance columns, validated, backed up), im0 `_i2d` mosaics stale-tagged `*_im0_badastrom.fits`, run STOPS (`AstrometryCorrectionRequiredError`) — the crf frames must be regenerated before any further cataloging |
| **m3, m4, m5, m6** | same measurement | **RED FLAG**: the solution is frozen after m2; positions come from the same crf GWCS, so a shift here is a real defect (centroiding systematics, seed drag, stale frame). `AstrometryRegressionError`, blocking |
| **m7 cross-band merge** | cross-filter agreement: anchor = filter nearest VIRAC2 Ks (2.149 µm); every filter vs anchor < **5 mas** bulk; matched-pair local residual map, no significant **2″** cell > **15 mas** (error bars mandatory) | `CrossFilterAstrometryError`, blocking, before the merge pools positions |

### What the frozen (m3–m6) per-exposure gate compares against

The gate is a **movement** check, not an absolute-magnitude one: an exposure fails
only when its vs-consensus offset moved since the m2 freeze by more than
`STAGE_STABILITY_TOL_MAS`.  An exposure with no m2 baseline is judged by *why* it
has none:

| m2 state | frozen-stage verdict |
|---|---|
| recorded in m2's `exposures` | delta vs that baseline; > tol ⇒ `AstrometryRegressionError` |
| recorded in m2's `consensus.skipped` (too few reliable stars — m2 found and reported the defect) | **UNVERIFIED**, not a failure. It never had a frozen solution, so this is its first measurement and cannot be a movement. `all_verified` goes false — see the caveat below |

> ⚠ **`all_verified` is not an enforced gate.** It is written into the checkpoint
> JSON and has **no non-test reader**; `stage_release.py` never opens
> `astrometry_checkpoints/`. Item 0b of `RELEASE_DEPLOYMENT_CHECKLIST.md` asks a
> human to check it. So routing something to `unverified` rather than `failures`
> converts an automatic `AstrometryRegressionError` into a manual checklist item,
> which is a real reduction in enforcement — worth knowing when reading the tables
> above. Automating the walk is its own change: the naive rule (`passed is not
> True or all_verified is not True` over every `*_latest.json`) refuses 12 of 14
> fields today, so the triage of those records is the actual deliverable, and the
> gate additionally needs a run token (records carry only `date`, and brick holds
> 89 `checkpoint_m2_*` files), a scope rule for stale `_latest` records from
> abandoned runs, and an assertion that each record's tolerances match the module
> defaults — w51's 60 records were written at `stage_stability_tol_mas: 20.0`
> against a module value of 2.0.
| absent for no recorded reason (new or renamed frame after the freeze) | `AstrometryRegressionError` — fail closed |

The middle row exists because of arches F212N (2026-08-02): a snowball storm in
exposure 4 (JUMP_DET 1.2 % → 7.6 %, 261 blobs > 100 px vs 9) cut its source count
~31 % on all eight detectors, m2 skipped all eight and said so in
`consensus.skipped`, and m3 then read them 12–18 mas off the consensus and killed
the m4–m8 chain over a defect m2 had already handled.  The exposure's own data
quality is the thing to investigate; a frozen-solution regression is not what
happened.

The **consensus→reference** tie is gated the same way, with the same distinction:

| m2 state | frozen-stage verdict |
|---|---|
| tie applied (`apply_ok: true`) | delta vs the reported bulk; > tol ⇒ `AstrometryRegressionError` |
| tie measured but **refused** (`apply_ok: false` — no coherent dense peak, gross sparse-Gaia split, failed per-tile / same-star gate), and the later stage lands **within** tol of it | **STABLE**. A refused tie is still a *measurement*: two readings of the same quantity agreeing is evidence the solution did not move. `apply_ok: false` says the *absolute* tie is uncertified, not that the consensus was free to move — so the pass is kept and the message notes the tie remains uncertified. (sgra F212N: m2 refused 48.49 mas, m3 reads 48.09 — a 0.41 mas delta) |
| tie measured but **refused**, and the later stage lands **beyond** tol | **UNVERIFIED**. It moved, but away from something that was never applied, so this is not a frozen-solution regression. The delta and m2's value are both named in the message; the field's *absolute* tie is the thing to investigate |
| no m2 record at all | `AstrometryRegressionError` — fail closed |

w51 F140M (2026-08-02) is the case: m2 measured a **7827 mas** consensus→reference
offset, judged it untrustworthy (`per_tile clean: false`, `swept: true`, a
window-limited histogram peak), refused to apply it, and recorded it in
`unverified`.  m3 then measured a clean **32 mas** same-star tie
(`apply_ok: true`, `swept: false`) and raised
`consensus->reference MOVED 7794.98 mas since the m2 freeze` — the field was
blocked because the measurement got *better*.  Note the failure only fires in
that direction: w51 F162M and F182M, whose m3 ties were *also* refused, passed,
because a refused m3 tie never reaches the baseline comparison at all.

Stage-name mapping: the user-facing plan's "m1 pass" = the repo's m12 phase
(iter1+iter2); its merge is labeled **m2** — that is the correcting
checkpoint.  "m2..m5" of the plan = merge tokens m3..m6 here.  "m6
cross-filter" = the m7 cross-band merge.

## The per-filter JWST consensus catalog

The consensus is MEASURED per `(visit, filter)` -- detecting that one exposure
is misaligned means comparing it against its own visit's other exposures -- and
then POOLED across visits into one catalog per filter, written at m2 as
`catalogs/<filter>_consensus.fits`.  That catalog is what the rest of the
pipeline ties to; see [`../../docs/JWST_CONSENSUS_CATALOG.md`](../../docs/JWST_CONSENSUS_CATALOG.md).

## The consensus measurement (`visit_consensus.py`)

1. Reliable-star cut per exposure catalog (qfit ≤ 0.1, S/N ≥ 10, excluding
   replaced-saturated fits).
2. Anchor = exposure with the most reliable stars; every exposure's offset to
   the anchor measured with `measure_offset` (histogram + **sweep**, so a 20″
   shifted exposure is found).
3. Exposures shifted into the anchor frame, stars associated
   (`search_around_sky` nearest pair — unambiguous only BECAUSE the relative
   offsets were removed first), consensus = per-star median over
   ≥ 2 exposures.
4. Consensus frame re-centred by the **median** of the per-exposure offsets
   (the median limits the pull of any one bad exposure).
5. Every exposure re-measured against the consensus; misaligned = off > 2 mas
   AND significant vs the peak error bars.

## The reference tie (`measure_reference_tie`) — five checks

A and B are independent measurements; C compares them, and D and E test the
same peak for stability.  Agreement among all five is therefore weaker evidence
than five independent checks would be.

* A: `measure_offset` vs the full (dense **VIRAC2**) refcat, sweep on — **this is
  the reference tie**;
* B: vs the sparse Gaia-only subset — a **diagnostic** cross-check;
* C: `agree_across_references` (A vs B).  In the GC **Gaia is the frame, never the
  reference catalog, and is too sparse to BLOCK** (memory: `gc-gaia-frame-not-catalog`;
  Gaia↔VIRAC2 agree ~2.3 mas over the Brick, so a fine ~5–10 mas split is a JWST-side
  population effect).  Two tolerances:
  * **fine** (`REFERENCE_AGREE_TOL_MAS`, 5 mas) — recorded as `cross_reference.agree`,
    **diagnostic only**;
  * **gross** (`REFERENCE_CROSSCHECK_GROSS_MAS`, ~100 mas) — `cross_reference_gross_ok`;
    the only cross-check that gates `apply_ok`, catching a spurious/window-limited
    VIRAC peak (brick-1182 v001 ~700 mas tell);
* D: per-tile map (`measure_offset_grid`) must be clean;
* E (bands overlapping VIRAC2, 1.0–2.5 µm): flux-cut source-by-source residual
  (both catalogs bright-cut until the estimated spacing ≥ 3× the match radius,
  then `local_residual_map` — which itself REFUSES to run without a verified
  small global tie).

A correction is applied **only** when A is coherent AND the gross cross-check
passes AND D is clean (`apply_ok`).  Anything else is recorded as
*could-not-verify* — loud and audited.

## Corrections & provenance

* The offsets table is the ONLY authoring channel (see
  `../reduction/ASTROMETRY_WCS_CORRECTION_FLOW.md`; which table a field reads is
  declared in `../reduction/alignment_config.py` and resolved by
  `unified_alignment.resolve_shift`).
  `update_offsets_table` converts on-sky mas → the table's Δα-coordinate
  convention, refuses corrections that match no row, refuses a per-exposure
  correction against a per-visit (module-locked) table, validates the result
  with the collapsed-visit guard (which raises on the brick-1182 signature),
  keeps a timestamped backup, and stamps provenance
  columns (`prov_stage`, `prov_date`, `prov_dra_added_mas`,
  `prov_ddec_added_mas`, `prov_source`).
* **`Vgroup` is part of a per-exposure row's identity.** A visit can dither
  across several visit groups (physically disjoint sky tiles) and the exposure
  number RESTARTS in each, so `(visit, filter, exposure, module)` names two
  different pointings — cloudc has 2 groups in every filter, gc2211 has 6,
  sgrb2 F187N has 2. Tables built by `build_virac2_offsets` and seeded by
  `seed_offsets_table_from_consensus` carry the column; `update_offsets_table`,
  `lookup_consensus_offset` and `fix_alignment` narrow on it
  (`same_vgroup` — a CSV round-trip makes astropy read `"06201"` back as
  `6201`).
  Two rules keep an OLDER table safe:
  * an EMPTY `Vgroup` cell means "group unknown", so such a
    row (written before the column existed, or preserved by the builder's
    field-safe merge) keeps applying exactly as it did — see
    `vgroup_row_matches`. Dropping it would silently discard an accumulated
    correction;
  * the consensus upsert MIGRATES a pre-`Vgroup` row: it adopts the row and
    backfills the group. If two groups claim
    the same legacy row, that row blended two pointings and the upsert REFUSES.
* **Magnitude ceilings on what may be written** (`_assert_correction_magnitudes`).
  A correction is bounded by its KIND:

  | correction kind | constant | default | override |
  |---|---|---|---|
  | per-exposure jitter (has an `exposure` or a `module`) | `MAX_CORRECTION_ARCSEC` | **0.5″** | `ASTROM_MAX_CORRECTION_ARCSEC` |
  | per-visit BULK tie (`exposure is None` **and** `module is None`) | `MAX_BULK_CORRECTION_ARCSEC` | **60″** | `ASTROM_MAX_BULK_CORRECTION_ARCSEC` |

  Jitter is mas-scale by construction, so 0.5″ is already generous — that is the
  gate cloudef needed (its +102″ runaway was written to a per-EXPOSURE row). The
  bulk tie is deliberately loose because a wrong-guide-star visit really is
  arcseconds off (brick-1182 visit-001 ~17–20″) and correcting it is the job; 60″
  is `measure_offset`'s sweep ceiling, so any larger value lies outside what a
  swept peak can produce. A non-positive or unparseable override raises.
* **Cumulative drift bound.** The per-correction ceiling bounds one call at a
  time, so creep accumulates across successive calls (five legal 0.4″ corrections
  = 2″ of silent drift; cloudef reached 105″ that way). Because
  `prov_dra/ddec_added_mas` accumulate, the write
  path also rejects any ROW whose total accumulated correction exceeds the **bulk**
  limit. ⚠ That bounds accumulation at 60″, so a table can still reach tens of
  arcseconds of `prov_*_added_mas` inside the bound. **The diagnostic for a
  poisoned table is `prov_*_added_mas`, not the total `|offset|`** — an m2 visit-consensus correction is
  mas-scale by construction, so arcsecond-scale `prov_*_added_mas` is a category
  error, while a large *total* `|offset|` can be perfectly correct (brick-1182's
  released table is median 12.1″ with 68/7.6 mas of `prov_*` additions).
  ⚠ **Blind spot:** `update_offsets_table` zero-fills the `prov_*` columns when
  they are absent, so a rebuilt or legacy table restarts the accumulator at 0 —
  and cloudef's live table, the field whose +102″ runaway motivates this whole
  paragraph, has **no `prov_*` columns at all**. On such a table both the drift
  bound and the `prov_*` diagnostic are blind.
* **Module-granularity refusal** (`_assert_module_granularity`). Corrections are
  keyed `(visit, exposure, module)`, but the apply loop skips module narrowing on a
  table with no `Module` column — so every detector's correction for one exposure
  lands on the SAME row, additively (8 × 0.4″ → +3.2″, each part legal). Such a
  correction set is refused. It protects the tables with **no `Module` column**:
  brick's two (1182, 2221), gc2211's, sgra's and **sgrb2's**.
* **One-correction-per-row refusal** (`_assert_one_correction_per_row`). Having a
  `Module` column is NOT the same as having it at the corrections' granularity,
  and the guard above returns early whenever the column merely exists. cloudc's,
  sgrc's, cloudef's and quintuplet's tables hold module **families**
  (`nrca`/`nrcb`/`nrcalong`/`nrcblong`) while the consensus emits one correction
  per **detector** (`nrca1`…), so four corrections landed on one row and were
  summed — sgrc's table ran 185.7 → 525.7 → 1678.5 mas over three re-tie
  iterations this way (2026-07-30/08-01). This guard checks the rows the apply
  loop would actually touch, so it sees a column present at the wrong
  granularity. Whole-visit bulk ties (`exposure=None, module=None`) are exempt:
  they are broad by design and compose with each exposure's jitter, exactly as
  `lookup_consensus_offset` sums a BULK row and a jitter row.
* **Pooling** (`pool_corrections_to_table_granularity`) is the remedy the refusal
  names — the m2 checkpoint applies it automatically on the `locked` channel
  before the actionability floor, and `update_offsets_table(..., pool=True)` /
  `--pool` on `apply_m2_checkpoint_corrections.py` and
  `run_astrometry_checkpoint.py` expose it to the recovery tooling. It takes the
  **median** of the corrections sharing a row, because a family row can only
  express the module-common shift; the per-detector spread is a
  distortion/DVA-class systematic the row has no freedom to remove. Pooling
  before the floor is what makes the loop converge — residuals that largely
  cancel pool to a sub-floor shift and the checkpoint PASSES instead of writing
  their sum. It is deliberately narrow and **refuses** rather than guessing:
  across module families (so a `Module`-less table still gets the actionable
  "rebuild `--per-module`" refusal instead of a silent A/B average), a module
  contributing twice to one row (two vgroups against a `Vgroup`-less table), a
  group whose members disagree by more than `ASTROM_MAX_POOL_SPREAD_MAS`
  (50 mas), and any member over the magnitude ceiling — which is checked on the
  **members**, since `median ≤ max` means pooling would otherwise average a
  blown-up detector out of existence. What was collapsed is written to the
  checkpoint record under `pooling` (the correction's `source` is truncated to
  64 characters, shorter than a real member list).
* Stale im0 mosaics are RENAMED `*_i2d_im0_badastrom.fits` and kept intact (+ a
  `.why.json` sidecar and a ledger in `astrometry_checkpoints/`).
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
| `ASTROM_MAX_CORRECTION_ARCSEC=<f>` | raise/lower the per-exposure ceiling (default 0.5″) |
| `ASTROM_MAX_BULK_CORRECTION_ARCSEC=<f>` | raise/lower the per-visit bulk ceiling **and** the cumulative-drift bound (default 60″) |
| `ASTROM_ALLOW_MISSING_PERFRAME=1` | demote the missing-per-frame-catalog stop (`cataloging.py`) |
| `CATALOG_ALLOW_UNVETTED_FALLBACK=1` | allow the unvetted-catalog fallback |
| `OFFSETS_TABLE_COLLAPSE_RAISE=1` | make the collapsed-visit guard raise instead of warn (`reduction/validate_offsets_table.py`) |
| `FORCE_REALIGN_ON_DISAGREE=1` | hard-stop when a frame's baked `RAOFFSET` disagrees with the current table (`reduction/unified_alignment.py`) |

Set an override to record a decision you have already justified by other
means. A red gate stays red: the override is the record, not the justification (same policy as
`ALLOW_REGISTRATION_FAIL`).

## Relationship to the other astrometry shields

* `measure_offset` / `measure_offset_grid` — the sanctioned measurement
  (CLAUDE.md rule #1); everything here builds on it.  `measure_offset` also
  returns `dra_err/ddec_err/n_peak` (MAD-based peak standard errors) and
  subsamples internally above `MAX_PAIRS_PER_WINDOW` so dense-catalog sweeps
  are memory-safe.
* `local_residual_map` — fine-scale (2″) matched-pair residual mapping,
  precondition-gated on a verified small global tie (it raises
  `GlobalTieNotVerifiedError` otherwise, which is what keeps it from becoming
  an ad-hoc dense-NN shortcut).
* `registration_failsafes.py` / the inter-frame overlap gate (PR #85) —
  release-time product checks.  The checkpoints complement them: they run
  DURING cataloging, at mas-level tolerances, and stop the error at its
  source.
