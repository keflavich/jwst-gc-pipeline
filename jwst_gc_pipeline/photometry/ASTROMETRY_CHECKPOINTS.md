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
| **m7 cross-band merge** | cross-filter agreement: anchor = filter nearest VIRAC2 Ks (2.149 µm); every filter vs anchor < **5 mas** bulk; matched-pair local residual map, no significant **2″** cell > **15 mas** (error bars mandatory). Plus a recorded, non-gating **residual-field** measurement (below) | `CrossFilterAstrometryError`, blocking, before the merge pools positions |

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

## WHERE a frozen-stage failure stops the pipeline

`ASTROM_CHECKPOINT_ENFORCE` — `release` (default) or `stage`.

At `release`, a failure at m3–m6 or at the m7 cross-filter check is **measured,
recorded with `passed: false`, and printed**, and the chain continues. The field
is then refused by

```bash
python scripts/release/check_astrometry_checkpoints.py --field <f>
```

which `stage_release.py` runs before staging. At `stage`, the failure raises
inside the stage that measured it, which is what it did before.

**Why the default moved.** m3 and later cannot change the astrometry — the
solution is frozen, which is what makes a shift there a defect — so the check is
a *measurement* wired up as a *control*. Raising inside the stage bought one
thing, not spending compute on a run that would be refused, and cost three:

* the chain is `afterok`, so one filter's raise discarded every other filter's
  finished stages. cloudef 002 spent ten re-tie iterations reaching m2 and lost
  all of it at m3 (2026-08-18);
* the products an investigator needs were never made, so every diagnosis began
  by re-running the chain to get them back;
* every frozen-stage failure diagnosed so far has been a *comparison artefact*
  rather than movement — the one-sided star restriction (#285), the
  full-set-vs-shared baseline (#430: sickle F335M read 2.23 mas of "movement"
  and 0.637 on the stars both stages carry), the refused-m2-tie inversion (w51
  F140M), the absolute-vs-delta per-exposure gate (brick F115W).

**Nothing is waived.** The stop moved; it did not disappear. The gate fails
closed on a record it cannot read (rc 2) and on a field with no records at all
(rc 3) — a field that never ran the checkpoint is unverified, not verified.
`ALLOW_LATE_STAGE_ASTROM_SHIFT` and `ALLOW_CROSSFILTER_ASTROM_FAIL` still work
as before and are still the only way to make a failure non-blocking.  Using
either is now RECORDED: the checkpoint record carries a `gate_override` block
naming the variable, whether it was set, and the justification read from
`<VAR>_REASON`.  Before that, a record walked past and a record that stopped
the chain were byte-identical (`passed: false` on both), so the only trace of a
waiver was one WARNING line in a SLURM log -- which is why brick's m5 F200W
failure could not be attributed two weeks later (issue #258).  The release gate
prints the block beside the failure, and says `not recorded` for records that
predate the field rather than calling them un-overridden.

**m2 is untouched.** It is the one stage where the astrometry can still change:
its response to a measured offset is to correct the offsets table, stale-tag the
mosaics, and have its caller stop the run for regeneration. That is a control
action no later gate can perform, so the deferral does not apply to it.

A typo in the variable enforces at the **stage** — anything that is not exactly
`release` is read as `stage`, so a misspelling costs a stopped chain rather than
a shipped misalignment.

The **consensus→reference** tie is gated the same way, with the same distinction:

| m2 state | frozen-stage verdict |
|---|---|
| tie applied (`apply_ok: true`) | delta vs the reported bulk; > tol ⇒ **re-measured on the stars the two consensi share** (below), and only a delta that survives that raises `AstrometryRegressionError` |
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

### The comparison is made on the stars both stages have

A delta over tolerance is **not** reported as movement until both ties have been
re-measured over the stars the two consensi share.  Issue #285 restricted the
later stage's consensus to m2's star *list*, but the number it is differenced
against was measured over m2's *full* set — so the stars a later stage cannot
re-detect drag the baseline and nothing else.  From the live sickle records
(2026-08-16), F335M m2 vs m5:

```
m2 over its full 2964 stars    (-0.013, +0.014) mas
m5 over its      2644 stars    (+0.457, +2.194) mas    raw delta 2.230
both over the shared 2642      (-0.013, +1.764) mas -> delta 0.637  PASS
```

Those are the numbers the on-disk records carry, measured before the *stage*
side was restricted as well. With both sides restricted the same two cases read
0.651 and 2.536 — same verdicts, slightly larger numbers, because the stage side
now drops the one or two stars m2 does not carry.

`STAGE_STABILITY_TOL_MAS` is unchanged at 2.0; what changes is which two
numbers it is applied to.  The re-measure runs **only** when the raw comparison
is over tolerance, so a passing stage never pays for it.

It is not a way to get a pass.  It **refuses**, leaving the raw comparison to
raise, when

* m2's pooled consensus catalog is missing or unreadable;
* either consensus holds fewer than `SURVIVOR_MIN_STARS` (50);
* the m2 consensus catalog pools more than one visit — its positions are
  visit-averaged while the stage measures one visit, and on brick F115W that
  substitution alone is 1.7–1.9 mas against a 2.0 mas budget;
* the shared set is below `max(50, SURVIVOR_MIN_FRACTION × max(n_m2, n_stage))`
  — two 90,000-star catalogs sharing 0.07% of their stars clear any absolute
  floor while saying nothing about each other;
* either re-measure returns a non-finite tie, sets `apply_ok: false`, or had to
  **sweep** its window — a measurement the estimator declined to sign cannot
  overturn a blocking failure;
* the two re-measures used different estimators (`bulk_source` `same-star` vs
  `histogram`), whose difference against a dense reference is several mas of
  method rather than a shift.

Matching is mutual (`_mutual_match_mask`) at `SURVIVOR_MATCH_TOL_MAS` = 150 mas,
mirroring `build_visit_consensus(restrict_radius=0.15")`, which is what produced
the stage consensus in the first place.

The record grows `visits[].symmetric_baseline`: both ties on the shared stars,
`delta_mas`, `raw_delta_mas`, the three counts, the floor, `bulk_source`, the
refusal `reason`, and `mag_split` — the median magnitude of the kept and dropped
stars. The intersection is a **biased sample** and the direction is not fixed:
sickle's F335M drop-outs are ~1.1 mag fainter than its survivors, its F187N
drop-outs ~1.9 mag brighter (1.86 as `mag_split` records it). It characterises
the **m2 side only**, so the stars the later stage carries and m2 does not are
not described. Displacement confined to the stars a later stage
drops is not visible to this comparison; that is a real reduction in coverage,
and `mag_split` is what makes the skew visible to whoever reads the pass.

## A VIRAC2-framed field stops when its reference catalog is missing

The checkpoints load `{basepath}/catalogs/gaia_virac2_refcat*.fits` (or
`ASTROM_REFCAT`).  Without it they still run, and what they check collapses to
internal consistency: the exposures agree WITH EACH OTHER, and nothing says
where that agreed frame sits on the sky.  A field can reduce, pass every check,
and ship arcseconds off — proposal 1939's sgra mosaics sat ~14.8″ from VIRAC2
with its offsets table unread.

So a field whose `ALIGNMENT_CONFIG` declares **VIRAC2** now raises
`MissingReferenceCatalogError` rather than printing and continuing.  The scope
is deliberate, and measured (2026-08-20): every VIRAC2-framed field carries a
refcat today except a newly-registered one, and no Gaia-framed field carries
one at all, so the gate stops the field that needs stopping and leaves m4, m92,
ngc6397 and w51 running as before.

Build the missing catalog with

```bash
python -m jwst_gc_pipeline.reduction.build_gaia_virac2_refcat_byquery \
    --base <basepath> --epoch <YYYY.Y> --ra <deg> --dec <deg> --radius <deg> \
    [--obs-token <NNN>]
```

`--obs-token` stamps `_o<NNN>` into the filename, which is what `pick_refcat`
matches on to give each observation its OWN catalog in a shared field
directory.  A field whose observations share one directory (the treasury tiles,
gc2211) needs it: an untokened catalog is handed to every observation alike,
and for pointings arcminutes apart that is the wrong sky — the way gc2211 o023
took a −9.28″ correction.

`ALLOW_CONSENSUS_ONLY_ASTROMETRY=1` runs without one, for a deliberate first
look at a field whose catalog has not been built.  It never lets a run borrow
another observation's catalog: `pick_refcat` refuses that separately, override
or not.

Absent `symmetric_baseline` means the re-measure never ran — the raw comparison
passed, or the stage is a correcting one, or there is no reference catalog, or
the offset is under `REFERENCE_APPLY_MIN_MAS`, or m2 refused its own tie.

The check removes a *baseline artefact*; it does not soften the gate. sickle
F187N m3 fails either way — 2.342 mas raw, 2.536 mas on the shared stars at
this head —
because there the drop-outs are brighter than the survivors and carry no
artefact to remove.

Stage-name mapping: the user-facing plan's "m1 pass" = the repo's m12 phase
(iter1+iter2); its merge is labeled **m2** — that is the correcting
checkpoint.  "m2..m5" of the plan = merge tokens m3..m6 here.  "m6
cross-filter" = the m7 cross-band merge.

## The cross-filter residual field (m7, measurement only)

Alongside its two gates the m7 checkpoint MEASURES the coherent,
position-dependent part of each filter-to-anchor residual
(`measure_residual_field`), stored as `filters[i]["field"]` in
`checkpoint_m7_crossfilter_*.json` and printed as one `ASTROM CROSSFILTER
FIELD:` line per filter. **Nothing gates on it**; the two tolerances above are
unchanged.

It exists because both gates are structurally blind to that term:

* `CROSSFILTER_TOL_MAS = 5.0` is on the **bulk**, which is ~0 for a field whose
  mean is zero by construction;
* `LOCAL_CELL_TOL_MAS = 15.0` at `LOCAL_CELL_SIZE_ARCSEC = 2.0` does not merely
  fail to reach significance — on a dense field the reliability cut leaves ~1.2
  stars per 2″ cell against `LOCAL_CELL_MIN_STARS = 10` (measured on brick
  F212N/F182M), so the map returns `n_cells = 0`. An injection sweep on Brick
  geometry never trips it at any amplitude up to 30 mas/arcmin.

  That silence used to score as a **pass**. It is now reported as
  **unverified**: `run_crossfilter_checkpoint` records `unverified` and
  `all_verified` (matching `run_visit_checkpoint`) and prints
  `ASTROM CHECKPOINT [m7-crossfilter] COULD NOT VERIFY:` per entry. Three
  situations produce one — an empty map, a map with fewer than
  `LOCAL_CELL_MIN_CELLS` (4) populated cells, and a bulk tie that skipped the
  local map entirely (swept, or ≥ 100 mas). None is a *failure*: measuring
  nothing is a coverage fact about the field, not evidence of a misalignment,
  and failing it would block every dense field on a cell size that cannot work
  there.

  **`all_verified` still has no non-test reader** — `stage_release.py` never
  opens `astrometry_checkpoints/` and `monitoring/scan.py` globs `checkpoint_m2_*`
  only — so today this buys a printed line and a JSON field a human may read.

New `tolerances` keys: `field_cell_arcsec` (45″) and `field_min_stars` (40) —
cells large enough to hold hundreds of stars, so the per-cell SEM falls far
below the signal.

Every amplitude in the `field` block is **per-component** (per axis), recorded
as `rms_convention`; the only 2-D vector magnitude is `max_cell_off_mas`. Keys:
`rms_mas`, `median_sem_mas`, `coherent_mas` (= `sqrt(rms² − <sem²>)`, the
number to quote), `rms_after_affine_mas`, `affine_absorbed_fraction` with its
`affine_absorbed_chance` (`6/(2n)`) and `affine_absorbed_adjusted`,
`gradient_mas_per_arcmin` (Frobenius norm of the fitted 2×2 Jacobian),
`n_pairs` / `matched_fraction` / `match_radius_mas`, and
`n_cells` / `n_cells_in_bbox` / `n_cells_dropped`.

Brick m7, 45″ cells, per-component rms: F212N/F187N 0.54, F405N/F466N 0.51,
F182M/F187N 0.98, F212N/F182M 1.40, F212N/F405N 2.50, F182M/F466N 2.53,
F212N/F200W 3.42, F182M/F115W 4.47 mas — 7–45× the per-cell SEM. This is the
field-wide astrometric floor; see issue #296 for what causes it and #299 for
the correction.


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
  columns (`prov_stage`, `prov_date`, `prov_dra_onsky_mas`,
  `prov_ddec_onsky_mas`, `prov_dec_deg`, `prov_source`).

  The `_onsky_` in those names is load-bearing. Right ascension has two
  quantities that are both angles and differ by cos(declination) — about 14% at
  Galactic Centre declinations: an **on-sky separation** (how far the source
  actually moved) and a **coordinate offset** (how much the right-ascension
  number changed). The table's own `dra` columns hold the coordinate one; these
  provenance columns hold the on-sky one, and now say so. `prov_dec_deg` records
  the declination each conversion used, so the coordinate offset a provenance
  entry implies can be re-derived exactly rather than bounded.

  Tables written before that convention use `prov_dra_added_mas` /
  `prov_ddec_added_mas`; they are renamed on their next correction, values
  untouched.
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
  `prov_dra/ddec_onsky_mas` accumulate, the write
  path also rejects any ROW whose total accumulated correction exceeds the **bulk**
  limit. ⚠ That bounds accumulation at 60″, so a table can still reach tens of
  arcseconds of `prov_*_onsky_mas` inside the bound. **The diagnostic for a
  poisoned table is `prov_*_onsky_mas`, not the total `|offset|`** — an m2 visit-consensus correction is
  mas-scale by construction, so arcsecond-scale `prov_*_onsky_mas` is a category
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
  **mean** of the corrections sharing a row, because a family row can only
  express the module-common shift; the per-detector spread is a
  distortion/DVA-class systematic the row has no freedom to remove. Pooling
  before the floor is what makes the loop converge — residuals that largely
  cancel pool to a sub-floor shift and the checkpoint PASSES instead of writing
  their sum. It is deliberately narrow and **refuses** rather than guessing:
  across module families (so a `Module`-less table still gets the actionable
  "rebuild `--per-module`" refusal instead of a silent A/B average), a module
  contributing twice to one row (two vgroups against a `Vgroup`-less table), a
  group whose members disagree by more than `ASTROM_MAX_POOL_SPREAD_MAS`
  (50 mas) measured as the largest separation between any two of them **as
  vectors**, and any member over the magnitude ceiling — which is checked on the
  **members**, since a mean cannot exceed the largest of them and pooling would
  otherwise average a blown-up detector out of existence. What was collapsed is written to the
  checkpoint record under `pooling`. The correction's `source` string survives in
  the offsets table -- its provenance column is widened to fit the value being
  written rather than left at whatever width the table happened to have, so the
  detector list is not silently cut (there is still an outer bound,
  `PROV_TEXT_MAX_CHARS` = 256 characters, but reaching it is announced, and
  everything the checkpoint writes is far short of it — a stage token, one of two
  fixed bases, and a pooled suffix naming at most one module's four detectors;
  `PROV_TEXT_MAX_CHARS` says why no exact figure is quoted, since the spread
  field's width depends on `ASTROM_MAX_POOL_SPREAD_MAS`) — but a one-line summary
  is not the per-detector measurements, which is why the record carries them
  separately.
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

### Naming: records are keyed on the OBSERVATION

`checkpoint_{stage}_{filter}{obs_token}_latest.json`, where `obs_token` comes
from `consensus_obs_token(proposal_id, obsid)` — `_o002`, `_j7213`, or
`_o002-998` for a registered joint obsid.  A mixed-filter run
(`filtername=None`) writes `checkpoint_{stage}_all{obs_token}`.  The m7
cross-filter record carries it too.

The token is not decorative.  cloudef's 2092 observations 002 and 005 share one
`astrometry_checkpoints/` directory, so the untokened
`checkpoint_m2_F360M_latest.json` written by one run replaced the other's, and
every frozen-stage reader then compared one observation's exposures against the
other's baseline — which is not a movement measurement of anything (issue
#281).  An untokened record *body* carries no observation identity either: its
`visit` field is `"1"` for both `jw02092002001` and `jw02092005001`.

**Legacy untokened records, and what happens to them.**  Every record written
before this existed is untokened.  Counted precisely, because two earlier
numbers in this section disagreed with each other: **80**
`checkpoint_m2_*_latest.json` across **13** directories, **0** of them tokened
(638 files including the timestamped history, and symlinked field directories
resolved — brick and cloudc live under `/blue`, the rest under `/orange`).
A reader
that wants `_oNNN` accepts the untokened spelling **only where it cannot be
another observation's**: where the registry says exactly one observation of this
field images this filter.  brick's two observations use disjoint filter sets, so
its legacy records are read normally, with a line saying so.  Where more than
one observation images the filter — and for a `None` filtername, which cannot be
tested at all — the untokened record is **refused**.

That refusal is a **stop, not a downgrade to "unverified"**.  With no m2
baseline the exposure has no frozen-stage comparison, so
`_run_astrometry_stage_checkpoint` appends a "no m2 per-exposure baseline"
failure and raises `AstrometryRegressionError`.  On cloudef, gc2211, ngc6334 and
sickle that halts m3–m8 until m2 is re-run.  **There is no migration**, and
there should not be: renaming an existing record would assert an observation
identity its contents do not carry.  Re-run m2.

To apply corrections from a directory holding more than one observation, pass
`scripts/reduction/apply_m2_checkpoint_corrections.py --obs-token _oNNN`.  It
refuses to run without one rather than union two observations' corrections into
the same table rows.

## Environment switches

| var | effect |
|---|---|
| `ASTROM_CHECKPOINT=0` | disable all checkpoints (emergencies only) |
| `ASTROM_CHECKPOINT_WARN_ONLY=1` | demote blocking failures to loud warnings |
| `ASTROM_CHECKPOINT_APPLY=1` | at m2, auto-apply corrections to the offsets table + stale-tag im0 |
| `ASTROM_REFCAT=<path>` | reference catalog override (default: `{basepath}/catalogs/gaia_virac2_refcat*.fits`) |
| `ALLOW_CONSENSUS_ONLY_ASTROMETRY=1` | run a **VIRAC2-framed** field with no reference catalog on disk; the checkpoints then verify only that the exposures agree with each other, so the absolute frame is UNVERIFIED and the result is not releasable |
| `ALLOW_LATE_STAGE_ASTROM_SHIFT=1` | override the m3+ frozen-solution gate |
| `ALLOW_LATE_STAGE_ASTROM_SHIFT_REASON=<text>` | the written justification CLAUDE.md requires; stored in the record |
| `ALLOW_CROSSFILTER_ASTROM_FAIL=1` | override the cross-filter gate |
| `ALLOW_CROSSFILTER_ASTROM_FAIL_REASON=<text>` | as above, for the cross-filter gate |
| `ASTROM_MAX_CORRECTION_ARCSEC=<f>` | raise/lower the per-exposure ceiling (default 0.5″) |
| `ASTROM_MAX_BULK_CORRECTION_ARCSEC=<f>` | raise/lower the per-visit bulk ceiling **and** the cumulative-drift bound (default 60″) |
| `ASTROM_ALLOW_MISSING_PERFRAME=1` | demote the missing-per-frame-catalog stop (`cataloging.py`) |
| `CATALOG_ALLOW_UNVETTED_FALLBACK=1` | allow the unvetted-catalog fallback |
| `OFFSETS_TABLE_COLLAPSE_RAISE=1` | make the collapsed-visit guard raise instead of warn (`reduction/validate_offsets_table.py`) |
| `FORCE_REALIGN_ON_DISAGREE=1` | hard-stop when a frame's baked `RAOFFSET` disagrees with the current table (`reduction/unified_alignment.py`) |
| `ASTROM_M2_CORRECTION_FLOOR_MAS=<f>` | at m2, MEASURE and RECORD every residual as usual but only ACT on those at or above this magnitude (default 0 = act on all). See below. |
| `ALLOW_UNVERIFIED_ASTROM_CHECKPOINT=1` | let a checkpoint that measured a shift and then refused to apply it count as a pass |

### The m2 correction floor, and what it may not suppress

`ASTROM_M2_CORRECTION_FLOOR_MAS` exists for ONE class of residual: a
per-detector distortion term (instrument aperture model, velocity aberration)
that the module-locked offsets table has no way to express. Applying its
detector mean is what the previous cycle already did, so the re-tie loop never
converges. That argument is about the SHAPE of the residual, not its size.

Two consequences follow, and both are enforced in code:

- **The consensus-to-reference tie is never floored.** It is one rigid shift of
  a whole visit onto the reference catalog, which the table expresses exactly,
  so the floor's rationale does not reach it. Flooring it would let an absolute
  frame error up to the floor through with nothing downstream to catch it: a
  common-mode shift moves every band equally, so the m7 cross-band gate sees
  agreement, and the ~100 mas gross gate is two orders of magnitude away.
  (`cataloging._is_whole_consensus_shift`.)
- **The value in force is written into the record**, as
  `tolerances.correction_floor_mas`. Without it, a pass that passed only
  because the floor had been raised is indistinguishable from a clean one, and
  the only trace is a line in a SLURM log. `run_field_retie_loop.sh` can now
  raise the floor by itself (`RETIE_ACCEPT_RESIDUAL_MAS`), which makes that
  distinction matter more, not less.

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
