# Per-detector affine tie at m2 — design

**Status:** v1 (translation-only), default **OFF** (`ASTROM_M2_PER_DETECTOR_TIE=1`
to enable).  m2-only by construction (the m3+ frozen-stage rule applies
unchanged).  Implementation: `detector_tie.py`; applied through the existing
consensus offsets-table + m2 checkpoint stop/regenerate flow
(`astrometry_checkpoint.py`, `ASTROMETRY_CHECKPOINTS.md`).

---

## 1. The residual this targets (and what it is NOT)

The GDC (distortion-map swap) experiment (2026-07, scratchpad
`gdc_experiment/`, `results/report_tables.md`) isolated the surviving
astrometric residual after the DVA inter-detector correction: swapping the
CRDS distortion solution for the GDC one changed **nothing** (every metric
identical to ≲0.1 mas), while the cross-module (A/B seam) same-star offsets
stayed at 2.7–5.4 mas and are **detector-pair- and filter-epoch-dependent**:

| field/filter | detector pair | dRA (mas) | dDec (mas) | n pair-meas |
|---|---|---|---|---|
| brick F212N | nrca3–nrcb4 | −0.2 | **−5.4** | 64 |
| brick F212N | nrca4–nrcb3 | **+2.5** | **−2.5** | 55 |
| brick F182M | nrca3–nrcb4 | +1.5 | −1.3 | 64 |
| brick F182M | nrca4–nrcb3 | +1.1 | +2.7 | 64 |

(same-star matched-pair offsets in the seam overlap, `stage_a_*.json`,
normalized to the sorted pair; the same-detector control pairs read ~1 mas.)

This is **inter-detector rigid PLACEMENT** (SIAF-class), not distortion:
within-detector affine residual rms is 0.77–0.96 mas (report "frames" table),
and the network self-calibration (astrometry paper, `siaf_accuracy`; memory
`siaf-accuracy-network-selfcal`) decomposes the residual into
(a) a deterministic, epoch-dependent DVA scale term (already corrected by
`dva_correction.py`, ON by default) and (b) **static per-detector placements
of 1–2.5 mas** — whose pairwise differences are exactly the 2.7–5.4 mas seam
terms above.

The existing machinery cannot express this term:

* the **module-locked** tables (brick/cloudc) apply ONE shift per
  (visit, filter) to all 8–10 detectors — deliberately (module-lock policy in
  `ASTROMETRY_WCS_CORRECTION_FLOW.md`), so a per-detector placement passes
  through untouched;
* the **per-exposure jitter rows** of the consensus table are per
  (exposure, detector) frame and are gated at 2 mas per frame with per-frame
  measurement noise of ~1–2 mas — a static 2.7 mas detector term is at the
  detection floor per frame, so it is flagged erratically (or not at all) and,
  when flagged, is corrected with per-frame noise re-injected.  This is the
  non-convergence documented at `cataloging.py` (`ASTROM_M2_CORRECTION_FLOOR_MAS`
  comment: "SIAF/DVA-class systematics that the module-locked offsets tables
  cannot express").

The fix: measure the term at the (visit, **detector**) level from the
**cross-detector overlap network** — thousands of same-star pairs per
detector, sem ≲0.3 mas on a 2–5 mas term — and write it as ONE per-detector
offsets-table row.

## 2. Measurement (`detector_tie.py`)

### ⚠ Why NOT "detector vs the visit consensus" (measured self-cancellation)

The first-draft measurement — each detector's pooled stars against the
ALL-detector visit consensus — **self-cancels**.  NIRCam dithers are small
compared to a detector, so an interior star is observed by the *same*
detector in every exposure: its consensus position is built almost entirely
from that detector's own measurements and carries the detector's placement
error with it.  Measured on brick F212N (2026-07-23, the validation harness):
every per-detector vs-consensus offset read **0.3–0.9 mas** (sem ~0.15) while
the seam terms are 2.7–5.4 mas — a ~15–20 % diluted echo carried only by the
seam-strip stars.  Same self-referential trap class as
`registration_failsafes` matching a mosaic against its own catalog (CLAUDE.md
release-gate blind spots) and the brick F182M 15 mas self-cancellation.  The
vs-consensus quantity is retained ONLY as a recorded diagnostic
(`vs_consensus_diag`): it must stay well below the network tie, or the
grouping is wrong.

### The network measurement

Per (visit, filter):

1. **Overlap pairs.** Every overlapping frame pair from DIFFERENT detectors
   (footprint-box intersection via the `interframe_overlap` geometry gate —
   a module gap / disjoint tiles produce no box, so no structural false
   peak; ≥25 stars of both frames in the box; capped at 40 pairs per
   detector pair, deterministic subsample).
2. **Per-pair same-star offset** (the sanctioned two-step, CLAUDE.md
   rule #1): `measure_offset(a, b, sweep=True)` histogram detection, then
   giant-cell `local_residual_map` matched-pair refinement (legal only after
   the verified small, un-swept histogram tie — removes the histogram bias,
   memory `histogram-vs-samestar-offset-bias`).  A swept or >90 mas pair is
   SKIPPED: gross misalignment is the per-exposure consensus machinery's
   job, not a placement trim.
3. **Network solve** (`solve_detector_network`): model `D_ab = p_b − p_a`,
   weighted least squares (weights 1/sem², sem floor 0.05 mas) per connected
   component; per-detector sems from the pseudo-inverse covariance scaled by
   the reduced χ² (never deflated below 1).  **Gauge: the per-component,
   per-axis MEDIAN of `p` is zero** — robust (one bad detector cannot drag
   the gauge, mirroring the consensus median re-centering) and
   bulk-preserving (the corrections have ~zero net effect on the visit's
   absolute tie).  The correction for detector *d* is `−p_d`.
4. VIRAC2 **cross-check** (never Gaia, never blocking — GC rule
   `gc-gaia-frame-not-catalog`): each applying detector's frames are
   same-star tied to VIRAC2 per frame (pair-weighted), minus the visit-bulk
   VIRAC2 tie, giving an independent detector-DIFFERENTIAL estimate.  The
   check is GROSS-only: it must agree with the network tie within
   `DETECTOR_REF_GROSS_TOL_MAS` (15 mas — far above VIRAC2's ~5–10 mas local
   wander so the reference's own systematics can never veto the internal
   measurement; it exists to catch a spurious solve / wrong grouping).
   Disagreement ⇒ REFUSE (leave uncorrected, loud warning), never "use the
   VIRAC2 number instead".

### Refusal floor (leave detector uncorrected + loud warning)

A detector tie is only emitted when ALL of:

* **connectivity**: the detector appears in ≥1 usable overlap measurement
  (an isolated detector, or one whose every pair was swept/gross, is
  REFUSED — never guessed);
* `n_pairs >= DETECTOR_TIE_MIN_PAIRS` (**50**, summed over its network
  measurements): below ~50 same-star pairs the sem of a ~5 mas-scatter
  population is ≳0.7 mas and a 2 mas term is barely 3σ; 50 is also the
  existing `min_stars` of the consensus builder;
* `sem <= DETECTOR_TIE_MAX_SEM_MAS` (**1.0 mas**, χ²-scaled network sem):
  the term being corrected is 2–5 mas; a 1 mas sem keeps every applied
  correction ≥2σ and the applied noise ≤¼ of the smallest real term.  (In
  practice a brick-class detector has 10³–10⁴ pairs across ≥60 overlap
  measurements and sem ≲0.3 mas; the floor triggers on pathological inputs.)
* significance + floor: `|tie| >= DETECTOR_TIE_APPLY_MIN_MAS` (**1.0 mas**)
  and `|tie| >= 3*sem` — sub-mas placements are below the SIAF static floor
  (1–2.5 mas) and below what re-drizzling can use.
* the VIRAC2 gross cross-check above (when measurable).

A refused detector keeps its current alignment (identical to the flag being
off for that detector) and is recorded in the checkpoint record with the
refusal reason.

### Why translation-only in v1 (path to full affine = v2)

The measured seam terms ARE translations: the within-detector affine residual
is <1 mas rms (GDC report), and the network self-cal placement decomposition
finds per-detector rotation/scale residuals below the per-star noise once the
DVA scale is removed.  A 6-param per-detector affine fit against the internal
consensus is straightforward to add (same pooled matched pairs, least-squares
about the detector center), but (a) it cannot be expressed in the offsets
table (`fix_alignment` applies `adjust_wcs(delta_ra, delta_dec)` only — a
rotation/scale would need a new apply mechanism on the GWCS, a much larger
blast radius), and (b) validation of 4 more parameters per detector against a
1 mas noise floor needs the multi-filter network solve, not a per-visit m2
measurement.  v2, if the translation-corrected seam still shows a coherent
residual: extend `adjust_wcs` usage in `fix_alignment` with a per-detector
rotation about the detector fiducial + scale, keyed by new
`drot (deg)`/`dscale` table columns, measured by the same pooled matched-pair
fit with the identical refusal ladder.

## 3. Application — offsets-table rows, no new mechanism

The consensus offsets table (`Offsets_JWST_Brick<pid>_consensus.csv`, keyed
(Visit, Filter, Exposure, Module)) gains a **third row kind**:

| row kind | Exposure | Module | applies to | measures |
|---|---|---|---|---|
| per-exposure jitter | ≥1 | detector | that frame | frame → consensus (guide-star jitter; placement-free by self-cancellation) |
| **per-detector tie** | **−1 (BULK_EXPOSURE)** | **detector token** | every exposure of that detector | detector placement, cross-detector network (this design) |
| per-visit bulk | −1 | `all` | every exposure | consensus → VIRAC2 |

`lookup_consensus_offset` sums **jitter + detector + bulk** for a frame.  The
detector row matching uses the same `_module_variants` semantics as
everywhere else, with exact-detector preference: a frame matching BOTH a
detector-level row and a module-family row uses the exact-detector row alone
(REPLACES, never sums), and any ambiguity beyond that raises.  Application to the frames is unchanged —
`fix_alignment → _apply_consensus_offsets_table → adjust_wcs`, idempotent via
`RAOFFSET`, regenerate-from-`_cal` on correction, exactly the existing m2
stop/regenerate flow.

### No-double-correction rules

* **Detector row vs module row:** a per-detector row REPLACES any
  module-family row for that detector's frames — it never adds.  Enforced at
  authoring time (`update_offsets_table` narrows to the exact-detector row
  when one exists, and refuses a detector-level correction on a table that
  cannot express it) and at apply time (`fix_alignment`'s module narrowing
  prefers an exact detector match; an unresolvable detector+family double
  match still hard-fails with `match.sum() != 1`).
* **Detector row vs per-exposure jitter rows:** the two are COMPLEMENTARY by
  the self-cancellation above — the frame-vs-consensus offset (the jitter
  measurement) does NOT contain the placement term, and the network tie
  contains no per-frame jitter (it is a per-detector combination over
  exposures with a median gauge).  They must NOT be residualized against
  each other: subtracting the tie from the frame offset would fabricate an
  opposite-sign jitter row that cancels the detector row.  When a detector
  row is applied in a cycle, its frames' jitter corrections are **DEFERRED**
  (recorded `jitter_deferred`, loud log line): the run stops/regenerates
  anyway, and the next m2 pass re-measures jitter on the corrected frames —
  the normal retie-loop convergence, with no geometry-dependent
  double-counting possible.
* **Detector row vs `STATIC_PLACEMENT_CORRECTION` /
  `APPLY_DVA_CORRECTION`:** self-consistent by construction — the tie is
  measured at m2 on the frames AS THEY ARE, so whatever placement/DVA
  correction is already baked into the crf is not re-measured.  The standing
  warning from `static_placement_correction.py` applies unchanged: toggling a
  frame-level correction AFTER a table row was solved invalidates the row
  (the genlock base-stamp guard catches the DVA case).
* **Module-locked fields (brick/cloudc):** v1 does NOT author per-detector
  rows into the VIRAC2locked tables — `update_offsets_table` refuses a
  detector-level correction on a table without the row to carry it (mirroring
  its existing per-exposure refusal), with the error message pointing at the
  table-extension step.  Rollout to those fields = extend the locked-table
  builder (`build_virac2_locked_perexp.py`) with detector rows solved by this
  module; until then the flag simply has no apply path there and the
  measurement is record-only.

### Module-lock policy compatibility

`ASTROMETRY_WCS_CORRECTION_FLOW.md` forbids per-module offsets-table splits
because a per-module tweak "injects VIRAC2 noise and breaks the lock".  This
design does not violate that rule's substance: the per-detector term is
solved from the **INTERNAL cross-detector overlap network** (10³–10⁴
same-star pairs per detector, sem ≲0.3 mas, median-gauge = zero net shift) —
VIRAC2 appears only as a gross cross-check and only the per-visit bulk row
touches VIRAC2, the same argument that admitted the per-exposure jitter
rows.  What was actually banned — fitting each module independently to
VIRAC2 — remains banned.

## 4. Gating & flow

* `ASTROM_M2_PER_DETECTOR_TIE=1` enables (default OFF, zero behavior change).
* Measured ONLY at correcting stages (m2/m12); m3+ frozen stages are
  untouched — after the m2 correct-and-stop, the regenerated frames carry the
  detector rows in their GWCS and the frozen-stage movement checks see a
  stable solution (the m2 record baseline is re-written on the re-run, as for
  every other m2 correction).
* When corrections are emitted they ride the EXISTING flow:
  `run_visit_checkpoint` → corrections list → (`ASTROM_CHECKPOINT_APPLY=1`)
  `seed_offsets_table_from_consensus` upsert → stale-tag im0 mosaics →
  `AstrometryCorrectionRequiredError` stop → regenerate from `_cal`.  No new
  flow is invented.

## 5. Validation

`scripts/reduction/validate_detector_tie.py` re-uses the GDC experiment's
cached per-frame m1 catalogs (brick F212N + F182M, visit 001, 192 frames
each): builds the all-detector visit consensus (for the self-cancellation
diagnostic), measures the per-detector network ties, applies them
**in-memory**, and re-measures (a) every cross-module detector-pair seam
(same-star, per overlapping frame pair — the stage-A methodology, sign
convention matched) and (b) the per-frame same-star VIRAC2 tie,
pair-weighted over frames, before vs after.  Acceptance: the headline seam
terms (F212N nrca3–nrcb4 Dec −5.4 mas) collapse toward the ~1 mas
same-detector control floor, and the VIRAC2 bulk moves by ≲ the applied
median-gauge residual (~sub-mas — the correction is internal and must not
drag the absolute frame).  Results table in the PR body.

## 6. Rollout

1. Land default-OFF (this PR).  No behavior change anywhere.
2. Shadow run on a consensus-table field (sgrc): enable
   `ASTROM_M2_PER_DETECTOR_TIE=1` with `ASTROM_CHECKPOINT_APPLY` unset —
   measurements land in the checkpoint records only; audit sem/stability.
3. Enable apply on that field; verify the m2→regenerate→m2 loop converges
   (second pass measures ~0 detector ties) and m3+ stays frozen.
4. Brick/cloudc: extend `build_virac2_locked_perexp.py` with detector rows
   (separate PR), re-tie, re-drizzle, compare the release-gate seam metrics.
5. v2 (rotation/scale) only if the corrected seam still shows coherent
   structure.
