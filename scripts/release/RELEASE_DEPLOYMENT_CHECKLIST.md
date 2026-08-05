# Release / deployment checklist (JWST-GC)

Gate for publishing any mosaic + catalog. Every item is **blocking** unless marked
optional. Run before `stage_release.py --stage`; do not stage around a red gate.

---

## ⛔ 0. Inter-frame overlap astrometry (BLOCKING — the #1 recurring failure)

**Verify that, everywhere two different observations / visits / pointings / dithers
overlap on sky, their stars match to `< 30 mas`.**

This is the persistent, insidious error that has bitten us repeatedly: two overlapping
frames carry a relative astrometric offset **greater than one pixel (usually greater
than one arcsecond)**, so in the overlap region the *same* star is drizzled to two
positions. The result is that the overlap strip **loses all its stars** (they smear /
cancel / fail to co-detect), and — because a whole-mosaic bulk offset can read ~0 while
half the field is untied — the field-average checks pass while the data are corrupt.
Concrete case: Brick 1182 **visit-001 sat ~21″ off visit-002** for weeks; the overlap
was junk while the bulk offset looked fine.

**How to check (reference-free, JWST-internal — this is the sensitive probe):**
- For **every pair of overlapping frames** (visit×visit, obs×obs, pointing×pointing,
  module×module), match detections *in the overlap footprint* and measure the offset
  with the **window-swept offset-histogram** (`astrometry_offsets.measure_offset`,
  sweep to ≥60″). **Never** a nearest-neighbour median — a >search-radius offset
  collapses NN-median to ~0 and hides the error (this is HOW the bug stays invisible).
- Cross-check against **VIRAC2 *and* Gaia** per visit/tile; a real tie agrees across
  both references, a spurious peak does not.
- **BLOCK if** any overlap-pair offset `> 30 mas`, **or** if the offset only appears
  after the sweep widens past the initial window (`swept=True` on a tie you expected
  small ⇒ the frame is grossly shifted), **or** if the overlap region shows an
  anomalous **deficit of matched stars** vs its surroundings (the "lost stars"
  symptom — check per-tile match counts, not just the offset).
- Map it **per tile / per visit**, never a single global number. A good half hides a
  broken half.

**Tooling:** `scripts/reduction/run_astrometry_checkpoint.py` (visit-consensus
per-exposure + reference multi-check, sweep-aware) and
`scripts/reduction/astrometry_audit.py` (inter-module).
**⚠ Gate gap:** the stock `registration_failsafes.py --scan` searches only ±2.5″ with
no sweep, so **it cannot detect a >2.5″ overlap offset** (zero pairs → "can't verify",
not FAIL). Until it is given a sweep / wide window, a passing `registration_failsafes`
is **not sufficient** — the swept per-visit/overlap check above must also pass.

### What `registration_failsafes --scan` gates, per module geometry

The script decides what to check from the field's **module geometry**, measured
from the mosaics themselves (shared pixels carrying real data in both modules,
not touching bounding boxes):

| geometry | example | what is gated |
|---|---|---|
| modules **overlap** | brick, cloudc, sgrc, sgrb2, w51, ngc6334 | the **merged** mosaic — the only product where the two modules are combined, so the inter-module seam appears there and nowhere else |
| modules **disjoint** (adjacent, non-overlapping sky) | arches, quintuplet | **per module**: each module's own mosaic is a complete object, there is no seam, and **every module must pass on its own** |
| **one module** only | sickle (nrcb) | per module, same as above |

**Exit codes are tri-state and only 0 is a pass:**

| exit | `PASS` | meaning |
|---|---|---|
| 0 | `true` | verified |
| 1 | `false` | a band is locally misregistered — BLOCK |
| 2 | `null` | **could not verify** — BLOCK |

`PASS: null` means the gate could not reach a verdict: no mosaics on disk, fewer
than two bands in a channel to cross-match, or — the common one — **a band with
no merged mosaic in a field whose modules overlap**, whose seam therefore went
unchecked. Ambiguity is not a pass. As of 2026-08-03 that last case covers
cloudc F182M, sgrc F115W/F162M, cloudef F162M/F210M and sgrb2 F150W; those bands
used to be omitted from the scan entirely and the field reported green.

---

## 0b. Stage astrometry checkpoints all green (BLOCKING)

Every cataloging run now writes checkpoint records under
`{basepath}/astrometry_checkpoints/` (see
`jwst_gc_pipeline/photometry/ASTROMETRY_CHECKPOINTS.md`): m2 visit-consensus
(per-exposure ≤ 2 mas + multi-check reference tie), m3–m6 frozen-solution, m7
cross-filter (≤ 5 mas per filter, no significant 2″ cell > 15 mas). Before
staging, confirm the `_latest` record of every band has `passed: true` **and**
`all_verified: true` — a could-not-verify is not a pass — and that no
checkpoint was run with `ASTROM_CHECKPOINT=0`, `ASTROM_CHECKPOINT_WARN_ONLY=1`,
or the `ALLOW_*` overrides without a written justification.

---

## 1. Absolute frame
- Each mosaic ties to VIRAC2 (PM-propagated to obs epoch) bulk, per-tile, high
  contrast; per-visit (not just whole-mosaic). VIRAC2 & Gaia agree.
- **Enforced tolerance: `FRAME_TOL_MAS = 15.0`** in `stage_release.py`
  (`check_catalog_on_frame`, catalog vs the Gaia-tied refcat) — a 25 mas tie is
  hard-refused by the gate.
- ⚠ **This gate only runs where a refcat is mapped.** `FRAME_REFCAT` in
  `stage_release.py` currently contains **`brick` only**; for every other field
  `check_catalog_on_frame` returns `None` ("can't enforce → caller warns"), so item
  1 is a warning, not a block. Add the field's Gaia-tied refcat there before
  treating this as enforced.

## 1b. Astrometric frame + epoch declaration (BLOCKING)

The release notes / README **must state the astrometric reference frame and the
position epoch** of every catalog (e.g. "Gaia DR3 frame via Gaia+VIRAC2 refcat,
positions at observation epoch 2022.655, not PM-propagated"), and whether
per-star proper-motion propagation was applied. Catalog `meta` should carry the
same (`REFFRAME`, `REF_EPOCH`). Downstream target lists (MSA plans, slit masks,
TA reference sets) MUST copy that declaration forward.

Why blocking: NIRSpec program 6927's MSA plan v11 was built from a source list
on the deprecated crowdsource-F405N frame (~90 mas off Gaia) with no frame
declaration; its Gaia-based TA candidates therefore sat (+47, +73) mas off the
science targets — a systematic half-shutter slit miss that no acquisition step
can remove. A one-line frame/epoch statement makes this class of error visible
at plan time.

## 2. Image ↔ catalog agreement
- Released mosaic and its released catalog agree — **enforced tolerance
  `SAME_RUN_TOL_MAS = 30.0`, whole-image** (item 2b is the per-tile check, which
  blocks above 30 mas) — and both agree with the reference: a shared offset passes
  item 2 but fails item 1.

## ⛔ 2b. Same-run image↔catalog provenance (BLOCKING)

**A release that ships BOTH images and per-filter catalogs MUST have them from the
same pipeline / cataloging run.** A catalog built before (or after) an image re-drizzle
sits on a *different* astrometric solution, so image and catalog disagree by
construction — it looks like an astrometry bug but is a provenance mismatch.
Concrete case: Brick 2221 F182M — the `..._m7_..._vetted` catalog (2026-07-08) vs the
`...-f182m-merged_i2d` mosaic (2026-07-11) sat ~10–15 mas apart purely because the
catalog predated the image re-drizzle. (Note `-merged_i2d` and `-merged_data_i2d` are
the *same* mosaic under two names — comparing those is fine.)

Enforced in `stage_release.py`: each shipped science image is matched to its shipped
per-filter catalog of the same `(filter, observation)` with the swept offset-histogram;
**BLOCK if any pair disagrees `> 30 mas`.** This is a direct astrometric proxy for
"same run"; a run-id / build-stamp in the catalog `meta` would make it exact (follow-up).
The absolute arbiter is always VIRAC2 — image↔catalog agreement is only meaningful
*within one run*.

## 3. Inter-module (PM-grade)
- NRCA↔NRCB residual mapped (reference-free overlap); flag `> 15 mas` (spurious PM).

## 4. Catalog provenance
- Release uses the current complete vetted products
  (`*_resbgsub_m7_dao_basic_vetted`), **not** the stale `*_LOCKED_*` per-filter catalogs
  (`best_dao_basic()` — the catalog-selection helper in **brick-jwst-2221**, absent from this repo — can
  return a stale LOCKED file that is ~1.9″/21″ off).

## 5. Versioning & provenance
- MANIFEST per-file version bumped; webpage version column updated.

---
*Add this same inter-frame overlap item to the per-observation QA issue template
(`JWST-GC/data-qa`).*

## 0c. Position-vs-brightness systematics (BLOCKING for pointing-reference catalogs)

A catalog used as a pointing (NIRSpec MSA/TA) reference must show **no
systematic variation of position with source brightness** — saturation-core
centroid bias, satstar wing-fit substitution, and nonlinearity all imprint
exactly such a trend, and the bulk reference tie (dominated by faint stars)
cannot see it.  Run
`scripts/reduction/run_astrometry_checkpoint.py --brightness <catalog> --refcat <refcat>`:
no magnitude bin's mean residual above tolerance (default 5 mas, significance-
gated), no significant mas/mag slope.

## 0d. Photometric continuity (BLOCKING, certified on the SCIENCE subset)

`stage_release.check_photometric_continuity` gates the shipped combined merged
catalog on (a) saturation-boundary continuity (`CONTINUITY_PAIRS`) and (b)
degenerate-pair color flatness (`DEGENERATE_PAIRS`: F405N-F410M, F182M-F187N),
floor `CONTINUITY_TOL_MAG = 0.10` mag.

**Flatness is certified on the SCIENCE subset** (`degenerate_pair_flatness(...,
science_only=True)`): the rows a user analyses after cutting every saturation
flag (`is_saturated`, `replaced_saturated`, `forced_filled`, either band), and
only over bins BRIGHTER than the 40th percentile of the reference-band magnitude
(where a suppression strip lives). The recovered / deep-core satstar rows stay in
the released table under their flags but are NOT required to be color-flat — some
carry a recovered-satstar color bias the flags exist to signal. The gate logs the
flag-inclusive drift too, so a regressing satstar flux scale stays visible.

**No per-pair exemption. Both flatness pairs hard-block, at `min_n=200`.** The
worst per-bin deviation is measured only over bins holding ≥200 stars
(`DEGENERATE_FLATNESS_MIN_N`), so a sparse saturation-onset bin cannot decide the
release. On Brick 2026-08 m8:
- **F405N-F410M** — raw science metric is 0.386, but that is an **n=33** bin at
  F410M=12.75 (and an n=184 bin at 13.0, dev 0.246). At `min_n=200` the worst
  qualifying bin is F410M=14.25 (**n=1418, dev 0.083**) → **passes** (<0.10,
  margin ~17%). Every bin with n≥800 has |dev| ≤ **0.083**.
- **F182M-F187N** — science metric **0.049** ("~0.05"); flag-inclusive **0.227**,
  logged not gated (the recovered-satstar color offset that `science_only` cuts).

The flag-inclusive figures the gate logs (**F405N-F410M 0.333, F182M-F187N 0.227**)
are computed at the DEFAULT `min_n`, not `min_n=200`: they are a satstar
flux-scale diagnostic, and such an offset lives in exactly the sparse bins
`min_n=200` suppresses, so measuring them at 200 would hide what they exist to
show (W51 m8: F405N-F410M flag-inclusive is 2.01 at the default min_n, 0.01 at
200). These match the numbers `scripts/analysis/regenerate_satstar_cmds.py`
annotates the published CMDs with.

The 2026-07-11 suppression-strip guard (`test_suppression_strip_refused`) still
fails at 0.352 because its strip bins hold ~800 stars, so `min_n=200` suppresses
the sparse onset bin without whitelisting a real, well-populated strip in either
pair. The gate never prints "ok" for a population it declined to measure: if the
science subset is too small to measure at `min_n=200` (nan), the pair fails
`NOT-CERTIFIED` unless the flag-inclusive metric is itself measurable AND clean —
so a small catalog carrying a real strip (both metrics nan) still **blocks**
rather than slipping through the raised floor.

**Saturation-BOUNDARY continuity is a SEPARATE gate.** `CONTINUITY_PAIRS` runs
`saturation_continuity(cat, band_sat='f410m', band_ref='f405n')` (band_sat = the
`replaced_saturated` band, band_ref = the reference binned in `mag_ref`) =
**0.170 mag (C1-boundary-jump)** on Brick m8: the recovered-F410M-satstar rows sit
~0.17 mag off the unflagged locus in the two bright transition bins **F405N
12.0–13.0** (n_sat≈38, jumps −0.170 / −0.154). This is the same recovered-satstar
color bias seen in F182M-F187N flatness, but the boundary metric measures the
satstar↔normal transition and therefore CANNOT exclude the satstar rows;
`science_only` does not apply to it. (The 0.04 mag figure is the reverse argument
order `saturation_continuity(band_sat='f405n', band_ref='f410m')`, which the gate
does not evaluate — do not quote it as a pass.)

**Known limit (`CONTINUITY_BOUNDARY_KNOWN_LIMITS` — WARN, triply scoped, not
whole-pair).** The 0.170 is an observation-design floor: a NIRCam-LONG field read
out at **NGROUPS≤2 (BRIGHT2)** cannot recover the deepest saturated cores (railed
at group 0), so the recovered F410M–F405N color carries an irreducible residual
in the saturation-onset bins. The gate WARNs instead of blocking **only** when all
three hold, else it FAILS:
1. the pair is F410M–F405N (the only entry);
2. the field's F410M readout is NGROUPS≤2, **read from the shipped science mosaic
   at gate time** (fail-closed if the mosaic/NGROUPS is unreadable);
3. the jump is below `CONTINUITY_BOUNDARY_FLOOR_CEILING_MAG` (default 0.35) — a
   gross break is not the floor.

Measured (2026-08): **brick 0.170, NGROUPS=2 → WARN** (gate green). **cloudc 2.84,
NGROUPS=2 → FAIL** (gross, above the ceiling). **w51 0.577, NGROUPS=5 → FAIL** (not
the railed regime). This is **PROVISIONAL** — saturated-star recovery photometry
improvement is under active investigation; remove the entry once the recovered
deep-core flux scale is fixed. The biased rows stay in the catalog under
`replaced_saturated` for anyone who cuts them.
