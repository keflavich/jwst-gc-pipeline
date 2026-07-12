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
- Each mosaic ties to VIRAC2 (PM-propagated to obs epoch) `< ~30 mas` bulk, per-tile,
  high contrast; per-visit (not just whole-mosaic). VIRAC2 & Gaia agree.

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
- Released mosaic and its released catalog agree `< ~15 mas`, per tile — and both agree
  with the reference (not just with each other; a shared offset passes item 2 but fails
  item 1).

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
  (`best_dao_basic()` can return a stale LOCKED file that is ~1.9″/21″ off).

## 5. Versioning & provenance
- MANIFEST per-file version bumped; webpage version column updated.

---
*Add this same inter-frame overlap item to the per-observation QA issue template
(`JWST-GC/data-qa`).*
