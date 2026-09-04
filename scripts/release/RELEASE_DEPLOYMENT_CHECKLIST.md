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
**⚠ Gate gap:** `registration_failsafes.py --scan` searches ±2.5″ per cell, and a
passing `registration_failsafes` is **not sufficient** on its own — the swept
per-visit/overlap check above must also pass, because this one matches a mosaic against
its own catalog (self-referential) rather than frame-vs-frame.
Since #588 a cell whose peak lands beyond half that window (`WINDOW_EDGE_FRAC = 0.5`,
i.e. > 1250 mas) is not graded on that number alone — at that radius a real large
offset and the arg-max of the disk's own wrong-pair background are the same number.
It is resolved first, by contrast and by **sweeping** the cell to 5″ and 10″
(`SWEEP_FACTORS`): an offset that reproduces across windows is graded at the swept
value, so the check now measures the 2.5–10″ regime it used to alias to ~1.8″. A cell
neither test can resolve measured nothing; it is reported as `n_window_edge` /
`window_edge_cells` (with each cell's `swept_windows`), and a 4-connected patch of
`MIN_SEAM_CELLS` such cells makes the verdict **could-not-verify** — exit 2, which
`stage_release.py` refuses on. It is never a pass. Above 10″, and for anything
frame-vs-frame, the gate is `check_interframe_overlap.py --scan`.
A displacement large enough to carry a region entirely **off** the truth footprint still
leaves that region with no pairs at all — cells that never verify, and so are reported
rather than failed. That hole is unchanged by #588 and is the standing reason a passing
`registration_failsafes` is not sufficient on its own.

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

- ⚠ **A pair the arbiter cannot settle stays "could not verify", and the field
  does not stage.** When an overlap is too thin to measure frame-against-frame
  *and* its own footprint holds too few reference stars to arbitrate, the check
  used to fall back to the FIELD-WIDE same-star map. That map is one verdict for
  a whole filter, so a seam confined to the sliver is a minority of every cell it
  measures and leaves the verdict clean — issue #174's conclusion, reproduced
  against the real gate on a 500 mas seam. That fallback is now **off by
  default**.

  `OVERLAP_ALLOW_FIELDWIDE_CLEAR=1` re-enables it. Setting it records a decision
  reached some other way rather than justifying one, and the log then reads:

  ```
  WARNING: pair <a> | <b> could NOT be arbitrated in its own overlap footprint
  (...); cleared ONLY by the FIELD-WIDE same-star map, which cannot resolve
  this pair's sliver -- because OVERLAP_ALLOW_FIELDWIDE_CLEAR=1 was set.
  ```

  Grep a staging log for `cleared ONLY by the FIELD-WIDE` before trusting a
  "0 FAIL" verdict. What removes the need for the override is a **denser arbiter
  star list** for the field.

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
- ⚠ **`FRAME_REFCAT` needs a DENSE catalogue; do not add a sparse one to it.**
  This gate measures whether a shipped catalogue sits on the right sky, and a
  sparse list gives a noisy bulk tie that would refuse good data. A field whose
  only star list is sparse (w51: ~9,500 rows, medNN 5.2″, against brick's
  ~115,000 at 1.1″) belongs in **`OVERLAP_ARBITER_REFCAT`** instead — a separate
  registry, read by `overlap_arbiter_refcat()`, whose list is used only to
  tie-break an inter-frame overlap too thin to measure frame-against-frame.
  `overlap_arbiter_refcat` falls back to `FRAME_REFCAT`, so a field with a dense
  catalogue needs one entry, not two.
- ⚠ **Known gap (issue #263).** The overlap check routes a star list into its
  gating or diagnostic slot by whether the file has a `source` column, not by
  how dense it is — so a sparse Gaia-only list without that column is used for
  gating. Routing by content has not landed. Until it does, read that check's
  log line for the catalogue it names rather than assuming VIRAC2.

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

## ⛔ 3b. The staged set is complete and from one generation (BLOCKING)

Both checks run in `stage_release.py` before anything is copied, and both show up in
a plain dry run (no `--stage` needed).

- **Every explicitly-listed src exists.** `nircam`/`miri` entries in `FIELDS` are
  curated by hand, so an absent one means the config is stale — usually because the
  m2 astrometry checkpoint renamed the product to `..._im0_badastrom.fits` (with a
  `.why.json` beside it) after correcting an offsets table. `discover_nircam` /
  `discover_miri` used to `continue` past a missing src, which staged a release
  short that band with nothing printed; they now collect it and `main` REFUSES
  (rc 2), naming the filter, the path, and the quarantined sibling. No override —
  the fix is to repoint the entry or re-drizzle.
- **One reduction generation per field.** `check_generation_span` reads `DATE` and
  `CRDS_CTX` from every staged science mosaic and reports, per instrument, a `DATE`
  span `> GENERATION_SPAN_DAYS` (7 d) or more than one `CRDS_CTX`. Report-only by
  default; `--refuse-mixed-generations` makes it rc 2. NIRCam and MIRI are compared
  separately (they are reduced in separate batches). v1.1's sickle set trips both
  legs: F210M 2026-04-19 / `jwst_1535.pmap` against four bands 2026-06-27 /
  `jwst_1537.pmap`.

## 4. Catalog provenance
- Release uses the current complete vetted products
  (`*_resbgsub_m7_dao_basic_vetted`), **not** the stale `*_LOCKED_*` per-filter catalogs
  (`best_dao_basic()` — the catalog-selection helper in **brick-jwst-2221**, absent from this repo — can
  return a stale LOCKED file that is ~1.9″/21″ off).

### The `oksep` quality-cut table carries the field's own proposal number

`oksep` is a hand-written label from program **2221**: a source's detections in
different exposures sit close enough together to call it one real star.  The
filtered table it names is written per field, and its suffix carries that
field's own registered proposal(s) --
`merge_catalogs._qualcuts_oksep_suffix()` builds `_qualcuts_oksep1905` for wd1 <!-- noqa: qualcuts-token -->
and `_qualcuts_oksep6151` for w51.  Fields that include program 2221 (brick, <!-- noqa: qualcuts-token -->
cloudc) keep the bare `2221` token so their existing catalogs are not renamed.

Two consequences for a release:

- Never match one program's token literally.  `stage_release.py` matches
  `QUALCUTS_RE`, and its catalog loops `continue` on a non-match, so a literal
  drops another field's quality-filtered table from the release **silently**.
  A grep-guard test (`jwst_gc_pipeline/tests/test_no_hardcoded_qualcuts_token.py`)
  fails CI on a new literal.
- 11 fields with no connection to program 2221 (arches, cloudef, gc2211,
  quintuplet, sgra, sgrb2, sgrc, sickle, w51, ...) still hold
  `_qualcuts_oksep2221` catalogs written before the suffix was per-field.  They <!-- noqa: qualcuts-token -->
  are mislabelled, not corrupt.  Check which token a field's staged table
  carries before quoting the program in release notes.

## 5. Versioning & provenance
- MANIFEST per-file version bumped; webpage version column updated.
- `exposures/` (the detector frames behind each mosaic; on by default,
  `--no-exposures` to omit) is **symlinks, never copies, even under `--copy`**,
  and is deliberately absent from `CHECKSUMS.sha256` — a re-reduction rewrites
  those frames' headers in place, so a frozen hash of one is a claim the
  release cannot keep. `MANIFEST.json` records this as `exposure_mode`. Two
  consequences to check:
  - A dangling link under `exposures/` means the pipeline moved a frame; a
    dangling link under `images/` is a real defect, because that tree must be
    `--copy`. Audit both with
    `find <field> -type l ! -exec test -e {} \; -print`.
  - `EXPOSURE PROVENANCE:` lines at staging name mosaics whose input list
    could not be established, so their frames are not offered. The list is read
    from the mosaic's own `HDRTAB.FILENAME` — `resample` writes one row per
    input, which is the drizzle's own record of what it consumed; the
    `ASNTABLE` association is a fallback for a mosaic with no `HDRTAB`.
    (Inferring the inputs from the association plus a `_crf` twin was wrong for
    25 of 170 staged mosaics, because the pipeline REPLACES the `_cal` suffix
    where that construction appended to it.) The mosaic still ships — the line
    exists so a field quietly offering frames for 9 of its 10 bands is visible
    at staging time rather than from the page.
  - `link mode symlink` in `MANIFEST.json` or at staging means the field's data
    are on a different filesystem from the release root, where a hardlink is
    impossible (brick and cloudc: `/blue` vs `/orange`). Those frames are
    Globus-transfer-only — the HTTPS data plane will not serve a symlink
    pointing out of the release tree — and the page and README say so. Audit a
    staged field with `stage_release.py --check-exposures --field <field>`,
    which also reports frames whose source has been rewritten since staging.

## 6. Publishing the site (do not hand-write the rsync)
- Deploy with `scripts/release/deploy_site.sh` (`--dry-run` first). The docroot
  `htdocs/jwst-gc/` is shared: the release pages are ours, `monitor/` belongs to
  `scripts/monitoring/deploy_monitor.sh` and is **not** in `releases/site/`, so a
  bare `rsync --delete releases/site/ …/htdocs/jwst-gc/` deletes the 194 MB
  monitor tree. It did, on 2026-08-06 — the monitor URL 404'd for five hours.
  The script protects that tree and fails (exit 4) if it is gone afterwards.
- After deploying, `https://starformation.astro.ufl.edu/jwst-gc/monitor/` still
  returns 200.

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

**Known limit (`CONTINUITY_BOUNDARY_KNOWN_LIMITS` — WARN, four-way scoped, not
whole-pair).** The 0.170 is an observation-design floor: a NIRCam-LONG field read
out at **NGROUPS≤2 (BRIGHT2)** cannot recover the deepest saturated cores (railed
at group 0), so the recovered F410M–F405N color carries an irreducible residual
in the saturation-onset bins. The gate WARNs instead of blocking **only** when all
four hold, else it FAILS:
1. the pair is F410M–F405N (the only entry);
2. the metric kind is `C1-boundary-jump` (the railed-core mechanism); a
   `C2-locus-offset` — a different defect — blocks;
3. the field's F410M readout is NGROUPS≤2, taken as the **deepest** readout across
   the shipped science mosaics, **read at gate time** (fail-closed if the
   mosaic/NGROUPS is unreadable or absent);
4. the jump is below **0.25 mag** — a hard-coded ceiling (deliberately NOT
   env-overridable: a single env knob would be a one-factor waiver of a blocking
   gate), set just above the measured 0.170 floor so a regression that worsens the
   jump (~0.30) still blocks.

The waiver (pair, metric, kind, NGROUPS, ceiling, reason) is recorded on the
catalog item and therefore in **MANIFEST.json** (per-catalog `continuity_waivers`
plus a top-level `continuity_gate: "waived"`), and `write_readme` emits a
plain-text "Known photometric limitations" section — so a shipped catalog carries
a durable record that the boundary gate was waived, not only a log line.

Measured (2026-08), and what actually stops each field:

| field | metric | kind | NGROUPS | verdict |
|---|---|---|---|---|
| brick | 0.170 | C1 | 2 | **WARN** (gate green) |
| w51 | 0.577 | C1 | 5 | FAIL — condition 3 (not the railed regime) |
| cloudc | 2.841 | C1 | **None** | FAIL — condition 3 (F410M mosaic quarantined `_i2d_im0_badastrom`, no shippable readout → fail-closed) |

Note this is **not** a demonstration that "every condition is needed" on real data:
today only the NGROUPS check separates the three fields, and cloudc is stopped by
the **missing-mosaic fail-closed** path, not the ceiling. The `C1`-kind guard and
the 0.25 ceiling (condition 4) cover failure modes the current fields do not
contain — they are exercised by the synthetic tests (a `C2-locus-offset`, a gross
0.8-mag break, a 0.30-mag regression). Stated plainly: this is a hypothesis-driven
guard set, not a set each validated against a real counter-example.

This is **PROVISIONAL**. The NGROUPS→railed-core direction is a *hypothesis the
current data cannot test* — n=3 confounded fields, and the one NGROUPS>2 field that
measures the pair (w51, 0.577) is 3.4× worse than brick's 0.170, while brick's own
F182M–F187N sits at 0.043 under the same BRIGHT2 readout. The exit path is a
per-row `railed_at_group0` flag from the reduction, which would let the boundary be
certified on the recoverable population and retire this entry. Remove the entry
once the recovered deep-core flux scale is fixed. The biased rows stay in the
catalog under `replaced_saturated` for anyone who cuts them.
