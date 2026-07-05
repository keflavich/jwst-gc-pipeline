# JWST pipeline upgrade playbook

Reusable procedure for moving the Galactic Center NIRCam reduction to a new
`jwst` version and/or CRDS context. Run this **every time** you consider an
upgrade. It is a **photometric re-calibration campaign**, not an astrometry
change — treat it as such.

The worked example at the bottom (2026-07: jwst 1.14 -> 1.21, CRDS 1253 -> 1581)
shows the expected shape of the answers. Redo the measurements each time; do not
trust old numbers.

---

## How to use this playbook

1. Work top to bottom. Phases 0-1 are cheap (no fan-out compute) and gate the
   expensive Phase 2.
2. Fill in the **Delta tables** with freshly measured numbers for the specific
   version/context pair you are considering.
3. Do not bundle an upgrade with an unrelated fix (astrometry, a single-band
   re-reduction, etc.). Keep the variables separated so a regression is
   attributable.
4. QOS `astronomy-dept-b` for all SLURM. NO email from jobs.
5. Never clobber the frozen public release (`/orange/adamginsburg/jwst/releases/`,
   live download targets) — reduce into a new versioned product tree.

## The one invariant worth memorizing

CRDS bumps almost always change **photom** (zeropoint) and often **flat**, while
leaving the ramp-level refs (dark/gain/linearity/superbias/readnoise) and the
**astrometric refs (distortion, filteroffset)** unchanged. So:

- **Photometry moves** -> every flux-derived product must be rebuilt.
- **Astrometry usually does NOT move** with a CRDS bump -> no re-tie, just
  re-verify. (A jwst *version* bump can still perturb resample/drizzle at the mas
  level — re-verify anyway.)

Confirm this invariant holds for your specific pair in Phase 0; if a CRDS delivery
DID change distortion/filteroffset, escalate — that is an astrometry campaign too.

---

## Phase 0 — Measure + pin + audit (no fan-out compute)

### 0a. Record the exact versions
- Current products' `CAL_VER` and `CRDS_CTX` (read from an existing `_cal` header).
- Candidate `jwst` tag (prefer a TAGGED release over a moving dev HEAD) and its
  install path. The repo is an editable checkout — `git -C <jwst repo> describe
  --tags`.
- Candidate CRDS context (latest cached, or newest on the server). Pin it
  explicitly; do not rely on "latest".

### 0b. Measure the reference delta (both pipeline stages, LW + SW)
For a representative LW and SW exposure, run `crds.getreferences` under the OLD and
NEW context for the full reftype set and diff. Fill:

| stage      | reftype | old ctx | new ctx | same? | impact |
|------------|---------|---------|---------|-------|--------|
| Detector1  | dark, gain, linearity, superbias, readnoise | | | | ramp / S/N |
| Detector1  | mask, saturation | | | | DQ / coverage |
| Image2     | flat | | | | per-pixel response |
| Image2     | photom | | | | absolute ZP |
| Image2     | area | | | | — |
| assign_wcs | distortion, filteroffset | | | | astrometry |

### 0c. Quantify the ZP shift
For each band, read `PHOTMJSR` from the old and new `photom` reference and record
the flux ratio (new/old). This is the headline scientific change. Tabulate all
bands.

### 0d. Audit custom code against the new jwst
The custom reduction steps (`PipelineRerunNIRCAM-LONG.py`, `destreak`,
`fix_alignment`, module-lock, `realign_to_catalog`) were written against an older
jwst datamodels/step API. Check for breakage:
- New reftypes the pipeline must handle or skip (e.g. jwst 1.21 Image2 added a
  `bkg` reftype absent in older contexts -> prefetch error unless `bkg_subtract`
  is skipped or the context provides it).
- datamodels attribute renames, step-signature changes, CRDS prefetch behavior.
- Dry-import + a single-exposure Detector1+Image2 smoke test before any fan-out.

**Gate:** do not proceed until 0b/0c are tabulated and 0d passes a smoke test.

---

## Phase 1 — Control-field pilot (one field)

Pick one field (brick is the reference). Full `SKIP=0` re-reduce from uncal under
the pinned jwst+CRDS into a NEW tree (e.g. `<field>/pipeline_v<N>/`), all bands.

Compare vs current products, per band:
- **photom**: measured flux ratio should match the `PHOTMJSR` deltas from 0c.
- **S/N + source counts**: expect ~unchanged if ramp refs were SAME; flag large
  deltas (points at a version-algorithm change).
- **astrometry**: re-run the astrometry verifiers; expect ~unchanged if
  distortion/filteroffset were SAME. Confirm any known WCS bug fixed by the new
  version is gone natively.
- **DQ/coverage**: effect of the mask/saturation change.

**Gate:** proceed only if every delta is explained and acceptable. Produce a
one-page old-vs-new diff report and get sign-off.

---

## Phase 2 — Fan out

- `SKIP=0` over all fields/filters (SLURM arrays, `astronomy-dept-b`, no email)
  into the new versioned tree.
- Rebuild catalogs (m1-m8) + cross-band merges per field.
- Re-run the astrometry audit + all flux-derived analysis (SEDs, color cuts,
  YSO/candidate selections, external-catalog flux ties).
- Astrometry: NO re-tie needed if distortion was stable — just re-verify each field.

## Phase 3 — Release

- Stage a NEW release version; keep the current frozen release untouched.
- Update webpage / distribution deliberately, as its own step.

---

## Decision guide

- **Upgrade needed for an astrometry bug?** Usually NO — astrometry is stable
  across CRDS bumps. Fix the astrometry directly (offsets tables, targeted
  assign_wcs re-run) with photometry preserved. See
  `BRICK_FILTEROFFSET_SWAP_FIX_NOTES.md` for the pattern (re-run Image2 under the
  OLD context so flat/photom are byte-identical while the WCS is corrected by the
  newer jwst code).
- **Upgrade to get newer absolute ZPs / calibration?** YES, but as a deliberate
  campaign per this playbook, not a side effect.
- **Cheap intermediate (newer ZPs, no version bump):** re-run **Image2 only**
  (assign_wcs+flat+photom) under the new context, keeping existing ramps. Caveat:
  mixes old-version ramps with new photom/flat — let the Phase-1 comparison decide
  if that hybrid is acceptable or a full `SKIP=0` is required.

---

## Worked example — 2026-07 (jwst 1.14.1.dev43 -> 1.21.0.dev314, CRDS 1253 -> 1581)

Measured on brick NIRCam.

**Versions:** products built with jwst `1.14.1.dev43` + `jwst_1253.pmap` (2024-07).
Candidate: jwst `1.21.0.dev314` (editable checkout; = 1.20.0rc1 +314 commits) +
`jwst_1581.pmap` (28 pmap deliveries newer).

**Reference delta (NIRCam):**

| stage      | reftype | 1253 -> 1581 | impact |
|------------|---------|--------------|--------|
| Detector1  | dark, gain, linearity, superbias, readnoise | SAME | ramp / S/N unchanged |
| Detector1  | mask | DIFF (0062->0074) | marginal DQ/coverage |
| Image2     | flat | DIFF (LW; some SW same) | small spatial photometric change |
| Image2     | photom | DIFF (all bands) | absolute ZP shift |
| Image2     | area | SAME | — |
| assign_wcs | distortion, filteroffset | SAME | astrometry unchanged |

**ZP shift (PHOTMJSR, NRCALONG):** F356W 0.421 -> 0.423 = **+0.5%**;
F444W 0.402 -> 0.391 = **-2.7%** (~0.03 mag). Band-dependent, up to ~3%.

**Headlines:**
1. Astrometry stable — distortion + filteroffset byte-identical -> the WCS
   (including the filteroffset module-swap fix) does not move with the CRDS bump. A
   full re-reduction under jwst 1.21 also auto-fixes the filteroffset module-swap
   (a jwst 1.14 bug) for every field/LW band.
2. Photometry changes (new photom + flat) -> rebuild all flux-derived products;
   ramp-level S/N unchanged.

**Custom-code breakage found:** jwst 1.21 Image2 has a `bkg` reftype absent in
1253 -> full Image2 under 1253 errors on prefetch; skip `bkg_subtract` (the 2024
cal had no background step, so this is faithful).

**Conclusion (2026-07):** NOT bundled with the brick astrometry fix. The
filteroffset fix was done under 1253 with photometry preserved. A full refresh is a
separate, pinned, piloted campaign for after the immediate deliverables.
