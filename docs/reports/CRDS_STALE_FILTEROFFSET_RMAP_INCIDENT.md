# CRDS stale-cache incident: module-swapped NIRCam LW filter offsets

**Audience:** anyone auditing NIRCam astrometry in this pipeline (human or agent).
**Status:** root-caused and remediated at the cache level 2026-07-11; per-field
LW WCS remediation tracked below.
**One-line summary:** local CRDS caches seeded in 2022-09 carry a
`jwst_nircam_filteroffset_0004.rmap` whose LONG-channel module assignment is
**swapped** (A↔B) relative to the CRDS server's current file *of the same
name*; every reduction run against such a cache applied the wrong module's
long-wavelength filter offsets — **independent of the installed `jwst`
version** — displacing each LW module by up to ~26 mas on sky.

---

## Why this document exists

A 2026-07 audit found (a) NIRCam module-A/B offsets that *differed between
reduction generations* (F410M ≈ 50 mas inter-module) and (b) an apparent
~78 mas offset between programs 2221 and 1182. Both were initially attributed
to "a stale `jwst` version and the associated SIAF". That attribution is
**wrong in a way that matters for remediation**: re-installing or upgrading
`jwst` does not fix (or cause) the problem, and no SIAF/distortion reference
was ever wrong. If you are auditing a field and see an inter-module offset in
an LW band, check the CRDS cache **content** (not just the context name)
before blaming software versions. This document records the true mechanism,
the exact signatures, and the checks.

## The mechanism

1. `assign_wcs` builds NIRCam imaging WCS from (i) CRDS `distortion`
   references, (ii) CRDS `filteroffset` references (a per-module,
   per-channel detector-frame shift in pixels), and (iii) the pointing
   keywords written by STScI SDP. The installed `jwst` package version
   changes none of the reference *content*.
2. The rmap `jwst_nircam_filteroffset_0004.rmap` was **corrected in place**
   by CRDS early in Cycle 1 (same file name, new content + new embedded
   sha1sum):
   - **2022-09-01 vintage** (sha1 `98d39dc5403e…`):
     `('LONG','A') → filteroffset_0008` and `('LONG','B') → 0007`.
     But 0008's own metadata says module **B**, and 0007 says module **A** —
     i.e. the mapping is swapped.
   - **Server/current** (sha1 `aade9b095a34…`, present in caches synced
     ≥2023-05): `('LONG','A') → 0007`, `('LONG','B') → 0008` — correct.
   - `SHORT` (SW) rows are **identical** in both generations; SW bands were
     never affected.
3. CRDS caches are trusted once populated: a cache seeded in 2022-09 and
   never re-synced serves the swapped mapping forever, under **any** context
   that references rmap 0004 (all of `jwst_1253`, `jwst_1533`, `jwst_1581`
   do). Whether a given reduction was poisoned therefore depends on its
   `CRDS_PATH`, *not* on its `CRDS_CTX` and not on its `CAL_VER`.

## The signature (how it shows up in data)

Applying module-B offsets to module A and vice versa displaces each module by
the (own − other) filter-offset difference, i.e. **equal-magnitude,
anti-symmetric sky shifts** with zero module-mean:

| Band (filter+pupil) | per-module error | A−B differential |
|---|---|---|
| F410M (CLEAR) | ±26.3 mas | 52.5 mas |
| F444W + F405N | ±11.0 mas | 22.0 mas |
| F444W + F466N | ±1.9 mas | 3.7 mas |

Measured re-assignment deltas on the Brick 2221 `_cal` files match these to
0.1 mas (e.g. F410M NRCALONG (−11.1, −23.8) mas vs NRCBLONG (+11.2, +23.8) at
pixel (1024,1024)). The anti-symmetric, filter-dependent, LW-only pattern is
the fingerprint; SW bands re-assign to 0.0 mas across `jwst` 1.14 → 1.21.

**Why it looked like a `jwst`-version bug:** reduction generations that used
old `jwst` also happened to use the stale caches; the surgical re-assignments
ran with `CRDS_PATH=/orange/adamginsburg/crds_cache` (fresh copy). CAL_VER
correlated perfectly with the poisoning, causing the misattribution. The
earlier "verified identical under jwst_1253 **and** jwst_1581" check compared
two *contexts* within the same (correct) cache — the stale-cache variable was
never varied.

## What the 78 mas "inter-observation offset" actually was

Unrelated to the rmap. Decomposed by per-tile offset-histogram audit
(2026-07-11):

- The 2221 F212N per-module quick-look mosaics are **untied a-priori
  products**: uniformly ~50–100 mas from VIRAC2 (2022-era blind pointing is
  0.1″ 1σ radial, plus pre-PRDOPSSOC-065 guider-dependent SIAF biases up to
  ~140 mas). Expected; absorbed by the per-exposure VIRAC2 tie.
- The 1182 F200W per-module mosaics on disk at audit time were written
  **mid-re-reduction** with the two visits not yet co-aligned (~100 mas Dec
  discontinuity across each module, confirmed per tile).
- Cross-matching those products produced "A modules disagree by 78 mas, B
  modules agree" and a 99.6 mas loop-closure failure — correctly diagnosing
  *non-rigid inputs*, not a frame error. **No released catalog carries this
  offset** (per-band Gaia DR3 validation: 1–8 mas).

There remains a real ~35 mas A−B declination term in the *untied* 2221 F212N
2024-generation module mosaics — the documented JDox-class NIRCam
inter-module registration residual — which the per-exposure tie removes
downstream. This is why module-level ties and per-tile
`registration_failsafes` checks, not whole-mosaic ties, gate releases.

## Remediation state

**Cache level (done 2026-07-11):** the stale rmap was replaced by the server
copy in all eight poisoned caches — `arches`, `arches_quintuplet`, `brick`,
`cloudef`, `crds`, `sgra`, `sgrb2`, `sickle` (under
`/orange/adamginsburg/jwst/<field>/crds`). The stale copies are preserved
beside them as `jwst_nircam_filteroffset_0004.rmap.stale_20220901_swappedAB`.
`cloudc`, `gc2211`, `m4`, `m92`, `ngc6334`, `ngc6397`, `omegacen`,
`quintuplet`, `sgrc`, `w51`, `wd1`, `wd2` were already correct.

**Data level:**
- Brick + Cloud C 2221 LW `_cal` files: surgically re-assigned 2026-07-04
  (`AssignWcsStep` re-run in place; SCI pixels byte-identical, WCS only);
  offsets tables regenerated afterward.
- Brick F466N (2026-04 rebuild through the then-stale brick cache): carries
  the swap, but the F466N A−B difference is 1.9 mas — below the noise floor;
  no action.
- **Open:** any LW products of `arches`, `arches_quintuplet`, `cloudef`,
  `sgra`, `sgrb2`, `sickle` reduced before 2026-07-11 through their local
  caches carry the swap (F410M-class bands ±26 mas/module; F405N-class
  ±11 mas). Remediate with the same surgical re-assign + re-drizzle +
  re-catalog; fold into the stale-field rerun campaign.

## Auditor checklist

1. **Verify the cache before trusting any LW astrometry:**
   ```
   sha1sum $CRDS_PATH/mappings/jwst/jwst_nircam_filteroffset_0004.rmap
   # aade9b095a34…  = correct
   # 98d39dc5403e…  = stale/swapped — stop; see this document
   ```
2. **Check which file a frame actually used:** `R_FILOFF` in the `_cal`
   header. For LONG detectors: NRCALONG must reference `filteroffset_0007`,
   NRCBLONG `0008`. (SW: NRCA* → 0005, NRCB* → 0006.)
3. **Fingerprint test for the swap:** re-run `AssignWcsStep` with a pinned
   context and a fresh cache, diff the WCS at pixel (1024,1024) per detector.
   Anti-symmetric A/B shifts matching the table above = the swap. All-zero =
   clean. (Script pattern: `_jwst_version_delta.py`, 2026-07 audit.)
4. **Never conclude "version regression" from CAL_VER correlation alone** —
   vary the cache, not just the context/version, before attributing.
5. Measure offsets only with the sanctioned histogram-stacking helpers
   (`astrometry_offsets.measure_offset[_grid]`); map PER TILE before
   interpreting any cross-product comparison, and treat loop-closure failure
   as "inputs not rigid", which includes *mid-rebuild products* — check job
   queues and file mtimes before comparing mosaics.

## Cross-references

- `jwst_gc_pipeline/reduction/ASTROMETRY_WCS_CORRECTION_FLOW.md` — module-lock
  policy section (updated to point here; supersedes its earlier
  "stale-jwst-version" root cause).
- Overleaf astrometry paper, `wcs_provenance.tex` — publication-facing summary
  with the per-program provenance table (SDP/PRD/CRDS/jwst versions, guide
  stars).
- Memory note `crds-stale-filteroffset-rmap-swap` (agent memory).
- JDox: "JWST Pointing Accuracy"; "NIRCam Image Alignment Best Practices";
  STScI memo JWST-STScI-008783 (Sohn 2024) for the 2022-era pointing/guider
  systematics; Griggio, Nardiello & Bedin (2023) for the ~0.2 mas distortion
  residual floor.
