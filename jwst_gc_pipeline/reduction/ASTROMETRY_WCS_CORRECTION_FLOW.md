# Astrometric WCS correction flow — which files get corrected, and how

**Audience:** anyone (human or agent) modifying the NIRCam reduction/alignment.
**Why this exists:** to keep an unambiguous, reproducible path from the archive
L2 products (`*_cal.fits`) to the final mosaics and catalogs, and to prevent
**double-correction** of the astrometric WCS.

Implemented in:
- `PipelineRerunNIRCAM-LONG.py` — `fix_alignment()` (per-exposure), Image3 call, `realign_to_catalog()` call sites.
- `align_to_catalogs.py` — `realign_to_catalog()`, `sync_gwcs_to_fits_wcs()`.
- `photometry/merge_catalogs.py` — `shift_individual_catalog()` (catalog side).

---

## ⛔ FORBIDDEN: dense-nearest-neighbour-median astrometry

**Never compute or apply an astrometric offset as the MEDIAN (or mean) of
nearest-neighbour matches (`match_to_catalog_sky` / `search_around_sky`) against a
DENSE reference catalog (VIRAC2 / VVV / GNS, median NN spacing ≲ 3").** When the
true shift exceeds the reference's nearest-neighbour spacing, NN pairs the WRONG
star and the median **collapses toward ~0** (or a spurious value). It fabricates
false agreement and has repeatedly fooled *validation* of the GC fields (a
NN-median check "confirms 0.00 fine" on a frame that is really off). (Note: the
brick-1182 v001 ~20" error itself was an offsets-table CURATION collapse, not a
NN-median measurement — see the brick-1182 note — but NN-median is the same class
of failure and must never be used.)

This is now enforced in code:
`jwst_gc_pipeline.photometry.measure_offsets.assert_sparse_reference_for_nn_median`
**raises `DenseNNMedianAstrometryError`** on a dense reference. It guards
`measure_offsets`, `realign_to_catalog`, `bootstrap_reference_catalog`,
`combine_singleframe(realign=True)`, and the `generate_offsets_table` validation.

**Use instead:**
- **2D offset-histogram stacking** — histogram *all* pairwise offsets within ~3",
  take the peak (robust no matter how large the shift). Public helper:
  `jwst_gc_pipeline.photometry.astrometry_offsets.measure_offset` (and
  `measure_offset_grid` for the mandatory per-tile map). See also
  `scripts/reduction/astrometry_audit.py::xcorr` and
  `scripts/miri_reduction/apply_measured_miri_wcs_offsets.py::refine_offset`.
- **a SPARSE reference** — the Gaia-only subset (`source == b'GaiaDR3'`, medNN
  ~5.7"), never the full dense catalog.

**A bulk offset ≈ 0 is NOT sign-off.** A half-mosaic can be grossly SHIFTED while
the field-average reads ~0 (brick-1182 visit-001: a clean ~20" rigid step across the
y=0.5 seam). Always map the offset PER TILE
(`measure_offset_grid`, `registration_failsafes.py`) and require per-tile peak
contrast ≳ 5 everywhere.

A grep-guard test (`jwst_gc_pipeline/photometry/tests/test_no_adhoc_nn_median_astrometry.py`)
fails CI if a new file pairs a NN match with a median/mean — do not write ad-hoc
`match_to_catalog_sky(...).median()`; call `measure_offset` instead.

---

## TL;DR — where astrometric corrections live

| Product (role) | Gets a WCS correction? | Mechanism | Idempotent? |
|---|---|---|---|
| `*_cal.fits` (archive L2b, per-exposure) | **No** — never modified in place | — | (immutable input) |
| `*_destreak*.fits` / `*_align.fits` (per-exposure working copy) | **Yes — GWCS** | `fix_alignment()` → `jwst.tweakreg.utils.adjust_wcs` | **Yes** (`RAOFFSET` header guard) |
| `*_crf.fits` (CR-flagged per-exposure, from Image3) | inherits the corrected GWCS | (produced by Image3 from the corrected input) | n/a |
| `*_i2d.fits` (resampled mosaic) | inherits, **not separately shifted** | resample of the corrected exposures | (pristine) |
| `*_realigned-to-refcat.fits` (i2d copy) | **Yes — FITS hdr + GWCS** | `realign_to_catalog()` + `sync_gwcs_to_fits_wcs()` | regenerated from pristine `_i2d` each run; GWCS sync is an absolute set |
| per-frame catalogs (`*_daophot_basic.fits`) | use the corrected crf GWCS | (read crf GWCS) | n/a |
| merged catalog | **Yes — table-space** | `shift_individual_catalog()`: `final = centroid − RAOFFSET_meta + dra_table` | re-derivable from any offsets table |

**The astrometric solution has exactly two authoring points:**
1. **Per-exposure** (`fix_alignment` → `adjust_wcs`): the science-bearing tie. Catalogs (on crf) and the `_i2d` mosaic both inherit it.
2. **Post-resample rigid tie** (`realign_to_catalog`): a final whole-mosaic CRVAL nudge, applied **only** to the `_realigned-to-refcat` copy of the i2d.

---

## The reproducible path (per-exposure → final)

```
archive  jw…_cal.fits                          (MAST L2b; assign_wcs GWCS; NEVER edited in place)
   │  destreak()  →  jw…_destreak_oNNN.fits     (working copy)
   │  fix_alignment(...)                        (per-exposure GWCS shift via adjust_wcs;
   │                                             reads offsets/Offsets_JWST_Brick<pid>_<ref>[_average].csv;
   │                                             writes RAOFFSET/DEOFFSET + OLCRVAL → IDEMPOTENT)
   ▼
Image3Pipeline.call(..., tweakreg skip=True)    (TweakRegStep is SKIPPED — see note)
   ├─►  jw…-<filt>-merged_crf.fits  (per-exposure, CR-flagged, corrected GWCS)  ──► CATALOGS (crf-space)
   └─►  jw…-<filt>-merged_i2d.fits  (resampled mosaic, corrected GWCS; pristine)
            │  shutil.copyfile → …_realigned-to-refcat.fits
            │  realign_to_catalog(...)           (rigid CRVAL shift to external refcat; FITS hdr)
            │  sync_gwcs_to_fits_wcs(...)         (propagate that shift into the i2d GWCS)
            ▼
        jw…-<filt>-merged_realigned-to-refcat.fits   ← FINAL IMAGE DELIVERABLE
```

**TweakRegStep is intentionally skipped** (`tweakreg_parameters['skip'] = True`).
All absolute alignment is done by our `fix_alignment` (per-exposure) + the
post-resample `realign_to_catalog`, **not** by the pipeline TweakReg step. Do not
re-enable TweakReg without removing one of these, or you will double-correct.

---

## Why two stages, and why no double-correction

- `fix_alignment` ties **each exposure** to the reference using the per-frame
  offsets table (relative frame-to-frame + bulk). It is **idempotent**: the first
  thing it does is check for a `RAOFFSET` keyword and bail if present
  (`align_to_catalogs.py` / `PipelineRerun … fix_alignment`, the `if 'RAOFFSET' in header` guard).
  Re-running the pipeline therefore never stacks shifts on the per-exposure files.
- `realign_to_catalog` applies a single rigid whole-mosaic CRVAL shift to set the
  absolute zero point of the **mosaic**. It always operates on a **fresh
  `shutil.copyfile` of the pristine `_i2d.fits`** (the `_i2d` is never edited), so
  it is reproducible run-to-run and cannot stack. It records `OLCRVAL1/2`.
- `sync_gwcs_to_fits_wcs` sets the i2d GWCS tangent point **equal to** the
  realigned SCI-header CRVAL (an **absolute set**, not a relative shift), so calling
  it twice is a no-op — idempotent by construction.
- Catalogs never read the i2d. They read the crf GWCS and then re-express the tie
  in table space: `shift_individual_catalog` does `centroid − RAOFFSET_meta + dra_table`,
  i.e. it *removes* the GWCS-baked `RAOFFSET` and re-applies the current offsets
  table value. This makes the catalog frame re-derivable from any offsets table
  **without** re-running the pipeline, and keeps catalog ↔ mosaic ties consistent
  (both ultimately trace to the same offsets table + refcat).

**Single rule to avoid double-correction:** correct the astrometry at *exactly one*
of {per-exposure `fix_alignment`, post-resample `realign_to_catalog`} for a given
effect. `fix_alignment` owns the per-frame solution; `realign_to_catalog` owns only
the residual rigid mosaic zero-point. Never add a third corrector, and never edit
`_cal.fits` or `_i2d.fits` in place.

---

## Tooling: use STScI tools; the one documented exception

- **Per-exposure GWCS shifts MUST use `jwst.tweakreg.utils.adjust_wcs`.** It applies
  the shift on the `v2v3`/tangent frame of a *calibrated* (`_cal`/`_tweakreg`/`_skymatch`)
  GWCS — the supported, correct path. `fix_alignment` already does this. Do **not**
  hand-edit `crval`/`pc` of a per-exposure GWCS.

- **Resampled (i2d) GWCS: STScI provides NO tool.** `adjust_wcs`'s own docstring
  states it is *"not designed to handle … GWCS of resampled images."* So for the
  `_realigned-to-refcat` mosaic we cannot use `adjust_wcs`. `sync_gwcs_to_fits_wcs`
  therefore rebuilds the resampled GWCS's terminal `gwcs.fitswcs.FITSImagingWCSTransform`
  with the new `crval` (keeping `crpix`/`cdelt`/`pc`/`projection`) using gwcs's own
  public model API — this is the minimal, supported way to set a resampled tangent
  point, and it is verified to make GWCS == FITS-header WCS to < 0.01 mas. It is **not**
  a free-form transform hack and it is idempotent (absolute set).

  > If a future jwst/gwcs release ships a sanctioned resampled-WCS shifter, replace
  > `sync_gwcs_to_fits_wcs` with it.

- **Preferred long-term simplification (not yet implemented):** fold the
  `realign_to_catalog` rigid offset into the per-exposure offsets table so the tie
  is applied once, at the `_cal` level via `adjust_wcs`, and the resampled i2d is
  correct *by construction* (no post-resample GWCS edit needed at all). This would
  make the i2d and catalogs share a single tie mechanism. Left as a TODO because it
  requires regenerating the offsets table to absorb the current rigid residual.

---

## Offsets-table provenance (how each `offsets/Offsets_*.csv` is built)

`fix_alignment` reads a per-frame offsets table
`offsets/Offsets_JWST_Brick<pid>_<refname>[_average].csv` (see
`PipelineRerunNIRCAM-LONG.py` ~L1047–L1081). Those tables are **inputs** the
pipeline consumes but does not itself generate — so the builders are tracked for
provenance. Losing a builder means a correction can no longer be reproduced or
audited from first principles even though catalogs/mosaics already carry it.

| reference frame (`refname`) | offsets table | builder |
|---|---|---|
| Gaia/VIRAC2 (brick/cloudc) | `Offsets_JWST_Brick<pid>_VIRAC2[locked].csv` | `build_gaia_virac2_refcat.py` (seed refcat) + the per-frame measure in the reduction |
| **GNS (sickle, prop 3958)** | `Offsets_JWST_Brick3958_GNS.csv` | `brick2221/reduction/build_sickle_gns_offsets.py` (in **brick-jwst-2221** — region-specific) |

**Sickle → GNS:** a 2026-06-20 audit found sickle catalogs sit at the **raw
`assign_wcs` frame** (`RAOFFSET=0`, no offsets table for 3958) — ~91 mas off the
GNS reference the mosaics are tied to. The user chose the GNS frame.
`brick2221/reduction/build_sickle_gns_offsets.py` (in the **brick-jwst-2221**
repo, where sickle-specific code lives) measures the per-filter bulk sickle→GNS
correction and writes the per-frame table in the `shift_individual_catalog`
convention (`dra_table = corr_dRA_onsky / cos(dec)`). Its output
`Offsets_JWST_Brick3958_GNS.csv` is dropped into the sickle data tree's
`offsets/` dir and consumed by `fix_alignment` like any other table.
`brick2221/shellscripts/sickle_gns_reduce_retry.sh` is the companion submitter
that re-reduces sickle with the GNS table. (See the `PipelineRerunNIRCAM-LONG.py`
~L1140 note.)

**MIRI registration** is a separate, manual pre-step — MIRI does **not** use the
NIRCam offsets-table path. The sickle MIRI frames are registered to the NIRCam
F480M frame by region-specific scripts in **brick-jwst-2221**
(`brick2221/reduction/register_sickle_miri_o001_o002.py`,
`register_o002_f770w_per_frame_to_f480m.py`,
`register_o002_f770w_gwcs_to_f480m.py`, `merge_sickle_miri_o001_o002.py`). They
edit the per-frame FITS WCS / embedded gwcs in place (idempotent via
`MIRIDRA`/`MIRIDDE`/`MIRIWCSN`) and **must be run before cataloging** a sickle
MIRI obs, or its mosaics sit ~3.3″ off truth while the catalog underneath is
correct. The brick F2550W reduction-tool scripts in `scripts/miri_reduction/`
are region-general examples; the operational MIRI scripts live in brick-jwst-2221.

## Module-lock policy (NRCA == NRCB) and the F410M inter-module offset

`fix_alignment` applies **one shift per (visit, filter) to BOTH modules** (NRCA and
NRCB) — the locked table (`Offsets_JWST_Brick<pid>_VIRAC2locked.csv`) is keyed on
(Visit, Filter), not Module. This is deliberate: NIRCam's SIAF/`assign_wcs` solution
co-registers the two long-wave detectors (NRCA5, NRCB5) to <~1 mas *when assign_wcs is
run against a correct CRDS cache*, so an independent per-module tweak would normally
inject VIRAC2 noise and break the lock.

**ROOT CAUSE CORRECTION (2026-07-11): the F410M inter-module offset was a STALE LOCAL
CRDS CACHE serving a module-swapped LW `filteroffset` mapping — NOT a jwst-version bug
and NOT a SIAF/distortion issue.** Full incident report, fingerprint table, and
auditor checklist:
**[docs/reports/CRDS_STALE_FILTEROFFSET_RMAP_INCIDENT.md](../../docs/reports/CRDS_STALE_FILTEROFFSET_RMAP_INCIDENT.md)**.
Short version: `jwst_nircam_filteroffset_0004.rmap` was corrected in place by CRDS
early in Cycle 1; local caches seeded 2022-09 (brick, arches, arches_quintuplet,
cloudef, sgra, sgrb2, sickle, crds — all repaired 2026-07-11, stale copies kept as
`*.stale_20220901_swappedAB`) mapped `('LONG','A')→filteroffset_0008` (the module-B
file) and vice versa. Result: anti-symmetric per-module sky errors equal to the
(own−other) filter-offset difference — F410M ±26.3 mas/module (52.5 mas A−B
differential), F405N (F444W+F405N) ±11.0 (22.0), F466N ±1.9 — **independent of the
installed jwst version**. Once the cache is correct, every band re-assigns to 0.0 mas
across jwst 1.14→1.21; SW mappings were identical in both rmap generations, so SW was
never affected. The earlier attribution to "jwst 1.14 applied the distortion wrong"
came from CAL_VER correlating with which `CRDS_PATH` each reduction generation used;
the "verified under jwst_1253 AND jwst_1581" cross-check varied the *context* but not
the *cache*, so it could not detect the difference. Checks:
`sha1sum $CRDS_PATH/mappings/jwst/jwst_nircam_filteroffset_0004.rmap`
(`aade9b095a34…` correct, `98d39dc5403e…` stale/swapped) and the `_cal` header
`R_FILOFF` (NRCALONG must use 0007, NRCBLONG 0008).

**Correct fix (surgical): re-run Image2 (assign_wcs) with a verified-fresh CRDS cache.**
Pinning the context keeps flat/photom references identical → photometry is preserved
(no flux perturbation mid-release) while astrometry is fixed. This needs NO
offsets-table hack. (Applied to brick+cloudc LW 2026-07-04.)

**Interim workaround currently in place (must be reverted if Image2 is re-run):** F410M was
given per-module rows (a `Module` column = `nrcalong`/`nrcblong`) in the locked table with a
~48 mas extra shift on NRCALONG, and `fix_alignment` (PipelineRerunNIRCAM-LONG.py ~L1180)
narrows the match by module when >1 row matches and a `Module` column is present. That
band-aid empirically reproduces what correct assign_wcs does (verified: F410M mosaic
module step 68 → ~0 mas). **DANGER:** it is applied on top of the OLD (buggy) `_cal` WCS.
If the `_cal` is regenerated with current jwst, remove the F410M split (revert to a single
both-module row) or it will double-correct by ~48 mas.

**Rule for the offsets-table builder:** a per-module split is a LAST-RESORT workaround for a
frame that cannot be reprocessed. Before splitting, first verify the CRDS cache
(sha1sum check above) and re-run assign_wcs against a fresh cache — a stale local CRDS
cache (module-swapped LW filteroffset mapping), not jwst version and not CRDS
distortion content, was the actual cause for F410M. (The brick 2221 locked table was
rebuilt with a single both-module F410M row after the 2026-07-04 re-assign; the
band-aid split is gone.)

## Reference epochs (so propagation is reproducible)

- Gaia DR3 reference epoch = **2016.0**.
- VIRAC2 (VizieR II/387) reference epoch = **2014.0** (Smith+2025: *"fixed at the
  reference epoch, 2014.0"*). Not 2014.3, not 2016.0.
- The seed refcat (`build_gaia_virac2_refcat.py`) propagates each to the F115W
  observation epoch **2022.70** with per-source PM. `GAIA_EPOCH`/`VIRAC2_EPOCH`
  constants live at the top of that script.

---

_Last updated 2026-07-03 (added module-lock policy + F410M exception). See also `align_to_catalogs.py:sync_gwcs_to_fits_wcs`
docstring, the offsets-table builders (`_bench/build_sickle_gns_offsets.py`,
`scripts/miri_reduction/` registration scripts), and `f115w-astrometry-*`
analysis writeups in brick-jwst-2221._
