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

| reference frame (`refname`) | offsets table | builder (tracked) |
|---|---|---|
| Gaia/VIRAC2 (brick/cloudc) | `Offsets_JWST_Brick<pid>_VIRAC2[locked].csv` | `build_gaia_virac2_refcat.py` (seed refcat) + the per-frame measure in the reduction |
| **GNS (sickle, prop 3958)** | `Offsets_JWST_Brick3958_GNS.csv` | **`_bench/build_sickle_gns_offsets.py`** |

**Sickle → GNS (`_bench/build_sickle_gns_offsets.py`):** a 2026-06-20 audit found
sickle catalogs sit at the **raw `assign_wcs` frame** (`RAOFFSET=0`, no offsets
table for 3958) — ~91 mas off the GNS reference the mosaics are tied to. The user
chose the GNS frame. This script measures the per-filter bulk sickle→GNS
correction and writes the per-frame table in the `shift_individual_catalog`
convention (`dra_table = corr_dRA_onsky / cos(dec)`). Its output
`Offsets_JWST_Brick3958_GNS.csv` is dropped into the sickle data tree's
`offsets/` dir and consumed by `fix_alignment` like any other table.
`_bench/sickle_gns_reduce_retry.sh` is the companion submitter that re-reduces
sickle with the GNS table once the group QOS frees and releases the held
cataloging jobs. (See the `PipelineRerunNIRCAM-LONG.py` ~L1140 note.)

**MIRI registration** is a separate, manual pre-step — MIRI does **not** use the
NIRCam offsets-table path. The sickle/cloudef MIRI frames are registered to the
NIRCam F480M frame by the scripts in `scripts/miri_reduction/` (see that
directory's README). They edit the per-frame FITS WCS / embedded gwcs in place
(idempotent via `MIRIDRA`/`MIRIDDE`/`MIRIWCSN`) and **must be run before
cataloging** a MIRI obs, or its mosaics sit ~3.3″ off truth while the catalog
underneath is correct.

## Reference epochs (so propagation is reproducible)

- Gaia DR3 reference epoch = **2016.0**.
- VIRAC2 (VizieR II/387) reference epoch = **2014.0** (Smith+2025: *"fixed at the
  reference epoch, 2014.0"*). Not 2014.3, not 2016.0.
- The seed refcat (`build_gaia_virac2_refcat.py`) propagates each to the F115W
  observation epoch **2022.70** with per-source PM. `GAIA_EPOCH`/`VIRAC2_EPOCH`
  constants live at the top of that script.

---

_Last updated 2026-06-25. See also `align_to_catalogs.py:sync_gwcs_to_fits_wcs`
docstring, the offsets-table builders (`_bench/build_sickle_gns_offsets.py`,
`scripts/miri_reduction/` registration scripts), and `f115w-astrometry-*`
analysis writeups in brick-jwst-2221._
