# Bulk astrometry update — correct a field's tie WITHOUT re-cataloging

**The savings:** cataloging (detection + per-frame PSF fitting + merge, ×6–7 stages)
is the expensive part of the pipeline — hours per filter. A bulk astrometric
correction touches *none* of it. If the cross-exposure (relative) astrometry is
already right, fixing the absolute tie is a **pure coordinate relabel**: add one
rigid `(dRA, dDec)` on-sky vector to every catalog's sky positions. Detector
`x_fit`/`y_fit`, fluxes, and all *relative* geometry are untouched. It takes
minutes.

This is the operational form of the versioning **`REPROJECT`** verdict
(`versioning/VERSIONING_PROVENANCE.md`): the product's science-data facet is
unchanged, only its WCS/coordinate facet moves.

Tool: `jwst_gc_pipeline/photometry/bulk_astrometry_update.py`.

---

## When this path is valid, and when it is not

Valid **iff the residual tie error is a single rigid offset over the whole field.**
That holds when the relative astrometry is internally consistent — every exposure
correctly registered to every other — so the only thing wrong is a global shift
of the whole mosaic against the absolute reference (VIRAC2/Gaia).

A residual that VARIES across the field (one part needs a different shift than
another) needs a **per-exposure re-tie + re-catalog** instead. A single rigid
shift there corrects one region and *breaks* another — the brick-1182 failure
mode (a bulk ≈ 0 hiding a shifted half-mosaic).

The tool **enforces** this: `measure_bulk_offset` runs the sanctioned per-tile
map (`astrometry_offsets.measure_offset_grid`) and raises
`NonUniformResidualError` if any tile's offset deviates from the global by more
than `--uniformity-tol-mas` (default 15). **Do not bypass the gate.**

Offsets are measured only with the density-immune offset-histogram estimator
(`measure_offset`, swept window). NN-median against a dense catalog is banned
(repo `CLAUDE.md`, ASTROMETRY RULE #1).

---

## Decide whether you even need it

```bash
# What does the provenance planner say for this field?
python -m jwst_gc_pipeline.versioning.rerun plan --field /path/to/field/catalogs
```

If the stages read **`REPROJECT`** (WCS moved, science data identical), this is
exactly the path to take. If they read `REFIT`/`RE_REDUCE`, the data changed and
the products must be refit.

---

## Usage

### A. Measure the offset and apply it (recommended)

```bash
python -m jwst_gc_pipeline.photometry.bulk_astrometry_update \
    --catalogs '/path/to/field/catalogs/*_m7*.fits' '/path/to/field/catalogs/*_m8*.fits' \
    --measure \
    --reference /path/to/virac2_or_gaia_only.fits \
    --measure-catalog /path/to/field/catalogs/f200w_nrca_m7.fits \
    --ref-ra-col RA --ref-dec-col DEC \
    --uniformity-tol-mas 15
```

It measures `(dRA, dDec)` (histogram-stacked, swept), **gates on per-tile
uniformity**, prints the tie diagnostics (offset, contrast, window, `swept`,
worst-tile deviation), then shifts every matched catalog by that one vector.
A `swept=True` or a large `window` means the field is grossly shifted — inspect
before trusting it.

### B. Apply a known offset (e.g. from a corrected offsets-table delta)

```bash
python -m jwst_gc_pipeline.photometry.bulk_astrometry_update \
    --catalogs '/path/to/field/catalogs/*_m7*.fits' \
    --dra-mas -17.5 --ddec-mas 4.2
```

Skips measurement and the uniformity gate — you are asserting the offset. Use
only when the offset is known-uniform.

### Options
- `--dry-run` — report what would change; write nothing.
- `--no-backup` — skip the `.pre_bulkastrom` sidecar copy (kept by default).
- `--no-restamp` — skip the provenance re-stamp.
- `--ra-col/--dec-col` — coordinate columns of the measured catalog (auto-detected otherwise).

---

## What it changes, and what it leaves alone

| Touched | Untouched |
|---|---|
| every sky-coordinate column (`skycoord` mixin, `ra`/`dec`, `skycoord_centroid_*`) — shifted by the SAME rigid vector | `x_fit`/`y_fit` (detector positions) |
| `.meta` provenance cards `ABULKDRA/ABULKDDE/ABULKUTC/ABULKMTH/ABULKREF` | all flux/PSF/quality columns |
| provenance sidecar (WCS facet; data facet unchanged) — best-effort re-stamp | relative geometry (all pairwise separations preserved exactly) |

The offset is applied with `astropy` `SkyCoord.spherical_offsets_by`, so the
cos(dec) handling is astropy's (hand-rolling it caused the ~69 mas cos(dec) bug).
A `.pre_bulkastrom` backup of each catalog is written by default (revertible).

---

## Provenance outcome

Each updated catalog keeps its recorded `stage` and gets re-stamped: the **data
facet is identical**, the **WCS/coordinate facet changes**, and the `ABULK*` meta
cards record the applied shift + UTC + method + reference. A subsequent
`rerun plan --field` therefore shows the stage as consistent (`SKIP` once the new
coordinates are the recorded state), and the change is fully traceable — you can
always diff against the previous tagged version or the `.pre_bulkastrom` backup.

> This path is a **post-hoc** correction of finished products (the `posthoc`
> WCS-change mode): it leaves the offsets table and all seeding as they are.
> Re-tying at **m12** (the `reseed` mode) changes seeding and requires
> re-cataloging downstream — a separate operation.

---

## Also updating the mosaics (optional, manual)

The tool updates catalogs. To carry the same correction into the `_i2d` mosaics,
nudge their WCS with the reduction's `adjust_wcs` path (a `CRVAL` shift), the
same primitive `fix_alignment` uses. The catalogs and mosaics then agree. The
catalogs are the science deliverable and the common case, so this step stays
manual.
