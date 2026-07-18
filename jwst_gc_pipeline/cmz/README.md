# `jwst_gc_pipeline.cmz` — CMZ-wide release products

Growing two-color HiPS + shareable giant catalog. Implements
`scripts/release/CMZ_HIPS_AND_CATALOG_SHARING_PLAN.md`.

**Pure-Python is the default path.** The two-color pair is **F212N (blue) + F480M
(red)** — both CMZ-wide with program 10678; **F405N is a legacy per-field
fallback** where a field predates F480M coverage.

## Layers (complementary, not alternatives)

| Need | Module | Tool | Dep |
|---|---|---|---|
| CMZ-wide catalog (one table) | `catalog_assembly` | astropy | — |
| growing color mosaic (**incremental, pure-Python**) | `hips` | `reproject.hips` + Pillow | `reproject`, `Pillow` |
| WHERE data is (footprint) | `coverage_moc` | `mocpy` | `mocpy` |
| ANALYSE / cross-match at scale | `hats_export` | `hats-import` + LSDB | `hats-import` |
| VIEW catalog in Aladin | `hipsgen.build_catalog_hips` | `Hipsgen-cat.jar` (only piece with no pure-Python equivalent) | Java + jar |

`hipsgen.py` (Java) is **optional** — used for the progressive catalog HiPS, or as
an alternative image builder if you specifically want CDS Hipsgen.

## End-to-end (as program 10678 rolls in)

```bash
# 1. Assemble the CMZ-wide catalog from per-field combined catalogs
python -m jwst_gc_pipeline.cmz.catalog_assembly --spec fields.json --out cmz_catalog
#    fields.json: [{"path": ".../basic_merged_..._resbgsub_m7...fits",
#                   "field": "brick", "program": "2221", "obsid": "001"}, ...]
#    -> cmz_catalog.{fits,ecsv,parquet}  (+ cmz_field/program/obsid/src_tag, cmz_n_bands)

# 2. Coverage MOC (footprint tracker + drives incremental rebuilds)
python -m jwst_gc_pipeline.cmz.coverage_moc --i2d '.../ *-f212n-merged_i2d.fits' \
       --out cmz_f212n_coverage.fits

# 3. Two-color HiPS (pure-Python, incremental) — B=F212N, R=F480M
python - <<'PY'
from jwst_gc_pipeline.cmz import hips
# Grow each mono HiPS one field at a time (only overlapping master tiles rewrite):
hips.add_field_to_mono_hips('HiPS/F212N', ['.../brick-f212n-merged_i2d.fits'], 'brick')
hips.add_field_to_mono_hips('HiPS/F212N', ['.../sgrc-f212n-merged_i2d.fits'],  'sgrc')
hips.add_field_to_mono_hips('HiPS/F480M', ['.../sgrc-f480m-merged_i2d.fits'],  'sgrc')
# (for a legacy field lacking F480M, add its F405N to a separate HiPS/F405N and
#  point the red channel there for that region)
hips.derive_two_color_hips('HiPS/F212N', 'HiPS/F480M', 'HiPS/CMZ_color')
PY

# 4. HATS for LSDB (scalable) — analysis/cross-match
python -m jwst_gc_pipeline.cmz.hats_export --parquet cmz_catalog.parquet \
       --out hats/ --name cmz_jwst
# Catalog HiPS for Aladin (the one Java piece; no pure-Python generator exists):
#   export HIPSGENCAT_JAR=/path/Hipsgen-cat.jar
#   python -c "from jwst_gc_pipeline.cmz import hipsgen; \
#              hipsgen.build_catalog_hips('cmz_catalog.fits','HiPS/cmz_cat')"
```

## Incremental (pure-Python)

`add_field_to_mono_hips` builds the new field's HiPS in scratch and
`merge_hips_trees` folds it into the master **per order** (nan-aware combine).
reproject.hips already writes a correct all-order pyramid per field, so the merge
does **no** pyramid re-derivation (hence no HEALPix orientation math) and rewrites
only the master tiles the field overlaps. `derive_two_color_hips` is per-tile over
the shared tiles — cheap to re-derive at any stretch.

## Provenance

The two-color HiPS `properties` and the assembled catalog carry the pipeline tag
(`jwst_gc_pipeline.versioning`) and, per source, the field/program/obsid/tag it
came from — so a growing CMZ product stays traceable to the tagged runs behind it.
