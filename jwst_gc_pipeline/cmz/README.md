# `jwst_gc_pipeline.cmz` — CMZ-wide release products

Growing two-color HiPS + shareable giant catalog. Implements
`scripts/release/CMZ_HIPS_AND_CATALOG_SHARING_PLAN.md`.

## Layers (complementary, not alternatives)

| Need | Module | Tool | Dep |
|---|---|---|---|
| CMZ-wide catalog (one table) | `catalog_assembly` | astropy | — |
| growing color mosaic | `hips` | `reproject.hips` + Pillow | `reproject`, `Pillow` |
| — native incremental / RGB / Allsky | `hipsgen` | CDS `Hipsgen.jar` | Java + jar |
| WHERE data is (footprint) | `coverage_moc` | `mocpy` | `mocpy` |
| VIEW catalog in Aladin | `hipsgen.build_catalog_hips` | `Hipsgen-cat.jar` | Java + jar |
| ANALYSE / cross-match at scale | `hats_export` | `hats-import` + LSDB | `hats-import` |

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

# 3a. Two-color HiPS (pure-python; no Java) from two mono HiPS
python - <<'PY'
from jwst_gc_pipeline.cmz import hips
hips.build_mono_hips(['.../brick-f212n-merged_i2d.fits', '.../sgrc-f212n-...fits'],
                     'HiPS/F212N')                 # blue substrate (F212N)
hips.build_mono_hips(['.../sgrc-f480m-...fits'],    'HiPS/LONG')  # F480M/F405N
hips.derive_two_color_hips('HiPS/F212N', 'HiPS/LONG', 'HiPS/CMZ_color')
PY

# 3b. OR native-incremental image HiPS + RGB via CDS Hipsgen (Java)
export HIPSGEN_JAR=/path/Hipsgen.jar
python - <<'PY'
from jwst_gc_pipeline.cmz import hipsgen
hipsgen.build_mono_hips('fits_in/F212N', 'HiPS/F212N')   # re-run to grow
PY

# 4. Catalog HiPS for Aladin (Java) + HATS for LSDB
export HIPSGENCAT_JAR=/path/Hipsgen-cat.jar
python -m jwst_gc_pipeline.cmz.hats_export --parquet cmz_catalog.parquet \
       --out hats/ --name cmz_jwst
```

## Incremental note

`hips.build_mono_hips` (reproject.hips) rebuilds from the member list —
`MemberRegistry` tracks members so you re-run with the full set. **Native
incremental** tiling (rewrite only affected tiles) is CDS Hipsgen (`hipsgen.py`);
the plan recommends it for the survey substrate. Two-color derivation
(`derive_two_color_hips`) is per-tile over the shared tiles — cheap to re-derive.

## Provenance

The two-color HiPS `properties` and the assembled catalog carry the pipeline tag
(`jwst_gc_pipeline.versioning`) and, per source, the field/program/obsid/tag it
came from — so a growing CMZ product stays traceable to the tagged runs behind it.
