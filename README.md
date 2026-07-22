# jwst-gc-pipeline

A JWST photometry pipeline for crowded fields, originally developed for
NIRCam and MIRI observations of the Galactic Center (the Brick, Sgr B2,
Cloud C, Sgr A*, and similar fields). The pipeline is field-agnostic and
suitable for reduction and processing of any crowded JWST Galactic Center
program.

This package was extracted from
[brick-jwst-2221](https://github.com/keflavich/brick-jwst-2221) so that the
generic pipeline code could be shared with other Galactic Center JWST
projects, while the Brick-specific science analysis (ice analyses, CO
modeling, paper figures) remains in `brick2221`.

## Layout

- `jwst_gc_pipeline.reduction` — pipeline stages
  - `PipelineRerunNIRCAM-LONG.py` / `PipelineRerunNIRCAM-SHORT.py` — JWST
    `calwebb_image3` runners with tweakwcs modifications
  - `PipelineRerunF212N.py`, `PipelineMIRI.py` — additional pipeline runners
  - `destreak.py` — percentile-subtraction destriper for NIRCam horizontal
    quadrants
  - `align_to_catalogs.py` — astrometric re-alignment (VVV crossmatch and
    generic catalog matching)
  - `saturated_star_finding.py` — PSF fitting and removal of saturated stars
  - `filtering.py` — filter / FWHM / instrument utilities
  - `make_merged_psf.py` — gridded PSF construction
  - `merge_a_plus_b.py`, `realign_and_merge.py` — module merging and
    reprojection helpers
  - `run_notebook.py` — utilities

- `jwst_gc_pipeline.photometry` — catalog-level processing
  - `catalog_long.py` (also handles short) — crowdsource
    photometry extraction and the PSF-photometry driver / `main()`
  - `cataloging.py` — the PSF-photometry pipeline (Hosek-style iterative
    detect/fit/reseed); the default path. See `PHOTOMETRY_PIPELINE.md`.
  - `crowdsource_catalogs_short.py` — deprecated short-wave variant
  - `make_reftable.py` — astrometric reference table construction
  - `merge_catalogs.py` — multi-wavelength catalog merger

The default PSF-photometry pipeline is implemented in `cataloging.py` and
documented in [`PHOTOMETRY_PIPELINE.md`](PHOTOMETRY_PIPELINE.md). Pass
`--legacy-iterations` to use the old `IterativePSFPhotometry` path instead.

- `jwst_gc_pipeline.plotting` — generic plotting helpers
  - `plot_tools.py` — color-color, color-magnitude, extinction-vector
    templates

- `jwst_gc_pipeline.data` — small reference tables (FWHM lookup tables)

## Reduction process

1. `PipelineRerunNIRCAM-LONG.py` — run JWST `calwebb_image3` with modified
   tweakwcs. Uses `destreak.destreak`,
   `align_to_catalogs.realign_to_catalog`, and
   `saturated_star_finding.iteratively_remove_saturated_stars` internally.
2. `catalog_long.py` — extract long-wavelength catalogs.
3. `make_reftable.py` — build the F410M-based reference table for
   short-wavelength alignment (called from `merge_catalogs`).
4. `PipelineRerunNIRCAM-SHORT.py` — run pipeline on short-wave data using
   that reference catalog.
5. `catalog_long.py` — extract short-wave catalogs (the same
   module handles both).
6. `merge_catalogs.py` — merge multi-wavelength catalogs.

## Setup

For each field/program:
- Set up a `crds/` directory under your project's working directory (e.g.
  `/orange/adamginsburg/<field>/crds`).
- Provide a region file for selecting reference stars.
- Configure the field-name → proposal-ID mappings used by
  `merge_catalogs` and `align_to_catalogs`.

## Astrometric WCS corrections

Before touching any alignment / WCS code, read
[`jwst_gc_pipeline/reduction/ASTROMETRY_WCS_CORRECTION_FLOW.md`](jwst_gc_pipeline/reduction/ASTROMETRY_WCS_CORRECTION_FLOW.md).
It documents which files get WCS corrections, the reproducible `_cal` → mosaic/catalog
path, how double-correction is prevented, and the rule that per-exposure GWCS shifts
use `jwst.tweakreg.utils.adjust_wcs` (resampled-image GWCS has no STScI shifter).

## License

BSD 3-clause. See `licenses/LICENSE.rst`.
