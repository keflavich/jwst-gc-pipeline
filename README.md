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
  - `PipelineRerunNIRCAM-LONG.py` — the NIRCam `calwebb_image3` runner (handles
    both long- and short-wavelength filters; TweakRegStep is skipped and the
    astrometric tie is applied per-exposure instead — see below)
  - `PipelineMIRI.py`, `PipelineRerunNIRISS.py` — the MIRI and NIRISS runners
  - `alignment_config.py` — **the per-field alignment registry**: which absolute
    reference frame and which shift source each `(proposal, observation)` uses
  - `unified_alignment.py` — resolves that declaration for one exposure
    (`resolve_shift`), applies it, and stamps the provenance header cards; the
    one path every NIRCam field goes through
  - `validate_offsets_table.py` — offsets-table sanity guards
    (`flag_collapsed_visits` / `assert_offsets_table_sane`)
  - `build_virac2_offsets.py`, `build_gaia_virac2_refcat.py`,
    `build_gaia_virac2_refcat_byquery.py` — reference-catalog and offsets-table
    builders
  - `bulk_offset_step0.py` — bulk-offset VERIFY step (importable + tested; not
    yet wired into the reduction)
  - `destreak.py` — percentile-subtraction destriper for NIRCam horizontal
    quadrants
  - `align_to_catalogs.py` — generic catalog matching. The post-resample mosaic
    realign (`realign_to_catalog` / `realign_to_vvv`) was retired 2026-07-11 and
    both names now raise `NotImplementedError`.
  - `saturated_star_finding.py` — saturated-star detection, PSF fitting and
    removal (driven from the cataloging stage, not from the reduction)
  - `satstar_deblend.py` — ZEROFRAME core deblending for crowded fields
  - `dva_correction.py`, `static_placement_correction.py`, `fits_wcs_sync.py` —
    inter-detector DVA, SIAF placement, and FITS↔GWCS header sync
  - `filtering.py` — filter / FWHM / instrument utilities
  - `make_merged_psf.py` — gridded PSF construction
  - `run_notebook.py` — utilities

- `jwst_gc_pipeline.photometry` — catalog-level processing
  - `crowdsource_catalogs_long.py` — CLI / `main()` for both short and long
    filters, plus the crowdsource path
  - `cataloging.py` — the PSF-photometry pipeline (explicit m12→m8 sequence of
    single-pass detect/fit/reseed stages); the default path. See
    `PHOTOMETRY_PIPELINE.md`.
  - `manual_defaults.py` — single source of truth for the tunable defaults
  - `astrometry_checkpoint.py`, `visit_consensus.py`, `measure_offsets.py`,
    `astrometry_offsets.py` — the in-pipeline astrometry failsafe ladder. See
    `photometry/ASTROMETRY_CHECKPOINTS.md`.
  - `interframe_overlap.py`, `registration_gate.py` — registration gates
  - `make_reftable.py` — astrometric reference table construction
  - `merge_catalogs.py` — multi-wavelength catalog merger
  - `forced_fill.py` — the m8 forced cross-band fill

The default PSF-photometry pipeline is implemented in `cataloging.py` and
documented in [`PHOTOMETRY_PIPELINE.md`](PHOTOMETRY_PIPELINE.md). Pass
`--legacy-iterations` to use the old `IterativePSFPhotometry` path instead.

- `jwst_gc_pipeline.astrometry_gdc` — geometric-distortion-correction
  experiments (`stdgdc.py`, `gdc_wcs.py`, `distortion_floor_diagnostic.py`)
- `jwst_gc_pipeline.cmz` — CMZ-wide products: catalog assembly, HiPS, HATS,
  coverage MOCs
- `jwst_gc_pipeline.versioning` — product versioning / provenance verdicts. See
  `versioning/VERSIONING_PROVENANCE.md`.
- `jwst_gc_pipeline.plotting` — generic plotting helpers
  - `plot_tools.py` — color-color, color-magnitude, extinction-vector
    templates
- `jwst_gc_pipeline.data` — small reference tables (FWHM lookup tables)

## Reduction process

1. `PipelineRerunNIRCAM-LONG.py` — run JWST `calwebb_image3`. Per exposure it
   calls `destreak.destreak` to write the working copy, then `fix_alignment`,
   which resolves this exposure's shift through
   `unified_alignment.resolve_shift` (driven by `alignment_config.py`) and bakes
   it into the GWCS. `TweakRegStep` is **skipped**: the tie is applied
   per-exposure, exactly once, so the `_crf` frames and the `_i2d` mosaic both
   inherit it. There is no post-resample mosaic realign.
2. `crowdsource_catalogs_long.py` → `cataloging.run_manual_pipeline` — the
   per-filter photometry stages (m12 → m6), the cross-band m7 merge and the m8
   forced fill. Saturated-star fitting/removal
   (`saturated_star_finding.remove_saturated_stars`) runs **here**, not in the
   reduction. The same module handles short- and long-wavelength filters.
3. `merge_catalogs.py` — merge multi-wavelength catalogs.
   `make_reftable.py` builds the F410M-based reference table and is called from
   `merge_catalogs`.

## Setup

For each field/program:
- Set up a `crds/` directory under your project's working directory (e.g.
  `/orange/adamginsburg/<field>/crds`).
- Provide a region file for selecting reference stars.
- Configure the field-name → proposal-ID mappings used by
  `merge_catalogs` and `align_to_catalogs`.
- **Declare the field's alignment in
  [`jwst_gc_pipeline/reduction/alignment_config.py`](jwst_gc_pipeline/reduction/alignment_config.py)**
  (reference frame + shift source). A NIRCam field with no entry is refused
  rather than silently left at the raw `assign_wcs` frame.

## Astrometric WCS corrections

Before touching any alignment / WCS code, read
[`jwst_gc_pipeline/reduction/ASTROMETRY_WCS_CORRECTION_FLOW.md`](jwst_gc_pipeline/reduction/ASTROMETRY_WCS_CORRECTION_FLOW.md).
It documents which files get WCS corrections, the reproducible `_cal` → mosaic/catalog
path, how double-correction is prevented, and the rule that per-exposure GWCS shifts
use `jwst.tweakreg.utils.adjust_wcs` (resampled-image GWCS has no STScI shifter).
`CLAUDE.md` carries the hard rules (never NN-median against a dense catalog; read
the GWCS, not the SIP header).

## License

BSD 3-clause. See `licenses/LICENSE.rst`.
