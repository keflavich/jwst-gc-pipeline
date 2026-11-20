# MIRI F770W 4QPM imaging

`jwst_gc_pipeline.miri_qpm` is an independent workflow for MIRI F770W FULL imaging. It gathers local exposures, runs required JWST calibration stages, obtains an imager reference for each configured field, calibrates and de-smears the three 4QPM panes, and writes one combined imager-plus-pane FITS image per field.

The workflow is invoked separately from `run_pipeline`. It does not submit SLURM jobs or download observations. Fields are defined by explicit proposal/observation lists, so several observations can contribute to one field. Additional 10678 observations can be included in a new run by updating these lists and input folders.

## Installation and commands

Use a JWST environment; Python 3.12 is tested.

```bash
pip install -e '.[qpm,test]'
python -m jwst_gc_pipeline.miri_qpm plan --config examples/miri_qpm/fields.toml
python -m jwst_gc_pipeline.miri_qpm run --config examples/miri_qpm/fields.toml
```

If `setuptools_scm` cannot parse the checkout's Git tag, a local installation can use `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_JWST_GC_PIPELINE=0.0.0+qpm.local pip install -e '.[qpm,test]'`. Repository tag/version policy requires maintainer review.

The installed command is `gc-miri-qpm`. Configuration paths resolve relative to the TOML file. Edit the example's paths before running it.

| Command | Result |
| --- | --- |
| `plan` | Print the selected files and exclusions without writing products. |
| `gather` | Write a manifest of paired exposures from the input folders. |
| `prepare` | Gather, run required Detector1/Image2 stages, and obtain each field's imager reference. |
| `image` or `run` | Prepare and write each field's pane image with the selected calibration. |
| `calibrate` | Prepare and refit a separate candidate calibration bundle. |

Discovery selects MIRI `MIR_IMAGE`, `F770W`, `FULL` products. A rate file is required for the panes; an uncal file can supply it through Detector1. Image2 supplies a missing cal file. Conflicting copies of an exposure require narrower input roots; byte-identical copies are deduplicated. Background-subtracted cal files are excluded. Every requested observation must have usable inputs.

Each field can supply `reference`, a rectified celestial TAN imager mosaic. Otherwise Image3 builds it from the field's calibrated exposures. Standard JWST steps can be configured under `detector1.steps`, `image2.steps`, and `image3.steps`; CRDS must be configured for the JWST stages. Image3 works on separate copies, preserving the cal pixel values required by `median(cal/rate)`.

The pane workflow registers the detector images to the reference by default. Set `register=false` only when their GWCS already matches that reference. A newly built imager reference uses the configured Image3 astrometric frame; an absolute tie needs a suitable catalog or a reference produced by the main GC reduction.

## Imaging products

Each field writes `<field>/qpm/<field>_f770w_qpm_i2d.fits`, an exposure table, a JSON report, and a quicklook. Per-exposure pane FITS files are enabled by default. `products.json` lists the field images.

| Extension | Meaning |
| --- | --- |
| `SCI` | MJy/sr. Finite imager reference values take priority; panes fill uncovered pixels. |
| `ERR` | Propagated pixel errors. Calibration systematics and correlated noise are excluded. |
| `WHT` | Pane drizzle weight. |
| `SOURCE` | 0 empty; 1 imager; 2 pane coverage; 3 pane-filled imager hole; 4 bright hole left empty. |
| `FLAGS` | Bright/faint validity, RL edges, local corrections, exposure warnings, absent sky tie, saturation and fluence flags. |

Imaging uses the supplied R1 calibration unless `qpm.calibration_dir` names a bundle. Anchoring and dataset gain are off. Bright-core filling and flagged RL edge pixels are enabled. Point-source fluxes are not corrected for the measured deficit.

For the supplied calibration, pane point-source fluxes are **0.942 ± 0.019** of the imager fluxes relative to extended emission, about 6% low. This affects `SOURCE=2` and `SOURCE=3`; it does not change imager pixels (`SOURCE=1`). `QPMPSB` records the ratio, and `QPMPSBE` its systematic uncertainty. The error image does not include this uncertainty. A candidate refit's inherited point-source ratio is a diagnostic awaiting validation.

Bright-core filling remains enabled with flags. A saturated imager core filled by pane data (`SOURCE=3`) can appear unsaturated. Its value may exceed the supplied calibration's validated 30–450 MJy/sr range. Use the reference image or mask `SOURCE=3` for procedures that identify saturated sources by their missing cores. A finite pixel alone does not establish valid pane photometry.

`FLAGS` is a bit mask: a pixel can carry several bits. A bit is set when any contributing pane pixel has that property. Brightness limits are evaluated at the exposure's calibration level, including the sky offset; testing the final `SCI` value against 450 alone is insufficient.

| Bit | Meaning |
| --- | --- |
| 1 | Above 450 MJy/sr; outside the supplied validated range. |
| 2 | Richardson–Lucy edge margin; possible surface-brightness deficit. |
| 4 | Optional anchoring or dataset gain applied. |
| 8 | Below 30 MJy/sr; outside the supplied validated range. |
| 16 | Exposure warning, including a candidate calibration. |
| 32 | No measured imager sky offset to the reference. |
| 64 | Saturated rate ramp with a finite fitted slope; retained and unvalidated. |
| 128 | Rate × NGROUPS × TGROUP ≥ 40,000 DN; retained and unvalidated. |

`grid="footprint"` makes a field-sized image, including pane coverage. Several fields are processed independently. This workflow does not stitch separate field mosaics into a survey image. The pane engine can combine many exposures against one already combined reference mosaic.

A completed run checks that every selected exposure was processed, pane pixels reached the image, and reference science pixels were preserved. Reports retain exposure skips and scientific warnings. Unchanged runs can resume. Changed inputs/settings require a new `output_dir`, preventing a run from silently omitting newly arrived data.

## Recalibration

```bash
python -m jwst_gc_pipeline.miri_qpm calibrate --config examples/miri_qpm/refit.toml
```

Calibration uses the same local-file preparation path. Provide at least two training fields, at least four training visits, and a distinct field with `train=false` for validation. Each field requires an imager `reference` and a `starless` image on precisely the same grid. The star-subtracted images are inputs from photometry; this module does not fit a crowded-field star catalog. Choose independent sky fields and reference images for meaningful validation.

The fitter performs:

1. Isolated, unsaturated star selection and normalized 25×25 PSF stacking; non-negative 5×21 kernel fitting against the imager PSF.
2. Pane-star offset voting and an affine detector-coordinate geometry fit.
3. Diffuse overlap sampling with saturation, fluence, star-model, gradient and static bad-pixel cuts.
4. Pane-normalized alternating least squares for the flat, followed by relative affine response fitting on 8×8 blocks.
5. Field-cluster bootstrap uncertainties, visit-half flat uncertainty, and white-noise RL variance estimates.
6. Held-out diffuse, imager-control, geometry and PSF diagnostics, with sample counts.

Two fitting passes are the default. Inadequate constraints cause an explicit failure. Pixels with fewer than three clean flat samples receive their pane's median; `NSAMP` and the calibration report identify their support. The detector pane mask and static bad-pixel mask come from the selected seed bundle. The supplied point-source deficit remains a diagnostic and is explicitly marked as requiring validation for a candidate calibration.

The output contains `bundle/`, `calibration_report.json`, and fit evidence. The bundle has FITS flat/error/mask/region data, JSON response and geometry, NPZ kernels, and SHA256 checksums. Refitting writes to a new directory and never selects its result automatically. For imaging with a reviewed candidate, set `qpm.calibration_dir` to its `bundle/` and use a new image run directory. Products record the bundle checksum, recipe and candidate status; candidate pixels carry an exposure-warning flag.

The quoted R1 accuracy does not transfer automatically to a refit. Review held-out sample coverage, residuals, geometry, PSFs and flat support before scientific use. Empty brightness bins carry no validation result.

## Scientific behavior requiring review

- F770W FULL only. Other filters and subarrays have no supported calibration here.
- The supplied pane point-source flux ratio is 0.942 ± 0.019 relative to imager photometry. No correction is applied.
- The supplied validation range is 30–450 MJy/sr and NGROUPS 5–8. Brighter, saturated and high-fluence pixels can remain in the image with flags. Bright-hole filling can make a saturated star appear to have a measured core.
- RL edge pixels remain flagged; their surface brightness can be depressed. `edge_mode="drop"` removes them.
- Registration can fall back to a median solution from the input group, or to the cal GWCS with a warning. Missing sky offsets also produce flagged pixels. Inspect these warnings when assessing a field.
- Pane overlap measurements are QA diagnostics. The main pipeline's astrometric release gates still apply when preparing a scientific release.
- `ERR` excludes correlated RL/drizzle noise and calibration-law systematics. Candidate RL variance factors are white-noise simulation estimates.

## Credit

Nazar Budaiev proposed imaging with the MIRI 4QPM coronagraph panes and derived the R1 calibration shipped in `jwst_gc_pipeline/miri_qpm/data/F770W`.
