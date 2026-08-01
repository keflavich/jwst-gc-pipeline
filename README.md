# jwst-gc-pipeline

JWST imaging photometry for crowded fields. It reduces the exposures, ties them
to an absolute astrometric frame, fits PSF photometry exposure by exposure, and
merges the results into one multi-wavelength catalog. It was built for NIRCam,
MIRI and NIRISS observations of the Galactic Center — the Brick, Sgr B2,
Cloud C, Sgr A\* and their neighbours — and runs on any field registered in
`fields.yaml`.

## Install

See **[GETTING_STARTED.md](GETTING_STARTED.md)** — installation, the reference
data each stage needs, a worked end-to-end example, and what changes when you
run somewhere other than HiPerGator.

## Run it

One command reduces, catalogs and merges an observation:

```bash
python -m jwst_gc_pipeline.run_pipeline --proposal 2221 --obsid 001
```

Under SLURM it submits each stage to wait on the one before; otherwise it runs
them here, in order. Add `--dry-run` to print the commands and submit nothing.

To exercise the whole chain in minutes, give it a small region. This runs in
your shell:

```bash
python -m jwst_gc_pipeline.run_pipeline --proposal 2221 --obsid 001 \
       --filters F410M --cutout-region 266.5350,-28.7050,20
```

From Python, the same thing:

```python
from jwst_gc_pipeline.run_pipeline import run_pipeline
run_pipeline(proposal=2221, obsid=1)
```

## What comes out

Under the field's directory (`fields.yaml` says where each field lives):

| | |
|---|---|
| `<FILTER>/pipeline/*-merged_i2d.fits` | the mosaic for one filter |
| `<FILTER>/*_resbgsub_m*_daophot_basic.fits` | per-exposure catalogs, one per fitting stage |
| `catalogs/basic_merged_indivexp_photometry_tables_merged_resbgsub_m8_dedup.fits` | the multi-wavelength catalog — the science product |

## The two files you edit

- **[`jwst_gc_pipeline/fields.yaml`](jwst_gc_pipeline/fields.yaml)** — every
  target: its data directory, observations, filters, and reference catalog.
  Adding a target is an edit here and nothing else; running an unregistered one
  prints the block to paste. See [`docs/FIELDS.md`](docs/FIELDS.md).
- **[`jwst_gc_pipeline/config.yaml`](jwst_gc_pipeline/config.yaml)** — where and
  how it runs: account, QOS, per-stage CPUs, memory, walltime, and how far each
  stage fans out. It ships with HiPerGator's settings; copy it and point
  `GC_PIPELINE_CONFIG` at your copy to run elsewhere.

A NIRCam field also declares how it is aligned — which absolute frame, and where
its shifts come from — in
[`reduction/alignment_config.py`](jwst_gc_pipeline/reduction/alignment_config.py).
A field with no entry is reduced at the raw `assign_wcs` pointing and says so in
the log.

## The three stages

| | what it does | driver |
|---|---|---|
| **reduce** | JWST Image3 per filter, with the astrometric tie applied to each exposure before resampling. Writes `*_crf.fits` and the mosaic. | `reduction/PipelineRerunNIRCAM-LONG.py`, `PipelineMIRI.py`, `PipelineRerunNIRISS.py` |
| **catalog** | PSF photometry on every exposure, in stages m12 → m8: detect, fit, subtract, re-seed, then fill across bands. Saturated stars are fitted and removed here. | `photometry/crowdsource_catalogs_long.py` → `photometry/cataloging.py` |
| **merge** | Combines the filters into one catalog. | `photometry/merge_catalogs.py` |

[`PHOTOMETRY_PIPELINE.md`](PHOTOMETRY_PIPELINE.md) documents the fitting stages.

## Astrometry

Crowded-field astrometry has two rules that this package enforces in code, and
breaking either produces a catalog that looks right and is not:

1. Measure an offset by **stacking all pairwise offsets and taking the peak**
   (`photometry.astrometry_offsets.measure_offset`). Matching each source to its
   nearest neighbour and taking the median fails silently against a dense
   reference catalog.
2. Read a frame's **GWCS**, through
   [`frame_wcs()`](jwst_gc_pipeline/frame_wcs.py). The FITS SIP header beside it
   is a fitted approximation, wrong by 5–8 mas on older products.

Before changing anything that touches alignment, read
[`reduction/ASTROMETRY_WCS_CORRECTION_FLOW.md`](jwst_gc_pipeline/reduction/ASTROMETRY_WCS_CORRECTION_FLOW.md);
`CLAUDE.md` states the rules in full, with the failures that produced them.

## The rest of the package

`jwst_gc_pipeline.cmz` builds CMZ-wide products (catalog assembly, HiPS, HATS,
coverage maps); `versioning` decides whether a product needs regenerating;
`astrometry_gdc` holds distortion-correction experiments; `plotting` has
color-magnitude and extinction templates.

## License

BSD 3-clause. See `licenses/LICENSE.rst`.

This package was extracted from
[brick-jwst-2221](https://github.com/keflavich/brick-jwst-2221) so the pipeline
could be shared across Galactic Center programs; the Brick science analysis
stayed there.
