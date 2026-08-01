# jwst-gc-pipeline

JWST imaging photometry for crowded fields. It reduces the exposures, ties them
to an absolute astrometric frame, fits PSF photometry exposure by exposure, and
merges the results into one multi-wavelength catalog. It was built for NIRCam,
MIRI and NIRISS observations of the Galactic Center — the Brick, Sgr B2,
Cloud C, Sgr A\* and their neighbours — and works on any field you register.

## Install

```bash
git clone https://github.com/keflavich/jwst-gc-pipeline.git
cd jwst-gc-pipeline
pip install -e .
```

**[GETTING_STARTED.md](GETTING_STARTED.md)** covers the rest: the reference data
each stage needs, a worked end-to-end example, and what changes when you run
somewhere other than HiPerGator (UF's HPC cluster, where this was written).

## Run it

The exposures have to be on disk already — this reduces and measures them, it
does not fetch them. Then one command reduces, catalogs and merges an
observation:

```bash
python -m jwst_gc_pipeline.run_pipeline --proposal 2221 --obsid 001
```

On a SLURM cluster each stage is submitted to wait on the one before; otherwise
they run here, in order. `--dry-run` prints the commands and submits nothing.

To try reduction and photometry on a small piece first, give it a region — a
DS9 region file, or RA, Dec and a size in arcseconds. This runs in your shell
and takes minutes:

```bash
python -m jwst_gc_pipeline.run_pipeline --proposal 2221 --obsid 001 \
       --filters F410M --cutout-region 266.5350,-28.7050,20
```

A cutout stops after cataloging.

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
| `<FILTER>/*_resbgsub_m*_daophot_basic.fits` | per-exposure catalogs, one per fitting pass |
| `catalogs/basic_merged_indivexp_photometry_tables_merged*.fits` | the multi-wavelength catalog |

The deepest catalog — `..._resbgsub_m8_dedup.fits`, which fits every band at
each source's position even where that band did not detect it — needs the
cross-band passes, and those need every filter cataloged in **one** job. The
default submits one job per filter, so it produces the merged catalog above but
not the cross-band one.
[`scripts/reduction/submit_cataloging_chain.sh`](scripts/reduction/submit_cataloging_chain.sh)
runs the per-filter jobs and then the cross-band pass over all of them.

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
| **reduce** | JWST's `calwebb_image3` mosaicking, one filter at a time. Each exposure's WCS is corrected to the reference frame first, so the mosaic and the per-exposure products inherit the same astrometry. | `reduction/PipelineRerunNIRCAM-LONG.py`, `PipelineMIRI.py`, `PipelineRerunNIRISS.py` |
| **catalog** | PSF photometry on every exposure, in five numbered passes (`m12`, `m3`…`m6`): detect, fit, subtract, re-seed on the residual. Saturated stars are fitted and removed here. Two further passes, `m7` and `m8`, work across bands and need every filter in one job. | `photometry/crowdsource_catalogs_long.py` → `photometry/cataloging.py` |
| **merge** | Combines each filter's per-exposure catalogs into one catalog per filter, then those into the multi-wavelength catalog. | `photometry/merge_catalogs.py` |

[`PHOTOMETRY_PIPELINE_BRIEF.md`](PHOTOMETRY_PIPELINE_BRIEF.md) describes each
pass and its parameters; [`PHOTOMETRY_PIPELINE.md`](PHOTOMETRY_PIPELINE.md) has
the flags and filenames.

## Astrometry

Crowded-field astrometry has two rules that this package enforces in code, and
breaking either produces a catalog that looks right and is not:

1. Measure an offset by **stacking all pairwise offsets and taking the peak**
   (`photometry.astrometry_offsets.measure_offset`), and map it per tile — one
   number for a whole field hides half a field being wrong. Matching each source
   to its nearest neighbour and taking the median fails silently against a dense
   reference catalog. Refining a small offset has its own rules; read
   `CLAUDE.md` before trusting a number.
2. Read a frame's **GWCS** — the exact distortion model in its ASDF extension —
   through [`frame_wcs()`](jwst_gc_pipeline/frame_wcs.py). The approximate WCS
   in the FITS header beside it (SIP) is off by 5–8 mas on older products.

Before changing anything that touches alignment, read
[`reduction/ASTROMETRY_WCS_CORRECTION_FLOW.md`](jwst_gc_pipeline/reduction/ASTROMETRY_WCS_CORRECTION_FLOW.md);
`CLAUDE.md` states the rules in full, with the failures that produced them.

## The rest of the package

`jwst_gc_pipeline.cmz` assembles survey-wide products across the Central
Molecular Zone; `versioning` decides whether a product needs regenerating;
`plotting` has color-magnitude and extinction templates.

To work on the pipeline itself, `pip install -e '.[test]'` and run
`pytest jwst_gc_pipeline tests`.

## License

BSD 3-clause. See `licenses/LICENSE.rst`.
