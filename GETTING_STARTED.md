# Getting started

This pipeline turns JWST imaging of crowded fields into PSF-photometry
catalogs. Three stages, in order:

| stage | what it does | entry point |
|---|---|---|
| **reduce** | Image3 + astrometric alignment → `*_crf.fits` per exposure | `reduction/PipelineRerunNIRCAM-LONG.py` |
| **catalog** | detect + PSF-fit each exposure, iteratively → per-frame catalogs | `python -m jwst_gc_pipeline.photometry.crowdsource_catalogs_long` |
| **merge** | combine per-frame catalogs across exposures and filters | `python -m jwst_gc_pipeline.photometry.merge_catalogs` |

**Read this first if you are not on HiPerGator: [Running elsewhere](#running-elsewhere)
— the honest answer is "partly".**

---

## Install

```bash
git clone https://github.com/keflavich/jwst-gc-pipeline
cd jwst-gc-pipeline
pip install -e .
```

Then set, in your shell or job script:

```bash
export CRDS_PATH=/somewhere/with/20GB          # JWST reference files land here
export CRDS_SERVER_URL=https://jwst-crds.stsci.edu
export STPSF_PATH=/path/to/stpsf-data          # PSF models (WEBBPSF_PATH on older installs)
```

`STPSF_PATH` is required — the saturated-star finder raises without it.

There are no console scripts. Everything runs as `python -m <module>` or, for
the reduction driver, as a path — its filename contains a hyphen, so `python -m`
cannot load it.

## Data layout

Everything hangs off one **basepath** per target. Stage 1 creates the rest:

```
<basepath>/                     e.g. /orange/adamginsburg/jwst/sickle/
├── F212N/                      one directory per filter, uppercase
│   └── pipeline/               reduced products: *_crf.fits, *_i2d.fits
├── catalogs/                   merged catalogs (the science output)
├── offsets/                    astrometric offset tables
├── reduction/                  FWHM tables and reduction bookkeeping
├── psfs/                       gridded PSF models
└── regions_/                   DS9 region files
```

Put your uncalibrated or `_cal` frames under `<basepath>/<FILTER>/pipeline/`
before running stage 1.

---

## On HiPerGator

The supported path. Submit the three stages as SLURM arrays, one task per
filter:

```bash
# 1. reduce — one array task per filter
sbatch --array=0-3 --export=ALL,PROPOSAL=2221,FIELD=001,FILTERS="F405N F410M F466N F212N" \
       --job-name=brick2221-o001-reduce \
       scripts/reduction/submit_reduction.sbatch

# 2. catalog — after stage 1 has written the *_crf.fits
sbatch --array=0-3 --export=ALL,PROPOSAL=2221,FIELD=001,TARGET=brick,\
FILTERS="F405N F410M F466N F212N" \
       --job-name=brick2221-o001-cat \
       scripts/reduction/submit_cataloging.sbatch

# 3. merge — one job, not an array
python -m jwst_gc_pipeline.photometry.merge_catalogs --target=brick --merge-singlefields
```

`--list-jobs` on stage 3 prints the array index → (program, filter) map, so you
can check `--array` bounds before submitting.

**Two conventions that are not optional:**

- Use `--account=astronomy-dept --qos=astronomy-dept-b`. The default
  `adamginsburg` QOS caps you at 10 CPUs and a 16-CPU task will sit PENDING
  forever. The submit scripts already set this; keep it if you write your own.
- Name jobs `<target><program>-o<obsid>-<stage>[-FILTER]` **at submit time**.
  The scripts rename themselves once a job *starts*, which is no help while it
  is queued — and queued is exactly when you are looking.

---

## Adding a new dataset

A new target is not configuration — it is a code change in **six places**. All
of them are dictionaries keyed by target or proposal id:

| what | where |
|---|---|
| filters per program | `obs_filters` in `photometry/merge_catalogs.py` |
| visits per program | `nvisits`, inside `main()` in `photometry/crowdsource_catalogs_long.py` |
| observation number | `project_obsnum` in `photometry/merge_catalogs.py` |
| astrometric offsets | `offsets_tables`, inside `main()` in `photometry/merge_catalogs.py` |
| alignment strategy | `ALIGNMENT_CONFIG` in `reduction/alignment_config.py` |
| starless-image targets | `TARGETS` in `photometry/make_starless_image.py` |

Plus the basepath itself, which is an `if target in (...)` branch in
`merge_catalogs.main()` choosing between two hard-coded roots.

Two of those dictionaries live *inside functions*, so they cannot be imported,
inspected, or overridden — you edit the source. `obs_filters` also exists in a
second copy in `reduction/make_merged_psf.py`.

> This is the single biggest obstacle to using the pipeline on a new field, and
> it is a known problem rather than a design. A registry that these six read
> from would make adding a target a data change instead of a patch.

---

## Running elsewhere

**Status: partial.** The pipeline was built on HiPerGator and still assumes it
in places. Be aware before you invest time:

- **32 files** in the package contain hard-coded `/orange/adamginsburg` or
  `/blue/adamginsburg` paths.
- `GC_BASEPATH_OVERRIDE=/your/data/target/` redirects the basepath. The
  NIRCam-LONG reduction driver, the cataloging driver and the merge honour it.
  The **MIRI and NIRISS reduction drivers ignore it** and write to `/orange`.
- A portability layer was proposed and closed unmerged (PR #98,
  `paths.py` / `JWST_GC_DATAROOT`). `scratch_basepath.py` still refers to it.

So today, off HiPerGator:

| stage | NIRCam long-wavelength | MIRI / NIRISS |
|---|---|---|
| reduce | redirects | **writes to `/orange/...`** |
| catalog | redirects | redirects |
| merge | redirects | redirects |

That is: you can run NIRCam long-wavelength end to end against your own tree.
For MIRI and NIRISS you must reduce elsewhere, or patch those two drivers the
way the NIRCam one is patched — one call to `apply_basepath_override`.

The SLURM scripts are HiPerGator-specific throughout — partition names, the
CRDS cache path, and an absolute path to one conda environment. Treat them as
worked examples, not portable tools.

---

## Where to go next

- [`PHOTOMETRY_PIPELINE_BRIEF.md`](PHOTOMETRY_PIPELINE_BRIEF.md) — what each
  photometry stage does and the parameters it uses. Start here.
- [`PHOTOMETRY_PIPELINE.md`](PHOTOMETRY_PIPELINE.md) — flags, filenames, output
  trees, the distributed fan-out.
- [`CLAUDE.md`](CLAUDE.md) — the rules that matter before you touch astrometry.
  The first one exists because ignoring it silently produced wrong answers more
  than once.
