# Getting started

This pipeline measures PSF photometry in crowded JWST fields. You give it
calibrated exposures of one target; it gives you a catalog of stars with
positions and fluxes in every filter you ran.

Work through it in three stages, in order:

| stage | what it does | what you run |
|---|---|---|
| **1. reduce** | mosaics the exposures (JWST Image3) and corrects their astrometry, writing one `*_crf.fits` per exposure | `python jwst_gc_pipeline/reduction/PipelineRerunNIRCAM-LONG.py` |
| **2. catalog** | detects stars and fits a PSF to each one, exposure by exposure, writing one catalog per exposure | `python -m jwst_gc_pipeline.photometry.crowdsource_catalogs_long` |
| **3. merge** | combines those catalogs across exposures, and assembles the all-filter products | `python -m jwst_gc_pipeline.photometry.merge_catalogs` |

Stage 1 differs by instrument; stages 2 and 3 are shared.

| instrument | stage-1 script | stage 2 adds | stage 3 |
|---|---|---|---|
| NIRCam | `reduction/PipelineRerunNIRCAM-LONG.py` — short and long filters both | `--modules=merged` | as above |
| MIRI | `reduction/PipelineMIRI.py` | `--instrument=miri --modules=mirimage` | `GC_INSTRUMENT_OVERRIDE=miri` |
| NIRISS | `reduction/PipelineRerunNIRISS.py` | `--instrument=niriss --modules=nis` | `GC_INSTRUMENT_OVERRIDE=niriss` |

Stage 3 takes the instrument from the environment rather than a flag.

Away from HiPerGator, stage 1 for NIRCam writes wherever you point it; MIRI and
NIRISS write to a hard-coded `/orange` path until someone patches them. See
[Running somewhere else](#running-somewhere-else).

---

## Install

```bash
git clone https://github.com/keflavich/jwst-gc-pipeline
cd jwst-gc-pipeline
pip install -e .
```

Four environment variables configure it.

| variable | what it is |
|---|---|
| `CRDS_PATH` | where JWST reference files are cached (tens of GB) |
| `CRDS_SERVER_URL` | `https://jwst-crds.stsci.edu` |
| `STPSF_PATH` | the [stpsf](https://stpsf.readthedocs.io/) reference data, a separate download |
| `GC_ALLOW_DEV` | set to `1` if you are working on an unreleased version, or if you make edits yourself |

### On HiPerGator

The caches already exist, so point at them rather than fetching your own. The
submit scripts export `CRDS_PATH`, `CRDS_SERVER_URL` and `GC_ALLOW_DEV`;
`STPSF_PATH` you set yourself either way. To run the modules by hand, export
all four:

```bash
export CRDS_PATH=/orange/adamginsburg/jwst/crds
export CRDS_SERVER_URL=https://jwst-crds.stsci.edu
export STPSF_PATH=/orange/adamginsburg/jwst/stpsf-data
export GC_ALLOW_DEV=1
```

Set `PYTHON=/path/to/your/python` to choose which interpreter the submit
scripts use; they fall back to a site default.

### Anywhere else

```bash
export CRDS_PATH=/somewhere/with/room
export CRDS_SERVER_URL=https://jwst-crds.stsci.edu
export STPSF_PATH=/path/to/stpsf-data
export GC_ALLOW_DEV=1
```

### A MAST token

Stage 1 needs one, whether or not it downloads anything. Get one from
<https://auth.mast.stsci.edu/token> and save it:

```bash
echo '<your-token>' > ~/.mast_api_token
chmod 600 ~/.mast_api_token
```

---

## Set up a target directory

One directory holds everything for one target: the exposures going in, the
mosaics and per-exposure products in the middle, and the catalogs coming out.
The code calls it the **basepath**.

```
<basepath>/                     e.g. /orange/adamginsburg/jwst/sickle/
├── F212N/                      one directory per filter, uppercase
│   └── pipeline/               put your exposures here; stage 1 writes *_crf.fits here
├── psfs/                       stage 2 writes PSF models here
├── catalogs/                   stages 2 and 3 write catalogs here; also holds
│                               the reference catalog you supply
├── offsets/                    only for fields aligned from a table (see alignment_config.py)
└── regions_/nircam_<target>_fov.reg    only for MIRI
```

The pipeline creates `psfs/` and `catalogs/`. You supply two things:

**The exposures.** Put the `_cal` files (or `_rate`, if you want stage 1 to
re-fit the ramps) in `<basepath>/<FILTER>/pipeline/`.

**An Image3 association file** in the same directory. This is the `.json` that
lists which exposures belong to one mosaic; MAST ships one with every
observation, so if you downloaded the data from MAST you already have it. Stage
1 looks for

```
jw0<proposal>-o<obs>*_image3_*0[0-9][0-9]_asn.json
```

That pattern needs a run of digits before `_asn.json`, which MAST filenames have
and a hand-written name usually lacks. A real one:

```
jw02221-o001_20221007t121022_image3_007_asn.json
```

Finding none, stage 1 downloads the observation's association files from MAST.
It does that only when none is on disk.

**A reference catalog** at `<basepath>/catalogs/`, and, for a field aligned from
a table, **an offsets table** at `<basepath>/offsets/`. Which reference catalog a
field uses is registered in `REFERENCE_ASTROMETRIC_CATALOG_BY_FIELD`; stage 1
stops with a clear error naming the file it wanted. Both paths are relative to
the basepath, so they follow `GC_BASEPATH_OVERRIDE`.

The FOV region file is named `nircam_<target>_fov.reg` whatever the instrument.
Only the NIRCam driver reads it.

You no longer supply a `reduction/fwhm_table.ecsv`: the PSF width of each
filter is an instrument constant, so the pipeline reads the table that ships in
`jwst_gc_pipeline/reduction/`. A target directory that has its own copy still
uses it.

---

## A worked example

Stages 1 and 2 on one filter of the Brick, writing to a scratch directory of
your own so nothing can touch released products. Measured on HiPerGator: about
18 minutes for stage 1 and 34 for stage 2, on a 20-arcsec cutout.

Stage 1 reads three things from the target directory besides the exposures: the
association file, the reference catalog, and (for this field) an offsets table.
`stage_scratch_basepath.sh` copies exactly those out of the real tree and
nothing the pipeline writes, so start with it:

```bash
export GC_BASEPATH_OVERRIDE=$HOME/scratch/brick-demo/
scripts/reduction/stage_scratch_basepath.sh \
    /blue/adamginsburg/adamginsburg/jwst/brick \
    "$GC_BASEPATH_OVERRIDE" reduce destreak_o001_crf F410M
```

Then:

```bash
# Stage 1 -- reduce.  -s reuses the *_cal files on disk; drop it to re-fit the
# ramps from the raw data instead, which is much slower.
python jwst_gc_pipeline/reduction/PipelineRerunNIRCAM-LONG.py \
       -p 2221 -d 001 -f F410M -m merged -s

# Stage 2 -- catalog.  --each-suffix names the stage-1 products to photometer
# and carries the observation number.  --cutout-region keeps it to minutes.
python -m jwst_gc_pipeline.photometry.crowdsource_catalogs_long \
       --proposal_id=2221 --field=001 --target=brick \
       --filternames=F410M --modules=merged \
       --each-exposure --each-suffix=destreak_o001_crf \
       --cutout-region=266.5350,-28.7050,20
```

Stage 1 writes 48 `*_destreak_o001_crf.fits` into
`$GC_BASEPATH_OVERRIDE/F410M/pipeline/`. Stage 2 writes per-exposure catalogs
into `$GC_BASEPATH_OVERRIDE/cutouts/<label>/catalogs/`.

**Stage 3 needs the whole filter set, so the one-filter example stops here.**
The merge combines filters, and it looks in `<basepath>/catalogs/` rather than
under `cutouts/`. Run it after a full-field, all-filter stage 2 — see
[Running on HiPerGator](#running-on-hipergator). The same is true of the last two
phases of stage 2: the cross-band merge and the forced fill both need at least
two filters, so a one-filter run ends after the per-exposure phases.

`GC_BASEPATH_OVERRIDE` redirects the basepath for all three stages. Leave it
unset to work in the target's own directory.

---

## Running on HiPerGator

The same three stages, as SLURM jobs. Stages 1 and 2 are arrays with one task
per filter. Stage 3 is an array with one task per (program, filter), followed by
a single job that combines the filters.

```bash
# Stage 1 -- reduce
sbatch --array=0-3 \
       --export=ALL,PROPOSAL=2221,FIELD=001,FILTERS="F405N F410M F466N F212N" \
       --job-name=brick2221-o001-reduce \
       scripts/reduction/submit_reduction.sbatch

# Stage 2 -- catalog, once stage 1 has written the *_crf.fits.
# PROPOSAL, FIELD, TARGET, EACH_SUFFIX and MODULES travel together: set one and
# you must set all five, or the script exits 64.
sbatch --array=0-3 \
       --export=ALL,PROPOSAL=2221,FIELD=001,TARGET=brick,EACH_SUFFIX=destreak_o001_crf,MODULES=merged,FILTERS="F405N F410M F466N F212N" \
       --job-name=brick2221-o001-cat \
       scripts/reduction/submit_cataloging.sbatch

# Stage 3 -- merge.  Two submissions: an array for the per-filter merges, then
# one job that combines the filters, after the whole array has finished.
jid=$(sbatch --parsable --array=0-10 \
      --export=ALL,TARGET=brick,MERGE_SINGLEFIELDS=1 \
      --job-name=brick2221-o001-merge \
      scripts/reduction/submit_merge.sbatch)
sbatch --dependency=afterok:$jid --export=ALL,TARGET=brick \
       --job-name=brick2221-o001-mergeall \
       scripts/reduction/submit_merge.sbatch
```

Stage 1's array leaves `MODULES` at its default, `nrca,nrcb,merged` — three
mosaics per filter, where the worked example above builds only `merged`. Set
`MODULES=merged` to match it.

Stage 3's array bounds come from the target's (program, filter) list. Print it
first:

```bash
python -m jwst_gc_pipeline.photometry.merge_catalogs \
       --target=brick --merge-singlefields --list-jobs
```

Eleven lines for the Brick, so `--array=0-10`. Without `MERGE_SINGLEFIELDS=1`
the job does the all-filter merges only, which is what the second submission
above is.

Stage 2 runs the per-exposure phases and, when given two or more filters, the
cross-band merge that follows them. The forced fill that measures each star in
the filters where it went undetected is a separate invocation
(`--manual-start-phase=m8`); the `submit_cataloging_m8*.sbatch` scripts run it,
split across two jobs for a field too large for one walltime.

### What the jobs ask for

| stage | script | CPUs | memory | walltime |
|---|---|---|---|---|
| reduce | `submit_reduction.sbatch` | 16 | 128 GB | 24 h |
| catalog | `submit_cataloging.sbatch` | 32 | 128 GB | 48 h |
| merge | `submit_merge.sbatch` | 4 | 64 GB | 8 h |

Those are per array task, and the catalog defaults are sized for the Brick and
Sgr C — the densest fields in the survey, ~78k sources per exposure. A sparse
field finishes far inside them.

Two conventions the submit scripts already follow, and that you should carry
into any script of your own:

- **Submit with `--account=astronomy-dept --qos=astronomy-dept-b`.** The
  default `adamginsburg` QOS caps a job at 10 CPUs, so a 16- or 32-CPU task
  submitted under it stays PENDING forever.
- **Name the job at submit time**, as
  `<target><program>-o<obsid>-<stage>[-FILTER]`. Several reduce and catalog
  jobs are usually in flight at once, and a job renamed once it starts shows
  the generic name for exactly the hours it spends queued.

---

## Adding a new target

Every target is listed by hand in nine places for NIRCam — eleven if you also
run MIRI — each keyed by target name or proposal id. In the order a run hits
them:

| stage | registry | file |
|---|---|---|
| reduce | `field_to_reg_mapping` — proposal + obs → target | `reduction/PipelineRerunNIRCAM-LONG.py`, inside `if __name__` |
| reduce | `refnames` — astrometric reference frame token | `reduction/PipelineRerunNIRCAM-LONG.py` |
| reduce | `REFERENCE_ASTROMETRIC_CATALOG_BY_FIELD` — which reference catalog file | `reduction/PipelineRerunNIRCAM-LONG.py` |
| reduce | `ALIGNMENT_CONFIG` — how the field is aligned | `reduction/alignment_config.py` |
| reduce (MIRI) | `fov_regname` — target → FOV region file | `reduction/PipelineMIRI.py` |
| reduce (MIRI) | `field_to_reg_mapping` again | `reduction/PipelineMIRI.py`, inside `if __name__` |
| catalog | `field_to_reg_mapping` again | `photometry/crowdsource_catalogs_long.py`, inside `main()` |
| catalog | `nvisits` — visits per program | `photometry/crowdsource_catalogs_long.py`, inside `main()` |
| merge | `obs_filters` — filters per program | `photometry/merge_catalogs.py` |
| merge | `project_obsnum` — observation number | `photometry/merge_catalogs.py` |
| merge | `offsets_tables` — astrometric offsets | `photometry/merge_catalogs.py`, inside `main()` |

Plus the basepath: two `if target in (...)` branches, one in the catalog driver
and one in the merge, choosing between `/orange` and `/blue`. Keep them
identical. They disagreed on `wd1` and `wd2` until 2026-07-31: the catalog
stage wrote to `/orange` while the merge read from `/blue`.

Three sit **inside a function** and two more inside an `if __name__` block, so
importing, printing or overriding them means editing the source first.

Miss one and you get:

| missed | what happens |
|---|---|
| `field_to_reg_mapping` | `KeyError: '<proposal>'`, in whichever stage you reached first |
| `REFERENCE_ASTROMETRIC_CATALOG_BY_FIELD` | `KeyError: No reference catalog mapping configured for proposal_id=...` |
| `obs_filters` | the merge raises `AttributeError: 'NoneType' object has no attribute 'items'` |
| `refnames` | nothing visible — it is read with `.get()`, so the alignment step quietly receives `None` |

Two more registries duplicate this information and are worth knowing about:
`obs_filters` and `obs_ids` in `reduction/make_merged_psf.py`, and
`obs_filters_niriss` in `photometry/merge_catalogs.py` for NIRISS runs. Also
`TARGETS` in `photometry/make_starless_image.py` if you want starless images.

> Known design flaw, and the biggest obstacle to adding a field. One shared
> registry would turn all of this into a data change. Proposed in #220.

---

## Running somewhere else

The pipeline was built on HiPerGator and still assumes it in places. What
works today:

| stage | NIRCam | MIRI / NIRISS |
|---|---|---|
| reduce | writes where you point it | writes to `/orange/adamginsburg/...` |
| catalog | writes where you point it | writes where you point it |
| merge | writes where you point it | writes where you point it |

`GC_BASEPATH_OVERRIDE=/your/data/target/` sets the basepath. The NIRCam
reduction driver, the catalog driver and the merge all honour it. The MIRI and
NIRISS reduction drivers build their basepath directly and ignore it; giving
them the one `apply_basepath_override` call the NIRCam driver already has would
fix that.

32 files in the package contain a literal `/orange/adamginsburg` or
`/blue/adamginsburg` path, so a full run off HiPerGator still needs manual work.
The reference catalogs are not among them: those are basepath-relative and
follow the override, which makes them inputs you have to supply rather than
paths you have to patch.

A portability layer was proposed in #98 and closed unmerged; `scratch_basepath.py`
still refers to it.

The SLURM scripts carry HiPerGator specifics throughout — partition names, the
CRDS cache path, one absolute conda path. Treat them as worked examples to
adapt.

---

## Where to go next

- [`PHOTOMETRY_PIPELINE_BRIEF.md`](PHOTOMETRY_PIPELINE_BRIEF.md) — what each
  photometry stage does, and the parameters it uses. Read this next.
- [`PHOTOMETRY_PIPELINE.md`](PHOTOMETRY_PIPELINE.md) — flags, filenames, output
  trees, how the work is distributed.
- [`jwst_gc_pipeline/reduction/ASTROMETRY_WCS_CORRECTION_FLOW.md`](jwst_gc_pipeline/reduction/ASTROMETRY_WCS_CORRECTION_FLOW.md)
  — required reading before you change anything astrometric. Its rules exist
  because breaking them produced wrong answers that looked right.

To work on the pipeline itself, `pip install -e '.[test]'` adds the test suite;
run it with `pytest jwst_gc_pipeline tests`.
