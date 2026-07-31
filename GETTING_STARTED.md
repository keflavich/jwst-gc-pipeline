# Getting started

This pipeline measures PSF photometry in crowded JWST fields. You give it
calibrated exposures of one target; it gives you a catalog of stars with
positions and fluxes in every filter you ran.

Work through it in three stages, in order:

| stage | what it does | what you run |
|---|---|---|
| **1. reduce** | mosaics the exposures (JWST Image3) and corrects their astrometry, writing one `*_crf.fits` per exposure | `python jwst_gc_pipeline/reduction/PipelineRerunNIRCAM-LONG.py` |
| **2. catalog** | detects stars and fits a PSF to each one, exposure by exposure, writing one catalog per exposure | `python -m jwst_gc_pipeline.photometry.crowdsource_catalogs_long` |
| **3. merge** | combines those catalogs across exposures, then across filters, into the catalog you use | `python -m jwst_gc_pipeline.photometry.merge_catalogs` |

Stage 1 is a script path rather than `python -m`, because `python -m` rejects
the hyphen in its filename.

Stage 1 differs by instrument; stages 2 and 3 are shared.

| instrument | stage-1 script | stages 2 and 3 |
|---|---|---|
| NIRCam | `jwst_gc_pipeline/reduction/PipelineRerunNIRCAM-LONG.py` — short and long filters both; the `-LONG` is left over from when it handled only the long ones | as above |
| MIRI | `jwst_gc_pipeline/reduction/PipelineMIRI.py` | add `--instrument=miri` |
| NIRISS | `jwst_gc_pipeline/reduction/PipelineRerunNIRISS.py` | add `--instrument=niriss` |

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
submit scripts export all four, so submitting through them leaves nothing to
set. To run the modules by hand (not using submission scripts that include
these variables), export:

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
├── catalogs/                   stage 3 writes merged catalogs here
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

— three digits before `_asn.json`, which is what a MAST filename has and a
hand-written one usually lacks. A real one:

```
jw02221-o001_20221007t121022_image3_007_asn.json
```

Finding none, stage 1 downloads the observation's association files from MAST.
It does that only when none is on disk.

The MIRI FOV region file is named `nircam_<target>_fov.reg` regardless of
instrument — the MIRI driver reuses the NIRCam footprint.

You no longer supply a `reduction/fwhm_table.ecsv`: the PSF width of each
filter is an instrument constant, so the pipeline reads the table that ships in
`jwst_gc_pipeline/reduction/`. A target directory that has its own copy still
uses it.

---

## A worked example, start to finish

This runs all three stages on a 20-arcsec cutout of the Brick in one filter, in
minutes rather than in a day. Everything lands in a scratch directory of your
own, well clear of released products.

```bash
export GC_BASEPATH_OVERRIDE=$HOME/scratch/brick-demo/
mkdir -p "$GC_BASEPATH_OVERRIDE"

# Stage 1 -- reduce.  -s reuses the *_cal files already on disk; drop it to
# re-fit the ramps from the raw data instead (much slower).
python jwst_gc_pipeline/reduction/PipelineRerunNIRCAM-LONG.py \
       -p 2221 -d 001 -f F410M -m merged -s

# Stage 2 -- catalog.  --each-suffix names the stage-1 products to photometer
# and carries the observation number.
python -m jwst_gc_pipeline.photometry.crowdsource_catalogs_long \
       --proposal_id=2221 --field=001 --target=brick \
       --filternames=F410M --modules=merged \
       --each-exposure --each-suffix=destreak_o001_crf \
       --cutout-region=266.5350,-28.7050,20

# Stage 3 -- merge.
python -m jwst_gc_pipeline.photometry.merge_catalogs \
       --target=brick --merge-singlefields
```

Results land in `$GC_BASEPATH_OVERRIDE/cutouts/<label>/catalogs/`. Drop
`--cutout-region` for the real thing, and expect stage 2 to take hours.

`GC_BASEPATH_OVERRIDE` redirects the basepath for all three stages. Leave it
unset to work in the target's own directory.

---

## Running on HiPerGator

The same three stages, as SLURM jobs. Stages 1 and 2 are arrays with one task
per filter; stage 3 is an array with one task per (program, filter).

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

# Stage 3 -- merge
sbatch --array=0-10 --export=ALL,TARGET=brick,MERGE_SINGLEFIELDS=1 \
       --job-name=brick2221-o001-merge \
       scripts/reduction/submit_merge.sbatch
```

Stage 3's array bounds come from the target's (program, filter) list. Print it
first, which runs nothing:

```bash
python -m jwst_gc_pipeline.photometry.merge_catalogs \
       --target=brick --merge-singlefields --list-jobs
```

Eleven lines for the Brick, so `--array=0-10`. Without `MERGE_SINGLEFIELDS=1`
the job does the all-filter merges only, and needs no array.

Stage 2 runs the whole photometry ladder in one job, ending with the cross-band
merge and a forced fill that measures each star in the filters where it went
undetected. For a field too large to finish in one walltime, the
`submit_cataloging_m8*.sbatch` scripts split that final fill across two jobs.

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

Every target is listed by hand in eight places for NIRCam — ten if you also run
MIRI — each keyed by target name or proposal id. In the order a run hits them:

| stage | registry | file |
|---|---|---|
| reduce | `field_to_reg_mapping` — proposal + obs → target | `reduction/PipelineRerunNIRCAM-LONG.py`, inside `if __name__` |
| reduce | `refnames` — astrometric reference catalog | `reduction/PipelineRerunNIRCAM-LONG.py` |
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

Four of these registries sit **inside a function**, so importing, printing or
overriding one means editing the source first.

Miss one and you get:

| missed | what happens |
|---|---|
| `field_to_reg_mapping` | `KeyError: '<proposal>'`, in whichever stage you reached first |
| `obs_filters` | the merge raises `TypeError` on `None` |
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

Two things stay absolute even under the override, so a full run off HiPerGator
still needs manual work:

- 32 files in the package contain a literal `/orange/adamginsburg` or
  `/blue/adamginsburg` path.
- The reference catalogs used for astrometry, and several diagnostic outputs,
  are among them.

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
run it with `pytest jwst_gc_pipeline`.
