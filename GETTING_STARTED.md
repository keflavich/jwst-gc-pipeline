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
└── regions_/nircam_<target>_fov.reg    read by the NIRCam driver
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
field uses is registered in `fields.yaml`; stage 1
stops with a clear error naming the file it wanted. Both paths are relative to
the basepath, so they follow `GC_BASEPATH_OVERRIDE`.

The FOV region file is named `nircam_<target>_fov.reg` whatever the instrument.
Only the NIRCam driver reads it.

You no longer supply a `reduction/fwhm_table.ecsv`: the PSF width of each
filter is an instrument constant, so the pipeline reads the table that ships in
`jwst_gc_pipeline/reduction/`. A target directory that has its own copy still
uses it.

---

## One command

For a dataset that is registered in `fields.yaml`, this reduces, catalogs and
merges the whole observation, submitting each stage so it waits for the one
before:

```bash
python -m jwst_gc_pipeline.run_pipeline --proposal 2221 --obsid 001
```

From Python:

```python
from jwst_gc_pipeline.run_pipeline import run_pipeline
run_pipeline(proposal=2221, obsid=1)
```

Add `--dry-run` to see the `sbatch` lines without submitting. To try the whole
chain in minutes, give it a cutout — that runs in your shell rather than the
queue:

```bash
python -m jwst_gc_pipeline.run_pipeline --proposal 2221 --obsid 001 \
       --filters F410M --cutout-region 266.5350,-28.7050,20
```

**What the observation is** — target, filters, data directory, reference
catalog — comes from [`fields.yaml`](jwst_gc_pipeline/fields.yaml). A proposal
that is not registered stops with the block to add; see
[`docs/FIELDS.md`](docs/FIELDS.md).

**Where it runs** — account, QOS, CPUs, memory, walltime and how far each stage
fans out — comes from [`config.yaml`](jwst_gc_pipeline/config.yaml), which ships
with HiPerGator's settings. To run elsewhere, copy it and point
`GC_PIPELINE_CONFIG` at your copy:

```bash
cp jwst_gc_pipeline/config.yaml ~/my-cluster.yaml
export GC_PIPELINE_CONFIG=~/my-cluster.yaml
```

A copy need only contain what differs.

The rest of this page is what that command does, stage by stage, for when you
need to run one on its own.

---

## A worked example

Stages 1 and 2 on one filter of the Brick, writing to a scratch directory of
your own so nothing can touch released products. Measured on HiPerGator: about
18 minutes for stage 1 and 34 for stage 2, on a 20-arcsec cutout.

Stage 1 reads three things from the target directory besides the exposures: the
association file, the reference catalog, and (for this field) an offsets table.
`stage_scratch_basepath.sh` copies the inputs out of the real tree and nothing
the pipeline writes, so start with it:

```bash
# The scratch directory must end in the target's own name, so the staging
# script can tell it is staging `brick`.
export GC_BASEPATH_OVERRIDE=$HOME/scratch/demo/brick/
scripts/reduction/stage_scratch_basepath.sh \
    /blue/adamginsburg/adamginsburg/jwst/brick \
    "$GC_BASEPATH_OVERRIDE" reduce destreak_o001_crf F410M
```

It stages the exposures, the offsets table and the reference catalog. It leaves
the association file, so stage 1 fetches that from MAST — a minute, and the one
thing in this example that needs your MAST token.

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

It prints one line per array task, numbered from zero: eleven of them for the
Brick, so `--array=0-10`. Without `MERGE_SINGLEFIELDS=1` the job does the
all-filter merges only, which is what the second submission above is.

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
  default QOS has a much lower CPU cap, and a job that asks for more than the
  cap never starts.
- **Name the job at submit time**, as
  `<target><program>-o<obsid>-<stage>[-FILTER]`. Several reduce and catalog
  jobs are usually in flight at once, and a job renamed once it starts shows
  the generic name for exactly the hours it spends queued.

---

## Adding a new target

One file: [`jwst_gc_pipeline/fields.yaml`](jwst_gc_pipeline/fields.yaml). Add a
block, and the reduce, catalog and merge stages all pick the target up.

```yaml
  mynewfield:
    root: orange                       # which data tree; see `roots:` at the top
    observations:
      '9999':                          # the proposal id
        nvisits: 2
        reference_frame: VIRAC2        # names the offsets table and realign gate
        obsids:
          nircam: ['001']              # every observation of this field
        reference_catalog:             # what the astrometry ties TO
          '001': catalogs/gaia_virac2_refcat_epoch2025.5.fits
        filters: [f115w, f212n, f405n]
```

`run_pipeline` on an unregistered proposal prints this block with your
proposal and observation filled in, so the quickest way to start is to run it
and paste what it gives you.

Order in the file means nothing: the loader sorts proposals numerically and
filters by wavelength, so where you write an entry has no effect — including on
the SLURM array index. [`docs/FIELDS.md`](docs/FIELDS.md) describes every key.

**Two things live outside the registry.** `ALIGNMENT_CONFIG` in
`reduction/alignment_config.py` says *how* a field is aligned rather than what a
field is, and is already one typed registry with its own tests. And three
smaller target lists are only reached by the tool that reads them:
`stage_release.py`'s `FIELDS`, `make_starless_image.py`'s `TARGETS`, and
`build_virac2_offsets.py`'s `REGION`.

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

The SLURM scripts take `CRDS_PATH`, `CRDS_SERVER_URL`, `STPSF_PATH` and
`PYTHON` from the environment when they are set, so `config.yaml` supplies those
and no edit is needed. What is still written into the scripts is the partition
name and the `#SBATCH --output=` log path — SLURM reads directives before any
shell runs, so a variable there is never expanded. `slurm.log_dir` in
`config.yaml` covers the log path for anything the runner submits, by passing
`--output` on the command line; a script you submit by hand uses the directive.

---

## Where to go next

- [`PHOTOMETRY_PIPELINE_BRIEF.md`](PHOTOMETRY_PIPELINE_BRIEF.md) — what each
  photometry stage does, and the parameters it uses. Read this next.
- [`PHOTOMETRY_PIPELINE.md`](PHOTOMETRY_PIPELINE.md) — flags, filenames, output
  trees, how the work is distributed.
- [`docs/RACE_CONDITIONS.md`](docs/RACE_CONDITIONS.md) — what breaks when array
  tasks touch the same file at once, and the rule for writing one. Read it
  before adding anything that writes a path more than one task can reach.
- [`jwst_gc_pipeline/reduction/ASTROMETRY_WCS_CORRECTION_FLOW.md`](jwst_gc_pipeline/reduction/ASTROMETRY_WCS_CORRECTION_FLOW.md)
  — required reading before you change anything astrometric. Its rules exist
  because breaking them produced wrong answers that looked right.

To work on the pipeline itself, `pip install -e '.[test]'` adds the test suite;
run it with `pytest jwst_gc_pipeline tests`.
