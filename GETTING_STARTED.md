# Getting started

This pipeline turns JWST imaging of crowded fields into PSF-photometry
catalogs. Three stages, in order:

| stage | what it does | entry point |
|---|---|---|
| **reduce** | Image3 + astrometric alignment → `*_crf.fits` per exposure | `reduction/PipelineRerunNIRCAM-LONG.py` |
| **catalog** | detect + PSF-fit each exposure, iteratively → per-frame catalogs | `python -m jwst_gc_pipeline.photometry.crowdsource_catalogs_long` |
| **merge** | combine per-frame catalogs across exposures and filters | `python -m jwst_gc_pipeline.photometry.merge_catalogs` |

Running somewhere other than HiPerGator? Read
[Running elsewhere](#running-elsewhere) first: NIRCam long-wavelength works
against your own data tree today, and MIRI and NIRISS need one patch each.

---

## Install

```bash
git clone https://github.com/keflavich/jwst-gc-pipeline
cd jwst-gc-pipeline
pip install -e .
```

### On HiPerGator

Point at the group's existing caches rather than fetching your own. The submit
scripts export `CRDS_PATH`, `CRDS_SERVER_URL` and `GC_ALLOW_DEV` for you, so
submitting through them leaves one variable to set:

```bash
export STPSF_PATH=/orange/adamginsburg/jwst/stpsf-data
```

To run the modules by hand, also export:

```bash
export CRDS_PATH=/orange/adamginsburg/jwst/crds     # 147 GB, already populated
export CRDS_SERVER_URL=https://jwst-crds.stsci.edu
export GC_ALLOW_DEV=1
```

Set `PYTHON=/path/to/your/python` to choose the interpreter the submit scripts
use; they fall back to a site default.

### Anywhere else

```bash
export CRDS_PATH=/somewhere/with/room           # JWST reference files land here
export CRDS_SERVER_URL=https://jwst-crds.stsci.edu
export STPSF_PATH=/path/to/stpsf-data           # separate download; required
export GC_ALLOW_DEV=1                           # see below
```

**Set `GC_ALLOW_DEV=1` whenever you run from a working checkout.** Both entry
points call `assert_runnable_version`, which requires HEAD to sit exactly on a
`YYYY-MM-DD_PR<n>` release tag with a clean tree:

```
UntaggedPipelineError: refusing to run a PRODUCTION stage on an untagged
or dirty tree
```

One local edit, or a few commits past the last tag, triggers it. The submit
scripts export it; export it yourself when you call the modules directly.

You also need a MAST token at `~/.mast_api_token` — the reduction driver opens
it unconditionally, before doing any work, on HiPerGator as well as off it.

Run `pip install -e '.[test]'` to get the test suite. `STPSF_PATH` points at
the stpsf reference data, a separate download.

The saturated-star finder reads `STPSF_PATH` by name; `WEBBPSF_PATH` feeds a
different code path.

Run everything as `python -m <module>`. The one exception is the reduction
driver: its filename contains a hyphen, so give its path directly.

## Data layout

Everything hangs off one **basepath** per target. The pipeline creates
`<FILTER>/pipeline/`. You supply `reduction/fwhm_table.ecsv`, plus `offsets/`
and `regions_/` where your field needs them. `psfs/` and `catalogs/` must exist
as empty directories before the run — a known gap: the writers skip `makedirs`
and fail on a missing directory.

```
<basepath>/                     e.g. /path/to/data/sickle/
├── F212N/                      one directory per filter, uppercase
│   └── pipeline/               your _cal/_rate frames in, *_crf.fits out
├── reduction/fwhm_table.ecsv   INPUT.  Copy the one shipped in the package
├── offsets/                    INPUT for table-locked fields; see alignment_config
├── regions_/<inst>_<target>_fov.reg   INPUT for MIRI only (see below)
├── psfs/                       must exist; PSF models are written here
└── catalogs/                   must exist; merged catalogs land here
```

Before running stage 1:

- **Frames.** Put your `_cal` (or `_rate`) files in `<basepath>/<FILTER>/pipeline/`.
- **An Image3 association file** next to them. Stage 1 globs for
  `jw0<proposal>-o<obs>*_image3_*0[0-9][0-9]_asn.json` — note the three digits
  before `_asn.json`, which a hand-made name is unlikely to have. A real one
  looks like `jw02221-o001_20221007t121022_image3_007_asn.json`. Finding none,
  stage 1 queries MAST and starts downloading the whole program, with no warning
  that it is doing so.
- **The FWHM table.** Copy `jwst_gc_pipeline/reduction/fwhm_table.ecsv` from the
  package to `<basepath>/reduction/`. Stage 1 reads it before writing anything
  and stops at once if it is absent.
- **`mkdir psfs catalogs`.** The code writes into both without creating them
  (only cutout runs call `makedirs`).

The FOV region file is a **MIRI** input. (The NIRCam driver looks it up too,
but feeds it to a realignment step retired in July 2026, so an absent entry
there does nothing.)

---

## On HiPerGator

Submit each stage as a SLURM array with one task per filter:

```bash
# 1. reduce — one array task per filter
sbatch --array=0-3 --export=ALL,PROPOSAL=2221,FIELD=001,FILTERS="F405N F410M F466N F212N" \
       --job-name=brick2221-o001-reduce \
       scripts/reduction/submit_reduction.sbatch

# 2. catalog — after stage 1 has written the *_crf.fits
#    PROPOSAL, FIELD, TARGET, EACH_SUFFIX and MODULES travel together: set one
#    and you must set all five, or the script exits 64.  EACH_SUFFIX names the
#    reduced products to catalog and embeds the obs number.
sbatch --array=0-3 --export=ALL,PROPOSAL=2221,FIELD=001,TARGET=brick,\
EACH_SUFFIX=destreak_o001_crf,MODULES=merged,\
FILTERS="F405N F410M F466N F212N" \
       --job-name=brick2221-o001-cat \
       scripts/reduction/submit_cataloging.sbatch

# 3. merge — a single job
python -m jwst_gc_pipeline.photometry.merge_catalogs --target=brick --merge-singlefields
```

`--list-jobs` on stage 3 prints the array index → (program, filter) map, so you
can check `--array` bounds before submitting.

The catalog stage runs the whole phase ladder itself, including the m7
cross-band merge and the **m8** forced fill, which is on by default
(`--no-forced-fill-m8` turns it off). The `submit_cataloging_m8*.sbatch` scripts
split that work across two jobs for fields too large for one walltime.

The submit scripts already carry this group's SLURM account, QOS and
job-naming conventions. Writing your own scripts means carrying them too —
[`CLAUDE.md`](CLAUDE.md) states both rules.

---

## Adding a new dataset

Adding a target is a code change in **seven places** for NIRCam, eight with
MIRI, each keyed by target or proposal id. In the order you hit them:

| stage | what | where |
|---|---|---|
| reduce | proposal → obs → target | `field_to_reg_mapping`, inside `__main__` of `reduction/PipelineRerunNIRCAM-LONG.py` |
| reduce | alignment strategy | `ALIGNMENT_CONFIG` (a tuple of `FieldAlignment`) in `reduction/alignment_config.py` |
| reduce (MIRI) | target → FOV region file | `fov_regname` in `reduction/PipelineMIRI.py` |
| catalog | visits per program | `nvisits`, inside `main()` of `photometry/crowdsource_catalogs_long.py` |
| merge | filters per program | `obs_filters` in `photometry/merge_catalogs.py` |
| merge | observation number | `project_obsnum` in the same file |
| merge | astrometric offsets | `offsets_tables`, inside `main()` in the same file |
| starless | target list | `TARGETS` in `photometry/make_starless_image.py` |

Plus the basepath, an `if target in (...)` branch in `merge_catalogs.main()`
choosing between two hard-coded roots.

**Three** of these live *inside functions*, reachable only by editing the
source — importing, inspecting or overriding them requires moving them out
first. `obs_filters` also has a second copy in
`reduction/make_merged_psf.py`. Miss `field_to_reg_mapping` and stage 1 dies
with a bare `KeyError`; miss `obs_filters` and the merge dies with a `TypeError`
on `None`.

Known gap: `refnames` (`PipelineRerunNIRCAM-LONG.py`) is read with `.get()`, so
an unregistered proposal passes `refname=None` into the alignment step
silently.

> Known design flaw, and the largest single obstacle to a new field. A shared
> registry backing these entries would make adding a target a data change.
> Proposed in #220.

---

## Running elsewhere

**Status: partial.** The pipeline was built on HiPerGator and still assumes it
in places. Be aware before you invest time:

- **32 files** in the package contain hard-coded `/orange/adamginsburg` or
  `/blue/adamginsburg` paths.
- `GC_BASEPATH_OVERRIDE=/your/data/target/` redirects the basepath. The
  NIRCam-LONG reduction driver, the cataloging driver and the merge honour it.
  The **MIRI and NIRISS reduction drivers ignore it** and write to `/orange`.
- A portability layer exists as closed PR #98 (`paths.py` /
  `JWST_GC_DATAROOT`); `scratch_basepath.py` still references it.

So today, off HiPerGator:

| stage | NIRCam long-wavelength | MIRI / NIRISS |
|---|---|---|
| reduce | redirects | **writes to `/orange/...`** |
| catalog | redirects | redirects |
| merge | redirects | redirects |

NIRCam long-wavelength **writes** where you tell it. Two classes of **input**
remain absolute, so a full off-HiPerGator run still needs manual work:

- The MAST login noted under Install is unconditional — even with `-s`.
- Reference catalogs and several diagnostic paths are hard-coded under
  `/orange`.

For MIRI and NIRISS, reduce elsewhere or add the single
`apply_basepath_override` call the NIRCam driver already has.

The SLURM scripts carry HiPerGator specifics throughout — partition names, the
CRDS cache path, and one absolute conda path. Treat them as worked examples to
adapt.

---

## Where to go next

- [`PHOTOMETRY_PIPELINE_BRIEF.md`](PHOTOMETRY_PIPELINE_BRIEF.md) — start here:
  what each photometry stage does and the parameters it uses.
- [`PHOTOMETRY_PIPELINE.md`](PHOTOMETRY_PIPELINE.md) — flags, filenames, output
  trees, the distributed fan-out.
- [`CLAUDE.md`](CLAUDE.md) — the rules that matter before you touch astrometry.
  The first one exists because ignoring it silently produced wrong answers more
  than once.
