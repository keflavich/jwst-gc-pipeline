# Getting started

This pipeline turns JWST imaging of crowded fields into PSF-photometry
catalogs. Three stages, in order:

| stage | what it does | entry point |
|---|---|---|
| **reduce** | Image3 + astrometric alignment → `*_crf.fits` per exposure | `reduction/PipelineRerunNIRCAM-LONG.py` |
| **catalog** | detect + PSF-fit each exposure, iteratively → per-frame catalogs | `python -m jwst_gc_pipeline.photometry.crowdsource_catalogs_long` |
| **merge** | combine per-frame catalogs across exposures and filters | `python -m jwst_gc_pipeline.photometry.merge_catalogs` |

The catalog stage runs the whole phase ladder itself, including the m7
cross-band merge and the **m8** forced fill, which is on by default
(`--no-forced-fill-m8` turns it off). The `submit_cataloging_m8*.sbatch`
scripts are a walltime split for fields too big for one job — not an extra step
you have to remember.

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
export STPSF_PATH=/path/to/stpsf-data          # PSF models; required
export GC_ALLOW_DEV=1                          # see below -- you need this
```

**`GC_ALLOW_DEV=1` is not optional for a working checkout.** Both entry points
call `assert_runnable_version`, which refuses to run unless HEAD sits exactly on
a `YYYY-MM-DD_PR<n>` release tag with a clean tree:

```
UntaggedPipelineError: refusing to run a PRODUCTION stage on an untagged
or dirty tree
```

One local edit, or being a few commits past the last tag, is enough. The submit
scripts set it for you; running the modules directly, you must.

You also need a MAST token at `~/.mast_api_token` — the reduction driver opens
it unconditionally, before doing any work, on HiPerGator as well as off it.

`pip install -e '.[test]'` if you want to run the test suite. `STPSF_PATH` needs
the stpsf reference data, which is a separate download.

`STPSF_PATH` is required by name — the saturated-star finder raises without
it, and setting `WEBBPSF_PATH` instead does not satisfy that check.

There are no console scripts. Everything runs as `python -m <module>` or, for
the reduction driver, as a path — its filename contains a hyphen, so `python -m`
cannot load it.

## Data layout

Everything hangs off one **basepath** per target. Only `<FILTER>/pipeline/` is
created for you. Two of the rest are **inputs you must supply**; the others must
simply **exist**, empty — nothing creates them and you get a write error if they
are missing:

```
<basepath>/                     e.g. /orange/adamginsburg/jwst/sickle/
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

The FOV region file is a **MIRI** input. The NIRCam driver still looks it up,
but the value feeds the VVV realignment step that was retired in July 2026 and
is now read by nothing; a missing entry there is a silent no-op.

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
#    PROPOSAL, FIELD, TARGET, EACH_SUFFIX and MODULES travel together: set one
#    and you must set all five, or the script exits 64.  EACH_SUFFIX names the
#    reduced products to catalog and embeds the obs number.
sbatch --array=0-3 --export=ALL,PROPOSAL=2221,FIELD=001,TARGET=brick,\
EACH_SUFFIX=destreak_o001_crf,MODULES=merged,\
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

A new target is not configuration — it is a code change in **seven places** for
NIRCam, eight with MIRI, keyed by target or proposal id. In the order you hit
them:

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

**Three** of these live *inside functions* — they cannot be imported, inspected
or overridden, so you edit the source. `obs_filters` also has a second copy in
`reduction/make_merged_psf.py`. Miss `field_to_reg_mapping` and stage 1 dies
with a bare `KeyError`; miss `obs_filters` and the merge dies with a `TypeError`
on `None`.

One more worth knowing about, though it will not stop you: `refnames`
(`PipelineRerunNIRCAM-LONG.py`) is read with `.get()`, so an unregistered
proposal silently passes `refname=None` into the alignment step rather than
failing.

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

That is: NIRCam long-wavelength **writes** where you tell it. It is not yet a
clean end-to-end run off HiPerGator, because some **inputs** are still absolute:

- The MAST login noted under Install is unconditional — even with `-s`.
- Reference catalogs and several diagnostic paths are hard-coded under
  `/orange`.

For MIRI and NIRISS you must reduce elsewhere, or add the one
`apply_basepath_override` call the NIRCam driver has.

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
