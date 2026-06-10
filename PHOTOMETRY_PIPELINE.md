# Photometry pipeline

This is **the** PSF-photometry pipeline (`jwst_gc_pipeline.photometry.cataloging`),
the default path as of 2026-06-09. It implements the recipe in
`PSFPhotometryPlan2026-06-09.md`. The previous `IterativePSFPhotometry` path is
retained only as an explicit opt-out (`--legacy-iterations`).

## Why this replaced `IterativePSFPhotometry`

photutils `IterativePSFPhotometry` (free position + LevMar + internal
re-detection) is numerically unstable for isolated bright stars: the centroid
walks and settles on an inflated-flux minimum whose **model peak exceeds the
data peak** (impossible for a single positive PSF; `qfit` does not catch it).
This pipeline makes every `daofind → fit → residual → reseed` step explicit,
uses single-pass BASIC `PSFPhotometry`, and adds a physical model/data-peak
overshoot check and a strict ban on negative-peak sources.

## How to run

Example (single filter):

```
python -m jwst_gc_pipeline.photometry.crowdsource_catalogs_long \
    --filternames=F480M --modules=nrcb \
    --proposal_id=3958 --field=007 --target=sickle \
    --each-suffix=destreak_o007_crf --each-exposure --daophot --skip-crowdsource \
    --group --max-group-size=10 --manual-group-min-sep-fwhm=3.0 \
    --cutout-region=/path/to/region.reg --cutout-label=my_cutout
```

Multi-filter (adds the cross-band iter7); one module token serves a SW and a LW
filter (e.g. `nrcb` → `nrcblong` for F480M and `nrcb1..4` for F210M):

```
    --filternames=F480M,F210M --modules=nrcb ...
```

`--cutout-region` accepts a DS9 `.reg` file or an `'ra,dec,size_arcsec'` string;
`--cutout-label` names the output tree under `<basepath>/cutouts/<label>/`.

Full-frame is the same command without `--cutout-region`/`--cutout-label`:

```
python -m jwst_gc_pipeline.photometry.crowdsource_catalogs_long \
    --filternames=F480M --modules=nrcb \
    --proposal_id=3958 --field=007 --target=sickle \
    --each-suffix=destreak_o007_crf --each-exposure --daophot --skip-crowdsource \
    --group --max-group-size=10 --manual-group-min-sep-fwhm=3.0
```

To use the old `IterativePSFPhotometry` path instead:

```
    --legacy-iterations ...
```

**Single in-process job (not a SLURM array).** Each phase detects on the
*previous* phase's merged residual mosaic, so the phases are strictly sequential
and the whole run executes in one process — full-frame or cutout. Run it as a
single (non-array) job: with `SLURM_ARRAY_TASK_ID` set the run aborts with a
message telling you to drop `--array` (or pass `--legacy-iterations` for the old
array-parallel per-exposure path). Full-frame outputs land in place under
`<basepath>/<FILTER>/pipeline/` and `<basepath>/catalogs/`; cutout runs are
namespaced under `<basepath>/cutouts/<label>/`.

## The iterations

`m12` runs per-frame iter1+iter2; `m3..m7` are merged-i2d-seeded passes. Source
*detection* uses a progressively cleaner co-add so sources hidden in one stage
surface in the next; the *fit* always runs on the raw or background-subtracted
**frames**, never on the i2d.

| iter | phase | detection co-add | frames fit | background |
|------|-------|------------------|------------|------------|
| 1,2  | m12   | per-frame data / residual | raw | — |
| 3    | m3    | raw merged i2d | raw | — |
| 4    | m4    | source-subtracted residual i2d | raw | builds first bg |
| 5    | m5    | residual i2d − background | bg-subtracted | recompute |
| 6    | m6    | residual i2d − background | bg-subtracted | recompute |
| 7    | m7    | cross-filter seed (multi-filter only) | bg-subtracted | — |

Each pass: build seed (previous vetted catalog ∪ daofind on the detection
co-add) → satstar-wing rejection + dedup → single-pass BASIC `PSFPhotometry` →
post-fit dedup → near-saturation/satstar-wing rejection → model/data-peak
overshoot QC → **drop non-positive-flux sources** → render model/residual. Then
merge across frames, tag provenance, vet, build the residual i2d and the
source-masked smoothed background fed to the next pass.

## Key behaviours

- **Negative-peak ban.** A PSF is strictly positive, so any fit with
  `flux_fit <= 0` is a negative-peak model. These are dropped at the fitter and
  excluded from seeds — they never enter a catalog and cannot breed more
  negatives at over-subtracted spots.
- **Overshoot QC + forced refit.** Fits whose rendered model peak exceeds
  `--manual-overshoot-ratio` × the local data peak are, by default, re-fit as
  forced photometry pinned at the trusted seed position (flux free), then the
  result is positivity-checked.
- **Grouping.** With `--group`, sources closer than
  `--manual-group-min-sep-fwhm` × FWHM are fit jointly. `min_separation` is a
  grouping *radius* (larger = group wider pairs); use ~3·FWHM to jointly fit
  close blends that otherwise over-subtract in the valley between them.
- **Source-masked background.** The smoothed background masks fitted source
  disks before smoothing, so it represents the diffuse background only and does
  not feed source-core holes back into the next fit.
- **`iter_found` column.** Every merged catalog records the first iteration
  (2..7) each source appears in, matched across phases by sky position.
- **Cross-band (iter7).** Multi-filter runs union the per-filter vetted m6
  catalogs into a cross-band seed. (The stringent ≥2-filter / <10 mas / S/N>5
  cut is a documented TODO; currently a plain union.)

## Outputs

Under `<basepath>/cutouts/<label>/`:

- `catalogs/<filt>_<module>_indivexp_merged[...]_m{N}_dao_basic.fits` — merged
  per-phase catalogs (`m2`,`m3`,`m4`,`_resbgsub_m5`,`_resbgsub_m6`,`_resbgsub_m7`),
  plus `_vetted` and `_allcols` variants. Carry `skycoord`, `flux`, `qfit`,
  `iter_found`, etc.
- `<filt>/pipeline/...-<module>_data_i2d.fits` — input data mosaic.
- `..._m{N}_..._mergedcat_residual_i2d.fits` — residual mosaic per phase
  (point-source models subtracted; saturated stars already removed via the
  per-frame satstar model).
- `..._m{N}_..._mergedcat_model_i2d.fits` — model mosaic per phase. **For display
  it includes the saturated-star model added back** on top of the fitted
  point-source model, so it overlays against the data; the residual above is
  unaffected (satstars are not re-subtracted).
- `..._m{N}_..._mergedcat_residual_smoothed_bg_i2d.fits` — background map.

The iteration tokens (`_m1.._m7`, `_dao_basic`) are disjoint from the legacy
`iter2/iter3/iter4`, `_daoiterative` products, so the two paths coexist in one
tree without collision.

## Flags (defaults)

The `--manual-*` flag names are retained for back-compatibility; the path they
control is the default.

| flag | default | meaning |
|------|---------|---------|
| `--manual-iterations` | on | use this pipeline (default) |
| `--legacy-iterations` | — | opt out, use the legacy `IterativePSFPhotometry` cutout path |
| `--manual-overshoot-ratio` | 1.2 | model-peak / local-data-peak flag threshold |
| `--manual-overshoot-action` | refit | `flag` \| `drop` \| `refit` (forced photometry at seed) |
| `--manual-iter2-local-snr` | 3.0 | local-S/N cut for residual-seeded passes |
| `--manual-ext-qfit-max` | 0.2 | extended-emission vetting: keep if qfit ≤ this |
| `--manual-ext-peak-over-bkg` | 20 | …or peak surface brightness > this × local bkg |
| `--manual-ext-local-snr-min` | 5.0 | …and local S/N ≥ this; also the i2d-detection S/N cut |
| `--manual-group-min-sep-fwhm` | 2.0 | grouping radius in FWHM (use ~3.0 for blends) |
| `--group` / `--max-group-size` | off / — | enable joint fitting; cap group size |

## Known limitations

- Runs as a single in-process job; it does not (yet) shard per-exposure fits
  across a SLURM array, so a large full-frame multi-filter run is long-but-linear
  in one process rather than array-parallel.
- Cross-band seed is a plain union (stringent multi-filter cut is a TODO).
- Faint sources blended on the wings of much brighter or saturated stars may be
  detected but dropped by the fit/dedup; resolving these needs fit-side
  deblending (forced photometry at detected positions), not detection changes.
