# Photometry pipeline

<!--
  DOC SYNC — last reviewed 2026-06-28 (added m8 forced cross-band fill; documented
  the per-frame distributed fan-out + monolith equivalence; cross-band seed is a
  deduped union, no longer a "plain union").  Prior review 2026-06-10, commit 17d0861.
  This is the IMPLEMENTATION-facing doc (flags, tokens, file names, behaviours).
  Its science-narrative companion is PIPELINE_METHODS.md (publication-style, no
  code). KEEP THE TWO IN SYNC: when the algorithm changes, update BOTH and bump
  the stamp in BOTH. PIPELINE_METHODS.md = "what & why"; this file = "how & where".
-->

This is **the** PSF-photometry pipeline (`jwst_gc_pipeline.photometry.cataloging`),
the default path as of 2026-06-09. It implements the recipe in
`PSFPhotometryPlan2026-06-09.md`. The previous `IterativePSFPhotometry` path is
retained only as an explicit opt-out (`--legacy-iterations`). For the
science-method narrative (publication style, no code), see
[`PIPELINE_METHODS.md`](PIPELINE_METHODS.md).

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

**Monolithic in-process run (the simple path).** Each phase detects on the
*previous* phase's merged residual mosaic, so the phases are strictly sequential.
The simplest way to run is one process that executes every phase in order —
full-frame or cutout. Run it as a single (non-array) job: with
`SLURM_ARRAY_TASK_ID` set the run aborts with a message telling you to drop
`--array` (or pass `--legacy-iterations` for the old array-parallel per-exposure
path). Full-frame outputs land in place under `<basepath>/<FILTER>/pipeline/`
and `<basepath>/catalogs/`; cutout runs are namespaced under
`<basepath>/cutouts/<label>/`.

**Distributed per-frame fan-out (recommended at scale).** The monolith is a
single long job and is recommended *against* for large full-frame multi-filter
runs (one 32-core/48 h job sits a long time waiting for a big node). The same
pipeline can instead be sharded below the filter boundary: for each phase, many
tiny per-frame "fan-out" worker jobs fit a frame shard, then one "finalize"
barrier job merges/vets/builds the residual + background and (for the final
phase) the cross-band catalogs — chained `afterok`, phase after phase. This is
controlled by `--manual-frame-shard=I/N`, `--manual-skip-finalize` (worker),
`--manual-finalize-only` (barrier), `--manual-start-phase`, and
`--manual-stop-after-phase`; all default off, so a monolithic run is unchanged.
The fan-out and finalize jobs run the **same** per-frame fit and per-phase
barrier code as the monolith, so the two produce the **same** science output —
the distributed path just splits it across small, backfill-friendly jobs. See
[`scripts/reduction/README.md`](scripts/reduction/README.md) for the submitters
(`submit_cataloging_perframe.sh`, the per-filter chain, the m7 finalize) and
`scripts/reduction/validate_perframe_equivalence.sh`, which diffs a monolithic
vs per-frame run on the same cutout to prove equivalence (it diffs the m6 vetted
catalogs; the cross-band *merge* and the m8 fill are full-frame-only — see below).

## The iterations

`m12` runs per-frame iter1+iter2; `m3..m7` are merged-i2d-seeded passes; `m8` is
the forced cross-band fill (multi-filter only). Source *detection* uses a
progressively cleaner co-add so sources hidden in one stage surface in the next;
the *fit* always runs on the raw or background-subtracted **frames**, never on
the i2d.

| iter | phase | detection co-add | frames fit | background |
|------|-------|------------------|------------|------------|
| 1,2  | m12   | per-frame data / residual | raw | — |
| 3    | m3    | raw merged i2d | raw | — |
| 4    | m4    | source-subtracted residual i2d | raw | builds first bg |
| 5    | m5    | residual i2d − background | bg-subtracted | recompute |
| 6    | m6    | residual i2d − background | bg-subtracted | recompute |
| 7    | m7    | cross-filter seed (multi-filter only) | bg-subtracted | — |
| 8    | m8    | m7 merged positions (no new detection) | bg-subtracted | — |

`m8` is not a detect/fit pass — it is a **forced cross-band fill** run right after
the m7 cross-band merge. For every m7 merged source that is a *non-saturated
non-detection* in some band, it force-fits the flux at the merged reference
position in that band (`forced_psf_photometry`, position pinned, flux-only),
recovering the phantom non-detections the independent-detection cross-match
leaves behind or producing a real per-source noise limit. It is self-contained:
it writes a sibling `..._resbgsub_m8` catalog, never mutates m7, and a failure is
non-fatal. It is on by default (disable the inline fill with `--no-forced-fill-m8`)
and, like the m7 cross-band merge, is **full-frame only** — cutout runs (which
lack the per-filter reduction i2d mosaics) stop at the per-filter m6/m7 catalogs.
The combined m8 is then **de-duplicated** into a `..._resbgsub_m8_dedup.fits`
sibling (`--no-m8-dedup` to skip): the cross-band merge can split one star into
two reference rows in crowded fields, and the dedup collapses complementary-
coverage pairs while preserving resolved binaries. Because the monolithic fill
sweeps every frame serially and can overrun the wall, m8 can instead be fanned
out one band per job with `--manual-m8-partial` (pair the m7 job with
`--no-forced-fill-m8`); `scripts/reduction/submit_cataloging_m8.sh` schedules the
per-band partials + the merge (`m8_merge_partials.py`, which also dedups).

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
- **Overshoot QC + forced refit.** After each single-pass fit, every source's
  rendered model **peak** is compared to the local **data** peak. A model peak
  exceeding `--manual-overshoot-ratio` × the data peak is physically impossible
  for one positive PSF and signals that the free-position fit let the centroid
  **walk off the star** and settle at an inflated flux (the exact instability
  that retired `IterativePSFPhotometry`). By default (`--manual-overshoot-action=refit`)
  each flagged source is re-measured by **forced photometry**
  (`forced_psf_photometry`): the position is **pinned at the trusted daofind/seed
  position** (not the drifted fit position) and only the flux is solved. With
  position fixed the model is linear in flux, so the flux is the exact
  closed-form weighted least-squares value `f = Σ(d·p·w)/Σ(p²·w)` — one cheap
  step (~80× faster than a fixed-position LM fit, which matters when dozens of
  sources/frame are flagged), with no opportunity to re-drift. That closed form
  is unbounded, so on background-/neighbour-subtracted data it can go negative;
  the refit therefore clamps flux ≥ 0 (`nonnegative=True`) and the negative-peak
  ban is the final net. See `forced_psf_photometry`'s docstring for the full
  rationale.
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
  catalogs into a cross-band seed, then **dedup** co-located positions
  (`--manual-crossband-seed-dedup-mas`, default 30 mas) so a star seen in several
  bands seeds the m7 fit once, not N times (a degenerate co-located group
  otherwise over-subtracts). The stringent ≥2-filter / S-N coincidence
  *requirement* (vs the current union) is still a documented TODO.
- **Forced cross-band fill (iter8).** After the m7 cross-band merge, m8
  force-fits every band at the merged position of sources that are non-saturated
  non-detections there (see the iterations table). Full-frame only.

## Outputs

Under `<basepath>/cutouts/<label>/`:

- `catalogs/<filt>_<module>_indivexp_merged[...]_m{N}_dao_basic.fits` — merged
  per-phase catalogs (`m2`,`m3`,`m4`,`_resbgsub_m5`,`_resbgsub_m6`,`_resbgsub_m7`),
  plus `_vetted` and `_allcols` variants. Carry `skycoord`, `flux`, `qfit`,
  `iter_found`, etc.
- `catalogs/basic_<module>_indivexp_photometry_tables_merged_resbgsub_m7<obs>.fits`
  — the multi-filter **cross-band** table (per-band fluxes/mags matched within
  `max_offset`, anchored on the reference filter). Full-frame only.
- `..._resbgsub_m8<obs>.fits` — the **forced-fill** sibling of the m7 cross-band
  table: same rows, but every band's flux is force-fit at the merged position so
  cross-band non-detections carry a measured value / noise limit instead of a
  gap. Full-frame only; never overwrites m7.
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

- The monolith runs as a single long in-process job. For large runs use the
  per-frame distributed fan-out (above) instead, which shards the per-exposure
  fits across many small SLURM jobs and reproduces the monolith's output; the
  phases themselves remain strictly ordered (each detects on the previous phase's
  residual mosaic), so the *finalize* barriers are still serialized phase-to-phase.
- Cross-band seed is a deduped union; the stringent ≥2-filter coincidence
  *requirement* is still a TODO.
- Faint sources blended on the wings of much brighter or saturated stars may be
  detected but dropped by the fit/dedup; the m8 forced cross-band fill recovers
  such sources only where they were already detected in another band, not where
  they are lost in every band.
