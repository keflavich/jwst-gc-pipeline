# Photometry pipeline

(most of this is written by AI but Adam is editing in some comments to explain what... frankly is nonsensical)
<!--
  DOC SYNC — last reviewed 2026-07-07 (expanded: per-iteration step-by-step,
  explicit saturated-star finding + daofind parameters, simplified all-defaults
  table, and a full per-parameter table showing NIRCam / MIRI / extended-emission
  deployment).  Prior review 2026-06-28 (m8 forced cross-band fill; per-frame
  fan-out equivalence).  This is the IMPLEMENTATION-facing doc (flags, tokens,
  file names, behaviours).  Its science-narrative companion is PIPELINE_METHODS.md
  (publication-style, no code). KEEP THE TWO IN SYNC: when the algorithm changes,
  update BOTH and bump the stamp in BOTH. PIPELINE_METHODS.md = "what & why"; this
  file = "how & where".  All numbers below are the actual deployed defaults; each
  is cited to cataloging.py / crowdsource_catalogs_long.py / saturated_star_finding.py.
-->

## How to run

Single filter (cutout):

```
python -m jwst_gc_pipeline.photometry.crowdsource_catalogs_long \
    --filternames=F480M --modules=nrcb \
    --proposal_id=3958 --field=007 --target=sickle \
    --each-suffix=destreak_o007_crf --each-exposure --daophot --skip-crowdsource \
    --group --max-group-size=10 --manual-group-min-sep-fwhm=3.0 \
    --cutout-region=/path/to/region.reg --cutout-label=my_cutout
```

Multi-filter (adds the cross-band m7 + m8 steps); one module name serves a SW and a
LW filter (e.g. `nrcb` → `nrcblong` for F480M and `nrcb1..4` for F210M), but usually you should use `modules=merge` because reducing individual frames independently makes no sense:

```
    --filternames=F480M,F210M --modules=nrcb ...
```

`--cutout-region` accepts a DS9 `.reg` file or an `'ra,dec,size_arcsec'` string; it is intended for faster runs for development testing.
`--cutout-label` names the output tree under `<basepath>/cutouts/<label>/`.
Full-frame is the same command without `--cutout-region`/`--cutout-label` ('frame' here means a single H2RG detector module, like NRCA1 or NRCAlong). 

**Monolithic in-process run (simpler, but may wait in queue longer).** Each phase detects on the
*previous* phase's merged residual mosaic, so the phases are sequential.
The simplest way to run is one process that executes every phase in order —
full-frame or cutout, as a single (non-array) job. Full-frame outputs land under
`<basepath>/<FILTER>/pipeline/` and `<basepath>/catalogs/`; cutout runs are
 under `<basepath>/cutouts/<label>/`.

**Distributed per-frame fan-out (recommended at scale).** (edited by Adam to *slightly* de-slopify.  Mop up after the AI?) The same pipeline can
run individual images on separate nodes/processes for a single filter: for each phase, many per-frame worker
jobs run in parallel, then one finalize-barrier job merges/vets/builds the
residual + background (and, for the final phase, the cross-band catalogs).
These are chained together with slurm using `afterok`. 
The AI tool likes the word 'shard' for this, so each subprocess is a 'shard'.
Controlled by `--manual-frame-shard=I/N`,
`--manual-skip-finalize` (worker), `--manual-finalize-only` (barrier),
`--manual-start-phase`, `--manual-stop-after-phase`; all default off such that monolithic is the default.
Fan-out and finalize run the **same** per-frame fit
and per-phase barrier code as the monolith, so they produce the **same** science
output. See [`scripts/reduction/README.md`](scripts/reduction/README.md) for the
submitters and `scripts/reduction/validate_perframe_equivalence.sh`, which diffs
a monolithic vs per-frame run on the same cutout to prove equivalence.

---

## What each iteration actually does

The pipeline is a sequence of phases. Detection uses a progressively cleaner
co-add so sources hidden in one stage surface in the next; **the fit always runs
on the raw or background-subtracted frames, never on the i2d.** The phase list is
`['m12', 'm3', 'm4', 'm5', 'm6']`, plus `'m7'` for multi-filter runs
(`cataloging.py:2633`); `m8` runs after the m7 cross-band merge.

Every phase, for every frame, runs the same skeleton
(`do_photometry_step_manual` → `_prepare_frame_for_photometry` →
`_manual_phot_pass`):

1. **Load the frame** (cal/crf), build its WCS, DQ plane, source mask, and
   per-filter FWHM (from `reduction/fwhm_table.ecsv`).
2. **Find and subtract saturated stars** → the satstar model (see below). This
   model is subtracted from the fit frame so bright stars' wings don't corrupt the
   fits, and (for display) added back into the model mosaic.
3. **Build the detection seed** = `daofind` on this phase's detection co-add image.
   The DAOfind catalog is filtered by local S/N, round/sharp, and is unioned with the previous phase's vetted
   catalog (see "daofind parameters" below).
4. **Fit** every seed with single-pass BASIC `PSFPhotometry` on the fit frame using STPSF PSFs.
5. **Post-fit QC**: model/data-peak overshoot check (→ forced refit),
   non-positive-flux ban, near-saturation/satstar-wing rejection, dedup.
6. **Render** the per-frame model + residual.

Then a per-phase **barrier** merges the per-frame catalogs across frames, tags
`iter_found`, vets extended emission, and builds the residual i2d + the
source-masked smoothed background that seeds the next phase.

### Saturated-star finding (`saturated_star_finding.py`)

Done once per frame, cached as `<frame>_..._satstar_catalog.fits`; recomputed
only when the off-FOV cross-frame reconciliation supplies flux overrides/drops,
or `overwrite=True` (`cataloging.py:1742`). Steps:

1. **Identify saturated pixels** from the DQ plane: the `SATURATED` bit
   (`saturated_star_finding.py:282`), plus `DO_NOT_USE` for truly-lost pixels
   (0 good ramp groups) and `JUMP_DET` cosmic-ray handling (saturated clusters
   ≥3 px are protected). Connected components → one candidate saturated star per
   blob.
2. **Fit each blob** with a gridded WebbPSF/STPSF model (`npsf=16`, `oversample=2`;
   `saturated_star_finding.py:76-77`). In-FOV field of view: **fovp 512** for SW
   (NRC?1-4) and MIRI, **fovp 1024** for LW (NRC?LONG) (`:1639`). Grids live in
   `<basepath>/psfs`, named `{inst}_{detector}_{filter}_fovp{N}_samp2_npsf16.fits`
   (`:93`). The LW detector token is `nrcb5`/`nrca5` (WebbPSF naming), **not**
   `nrcblong` — the per-frame cache lookup maps `nrcXlong → nrcX5`.
3. **Keep-gate** each fit (`accept_satstar_fit`, `:1061`): NIRCam requires
   `qfit ≤ 5`, `snr ≥ 3`, `sidelobe ≥ -10 σ`, and `ssr_ratio ≤ 1` — but the ssr
   gate is applied **only to low-confidence fits** (`snr ≤ 10`); high-S/N,
   good-qfit satstars bypass it so real bright stars are never dropped. MIRI uses
   looser bounds (`qfit ≤ 15`, `snr ≥ 2`, `sidelobe ≥ -40`, `ssr ≤ 2`).
4. **MIRI-only seed gates** reject false blobs in extended emission:
   `seed_prominence_min=8.0`, `seed_core_min=1000.0`, `seed_conc_min=1.3`
   (`:1273`; F770W/Sickle-calibrated, env-overridable), plus a spike-merge that
   folds diffraction-spike satellites into their core.
5. **Off-FOV bright stars** (spikes bleeding in from outside the frame) use a
   **large** forced grid — fovp **2048** (SW) / **1024** (LW/MIRI) (`:1679`) — a
   position prior, an integer grid search (`radius=5 px`), a `model ≤ data` clamp
   at the 10th percentile (`oversub_clamp_percentile=10.0`), and a cross-frame
   flux reconciliation that pins runaway far-detector fits to the nearest-detector
   flux.
6. **ZEROFRAME / frame-0 recovery (both OFF by default):**
   - `--deblend-satstars`: in crowded fields (gc2211) touching saturated cores
     label as one blob; the sibling `_ramp.fits` ZEROFRAME splits the merged
     component into one seed per star. Falls back to the ramp first read
     `SCI[0,0]` as a pseudo-zeroframe when no `ZEROFRAME` extension exists, and
     no-ops entirely where no `_ramp.fits` exists.
   - `--satstar-zeroframe-recover`: replaces the charge-migration-inflated
     saturated *rim* with `R × group0` (de-inflated truth) so the subtracted
     residual leaves no positive ring; `--satstar-zeroframe-dilate=3` px buffers
     the DQ mask for charge migration.

During the point-source fit, daophot detections that land on a satstar are
rejected: `--satstar-artifact-ratio=1.0` drops fits dimmer than the local satstar
model (gated where `satstar_model > 3 σ`, `--satstar-artifact-sigK`), and
detections within `~1.5 × FWHM` of a hard-saturated pixel are dropped.

### daofind parameters

Detection is `photutils.DAOStarFinder` (`cataloging.py:563`). The `threshold`
passed to DAOStarFinder is the **global minimum** of the local-noise map — i.e.
deliberately permissive "detect everything" — and the **real** selection is the
subsequent per-source **local-S/N filter** plus the round/sharp bounds:

| pass | round lo/hi | sharp lo/hi | local-S/N cut | S/N filter | seeded from |
|------|-------------|-------------|---------------|------------|-------------|
| m1 (iter1) | −1.0 / +1.0 | 0.30 / 1.40 | 5.0 | **off** (unseeded discovery) | nothing |
| m2 (iter2) | −0.3 / +0.3 | 0.50 / 1.00 | 3.0 | on | m1 catalog + m1 residual |
| m3..m7 (per-frame) | −0.3 / +0.3 | 0.50 / 1.00 | 3.0 | on | prev vetted catalog |

(`cataloging.py:1683` m1, `:1699` m2, `:1717` m3+.) The wider round/sharp window
on **m1** is intentional: the first unseeded pass casts a wide net; **m2 onward**
tighten toward star-like shapes because they reseed on residuals where extended
emission dominates the false positives.

For the merged-i2d-seeded phases (m3+), a second `daofind` runs on the **detection
co-add** itself (`_build_i2d_augmented_seed`) with its own bounds —
`--manual-seed-round-max=0.5`, `--manual-seed-sharp-lo/hi=0.4/1.2` — and that
result is unioned with the previous phase's vetted merged catalog and deduped at
`0.5 × FWHM`. FWHM is per-filter from `reduction/fwhm_table.ecsv` (F210M 2.30,
F212N 2.34, F480M 2.57 px).

### The fit (identical every phase)

Single-pass **BASIC** `photutils.PSFPhotometry` (`cataloging.py:213`):
`fitter=LevMarLSQFitter()`, `fit_shape=(5, 5)`, `aperture_radius = 2.0 × FWHM`,
a `LocalBackground` annulus (`inner ≈ aperture + 0.5·FWHM`, width `≈ FWHM`), and
— only with `--group` — a `SourceGrouper(min_separation = manual_group_min_sep_fwhm × FWHM)`
for joint fitting of blends. It is **not** iterative: reseeding happens across
phases, not inside a phase.

Post-fit, in order: **overshoot QC** (rendered model peak vs local data peak; if
`model_peak > 1.2 × data_peak` the free-position fit walked off the star →
`--manual-overshoot-action=refit` re-solves flux-only at the pinned seed position
via the closed-form `f = Σ(d·p·w)/Σ(p²·w)`, clamped ≥0); **non-positive-flux ban**
(`flux_fit ≤ 0` dropped at `cataloging.py:465` — a positive PSF cannot have a
negative peak); **dedup**; **near-saturation / satstar-wing rejection**.

### Phase-by-phase detection & fit surfaces

| phase | detection co-add | frames fit | background |
|-------|------------------|------------|------------|
| m12 (iter1+2) | m1: raw frame · m2: raw − m1 model | raw (satstar-subtracted) | — |
| m3 | raw merged data i2d | raw | — |
| m4 | m3 source-subtracted residual i2d | raw | **builds 1st bg** (after merge) |
| m5 | m4 residual i2d − m4 bg | bg-subtracted | recompute |
| m6 | m5 residual i2d − m5 bg | bg-subtracted | recompute |
| m7 | cross-filter seed (multi-filter only) | bg-subtracted (m6 bg) | — |
| m8 | m7 merged positions (no detection) | bg-subtracted | — |

- **m12** is per-frame only (no merged mosaic yet): iter1 discovers unseeded on
  the raw frame; iter2 reseeds on that frame's own residual plus the m1 catalog.
  Both fit the raw, satstar-subtracted frame. The satstar model is built here.
- **m3** is the first merged-seeded pass: detect on the raw data i2d coadd, fit
  raw frames.
- **m4** detects on m3's residual mosaic, fits raw frames, and after its merge
  builds the **first** source-masked smoothed background.
- **m5, m6** detect on the previous residual mosaic **minus** the current
  background, and fit **background-subtracted** frames, recomputing the
  background each time. m6 is the final per-filter pass.
- **m7** (multi-filter only): the seed is the cross-band merge of every filter's
  m6 vetted catalog, deduped so a star seen in N bands seeds once. Fit on m6
  background-subtracted frames.
- **m8** is **not** a detect/fit pass — it is a **forced cross-band fill** after
  the m7 merge. For every m7 merged source that is a *non-saturated non-detection*
  in some band, it force-fits flux at the merged position in that band (position
  pinned, flux-only), recovering phantom non-detections or producing a per-source
  noise limit. On by default (`--no-forced-fill-m8` disables); full-frame only;
  writes a sibling `..._resbgsub_m8` catalog, never mutates m7. The result is then
  de-duplicated to `..._resbgsub_m8_dedup.fits` (`--no-m8-dedup` to skip).

### Merge + vet (per-phase barrier)

After each phase the per-frame catalogs are merged, `iter_found` is tagged, and
the merged catalog is vetted by `_filter_extended_emission` (`cataloging.py:761`,
called `:3474`). NIRCam keeps a source if it is **star-like**

```
star_like = (qfit ≤ 0.2) OR (flags in keep_flags) OR (peakSB > 20 × local_bkg)
            OR bright_isolated
bright_isolated = (snr ≥ 20) AND (qfit < 0.4) AND (group_size ≤ 1)
```

**and** it clears the local-S/N floor (`local_snr_min = 5`), then drops anything
flagged `model_overshoot`. MIRI instead vets purely on data-i2d **prominence**
`(core_peak − annulus_median)/annulus_MAD ≥ min_prominence`. The pipeline then
builds the residual i2d and the source-masked smoothed background for the next
phase.

---

## Table A — simplified, all defaults (NIRCam, non-extended field)

The complete pipeline with every value at its shipped default. This is what runs
for a plain single-filter NIRCam field with no tuning flags.

| step | parameter | default |
|------|-----------|---------|
| **satstar** | in-FOV PSF fovp (SW / LW) | 512 / 1024 |
| | off-FOV forced PSF fovp (SW / LW) | 2048 / 1024 |
| | PSF grid `npsf` / `oversample` | 16 / 2 |
| | keep-gate `qfit / snr / sidelobe / ssr` | ≤5 / ≥3 / ≥−10σ / ≤1 (ssr only if snr≤10) |
| | mask dilation buffer | 2 px |
| | wing-rejection ratio / sigK | 1.0 / 3.0 |
| | off-FOV model≤data clamp percentile | 10 |
| | ZEROFRAME deblend / rim-recover | off / off |
| **daofind m1** | round / sharp / local-S/N | ±1.0 / 0.30–1.40 / 5.0, no S/N filter |
| **daofind m2+** | round / sharp / local-S/N | ±0.3 / 0.50–1.00 / 3.0 |
| **i2d seed** | round-max / sharp lo-hi / dedup | 0.5 / 0.4–1.2 / 0.5·FWHM |
| | struct-noise prune (x, y) | 0.0, 0.0 (off) |
| | coarse-bg box | 0 (off) |
| **fit** | engine | BASIC PSFPhotometry, single pass |
| | fitter / fit_shape / aperture | LevMar / (5,5) / 2·FWHM |
| | grouping | off (`--group` to enable) |
| | group min-sep | 2.0·FWHM |
| | max group size | unlimited |
| **post-fit** | overshoot ratio / action | 1.2 / refit |
| | negative-flux | banned |
| **vetting** | qfit_max | 0.2 |
| | peak-over-bkg | 20 |
| | local-S/N min | 5.0 |
| | bright-isolated keep (snr / qfit) | ≥20 / <0.4 |
| | prominence gate | 0 (off; MIRI only) |
| **cross-band** | seed dedup / min-filters / snr / qfit | 30 mas / 2 / 5.0 / 0.2 |
| **m8** | forced fill / dedup | on / on |

---

## Table B — every configurable parameter, and what is actually used

"What's actually used" differs by regime. **NIRCam-std** = plain NIRCam field.
**Extended** = a NIRCam field auto-detected as extended emission (`w51`, `sickle`,
`wd2`, `ngc6334`; `--extended-emission` forces it on/off). **MIRI** = the
`miri_tuning` per-phase schedule. Blank = same as NIRCam-std.

| flag | dest / default | NIRCam-std | Extended | MIRI |
|------|----------------|-----------|----------|------|
| *(path)* `--manual-iterations` / `--legacy-iterations` | `True` | this pipeline | | |
| `--daophot`, `--skip-crowdsource`, `--each-exposure` | off | required for this path | | |
| `--manual-overshoot-ratio` | 1.2 | 1.2 | 1.2 | 1.2 |
| `--manual-overshoot-action` | `refit` | refit | refit | refit |
| `--manual-iter2-local-snr` | 3.0 | 3.0 (m2+) | 3.0 | 3.0 |
| `--manual-seed-round-max` | 0.5 | 0.5 | 0.5 | 0.5 |
| `--manual-seed-sharp-lo` / `-hi` | 0.4 / 1.2 | 0.4 / 1.2 | | |
| `--manual-struct-noise-x` (`struct_x`) | 0.0 | 0.0 (off) | **1.0** (auto) | **5.0** m12–m4, **3.0** m5–m6 |
| `--manual-struct-noise-y` (`struct_y`) | 0.0 | 0.0 (off) | **2.0** (auto) | **8.0** all phases |
| `--manual-coarse-bg-box` (`coarse_bg_box`) | 0 | 0 (off) | 0 | **51** m12–m4, 0 m5–m6 |
| `--manual-ext-qfit-max` | 0.2 | 0.2 | 0.2 | **0.4** |
| `--manual-ext-peak-over-bkg` | 20.0 | 20 | 20 | 20 |
| `--manual-ext-local-snr-min` | 5.0 | 5.0 | 5.0 | **8.0** m12–m4, **3.0** m5–m6 |
| `--manual-ext-snr-high-keep` | 20.0 | 20 | 20 | 20 |
| `--manual-ext-qfit-high-keep-max` | 0.4 | 0.4 | 0.4 | 0.4 |
| `--manual-ext-qfit-recover-max` | 0.2 | 0.2 (= qfit_max ⇒ **no-op**) | set 0.5 to enable | |
| prominence gate (`miri_prominence_snr`) | 0.0 | 0 (off) | 0 (off) | **8→3 progressive** (m12:8, m4:5.5, m6:3) |
| `--nircam-prom-m1` / `-m2` / `-m3plus` | 0.0 | 0 (off) | 0 (off; opt-in) | n/a |
| `--group` | off | off (pass `--group`) | | |
| `--manual-group-min-sep-fwhm` | 2.0 | 2.0 (use ~3.0 for blends) | | |
| `--max-group-size` | `unlimited` | unlimited (cap for dense) | | |
| `--fit-satstar-outside-fov` | `None`(auto) | on full-frame, off cutout | | |
| `--satstar-artifact-ratio` | 1.0 | 1.0 | 1.0 | 1.0 |
| `--satstar-artifact-sigK` | 3.0 | 3.0 | 3.0 | 3.0 |
| `--satstar-oversub-clamp-percentile` | 10.0 | 10 | 10 | 10 |
| `--deblend-satstars` | off | off (on for gc2211) | | off |
| `--satstar-zeroframe-recover` | off | off | | off |
| `--satstar-zeroframe-dilate` | 3 | 3 | | |
| `--manual-crossband-seed-dedup-mas` | 30.0 | 30 | 30 | 30 |
| `--manual-crossband-seed-min-filters` | 2 | 2 | 2 | 2 |
| `--manual-crossband-seed-snr-min` | 5.0 | 5.0 | | |
| `--manual-crossband-seed-qfit-max` | 0.2 | 0.2 | | |
| `--no-forced-fill-m8` (`forced_fill_m8`) | `True` | on | on | on |
| `--no-m8-dedup` (`m8_dedup`) | `True` | on | on | on |
| `--manual-frame-shard`, `--manual-skip-finalize`, `--manual-finalize-only` | off | monolith | | |
| `--manual-start-phase` / `--manual-stop-after-phase` | `''` | full run | | |
| `--parallel-workers` / `--parallel-chunk-size` | 1 / 100 | serial (experimental) | | |
| `--each-suffix` | `destreak_o001_crf` | per-reduction | | |
| `--cutout-region` / `--cutout-label` / `--cutout-size-arcsec` | `''`/`''`/5.0 | full-frame | | |

Notes on the tri-state and env-driven values:
- **`--extended-emission`** is tri-state: default `None` → auto by target
  membership (`w51`, `sickle`, `wd2`, `ngc6334`); `--extended-emission` /
  `--no-extended-emission` force it. Turning it on auto-sets the structure-noise
  prune to `(1.0, 2.0)` unless you override `--manual-struct-noise-x/-y`.
- **MIRI** per-phase values come from the `miri_tuning` schedule
  (`cataloging.py:2916-2990`); the prominence schedule is env-tunable
  (`MIRI_PROM_SNR_PROGRESSIVE`, `MIRI_PROM_SNR_HI/LO`), defaulting to 8→3 across
  m12→m6.
- **Satstar seed gates** (`seed_prominence_min=8`, `seed_core_min=1000`,
  `seed_conc_min=1.3`) are MIRI-only and env-overridable
  (`MIRI_SATSTAR_SEED_*`); they do not fire on NIRCam.

---

## Key behaviours (rationale)

- **Negative-peak ban.** Any fit with `flux_fit ≤ 0` is a negative-peak model
  (impossible for a positive PSF). Dropped at the fitter and excluded from seeds,
  so they never breed more negatives at over-subtracted spots.
- **Overshoot QC + forced refit.** A model peak exceeding `1.2 ×` the local data
  peak signals the free-position fit walked off the star (the instability that
  retired `IterativePSFPhotometry`). The refit pins the position at the trusted
  seed and solves flux-only in closed form (~80× faster than a fixed-position LM
  fit), clamped ≥0.
- **Source-masked background.** The smoothed background masks fitted source disks
  before smoothing, so it represents diffuse emission only and does not feed
  source-core holes back into the next fit.
- **`iter_found` column.** Every merged catalog records the first iteration
  (2..7) each source appears in, matched across phases by sky position.
- **Cross-band (m7).** Multi-filter runs union the per-filter vetted m6 catalogs,
  dedup co-located positions (`--manual-crossband-seed-dedup-mas=30`), and can
  require independent ≥`--manual-crossband-seed-min-filters` confirmation.
- **Forced cross-band fill (m8).** Force-fits every band at the merged position of
  cross-band non-detections; full-frame only.

## Outputs

Under `<basepath>/cutouts/<label>/` (or in place for full-frame):

- `catalogs/<filt>_<module>_indivexp_merged[...]_m{N}_dao_basic.fits` — merged
  per-phase catalogs (`m2`,`m3`,`m4`,`_resbgsub_m5`,`_resbgsub_m6`,`_resbgsub_m7`),
  plus `_vetted` and `_allcols` variants (carry `skycoord`, `flux`, `qfit`,
  `iter_found`, …).
- `catalogs/basic_<module>_indivexp_photometry_tables_merged_resbgsub_m7<obs>.fits`
  — the multi-filter **cross-band** table (per-band fluxes/mags, anchored on the
  reference filter). Full-frame only.
- `..._resbgsub_m8<obs>.fits` + `..._resbgsub_m8_dedup.fits` — the **forced-fill**
  sibling of the m7 table; every band's flux is force-fit at the merged position.
  Full-frame only; never overwrites m7.
- `<filt>/pipeline/...-<module>_data_i2d.fits` — input data mosaic.
- `..._m{N}_..._mergedcat_residual_i2d.fits` — residual mosaic per phase
  (point-source models subtracted; saturated stars already removed).
- `..._m{N}_..._mergedcat_model_i2d.fits` — model mosaic per phase. **For display
  it adds the saturated-star model back** on top of the fitted point-source model;
  the residual above is unaffected.
- `..._m{N}_..._mergedcat_residual_smoothed_bg_i2d.fits` — background map.
- `<frame>_..._satstar_catalog.fits` (+ `_extended_` variant) — cached per-frame
  saturated-star fits.

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
| `--manual-no-sky-clean-keep` | (tier on) | disable the sky-clean keep tier: on emission-free sky (deep-i2d local floor ≈ dark-sky ref) keep on prominence ≥ `--manual-sky-clean-prom-min` (5) + S/N ≥ `--manual-sky-clean-snr-min` (3), qfit ignored; inert where emission is measured |
| `--manual-group-min-sep-fwhm` | 2.0 | grouping radius in FWHM (use ~3.0 for blends) |
| `--group` / `--max-group-size` | off / — | enable joint fitting; cap group size |

## Known limitations

- The monolith runs as a single long in-process job; for large runs use the
  per-frame distributed fan-out. The phases remain strictly ordered (each detects
  on the previous phase's residual mosaic), so the finalize barriers are
  serialized phase-to-phase.
- Cross-band seed is a deduped union; the stringent ≥2-filter coincidence
  *requirement* (vs the union) is available via
  `--manual-crossband-seed-min-filters` but the default union path is what most
  runs use.
- Faint sources blended on the wings of much brighter or saturated stars may be
  detected but dropped by the fit/dedup; the m8 forced cross-band fill recovers
  such sources only where they were already detected in another band, not where
  they are lost in every band.
- `--satstar-zeroframe-recover` / `--deblend-satstars` need the sibling
  `_ramp.fits` (Detector1 `save_calibrated_ramp`); they no-op on frames that lack
  it. A literal `ZEROFRAME` extension is preferred; the ramp first read `SCI[0,0]`
  is used as a pseudo-zeroframe fallback.
</content>
</invoke>


## History Notes

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

