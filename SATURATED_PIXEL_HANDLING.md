# Saturated-Pixel Handling

How the pipeline detects, models, and photometers **saturated stars** — the
bright sources whose cores exceed the detector full-well and are flagged
`SATURATED` in the DQ plane. Saturated cores carry no usable flux, so the star's
brightness and position must be recovered from its **unsaturated wings and
diffraction spikes** (PSF fit with the core masked), and optionally from the
**ramp first read (ZEROFRAME / group-0)**, which saturates at a much higher flux.

NIRCam and MIRI are handled differently — different detector physics (PSF width,
brighter-fatter, extended emission, edge glow) drive different masking, fitting,
and vetting. Both share one engine:
`jwst_gc_pipeline/reduction/saturated_star_finding.py`.

File:line references are current as of this audit (2026-07); treat them as
starting points, not fixed addresses.

---

## 1. Pipeline of a saturated star

1. **Detect** — read the `SATURATED` DQ bit, reject `JUMP_DET` cosmic rays, keep
   clusters ≥ 3 px (`find_saturated_stars`, `saturated_star_finding.py:282`).
2. **Filter spurious flags** — drop DQ-SATURATED components sitting on *faint*
   data (persistence / JUMP mis-tags) via a per-filter **data floor**
   (`_SATSTAR_DATA_FLOOR`, `:221`; test at `:325`).
3. **Label + refine** — connected components (`ndimage.label`), remove large
   edge bleeds (preserving genuine NaN-variance cores), refine centroids on the
   real core (`_refine_coms_by_data`, `:484`).
4. **(MIRI) merge spike satellites** — fold diffraction-spike fragments into the
   parent core (`_merge_spike_satellites`, `:161`; off for NIRCam).
5. **(opt) Deblend touching cores** — split merged saturated blobs into one seed
   per star using the ZEROFRAME (`--deblend-satstars`, `satstar_deblend.py`).
6. **Fit** — PSF fit with the core masked, adaptive mask buffer + background
   annulus, inverse-variance (1/ERR²) weighting, brightest-first with iterative
   subtraction (`get_saturated_stars`, `:1116`).
7. **(MIRI) seed gate** — drop extended-emission "phantom" components by
   prominence / core-brightness / concentration (`:1292`).
8. **Accept gate** — keep the fit on qfit / sidelobe / ssr / snr (`:1018`,
   defaults `:1558`).
9. **(opt) ZEROFRAME rim recovery** — de-inflate the brighter-fatter rim using
   group-0 (`zeroframe_recover_saturated`, `:389`; `--satstar-zeroframe-recover`).
10. **(opt) Off-FOV stars** — fit bright stars whose *centers* are outside the
    frame but whose spikes reach in; reconcile flux across frames
    (`reconcile_outside_fov_satstar_fluxes`, `:708`; `--fit-satstar-outside-fov`).
11. **Output** — `*_satstar_catalog.fits`, `*_satstar_model.fits`,
    `*_satstar_residual.fits`, `*_satstar_flags.fits`.
12. **Photometry integration** — daophot fits on the satstar-model-subtracted
    data; artifact/coincidence gates keep daophot off the saturated cores; the
    satstar catalog is merged in with `replaced_saturated=True`
    (`cataloging.py`, `merge_catalogs.py`).

---

## 2. Detection & DQ flagging (shared)

- **SATURATED bit + CR rejection** (`saturated_star_finding.py:282`): extract
  `dqflags.pixel['SATURATED']`, reject `JUMP_DET`, require ≥3-px clusters to
  separate ramp non-linearity from single-pixel CRs.
- **Data floor** (`_SATSTAR_DATA_FLOOR`, `:221`; `_resolve_satstar_data_floor`,
  `:227`): a DQ-SATURATED component is only trusted if the max data value in its
  wings exceeds the per-filter floor (MJy/sr), or its core is NaN-variance
  (genuinely unrecoverable). Guards against persistence / JUMP artifacts
  mis-tagged SATURATED on faint sources.

  | filters | satstar-finder floor | daophot-mask floor (`_NIRCAM_SAT_DATA_FLOOR`, `cataloging.py:1059`) |
  |---|---|---|
  | F140M F162M F182M F187N F210M F405N F480M | 1000 | 5000 |
  | F335M F360M F410M | 800 | 2500 |
  | others | 0 (off) | — |

  Override: `--saturation-data-floor` (photometry), env `SATSTAR_DATA_FLOOR`
  (finder). `-1` = per-filter auto; `0` = mask all SATURATED; `>0` = explicit.

---

## 2b. KNOWN DEFECT — detection uses the *any-group* SATURATED bit

**The saturation detection flags every pixel that saturates in *any* ramp group,
including pixels the ramp fitter fully recovered from good earlier groups.** This
over-flags the saturated set, masks recoverable data, and (worse) sweeps real
stars on bright emission into the satstar channel while vetoing them from daophot
— so they vanish from the catalog.

**Where:** `find_saturated_stars` (`saturated_star_finding.py:282`):
```python
saturated = (dq & dqflags.pixel['SATURATED']) > 0
```
`dq` is the 2-D cal/crf `DQ`, whose `SATURATED` bit (per the JWST pipeline) is set
if the pixel saturated in *any* group. Only a pixel whose **first** group
saturates is genuinely unrecoverable; any later-group saturation still yields a
valid ramp-fit slope. The code documents this itself
(`first_group_saturation_mask`, `:3189`; `correct_dq_first_group_saturation`,
`:3218`).

**Empirical scale** (MIRI sgrb2 F770W, `jw05365998001_02101_00003`): the cal DQ
flags 3,114 px SATURATED; only **345 (11%)** are first-group (truly
unrecoverable); **2,769 (89%)** saturated only in later groups, and **62% of
those carry finite recovered flux** (median ~760 MJy/sr). So ~9× over-flagging.
Documented downstream harm: cloudc 2526 F770W — 28 by-eye-real stars fused into
saturated blobs and lost from the catalog (commit `e4039e6`).

**When created:** commit **`8837408` (2026-05-04, "first real commit")** — the
detection has used the any-group bit since the repository's first commit (the
satstar finder predates the git history; this is the earliest tracked commit).

**Affected commit range:** **`8837408` (2026-05-04) → HEAD** — *every* commit. The
default behaviour on `main` is affected. **A proper fix is now in flight (PRs #40
+ #41, see below)**; until merged, on `main`:
- **NIRCam: entirely unmitigated.** The only correction returns `dq` unchanged
  unless the instrument is MIRI (`correct_dq_first_group_saturation`, `:3227`).
- **MIRI: unmitigated by default.** That correction is env-gated
  `MIRI_FIRSTGROUP_SAT_DQ` (default `0`) and needs a sibling `_ramp.fits`.

**Mitigation / fix history:**
- `e4039e6` (2026-06-30) — added `first_group_saturation_mask` /
  `correct_dq_first_group_saturation` (reads `_ramp.fits` GROUPDQ, clears
  SATURATED on later-group-only px). **MIRI-only, opt-in, default off** → does not
  change the shipped default for any instrument.
- `e08952d` (2026-07-02) — "decouple detection vs fit mask" — **reverted 15 min
  later by `b60a16c`** (no stated reason). Correct instinct (separate the
  detection mask from the fit mask) but ramp-based and MIRI-scoped.
- **`#40` `satstar-truly-saturated`** — the real fix, cleaner route: the *finder*
  detects on **`SATURATED & DO_NOT_USE`** (truly-lost cores; `DO_NOT_USE` = the
  ramp produced no valid value = 0 good groups) + a min-lost-core-size gate. Uses
  the crf `DO_NOT_USE` bit → **no ramp file, all instruments, on by default**
  (fallback to full SATURATED when `DO_NOT_USE` is absent; gate
  `SATSTAR_REQUIRE_DO_NOT_USE`). Recovered wings are now *unmasked and fit*, so
  there is no divot/ring — the pixels e08952d re-masked shouldn't have been masked
  at all. Frame-wide 209 → 84 seeds.
- **`#41` `fix-daophot-truly-lost-sat`** (stacked on #40) — same restriction on
  the **daophot** path (`_prepare_frame_for_photometry`): `is_saturated =
  truly_lost_saturated_mask(dq)`, so the daophot fit mask *and* the
  `_filter_near_saturation` veto stop dropping recoverable point sources.

**Bottom line:** the mistake was using one **any-group** mask for both detecting
and fitting. The fix keys both channels off the **truly-lost** core
(`SATURATED & DO_NOT_USE`) and fits the recovered wings — instrument-agnostic,
default-on, no ramp file. Landed across PRs #40 (finder) + #41 (daophot); merge
both to close it on `main`.

---

## 3. NIRCam handling

- **Fit engine** (`get_saturated_stars`, `:1116`): per component, mask the
  saturated core (dilated by an **adaptive buffer** scaling with core area —
  NIRCam `scale=0.4, cap=6, min=2`, `compute_adaptive_mask_buffer:610`),
  estimate a local background in an **adaptive annulus** (`:657`), and PSF-fit
  the wings with 1/ERR² weighting. Sources are fit **brightest-core-first** and
  each accepted model is subtracted from the working image before the next fit
  (`:1400`), so neighbouring wings don't double-count.
- **PSF grid by size** (`:1447`): LW (NRCA5/NRCB5) in-FOV stars use `fov=1024`
  (a 512-px grid under-estimates bright LW flux by 50–70%); SW use 512. Off-FOV
  (forced) stars require the **large grid** (2048 SW / 1024 LW) to carry the
  diffraction spikes that reach ~40″ into the frame.
- **Accept gate** (`:1018`, defaults `:1558`): keep if finite `0 < qfit < 5.0`
  (with sidelobe/ssr backstops); `snr > 3.0`. The `ssr_ratio < 1.0` gate is
  **confidence-subordinated** — a high-S/N (>10), good-qfit fit is kept even if
  ssr fails (BFE makes the STPSF first sidelobe brighter than the real star).
- **No** MIRI seed gate, spike-merge, 2-D background, or satstar-coincidence
  exclusion — those are MIRI-only (§4).
- **Detector1 provides the ramp** (`PipelineRerunNIRCAM-LONG.py`): the sibling
  `_ramp.fits` (ZEROFRAME + GROUPDQ) is what the depth-recovery options read.

---

## 4. MIRI handling — differences from NIRCam

MIRI's broader PSF (7–25 µm), stronger extended emission, higher detector-edge
glow, and lack of BFE/IPC sidelobes drive a different configuration. Triggered by
`miri_tuning` (`cataloging.py:2590`, on for all-MIRI runs unless `--no_miri_tuning`).

| mechanism | NIRCam | MIRI | file:line |
|---|---|---|---|
| Accept: qfit_max | 5.0 | **15.0** | `saturated_star_finding.py:1558` |
| Accept: sidelobe_min | −10.0 | **−40.0** | `:1559` |
| Accept: ssr_ratio_max | 1.0 | **2.0** | `:1560` |
| Accept: snr_min | 3.0 | **2.0** | `:1561` |
| ssr gate | confidence-subordinated | **dropped** (finite qfit only) | `:1039` |
| Mask buffer | scale 0.4 / cap 6 | **scale 0.8 / cap 12** (wider charge bleed) | `:1720` |
| Spike-satellite merge | off | **on** (`gap=3`, ratio 3:1) | `:161`, `:1230` |
| PSF grid in-FOV | small unless LW | **large when sat-area ≥ 200 px** (spikes set amplitude) | `:1477` |
| 2-D local background | off | **on in-FOV** (median-filter, removes emission) | `:1811` |
| Extended-emission seed gate | off | **on** (prominence / core / concentration) | `:1292` |
| Position fit | bounded (+size term) | **bounded to 1.5×FWHM** (env `MIRI_SATSTAR_BOUNDED_FIT`) | `:2172` |
| Satstar-coincidence daophot exclusion | off (`0`) | **on** (`1.5×FWHM`) | `cataloging.py:1571` |
| Prominence-SNR schedule | n/a | **8.0 (m12–m4) → 3.0 (m5–m6)** | `cataloging.py:2856` |
| Coarse-bg detection | off | **51-px median on raw phases** | `cataloging.py:2601` |
| m7 cross-band merge | run | **skipped** (per-phase schedule replaces it) | `cataloging.py:2591` |

**MIRI seed gate (phantom rejection, `:1292`):** MIRI broadbands (F770W, F2100W)
saturate on *nebulosity*, sprouting dozens of non-stellar DQ-SATURATED
components that, fit as PSFs, become phantom bright stars (deep negative pits in
the residual). The gate drops a component unless it is a compact bright star:
`_seed_prominence` (`:876`) ≥ `seed_prominence_min` (8.0), core ≥ `seed_core_min`
(1000), concentration ≥ `seed_conc_min` (1.3), measured on the **deep coadd**
(frame-invariant) when available. `robust=True` uses a neighbour-immune metric
(25th-pct + lower-half MAD). Env overrides: `MIRI_SATSTAR_SEED_{PROM,CORE,CONC}_MIN`,
`MIRI_SATSTAR_SEED_PROM_ROBUST`.

**MIRI first-group DQ** (`correct_dq_first_group_saturation`, `:3218`, env
`MIRI_FIRSTGROUP_SAT_DQ`, default off): clears the SATURATED bit on pixels that
saturate only in *later* ramp groups (recoverable), keeping only truly
unrecoverable first-group saturation.

---

## 5. Photometry depth: frame-0 (ZEROFRAME) & filter choice

Saturation depth — how bright a star can be before its core is unrecoverable —
is set by **two levers**: which frame reads you use, and which filter.

### 5a. The ramp-read ladder (deep → bright ceiling)

A JWST integration is a ramp of N group reads. The **full ramp** (default `_cal`)
is the deepest but saturates at the lowest flux. Earlier reads saturate at higher
flux, trading depth for a brighter saturation ceiling:

| tier | source | saturation ceiling | flag | when |
|---|---|---|---|---|
| **full ramp** | `_cal` / `_crf` | lowest (deepest) | — (default) | faint-star photometry |
| **ZEROFRAME rim recover** | group-0 rim of `_ramp.fits` | ~N_group × higher on the *rim* | `--satstar-zeroframe-recover` (+`--satstar-zeroframe-dilate`, default 3) | bright stars leaving a positive ring/dot in the residual |
| **ZEROFRAME core deblend** | group-0 peaks of `_ramp.fits` | resolves cores that touch in `_cal` | `--deblend-satstars` | crowded fields where bright cores merge (gc2211) |
| **(MIRI) first-group DQ** | group-0 GROUPDQ | keep only unrecoverable core | env `MIRI_FIRSTGROUP_SAT_DQ` | MIRI over bright background |

- **`--satstar-zeroframe-recover`** (`zeroframe_recover_saturated`, `:389`):
  the `_cal` rim is *inflated* above truth by charge migration (brighter-fatter);
  group-0 (read before migration) gives the true profile. Rewrites inflated rim
  pixels with `R×group0` (`R` = median `cal/group0` over bright unsaturated px).
  Where group-0 itself saturates (deep core), unrecoverable → PSF-model fallback.
- **`--deblend-satstars`**: in crowded GC fields two bright cores can share one
  DQ blob so the single seed lands *between* the stars. The ZEROFRAME (saturates
  ~N_group higher) resolves the individual cores → one seed per star. Auto-
  degrades to legacy when a frame has no sibling `_ramp.fits`.
- **Requires `_ramp.fits`** next to the `_cal`/`_crf` (a Detector1 product). All
  three are no-ops when it's absent (e.g. the sickle reduction has none).

### 5b. Filter choice

A narrower filter saturates at a **brighter** magnitude than a broad/medium band
(less in-band flux per pixel), so filter selection is itself a depth-vs-
saturation-ceiling choice. Set with `--filternames`; the per-filter
`_SATSTAR_DATA_FLOOR` / `_NIRCAM_SAT_DATA_FLOOR` (§2) and per-filter clamp tuning
(`--satstar-oversub-clamp-percentile`, lower for single-detector LW over bright
background e.g. F335M) reflect this. Example: F466N (narrow) keeps stars
unsaturated several mag brighter than F410M (medium) at the same exposure.

### 5c. Suggested depth presets

There is **no single `--depth` convenience flag** today — depth is assembled from
the filter + ZEROFRAME flags. Practical combinations:

- **Standard** (faint-limited): full ramp, no ZEROFRAME. Default.
- **Bright-preserving**: `--satstar-zeroframe-recover` (+ narrow filter) to keep
  the brightest stars clean.
- **Crowded + bright** (GC cores): `--deblend-satstars --satstar-zeroframe-recover
  --group --max-group-size=10..15`.
- **MIRI over nebulosity**: `miri_tuning` defaults + tune the seed gate
  (`MIRI_SATSTAR_SEED_*`) and, if needed, `MIRI_FIRSTGROUP_SAT_DQ`.

*(A future `--satstar-depth={standard,bright,crowded}` preset that expands to
these flag sets would remove the footgun of setting them individually.)*

---

## 6. Downstream integration

- **daophot stays off saturated cores**: daophot fits run on the
  **satstar-model-subtracted** data (`cataloging.py:165`), and two gates reject
  daophot fits that are really satstar-wing artifacts:
  - `--satstar-artifact-ratio` (default 1.0) + `--satstar-artifact-sigK` (3.0):
    drop a daophot fit where `dao_model < ratio × satstar_model` inside the gate
    (`_filter_satstar_artifacts`, `:249`).
  - **(MIRI)** coincidence exclusion: drop daophot fits within `1.5×FWHM` of a
    satstar entry (`cataloging.py:1571`; off for NIRCam).
- **Off-FOV over-subtraction clamp**: forced (off-field) satstar models are
  clamped to the data (`--satstar-oversub-clamp-percentile`, default 10 → clamp
  90% of the >5σ footprint) so deep spikes don't leave negative pits.
- **Cross-frame flux reconciliation** (`:708`): the same off-FOV star fit in many
  frames is reconciled — trust the frame whose detector centre is closest (sees
  the highest-S/N spikes), reject high runaways, floor against single-frame
  under-subtraction.
- **Merged catalog** (`merge_catalogs.py:1703`+): per-exposure satstar catalogs
  are consolidated and deduped (~0.15″, keep brightest), then merged into the
  daophot catalog; satstar-only rows are marked **`replaced_saturated=True`**
  (per-filter `replaced_saturated_{FILTER}` in cross-filter merges).
- **Products**: `*_satstar_{catalog,model,residual,flags}.fits`. The flags image
  is a uint8 bitmask — bit 1 partly saturated (recoverable), bit 2 totally
  saturated (NaN-variance core), bit 4 included in an accepted satstar fit.

---

## 7. Flags & environment reference

### CLI options (`crowdsource_catalogs_long.py`)

| flag | dest | default | purpose |
|---|---|---|---|
| `--saturation-data-floor` | `saturation_data_floor` | −1.0 | mask a SATURATED px only if data > floor; −1 = per-filter auto, 0 = mask all, >0 = explicit |
| `--desaturated` / `-d` | `desaturated` | False | use the satstar-removed image |
| `--fit-satstar-outside-fov` / `--no-…` | `fit_satstar_outside_fov` | None (auto) | fit stars whose centers are off-frame (auto: on full-frame, off cutout) |
| `--deblend-satstars` | `deblend_satstars` | False | ZEROFRAME-deblend touching saturated cores |
| `--satstar-zeroframe-recover` | `satstar_zeroframe_recover` | False | de-inflate brighter-fatter rim from group-0 |
| `--satstar-zeroframe-dilate` | `satstar_zeroframe_dilate` | 3 | DQ-mask dilation (px) for rim recovery |
| `--satstar-artifact-ratio` | `satstar_artifact_ratio` | 1.0 | reject daophot fits dimmer than the satstar wing; 0 = off |
| `--satstar-artifact-sigK` | `satstar_artifact_sigK` | 3.0 | gate applies where satstar_model > sigK × median(err) |
| `--satstar-oversub-clamp-percentile` | `satstar_oversub_clamp_percentile` | 10.0 | off-FOV clamp scale; lower (1–2) for bright-bg LW |
| `--manual-ext-recover-satstar-guard-arcsec` | … | 2.0 | drop recovered sources within N″ of a satstar (spike guard) |
| `--group` / `--manual-group-min-sep-fwhm` / `--max-group-size` | … | off / 2.0 / unlimited | joint fitting of close pairs (cleaner cores in crowded fields) |
| `--filternames` / `-f` | `filternames` | (list) | filter set — sets the intrinsic saturation ceiling |

### Environment variables (`saturated_star_finding.py`, `PipelineMIRI.py`)

| var | default | effect |
|---|---|---|
| `SATSTAR_DATA_FLOOR` | per-filter | override the finder data floor |
| `MIRI_FIRSTGROUP_SAT_DQ` | 0 | MIRI: keep only first-group (unrecoverable) saturation |
| `MIRI_SATSTAR_SPIKE_MERGE` / `…_RATIO` | 3 / 3.0 | spike-satellite merge gap / size ratio |
| `MIRI_SATSTAR_SEED_{PROM,CORE,CONC}_MIN` | 8.0 / 1000 / 1.3 | seed-gate thresholds |
| `MIRI_SATSTAR_SEED_PROM_ROBUST` | 0 | neighbour-robust prominence metric |
| `MIRI_SATSTAR_BOUNDED_FIT` | 1 | bounded (1) vs locked (0) position fit |
| `MIRI_DROP_OFFFP_SATSTAR` | 1 | drop off-footprint auto-detected satstars |
| `MIRI_PROM_SNR` / `…_PROGRESSIVE` / `…_HI` / `…_LO` | — / 0 / 8.0 / 3.0 | prominence-SNR schedule |
| `MIRI_TRIM_*` | E0/W16/R12 (+adaptive) | detector edge-glow trim |
| `STPSF_PATH` | (required) | WebbPSF grid data (set before import) |

---

## 8. Key files

| file | responsibility |
|---|---|
| `reduction/saturated_star_finding.py` | detection, PSF fitting, accept gate, ZEROFRAME recovery, off-FOV reconciliation |
| `reduction/satstar_deblend.py` | ZEROFRAME core deblending (crowded fields) |
| `reduction/PipelineRerunNIRCAM-LONG.py` / `PipelineMIRI.py` | Detector1 → `_ramp.fits` (ZEROFRAME/GROUPDQ); MIRI edge trim |
| `photometry/cataloging.py` | per-frame satstar+daophot integration, `miri_tuning` schedule, artifact/coincidence gates |
| `photometry/crowdsource_catalogs_long.py` | CLI options, manual m12→m8 pipeline |
| `photometry/merge_catalogs.py` | satstar catalog consolidation, dedup, `replaced_saturated` merge |

*See also `PERFORMANCE_BRICK.md` (satstar models are cached/reused) and
`NOTES_star_vs_extended_emission.md`.*
