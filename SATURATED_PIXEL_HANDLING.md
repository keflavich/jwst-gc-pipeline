# Saturated-Pixel Handling

How the pipeline detects, models, and photometers **saturated stars** — the
bright sources whose cores exceed the detector full-well and are flagged
`SATURATED` in the DQ plane. Saturated cores carry no usable flux, so the star's
brightness and position must be recovered from its **unsaturated wings and
diffraction spikes** (PSF fit with the core masked), and optionally from the
**ramp first read (ZEROFRAME / group-0)**, which saturates at a much higher flux.

NIRCam and MIRI are handled differently — different detector physics (PSF width,
IPC / charge migration, extended emission, edge glow) drive different masking,
fitting, and vetting. Both share one engine:
`jwst_gc_pipeline/reduction/saturated_star_finding.py`.

Code references here are **symbol names**, not `file.py:NNN` line numbers —
line citations went stale silently across refactors (a 2026-07-30 audit found
most of them pointing at unrelated code). `git grep -n '<symbol>'` resolves any
of them; `tests/test_doc_code_references.py` keeps line citations from coming
back.

---

## 1. Pipeline of a saturated star

1. **Detect** — read the `SATURATED` DQ bit, reject `JUMP_DET` cosmic rays, keep
   clusters ≥ 3 px (`find_saturated_stars`).
2. **Filter spurious flags** — drop DQ-SATURATED components sitting on *faint*
   data (persistence / JUMP mis-tags) via a per-filter **data floor**
   (`_SATSTAR_DATA_FLOOR`; tests in `reduction/tests/test_satstar_data_floor.py`).
3. **Label + refine** — connected components (`ndimage.label`), remove large
   edge bleeds (preserving genuine NaN-variance cores), refine centroids on the
   real core (`_refine_coms_by_data`).
4. **(MIRI) merge spike satellites** — fold diffraction-spike fragments into the
   parent core (`_merge_spike_satellites`; off for NIRCam).
5. **(opt) Deblend touching cores** — split merged saturated blobs into one seed
   per star using the ZEROFRAME (`--deblend-satstars`, `satstar_deblend.py`).
6. **Fit** — PSF fit with the core masked, adaptive mask buffer + background
   annulus, inverse-variance (1/ERR²) weighting, brightest-first with iterative
   subtraction (`get_saturated_stars`).
7. **(MIRI) seed gate** — drop extended-emission "phantom" components by
   prominence / core-brightness / concentration (`_seed_prominence` /
   `_seed_concentration`, gated inside `get_saturated_stars`).
8. **Accept gate** — keep the fit on qfit / sidelobe / ssr / snr
   (`accept_satstar_fit`; defaults are that function's keyword defaults).
9. **(opt) ZEROFRAME rim recovery** — de-inflate the charge-migration rim using
   group-0 (`zeroframe_recover_saturated`; `--satstar-zeroframe-recover`).
10. **(opt) Off-FOV stars** — fit bright stars whose *centers* are outside the
    frame but whose spikes reach in; reconcile flux across frames
    (`reconcile_outside_fov_satstar_fluxes`; `--fit-satstar-outside-fov`).
11. **Output** — `*_satstar_catalog.fits`, `*_satstar_model.fits`,
    `*_satstar_residual.fits`, `*_satstar_flags.fits`.
12. **Photometry integration** — daophot fits on the satstar-model-subtracted
    data; artifact/coincidence gates keep daophot off the saturated cores; the
    satstar catalog is merged in with `replaced_saturated=True`
    (`cataloging.py`, `merge_catalogs.py`).

---

## 2. Detection & DQ flagging (shared)

- **SATURATED bit + CR rejection** (`find_saturated_stars`): extract
  `dqflags.pixel['SATURATED']`, reject `JUMP_DET`, require ≥3-px clusters to
  separate ramp non-linearity from single-pixel CRs.
- **Data floor** (`_SATSTAR_DATA_FLOOR` / `_resolve_satstar_data_floor` in
  `saturated_star_finding.py`): a DQ-SATURATED component is only trusted if the
  max data value in its wings exceeds the per-filter floor (MJy/sr), or its core
  is NaN-variance (genuinely unrecoverable). Guards against persistence / JUMP
  artifacts mis-tagged SATURATED on faint sources.

  | filters | satstar-finder floor (`_SATSTAR_DATA_FLOOR`) | daophot-mask severity floor (`SAT_SEVERITY_FLOOR`) |
  |---|---|---|
  | F187N | 1000 | **8000** |
  | F140M F162M F405N F480M | 1000 | 5000 |
  | F182M F210M | 1000 | 4000 |
  | F335M F360M F410M | 800 | 2500 |
  | F466N | 0 (off) | 5000 † |
  | F115W F200W F212N | 0 (off) | 4000 † |
  | F356W F444W | 0 (off) | 2500 † |
  | anything else | 0 (off) | 0 (gate off) |

  † channel-sibling estimate, not yet measured from per-field crf data.

  Both tables live in `reduction/saturated_star_finding.py`; the severity floor is
  resolved by `_resolve_satstar_severity_floor` (explicit arg > env
  `SATSTAR_SEVERITY_FLOOR` > per-filter table > 0 = off).

  Override: `--saturation-data-floor` (photometry), env `SATSTAR_DATA_FLOOR`
  (finder). `-1` = per-filter auto; `0` = mask all SATURATED; `>0` = explicit.

---

## 2b. The *any-group* SATURATED bit — detection and fitting are DECOUPLED

**Status: resolved 2026-07-04, but not the way the plan below expected — read this
before "fixing" the detection mask again.**

The cal/crf `SATURATED` bit is set on every pixel that saturates in *any* ramp
group, including pixels the ramp fitter fully recovered from good earlier groups.
Only a pixel whose **first** group saturates is genuinely unrecoverable, and only
those carry `DO_NOT_USE` / a NaN `VAR_POISSON`.

Using that single any-group mask for BOTH detection and fitting was the original
defect: it over-flagged the saturated set, masked recoverable data, and swept real
stars on bright emission into the satstar channel while vetoing them from daophot.
The fix was **not** to restrict detection — it was to split the two uses:

| channel | mask used today | where |
|---|---|---|
| satstar **seeding** | the **full** any-group `SATURATED` mask (≥3-px clusters, `JUMP_DET` disambiguated, per-filter data floor) | `find_saturated_stars` |
| satstar **fit** | `saturated & _unrecoverable`, with `_unrecoverable = isnan(VAR_POISSON)`; the recovered wings are **kept and fit** | `get_saturated_stars` |
| daophot fit mask + `_filter_near_saturation` veto | any-group `SATURATED`, narrowed to pixels whose data exceeds the per-filter `SAT_SEVERITY_FLOOR` | `_prepare_frame_for_photometry` (`cataloging.py`) |

`get_saturated_stars` also records the split per pixel in its flag image: bit 1 =
saturated-but-recovered, bit 2 = unrecoverable.

**Why seeding must stay on the full mask.** A moderately saturated star whose core
was frame-0-recovered has NO truly-lost pixel, so a seed restricted to
`SATURATED & DO_NOT_USE` never seeds it — and daofind cannot fit its saturated
core either, so it is never cataloged at all. That dropped real W51 cluster stars
(A/B). The truly-lost seed restriction therefore survives only as an opt-in debug
switch, **`SATSTAR_SEED_REQUIRE_DO_NOT_USE=1`, default OFF** (note the `_SEED_`;
the earlier plan called it `SATSTAR_REQUIRE_DO_NOT_USE`).

**Empirical scale of the over-flagging** (MIRI sgrb2 F770W,
`jw05365998001_02101_00003`): the cal DQ flags 3,114 px SATURATED; only **345
(11%)** are first-group (truly unrecoverable); **2,769 (89%)** saturated only in
later groups, and **62% of those carry finite recovered flux** (median ~760
MJy/sr). That is why the any-group bit is unusable as a *fit* mask — and why the
89% recovered wings are now fit rather than masked. Documented downstream harm
from the old single-mask behaviour: cloudc 2526 F770W — 28 by-eye-real stars fused
into saturated blobs and lost from the catalog (commit `e4039e6`).

**Still opt-in / instrument-scoped:** the first-group DQ correction
(`first_group_saturation_mask` / `correct_dq_first_group_saturation`) clears the
SATURATED bit on later-group-only pixels using a sibling `_ramp.fits` GROUPDQ. It
is **MIRI-only and env-gated `MIRI_FIRSTGROUP_SAT_DQ` (default `0`)**, and returns
`dq` unchanged when disabled, non-MIRI, or when no ramp file exists. NIRCam relies
on the decoupling above plus the severity floor, not on this correction.

**History (why the code looks like this):**
- `8837408` (2026-05-04, first tracked commit) — the finder used the any-group bit
  for everything; the satstar finder predates the git history.
- `e4039e6` (2026-06-30) — added `first_group_saturation_mask` /
  `correct_dq_first_group_saturation`. MIRI-only, opt-in, default off — still true
  today.
- `e08952d` (2026-07-02) — "decouple detection vs fit mask", **reverted 15 min
  later by `b60a16c`**. The instinct was right (that decoupling is what shipped),
  but it was ramp-based and MIRI-scoped.
- `#40` `satstar-truly-saturated` — proposed restricting the *finder* to
  `SATURATED & DO_NOT_USE` on by default. **Superseded:** that seed restriction is
  now opt-in and OFF (see above); what survived is the *fit*-side truly-lost mask
  and the unmasked, fitted recovered wings.
- `#41` `fix-daophot-truly-lost-sat` — proposed `is_saturated =
  truly_lost_saturated_mask(dq)` on the daophot path. **Not what shipped:** no
  `truly_lost_saturated_mask` exists. The daophot path keeps the any-group mask
  and narrows it with the per-filter data-severity floor
  (`--saturation-data-floor`, default `-1` = per-filter auto), which addresses the
  same harm — real point sources mis-tagged SATURATED on bright emission — without
  dropping genuinely saturated cores from the mask.

**Bottom line:** the mistake was using one **any-group** mask for both channels.
Seeding uses the full mask (or moderate saturated stars vanish); the fit excludes
only NaN-variance truly-lost pixels; daophot gates on data severity. Do not
re-restrict the seeding mask.

---

## 3. NIRCam handling

- **Fit engine** (`get_saturated_stars`): per component, mask the
  saturated core (dilated by an **adaptive buffer** scaling with core area —
  NIRCam `scale=0.4, cap=6, min=2`, `compute_adaptive_mask_buffer:610`),
  estimate a local background in an **adaptive annulus**
  (`compute_adaptive_bkg_annulus`), and PSF-fit
  the wings with 1/ERR² weighting. Sources are fit **brightest-core-first** and
  each accepted model is subtracted from the working image before the next fit,
  so neighbouring wings don't double-count.
- **PSF grid by size** (`get_psf(..., fov_pixels=1024)`): LW (NRCA5/NRCB5) in-FOV stars use `fov=1024`
  (a 512-px grid under-estimates bright LW flux by 50–70%); SW use 512. Off-FOV
  (forced) stars require the **large grid** (2048 SW / 1024 LW) to carry the
  diffraction spikes that reach ~40″ into the frame.
- **Accept gate** (`accept_satstar_fit`, defaults in its signature): keep if finite `0 < qfit < 5.0`
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
`miri_tuning` (`cataloging.run_manual_pipeline`, on for all-MIRI runs unless `--no_miri_tuning`).

| mechanism | NIRCam | MIRI | file:line |
|---|---|---|---|
| Accept: qfit_max | 5.0 | **15.0** | `accept_satstar_fit` |
| Accept: sidelobe_min | −10.0 | **−40.0** | `accept_satstar_fit` |
| Accept: ssr_ratio_max | 1.0 | **2.0** | `accept_satstar_fit` |
| Accept: snr_min | 3.0 | **2.0** | `accept_satstar_fit` |
| ssr gate | confidence-subordinated | **dropped** (finite qfit only) | `accept_satstar_fit` |
| Mask buffer | scale 0.4 / cap 6 | **scale 0.8 / cap 12** (wider charge bleed) | `compute_adaptive_mask_buffer` |
| Spike-satellite merge | off | **on** (`gap=3`, ratio 3:1) | `_merge_spike_satellites` |
| PSF grid in-FOV | small unless LW | **large when sat-area ≥ 200 px** (spikes set amplitude) | `_use_large_infov` in `get_saturated_stars` |
| 2-D local background | off | **on in-FOV** (median-filter, removes emission) | `bg2d` block in `get_saturated_stars` |
| Extended-emission seed gate | off | **on** (prominence / core / concentration) | `_seed_prominence` / `_seed_concentration` |
| Position fit | bounded (+size term) | **bounded to 1.5×FWHM** (env `MIRI_SATSTAR_BOUNDED_FIT`) | `_miri_bounded` block in `get_saturated_stars` |
| Satstar-coincidence daophot exclusion | off (`0`) | **on** (`1.5×FWHM`) | "MIRI satstar-coincidence drop" in `cataloging.py` |
| Prominence-SNR schedule | n/a | **8.0 (m12–m4) → 3.0 (m5–m6)** | `MIRI_PROM_SNR*` block in `cataloging.py` |
| Coarse-bg detection | off | **51-px median on raw phases** | `coarse_bg_box` in `cataloging.py` |
| m7 cross-band merge | run | **skipped** (per-phase schedule replaces it) | `run_manual_pipeline` (`phases.remove('m7')`) |

**MIRI seed gate (phantom rejection, `_seed_prominence` / `_seed_concentration`):** MIRI broadbands (F770W, F2100W)
saturate on *nebulosity*, sprouting dozens of non-stellar DQ-SATURATED
components that, fit as PSFs, become phantom bright stars (deep negative pits in
the residual). The gate drops a component unless it is a compact bright star:
`_seed_prominence` ≥ `seed_prominence_min` (8.0), core ≥ `seed_core_min`
(1000), concentration ≥ `seed_conc_min` (1.3), measured on the **deep coadd**
(frame-invariant) when available. `robust=True` uses a neighbour-immune metric
(25th-pct + lower-half MAD). Env overrides: `MIRI_SATSTAR_SEED_{PROM,CORE,CONC}_MIN`,
`MIRI_SATSTAR_SEED_PROM_ROBUST`.

**MIRI first-group DQ** (`correct_dq_first_group_saturation`, env
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

- **`--satstar-zeroframe-recover`** (`zeroframe_recover_saturated`):
  the `_cal` rim is *inflated* above truth because charge from the saturating core
  **migrates/blooms outward** during the integration — a near-saturation
  well-overflow effect, **not** the classical brighter-fatter effect (BFE) and
  not IPC. Verified on **NIRCam** (sickle F210M: rim `_cal` sits ~15% above the
  ramp first read, `recovered/cal ≈ 0.85`), so it is not MIRI-specific; it
  applies to any saturating source with a `_ramp.fits`. Group-0 (read before the
  migration) gives the true profile → rewrite inflated rim pixels with `R×group0`
  (`R` = median `cal/group0` over bright unsaturated px).
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
  **satstar-model-subtracted** data (`_filter_satstar_artifacts` call site in
  `cataloging.py`), and two gates reject
  daophot fits that are really satstar-wing artifacts:
  - `--satstar-artifact-ratio` (default 1.0) + `--satstar-artifact-sigK` (3.0):
    drop a daophot fit where `dao_model < ratio × satstar_model` inside the gate
    (`_filter_satstar_artifacts`).
  - **(MIRI)** coincidence exclusion: drop daophot fits within `1.5×FWHM` of a
    satstar entry ("MIRI satstar-coincidence exclusion" in `cataloging.py`; off for NIRCam).
- **Off-FOV over-subtraction clamp**: forced (off-field) satstar models are
  clamped to the data (`--satstar-oversub-clamp-percentile`, default 10 → clamp
  90% of the >5σ footprint) so deep spikes don't leave negative pits.
- **Cross-frame flux reconciliation** (`reconcile_outside_fov_satstar_fluxes`): the same off-FOV star fit in many
  frames is reconciled — trust the frame whose detector centre is closest (sees
  the highest-S/N spikes), reject high runaways, floor against single-frame
  under-subtraction.
- **Merged catalog** (`merge_catalogs.py`, the satstar consolidation +
  `replace_saturated`): per-exposure satstar catalogs
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
| `--satstar-zeroframe-recover` | `satstar_zeroframe_recover` | False | de-inflate charge-migration rim from group-0 |
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
| `MIRI_TRIM_*` | E0/W16/R12 (+adaptive: `MIRI_TRIM_EAST_ADAPT` 1, `_THRESH` 0.08, `_MAX` 96) | detector edge-glow trim |
| `MIRI_SATSTAR_RENDER_FOOTPRINT` | 1 | render the model only inside the star's footprint |
| `MIRI_SATSTAR_WING_FLOOR` | 5.0 | wing data floor for the render |
| `MIRI_SATSTAR_FLATTOP` / `…_PLATEAU_FRAC` | 0 / 0.15 | replace the model inside a flat-topped core with the data |
| `MIRI_SATSTAR_PHANTOM_FLUX_FLOOR` / `…_SSR_MAX` / `…_RATIO_MAX` | 0 (off) / 50 / 50 | post-fit bright-phantom rejection (F770W-calibrated floor 1e5 set in the launcher) |
| `MIRI_DAOPHOT_PROM_ROBUST` | 0 | neighbour-robust prominence on the daophot side |
| `MIRI_EDGE_DETECT_MARGIN` | 8 | detection margin at the detector edge |
| `MIRI_RESID_PIT_NMAD` / `…_DILATE` | 15.0 / 2 | residual-pit masking |
| `SATSTAR_SEED_REQUIRE_DO_NOT_USE` | 0 (**off**) | restrict SEEDING to truly-lost cores (debug only — see §2b) |
| `SATSTAR_SEVERITY_FLOOR` | per-filter `SAT_SEVERITY_FLOOR` | override the daophot-mask severity floor |
| `SATSTAR_MIN_LOST_CORE` | 5 | min truly-lost core size (px) |
| `SATSTAR_SUBFLOOR_SEED_FRAC` | 0.35 | sub-floor seeding fraction |
| `SATSTAR_COMPONENT_OVERLAP_FRAC` / `…_MIN_PX` | (see code) / 250 | merge overlapping saturated components |
| `SATSTAR_WINGCAL` | 1 | wing self-calibration (`apply_wing_selfcal`) |
| `SATSTAR_ZEROFRAME_FIT` | 1 | fit using the ZEROFRAME where available |
| `SATSTAR_LOG_VERBOSE` | 0 | verbose finder logging |
| `SATSTAR_DEDUP_ARCSEC` | 0.15 | consolidation dedup radius (`merge_catalogs`) |
| `SATSTAR_REPLACE_RADIUS_ARCSEC` | (see code) | satstar→daophot replacement radius |
| `SATSTAR_FP_*` (14 vars: `_REJECT`, `_REJECT_RATIO`, `_REJECT_MIN_N`, `_REJECT_BRIGHTFRAC`, `_FLUXRATIO`, `_MERGE_MAX_ARCSEC`, `_COMP_ARCSEC`, `_BIG_ARCSEC`, `_BIGCORE_ARCSEC`, `_BIGCORE_MERGE_ARCSEC`, `_PIXSCALE_ARCSEC`, …) | see `merge_catalogs.py` | false-positive rejection / merging at consolidation |
| `STPSF_PATH` | (required) | WebbPSF grid data (set before import) |

The finder/merger read ~50 `SATSTAR_*`/`MIRI_*` variables in total; the table above
covers the ones that change shipped behaviour. `git grep "environ.get('SATSTAR"`
and `…'MIRI` is the authoritative list.

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
