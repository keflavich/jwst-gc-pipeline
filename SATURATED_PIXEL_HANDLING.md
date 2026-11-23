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

Code references here are **symbol names**. Line citations went stale silently
across refactors (a 2026-07-30 audit found most of them pointing at unrelated
code), so `git grep -n '<symbol>'` resolves any reference here, and
`tests/test_doc_code_references.py` holds the symbol-name form in place.

---

## 1. Pipeline of a saturated star

1. **Detect** — read the `SATURATED` DQ bit, reject `JUMP_DET` cosmic rays, keep
   clusters ≥ 3 px (`find_saturated_stars`).
2. **Filter spurious flags** — drop DQ-SATURATED components sitting on *faint*
   data (persistence / JUMP mis-tags) via a per-filter **data floor**
   (`_SATSTAR_DATA_FLOOR`; tests in `reduction/tests/test_satstar_data_floor.py`).
3. **Label + refine** — connected components (`ndimage.label`), remove large
   edge bleeds (preserving genuine NaN-variance cores), refine centroids on the
   real core (`_refine_coms_by_data`). The refinement's core leaves out
   NaN-variance pixels flagged OUTLIER or by the bad-pixel mask, and the
   pieces of fewer than 3 px left after that (`SATSTAR_SEED_CORE_DQ`, #1098).
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
   (`accept_satstar_fit`). Its keep thresholds are **required** keyword args;
   the per-instrument values are set by the caller in
   `get_saturated_stars` (`_qfit_max_keep = 15.0 if _is_miri else 5.0`, etc.).
   `ssr_trust_snr=10.0` is the one signature default.
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

  | filters | finder wing floor (`_SATSTAR_DATA_FLOOR`) | data-severity floor (`SAT_SEVERITY_FLOOR`) |
  |---|---|---|
  | F187N | 1000 | **8000** |
  | F140M F162M F405N F480M | 1000 | 5000 |
  | F182M F210M | 1000 | 4000 |
  | F335M F360M F410M | 800 | 2500 |
  | F164N | 1000 † | 8000 † |
  | F323N F470N | 1000 † | 5000 † |
  | F466N | 0 (off) | 5000 † |
  | F115W F200W F212N | 0 (off) | 4000 † |
  | F356W F444W | 0 (off) | 2500 † |
  | anything else | 0 (off) | 0 (gate off) |

  † channel-sibling estimate, awaiting measurement from per-field crf data.

  Both tables are keyed by the science bandpass, which is the PUPIL value for
  pupil-wheel exposures (`FILTER=F444W, PUPIL=F405N` is F405N; `F322W2/F323N` is
  F323N; `F150W2/F164N` and `F150W2/F162M` are F164N and F162M), resolved by
  `_satstar_floor_band` through `mast_names.filtername_from_header`. The FILTER
  keyword alone gave those frames the blocker's row. A NIRCam narrow or medium
  band in neither table logs one warning per run and runs ungated.

  Both tables live in `reduction/saturated_star_finding.py`; the severity floor is
  resolved by `_resolve_satstar_severity_floor` (explicit arg > env
  `SATSTAR_SEVERITY_FLOOR` > per-filter table > 0 = off).

  The two columns are **different measurements**: the finder wing floor is
  compared to the component max of a 5-px maximum-filtered SCI (a
  wing/neighbourhood statistic), while the severity floor is compared to raw SCI
  inside the component. Besides narrowing the daophot mask, the severity floor
  drives four finder behaviours (the seed severity gate plus peak-, sub-floor- and
  partner-band seeding, §2b), which is where its largest effect is. MIRI filters
  have no entry, so MIRI daophot masks EVERY any-group SATURATED pixel and the
  original harm (real sources vetoed on bright emission; cloudc 2526 F770W) is
  unmitigated on MIRI.

  **Readout bound** (`satstar_readout_data_floor`): a pixel that reached the
  saturation level S (DN) by the last group has a rate of at least S / t_last, so
  its crf value is at least about S · PHOTMJSR / t_last. On NIRCam the finder wing
  floor is lowered to `FRAC × 50000 × PHOTMJSR / t_last` (MJy/sr) when that value
  is below the table entry; it never raises a table entry or turns on a filter
  whose entry is 0. t_last is the mean time of the last group,
  `((NGROUPS − 1)(NFRAMES + GROUPGAP) + (NFRAMES + 1)/2) × TFRAME`, and FRAC is
  0.5 (env `SATSTAR_DATA_FLOOR_READOUT_FRAC`; 0 restores the table). The table
  values were tuned on a long-ramp field (W51 F480M, SHALLOW2 × 5 groups, rule
  value 213). On the shallow wd2 F410M ramp (SHALLOW4 × 7 groups, t_last 349 s)
  the rule value is 62 against the table 800, and the 800 floor dropped 1037
  components (first saturated group 1–6), most of them 15–16 mag stars that
  daophot then masked.
  On both fields the components with a SATURATED flag split into a group at
  10–80 MJy/sr (flagged pixels that are not stars) and a group that tracks
  S · PHOTMJSR / t_g of their first saturated group; the rule value falls
  between the two. An explicit floor or env `SATSTAR_DATA_FLOOR` takes
  precedence, and the satstar cache key carries `dfr<FRAC>` when the readout
  bound is active.

  Override: `--saturation-data-floor` (photometry), env `SATSTAR_DATA_FLOOR`
  (finder). `-1` = per-filter auto; `0` = mask all SATURATED; `>0` = explicit.

---

## 2b. The *any-group* SATURATED bit — detection and fitting are DECOUPLED

**Status: resolved 2026-07-04. Read this before "fixing" the detection mask again
— what shipped differs from the plan in issues #40/#41 below.**

The cal/crf `SATURATED` bit is set on every pixel that saturates in *any* ramp
group, including pixels the ramp fitter fully recovered from good earlier groups.
Only a pixel whose **first** group saturates is genuinely unrecoverable.

⚠ **`isnan(VAR_POISSON)` means different things at different pipeline stages.**
Measured on one brick F200W exposure (2026-07-30): the `_cal` product has 3200
`SATURATED` px, 43539 `DO_NOT_USE` px and **zero** NaN-`VAR_POISSON` px, while its
`_destreak_o004_crf` sibling has the same 3200 `SATURATED` px with **all 3200**
NaN-`VAR_POISSON`. Check which product you are on before building a gate on it.

Using one any-group mask for BOTH seeding and the daophot veto was the original
defect: it over-flagged the saturated set and swept real stars on bright emission
into the satstar channel while vetoing them from daophot. The fix has two parts:
(a) seeding keeps the full mask and adds positive seeding paths, and (b) daophot
gates on data *severity*:

| channel | mask used today | where |
|---|---|---|
| satstar **seeding** | starts from the full any-group `SATURATED` mask (≥3-px clusters, `JUMP_DET` disambiguated, per-filter wing data floor), then a severity gate **removes** components and three blocks **add** seeds — see below | `find_saturated_stars` |
| satstar **fit** | the current source's saturated component **dilated** by the adaptive mask buffer, plus other sources' saturated pixels — for a **forced / off-FOV** source the *whole* saturated mask is dilated as one mask for all sources, and wing self-cal is skipped. With a ZEROFRAME anchor `zf_deep_core` is substituted for `saturated` and goes through the same dilation. The mask also carries `cutout == 0` and `isnan(cutout)`, and `data` is zeroed where `VAR_POISSON` is NaN — which is *why* truly-lost pixels are excluded in practice. | `get_saturated_stars` |
| daophot fit mask | any-group `SATURATED` (the `_bad_dq_bitmask` OR re-adds every `SATURATED` pixel, so the per-filter `SAT_SEVERITY_FLOOR` narrows only the pixels set to NaN in the detection image, not the fit mask). With `--truly-lost-saturation-in-fit` (default off, #567): `SATURATED & DO_NOT_USE` only, taken before the floor, plus the non-`SATURATED` bad flags | `_prepare_frame_for_photometry` (`cataloging.py`) |
| `_filter_near_saturation` veto | the **raw** any-group `SATURATED` bit from `ctx.dqarr`, recomputed inside the veto; drops fits within `near_sat_dist_pix` (1.0 px on NIRCam). **This veto uses the unnarrowed bit.** With `--truly-lost-saturation-in-fit` the manual path hands the veto a copy of the DQ with `SATURATED` cleared outside the truly-lost set (`ctx.dq_for_satveto`); `ctx.dqarr` stays raw. Opt-in `SATPROX_SEEDED_COMPANION` (default off) exempts a fit at a seed-catalog position when every SATURATED pixel that close is one the hand-off restored to the fit and the fit has finite `flux_fit > 0` and `qfit` (`crowdsource_catalogs_long.py::_seeded_companion_mask`) | `crowdsource_catalogs_long.py` |

`_unrecoverable = isnan(VAR_POISSON)` feeds the per-pixel flag image (bit 1 =
saturated-but-recovered, bit 2 = unrecoverable), centroid refinement, and the
`NIRCAM_SATSTAR_RECOVERED_CAP` flux cap. That cap is off in the bare
function but **cataloging sets it to `1` on the extended-emission NIRCam fields**
(`w51`, `sickle`, `wd2`, `ngc6334`) unless the variable is already in the
environment, so on those four fields `_unrecoverable` does change reported
fluxes. The cap bounds the model peak by the brightest recovered pixel of the
star's own region when the model's peak pixel is itself measured (finite and
recoverable, of either sign), or when the measured pixel nearest the model peak
is at or below zero (a refit of an already-subtracted star, capped to its
residual); otherwise (an unmeasured core) it compares the model with the data
at the measured pixel nearest the model peak (`recovered_cap_flux`: flux ≤
data / unit PSF there), so a recovered ring bounds the model at the ring.
For a blended component (several deblended seeds share one label) the star's
own region is the part of the component nearer its seed than any sibling
seed (`nearest_seed_cell`); without that split, a faint star's region holds
its brighter sibling's core and the unmeasured-core branch reads the
sibling's model-subtracted residual. The unmeasured-core branch is also
skipped when the pixel it reads has a unit PSF below
`NIRCAM_SATSTAR_RECOVERED_MIN_PSF_FRAC` (0.005) of the peak. The
`flux_fit_precap` and `cap_psf_frac` columns record the flux before the cap
and where the cap read the model. One
open question: should frame-0-recovered wings be fit rather than masked?
Tracked in
[#213](https://github.com/keflavich/jwst-gc-pipeline/issues/213). Evidence that the
current masked-core behaviour is deliberate: the wing
self-calibration (`apply_wing_selfcal`, `SATSTAR_WINGCAL`, default on) exists
specifically to correct the bias of masked-core wing fits, and
`NIRCAM_SATSTAR_RECOVERED_CAP` exists because the recovered core is masked.

**Why seeding starts from the full mask.** A moderately saturated star whose core
was frame-0-recovered has no truly-lost pixel, so a seed restricted to
`SATURATED & DO_NOT_USE` misses it — and daofind cannot fit its saturated core
either, so it goes uncataloged. That dropped real W51 cluster stars (A/B). The
restriction survives as an opt-in debug switch,
**`SATSTAR_SEED_REQUIRE_DO_NOT_USE=1`, default OFF**.

**The seed set is the DQ mask reshaped by four blocks.** With a per-filter
`SAT_SEVERITY_FLOOR` in force, `find_saturated_stars` also:

- **removes** components with no NaN-variance core whose peak is below the floor
  (bright-star over-flags — the comment cites brick F182M carrying 362 fake
  satstars, 45% of that catalog);
- **adds** peak-based seeds at bright *unflagged* pixels above the floor;
- **adds** sub-floor seeds above `SATSTAR_SUBFLOOR_SEED_FRAC` × floor (0.35);
- **adds** partner-band seeds where the same star was accepted in a
  near-degenerate partner band (F410M↔F405N, F182M↔F187N).

Measured with the shipped defaults: brick F200W 157 → **350 seeds** (2.23×; mask
3200 → 9140 px; `dqsat` 157 / `peak` 33 / `subfloor` 160, so **55%** of seeds
contain no DQ-saturated pixel at all), brick F182M 161 → **420** (2.61×; 62%).
⚠ F200W's severity floor (4000) is one of the unmeasured †channel-sibling
estimates, so its 2.23× is contingent on a guessed floor; F182M's 4000 is measured.

Those unflagged seeds are intended where the severity floor is the real onset of
core suppression (brick F182M). Where the floor sits far below the flagging
onset they instead pick up ordinary stars: gc-treasury o043 F212N (floor 4000,
measured no-NaN SATURATED onset 38017) has 91.2% of its accepted satstars (915
per frame) with no SATURATED pixel within 2 px (#991). Setting
`SATSTAR_SEED_REQUIRE_SAT_DQ_FILTERS=F212N` drops the `peak` / `subfloor`
seeds of that band whose component holds no DQ-SATURATED pixel; the test reads
the DQ array, not `seed_kind`, because a flagged component the severity gate
dropped is re-seeded as `subfloor`. Partner and sibling seeds are not affected.
Reason about the seed list from these four blocks.

**Empirical scale of the over-flagging** (MIRI sgrb2 F770W,
`jw05365998001_02101_00003`): the cal DQ flags 3,114 px SATURATED; only **345
(11%)** are first-group (truly unrecoverable); **2,769 (89%)** saturated only in
later groups, and **62% of those carry finite recovered flux** (median ~760
MJy/sr). That is the scale by which the any-group bit overstates the truly-lost
set; those recovered wings are masked in the fit today (open question, #213).
Documented downstream harm from the old single-mask behaviour: cloudc 2526 F770W — 28 by-eye-real stars fused
into saturated blobs and lost from the catalog (commit `e4039e6`).

**Still opt-in / instrument-scoped:** the first-group DQ correction
(`first_group_saturation_mask` / `correct_dq_first_group_saturation`) clears the
SATURATED bit on later-group-only pixels using a sibling `_ramp.fits` GROUPDQ. It
is **MIRI-only and env-gated `MIRI_FIRSTGROUP_SAT_DQ` (default `0`)**, and returns
`dq` unchanged when disabled, on non-MIRI data, or when the ramp file is missing.
NIRCam relies on the decoupling above plus the severity floor.

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
  now opt-in and OFF (see above), and the fit masks the dilated saturated
  component. What survived is the *decoupling* of the two channels and the
  data-severity gate.
- `#41` `fix-daophot-truly-lost-sat` — proposed `is_saturated =
  truly_lost_saturated_mask(dq)` on the daophot path. **Superseded:** the daophot
  path keeps the any-group mask and narrows it with the per-filter data-severity
  floor (`--saturation-data-floor`, default `-1` = per-filter auto), which
  addresses the same harm — real point sources mis-tagged SATURATED on bright
  emission — while genuinely saturated cores stay in the mask.

**Bottom line:** the mistake was using one **any-group** mask for both channels.
Seeding starts from the full mask (restricting it to truly-lost cores makes
moderate saturated stars vanish) and is then reshaped by the severity gate and the
three additive blocks; the satstar fit masks the **dilated** saturated component;
daophot gates on data severity. Before changing any of these, read the table above:
the code deliberately avoids all three plausible-sounding "fixes" (restrict seeding
to `DO_NOT_USE`, mask only truly-lost pixels in the fit, narrow
`_filter_near_saturation` with the severity floor). The one exception is the opt-in
`--truly-lost-saturation-in-fit` (default off): it applies the second of those
(truly-lost pixels only in the fit mask, and in the DQ the veto reads) so the B arm
of the #418 comparison can be produced; it is not a default and is not evidence that
the restriction is safe.

---

## 2c. Single-group (G1-only) pixels — `G1ONLY_SAT_DQ` (NIRCam)

A pixel whose ramp has exactly one usable group (group 1 good, every later
group SATURATED) gets its rate from group 1 alone, and the crf DQ stays clean
there (SATURATED reaches the rate DQ only when all groups saturate). Group 1
is the mean of NFRAMES frames, so its effective exposure time is
`t1 = (NFRAMES+1)·TFRAME/2`, not TGROUP. Which time the ramp fit used depends
on the stcal code that ran Detector1:

| Detector1 | ramp | single-group rate | m (below) |
|---|---|---|---|
| jwst 1.14.1 (stcal 1.7.3, python OLS), NGROUPS=2 | `fit_2_group`: group 1 / group_time | G1/TGROUP (0.75x for BRIGHT2) | 1 |
| jwst 1.14.1 (stcal 1.7.3), NGROUPS>2 | `fit_single_read`: group 1 / one_groups_time | G1/t1 (correct) | r |
| jwst 1.21 (stcal 1.18, C fitter), any NGROUPS | `is_0th` (0th group the only good one, its GROUPDQ 0) → one_group_time | G1/t1 (correct) | r |

(Code read from the installed stcal 1.7.3 and 1.18.0. jwst 1.12.1, used for
cloudc 2221, was not read; its NGROUPS=2 frame measures m = 0.99, the biased
value. The table describes code paths; the gate below measures each frame.)

`r = TGROUP / t1` is 1.333 for BRIGHT2 and 3.333 for SHALLOW2.

Measured m (median; 5-95%) on one frame each:

| frame | readout | Detector1 | m | 5-95% | n px | gate |
|---|---|---|---|---|---|---|
| brick 2221 o001 F182M nrcb4 | BRIGHT2, NG=2 | 1.14.1 | 1.000 | 0.972-1.037 | 1538 | flag |
| brick 2221 o001 F410M nrcblong | BRIGHT2, NG=2 | 1.14.1 | 0.998 | 0.963-1.060 | 22724 | flag |
| cloudc 2221 o002 F182M nrcb1 | BRIGHT2, NG=2 | 1.12.1 | 0.991 | 0.960-1.063 | 2589 | flag |
| gc-treasury 10678 o105 F480M nrcblong | BRIGHT2, NG=4 | 1.21 | 1.344 | 1.291-1.447 | 35242 | skip |
| gc-treasury 10678 o078 F480M nrcblong | BRIGHT2, NG=4 | 1.21 | 1.357 | 1.271-1.448 | 44530 | skip |
| gc-treasury 10678 o105 F212N nrcb4 | BRIGHT2, NG=4 | 1.21 | 1.325 | 1.281-1.383 | 1670 | skip |
| gc-treasury 10678 o105 F212N nrcb1 | BRIGHT2, NG=4 | 1.21 | 1.325 | 1.288-1.432 | 1476 | skip |
| brick 1182 o004 F200W nrcb1 | SHALLOW2, NG=7 | 1.14.1 | 3.287 | 3.189-3.533 | 31811 | skip |
| sgrb2 5365 o001 F182M nrcb1 | SHALLOW2, NG=3 | 1.18.0 | 3.317 | 3.212-3.573 | 5875 | skip |

`reduction/g1only_dq.py` reads GROUPDQ from the sibling `_ramp.fits` and, on
frames that need it, ORs `SATURATED | DO_NOT_USE` into the frame DQ at the
single-group pixels and sets SCI/ERR/VAR_* to NaN there. The satstar finder
then seeds them, daophot masks them (`truly_lost_saturated_mask`), and the
satstar first-read recovery rebuilds the core from group 0. It is applied in
`write_perexposure_crf` (persisted in the crf) and in memory by cataloging's
`load_data` and satstar's `remove_saturated_stars`; the satstar fit-switch
key and cache content key carry `'g1'` for flagged frames only.

**The gate.** `G1ONLY_SAT_DQ=auto` (default; also when unset) decides per
frame. It fits `SCI = C·(G2−G1)/TGROUP + B` on clean pixels (GROUPDQ 0 in all
groups, crf DQ clean) through two robust points (median of the faint half,
median of pixels with G2−G1 ≥ 200 DN in the top 10%), then takes the median of
`m = (SCI − B) / ((G1/TGROUP)·C)` over single-group pixels with G1 ≥ 200 DN.
m < (1+r)/2 → flag; otherwise skip. B absorbs destreak / background offsets:
subtracting the median sky, a sky-scaled gradient, or adding 5× sky moves m
by ≤ 0.03 on every frame above.

Fallbacks: fewer than 20 measurable pixels, fewer than 100 bright reference
pixels, no ramp SCI, no NFRAMES/TFRAME, or a non-finite m → **flag** (the pre-gate behaviour;
such frames have few single-group pixels). r < 1.05 (NFRAMES=1, where
G1/TGROUP is correct under either timing) → skip. A crf header that already
carries `G1ONLYDC` reuses that decision (a flagged crf has NaN where m is
measured); `G1ONLYDQ > 0` without `G1ONLYDC` (flagged before the gate) →
flag. An m farther than (r−1)/4 from both 1 and r prints a WARNING
(`G1OnlyAmbiguousWarning`) and still decides at the midpoint.

Header keys: `G1ONLYDQ` (px flagged; 0 on skip), `G1ONLYDC` (flag/skip),
`G1ONLYWY` (source: reason), `G1ONLYM` (median m), `G1ONLYN` (px used),
`G1ONLYR` (r). One log line per application, e.g.
`G1-only DQ (load_data): <file>: skip 36567 single-group px: m=1.344
(n=35242, 5-95% 1.291-1.447) r=1.333 [measured: m>=1.167: unbiased]`.

Effect of the choice on gc-treasury F480M (m7, one full frame, e4 nrcblong):
flagging the unbiased o105 / o078 frames (`G1ONLY_SAT_DQ=1`) gives 52865 /
49601 rows against 53004 / 49732 unflagged, with about 97 / 103 bright
(F ≥ 1e4) stars per frame left without a row. `auto` skips both frames and
returns 53004 / 49732 rows, identical in every column to the unflagged run
(F212N o105 nrcb4: 46271 unflagged, 46237 flagged, 46271 auto).

---

## 3. NIRCam handling

- **Fit engine** (`get_saturated_stars`): per component, mask the
  saturated core (dilated by an **adaptive buffer** scaling with core area —
  NIRCam `scale=0.4, cap=6, min=2`, `compute_adaptive_mask_buffer`),
  estimate a local background in an **adaptive annulus**
  (`compute_adaptive_bkg_annulus`), and PSF-fit
  the wings with 1/ERR² weighting. Sources are fit **brightest-core-first** and
  each accepted model is subtracted from the working image before the next fit,
  so each wing's flux is counted once.
- **Background-scatter error floor** (`SATSTAR_ERR_BKG_SCATTER`, on by default
  for extended-emission NIRCam, `bkg_scatter_fit_error`): the crf ERR holds
  read and Poisson noise only, so in a crowded or nebular field the background
  pixels of the fit box carry far too much weight (wd2 F150W: annulus scatter
  ~12 against a median ERR of ~0.5, ~6500 background pixels per 81×81 box) and
  15–18 mag stars come out 0.4–0.8 mag faint. The robust scatter (1.4826 MAD)
  of the unmasked local-background annulus is added in quadrature to every
  pixel's error before the fit. `flux_err` and `reduced_chi2` are computed
  with these errors: on wd2 F150W nrcb3 the median `flux_err` grows 1.6×
  (3.7× for stars brighter than 13 mag, whose masked cores leave mostly
  background pixels in the fit) and the median `reduced_chi2` falls from 177
  to 44, so neither column is comparable between runs with and without the
  switch. No gate reads `reduced_chi2`; the accept gate's S/N > 3 floor and
  its S/N > 10 bypass of the `ssr_ratio` test do read `flux_err`. The
  floor also largely cancels the optional near-core downweighting
  (`satstar_central_downweight_sigma`, default off) wherever the scatter
  exceeds the downweighted ERR. The switch is part of the satstar cache key
  (`es` in `SATFITSW`), so turning it on refits cached catalogs, including
  those of frames without a ramp.
- **PSF grid by size** (`get_psf(..., fov_pixels=1024)`): LW (NRCA5/NRCB5) in-FOV stars use `fov=1024`
  (a 512-px grid under-estimates bright LW flux by 50–70%); SW use 512. Off-FOV
  (forced) stars require the **large grid** (2048 SW / 1024 LW) to carry the
  diffraction spikes that reach ~40″ into the frame.
- **Accept gate** (`accept_satstar_fit`; thresholds passed in by
  `get_saturated_stars`, not signature defaults). On **NIRCam**: require
  `snr > 3.0` and `flux > 0`; reject on `qfit` only when it is finite, so a
  **non-finite `qfit` passes** (and a negative one does too); `sidelobe > -10σ`
  when finite; the `ssr_ratio < 1.0` backstop is bypassed for a trustworthy fit
  (`snr > 10` and `qfit < 5.0`). MIRI instead requires
  `isfinite(qfit) and 0 < qfit < 15.0` and drops the ssr test entirely
  (`snr > 2.0`, `sidelobe > -40σ`). Keep if finite `0 < qfit < 5.0`
  (with sidelobe/ssr backstops); `snr > 3.0`. The `ssr_ratio < 1.0` gate is
  **confidence-subordinated** — a high-S/N (>10), good-qfit fit is kept even if
  ssr fails (BFE makes the STPSF first sidelobe brighter than the real star).
- The MIRI seed gate, spike-merge, 2-D background, and satstar-coincidence
  exclusion are MIRI-only (§4).
- **Detector1 provides the ramp** (`PipelineRerunNIRCAM-LONG.py`): the sibling
  `_ramp.fits` (ZEROFRAME + GROUPDQ) is what the depth-recovery options read.

---

## 4. MIRI handling — differences from NIRCam

MIRI's broader PSF (7–25 µm), stronger extended emission, higher detector-edge
glow, and lack of BFE/IPC sidelobes drive a different configuration. Triggered by
`miri_tuning` (`cataloging.run_manual_pipeline`, on for all-MIRI runs unless `--no_miri_tuning`).

| mechanism | NIRCam | MIRI | file:line |
|---|---|---|---|
| Accept: qfit_max | 5.0 | **15.0** | set in `get_saturated_stars`, applied by `accept_satstar_fit` |
| Accept: sidelobe_min | −10.0 | **−40.0** | set in `get_saturated_stars` |
| Accept: ssr_ratio_max | 1.0 | **2.0** | set in `get_saturated_stars` |
| Accept: snr_min | 3.0 | **2.0** | set in `get_saturated_stars` |
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
  well-overflow effect, distinct from the classical brighter-fatter effect (BFE)
  and from IPC. Verified on **NIRCam** (sickle F210M: rim `_cal` sits ~15%
  above the ramp first read, `recovered/cal ≈ 0.85`); it applies to any saturating
  source with a `_ramp.fits`. Group-0 (read before the
  migration) gives the true profile → rewrite inflated rim pixels with `R×group0`
  (`R` = median `cal/group0` over bright unsaturated px).
  Where group-0 itself saturates (deep core), unrecoverable → PSF-model fallback.
  Group-0 saturation is read from the ramp GROUPDQ (`SATSTAR_ZF_G0_GROUPDQ`,
  on by default wherever the recovered-core cap is on, i.e. extended-emission
  NIRCam) as well as from the frame-wide ceiling (0.9 × p99 of group 0 at
  SATURATED pixels). With a multi-frame readout (SHALLOW4, MEDIUM8, …) group 0
  averages several frames, and its pile-up level varies by pixel: on wd2 F150W
  nrcb3, 3699 group-0-saturated pixels sat below the ceiling and were rewritten
  to about half the rate the ramp fit took from the ZEROFRAME (#1065).
  Group 0 also flattens once the later frames of a multi-frame group reach the
  pile-up level: group 0 / first frame falls from 2.66 at a predicted
  last-frame fill (NFRAMES × first frame / saturation) of 0–0.2 to 2.44 at
  fill 1.0–1.1 and 1.69 at fill 2–3 (wd2 F150W nrcb3; nrcb1 2.60 → 2.38 →
  1.62). The jwst saturation step flags these reads: an averaged group whose
  next group saturates and whose value exceeds the threshold × mean(frames) /
  last frame (0.625 for SHALLOW4) gets DO_NOT_USE, and every SATURATED pixel
  at fill ≥ 0.9 carries SATURATED or DO_NOT_USE in group 0 (nrcb3 exposure 1:
  1418 SATURATED, 5087 DO_NOT_USE only). `ramp_fit` then has no good group and
  takes the rate from the ZEROFRAME. The anchor without the next switch reads
  only the SATURATED bit, so it rewrites the DO_NOT_USE pixels as R × (flat
  group 0): recovered / crf falls from 1.05 to 0.82 at fill 1.5–2 on nrcb3
  and to 0.77 on nrcb1. With `SATSTAR_ZF_FIRST_FRAME` (on by default with the
  cap) `remove_saturated_stars` loads the ramp ZEROFRAME extension, which
  holds the first frame alone (`_find_first_frame_for`, only for NFRAMES > 1),
  and the group-0 mask with DO_NOT_USE included
  (`_find_group0_saturation_for(do_not_use=True)`). `first_frame_group0`
  measures k = group 0 / first frame as binned medians in log first frame
  over the frame's unflagged SATURATED-buffer pixels with group 0 in
  [R_g0_min, ceiling) (2.675 → 2.561 on nrcb3, 2.622 → 2.502 on nrcb1; the
  ratio drifts with brightness below the flag threshold too), and replaces
  group 0 by k(first frame) × first frame wherever group 0 is flagged and the
  first frame is positive. Pixels whose first frame is 0 (jwst zeroes the
  ZEROFRAME where the first frame saturates) stay deep core. On nrcb3
  exposure 1 it replaces 6617 pixels, and the recovered / crf ratio of the
  flagged pixels stays at 1.04–1.06 from fill 0.2 to 5 (nrcb1: 1.00–1.01);
  unflagged pixels read 1.02 (nrcb1 1.00) with or without the switch.
  DEAD, HOT and REFERENCE_PIXEL pixels are left out of the rewrite
  (`SATSTAR_ZF_RIM_BADPIX`, default on): dead/hot clumps carry the SATURATED bit
  with a group-0 of ~0.001–1.2 DN, and their `R×group0` rewrite, weighted at 5%
  of its value, drove four 16–18 mag wd2 F150W stars to negative flux
  (#1071). A rewritten pixel's fit error is also at least the frame's
  median ERR.
  Wherever the cap is on, SATURATED pixels with a finite, nonzero crf rate
  and no DO_NOT_USE keep that rate (`SATSTAR_ZF_KEEP_FINITE`): the crf
  SATURATED bit is any-group, and `ramp_fit` measured these pixels from their
  unsaturated groups or, for a flagged group 0, from the first frame. Only
  pixels with no rate (SATURATED and DO_NOT_USE, SCI zeroed) are rewritten
  from group 0 or the first frame. On wd2 F150W nrcb3 the first-frame rewrite
  read 5–8% above the crf at the replaced pixels, and the 14.5–16.5 mag
  saturated stars read 0.05 mag brighter than the unsaturated ones against
  dolphot; keeping the crf brought every 0.5 mag bin from 14.5 to 18 mag
  within 0.015 mag of the unsaturated stars (benchmark |dm| < 0.3: 862 →
  863 of 876, MAD 0.025 → 0.018). On F277W nrcblong the 14–17 mag offset went
  from +0.03 to −0.01 (3031 → 3042 of 3257) with the per-magnitude scatter
  0.003–0.006 mag larger (#1092); the default accepts that scatter increase
  for the removal of the offset.
- **`--deblend-satstars`**: in crowded GC fields two bright cores can share one
  DQ blob so the single seed lands *between* the stars. The ZEROFRAME (saturates
  ~N_group higher) resolves the individual cores → one seed per star. Auto-
  degrades to legacy when a frame has no sibling `_ramp.fits`, and the cache
  stamp (`SATRECOV`) then omits the `_dbl2` token for that frame; the
  `SATFITSW` stamp likewise omits its `dv<margin>` token for such a frame. A
  missing scikit-image raises `ImportError` (from `build_deblended_source_records`
  and from a direct `deblend_blob_zeroframe` call), which stops the frame's
  photometry task, and only for a `--deblend-satstars` run on a frame that has
  a ramp. The satstar catalogue meta carries `DBLNCMP` (components deblended),
  `DBLNSPL` (split into more than one star, after the PSF veto), `DBLFAIL`
  (deblend raised, centroid kept) and `DBLNVET` (residual peaks the PSF veto
  rejected; 0 when `SATSTAR_DEBLEND_PSF_MARGIN` is off) for a frame the deblend
  ran on.
  search also reads a PSF ring or spike bead of one bright star as a companion
  when a catalogue source sits on it; the opt-in `SATSTAR_DEBLEND_PSF_MARGIN`
  (default 0 = off) fits the accepted stars with PSF stamps and keeps only the
  residual peaks above that fraction of the model (#1195, synthetic scenes
  only so far; not calibrated on 10678 data). It splits MORE than the plain
  search where the accepted-star positions are wrong: a core clipped by the
  image edge, or two cores merged into one invalid region. Set
  `OMP_NUM_THREADS=1` in any job that turns it on.
- **Requires `_ramp.fits`** next to the `_cal`/`_crf` (a Detector1 product). All
  three are no-ops when it's absent (e.g. the sickle reduction has none).

### 5b. Filter choice

A narrower filter saturates at a **brighter** magnitude than a broad/medium band
(less in-band flux per pixel), so filter selection is itself a depth-vs-
saturation-ceiling choice. Set with `--filternames`; the per-filter
`_SATSTAR_DATA_FLOOR` / `SAT_SEVERITY_FLOOR` (§2) and per-filter clamp tuning
(`--satstar-oversub-clamp-percentile`, lower for single-detector LW over bright
background e.g. F335M) reflect this. Example: F466N (narrow) keeps stars
unsaturated several mag brighter than F410M (medium) at the same exposure.

### 5c. Suggested depth presets

Depth is assembled from the filter + ZEROFRAME flags. Practical combinations:

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
  90% of the >5σ footprint) to keep deep spikes from digging negative pits.
- **Cross-frame flux reconciliation** (`reconcile_outside_fov_satstar_fluxes`): the same off-FOV star fit in many
  frames is reconciled — trust the frame whose detector centre is closest (sees
  the highest-S/N spikes), reject high runaways, floor against single-frame
  under-subtraction.
- **Merged catalog** (`merge_catalogs.py`, the satstar consolidation +
  `replace_saturated`): per-exposure satstar catalogs
  are consolidated and deduped (~0.15″; the brightest exposure's row
  represents the star, and its `flux_fit` is the median over exposures of
  the per-exposure fluxes, `SATSTAR_FLUX_STAT`), then merged into the
  daophot catalog; satstar-only rows are marked **`replaced_saturated=True`**
  (per-filter `replaced_saturated_{FILTER}` in cross-filter merges).
  The pooled wing calibration (`apply_pooled_wingcal`, for rows whose
  per-frame self-cal was skipped) is applied to every per-exposure row
  before the dedup, so the representative and the median see fluxes in one
  calibration state.  On the wd2 dolphot benchmark (#1032) the median moves
  9 of 10 saturated bands 0.02–0.12 mag toward dolphot and F150W 0.12 mag
  away (+0.10 → +0.22 mag fainter than dolphot); the F150W per-exposure fits
  are biased faint, and the brightest-of-N selection happened to offset part
  of that.  `SATSTAR_FLUX_STAT=brightest` restores the old statistic.
- **Post-merge off-FOV cleanup** (`_clean_offfov_dups_and_offfield`): off-FOV
  `replaced_saturated` rows within 1.0″ collapse to one row (the per-frame fits of
  an off-FOV star scatter wider than the 0.15″ dedup). In-field rows follow
  `_infield_dedup_settings`: NIRCam reads `SATSTAR_INFIELD_DEDUP` — `cofit`
  (default; collapse only rows that no per-exposure satstar run fit side by side,
  `_never_cofit_duplicates`; every collapsed set has all its pairs within 1.0″
  and never co-fit), `cofit_residual` (opt-in, below), `none` (keep all), or
  `legacy` (1.0″
  friends-of-friends over every row, which deleted distinct stars, #972). MIRI and
  NIRISS read `SATSTAR_INFIELD_DEDUP_MIRI` / `SATSTAR_INFIELD_DEDUP_NIRISS`
  (default `legacy`; the co-fit rule is unvalidated there). The instrument is
  the data i2d's (or first frame's) `INSTRUME` header when that says MIRI or
  NIRISS, else the module token / `GC_INSTRUMENT_OVERRIDE` / filter name.
  `run_manual_pipeline` rejects a bad value of any of the three variables at
  entry. The extended-emission targets always collapse off-FOV rows only.
- **`SATSTAR_INFIELD_DEDUP=cofit_residual`** (opt-in): the satstar finder fits
  one star per saturated DQ component, so two stars that share a component in
  every frame are never fit side by side and `cofit` collapses them
  (gc-treasury 10678 m7 o078 + o043: about 21 of the 32 collapsed F480M rows
  are such stars, at 0.2–1.0″). In this mode a never-co-fit pair is
  kept as two stars when, in the accepted runs that fit only the better-ranked
  member (by support, then flux), the frame's `*_satstar_residual.fits` at the
  other member holds ≥ `SATSTAR_INFIELD_RESID_RATIO_MIN` (default 0.5) of its
  peak above the local background **and**, summed within 3 px,
  ≥ `SATSTAR_INFIELD_RESID_FLUX_MIN` (default 0.25) of its fitted flux, both in
  the same run (`_satstar_residual_distinct_pairs`). The member must sit
  ≥ 20 px inside the frame. Pairs without such a measurement keep the `cofit`
  answer, and a missing, unreadable, truncated or wrongly shaped residual/frame
  is logged and falls back to `cofit` for the pairs it would have measured. A
  row whose every fit lies within 10 px of the science area, i.e. within 14 px
  of the array edge (4 reference pixels + 10), is ranked last when a collapsed
  set picks its survivor (`_satstar_edge_only_rows`), so an edge-truncated fit no
  longer outranks the interior fit of the same star. On those two observations
  the mode restores 21 F480M and 41 F212N rows and swaps the survivor in 1 / 9
  edge-truncated sets. The peak test alone would keep 103 F212N rows, 62 with
  < 0.1 of their flux left (rows whose every fit is at an edge, where the
  neighbour's model is negligible). The 0.25 flux threshold was tuned on those
  two observations only (kept rows ≥ 0.29, rejected ≤ 0.067) and is untested
  on other fields or filters.
- **`SATSTAR_DAO_RESTORE_FRAC_MIN`** (opt-in; `satstar_dao_restore`; off unless
  above 0): keep the daophot row that a satstar overwrote when the in-field dedup
  (`cofit` or `cofit_residual`) would drop the satstar row.  `replace_saturated`
  writes the satstar's flux and position over the daophot row it pairs with the
  satstar, so the row the dedup drops is the only row the star had.  gc-treasury
  10678 o105 F212N m7: a replay of the `cofit` dedup on the 10354 in-field rows of
  the consolidated satstar catalog drops 57; 55 of the 57 have no row in the
  production m7 merged catalog (the other two were not examined) and 24 of the 55
  have a daophot cluster (1-5 frames) within 80 mas.  Five of the 24 lie 63-79 mas
  from the satstar at 0.002-0.031 of its flux (`dqsat` seeds of single-exposure
  transients, rightly dropped); the other 19 lie within 21 mas of it (18 within
  5 mas) at 0.80-1.06 of its flux.  With the switch on:
  1. the per-frame merge (`merge_individual_frames` -> `replace_saturated`) records
     the flux, flux error and position of each daophot row it overwrites in the
     columns `satstar_dao_flux`, `satstar_dao_flux_err`, `satstar_dao_ra` and
     `satstar_dao_dec` (float64, NaN on every other row and on appended satstar rows;
     32 bytes per row; 18.6 MB for the 579,673 rows of the o105 F212N raw merged
     table);
  2. the dedup keeps a row it would drop, as the daophot row it was, when the
     recorded flux is in [`SATSTAR_DAO_RESTORE_FRAC_MIN`,
     `SATSTAR_DAO_RESTORE_FRAC_MAX`) of the satstar flux (default upper end 1.25,
     exclusive), the recorded position is within `SATSTAR_DAO_RESTORE_SEP_MAX_ARCSEC`
     of the satstar (default 0.03") and no surviving satstar row lies within
     `SATSTAR_DAO_RESTORE_NEIGHBOUR_MIN_ARCSEC` of the recorded position (default
     0.1"; 0 turns the check off).  The kept row has the recorded flux, flux error
     and position, `replaced_saturated` False, `is_saturated` unchanged (the vetting
     force-keep reads it), the satstar ensemble columns (`satstar_nframes`,
     `satstar_nmeas`, `satstar_std_ra`, `satstar_std_dec`) blank, and
     `satstar_dao_restored` True.  Every other row is dropped or kept as without the
     switch.
  The three limits other than `FRAC_MIN` are read only when `FRAC_MIN` is above 0,
  and `run_manual_pipeline` and `merge_individual_frames` reject a bad value of any
  of the four before any work (`FRAC_MAX` must exceed `FRAC_MIN`).  A merged table
  written without the switch has no record: the dedup prints a WARNING and drops
  as before.  A table with pixel or magnitude columns (`x`, `y`, `dx`, `dy`,
  `x_fit`, `y_fit`, `x_err`, `y_err`, `mag_ab`, `mag_vega`, `emag_ab`) restores
  nothing, because the record does not cover them; the per-frame merge's
  `minimal_table` has none.  The restored row carries the daophot photometry.  In
  the replay below the 18 appended daophot rows added no crater, while
  `SATSTAR_INFIELD_DEDUP=cofit_residual`, which restores satstar fits (a larger and
  different set of rows), added four (27 craters deeper than 500 MJy/sr against 23);
  those restored single-frame satstar fits over-subtract by 25-40 % or lie about
  0.2" off.  The defaults come from the o105 F212N recon table of the
  55 dropped rows (the per-frame daophot cluster stands in for the overwritten row,
  which is not on disk): the code with the defaults restores 19 of the 55, namely
  the 24 with a cluster except the five transients (ratio 0.002-0.031, 63-79 mas);
  their nearest surviving satstar row is 0.17-0.90" away, so the neighbour check
  does not change this set; `SEP_MAX_ARCSEC=0.01` restores 18; a ratio window of
  0.0001-2 with `SEP_MAX_ARCSEC=0.1` restores all 24 with a cluster, the transients
  included.  The defaults are untested on any other field or filter, and the
  full-merge effect (m7 + finalize with the switch on, then the residual render) has not
  been measured: the replay below appended the rows to the vetted catalog.
- **Opt-in guards in the merged-catalog residual render**
  (`build_mergedcat_residuals`; each switch is off unless set).  Two kinds of
  artifact stay in the `*_mergedcat_residual_i2d` of a saturated field: craters,
  where the render subtracts a row at a flux the frame's data do not carry, and
  single-exposure transients, which enter the coadd at weight 1/N while no catalog
  model subtracts them.  The counts below are gc-treasury 10678 o105 F212N m7,
  whose production residual holds 489 craters (a crater is a residual minimum
  more than 50 MJy/sr below zero and more than 0.3 of the merged model's peak
  there; 71 are deeper than 500 MJy/sr) and 845 whole-star leftovers (a residual
  peak above 50 MJy/sr that holds more than 0.7 of the data peak; 133 are above
  500 MJy/sr).  Each count is a replay of `build_mergedcat_residuals` on the
  production m7 catalog, so a difference is the switch's own effect.
  - `MERGE_SATROW_DATA_SUPPORT_MIN` (0-1).  A `replaced_saturated` row that a
    frame's own satstar model does not cover is rendered at the merged flux,
    the flux of the frames whose satstar pass fit it.  A frame whose data hold
    less than this fraction of that flux at the row (`_satrow_data_flux`: the
    least-squares amplitude of the unit-flux PSF over the central 5x5 pixels,
    after the median of the 6-10 pixel annulus; a core with a non-finite pixel
    cannot be tested and keeps the merged flux) renders the row at the data
    flux, floored at 0.  Craters deeper than 500 MJy/sr: 71 -> 67 at 0.3 and
    65 at 0.6.
  - `MERGE_ROW_RESID_SUPPORT_MIN` (0-1) with `MERGE_ROW_RESID_NMATCH_MAX`
    (default 99).  Per frame, the unit PSF of every uncovered
    `replaced_saturated` row and of every ordinary row matched in at most
    `NMATCH_MAX` frame catalogs, with a rendered peak of at least 25 MJy/sr, is
    fit to that frame's merged-catalog residual (`_resid_supported_shrink`:
    central 5x5 pixels, 6-10 pixel annulus background, at least 12 finite core
    and 20 finite annulus pixels).  A row whose fitted amplitude is below
    -(1 - MIN) x its flux is re-rendered at flux + amplitude, floored at 0,
    largest first, with the residual updated after each row.  At 0.3 the craters
    go 489 -> 147 and those deeper than 500 MJy/sr 71 -> 32, with no crater
    where production had none; `NMATCH_MAX=1` (single-detection rows only)
    gives 489 -> 209 and 71 -> 39.  The whole-star leftovers above 500 MJy/sr
    go 133 -> 134.
  - `SINGLE_EXPOSURE_EVENT_MASK=1` (`single_exposure_events`).  Sets DO_NOT_USE
    on the pixels of a compact SATURATED core that one frame carries and the
    other covering frames do not, with its wings, in the data mosaic and in each
    per-frame residual's DQ before resampling (94 events in 38 of 48 frames,
    10575 pixels).  With the shrink above, the leftovers above 500 MJy/sr go
    133 -> 58; the 17 "vetted but unsubtracted" and 9 "vetted out" ones are
    gone.  Measured against the replay with `MERGE_SATROW_DATA_SUPPORT_MIN=0.3`
    alone, the mask removes 8 craters and adds 4 (248, 154, 101 and 54 MJy/sr
    deep, each 0.77-0.97 of the model peak); with the shrink above it adds none
    and removes 8 as well.
  - `NEGATIVE_PIXEL_MASK_NSIGMA` (a number >= 0; blank or 0 is off).  Sets
    DO_NOT_USE on pixels with no SATURATED or DO_NOT_USE flag and SCI below
    -N x ERR, in the same two places.  At 50 it masks 10 pixels in 9 of 48
    frames; the deepest is a detector pixel (nrca4, x=1140, y=1644) with SCI
    -19079 to -19095 MJy/sr, 166.5 ERR below zero and DQ = JUMP_DET only, in
    each of the six exposures of its pointing.  With the shrink and the event
    mask in place, craters deeper than 500 MJy/sr go 30 -> 24 (7 craters, 259 to
    2723 MJy/sr deep, are gone and none appear).
  - The last three together (`SINGLE_EXPOSURE_EVENT_MASK=1
    NEGATIVE_PIXEL_MASK_NSIGMA=50 MERGE_ROW_RESID_SUPPORT_MIN=0.3`): craters
    489 -> 132 (deeper than 200 MJy/sr 166 -> 52, than 500 71 -> 24, than 1000
    37 -> 8); whole-star leftovers 845 -> 771 (above 200 MJy/sr 178 -> 103,
    above 500 133 -> 58, above 1000 116 -> 44).  Of the 24 craters left above
    500 MJy/sr, 8 lie within 31 mas of a satstar row fit in one frame (the
    crater is in that frame's own satstar model, which the shrink does not
    reach) and 16 have no catalog row within 40 mas.
  - Shrink threshold: `MERGE_ROW_RESID_SUPPORT_MIN` 0.6 and 0.8 instead of 0.3, with
    the event mask, the negative-pixel mask and `MERGE_SATROW_DATA_SUPPORT_MIN=0.3`
    unchanged (o105 F212N replays).  Craters deeper than 50 MJy/sr 132 (0.3), 121
    (0.6), 120 (0.8); deeper than 500 MJy/sr 24, 22, 21; whole-star leftovers
    above 500 MJy/sr 58, 59, 59.  A higher threshold removes more craters and
    changes the leftovers by one star.  A row lowered although it was right would
    show up as a leftover; the count above 500 MJy/sr goes from 58 to 59 between
    0.3 and 0.8, and the replay cannot say whether that star is one of them.
  - `SATSTAR_SECOND_PASS_NEIGHBOUR_FRAC_MIN` / `_MAX` (merge time,
    `replace_saturated`; off unless `_MIN` is above 0, `_MAX` defaults to 0.8).
    The mutual-nearest second pass pairs a satstar with the nearest daophot row
    out to min(0.5", 1.5 FWHM), 0.105" in F212N, and overwrites that row's flux and
    position with the satstar's.  A satstar whose own daophot row was clipped pairs
    with it at 0.5-3.6 mag below the satstar's flux.  A neighbouring star
    pairs at a comparable flux and is erased: the leftover (100-3000 MJy/sr) then
    sits beside the satstar.  With the guard on, a pairing whose row carries `_MIN`
    <= row flux / satstar flux < `_MAX` of the satstar's flux is not made; the row
    stays and the satstar is appended as its own row.  In o105 F212N 327
    `replaced_saturated` rows were paired at a pre-overwrite separation of at least
    0.05".  The overwritten rows were reconstructed from the per-frame m7 tables
    (rows within 0.16" of the satstar in each frame, shifted by the frame's median
    offset to the unreplaced merged rows; the candidate whose distance to the
    satstar matches `satstar_match_sep` to 0.02" and that appears in the most frames
    within 0.03" is the overwritten row).  59 of them carry 0.2-0.8 of the satstar's
    flux, and for those the data light centroid lies at the flux-weighted blend of
    the satstar and the row, 0.04-0.2 pixel from it: they are real neighbours.
    Replay: the reconstructed rows with 0.2 <= flux ratio < 0.8 appended to the
    production vetted catalog on top of the stack above.  Leftovers above 500 MJy/sr
    58 -> 27 (above 1000: 44 -> 19; above 2000: 25 -> 15) at shrink 0.3, with
    craters deeper than 50 MJy/sr 132 -> 140 and deeper than 500 MJy/sr 24 -> 27; at
    shrink 0.8, 59 -> 28 leftovers above 500 MJy/sr, craters deeper than 50 MJy/sr
    120 -> 127, deeper than 500 MJy/sr 21 -> 23.  Appending every reconstructed row
    whatever its flux ratio gives 26 leftovers above 500 MJy/sr and 149 / 27 craters
    (shrink 0.3), so the window drops no leftover the unrestricted set clears and
    adds 9 fewer craters deeper than 50 MJy/sr.  Caveat: the replay appends rows to
    the vetted catalog; `replace_saturated` and the dedup and vetting after it
    have not been rerun with the guard on, so a restored row that vetting would
    remove is counted as kept here.
  - What the guards leave (o105 F212N, stack + neighbour guard + shrink 0.8: 28
    leftovers above 500 MJy/sr, 23 craters deeper than 500 MJy/sr).  The craters
    classify (a row within 40 mas counts as at the crater) as 16
    with no catalog row there (5 lie 117-136 mas from a satstar row, 11 lie
    0.46-1.07" from one; the satstars are 1.3e7-3.8e7 in flux), 5 in the satstar
    model of a single frame (the satstar fit of one exposure, which the shrink does
    not reach) and 2 at a satstar row in 4-5 frames.  The montages of the 16 show
    ring-shaped minima around the brightest satstars, which is what a mismatch
    between the PSF model and the measured PSF structure would leave; no catalog
    change reaches them and that reading is not tested further here.  The brightest
    leftovers (80606, 35239 and 11796 MJy/sr) are stars that the satstar finder
    joined to a brighter satstar.  Example:
    the star at (266.646137, -28.66411) in nrca1 exposure 4 has its own raw DQ
    SATURATED component (45 pixels, every pixel finite, maximum 91717 MJy/sr) 14.6
    pixels (6.6 FWHM) from the accepted satstar, whose component holds 231
    pixels.  The finder's mask for the satstar is its DQ component plus the
    unflagged pixels above the severity floor (dilated by 2 pixels), which are
    joined with the neighbour's: one component (`sat_area` 1199), one satstar row,
    and the daophot fit has no row there because the hand-off of SATURATED
    components without an accepted satstar (`DAOPHOT_HANDOFF_UNACCEPTED_SAT`) is
    off for F212N in the production m7 run (`DAOPHOT_HANDOFF_FILTERS='F480M'`, the
    per-frame catalog header says `HNDOFFSA = False`).
  - Other tiles and filters (same replay, `SINGLE_EXPOSURE_EVENT_MASK=1
    NEGATIVE_PIXEL_MASK_NSIGMA=50 MERGE_ROW_RESID_SUPPORT_MIN=0.3`, production
    vetted catalog).  Whole-star leftovers above 500 MJy/sr / craters deeper than
    500 MJy/sr, production -> replay: o045 F212N 140 -> 75 / 84 -> 20; o118 F212N
    72 -> 17 / 33 -> 3; o130 F212N 90 -> 16 / 28 -> 5; o138 F212N 89 -> 32 /
    52 -> 21; o105 F480M 111 -> 110 / 16 -> 13.  The F480M leftovers are mostly the
    rings below, which the switches do not touch.  With the footprint mask on top
    (`KFAC` 1, `MARGIN` 2, the production consolidated satstar catalog of each tile,
    applied by an emulation script to scratch copies of the replays; the module was
    not run on these tiles) the same two counts, replay -> replay with the mask, are
    o045 F212N 75 -> 19 / 20 -> 4; o118 F212N 17 -> 8 / 3 -> 2; o130 F212N 16 -> 9 /
    5 -> 0; o138 F212N 32 -> 17 / 21 -> 3; o045 F480M 120 -> 19 / 6 -> 0; o118 F480M
    47 -> 10 / 4 -> 0; o130 F480M 19 -> 13 / 1 -> 0; o138 F480M 43 -> 12 / 2 -> 0;
    o105 F480M 110 -> 9 / 13 -> 0.  The largest leftovers on o045 F212N ((266, 1113) at
    48094 MJy/sr, (9317, 433) at 30406, (6296, 2636) at 10826) coincide with satstar
    candidates that `fit_quality_gate` rejected (qfit 13-60).  For the first, in frame
    `jw10678045001_02101_00004_nrca1`, the seed lies 5.5 pixels off the core of a
    component of 143 pixels with about 40 pixels of saturated DQ in view (class D: a
    union of components), and the fit collapses (flux 768076 -> 37910, implied peak
    4106 against an observed 52625 MJy/sr).  In a 5-frame test of o045 F212N, 69 of
    the 506 gate-rejected candidates with `flux_fit >= 1e4` have a
    `--deblend-satstars` row within 8 pixels.
  - Rows that the pipeline drops, found by replay (o105 F212N, on top of the stack
    and neighbour guard above).  (1) `--deblend-satstars` (off in production; needs
    each frame's `_ramp.fits` and `_cal.fits`) run on all 48 F212N frames adds 1104
    satstar rows and loses 209; consolidated, 227 new and 103 lost.  Appending the
    consolidated rows seen in at least two frames (`n_frames_fit >= 2`) takes the
    leftovers above 500 / 1000 / 5000 MJy/sr from 28 / 20 / 5 to 24 / 16 / 2 and
    leaves the craters unchanged (23 deeper than 500); including single-frame rows
    adds one crater deeper than 2000; also removing the 73 base rows that match the
    lost rows raises the leftovers above 500 to 44.  (2) The in-field dedup (`cofit`)
    leaves 55 consolidated satstar rows with no row in the merged catalog.  The
    "never co-fit" test cannot reject a star that was satstar-fit in one frame
    only.  19 of the 55 are real stars: peak or sub-floor seeds, area 9-32 pixels,
    `n_frames_fit` 1-2, with daophot clusters in 1-5 other frames at 0.80-1.06 of the
    satstar flux and, in 18 of them, within 5 mas of the satstar position, and a
    satstar neighbour 0.17-0.9" away.  About 36 are transient-like `dqsat` seeds (no
    daophot cluster, or only neighbours at 0.01-0.03 of the flux) that the dedup
    rightly drops.  Appending the 18 daophot-supported ones (flux ratio 0.75-1.25,
    position within 10 mas, from the per-frame m7 daophot tables) to the deblend stack gives
    17 / 9 / 2 leftovers above 500 / 1000 / 5000 MJy/sr and no new crater (23
    deeper than 500); `SATSTAR_INFIELD_DEDUP=cofit_residual` on the same stack
    gives 15 / 8 / 2 with 27 craters deeper than 500; the extra craters are restored
    single-frame fits that over-subtract by 25-40 % or lie about 0.2" off.  The
    replay needs no minimum on the number of daophot frames (four of the 18 rows
    have a daophot detection in one frame, at 0.86-0.99 of the satstar flux) and
    `SATSTAR_DAO_RESTORE_FRAC_MIN` (above) is the opt-in rule with those criteria.
    Its default separation limit of 0.03" adds one row to the replay's 18 (idx 8131,
    21.4 mas, 1.058 of the satstar flux); appending those 19 rows gives 727 / 59 /
    16 / 9 / 7 / 2 leftovers above 50 / 200 / 500 / 1000 / 2000 / 5000 MJy/sr and the
    craters of the 18-row stack (127 / 53 / 23 / 7 / 6 deeper than 50 / 200 / 500 /
    1000 / 2000).  The rows in these replays were appended to the vetted catalog;
    they did not pass through `replace_saturated`, the dedup and the vetting.
  - Deblend rows on F480M (o105, the same replay on all 12 frames; the event mask,
    negative-pixel mask and shrink 0.3 are on).  The consolidated satstar catalog goes
    from 1703 rows (production) to 1559 (deblend): 77 new, 221 lost.  The lost rows are
    the poorer fits (median qfit 5.5; n_frames_fit 1 / 2 / 3 or more = 105 / 60 / 56); the
    new rows have median qfit 2.9.  Appending the 36 new rows with
    `n_frames_fit >= 2` to the production vetted catalog takes the leftovers above 50 / 200 /
    500 / 1000 / 2000 / 5000 MJy/sr from 274 / 165 / 110 / 84 / 24 / 2 to 264 / 156 / 101 /
    77 / 22 / 0 and the craters deeper than 50 / 200 / 500 / 1000 / 2000 from
    30 / 17 / 13 / 7 / 1 to 34 / 20 / 15 / 8 / 2.  Appending all 77 gives 262 / 154 / 99 / 75 /
    22 / 0 and 36 / 22 / 17 / 10 / 3.  Also dropping the 213 base rows that match the lost
    rows gives 311 / 202 / 147 / 99 / 22 / 0 and 31 / 19 / 14 / 8 / 2.  Of the 165 F480M
    leftovers above 200 MJy/sr on the shrink-0.3 stack, 36 remain with the footprint mask
    below, so most of them lie inside the mask footprint; the deblend rows change 165 to 156.
  - `MERGEDCAT_FOOTPRINT_MASK_KFAC` with `MERGEDCAT_FOOTPRINT_MASK_MARGIN_PX`
    (default 2) and `MERGEDCAT_FOOTPRINT_MASK_PASSES` (default `m7`;
    `residual_footprint_mask`; off unless `KFAC` is above 0).  After
    the residual is resampled, the SCI plane of the residual i2d is set to NaN in
    a disk of radius `KFAC` x sqrt(`sat_area` / pi) native detector pixels +
    `MARGIN` mosaic pixels around the fitted position of every row of the
    consolidated satstar catalog that the merge read (the native pixel is 0.031"
    in the NIRCam short-wave filters, 0.063" in the long-wave ones and 0.11" in
    MIRI).  A row without a positive `sat_area` has no footprint.  The SCI header
    records `FPMASKK`, `FPMASKM` and `FPMASKN` (rows masked).  The mask is skipped
    with a warning, and the image kept, when the consolidated catalog does not
    exist or lacks `skycoord_fit` or `sat_area`, or when the merge read the NIRCam
    `merged_i2d` primary catalog, which has no consolidated cache.  The model i2d,
    the data i2d and the catalogs are as before.  The next phase of a manual run
    takes its i2d seeds and its smoothed background map from this residual, so a
    mask in an intermediate pass would change those inputs, and the counts below
    are for the final residual only.  The mask is therefore applied to the passes
    that `MERGEDCAT_FOOTPRINT_MASK_PASSES` lists (`m2` ... `m7`, comma- or
    space-separated; unset or blank gives `m7`, as for `GC_RESID_NAN_SATSTAR_PASSES`)
    and a run with the switch on prints which pass it left alone; an unknown label
    raises before any residual is rendered.  `GC_RESID_NAN_SATSTAR_FOOTPRINT`
    (section 7) blanks the DQ-SATURATED pixels of accepted satstars in the per-frame
    residuals of a manual pass; this switch acts on the coadd, on a disk with a margin
    around every consolidated row, and needs no per-frame rerun.  The two have not
    been compared.  The counts below were measured with an
    emulation script before the switch existed.  The switch, applied to the same
    scratch copies of the residual i2d (o105 F212N with the deblend rows and the
    18 supported rows; o105 F480M with the shrink-0.3 stack, without and with the
    deblend rows) at `KFAC` 1 and `MARGIN` 2, sets the same pixels to NaN
    (3,227,541 NaN pixels of the F212N image and 667,561 of the F480M one on both
    sides, none in only one) and leaves every other pixel unchanged (maximum
    difference 0), so the counts of those three stacks are those of the switch;
    the other counts came from the same emulation script.  Counts for o105 F212N
    (leftovers above 50 / 200 / 500 / 1000 / 2000 / 5000 MJy/sr | craters deeper
    than 50 / 200 / 500 / 1000 / 2000), without the mask and then with it at `KFAC`
    1 and `MARGIN` 2:
    - production 845 / 178 / 133 / 116 / 66 / 18 | 489 / 166 / 71 / 37 / 18; it was
      not replayed with the mask.
    - the best stack above (event mask, negative-pixel mask 50, shrink 0.8, vetted
      catalog with the reconstructed rows) 740 / 72 / 28 / 20 / 16 / 5 |
      127 / 53 / 23 / 7 / 6; with the mask 752 / 51 / 12 / 9 / 2 / 1 |
      89 / 35 / 5 / 0 / 0.
    - the best stack with the deblend rows and the 18 supported rows
      728 / 60 / 17 / 9 / 7 / 2 | 127 / 53 / 23 / 7 / 6; with the mask
      741 / 46 / 7 / 4 / 2 / 1 | 89 / 35 / 5 / 0 / 0.

    o105 F212N's count above 50 MJy/sr rises with the mask (740 to 752, 728 to 741);
    the cause was not examined.  o105 F480M:
    - production 303 / 168 / 111 / 84 / 24 / 2 | 69 / 28 / 16 / 7 / 1; with the mask
      174 / 39 / 10 / 4 / 1 / 1 | 31 / 4 / 2 / 0 / 0.
    - event mask, negative-pixel mask and shrink 0.3 274 / 165 / 110 / 84 / 24 / 2 |
      30 / 17 / 13 / 7 / 1; with the mask 145 / 36 / 9 / 4 / 1 / 1 | 3 / 0 / 0 / 0 / 0.
    - the same with the 36 deblend rows 264 / 156 / 101 / 77 / 22 / 0 |
      34 / 20 / 15 / 8 / 2; with the mask 141 / 33 / 6 / 1 / 0 / 0 |
      4 / 0 / 0 / 0 / 0.

    The mask sets 1.66 % (F212N) and 2.19 % (F480M) of the finite pixels of the
    residual i2d to NaN, in disks of median radius 4.7 (F212N) and 6.3 (F480M)
    mosaic pixels and maximum radius 33.9 and 33.3.  The counts of a masked image
    leave out whatever lies inside the disks: a leftover or a crater there is not
    measured.  The mask hides the model mismatch inside the footprint; it does not
    make the model fit.
  - Ring-zone extension (`MERGEDCAT_FOOTPRINT_MASK_RING_FLUX_MIN` with
    `MERGEDCAT_FOOTPRINT_MASK_RING_RADIUS_PX`; `residual_footprint_mask`).  Off unless
    `KFAC` and `RING_FLUX_MIN` are above 0; neither ring variable is read otherwise.  A
    row of the consolidated satstar catalog whose `flux_fit` is at least `RING_FLUX_MIN`
    gets a disk of at least `RING_RADIUS_PX` mosaic pixels (the larger of that and its
    footprint radius above); every other row keeps its footprint disk.  A row without a
    positive `sat_area` has no disk with or without the extension, and a NaN `flux_fit`
    is not widened.  A consolidated cache without a `flux_fit` column skips the mask
    with a warning and keeps the image.  `RING_FLUX_MIN` without a `RING_RADIUS_PX`
    above 0, and a bad value of either, raise before any residual is rendered.  The SCI
    header adds `FPRINGF` (flux), `FPRINGR` (radius in mosaic pixels) and `FPRINGN`
    (rows at or above the flux).  The reason: 26 of the 39 craters deeper than 200
    MJy/sr near a satstar (above) lie 12-20 pixels from it, outside the footprint disks
    of median radius 4.7 pixels.  Emulation on scratch copies of the masked replays
    (`RING_FLUX_MIN` 1e6; `RING_RADIUS_PX` 25 for F212N and 20 for F480M).  Craters
    deeper than 50 / 200 / 500 MJy/sr and masked fraction of the finite pixels, mask ->
    mask + ring:
    - o045 F212N 139 / 57 / 4 -> 26 / 1 / 0, 1.90 -> 2.91 %.
    - o118 F212N 75 / 26 / 2 -> 45 / 9 / 1, 0.57 -> 0.86 %.
    - o130 F212N 55 / 9 / 0 -> 32 / 3 / 0, 0.47 -> 0.68 %.
    - o138 F212N 92 / 27 / 3 -> 35 / 8 / 2, 0.86 -> 1.43 %.
    - o105 F212N (deblend rows + 18 supported rows, mask 89 / 35 / 5): 43 / 13 / 3 at
      20 pixels (+0.4 percentage points masked), 39 / 10 / 1 at 25 pixels (+0.7) and
      38 / 8 / 1 at 34 pixels.
    - o045, o118, o130, o138 and o105 F480M at 20 pixels: the mask leaves 0-3 craters
      deeper than 50 MJy/sr, and the ring changes no crater count.
    The leftovers above 500 MJy/sr change by at most two per image (o045 F212N 19 -> 18,
    o138 F212N 17 -> 16, o045 F480M 19 -> 17, the others unchanged; o105 F212N 7 -> 8
    at 20 pixels and 7 at 25 and 34).  The ring disk hides what is inside it: a real star
    within 25 pixels (0.78") of a satstar with `flux_fit` of 1e6 and more is blanked
    with it.  The module, applied to the unmasked scratch copy of the o105 F212N replay
    (deblend rows + 18 supported rows) at 1e6 and 25 pixels, widens 239 of the 10356
    rows, sets 1245111 more pixels to NaN (2.40 % of the 51825305 finite pixels) and gives
    the 3611086 NaN pixels of the emulation, with every finite pixel identical (maximum
    difference 0).  No full m7 + finalize run with the extension exists, and its radius
    has no default.
  - Gate-rejected fits (`MERGEDCAT_FOOTPRINT_MASK_REJECTED`; `residual_footprint_mask`).
    A bright saturated star whose fit the satstar gates reject has no row in the
    consolidated catalog and no model, so its residual is the whole star, and the disks
    above do not cover it.  The extension adds the disks of those fits.  Off unless
    `KFAC` is above 0 and the variable is `1/true/yes/on`; it is not read otherwise
    (`0/false/no/off`, unset and blank are off; any other value raises before any
    residual is rendered).  The rows come from the `*_satstar_rejected.fits` files that
    the merge reads with `load_rejected_satstar_catalog`: this observation's files
    (observation scope), one phase per exposure (the pass being built, as for the
    consolidated catalog), and the frame variants of the frames being merged.  A file
    contributes the rows whose `sidelobe_maskable` is true.  That column is written by
    the satstar stage when `SATSTAR_SIDELOBE_MASK_FRAC_MIN` is set: a fit
    that only the sidelobe cut rejects, and whose annulus fraction the mask fraction
    keeps.  A file written without it (the switch was off, or an older run) contributes
    no row.  If no file has the column the extension is skipped with a warning and the
    consolidated rows are still masked, so that a run which turned the extension on is
    not left with nothing masked in silence; the same holds when the merge read the
    NIRCam `merged_i2d` primary catalog (the rejected file beside it is not read), when
    there are no rejected files, and, with the ring extension on, when the rejected rows
    have no `flux_fit`.  The disk of a row is the footprint radius above, with the
    ring-zone extension applied to the union of the consolidated and the rejected rows.
    Unlike the accepted channel, `load_rejected_satstar_catalog` serves `skycoord_fit` as
    stored; the mask reads it with `refresh_sky=True`, which re-projects each file's
    pixel centroids through its frame's current GWCS as the accepted channel does on
    every read (a stored position can be tens of mas, up to about three F212N mosaic
    pixels, off after an offsets-table correction; issue #193).  The SCI header adds
    `FPREJN` (rejected rows masked) when the extension is on.  The passes the mask skips
    do not read the rejected files.  The default of the loader is unchanged.  The model
    i2d, the data i2d and the catalogs are as before: a rejected row has no model row
    and no catalog entry.
  - DQ_SATURATED footprint (`MERGEDCAT_FOOTPRINT_MASK_DQSAT_PX` with
    `MERGEDCAT_FOOTPRINT_MASK_DQSAT_MIN_PX`; `residual_footprint_mask`).  The disks above hang
    on satstar rows, so a saturated star that has no row (its fit failed the gates and the
    sidelobe mask fraction did not keep it), and a second saturated star inside the blob that
    one row fitted, have none.  This extension blanks the saturated pixels of the frames
    themselves.  Off unless `KFAC` and `DQSAT_PX` are above 0 (a blank or 0 `DQSAT_PX` is
    off; neither variable is read while `KFAC` is off).  `DQSAT_MIN_PX` (a whole number
    of detector pixels >= 1, default 4) is read only with `DQSAT_PX` on.  A bad value of
    either raises before any residual is rendered.  After the residual is resampled and
    the disks are applied, every per-frame residual of the pass (`*_mergedcat_residual.fits`,
    the files the resample reads; each keeps the DQ of its crf) contributes its pixels with
    the SATURATED bit (2) set that belong to a connected component (8-connectivity) of at
    least `DQSAT_MIN_PX` pixels.  The pixels go through the frame GWCS (`frame_wcs`) to the
    pixel grid of the residual i2d and are rounded to the nearest mosaic pixel; the union over
    the frames is dilated by a disk of radius `DQSAT_PX` mosaic pixels, and the residual i2d
    is NaN inside it.  A star that saturates in only some of the dithers is blanked
    wherever one of them saturates; a mask applied to each frame before the resample
    would leave the flux of the exposures that did not saturate.  The SCI header adds
    `FPDQSATD` (radius in mosaic pixels), `FPDQSATM` (smallest component) and `FPDQSATN`
    (frames).  The step does not read the consolidated satstar catalog, so it runs when the
    catalog is missing and the disks are skipped.  A frame without a `DQ` extension, or whose
    GWCS cannot be read, skips this step with a warning and keeps the image with its disks.
    The passes are those of the disks (`MERGEDCAT_FOOTPRINT_MASK_PASSES`).  The model i2d,
    the data i2d and the catalogs are as before.  Emulation on scratch copies of the residual
    i2d of replays that carry the disks and the ring zone (`KFAC` 1, `MARGIN_PX` 2,
    `RING_FLUX_MIN` 1e6; `RING_RADIUS_PX` 25 for F212N in the replay `mkrswapdblfs3` and 20
    for F480M in `mkrsdbl`), at `DQSAT_PX` 4 for F212N (mosaic pixel 0.031") and 3 for F480M
    (0.063"), `DQSAT_MIN_PX` 4.  The emulation read the DQ of the production crf files; the
    per-frame residual of o138
    F212N nrca1 exposure 1 has the same 3989 DQ_SATURATED pixels at the same positions as its
    crf.  Leftovers above 50 / 200 / 500 / 1000 / 2000 / 5000 MJy/sr | craters deeper than
    50 / 200 / 500 / 1000 / 2000 MJy/sr, with the disks and then with the DQ_SAT mask added,
    and the percentage of the finite pixels of the replay's residual i2d that it sets to NaN:
    - o045 F212N 932 / 43 / 8 / 2 / 1 / 1 | 27 / 1 / 0 / 0 / 0 -> 925 / 42 / 7 / 1 / 0 / 0 |
      18 / 0 / 0 / 0 / 0; 0.327 %.
    - o118 F212N 124 / 12 / 7 / 4 / 2 / 1 | 45 / 9 / 1 / 0 / 0 -> 120 / 10 / 5 / 2 / 1 / 0 |
      25 / 6 / 1 / 0 / 0; 0.313 %.
    - o130 F212N 94 / 21 / 9 / 7 / 6 / 3 | 32 / 3 / 0 / 0 / 0 -> 91 / 19 / 6 / 4 / 3 / 0 |
      8 / 0 / 0 / 0 / 0; 0.325 %.
    - o138 F212N 373 / 11 / 2 / 2 / 2 / 2 | 37 / 8 / 2 / 2 / 0 -> 369 / 9 / 0 / 0 / 0 / 0 |
      24 / 5 / 1 / 1 / 0; 0.312 %.
    - o045 F480M 154 / 36 / 15 / 9 / 5 / 1 | 1 / 0 / 0 / 0 / 0 -> 129 / 17 / 1 / 0 / 0 / 0 |
      1 / 0 / 0 / 0 / 0; 2.026 %.
    - o118 F480M 143 / 29 / 9 / 7 / 3 / 0 | 0 / 0 / 0 / 0 / 0 -> 130 / 23 / 3 / 0 / 0 / 0 |
      0 / 0 / 0 / 0 / 0; 1.266 %.
    - o130 F480M 137 / 32 / 8 / 4 / 2 / 0 | 1 / 0 / 0 / 0 / 0 -> 126 / 24 / 4 / 0 / 0 / 0 |
      0 / 0 / 0 / 0 / 0; 1.081 %.
    - o138 F480M 116 / 19 / 4 / 0 / 0 / 0 | 0 / 0 / 0 / 0 / 0 -> 115 / 18 / 3 / 0 / 0 / 0 |
      0 / 0 / 0 / 0 / 0; 1.126 %.
    - o105 F212N 740 / 46 / 7 / 4 / 2 / 1 | 39 / 10 / 1 / 0 / 0 -> 726 / 45 / 6 / 3 / 1 / 0 |
      28 / 9 / 1 / 0 / 0; 0.331 %.
    - o105 F480M 145 / 36 / 9 / 4 / 1 / 1 | 3 / 0 / 0 / 0 / 0 -> 120 / 24 / 3 / 0 / 0 / 0 |
      1 / 0 / 0 / 0 / 0; 2.068 %.
    The two o105 replays are older stacks than those of the other tiles.  o105 F212N carries
    the deblend rows, the DAO restore, the disks and the ring zone (`RING_FLUX_MIN` 1e6,
    `RING_RADIUS_PX` 25); o105 F480M carries the disks and the ring zone (`RING_RADIUS_PX` 20)
    and no deblend rows.  The module was not run on o105.

    The F212N masks depend on the component size and the radius.  At `DQSAT_MIN_PX` 15 the
    F212N leftovers above 500 MJy/sr are 8 / 7 / 7 / 2 for o045 / o118 / o130 / o138, against
    8 / 7 / 9 / 2 without the DQ_SAT mask and 7 / 5 / 6 / 0 at `DQSAT_MIN_PX` 4; the craters
    deeper than 200 MJy/sr are 0 / 6 / 0 / 5 at both sizes.  At `DQSAT_MIN_PX` 15 a dilation
    of 4 mosaic pixels (0.124") sets 0.175 / 0.162 / 0.165 / 0.157 % of the finite pixels to
    NaN and one of 8 pixels 0.579 / 0.436 / 0.428 / 0.455 %; the leftovers above 500 MJy/sr are
    8 / 7 / 7 / 2 at 4 pixels and 8 / 7 / 6 / 2 at 8.  A NaN-component dilation of the data
    i2d (the compact NaN components of 6 to 3000 mosaic pixels, dilated by 4 pixels) was
    also emulated on F480M: it sets 0.007 % of the finite pixels to NaN and leaves
    11 / 5 / 5 / 4 leftovers above 500 MJy/sr for o045 / o118 / o130 / o138, against
    1 / 3 / 4 / 3 for the DQ_SAT mask at 1.1-2.0 %.  On F212N, applied to the DQ_SAT-masked
    residual, the same dilation of 4 pixels sets 1303 / 1386 / 1390 / 1245 more pixels to
    NaN (0.002-0.003 % of the finite pixels) for o045 / o118 / o130 / o138 and leaves every
    count above unchanged; the 41 x 41 pixel boxes of the data i2d around o130 (349, 2907)
    and (4934, 195) hold no NaN pixel.  It is not in the module.
    Module against the emulation.  `mask_residual_dqsat`, run on scratch copies of the
    residual i2d of the eight tile-bands o045 / o118 / o130 / o138 x F212N / F480M with the
    per-frame residual files of the same replays (F212N `swapdblfs3`, 48 frames; F480M
    `sdbl`, 12 frames; `DQSAT_PX` and `DQSAT_MIN_PX` as above), gives the NaN pixels of the
    emulation exactly: in each of the eight no pixel differs in NaN state and every finite
    pixel is identical.  Saturated components of at least 4 pixels / saturated detector
    pixels in them / flagged mosaic pixels / pixels set to NaN (percentage of the finite
    pixels of the copy):
    - o045 F212N 5193 / 193245 / 83819 / 164621 (0.327 %).
    - o118 F212N 2738 / 79778 / 44676 / 160957 (0.313 %).
    - o130 F212N 2556 / 65068 / 39934 / 167418 (0.325 %).
    - o138 F212N 3558 / 115890 / 57455 / 159492 (0.312 %).
    - o045 F480M 10089 / 722880 / 232716 / 246159 (2.026 %).
    - o118 F480M 4927 / 322422 / 107186 / 156425 (1.266 %).
    - o130 F480M 4109 / 225779 / 78851 / 133880 (1.081 %).
    - o138 F480M 4382 / 271664 / 93746 / 139252 (1.126 %).
    End to end.  The F480M residual i2d of o045 / o118 / o130 / o138 was rendered twice
    with `build_mergedcat_residuals` (12 frames each; `KFAC` 1, `MARGIN_PX` 2), without and
    with `DQSAT_PX` 3.  The scratch render root has no consolidated satstar catalog, so the
    disks are skipped and the render without the setting carries no mask.  In each tile the
    render with the setting has the NaN pixels of the emulation applied to the render
    without it (0 pixels in only one of them; 932593 / 658388 / 602473 / 627223 NaN pixels),
    keeps every NaN pixel of the render without it, and has the same value in every pixel
    that is finite in both (0 differ).  The step reports the component, saturated pixel and
    flagged pixel counts of the module run above, the header has `FPDQSATD` 3, `FPDQSATM` 4
    and `FPDQSATN` 12, and the render without the setting has none of the three cards.  The
    mask sets 4.325 / 2.164 / 1.732 / 1.925 % of the finite pixels of the unmasked render to
    NaN, against 2.026 / 1.266 / 1.081 / 1.126 % on the replays that already have the disks
    and the ring zone.  The same pair for o138 F212N (48 frames, `DQSAT_PX` 4, `FPDQSATN` 48)
    gives the same result: 2631358 NaN pixels in the render with the setting and in the
    emulation applied to the render without it (0 pixels in only one of them; 2363634 NaN
    pixels without the setting), every NaN pixel of the render without it kept, 0 of 51559922
    pixels that are finite in both differ, and 0.517 % of the finite pixels of the unmasked
    render set to NaN (0.312 % on the replay that already has the disks and the ring zone).
    No full m7 + finalize run with the disks, the ring zone and the extension on the real
    consolidated satstar catalog exists.
    What the DQ_SAT mask leaves on F212N (`DQSAT_PX` 4, `DQSAT_MIN_PX` 4).  The craters deeper
    than 200 MJy/sr (o118 6, o138 5) belong to stars whose nearest vetted row is not a satstar
    row: model peaks of 423-3247 MJy/sr, depths of 30-56 % of the model peak, 0.53-1.46" from the
    nearest satstar row.  The footprint does not reach them.  The leftovers above 1000 MJy/sr
    (o045 1, o118 2, o130 4, o138 0), by the class that the leftover list assigns:
    - Not in the m7 catalog, 2.3-2.4" from the nearest satstar row: o130 (349, 2907) at 3655
      and (4934, 195) at 2515 MJy/sr.  Two frames hold (349, 2907) (nrca2 exposures 4 and 5).
      In the disk of radius 6 detector pixels (113 pixels) around it the peaks are 4655 and
      4592 MJy/sr, there is no DQ_SATURATED pixel and there is 1 NaN pixel in each frame; the
      per-frame m7 residuals keep both peaks.  Two frames hold (4934, 195) (nrca3 exposures 2
      and 3).  The disk around it peaks at 6284 MJy/sr in exposure 3, with 53 JUMP_DET pixels
      and no NaN or DQ_SATURATED pixel, and has a maximum of 9.1 MJy/sr in exposure 2.
    - Vetted row without a subtraction, 2.8-3.0" from the nearest satstar row: o118 (5457, 2808)
      at 3201 and o130 (5065, 68) at 1244 MJy/sr.  Two frames hold (5457, 2808) (nrca4
      exposures 1 and 2).  The disk around it peaks at 1.078e4 MJy/sr in exposure 1, with 14
      JUMP_DET pixels and no NaN or DQ_SATURATED pixel, and has a maximum of 9.5 MJy/sr in
      exposure 2.
    - Within 1" of a satstar row: o045 (1253, 131) at 1039, o118 (5014, 2112) at 1231 and o130
      (3091, 4576) at 2099 MJy/sr, at 0.90", 0.93" and 0.24".
  - What is left with the mask (o105 F212N, deblend rows + 18 supported rows).  Seven
    leftovers above 500 MJy/sr.  Three are within 0.22-0.30" of a satstar row and
    within 20 pixels of the left mosaic edge (x = 12-18, y = 1208-1225).  Two have no
    catalog row, sit at y = 9 (x = 452 and 461), and have no covering frame in the
    per-frame probe.  One (11796 MJy/sr) is a star saturated in the one exposure
    that covers it, whose satstar candidate the fit-quality gate rejected (qfit 7.39,
    flux 12572).  One (3896 MJy/sr) is an unsaturated star covered by one exposure
    with no daophot row.  Of the 58 leftovers above 200 MJy/sr in the cofit_residual
    + deblend replay, 7 lie within 0.25" of a gate-rejected satstar candidate in some
    frame (six rejected by `fit_quality_gate`, one by `implied_peak_gate`).
  - The craters that the mask leaves (same stack): 5 deeper than 500 MJy/sr and 35
    deeper than 200, 34 of them with no catalog row within 40 mas.  The nearest
    satstars of the five are 0.46-1.07" away, with `flux_fit` 4.4e5, 6.0e5, 1.1e7,
    1.2e7 and 1.3e7.  Without the mask, in the best stack 39 craters deeper than
    200 MJy/sr lie within 1.5" of a satstar with `flux_fit >= 1e5`; their distance
    to it has quartiles 15.4, 16.0 and 16.3 pixels, 26 are 12-20 pixels away, and
    23 of those 26 lie at position angles 105-135 degrees (mod 180) in the mosaic
    pixel frame.  In the median of residual / `flux_fit` over the 147 isolated
    satstars (no other satstar within 45 pixels) with `flux_fit >= 1e6`, the
    production residual at 14-18 pixels is 0.4 and 1.0 (x 1e-6) in the position-angle
    sectors 0-15 and 105-135 degrees against 5.1, 6.3 and 6.8 in the sectors 15-45,
    75-105 and 135-165 degrees.  A template (the median of residual / `flux_fit`)
    made from half of those stars correlates 0.992 over 6-24 pixels with the other
    half's.  Subtracting it from the other half, scaled by each star's flux, lowers
    the median rms of the residual in the 6-24 pixel annulus from 37 to 24 MJy/sr and
    the number of stars with a maximum above +500 MJy/sr from 37 to 20, and leaves the
    number with a minimum below -500 MJy/sr at 23 -> 25.  A median template therefore
    does not predict the dips.  Over the 177 isolated satstars (no other satstar
    within 37 pixels) with `flux_fit >= 1e6` the median of residual / model in
    1-pixel annuli is -0.059 (0.5 pixel), -0.078 (2.5), -0.085 (3.5), -0.046 (4.5),
    +0.054 (5.5), +0.225 (6.5), +0.143 (7.5) and +0.080 (8.5): the model is 5-9 % too
    bright inside 5 pixels and 5-23 % too faint at 5.5-8.5 pixels (beyond that the
    annulus holds other stars' model).
  - Core pits and a radial profile correction (emulation only, o105 F212N).  The crater
    counts above require a depth above 50 MJy/sr and above 0.3 x the model peak, which
    leaves out the pits in the saturated cores of bright stars.  Over the 1522 isolated
    satstars (no other satstar within 17 pixels) with `flux_fit >= 1e5` the minimum of
    the residual within 4 pixels of the star has a median of -0.020 and a 5th percentile
    of -0.083 times the model peak (-0.038 and -0.120 for the 226 with `flux_fit >= 1e6`);
    194 stars have a minimum below -5000 MJy/sr, 64 below -10000 and 14 below -20000, the
    same for the production residual and for the best stack with the deblend rows.  A
    correction built from the data subtracts rho(r, flux) x (the star's own median model
    profile x `flux_fit`) from the residual out to 14 pixels (tapering to zero at 16),
    where rho is the median over isolated stars (no other satstar within 15 pixels) in six
    `flux_fit` bins (1e5, 2e5, 5e5, 1e6, 2e6, 5e6, above) of the median residual / median
    model in 1-pixel annuli (-0.061, -0.093 and +0.218 at 0.5, 3.5 and 6.5 pixels for the
    1e6-2e6 bin).  Fitting rho on half of the stars (the even rows of the catalog) gives the
    same result as fitting it on all.  The core minima then number 65 / 19 / 3 below -5000 /
    -10000 / -20000 MJy/sr (65 / 21 / 2 for the half fit), the stars with a maximum above
    +500 MJy/sr at 6-16 pixels fall from 46 to 31, and the stars with a minimum below
    -500 MJy/sr there rise from 29 to 43.  The tile counts do not improve: leftovers above
    500 MJy/sr 17 -> 17, craters deeper than 500 MJy/sr 23 -> 26 (27), craters deeper
    than 50 MJy/sr 127 -> 150 (152).  With the footprint mask the same stars have no core
    pixels left and 1 minimum below -500 MJy/sr and 3 maxima above +500 MJy/sr at
    6-16 pixels.  An azimuthally averaged profile correction therefore removes about
    two thirds of the deep pits and a third of the ring excess, and adds ring dips.
  - Caveat for every replay row above: each is the production m7 catalog with rows
    appended or changed; `replace_saturated`, the dedup and the vetting have not been
    rerun, so a restored row that vetting would remove is counted as kept.
- **Products**: `*_satstar_{catalog,model,residual,flags}.fits`. The flags image
  is a uint8 bitmask — bit 1 partly saturated (recoverable), bit 2 totally
  saturated (NaN-variance core), bit 4 included in an accepted satstar fit.

### 6a. Saturated-source astrometry: the sky column is recomputed on every read

A per-exposure `*_satstar_catalog.fits` is a **cache of a FIT**, and the fit is
the pixel centroid (`xcentroid`/`ycentroid`).  Its `skycoord_fit` is that
centroid projected through the frame's WCS **at fit time**, which is a different
thing: the frame's WCS changes underneath the cache whenever the offsets table
is corrected and the working copy is regenerated from `_cal`, and the way the
pipeline reads that WCS changed once (GWCS-first, 2026-07-29).  The cache
survives both — `_satstar_recovery_signature` keys it on the recovery/deblend
config only, deliberately, so a plain re-run does not refit every field.

So a stored sky position is never trusted.  Both readers
(`load_or_make_satstar_catalog` on a cache hit, and the consolidated-catalog
build in `merge_catalogs.load_satstar_catalog`) re-project the stored pixel
centroid through `frame_wcs()` of the frame the catalog sits beside
(`jwst_gc_pipeline.photometry.satstar_wcs_refresh`).  On a field whose frames
have not moved since the fit this is a no-op to <0.01 mas.

The component anchor `sat_com_ra`/`sat_com_dec` is stored as sky only, so there
is no pixel to re-project; it is **transported** by the same tangent-plane offset
the row's own `skycoord_fit` just moved by.  It is the bbox centre of the
component that star was fit in, so the WCS-difference gradient over that
separation is the whole error: 0.001 mas median / 0.003 mas max against the exact
pixel transport (brick F182M nrcb1, 411 anchors, frame displaced 2″ with a 0.05°
roll).  Do **not** recover the anchor pixel by rebuilding the fit-time WCS from
the header cards in the catalog's meta — that is ASTROMETRY RULE #2's forbidden
SIP-header inversion, and a linear-card whitelist that drops `A_ORDER`/`A_i_j`/
`B_ORDER`/`B_i_j` inverts a `RA---TAN-SIP` projection through a distortion-free
TAN: 54.87 mas median / 224.30 mas max (1.79 / 7.26 px) on an UNMOVED frame.

Without the refresh, a June fit publishes June's astrometry through an August
frame: brick F200W's caches read +56.8 / +88.7 mas away from their own pixels'
current sky positions, and the m6 catalog built from them showed a matching
+58.7 / +88.2 mas saturated-versus-unsaturated position excess (issue #193).  A
cache mtime does not detect this — the comparison that does is stored
`skycoord_fit` versus `frame_wcs(frame).pixel_to_world(xcentroid, ycentroid)`.

The **consolidated** per-filter satstar catalog is itself a cache, and it is
invalidated on the same axis: its freshness key carries `SATFRMSG`, a stat-only
digest of the resolved frames' name/mtime/size
(`satstar_wcs_refresh.satstar_frame_state_signature`), alongside the source
count, dedup radius and dedup algorithm.  Those other terms are all properties of
the per-exposure satstar catalogs, which an offsets-table correction plus
regeneration from `_cal` does not touch — so without the frame term the
consolidated catalog silently goes stale again the next time a frame moves.

---

## 7. Flags & environment reference

### CLI options (`crowdsource_catalogs_long.py`)

| flag | dest | default | purpose |
|---|---|---|---|
| `--saturation-data-floor` | `saturation_data_floor` | −1.0 | mask a SATURATED px only if data > floor; −1 = per-filter auto, 0 = mask all, >0 = explicit |
| `--truly-lost-saturation-in-fit` | `truly_lost_saturation_in_fit` | False | manual path only: fit mask, near-sat veto DQ and the satstar saturated-pixel fill use truly-lost saturation (SATURATED & DO_NOT_USE) instead of the any-group SATURATED flag (#567). Recorded per frame as catalog header `TLSATFIT`. The A and B arms write the **same** per-frame product names and `--skip-if-done` markers are judged by mtime only (`_marker_is_current`), so run the B arm in its own basepath and do not combine it with `--skip-if-done` over an A-arm tree (a marker does not record the arm). The legacy crowdsource step, `artificial_stars.py` and `xexp_reconcile` keep the raw flag |
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
| `SATSTAR_DATA_FLOOR_READOUT_FRAC` | 0.5 | NIRCam: lower a per-filter finder floor to FRAC × 50000 × PHOTMJSR / t_last when smaller; 0 keeps the table |
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
| `MIRI_SATSTAR_PHANTOM_FLUX_FLOOR` / `…_SSR_MAX` / `…_RATIO_MAX` | 0 (off) / 50 / 50 | post-fit bright-phantom rejection; the code suggests an F770W-calibrated floor of 1e5, and every launcher in this repo leaves it at 0 |
| `MIRI_DAOPHOT_PROM_ROBUST` | 0 | neighbour-robust prominence on the daophot side |
| `MIRI_EDGE_DETECT_MARGIN` | 8 | detection margin at the detector edge |
| `MIRI_RESID_PIT_NMAD` / `…_DILATE` | 15.0 / 2 | residual-pit masking |
| `MERGEDCAT_FOOTPRINT_MASK_KFAC` / `MERGEDCAT_FOOTPRINT_MASK_MARGIN_PX` / `MERGEDCAT_FOOTPRINT_MASK_PASSES` | off / 2 / `m7` | NaN disk of radius KFAC x sqrt(`sat_area` / pi) native pixels + MARGIN mosaic pixels around every row of the consolidated satstar catalog, in the SCI plane of the mergedcat residual i2d of the listed passes (`m2` ... `m7`); off unless KFAC is above 0; the model i2d, the data i2d and the catalogs do not change; a bad value raises before any residual is rendered (guards section) |
| `MERGEDCAT_FOOTPRINT_MASK_RING_FLUX_MIN` / `MERGEDCAT_FOOTPRINT_MASK_RING_RADIUS_PX` | off / none | with the footprint mask on: a row of the consolidated satstar catalog with `flux_fit` of at least `RING_FLUX_MIN` gets a disk of at least `RING_RADIUS_PX` mosaic pixels (the ring craters lie 12-20 pixels out); off unless `KFAC` and `RING_FLUX_MIN` are above 0, and `RING_RADIUS_PX` has no default and must then be above 0; a bad value raises before any residual is rendered; the SCI header records `FPRINGF`, `FPRINGR`, `FPRINGN` (guards section) |
| `MERGEDCAT_FOOTPRINT_MASK_REJECTED` | off | with the footprint mask on: also NaN the disks (same radius, same ring-zone extension) of the gate-rejected satstar fits whose `sidelobe_maskable` is true in the `*_satstar_rejected.fits` the merge reads (observation scope, phase of the pass, frame variants of its frames; `skycoord_fit` re-projected through each frame's current GWCS); `1/true/yes/on` or `0/false/no/off`, anything else raises before any residual is rendered; not read while `KFAC` is off; rows are marked by `SATSTAR_SIDELOBE_MASK_FRAC_MIN` in the satstar stage, and a missing column warns and keeps the consolidated rows' mask; the SCI header records `FPREJN` (guards section) |
| `SATSTAR_DAO_RESTORE_FRAC_MIN` / `SATSTAR_DAO_RESTORE_FRAC_MAX` / `SATSTAR_DAO_RESTORE_SEP_MAX_ARCSEC` / `SATSTAR_DAO_RESTORE_NEIGHBOUR_MIN_ARCSEC` | off / 1.25 / 0.03 / 0.1 | keep the daophot row a satstar overwrote when the in-field dedup would drop the satstar row: recorded flux in [`FRAC_MIN`, `FRAC_MAX`) of the satstar's, recorded position within `SEP_MAX_ARCSEC` of the satstar, no surviving satstar row within `NEIGHBOUR_MIN_ARCSEC` of it (0 turns that check off); off unless `FRAC_MIN` is above 0 (the other three are then not read); needs a merge that ran with the switch on (`satstar_dao_*` columns, section 6); a bad value raises at the start of `run_manual_pipeline` and `merge_individual_frames` |
| `SATSTAR_SEED_REQUIRE_DO_NOT_USE` | 0 (**off**) | restrict SEEDING to truly-lost cores (debug only — see §2b) |
| `SATSTAR_SEVERITY_FLOOR` | per-filter `SAT_SEVERITY_FLOOR` | override the daophot-mask severity floor |
| `SATSTAR_MIN_LOST_CORE` | 5 | min truly-lost core size (px) |
| `SATSTAR_SUBFLOOR_SEED_FRAC` | 0.35 | sub-floor seeding fraction |
| `SATSTAR_SEED_REQUIRE_SAT_DQ_FILTERS` | unset (off) | bandpasses (comma/space-separated, e.g. `F212N`) whose peak-based and sub-floor seeds must hold a DQ-SATURATED pixel; others are unaffected, a non-bandpass name raises, listed bands carry `sd` in the satstar cache key (#991) |
| `SATSTAR_WING_MULT_FILTERS` | unset (off) | bandpasses (comma/space-separated; only `F150W` has a table entry in `photometry/wingcal.py::WING_MULT_TABLE`, any other name raises) whose satstar catalog `flux_fit`/`flux_err` are divided by a ratio that ramps from 1 at `wingcal_rmask` 3.5 px to 10^(0.4·0.275) ≈ 1.29 at 5 px; the ratio is recorded in `wingmult_ratio` (listed bands only), the model, residual, `flux_fit_raw` and `wingcal_ratio` are unchanged, and listed bands carry `wm` in the satstar `SATFITSW` cache key (#1036). `satstar_content_key` hashes every `SATSTAR*` variable, so setting this one also changes the content key of unlisted bands (it feeds only the opt-in cross-phase reuse). The ratio is kept out of `wingcal_ratio`, so `apply_pooled_wingcal` still divides rows with `wingcal_ratio == 1.0` by the pooled C(r) on top of it; whether the 0.275 mag amplitude was measured before or after wing self-cal/pooled calibration is not established. Fitted on wd2 F150W against narrowband predictions; see the issue for the open evidence questions |
| `SATSTAR_COMPONENT_OVERLAP_FRAC` / `…_MIN_PX` | (see code) / 250 | merge overlapping saturated components |
| `SATSTAR_WINGCAL` | 1 | wing self-calibration (`apply_wing_selfcal`) |
| `SATSTAR_WINGCAL_MAX_SE` | 0.05 | largest standard error (1.2533 madstd/√n) of a C(r) bucket that wing self-calibration applies, per-frame and pooled; ≤ 0 disables the gate. madstd is floored at the fractional scatter of the smallest r_mask bucket (≥ 5 stars) times the bucket ratio, and C(r) is anchored at C(0) = 1 (#1041) |
| `SATSTAR_ZEROFRAME_FIT` | 1 | fit using the ZEROFRAME where available; blank = default, `1/true/yes/on` or `0/false/no/off` (any case), any other value raises |
| `SATSTAR_ZF_RCURVE_GUARD` / `…_MAXSTEP` | 1 (**on**) / 1.3 | truncate the ZEROFRAME R(g0) curve at the first bin-to-bin step larger than MAXSTEP, up or down; 0 restores the untruncated curve, which collapses on F480M (#972) |
| `SATSTAR_ZF_RCURVE_SATCHECK` | 1 (**on**) | compare the measured R(g0) curve with cal/group0 of the SATURATED pixels that keep a finite ramp-fit rate (no DO_NOT_USE, clean group 0, ≥ 50 pixels); when they differ by more than 2×, rebuild the curve from those pixels.  On sparse frames every calibration pixel above 2000 DN can be a hot or offset pixel, and the guard keeps that smooth junk curve (wd2 F164N/F187N/F212N: R 30–3000× low; w51 F140M/F162M/F187N/F210M/F405N in a 49-frame sample: 20 of 49 collapsed).  When fewer than 50 such pixels exist (NIRCam frames only), a curve whose bright-end R differs by more than 2× from PHOTMJSR / t(group 0), t(group 0) = TFRAME × (NFRAMES + 1) / 2, is replaced by that header rate (w51 F187N nrca1: 13 pixels, R 0.0026 vs 2.66), and the rim is R_header × group 0 minus the cal − crf level as under `SATSTAR_ZF_R_HEADER`, with that switch on or off.  Healthy curves sit at 0.88–1.04× the header rate in 46 frames (brick, sgrb2, gc-treasury, w51, wd2) |
| `SATSTAR_ZF_R_HEADER` | = `NIRCAM_SATSTAR_RECOVERED_CAP` (**on** for extended-emission NIRCam) | rewrite the ZEROFRAME rim with PHOTMJSR / t(group 0) in place of the measured R(g0) curve, on NIRCam frames where an R was measured; the curve is still measured, checked and logged.  Above 2000 DN the curve's calibration pixels are mostly the wings of saturated stars, which charge migration inflates: on wd2 the SW curve sat 4–9 % above the header rate on nrcb3/nrcb4 and 1–2 % on nrcb1, while pixels ≥ 25 px from any SATURATED pixel sit within 1–2 % of it.  In an offline refit of the wd2 F150W/F200W nrcb1+nrcb3 and F250M/F300M nrcblong satstars against dolphot it moved the satstar dm from −0.047 / −0.066 / +0.030 / +0.018 to −0.028 / −0.034 / −0.009 / −0.013 (unsaturated stars: +0.004 / −0.017 / +0.003 / −0.003) and the F150W nrcb3 − nrcb1 offset from −0.058 to 0.000 (#1142).  The rewrite is R_header × group 0 minus the level a destreak removed after calibration, cal − crf per row and amplifier (`zeroframe_cal_offset`; 0 without a sibling cal or when nothing was removed, as on the wd2 align products).  On w51 LW (legacy destreak, median cal − crf 7–17 MJy/sr) the header rim read 27–34 % above the kept SATURATED pixels below 2000 DN without it and 2–4 % from them with it (measured curve: 16–23 % above below 2000 DN, 5–8 % below above); on brick F356W nrcblong it restores 2501 of the curve's 3110 buffer rewrites (header alone: 261) (#1148) |
| `SATSTAR_ZF_FLAT` | 0 | divide the ZEROFRAME rim rewrite by the pixel flat the calibration applied (`R_FLAT`, read from the CRDS cache under `CRDS_PATH`; NaN, zero, negative, DO_NOT_USE and NO_FLAT_FIELD flat values read as 1, as in the jwst flat_field step), and measure the R(g0) curve and the R-curve check on cal × flat / group 0.  The crf is rate × PHOTMJSR / flat, while group 0 is raw DN, so R × group 0 leaves each rewritten pixel off by its flat value.  On wd2 the rim pixels' flat spans 0.97–1.02 (16–84 %), with satstar medians of 0.988 on nrcb3 and 1.005 on nrcb1 (F150W/F200W).  In an offline refit of the wd2 satstars under `SATSTAR_ZF_R_HEADER` the per-star slope of dm against 2.5 log10(flat) moved from −0.56 to −0.85 to +0.08 to +0.31 within each SW detector, and the MAD within one detector fell (F200W nrcb1 0.025 → 0.020, nrcb3 0.026 → 0.021; F150W 14–17 mag 0.028–0.043 → 0.024–0.037; F250M 13.5–15 mag 0.033–0.034 → 0.019–0.026).  The median moved by −0.013 on nrcb3 and +0.006 on nrcb1, which widens the F200W nrcb3 − nrcb1 satstar offset from −0.024 to −0.045 (unsaturated stars: −0.025) (#1149, #1142) |
| `SATSTAR_ZF_KEEP_FINITE` | = `NIRCAM_SATSTAR_RECOVERED_CAP` (**on** for extended-emission NIRCam) | leave SATURATED pixels with a finite ramp-fit rate and no DO_NOT_USE alone (not rewritten, not masked); on wd2 it removed the 0.05 mag bright offset of the first-frame rewrite on F150W (#1092) |
| `SATSTAR_OBS_PK_FROM_CRF` | 0 | the implied-peak gate reads its observed peak from the crf values, not the ZEROFRAME rewrite |
| `NIRCAM_SATSTAR_RECOVERED_MIN_PSF_FRAC` | 0.005 | the recovered-core cap skips a star whose model peak pixel is unmeasured when the measured pixel it reads has a unit PSF below this fraction of the peak; the data there belong to other sources (§2b) |
| `SATSTAR_ZF_G0_GROUPDQ` | = `NIRCAM_SATSTAR_RECOVERED_CAP` (**on** for extended-emission NIRCam) | first-read pixels flagged SATURATED in the ramp GROUPDQ (integration 0, group 0) are invalid for the ZEROFRAME anchor: deep core, not rewritten from their clipped value (#1065) |
| `SATSTAR_ZF_FIRST_FRAME` | = `NIRCAM_SATSTAR_RECOVERED_CAP` (**on** for extended-emission NIRCam) | with NFRAMES > 1, the ZEROFRAME anchor reads k(first frame) × the ramp ZEROFRAME extension (the first frame) where the ramp GROUPDQ flags group 0 SATURATED or DO_NOT_USE; k = binned median group 0 / first frame on the frame's unflagged pixels (`first_frame_group0`) |
| `SATSTAR_ZF_RIM_BADPIX` | 1 (**on**) | leave DEAD, HOT and REFERENCE_PIXEL pixels out of the ZEROFRAME rim rewrite, and floor a rewritten pixel's fit error at the frame's median ERR; 0 restores the rewrite that weighted dead pixels at 400–5e8× a normal pixel (#1071) |
| `SATSTAR_SEED_CORE_DQ` | 1 (**on**) | leave NaN-variance pixels carrying OUTLIER, DEAD, HOT, WARM, LOW_QE, RC, TELEGRAPH, NO_LIN_CORR, NO_SAT_CHECK, NO_GAIN_VALUE, NO_FLAT_FIELD, UNRELIABLE_BIAS, OTHER_BAD_PIXEL or REFERENCE_PIXEL out of the core that `_refine_coms_by_data` centres a seed on (`seed_saturation_core`), then the pieces of fewer than 3 px that remain (a HOT clump on a gc-treasury F480M star's peak otherwise left a 2-px edge pair to win, #1101); 0 restores the seed that an OUTLIER pair or a WARM/HOT clump on a star's edge moved 2.4–3.6 px off wd2 stars (#1098) |
| `SATSTAR_DEBLEND_PSF_MARGIN` | 0 (**off**) | with `--deblend-satstars`, a secondary peak of the ZEROFRAME residual must exceed this fraction of the PSF model of the accepted stars at its position (`satstar_deblend._psf_veto_peaks`, #1195); needs a gridded in-FOV PSF, falls back to the plain peak search above 5000 saturated px, under 30 fit px, above 40 model stars, when the fit fails or the stamp is degenerate (NaN, zero, negative); keys the cache as `dv<margin>` only for frames whose deblend runs (sibling `_ramp.fits`); can split more than the plain search where a saturated core is clipped by the image edge or two cores merge; set `OMP_NUM_THREADS=1` |
| `SATSTAR_QFIT_LOCAL_GATE` / `…_R` / `…_MAX` | 0 / 0 (10 when the gate is on) / 5.0 (the box qfit cap; 1.0 before #1058) | qfit over r < R px as a `qfit_local` column; with the gate on, NIRCam in-FOV fits of components carrying SATURATED DQ in the frame's own DQ are judged on it (whatever their `seed_kind`: a severity-dropped SAT component re-seeded as `subfloor` counts), and components with no SATURATED pixel keep the box qfit, which on sgra F405N kept thousands of unsaturated stars out of the satstar channel |
| `SATSTAR_DEBLEND_ORDER` | 0 | with the ZEROFRAME deblend, fit the seeds of one component largest ZF-saturated core first, then the unsaturated peaks in detection order.  The deblended seeds of a component all carry its `sat_area`, so the `-sat_area` sort leaves their order at the raster order of the cores; this adds the core radius (`order_key`) as a second sort key.  `sat_area` is unchanged.  The satstar cache key gains `do` on deblended frames.  Not yet measured on wd2: whether any over-bright row has a larger-core neighbour fitted after it is the census of #1094 |
| `SATSTAR_OPTIN_FILTERS` | (unset: every bandpass) | bandpasses (`F480M` or `F480M,F405N`) that keep the opt-in fit switches `SATSTAR_ZF_KEEP_FINITE`, `SATSTAR_OBS_PK_FROM_CRF`, `SATSTAR_QFIT_LOCAL_GATE`, `SATSTAR_QFIT_LOCAL_R` and `SATSTAR_DEBLEND_ORDER`; for any other bandpass those five are unset for the whole per-frame fit (`satstar_optin_scoped` on `load_or_make_satstar_catalog`), so the frame fits and keys its cache (`SATFITSW`) as if they were never exported.  A chain takes one environment, so gc-treasury sets the switches once with `SATSTAR_OPTIN_FILTERS=F480M`; a malformed name raises |
| `SATSTAR_ERR_BKG_SCATTER` | 1 for extended-emission NIRCam (set by `cataloging.py`), else 0 | add the robust local-background-annulus scatter in quadrature to the satstar fit errors (`bkg_scatter_fit_error`); NIRCam in-FOV fits only |
| `DAOPHOT_HANDOFF_UNACCEPTED_SAT` / `DAOPHOT_HANDOFF_DATA_FLOOR` | on for the targets in `_DAOPHOT_HANDOFF_UNACCEPTED_SAT_TARGETS` (gc-treasury), off elsewhere; an export wins / 0 | NIRCam only: hand every SATURATED component without an accepted satstar to daophot and return its valid-rate late-group pixels to the fit; limited to the bandpasses in `DAOPHOT_HANDOFF_FILTERS` when that is set; recorded per frame as catalog header `HNDOFFSA` |
| `SATPROX_SEEDED_COMPANION` | off | NIRCam m3..m7 daophot passes on frames whose hand-off restored pixels: a fit within 0.5 px of a seed-catalog position (the pass's `prev_seed_catalog`) is exempt from the `_filter_near_saturation` veto when its nearest SATURATED pixel NOT restored to the fit (`_handoff_restore_pixels`) is more than `near_sat_dist_pix` away and it has finite `flux_fit > 0` and finite `qfit`. Keeps companions 4-5 px from a handed-off bright star whose restored late-group footprint reaches within 1 px of them (gc-treasury F480M); fresh, unseeded fits on the restored footprint stay vetoed (`cataloging.py::_seeded_companion_kwargs`). On/off values follow the hand-off convention; an unknown value raises |
| `DAOPHOT_BRIGHT_PRESUB_PEAK_NOISE` | 0 (off) | m3..m7 daophot pass: seeds whose 3x3 data peak exceeds this multiple of the frame's median local noise are fit first and their model is subtracted before the detection that builds the seed, so faint neighbours within ~14 px of a bright daophot-fit star are not lost to the bright star's inflated local-noise map (the satstar channel already subtracts its stars first); the bright seeds plus the residual detections then go to the one full fit; non-numeric, negative or non-finite raises (#991) |
| `GC_RESID_NAN_SATSTAR_FOOTPRINT` / `GC_RESID_NAN_SATSTAR_PASSES` / `GC_RESID_NAN_SATSTAR_MIN_MODEL` | off / `m7` / 20 | report the DQ-SATURATED pixels under an accepted satstar's subtracted model (model > `…_MIN_MODEL` MJy/sr) as NaN in the per-frame `*_daophot_basic_residual.fits` written by the manual passes listed in `…_PASSES` (`m1` … `m7`, comma- or space-separated; the m3–m6 residuals seed the next pass and have not been run with NaN in them). The pass's catalog, `_model.fits` and `modelsub_bkg` columns are written from the unmasked arrays (`_save_manual_pass`, `_satstar_footprint_nan_mask`); the residual i2d, its smoothed background and the merged-catalog residual inherit the NaN.  The on/off switch follows the hand-off convention; an unknown pass label or a bad `…_MIN_MODEL` raises |
| `SATSTAR_LOG_VERBOSE` | 0 | verbose finder logging |
| `SATSTAR_DEDUP_ARCSEC` | 0.15 | consolidation dedup radius (`merge_catalogs`) |
| `SATSTAR_FLUX_STAT` | median | consolidated `flux_fit`: `median` of the per-exposure fluxes, or `brightest` (the representative's own flux, the old behaviour; kept as `flux_brightest_fit` either way) |
| `SATSTAR_REPLACE_RADIUS_FWHM` | 1.5 | radius of `replace_saturated`'s mutual-nearest second pass (a satstar the tight per-band radius missed pairs with its clipped daophot row) in units of the band's PSF FWHM, capped at 0.5"; a fixed 0.5" (3–17 FWHM in NIRCam) let a satstar with no daophot row of its own overwrite its nearest neighbour star |
| `SATSTAR_REPLACE_RADIUS_ARCSEC` | (unset) | fixed second-pass radius in arcsec, overriding `SATSTAR_REPLACE_RADIUS_FWHM`; 0.5 restores the radius used before the FWHM scaling, 0 disables the second pass |
| `SATSTAR_FP_*` (11 vars: `_REJECT`, `_REJECT_RATIO`, `_REJECT_MIN_N`, `_REJECT_BRIGHTFRAC`, `_FLUXRATIO`, `_MERGE_MAX_ARCSEC`, `_COMP_ARCSEC`, `_BIG_ARCSEC`, `_BIGCORE_ARCSEC`, `_BIGCORE_MERGE_ARCSEC`, `_USE_ANCHOR`) | see `merge_catalogs.py` | false-positive rejection / merging at consolidation |
| `SATSTAR_PIXSCALE_ARCSEC` | 0.063 | pixel scale used by those radii (no `FP_` in the name) |
| `STPSF_PATH` | (required) | WebbPSF grid data (set before import) |
| `G1ONLY_SAT_DQ` | `auto` | NIRCam single-group (G1-only) pixels → SATURATED\|DO_NOT_USE + NaN from the sibling `_ramp.fits`: `auto` flags only frames whose measured m = SCI/(G1/TGROUP) is nearer 1 (biased ramp fit) than r = TGROUP/((NFRAMES+1)·TFRAME/2) (see §2c); `1`/`force` flags every frame; `0`/`off` (or empty) disables; any other value raises |

The finder/merger read ~50 `SATSTAR_*`/`MIRI_*` variables in total; the table above
covers the ones that change shipped behaviour. `git grep "environ.get('SATSTAR"`
and `…'MIRI` is the authoritative list.

The seven `SATSTAR_ZF_*` / `SATSTAR_OBS_PK_*` / `SATSTAR_QFIT_LOCAL_*` rows,
`SATSTAR_SEED_CORE_DQ` and `SATSTAR_ERR_BKG_SCATTER` are read by
`satstar_fit_switches`, and the
per-exposure satstar cache is keyed on them
(meta `SATFITSW`, `satstar_fit_switch_signature`) beside `SATRECOV`, so changing
one refits the cached catalogs it affects. Their on/off values follow the
daophot hand-off convention: unset or blank gives the default, `1/true/yes/on`
and `0/false/no/off` are accepted in any case, and any other value raises.
A cap-on run (`NIRCAM_SATSTAR_RECOVERED_CAP=1`, read as an integer like the
fitter reads it) adds `cs<NIRCAM_SATSTAR_RECOVERED_MIN_PSF_FRAC>` to the key, so
catalogs fitted before the cap read only the seed's share of a blended
component are refit once.

The PSF is a cache key too.  With `PSF_EPSF_CORE_DIR` set, NIRCam satstar fits
use the hybrid PSF (ePSF core inside 10 px, STPSF outside 12 px; see
`PHOTOMETRY_PIPELINE.md`), and every satstar catalog records `PSFMODEL`
(`STPSF`, or `STPSF+EPSFCORE` for a hybrid fit) and, for a hybrid fit, `EPSFSHA` (sha256 of the core file).
`load_or_make_satstar_catalog` compares `EPSFSHA` with the core the run would use
before it looks at `SATRECOV`/`SATFITSW` (including when `recovery_signature` is
`None`), so enabling the variable, or editing a core in place, refits the
affected filter's catalogs once, and an STPSF run does not adopt a hybrid
catalog.  The extended (`force_union`) catalogs go through the same check; the
script that writes them (`force_union`) is not in this repo, so whether it copies
`EPSFSHA` is unchecked, and one that does not would be refit on every hybrid call.  The
cross-phase content key carries `psf_core=<sha>` only when a core applies.

---

## 8. Key files

| file | responsibility |
|---|---|
| `reduction/saturated_star_finding.py` | detection, PSF fitting, accept gate, ZEROFRAME recovery, off-FOV reconciliation |
| `reduction/satstar_deblend.py` | ZEROFRAME core deblending (crowded fields) |
| `reduction/g1only_dq.py` | single-group (G1-only) pixel flag and its per-frame `auto` gate (§2c) |
| `reduction/PipelineRerunNIRCAM-LONG.py` / `PipelineMIRI.py` | Detector1 → `_ramp.fits` (ZEROFRAME/GROUPDQ); MIRI edge trim |
| `photometry/cataloging.py` | per-frame satstar+daophot integration, `miri_tuning` schedule, artifact/coincidence gates |
| `photometry/crowdsource_catalogs_long.py` | CLI options, manual m12→m8 pipeline |
| `photometry/merge_catalogs.py` | satstar catalog consolidation, dedup, `replaced_saturated` merge |

*See also `PERFORMANCE_BRICK.md` (satstar models are cached/reused) and
`NOTES_star_vs_extended_emission.md`.*
