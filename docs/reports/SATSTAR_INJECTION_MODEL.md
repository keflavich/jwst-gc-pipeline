# Saturated-star injection: a physically-motivated forward model for measuring recovery bias vs saturation regime

**Goal.** Inject known-flux stars spanning unsaturated → mildly → deeply → hard-saturated, run the full satstar recovery, and plot **recovered/injected flux vs saturation depth**. This is the direct, absolute test of "unbiased recovery across saturation regimes" that we have never had (`artificial_stars.py` explicitly injects *no* saturated stars).

## 1. The three physical regimes (literature-grounded)

JWST NIR detectors are Teledyne **H2RG** HgCdTe arrays. Three distinct effects bias the ramp of a bright/saturated source; they must be modelled separately because the recovery must be unbiased across all of them.

### (a) Per-pixel nonlinearity — the *isolated* effect
Intrinsic to a single pixel: the p-n junction capacitance changes as charge accumulates, so measured counts roll over below the true fluence. Sub-% at low signal → tens-of-% near full well (Canipe, Robberto & Hilbert 2017, *A New Non-Linearity Correction Method for NIRCam*, JWST-STScI-005167; correctable to 0.2% over 97% of the range). Happens **even for an isolated pixel**. Modelled with the **CRDS `linearity` reference** (per-pixel polynomial) run in reverse.

### (b) Hard saturation
The pixel reaches full well and clips. From the group at which cumulative charge ≥ the **CRDS `saturation` reference** (per-pixel full-well, DN; NRCA3 median ≈ 53,000 DN, hard 16-bit clip 65,535), the pixel is flagged `SATURATED`. A pixel saturated **in group 0** is truly lost → `DO_NOT_USE`, `VAR_POISSON = NaN`. A pixel saturated later is *ramp-recoverable* from its good early groups.

### (c) Charge migration — the neighbor-coupled effect (the one the user means)
This is the JWST-documented **`charge_migration`** step (calwebb_detector1). Mechanism, quoting the pipeline docs: *"the spilling of charge from a saturated pixel into its neighboring pixels,"* which makes *"group-to-group differences decrease significantly once the signal level is greater than ~25,000 ADU."* So a pixel **next to** a deeply-saturated pixel receives **excess** charge after the neighbor saturates — exactly the user's description: excess flux appears in neighbors *after* the core saturates, distinct from (a) which needs no neighbor.

Two consequences, and they point in opposite directions depending on what you measure:
- **At the count level:** the neighbor reads **high** (excess migrated charge).
- **At the ramp-fit level:** because the excess arrives late and then the group-to-group differences flatten, a naive slope fit is **depressed**; the pipeline instead flags all groups above `signal_threshold` (≈25,000 ADU) as `CHARGELOSS` + `DO_NOT_USE` and drops them from the slope.

Physically this is the saturation-driven limit of the **brighter-fatter effect** (Plazas et al. 2018, *Laboratory Measurement of the BFE in an H2RG*, PASP 130, 065004): during integration, newly-arrived charge is preferentially attracted to *less-full* neighbors and repelled from *more-full* ones, so the PSF grows fatter with brightness; once a pixel saturates, the "spill to less-full neighbors" becomes the gross charge-migration excess.

**Critical: `charge_migration` is NOT applied to 1–2 group integrations.** This directly explains the #210 field-dependence the review flagged as confounded:

| field | NGROUPS | charge_migration | #210 fit-flux vs footprint |
|---|---|---|---|
| brick F200W | 7 | **ON** (flags the excess) | R ≈ 1.0 (footprint-stable) |
| gc2211 F200W | 2 | **OFF** (excess persists in wings) | R up to 1.9 (footprint-sensitive) |

At NGROUPS=2 the migrated excess is never flagged, so it sits in the wing pixels the amplitude fit integrates → a larger fit box pulls in more excess → flux climbs with footprint. This is a **testable prediction**: the injection model with charge migration must reproduce "footprint-sensitive flux at NGROUPS=2, stable at NGROUPS≥3."

## 2. Forward model (ramp-level injection)

For an injected scene of known-flux PSFs, simulate the up-the-ramp accumulation in electrons, apply the three effects, then ramp-fit exactly as the pipeline does, producing cal-like `SCI` / `DQ` / `VAR_POISSON`:

```
per group g in 0..NGROUPS-1:
  Q_true[g] = Q_true[g-1] + rate_e * t_group          # linear accumulation (e-)
  # (c) brighter-fatter / charge migration redistribution:
  #   move dQ from each pixel to its 4-neighbours ∝ (fullness_self - fullness_nbr)_+,
  #   with coupling f_bf that RISES steeply once a pixel > migration_threshold
  Q_mig[g] = redistribute(Q_true[g], f_bf, migration_threshold)
  # (a) per-pixel nonlinearity (CRDS linearity poly, applied forward):
  C[g] = nonlinearise(Q_mig[g] / gain)                 # DN
  # (b) hard saturation (CRDS saturation ref):
  SAT[g] = C[g] >= fullwell_DN ; C[g] = min(C[g], fullwell_DN)
ramp-fit slope over groups that are NOT SATURATED and (if NGROUPS>=3) NOT CHARGELOSS(>signal_threshold)
-> SCI (DN/s); DO_NOT_USE + VAR_POISSON=NaN where group-0 saturated
```

Parameters come from CRDS (`gain`, `saturation`, `linearity`) + the exposure header (`NGROUPS`, `TGROUP`). The **one free parameter is the charge-migration coupling `f_bf`**; calibrate it so injected NGROUPS=2 stars reproduce the observed #210 R-vs-footprint curve, then hold it fixed and validate on held-out fields.

## 3. What we measure

Inject a ladder of magnitudes (unsaturated → group-0-saturated) at many positions, recover with `get_saturated_stars` (production settings), and plot **recovered/injected vs saturation depth** (parametrised by `sat_area` / peak-group-of-saturation). A flat line at 1.0 = unbiased; the shape of any deviation is the bias-vs-regime curve — the pass/fail for the goal, and the thing to test the wing-selfcal / zeroframe / (upcoming) empirical-PSF-wing fixes against.

## 4. Field matrix (spin-out)

Chosen to span background × crowding × filter-width × NGROUPS (charge_migration on/off):

| field | filters | regime |
|---|---|---|
| gc2211 | wide (F200W…) | super-saturated, high bkg, **NGROUPS=2 (cm OFF)** |
| brick / other GC (1182/2221) | narrow+med+wide | tons of sat+unsat, NGROUPS=7 (cm ON) |
| ngc6334 / sgr* | med+narrow, extended emission | structured background |
| m4 / m92 / ngc6397 | wide+med | crowded, **negligible background** |

Low-background globulars isolate the crowding/PSF axis from the diffuse-background axis; the GC fields add the background + deep-saturation axes; gc2211 is the charge-migration-OFF extreme.

## 4b. First fleet results (10/18 runs; F200W+F212N across brick/gc2211/ngc6334/sgra)

Recovered − injected mag (+ve = recovered too faint), median per saturation regime:

| field | filt | NG | mild | moderate | deep | hard |
|---|---|---|---|---|---|---|
| brick | F200W | 7 | – | +59 (4.5σ) | +521 | **+1161 (14σ)** |
| ngc6334 | F200W | 7 | +182 | +151 | +314 | +720 |
| brick | F212N | 7 | +60 | – | +148 | – |
| sgra | F212N | 7 | +28 | +43 | +77 | −20 |
| gc2211 | F200W | 2 | +41 | +33 | +33 | +91 (all ±500–840 scatter) |

**Findings.**
1. **Recovery is biased FAINT and the bias GROWS with saturation depth** — clean and
   high-significance in brick/ngc6334 F200W (up to ~1.16 mag for hard-saturated).
   Narrow-band (F212N) is milder (saturates less). This is a clear, previously
   unquantified bias-vs-regime signal.
2. **gc2211 (NGROUPS=2, super-saturated wide-band)** has a small median bias but
   ENORMOUS scatter (±0.5–0.8 mag) and no clean trend — that regime is recovered
   imprecisely rather than cleanly biased.
3. **Charge migration (f_bf 0 vs 0.3) barely moves the recovered-flux bias**
   (#210 test: 3–4 mmag shift for BOTH NGROUPS=2 and 7). So the depth-dependent
   faint bias is driven by **saturation flux-loss + wing-extrapolation shortfall**,
   not charge migration at f_bf=0.3. (Charge migration may still drive the #210
   *footprint* sensitivity — a different metric — and/or need a larger f_bf.)

**Caveat RESOLVED 2026-08-02 — the +1 mag bias is REAL recovery under-correction.**
Two checks:
- *Wing-truth* (`wing_truth_check.py`): the forward model PRESERVES the true PSF
  amplitude in the surviving unsaturated wings — forward/true = 0.96–1.00 in every
  annulus up to hard saturation. So the injected star carries the true flux; the
  harness is not over-suppressing.
- *Wing-selfcal engaged but under-corrects*: the recovery ran wing self-cal (9
  real-star calibrators on the injected brick F200W frame), but the truth/masked
  ratio grows with mask radius — r5≈1.10, r15≈1.59, r18≈1.95, **r27≈4.0** — while
  the recovery applied only a **median 1.096×** correction. Deep/hard stars (large
  mask radii) are therefore under-corrected by exactly the amount of the measured
  +0.5–1.2 mag bias.

**So the injection harness has diagnosed a specific, fixable RECOVERY deficiency:**
wing self-cal applies a near-global (median) correction, but the masked-core bias
scales steeply with saturation depth (mask radius). The fix belongs in
`apply_wing_selfcal` — apply a per-star, mask-radius-dependent correction, not a
single median. This is the likely origin of the CMD "hook."

## 4c. f_bf calibration — NEGATIVE result (2026-08-02)

Injected into gc2211 (NGROUPS=2) at f_bf ∈ {0, 0.5, 1, 2}, recovered at fit_shape
31 vs 81. R = flux₈₁/flux₃₁ (`calibrate_fbf.py`):

| f_bf | R(81/31) | N |
|---|---|---|
| 0.0 | 1.007 | 90 |
| 0.5 | 1.010 | 83 |
| 1.0 | 1.009 | 77 |
| 2.0 | 1.014 | 79 |

Target (real gc2211, #210 matched-r_core) = **1.10–1.18**. The modelled
nearest-neighbour charge migration is **flat in f_bf** and never approaches the
real value — injected gc2211 stars behave brick-like (R≈1.0), NOT like real
gc2211 (R≈1.1–1.18). Consistent with the fleet (f_bf 0→0.3 moved recovery bias
only 3–4 mmag). **Conclusion:** charge migration *as modelled* is not the driver
of the #210 footprint sensitivity; that sensitivity is a **real-data effect** the
forward model does not reproduce — most likely PSF-wing **model mismatch** (STPSF
theoretical wings vs the real, broader wings — the Jay empirical-PSF-wing work),
and/or real charge migration that is stronger/longer-range than a 4-neighbour
spill. So f_bf stays a nuisance knob (kept small); the recovery-bias story (§4b)
is driven by wing-selfcal under-correction, not charge migration.

## 4d. CORRECTION (2026-08-02) — §4b was wrong; the bias is a harness/wing issue, and it's UNRESOLVED

The wing-selfcal ON/OFF test (`test_wingcal_onoff.py`, brick F200W, 11 hard stars):

| regime | bias wingcal ON | bias wingcal OFF |
|---|---|---|
| hard | **+1161 mmag** | **+6 mmag** |
| deep | +521 | −48 |
| moderate | +9 | −56 |

**The masked-core FIT itself is unbiased** (wingcal OFF ≈ 0). The entire
depth-growing bias enters through the **wing-selfcal correction** (÷ a ratio that
reaches ~4× at r≈27 px, with ±3× scatter). So §4b's "real recovery
under-correction" was the wrong reading — the correction is *over*-doing it here,
and there is no `apply_wing_selfcal` "median-vs-per-star" bug (it already
interpolates per rmask).

**But whether that over-correction is REAL (a genuine deep-star failure = the CMD
hook) or a HARNESS ARTIFACT is not cleanly resolved:**
- STPSF-wing injections truly need ratio≈1 but the recovery applies ~4× → they
  are over-corrected → but that under-represents real stars (real wings *are*
  broad, so real stars genuinely need a large correction).
- A crude radial ×1.5 "empirical wing" boost (`empirical_wings.py`) did NOT fix
  it (hard +1656, N=9 thin) — a radial surface-brightness boost does not
  reproduce the wing-selfcal's integrated *flux* ratio (~4×).

**What is solid:** (a) the core fit is unbiased; (b) the deep-star bias lives in
the wing-selfcal correction at large mask radius, where its ratio is large and
NOISY (±3×). That noisy large-r extrapolation is the prime suspect for the hook.

**To resolve needs a FAITHFUL empirical-PSF injection** — inject a full 2-D
empirical PSF stacked from the frame's real bright unsaturated stars (the same
population the wingcal calibrates on), not a radial approximation. Then the
injected stars reproduce the real masked-fit/truth ratio by construction and the
harness measures the true recovery bias. This ties directly to
`scripts/analysis/wing_calibration/` (stack_psf_2d.py) and the Jay empirical-wing
effort. Deferred to a decision.

## 5. Status / next steps
1. [ ] `saturation_forward_model.py` — the physics above, CRDS-driven (this doc's §2).
2. [ ] injection+recovery driver (extend `artificial_stars.py` path) → recovered/injected vs sat-depth.
3. [ ] calibrate `f_bf` on gc2211 NGROUPS=2 #210 curve; validate charge-migration prediction on brick NGROUPS=7.
4. [ ] multi-field matrix (§4) via SLURM array.
5. [ ] bias-vs-regime curves per field/filter; feed the wing-selfcal / empirical-wing work.
