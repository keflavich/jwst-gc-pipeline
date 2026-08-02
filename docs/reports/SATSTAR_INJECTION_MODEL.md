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

**Critical caveat (unresolved).** The absolute magnitude (~1 mag hard-sat) depends
on (a) forward-model realism — does it remove the right amount of flux? — and
(b) whether injected stars receive the same **wing self-calibration** as production
stars. The paper's continuity fix reached ~0.11 mag residual, so a +1.16 mag
hard-sat bias is either a real deep-regime failure the continuity metric never
probed, OR a harness artifact (injected stars not wing-self-calibrated). Resolving
this is the top next step.

## 5. Status / next steps
1. [ ] `saturation_forward_model.py` — the physics above, CRDS-driven (this doc's §2).
2. [ ] injection+recovery driver (extend `artificial_stars.py` path) → recovered/injected vs sat-depth.
3. [ ] calibrate `f_bf` on gc2211 NGROUPS=2 #210 curve; validate charge-migration prediction on brick NGROUPS=7.
4. [ ] multi-field matrix (§4) via SLURM array.
5. [ ] bias-vs-regime curves per field/filter; feed the wing-selfcal / empirical-wing work.
