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

## 5. Status / next steps
1. [ ] `saturation_forward_model.py` — the physics above, CRDS-driven (this doc's §2).
2. [ ] injection+recovery driver (extend `artificial_stars.py` path) → recovered/injected vs sat-depth.
3. [ ] calibrate `f_bf` on gc2211 NGROUPS=2 #210 curve; validate charge-migration prediction on brick NGROUPS=7.
4. [ ] multi-field matrix (§4) via SLURM array.
5. [ ] bias-vs-regime curves per field/filter; feed the wing-selfcal / empirical-wing work.
