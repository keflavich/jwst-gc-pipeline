# Saturated-star fit footprint: decoupling, cost, and the flux-correction method

**Status:** in progress (calibration running). **Branch:** `feature/adaptive-satstar-fitshape`.
**Scripts:** `scripts/satstar_deblend/{fit_footprint_sweep,adaptive_fit_shape_ab,sweep_env}.*`,
correction calibration + validation (below).

## 1. Motivation

Fitting saturated stars over a large footprint is (a) computationally costly and
(b) risks biasing the fit by pulling faint, contaminated neighbours into the
amplitude solution. Matt (PEPPAR) raised decoupling the *fit* footprint from the
*model* footprint (the latter used for PSF-wing rejection and background maps).
Question: how much worse are our fits with a smaller fitting footprint, and can
we fit small while modelling big?

## 2. The decoupling already exists

In `get_saturated_stars` (`reduction/saturated_star_finding.py`) the fit
footprint `size` is passed only to `PSFPhotometry(fit_shape=size)` (~line 3012).
The cutout, PSF-model grid (`fov_pixels`), saturated-core mask, background
annulus, and the rendered model image used for wing rejection + background maps
are all keyed to `pad`/`fov_pixels` alone. So a small `size` with `pad`
unchanged fits small while modelling big — no structural change needed. (The
normal-star daophot path is already decoupled: `fit_shape=(5,5)`, `fovp101`
model, `(21,21)` render.)

## 3. Fit-footprint sweep — fit-small shifts satstar FLUX; position holds

`fit_footprint_sweep.py`, one gc2211 F200W frame, top-20 saturated stars, `pad`/
model/mask held at 81, only `size` swept:

| size | median Δflux vs 81 | median Δpos | wall-clock | rel cost |
|---|---|---|---|---|
| 11 | −88.5% | 73 mas | 91 s | 0.16× |
| 17 | −32% | 3.7 mas | 100 s | 0.17× |
| 21 | −30% | 5.0 mas | 110 s | 0.19× |
| 31 | −16% | 2.1 mas | 148 s | 0.25× |
| 51 | −5.2% | 0.7 mas | 273 s | 0.47× |
| 81 | 0 (ref) | 0 | 585 s | 1.00× |

A small box excludes most of the fitted pixels (`n_pixels_fit` 3324 → 44 as size
81 → 11) and the fitted flux **departs from the size-81 value by 5–90%**,
monotonic and systematic (the small box gives *lower* flux). The brightest star
is unfittable below size 21 (box inside the masked core). Position is
footprint-robust (<5 mas by size ≈17). Which footprint is *correct* is a
separate question — see §7.1; the leverage-vs-contamination interpretation below
is a hypothesis pending those controls. (Note: `flux₈₁/flux_size` is written `R`
in §7; here the same effect is quoted as the signed percentage change.)

## 4. Adaptive-by-core leaves the flux change in place

`adaptive_fit_shape_ab.py`: opt-in `adaptive_fit_shape=True` sets
`fit_shape = clip(2.83·r_core + 17, 21, 81)` (r_core = √(sat_area/π), forced
odd). A/B vs flat-81 on one gc2211 F200W frame: **80% faster** (117 s vs 585 s),
position free (1.4 mas), and a **systematic −11% flux change at all brightnesses**
(a signed shift) vs the size-81 fit. Scaling the box to the core spreads
the change evenly, and the change survives. (Which footprint is correct stays
open — §7.1; on gc2211's deep-crowded field the two disagree.)

## 5. Method under validation — fit-small + size-dependent flux correction

The bias in §3 is *deterministic* in the fit footprint. So: fit at the small
adaptive box (keep the speedup + reduced contamination), then multiply the flux
by a correction `C(size_used, r_core, filter)` calibrated to recover the flat-81
flux — an aperture-correction analog for the fit footprint.

- **Calibration:** run the size sweep across filters (SW: F200W, F212N; LW:
  F405N, F410M) and ≥2 frames each; measure `flux_81 / flux_size` vs
  `(size, r_core, filter)`; fit a smooth monotonic correction per filter.
- **Correction lives in** `get_saturated_stars` (applied after the fit when
  adaptive + correction enabled).

## 6. Validation suite (targets)

| # | test | target |
|---|---|---|
| 1 | flux corrected-small vs flat-81, binned by brightness AND core size | median \|Δ\| <2%, \|bias\| <1%, 90th <5% |
| 2 | position corrected-small vs flat-81 | <5 mas |
| 3 | wall-clock speedup | report |
| 4 | robustness (failed fits, accepted count) | no regression |
| 5 | **end-to-end** `assert_saturation_continuity` (paper CMD metric ~0.11) | preserved |
| 6 | held-out generalization (calibrate set A, validate set B) | targets hold on B |

Datasets: brick {F200W, F212N (SW); F405N, F410M (LW)} for calibration; gc2211
F200W + brick {F182M, F466N} held out.

## 7. Results — the correction is dataset-dependent

Calibration array (job 38335896, brick 2 SW + 2 LW × 2 frames, 17,306 matched
star-size rows) + the gc2211 F200W sweep. Reference throughout is the size-81
fit; **"R" measures departure from that reference, whose status as truth is
open** (§7.1).

`R = flux₈₁ / flux_size`, F200W:

| dataset | R @ 11 | R @ 31 | R @ 51 | local_bkg | NGROUPS / EXPTIME |
|---|---|---|---|---|---|
| brick F200W (prop 1182) | 1.02 | 1.01 | 1.00 | 0.75 | **7 / 344 s** |
| gc2211 F200W (prop 2211) | 1.91 | 1.12 | 1.03 | 1.8 | **2 / 97 s** |

The two datasets diverge strongly, and **the divergence survives at *matched*
r_core** (`R@31`, N in parentheses):

| r_core | gc2211 R@31 (N) | brick R@31 (N) |
|---|---|---|
| 3–4 | 1.10 (38) | 1.03 (54) |
| 4–5 | 1.11 (86) | 0.97 (4) |
| 5–7 | 1.18 (62) | 0.98 (20) |

(gc2211 = 1 frame, 189 in-FOV satstars; brick = both F200W frames pooled. The
r_core 4–5 brick bin stays thin — brick simply has few stars there.) Where the
two overlap in r_core, gc2211 still sits ~10–18% higher. So R depends on more
than `(size, r_core, filter)`; a correction calibrated on one
dataset (≈1.0 for brick) would mis-correct the other by 10–20%.

### 7.1 Open confounds

- **Which footprint is truth stays open.** R > 1 in gc2211 is equally consistent
  with the *small* box being right and the *large* box over-counting (blended
  neighbours / diffuse background in a crowded field). The sign alone leaves the
  correct footprint undetermined; an independent flux reference (unsaturated
  curve-of-growth, aperture photometry, or synthetic-injection recovery) would
  settle it and remains to be run. The established result here is that the fits
  *differ*.
- **Field and saturation depth are confounded.** brick F200W (prop 1182) is
  NGROUPS=7 / 344 s; gc2211 F200W (prop 2211) is NGROUPS=2 / 97 s. The two
  datasets differ in saturation depth and exposure as well as crowding/background,
  so the driving axis (crowding, background, ramp depth, or a mix) remains
  unidentified. Both are per-observation quantities.
- **The two PSF grids share a name and differ in content.** The
  `nircam_nrca1_f200w_fovp512…fits` files differ between the two `psfs/` dirs
  (different md5), so the PSF model is an additional uncontrolled variable.
- **Thin coverage in the large-r_core / bright regime**, small N per bin, single
  frame per dataset for the cross-dataset table, no error bars.

**Conclusion:** the departure from size-81 is **dataset/observation-dependent
beyond `(size, r_core, filter)`**, so the evidence rules out a *universal*
flux correction (approach 1 as originally scoped). A richer, depth/crowding-aware
correction — and which footprint is truth — stays **open**, pending the controls
in §7.1.

## 8. Recommendation

1. **Keep the current default (full fit box) for saturated-star photometry** —
   the size-dependence is large and its truth/confounds are unresolved, so a
   change would be unvalidated. Production already uses the full box.
2. **Fit-small is safe for saturated-star ASTROMETRY** in both datasets tested
   (position robust to <5 mas by size ≈17 across thousands of stars, ~6×
   faster) — the one clean, well-supported win. Use small boxes for
   position-only passes.
3. **Before any photometry speedup**, run the §7.1 controls: (a) an independent
   flux truth to fix the direction of the effect; (b) disentangle field vs
   NGROUPS/depth (e.g. same field at two ramp depths, or match sat_area AND
   NGROUPS); (c) confirm/rebuild matched PSF grids. Only then is a
   depth/crowding-aware correction worth attempting.

The opt-in `adaptive_fit_shape` (default off) is retained as the astrometry-pass
lever and to reproduce the A/B.

Net: the decoupling exists and position is cheap; saturated-star **flux** needs
the full box until a dataset-aware correction is validated.
