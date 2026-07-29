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
are all keyed to `pad`/`fov_pixels`, **not** `size`. So a small `size` with `pad`
unchanged fits small while modelling big — no structural change needed. (The
normal-star daophot path is already decoupled: `fit_shape=(5,5)`, `fovp101`
model, `(21,21)` render.)

## 3. Fit-footprint sweep — fit-small biases satstar FLUX, not position

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

The amplitude leverage of a saturated star lives in its extended unsaturated
wings/spikes (the core is masked). A small box excludes them (`n_pixels_fit`
3324 → 44 as size 81 → 11) and **underestimates the flux by 5–90%**, monotonic
and systematic. The brightest star is unfittable below size 21 (box inside the
masked core). Position, by contrast, is footprint-robust (<5 mas by size ≈17).
The contamination worry is *inverted* for these bright stars: a small box gives
*lower* flux, not higher — wing leverage dominates faint contamination.

## 4. Adaptive-by-core alone does not fix it

`adaptive_fit_shape_ab.py`: opt-in `adaptive_fit_shape=True` sets
`fit_shape = clip(2.83·r_core + 17, 21, 81)` (r_core = √(sat_area/π), forced
odd). A/B vs flat-81 on one gc2211 F200W frame: **80% faster** (117 s vs 585 s),
position free (1.4 mas), but a **systematic −11% flux bias at all brightnesses**
(signed, not scatter). Scaling the box to the core spreads the loss evenly but
does not remove it — the wing leverage genuinely needs the area.

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

## 7. Results — the correction does NOT generalize (approach-1 premise disproven)

Calibration array (job 38335896, brick 2 SW + 2 LW × 2 frames, 17,306 matched
star-size rows) + the earlier gc2211 F200W sweep.

**The flux-footprint bias is field/environment-intrinsic, not a function of
(size, r_core, filter).** Median R = flux₈₁/flux_size:

| field / regime | R @ size 11 | R @ 31 | R @ 51 | local_bkg |
|---|---|---|---|---|
| brick F200W (low bkg) | 1.02 | 1.007 | 1.001 | 0.75 |
| gc2211 F200W (dense GC, high bkg) | 1.91 | 1.12 | 1.03 | 1.8 |

Same filter, same PSF grid type (`nircam_..._fovp512`), similar r_core
distributions (median ~4.6) — yet brick shows ~0% footprint sensitivity while
gc2211 shows −47% at size 11. **At matched flux** the fields still differ:

| flux bin | gc2211 R@31 | brick R@31 |
|---|---|---|
| 3e4–1e5 | 1.11 | 1.00 |
| 1e5–3e5 | 1.17 | 0.90 |
| 3e5–1e7 | 1.19 | 0.98 |

So flux, r_core, and local_bkg do **not** predict R across fields (within gc2211
R tracks flux/r_core at ρ≈0.5 but is flat in local_bkg; the field offset is
unexplained by any per-star quantity). gc2211's brighter stars on a ~2.3× higher
background genuinely carry more wing flux outside a small box (flux doubles from
size 11→81); brick's stars are already well-determined by a small box.

**Consequence:** a universal `C(size, r_core, filter)` correction calibrated on
one field is ≈1.0 and would leave a dense field's photometry biased −15–20%.
Approach 1 (fit-small + universal flux correction) is therefore **not viable**.

## 8. Recommendation

1. **Keep the full fit box (81) for saturated-star photometry.** It is genuinely
   needed in dense/high-background fields (gc2211-type) — not just contamination
   — and is harmless (only slower) where it isn't. Production already does this.
2. **Fit-small is free for saturated-star ASTROMETRY** in every field (position
   is footprint-robust to <5 mas by size ≈17, ~6× faster). Use small boxes for
   position-only passes.
3. **Speedup for photometry only via a validated environment gate** (small box
   where background/crowding is demonstrably low). The predictor is field/local-
   environment level (not per-star flux/r_core/local_bkg), so this needs its own
   calibration + per-field verification; modest payoff, deferred.

Net: the decoupling exists and position is cheap, but saturated-star **flux**
cannot be shrunk with a universal correction — the extensive validation caught
this before it could silently bias dense-field catalogs.
