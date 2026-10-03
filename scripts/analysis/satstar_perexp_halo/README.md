# Per-exposure scattered halo of saturated F480M stars (program 10678)

The leave-one-dither-out (LOO) model of `scripts/analysis/satstar_poisson` leaves
~5% of the local halo intensity around every saturated F480M star, because the halo
changes between dithers (#993/#994/#998). Here that change is measured for every
saturated star in the program that is bright enough. The question is whether it
follows a pattern that a model can remove.

## Pipeline

| step | script | output |
|---|---|---|
| 1 | `halocal_extract.py <dir>` | Streams all 808 LW F480M `_cal` frames from the public S3 mirror. Keeps star-centred cutouts of every saturated region of ≥250 px, plus the frame's GWCS. |
| 2 | `halocal_measure.py` | `meas.npz`, with 16,361 cutouts of 4,106 linked stars. Holds the off-spike median halo `H` in 12 annuli (15–250 px) × 12 sectors, with its noise `He`. |
| 3 | `perexp_measure.py meas.npz pxh.npz` | The per-exposure change of each star-visit with ≥4 exposures. See the definitions below. |
| 4 | `perexp_models.py pxh.npz models.json` | Cross-validated models of the change (5-fold by star). |
| 5 | `perexp_figs.py pxh.npz models.json <out> 0.02 <epsf_map dir>` | Figures 1–9 and `tests.json`. Includes the self-constrained tests. |

Steps 1–2 are copied unchanged from the halocal branch (satstar_poisson README §7).
Step 1 reads its key list from `lw_cal_keys.json` in the *parent* of `<dir>`.
That list is committed here: the 808 `[S3 key, size]` pairs, 404 NRCALONG and
404 NRCBLONG. Copy it there before running.

The last argument of step 5 is optional. It is the directory with the epsf_map
`result_nrc{a,b}long.npz` (#1002/#1007), which supplies the field-star wing-vs-column
comparison.

## Definitions

- **Star-visit:** one star in one observation. It has 6 dithers, thrown ±385 px in x and ±85 px in y, all at one roll. So the sky scene is common to all of a star-visit's exposures and cancels in their difference.
- **Reference:** `Hbar` is the mean of `H` over the star-visit's exposures.
- **Halo profile:** fit `p(r) = B + A (r/30)^-α` to the sector-median reference. The halo is `I(r) = A (r/30)^-α`, with the scene `B` removed.
- **Fractional change:** δ_e(r,θ) = (H_e − Hbar)/I(r).
  - A cell is kept only where `I > 0.5 B` and `I > 5 He`.
  - Star-visits whose scene fit degenerated (`B ≤ 0`, or α outside 0.5–8) are dropped.
- **Amplitude:** `a` is the inverse-variance mean of δ over r = 15–80 px. `prof` is its mean over azimuth in each annulus. `dip` is a cos/sin fit over r = 20–65 px.
- **Core area:** `area` is the saturated-core size (DQ SATURATED pixel count) relative to the star-visit mean. It is set from the raw ramp, independently of the halo pixels.
- **Model weights:** 1/(σ² + 0.08²). The change scatters by ~0.1 from star to star, while σ(a) is ~0.003, so inverse-variance weights would hand the fit to a few of the brightest stars.

## Results (8,933 exposures, 1,806 star-visits, 1,739 stars)

- **rms of a:** 0.095, against a formal noise of 0.003.
- **Dither index:** explains ≤5% (per-dither medians ±2%).
- **Shared per-exposure term:** a term shared by all stars of one exposure explains −7%, i.e. less than nothing, cross-validated.
- **Dipole:** none (<1%).
- **Detector column (the pattern):** a free function f(x) per detector, on 64-px knots, explains **43%** cross-validated.
  - NRCBLONG: a deficit centred at x ≈ 400 px, about 300 px wide.
    - Binned median `a` (`tests.json`, 64-px bins): −0.20 at the trough. The mean is −0.15 over x = 250–550 against +0.02 at x < 150 or > 750.
    - Core area: −0.25 at the trough; −0.19 against +0.02.
  - NRCALONG: a U shape.
    - Binned `a`: +0.08 at x < 300 or > 1750 against −0.03 at x = 700–1350.
    - Core area: +0.10 against −0.04.
  - The fitted spline `f(x)` (`tests.json` `xprofile.f`) swings further: −0.27 for the NRCBLONG trough and +0.30 for the NRCALONG edges. Those amplitudes are model-defined.
    - δ is relative to each star-visit mean, so `f` is fixed only up to a constant and over the ±385 px dither span.
    - Its edge values are pinned to 0, and the curvature penalty (`PEREXP_LAM`) shapes the contrast.
    - Quote the binned numbers.
  - The dependence on detector row is weak.
  - The low-order Legendre "position" model reaches only 17%, because it cannot follow the NRCBLONG feature.
  - The saturated-core area traces the same f(x) independently.
  - **Unsaturated field-star PSF wings** (epsf_map, `tests.json` `field_wing`) are flat in x at 4–15 px to ±4%, except the NRCBLONG x > 1920 bin (−7%). So the core and inner wings do not follow the halo dip.
    - This does not directly test the field PSF at the halo radii. At 15–29 px the field-wing bins are noisy (NRCBLONG −19% to +9%).
  - **Cause:** see the next section. The deficit is confined to saturated stars, and it is already present in the raw ramp. The SATURATION reffile is excluded. The mechanism is open.
- **Core area:** the area of the same exposure explains **67%** (a = 0.73 × (area − 1), r = 0.82). With f(x) added it reaches **70%**.
- **Self-constrained:** the exposure's own inner halo at 15–40 px predicts its 40–80 px halo with slope 0.90 (85% explained), and its 80–150 px halo with slope 0.49 (76%).
  - The radial shape is ~80% one mode, and that mode peaks at r ≈ 25–40 px.
  - Two free radial modes per exposure leave a 3–4.5% rms azimuthal mean at r = 20–100 px.
  - At the 30°-sector level 5–9% is left (split-half r = 0.81, against a formal noise that would allow 0.999). That residual is real azimuthal structure, or scene-sampling noise that `He` does not capture.

## Cause of the NRCBLONG x ≈ 400 deficit (#1013 follow-up)

| step | script | output |
|---|---|---|
| 6 | `perexp_cause.py` | the deficit and the a-vs-area slope by brightness; the SATURATION reffile threshold maps against f(x) (`cause_*.png`, `cause.json`) |
| 7 | `perexp_fieldstar_flux.py <prefix> <visit glob>...` | unsaturated field-star aperture flux vs detector x (`fieldstar.*`) |
| 8 | `perexp_ramp.py measure <rowdir> <saturation reffile> <cal key>...` then `analyze <prefix> <rowdir>` | the halo and the saturated-core area in the RAW ramp (`ramp_10678.*`, `ramp_2221.*`) |
| 9 | `RAMP_ZEROFRAME=1 perexp_ramp.py measure ...` | the same, with the `_uncal` ZEROFRAME (the first ~10 s read) as an extra group (`ramp_10678_zf.*`) |
| 10 | `perexp_column_map.py pxh.npz <prefix>` | 32-px column profile against the amplifier boundaries, 256-px detector map, date quartiles (`column_map.*`) |

- **Saturated stars only** (`cause.json`, NRCBLONG).
  - Brightness terciles: split by star-visit mean `A30`, 2,235–2,239 exposures each.
  - Halo: the deficit is −0.14 / −0.18 / −0.20. This is `column_feature`: the mean of the 128-px-binned median `a` over the bins centred at x = 320 and 448, minus its mean over bins centred at x < 150 or > 750.
  - Core area: the same contrast of `area_prof` is −0.18 / −0.24 / −0.24.
  - Field stars: 5,414 unsaturated stars give 25,456 star-exposure ratios, from 9 NRCBLONG visits (obs 042, 046, 061, 063, 069, 070, 078, 116, 126; listed in `fieldstar.json`). Their aperture flux (r = 3 px) is flat in x: every 128-px bin is within 0.990–1.002.
- **Not the SATURATION reffile** (`cause.json` `saturation_ref`, `jwst_nircam_saturation_0115`).
  - Its 128-px column medians vary 0.9% rms.
  - They correlate +0.65 with `a`, so the threshold is lowest where the halo is lowest. A lower threshold saturates *more* pixels, while the core area is *smaller* there.
- **Already in the raw ramp, so no calibration step causes it.** Step 8 works on the `_uncal` data, before superbias, linearity, dark and flat. The halo is the median group difference D_k at 15–80 px minus its value at 150–250 px; the area counts core pixels at or above the threshold in each group.
  - **Program 10678 (F480M):** 13 observations, the 12 with the most in-band saturated stars plus obs 061.
    - In-band / out-of-band (`ramp_10678.json` `split_band`): halo 0.81, 0.79, 0.75 for D1, D2, D3 (±0.02–0.03, 49 stars). The exposure-1–2 vs 3–6 comparison for stars whose exposures 1–2 are in the band (`band`) reads 0.80, 0.74, 0.72 (30 stars), with area 0.59 → 0.50.
    - Saturated area: 0.61 at group 0, falling to 0.54 at group 3.
    - Null band at x = 1300–1600: 0.97–1.00.
    - With the 10678 pattern (+385 px in x per step) a star is in the band almost only in exposures 1–2. The control, stars whose exposures 1–2 are *outside* the band, (`control`) reads 0.99–1.03 for the halo and 0.97–0.99 for the area, so this is a column effect and not an exposure-order one.
  - **Program 2221 (F405N / F410M / F466N, 24-point FULLBOX, 2022):** 96 frames, cores ≥20 px.
    - Halo 0.90 ± 0.03 (73 stars); area 0.73–0.75 (113 stars); null band 0.99–1.00.
    - So the effect is present in other LW filters and epochs, weaker for these smaller cores.
    - Measured with the threshold reffile the 2221 frames name, `jwst_nircam_saturation_0098`. A first run used the 10678 file (0115); the two NRCBLONG maps (all 96 frames are NRCBLONG) differ on 0.16% of pixels, evenly in and out of the band (0098/0115 = 1.0000 in every 128-px column bin). Rerun with 0098, every frame finds the same number of saturated cores, and one more null-band star passes the core-area cut (167 → 168). The halo moves 0.889 → 0.898 (0.3σ); the area, null band and exposure-order control are unchanged.
  - **The deficit grows along the ramp** in both the halo (0.81 → 0.75) and the area (0.61 → 0.54). Charge accumulation, not photon arrival, is implicated.
  - **The flat field** is only ~5% low here (rate/cal), so raw 0.78–0.81 corresponds to the ~0.83–0.85 seen in `cal`.
- **Not optical.** Unsaturated PSF wings at 4–15 px are flat in x (±4%).
- **Present from the first read** (step 9, obs 041/043/075/086, 24 frames). In the ZEROFRAME the in-band / out-of-band saturated-core area is already 0.64 ± 0.06 (19 stars; null band 1.00 ± 0.01; the exposure-1–2 control 0.96).
  It is 0.65 at group 0 and 0.56 at group 3, and the halo in G0 − ZEROFRAME is 0.80 ± 0.11.
  So almost all of the deficit is there in the first ~10 s; the growth along the ramp is second order.
  The exposure-order control is 0.957 ± 0.009 in the ZEROFRAME, against 0.97–0.98 in the later groups: a small first-read effect in exposures 1–2 (persistence, or a first-exposure reset effect) that does not depend on x. It is small next to the in-band 0.64.
- **Where it sits** (step 10, the 8,933-exposure table).
  - Not readout: the 32-px profile is a smooth trough from x ≈ 200 to ≈ 560, deepest at x ≈ 390 (core area −0.26, halo −0.20), with no step at the x = 512 amplifier boundary.
  - A full-height column stripe: in the 256-px map every row of the x = 256–512 column reads −0.09 to −0.26. It is deepest at mid-height (−0.23 to −0.26 for y = 256–1792) and weaker at the top and bottom edges (−0.15, −0.09). At y < 512 it spreads to x = 512–768 (−0.07, −0.12). Every other cell is within ±0.07, apart from the x < 256, y < 256 corner (+0.11).
  - Stable in time: −0.18 to −0.23 in each date quartile of 2026-09-11 to 09-21 (out of band +0.01). It is also in the 2022 program 2221 (step 8), so it has persisted for ~4 years.
  - The open mechanism is therefore a detector property of the x ≈ 250–550 columns that acts only at very high illumination: a saturated core is smaller and its halo fainter there from the first read, while unsaturated stars are flat.
- **LINEARITY reffile:** not inspected. The raw-ramp result makes it moot, because the deficit exists before linearity is applied.

## Effect on the production saturated-star fluxes

`production_flux_vs_x.py fit <workdir> <psf dir> <cal key>...` runs `saturated_star_finding.remove_saturated_stars` unchanged on each `_cal` frame. It uses the fovp1024 F480M grid, and has no ZEROFRAME anchoring because no `_ramp.fits` is available here. `analyze <prefix> <workdir>` then compares each star's `flux_fit` across its dithers (`production_flux_vs_x.*`).

Note that the catalog's `x_fit`/`y_fit` are positions inside the fit box; the detector position is `x_0`/`y_0`, which agrees with `skycoord_fit` through the GWCS to 0.0005 px.

- **Data:** obs 041, 061, 075 and 086 of 10678, 24 NRCBLONG frames, 16,397 catalog rows. 921 stars have dithers both in the band (x = 250–550) and outside it. "Outside" excludes both the band and the null band (±50 px).
- **The production flux is low in the band, and more so for larger cores.** Median `flux_fit` in band / outside, by the star's median `sat_area`:

  | `sat_area` | band | null (x = 1300–1600) | band / null |
  |---|---|---|---|
  | < 50 | 1.003 ± 0.004 (360 stars) | 1.011 ± 0.006 | 0.993 ± 0.007 |
  | 50–150 | 0.957 ± 0.004 (404 stars) | 1.000 ± 0.002 | 0.957 ± 0.004 |
  | 150–500 | 0.910 ± 0.005 (138 stars) | 0.990 ± 0.003 | 0.919 ± 0.006 |
  | ≥ 500 | 0.842 ± 0.025 (19 stars) | 0.984 ± 0.006 | 0.856 ± 0.026 |

  This table bins on the star's median `sat_area` over all its dithers. The depth model below uses the in-band detection's own area, which is smaller.
  The null band has a weak size trend of its own (1.011 → 0.984), so band / null (`band_over_null_by_sat_area`) isolates the column term.

  This is smaller than the ~20% halo deficit: the masked-core fit also uses the inner wings, which are flat in x.
- **Depth model** (robust fit, `depth_model`): in band / outside = 1 − 0.147 · max(0, log10(sat_area / 50)). Here `sat_area` is the core area *of the in-band detection*, which is what a per-detection correction has.
- **Shape** (`xshape32`, cores ≥ 100 px): a trough from x ≈ 190 to 600, 0.90–0.93 over x = 320–512. It matches the core-area stripe of step 10. Outside it the flux is within ±1.5%.
- **Proposed correction (not applied):** `flux_fit / (1 − 0.147 · max(0, log10(sat_area / 50)) · s(x))`.
  - s(x) is the `xshape32` trough normalized to 1 in the band. It is NRCBLONG only.
  - It is calibrated on F480M 10678 only.
  - The mechanism is still open (JWST-GC/data-qa#349), so it is an empirical, per-detector term.
