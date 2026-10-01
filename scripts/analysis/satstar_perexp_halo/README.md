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
- **Dither index:** explains ≤5% (per-dither medians ±2%), and so does a term shared by all stars of one exposure (−7%). This change is **not** a property of the exposure.
- **Dipole:** none (<1%).
- **Detector column (the pattern):** a free function f(x) per detector, on 64-px knots, explains **43%** cross-validated.
  - NRCBLONG: a ~25% deficit centred at x ≈ 400 px, about 300 px wide.
  - NRCALONG: a U shape, +30% within ~300 px of both x edges.
  - The dependence on detector row is weak.
  - The low-order Legendre "position" model reaches only 17%, because it cannot follow the NRCBLONG feature.
  - The saturated-core area traces the same f(x) independently.
  - Unsaturated field-star PSF wings at 4–15 px are flat in x to ±4%. So the column structure goes with *saturation*; it is not the optical PSF.
  - Cause unknown.
- **Core area:** the area of the same exposure explains **67%** (a = 0.73 × (area − 1), r = 0.82). With f(x) added it reaches **70%**.
- **Self-constrained:** the exposure's own inner halo at 15–40 px predicts its 40–80 px halo with slope 0.90 (85% explained), and its 80–150 px halo with slope 0.49 (76%).
  - The radial shape is ~80% one mode, and that mode peaks at r ≈ 25–40 px.
  - Two free radial modes per exposure leave a 3–4.5% rms azimuthal mean at r = 20–100 px.
  - At the 30°-sector level 5–9% is left (split-half r = 0.81, against a formal noise that would allow 0.999). That residual is real azimuthal structure, or scene-sampling noise that `He` does not capture.
