# Saturated-star flux with a free per-exposure halo (#1013 follow-up)

The LW halo of a saturated star changes from one exposure to the next by ~10% rms
(#1013), but its diffraction spikes do not (#993). A masked-core PSF fit gets its
amplitude from the wings, so its flux follows the halo.
`jwst_gc_pipeline.photometry.satstar_halo_modes` fits

    m = F·P + F·Σ_k c_k b_k(r)·P̄(r) + B

where:

- P̄ is the model's azimuthal-median (smooth) halo.
- b_k are log-r hats.

Inside the halo range, F is therefore set by the spikes.

Production hook: `SATSTAR_HALO_MODES=1` makes `get_saturated_stars` record
`halomodes_ratio` and `flux_fit_halomodes = flux_fit × ratio`. It is opt-in and
record-only, and `flux_fit` itself is unchanged.

## Reproduce

1. `make_psf.py x y psf_<star>_e<k>.npy` once per exposure. The detector position
   is the cutout's `X0+xt`, `Y0+yt`.
2. `halo_photometry.py <cutout dir> <psf dir> <out> tgt trn_116 ...`.
   - The cutouts come from `satstar_poisson/cutout.py`.
   - `--replot <out>` remakes the figure from the json.

## Result (7 saturated F480M stars, 10678, all on NRCBLONG; 40 exposures)

`docs/evidence/satstar_halo_modes/halo_photometry.png`

| | standard F·P+B | halo modes |
|---|---|---|
| dither rms, median of the 7 stars, r ≤ 200 px | 4.9% | 2.7% |
| dither rms, median, r ≤ 100 px | 3.0% | 2.8% |
| mean flux at r ≤ 100 px / at r ≤ 200 px, median (range) | 0.88 (0.79–0.95) | 0.96 (0.94–0.99) |
| flux relative to halo modes, r ≤ 200 px | 1.21–1.31 | 1 |

- The halo-mode flux is ~3× less sensitive to the fit radius. The standard fit
  grows by ~12% (up to 21%) between r ≤ 100 and r ≤ 200 px, because the real halo
  at 80–200 px is 2–4× STPSF's.
- The halo light the fit absorbs, summed over the *fitted* pixels, is 0.65% of F
  (median; at most 1.8%). The flux changes by much more than that because the
  standard fit scales the whole PSF, core included, to match the halo.
  (A hat lying mostly under the masked core is constrained only by its few
  unmasked pixels, so its coefficient can be large; integrating it over the
  masked annulus would extrapolate, which an earlier version of this diagnostic
  did.)
- Field-star pixels are clipped at 5 × the residuals' robust (MAD) scatter, not
  at 5 formal σ: the real halos fit to χ²/pix ≈ 8, and an absolute cut removed
  most of the good pixels.
- **Not validated in absolute terms.** On bright *unsaturated* stars in these
  cutouts (≤10⁴ e⁻), masking even r < 6 px already leaves the standard fit
  unconstrained, so there is no truth set at the relevant radii. The halo-mode
  flux is right only to the extent that STPSF's spikes are, and its residual
  r ≤ 100 vs 200 px drift (~4%) bounds that.

## Production hook: no gain at the production radius

`SATSTAR_HALO_MODES=1` fits out to the photutils box half-width (40.5 px at
`pad=81`) unless `SATSTAR_HALO_MODES_RMAX` overrides it, and the ±`pad` cutout
still limits it. A saturated core wider than ~rmax/1.3 leaves no halo range, so
the ratio is NaN for the stars above (about half of these 7 at the default box).

`production_radius.py` checks the stars the hook actually sees: moderately
saturated ones, with 5–1000 saturated pixels (core radius ~1–18 px). It uses
every such star in the 6 `_cal` frames of 10678 obs 061 NRCBLONG and refines its
position on the standard fit's χ². The PSF is STPSF at the nearest node of a
4×4 grid, and both models are fitted on the same pixels with the hook's knots
(`satstar_halo_knots`) (`docs/evidence/satstar_halo_modes/production_radius_o061.*`):

| rmax | stars | dither rms, standard | dither rms, halo modes | stars better with halo modes | F_halo / F_standard (16–84%) |
|---|---|---|---|---|---|
| 40.5 px | 215 | 12.3% | 13.0% | 40% | 0.955 (0.87–1.07) |
| 60 px | 206 | 12.4% | 12.4% | 43% | 0.974 (0.86–1.08) |

At the production radius the halo modes do **not** reduce the dither scatter.
They move each flux by ~10% in either direction, with a −3 to −5% median.

- **What does improve:** only stars with a core of ≳100 saturated pixels trend
  to the 0.85–0.9 ratio of the large stars above.
- **The baseline:** the 12% baseline is this simplified fit's (no neighbours,
  one PSF node), not production's. The comparison is like-for-like.
- **Recommendation:** leave `SATSTAR_HALO_MODES` off. Use the halo modes only
  for strongly saturated stars fitted to r ≳ 150–200 px.

## Wide radius on production catalogs (r ≤ 200 px)

`wide_radius_production.py fit <catalog dir> <psf grid> <out npz> <cal key>...` starts from the production `_satstar_catalog.fits` of `satstar_perexp_halo/production_flux_vs_x.py`. For every star with ≥ 50 saturated px it cuts a 401 × 401 px stamp of the `_cal` frame around the fitted detector position (`x_0`, `y_0`). It evaluates the production PSF grid there, masks DQ DO_NOT_USE|SATURATED dilated by 3 px, and fits F·P+B (`F_S`) and the halo modes (`F_H`, knots from `satstar_halo_knots`, r_core = √(sat_area/π) + 3) on the same pixels. `analyze` links the dithers of each star (`docs/evidence/satstar_halo_modes/wide_radius.*`).

Data: 10678 obs 041, 061, 075 and 086, 24 NRCBLONG frames, 9,897 rows, 1,157 stars with ≥ 4 dithers.

| sat_area | stars | dither rms, production flux_fit | F_S | F_H | stars where F_H beats flux_fit | F_H / flux_fit |
|---|---|---|---|---|---|---|
| 50–300 | 990 | 3.8% | 7.0% | 5.9% | 32% | 0.82 |
| 300–1000 | 151 | 4.2% | 5.1% | 3.8% | 57% | 0.84 |
| 1000–3000 | 4 | 3.7% | 3.1% | 3.3% | 3 of 4 | 0.97 |

**The column deficit** (`column`). This is the real gain. It uses the per-star median in x-region / outside, where outside excludes both regions ±50 px.

| | flux_fit | F_S | F_H |
|---|---|---|---|
| band x = 250–550, 50–300 px | 0.940 ± 0.004 | 0.946 ± 0.006 | 0.959 ± 0.004 |
| null x = 1300–1600, 50–300 px | 0.995 | 0.992 | 0.995 |
| **band, ≥ 300 px** (54 stars) | **0.884 ± 0.008** | 0.889 ± 0.009 | **0.982 ± 0.006** |
| null, ≥ 300 px | 0.991 | 0.989 | 1.004 |

- **What the column test shows:** for cores ≥ 300 px, the wide halo-mode flux all but removes the NRCBLONG column deficit that the production flux carries (`satstar_perexp_halo`, data-qa#349). This fits the model: the deficit lives in the halo, and the halo terms absorb the halo while the spikes set F. Below 300 px the 200-px stamp is dominated by crowding: the dither scatter gets worse (5.9% against 3.8%) and the deficit is only partly removed.
- **Absolute scale:** F_H is ~0.84 × flux_fit for these cores. A halo-defined amplitude (production) and a spike-defined one disagree by that much because the real halo is brighter than STPSF's. Neither is validated in absolute terms (see above), so F_H is recorded beside `flux_fit`, never in place of it.
- **Production hook:** `SATSTAR_HALO_MODES_WIDE=1` records `halomodes_wide_ratio` = F_H / F_S on the full-frame stamp, and `flux_fit_halomodes_wide = flux_fit × ratio`, for cores ≥ 300 px (`SATSTAR_HALO_MODES_WIDE_AREA_MIN`). Smaller cores get NaN. It uses `satstar_halo_mode_ratios_wide`, evaluated at DETECTOR coordinates (#1055). It is opt-in and record-only.
- **End-to-end check:** I ran `remove_saturated_stars` with `SATSTAR_HALO_MODES_WIDE=1` on 10678 obs 061 exposure 1 (855 saturated sources).
  - `flux_fit` is identical to the run without it.
  - The 45 sources with ≥ 300 px get a ratio and every other source gets NaN.
  - The median ratio of 0.837 (16–84%: 0.715–0.975) matches the standalone script on the same frame (0.838).
