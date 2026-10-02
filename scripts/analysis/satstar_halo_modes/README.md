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
| dither rms, median of the 7 stars, r ≤ 200 px | 4.6% | 2.9% |
| dither rms, median, r ≤ 100 px | 2.8% | 2.8% |
| mean flux at r ≤ 100 px / at r ≤ 200 px | 0.87 (0.83–0.95) | 0.97 (0.94–0.99) |
| flux relative to halo modes, r ≤ 200 px | 1.20–1.26 | 1 |

- The halo-mode flux is ~4× less sensitive to the fit radius. The standard fit
  grows by ~20% between r ≤ 100 and r ≤ 200 px, because the real halo at
  80–200 px is 2–4× STPSF's.
- The halo light that the fit absorbs is only 1–3% of F per exposure. The flux
  changes by much more than that because the standard fit scales the whole PSF,
  core included, to match the halo.
- **Not validated in absolute terms.** On bright *unsaturated* stars in these
  cutouts (≤10⁴ e⁻), masking even r < 6 px already leaves the standard fit
  unconstrained, so there is no truth set at the relevant radii. The halo-mode
  flux is right only to the extent that STPSF's spikes are, and its residual
  r ≤ 100 vs 200 px drift (~4%) bounds that.
