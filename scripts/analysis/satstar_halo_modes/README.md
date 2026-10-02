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

## Production hook: untested at the production box

`SATSTAR_HALO_MODES=1` fits out to the photutils box half-width (40.5 px at
`pad=81`) unless `SATSTAR_HALO_MODES_RMAX` overrides it, and the ±`pad` cutout
still limits it. The evidence above shows a scatter gain only at r ≤ 200 px (none
at r ≤ 100), and a saturated core wider than ~rmax/1.3 leaves no halo range, so
the ratio is NaN for such stars — about half of these 7 at the default box. The
ratio is also applied to a `flux_fit` from photutils' own box and local
background, while the halo-mode fits use a circle and a constant background.
Treat `flux_fit_halomodes` as a diagnostic until it has been checked at the
radius it will be run with.
