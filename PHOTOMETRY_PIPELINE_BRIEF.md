# Photometry pipeline — brief

Terse, process-focused summary of the PSF-photometry pipeline: what each step
does and the parameters it uses. For flags, file names, output trees, and the
distributed fan-out, see [`PHOTOMETRY_PIPELINE.md`](PHOTOMETRY_PIPELINE.md).

The pipeline is a sequence of `daofind → fit → residual → reseed` passes.
Detection runs on a progressively cleaner co-add (so sources hidden in one pass
surface in the next); **the fit always runs on the frames, never on the co-add.**
All numbers below are NIRCam-standard defaults; regime deltas are at the end.

## Per-frame recipe (every pass)

1. Find and subtract saturated stars → satstar model.
2. `daofind` on this pass's detection image → seed, unioned with the previous
   pass's vetted catalog.
3. Fit every seed: single-pass BASIC PSF photometry on the frame.
4. Post-fit QC: overshoot refit, non-positive-flux ban, dedup, satstar-wing /
   near-saturation rejection.
5. (Barrier) merge per-frame catalogs, vet extended emission, build the residual
   co-add + source-masked smoothed background for the next pass.

## Saturated stars

- **Find:** connected components of DQ-`SATURATED` pixels (plus `DO_NOT_USE` for
  truly-lost cores; cosmic-ray `JUMP_DET` handled).
- **Fit:** gridded WebbPSF model, `npsf=16`, `oversample=2`; field of view
  `fovp=512` (SW) / `1024` (LW) in-FOV, `2048` (SW) / `1024` (LW/MIRI) for
  off-FOV stars whose spikes bleed into the frame.
- **Keep-gate (NIRCam):** `qfit ≤ 5`, `snr ≥ 3`, `sidelobe ≥ −10 σ`, `ssr ≤ 1`
  — `ssr` applied only to low-confidence fits (`snr ≤ 10`); high-S/N good-qfit
  stars bypass it. **MIRI:** looser — `qfit ≤ 15`, `snr ≥ 2`, `sidelobe ≥ −40`,
  `ssr ≤ 2`.
- **MIRI seed gates** (reject emission blobs): `prominence ≥ 8`, `core ≥ 1000`,
  `concentration ≥ 1.3`, plus diffraction-spike merge.
- **Off-FOV stars:** large PSF grid + position prior + integer grid search
  (`radius 5 px`) + `model ≤ data` clamp at the 10th percentile + cross-frame
  flux reconciliation (pin runaways to the nearest-detector flux).
- **Frame-0 recovery (off by default):** ZEROFRAME core deblend and
  charge-migration rim de-inflation (`R × group0`, DQ dilation `3 px`); need the
  sibling ramp cube, else no-op.
- Point-source detections that land on a satstar are rejected: drop fits dimmer
  than the local satstar model (`ratio 1.0`, gated where `model > 3 σ`), and drop
  anything within `~1.5 × FWHM` of a hard-saturated pixel.

## Detection (`daofind`)

Threshold is deliberately permissive (global-minimum local noise → detect
everything); the **real** selection is a per-source **local-S/N** cut plus
roundness/sharpness bounds.

| pass | round | sharp | local-S/N | S/N filter |
|------|-------|-------|-----------|------------|
| m1 (discover) | ±1.0 | 0.30–1.40 | 5.0 | off |
| m2+ | ±0.3 | 0.50–1.00 | 3.0 | on |

The co-add-augmented seed (m3+) runs a second `daofind` on the detection co-add
with `round ≤ 0.5`, `sharp 0.4–1.2`, deduped at `0.5 × FWHM`. FWHM is per-filter
(F210M 2.30, F212N 2.34, F480M 2.57 px).

## Fit

Single-pass **BASIC** `PSFPhotometry` (not iterative — reseeding is across
passes): `LevMarLSQFitter`, `fit_shape = (5, 5)`, `aperture = 2 × FWHM`, a
`LocalBackground` annulus, and — with `--group` — joint fitting of sources closer
than `min_separation = 2 × FWHM` (use `~3 × FWHM` for blends).

Post-fit, in order:
- **Overshoot:** if rendered `model_peak > 1.2 × data_peak`, the free fit walked
  off the star → refit flux-only at the pinned seed position (closed-form, clamped
  ≥ 0).
- **Negative-flux ban:** `flux ≤ 0` dropped (a positive PSF cannot have a negative
  peak).
- Dedup, satstar-wing / near-saturation rejection.

## Passes

- **m12** (per-frame, raw): iter1 discovers unseeded; iter2 reseeds on the frame's
  own residual + iter1 catalog. Builds the satstar model.
- **m3:** detect on the raw data co-add, fit raw frames.
- **m4:** detect on m3's residual co-add, fit raw frames; builds the first
  source-masked background.
- **m5, m6:** detect on `residual − background`, fit **background-subtracted**
  frames; recompute the background each pass. m6 is the final per-filter pass.
- **m7** (multi-filter): seed from the cross-band merge of every filter's m6
  catalog (deduped, optionally requiring ≥2-filter confirmation), fit on m6
  background-subtracted frames.
- **m8:** no detection — **force-fit** flux at each cross-band merged position that
  is a non-saturated non-detection in some band (position pinned, flux-only), then
  dedup. Recovers non-detections / sets noise limits.

## Vetting (per-pass, after merge)

**NIRCam** keeps a source if it is star-like

```
(qfit ≤ 0.2) OR (peakSB > 20 × local_bkg) OR (snr ≥ 20 AND qfit < 0.4 AND solo)
```

**and** it clears `local-S/N ≥ 5`; then drops anything flagged overshoot.
**MIRI** vets on data-co-add `prominence = (core − annulus_median)/annulus_MAD ≥
threshold`.

## Regime deltas

| | structure-noise prune (x, y) | coarse-bg box | vet qfit_max | vet local-S/N | prominence gate |
|--|--|--|--|--|--|
| **NIRCam-std** | off (0, 0) | off | 0.2 | 5.0 | off |
| **Extended emission** (w51, sickle, wd2, ngc6334) | (1, 2) | off | 0.2 | 5.0 | off |
| **MIRI** | (5, 8) m12–m4, (3, 8) m5–m6 | 51 (raw passes) | 0.4 | 8 → 3 | 8 → 3 across m12→m6 |

Extended-emission handling auto-engages on those targets (`--extended-emission` /
`--no-extended-emission` to force). Grouping, frame-0 recovery, and cross-band
confirmation are opt-in everywhere.
</content>
