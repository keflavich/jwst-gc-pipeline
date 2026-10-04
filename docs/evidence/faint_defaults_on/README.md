# Faint-star options on by default (AUTO)

Two faint-star options that were opt-in become AUTO.  Each is on for a
star-dominated NIRCam field and off on an extended-emission target
(`_is_extended_emission`: `--extended-emission`, or the target list
w51/sickle/wd2/ngc6334) and for a MIRI filter.

| option | before | AUTO, star-dominated NIRCam | AUTO, extended emission | AUTO, MIRI filter |
|---|---|---|---|---|
| `manual_m7_seed_own_band` (#1015: each band's m7 seed adds the band's own m6 vetted sources and its m6-residual detections) | off | on | off | off (all-MIRI runs drop m7) |
| `manual_seed_round_loose_max` (#1020: residual seeds with roundness up to ±0.8 pass when their annulus prominence is ≥ 5) | 0 | 0.8 | 0 | 0 |

`--manual-m7-seed-own-band` / `--no-manual-m7-seed-own-band` and an explicit
`--manual-seed-round-loose-max` override AUTO.

## Reference-field results

The four reference fields and their thresholds are described in
`docs/evidence/faint_reference_fields`.  All numbers below are phase m7,
inner box, pooled over the clean run and the ten injection runs of each
field.

### Calibration runs: `int3` and `int3on` (code 94033cf0)

`int3` is the integration branch at its shipped defaults (both options
off).  `int3on` is the same code with own-band on for all four fields and the
loose window at 0.8 on `superdense`, `dense_bright` and `dark`.  On
`bright_modest` (W51) `int3on` has own-band on and the loose window off.
The thresholds in `fields.yaml` were calibrated on `int3on`.

**superdense (NSC, F212N)**

| variant | S/N 40-80 | S/N 80-160 | S/N 160-320 | S/N 320-640 | resid excess /as² (+/−) | over-subtracted /as² (n) | bias mag (n) |
|---|---|---|---|---|---|---|---|
| int3 | 1/51 | 5/64 | 18/74 | 26/51 | 1.66 (27/3) | 1.25 (18) | -0.018 ± 0.023 (50) |
| int3on | 1/51 | 11/64 | 29/74 | 29/51 | 1.52 (28/6) | 1.25 (18) | +0.012 ± 0.025 (70) |

**dense_bright (Sgr B2, F187N)**

| variant | S/N 5-10 | S/N 10-20 | S/N 20-40 | S/N 40-80 | resid excess /as² (+/−) | over-subtracted /as² (n) | bias mag (n) |
|---|---|---|---|---|---|---|---|
| int3 | 0/49 | 1/69 | 12/58 | 32/64 | 2.29 (33/0) | 1.18 (17) | +0.032 ± 0.015 (44) |
| int3on | 0/49 | 1/69 | 15/58 | 36/64 | 2.15 (31/0) | 0.90 (13) | +0.049 ± 0.013 (51) |

**bright_modest (W51, F187N)**

| variant | S/N 5-10 | S/N 10-20 | S/N 20-40 | S/N 40-80 | resid excess /as² (+/−) | over-subtracted /as² (n) | knots cataloged | bias mag (n) |
|---|---|---|---|---|---|---|---|---|
| int3 | 0/53 | 0/56 | 3/68 | 34/62 | 0.21 (4/1) | 0.07 (1) | 1/4 | -0.130 ± 0.040 (37) |
| int3on (own-band only) | 0/53 | 0/56 | 3/68 | 35/62 | 0.48 (11/4) | 0.07 (1) | 1/4 | -0.135 ± 0.039 (38) |

**dark (Brick, F182M)**

| variant | S/N 5-10 | S/N 10-20 | S/N 20-40 | S/N 40-80 | resid excess /as² (+/−) | over-subtracted /as² (n) | labels | bias mag (n) |
|---|---|---|---|---|---|---|---|---|
| int3 | 0/62 | 18/55 | 42/61 | 47/62 | 1.32 (21/2) | 1.11 (16) | 15/33 | +0.031 ± 0.009 (89) |
| int3on | 2/62 | 29/55 | 45/61 | 48/62 | 0.97 (16/2) | 1.04 (15) | 19/33 | +0.041 ± 0.010 (93) |

`int3` fails `dark` (completeness 10-20 = 18/55 < 0.35; residual excess 1.32 >
1.25).  `int3on` passes all four fields.

Injected stars gained / lost, `int3on` vs `int3` (exact sign test):

| field | bin 1 | bin 2 | bin 3 | bin 4 |
|---|---|---|---|---|
| `superdense` | S/N 40-80: +0/−0 | S/N 80-160: +6/−0 (p=0.031) | S/N 160-320: +11/−0 (p=0.00098) | S/N 320-640: +3/−0 (p=0.25) |
| `dense_bright` | S/N 5-10: +0/−0 | S/N 10-20: +0/−0 | S/N 20-40: +4/−1 (p=0.38) | S/N 40-80: +5/−1 (p=0.22) |
| `bright_modest` | S/N 5-10: +0/−0 | S/N 10-20: +0/−0 | S/N 20-40: +0/−0 | S/N 40-80: +1/−0 (p=1) |
| `dark` | S/N 5-10: +2/−0 (p=0.5) | S/N 10-20: +12/−1 (p=0.0034) | S/N 20-40: +4/−1 (p=0.38) | S/N 40-80: +2/−1 (p=1) |

### Same-commit runs: `defoff`, `defown`, `defon` (code d1ea6b73)

All three ran this branch's code (d1ea6b73, clean checkout).  `defon` is
the branch at its defaults (AUTO).  `defoff` adds
`--no-manual-m7-seed-own-band --manual-seed-round-loose-max=0` on all four
fields, which is the behavior before this PR.  `defown` adds
`--manual-seed-round-loose-max=0` on the three star fields, so own-band is
on and the loose window off; `defon` vs `defown` isolates the loose window.
`JWST_GC_REFFIELD_VARIANT=defon pytest -k test_reference_field_passes`
passes all four fields (4 passed).

| field | variant | injected completeness per S/N bin | resid excess /as² | over-subtracted /as² | bias mag | labels | result |
|---|---|---|---|---|---|---|---|
| `superdense` | defoff | 1/51, 5/64, 18/74, 26/51 | 1.59 | 1.25 | −0.030 | | pass |
| | defown | 1/51, 11/64, 29/74, 29/51 | 1.39 | 1.18 | −0.004 | | pass |
| | defon | 1/51, 11/64, 30/74, 29/51 | 1.32 | 1.11 | −0.007 | | pass |
| `dense_bright` | defoff | 0/49, 1/69, 12/58, 32/64 | 2.35 | 1.25 | +0.032 | | pass |
| | defown | 0/49, 2/69, 14/58, 36/64 | 1.73 | 1.25 | +0.046 | | pass |
| | defon | 0/49, 1/69, 15/58, 37/64 | 2.08 | 0.90 | +0.052 | | pass |
| `bright_modest` | defoff | 0/53, 0/56, 3/68, 34/62 | 0.21 | 0.07 | −0.130 | knots 1/4 | pass |
| | defon | 0/53, 0/56, 3/68, 34/62 | 0.21 | 0.07 | −0.130 | knots 1/4 | pass |
| `dark` | defoff | 0/62, 18/55, 42/61, 47/62 | 1.32 | 1.11 | +0.031 | 15/33 | **fail** |
| | defown | 2/62, 28/55, 45/61, 49/62 | 0.90 | 0.90 | +0.041 | 19/33 | pass |
| | defon | 2/62, 29/55, 45/61, 48/62 | 0.97 | 0.90 | +0.041 | 19/33 | pass |

The completeness bins are S/N 40–80, 80–160, 160–320 and 320–640 on
`superdense` and S/N 5–10, 10–20, 20–40 and 40–80 on the other fields.
`defoff` fails `dark` on the same two thresholds as `int3`.

Injected stars gained / lost (exact sign test):

| field | `defon` vs `defoff` | `defon` vs `defown` |
|---|---|---|
| `superdense` | 80–160: +6/−0 (p=0.031); 160–320: +12/−0 (p=0.00049); 320–640: +3/−0 (p=0.25) | 160–320: +1/−0; others +0/−0 |
| `dense_bright` | 20–40: +4/−1 (p=0.38); 40–80: +5/−0 (p=0.062) | 10–20: +0/−1; 20–40: +1/−0; 40–80: +2/−1 |
| `bright_modest` | +0/−0 in every bin | (not run) |
| `dark` | 5–10: +2/−0 (p=0.5); 10–20: +12/−1 (p=0.0034); 20–40: +4/−1 (p=0.38); 40–80: +2/−1 (p=1) | 10–20: +1/−0; 40–80: +0/−1 |

- **AUTO wiring.**  On `bright_modest` (W51, an extended-emission target)
  `defon` and `defoff` write no own-band m7 seed file, their cross-band
  seed catalogs are identical, and their m7 vetted catalogs have the same
  rows in all 11 runs.  Positions agree to 10⁻⁸″ and fluxes to 2 × 10⁻⁷
  (relative); the largest relative difference in any column is 10⁻⁴, in
  the local background scatter (`mean_modelsub_bkg_std`).  These are
  floating-point rounding differences between two runs.
- **Own-band carries the gain.**  `defown` reproduces the `defon` gains
  against `defoff`, and `defon` vs `defown` moves at most one injected star
  per bin (every p = 1).  The loose window changes the residual metrics:
  `superdense` excess 1.39 → 1.32 and over-subtracted 1.18 → 1.11;
  `dense_bright` over-subtracted 1.25 → 0.90 and excess 1.73 → 2.08 (seed
  spread ±0.19); `dark` excess 0.90 → 0.97.
- **Ring companions.**  `defown` has one faint source 1.5–4.5 px from a
  bright star on the `dense_bright` clean run (1.4 expected by chance);
  `defoff` and `defon` have none.  `ring_ratio` carries no threshold.
- **Calibration runs agree.**  `defoff` matches `int3` and `defon` matches
  `int3on` to within a few injected stars per field; the remaining
  differences come from the code between 94033cf0 and d1ea6b73.

## Figures

`figures/defoff_vs_defon/<field>_<filter>_m7_s<seed>.png` (same code, this
PR's defaults against both options off) and
`figures/int3_vs_int3on/<field>_<filter>_m7_s<seed>.png` (calibration
runs), seeds 0 (clean) and 1 (injection).  Columns: data with the base
catalog, base residual, data with the proposed catalog, proposed residual,
and base − proposed residual.  The top row is the 3.9″ field; rows A–D are 1.2″ cutouts placed where
the two catalogs differ most.  Green circles are proposed-only
sources, red boxes base-only sources, yellow crosses injected stars and
orange crosses over-subtracted cores.  Each row's residual stretch is ±5σ of
the base residual.

- `superdense` and `dark`: most green sources sit on a compact positive
  residual in the base column that the proposed fit removes; in
  `superdense` seed 1 three of them are injected stars (cutouts A and B).
- `dense_bright`: the same, and over-subtracted cores fall from 18 to 12
  (`defoff_vs_defon` seed 1) or 17 to 12 (`int3_vs_int3on` seed 1).  Next
  to the bright star in cutout A some of the new sources carry
  over-subtracted marks themselves.
- `bright_modest`, `defoff_vs_defon`: identical catalogs and a blank
  difference column (AUTO turns both options off).
- `bright_modest`, `int3_vs_int3on`: one `int3on`-only source (cutout A)
  with little signal in the data, and dark rings at about ten other places
  in the difference column.  Those are background holes (next section).

## Background holes at dropped m7 seeds

Every m7 seed position is masked out of the m7 smoothed background.  An
own-band seed that vetting then drops leaves a hole in the background map,
and `_residual_snr` (the residual excess metric) subtracts that map.
`scripts/bg_holes.py` takes the seeds own-band adds at m7 in the inner box
and reads the smoothed background of both variants at each one (`int3on` −
`int3`).  σ is the pixel scatter (1.4826 MAD) of the `int3` m7 residual
mosaic in the inner box.

| field | added m7 seeds per run | kept in final catalog | dropped: bg difference, median (p10) | dropped, in σ: median (p10) | random positions: median |
|---|---|---|---|---|---|
| `superdense` | 123 | 947/1355 | −25.2 (−311) | −0.96 (−11.8) | −0.0 |
| `dense_bright` | 128 | 839/1409 | −0.7 (−2.1) | −1.43 (−4.0) | −0.0 |
| `bright_modest` | 19 | 74/212 | −22.5 (−37) | −0.51 (−0.84) | +0.0 |
| `dark` | 84 | 791/926 | −0.8 (−2.5) | −2.65 (−8.5) | +0.0 |

The same-commit pair `defoff` → `defon` (`defon_bg_holes_<field>.json`;
`defon` also has the loose window on) gives the same picture on the star
fields:

| field | added m7 seeds per run | kept in final catalog | dropped: bg difference, median (p10) | dropped, in σ: median (p10) | random positions: median |
|---|---|---|---|---|---|
| `superdense` | 123 | 930/1349 | −22.7 (−253) | −0.87 (−9.7) | −0.0 |
| `dense_bright` | 127 | 845/1393 | −0.7 (−2.1) | −1.43 (−4.1) | −0.0 |
| `dark` | 84 | 789/919 | −0.9 (−2.7) | −2.90 (−8.9) | +0.0 |

On `bright_modest` `defon` adds no m7 seeds.

Holes appear on every field, and in units of the residual scatter they are
deeper on the star fields than on W51.  The fields differ in what the union
buys.  On the star fields own-band gains injected stars and the net
residual excess falls.  On
W51 it gains one injected star, leaves the knot count at 1/4, keeps 74 of 212
added seeds and raises the residual excess from 0.21 to 0.48 per arcsec².  A
hole at an unfitted star leaves the star's light in the residual, which is
what the residual should show.  A hole at an emission knot moves part of the
knot from the background into the residual.  AUTO turns own-band off on
extended-emission targets for that reason.

## Loose-only seeds (`int3on`, `defon`)

`scripts/loose_survival.py` follows every new i2d residual seed of the scored
band (all phases, deduplicated at one pixel) inside the inner box.  A seed
is loose-only when every detection of it was admitted only by the loose
window.

| field | seeds | distinct new i2d seeds | reach m7 vetted | at an injected star (injection runs, any fate) | survivors at an injected star (chance) |
|---|---|---|---|---|---|
| `superdense` | loose | 113 | 15 (0.13) | 9/104 (0.087) | 1/14 (0.07) |
| | tight | 610 | 44 (0.07) | 27/553 (0.049) | 3/40 (0.20) |
| `dense_bright` | loose | 229 | 18 (0.08) | 20/209 (0.096) | 2/18 (0.09) |
| | tight | 1029 | 91 (0.09) | 53/935 (0.057) | 8/83 (0.42) |
| `dark` | loose | 101 | 16 (0.16) | 11/93 (0.118) | 3/15 (0.08) |
| | tight | 921 | 125 (0.14) | 57/835 (0.068) | 16/112 (0.56) |

Pooled over the three fields:

- reach m7 vetted: loose 49/443 (0.111), tight 260/2560 (0.102), Fisher p = 0.55;
- at an injected star, any fate: loose 40/406 (0.099), tight 137/2323 (0.059), p = 0.0043;
- survivors at an injected star: loose 6/47 (0.128), tight 27/235 (0.115), p = 0.8.

`defon` (`defon_loose_survival.json`) gives the same pooled rates:

- reach m7 vetted: loose 46/421 (0.109), tight 256/2518 (0.102), Fisher p = 0.66;
- at an injected star, any fate: loose 38/388 (0.098), tight 131/2286 (0.057), p = 0.0045;
- survivors at an injected star: loose 6/44 (0.136), tight 26/233 (0.112), p = 0.61.

Loose-only seeds reach the final catalog at the same rate as tight seeds,
and those that do land on injected stars at a similar rate.  The last
comparison has 47 loose survivors and cannot separate rates that differ by
less than about a factor of two.

## Caveats

- **Own-band realness.**  On the full Brick frame the restored own-band
  sources are confirmed by an independent visit 0.4–0.8× as often as the m6
  vetted sources m7 already has (`docs/evidence/faint_m7_seed_union`).
  The reference fields measure completeness on injected stars and residual
  structure; they do not measure that realness ratio.
- **Loose window on bright background.**  On the full Brick and Sgr B2 frames
  29–31% of the loose-only seeds sit in the brightest tenth of the background,
  and they cluster at diffraction-spike position angles
  (`docs/evidence/faint_seed_roundness`).  Each star-field reference field
  covers 3.9″ and does not sample those full-frame distributions.
- **Sgr B2.**  `sgrb2` is not an extended-emission target, so both options
  are on there.  `dense_bright` (in the Sgr B2 envelope) gains stars and
  loses over-subtracted cores, and also has background holes of −1.4σ
  (median) at dropped seeds.
- **Injection.**  Injected stars are drawn from the same PSF model the fit
  uses, so completeness on them is an upper bound for real stars of the same
  flux.
- **Residual excess depends on the background mask.**  The metric subtracts
  the m7 smoothed background, so a change in which positions are masked
  changes it without any change in the residual image itself.
- **MIRI.**  Neither option was measured on MIRI; AUTO leaves both off there.

## Reproduce

```
# scores and paired comparison (evaluate.py writes the JSON used by --baseline)
python -m jwst_gc_pipeline.photometry.reference_fields.evaluate --variant int3 --json int3.json
python -m jwst_gc_pipeline.photometry.reference_fields.evaluate --variant int3on --json int3on.json \
    --baseline int3.json
# figures
python -m jwst_gc_pipeline.photometry.reference_fields.figures --variant int3on --base int3 \
    --phase m7 --seeds 0,1 --out figures/int3_vs_int3on/
# loose-only seed fate and background holes
python scripts/loose_survival.py int3on int3on_loose_survival.json
python scripts/bg_holes.py <field> int3 int3on int3on_bg_holes_<field>.json

# same-commit runs (from a checkout of this branch; the runs go to SLURM)
python -m jwst_gc_pipeline.photometry.reference_fields.run --variant defon --submit
python -m jwst_gc_pipeline.photometry.reference_fields.run --variant defoff --submit \
    '--extra=--no-manual-m7-seed-own-band --manual-seed-round-loose-max=0'
python -m jwst_gc_pipeline.photometry.reference_fields.run --variant defown --submit \
    --fields superdense,dense_bright,dark '--extra=--manual-seed-round-loose-max=0'
python -m jwst_gc_pipeline.photometry.reference_fields.evaluate --variant defoff --json defoff.json
python -m jwst_gc_pipeline.photometry.reference_fields.evaluate --variant defon --json defon.json \
    --baseline defoff.json
python -m jwst_gc_pipeline.photometry.reference_fields.figures --variant defon --base defoff \
    --phase m7 --seeds 0,1 --out figures/defoff_vs_defon/
python scripts/loose_survival.py defon defon_loose_survival.json
python scripts/bg_holes.py <field> defoff defon defon_bg_holes_<field>.json
JWST_GC_REFFIELD_VARIANT=defon python -m pytest \
    jwst_gc_pipeline/photometry/tests/test_reference_fields.py -k test_reference_field_passes
```

The evaluate outputs are `reffield_<variant>.json` here.
