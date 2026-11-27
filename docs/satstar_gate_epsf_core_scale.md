# Satstar implied-peak gate on the ePSF-core PSF (`SATSTAR_GATE_EPSF_CORE_SCALE`)

Opt-in, default off.  With the switch on, the model-implied peak that the satstar
post-fit gate compares with half the severity floor is compared with a detector-specific
fraction of it, so that a star near the gate keeps the decision it had with the STPSF-only
PSF when the fit uses the hybrid PSF grid with the ePSF core (`PSF_EPSF_CORE_DIR`).

## What the gate does

`get_saturated_stars` in `saturated_star_finding.py` fits each satstar candidate with
the PSF grid and, after the fit, rejects it (`reject_reason = implied_peak_gate`) when

* the model-implied peak, `satstar_implied_peak`, is below
  `0.5 * SAT_SEVERITY_FLOOR[band]` (2500 MJy/sr in F480M, 2000 in F212N), and
* the observed peak of the star is not itself at or above that half floor.

A rejected star goes to the daophot channel (`cataloging._gate_reject_handoff_xy`).  The
floors were chosen with STPSF-only fits.

## Why the gate moves with the ePSF core

The hybrid grid (`epsf_hybrid.py`) replaces the STPSF core inside 10 px with the
empirical core, matched in flux per sub-pixel phase.  In F480M the core has a lower peak
per unit flux than STPSF, and the fit then takes a lower flux.  The implied peak of one
star is the product of the two ratios:

```
implied(core) / implied(STPSF) = [peak per unit flux, core / STPSF] x [flux_fit, core / STPSF]
```

Stars whose STPSF implied peak is within a few percent above the gate fall below it with
the core and change channel.  Example, o105 F480M, the star at pixel (1268.97, 632.68) in
the five `nrcblong` exposures 2 to 6 (observed peaks 1959, 1975, 1783, 2022, 1996):

| exposure | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| implied peak, production (STPSF only) | 2547 | 2560 | 2429 | 2506 | 2554 |
| implied peak, ePSF core | 2411 | 2439 | 2339 | 2409 | 2418 |

Production accepts four of the five (gate 2500); with the core all five are rejected and
the star goes to the daophot channel.  A neighbour of the star, 4.8 px away at
(1264.48, 634.44), is fitted in the same group there (exposure 6 only), the group fit moves
it by 0.06 px, and the near-saturation filter then drops the row, which leaves the
403 MJy/sr leftover at pixel (4452, 858) of the tile render.  The second step belongs to the
near-saturation distance rule and is the subject of its own switch.

## Measurement

Matched rows: frame by frame, mutual nearest neighbours within 0.05 arcsec between the
per-frame satstar tables (accepted plus rejected) of the production run (STPSF-only PSF,
no `EPSFSHA` in the table header) and a run of the same frames with the ePSF core.  "Near
the gate" is a production implied peak within 0.8 to 1.25 times the gate.  Rows need a
finite implied peak and a positive flux in both runs.

```python
ib, sep, _ = a['skycoord_fit'].match_to_catalog_sky(b['skycoord_fit'])
ia, _, _ = b['skycoord_fit'].match_to_catalog_sky(a['skycoord_fit'])
pairs = [(i, j) for i, j in enumerate(ib) if ia[j] == i and sep[i].arcsec < 0.05]
near = (imp_a > 0.8 * gate) & (imp_a < 1.25 * gate)
ratio = np.median(imp_b[near] / imp_a[near])           # the scale
reject_a = (imp_a < gate) & ~(obs_a >= gate)
reject_b = (imp_b < s * gate) & ~(obs_b >= gate)        # s = 1 is the unscaled gate
```

### F480M (gate 2500 MJy/sr), five runs pooled by detector

Runs: the o105 control chain (every opt-in switch off, ePSF core on) and the four tile
runs of the e2e5 opt stack (o045, o118, o130, o138).  Each run has six frames per
detector.

| detector | matched rows | near the gate | implied peak, run / production | peak per unit flux | flux | decisions that differ at s = 1.0 (accept to reject, reject to accept) | at the table scale |
|---|---|---|---|---|---|---|---|
| NRCA5 (`nrcalong`) | 11,282 | 2,863 | 0.9755 | 0.9866 | 0.9890 | 135, 8 | s = 0.976: 26, 26 |
| NRCB5 (`nrcblong`) | 13,892 | 2,932 | 0.9547 | 0.9707 | 0.9836 | 232, 2 | s = 0.955: 28, 14 |

The medians over the near-gate rows of the single runs:

| run | NRCA5 | NRCB5 |
|---|---|---|
| o105 control | 0.9776 | 0.9562 |
| o045 | 0.9766 | 0.9558 |
| o118 | 0.9749 | 0.9546 |
| o130 | 0.9728 | 0.9521 |
| o138 | 0.9716 | 0.9509 |

The ratio is a property of the detector: both factors come from the PSF grid, and the
near-gate medians of the five runs span 0.6 % (NRCA5) and 0.5 % (NRCB5).  One scale per
detector lowers the number of decisions that differ from production from 377 (143 + 234)
at s = 1.0 to 94 (52 + 42).  What remains is the star-to-star scatter of the ratio: the
16th to 84th percentile of the near-gate rows is 0.9679 to 0.9838 (NRCA5) and 0.9467 to
0.9628 (NRCB5), about +-0.8 %, which no single scale removes.  Of the 94, 40 are stars
that production rejected and the scaled gate accepts.

The scale for a star at the gate is the near-gate median, not the median over all matched
rows: the all-rows medians are 0.9693 (NRCA5) and 0.9532 (NRCB5), and the near-gate
medians 0.9755 and 0.9547.

### F212N (gate 2000 MJy/sr), same runs

The sign is the opposite one.  The core raises the implied peak of the near-gate rows by
0.2 to 1.5 % (median, by detector):

| detector | NRCA1 | NRCA2 | NRCA3 | NRCA4 | NRCB1 | NRCB2 | NRCB3 | NRCB4 |
|---|---|---|---|---|---|---|---|---|
| implied peak, run / production | 1.0060 | 1.0081 | 1.0017 | 1.0128 | 1.0059 | 1.0046 | 1.0127 | 1.0152 |

A star near the gate therefore does not drop out of the satstar channel through this
mechanism in F212N; the run accepts more of them than production does (pooled over the
eight detectors and five runs, 301 stars that production rejected and the run accepts, 153
that production accepted and the run rejects).  The table has no F212N entry.  `SATSTAR_GATE_EPSF_CORE_SCALE_VALUE` accepts a
scale above 1, but no value was measured or tested for it.

## The switch

`saturated_star_finding.py`:

* `SATSTAR_GATE_EPSF_CORE_SCALE=1` turns it on.  Off, `satstar_implied_peak_gate_rejects`
  evaluates the former expression and the cache key is that of a run that never set the
  variable.
* `SAT_GATE_EPSF_CORE_SCALE` holds the scale by `(band, detector)`: `('f480m', 'NRCA5')`
  0.976 and `('f480m', 'NRCB5')` 0.955 (the pooled medians above, rounded to the third
  decimal).  A band or detector without an entry keeps the unscaled gate.
* The scale applies to a PSF that carries the core (`epsf_core_sha256` in the grid
  meta).  A run without `PSF_EPSF_CORE_DIR`, or a detector with no core file, keeps the
  gate.
* `SATSTAR_GATE_EPSF_CORE_SCALE_VALUE=<x>` replaces the table with one positive scale for
  every band and detector (for a measurement at another value).  A value that is not a
  positive finite number raises `ValueError` naming the variable.
* The scale multiplies only the model-implied peak.  The observed-peak second chance
  (an observed peak at or above half the floor keeps the star) is data and stays at the
  unscaled half floor.
* `satstar_fit_switch_signature` adds the part `gs<scale>` to the cache key (`SATFITSW`)
  when the effective scale of the frame is not 1, so a cached satstar catalog written
  with another scale, or without the switch, is refit and not reused.  The columns
  `satstar_implied_peak`, `satstar_observed_peak` and `sat_severity_floor` keep their
  unscaled values.
* The fit-quality hand-off in `cataloging.py` that compares implied and observed peaks with
  the floor (`DAOPHOT_HANDOFF_FAINT_FIT_QUALITY`) stays unscaled; no o105 F480M row lies
  between the scaled and unscaled gate there.

## Limits of this change

* One scale per detector corrects the median, not the star-to-star scatter.  94 of
  25,174 matched rows (0.4 %), 40 of them in the accepting direction, still differ from
  the production decision.
* The scale was measured on the o105, o045, o118, o130 and o138 tiles of the
  gc-treasury 10678 F480M frames.  Another field, filter or core version needs the
  measurement again; the recipe above is all of it.
* It changes which stars the satstar channel owns, so a star near the gate can come back
  to the satstar channel with a flux that differs from the daophot one.  The tile test
  in the pull request description counts the consolidated satstar rows gained and lost.
