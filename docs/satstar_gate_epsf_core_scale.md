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
near-saturation distance rule and is the subject of its own switch
(`near_saturation_continuous_distance.md`); the replays below show that this switch alone
does not bring the neighbour back, and that it restores the channel of the star.

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

## One-frame replays

The m7 pass of two o105 F480M frames was replayed with the integration code plus the two
switches (ePSF core on, every other setting that of the end-to-end control run), with four
arms: no switch (control), this switch alone, the near-saturation distance switch alone
(`NEARSAT_CONTINUOUS_DIST`) and both.  The rows are those of the final daophot table (after
the near-saturation filter) within 3 px of the position, and the satstar rows within
1 arcsec of the star.

Frame `nrcblong` exposure 6: the F5 star (1268.97, 632.68) and its neighbour at (1264.5, 634.4).

| arm | star, satstar channel | star, daophot row | neighbour, daophot row (x_fit, flux, flags) |
|---|---|---|---|
| production (STPSF only) | accepted: implied peak 2553.5, observed peak 1996.0, flux 29125.9 | none | 1264.4819, 5232.8, 1 |
| control | rejected by the gate: implied peak 2418.3, observed peak 1996.0, flux 28636.7 | flux 23597.7 | none |
| this switch | accepted, the same fit (flux 28636.7) | none | none |
| near-saturation switch | rejected, as in the control | flux 23597.7 | 1264.5422, 5360.7, 0 |
| both | accepted | none | 1264.5107, 5291.3, 1 |

Frame `nrcalong` exposure 1: the F4 row at (1834.5, 1633.2).

| arm | daophot row (x_fit, y_fit, flux, flags) |
|---|---|
| production | 1834.5003, 1633.1601, 13850.3, 1 |
| control | none |
| this switch | none |
| near-saturation switch | 1834.4991, 1633.1712, 13502.2, 1 |
| both | 1834.4991, 1633.1712, 13502.2, 1 |

The switch changes which channel owns the star and nothing else about its fit: the satstar
flux is 28636.7 with and without it, and the daophot fit that the control makes of the same
star gives 23597.7.  It does not return the neighbour: in the arm with this switch alone the
neighbour has no row, and in the arm with both it is present at x_fit 1264.5107, a position
that rounds to the pixel next to the saturated blob and that the rounded-pixel rule would
drop.  Production has that row at 1264.4819.  The F4 row does not return with this switch
either.

The scale moves whole-frame decisions as the measurement says.  Satstar catalog rows
(rejected rows) in the control and with this switch: `nrcblong` exposure 6, 513 (216) and
519 (210); `nrcalong` exposure 1, 438 (127) and 445 (120).  Against the production satstar
catalog of the frame (519 and 444 rows; positions matched within 0.05 arcsec) the control
lacks 10 and 9 production rows and has 4 and 3 rows that production does not; with this
switch the numbers are 4 and 4 (`nrcblong`), 4 and 5 (`nrcalong`).  A satstar-only replay of
exposure 2 of the same detector accepts the F5 star (implied peak 2410.9 against 2387.5,
the same flux 28504.9) and grows the catalog from 510 (233 rejected) to 521 (222) rows; its
cache key gains the part `gs0.955`.  The control arm reproduces the daophot table of the
earlier control replay of each frame (54060 and 52982 rows; `x_fit`, `y_fit`, `flux_fit`,
`flags` and `qfit` identical in every row), so a run without the switches is unchanged.

## Tile test (o105)

The o105 tile was rebuilt end to end (stage, 8-shard m7 fan-out over both bands, finalize,
scoring) with the integration code of the end-to-end tests plus this switch and the
near-saturation distance switch, in the environment of the end-to-end control (production
recipe, ePSF core on).  One arm set
`SATSTAR_GATE_EPSF_CORE_SCALE=1` and `NEARSAT_CONTINUOUS_DIST=1` ("both switches"); a second arm
set `NEARSAT_CONTINUOUS_DIST=1` alone (this switch off in a tree that contains it).  No arm
with this switch alone was built: in the
replays this switch alone returns neither the F4 row nor the F5 neighbour, so the difference
of the two arms measures what this switch adds to the distance switch.  The control column is
the control render of the end-to-end tests, built earlier in another worktree of the same
code.  The production column is the production residual image, scored with the same tools as
the other renders.

Counts are by peak or depth in MJy/sr.  A leftover is a residual with residual / data > 0.7
and a peak of at least the stated value; a big crater is a negative residual deeper than
50 MJy/sr whose depth exceeds 0.3 times the model peak.

| band | render | leftovers >= 50 | >= 100 | >= 200 | >= 500 | >= 1000 | craters >= 50 | >= 200 | >= 500 |
|---|---|---|---|---|---|---|---|---|---|
| F480M | production | 303 | 202 | 168 | 111 | 84 | 69 | 28 | 16 |
| F480M | control | 297 | 200 | 168 | 107 | 77 | 92 | 38 | 17 |
| F480M | fit-time switches, unmasked | 221 | 155 | 146 | 104 | 78 | 20 | 12 | 8 |
| F480M | fit-time switches, masked | 49 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| F480M | distance switch alone | 277 | 180 | 150 | 103 | 76 | 94 | 39 | 17 |
| F480M | both switches | 277 | 180 | 150 | 103 | 76 | 95 | 40 | 18 |
| F212N | production | 845 | 317 | 178 | 133 | 116 | 489 | 166 | 71 |
| F212N | control | 830 | 315 | 176 | 134 | 117 | 454 | 152 | 65 |
| F212N | fit-time switches, unmasked | 610 | 118 | 38 | 13 | 8 | 102 | 44 | 18 |
| F212N | fit-time switches, masked | 475 | 52 | 8 | 0 | 0 | 10 | 0 | 0 |
| F212N | distance switch alone | 827 | 312 | 175 | 133 | 116 | 452 | 151 | 64 |
| F212N | both switches | 827 | 312 | 175 | 133 | 116 | 452 | 151 | 64 |

The two further renders are the end-to-end render with the fit-time switches of the other
open pull requests and no footprint mask ("unmasked") and the same with the footprint masks
("masked"); neither contains the two switches of this test.  The four features followed through the
end-to-end tests:

| feature | production | control | fit-time switches, unmasked | fit-time switches, masked | distance switch alone | both switches |
|---|---|---|---|---|---|---|
| F480M leftover (2614, 1826), "F4" | none | 904 | 904 | none | none | none |
| F480M leftover (4452, 858), "F5" | none | 403 | 403 | none | none | none |
| F212N crater (8306, 1270), depth | 978 | 966 | 1590 (ratio 0.32, big) | none | 966 | 966 |
| F212N crater (4469, 1280), depth | 463 | 465 | 755 (ratio 0.30, big) | none | 465 | 465 |

"None" is no entry in the leftover or crater list within 3 px.  The 13 x 13 pixel box maxima
of F4 and F5 are 33 and 59 MJy/sr with the distance switch alone and 33 and 26 with both.

Both switches against the distance switch alone:

* F480M leftovers: every count in the table is the same.  The 21 leftovers >= 100 that the
  distance switch clears against the control are cleared in both arms, with the same new one
  at (5325, 1803) (327 MJy/sr).  The F4 feature at (2614, 1826) is 33 MJy/sr in both arms;
  the F5 feature at (4452, 858) is 59 MJy/sr with the distance switch alone and 26 with both.
* F480M craters: 94 to 95 at >= 50, 39 to 40 at >= 200 and 17 to 18 at >= 500.  The added
  crater is at (1597, 1877): depth 881, ratio 0.44 and a 13 x 13 pixel minimum of -867
  MJy/sr with both switches, against depth 475 (ratio 0.29, below the 0.3 limit) and a
  minimum of -462 in the control and with the distance switch alone, and depth 479 and
  -466 in production.  A returned satstar row (below) lies 0.04 arcsec from it (flux
  28,577.2; `qfit` 0.252 at (1596.6, 1877.5); the other arms have a row there with flux
  22,842.9).
* The consolidated F480M satstar catalogs, matched within 0.15 arcsec:

  | render | rows | against production |
  |---|---|---|
  | production | 1703 | |
  | control | 1686 | 21 production rows lost (median flux 28,711), 4 rows not in production gained |
  | distance switch alone | 1686 | the same as the control: none lost, none gained |
  | both switches | 1701 | 7 production rows lost (median flux 24,928), 5 rows not in production gained |

  Control to both switches: none lost and 15 gained, with fluxes from 28,357 to 29,237
  (median 28,735).  These are 14 of the 21 production rows that the control lost and one row
  that production does not have.  The median flux ratio of the 1686 rows the two share is
  1.0000 (16th to 84th percentile 1.0000 to 1.0000; no row differs by more than 20 per cent).
  The F5 star has a satstar row with both switches (flux 28,581; production 28,728) and none
  in the control or with the distance switch alone (the nearest row is 3.8 arcsec away).
* The F480M residual images of the two arms differ by more than 5 MJy/sr in 1786 pixels (117
  regions after a 3 pixel dilation).  17 regions have a peak difference above 50 MJy/sr; 13 of
  them lie within 3 px of one of the 15 returned rows (the largest, 466 MJy/sr, at (4837, 360)
  and 464 at (1596, 1877)); the other four are at (5236, 916), (3673, 107), (721, 1514) and
  (5421, 1177), with peak differences of 114, 55, 52 and 52 MJy/sr, 122 to 1085 px from the
  nearest returned row.  Each lies within 2.3 px of a satstar row that both arms have and
  whose flux differs between the arms; the last table below lists the five largest of
  these regions.
* F212N: the switch has no F212N entry.  The two arms have the same counts, the same satstar
  rows (10,399; production 10,356) and residual images that differ in 4433 of 51,825,305
  pixels by at most 6.3e-5 MJy/sr.

The 15 returned rows and the residual around them.  The table lists the 15 consolidated
satstar rows that the arm with both switches has and the control lacks.  The last four columns
give the minimum and maximum of the residual (MJy/sr, written min / max) in the 13 x 13 pixel
box around each row.  Positions are tile pixels and the flux is `flux_fit` of the consolidated
satstar catalog.

| x | y | satstar flux | production | control | distance switch alone | both switches |
|---|---|---|---|---|---|---|
| 1579 | 1960 | 28,357 | -100 / 34 | 4 / 70 | 4 / 70 | -101 / 64 |
| 4837 | 360 | 28,504 | -480 / 206 | -20 / 225 | -20 / 226 | -468 / 205 |
| 1597 | 1877 | 28,577 | -466 / 279 | -462 / 279 | -462 / 279 | -867 / 280 |
| 4457 | 856 | 28,581 | -425 / 30 | -1 / 418 | -1 / 59 | -349 / 26 |
| 1545 | 2079 | 28,671 | -71 / 36 | -59 / 43 | -59 / 43 | -61 / 41 |
| 3729 | 1334 | 28,678 | -348 / 81 | 1 / 83 | 1 / 83 | -293 / 79 |
| 4664 | 1518 | 28,705 | -343 / 75 | -34 / 92 | -34 / 92 | -312 / 77 |
| 2684 | 1378 | 28,735 | -240 / 58 | -61 / 62 | -61 / 62 | -217 / 50 |
| 2772 | 1062 | 28,769 | -259 / 40 | -39 / 74 | -39 / 74 | -209 / 43 |
| 5399 | 1040 | 28,789 | -173 / 53 | -22 / 71 | -22 / 71 | -126 / 47 |
| 1150 | 142 | 28,914 | -357 / 24 | -44 / 56 | -44 / 56 | -348 / 45 |
| 5364 | 1285 | 28,990 | -29 / 67 | -25 / 48 | -25 / 48 | -25 / 118 |
| 8 | 47 | 29,044 | -163 / 38 | -2 / 47 | -2 / 47 | -8 / 74 |
| 5360 | 1429 | 29,145 | -93 / 30 | -8 / 39 | -8 / 39 | -57 / 27 |
| 5363 | 88 | 29,237 | -136 / 39 | -6 / 49 | -6 / 49 | -45 / 46 |

The box minimum is below -200 MJy/sr at 8 of the 15 rows in production, 1 in the control,
1 with the distance switch alone and 8 with both switches.  Its median over the 15 rows
is -240 MJy/sr in production, -22 in the control, -22 with the distance switch alone and
-209 with both switches.  With both switches the box minimum is lower than with the distance
switch alone by more than 50 MJy/sr at 10 of the 15 rows.  The box maximum is above 200 MJy/sr
at 2 rows in production, 3 in the control, 2 with the distance switch alone and
2 with both switches.  The F5 star is the row at (4457, 856): its box maximum is
418 in the control, 59 with the distance switch alone and 26 with both switches, and its
box minimum is -1, -1 and -349 (production: 30 and -425).  At these stars the
satstar fit that the switch returns subtracts about as deeply as the production fit
and more deeply than the daophot fit of the control.  Of the 15 rows only the star at
(1597, 1877) is the added crater counted above: against the control the arm with the distance
switch alone has 4 new and 2 gone craters at >= 50 MJy/sr and the arm with both switches has
5 new and 2 gone.

The five largest regions of the difference image that lie more than 3 px from a returned row
each lie within 2.3 px of a matched satstar row, a row that the control and the arm with both
switches both have, whose flux differs between the two:

| region (x, y) | peak difference (MJy/sr) | nearest matched row | satstar flux, control to both switches |
|---|---|---|---|
| (5236, 916) | 114.2 | 2.3 px | 30,381 to 29,859 (-1.7 %) |
| (3673, 107) | 55.4 | 0.3 px | 28,100 to 28,465 (+1.3 %) |
| (721, 1514) | 52.3 | 2.1 px | 29,187 to 28,853 (-1.1 %) |
| (5421, 1177) | 52.0 | 0.8 px | 29,548 to 29,039 (-1.7 %) |
| (2036, 623) | 49.0 | 0.4 px | 29,646 to 28,420 (-4.1 %) |

The arm with the distance switch alone has the satstar rows and fluxes of the control in every
one of the 1686 matched rows.  With both switches 56 of the 1686 differ in `flux_fit`: 49 by
more than 1e-6 of the flux, 30 by more than 0.1 %, 11 by more than 1 % and none by more than
5 %.  The cause of these flux changes was not traced.

On this tile the switch returns 14 of the 21 production satstar rows that the control had
moved to the daophot channel, changes no leftover count in either band and adds one deeper
crater at one of the returned rows.  At the F5 star it lowers the box maximum from 59 to 26
MJy/sr and the residual at the peak pixel (4457, 856) from +34 to -349 MJy/sr (production
-425, control +48).  The box minimum is below -200 MJy/sr at 8 of the 15 returned rows, as
in production, and at 1 of the 15 in the control.  The tile test shows no gain in the counts
of leftovers or craters.  This is one tile, and the arm with this switch also has the
distance switch.

Evidence files are in `docs/evidence/satstar_gate_tile_o105/`.  `o105_both_summary.txt` and
`o105_near_summary.txt` hold the counts and the paired changes of the two arms against the
production, control and two other renders; `o105_both_compare_F480M.txt` and
`o105_both_compare_F212N.txt` list every cleared, reduced, worse and new feature of the arm
with both switches; `o105_both_scan_ctlfin_F480M.txt` and `o105_both_scan_ctlfin_F212N.txt`
hold the position-independent scan against the control (13 pixel boxes, threshold 200
MJy/sr); `o105_both_satcons.txt` and `o105_near_satcons.txt` compare the consolidated satstar
catalogs; `o105_satstar_flux_changes_both.txt` lists the matched rows whose flux changes;
`o105_gained_rows_boxes.txt` is the table of the 15 returned rows; `o105_residual_both_minus_near.txt`
lists the regions where the two F480M images differ; `o105_both_features.txt`,
`o105_boxes_at_features.txt` and `o105_rows_at_features.txt` give the features F1, F2, F4
and F5, the residual extrema in the boxes around the features of both tile tests and the
rows of the merged catalog there.  The files label the renders `prod5` (production),
`ctlfin` (control), `optnm` and `optfin` (the fit-time switches of the other open pull
requests, unmasked and masked), `n1xnear` (the distance switch alone) and `n1xboth` (both
switches).  The survive files named in line 16 of the compare files are scratch files of the
end-to-end run and are not part of the evidence.

## Limits of this change

* One scale per detector corrects the median, not the star-to-star scatter.  94 of
  25,174 matched rows (0.4 %), 40 of them in the accepting direction, still differ from
  the production decision.
* The scale was measured on the o105, o045, o118, o130 and o138 tiles of the
  gc-treasury 10678 F480M frames.  Another field, filter or core version needs the
  measurement again; the recipe above is all of it.
* It changes which stars the satstar channel owns, so a star near the gate can come back
  to the satstar channel with a flux that differs from the daophot one.  The tile test
  above counts the consolidated satstar rows gained and lost.
* The tile test is one tile, and the arm with this switch also has the distance switch of
  #1394.  No arm with this switch alone was built, so its effect is the difference of two
  arms that share the other switch.
