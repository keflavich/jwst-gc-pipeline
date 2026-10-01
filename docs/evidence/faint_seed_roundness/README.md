# i2d residual seed: admit roundness to ±0.8 above local structure

The coadd i2d-seed daofind keeps `|roundness| ≤ 0.5`.  On the Brick F182M m7
residual 55% of the S/N > 7 residual peaks pass that cut and 78% pass ±0.8: a
faint star distorted by noise or by a neighbour's subtracted wing fails it and
never enters the seed.  This branch admits detections with roundness in
(0.5, 0.8] when their annulus prominence on the detection image is ≥ 5.
The loose cut is AUTO-off (tight ±0.5 only) on extended-emission targets
(`_is_extended_emission`), where elongated knots and filament points pass the
prominence test.

Figure layout and metric definitions:
[../faint_reference_fields/README.md](../faint_reference_fields/README.md).
Comparison: `m7seed` (the base branch) → `seedround4` (this branch).

### Super-high density (NSC, F212N), injection seed 1
![](superdense_s1.png)

9 added, 2 dropped; the seed's injected S/N 160–320 bin goes 3/9 → 4/9.
Row A: an injected star and two faint neighbours enter the seed.  Row B:
three faint stars in a gap between bright ones.

### Dark cloud (Brick, F182M), injection seed 1
![](dark_s1.png)

1 added, 2 dropped; S/N 10–20 bin 2/6 → 3/6; over-subtracted 13 → 11.
Row B: an injected star blended with a neighbour (distorted shape) enters the
seed and is fitted.  Row A: the dropped source left a negative core in the
base branch's residual.

### High density on bright background (Sgr B2, F187N), clean run
![](dense_bright_s0.png)

8 added, 5 dropped; residual excess 1.80 → 2.08 per arcsec², over-subtracted
unchanged (21).  Row A: two base-branch sources on an elongated residual bar
are gone.  Row B: the bright star is refitted at a different position and its
core stays over-subtracted (PSF mismatch).  Row C: a faint star added.

## Metrics (m7)

**superdense (NSC, F212N)** (phase m7)

| variant | S/N 40-80 | S/N 80-160 | S/N 160-320 | S/N 320-640 | resid excess /as² (+/−) | over-subtracted /as² (n) | labels | bias mag |
|---|---|---|---|---|---|---|---|---|
| m7seed | 1/11 | 5/13 | 5/14 | 8/10 | 1.32 (22/3) | 1.32 (19) | — | 0.018 |
| seedround4 | 1/11 | 5/13 | 7/14 | 8/10 | 1.39 (23/3) | 1.32 (19) | — | -0.048 |

**dense + bright bg (Sgr B2, F187N)** (phase m7)

| variant | S/N 5-10 | S/N 10-20 | S/N 20-40 | S/N 40-80 | resid excess /as² (+/−) | over-subtracted /as² (n) | labels | bias mag |
|---|---|---|---|---|---|---|---|---|
| m7seed | 0/17 | 0/11 | 2/9 | 5/11 | 1.80 (26/0) | 1.45 (21) | — | 0.001 |
| seedround4 | 0/17 | 0/11 | 2/9 | 6/11 | 2.08 (30/0) | 1.45 (21) | — | -0.004 |

**modest density + bright bg (W51, F187N)** (phase m7)

| variant | S/N 5-10 | S/N 10-20 | S/N 20-40 | S/N 40-80 | resid excess /as² (+/−) | over-subtracted /as² (n) | labels | emission knots cataloged | bias mag |
|---|---|---|---|---|---|---|---|---|---|
| m7seed | 0/11 | 0/6 | 0/13 | 11/18 | 0.55 (10/2) | 0.07 (1) | — | 2/4 | 0.138 |
| seedround4 | 0/11 | 0/6 | 1/13 | 11/18 | 1.11 (19/3) | 0.07 (1) | — | 2/4 | 0.154 |

**dark cloud (Brick, F182M)** (phase m7)

| variant | S/N 5-10 | S/N 10-20 | S/N 20-40 | S/N 40-80 | resid excess /as² (+/−) | over-subtracted /as² (n) | labels | bias mag |
|---|---|---|---|---|---|---|---|---|
| m7seed | 0/10 | 4/11 | 9/15 | 11/12 | 1.18 (19/2) | 1.11 (16) | 0.55 | 0.050 |
| seedround4 | 0/10 | 6/11 | 9/15 | 11/12 | 1.18 (19/2) | 1.04 (15) | 0.55 | 0.050 |

The W51 row above is the first version of this branch, with the loose cut
active on W51: 60–70 loose detections per phase at m5–m7, 25–29 passing
prominence ≥ 5, residual excess 0.55 → 1.11 for +1/48 injected stars.  That
result is why the loose cut is AUTO-off on extended-emission targets; with
the current branch W51 is identical to `m7seed`.

## Caveats

- Sgr B2 residual excess rises 1.80 → 2.08 (26 → 30 positive peaks).  The
  extra peaks come from refits around bright stars (row B) more than from the
  added sources.
- The prominence ≥ 5 requirement is the only guard against elongated
  structure on star fields; a dense field with filamentary emission and no
  `_is_extended_emission` entry would need the AUTO list extended.
