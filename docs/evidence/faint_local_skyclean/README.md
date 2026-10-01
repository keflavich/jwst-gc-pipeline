# Sky-clean tier: local 3″ structure reference

The sky-clean keep tier (prominence + S/N, qfit ignored) applies where a
source's annulus floor is within 2 dark-sky σ of the field's dark-sky
reference, the 5th percentile of the whole mosaic.  Outside the darkest lane
of a field every source reads as "on emission" and the tier is inert (Brick
F182M m7: 29k of 35k dropped S/N 5–20 sources failed only this test).  This
branch also marks a source clean when its annulus floor is within 2 i2d ERR
of the 5th percentile of its own 3″ tile.  Tile p25 − p5 measures ~0.8 ERR in
the Brick dark cloud, ~1.8 in the Sgr B2 envelope, ~9 on W51 F187N emission
and ~13 in the Sgr A* cluster, so filaments and crowding stay outside the
tier.  The global test is kept as an OR.

Figure layout and metric definitions:
[../faint_reference_fields/README.md](../faint_reference_fields/README.md).
Comparison: `m7seed` (the base branch) → `skylocal4` (this branch).

### Dark cloud (Brick, F182M), injection seed 1
![](dark_s1.png)

7 added, 0 dropped; the seed's injected S/N 20–40 bin goes 5/10 → 7/10.
Row B: two injected stars (yellow +) are fitted and leave the residual.
Row A: two added sources next to a brighter star carry over-subtracted cores
(13 → 15 in this seed).

### Dark cloud (Brick, F182M), clean run
![](dark_s0.png)

6 added, 1 dropped; residual excess 1.18 → 0.97 per arcsec², over-subtracted
16 → 19.  Row A: the added sources sit in and around a compact group; after
the refit two existing members and one added star have over-subtracted cores.
This compact group is the same one the qfit and S/N-floor branches touch.

### High density on bright background (Sgr B2, F187N), clean run
![](dense_bright_s0.png)

2 added, 0 dropped; residual excess 1.80 → 1.59, over-subtracted unchanged
(21).  Row A: one source next to a bright star, on the same residual bar the
qfit branch picks up, and one faint star.

## Metrics (m7)

**superdense (NSC, F212N)** (phase m7)

| variant | S/N 40-80 | S/N 80-160 | S/N 160-320 | S/N 320-640 | resid excess /as² (+/−) | over-subtracted /as² (n) | labels | bias mag |
|---|---|---|---|---|---|---|---|---|
| m7seed | 1/11 | 5/13 | 5/14 | 8/10 | 1.32 (22/3) | 1.32 (19) | — | 0.018 |
| skylocal4 | 1/11 | 5/13 | 5/14 | 8/10 | 1.32 (22/3) | 1.32 (19) | — | 0.018 |

**dense + bright bg (Sgr B2, F187N)** (phase m7)

| variant | S/N 5-10 | S/N 10-20 | S/N 20-40 | S/N 40-80 | resid excess /as² (+/−) | over-subtracted /as² (n) | labels | bias mag |
|---|---|---|---|---|---|---|---|---|
| m7seed | 0/17 | 0/11 | 2/9 | 5/11 | 1.80 (26/0) | 1.45 (21) | — | 0.001 |
| skylocal4 | 0/17 | 0/11 | 3/9 | 5/11 | 1.59 (23/0) | 1.45 (21) | — | -0.004 |

**modest density + bright bg (W51, F187N)** (phase m7)

| variant | S/N 5-10 | S/N 10-20 | S/N 20-40 | S/N 40-80 | resid excess /as² (+/−) | over-subtracted /as² (n) | labels | emission knots cataloged | bias mag |
|---|---|---|---|---|---|---|---|---|---|
| m7seed | 0/11 | 0/6 | 0/13 | 11/18 | 0.55 (10/2) | 0.07 (1) | — | 2/4 | 0.138 |
| skylocal4 | 0/11 | 0/6 | 0/13 | 11/18 | 0.55 (10/2) | 0.07 (1) | — | 2/4 | 0.138 |

**dark cloud (Brick, F182M)** (phase m7)

| variant | S/N 5-10 | S/N 10-20 | S/N 20-40 | S/N 40-80 | resid excess /as² (+/−) | over-subtracted /as² (n) | labels | bias mag |
|---|---|---|---|---|---|---|---|---|
| m7seed | 0/10 | 4/11 | 9/15 | 11/12 | 1.18 (19/2) | 1.11 (16) | 0.55 | 0.050 |
| skylocal4 | 0/10 | 4/11 | 11/15 | 10/12 | 0.97 (16/2) | 1.32 (19) | 0.55 | 0.050 |

## Caveats

- Over-subtracted cores rise on the dark field (16 → 19 clean, 13 → 15
  seed 1), all inside crowded groups next to brighter stars.
- Unchanged on superdense and W51 at m7, as intended (their tiles have
  large p25 − p5).
- The local reference is the tile's 5th percentile; a tile that is entirely
  inside a smooth bright plateau counts as clean.  A smooth plateau cannot
  imitate a PSF, and the prominence and S/N requirements of the tier still
  apply.
