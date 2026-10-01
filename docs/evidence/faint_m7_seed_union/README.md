# m7 seed = cross-band seed ∪ own-band m6 vetted catalog

`main` builds the m7 seed from the ≥2-filter cross-band seed only, so every
source that one band's m6 vetting accepted and no other band confirmed leaves
the m7 model and returns to the final residual (a third of the m6 vetted
catalog in Brick F182M, Sgr B2 and Sgr A*).  This branch unions each band's own
m6 vetted catalog, and daofind detections on the m6 residual − background, into
that band's m7 seed.

The other faint-star vetting branches stack on this one: without it, a source
that a vetting fix admits at m6 is dropped again at m7.

Figure layout and metric definitions:
[../faint_reference_fields/README.md](../faint_reference_fields/README.md).
Comparison: `main` (current) → `m7seed` (this branch).

### Super-high density (NSC, F212N), injection seed 1
![](superdense_s1.png)

41 sources added, 1 dropped.  Rows A–D: faint stars between the bright ones
are fitted and leave the residual.  Injected recovery at S/N 80–320 goes from
1/16 to 5/16 in this seed.

### Dark cloud (Brick, F182M), injection seed 1
![](dark_s1.png)

16 added, 3 dropped; over-subtracted cores 16 → 13.  The dropped sources (red
boxes, rows A–B) leave negative cores in `main`'s residual (black in the
difference column).

### High density on bright background (Sgr B2, F187N), clean run
![](dense_bright_s0.png)

40 added; residual excess 3.05 → 1.80 per arcsec².  Rows A and D: some new
sources next to bright stars carry over-subtracted cores (17 → 21 field-wide);
see caveats.

### Modest density on bright, filamentary emission (W51, F187N), clean run
![](bright_modest_s0.png)

One source added.  The residual gains compact Pa-α peaks, shown next.

![](w51_m7_knots.png)

Six residual peaks (S/N 22–40) are in `main`'s m6 residual and in this
branch's m7 residual.  `main`'s m7 residual hides them because its 7-position
seed leaves them unmasked and its smoothed background absorbs them.  None has
an F210M counterpart; all are vetted out of the catalog.

## Metrics (m7)

**superdense (NSC, F212N)** (phase m7)

| variant | S/N 40-80 | S/N 80-160 | S/N 160-320 | S/N 320-640 | resid excess /as² (+/−) | over-subtracted /as² (n) | labels | bias mag |
|---|---|---|---|---|---|---|---|---|
| main | 1/11 | 1/13 | 2/14 | 7/10 | 1.66 (27/3) | 1.11 (16) | — | -0.048 |
| m7seed | 1/11 | 5/13 | 5/14 | 8/10 | 1.32 (22/3) | 1.32 (19) | — | 0.018 |

**dense + bright bg (Sgr B2, F187N)** (phase m7)

| variant | S/N 5-10 | S/N 10-20 | S/N 20-40 | S/N 40-80 | resid excess /as² (+/−) | over-subtracted /as² (n) | labels | bias mag |
|---|---|---|---|---|---|---|---|---|
| main | 0/17 | 0/11 | 2/9 | 5/11 | 3.05 (44/0) | 1.18 (17) | — | -0.058 |
| m7seed | 0/17 | 0/11 | 2/9 | 5/11 | 1.80 (26/0) | 1.45 (21) | — | 0.001 |

**modest density + bright bg (W51, F187N)** (phase m7)

| variant | S/N 5-10 | S/N 10-20 | S/N 20-40 | S/N 40-80 | resid excess /as² (+/−) | over-subtracted /as² (n) | labels | emission knots cataloged | bias mag |
|---|---|---|---|---|---|---|---|---|---|
| main | 0/11 | 0/6 | 1/13 | 11/18 | 0.21 (4/1) | 0.07 (1) | — | 2/4 | 0.154 |
| m7seed | 0/11 | 0/6 | 0/13 | 11/18 | 0.55 (10/2) | 0.07 (1) | — | 2/4 | 0.138 |

**dark cloud (Brick, F182M)** (phase m7)

| variant | S/N 5-10 | S/N 10-20 | S/N 20-40 | S/N 40-80 | resid excess /as² (+/−) | over-subtracted /as² (n) | labels | bias mag |
|---|---|---|---|---|---|---|---|---|
| main | 0/10 | 3/11 | 9/15 | 10/12 | 1.39 (22/2) | 1.18 (17) | 0.45 | 0.045 |
| m7seed | 0/10 | 4/11 | 9/15 | 11/12 | 1.18 (19/2) | 1.11 (16) | 0.55 | 0.050 |

## Caveats

- Sgr B2 over-subtracted cores rise 17 → 21.  Some added sources sit on a
  bright neighbour's PSF wing, where PSF-model mismatch leaves a positive
  ring.  The prominence-keep branch adds a concentration guard for these.
- W51 residual excess rises 0.21 → 0.55 per arcsec².  The added peaks are the
  Pa-α knots above, masked from the m7 background because the m6 i2d
  augmentation now detects them.  This is the existing "mask vetted ∪ seed"
  background behaviour applied to a fuller seed.  Knots cataloged stays 2/4.
- The faint-star gains here are modest because the m6 vetting still rejects
  most S/N < 20 stars; the vetting branches stacked on this one carry the rest.
