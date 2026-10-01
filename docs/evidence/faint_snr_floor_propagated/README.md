# Vetting S/N floors on the merged-flux uncertainty (`flux_err_prop`)

A merged catalog's `flux` is the mean over `nmatch` per-frame fits, but its
`flux_err` is the weighted mean of the per-frame errors (one frame's
uncertainty).  The vetting floor `local_snr_min = 5` on `flux/flux_err`
therefore acts as ~5√nmatch on the merged flux (Brick F182M m6: median
`flux_err/flux_err_prop` 3.16; 66,913 of 506,114 sources below the per-frame
floor and above it on `flux_err_prop`).  This branch applies the floors to
`flux/flux_err_prop`.

Figure layout and metric definitions:
[../faint_reference_fields/README.md](../faint_reference_fields/README.md).
Comparison: `m7seed` (the base branch) → `snrprop` (this branch).

### Dark cloud (Brick, F182M), injection seed 1
![](dark_s1.png)

9 added, 0 dropped; the seed's injected S/N 10–20 bin goes from 2/6 to 4/6.
Row A, right: an injected star (yellow +) that the per-frame floor rejects is
fitted and leaves the residual.  Row A, left: one added source next to a
brighter star has an over-subtracted core (13 → 14 in this seed).

### Dark cloud (Brick, F182M), clean run
![](dark_s0.png)

12 added, 0 dropped; residual excess 1.18 → 0.83 per arcsec², over-subtracted
cores 16 → 14, hand-labelled clump stars 0.55 → 0.67.  Row B: isolated faint
stars on the dark cloud.

### High density on bright background (Sgr B2, F187N), clean run
![](dense_bright_s0.png)

17 added, 2 dropped; residual excess 1.80 → 1.39, over-subtracted 21 → 20.
Rows A–C: faint stars between brighter ones.

## Metrics (m7)

**superdense (NSC, F212N)** (phase m7)

| variant | S/N 40-80 | S/N 80-160 | S/N 160-320 | S/N 320-640 | resid excess /as² (+/−) | over-subtracted /as² (n) | labels | bias mag |
|---|---|---|---|---|---|---|---|---|
| m7seed | 1/11 | 5/13 | 5/14 | 8/10 | 1.32 (22/3) | 1.32 (19) | — | 0.018 |
| snrprop | 1/11 | 5/13 | 5/14 | 8/10 | 1.32 (22/3) | 1.32 (19) | — | 0.018 |

**dense + bright bg (Sgr B2, F187N)** (phase m7)

| variant | S/N 5-10 | S/N 10-20 | S/N 20-40 | S/N 40-80 | resid excess /as² (+/−) | over-subtracted /as² (n) | labels | bias mag |
|---|---|---|---|---|---|---|---|---|
| m7seed | 0/17 | 0/11 | 2/9 | 5/11 | 1.80 (26/0) | 1.45 (21) | — | 0.001 |
| snrprop | 0/17 | 1/11 | 2/9 | 5/11 | 1.39 (20/0) | 1.39 (20) | — | 0.001 |

**modest density + bright bg (W51, F187N)** (phase m7)

| variant | S/N 5-10 | S/N 10-20 | S/N 20-40 | S/N 40-80 | resid excess /as² (+/−) | over-subtracted /as² (n) | labels | emission knots cataloged | bias mag |
|---|---|---|---|---|---|---|---|---|---|
| m7seed | 0/11 | 0/6 | 0/13 | 11/18 | 0.55 (10/2) | 0.07 (1) | — | 2/4 | 0.138 |
| snrprop | 0/11 | 0/6 | 0/13 | 11/18 | 0.55 (10/2) | 0.07 (1) | — | 2/4 | 0.138 |

**dark cloud (Brick, F182M)** (phase m7)

| variant | S/N 5-10 | S/N 10-20 | S/N 20-40 | S/N 40-80 | resid excess /as² (+/−) | over-subtracted /as² (n) | labels | bias mag |
|---|---|---|---|---|---|---|---|---|
| m7seed | 0/10 | 4/11 | 9/15 | 11/12 | 1.18 (19/2) | 1.11 (16) | 0.55 | 0.050 |
| snrprop | 1/10 | 8/11 | 9/15 | 11/12 | 0.83 (14/2) | 0.97 (14) | 0.67 | 0.050 |

## Caveats

- Superdense and W51 are unchanged at m7: in the NSC the injected stars are
  well above both floors, and in W51 the extended-emission prominence floor
  (see the prom-floor-bright branch) decides.
- Dark clean run, row A: three new sources on one compact residual blob next
  to a diagonal dust/emission edge.  It is fitted as a close group; it may be
  one extended source split three ways.
- Sgr B2 clean run, row D: two new sources on a diffuse residual patch;
  the F212N companion band does not settle whether they are stars.
- The qfit noise term and the bright-isolated keep stay on the per-frame S/N,
  since `qfit` is itself a per-frame mean.
