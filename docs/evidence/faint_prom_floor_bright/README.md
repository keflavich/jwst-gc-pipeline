# Exempt tight, bright fits from the extended-emission prominence floor

On extended-emission targets (w51, sickle, wd2, ngc6334) vetting applies a
hard prominence floor of 3.0.  Prominence is normalized by the MAD of the
4–10 px annulus, and on a nebular field that MAD is set by emission structure,
so a real bright star can read prominence 2–3.  In the W51 F187N reference
field, injected stars at S/N_true 52 and 76 were fitted with qfit 0.17 and
0.09 and deleted by the floor alone (prominence 2.55 and 2.71).  This branch
passes sources with `qfit ≤ 0.2` AND merged S/N ≥ 30 at prominence ≥ 2.0.

Chance-corrected continuum-match purity of qfit ≤ 0.2, prominence 2–3
sources in the W51 m6 catalogs: F187N vs F210M ~1.0 at S/N_prop ≥ 30 (n = 33);
F480M vs F410M 0.38–0.51 at S/N_prop 20–30 (n = 40) and 0.88–1.00 above 30.
The S/N 20–30 bin sets the S/N 30 requirement.

Figure layout and metric definitions:
[../faint_reference_fields/README.md](../faint_reference_fields/README.md).
Comparison: `snrprop` (the base branch, S/N-floor fix) → `promfloor` (this
branch).

### Modest density on bright, filamentary emission (W51, F187N), injection seed 1
![](bright_modest_s1.png)

2 added, 0 dropped; the seed's injected S/N 40–80 bin goes 5/9 → 7/9.  Rows
A and B: the two injected stars the floor deleted are fitted and leave the
residual.  Neither is near a hand-labelled knot (magenta ×).

### W51, clean run
![](bright_modest_s0.png)

No catalog change: the exemption admits no source in the clean field, and
emission knots cataloged stays 2/4.

## Metrics (m7)

**modest density + bright bg (W51, F187N)** (phase m7)

| variant | S/N 5-10 | S/N 10-20 | S/N 20-40 | S/N 40-80 | resid excess /as² (+/−) | over-subtracted /as² (n) | labels | emission knots cataloged | bias mag |
|---|---|---|---|---|---|---|---|---|---|
| snrprop | 0/11 | 0/6 | 0/13 | 11/18 | 0.55 (10/2) | 0.07 (1) | — | 2/4 | 0.138 |
| promfloor | 0/11 | 0/6 | 0/13 | 13/18 | 0.55 (10/2) | 0.07 (1) | — | 2/4 | -0.025 |

## Caveats

- W51 completeness below S/N_true 40 stays 1/30 or lower in every variant.
  This branch changes only fits with merged S/N ≥ 30; extending the
  exemption to S/N_prop 20–30 would admit emission at purity 0.38–0.51.
- Flux bias improves 0.138 → −0.025 mag because the bright injected stars
  are now in the catalog.
- Fields outside `_EXTENDED_EMISSION_TARGETS` are unchanged.
