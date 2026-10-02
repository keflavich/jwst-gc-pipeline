# Vetting S/N floors on the merged-flux uncertainty (`flux_err_prop`)

A merged catalog's `flux` is the mean over `nmatch` per-frame fits, but its
`flux_err` is the weighted mean of the per-frame errors (one frame's
uncertainty).  The vetting floor `local_snr_min = 5` on `flux/flux_err`
therefore acts as ~5√nmatch on the merged flux (Brick
`f182m_merged_o001_indivexp_merged_resbgsub_m6_dao_basic.fits`: median
`flux_err/flux_err_prop` 3.16, equal to the median √nmatch_good; 66,913 of
506,114 sources below the per-frame floor and above it on `flux_err_prop`).
This branch applies the local S/N floor to `flux/flux_err_prop`.

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

## Full-field replay (Brick, Sgr B2)

The m6 vetting of this branch and of the base branch was replayed on the
production m6 merged catalogs (`scripts/vet_variant.py`, which executes each
worktree's own vetting call with pipeline defaults).

**Keep path.**  The local S/N floor moves to `flux_err_prop`; the sky-clean
floor stays on the per-frame S/N (see "Sky-clean floor" below).

| field, band | base kept | this branch | added | via local floor | of which qfit > 0.2 | lost |
|---|---|---|---|---|---|---|
| Brick F182M | 377,837 | 408,951 | 31,114 | 31,114 (31,062 peakSB, 52 flags) | 31,090 | 0 |
| Brick F212N | 117,039 | 149,763 | 32,724 | 32,724 (32,595 peakSB, 129 flags) | — | 0 |
| Brick F405N | 92,291 | 102,959 | 10,668 | 10,668 (10,632 peakSB, 36 flags) | — | 0 |

The added F182M sources have median per-frame S/N 3.6, median S/N on
`flux_err_prop` 12.1, median `nmatch` 16 and median prominence 4.8.

**Realness.**  Added sources are matched against an independent catalog
within 60 mas.  The chance rate comes from the same positions shifted by
~2″.  The expectation is the match rate of base-kept sources of the same
flux, in the same bin of distance to the nearest saturated star:
rel = (match − chance) / (expected − chance); 1 means as real as the
current catalog at that flux, 0 means chance.

| field (reference) | sat. distance | added | match | chance | expected | rel |
|---|---|---|---|---|---|---|
| Brick F182M (F200W 1182/o004 m7, independent visit) | 0–1″ | 5,239 | 0.37 | 0.07 | 0.28 | 1.44 |
| | 1–2″ | 15,145 | 0.51 | 0.07 | 0.34 | 1.62 |
| | > 2″ | 11,028 | 0.49 | 0.07 | 0.49 | 0.99 |
| Sgr B2 F187N (F182M m6, same visit) | 0–1″ | 1,340 | 0.26 | 0.10 | 0.25 | 1.18 |
| | 1–2″ | 5,550 | 0.46 | 0.10 | 0.35 | 1.45 |
| | > 2″ | 25,365 | 0.54 | 0.09 | 0.63 | 0.83 |

(This replay predates the sky-clean correction below and includes its 298
Brick F182M sky-clean additions; they change these rates by < 0.01.)
Within 2″ of a saturated star the base catalog itself is confirmed less
often than far from one, so rel > 1 there reflects a lower expectation; the
added sources' match rate (0.37–0.54) is close to the far-field base rate.
The Sgr B2 reference is another filter of the same visit, so PSF artifacts
of bright stars can match in both; it overstates realness.

**Realness by keep path and prominence.**  The same replay, with the
expectation taken over all saturated-star distances (`scripts/slices.py`).
Keep path is the first of qfit ≤ 0.2, flags = 1, peakSB > 20 × local_bkg,
other (sky-clean) that applies.

| slice | Brick n | Brick rel | Sgr B2 n | Sgr B2 rel |
|---|---|---|---|---|
| all added | 31,412 | 1.02 | 32,255 | 0.81 |
| via flags = 1 | 759 | 0.44 | 901 | 0.55 |
| via peakSB | 30,446 | 1.03 | 31,087 | 0.81 |
| via other (sky-clean) | 207 | 1.22 | 267 | 1.04 |
| peakSB, prominence < 4 | 10,264 | 0.52 | 16,440 | 0.54 |
| peakSB, prominence ≥ 4 | 20,182 | 1.28 | 14,647 | 1.09 |
| prominence 0–2 | 1,767 | 0.20 | 3,404 | 0.17 |
| prominence 2–3 | 3,564 | 0.37 | 6,021 | 0.44 |
| prominence 3–4 | 5,296 | 0.70 | 7,575 | 0.77 |
| prominence 4–5 | 5,884 | 0.94 | 6,916 | 1.00 |
| prominence 5–7 | 8,409 | 1.29 | 5,310 | 1.17 |
| prominence 7–10 | 4,679 | 1.47 | 2,114 | 1.19 |
| prominence ≥ 10 | 1,805 | 1.63 | 857 | 1.04 |
| qfit 0.2–0.4 | 230 | −0.11 | 363 | 0.17 |
| qfit 0.4–0.6 | 288 | −0.05 | 395 | 0.29 |
| qfit 0.6–1 | 12,394 | 1.12 | 8,950 | 0.94 |
| qfit ≥ 1 | 18,476 | 0.98 | 22,522 | 0.76 |

Nearly every addition enters through the peakSB test.  Their realness falls
with prominence: below prominence 3 the additions are confirmed at 0.2–0.4
of the rate of base-kept stars of the same flux.  The prominence guard on the
peakSB branch in the prominence-keep PR (#1018, guard 4) removes the
prominence < 4 additions; the remaining 20,182 Brick and 14,647 Sgr B2
additions have rel 1.28 and 1.09.  The two branches are therefore intended to
land together.  The 518 Brick additions with qfit 0.2–0.6 are at chance
(rel −0.11 and −0.05); they are 1.6% of the additions.

![](added_brick_snr.png)

Random added sources in Brick F182M, four per bin of distance to the
nearest saturated star (2″ stamps).  Per source: F182M data with the current
catalog (cyan dots), other added sources (orange circles) and saturated
stars (red ×); the production m7 residual (the current final residual); the
production m6 residual, in which every m6 fit, the added source included, is
subtracted; the F200W image of the independent visit.  Green labels have an
F200W counterpart within 60 mas: 6 of the 12 drawn here, and 0.48 of all
28,962 added sources in the F200W footprint (chance 0.07).  Row 2 left: a
faint star left in the current m7 residual and removed by the m6 fit.  The
red-labelled sources include one beside a bright star (row 2 right), one on
a diffraction spike (row 5 right) and faint isolated peaks (rows 4 and 6,
right).  `scripts/` holds the replay (`vet_variant.py`), the comparison
(`compare.py`), the realness slices (`slices.py`) and this gallery
(`added_gallery.py`).

![](added_sgrb2_snr.png)

The same for Sgr B2 F187N, with the same-visit F182M image as reference.

**Sky-clean floor.**  The sky-clean tier keeps sources on clean sky
regardless of `qfit`, with S/N ≥ 3 as its only fit-quality cut.  Applying the
propagated S/N to that floor as well admitted 298 Brick F182M sources (678
F212N, 344 F405N) with median per-frame S/N 2.6, so this branch leaves
`sky_clean_snr_min` on `flux / flux_err`
(`test_sky_clean_floor_stays_per_frame`).

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
- The bright-isolated keep (`snr_high_keep`) stays on the per-frame S/N
  (`test_bright_isolated_keep_stays_per_frame`), and so does the m7
  cross-band seed confirmation (`--manual-crossband-seed-snr-min`), which
  force-fits its positions in every band.
- `flux_err_prop` propagates the per-frame formal errors as independent.
  Terms common to every frame (the shared background model, the shared
  neighbour model, the shared seed position) do not average down; they are
  largest for faint stars on structured background, which is where this
  branch adds sources.  The realness table above measures the net effect.
