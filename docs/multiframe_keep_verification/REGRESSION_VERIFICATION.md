# Multi-frame-confirmation keep — verification & regression notes

> **Figures moved:** image files are no longer tracked in this repo; they live in the Overleaf astrometry-paper project (https://www.overleaf.com/project/6a521006b63a11a7e0d80fa0) under `figures/keep_verification/` (same filenames).

This documents the depth benchmark and the extended-emission regression checks for
the recover tier + multi-frame-confirmation keep (Hosek `ndet≥3` style) added to
`_filter_extended_emission`.

## What & why

Benchmarking the Arches F212N NRCA4 catalog against Matt Hosek's UCLA pipeline
showed we were ~1–2 mag shallower. Investigation split the deficit in two: a
**survival** loss (fixable here) and a **detection** floor (residual). Our
per-frame detection already finds ~68% of Hosek's stars (a DAOStarFinder pass on
the deep coadd is *worse*, 43%, from crowding), but the first extended-emission
vetting then drops faint *detected* stars (completeness 0.64 → 0.42). Hosek keeps
those faint marginal stars via `ndet≥3`; we were dropping them per-vetting on
single-catalog `qfit`/`snr`. This change recovers the survival loss. The ~32% of
Hosek we never detect (faint / crowded / spike-blocked) is the detection floor and
is NOT addressed here — see the remaining-miss anatomy at the end.

The fix (opt-in, default OFF): keep any source with `nmatch ≥ N` and
`qfit ≤ cap`, regardless of the qfit/snr cuts. An optional position-stability
guard (`--manual-ext-nmatch-confirm-maxpos-mas`) rejects position-*unstable*
spurious (cosmic-ray / noise coincidences). **It does NOT reject extended
emission** — emission is fixed on-sky, so an emission-knot daofind bump also
repeats across dithers *with a stable centroid* and passes the guard. So this is
a **star-field tool**: a clean win on star-dominated fields (Arches), but it
re-admits emission on emission-dominated fields (see the cutout section) and must
be left OFF there (its default). This scope limit is the honest result of the
verification below.

## Depth benchmark vs Hosek (Arches F212N NRCA4, 74,896 Hosek sources)

Completeness ladder (fraction of Hosek recovered, full pipeline m1–m6):

| catalog | ours in footprint | matched to Hosek | completeness | Hosek-only |
|---|---|---|---|---|
| baseline (no recover) | 32,343 | 28,931 | 0.386 | 45,965 |
| recover tier | 36,963 | 32,502 | 0.434 | 42,394 |
| **+ multi-frame keep** | **51,102** | **43,650** | **0.583** | **31,246** |

**+14,719 real Hosek matches over baseline (0.386 → 0.583).** Positions stay
tight (median separation 3.1 mas, robust 2.2–2.8 mas). Ours-only = 7,452
(spurious, acceptable on a dense field where Hosek carries spurious too). The
remaining ~31k Hosek-only is ≈ half his own artifacts (spikes/emission) plus his
sub-3-frame faint detections — near the fair-match ceiling.

> *Figure:* `figures/keep_verification/arches_hosek_nmatch_m6.png` (Overleaf)

### Detection is not the bottleneck — survival is

Completeness vs Hosek instrumental magnitude. We fully match Hosek at the bright
end (m<−6 ≈ 1.0). `m2-raw` (green, all detections) stays high even at the faint
end — we *detect* the faint stars — but the final catalogs collapse; that gap is
the detected-but-dropped population the multi-frame keep recovers. The deep-coadd
daofind (red) is worse than per-frame, so detecting harder does not help.

> *Figure:* `figures/keep_verification/completeness_vs_hosek_by_mag.png` (Overleaf)

## Regression verification

### Unit tests — `test_recover_tier_vetting.py` (18 pass)
- Recover tier: no-op default; admit blended real; reject emission/low-snr/near-satstar;
  sloped-prominence admit/reject on a synthetic i2d; persisted `prominence`/`peak_sb` columns.
- Multi-frame keep (5 new): default-off; admit faint multi-frame star; respect qfit
  ceiling; require ≥N frames; position guard rejects a synthetic wandering (100 mas)
  source and keeps the tight (3 mas) one. (Note: this exercises the guard mechanism;
  it does NOT imply emission is rejected — real extended emission has a *stable*
  centroid, see the cutout section.)

### Full photometry suite
`pytest jwst_gc_pipeline/photometry/tests/` → **205 passed** (0 failures). The
merge, satstar, box-artifact, off-FOV, and integration regressions all stay green.

### Extended-emission cutouts — the multi-frame keep is NOT emission-safe

All 5 cutouts re-run end-to-end with recover + multi-frame keep *enabled*
(`--manual-ext-nmatch-confirm=3 --manual-ext-nmatch-confirm-qfit-max=0.6
--manual-ext-nmatch-confirm-maxpos-mas=20`), compared to the recover-off baseline:

| cutout (type) | vetted base→nm | added | verdict |
|---|---|---|---|
| low_background (star field) | 46 → 66 | +20 | real stars (star-dominated field) |
| pillar_head (sickle emission) | 12 → 15 | +3 | includes ~1+ fake emission source |
| pillar_with_satstar (sickle) | 37 → 53 | +16 | mixed |
| **w51 darkfilament F480M (emission)** | 106 → **186** | **+110** | **mostly EMISSION** |
| w51 darkfilament F187N (emission) | 69 → 84 | +15 | mixed |

**On emission-heavy fields the multi-frame keep RE-ADMITS EMISSION.** The W51
dark-filament +110 sources are emission knots, not stars: median prominence 0.9
(95% below 3), sitting on bright background — the same low-prominence emission the
recover prominence gate rejects. pillar_head gained a fake as well (it got worse,
not "unchanged").

**Why the position guard fails:** on-sky emission is spatially *fixed*, so a
daofind bump on an emission knot has a *stable* centroid across dithers
(measured posscatter median ~0 mas) and passes `nmatch-confirm-maxpos-mas`. The
guard assumption ("emission centroids wander") is wrong for compact emission.

The bg medians going *down* on these fields is NOT a safety pass — it is the
emission being (wrongly) subtracted into the model, which lowers the residual.

> *Figure:* `figures/keep_verification/w51_darkfilament_emission_safety.png` (Overleaf)

> *Figure:* `figures/keep_verification/pillar_head_emission_safety.png` (Overleaf)

**Scope of the feature (revised):** the multi-frame keep is a *star-field* tool.
On star-dominated fields (Arches: +14,719 real Hosek matches; sickle
low_background: real stars) it is a large, clean win. On emission-dominated fields
(W51 dark filament; the sickle pillars to a lesser degree) it re-admits emission
and should be left OFF (its default) or paired with a real emission discriminator
— prominence is the one that separates them, but a flat prominence floor also
drops faint real stars (the tension the recover sloped gate manages). A robust
per-source star-vs-emission cut for the multi-frame tier is future work.

### Recover-tier prominence boundary (context)

The recover tier's sloped `(qfit, log prominence)` gate, fit on labelled sickle +
W51 cutouts (real = blue, emission = red). Keeps 63/69 real at 5/142 emission.

> *Figure:* `figures/keep_verification/recover_qfit_prominence_boundary.png` (Overleaf)

## Reproduce

```
# depth ladder + detection ceiling
python .../depth_ceiling_test.py
# emission-cutout regression (recover + nmatch), base vs nm
sbatch cutout_nm.sbatch            # 5 cutouts, --manual-ext-nmatch-confirm=3 ...
# full-frame Arches (definitive)
sbatch arches_nmatch.sbatch        # slope-5.6 recover + nmatch=3 + maxpos=20
```

Default OFF (`--manual-ext-nmatch-confirm 0`) is byte-identical to prior behaviour.

## Anatomy of the remaining Hosek-only (~28.6k), recover+nmatch m6

Diagnostic (`diagnose_remaining_misses.py`):

- **(c) recovery vs OUR nmatch** — the keep works: `our_nmatch≥3` recovers 76–99%.
  The residual misses are dominated by `our_nmatch=0` (24,001 — never detected,
  0.16 recovery) and `our_nmatch=1–2` (8,907 — sub-3-frame, below the keep).
- **(d) satstar spikes** — 11,416 of 28,648 (40%) are within 1.5″ of a saturated
  star (unmeasurable spike zones / Hosek artifacts). Significant but not the majority.
- **(b) recovery vs Hosek ndet** — rises 0.35 (ndet=3) → 0.85 (ndet=11): we recover
  Hosek's robust stars and miss 65% of his marginal `ndet=3` (his minimum confidence).
- **(a) spatial** — the misses cluster in the densest sub-region (crowding-limited).

**Conclusion:** the multi-frame keep closed the *survival* loss (detected-but-dropped
faint stars). The remaining gap is the *detection* floor — ~24k never detected
(faint / crowded / spike-blocked; a deep-coadd daofind is worse, 0.43, so they are
genuinely hard), ~9k sub-3-frame, ~40% in spike zones, plus Hosek's own marginal
ndet=3. Closing further needs better detection in crowded/spike regions, not more
vetting relaxation.

> *Figure:* `figures/keep_verification/remaining_hosek_only_anatomy.png` (Overleaf)
