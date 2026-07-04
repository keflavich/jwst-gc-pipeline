# Brick Pipeline Performance Report

Where the wall-clock goes, measured from the current-code Brick 1182 re-reduction
+ re-catalog (July 2026). Numbers are from SLURM `sacct` elapsed times and the
per-line ISO timestamps in the job logs under
`/blue/adamginsburg/adamginsburg/logs/jwst/`.

**Reference run:** F444W (LW), proposal 1182 field 004, 48 exposures, 8 parallel
workers.
- reduction: `reduce_reduce_36306458_*.out`
- cataloging (m12→m6): `catalog_brick-catalog_36240928_3.out` (elapsed 1d 09:30:58)

---

## TL;DR — where the time goes

| stage | time (LW / F444W) | note |
|---|---|---|
| **Reduction (imaging)** | ~1 h | fast; SW ~4–5 h |
| **Cataloging m12→m6** | **33.5 h** | the whole ballgame |
| m7 (cross-band merge, per-proposal) | ~16 h | one job, all filters |
| m8 (forced cross-band fill, per-proposal) | ~14 h | one job, all filters |

**Cataloging dominates by ~30×.** Inside cataloging, essentially all the time is
**per-frame photometry fitting**. The catalog merge (chunked) is now trivial
(<5 min/phase). The single biggest block is **m12, the first pass (18 h)**, and
within m12 a **dense-frame long tail** (~13 h where the densest Brick-center
frames fit while the rest sit idle — worker load imbalance).

---

## 1. Imaging (reduction)

Per-filter Image3 wall-clock (`sacct`, job 36306458):

| filter | frames | reduction |
|---|---|---|
| F115W (SW) | 192 | 4:12 |
| F200W (SW) | 192 | 4:55 |
| F444W (LW) | 48 | 0:56 |

Per **module** Image3 chain (F115W, ~26 min/module), from stpipe step
timestamps:

| step | duration |
|---|---|
| tweakreg | skipped (skip=True; alignment done up-front by `fix_alignment`) |
| skymatch (`method='match'`) | ~4.5 min |
| outlier_detection | ~10 min |
| resample | ~10 min |
| source_catalog | ~1 min |

Times three modules (`nrca`, `nrcb`, `merged`) plus destreak and the
`realign_to_catalog` up-front alignment → the ~4 h SW total. SW is ~4–5× the LW
cost because SW has 8 detectors/exposure (vs 2 LW) → 192 vs 48 frames, plus
per-frame destreak.

**Reduction is not the bottleneck.**

---

## 2. Satstar "cataloging"

Saturated-star models (`*_satstar_model.fits`) are built per-frame by
`remove_saturated_stars` (`crowdsource_catalogs_long.py:1711`) during the first
cataloging pass, then **cached on disk and reused** across all later phases and
across re-catalog reruns. In the reference run they were reused from a prior
build (file mtimes predate the run), so satstar contributed ~0 to the 33.5 h.

The per-frame `Satstar-artifact filter` gate seen throughout the photometry logs
is a millisecond-scale post-fit cull, not the satstar fit.

**Satstar is not a steady-state bottleneck** (it is paid once, then cached). It
only matters on a cold run with no cached models.

---

## 3. Per-iteration breakdown (F444W, 48 frames, 8 workers)

Phase boundaries from the `mergedcat:` and `combine_singleframe_chunked:`
timestamps. "fit" = per-frame photometry across all 48 frames; "merge" =
`combine_singleframe_chunked` (spatial-tile parallel merge + dedup).

| phase | whole | per-frame fit | merge | per-frame (CPU-h) | what it is |
|---|---|---|---|---|---|
| **m12** | **18.2 h** | 17.7 h | 0.5 h | ~3.0 (skewed) | first pass: detect + fit from scratch |
| m3 | 3.47 h | 3.44 h | 2 min | 0.57 | re-fit, seeded |
| m4 | 3.62 h | 3.56 h | 4 min | 0.59 | re-fit, seeded |
| m5 | 3.78 h | 3.74 h | 3 min | 0.62 | re-fit, seeded |
| m6 | 3.82 h | 3.79 h | 1 min | 0.63 | re-fit, seeded |
| finalize | 0.6 h | — | — | — | vetting + write |
| **total** | **33.5 h** | | | | |

Per-proposal (all filters in one job):

| phase | time | scope |
|---|---|---|
| m7 (cross-band merge) | ~16 h | one job over all filters (job 36013572) |
| m8 (forced cross-band fill) | ~14 h | one job over all filters (job 35882156) |

### Reading the table
- **Merge is solved.** The chunked spatial-tile merge is <5 min even on
  ~2.1 M detections/phase. Not worth further work.
- **m12 is 5× the cost of m3–m6.** m12 detects and fits every source from
  scratch; m3–m6 re-fit from the previous phase's seed catalog and converge far
  faster. Same 48 frames, same fitter — the difference is from-scratch detection
  + first-fit vs seeded re-fit.
- **Per-image is highly non-uniform.** m12's 17.7 h of fit is not 48 equal
  frames: the log shows fast waves finishing in minutes, then one ~13 h stretch
  with no output — the densest Brick-center frames fitting while the other
  workers are idle. m3–m6 are ~35 min/frame and much more even.

---

## 4. Bottleneck analysis

1. **Per-frame photometry fitting is ~99% of cataloging.** Everything else
   (detection merge, satstar, vetting, I/O) is minor.
2. **m12 first-pass dominates** (18 h of 33.5 h). Seeded re-fits (m3–m6) are 5×
   cheaper, so anything that makes m12 start from a seed instead of from scratch
   pays back hugely.
3. **Worker load imbalance on dense frames.** Frames are dispatched in
   chunk/order, not by cost. The Brick-center frames carry most of the sources;
   when they land in the same wave, 7 workers finish and idle while 1 grinds for
   hours. This is the ~13 h tail.
4. **SW is 2–4× the LW cataloging cost** (192 vs 48 frames), which is why
   F115W/F200W were still running >1.5 days when F444W finished at 33.5 h.

---

## 5. Recommendations (ranked by expected payoff)

1. **Cost-balanced frame scheduling.** Sort frames by source count (or a cheap
   density proxy) and hand them to workers longest-first (LPT). Removes the
   idle-workers-behind-one-dense-frame tail; could cut m12 wall-clock materially
   without more CPUs.
2. **Seed m12.** Give the first pass a coarse detection/seed (e.g. the deep i2d
   detection already computed) so it behaves like the 5×-faster m3–m6 re-fit
   instead of a from-scratch fit.
3. **Per-frame sharding across nodes.** `--manual-frame-shard` already exists;
   split the frame set across array tasks so the dense frames run on separate
   nodes in parallel rather than serially inside one 8-worker job. Biggest lever
   for SW (192 frames).
4. **More workers for m12 specifically.** The dense tail is effectively serial at
   8 workers; m12 alone benefits from a wider pool (m3–m6 less so).
5. **Leave the merge alone.** The chunked merge is already <5 min/phase.

---

*Generated from the July 2026 Brick 1182 current-code run. To refresh: re-read
the `mergedcat:` / step timestamps in the cataloging and reduction logs under
`/blue/adamginsburg/adamginsburg/logs/jwst/` for the run of interest.*
