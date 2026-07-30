# Brick Pipeline Performance Report

Where the wall-clock goes, measured from the current-code Brick 1182 re-reduction
+ re-catalog (July 2026), cross-checked against earlier Brick cataloging runs.
Numbers are from SLURM `sacct` elapsed times and the per-line ISO timestamps in
the job logs under `/blue/adamginsburg/adamginsburg/logs/jwst/`.

**Runs used:**
- reduction: `reduce_reduce_36306458_*.out`
- cataloging (stalled): `catalog_brick-catalog_36240928_{0,1,3}.out`
- cataloging (clean baseline, no stall): `catalog_brick-catalog_36013569_0.out`
  (SW, 192 frames), `catalog_brick-catalog_36013571_3.out` (LW, 48 frames)

---

## TL;DR

| stage | clean time | note |
|---|---|---|
| **Reduction (imaging)** | ~1 h LW / ~4–5 h SW | Image3 per module; not the bottleneck |
| **Cataloging (full, LW)** | **~10.5 h** | clean run 36013571_3 |
| **Cataloging (full, SW)** | **~18–20 h** | ~2× LW (192 vs 48 frames) |
| m7 (cross-band merge, per-proposal) | ~16 h | one job, all filters |
| m8 (forced cross-band fill, per-proposal) | ~14 h | one job, all filters |

Cataloging dominates reduction, and inside cataloging it is **per-frame PSF
fitting** that costs the time. *Every* phase runs its own per-frame `daofind` +
fit — m12 is not "unseeded." m12 costs the most only because it is **two fit
passes** (iter1, then iter2 on the residual) instead of one, so it is ~2× a
single seeded phase, not 5×. Detection (`daofind`) is cheap; the fit is the
cost.

### Important correction: the "33.5 h" F444W run was an I/O anomaly, not the norm

The first draft of this report measured job `36240928` and saw m12 take 18 h with
a ~13 h "tail," and wrongly attributed it to dense Brick-center frames. **That is
not supported** — the Brick stellar density is nearly uniform, and the data prove
it. That 18 h was a **one-time I/O stall**, not compute (see §4). The clean-run
numbers above are the real cost.

---

## 1. Imaging (reduction)

Per-filter Image3 wall-clock (`sacct`, job 36306458):

| filter | frames | reduction |
|---|---|---|
| F115W (SW) | 192 | 4:12 |
| F200W (SW) | 192 | 4:55 |
| F444W (LW) | 48 | 0:56 |

Per **module** Image3 chain (F115W, ~26 min/module), from stpipe step timestamps:

| step | duration |
|---|---|
| tweakreg | skipped (skip=True; alignment done up-front by `fix_alignment`) |
| skymatch (`method='match'`) | ~4.5 min |
| outlier_detection | ~10 min |
| resample | ~10 min |
| source_catalog | ~1 min |

× three modules (`nrca`, `nrcb`, `merged`) + destreak + `realign_to_catalog`.
SW is ~4–5× LW because SW has 8 detectors/exposure (vs 2 LW) → 192 vs 48 frames.

**Reduction is not the bottleneck.**

---

## 2. Satstar "cataloging"

Saturated-star models (`*_satstar_model.fits`) are built per-frame by
`remove_saturated_stars` (`crowdsource_catalogs_long.py:1711`) during the first
cataloging pass, then **cached on disk and reused** across later phases and across
re-catalog reruns. In the reference run they were reused from a prior build, so
satstar contributed ~0. The per-frame `Satstar-artifact filter` gate in the
photometry logs is a millisecond-scale post-fit cull, not the fit.

**Satstar is not a steady-state bottleneck** (paid once, then cached).

---

## 3. Per-iteration breakdown

Same 48 (LW) / 192 (SW) frames are fit every phase, each with its own per-frame
`daofind` + PSF fit. m12 runs **two** passes (iter1, then iter2 on the residual);
m3–m6 run one pass each, seeded by that phase's `daofind` + the projected
previous/cross-band catalog. Clean-run per-phase (no I/O stall):

| phase | LW (48 fr) | SW (192 fr) | what it is |
|---|---|---|---|
| **m12** | ~2.4 h | ~9 h | daofind+fit ×2 (iter1 + iter2-on-residual) |
| m3 | ~0.2–2 h | ~0.4–2 h | daofind+fit ×1, seeded |
| m4 | ~0.2–2 h | ~0.4–2 h | daofind+fit ×1, seeded |
| m5 | ~2 h | ~2 h | daofind+fit ×1, seeded |
| m6 | ~0.2–1 h | ~1 h | daofind+fit ×1, seeded |
| finalize | ~0.6 h | ~0.6 h | vetting + write |
| **full total** | **~10.5 h** | **~18–20 h** | |
| m7 (cross-band, per-proposal) | ~16 h (all filters, one job) | |
| m8 (forced fill, per-proposal) | ~14 h (all filters, one job) | |

**Per-image:** with 8 workers, a phase runs in `frames/8` waves. Clean m3–m6 show
**~0.5 h/wave, and every wave is within 0.06 h of every other** — i.e. the frames
cost the same. There is no dense-frame subset.

**Merge is solved.** The chunked spatial-tile merge (`combine_singleframe_chunked`)
is <5 min even on ~2 M detections/phase. Not worth further work.

---

## 4. The I/O stall (why the F444W run took 33.5 h)

Per-frame completion timestamps, gap analysis:

| run | frames | m12 (m1) span | m12 max single gap | m3 max gap |
|---|---|---|---|---|
| 36240928_0 (F115W) | 192 | 23.6 h | **13.8 h** | 0.06 h |
| 36240928_1 (F200W) | 192 | 18.2 h | **14.8 h** | 0.06 h |
| 36240928_3 (F444W) | 48 | 17.9 h | **11.8 h** | 0.5 h |
| 36013569_0 (F115W, earlier) | 192 | **8.4 h** | **0.38 h** | 0.06 h |
| 36013571_3 (LW, earlier) | 48 | **2.1 h** | **0.42 h** | 0.36 h |

The diagnosis:

1. **Frame density is uniform.** In every run, m3–m6 finish with max inter-frame
   gaps of ~0.06 h — all frames cost the same. If a subset of frames were genuinely
   dense, they would be slow in *every* phase; they are not. The earlier runs'
   m12 is also uniform (max gap ~0.4 h).
2. **The 33.5 h run had a single 12–15 h stall in m12**, and it hit **all three
   array tasks (F115W, F200W, F444W) at once**. A simultaneous multi-hour stall
   across independent tasks is not a property of any one frame — it is a **shared
   resource**: the node / NFS / scratch filesystem hung.
3. **m12 is the peak-I/O phase.** It is the only phase that reads every `_crf`
   frame fresh, loads the PSF grids, and writes every per-frame residual/model
   FITS for the first time. Three array tasks × 8 workers = 24 processes all
   hammering shared `/orange` + `/blue` at once during m12 is what tipped it over;
   m3–m6 reuse cached inputs and write less, so they never stalled.
4. **The earlier runs prove m12 without the stall is ~2 h (LW) / ~8 h (SW).** The
   ~12–15 h is pure overhead from one bad I/O episode.

So: cataloging is per-frame-photometry-bound (§3), but the specific 33.5 h figure
was an **I/O incident**, not the steady-state cost, and had nothing to do with
Brick-center density.

---

## 5. Recommendations (ranked)

1. **Fix the m12 I/O contention.** This is the real lever for the observed
   blow-ups.
   - Build per-frame products on **node-local scratch** (`$SLURM_TMPDIR`) and
     copy the finished file to shared `/orange` `/blue` in a single sequential
     write, instead of doing the many-small-write FITS/table construction
     directly on the shared mount. Implemented via `write_via_local_scratch`
     (applied to the per-frame catalog write); extend to the per-frame
     residual/model image writers as well.
   - **Stagger array-task starts** (or cap concurrent tasks per node) so N tasks ×
     8 workers don't all peak-I/O the same mount simultaneously.
   - Consider fewer workers for m12 specifically, or spread the frame set across
     nodes, to cut peak I/O concurrency.
2. **Per-frame sharding across nodes** (`--manual-frame-shard` exists) — biggest
   lever for SW (192 frames), and it also spreads I/O across nodes.
3. **Leave the merge alone.** The chunked merge is already <5 min/phase.
4. **Do NOT special-case "dense" frames.** Density is uniform; cost-by-density
   scheduling would buy nothing.
5. **Seeding won't help m12.** Every phase already runs `daofind`, and detection
   is cheap next to the PSF fit — the cost is fitting every source, not finding
   them. m12's extra cost is its second fit pass (iter2), not a lack of seed.
   Cutting m12 means cheaper fitting (fewer sources / faster fitter) or dropping
   a pass, not "seeding."

---

*Generated from the July 2026 Brick 1182 current-code run + earlier-run baselines.
To refresh: parse the `mergedcat:` / stpipe step timestamps in the cataloging and
reduction logs under `/blue/adamginsburg/adamginsburg/logs/jwst/`, and always
check per-frame gap uniformity before attributing cost to "hard" frames.*
