# Satstar finder: spurious DQ-SATURATED flags → fake-star clusters

**TL;DR:** The saturated-star finder trusted the DQ `SATURATED` flag alone. Many
DQ-`SATURATED` pixels are spurious (persistence, JUMP mis-tag, bad pixels). The finder invented a "saturated star" at each and
extrapolated a huge flux from a faint pixel. These fake satstars bypass every
downstream quality gate. Fix = a per-filter **data floor** in
`find_saturated_stars` (`_SATSTAR_DATA_FLOOR`, env `SATSTAR_DATA_FLOOR`).

## Symptom
W51 F480M cataloged a "runaway cluster" of fake sources around bright/extended
sources — e.g. around 19:23:44.49 +14:29:47 (an *extended* source) and the real
cluster near +14:29:56. The rendered mergedcat MODEL showed several
bright blobs where the data has one blended source; the RESIDUAL was a black/
white over-subtraction mess that daofind then fit MORE fakes onto.

## How it was diagnosed (so you can repeat it)
1. Counted catalog sources within 1" of the bright source: 26–39, most flagged
   `is_saturated`/`replaced_saturated`.
2. Sampled the DQ `SATURATED` flag + SCI data at a specific fake
   (19:23:44.499 +14:29:56.24, flux 104948): **SATURATED in 8/8 frames but data
   peak ~127 MJy/sr** (real saturation is thousands; the extended source is
   ~4441). → the flag is spurious.
3. Data-peak distribution of the consolidated satstars around the two sources:
   **~80% had i2d data < 500** (spurious), only ~2 each were genuinely bright.
4. `find_saturated_stars`/`get_saturated_stars` accept every flagged component
   (the `_NIRCAM_SAT_DATA_FLOOR` in `cataloging.py` serves the photometry mask
   alone).

## How it reached the catalog (important gotcha)
**Satstars bypass the daophot fit.** `replace_saturated` *injects* a row with the
satstar flux and leaves `qfit`/`cfit`/`nmatch`/`dra` at defaults — so the fake
carried the default **`cfit = 0`**, where a real fit with model≫data reads
wildly negative. The "model==catalog invariant" then FORCES every
`replaced_saturated` row back into the vetted catalog, bypassing
`_filter_extended_emission` and the overshoot QC. So a fake satstar with model peak ≫ data peak survives untouched.

Corollary: **do not** try to reject these with `cfit`/`qfit`/overshoot filters —
those filters skip satstars. And **do not** blanket-reject `model > data`: that
is *expected* for genuine saturated stars (the data core is clipped/NaN while the
model reconstructs the true, higher flux). The clean discriminator is the DATA
value at the flag (faint = spurious).

## The fix
`find_saturated_stars(..., sat_data_floor)`: after the initial connected-component
labeling (before spike-merge/edge logic), drop each DQ-`SATURATED` component
UNLESS
  (a) its data — or its 5px charge-migration wings — rises above the floor, or
  (b) it overlaps a NaN-variance (`VAR_POISSON` NaN) *unrecoverable* core (a
      genuine deep-saturated star reads low/NaN in the core).
`get_saturated_stars` resolves the floor: explicit arg > env `SATSTAR_DATA_FLOOR`
> per-filter `_SATSTAR_DATA_FLOOR` default > 0 (a floor of 0 leaves unlisted
filters as before). Genuine *moderate* stars below the floor are cataloged by
the normal daophot channel; only the satstar channel skips them.

## Dead ends we tried first (don't repeat)
- **Satstar consolidation dedup radius** (0.15→0.5"): real but partial — collapses
  per-frame position scatter, and the fakes remained. They are a *count* problem.
- **Median-position consolidation + detection guard** (approach A) and **coadd-core
  peak cap** (approach B): both regression-safe and marginally helpful, and the
  fakes remained. They treat the symptom (bad subtraction) rather than the cause
  (the satstar's existence). Peak-cap does flatten the over-sub pit if you want
  the subtraction safer; the fake count stays.

## Infra gotchas learned along the way
- **Run jobs from a NEUTRAL cwd.** `python -m jwst_gc_pipeline...` puts the cwd on
  `sys.path[0]`; if you `cd` into the main repo before `sbatch`, the job imports
  the main repo and **shadows `PIPE_ROOT`/`PYTHONPATH`** (the job silently runs
  main-repo code). Submit from e.g. `/orange/adamginsburg/jwst/w51`.
- **The repo auto-sync can drop/rewrite local commits.** Verify with
  `git merge-base --is-ancestor <sha> HEAD`, cherry-pick back if dropped, and grep
  the change in the pinned worktree file before submitting.
- The consolidated satstar cache is keyed by NSATSRC + SATDDUPR alone, so a
  data-floor change reuses the old cache. Clear it (or use a fresh cutout label)
  when A/B-testing the floor.
