# Satstar finder: spurious DQ-SATURATED flags → fake-star clusters

**TL;DR:** The saturated-star finder trusted the DQ `SATURATED` flag with **no
data-value check**. Many DQ-`SATURATED` pixels are spurious (persistence, JUMP
mis-tag, bad pixels). The finder invented a "saturated star" at each and
extrapolated a huge flux from a faint pixel. These fake satstars bypass every
downstream quality gate. Fix = a per-filter **data floor** in
`find_saturated_stars` (`_SATSTAR_DATA_FLOOR`, env `SATSTAR_DATA_FLOOR`).

## Symptom
W51 F480M cataloged a "runaway cluster" of fake sources around bright/extended
sources — e.g. around 19:23:44.49 +14:29:47 (an *extended* source, not a star)
and the real cluster near +14:29:56. The rendered mergedcat MODEL showed several
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
4. `find_saturated_stars`/`get_saturated_stars` apply **no** data floor (the
   `_NIRCAM_SAT_DATA_FLOOR` in `cataloging.py` is only for the photometry mask).

## Why nothing downstream caught it (important gotcha)
**Satstars bypass the daophot fit.** `replace_saturated` *injects* a row with the
satstar flux and leaves `qfit`/`cfit`/`nmatch`/`dra` at defaults — so the fake
had **`cfit = 0`, not the wildly-negative value you'd expect** for model≫data.
The "model==catalog invariant" then FORCES every `replaced_saturated` row back
into the vetted catalog, bypassing `_filter_extended_emission` and the overshoot
QC. So a fake satstar with model peak ≫ data peak survives untouched.

Corollary: **do not** try to reject these with `cfit`/`qfit`/overshoot filters —
those never run for satstars. And **do not** blanket-reject `model > data`: that
is *expected* for genuine saturated stars (the data core is clipped/NaN while the
model reconstructs the true, higher flux). The clean discriminator is the DATA
value at the flag (faint = spurious), not the model/data ratio.

## The fix
`find_saturated_stars(..., sat_data_floor)`: after the initial connected-component
labeling (before spike-merge/edge logic), drop each DQ-`SATURATED` component
UNLESS
  (a) its data — or its 5px charge-migration wings — rises above the floor, or
  (b) it overlaps a NaN-variance (`VAR_POISSON` NaN) *unrecoverable* core (a
      genuine deep-saturated star reads low/NaN in the core).
`get_saturated_stars` resolves the floor: explicit arg > env `SATSTAR_DATA_FLOOR`
> per-filter `_SATSTAR_DATA_FLOOR` default > 0 (off → unchanged for unlisted
filters). Genuine *moderate* stars that fall below the floor are NOT lost — they
are cataloged by the normal daophot channel; only the satstar channel skips them.

## Dead ends we tried first (don't repeat)
- **Satstar consolidation dedup radius** (0.15→0.5"): real but partial — collapses
  per-frame position scatter, but the fakes remained (they're a *count* problem,
  not a dedup problem).
- **Median-position consolidation + detection guard** (approach A) and **coadd-core
  peak cap** (approach B): both regression-safe and marginally helpful, but neither
  removes the fakes — they treat the symptom (bad subtraction) not the cause (the
  satstar shouldn't exist). Peak-cap does usefully flatten the over-sub pit if you
  want the subtraction safer, but it is not the fix for the fake count.

## Infra gotchas learned along the way
- **Run jobs from a NEUTRAL cwd.** `python -m jwst_gc_pipeline...` puts the cwd on
  `sys.path[0]`; if you `cd` into the main repo before `sbatch`, the job imports
  the main repo and **shadows `PIPE_ROOT`/`PYTHONPATH`** (worktree code silently
  never runs). Submit from e.g. `/orange/adamginsburg/jwst/w51`.
- **The repo auto-sync can drop/rewrite local commits.** Verify with
  `git merge-base --is-ancestor <sha> HEAD`, cherry-pick back if dropped, and grep
  the change in the pinned worktree file before submitting.
- The consolidated satstar cache is keyed by NSATSRC + SATDDUPR; it does **not**
  rebuild on a data-floor change, so clear it (or use a fresh cutout label) when
  A/B-testing the floor.
