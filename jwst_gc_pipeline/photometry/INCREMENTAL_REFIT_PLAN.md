# Incremental refit — reuse unchanged per-source fits across manual phases

**Status (2026-07-12):**
- ✅ **Mechanism proven + merge-ready:** the pure decision logic
  (`incremental_refit.py`) and, crucially, the **core numerical invariant** it
  relies on are unit-tested — an independent (`group=False`) single-source PSF fit
  is **bit-identical** when the background changes only outside the source's
  footprint (`test_incremental_refit.py::test_far_bg_change_leaves_independent_fit_bit_identical`),
  and DOES change for a localized in-footprint change (control).
- ⏭️ **Fit-loop wiring is the next, VERIFY-GATED step** (see "Wiring" below). It is
  deliberately NOT in this PR: it is surgery on the core per-frame fit path
  (`do_photometry_step_manual`), including `modsky` reconstruction, and must be
  landed behind `--manual-incremental-refit` (default OFF) and certified by a
  `--manual-incremental-refit-verify` run on a real field (fit both ways, assert
  reused == full) before the fast path is trusted. That verification run cannot be
  done in the authoring session, so the wiring ships separately once it can be run.

## The waste

The manual pipeline runs ~6 phases (`m12, m3, m4, m5, m6, m7`). **Every phase
re-fits every seeded source in all 192 frames** — the dominant cost (~2 h/phase
at 32 workers, profiled on the live brick runs). But between two consecutive
phases most fits are *identical*: nothing that determines them changed.

## When is a per-source fit provably identical between phases?

The default cataloging fit is **`group=False`** — photutils `PSFPhotometry` with
**no grouper**, so **each source is fit independently** over its `fit_shape`
(5×5) window, with a `LocalBackground` annulus out to `localbkg_outer`
(~3.5·FWHM). An independent single-source fit is a deterministic function of
exactly three things:

1. the source's **seed position** (`x_init, y_init`),
2. the **PSF model** (fixed across phases), and
3. the **image data within the source's footprint** = `raw − bg`, over the disk
   of radius `localbkg_outer` around the seed (fit window + localbkg annulus).

`raw` is the same every phase (the frame never changes). The only per-phase input
that varies is the **background map** `bg` (source-masked smoothed bg, rebuilt
from the previous phase's residual; bg subtraction is active for `m5, m6, m7`).
Therefore, for a source that is still seeded at the same position:

> **fit(phase N) == fit(phase N−1)  ⇔  max |bg_N − bg_{N−1}| ≈ 0 over the
> source's footprint disk.**

A **new nearby seed does NOT change** an existing source's independent fit: with
no grouper, the neighbour's flux is already in the data window either way (it is
never *modelled* in the independent fit), so adding it as a seed changes nothing
for the incumbent. Hence — for `group=False` — there is **no group-membership
condition**; the reuse test is purely (same seed) ∧ (bg unchanged over footprint).

(`group=True` couples a group's members, so a seed added to a group changes the
joint fit. Incremental refit **disables itself and falls back to a full fit when
`group=True`** — conservative.)

## Algorithm (per frame, per phase N≥2, group=False)

Inputs available **on disk** (no in-memory threading): phase N's bg map
(`resbg_path`), phase N−1's saved per-frame catalog, and phase N−1's bg map.

1. Load `bg_N` and `bg_{N-1}`; compute `dbg = |bg_N − bg_{N-1}|`.
2. `dirty_px = dbg > bg_delta_thresh`, dilated by `localbkg_outer` (a bg change
   at pixel p taints any source whose footprint reaches p).
3. For each seed of phase N: **reusable** iff
   - it matches a phase-(N−1) *fitted* source within `match_tol_pix` (≤0.01 px —
     the seed is carried forward unchanged, not re-detected off-position), and
   - its footprint disk touches **no** dirty pixel, and
   - its footprint is fully on valid (unmasked) data in both phases.
4. **Fit only the non-reusable seeds** via the normal `_manual_phot_pass` (its
   `init_params = seed[~reusable]`).
5. **Splice**: copy the reused rows verbatim from the phase-(N−1) catalog
   (`splice_reused_rows`), append the freshly-fit rows, reorder to seed order.
   Order/ids reconciled so downstream merge is unaffected.
6. **`modsky` reconstruction (the load-bearing wiring detail).** `_manual_phot_pass`
   returns `modsky` = the rendered model of *the sources it fit*. The subset pass
   renders only the refit sources, so the reused sources' model stamps must be
   **re-rendered from their cached `x_fit/y_fit/flux_fit`** (using the same PSF and
   render path the fitter uses) and **added** to `modsky` — otherwise the residual
   (`data − modsky`) is wrong exactly where reuse happened, corrupting the
   next-phase source-masked bg and cascading. Rendering N sources is far cheaper
   than fitting them, so the saving is preserved. This step is what the
   verify-mode must certify (the residual/model i2d must match the full-fit run).

## Correctness safeguards

- **`--manual-incremental-refit-verify`**: fit BOTH ways (full + incremental) and
  assert the reused rows are bit-identical to the full-fit rows; use the full
  result. Run this on a real field once to certify before trusting the fast path.
  (The unit tests already prove the invariant on synthetic frames — a bg change
  beyond `localbkg_outer` leaves the incumbent fit unchanged to <1e-9.)
- **Conservative thresholds**: `bg_delta_thresh` defaults to a small fraction of
  the frame's pixel noise; `match_tol_pix ≤ 0.01`. A source is refit whenever in
  doubt.
- **Fail-open**: any error in the reuse path (missing prev catalog, shape
  mismatch, NaN bg) falls back to a full fit for that frame — never a silent drop.
- **Disabled** for `group=True`, for phase 1 (no prior), and when the prev
  catalog / prev bg is absent.

## Expected savings

Away from sources whose mask changed, the source-masked smoothed bg is ~identical
phase-to-phase, so `dirty` is a thin halo around changed sources. In late phases
(few new detections) the reusable fraction is high; per-frame photometry time
scales with the *refit* count. Order-of-magnitude: if 60–80 % of sources are
reusable in `m5→m6→m7`, per-frame time in those phases drops ~3–5×, i.e. several
hours per filter. Measured savings are reported by the verify/among-run logs.

## Files

- `incremental_refit.py` — pure `classify_reusable_seeds`, `dirty_bg_mask`,
  `splice_reused_rows` (no I/O; unit-tested).
- `cataloging.py` — `run_manual_pipeline` passes the prev phase label + prev bg
  path per frame; `do_photometry_step_manual` calls the classifier, subsets the
  seed, and splices (all behind `--manual-incremental-refit`).
