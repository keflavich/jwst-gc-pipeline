# gc2211 saturated-star deblending (merged saturated cores)

**Worktree:** `jwst-gc-pipeline-wt-satdeblend` (branch `satstar-deblend-gc2211`).

## Problem
gc2211 GC fields are so crowded that bright stars' SATURATED-DQ cores TOUCH and
merge into one connected component. `saturated_star_finding.find_saturated_stars`
+ `_refine_coms_by_data` fit ONE PSF per component at the bbox-centre, which for a
merged double sits BETWEEN the stars -> both mis-placed and badly subtracted.
Quantified (F200W o023 nrca1): **~29% of saturated components contain >=2 stars.**

## Key finding: frame zero resolves the cores
These exposures have only **ngroup=2** and the cores saturate at group 0. BUT the
`_ramp.fits` carries a **ZEROFRAME** (single reset/first read, ~1 frame time ->
~Ngroup-higher saturation ceiling). At the merged cores the ZEROFRAME peaks at
5k-36k DN, BELOW the ~50k ceiling, so the PSF + diffraction spikes are intact
where the cal slope is a saturated blob. The ZEROFRAME therefore RESOLVES the
individual saturated cores. (It is raw uncalibrated DN, noisy, with a pedestal ->
use for POSITIONS, not flux. The very brightest stars still saturate a SMALL
central ZF region; their spikes still pin the centre.)

## Approach (ZEROFRAME-primary, UNIFIED two-regime deblend) — `deblend_zeroframe.py`
ZF-saturated cores are stored as ZEROS (black holes), so two regimes coexist even
within one frame.  `deblend_blob_zeroframe` handles both:
1. **ZF-SATURATED cores (deep stars).** `invalid = (zf==0)|(zf>=ceiling)|~finite`;
   close + fill-holes, label, take each cluster's CENTROID = one centre per
   ZF-saturated star.  The cluster union is the CLAIMED region.  Robust to
   ring/spike asymmetry (a single big core stays ONE centre); two touching cores
   stay separate while an unsaturated gap divides them (L168).
2. **ZF-RECOVERABLE cores (moderate stars / sparse-shallow frames).** Nearest-valid
   fill (per-hole, no bridging) + Gaussian (~FWHM) smooth + `peak_local_max`.  Peaks
   INSIDE a claimed core are ABSORBED (that core star's own filled flux -> avoids
   the L31 single-core split); peaks OUTSIDE are candidate companions.
3. **CONFIRM** outside-peaks: keep only those with a deduped daophot source within
   `snap_frac*FWHM` (cataloged star) — or, with no catalog, a COMPACT peak
   (`_is_compact`) — to reject Airy-ring/spike/bridge structure of bright stars.
   The brightest peak in a coreless (regime-2) blob is always kept.
4. **SNAP** to daophot for sub-pixel astrometry; FoF-merge within the core radius.
`robust_zf_ceiling()` returns the saturation pile-up value or +inf when cores are
zeros / nothing saturates (robust to the LW 9e8 hot-pixel outlier).

### Validation
`validate_deblend.py` (single frame, panels in `out/`):
- L31 single -> 1, L168 double -> 2, L15/L138 -> 1 (companions unsaturated/marginal);
  whole frame `{1:260, 2:62, 3:14}`; 336 blobs -> 408 centres (+72 recovered).
`batch_validate.py` (7 frames, all filters/modules/depths, panels in `out_batch/`):
no crashes; multi-fraction tracks crowding — deep o023/o028 19-24%, sparse
o046/o049 0-4%, F277W LW 5%.  Both regimes confirmed working on real doubles.

## Scripts
- `find_merged_satblobs.py` — locate/cross-check merged-double blobs (GNS + daophot).
- `probe_ramp_cores.py`, `probe_zeroframe.py` — the frame-zero investigation.
- `deblend_zeroframe.py` — the deblender (standalone; to be ported into
  `jwst_gc_pipeline/reduction/saturated_star_finding.py`).
- `validate_deblend.py` — example-blob + whole-frame validation, writes `out/`.
- `make_carta_artifacts.py` — writes ZEROFRAME-with-WCS, NEW deblend catalog,
  OLD one-per-blob catalog next to the dev frame's pipeline products.

## CARTA snippets (`~/.carta/config/snippets/`)
- `gc2211-satstar-deblend-debug` — crf data | ZEROFRAME | satstar model |
  residual + satstar/NEW/OLD catalogs, with jump coords for L168/L15/L31.
- `gc2211-merged-images-catalog` — merged i2d deep images + merged daophot cat.

## TODO (integration)
Port `deblend_blob_zeroframe` into `get_saturated_stars`: load the matching
`_ramp.fits` ZEROFRAME per frame, replace the one-record-per-component loop with
one `source_record` per returned centre (sharing the blob's sat-mask), keep the
existing brightest-first iterative-subtraction fit. Then a residual re-detection
pass for marginally-saturated companions.
