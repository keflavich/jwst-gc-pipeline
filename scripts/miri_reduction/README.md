# MIRI reduction scripts (2026-06)

One-off but preserved scripts from the MIRI imaging + photometry pipeline
work of 2026-06-10/11 (brick F2550W background-matching/NaN fix, MIRI
astrometric registration, sickle-background source validation).  The
canonical copies are tracked here; the operational copies live in the data
trees and are run from there.

All scripts use **absolute paths** to the data trees, so they can be run
either from this directory or from their data-tree locations -- "copying"
is only needed to keep the operational copy next to its products.

The SLURM job runners that submit these (and the MIRI cataloging runs) are
tracked in `brick-jwst-2221/brick2221/shellscripts/` (brick_miri_*,
cloudc_miri_*, sickle_miri_*), per the convention that all job runners live
there even for non-brick datasets.

| script | data-tree location | purpose |
|---|---|---|
| `miri_f2550w_image3_rerun_v2.py` | `/orange/adamginsburg/jwst/brick/reduction_scripts/` | Re-run image3 only (tweakreg skipped) on brick F2550W with skymatch(subtract=True) + outlier snr='30.0 25.0', writing to `pipeline_v2/`; the experiment that diagnosed/fixed the OUTLIER-overflag NaN holes and background seams.  Edit `asn_file`, `pipedir`, `variants` to adapt to another target. |
| `apply_measured_miri_wcs_offsets.py` | `/orange/adamginsburg/jwst/brick/reduction_scripts/` | Apply offset-histogram-measured astrometric corrections to MIRI i2d FITS headers (CRVAL shift; ASDF gwcs left untouched, flagged with MIRIWCSN).  Idempotent via MIRIDRA/MIRIDDE keywords.  Contains `refine_offset()` -- the density-robust offset-histogram registration that should eventually move into `align_to_catalogs.py`. |
| `miri_f2550w_tile_homogenize_v3.py` | `/orange/adamginsburg/jwst/brick/reduction_scripts/` | Per-visit-tile background-plane homogenization (fit each tile's plane deviation from the consensus mosaic, subtract, rebuild).  **Negative result**: did NOT reduce the x=1609 seam -- the plane mostly re-measured skymatch's constants, and the edge artifact varies along the boundary on ~50 px scales, which a plane cannot represent.  Kept for the record. |
| `miri_f2550w_edgetrim_v4.py` | `/orange/adamginsburg/jwst/brick/reduction_scripts/` | Detector edge-trim: DQ-flag 40 east columns / 16 west columns / 12 rows per frame (the east-edge glow reaches +150..+1400 MJy/sr and varies frame-to-frame and along the edge), then rebuild image3 with the v2a configuration.  In overlap strips neighbors' clean interiors cover the trimmed pixels, removing the seam at its source. |

### Sickle MIRI astrometric registration (prop 3958) — see brick-jwst-2221

The sickle MIRI → NIRCam-F480M registration scripts are **region-specific** and
live in the **brick-jwst-2221** repo, not here:
`brick2221/reduction/register_sickle_miri_o001_o002.py`,
`register_o002_f770w_per_frame_to_f480m.py`,
`register_o002_f770w_gwcs_to_f480m.py`, `merge_sickle_miri_o001_o002.py`
(+ `brick2221/reduction/build_sickle_gns_offsets.py` and
`brick2221/shellscripts/sickle_gns_reduce_retry.sh` for the NIRCam sickle→GNS
offsets). They are **manual pre-steps** — run before cataloging a sickle MIRI obs
or its mosaics sit ~3.3″ off truth. See the "Offsets-table provenance" section of
`jwst_gc_pipeline/reduction/ASTROMETRY_WCS_CORRECTION_FLOW.md`.

Analysis scripts (source validation/classification, CMDs) live in
`brick-jwst-2221/brick2221/analysis/` (`f2550w_source_validation_o003.py`,
`miri_cmd.py`) -- jwst-gc-pipeline is for pipeline work, brick2221 for
analysis work.

## Key lessons encoded in these scripts

- **Background matching and NaN holes are one problem**: with
  `skymatch(subtract=False)`, outlier_detection's median image sees
  inter-visit thermal background jumps and OUTLIER-flags whole regions in
  every frame; resample then has no valid inputs.  `subtract=True` fixes
  both (now the PipelineMIRI default).
- **Never validate astrometry by nearest-neighbor median against a dense
  catalog** -- it returns ~0 for any true shift.  Use the offset-histogram
  stacking in `refine_offset()`, and require the peak to stand far above
  the median bin count.  For bands the refcat cannot register (F2550W),
  chain through the nearest shorter-wavelength MIRI band.
- The historical hard-coded MIRI shift (-3.895", +1.28") was itself the
  dominant astrometric error on sickle/cloudc; `fix_alignment` now defaults
  to zero shift.
