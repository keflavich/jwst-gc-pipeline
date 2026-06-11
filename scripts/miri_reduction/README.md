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
| `f2550w_source_validation_o003.py` | `/orange/adamginsburg/jwst/brick/reduction_scripts/` | Validate brick F2550W point sources against the sickle SICKLE-MIR-BACKGROUND (jw03958-o003) F770W/F1130W/F1500W frames (which lie inside the F2550W mosaic): forced aperture photometry in all four bands at corrected positions, 7.7-25.5 um spectral index, YSO-vs-evolved classification.  Writes `brick/catalogs/f2550w_sources_o003_validation.fits`. |

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
