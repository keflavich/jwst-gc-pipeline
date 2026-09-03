# NIRISS Sgr C (4147 / obs 012) — first-light reduction + cataloging

New instrument path (first NIRISS in this pipeline). Ported from the MIRI
single-detector runner: NIRISS has one NIS detector, so the module / destreak /
SW-LW split plays no part.

## What runs
- Reduction: `reduction/PipelineRerunNIRISS.py` (Detector1+Image2 from uncal →
  per-exposure fix_alignment (zero baseline) → Image3 with abs_refcat → VIRAC2).
  Products under `<target>/niriss/<FILTER>/pipeline/`.
  Submit: `scripts/reduction/submit_reduction_niriss.sbatch`.
- Cataloging: `crowdsource_catalogs_long --each-exposure --instrument niriss
  --modules nis --each-suffix o012_crf` (the standard manual PSF pipeline
  m12/m3–m6). Submit: `scripts/reduction/submit_cataloging_niriss.sbatch`.
  Prereq: `scripts/reduction/stage_niriss_basepath.sh <target>` (once).

## Instrument disambiguation (NIRISS shares filter names with NIRCam)
`GC_INSTRUMENT_OVERRIDE=niriss` (via `--instrument niriss`) — a process-global
override (same safety rationale as `GC_BASEPATH_OVERRIDE`) consulted by
`naming._instrument_from_filter/_inst_token/_svo_filter_id`, `fwhm_table_path`,
and the basepath `niriss/` insertion. Header-driven PSF builders
(`saturated_star_finding.get_psf`, `filtering.get_fwhm/get_filtername`) key off
`INSTRUME='NIRISS'` directly.

## First-light m2 astrometry-checkpoint override (JUSTIFIED)
Run with `ASTROM_CHECKPOINT_WARN_ONLY=1`.

The m2 checkpoint hard-stops NIRISS F200W: it finds small per-exposure jitter
(2–10 mas vs visit consensus, normal) AND reports the consensus→VIRAC2 bulk tie
as "COULD NOT VERIFY" because the sparse-Gaia cross-ref gross gate reads
**18708 mas (18.7″)** → `gross_ok=False`.

That cross-ref is spurious. TWO independent density-immune
`measure_offset` histogram ties (mosaic and the m6 catalog) put JWST F200W on
VIRAC2 at **~9 mas** (contrast 32, ~6×10^5 pairs) — a coherent tie. Per
`CLAUDE.md` ("the sparse-Gaia cross-check must never BLOCK a coherent VIRAC
tie"), this is the documented failure mode where a sparse/spurious Gaia
cross-check blocks a good VIRAC tie (Sgr C is deep; the JWST-detectable Gaia
subset is small). So for first light we demote the checkpoint to a warning and
ship the tweakreg-abs VIRAC tie (~9 mas).

The same switch also demotes the m2 no-channel refusal, which is what a NIRISS
run now meets first: `offsets_channel(..., instrument='niriss')` is `none`
because `PipelineRerunNIRISS.fix_alignment` reads no offsets table, so an
above-floor per-exposure correction has nowhere to go and the checkpoint says so
by name. Under `ASTROM_CHECKPOINT_WARN_ONLY=1` that is a warning and the run
proceeds, exactly as this section already describes; with
`ASTROM_CHECKPOINT_APPLY=1` the measured-misaligned im0 mosaics are still
stale-tagged `*_i2d_im0_badastrom.fits` and nothing is written to any offsets
table.

DEFERRED (after first light): (a) the 2–10 mas per-exposure consensus
refinement (would need NIRISS `fix_alignment` to consume a consensus offsets
table, à la the NIRCAM sgrc `_apply_consensus_offsets_table` path, then
regenerate + re-catalog); (b) root-causing the 18.7″ sparse-Gaia cross-ref for
single-detector NIRISS (refcat Gaia-subset epoch/PM to obs 2024.49? checkpoint
pairing for one detector?). Both are astrometry-refinement follow-ups.

## Status (2026-07-25)
F200W: reduction + full m6 catalog (104,816 sources; clean model/residual;
VIRAC2 tie 9.1 mas). F158M/F356W/F480M: reduced; cataloging queued (same path).
