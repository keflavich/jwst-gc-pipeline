# Proposal for a sky-preserving Treasury F480M diffuse-emission release

The next quantitative F480M map should start from the no-destreak exposure lineage. The released science quantity would be bandpass-calibrated, star-subtracted surface brightness in MJy/sr on the native F480M mosaic grid. A downstream line or color product would additionally require matched PSFs, a bandpass-aware continuum estimate, and propagated scaling and zero-point errors. [NIRCam passbands](https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-filters)

## Evidence

- On 2026-11-20, the survey recipe selected 66 hybrid-m7 F480M residual tiles. Their `*_perframe_inputs.json` ledgers named 784 exposure records, all `destreak_…_crf`. None of those exposure stems had a matching `align_…_crf` next to it in the production directory. The survey FITS coadd carries a primary brightness image and a WHT extension. The selector describes the currently available inputs, and the existing FITS coadd has no manifest that identifies the inputs used for its build. The current pipeline policy already turns off destreaking for Treasury F480M. [Policy and original rerun](https://github.com/keflavich/jwst-gc-pipeline/pull/1271)
- A paired injection into the six NRCBLONG calibrated exposures of each of o065, o072, and o079 measured the transfer of sky structure. The synthetic sky had a peak amplitude of 0.6 MJy/sr. The row filter was applied to both the injected and uninjected calibration and the difference was reprojected with the same aligned GWCS, bilinear interpolation, DQ mask and equal exposure weights. Scoring used the central 768 × 768 pixels where at least three exposures contributed. The no-destreak injection on this same grid defines unit gain. Contrast gain is the slope of the recovered image against the injected template with a free intercept. The experiment held star fitting and coverage fixed. [Earlier controlled three-observation rerun](https://github.com/keflavich/jwst-gc-pipeline/issues/1268#issuecomment-6604672711)

| Injected structure | 256-row filter | 2048-row filter |
| --- | ---: | ---: |
| Row-parallel filament, 8 px FWHM | 27% | 11% |
| Row-parallel filament, 32 px FWHM | 54% | 11% |
| Row-parallel filament, 128 px FWHM | 80% | 11% |
| Row-perpendicular filament, 32 px FWHM | 100% | 98% |
| Row-parallel wave, 512 px wavelength | 82% | 4% |
| Circular knot, 30 px FWHM | 94% | 89% |

Values are medians across the three observations. The 32 px parallel filament recovered 19–58% of contrast with the 256-row window and 3–15% with the 2048-row window. Its median aperture flux recovery was 70% and 18%, respectively. Detector orientation and morphology matter even when the mosaic seams look small.

A separate test used the archived F480M image mosaics from the original rerun, before stellar subtraction. It fit a robust additive offset per tile on overlap block medians, with equal weight per edge and a soft-L1 loss with a 0.25 MJy/sr scale. Each of the 122 test edges was excluded from its fit and had an alternate path to connect its endpoints. Seven other overlaps could not be scored by this test.

| Image-mosaic lineage | Median absolute held-out offset before | After | Units |
| --- | ---: | ---: | --- |
| Archived production with destreak | 0.586 | 0.904 | MJy/sr |
| No destreak | 0.186 | 0.097 | MJy/sr |
| Joint row correction prototype | 0.179 | 0.148 | MJy/sr |

The no-destreak result supports testing additive matching on rebuilt residuals. For the old destreak products, a tile constant cannot recover the lost row-aligned sky signal. The archival production products varied in age. [Overlap rerun and caveats](https://github.com/keflavich/jwst-gc-pipeline/pull/1271)

## Proposed build

1. **Regenerate the F480M residual inputs.** Start with o065/o072/o079 for seam tests and o078/o100/o105 plus o102/o109 for crowding and saturation checks. Pin the calibrated exposure inventory, code and WCS. Build the `align` CRFs, then run the normal JICAMA phases through the merged hybrid-m7 residual. Resolve exposure lineage from its ledger and verify the residual, model, masks and WCS state describe the same build. Apply the same controlled test to F212N before changing its policy. Audit historical destreak provenance for fields outside the Treasury. [Historical destreak audit](https://github.com/JWST-GC/cataloging-pre-treasury/blob/main/reports/report_12_destreak_vs_nodestreak.md)
2. **Fit relative brightness on matched sky pixels.** Compare overlapping residuals on a common WCS using well-covered pixels and masks for unreliable edges and bright-star cores and wings. Start with additive constants, report held-out block or edge residuals, and record component membership and offset uncertainty. Disconnected components have independent zero-point freedoms. Any absolute calibration needs an independent bandpass-aware anchor. Test detector gradients or gains only where repeat observations justify them. [Pairwise MIRI solver](https://github.com/keflavich/jwst-gc-pipeline/blob/main/jwst_gc_pipeline/reduction/miri_skymatch.py) · [PHANGS-JWST background anchoring](https://arxiv.org/abs/2401.15142)
3. **Make the FITS coadd reproducible.** Move the survey coadd recipe into the pipeline. Drive it with an explicit manifest of eligible inputs and exclusions, including parent CRFs, PSF and saturation settings, calibration and WCS generation, offsets and their uncertainty. Record coverage, statistical uncertainty and masks for unmeasured or unreliable pixels. Estimate aperture-scale noise using split coadds to include resampling correlations. Tie the manifest to the delivered FITS bytes. [Residual WCS lineage incident](https://github.com/keflavich/jwst-gc-pipeline/issues/1141)
4. **Treat saturated cores and blends explicitly.** Pilot F212N-resolved seed positions for the remaining F480M close pairs, fitting F480M fluxes independently. Measure signed core and wing residuals on held-out examples and record area lost to masking. Preserve valid dither measurements for a sky pixel wherever possible. The o102 footprint-mask experiment reduced the number of strong F480M core residuals while creating more masked area. A mask in the science product makes that trade-off measurable. [Bright-star pilot](https://github.com/keflavich/jwst-gc-pipeline/issues/1346) · [Saturated-footprint experiment](https://github.com/keflavich/jwst-gc-pipeline/pull/1274)

The first end-to-end check should repeat the paired injections with star fitting enabled, using the same starting exposures for both arms. Report contrast and aperture flux by shape, scale and orientation together with seam offsets, directional power, and valid area. The calibration-level results above establish why the input lineage should change. The new run would establish how much of that gain survives source fitting and the final mosaic.

## QA and publication integration

- Stage 15 can report the transfer and seam metrics beside its photometry-tool panels. [QA methods](https://github.com/JWST-GC/data-qa/blob/main/docs/qa_methods.md#stage-15--residual-maps-by-photometry-tool)
- The Treasury HiPS builder can render from the approved FITS and manifest. [Survey mosaic viewer builder](https://github.com/keflavich/jwst_scripts/blob/main/scripts/gc_treasury_mosaic_rgb_hips.py)
