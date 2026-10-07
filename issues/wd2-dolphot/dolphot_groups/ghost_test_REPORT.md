# Fixed-offset dolphot sources near saturated stars: pixel-stack test

Scripts: stack.py (stamps from crf frames), analyze_stack.py (sky-aligned stack), psf_compare.py (detector-aligned stack minus stpsf grid PSF), dolphot_side.py (catalog-side tests, writes ghost_candidates_<band>.ecsv).
Figures: fig/ghost_stack_<band>.png, fig/ghost_psfcompare_<band>.png, fig/dolphot_side_<band>.png (bands 150W, 162M, 182M). F164N and F187N were not stacked.
Numbers: stack_results_*.json, psf_compare_*.json, stack_detaligned_*.json, dolphot_side_*.json.

Hypotheses: (a) an optical ghost present in the pixel data; (b) a spurious dolphot detection with no counterpart flux in the data.

## Method
Isolated saturated stars (no other saturated star within 2"), parent mag 15.5-18: 993 stamps (F150W), 1294 (F162M), 553 (F182M, mostly 15.5-16.5) from the 32 crf frames per band. Each stamp is background-subtracted, normalised by the 0.25-0.6" annulus flux, and has its azimuthal median removed. Stack = median. Measurement = sum in r<0.08" aperture at the expected position versus the same aperture at 31 positions on a ring of equal radius (+-25 deg excluded); z = (value - ring mean)/ring std. Ghost amplitude in total-flux fraction = annulus-normalised value x 0.071 (annulus holds 7% of the stpsf PSF flux; F162M 6.8%, F182M 6.9%).
A dolphot flux ratio of 0.15% (7.1 mag) corresponds to about 0.020 in annulus-normalised units.

## Results
1. All frames of all three bands have one PA_V3 = 140.82 deg, so a fixed sky offset is a fixed detector offset (about +12.2, +9.8 px for the E offset; same for all 8 SW detectors within 0.3 px). The test cannot separate "sky-fixed" from "detector-fixed" origin.
2. Raw sky stack (F150W, E position): 0.0054 +- 0.0039 (z=1.3); the PSF wings have strong diffraction-spike structure at this radius (ring std 0.0039 is the PSF structure, not noise alone). The stpsf model processed identically gives 0.0083 +- 0.0068 at the same position: the data bump equals the model PSF structure.
3. Data minus stpsf model (detector-aligned, all detectors), aperture sum at expected position minus ring mean, 1 sigma = ring std:
   - F150W: +0.0020 +- 0.0021 (z=1.0); 3 sigma limit 0.0062 = 0.044% of parent flux (8.4 mag fainter). Expected from dolphot 0.020.
   - F162M: -0.0019 +- 0.0023 (z=-0.8).
   - F182M (W offset): -0.0018 +- 0.0015 (z=-1.2).
   - By parent mag (F150W, 15.5-16.5/16.5-17.25/17.25-18): z = 1.1, 0.8, 1.1; no trend.
   - Per detector (sky stack, F150W): z between 1.0 and 1.7 for all 7 populated detectors; per exposure z 1.3.
   - North feature (+0.05,+0.69): data-minus-model -0.0009 +- 0.0009 (F150W), 0.0002 (F162M), 0.0005 (F182M); no excess.
   The pixel data show no excess flux at the dolphot offset at a level roughly 3-10x below the dolphot-implied flux (limit 0.044% vs 0.15% for F150W). Caveat: ring std is dominated by PSF-model mismatch (data structure is 0.75x the model amplitude in F150W), so the limit assumes a point-like ghost and uses the ring scatter as the noise.
4. Dolphot side (unmatched in both arms, 0.12" circle): F150W 224 vs 11 mirror, F162M 144 vs 4, F182M 44 vs 16 (W offset; the F182M mirror control is not clean, 16).
   - Centroids by parent detector are the same (F150W: +0.484 to +0.489", dDec -0.006 to +0.004", for NRCA4, B1, B2, B3, B4; F182M B1/B3/B4 -0.496 to -0.500, -0.076 to -0.087). Module A: (+0.485,-0.004), module B: (+0.487,-0.000). Detector assigned from dither 1; 90% of parents are on module B.
   - dolphot mag minus parent mag: median 7.14 (std 0.41) F150W, 6.86 F162M, 6.65 F182M; fit slope of dolphot mag vs parent mag 0.97 (F150W), 0.94, 0.95, i.e. nearly constant ratio, drifting from 7.4 (parents 14-15.5) to 7.1 (parents 18-19) in F150W.
   - Fraction of saturated parents with a circle source (F150W): 0/53 (12-13), 3/79 (13-14), 12/122 (14-15), 7/94, 11/140, 41/530 (16-17), 83/830 (17-18), 66/486 (18-19): about 8-10% from 14 to 19 mag and lower (0-4%) brighter than 14. Only about 10% of parents carry such a source, so it is not a universal fixed-ratio companion; the lower rate for bright parents is not explained by detectability (the implied source would be brighter, mag 20 for a mag-13 parent). The cause is not determined (saturated-star handling in dolphot is a candidate).
5. Documentation: searches of jwst-docs found NIRCam scattered-light artifacts (wisps, claws, dragon's breath, https://jwst-docs.stsci.edu/known-issues-with-jwst-data/nircam-known-issues/nircam-scattered-light-artifacts) and grism ghosts (tadpole, shell); no imaging ghost at +0.485" E in F150W/F162M/F164N or at 0.5" W in F182M/F187N was found. No documented ghost is claimed.

## Conclusion
The pixel stacks favour (b) over (a): no flux excess above PSF-model structure at the expected position in F150W, F162M or F182M, at 3-10x below the dolphot-implied flux for F150W (3 sigma limit). The expected position lies on the PSF's own diffraction structure (the stpsf model has the same lobe), which is consistent with spurious dolphot detections where the dolphot PSF model mismatches the data, though this experiment does not test that mechanism directly. Strength: moderate; limited by PSF-mismatch scatter (ring std 0.002 annulus-normalised) and by the single PA. The constant magnitude difference and constant offset across detectors are expected for either a ghost or a PSF-shape residual fixed relative to the parent. The north feature shows no excess either (no pixel evidence; single-PA caveat applies).
