# F770W reference calibration

Recipe R1 uses GO 10678 F770W FULL imaging. The bundle contains the overlap flat, flat uncertainty, static bad-pixel mask, pane response law, non-negative PSF kernels, and affine detector geometry.

The response is `Q = (rate * k / FLAT - b) / a`, where `k = median(cal / rate)` on the imager. The quoted validation covers 30–450 MJy/sr and NGROUPS 5–8. The point-source flux ratio is 0.942 ± 0.019 and is recorded without applying a correction.

`SHA256SUMS` identifies the numerical reference files. The supplied calibration is credited to Nazar Budaiev.
