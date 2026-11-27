# Positive-spike finder: tolerating one unusable neighbour

`POSITIVE_SPIKE_MASK_MAX_UNUSABLE` is an opt-in extension of the one-pixel spike finder of `POSITIVE_SPIKE_MASK_MIN_EXCESS` (`jwst_gc_pipeline/photometry/single_exposure_events.py::find_positive_spikes`). Unset, blank or 0, the finder is the shipped one, and `jwst_gc_pipeline/photometry/tests/test_single_exposure_events.py` pins the equality on eight synthetic noisy frames. With 1, a spike may have one of its eight neighbours unusable. The switch changes only the pixels that `jwst_gc_pipeline/photometry/crowdsource_catalogs_long.py::single_exposure_event_masks` drops from the per-frame images before they are resampled into a mosaic. It has no effect on any fit or catalog.

This directory holds the measurements behind the rule, the largest tolerance that the data allow, and a tile test on o130. Every number below is in a file of `results/`, written by a script of `scripts/`.

## The case

Pixel (1525, 141) of `jw10678130001_02101_00003_nrcalong_destreak_o130_crf.fits` has 1572.0 MJy/sr. The median of the pixels 3 to 8 pixels from it is 6.4 MJy/sr with a sigma of 2.1 MJy/sr, so its excess is 1565.7 MJy/sr, 738 sigma. Its brightest usable neighbour has 15.0 MJy/sr, 0.0055 of the excess. The lower left neighbour is NaN with DQ 5 (DO_NOT_USE and JUMP_DET). The shipped finder needs all eight neighbours usable and returns nothing here. No other single-exposure mask covers the box either: the cross-frame events, the ramp events and the negative pixels cover 0 of its 9 pixels (`results/which_mask_o130_F480M_1525_141.json`). The pixel stays in the mosaic and leaves a leftover at mosaic pixel (1932, 146) with a residual peak of 358.8 MJy/sr, a data peak of 361.3 and a model peak of 4.9, class VETTED_BUT_UNSUBTRACTED, in the masked render of the four-tile run. The unmasked render has the same feature at the same pixels (`results/o130_F480M_cleared_leftovers.json`, `results/pixcmp_o130.txt`).

## The rule

A pixel is a candidate when it is finite, carries neither DO_NOT_USE nor SATURATED, and stands at least `MIN_EXCESS` and 5 sigma above the median of the finite pixels 3 to 8 pixels from it. The shipped finder then needs all eight neighbours usable and the brightest of them at most 0.15 of the excess. With a tolerance of k:

- at most k neighbours may be unusable, that is not finite, DO_NOT_USE or outside the array;
- no neighbour may be SATURATED, whatever k is;
- the isolation test of 0.15 is judged on the usable neighbours;
- the 3x3 box of a spike is masked as before, including the unusable neighbour.

`jwst_gc_pipeline/photometry/single_exposure_events.py::SPIKE_MAX_UNUSABLE_LIMIT` is 1. The variable raises `ValueError` naming itself for a value above the limit, a value below 0 and anything that is not an integer.

The scan below measures missing neighbours that are NaN with DO_NOT_USE. A SATURATED neighbour is a different configuration, which the last section describes.

## Why the limit is 1

A tolerance of k lets through a star core whose k brightest neighbours are missing, if its next brightest neighbour is at most 0.15 of the peak. The scan therefore takes every star core of every frame and asks which tolerance would make it look isolated, with no assumption about which neighbours are really missing:

1. A core is a local maximum of a complete 3x3 box with an excess of at least 300 MJy/sr and at least 5 sigma. The scan finds 8549 (o130 F480M) to 142336 (o105 F212N) of them per tile-band (Table 2 of `results/spike_tolerance_summary.txt`).
2. Sorting its eight neighbours from the brightest, it passes at k when the neighbour of rank k+1 is at most 0.15 of the excess. A core that fails the shipped test and passes at k is at risk.
3. Each core at risk with k up to 4 is looked up at the same sky position in every other frame that covers it, as the 3x3 maximum of the excess over the local background. When another frame holds 0.5 or more of the excess, the core is a source seen in more than one exposure. When no other frame covers the position, the core cannot be tested.

All eight tile-bands of the four-tile run are scanned, every frame (`results/spikes_<tile>_<band>.json`). Cores at risk, summed over the tile-bands (Table 3):

| k | cores at risk | another frame holds 0.5 or more of the excess | no other frame covers the position | another frame covers it and holds less than 0.5 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 15 | 0 | 0 | 15 |
| 2 | 19 | 1 | 0 | 18 |
| 3 | 81 | 62 | 1 | 18 |
| 4 | 1332 | 1295 | 18 | 19 |

The largest k at which no tile-band has a source or an untestable core is 1. The one source at k = 2 is o138 F480M, exposure 1 nrcblong, pixel (1096, 92): excess 691 MJy/sr, brightest neighbours at 0.363, 0.183 and 0.081 of it, five other frames, the largest holding 0.56 of the excess, nearest multi-frame catalog row at 0.44 FWHM.

The other-frame fraction of the cores at risk has two groups (Table 6). Of the 1314 cores at k = 4 that another frame covers, 1295 hold 0.5 or more, one holds 0.304 and 18 hold 0.243 or less. At k = 1 the largest fraction is 0.180, and at k = 2 the o138 core holds 0.556. Any cut between 0.180 and 0.556 gives the same limit of 1.

A Gaussian star gives a weaker bound. A star of PSF sigma 0.95 pixel (the F212N value) on a pixel corner has three neighbours at 1.0 of its peak pixel, four at 0.33 and one at 0.109, so it passes the isolation test only when seven neighbours are missing. The model alone would allow a tolerance of 6, and the data show a star core that passes at 2. The limit follows the data.

## What a tolerance of 1 adds

The scan runs the shipped finder and the finder with every tolerance up to 6 on every frame of the eight tile-bands (`MIN_EXCESS` 300 MJy/sr). Spikes found, shipped and with a tolerance of 1 (Table 1):

| tile | band | frames | shipped | tolerance 1 | added |
| :--- | :--- | ---: | ---: | ---: | ---: |
| o105 | F212N | 48 | 15 | 16 | 1 |
| o118 | F212N | 48 | 12 | 12 | 0 |
| o130 | F212N | 48 | 9 | 10 | 1 |
| o138 | F212N | 48 | 5 | 5 | 0 |
| o105 | F480M | 12 | 54 | 57 | 3 |
| o118 | F480M | 12 | 60 | 66 | 6 |
| o130 | F480M | 12 | 69 | 72 | 3 |
| o138 | F480M | 12 | 67 | 68 | 1 |
| all | | | 291 | 306 | 15 |

The 15 added spikes (Table 7) have excesses of 368 to 2011 MJy/sr and 91 to 1805 sigma. The brightest usable neighbour holds at most 0.0696 of the excess, median 0.0055, against the limit of 0.15. The missing neighbour is NaN with DO_NOT_USE in all 15 (DQ 5 nine times, DQ 1 five times, DQ 6145 once). Every spike is covered by at least one other frame, and no other frame holds more than 0.043 of the excess at its sky position. The closest SATURATED pixel is 13 pixels away.

Seven of the 15 lie within 1 FWHM (0.072 arcsec in F212N, 0.164 arcsec in F480M) of a row of the vetted catalog of the run, six of those seven within 1 FWHM of a row with nmatch of 2 or more, and none within 1 FWHM of a satstar row. The vetted catalogs hold 140067 to 523550 rows per tile-band, so a row near a position is common, and the scan does not estimate how many of the seven are chance coincidences. The test that the mask removes no source is the other-frame fraction.

Two of the three F480M spikes that o130 gains are the same detector pixel, (760, 303) of NRCALONG, in exposures 3 and 4 (1275 and 1276 MJy/sr). The scan finds that pixel in o118 F480M as well, in exposures 2 and 6 (2011 and 1277 MJy/sr). Each time one neighbour is NaN with DQ 1, and no other frame holds more than 0.00 of the excess at the sky position (Table 4).

For contrast, the tolerances above 1 add 14 more spikes over the eight tile-bands (Table 5): two at 2, one at 3, two at 4, five at 5 and four at 6. Nine of the 14 have another frame that holds 0.5 or more of the excess (one at 4, four at 5, four at 6) and seven lie within 1 FWHM of a satstar row (one at 4, three at 5, three at 6).

## Tile test on o130

The four-tile run renders the residual mosaic of each tile-band twice: with the footprint masks of the stacked opt-in pull requests (masked) and without them (unmasked). Both renders drop the pixels of `single_exposure_event_masks` from the per-frame images. Both were rendered again with `POSITIVE_SPIKE_MASK_MAX_UNUSABLE=1` from the same catalogs, for o130 F480M and o130 F212N. The code is the code of the four-tile run with this branch merged in. A render with the switch unset on that code is the control. A leftover is a residual peak above 0.7 times the data peak and above the threshold in MJy/sr, and a crater is a depth above the threshold in MJy/sr, as in the four-tile run (`results/compare_o130_F480M.json`, `results/compare_o130_F212N.json`).

| render | leftovers >= 50 | >= 100 | >= 200 | >= 500 | craters >= 50 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| F480M masked, run | 56 | 7 | 1 | 0 | 0 |
| F480M masked, switch 1 | 53 | 5 | 0 | 0 | 0 |
| F480M unmasked, run | 85 | 28 | 19 | 14 | 1 |
| F480M unmasked, switch 1 | 82 | 26 | 18 | 14 | 1 |
| F212N masked, run | 52 | 3 | 1 | 1 | 5 |
| F212N masked, switch 1 | 52 | 3 | 1 | 1 | 5 |
| F212N unmasked, run | 73 | 16 | 13 | 6 | 56 |
| F212N unmasked, switch 1 | 73 | 16 | 13 | 6 | 56 |

The control reproduces the masked render of the run pixel for pixel in both bands (0 of 11430027 finite pixels differ in F480M, 0 of 49472138 in F212N), so every difference below comes from the switch (`results/pixcmp_o130.txt`).

**F480M.** The switch adds three spikes to the masks of the 12 frames: (1525, 141) and (760, 303) in exposure 3 nrcalong, and (760, 303) in exposure 4 nrcalong. It sets 27 pixels (3 boxes of 9) to DO_NOT_USE that no other mask had set (`results/union_masks_o130_F480M.json`: 8026 to 8053 pixels over the union of the frame masks). The NaN pattern of the masked render does not change. 49 of its 11430027 finite pixels differ, in three groups of 19, 15 and 15 pixels, with a largest change of 360.6 MJy/sr and a median of 0.033 MJy/sr. The unmasked render differs at the same 49 of 12472741 pixels. Each group contains one of three leftovers, which are gone with the switch (`results/o130_F480M_cleared_leftovers.json`):

| mosaic pixel | residual peak | data peak | model peak | class in the run | nearest vetted row, arcsec |
| :--- | ---: | ---: | ---: | :--- | ---: |
| (1932, 146) | 358.8 | 361.3 | 4.9 | VETTED_BUT_UNSUBTRACTED | 0.017 |
| (774, 312) | 159.8 | 162.6 | 4.5 | VETTED_BUT_UNSUBTRACTED | 0.030 |
| (1161, 314) | 83.8 | 84.8 | 2.8 | NOT_IN_M7_CATALOG | 0.196 |

Peaks are in MJy/sr. The leftovers of 50 MJy/sr and more number 56 without the switch and 53 with it, so none appears. The leftovers of 100 and more go from 7 to 5, and the one of 200 and more goes. No crater appears.

**F212N.** The switch adds one spike, in exposure 4 nrca2 at (1819, 1980), with an excess of 602 MJy/sr and 244 sigma, and the renders do not change at all (0 of 49472138 masked and 0 of 51826691 unmasked pixels differ). The union of the frame masks does not change either (`results/union_masks_o130_F212N.json`: 10644 to 10644 pixels, no frame changed), because the ramp test of the single-exposure events already masks the 3x3 box of that spike (`results/which_mask_o130_F212N_1819_1980.json`: 9 of 9 pixels in a ramp event of 1 pixel with a peak of 604 MJy/sr).

## What it does not do

- It changes no fit and no catalog. The vetted row at 0.003 arcsec of the pixel (1525, 141), which has nmatch 1, stays in the catalog.
- A spike beside a SATURATED pixel stays unjudged. An example in o130 F480M is exposure 4 nrcblong, pixel (821, 2037) with 3359.7 MJy/sr (`results/pixel_box_o130_F480M_821_2037.txt`). Its neighbours are finite and the brightest has 83.7 MJy/sr, about 0.017 of an excess of 3331.8 MJy/sr, but the upper left neighbour carries DQ 2 (SATURATED) with 59.8 MJy/sr, and neither tolerance 0 nor 1 returns the pixel. With the SATURATED flag cleared in a copy of the DQ, the shipped finder returns it. A SATURATED neighbour can mark the edge of a saturated star, so tolerating it needs its own measurement of the star cores.
- The limit of 1 rests on 15 cores at risk at k = 1 and one source at k = 2. A tolerance of 2 would add two spikes over the eight tile-bands, none of them covered by another frame at 0.5 or more, and would let through the one core that another frame confirms as a source. The scan covers four tiles in two bands and does not cover o045.
- The branch of the ramp events (pull request 1367) and the base of this branch (pull request 1374) conflict in `jwst_gc_pipeline/photometry/crowdsource_catalogs_long.py` and `jwst_gc_pipeline/photometry/single_exposure_events.py`, 3 hunks in each. This branch has the same 3 and 3 hunks, and the tolerance lines sit in the cache key, the docstring and the log line of `single_exposure_event_masks`. The tile test used the merge with the cache key `(files, events_on, ramp_on, nsigma, spike_min, spike_unusable)`.

## Files

| path | content |
| :--- | :--- |
| `scripts/scan_spikes.py` | the safety scan of one tile-band, writes `results/spikes_<tile>_<band>.json` |
| `scripts/aggregate_scans.py` | Tables 1 to 7 over the eight scans, writes `results/spike_tolerance_summary.txt` |
| `scripts/union_masks.py` | pixels that the switch adds to the union of the per-frame masks of a tile-band |
| `scripts/which_mask.py` | which per-frame mask covers the 3x3 box of one pixel |
| `scripts/pixel_box.py` | the 3x3 box of one pixel and what the finder makes of it |
| `results/spikes_*.json` | the eight scans, with every shipped and added spike, the star cores at risk and the other-frame look-ups |
| `results/spike_tolerance_summary.txt` | Tables 1 to 7 |
| `results/compare_o130_F480M.json`, `results/compare_o130_F212N.json` | leftover and crater counts and matchings of the six renders of the tile test |
| `results/pixcmp_o130.txt` | pixel comparison of the renders |
| `results/o130_F480M_cleared_leftovers.json` | the three leftovers that the switch clears |
| `results/union_masks_o130_F480M.json`, `results/union_masks_o130_F212N.json` | mask pixels that the switch adds |
| `results/which_mask_o130_F480M_1525_141.json`, `results/which_mask_o130_F212N_1819_1980.json` | the masks that cover the 3x3 box of the F480M case (none) and of the added F212N spike (a ramp event) |
| `results/pixel_box_o130_F480M_821_2037.txt` | the 3x3 box of the SATURATED-neighbour example |

## Reproduce

The scripts need the run tree of a tile: `ROOT` is its `gc-treasury` directory, which holds `F480M/pipeline/*_crf.fits` and `catalogs/` with the vetted merged catalog and the consolidated satstar catalog of each band. Scan and tables, for each tile and band:

```
python -I docs/evidence/spike_tolerance/scripts/scan_spikes.py o130 F480M --root ROOT --out docs/evidence/spike_tolerance/results
python -I docs/evidence/spike_tolerance/scripts/aggregate_scans.py docs/evidence/spike_tolerance/results --out docs/evidence/spike_tolerance/results/spike_tolerance_summary.txt
```

Mask pixels that the switch adds on a tile-band, and the masks that cover one pixel, with the mask switches of the run in the environment:

```
export POSITIVE_SPIKE_MASK_MIN_EXCESS=300 SINGLE_EXPOSURE_EVENT_MASK=1 SINGLE_EXPOSURE_EVENT_RAMP=1 NEGATIVE_PIXEL_MASK_NSIGMA=50
python -I docs/evidence/spike_tolerance/scripts/union_masks.py o130 F480M --root ROOT --out union_masks_o130_F480M.json
python -I docs/evidence/spike_tolerance/scripts/which_mask.py o130 F480M jw10678130001_02101_00003_nrcalong_destreak_o130_crf.fits 1525 141 --root ROOT --out which_mask.json
python -I docs/evidence/spike_tolerance/scripts/pixel_box.py ROOT/F480M/pipeline/jw10678130001_02101_00004_nrcblong_destreak_o130_crf.fits 821 2037 --out pixel_box.txt
```

The residual render with the switch is the run's render command with `POSITIVE_SPIKE_MASK_MAX_UNUSABLE=1` added to its environment. The ramp events of `which_mask.py` and `union_masks.py` need the checkout to hold the ramp-event branch; without it the ramp part is skipped.
