# Footprint mask at larger `KFAC`: leftovers, big craters and masked area on five tiles

`MERGEDCAT_FOOTPRINT_MASK_KFAC` (off unless above 0) sets the SCI plane of the
mergedcat residual i2d to NaN in a disk around every row of the consolidated satstar
catalog.  The radius is `KFAC` × sqrt(`sat_area` / π) (`sat_area` in native detector
pixels, converted to mosaic pixels) + `MARGIN` mosaic pixels, so `KFAC` 1 is the
footprint of the saturated core.  The PRs of the stack that add the ring zone, the
DQ_SAT dilation and the edge band measured the mask at `KFAC` 1.  This section measures
`KFAC` 2, 3 and 4 (and 5 on o105 F212N) on the production residual i2d of the five
tiles of 10678 (o045, o105, o118, o130, o138) in F212N and F480M, and reports what the
larger disks cost.  The mask code does not change.  The PR adds the value `KFAC` 4 to the
parsing test of the switch, this section and a bullet in `SATURATED_PIXEL_HANDLING.md`.
The scripts of this section (the mask emulation, the render and the scoring) are not part
of this repo; `results/*.txt` hold their outputs.

**Result.**  From `KFAC` 1 to `KFAC` 4 the craters ≥ 50 MJy/sr fall by 62.5–69.5 % on F212N and by
20.0–50.0 % on F480M, and the whole-star leftovers ≥ 200 MJy/sr go from 13–65 to 8–38 per
F212N tile and from 17–24 to 11–19 per F480M tile.  The cost is masked area: `KFAC` 4 masks
4.0–15.2 % of the finite pixels of an F212N tile and 7.2–22.9 % of an F480M tile, against
1.0–3.2 % and 1.8–4.5 % at `KFAC` 1.  On F480M the step from `KFAC` 3 to `KFAC` 4 removes
0–3 leftovers ≥ 200 MJy/sr per tile for 2.5–8.3 percentage points of area.  Of the
30 craters ≥ 500 MJy/sr that F212N keeps at `KFAC` 4, 28 are `data` craters (the data are deeper than the model).
A radial profile correction of the satstar model leaves the craters ≥ 200 MJy/sr under the `KFAC` 4 mask unchanged on the two tiles
where it was tried.

**How the counts are made.**  Each count comes from a scratch copy of the production mergedcat residual
i2d of the tile and band; the production tree is read and never written.  An emulation of
`residual_footprint_mask` sets to NaN

- the disks of the consolidated satstar catalog of the tile (radius `KFAC` × sqrt(`sat_area` / π) × native pixel /
  mosaic pixel + 2 mosaic pixels),
- the ring zone: rows with `flux_fit` ≥ 10⁶ get a disk of at least 25 mosaic pixels in F212N and 20 in F480M
  (68–324 such rows per F212N tile and 6–42 per F480M tile, `results/kfac_ring_rows.txt`),
- the DQ_SAT dilation, 4 mosaic pixels in F212N and 3 in F480M (in every mask of this section), and
- the edge band, 8 and 6 pixels (the table of the section "With the edge band").

The production residual carries no render flags, so the single-exposure negative events of the frames are in every
count; the subsection on the flagged render below shows their share.  Whole-star leftovers (residual peak ≥ N MJy/sr, at
least 50, and more than 0.7 of the data peak at the star) and big craters (depth > 50 MJy/sr and depth more than 0.3 of the
model peak, counted at depth ≥ 50, 200 and 500 MJy/sr) follow the definitions of the other evidence sections of the
stack.  A vetted row is masked when its pixel is NaN in the masked image and finite in the production residual; the
percentage is of the vetted rows that fall on a finite pixel of the production residual.

## Counts and cost by `KFAC`

F212N (ring zone 25 pixels, DQ_SAT dilation 4 pixels):

| tile | `KFAC` | masked pixels (% of finite) | vetted rows masked (%) | leftovers ≥ 200 | leftovers ≥ 500 | craters ≥ 50 | craters ≥ 200 | craters ≥ 500 |
|---|---|---|---|---|---|---|---|---|
| o045 | 1 | 3.2 | 2.7 | 65 | 15 | 272 | 59 | 10 |
|  | 2 | 5.8 | 3.6 | 47 | 9 | 170 | 30 | 9 |
|  | 3 | 9.9 | 6.2 | 38 | 7 | 106 | 23 | 7 |
|  | 4 | 15.2 | 10.2 | 38 | 7 | 83 | 18 | 7 |
| o105 | 1 | 2.7 | 2.6 | 48 | 6 | 280 | 44 | 9 |
|  | 2 | 5.0 | 3.5 | 38 | 4 | 199 | 26 | 8 |
|  | 3 | 8.6 | 5.8 | 34 | 3 | 136 | 12 | 6 |
|  | 4 | 13.3 | 9.2 | 31 | 3 | 105 | 9 | 6 |
|  | 5 | 18.9 | 14.0 | 30 | 3 | 76 | 10 | 6 |
| o118 | 1 | 1.2 | 1.5 | 13 | 5 | 185 | 38 | 13 |
|  | 2 | 2.0 | 1.8 | 13 | 5 | 144 | 26 | 11 |
|  | 3 | 3.3 | 2.6 | 12 | 5 | 98 | 15 | 6 |
|  | 4 | 5.0 | 3.9 | 12 | 5 | 63 | 8 | 6 |
| o130 | 1 | 1.0 | 1.6 | 20 | 6 | 110 | 16 | 6 |
|  | 2 | 1.7 | 1.8 | 19 | 6 | 79 | 8 | 6 |
|  | 3 | 2.7 | 2.4 | 18 | 6 | 43 | 7 | 5 |
|  | 4 | 4.0 | 3.5 | 17 | 5 | 35 | 6 | 5 |
| o138 | 1 | 1.7 | 1.5 | 25 | 14 | 219 | 35 | 8 |
|  | 2 | 2.9 | 1.9 | 18 | 9 | 163 | 23 | 7 |
|  | 3 | 4.8 | 3.0 | 9 | 2 | 104 | 13 | 6 |
|  | 4 | 7.3 | 4.8 | 8 | 1 | 69 | 10 | 6 |

F480M (ring zone 20 pixels, DQ_SAT dilation 3 pixels):

| tile | `KFAC` | masked pixels (% of finite) | vetted rows masked (%) | leftovers ≥ 200 | leftovers ≥ 500 | craters ≥ 50 | craters ≥ 200 | craters ≥ 500 |
|---|---|---|---|---|---|---|---|---|
| o045 | 1 | 4.5 | 2.8 | 17 | 1 | 35 | 6 | 0 |
|  | 2 | 8.0 | 4.5 | 14 | 1 | 32 | 5 | 0 |
|  | 3 | 14.6 | 9.7 | 13 | 1 | 24 | 3 | 0 |
|  | 4 | 22.9 | 17.7 | 11 | 0 | 23 | 3 | 0 |
| o105 | 1 | 4.4 | 2.8 | 24 | 3 | 19 | 1 | 0 |
|  | 2 | 7.8 | 4.3 | 20 | 3 | 17 | 1 | 0 |
|  | 3 | 14.1 | 9.2 | 17 | 2 | 11 | 1 | 0 |
|  | 4 | 21.9 | 16.6 | 14 | 2 | 11 | 1 | 0 |
| o118 | 1 | 2.2 | 1.4 | 23 | 3 | 10 | 1 | 0 |
|  | 2 | 3.6 | 1.9 | 23 | 3 | 10 | 1 | 0 |
|  | 3 | 6.4 | 3.5 | 20 | 2 | 8 | 0 | 0 |
|  | 4 | 10.0 | 6.5 | 19 | 2 | 8 | 0 | 0 |
| o130 | 1 | 1.8 | 1.2 | 24 | 4 | 14 | 0 | 0 |
|  | 2 | 2.7 | 1.5 | 22 | 4 | 10 | 0 | 0 |
|  | 3 | 4.7 | 2.6 | 20 | 3 | 7 | 0 | 0 |
|  | 4 | 7.2 | 4.6 | 19 | 3 | 7 | 0 | 0 |
| o138 | 1 | 2.0 | 1.2 | 18 | 3 | 10 | 0 | 0 |
|  | 2 | 3.2 | 1.6 | 16 | 2 | 9 | 0 | 0 |
|  | 3 | 5.7 | 2.9 | 14 | 2 | 7 | 0 | 0 |
|  | 4 | 8.9 | 5.5 | 14 | 2 | 6 | 0 | 0 |

The masked area is the pixels that are finite in the production residual i2d and NaN after the mask, as a percentage of the
finite pixels (51.8 million per F212N tile, 12.5 million per F480M tile;
`results/kfac_masked_area.txt`).  The counts fall or stay with larger `KFAC` in every column of `results/kfac_counts.txt`, with
one exception: the craters ≥ 200 MJy/sr of o105 F212N go from 9 at `KFAC` 4 to 10 at `KFAC` 5.

- Between `KFAC` 1 and 4 the craters ≥ 50 MJy/sr fall by 62.5–69.5 % on F212N
  (110–280 → 35–105 per tile) and by 20.0–50.0 % on F480M
  (10–35 → 6–23).  Craters ≥ 200 MJy/sr go from 16–59 to
  6–18 on F212N and from 0–6 to 0–3 on F480M; craters
  ≥ 500 MJy/sr from 6–13 to 5–7 on F212N, and F480M has 0 of them at
  `KFAC` 4.
- The whole-star leftovers ≥ 200 MJy/sr go from 13–65 to 8–38 per tile on F212N
  and from 17–24 to 11–19 on F480M; leftovers ≥ 500 MJy/sr from 5–15 to
  1–7 on F212N and from 1–4 to 0–3 on F480M.  On F212N the
  leftovers ≥ 200 MJy/sr fall most in o045, o105 and o138 (by 27, 17 and 17) and by
  1 and 3 in o118 and o130.
- The cost is area.  `KFAC` 1 masks 1.0–3.2 % of the finite pixels of an F212N tile and
  1.8–4.5 % of an F480M tile; `KFAC` 4 masks 4.0–15.2 % and 7.2–22.9 %.  At
  `KFAC` 4 the masked vetted rows are 3.5–10.2 % (F212N) and 4.6–17.7 % (F480M) of the
  vetted rows on finite pixels.
- At `KFAC` 1 the masks of the tables (with the ring zone and the DQ_SAT dilation) already hold nearly every vetted row
  with a flux of 10⁵ or more on a finite pixel (512–1888 such rows per F212N tile, 18–67 per F480M
  tile).  The exceptions are o045 F212N (5 of 1888 rows), o105 F212N (1 of 1637 rows), o130 F212N (1 of 512 rows) and o138 F212N (2 of 947 rows).  The vetted rows that the larger disks add therefore have a flux below 10⁵,
  apart from those few.

Step from `KFAC` 3 to `KFAC` 4:

| tile and band | masked pixels added (percentage points) | leftovers ≥ 200 removed | leftovers ≥ 500 removed | craters ≥ 50 removed | craters ≥ 200 removed | craters ≥ 500 removed |
|---|---|---|---|---|---|---|
| o045 F212N | 5.3 | 0 | 0 | 23 | 5 | 0 |
| o105 F212N | 4.7 | 3 | 0 | 31 | 3 | 0 |
| o118 F212N | 1.7 | 0 | 0 | 35 | 7 | 0 |
| o130 F212N | 1.4 | 1 | 1 | 8 | 1 | 0 |
| o138 F212N | 2.6 | 1 | 1 | 35 | 3 | 0 |
| o045 F480M | 8.3 | 2 | 1 | 1 | 0 | 0 |
| o105 F480M | 7.8 | 3 | 0 | 0 | 0 | 0 |
| o118 F480M | 3.6 | 1 | 0 | 0 | 0 | 0 |
| o130 F480M | 2.5 | 1 | 0 | 0 | 0 | 0 |
| o138 F480M | 3.2 | 0 | 0 | 1 | 0 | 0 |

On F212N the step removes 8–35 craters ≥ 50 MJy/sr per tile and
1–7 craters ≥ 200 MJy/sr for 1.4–5.3 percentage points more masked area, and
0–3 leftovers ≥ 200 MJy/sr.  On F480M it removes 0–3
leftovers ≥ 200 MJy/sr and 0–1 craters ≥ 50 MJy/sr for
2.5–8.3 percentage points more masked area.  `KFAC` 3 masks 4.7–14.6 % of an F480M tile
against 7.2–22.9 % at `KFAC` 4 and leaves 0–3 more leftovers ≥ 200 MJy/sr and
0–1 more craters ≥ 50 MJy/sr per tile, so it is the cheaper setting on F480M.  This PR documents
`KFAC` 4, the largest value measured on every tile.

## With the edge band

The edge band (8 mosaic pixels in F212N, 6 in F480M) is added to the masks of the tables above.  Each cell
gives `KFAC` 1 → `KFAC` 4.

| tile and band | masked pixels (% of finite) | leftovers ≥ 200 | leftovers ≥ 500 | craters ≥ 50 | craters ≥ 200 | craters ≥ 500 |
|---|---|---|---|---|---|---|
| o045 F212N | 3.7 → 15.6 | 60 → 34 | 10 → 3 | 270 → 82 | 58 → 17 | 10 → 7 |
| o105 F212N | 3.2 → 13.7 | 38 → 25 | 1 → 1 | 280 → 105 | 44 → 9 | 9 → 6 |
| o118 F212N | 1.7 → 5.5 | 9 → 9 | 4 → 4 | 183 → 61 | 38 → 8 | 13 → 6 |
| o130 F212N | 1.5 → 4.5 | 16 → 16 | 4 → 4 | 109 → 34 | 16 → 6 | 6 → 5 |
| o138 F212N | 2.2 → 7.8 | 18 → 5 | 10 → 1 | 219 → 69 | 35 → 10 | 8 → 6 |
| o045 F480M | 5.2 → 23.6 | 16 → 10 | 1 → 0 | 33 → 22 | 6 → 3 | 0 → 0 |
| o105 F480M | 5.1 → 22.6 | 20 → 11 | 2 → 1 | 19 → 11 | 1 → 1 | 0 → 0 |
| o118 F480M | 2.9 → 10.7 | 20 → 16 | 2 → 1 | 9 → 7 | 1 → 0 | 0 → 0 |
| o130 F480M | 2.5 → 7.9 | 23 → 19 | 3 → 3 | 14 → 7 | 0 → 0 | 0 → 0 |
| o138 F480M | 2.7 → 9.6 | 18 → 14 | 3 → 2 | 10 → 6 | 0 → 0 | 0 → 0 |

At `KFAC` 4 the edge band lowers the leftovers ≥ 200 MJy/sr by 1–6 per F212N
tile and 0–3 per F480M tile, and the craters ≥ 50 MJy/sr by
0–2 (F212N) and 0–1 (F480M), for
about 0.5 percentage points (F212N) and about 0.7 percentage points (F480M) more masked area.

## What is left at `KFAC` 4

The craters of the `KFAC` 4 render, split by what could have made them (`results/kfac_remaining_craters.txt`;
the cells give the counts at depth ≥ 50 / ≥ 200 / ≥ 500 MJy/sr):

- `data`: depth more than twice the model peak at the crater.  The data are deeper than the model, so the model
  cannot have over-subtracted them.
- `near`: not `data`, and the nearest consolidated satstar row lies within three times its own `KFAC` 4 disk radius
  (ring zone and margin included).
- `far`: the rest.

| tile and band | craters ≥ 50 / ≥ 200 / ≥ 500 | `data` | `near` | `far` |
|---|---|---|---|---|
| o045 F212N | 83 / 18 / 7 | 6 / 6 / 6 | 54 / 6 / 0 | 23 / 6 / 1 |
| o105 F212N | 105 / 9 / 6 | 7 / 7 / 6 | 76 / 2 / 0 | 22 / 0 / 0 |
| o118 F212N | 63 / 8 / 6 | 6 / 6 / 6 | 38 / 1 / 0 | 19 / 1 / 0 |
| o130 F212N | 35 / 6 / 5 | 4 / 4 / 4 | 14 / 0 / 0 | 17 / 2 / 1 |
| o138 F212N | 69 / 10 / 6 | 6 / 6 / 6 | 39 / 2 / 0 | 24 / 2 / 0 |
| o045 F480M | 23 / 3 / 0 | 0 / 0 / 0 | 16 / 1 / 0 | 7 / 2 / 0 |
| o105 F480M | 11 / 1 / 0 | 0 / 0 / 0 | 9 / 1 / 0 | 2 / 0 / 0 |
| o118 F480M | 8 / 0 / 0 | 0 / 0 / 0 | 2 / 0 / 0 | 6 / 0 / 0 |
| o130 F480M | 7 / 0 / 0 | 0 / 0 / 0 | 1 / 0 / 0 | 6 / 0 / 0 |
| o138 F480M | 6 / 0 / 0 | 0 / 0 / 0 | 2 / 0 / 0 | 4 / 0 / 0 |

On F212N, 28 of the 30 craters ≥ 500 MJy/sr at `KFAC` 4 are `data` craters (o045 6 of 7, o105 6 of 6, o118 6 of 6, o130 4 of 5, o138 6 of 6); the other
2 are `far`.  The `data` craters number 4–7 per F212N tile at depth
≥ 50 MJy/sr, and `KFAC` 5 on o105 F212N leaves its 6 craters ≥ 500 MJy/sr unchanged.  The craters ≥ 200 MJy/sr that are
not `data` craters number 12, 2, 2, 2 and 4 in o045, o105, o118, o130 and o138.  On F480M no crater is a `data` crater, and
0–3 craters ≥ 200 MJy/sr remain per tile.

### The same scan on a flagged render of o105 F212N

The production m7 render of o105 F212N was repeated with three render flags (`SATURATED_PIXEL_HANDLING.md`
documents them) and the production options and vetted catalog:

- `SINGLE_EXPOSURE_EVENT_MASK=1` sets DO_NOT_USE on the pixels of a compact saturated core that one frame carries and
  the other covering frames do not,
- `NEGATIVE_PIXEL_MASK_NSIGMA=50` sets DO_NOT_USE on pixels with SCI below −50 × ERR, and
- `MERGE_ROW_RESID_SUPPORT_MIN=0.3` re-renders a row at a lower flux when the PSF fit to the merged-catalog residual of its
  frame has an amplitude below −0.7 × the flux of the row.

The mask was applied at `KFAC` 1 to 5 with the same ring zone and DQ_SAT dilation (`results/kfac_counts.txt`, second
block).

| `KFAC` | leftovers ≥ 50 | ≥ 100 | ≥ 200 | ≥ 500 | craters ≥ 50 | ≥ 200 | ≥ 500 | with the edge band: leftovers ≥ 200 | craters ≥ 50 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 752 | 201 | 48 | 6 | 35 | 9 | 1 | 38 | 35 |
| 2 | 645 | 147 | 38 | 4 | 34 | 9 | 1 | 30 | 34 |
| 3 | 619 | 137 | 34 | 3 | 23 | 1 | 0 | 28 | 23 |
| 4 | 594 | 127 | 31 | 3 | 13 | 0 | 0 | 25 | 13 |
| 5 | 560 | 117 | 30 | 3 | 4 | 0 | 0 | 24 | 4 |

Against the production source of the same tile (first F212N table), the flagged render has the same leftovers ≥ 200 and
≥ 500 MJy/sr at every `KFAC` and fewer craters.  At `KFAC` 4 it has 13 against 105 craters ≥ 50 MJy/sr, 0 against
9 craters ≥ 200 MJy/sr and 0 against 6 craters ≥ 500 MJy/sr; at `KFAC` 1 it has 35 against 280 craters ≥ 50 MJy/sr.
In the flagged render the craters ≥ 50 MJy/sr fall from 35 at `KFAC` 1 to 4 at `KFAC` 5, and the leftovers ≥ 200 MJy/sr from
48 to 30.

## The switch against the emulation

The production function (`residual_footprint_mask` at `KFAC` 4, `MARGIN_PX` 2, ring zone `flux_fit` ≥ 10⁶ to 25 pixels)
was applied to a scratch copy of the production residual i2d of o105 F212N and compared with the emulation of the counts
above (`results/kfac_switch_vs_emulation.txt`; the DQ_SAT dilation and the edge band were compared in the PRs that add
them).  The copy has 9,123,221 NaN pixels and the emulation 9,123,221; 0 pixels are NaN only in the copy,
0 only in the emulation, and every finite value is identical.  Both mask 13.039 % of the finite pixels of the
production residual.  The switch masked 10,356 rows (median disk radius 13.0 mosaic pixels,
0.41″; largest 129.8 pixels) and 239 ring-zone rows, and the header cards are
`FPMASKK` 4.0, `FPMASKM` 2.0, `FPMASKN` 10356, `FPRINGF` 1000000.0, `FPRINGR` 25.0 and `FPRINGN` 239.  The
other tiles and bands were not run through the production function.

## A radial profile correction of the satstar model

A model-side alternative to the mask, tried on o045 and o105 F212N.  For isolated satstar rows (no other satstar row
within the correction radius + 1 pixels) with `flux_fit` ≥ 10⁵, the median ratio ρ(r; F) of the residual to the star's model
was measured in 1-pixel annuli in six flux bins (at least 26 rows per bin) and interpolated in log₁₀ `flux_fit`.  Of
the 11572 satstar rows of o045 and the 10356 of o105, 1822 and 1574 are isolated with
`flux_fit` ≥ 10⁵ at the 14-pixel setting, and 1130 and 1038 at the 30-pixel setting.  For each of the
2053 (o045) and 1746 (o105) rows with `flux_fit` ≥ 10⁵ the correction subtracts ρ(r; F) × (the median profile of
model / `flux_fit` of the isolated stars, in 1-pixel annuli) × `flux_fit` from the residual within 14 pixels (taper to zero
at 16) or within 30 pixels (taper to 34).  Counts on the production source without a mask, with the mask at `KFAC` 1 and
at `KFAC` 4 (ring zone and DQ_SAT dilation of the tables above):

| tile | mask | correction | leftovers ≥ 50 | ≥ 100 | ≥ 200 | ≥ 500 | craters ≥ 50 | ≥ 200 | ≥ 500 |
|---|---|---|---|---|---|---|---|---|---|
| o045 | none | none | 1085 | 376 | 190 | 140 | 554 | 220 | 84 |
|  |  | within 14 px | 1071 | 371 | 192 | 140 | 611 | 246 | 97 |
|  |  | within 30 px | 979 | 359 | 194 | 141 | 557 | 239 | 97 |
|  | `KFAC` 1 | none | 971 | 239 | 65 | 15 | 272 | 59 | 10 |
|  |  | within 14 px | 953 | 231 | 64 | 15 | 299 | 64 | 10 |
|  |  | within 30 px | 858 | 222 | 63 | 15 | 253 | 70 | 11 |
|  | `KFAC` 4 | none | 794 | 169 | 38 | 7 | 83 | 18 | 7 |
|  |  | within 14 px | 794 | 169 | 38 | 7 | 83 | 18 | 7 |
|  |  | within 30 px | 718 | 162 | 36 | 7 | 79 | 18 | 7 |
| o105 | none | none | 845 | 317 | 178 | 133 | 489 | 166 | 71 |
|  |  | within 14 px | 832 | 313 | 177 | 132 | 524 | 174 | 73 |
|  |  | within 30 px | 751 | 301 | 174 | 131 | 459 | 168 | 70 |
|  | `KFAC` 1 | none | 748 | 201 | 48 | 6 | 280 | 44 | 9 |
|  |  | within 14 px | 730 | 191 | 47 | 6 | 303 | 45 | 9 |
|  |  | within 30 px | 653 | 184 | 46 | 6 | 244 | 47 | 10 |
|  | `KFAC` 4 | none | 593 | 127 | 31 | 3 | 105 | 9 | 6 |
|  |  | within 14 px | 593 | 127 | 31 | 3 | 105 | 9 | 6 |
|  |  | within 30 px | 539 | 124 | 31 | 3 | 92 | 9 | 6 |

Without a mask the correction changes the craters ≥ 50 MJy/sr by +57 and +3 in o045 and +35 and −30 in o105 (14 and 30 pixels), and raises the craters
≥ 200 MJy/sr in all four cases (220 → 246 and 239 in o045,
166 → 174 and 168 in o105).  With the `KFAC` 4 mask the 14-pixel
correction leaves every count unchanged.  The 30-pixel correction lowers the craters ≥ 50 MJy/sr by
4 (o045) and 13 (o105), the leftovers ≥ 50 MJy/sr by
76 and 54, and the leftovers ≥ 200 MJy/sr by
2 and 0; it leaves the craters ≥ 200 MJy/sr
(18 and 9) and ≥ 500 MJy/sr (7 and 6) unchanged.  The ρ profiles and
the per-bin row counts are in `results/profile_correction.txt`.  The correction is not part of any PR of the stack.

## Setting for a run

The variables behind the tags of this section (the mask is off unless `KFAC` is above 0, and the ring zone, the DQ_SAT
dilation and the edge band act only with it; `MARGIN_PX` and the passes keep their defaults, 2 and `m7`), with the F480M
values where they differ:

```
MERGEDCAT_FOOTPRINT_MASK_KFAC=4              # 3 gives nearly the same F480M counts for less area (step table)
MERGEDCAT_FOOTPRINT_MASK_RING_FLUX_MIN=1e6
MERGEDCAT_FOOTPRINT_MASK_RING_RADIUS_PX=25   # 20 for F480M
MERGEDCAT_FOOTPRINT_MASK_DQSAT_PX=4          # 3 for F480M
MERGEDCAT_FOOTPRINT_MASK_EDGE_PX=8           # 6 for F480M
```

The ring zone, the DQ_SAT dilation and the edge band belong to other PRs of the stack (#1364, #1372, #1373).

## What this does not show

- The masks hide pixels.  The model, the catalogs and the photometry do not change, and a leftover or a crater that falls
  under the mask is not fixed.  At `KFAC` 4 the mask covers 4.0–15.2 % (F212N) and 7.2–22.9 % (F480M)
  of the finite pixels and 3.5–17.7 % of the vetted rows on finite pixels, so any use of the
  residual i2d that needs those pixels (completeness tests, residual photometry, diffuse-emission work) loses them.
- The counts come from an emulation on the production residual i2d.  Only the disks and the ring zone at `KFAC` 4 were
  compared with the production function, on o105 F212N; the DQ_SAT dilation and the edge band were compared in their own
  PRs.  No full m7 and finalize run with `KFAC` 4 exists.
- The ring-zone radius (25 pixels in F212N, 20 in F480M) comes from the ring-zone PR, where it was set on F212N replays and,
  at 20 pixels, changed no F480M crater count; it was not re-tuned for the larger disks.  The F480M ring zone widens
  6–42 rows per tile.
- The `data` craters at `KFAC` 4 (4–7 per F212N tile at depth ≥ 50 MJy/sr, none on F480M) are deeper than the model, so an
  over-subtracted model cannot make them; the flagged render of o105 F212N has 0 craters ≥ 500 MJy/sr at `KFAC` 4 against
  6 in the production source.  The `near` and `far` craters at `KFAC` 4 (31–98 per F212N tile and
  6–23 per F480M tile at depth ≥ 50 MJy/sr) have a depth of at most twice the model peak, so an over-subtracted model can
  make them, and a larger disk or a different model would have to address them.
- The flagged-render scan covers one tile and one band.  The profile correction is one parametrisation (a median radial
  profile in six flux bins) on two tiles in F212N; it does not exclude a per-star or two-dimensional correction.
