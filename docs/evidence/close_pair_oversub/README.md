# Close pairs leave negative holes in the m7 residual

## Finding

The m2->m8 production rerun (#1086 own-band m7 seed, #1107 companion flux-ratio
gate) catalogues most of the starlike residuals the previous m7 left behind.
The phase-loss audit (`results/*_before_after.json`, `scripts/compare.py`)
re-measures 300 pre-rerun starlike-residual positions per field and band in the
rerun m7 residual:

| field band | now in m7 vetted | resid S/N < -5 after | random positions S/N < -5 (of 100) |
|---|---|---|---|
| arches F212N | 204 | 52 | 4 |
| brick F182M | 215 | 56 | 1 |
| brick F212N | 217 | 40 | 0 |
| brick F405N | 167 | 35 | 7 |
| sgra F212N | 160 | 43 | 4 |
| sgrc F212N | 201 | 60 | 1 |

Before the rerun none of these positions had S/N < -5. The new negative
residuals sit on catalogued stars.

Across the whole m7 vetted catalogue (`scripts/nn_oversub.py`, 60,000 random
vetted stars per band), the fraction of stars whose m7 residual has
PSF-matched S/N < -5 depends on the separation to the nearest other vetted
star (`results/table.md`, `figures/curves.png`):

| field band (m7) | n vetted | 0-0.75 | 0.75-1.25 | 1.25-1.5 | 1.5-2 | 2-3 | 3-5 | 5-inf | all | NN < 1.5 FWHM |
|---|---|---|---|---|---|---|---|---|---|---|
| arches F212N after | 291,593 | 87% (38) | 63% (2268) | 39% (5964) | 18% (10995) | 6% (17641) | 5% (20467) | 8% (2619) | 13.5% | 13.8% |
| brick F182M after | 388,930 | 100% (66) | 76% (635) | 54% (2181) | 36% (10035) | 10% (15959) | 7% (23758) | 11% (7342) | 15.8% | 4.8% |
| brick F212N after | 195,636 | 99% (233) | 58% (1217) | 36% (3453) | 16% (6248) | 5% (11319) | 5% (22941) | 7% (14572) | 10.0% | 8.2% |
| brick F405N after | 88,722 | 95% (1572) | 69% (5273) | 38% (6116) | 24% (9293) | 6% (12242) | 4% (19520) | 7% (5921) | 19.3% | 21.6% |
| cloudc F212N after | 233,665 | 100% (56) | 54% (1002) | 32% (2486) | 16% (5347) | 4% (10229) | 4% (23584) | 4% (17207) | 7.0% | 5.9% |
| sgra F212N after | 241,300 | 82% (39) | 59% (1811) | 37% (5049) | 21% (9511) | 6% (16735) | 5% (22411) | 12% (4027) | 12.5% | 11.5% |
| sgrc F212N after | 358,790 | 97% (113) | 63% (2341) | 39% (4916) | 17% (9986) | 5% (17484) | 4% (21752) | 5% (3405) | 11.9% | 12.3% |
| brick F182M before | 264,057 | - (3) | - (16) | 65% (65) | 38% (1731) | 9% (9744) | 6% (31356) | 13% (17055) | 9.5% | 0.1% |
| brick F212N before | 111,909 | - (7) | 78% (118) | 60% (705) | 40% (1730) | 15% (6166) | 11% (24218) | 15% (27034) | 14.6% | 1.4% |
| brick F405N before | 67,553 | 96% (552) | 76% (2079) | 50% (2551) | 31% (6211) | 7% (12311) | 4% (26235) | 7% (9996) | 13.4% | 8.6% |

Columns 3-9: fraction with S/N < -5 (number of sampled stars) per
nearest-neighbour bin in units of the filter FWHM. "before" is the
2026-09-03 brick m7 (pre-#1086); "after" is the rerun m7. The cloudc row is the
2026-10-18 m7, re-catalogued after the #1185 cosmic-ray fix.

The over-subtracted fraction per separation bin is similar before and after.
The number of close pairs grew: in brick, the fraction of m7 vetted stars with
a neighbour closer than 1.5 FWHM went from 0.1% to 4.8% (F182M), 1.4% to 8.2%
(F212N) and 8.6% to 21.6% (F405N).

Every source in these catalogues has `group_size == 1`: production m2-m7
runs with `--group` off (cataloging.py passes the grouper to
`PSFPhotometry` only when `--group` is set), so each source is fitted on its
own.

Two regimes show up in the galleries (`figures/gallery_*_after.png`; random
S/N < -5 stars per separation bin, data above m7 residual, orange x = m7
vetted sources):

1. **< 0.75 FWHM: one star, two or three catalogue entries.** 82-100% of these
   are over-subtracted. The neighbour flux ratio has median 1.0
   (16-84%: 0.72-1.39 in F405N, 0.53-2.46 in F182M, 0.60-1.60 in sgrc F212N;
   for 3-5 FWHM pairs it is 0.21-4.3). The two entries often came from
   different iterations (`iter_found` equal in 30-53% of these pairs, against
   86-89% at 3-5 FWHM). They make up 2.6% of F405N m7 vetted stars and
   0.06-0.39% in the SW bands.
2. **0.75-2 FWHM: elongated blends and close real pairs.** 16-76% are
   over-subtracted. The likely mechanism is that each independent fit absorbs
   part of its neighbour's light, so the residual subtracts the shared light
   twice. `grouped_fit/README.md` tests this with grouped fits on the
   reference fields.

Isolated stars (> 5 FWHM) are over-subtracted at 5-12%; that floor is a
separate PSF-model / bright-star effect and is the reference level for the
pair excess.

## Where the near-duplicates come from

`_build_m7_band_seed` (cataloging.py) builds each band's m7 seed as the
cross-band seed plus that band's own m6 vetted sources:

* an own-band source joins when it lies at least
  `manual_crossband_seed_max_sep_mas` (30 mas) from every cross-band position;
* a cross-band position takes the flux of the own-band source within 30 mas,
  and flux 1.0 when there is none;
* the companion cut drops an own-band source only when a seed source within
  2.5 FWHM is more than 1/0.1 times brighter, so a flux-1.0 cross-band seed
  never triggers it.

When a star's own-band m6 position lies 30-70 mas from its cross-band position,
the seed therefore holds both. 30 mas is 0.22 FWHM in F405N (136 mas) and
0.48 FWHM in F182M (62 mas). Nothing in the union deduplicates at a fraction
of the FWHM; the 0.5 FWHM `DEDUPMAS` applies only to the residual-image
detections `_build_i2d_augmented_seed` adds afterwards.

`scripts/seed_pairs.py` on the brick m7 band seeds
(`results/brick_seed_pairs.json`, `figures/brick_seed_pairs.png`):

| band | seeds | NN < 0.5 FWHM | pairs < 0.5 FWHM | cross-band + own-band | of which cross-band flux = 1.0 | own + own | cross + cross | pair sep p10/50/90 (mas) |
|---|---|---|---|---|---|---|---|---|
| F182M | 388,282 | 0.04% | 69 | 41 | 41 | 26 | 2 | 18 / 30 / 31 |
| F212N | 195,164 | 0.39% | 380 | 187 | 180 | 181 | 12 | 22 / 32 / 36 |
| F405N | 98,604 | 2.3% | 1147 | 778 | 724 | 318 | 51 | 31 / 41 / 61 |

The cross-band + own-band pair separations start at 30 mas, the
`max_sep_mas` boundary. The own + own pairs come from the m6 vetted catalogue
itself (0.92% of F405N m6 vetted stars have a neighbour under 0.5 FWHM); the
in-pass post-fit dedup radius is 1.0 px, 0.46 FWHM in F405N.

## Candidate fixes

1. **Grouped fitting** (`--group --max-group-size=N`, grouping radius
   `manual_group_min_sep_fwhm` = 2.0 FWHM): fits sources that share light
   together. Addresses regime 2 and part of regime 1. Tested on the
   reference fields in `grouped_fit/README.md`.
2. **Seed deduplication in `_build_m7_band_seed`**: match own-band to
   cross-band positions within max(`max_sep_mas`, 0.5 FWHM), carry the
   own-band flux onto the matched cross-band position, and deduplicate
   own-band + own-band seeds at 0.5 FWHM. Addresses regime 1 at its source.
3. **Post-fit merge**: combine m7 vetted pairs closer than ~0.5 FWHM into one
   source before the residual is built.

## Data trees

Brick products come from `/blue/adamginsburg/adamginsburg/jwst/brick`, where
`fields.yaml` (`brick: root: blue`) sends every brick run since 2026-08-21
(see #937); the `/orange` brick tree holds the 2026-09-03 copy used as
"before". `scripts/nn_oversub.py` and `scripts/compare.py` read the rerun
products through `AFTER_REMAP=/orange/adamginsburg/jwst/brick=/blue/adamginsburg/adamginsburg/jwst/brick`.

## Scripts

Run from a scratch layout holding the pre-rerun snapshot
(`prod_before/<field>/<band>/{snapshot.json,cutouts.npz}`) and the phase-loss
audit's `phase_loss.py`; the `sys.path` lines at the top of each script point
there.

* `nn_oversub.py <field> <FILT> <before|after> <out-prefix> [nsample]` --
  residual S/N against nearest-neighbour separation.
* `table.py` -- `results/table.md` from the `nn_oversub.py` JSONs.
* `figs.py [<field> <FILT> <before|after>]` -- `curves.png` and one gallery.
* `seed_pairs.py <catalog-dir> <out-prefix> FILT:fwhm_mas ...` -- close
  pairs in the m7 band seeds by origin.
* `compare.py <field> <FILT> <out-prefix>` -- the phase-loss before/after
  figure and JSON (`figures/brick_f405n_before_after.png`;
  `figures/sgrc_f212n_neg_rows.png` is a crop of the sgrc figure's "most
  negative after" rows: data | before m7 residual | after m7 residual, red
  circle = the pre-rerun starlike-residual position).
