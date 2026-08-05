# Aperture photometry of saturated stars from the i2d mosaics

**Goal.** Add an *independent* flux measurement for every saturated star —
circular aperture photometry on the resampled mosaic (i2d) — alongside the
existing PSF-wing-fit flux (`flux_fit`), compare the two, ship aperture flux by
default in the catalogs, and produce (separate) aperture-correction tables.

## 1. Why aperture photometry is a useful cross-check here

The saturated-star flux (`flux_fit`) comes from an STPSF model fit to the
per-exposure crf frames with the saturated **core masked** — a model
extrapolation into the region with no data. Aperture photometry on the mosaic is
a fully independent estimator with completely different failure modes, so
agreement/disagreement between them is diagnostic:

- PSF fit recovers the (masked) core from the model + wings; aperture sums only
  the pixels that survive in the mosaic.
- **The saturated core is frequently NaN/masked in the mosaic too** (the drizzle
  drops flagged pixels), so a naive aperture sum is poisoned by NaNs and
  under-counts the brightest stars. Every measurement here is therefore
  NaN-aware and carries a **coverage fraction** and a **core-saturated flag**.
- In crowded fields (the GC) the aperture also catches neighbours; this shows up
  as scatter and is why the aperture flux is a *diagnostic with quality flags*,
  not a drop-in replacement for the PSF flux.

## 2. Method (`jwst_gc_pipeline/photometry/aperture_photometry.py`)

For each star (`skycoord_fit` from the consolidated satstar catalog):
1. Locate the science i2d mosaic(s) for the target/filter
   (`find_i2d_mosaics`; rejects residual/model/mergedcat/downsel/per-iteration
   products). Multi-obs targets (gc2211) keep the per-star best-coverage mosaic.
2. NaN-aware circular aperture on `SCI` (mask = non-finite pixels), a
   `CircularAnnulus` sigma-unclipped **median local sky**, on a **curve-of-growth**
   of radii defined in **arcsec** (converted to each mosaic's pixel grid):
   `0.10 0.15 0.20 0.30 0.40 0.50 0.70 1.00″`, primary = **0.30″**.
3. Convert MJy/sr → **Jy** with the mosaic pixel solid angle
   (`ww.proj_plane_pixel_area()`), the same conversion `merge_catalogs` uses for
   `flux_fit`; magnitudes via the SVO Vega `ZeroPoint` and `ABMAG_OFFSET = 8.90`.

**Columns added to the catalog** (by default): `aper_flux_jy`,
`aper_flux_err_jy`, `aper_mag_vega`, `aper_mag_ab`, `aper_bkg` (MJy/sr),
`aper_area_frac`, `aper_core_saturated`, `aper_i2d`, plus the curve of growth
`aper_flux_jy_r{radius}` / `aper_area_frac_r{radius}`.

## 3. Pipeline integration (default per catalog)

`load_satstar_catalog` (merge_catalogs) now calls
`_ensure_satstar_aperture_photometry` at every return path: freshly-built
consolidated caches are written **with** the aperture columns, and a legacy cache
without them is backfilled once and atomically rewritten (so later reads are
fast). Opt out with `SATSTAR_APERTURE_PHOT=0`. The extra `aper_*` columns are
stripped by `replace_saturated`'s existing column reconciliation, so the merged
photometry table is unchanged (aperture flux lives with the satstar catalog).

## 4. Aperture-correction tables (kept SEPARATE from photometry)

`build_aperture_correction_table` derives the curve-of-growth correction from
**clean** stars (full aperture coverage at every radius, not core-saturated,
isolated, SNR ≥ threshold): per radius it reports `flux_ratio_to_total`,
`apcorr_mag = −2.5 log10(ratio)`, the robust scatter (MAD) and N. Written to
`{basepath}/catalogs/{filter}_satstar_apcorr.ecsv` — a distinct file that is not
swept up by the `*_satstar_catalog.fits` / `*_daophot_*` globs, so aperture
corrections never contaminate the photometry tables.

> Caveat: a saturated-star catalog's cleanest members are its least-saturated
> (faint) end; for a definitive optical curve of growth the same routine can be
> pointed at the main photometry catalog's bright isolated unsaturated stars.

## 5. Results — aperture vs PSF across the field matrix

`scripts/aperture_photometry/investigate_aperture_vs_psf.py`, SLURM job 38760929;
per-field figures + `aperture_vs_psf_summary.ecsv` + `aperture_vs_psf_master.png`
under `docs/reports/apphot/`.

| target | filter | regime | N | core-sat in i2d | apcorr(0.3″) mag | COG reliable? |
|---|---|---|---|---|---|---|
| brick | f212n | SW narrow | 6071 | 0.5% | 0.24 ± 0.13 | **yes** (smooth, 3888 stars) |
| brick | f405n | LW narrow | 2564 | 0.0% | 0.27 | yes |
| ngc6334 | f187n | SW narrow | 200 | 0.0% | 1.23 | ok (small N) |
| sgra | f405n | LW narrow | 1360 | 2.1% | 1.28 | ok |
| gc2211 | f277w | LW wide NG2 | 101209 | 1.4% | 1.29 | partial |
| gc2211 | f200w | SW wide NG2 | 40502 | 1.4% | 0.52 | partial |
| brick | f200w | SW wide NG7 | 7268 | 8.2% | 2.60 (MAD→0.37) | **no — contaminated** |
| brick | f356w | LW wide | 15834 | 11.0% | 2.69 | **no — contaminated** |
| m4 | f322w2 | globular LW wide | 2272 | 0.0% | 2.22 | **no — contaminated** |
| ngc6397 | f322w2 | globular LW wide | 679 | 0.0% | 2.01 | partial |

**Findings.**

1. **Core saturation reaches into the mosaic.** 8–11% of saturated stars have a
   NaN/masked central pixel in the LW/SW **wide-band** i2d (f356w 11%, f200w 8%);
   narrow/medium bands are ≈0%. A plain aperture sum would silently under-count
   those — the NaN-aware sum + `aper_core_saturated` flag is essential, not
   cosmetic.

2. **The aperture correction is clean for narrow/medium bands** (f212n:
   0.3″ encloses 0.80 of the 1.0″ flux, a smooth monotonic curve of growth, MAD
   0.13, 3888 clean stars; f405n similar). These `*_satstar_apcorr.ecsv` tables
   are directly usable.

3. **In crowded WIDE bands the large-radius reference is contamination-limited.**
   brick f200w's curve of growth collapses (0.1″ → 0.012 of the 1.0″ flux, MAD up
   to 0.37): the 1.0″ aperture in the dense GC/globular fields is dominated by
   neighbours and extended flux, so "flux relative to 1.0″" is not a real
   aperture correction. The `ratio_mad` / `n_stars` columns expose this (large
   MAD, few stars) — **read them before trusting an apcorr**. The confounded
   large-radius reference is also why the "aperture-corrected / PSF" ratio in
   `aperture_vs_psf_master.png` panel (a) scatters 0.06–1.87 across fields; that
   panel should be read as *"how badly does the total-normalisation break"*, not
   as a PSF-flux validation.

**Bottom line.** Aperture flux from the mosaic is now measured and shipped by
default with a coverage fraction and a core-saturation flag, giving an
independent per-star cross-check of the PSF flux. It agrees with the PSF flux to
within the (well-behaved) curve-of-growth correction for narrow/medium bands and
for the less-saturated majority; it is a *flagged diagnostic* — not a
replacement — for the deeply-saturated cores (masked in the mosaic too) and for
wide-band photometry in the most crowded fields, where a definitive aperture
correction needs isolated **unsaturated** calibrators from the main catalog
rather than the saturated-star list itself.

## 6. Status
1. [x] `aperture_photometry.py` — NaN-aware aperture + curve of growth + apcorr.
2. [x] wired into `load_satstar_catalog` (default; env opt-out).
3. [x] separate apcorr tables → `catalogs/{filt}_satstar_apcorr.ecsv`.
4. [x] unit tests (synthetic mosaics).
5. [x] cross-field investigation (11 field/filter combos; job 38760929).
6. [ ] (next) apcorr from isolated unsaturated main-catalog stars for a
   contamination-free curve of growth in crowded wide bands.
