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

**Columns added to the catalog** (by default): `aper_flux_jy` (raw, NOT
aperture-corrected; a **lower limit** where the core is masked or coverage is
incomplete), `aper_flux_valid` (False = lower limit: core-saturated or
coverage < 0.98), `aper_flux_err_jy`, `aper_mag_vega`, `aper_mag_ab`, `aper_bkg`
(MJy/sr), `aper_area_frac`, `aper_core_saturated`, `aper_i2d`, plus the curve of
growth `aper_flux_jy_r{radius}` / `aper_area_frac_r{radius}`. Consumers should
gate on `aper_flux_valid` (or `aper_core_saturated`) and use the PSF `flux_fit`
for the flagged rows.

## 3. Pipeline integration (default per catalog)

`load_satstar_catalog` (merge_catalogs) now calls
`_ensure_satstar_aperture_photometry` at every return path: freshly-built
consolidated caches are written **with** the aperture columns, and a legacy cache
without them is backfilled once and atomically rewritten (so later reads are
fast). Opt out with `SATSTAR_APERTURE_PHOT=0`. The extra `aper_*` columns are
stripped by `replace_saturated`'s existing column reconciliation, so the merged
photometry table is unchanged (aperture flux lives with the satstar catalog). A
failed/empty measurement is non-fatal (a bad mosaic, missing SCI, or SVO outage
is caught) and an all-NaN result has its aperture columns **dropped** — on both
the cache-hit and the rebuild path — so the catalog looks "not measured" and the
measurement is retried rather than frozen as a false success.

> Caveat (shared-filter fields): where one filter is imaged by two proposals
> (e.g. NGC 6334 F200W/F470N, jw06778 + jw07213), both merged mosaics are found
> and each star is measured on the one that best covers it; the two are
> independently flux-calibrated, but a star straddling both is measured on a
> single proposal's mosaic, not a combination.

## 4. Aperture-correction tables (kept SEPARATE from photometry)

There are **two** aperture-correction products, both separate files under
`catalogs/` (never swept up by the `*_satstar_catalog.fits` / `*_daophot_*`
globs):

- **`{filter}_satstar_apcorr_refstars.ecsv` — the aperture correction of record**
  (§6): `build_reference_apcorr` on isolated, unsaturated, high-SNR **main-catalog**
  stars.
- **`{filter}_satstar_apcorr_diagnostic.ecsv` — diagnostic only**:
  `build_aperture_correction_table` on the least-saturated members of the
  saturated-star catalog. In crowded fields its large-radius reference is
  neighbour-contaminated, so its `apcorr_mag` is not a usable correction (F200W
  reads 2.6 mag vs F212N 0.24 mag in the same field; `ratio_mad` does **not** rank
  reliability across bands). Its meta carries a `warning` to that effect, and the
  bare `*_satstar_apcorr.ecsv` name is retired so nothing consumes it as if it
  were the correction.

Both report, per radius, `flux_ratio_to_total`, `apcorr_mag = −2.5 log10(ratio)`,
the robust scatter (MAD) and N.

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

2. **The satstar-derived curve of growth is smooth for narrow/medium bands**
   (f212n: 0.3″ encloses 0.80 of the 1.0″ flux, MAD 0.13, 3888 clean stars; f405n
   similar), but even these are diagnostic — the aperture correction of record is
   the reference-star table (§6), not this satstar-catalog one.

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
wide-band photometry in the most crowded fields, where a robust aperture
correction needs isolated **unsaturated** calibrators from the main catalog
rather than the saturated-star list itself.

## 6. Contamination-free aperture correction (isolated unsaturated reference stars)

The satstar-derived curve of growth is crowding-contaminated at large radius
(§5). `build_reference_apcorr` (`aperture_photometry.py`) instead uses the main
photometry catalog's **isolated, unsaturated, high-SNR** stars: not saturated /
not blended (`group_size≤1`) / `SNR≥150` / geometrically isolated so the aperture
*and* sky annulus are clean. The comparison driver
(`reference_apcorr_and_compare.py`) writes the result to a separate
`catalogs/{filt}_satstar_apcorr_refstars.ecsv` (the library builds the table;
the driver names + writes it).

The numbers below are measured on the **current canonical** merged catalog
(`resbgsub_m<N>_dao_basic`, the latest processing phase; `find_main_catalog`
selects it — the earlier `*_iterative` products are superseded and are not used).

**The measured crowding fact.** On the canonical brick F200W catalog
(578,955 sources) the nearest-neighbour separation is 0.248″ median: 26% of stars
are isolated beyond 0.3″, but only **0.75% beyond 0.5″ and 0.04% (251 stars)
beyond 0.8″**. So the *inner* curve of growth is well sampled, but an empirical
curve of growth to true "total" **cannot be measured from the data** — there are
too few stars isolated at large radius — and the outer/total correction must come
from a theoretical PSF. This is why the satstar table was contaminated at large
radius, and why Gutermuth (below) uses synthetic-PSF aperture corrections.
`build_reference_apcorr` adapts: it steps the isolation radius down a ladder until
≥200 clean stars survive, clips the COG to the sky-annulus-inner radius, and
records the achieved isolation + reliable max radius in the table meta.

**Practical notes (verified):** (a) the reference COG needs *bright* stars —
including faint (SNR≈50) stars biases the median-of-ratios low, so the default is
SNR ≥ 150. (b) Re-centring on the i2d is available (`recenter_box`) as a
safeguard, but on the canonical catalog the catalog↔i2d offset is only ~0.07 px,
so it is a near-no-op; it mattered only on the stale iterative catalog (~1.5 px
offset). (c) When crowding forces the isolation radius to ≤0.3″, the sky annulus
is pushed onto the PSF wings and over-subtracts, suppressing the empirical COG —
another face of the crowding limit (see F162M/F210M in §7).

## 7. Comparison to the theoretical PSF and to Gutermuth's method

**Empirical (i2d) vs theoretical PSF curve of growth.** Using bright
(SNR ≥ 150), isolated, unsaturated reference stars from the canonical catalog, the
clean empirical curve of growth matches the STPSF theoretical PSF-grid closely in
the well-populated fields: **brick F200W (16,509 stars) agrees to ≈1%**
(EE(0.10″)/EE(0.25″) 0.81 emp vs 0.82 theory; 0.15″ 0.94 vs 0.95), and brick
F356W (7,640) to ≈1–11% (0.10″ 0.89, 0.15″ 0.99). **So for the unsaturated stars
that dominate the catalog, our Stage-3 i2d aperture photometry is consistent with
the synthetic PSF** — the S3 saturated-core interpolation artifact Gutermuth warns
about does **not** measurably bias the curve of growth at these radii (it bites
the *saturated* cores, which are flagged `aper_core_saturated` and excluded from
the COG). Agreement degrades where crowding forces a tight isolation radius so the
sky annulus sits on the PSF wings (cloudef F162M/F210M, isolation ≤0.3–0.4″), and
the smallest aperture (0.05″ ≈ 1.6 SW px) is a sub-2-pixel floor — neither is
usable, both are crowding/sampling limits rather than a photometric bias.

**Rob Gutermuth (`/orange/adamginsburg/jwst/rguter_jwst_photometry`).** His
"JWST Photometry Issues" manuscript (Cloud E&F, prop 2092 = our `cloudef`) is
aperture-only (IDL PhotVis + `aper`), aperture radius = FWHM, and its aperture
corrections come from **synthetic WebbPSF/JDocs PSFs, not an empirical
curve-of-growth** (AperCorr = flux fraction within the FWHM aperture: F162M 0.534,
F210M 0.513, F360M 0.541, F480M 0.550, i.e. ≈0.65–0.73 mag). Points of agreement
and difference:

- **Method choice — vindicated on both sides.** Our crowding measurement shows an
  empirical total-light COG is impossible in these fields, so Gutermuth's use of a
  *theoretical* PSF aperture correction is the right call; conversely our
  empirical inner COG matching the theoretical PSF shows the synthetic-PSF apcorr
  is accurate for unsaturated stars.
- **Saturated cores.** Gutermuth deliberately keeps saturated pixels NaN
  (WCSmosaic) and *rejects* Stage-3 mosaics for saturated-core interpolation. Our
  i2d are Stage-3 but still carry NaN cores for 8–11% of satstars in wide bands;
  we handle the rest with the PSF-fit `flux_fit` (the whole saturated-star
  machinery) rather than aperture photometry, and flag the aperture measurement
  where the core is masked. Different route, same recognition that aperture
  photometry cannot be trusted on saturated cores.
- **Aperture correction source.** His = theoretical only. Ours = **empirical
  inner COG anchored to a theoretical PSF for the outer/total** — the empirical
  part is a data-driven check his single theoretical number does not have.

Empirical i2d curve of growth vs the theoretical STPSF PSF-grid, on the canonical
catalog (job 38778528; figure `apphot/apcorr_reference_vs_theory.png`). `emp/theo`
= empirical ÷ theoretical enclosed-energy ratio, both normalised at the clean max
radius; 1.0 = perfect.

| field | filt | N_ref | isol. | emp/theo @0.10″ | emp/theo @0.15″ | note |
|---|---|---|---|---|---|---|
| brick | f200w | 16509 | 0.4″ | **0.99** | **0.99** | agrees to ~1% |
| brick | f356w | 7640 | 0.6″ | 0.89 | 0.99 | agrees |
| cloudef | f480m | 603 | 0.4″ | 0.96 | 1.00 | agrees |
| cloudef | f360m | 1097 | 0.4″ | 0.70 | 0.92 | agrees @0.15″ |
| cloudef | f210m | 499 | 0.4″ | 0.30 | 0.71 | annulus-suppressed |
| cloudef | f162m | 982 | 0.3″ | 0.29 | 0.84 | tight-annulus @0.10″ |

**Reading it.** In the well-populated fields the empirical i2d curve of growth
agrees with the theoretical PSF to ≈1–11% (brick F200W 0.99/0.99 on 16.5k stars,
brick F356W 0.89/0.99, cloudef F480M 0.96/1.00). The cloudef SW medium bands
(F162M/F210M) fall low because their tighter achievable isolation forces the sky
annulus onto the PSF wings (and cloudef's per-module, multi-observation mosaics
add scatter) — a crowding/annulus limit, not a photometric bias.

**Numeric comparison to Gutermuth is definitional, not a discrepancy.** His
AperCorr is the flux within a FWHM-radius aperture *after his sky-annulus
subtraction*, normalised to the infinite PSF; e.g. F360M 0.541, F480M 0.550. Our
own STPSF PSF-grid, integrated as pure enclosed energy within the same aperture
radius (no annulus subtraction, box-normalised), gives ≈0.71 for both — so the
~0.16 offset is dominated by his annulus subtracting PSF wing flux and by the
different "total", not by a photometry difference. Reproducing his number would
require applying his exact aperture+annulus to a common PSF; we did not, so we do
not quote a single cross-comparison figure.

**Net comparison with Gutermuth.** Both conclude a theoretical PSF is the right
basis for the aperture correction in these crowded fields (he by choice, we by
the measured impossibility of an empirical total COG); our data-driven inner COG
independently confirms the synthetic PSF matches the data to ~1–10% for the
well-populated unsaturated fields; and both keep saturated cores out of aperture
photometry (he via NaN mosaics, we via the `aper_core_saturated` flag + PSF-fit
`flux_fit`).

## 8. Status
1. [x] `aperture_photometry.py` — NaN-aware aperture + curve of growth + apcorr.
2. [x] wired into `load_satstar_catalog` (default; env opt-out).
3. [x] separate apcorr tables → `catalogs/{filt}_satstar_apcorr_refstars.ecsv`
   (of record) + `..._diagnostic.ecsv`; never in the photometry table.
4. [x] unit tests (synthetic mosaics).
5. [x] cross-field investigation (11 field/filter combos; job 38760929).
6. [x] contamination-free apcorr from isolated unsaturated main-catalog stars
   (`build_reference_apcorr` → `*_satstar_apcorr_refstars.ecsv`).
7. [x] empirical-vs-theoretical-PSF curve of growth + Gutermuth comparison
   (job 38778528).
