# Empirical PSF (ePSF) maps across the NIRCam detectors — program 10678

Position-dependent, 4x-oversampled effective PSFs for F480M (NRCALONG, NRCBLONG)
and F212N (NRCA1-4, NRCB1-4) built from bright, unsaturated, isolated field stars in
the public `_cal` frames of JWST program 10678 (Galactic Center), compared with STPSF.
Motivation: a position-dependent PSF model (core, wings, spikes) good enough to model
the halos and spikes of super-saturated stars (issue #993).

## Pipeline

| step | script | what |
|---|---|---|
| 0 | `s3io.py` | list/fetch products, CRDS references and WSS OPDs from the MAST AWS mirror (`stpubdata`) |
| 1 | `run_extract.py` → `extract_stars.py` | stream every `_cal` frame (download → measure → delete), select stars, cut stamps |
| 2 | `stpsf_models.py` | STPSF on a 5x5 grid of detector positions per detector |
| 3 | `analyze_detector.py` (+ `epsf.py`, `catalog.py`) | wing + core ePSF, spatial grid with cross-validation, shape maps, per-star table |
| 4 | `figures.py` | figures and a JSON summary (one detector) |
| 5 | `sw_summary.py` | combined figures + JSON for several detectors (the 8 SW detectors, focal-plane layout) |
| 6 | `export_epsf_core.py` | write the full-sample spatial core as a pipeline core file (published in the separate `JWST-GC/epsfs` repository) for the hybrid PSF (`jwst_gc_pipeline.photometry.epsf_hybrid`, `PSF_EPSF_CORE_DIR`) |
| 7 | `fullfield_residual.py` → `fullfield_paired.py`, `fullfield_figures.py`, `fullfield_report_figs.py` | full-frame star subtraction with STPSF / held-out ePSF / hybrid on one frame (issue #1007) |
| 8 | `validate_hybrid_grid.py` | the pipeline's hybrid grid vs its STPSF grid with the production fitter (`PSFPhotometry`, `fit_shape=(5, 5)`) on held-out stars |

```
S=$SCRATCH/epsfmap
OMP_NUM_THREADS=1 python run_extract.py --detector nrcblong --scratch $S --workers 3
OMP_NUM_THREADS=1 python stpsf_models.py --detector NRCB5 --filter F480M --fov 65 --out $S/stpsf_NRCB5_F480M.npz
OMP_NUM_THREADS=1 python analyze_detector.py --detector nrcblong --scratch $S --stpsf $S/stpsf_NRCB5_F480M.npz --out $S/result_nrcblong.npz
python figures.py --result $S/result_nrcblong.npz --stars $S/result_nrcblong_stars.npz --label "NRCBLONG F480M" --prefix out/epsf
# SW (F212N): run_extract.py --every 3 (every 3rd observation), then per detector as above, then
python sw_summary.py --scratch $S --dets nrca1,nrca2,nrca3,nrca4,nrcb1,nrcb2,nrcb3,nrcb4 --prefix out/sw
```

`run_extract.py` needs two STPSF detector-frame PSFs in `$S` as neighbour templates
(`stpsf_test_F480M_NRCB5.fits`, `stpsf_test_F212N_NRCB1.fits`: `calc_psf(fov_pixels=61/81, oversample=4)`
at the detector centre).

## Star selection (`extract_stars.py`)

* **Not saturated in any group.** The cal DQ flags SATURATED only when a pixel lost
  (most of) its ramp; a star that saturates in groups 3–4 of the 4-group BRIGHT2 ramp is
  unflagged but was fitted on fewer groups.  So the peak pixel's accumulated signal at
  the last group, `SCI / PHOTMJSR × TGROUP × NGROUPS` [DN], must be below 50% of that
  pixel's CRDS saturation level (`R_SATURA`, fetched from the mirror; the level includes
  the bias, so 50% leaves a wide margin).  No SATURATED pixel anywhere within the core
  stamp + 3 px; no DO_NOT_USE/NaN within 3 px of the peak (others are masked).
* **Isolation / neighbours.** Every 3x3 local max > 5σ is a potential neighbour; it is a
  *real* neighbour if it exceeds 3× what the candidate's own PSF (2-D STPSF template,
  spikes included) predicts at that offset (so the candidate's own rings/spikes are not
  "neighbours").  Each real neighbour's light is predicted with the same template;
  stamp pixels where one neighbour exceeds 3σ are masked; the candidate is rejected if
  a neighbour adds >2% within 2 px or >10% within 3 px of the peak, if any pixel within
  3 px is masked, or if > 50% of the stamp is.  Saturated stars are masked with a disk
  of radius 6 + 2√(saturated area) px (wing stamps only; core stamps may not contain one).
* **Core stamps**: ±12 px, peak S/N > 40.  **Wing stamps**: ±30 px (LW) / ±40 px (SW),
  peak S/N > 150, up to 80 per frame, masked-fraction ≤ 70%.
* Sky positions from the **GWCS** (`datamodels.open(fn).meta.wcs`, ASTROMETRY RULE #2),
  with the local 2x2 Jacobian so the refined centroid is converted without the frame.

The field is confusion-limited: the per-pixel ERR is not the error budget beyond a few
px.  The per-star "confusion noise" is the robust rms of the stamp beyond 8 px minus the
ERR part; fits weight pixels by 1/(ERR² + conf² + (0.02·data)²).

## ePSF (`epsf.py`)

Anderson & King style.  The ePSF is the pixel-integrated PSF sampled at 4 nodes per
native pixel; a pixel whose centre is offset u from the star has model value f·P(u)+b.
Iterate: (1) fit each star (Gauss–Newton on (f, x, y, b) within 8 px, then linear
(f, b) over the stamp; clipped); (2) stack normalised residuals (d−b)/f − P(u) on the
nearest node, equal weight per sample, 3σ clip per node; (3) P += mean residual;
re-centre (centroid within 1.5 px at the origin) and re-normalise (flux within r=10 px
= 1).  Start: a Gaussian (not STPSF).

**Degeneracy.**  With a free background per star, P and P + c fit every star equally
well.  It is fixed by pinning the mean of the outermost 1.5 px annulus: the wing ePSF
(±30/40 px) to STPSF's value there (≈1e-5 of the r<10 flux per px — the only STPSF input
to the empirical model), the core ePSF (±12 px) to the wing ePSF's value at 10.5–12 px.

**Injection-recovery** (synthetic stars drawn from an STPSF ePSF broadened by a 0.3-px
Gaussian, realistic S/N, 6000 stars): FWHM recovered to +0.4% (STPSF start) / +0.7%
(Gaussian start) after 8 iterations, peak to −0.6%/−0.3%, max node error 1.7% of the
peak (at r≈1 px).  Differences below these levels are not significant.

**Spatial model.**  G×G cells, each ePSF built from its own stars starting from the
global one; evaluated piecewise-constant or bilinear between cell centres.  G is chosen
by cross-validation: 20% of the frames held out, median held-out reduced χ².

**Shape maps.**  Each star is refitted with the global ePSF plus three linear modes —
size (−(2P + u·∇P): a fractional FWHM change), and the two shears e1, e2 — and a
wing-excess ratio (3–10 px flux over the model's, same pixels).  Binned in 128-px cells;
noise from split halves (even vs odd frames); small-scale structure = map minus a
quadratic surface, and its half-vs-half correlation says whether it is real.

## STPSF (`stpsf_models.py`)

stpsf 2.2.0 with OPD `R2026091802-NRCA1_FP6-1.fits` (sensed 2026-09-18 05:22 UTC; the
program ran 2026-09-15..19), read from `s3://stpubdata/jwst/public/R2026091802/` because
MAST is not reachable.  `use_exact_wss_target_phase=False` (the FP6 target-phase map is
not in the available data files; the ISIM CV3 Zernike SI model is used instead).
`OVERDIST` (distortion + charge diffusion + IPC; its 4x4 block sum equals `DET_DIST`) is
convolved with the native-pixel box (exact, Fourier) to become an ePSF on the same grid.

**stpsf data files**: the official tarball hosts (stsci.box.com, data.science.stsci.edu)
are blocked here.  The files used (`jwst_pupil_RevW_npix1024.fits.gz`,
`JWST_OTE_OPD_cycle1_example_2022-07-30.fits`, `si_zernikes_isim_cv3.fits`,
`JWpupil_segments_RevW_npix1024.fits.gz`, `NIRCam/filters.tsv`, `NIRCam/filters/F212N|F480M_throughput.fits`,
`NIRCam/IPC/KERNEL_{IPC,PPC}_CUBE.fits`, `NIRCam/OPD/field_dep_table_nircam.fits`) came
from a public GitHub copy of the webbpsf-data **1.4.0** release; `version.txt` was set
to 2.1.0 so stpsf 2.2 would load them.  Filter curves / pupil / field-dependence tables
may differ slightly from the stpsf-data 2.x release.

## Outputs

* `<scratch>/stamps/<det>/<root>_stars.npz` — per frame: star table + stamps.
* `result_<det>.npz` — ePSFs (global core `P1`, wing `Pw`, grids `PG`/`P4`, split halves),
  STPSF ePSFs, CV metrics, per-frame wing excess.
* `result_<det>_stars.npz` — **per-star-exposure table** (columns in the issue / docstring
  of `analyze_detector.py`): `fname, detector, filter, expstart, obs, expnum, x, y`
  (0-based detector px, ePSF-fit centroid), `flux` (MJy/sr·px within r=10 px),
  `flux_dn_s`, `bkg` (MJy/sr), `ra, dec` (deg, GWCS), `peak_snr, satfrac, maskfrac, chi2, ok`,
  `dsize_*` (fractional size vs global/grid ePSF), `de1_*, de2_*`, `dwing_*`, `conf`, `frame`.
