# MIRI satstar / cataloging review — 2026-07-04

Findings from user (AG) visual inspection of the 8-field gate rollout
(re-cataloged against `satstar-bright-phantom-gate` = current `main` + PR #36).

**Scope note.** None of these are regressions from the bright-phantom gate
(PR #36) — the gate only *removes* super-bright phantoms on saturated emission
and left every real star verified in place. The problems below are **pre-existing
MIRI satstar / detection / flux behaviour** surfaced by careful inspection of the
freshly-regenerated products. They are documented here for follow-up (likely
separate PRs), not fixed in #36.

Each item: **observation → cutout → mechanism (confirmed / hypothesis) →
proposed solution(s) → risk of the solution**.

---

## 1. brick F2550W — negative cores at low-coverage / mosaic seams
![brick](01_brick_f2550w_negatives.png)
Panels: data | model | residual | WHT (exposure-coverage).

**Observation (AG).** Four bright sources along a ~vertical line are negative:
`17:46:08.53 -28:42:36.15`, `17:46:06.16 -28:42:21.1`,
`17:46:06.82 -28:42:34.2`, `17:46:12.28 -28:42:48.2`.

**Confirmed.** These are bright (mostly saturated) stars sitting where the
exposure coverage (WHT) collapses. Measured WHT vs field median (479):
`08.53` → **14 (3%)**, `06.16` → 256 (53%), `06.82` → 126 (26%),
`12.28` → 814 (170%, but on a sharp vertical WHT *seam*). The WHT panel shows a
round coverage **hole** at `08.53` (saturation drops those pixels from most
groups) and **vertical seams** (sharp left/right coverage steps) at `06.16` /
`12.28`. At low/edge coverage the drizzled data + noise are unreliable, so the
PSF/satstar amplitude over-predicts → deep negative core + over-sub ring
(`08.53` residual core −424). AG's "vertical line / n_exposures" read is correct.

**Proposed solutions.**
- (A) **Coverage-gated satstar subtraction**: where local WHT < f·median
  (e.g. f=0.2), do not fit/subtract a satstar (or subtract but flag+NaN the
  residual there). Cheap, targeted.
- (B) **Weight the fit by WHT** (extend the existing inverse-variance weighting
  to fold in coverage) so seam pixels contribute less to the amplitude.
- (C) Cosmetic: NaN-mask sub-threshold-WHT residual pixels so the display isn't
  polluted (does not fix the catalog flux).

**Risk.** (A) can *leave a real bright star unsubtracted* in a genuine low-cov
hole → a bright positive residual there instead (trade a negative for a
positive; arguably better). (B) is low-risk but won't fully fix a 3%-coverage
core. (C) hides information; the flux is still biased.

---

## 2. cloudc F770W — every saturated core UNDER-subtracted
![cloudc770](02_cloudc_f770w_undersubtraction.png)
4 brightest satstars; data | model | residual.

**Confirmed.** Non-saturated stars subtract cleanly, but **every star with a
saturated core** leaves a strong **positive** residual ring around a NaN centre
(resid max 5322 / 3718 / 3147 / 2550, all-positive — no over-sub). The model
core is visibly dimmer/smaller than the data → the satstar **amplitude is fit
too low**.

**Mechanism (hypotheses, ranked).**
1. **`--deblend-satstars` flux splitting** (this field runs it; sgrb2 does not):
   splitting a bright star's merged DQ blob into sub-components divides the flux
   among them → each model under-predicts → composite under-subtracts. Strongest
   suspect given it's the config difference vs the over-subtracting fields.
2. **STPSF (fovp101) core sharper than the undersampled MIRI data**: with the
   clipped core masked, the wing-driven amplitude is a compromise that
   under-fills the bright inner wings.
3. A flux cap / oversub-clamp mis-firing low (less likely — clamps target
   *over*-subtraction).

**Proposed solutions.**
- Confirm (1) by re-running this field **without** `--deblend-satstars` on a few
  bright stars and comparing the residual ring.
- If deblend is the cause: only deblend when genuinely blended (raise the
  deblend trigger), or re-normalise the summed deblended model to the parent
  blob's wing flux.
- If PSF-mismatch (2): fit amplitude on a **wing annulus** (robust wing
  photometry) rather than full-PSF LSQ, or use a broader/effective PSF.

**Risk.** Disabling deblend re-merges genuinely-blended saturated pairs (the
exact problem `--deblend-satstars` was added for on this field) → 2× over-sub
for real close pairs. Wing-annulus amplitude is noisier and emission-contaminated
in this field. Any amplitude bump risks flipping under-sub → over-sub pits.

---

## 3. cloudc F2550W — faint stars MASSIVELY OVER-subtracted (pockmarks)
![cloudc2550](03_cloudc_f2550w_oversubtraction.png)
data−bg | model | residual−bg (blue = over-sub pit). 7 AG-flagged stars.

> **Correction (AG, 2026-07-04).** An earlier draft of this section called these
> "fake stars on emission" / "under-subtracted." Both wrong. The residual reads
> below the ~1000 F2550W thermal-background **pedestal**; measured against zero it
> looked positive, but measured against the pedestal every one is a **deep
> negative pit = massive over-subtraction.** Corrected below.

**Confirmed.** Real bright stars subtract fine, but faint/moderate stars are
grossly **over-subtracted**. Measured against the local background (~1000):

| star | star flux (data−bg) | model peak | pit depth (resid−bg) | catalog flux |
|---|---|---|---|---|
| `17:46:13.71 -28:34:27.1` | **6** | 422 | **−420** | **37142** |
| `17:46:16.735 -28:35:17`  | 23 | 238 | −223 | 9781 |
| `17:46:16.16 -28:35:13.2` | 19 | 147 | −128 | 1498 |
| `17:46:16.64 -28:35:28.8` | 8 | 174 | −167 | 1248 |
| `17:46:13.58 -28:34:16.6` | 24 | 134 | −111 | 2104 |
| `17:46:13.224 -28:34:22.1`| 40 | 123 | −83  | 3747 |

The **catalog fluxes are inflated 5–70×** (37142 for a 6-count star), so the
rendered PSF hugely over-predicts and gouges a pit 80–420 below background. 6 of
7 are **non-saturated daophot** sources (not satstars), so this is a **daophot
flux/background failure at F2550W**, not a satstar-model issue. The fluxes are
also *erratic* (1248→37142 for near-identical ~1000-count peaks) → the fit is
ill-conditioned. Boxy render edges are also visible (secondary).

**Mechanism (hypotheses, to confirm).**
1. **Background pedestal absorbed into the flux.** F2550W has a huge (~1000
   MJy/sr) thermal background. If the local-background estimate under a faint
   source is too low, the PSF fit assigns the pedestal to the star → 10–70×
   flux over-estimate → over-subtraction pit. (Check `local_bkg` in the catalog
   for these rows; the erratic fluxes point here.)
2. **Ill-conditioned broad-PSF fit.** The F2550W PSF is very broad (~7–8 px
   FWHM); fitting it to a faint compact bump on structured background is
   degenerate → runaway amplitude.

**Proposed solutions.**
- Confirm (1): dump `local_bkg` + refit one star with a forced correct local
  background; if the pit vanishes, it's the background estimate.
- Fix the F2550W local-background estimation (larger/robust annulus, or subtract
  a proper coarse background before the fit — the `coarse_bg`/smoothed-bg paths
  already exist and are off here).
- Guard: reject / cap fits whose flux implies a model peak ≫ the local
  data-minus-background excess (an over-prediction gate analogous to the
  bright-phantom gate but for the faint over-sub regime).

**Risk.** A background-annulus change that works here can bias fluxes in the
crowded short-λ fields (F770W) where the annulus hits neighbours. An
over-prediction cap risks clipping genuine bright stars on faint background.
Must be filter-scoped and vetted per field. Ties to the F2550W model/residual
inconsistency + negative-pit tasks already open.

---

## 4. sgrb2 F1280W — three distinct problems
![sgrb2](05_sgrb2_f1280w_issues.png)
Rows: (a) sat→hole | (b),(c) pockmarks | (d) emission fake.

**(a) `17:47:18.15 -28:23:16.1` — bright sat star → NaN hole, "no star".**
Confirmed: model peak **7501 > data peak 3805** (data is saturation-clipped).
A full-amplitude PSF minus a clipped core goes deeply negative → NaN-masked →
looks like a masked hole with no fitted star (the star *is* fit and cataloged;
the residual just can't show it). This is the **clipped-core over-subtraction**
pit — the OVER-sub counterpart to cloudc F770W's UNDER-sub, same root
(amplitude vs clipped core, no deblend here).

**(b) `17:47:19.75 -28:23:49.6`, (c) `17:47:18.39 -28:23:57.0` — pockmarks.**
Faint sources, minor mis-subtraction (model 72/32 vs data 145/110); small
residual dimples. Low harm.

**(d) `17:47:17.805 -28:24:04.5` — fake on emission.** A bright emission ridge
(no compact source) is cataloged and PSF-subtracted → over-sub pit. (A genuine
emission-detection fake here, distinct from the cloudc F2550W flux-over-estimate
over-sub.)

**Proposed solutions.**
- (a) For clipped-core over-sub: **cap the subtracted model peak to the data's
  charge-migration wing level** (subtract only out to where model ≈ data), or
  NaN the clipped-core region consistently so it's a clean mask, not a
  variable-depth pit. Ties to the existing negative-pit / render tasks.
- (d) Filter-scoped detection stringency + shape/concentration vetting to reject
  emission ridges (broader than the PSF).

**Risk.** Capping the model peak *under*-subtracts the true wings of a genuinely
bright star (flips a hole into a positive halo). Detection stringency risks real
faint stars (as in §2/§3).

---

## Cross-cutting summary

| theme | fields | root | key risk of fixing |
|---|---|---|---|
| **faint-star flux over-estimate → deep over-sub pits** | **cloudc F2550W** | daophot flux inflated 5–70× (bg pedestal absorbed / ill-conditioned broad-PSF fit) | bg-annulus change biases crowded short-λ fields; over-prediction cap clips real bright stars |
| clipped-core over/under-sub | cloudc F770W (under), sgrb2 F1280W-a (over), brick F2550W | PSF amplitude vs saturation-clipped/NaN core; deblend splits flux | any amplitude change flips over↔under; deblend removal re-merges real pairs |
| emission-detection fakes | sgrb2 F1280W-d | long-λ broad PSF + bright emission; daofind + vetting too permissive | stringency kills real faint stars (cloudc F770W recovery) |
| low-coverage / mosaic seams | brick F2550W | few exposures → unreliable amplitude | coverage-gating leaves real stars unsubtracted |
| boxy render footprint | cloudc F2550W, others | model rendered only over fit cutout | compute cost of larger footprint |

**Recommendation.** These are filter- and field-dependent and the fixes are
mutually tensioned (stringency vs completeness, amplitude over vs under). Treat
as **separate follow-up PRs**, each with a by-eye truth region per field, rather
than bundling into the phantom-gate PR. Highest-value, lowest-risk first:
coverage-gating (brick), then confirm/repair the deblend flux-split (cloudc
F770W), then filter-scoped emission-fake vetting (F1280W/F2550W).
