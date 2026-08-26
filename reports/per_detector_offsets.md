# Are the per-detector offsets real? A calibration report

**Answer: no, not as a static term — so per-detector corrections must not be applied.**

Regenerate with:

```bash
python scripts/analysis/per_detector_offset_report.py --exclude-field gc2211
```

Last run 2026-08-07 · 34,672 per-detector measurements · 7,358 module-groups · 11 fields

![per-detector offsets](figures/per_detector_offsets.png)

## The question

The m2 retie loop measures each exposure's offset from its visit consensus **per detector**, but the offsets table holds one row **per module**, so the per-detector component is pooled away before anything is written. Issues #340/#342 proposed extending the table to per-detector rows rather than discarding it.

That is only justified if the per-detector term is a real, static, uncorrected distortion. The detectors are mechanically locked and are not moving at milliarcsecond level years into the mission, so a term that varies from one observation to the next is noise, and applying it would inject that noise into the astrometry.

## Method

For every exposure entry in every `checkpoint_m2_*.json`:

1. take the measured offset from the visit consensus (on-sky mas);
2. within one `(field, filter, visit, exposure, vgroup, module)` group — the detectors of one module in one exposure — subtract the group **mean**. That removes the common-mode per-exposure term (guide-star jitter and the like, which the module-locked table *can* express) and leaves only detector-to-detector structure;
3. aggregate per detector per field, sigma-clipped — the bulk-repair epochs put arcsecond outliers in the same array as the mas-scale term being tested;
4. compare each detector's **between-field scatter** against its **mean over fields**.

The mean is the estimator, not the median. With n=4 detectors `np.median` averages the middle two and discards the extremes: roughly twice the variance of the mean, and structurally unable to respond to one detector being genuinely offset.

## Result

| detector | fields | mean Δδ (mas) | between-field sd (mas) | verdict |
|---|---:|---:|---:|---|
| nrca1 | 9 | −0.187 | 0.267 | not static |
| nrca2 | 9 | −0.054 | 0.352 | not static |
| nrca3 | 9 | +0.110 | 0.396 | not static |
| nrca4 | 9 | +0.139 | 0.232 | not static |
| nrcb1 | 10 | +0.119 | 0.352 | not static |
| nrcb2 | 10 | −0.060 | 0.429 | not static |
| nrcb3 | 10 | +0.127 | 0.362 | not static |
| nrcb4 | 10 | −0.219 | 0.381 | not static |

**For all eight detectors the between-field scatter exceeds the mean.** Pooling all fields together, every detector is consistent with zero at ≤0.9σ.

The left panel is the whole argument: a static distortion term would put each detector's points on one horizontal line. Instead every detector's value **changes sign between fields** — e.g. `nrca2` reads +0.28 (sgrb2), −0.29 (sgrc), −0.60 (w51), +0.36 (gc2211).

## The roll degeneracy, and the second reading that closes it

The table above is measured on-sky, and on-sky sign reversal has an innocent explanation that the table cannot rule out: **a detector's placement error is fixed in the instrument frame, and this archive's fields do not share a roll.** From the `PA_V3` of one `_crf` per band:

```
arches 88.2   sgrb2 88.7-90.0   gc2211_o049 88.1   sgrc 91.4   brick 89.1 AND 275.5
w51 105.0-247.1   wd2 140.8   m92 170.0   ngc6397 267.3   cloudef 272.2
gc2211_o046 272.7   gc2211_o050 279.6   wd1 284.8   m4 290.1
```

Two clusters about 180° apart. A term that is *perfectly* static in the instrument frame **must** reverse sign on sky between a PA≈89 field and a PA≈275 one — so "it changes sign between fields" is also what a real per-detector term looks like in this archive, and the on-sky test alone cannot separate the two.

The report therefore runs the same estimator a second time on deviations de-rotated by each band's own `PA_V3`, and a third time with the angles **shuffled between bands** as the control. The control is needed because de-rotating by ~90° exchanges the two axes, so it averages a noisy axis into a quiet one and shrinks the scatter of both whatever angle was used.

Run 2026-08-25 over 73,673 measurements / 17,108 module-groups / 139 (field, band) angles:

| detector | sd on sky | sd de-rotated | sd shuffled (control) | verdict de-rotated |
|---|---:|---:|---:|---|
| nrca1 | 0.255 | 0.235 | 0.185 | not static |
| nrca2 | 0.264 | 0.190 | 0.208 | not static |
| nrca3 | 0.337 | 0.183 | 0.189 | not static |
| nrca4 | 0.264 | 0.166 | 0.159 | not static |
| nrcb1 | 0.351 | 0.209 | 0.232 | not static |
| nrcb2 | 0.629 | 0.205 | 0.287 | not static |
| nrcb3 | 0.371 | 0.157 | 0.175 | not static |
| nrcb4 | 0.441 | 0.335 | 0.493 | not static |
| **summed** | **2.912** | **1.679** | **1.928** | |

De-rotating does shrink the between-field scatter — and **the shuffled control reproduces almost all of that shrinkage** (1.93 against 1.68 for the true angles), which is the signature of axis mixing rather than of a recovered term. No detector reads STATIC in the instrument frame. **The "not static" verdict survives the one objection the on-sky test could not answer.**

Pinned by `jwst_gc_pipeline/photometry/tests/test_per_detector_report_roll_frame.py`, which synthesises a genuinely static instrument-frame term at two rolls 180° apart and requires the on-sky reading to miss it and the de-rotated reading to find it.

## Why the per-field significance is misleading

Within a single field these deviations are *highly* significant — up to 26σ (sickle nrcb3, +1.05 ± 0.04 mas). That is what makes the retie loop measure the same value every iteration, and it is what #342 detected when it reported "14 of 30 cells significant at >3σ".

Both are true and neither implies a distortion term. A per-field offset that reproduces within that field but reverses sign in the next field is a property of **that observation** — dither pattern, roll, source density, the consensus geometry — not of the detector. Correcting it would bake one observation's realisation into a table that the next observation reads back as an error of the same size and the opposite sign.

## Consequence for the offsets table

Keep the table module-keyed, and keep pooling. The per-detector term is:

- **not static** (this report),
- **sub-mas**: field means run −0.22 to +0.14 mas, consistent with #285's field-wide 0.53 mas (SW) / 0.19 mas (LW),
- **a small part of what the loop is chasing**: decomposing each correction into module-mean + per-detector residual, the module mean — which the table *can* express — is 70–95% of the total.

| field / filter | median \|total\| | module mean | per-detector | expressible |
|---|---:|---:|---:|---:|
| sgrc F115W | 1.87 | 1.77 | 0.90 | 95% |
| sgrc F162M | 2.27 | 1.59 | 2.16 | 70% |
| cloudef F210M | 1.85 | 1.46 | 0.80 | 79% |
| w51 F187N | 3.68 | 3.46 | 1.70 | 94% |
| sgrb2 F150W | 2.03 | 1.79 | 0.92 | 88% |
| brick F212N | 0.90 | 0.63 | 0.65 | 70% |

## One change that is worth making: pool by mean, not median

> **Landed (#381).** Pooling uses the mean now, and `prov_source` records
> `[mean of N, max_pair_sep <s>mas: ...]`. Everything below is the measurement that
> motivated the change, and describes the state before it.

`prov_source` records `m2 visit-consensus [median of N]`. With n=4 the median averages the middle two detectors and discards the outer two. Measured difference between the two estimators:

| field / filter | groups | median \|corr\| | mean \|corr\| | median \|difference\| | max |
|---|---:|---:|---:|---:|---:|
| sgrc F162M | 180 | 1.40 | 1.59 | 0.29 | 2.72 |
| sgrc F115W | 408 | 1.63 | 1.77 | 0.20 | 7.89 |
| cloudef F210M | 400 | 1.48 | 1.46 | 0.12 | 2.18 |
| w51 F187N | 160 | 3.55 | 3.46 | 0.38 | 1.15 |
| sgrb2 F150W | 192 | 1.85 | 1.79 | 0.23 | 0.78 |

Typically 0.1–0.4 mas, with a 7.9 mas tail. Small, but it is free accuracy and the tail is not. A clipped mean keeps the outlier rejection the median was chosen for without throwing away half the sample.

## What this report does not settle

Whether the retie loop terminates. That is a separate question and the evidence says it is **not** a granularity problem — corrections are landing (sgrc's table carries provenance dated through 2026-08-07) and the expressible fraction is 70–95%. Tracking sgrc F162M's module-level residual across iterations shows it converging where corrections were applied (`nrca/exp6`: 3.56 → 3.45 → 3.65 → **0.20** mas) and being perturbed elsewhere (`nrca/exp4`: 0.07 → 0.89), which is the coupling expected when the consensus is rebuilt from the same exposures being corrected. That belongs with #272.
