# ZEROFRAME saturated-star deblend: reliability gate and independent validation

**Date:** 2026-10-15. **Branch:** `deblend-gate` (PR #1183, draft).
**Scope:** JWST GC Treasury 10678, NIRCam F212N and F480M, `--deblend-satstars`
(ZEROFRAME-based splitting of merged DQ-SATURATED blobs into one seed per
star).

## 1. The problem

Scanning every DQ-SATURATED blob in all 4040 10678 crf frames at the seed
stage (data-qa, no PSF fit), ~65% of F480M blobs and ~31% of F212N blobs
split into >=2 proposed centres, but only ~0.5-1.3% of those centres are
confirmed by an independent signal (a second group-0 GROUPDQ-SATURATED core,
or a second VIRAC2 source, inside the blob). A synthetic single-star test
(an stpsf PSF with no second star present, run through the same deblender)
reproduces the dominant ~8-10 px secondary separation seen in the real data
once the saturated area is large enough — that separation is the PSF's own
diffraction-spike/sidelobe structure, not a second star. PSF fitting removes
most of this before a split reaches the shipped catalog (production m12
catalogs keep ~2 (F212N) / ~16 (F480M) DQ-SAT splits per frame, versus ~23 /
~322 raw candidates per frame at the seed stage), but a reliability gate was
proposed to filter the remainder.

## 2. The first gate, and why its validation was circular

The first version of this gate accepted a proposed >=2-centre split only if
the blob's group-0-saturated pixels formed >=2 connected clusters (bulk
count, no defect filtering). It was validated against a ground truth defined
as "g0-core count >= 2 OR VIRAC2 count >= 2" — the same g0-core signal the
gate itself used. Comparing the gate's criterion to a ground truth containing
that criterion is circular; the resulting "100% precision" claim measured
nothing beyond self-consistency. That claim has been removed from the module
docstring and the PR body, and this report replaces it with an independent
measurement (Section 4), a check of that measurement's own confound and a
needed-fraction check (Sections 5-6), and new shipped-catalog evidence
(Section 8).

## 3. Three code fixes independent of the validation question

Three defects were fixed regardless of what the independent measurement
found, because each is a bug in the gate's own logic:

**Defect filtering.** A raw group-0-saturated cluster can be a recurring
detector hot/fixed-pattern pixel rather than a real second star's core.
data-qa's component-level scan (g0_clean.py, a data-qa-repo script, not
tracked here) found 97% of raw F212N
g0>=2 components are exactly this (4989 raw components -> 165 after
filtering a (detector, 4-px bin) recurring in >=5 of 4040 frames).
`group0_core_positions` (in `satstar_deblend_gate.py`) now drops any cluster
whose centroid falls in such a bin, using a static per-detector registry
(`jwst_gc_pipeline/reduction/data/satstar_g0_defect_registry.json`, built by
`scripts/reduction/build_satstar_g0_defect_registry.py` from the same scan,
same bin size and recurrence threshold data-qa validated, but at per-CORE
rather than per-component granularity — flagged bins per detector: nrca1
3187, nrca2 2407, nrca3 1993, nrca4 2085, nrcalong 8816, nrcb1 1521, nrcb2
1457, nrcb3 1539, nrcb4 1551, nrcblong 16323). The registry loads lazily and
fails OPEN (no filtering) if the file is missing, so a missing registry never
itself suppresses a real split.

Measured directly on this session's own reprocessed sample (Section 4's 90
frames / 774 matched groups): comparing the RAW group-0-core count (no
filtering) against the new defect-filtered count, for groups that reached
the shipped catalog as a proposed >=2-centre split candidate —

| filter | raw-count passes (>=2) | defect-filtered passes (>=2) | passes that exist ONLY because of a defect cluster |
|---|---|---|---|
| F212N | 3 | 3 | 0 of 3 (sample too small to read a rate — see caveat below) |
| F480M | 162 | 123 | 39 of 162 (24.1%) |

F212N's own subsample is too small (3 raw-passing groups out of 98 sampled)
to state a shipped-catalog defect-pass rate with any precision; the reliable
number for F212N is data-qa's full-population, component-level scan (97%),
cited above, which this fix directly addresses. The F480M number comes from
a larger sample (162 raw-passing groups) and shows the defect filter is not
an F212N-only concern: it removes roughly a quarter of the raw passes that
would otherwise have reached the catalog on a hot-pixel coincidence alone.

**Per-centre matching, not a bulk threshold.** `gate_split_centers`'s
docstring claimed per-centre rejection, but the code only checked the bulk
count (`n_g0 >= 2`): a component proposed with 5 centres and only 2 confirmed
group-0 cores kept all 5. The gate now does a 1-to-1 nearest assignment
(`scipy.optimize.linear_sum_assignment`, cost = Euclidean pixel distance)
between proposed centres and detected cores, so one core cannot confirm two
centres, and only the centres with their own matched core (within
`max(1.5, 0.5*fwhm_pix)` px) survive; a 5-centre proposal with 2 confirmed
cores now keeps the 2 matched centres and drops the other 3. New tests cover
2 cores against 3-5 proposed centres, and a defect cluster standing in as the
second "core" (correctly excluded, collapsing the split).

**Fail-closed visibility.** `zeroframe_group0_saturated is None` (ZEROFRAME
fit off, a missing ramp, or a GROUPDQ read failure) makes every proposed
split on that frame collapse to the single blob centroid — correct
fail-closed behavior, but previously silent and indistinguishable from "the
gate ran and rejected". `build_deblended_source_records` now takes an
optional `gate_diagnostics` dict, updated with `n_multi_centre_proposed`,
`n_gate_unavailable`, `n_gate_rejected`, `n_gate_accepted`; `get_saturated_stars`
stamps these into the output table's FITS meta as `SATDBLG0`
(`off`/`n/a`/`nosignal`/`ok`) and `SATDBLGN`/`SATDBLGU`/`SATDBLGR`/`SATDBLGA`,
so a frame where the signal silently went missing stays greppable rather than
reading as a clean pass. In this session's own 90-frame reprocessed sample,
0 of 90 frames hit the signal-unavailable condition (every sampled frame had
a readable ramp and GROUPDQ) — consistent with it being an edge case rather
than a common occurrence, which is exactly why it needed an explicit counter
rather than reliance on noticing it by eye.

## 4. Independent re-measurement (this session): cross-dither and cross-filter

To avoid the circularity in Section 2, reliability of the SHIPPED survivors
was re-measured using two signals that share no code or criterion with the
gate:

- **Cross-dither repeatability**: a shipped split child's sky position
  reappears within 0.05" in a DIFFERENT exposure of the same tile (a real
  star should recur at the same place across overlapping dithers; a
  PSF-sidelobe artifact depends on that exposure's own saturation geometry
  and should not).
- **Cross-filter confirmation** (F480M only): an F480M split child's position
  matched within 0.1" against the independently-built F212N m12 catalog
  (`off_dbl2` tag) for the same tile, with a rotated-90-degree control (each
  split group's non-primary members reflected 90 degrees about the group's
  brightest member before matching) that bounds the false-positive rate from
  field crowding alone.

**Both signals share an unaddressed confound.** Within one visit the
position angle (PA) on sky is fixed and the dithers are small, so a PSF
diffraction-spike or strut-sidelobe artifact lands at the same offset from
its parent star in every dither of that visit, not only a real companion
does. data-qa's synthetic single-star test (no real companion by
construction) splits the same way for exactly this reason: the split is a
deterministic function of the PSF and the saturation geometry, not of
per-exposure noise, so it reproduces at matching precision across
independent realizations. Cross-dither repeat therefore cannot, by itself,
separate a real companion from a spike artifact. The rot90 control bounds
only coincidental field-crowding matches: rotating a child 90 degrees about
its primary moves it off the spike axis entirely, so a low rot90 rate
(0.2-0.7% here) does not bound how often an on-axis spike artifact would
pass the cross-dither test. The cross-filter check has an analogous gap: it
matches against the independently-built F212N `off_dbl2` catalog, but that
catalog is itself built with deblending on, at the same visit PA, so an
F480M artifact could be confirming against an F212N artifact rather than
against a real star. Sections 5 and 6 address this directly: Section 5
(Test A) checks whether confirmed children cluster on the PSF's own spike
directions and directly measures the single-star synthetic artifact's own
cross-dither pass rate under the production tolerance; Section 6 (Test B)
asks, independent of either confirmation signal, whether each confirmed
child is a source the deblend-off release catalog already contains.

Applied to the shipped m12 DQ-SAT split children of 19 production 10678
tiles (`m12_split_children_dqsat.ecsv`, 11115 rows / 5299 groups / 923
frames), re-opened and re-measured with the exact gate code this PR ships
(`satstar_deblend_gate.group0_core_positions`, defect-filtered). Reprocessing
every frame costs ~70-95s single-process; the full 923-frame population was
not tractable in this session, so a random, per-filter-stratified subsample
(45 frames/filter, fixed seed 20261015, 2 worker processes) was reprocessed:
90 frames, 779 groups attempted, 774 matched (5 `no_blob_match`, no other
failures). Each shipped child was bucketed by its own parent blob's
defect-filtered group-0-core count:

- **(a)** >=2 cores — this gate's accept criterion
- **(b)** <2 cores — this gate's reject criterion

| filter | bucket | n children | n groups | cross-dither confirmed | cross-filter confirmed (F480M only) | rot90 control |
|---|---|---|---|---|---|---|
| F212N | a | 6 | 3 | 0/6 (0%) | — | — |
| F212N | b | 193 | 95 | 34/193 (17.6%) | — | — |
| F480M | a | 259 | 123 | 82/259 (31.7%) | 256/259 (98.8%) | 1/136 (0.7%) |
| F480M | b | 1123 | 553 | 322/1123 (28.7%) | 1045/1123 (93.1%) | 1/570 (0.2%) |

**F480M is well-powered** (1382 bucketed children, a cross-filter check whose
rotated control confirms at 0.2-0.7% — essentially no false-positive floor)
and the result is unambiguous: bucket (a) and bucket (b) confirm as real at
nearly the same rate, 98.8% vs 93.1%. A second check at a looser accept
threshold (>=1 core instead of >=2, using the same reprocessed data, no new
reprocessing) gives 94.0% vs 94.5% — statistically indistinguishable. The
group-0-core count carries essentially no discriminating power between real
and fake F480M splits at either threshold. Per the decision rule this
measurement was designed against: bucket (b) confirms at 93%, far above the
~80% "the gate is too strict" line, not below it. Combining both buckets,
1314 F480M children are independently confirmed real (cross-dither OR
cross-filter); the gate (bucket b) would discard **1056 of them (80.4%)**.

**F212N is underpowered in this subsample.** Only 3 groups (6 children)
cleared the >=2-core bucket in the 45-frame sample — too few to read a
bucket-(a) confirmation rate at all (0/6 is as consistent with "none of
these particular 6 happened to repeat in this subsample" as with "none are
real"). Bucket (b) confirms at 17.6% by cross-dither alone (no cross-filter
check exists for F212N in this script, since F212N has no shorter-wavelength
reference catalog available here), which is markedly lower than F480M's
cross-dither rate (28-32%) — but cross-dither is a narrower test than
cross-filter (it requires the same saturation geometry to re-trigger the
same deblend split in a different dither, not just the star to be real), so
it is expected to undercount real F212N children, and 17.6% should not be
read as an estimate of the true real fraction. Combining both buckets by
position (cross-dither match only, since no cross-filter check exists for
F212N), 42 F212N children are independently confirmed real across the whole
subsample; the gate (bucket b) would discard 41 of them (97.6%) — directionally
consistent with F480M's finding, but resting on too few confirmed cases to
treat as a settled measurement on its own. Section 8 supplies a separate,
better-powered F212N check that does not depend on the group-0-core signal
at all.

**What the fraction of currently shipped real survivors the gate would remove
is, per filter:** F480M 80.4% (n=1314, well-powered); F212N 97.6% (n=42,
too few confirmed cases for a precise estimate, but the direction agrees).
Section 5 tests whether "confirmed real" itself is a reliable label, given
the fixed-PA confound stated above — it is not, which changes how much
weight this paragraph's numbers can carry.

## 5. Test A (spike-PA): does cross-dither/cross-filter confirmation distinguish real children from spike artifacts?

Two sub-tests, run on the same 774-group, 90-frame reprocessed subsample used
in Section 4.

**5.1 On-spike fraction.** For every bucketed child, the position angle (PA)
of the child relative to its group's primary (brightest child) was computed
in the DETECTOR PIXEL frame, reading the GWCS via `frame_wcs` per CLAUDE.md
astrometry rule #2 (0 of the 90 frames fell back to SIP). The JWST PSF's own
artifact-split directions were measured independently, by rerunning data-qa's
synthetic single-star test (synthetic_single_star.py, a data-qa-repo script
not tracked here; no real companion by construction, so any multi-centre
split is 100% a PSF/saturation artifact) and recording each synthetic
secondary's PA: 8 recurring
artifact-centroid directions emerged (not assumed from a theoretical
hex-mirror phase), each scored with a +/-7.5 deg on-spike window (matching
data-qa's own synthetic test and the coordinator's specified window). 8
separated 15-deg windows tile 33.3% of the circle regardless of where they
sit, so 33.3% is the uniform-PA (no preferred direction) expectation in the
pixel frame, independent of any particular frame's roll angle.

| filter | selection | n | on-spike fraction |
|---|---|---|---|
| F212N | all | 101 | 36.6% |
| F212N | confirmed real | 20 | 35.0% |
| F212N | unconfirmed | 81 | 37.0% |
| F212N | bucket a (>=2 cores) | 3 | 0.0% |
| F212N | bucket b (<2 cores) | 98 | 37.8% |
| F480M | all | 706 | 29.3% |
| F480M | confirmed real | 648 | 29.0% |
| F480M | unconfirmed | 58 | 32.8% |
| F480M | bucket a (>=2 cores) | 136 | 29.4% |
| F480M | bucket b (<2 cores) | 570 | 29.3% |
| synthetic (known artifact, self-check) | — | 739 | 99.5% (circular — the window is fit to this sample, not an independent check) |

Confirmed children do not sit on-spike above uniform (33.3%) or above
unconfirmed children, in either filter, in the raw pixel frame. Because
pooling raw pixel-frame PA across ~10 different NIRCam detectors with
different intrinsic rotations/parities relative to the telescope could mask
a real spike-direction signal, the same measurement was repeated in the
V2V3 (telescope-pupil-fixed) frame, using each frame's own
`detector -> v2v3` GWCS transform (a near-diagonal, parity-flipped, near-zero
rotation affine map, confirmed by direct finite-difference Jacobians on the
two detectors used in the synthetic test: `nrca1`
`[[-0.0311,-0.0003],[-0.0003,0.0313]]`, `nrcalong`
`[[-0.0628,-0.0001],[-0.0002,0.0631]]`). Pooled across all 10 detectors
present in the subsample (uniform expectation in this frame is 45.8%,
11 clusters at a window threshold chosen to match the pixel-frame analysis):
F212N confirmed 45.0% (n=20) vs unconfirmed 42.0% (n=81), bucket a 0.0%
(n=3) vs bucket b 43.9% (n=98); F480M confirmed 33.3% (n=648) vs unconfirmed
41.4% (n=58), bucket a 35.3% (n=136) vs bucket b 33.7% (n=570). A direct,
no-projection-needed cross-check restricted to the exact two detectors the
synthetic test used (`nrca1` for F212N, `nrcalong` for F480M) gives the same
qualitative answer: F212N/nrca1 29.4% (n=17, 0 confirmed in this small
same-detector subset); F480M/nrcalong confirmed 34.7% (n=300) vs unconfirmed
55.0% (n=20), against a 50.0% uniform expectation at this detector's cluster
geometry. A sweep of the cluster-window threshold (20/30/40, giving
7/6/5 clusters and uniform expectations of 29.2%/25.0%/20.8%) shows confirmed
children at a LOWER on-spike fraction than unconfirmed at every threshold
(e.g. threshold 40: confirmed 14.8% vs unconfirmed 20.1%, bucket a 12.2% vs
bucket b 16.5%) — the null result is not an artifact of one particular
window choice.

**Taken on its own, 5.1 does not support the confound**: if cross-dither
confirmation were simply re-confirming spike artifacts, confirmed children
should cluster on the spike directions more than unconfirmed children do, and
they do not, in either frame, at any window threshold tested.

**5.2 Direct cross-dither pass rate on pure artifacts.** 5.1 tests one
specific mechanism (recurrence confined to ~8 discrete spike angles). The
confound as stated does not require that mechanism — it only requires that a
single star's own deterministic PSF+saturation split geometry reproduce
across independent noise realizations at the production cross-dither
tolerance (0.05"), regardless of angle. That is a direct, assumption-free
measurement on the same synthetic single-star dataset: using the measured
plate scale (0.0311"/px F212N, 0.0630"/px F480M, from the Jacobians above),
0.05" converts to 1.61 px (F212N) / 0.79 px (F480M), and for each synthetic
secondary centre, any OTHER secondary centre from a DIFFERENT trial
(independent noise draw, standing in for a different dither) within that
tolerance counts as a "cross-dither pass." Result: **F212N 310/312 = 99.4%;
F480M 422/427 = 98.8%.** A population of 100% single-star PSF/saturation
artifacts, with no real companion anywhere in it, passes the exact
cross-dither criterion used to label shipped children "confirmed real" in
Section 4 essentially as often as a population of all-real children would.

**Conclusion.** 5.2 directly measures the confound the coordinator raised
and confirms it: the cross-dither test cannot discriminate real companions
from deterministic PSF/saturation-geometry artifacts, because such artifacts
reproduce within the production tolerance ~99% of the time regardless of
direction. 5.1's null result on spike-direction clustering does not
contradict this — it shows the artifact's recurrence is not confined to a
handful of discrete angles (so a PA-only check would not have caught it
either), which is a stronger problem for the cross-dither signal than the
coordinator's literal spike-angle framing, not a weaker one. Section 4's
"93-99% confirmed real" headline numbers and the F480M "80.4% of real
children discarded" removal-fraction are not reliable evidence that those
children are real: a 100%-fake population would score comparably under the
same test.

## 6. Test B (needed): do confirmed children already exist in the deblend-off release catalog?

For each of the 668 distinct confirmed children from Section 4 (deduplicated
across groups), the deblend-OFF v1.9 release joint-nircam catalog
(`/orange/adamginsburg/jwst/releases/v1.9-2026.10/gc-treasury/catalogs/gctreasury10678_joint_nircam.parquet`)
was read per tile/filter with a pyarrow box predicate on the per-filter
skycoord columns (prodDcheck/secondaries.py's method, a scratch script not
tracked here), then matched within 0.1". 0 of the 19 tiles' lookups failed.

| filter | n confirmed | already in v1.9 (<0.1") | NOT in v1.9 ("candidate needed") | dmag vs primary (med, p10-p90) | separation from primary (med, p10-p90) |
|---|---|---|---|---|---|
| F212N | 20 | 5 (25.0%) | 15 (75.0%) | 2.42 (0.83-5.25) | 0.39" (0.22"-0.63") |
| F480M | 648 | 265 (40.9%) | 383 (59.1%) | 2.05 (0.59-3.86) | 0.60" (0.39"-1.10") |

(Separations converted from the pixel values recorded in Section 5's PA
measurement using the same plate scales, 0.0311"/px F212N and 0.0630"/px
F480M.)

**This is not proof the NOT-in-v1.9 children are real.** v1.9 is itself a
different, independently-run reduction with its own detection limits near a
bright saturated core (the same crowding problem that makes any check this
close to a saturated star hard), so absence from v1.9 is consistent with a
real companion v1.9's own saturated-star handling missed, OR with it
continuing to be an artifact — and per Section 5, the "confirmed" label
feeding into this table is itself not a reliable real/fake discriminator.
59-75% "not in v1.9" is therefore an upper bound on how many of Section 4's
confirmed children would represent new, real detections if deblending were
reliable, not a measurement of how many of them are.

## 7. Reading Section 4 together with Section 3

Section 4 measures whether the group-0-core-COUNT signal discriminates real
from fake shipped children; it does not, for F480M, and is statistically
inconclusive for F212N. Section 3's fixes (defect filtering, per-centre
matching, fail-closed visibility) are independently correct regardless of
that answer — they fix bugs in how the count is computed and consumed, not
in whether the count is a good signal. The two conclusions are therefore not
in tension: the gate's bookkeeping is fixed, and its underlying signal is
shown, with reasonable power for F480M, not to be the discriminator the
original (circular) measurement claimed.

## 8. New evidence: shipped-catalog comparison, prodD (deblend ON, no gate) vs v1.9 released (deblend OFF)

A different, independent check compares the first prodD m7 catalog for tile
o075 (deblend on, this PR's gate not yet applied — prodD predates it) against
the released v1.9 joint catalog for the same tile (deblend off), same box.

**Satstar "secondaries"**: a `replaced_saturated` source counted as a
secondary when a BRIGHTER `replaced_saturated` source sits within radius R
(the signature a spurious split leaves: a real bright saturated star plus one
or more spurious fainter companions nearby).

| filter | catalog | nsat | secondaries <0.3" (q>1) | <0.6" (q>1) | <1.0" (q>1) |
|---|---|---|---|---|---|
| F212N | D (deblend on) | 11459 | 285 (158) | 1293 (789) | 3129 (1899) |
| F212N | P (deblend off, v1.9) | 11177 | 90 (13) | 888 (64) | 2699 (101) |
| F480M | D (deblend on) | 1776 | 1 (1) | 34 (26) | 111 (65) |
| F480M | P (deblend off, v1.9) | 1772 | 1 (1) | 14 (12) | 63 (34) |

F212N shows a large, bad-fit-dominated excess under deblending: the
bad-fit (`qfit > 1`) fraction of close secondaries jumps from 14% (<0.3") /
7% (<0.6") / 4% (<1.0") in P to 55% / 61% / 61% in D. F480M's totals are far
smaller (this tile has much less saturated area in F480M overall), but the
same direction is present at <0.6": the bad-fit secondary count rises from 12
(P) to 26 (D), more than double, while the raw (not-bad-fit-restricted) pair
count barely moves (14 to 34).

A second, independent tile (o067, `prodDcheck/secondaries_o067.txt`) shows
the identical pattern in both filters, at a larger sample:

| filter | catalog | nsat | secondaries <0.3" (q>1) | <0.6" (q>1) | <1.0" (q>1) |
|---|---|---|---|---|---|
| F212N | D (deblend on) | 9986 | 271 (140) | 1088 (618) | 2596 (1507) |
| F212N | P (deblend off, v1.9) | 9297 | 62 (10) | 660 (73) | 2041 (132) |
| F480M | D (deblend on) | 1705 | 1 (0) | 32 (23) | 99 (64) |
| F480M | P (deblend off, v1.9) | 1162 | 1 (1) | 4 (3) | 21 (13) |

F212N bad-fit secondaries <0.6" rise from 73 (P) to 618 (D), 8.5x; F480M's
rise from 3 (P) to 23 (D), 7.7x. The earlier framing of this check ("F480M
does not show the same deblend-specific excess ... that it shows for
F212N") does not hold once a second tile is checked: F480M's bad-fit-secondary
excess is present in both tiles, at a relative size (7-8x) comparable to
F212N's, just at absolute counts set by how little of F480M saturates in the
first place.

**Bright 2MASS-anchored check**: 2MASS Ks<12.5 stars with `Qflg` A/B and
`Cflg` 0, inside the merged i2d footprint, matched within 0.5" against each
catalog by plain nearest-neighbor; D and P share the same photometric
zeropoint run per filter. For F212N's 30 stars with Ks<9.5, all 30 have a
nearest match in both D and P, and the bulk `m - Ks` offset agrees to ~0.1
mag between them — but for **5 of the 30**, the nearest D match disagrees
with P by 3.3 to 9.2 mag:

| Ks (2MASS) | D: m-Ks-zp (sat, qfit, sep) | P: m-Ks-zp (sat, qfit) |
|---|---|---|
| 8.86 | +5.70 (sat0, q0.02) | +0.76 (sat1, q0.08) |
| 9.24 | +4.48 (sat1, q4.55) | -0.26 (sat1, q0.07) |
| 8.89 | +8.61 (sat0, q0.56) | -0.03 (sat1, q0.04) |
| 9.06 | +9.16 (sat0, q0.10) | -0.14 (sat1, q0.05) |
| 8.61 | +3.30 (sat1, q1.62) | -0.35 (sat1, q0.05) |

The disagreement is a matching artifact, not missing or corrupted primary
photometry: the D-side entries in this table are not the primary star —
they are a close (0.19-0.35") spurious extra source that a plain
nearest-neighbor match picked up instead, with its own `qfit` (0.56-4.55) far
worse than the clean primary fit (`qfit` 0.03-0.08 in P, and `qfit` 0.02-0.56
for the true primary in D once it is matched directly rather than by
nearest-neighbor). Re-matching D with a brightest-within-1" rule instead of
plain nearest-neighbor (prodDcheck/brightest1.py's method, a scratch script
not tracked here) finds the true primary for all 5 stars, with photometry
matching P. The o067 same-star brightest-match check
(`prodDcheck/brightest1_o067.txt`, matching the brightest 20% of P sources
to D within 0.1") confirms this directly at much
larger sample size: D-P is flat to +0.011 mag median (F212N, n=15342) and
-0.018 mag median with per-magnitude-bin deviations up to +0.031 mag (F480M,
n=14869). Deblending does not corrupt the primary stars' own photometry; it
adds a spurious close companion that a plain nearest-match crossmatch (as
tabulated above) can pick up instead of the real star.

Drilling into the two worst cases against D's own `basic`/`vetted` m7 catalog
confirms the spurious companions persist after standard vetting: e.g. the
Ks=9.24 star has a second D detection 0.19" away at `qfit=4.55` and a third
0.52" away at `qfit=2.59`, both far worse fits than the primary's `qfit=0.09`.
For F480M, all 6 of the tile's Ks<9.5 stars match cleanly (nearest-neighbor)
in both D and P in this particular tile — no bright-star mismatch of the
kind shown above turned up at this small sample size, consistent with F480M
having far fewer saturated pixels overall (secondaries tables above), not
with F480M's spurious-split secondaries being absent — the o075/o067
secondaries tables above show they are present in F480M too, just at smaller
absolute counts.

**Caveat (stated by the person who ran this check):** D also carries this
campaign's ePSF and residual-background changes alongside deblend, so
attributing the F212N damage to deblending specifically is a reasonable
inference from this comparison, not a proven one. The clean separation is to
re-catalog o075 (or another representative F212N-heavy tile) through m7 with
this PR's gate turned on (or with `--deblend-satstars` off entirely) and
check whether the secondaries/bright-star corruption falls back toward P's
level. That re-catalog was not run in this session: it is a full m2-m7
production pipeline pass, well beyond the scope of a small scratch-only test
job and outside the "no SLURM on production chains" limit this investigation
operated under. It is the concrete, well-defined follow-up this report
recommends before the gate (or a deblend on/off decision) ships to the rest
of the 10678 footprint.

## 9. Synthesis

Four measurements bear on whether, and how, satstar deblending should ship:

1. **Section 4** (group-0-core count vs cross-dither/cross-filter
   "confirmed real"): the gate's own accept/reject criterion does not
   discriminate real from fake for F480M (98.8% vs 93.1% confirmed at
   bucket a vs b) and is underpowered to judge for F212N (3 groups in
   bucket a).
2. **Section 5** (does "confirmed real" itself mean real): no — a
   population of 100% single-star PSF/saturation artifacts passes the same
   cross-dither criterion ~99% of the time (5.2), so Section 4's confirmed
   fractions and its F480M "80.4% of real children discarded" removal
   estimate are not reliable evidence of reality in either direction.
3. **Section 6** (needed): 59-75% of Section 4's "confirmed" children are
   absent from the deblend-off v1.9 catalog, but per Section 5 this bounds
   how many COULD be new real detections, not how many ARE.
4. **Section 8** (shipped-catalog prodD-vs-v1.9 comparison, reference-free,
   two tiles): deblending without a gate measurably damages the catalog in
   BOTH filters at <0.6" — bad-fit secondary counts (P to D) rise 8.5-12.3x
   in F212N (o067, o075) and 2.2-7.7x in F480M (o075, o067) (updated from
   this round's wording fix; the two filters differ in absolute count, set
   by how much of each filter saturates, not in direction) — and corrupts 5
   of 30 bright F212N 2MASS stars' naive nearest-match photometry (the
   primaries themselves are fine; a spurious close companion intercepts the
   match).

Reading these together: the evidence that previously argued FOR running
deblending ungated on F480M ("most of its shipped children are confirmed
real, and the gate would discard most of them for no benefit") is
undermined by Section 5 — that confirmation test cannot distinguish real
children from artifacts for either filter. The evidence that previously
argued the F480M damage in Section 8 was absent ("F480M shows none of
this") was wrong — Section 8, revised this round, shows comparable
relative damage to F212N. What survives, un-weakened, is: (a) the gate's
own group-0-core-count criterion is shown, with reasonable power for
F480M, not to discriminate real from fake; (b) ungated deblending
measurably damages the shipped catalog in both filters; (c) no measurement
in this report establishes that deblending recovers a known-real,
known-needed population large enough to justify that damage, in either
filter.

**Per-filter recommendation (off / gate / on):**

- **F212N: off.** Section 8 shows substantial, bad-fit-dominated catalog
  damage (including 5 of 30 bright stars) with deblending ungated, and
  nothing in this report establishes the gate removes that damage (bucket a
  is underpowered at 3 groups) or that the real-vs-fake split the gate is
  built on is reliable (Section 5).
- **F480M: off.** Section 8 (revised) shows the same direction of damage as
  F212N, at smaller absolute counts because less of F480M saturates.
  Section 4 shows the gate's group-0-core criterion carries no measurable
  discriminating power here at a well-powered sample size, so "gate" is not
  shown to do better than "on" for F480M specifically, and Section 5 removes
  the strongest argument ("the gate discards real confirmed children for no
  benefit") for leaving F480M deblending ungated.

This is consistent with, and reinforced by, B5+'s existing production
default of deblending OFF under the maintainer's rule "only if needed and
reliable": this round's results do not show reliability (Section 5) or
establish need (Section 6) for either filter, and do show damage from
running it ungated (Section 8).

This PR's own scope is unchanged by this recommendation: it fixes three
bugs in the gate's own logic (Section 3) that are correct independent of
whether or when deblending ships, and a per-filter on/off default remains
the maintainer's decision, not a change this PR makes. The concrete,
well-defined follow-up this report continues to recommend before any
change to a shipping default is the Section 8 re-catalog (o075 or another
representative tile) with deblending off entirely, to separate its effect
from this campaign's concurrent ePSF/background changes.

## 10. Test status

- `pytest jwst_gc_pipeline/reduction/tests/test_satstar_deblend_gate.py` — 31 passed
- `pytest jwst_gc_pipeline/reduction/tests/test_satstar_deblend_gate.py jwst_gc_pipeline/reduction/tests/test_satstar_deblend_seed_kind.py jwst_gc_pipeline/photometry/tests/test_satstar_recovery_cache.py` — 53 passed
- `pytest jwst_gc_pipeline/reduction/tests/ -k "satstar or deblend"` — 222 passed
- `pytest jwst_gc_pipeline/photometry/tests/ -k "satstar or recovery or cache"` — 601 passed
- grep-guard tests (`test_no_adhoc_nn_median_astrometry.py`, `test_no_sip_frame_astrometry.py`) unaffected

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01ATHHZxgVuacNKtoUeVXrKQ

Written by claude session_01ATHHZxgVuacNKtoUeVXrKQ
