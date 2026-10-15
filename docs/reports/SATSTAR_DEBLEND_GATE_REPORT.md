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
measurement (Section 4) and new shipped-catalog evidence (Section 6).

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
treat as a settled measurement on its own. Section 6 supplies a separate,
better-powered F212N check that does not depend on the group-0-core signal
at all.

**What the fraction of currently shipped real survivors the gate would remove
is, per filter:** F480M 80.4% (n=1314, well-powered); F212N 97.6% (n=42,
too few confirmed cases for a precise estimate, but the direction agrees).

## 5. Reading Section 4 together with Section 3

Section 4 measures whether the group-0-core-COUNT signal discriminates real
from fake shipped children; it does not, for F480M, and is statistically
inconclusive for F212N. Section 3's fixes (defect filtering, per-centre
matching, fail-closed visibility) are independently correct regardless of
that answer — they fix bugs in how the count is computed and consumed, not
in whether the count is a good signal. The two conclusions are therefore not
in tension: the gate's bookkeeping is fixed, and its underlying signal is
shown, with reasonable power for F480M, not to be the discriminator the
original (circular) measurement claimed.

## 6. New evidence: shipped-catalog comparison, prodD (deblend ON, no gate) vs v1.9 released (deblend OFF)

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
smaller (this tile has much less saturated area in F480M) and both D and P
already carry a high bad-fit fraction among the few close pairs present
(76-100%), so this check does not show the same deblend-specific excess for
F480M that it shows for F212N.

**Bright 2MASS-anchored check**: 2MASS Ks<12.5 stars with `Qflg` A/B and
`Cflg` 0, inside the merged i2d footprint, matched within 0.5" against each
catalog; D and P share the same photometric zeropoint run per filter. For
F212N's 30 stars with Ks<9.5, all 30 are present in both D and P, and the
bulk `m - Ks` offset agrees to ~0.1 mag between them — but **5 of the 30**
disagree by 3.3 to 9.2 mag between D and P, every one driven by a close
(0.19-0.35") spurious extra source in D with `qfit` 0.56-4.55 (versus a clean
match at `qfit` 0.03-0.08 in P for the same star):

| Ks (2MASS) | D: m-Ks-zp (sat, qfit, sep) | P: m-Ks-zp (sat, qfit) |
|---|---|---|
| 8.86 | +5.70 (sat0, q0.02) | +0.76 (sat1, q0.08) |
| 9.24 | +4.48 (sat1, q4.55) | -0.26 (sat1, q0.07) |
| 8.89 | +8.61 (sat0, q0.56) | -0.03 (sat1, q0.04) |
| 9.06 | +9.16 (sat0, q0.10) | -0.14 (sat1, q0.05) |
| 8.61 | +3.30 (sat1, q1.62) | -0.35 (sat1, q0.05) |

Drilling into the two worst cases against D's own `basic`/`vetted` m7 catalog
confirms the spurious companions persist after standard vetting: e.g. the
Ks=9.24 star has a second D detection 0.19" away at `qfit=4.55` and a third
0.52" away at `qfit=2.59`, both far worse fits than the primary's `qfit=0.09`.
For F480M, all 6 of the tile's Ks<9.5 stars match cleanly in both D and P
(no analogous corruption) — consistent with Section 4's finding that F480M's
shipped deblend output does not show the same severity of spurious-split
damage that F212N's does.

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

## 7. Synthesis

The two independent checks point the same way for F212N and in apparent
tension for F480M, and the resolution is that they are measuring different
things. Section 4 asks whether the group-0-core COUNT discriminates real
from fake among children that already reached the shipped catalog; for
F480M, it does not. Section 6 asks whether deblending changes what reaches
the shipped catalog at all, relative to not deblending; for F212N, it does,
substantially and for the worse on bright stars. F480M shows little of that
damage in Section 6, which is why Section 4's "the gate removes 80% of real
F480M children for no measurable precision gain" is not undercut by Section
6 — the F480M population the gate is operating on was already largely clean
before the gate, consistent with the gate's own near-zero discriminating
power there.

Recommendation: keep `SATSTAR_DEBLEND_GATE` default-on given the F212N
evidence of real, substantial shipped-catalog damage without any gate
(Section 6), while recording plainly that this session's independent
measurement does not confirm the group-0-core-count signal specifically
improves F212N reliability at the sample size reached here (Section 4), and
does show, at a well-powered sample size, that it removes the large majority
of real F480M children without a demonstrated precision benefit (Section 4).
A per-filter default is not implemented in this PR — it would be a
substantive new feature, not a bug fix — but is worth the maintainer's
explicit consideration once the Section 6 follow-up (re-catalog o075 with
the gate on) confirms or rules out the ePSF/background confound for F212N.

## 8. Test status

- `pytest jwst_gc_pipeline/reduction/tests/test_satstar_deblend_gate.py` — 31 passed
- `pytest jwst_gc_pipeline/reduction/tests/test_satstar_deblend_gate.py jwst_gc_pipeline/reduction/tests/test_satstar_deblend_seed_kind.py jwst_gc_pipeline/photometry/tests/test_satstar_recovery_cache.py` — 53 passed
- `pytest jwst_gc_pipeline/reduction/tests/ -k "satstar or deblend"` — 222 passed
- `pytest jwst_gc_pipeline/photometry/tests/ -k "satstar or recovery or cache"` — 601 passed
- grep-guard tests (`test_no_adhoc_nn_median_astrometry.py`, `test_no_sip_frame_astrometry.py`) unaffected

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01ATHHZxgVuacNKtoUeVXrKQ

Written by claude session_01ATHHZxgVuacNKtoUeVXrKQ
