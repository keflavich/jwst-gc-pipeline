# Astrometric WCS correction flow — which files get corrected, and how

**Audience:** anyone (human or agent) modifying the NIRCam reduction/alignment.
**Why this exists:** to keep an unambiguous, reproducible path from the archive
L2 products (`*_cal.fits`) to the final mosaics and catalogs, and to prevent
**double-correction** of the astrometric WCS.

Implemented in:
- `PipelineRerunNIRCAM-LONG.py` — `fix_alignment()` (per-exposure), Image3 call.
- `alignment_config.py` — the per-field registry: which reference frame and which
  shift source each `(proposal, observation)` uses (**NIRCam only**).
- `unified_alignment.py` — `resolve_shift()` resolves that declaration for one
  exposure; the single code path all NIRCam fields take.
- `photometry/merge_catalogs.py` — `shift_individual_catalog()` (catalog side).

> **Retired 2026-07-11:** the post-resample mosaic realign (`realign_to_catalog` /
> `realign_to_vvv` → `*_realigned-to-refcat.fits` / `*_realigned-to-vvv.fits` +
> `sync_gwcs_to_fits_wcs`) is **gone**. On the dense-refcat GC fields it was a
> guarded no-op (OLCRVAL = none, i.e. a byte-identical ~5 GB copy of `_i2d`), and
> `scripts/release/stage_release.py` publishes `_i2d.fits` as the release
> deliverable. The astrometric solution now has **exactly one** authoring point —
> per-exposure `fix_alignment` — and the `_i2d` mosaic is correct *by construction*.
> The `realign_to_catalog` / `realign_to_vvv` functions remain as `NotImplementedError`
> stubs so any stale caller fails loudly.

---

## ⛔ FORBIDDEN: dense-nearest-neighbour-median astrometry

**Never compute or apply an astrometric offset as the MEDIAN (or mean) of
nearest-neighbour matches (`match_to_catalog_sky` / `search_around_sky`) against a
DENSE reference catalog (VIRAC2 / VVV / GNS, median NN spacing ≲ 3").** When the
true shift exceeds the reference's nearest-neighbour spacing, NN pairs the WRONG
star and the median **collapses toward ~0** (or a spurious value). It fabricates
false agreement and has repeatedly fooled *validation* of the GC fields (a
NN-median check "confirms 0.00 fine" on a frame that is really off). The
brick-1182 v001 ~20" error came from an offsets-table CURATION collapse (see the
brick-1182 note) — the same class of silent false-agreement failure.

This is now enforced in code:
`jwst_gc_pipeline.photometry.measure_offsets.assert_sparse_reference_for_nn_median`
**raises `DenseNNMedianAstrometryError`** on a dense reference. It guards
`measure_offsets`, `bootstrap_reference_catalog`,
`combine_singleframe(realign=True)`, and the `generate_offsets_table` validation.
(The former `realign_to_catalog` guard-site was retired with the realign step.)

**Use instead:**
- **2D offset-histogram stacking** — histogram *all* pairwise offsets within ~3",
  take the peak (robust no matter how large the shift). Public helper:
  `jwst_gc_pipeline.photometry.astrometry_offsets.measure_offset` (and
  `measure_offset_grid` for the mandatory per-tile map). See also
  `scripts/reduction/astrometry_audit.py::xcorr` and
  `scripts/miri_reduction/apply_measured_miri_wcs_offsets.py::refine_offset`.
- **a SPARSE reference** — the Gaia-only subset (`source == b'GaiaDR3'`, medNN
  ~5.7"), never the full dense catalog.

**Sign-off requires a PER-TILE map.** A half-mosaic can be grossly SHIFTED while
the field-average reads ~0 (brick-1182 visit-001: a clean ~20" rigid step across the
y=0.5 seam). Always map the offset PER TILE
(`measure_offset_grid`, `registration_failsafes.py`) and require per-tile peak
contrast ≳ 5 everywhere.

A grep-guard test (`jwst_gc_pipeline/photometry/tests/test_no_adhoc_nn_median_astrometry.py`)
fails CI if a new file pairs a NN match with a median/mean — do not write ad-hoc
`match_to_catalog_sky(...).median()`; call `measure_offset` instead.

---

## TL;DR — where astrometric corrections live

| Product (role) | Gets a WCS correction? | Mechanism | Idempotent? |
|---|---|---|---|
| `*_cal.fits` (archive L2b, per-exposure) | **No** — never modified in place | — | (immutable input) |
| `*_destreak*.fits` / `*_align.fits` (per-exposure working copy) | **Yes — GWCS** | `fix_alignment()` → `jwst.tweakreg.utils.adjust_wcs` | **Yes** (`RAOFFSET` header guard) |
| `*_crf.fits` (CR-flagged per-exposure, from Image3) | inherits the corrected GWCS | (produced by Image3 from the corrected input) | n/a |
| `*_i2d.fits` (resampled mosaic — **final image deliverable**) | inherits | resample of the corrected exposures | (pristine) |
| per-frame catalogs (`*_daophot_basic.fits`) | use the corrected crf GWCS | (read crf GWCS) | n/a |
| merged catalog | **Yes — table-space** | `shift_individual_catalog()`: `final = centroid − RAOFFSET_meta + dra_table` | re-derivable from any offsets table |

**The astrometric solution now has exactly ONE authoring point:**
1. **Per-exposure** (`fix_alignment` → `adjust_wcs`): the science-bearing tie.
   Catalogs (on crf) and the `_i2d` mosaic both inherit it. The post-resample
   mosaic realign was retired 2026-07-11 (see the note at the top).

---

## The reproducible path (per-exposure → final)

```
archive  jw…_cal.fits                          (MAST L2b; assign_wcs GWCS; NEVER edited in place)
   │  destreak()  →  jw…_destreak_oNNN.fits     (working copy)
   │  fix_alignment(...)                        (per-exposure GWCS shift via adjust_wcs;
   │                                             shift resolved by unified_alignment.resolve_shift
   │                                             from the alignment_config.py declaration
   │                                             (locked / consensus table, or recorded bulk);
   │                                             writes RAOFFSET/DEOFFSET + OLCRVAL → IDEMPOTENT)
   ▼
Image3Pipeline.call(..., tweakreg skip=True)    (TweakRegStep is SKIPPED — see note)
   ├─►  jw…-<filt>-merged_crf.fits  (per-exposure, CR-flagged, corrected GWCS)  ──► CATALOGS (crf-space)
   └─►  jw…-<filt>-merged_i2d.fits  (resampled mosaic, corrected GWCS; pristine)  ← FINAL IMAGE DELIVERABLE
```

**TweakRegStep is intentionally skipped** (`tweakreg_parameters['skip'] = True`).
All absolute alignment comes from our `fix_alignment` (per-exposure). Do not
re-enable TweakReg, or you will double-correct.

---

## One correction stage, and why no double-correction

- `fix_alignment` ties **each exposure** to the reference using the per-frame
  offsets table (relative frame-to-frame + bulk). It is **idempotent**: the first
  thing it does is check for a `RAOFFSET` keyword and bail if present
  (`align_to_catalogs.py` / `PipelineRerun … fix_alignment`, the `if 'RAOFFSET' in header` guard).
  Re-running the pipeline therefore applies the shift exactly once.
- Because the tie is baked into every exposure's GWCS **before** Image3, both the
  `_crf` (→ catalogs) and the resampled `_i2d` mosaic inherit it. The mosaic is
  correct *by construction*: the single per-exposure shift is the only one it
  ever receives (the old `realign_to_catalog` step was retired 2026-07-11).
- Catalogs read the crf GWCS and re-express the tie in table space:
  `shift_individual_catalog` does `centroid − RAOFFSET_meta + dra_table`,
  i.e. it *removes* the GWCS-baked `RAOFFSET` and re-applies the current offsets
  table value. This makes the catalog frame re-derivable from any offsets table
  by a table edit alone, and keeps catalog ↔ mosaic ties consistent
  (both ultimately trace to the same offsets table + refcat).

**Single rule to avoid double-correction:** the astrometry is corrected at *exactly
one* place — per-exposure `fix_alignment`. Never add a post-resample mosaic
corrector, and never edit `_cal.fits` or `_i2d.fits` in place.

**Verification ladder (2026-07-12):** cataloging re-verifies this tie at every
merge stage — visit-consensus per-exposure checks (2 mas), frozen-solution
checks at m3–m6, and a cross-filter gate at the m7 merge.  A verified
misalignment is corrected HERE (offsets table + regenerate from `_cal`; the im0
mosaics are stale-tagged `*_im0_badastrom.fits`), never anywhere else.  See
`../photometry/ASTROMETRY_CHECKPOINTS.md`.  `fix_alignment` stamps `APROV*`
provenance cards (stage, method, applied shift, references, table) next to
`RAOFFSET/DEOFFSET` at apply time.

---

## Tooling: use STScI tools

- **Per-exposure GWCS shifts MUST use `jwst.tweakreg.utils.adjust_wcs`.** It applies
  the shift on the `v2v3`/tangent frame of a *calibrated* (`_cal`/`_tweakreg`/`_skymatch`)
  GWCS — the supported, correct path. `fix_alignment` already does this. Do **not**
  hand-edit `crval`/`pc` of a per-exposure GWCS.

- **The whole astrometric tie is applied at the `_cal`/per-exposure level** (via
  `adjust_wcs` in `fix_alignment`), so the resampled i2d is correct *by
  construction* and the i2d and catalogs share a single tie mechanism. This is
  also the only supported option: `adjust_wcs`'s own docstring states it is
  *"not designed to handle … GWCS of resampled images"*, and STScI ships no
  sanctioned resampled-WCS shifter. Any residual rigid mosaic zero-point must be
  absorbed into the per-exposure offsets table, never re-introduced as a
  mosaic-level GWCS edit.

---

## ⛔ The GWCS is authoritative; the FITS/SIP header is a fitted approximation

Every detector-frame product carries **two** WCS representations:

| representation | where | status |
|---|---|---|
| **GWCS** (ASDF extension, `model.meta.wcs`) | `_cal`, `_destreak`, `_align`, `_crf` | **authoritative** — the full SIAF distortion + velocity aberration + projection chain |
| FITS `RA---TAN-SIP` (SCI header) | same files | a *fitted low-order polynomial approximation* of the GWCS, for plain-`astropy.wcs` consumers (DS9/CARTA, `reproject`) |

SIP is a polynomial fit to the JWST distortion and cannot reproduce it exactly.
**Read the GWCS for anything
astrometric.** The SIP header exists for display and for external tools, and its
fidelity depends on how it was fitted.

### The 0.25 px default trap (found 2026-07-29)

`gwcs.WCS.to_fits()` / `to_fits_sip()` default to `max_pix_error=0.25` **pixels**.
STScI's `jwst.assign_wcs.util.update_fits_wcsinfo` uses `0.01`. Every place the
pipeline re-stamped a corrected GWCS with a bare `header.update(ww.to_fits()[0])`
therefore *replaced the delivered fit with one an order of magnitude worse*:

| product | `A_ORDER` | median | max FITS-vs-GWCS |
|---|---|---|---|
| brick F182M nrca1 `_cal` (MAST, 0.01 px) | 4 | 0.17 mas | 0.49 mas |
| brick F182M nrca1 `_destreak`/`_crf` | 3 | 0.83 mas | **5.5 mas** |
| brick F410M nrcalong `_crf` | 3 | 0.63 mas | **6.6 mas** |
| sickle F770W `_align`/`_crf` (MIRI) | 3 | 1.22 mas | **8.4 mas** |
| same GWCS refit at `max_pix_error=0.01` | 4–5 | 0.000 mas | 0.000 mas |

That 5–8 mas is *position-dependent* and differs per detector **and per filter**,
so a bulk tie leaves it in place: it injects spurious structure exactly where
the astrometric gates look (2 mas m2 per-exposure consensus, 5 mas m7
cross-filter, 30 mas inter-frame overlap).

Two further defects of the bare update, both now fixed:

1. `Header.update` **merges**. A degree-3 fit written over a delivered degree-4
   header leaves orphan `A_0_4`/`A_1_3`/`A_2_2`/`A_3_1`/`A_4_0` cards that
   contradict the written `A_ORDER=3`. Present on **nearly** every
   `_destreak.fits` in the archive (55 of 60 sampled from the 7,012 on disk).
   **These orphans are inert in astropy** — astropy sizes the SIP matrix from
   `A_ORDER` and reads only terms up to it (deleting all ten changes positions
   by 0.000000 mas). The entire 5–8 mas is attributable to `max_pix_error`
   alone. The orphans are stripped anyway, because a self-contradicting header
   is a trap for any reader that infers the order from the cards present, and
   for non-astropy consumers.
2. The written FITS WCS went unverified against the GWCS.
   `check_wcs` compared only the array **centre**, which agrees by construction
   — the distortion residual lives at the corners (brick F182M nrca1: centre
   0.0000 mas, (0,0) 5.117 mas, (2047,2047) 5.289 mas). A 25× loosening of the
   *requested* bound (0.25 px vs STScI's 0.01; the measured change is
   5.487 → 0.000 mas) survived because the check sat exactly where the error
   vanishes: a gate blind to the failure it is named for.

### The rule

- Any FITS WCS header written for a frame goes through
  **`jwst_gc_pipeline.reduction.fits_wcs_sync.sync_header_to_gwcs`**: it strips
  stale SIP coefficients, fits at `max_pix_error=0.01`, then **measures** the
  result against the GWCS over the whole array and raises
  `FitsGwcsMismatchError` above `FITS_GWCS_TOL_MAS` (0.5 mas). The achieved
  value is stamped as `SIPGWMAX`.
- Never call `ww.to_fits()` / `to_fits_sip()` with default arguments to write a
  header. (Reading `to_fits()[0]['CRVAL1']` for a log line is fine.)
- `astropy.wcs.WCS(header)` on a detector-frame product must always pass
  `relax=True` — a header whose CTYPE lost the `-SIP` suffix still carries the
  `A_*`/`B_*` terms, and without `relax` the distortion is silently dropped.
- Audit existing products with
  `python scripts/release/audit_fits_gwcs_agreement.py --field <field>`.
  Frames written before this fix read 5–8 mas and need regeneration *if*
  anything downstream reads their FITS header rather than their GWCS.

`i2d` mosaics are exempt: `resample` produces a rectified plain `RA---TAN` grid
with no SIP, so their FITS WCS is exact.

---

## Which shift a frame gets: the alignment registry (NIRCam)

**One path for every NIRCam field.** `fix_alignment` delegates the whole
decision: it calls `unified_alignment.resolve_shift(fn, proposal_id, field,
filtername, module, basepath, refname=…, use_average=…)`, which looks the field
up in **`alignment_config.py`** — the single source of truth for how each
`(proposal, observation)` is tied to an absolute frame. This replaced a
per-proposal `if/elif` chain whose `else` arm returned `(0, 0)`, so any field
missing a branch was silently left at the raw `assign_wcs` frame while the m2
checkpoint wrote corrections into a table nothing read (arches/2045,
quintuplet/2045, sgrb2/5365, cloudef/2092 obs 005 all sat in that state; a re-tie
loop on such a field re-measures the identical residual forever).

A field with **no entry** is now loud: `resolve_shift` prints
`NO CONFIGURED ALIGNMENT for proposal=… field=…` and returns
`AlignmentShift(configured=False)`; the frame stays at `(0, 0)` and says so.

⚠ **SCOPE: NIRCam only.** `PipelineMIRI.fix_alignment` carries its own dispatch
and its own inline policy constants (a `_PER_VISIT_SHIFT` map and a w51 rule);
`PipelineRerunNIRISS.fix_alignment` hardcodes
`rashift = decshift = 0 arcsec`. Both skip the component keywords and the
staleness guard. Folding them
in is follow-up work — read `alignment_config.py` as NIRCam-only.

Each entry declares two orthogonal things:

- **`reference_frame`** — WHICH absolute frame (`VIRAC2` / `Gaia` / `GNS`). GC
  fields use VIRAC2 (Gaia is the *frame* but far too sparse to be the reference
  *catalog* — see `CLAUDE.md`); halo/disk clusters outside the VVV footprint use
  Gaia directly. Each field declares its own frame.
- **`source`** — WHERE the numbers come from:

| `source` | reads | row kinds |
|---|---|---|
| `TABLE_LOCKED` | `offsets/Offsets_JWST_Brick<prop>_VIRAC2locked.csv` (curated, per-visit **or** per-exposure) | one total per matched row |
| `TABLE_CONSENSUS` | `offsets/Offsets_JWST_Brick<prop>_consensus.csv`, seeded/upserted by the m2 checkpoint | per-visit BULK sentinel + sparse per-exposure JITTER rows |
| `RECORDED_BULK` | constants in `alignment_config.py` itself | pure bulk; with `consensus_jitter=True` the consensus sentinel + jitter are summed on top |

Every applied shift decomposes as `total = bulk + jitter` — bulk is the
field/visit tie to the absolute frame (arcseconds when a guide-star acquisition
went wrong, known once, stable); jitter is the tens-of-mas per-exposure residual
around the visit consensus, re-measured every re-tie iteration.

### The configured fields

| proposal | obs | frame | source | reference filter | notes |
|---|---|---|---|---|---|
| 1182 | 004 | VIRAC2 | `TABLE_LOCKED` | — | brick, top-half visit-001 fix |
| 2221 | 001, 002 | VIRAC2 | `TABLE_LOCKED` | — | brick / cloudc |
| 4147 | all | VIRAC2 | `TABLE_LOCKED` | F212N | sgrc |
| 6151 | all | **Gaia** | `TABLE_CONSENSUS` | F210M | w51 (outside the VVV footprint); was F200W, which 6151 does not observe |
| 2045 | 001 | VIRAC2 | `TABLE_CONSENSUS` | F212N | arches |
| 2045 | 003 | VIRAC2 | `TABLE_LOCKED` | F212N | quintuplet |
| 5365 | all | VIRAC2 | `TABLE_LOCKED` | F212N | sgrb2 |
| 2211 | all | VIRAC2 | `TABLE_LOCKED` | F200W | gc2211 |
| 2092 | 005 | VIRAC2 | `TABLE_LOCKED` | F210M | cloudef obs 005 |
| 2092 | 002 | VIRAC2 | `RECORDED_BULK` + jitter | F210M | cloudef obs 002 |
| 3958 | 007 | VIRAC2 | `TABLE_LOCKED` | F210M | sickle; re-tied to VIRAC2 2026-08-04, GNS numbers deliberately NOT carried over |
| 1979 | all | **Gaia** | `RECORDED_BULK` | — | M4 (o002 + o003), halo cluster outside VVV |
| 1334 | all | **Gaia** | `RECORDED_BULK` | — | M92 (o001), pure per-visit shift |

Keep this table in step with `ALIGNMENT_CONFIG`; the module's own docstrings carry
the per-field provenance (`notes=`).

### How a locked-table row is selected

`_shift_from_locked` narrows on `(Visit, Filter)`, then by `Exposure` **only if
more than one row still matches** (so per-visit and per-exposure tables both
work), then by `Module` on the same condition (default OFF — filters lock
NRCA==NRCB together; F410M is the documented exception), and then by `Vgroup`
**unconditionally when the column exists** — a visit can dither across several
visit groups and the exposure number restarts in each, so a lone surviving row for
the *other* group is exactly the dangerous case. An **empty** `Vgroup` cell matches
any group (`vgroup_row_matches`), so pre-`Vgroup` rows keep applying. Narrowing
must leave exactly one row; any other count raises `ValueError`. The applied
numbers come from the `dra (arcsec)` / `ddec (arcsec)` columns.

⚠ **As of 2026-07-31 four tables on disk carry a `Vgroup` column**: cloudc's
2221, quintuplet's 2045, sgrc's 4147, and w51's `_consensus.csv`. Brick's 1182
and 2221 tables, and cloudef's and gc2211's, still use the pre-`Vgroup` layout,
so the narrowing stays inert exactly where it matters most: gc2211's own config
note records 6 visit groups reusing exposure numbers. Rebuild those tables with
the column before relying on the guard.

If no locked table exists, `_shift_from_locked` falls back to
`Offsets_JWST_Brick<prop>_<refname>_average.csv` (`use_average=True`, the
default) or `Offsets_JWST_Brick<prop>_<refname>.csv`, and requires a `refname`.
This fallback is **locked-source only**: a `TABLE_CONSENSUS` field whose table is
missing gets `(0, 0)` with `table_present=False`.

## Offsets-table provenance (how each `offsets/Offsets_*.csv` is built)

The `TABLE_LOCKED` tables are **inputs**: external builders write them and the
pipeline consumes them, so the builders are tracked for provenance. Losing a
builder makes a correction impossible to reproduce or audit from first
principles even though catalogs/mosaics already carry it. (`_consensus.csv` is
the exception: the m2 checkpoint writes it.)

| reference frame | offsets table | builder |
|---|---|---|
| Gaia/VIRAC2 (brick/cloudc) | `Offsets_JWST_Brick<pid>_VIRAC2[locked].csv` | `build_gaia_virac2_refcat_byquery.py` (seed refcat) + `build_virac2_offsets.py` |
| VIRAC2/Gaia, checkpoint-authored | `Offsets_JWST_Brick<pid>_consensus.csv` | `astrometry_checkpoint.seed_offsets_table_from_consensus` / `update_offsets_table` |
| VIRAC2 (sickle, prop 3958) | `Offsets_JWST_Brick3958_VIRAC2locked.csv` | `build_virac2_offsets.py` (REGION `'sickle'`) |
| ~~GNS (sickle, prop 3958)~~ | ~~`Offsets_JWST_Brick3958_GNS.csv`~~ | superseded 2026-08-04 — see below |

**Sickle → VIRAC2 (2026-08-04):** sickle was the last GC field whose recorded
bulk sat in the GNS frame while `refnames` already called it VIRAC2. GC policy is
that GC fields tie to VIRAC2, so it now has a `build_virac2_offsets` REGION entry
and a per-exposure `Offsets_JWST_Brick3958_VIRAC2locked.csv` (120 rows, 5 filters
× 24 exposures), declared `TABLE_LOCKED` — the same class as sgrc / quintuplet /
sgrb2.

The route was to BUILD the table rather than blank the recorded bulk and let step
0 re-measure: step 0 refuses to record a fresh tie for a field that is already
tied ("the field is tied; this (visit, band) is not", RC=4).

The measured GNS→VIRAC2 frame shift is **(+71.74, −70.09) mas** (std 1.5 / 1.45,
range < 4.8 mas across all five filters) — a coherent frame offset. The old GNS
numbers are deliberately NOT carried over: re-using them against VIRAC2 would
bake the frame difference in as if it were an astrometry correction.

*Historical:* a 2026-06-20 audit found sickle catalogs at the **raw `assign_wcs`
frame** (`RAOFFSET=0`, no offsets table for 3958), ~91 mas off the GNS reference
the mosaics were tied to, and the GNS frame was chosen at the time.
`brick2221/reduction/build_sickle_gns_offsets.py` (in the **brick-jwst-2221**
repo) built `Offsets_JWST_Brick3958_GNS.csv`, which is retained on disk but no
longer read.

**MIRI registration** is a separate, manual pre-step with its own path. The
sickle MIRI frames are registered to the NIRCam
F480M frame by region-specific scripts in **brick-jwst-2221**
(`brick2221/reduction/register_sickle_miri_o001_o002.py`,
`brick2221/reduction/register_o002_f770w_per_frame_to_f480m.py`,
`brick2221/reduction/register_o002_f770w_gwcs_to_f480m.py`,
`brick2221/reduction/merge_sickle_miri_o001_o002.py`). They
edit the per-frame FITS WCS / embedded gwcs in place (idempotent via
`MIRIDRA`/`MIRIDDE`/`MIRIWCSN`) and **must be run before cataloging** a sickle
MIRI obs, or its mosaics sit ~3.3″ off truth while the catalog underneath is
correct. The brick F2550W reduction-tool scripts in `scripts/miri_reduction/`
are region-general examples; the operational MIRI scripts live in brick-jwst-2221.

## Module-lock policy (NRCA == NRCB) and the F410M inter-module offset

`fix_alignment` applies **one shift per (visit, filter) to BOTH modules** (NRCA and
NRCB) — the locked table (`Offsets_JWST_Brick<pid>_VIRAC2locked.csv`) is keyed on
(Visit, Filter). This is deliberate: NIRCam's SIAF/`assign_wcs` solution
co-registers the two long-wave detectors (NRCA5, NRCB5) to <~1 mas *when assign_wcs is
run against a correct CRDS cache*, so an independent per-module tweak would
inject VIRAC2 noise and break the lock.

**ROOT CAUSE (2026-07-11): the F410M inter-module offset came from a STALE LOCAL
CRDS CACHE serving a module-swapped LW `filteroffset` mapping** — rather than
from the jwst version or the SIAF/distortion references, which were correct
throughout. Full incident
report, fingerprint table, and auditor checklist:
**[docs/reports/CRDS_STALE_FILTEROFFSET_RMAP_INCIDENT.md](../../docs/reports/CRDS_STALE_FILTEROFFSET_RMAP_INCIDENT.md)**.
Short version: `jwst_nircam_filteroffset_0004.rmap` was corrected in place by CRDS
early in Cycle 1; local caches seeded 2022-09 (brick, arches, arches_quintuplet,
cloudef, sgra, sgrb2, sickle, crds — all repaired 2026-07-11, stale copies kept as
`*.stale_20220901_swappedAB`) mapped `('LONG','A')→filteroffset_0008` (the module-B
file) and vice versa. Result: anti-symmetric per-module sky errors equal to the
(own−other) filter-offset difference — F410M ±26.3 mas/module (52.5 mas A−B
differential), F405N (F444W+F405N) ±11.0 (22.0), F466N ±1.9 — **independent of the
installed jwst version**. Once the cache is correct, every band re-assigns to 0.0 mas
across jwst 1.14→1.21; SW mappings were identical in both rmap generations, so SW
stayed correct throughout. Lesson for the auditor: a cross-check that varies the
CRDS *context* while reusing the same *cache* can only ever see context effects.
Checks:
`sha1sum $CRDS_PATH/mappings/jwst/jwst_nircam_filteroffset_0004.rmap`
(`aade9b095a34…` correct, `98d39dc5403e…` stale/swapped) and the `_cal` header
`R_FILOFF` (NRCALONG must use 0007, NRCBLONG 0008).

**Fix (surgical): re-run Image2 (assign_wcs) with a verified-fresh CRDS cache.**
Pinning the context keeps flat/photom references identical → photometry is preserved
through the release while astrometry is fixed. The offsets table stays untouched.
(Applied to brick+cloudc LW 2026-07-04.)

**Interim workaround currently in place (must be reverted if Image2 is re-run):** F410M was
given per-module rows (a `Module` column = `nrcalong`/`nrcblong`) in the locked table with a
~48 mas extra shift on NRCALONG, and the locked-table reader
(`unified_alignment._shift_from_locked`) narrows the match by module when >1 row
matches and a `Module` column is present. That
band-aid empirically reproduces what correct assign_wcs does (verified: F410M mosaic
module step 68 → ~0 mas). **DANGER:** it is applied on top of the OLD (buggy) `_cal` WCS.
If the `_cal` is regenerated with current jwst, remove the F410M split (revert to a single
both-module row) or it will double-correct by ~48 mas.

**Rule for the offsets-table builder:** a per-module split is a LAST-RESORT workaround for a
frame that cannot be reprocessed. Before splitting, first verify the CRDS cache
(sha1sum check above) and re-run assign_wcs against a fresh cache — a stale local CRDS
cache (module-swapped LW filteroffset mapping) was the cause for
F410M. (The brick 2221 locked table was
rebuilt with a single both-module F410M row after the 2026-07-04 re-assign; the
band-aid split is gone.)

## Reference epochs (so propagation is reproducible)

- Gaia DR3 reference epoch = **2016.0**.
- VIRAC2 (VizieR II/387) reference epoch = **2014.0** (Smith+2025: *"fixed at the
  reference epoch, 2014.0"*).
- The seed refcat (`build_gaia_virac2_refcat_byquery.py`) propagates each to the F115W
  observation epoch **2022.70** with per-source PM. `GAIA_EPOCH`/`VIRAC2_EPOCH`
  constants live at the top of that script.

---

## Inter-detector DVA correction (`dva_correction.py`, opt-in)

**What DVA is.** Velocity aberration: the spacecraft's ~30 km/s velocity
displaces apparent star positions toward the velocity apex (up to ~20.5"). The
bulk displacement is absorbed by the attitude (FGS guides on apparent
positions); only the *differential* part across the FOV matters for the WCS —
to first order a plate-scale factor `VA_SCALE` (|1−VA_SCALE| ≈ 1e-4, header
keyword). `assign_wcs` corrects it per detector by scaling V2V3 about *each
detector's own* reference point, which fixes the aberration WITHIN each
detector but leaves the aberration of the detector *separations* in the
delivered WCS. Residual = a per-detector rigid shift
`−(1−VA_SCALE)×(ref_d − C)` for any exposure-common point C: ±9–13 mas at
NIRCam module lever arms (the dominant part of the apparent "module A/B
offset"), and **epoch-dependent** (VA swings ±1e-4 over the year → module
separations move up to ~25 mas between epochs; a direct proper-motion hazard).
Measured on the Brick by network self-calibration: fitted inter-detector scale
= 9.7–9.9e-5 (2221, predicted 9.18e-5) and 1.05–1.06e-4 (1182, predicted
1.00e-4); after removal, static SIAF detector placements are 1–2.5 mas
(astrometry-paper `siaf_accuracy.tex`,
`scripts/analysis/siaf_selfcal/network_selfcal.py`).

**The correction.** `dva_correction.apply_dva_correction(fn)` applies the
per-detector rigid shift computed from the file's own header (`VA_SCALE`,
`RA_REF/DEC_REF`, with C = the V1 boresight `RA_V1/DEC_V1` — common to the
exposure by construction; the C-dependence is a common rigid shift absorbed by
the reference tie). Both the GWCS and the FITS SIP header are updated (same
mechanism as `fix_alignment`); idempotency via the `DVACORR` marker keyword
(`DVASHRA`/`DVASHDE` record the applied shift).

**Policy.**
- Opt-in: `APPLY_DVA_CORRECTION=1` enables the hook in `fix_alignment`
  (applied BEFORE the tie so offsets tables are measured on DVA-consistent
  frames). Default off = byte-identical behavior.
- Apply to `_cal`-derived working copies only; regenerating from
  `_cal` resets it (marker disappears with the overwrite) exactly like
  `RAOFFSET`.
- Do NOT "fix" the module offset with per-module offsets-table rows:
  the module separation error is deterministic and epoch-dependent — a fitted
  per-module shift goes stale as VA changes and injects reference noise (see
  the module-lock section above).

Shareable technical report on this issue (for STScI / upstream):
`docs/reports/DVA_INTERDETECTOR_REPORT.md`.

---

_Last updated 2026-07-11: retired the post-resample mosaic realign (`realign_to_catalog` /
`realign_to_vvv` / `sync_gwcs_to_fits_wcs` / `*_realigned-to-refcat.fits` gone — the tie is applied
once, per-exposure, and `_i2d` is the final image deliverable); added the opt-in inter-detector DVA
correction (section above); corrected the F410M module-lock root cause to a stale local CRDS
`filteroffset` cache (module-lock section). See also the offsets-table builders
(`_bench/build_sickle_gns_offsets.py`, `scripts/miri_reduction/` registration scripts) and
`f115w-astrometry-*` writeups in brick-jwst-2221._
