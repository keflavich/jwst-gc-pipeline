# CMZ HiPS mosaic + giant-catalog sharing — plan

Plan for (1) a **growing two-color CMZ HiPS** image that new fields (program
10678 as it rolls in) fold into without a full rebuild, and (2) **sharing the
full-CMZ giant catalog** with hierarchical/HiPS visualization in Aladin.

Written to sit next to the release machinery (`stage_release.py`,
`RELEASE_DEPLOYMENT_CHECKLIST.md`) — the release gate is the natural trigger for
regenerating the HiPS + catalog products.

---

## 0. Grounding — what exists today

**Image HiPS** (in `/orange/adamginsburg/jwst/jwst_scripts/`, the `jwst_rgb`
package — *not* this repo):
- Generator: **`reproject.hips.reproject_to_hips`** (astropy `reproject`), galactic
  frame, 512-px PNG tiles to order 13 (`jwst_rgb/save_rgb.py:191`). **Not** CDS
  `Hipsgen.jar`.
- Input is a **already-composited RGBA PNG** with embedded AVM/WCS — HiPS is built
  from the color PNG, not the FITS.
- **Full rebuild every time** (`shutil.rmtree` + regenerate). **No** incremental
  growth, **no** MOC, **no** catalog HiPS, **no** Allsky.
- The "third color by interpolation" already exists: `make_two_filter_rgb`
  (`scripts/gc2211_rgb_images.py:109`) sets R = long band, B = short band,
  **G = 0.5·(R+B)** of the asinh-stretched channels. Stretch = `simple_norm`
  asinh/log, percentile clips.

**Release** (`scripts/release/`): per-field `-merged_i2d.fits` mosaics + per-field
cross-band catalog (`basic_merged_indivexp_photometry_tables_merged_resbgsub_m7…`),
staged to `/orange/adamginsburg/jwst/releases/<version>/[<group>/]<field>/`, served
via Globus + HTTPS + a static webpage (`make_webpage.py`). **No cross-field CMZ
catalog** and **no footprint/coverage tracking** exist yet. **Program 10678 is
greenfield** — onboard via the `FIELDS` dict in `stage_release.py`.

**Filter reality (decisive for the two-color choice):**
- **F212N** — present CMZ-wide (sgra, sgrb2, sgrc, brick, cloudc): the reliable
  short-wave anchor.
- **F480M** — only sgrc, sgrb2, sickle. **Not** in brick/cloudc/gc2211.
- **F405N** — the CMZ-wide long-narrow band.
- ⇒ A literal *F212N–F480M* pair only covers part of the CMZ. Use a **per-field
  long-band policy** (below).

---

## Part 1 — Growing two-color CMZ HiPS

### 1.1 Design principle — separate the incremental substrate from the color

The current pipeline bakes color into a PNG *then* HiPS-tiles it, so any new field
forces a full recomposite + rebuild. Invert that:

> **Build one mono (grayscale, real-flux) HiPS per filter as the incremental
> substrate. Derive the color composite as a thin top layer.**

Because HiPS tiles are HEALPix-addressed, adding a field touches only the tiles it
overlaps (+ their ancestor tiles). Color (2-band + interpolated green) is then a
per-tile combine of the mono HiPS — cheap to re-derive, and re-derivable at
different stretches without recomputing the substrate.

```
per-field i2d ──► mono HiPS: HiPS/F212N   (incremental, real flux, fits tiles)
                  mono HiPS: HiPS/LONG     (F480M where present, else F405N)
                                  │
                                  ▼  (derive; cheap)
        color HiPS: R=LONG, B=F212N, G=0.5·(R+B)   ← 2-color + interpolated 3rd
                                  │
                                  ▼
                    Aladin Lite pane on the release webpage
```

### 1.2 The two long-band policy (F480M vs F405N)

Maintain **two** long-band mono HiPS and a coverage MOC per band:
- `HiPS/F480M` where F480M exists (sgrc, sgrb2, sickle, +10678 if it observes it),
- `HiPS/F405N` CMZ-wide.

The color layer picks, per sky region, the best available long band (prefer F480M,
fall back to F405N), recorded in a **per-band coverage MOC** (§1.4) so the
provenance of every colored region is explicit. This keeps a single seamless
CMZ-wide color mosaic without pretending F480M is everywhere.

(Alternative if a *uniform* long band is required for science-grade color: use
F405N everywhere — universal — and offer F480M as a separate overlay HiPS where
available. Decision for the author; see §4.)

### 1.3 Tooling: two viable substrates

| | **A. CDS `Hipsgen.jar`** (recommended for the survey substrate) | **B. `reproject.hips`** (keep for per-target quick-looks) |
|---|---|---|
| Incremental add | **Native** — re-run over an index incl. new files; only affected tiles rewrite | none (full rebuild); would need a custom tile-merge (§1.5) |
| Input | FITS i2d directly (real flux, mono) | composited PNG (current path) |
| MOC | **generates coverage MOC** | none |
| Color HiPS | **RGB HiPS from 2–3 mono HiPS** built-in | done upstream in the PNG |
| Allsky / low orders | yes | not written |
| Dep | Java (`Hipsgen.jar`, one jar) | pure Python (already in use) |

**Recommendation:** adopt **Hipsgen.jar** for the CMZ-wide growing mono+color
HiPS + MOC (it is purpose-built for exactly this: a survey that grows). Keep the
existing `reproject.hips` PNG path for per-target press-image RGBs. Wrap Hipsgen in
a thin python driver so invocation matches the repo's style.

If a Java dependency is unwanted, §1.5 gives the pure-Python incremental fallback.

### 1.4 Coverage MOC — also the footprint tracker the release lacks

For each field/filter, compute a **MOC** (Multi-Order Coverage, `mocpy`) from the
i2d valid-data footprint. The union MOC per filter = the survey coverage map. This:
- fills the "no footprint/coverage tracking" release gap,
- drives incremental rebuilds (the MOC of a *new* field = exactly the tiles to
  regenerate),
- is published as a `.fits`/`.moc` for Aladin overlay + machine-readable coverage,
- lets the webpage show "what's covered so far" as 10678 rolls in.

### 1.5 Pure-Python incremental fallback (if not adopting Hipsgen.jar)

`reproject_to_hips` per NEW field into a staging tree, then a `hips_merge` step:
for each `(Norder, Npix)` tile present in both the master and the new field,
combine (non-NaN priority / max / alpha-blend), write back; then regenerate
ancestor tiles bottom-up for the touched pixels only (and the Allsky). This is
scriptable (tiles are independent HEALPix PNGs) but re-implements what Hipsgen does
natively — hence the recommendation above.

### 1.6 Rollout / steps

1. **`cmz_hips/`** driver package (new, in `jwst_scripts` or this repo):
   `build_mono_hips(field, filter)`, `merge_or_update(master, field)`,
   `derive_color_hips(F212N, LONG)`, `build_coverage_moc(field, filter)`.
2. Backfill the current CMZ fields (F212N + F405N/F480M) into the master mono HiPS
   + MOC.
3. Derive the CMZ-wide color HiPS (2-color + interpolated green).
4. **Wire into the release**: after `stage_release.py` stages a field, call the
   incremental HiPS update for that field's F212N + long band, refresh the color
   HiPS + MOC. Gate on the same astrometry checks (a mis-registered field must not
   pollute the mosaic — reuse `check_interframe_overlap.py`).
5. **10678 onboarding**: add its `FIELDS` entry; the incremental path folds each
   new observation in automatically as it clears the release gate.
6. Embed an **Aladin Lite** pane on the release webpage (color HiPS + coverage MOC).

---

## Part 2 — Sharing the full-CMZ giant catalog

### 2.1 The gap

All catalog merging today is **within a field**. There is no CMZ-wide table and no
hierarchical catalog product. Three things to build: an **assembly**, a
**distribution format**, and a **visualization**.

### 2.2 Assemble the CMZ-wide catalog (new tooling)

`assemble_cmz_catalog.py`: vstack the per-field cross-band `resbgsub_m7` catalogs
into one table, with:
- **de-duplication in field-overlap regions** (same-star match across field
  boundaries within a tolerance; keep the higher-S/N / more-bands detection),
- a **provenance block** per source: `field`, `program`, `obsid`, source pipeline
  tag (ties into the versioning `GCTAG`/sidecar work — the CMZ catalog records
  which tagged run each field came from),
- **unified columns**: `ra`/`dec`, per-filter `mag_ab`/`mag_vega`/`flux_jy` +
  errors, `qfit`, `flags`, and a per-source filter-coverage bitmask (which bands
  actually measured it — filters differ by field).

Guard with the same astrometry sign-off as the release (cross-field seams are
exactly where a per-field tie error shows up).

### 2.3 Distribution format — HATS / Parquet (LSDB-native)

Publish the assembled catalog as a **HATS** (Hierarchical Adaptive Tiling Scheme,
formerly `hipscat`) **Parquet** dataset via `hats-import`: HEALPix-partitioned,
~equal objects per partition, Apache Parquet. Why:
- **LSDB** (LINCC Frameworks) reads it for parallel, out-of-core **cross-matching
  and analytics** — "80 TB from a laptop"; the format Rubin/STScI/IPAC/CDS/ESA are
  standardizing on.
- Cloud/Globus-friendly, partition-pruned queries (no full download to filter by
  sky region or magnitude).
- Keep the flat **FITS + ECSV** (current) *and* add a flat **Parquet** for the
  "just give me one file" user; HATS is the scalable path.

### 2.4 Visualization — HiPS catalog for Aladin

Generate a **HiPS catalog** (progressive catalog) with CDS **`Hipsgen-cat.jar`**
from the assembled table. This is the direct answer to "catalog visualization in
Aladin/HiPS": a zoom-dependent source layer that Aladin Lite/Desktop (and ESASky,
etc.) render natively — pan/zoom over millions of sources with no server. Embed it
in the release webpage's Aladin Lite pane, overlaid on the color image HiPS + the
coverage MOC.

> `mocpy` vs `Hipsgen-cat` vs `HATS`, to be explicit:
> - **`mocpy`** → *coverage* (footprint/where-is-data), not per-source display.
> - **`Hipsgen-cat` (HiPS catalog)** → *visualization* of sources in Aladin.
> - **`HATS`/`hipscat` + LSDB** → *analysis/distribution/cross-match* at scale.
> Use all three; they are complementary layers, not alternatives.

### 2.5 Release integration + steps

1. `assemble_cmz_catalog.py` (vstack + dedup + provenance + unified columns).
2. `hats-import` → HATS Parquet; also emit a flat Parquet + keep FITS/ECSV.
3. `Hipsgen-cat.jar` → HiPS catalog; `mocpy` → catalog coverage MOC.
4. `stage_release.py` gains a `--cmz-catalog` step that (re)assembles when any
   member field re-releases, and stages the HATS tree + HiPS catalog + MOC into the
   release tree next to the images.
5. `make_webpage.py` gains an **Aladin Lite** view: color image HiPS + catalog HiPS
   + coverage MOC, plus HATS/Parquet/FITS download links.

---

## 3. Cross-cutting: provenance tie-in

Both products stamp the **pipeline tag** (`jwst_gc_pipeline.versioning`): the HiPS
`properties` and the catalog HATS metadata record the tag + the MANIFEST of input
fields/tags they were built from. An incremental HiPS/catalog update records which
field/tag was folded in and when — so a growing CMZ product stays fully traceable
(and a re-released field triggers exactly the tiles/partitions it touches).

## 4. Decisions for the author

1. **Long band for the seamless color**: per-region best-available (F480M→F405N,
   §1.2) — or uniform F405N with F480M as an optional overlay?
2. **Hipsgen.jar (Java, native incremental + MOC + color) vs pure-Python
   incremental** (§1.3/1.5) for the survey substrate.
3. **HATS now or later**: ship HATS from the first CMZ-wide release, or start with
   flat Parquet + HiPS-cat and add HATS when catalog size/cross-match demand it?
4. Where the new tooling lives: `jwst_scripts` (with the existing HiPS/RGB code) vs
   this repo's `scripts/release/` (with the staging/gate machinery). Recommend the
   HiPS build in `jwst_scripts`, the release-integration + catalog assembly here.
